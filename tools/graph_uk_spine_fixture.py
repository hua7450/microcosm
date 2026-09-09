#!/usr/bin/env python3
"""Regenerate the hermetic charter-H2 UK spine parity fixture.

The oracle is the legacy :class:`microcosm.build.plan.StagePlan`: all 28
stages are the current production transform classes.  Private source files
are replaced only through their supported parsed-input seams.  The bundle's
``fixture.json`` is deliberately data-only so the graph's unbound UK registry
can reconstruct the same transforms; no recorded stage deltas are written or
replayed.

The oracle's identity is machine-specific, so it is never pinned:
``legacy_oracle_identity`` recomputes it in the acceptance test's own process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.source_manifest import SourceManifest, SourceStageSpec
from microcosm.build.uk_runtime.age_tail import UKAgeTailStageTransform
from microcosm.build.uk_runtime.cgt_imputation import (
    UKCGTPolicyParameters,
    uk_cgt_policy_parameters,
    uk_cgt_spine_stage_transform,
)
from microcosm.build.uk_runtime.cgt_structure import (
    UKCGTBandDonorStageTransform,
    UKCGTIncidenceCloneStageTransform,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.etb_services import UKETBServicesStageTransform
from microcosm.build.uk_runtime.etb_vat import UKETBVATStageTransform
from microcosm.build.uk_runtime.frs_brma import UKFRSBRMAStageTransform
from microcosm.build.uk_runtime.frs_council_tax import UKFRSCouncilTaxStageTransform
from microcosm.build.uk_runtime.frs_disability import UKFRSDisabilityStageTransform
from microcosm.build.uk_runtime.frs_education import UKFRSEducationStageTransform
from microcosm.build.uk_runtime.frs_education_grants import (
    UKFRSEducationGrantSplitStageTransform,
)
from microcosm.build.uk_runtime.frs_employment import UKFRSEmploymentStageTransform
from microcosm.build.uk_runtime.frs_household_draws import (
    UKFRSHouseholdDrawsStageTransform,
)
from microcosm.build.uk_runtime.frs_legacy_proxies import (
    UKFRSLegacyProxiesStageTransform,
)
from microcosm.build.uk_runtime.frs_person_draws import UKFRSPersonDrawsStageTransform
from microcosm.build.uk_runtime.frs_spine import (
    FRS_SPINE_TABLES,
    UKFRSSpineStageTransform,
    uk_frs_spine_seed_frame,
)
from microcosm.build.uk_runtime.frs_take_up import UKFRSTakeUpStageTransform
from microcosm.build.uk_runtime.graph import UK_SPINE_EXCLUSIONS, uk_spine_graph
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRCCapitalGainsBandTotal,
    HMRCCapitalGainsCell,
    HMRCCapitalGainsIncomeTotal,
    HMRCCapitalGainsJointDistribution,
    HMRCCapitalGainsSourceProvenance,
)
from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_BUILD_PERIOD,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRCIncomeBandTargetRecord,
    HMRCIncomeSourceProvenance,
    HMRCIncomeTargetSet,
)
from microcosm.build.uk_runtime.lcfs_consumption import (
    BUS_FARE_LCFS_CODES,
    UKLCFSConsumptionStageTransform,
)
from microcosm.build.uk_runtime.regional_uprating import (
    UKRegionalPropertyUpratingStageTransform,
)
from microcosm.build.uk_runtime.salary_sacrifice import (
    UKSalarySacrificeStageTransform,
)
from microcosm.build.uk_runtime.spi_income import SPI_DONOR_REQUIRED_COLUMNS
from microcosm.build.uk_runtime.spi_spine import (
    UKFRSHMRCSpineLeavesStageTransform,
    UKSPIIncomeSpineStageTransform,
    UKSPISupportChannelStageTransform,
)
from microcosm.build.uk_runtime.student_loans import UKStudentLoansStageTransform
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from microcosm.build.uk_runtime.uc_capital_coherence import (
    UKUCCapitalCoherenceStageTransform,
)
from microcosm.build.uk_runtime.uc_deduction_attributes import (
    UKUCDeductionAttributesStageTransform,
)
from microcosm.build.uk_runtime.uc_reporter_redraw import (
    UKUCReporterRedrawStageTransform,
)
from microcosm.build.uk_runtime.was_wealth import UKWASWealthStageTransform
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from microcosm.graph import graph_to_json

_REPOSITORY = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = (
    _REPOSITORY
    / "packages"
    / "microcosm-graph"
    / "tests"
    / "fixtures"
    / "parity"
    / "uk_spine"
)
_ROOT_HOUSEHOLDS = 135
_DONOR_ROWS = 64
_SPI_SAMPLE_FRACTION = _ROOT_HOUSEHOLDS / 10_000
_SPI_DONOR_SAMPLE_SIZE = 64
#: The packaged FRS spine roster the fixture exercises (manifest minus the
#: certified-pair exclusions); moves whenever a spine stage is added.
UK_FIXTURE_STAGE_COUNT = 28
_QRF_ESTIMATORS = 4

# These are the complete object-string surface observed in the unchanged
# legacy 28-stage output.  Graph storage uses pandas StringDtype/python.
_NORMALIZED_STRING_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "person": (
        "gender",
        "marital_status",
        "employment_status",
        "employment_sector",
        "aa_category",
        "dla_sc_category",
        "dla_m_category",
        "pip_m_category",
        "pip_dl_category",
        "current_education",
        "highest_education",
        "person_support_channel",
        "student_loan_plan",
    ),
    "benunit": ("benunit_support_channel", "uc_deduction_combination"),
    "household": (
        "region",
        "tenure_type",
        "accommodation_type",
        "council_tax_band",
        "brma",
        "household_support_channel",
        "source_household_key",
    ),
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _csv_codec_float(value: float) -> str:
    """Encode one float for exact decoding by the generic CSV codec.

    Pandas' default fast CSV float parser can move a correctly rounded
    decimal by one ULP.  An underscore between two mantissa digits keeps the
    token valid for Python/NumPy conversion while making CSV inference retain
    it as text; ``csv-tables`` then applies the declared ``float64`` dtype and
    gets the original binary value exactly.
    """

    mantissa, separator, exponent = format(value, ".17e").partition("e")
    if not separator:
        raise ValueError(f"Cannot encode non-finite fixture float {value!r}.")
    split = mantissa.index(".") + 2
    return f"{mantissa[:split]}_{mantissa[split:]}e{exponent}"


def _write_root_csv(
    path: Path,
    table: pd.DataFrame,
    dtypes: Mapping[str, str],
) -> None:
    """Write a root entity table and prove exact generic-codec float decode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format=_csv_codec_float,
    )
    decoded = pd.read_csv(path).astype(dict(dtypes))
    for column in table.select_dtypes(include=["floating"]).columns:
        expected = table[column].to_numpy(copy=False)
        actual = decoded[column].to_numpy(dtype=expected.dtype, copy=False)
        if expected.tobytes() != actual.tobytes():
            raise RuntimeError(
                f"Root CSV codec changed binary values in {path.name}:{column}."
            )


def _write_tab(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _artifact(path: Path, *, table: str) -> dict[str, object]:
    return {
        "role": "frs_table",
        "table": table,
        "kind": "licensed_microdata",
        "format": "tab",
        "vintage": "synthetic-h2",
        "locator": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "runtime_sha256_required": True,
    }


def _stage_payload(stage: SourceStageSpec) -> dict[str, object]:
    return {
        "stage": stage.stage,
        "survey": stage.survey,
        "source": stage.source,
        "grain": stage.grain,
        "artifacts": [dict(value) for value in stage.artifacts],
        "operations": [
            {"kind": operation.kind, **dict(operation.parameters)}
            for operation in stage.operations
        ],
        "outputs": list(stage.outputs),
        "nonnegative_outputs": list(stage.nonnegative_outputs),
        "rewrites": list(stage.rewrites),
        "notes": stage.notes,
    }


def _replace_operation(
    stage: SourceStageSpec, kind: str, **overrides: object
) -> SourceStageSpec:
    operations: list[dict[str, object]] = []
    found = False
    for operation in stage.operations:
        payload = {"kind": operation.kind, **dict(operation.parameters)}
        if operation.kind == kind:
            payload.update(overrides)
            found = True
        operations.append(payload)
    if not found:
        raise ValueError(f"{stage.stage!r} has no operation {kind!r}.")
    return SourceStageSpec.from_mapping(
        {**_stage_payload(stage), "operations": operations}
    )


def _frs_tables() -> dict[str, pd.DataFrame]:
    """Build 14 tiny FRS-shaped tabs with deterministic linked identities."""

    households: list[dict[str, object]] = []
    adults: list[dict[str, object]] = []
    children: list[dict[str, object]] = []
    benunits: list[dict[str, object]] = []
    accounts: list[dict[str, object]] = []
    benefits: list[dict[str, object]] = []
    jobs: list[dict[str, object]] = []
    pensions: list[dict[str, object]] = []
    penprov: list[dict[str, object]] = []
    oddjobs: list[dict[str, object]] = []
    maint: list[dict[str, object]] = []
    childcare: list[dict[str, object]] = []
    extchild: list[dict[str, object]] = []
    mortgages: list[dict[str, object]] = []
    for household_id in range(1, _ROOT_HOUSEHOLDS + 1):
        scotland = household_id % 4 == 0
        # Keep household composition independent of the region cadence so the
        # region-stratified SPI prior allocation changes weighted person mass.
        has_child = household_id % 5 == 0
        region = (8, 11, 12, 1)[household_id % 4]
        households.append(
            {
                "SERNUM": household_id,
                "GROSS4": 1.0 + household_id / 100.0,
                "GVTREGNO": region,
                "PTENTYP2": 5 + household_id % 2,
                "TYPEACC": 1 + household_id % 4,
                "BEDROOM6": 1 + household_id % 4,
                "CTANNUAL": 900.0 + household_id * 10.0,
                "CTBAND": 1 + household_id % 7,
                "CTREBAMT": float(household_id % 3),
                "ADULTH": 1,
                "CSEWAMT": "",
                "CWATAMTD": 0.0,
                "CWATAMT1": 4.0 if scotland else "",
                "CSEWAMT1": 5.0 if scotland else "",
                "WATSEWRT": 3.0,
                "NIRATLIA": 0.0,
                "RT2REBAM": 0.0,
                "HHRENT": 5.0,
                "SUBRENT": 0.0,
                "TENTYP2": 5,
                "MORTINT": 7.0,
                "STRUINS": 8.0,
                **{f"CHRGAMT{i}": float(i) for i in range(1, 10)},
            }
        )
        benunits.append(
            {
                "SERNUM": household_id,
                "BENUNIT": 1,
                "FAMTYPB2": 5,
                "DEPCHLDB": int(has_child),
                "TOTCAPB4": 100.0 + household_id,
            }
        )
        adult = {
            "SERNUM": household_id,
            "BENUNIT": 1,
            "PERSON": 1,
            "AGE80": 0,
            "AGE": 80 if household_id == 1 else 20 + household_id % 55,
            "SEX": 1 + household_id % 2,
            "TOTHOURS": 20 + household_id % 25,
            "HRPID": 1,
            "UPERSON": 1,
            "MARITAL": 1 + household_id % 3,
            "EMPSTATI": 1 + household_id % 8,
            "MJOBSECT": 1 + household_id % 2,
            "SIC": 10 + household_id % 80,
            "FTED": 2,
            "TYPEED2": 0,
            "EDUCQUAL": 1 + household_id % 18,
            "TRAIN": 10,
            "EMAAMT": 0.0,
            "CHEMAAMT": 0.0,
            "INEARNS": 100.0 + household_id,
            "SEINCAM2": float(household_id % 5),
            "MNTUS1": 2,
            "MNTUSAM1": 1.0,
            "MNTAMT1": 2.0,
            "MNTAMT2": 1.0,
            "CVPAY": 1.0,
            "ROYYR1": 2.0,
            "ROYYR2": 3.0,
            "ROYYR3": 4.0,
            "ROYYR4": 5.0,
            "ALLPAY2": 6.0,
            "ALLPAY3": 7.0,
            "ALLPAY4": 8.0,
            "CHAMTERN": 9.0,
            "CHAMTTST": 10.0,
            "APAMT": 11.0,
            "APDAMT": 12.0,
            "PAREAMT": 13.0,
            "REDAMT": 20.0,
            "SLREPAMT": 2.0,
            "SSPADJ": 1.0,
            "SMPADJ": 0.5,
            "TUBORR": 500.0,
            "ACCSSAMT": 1.0,
            "GRTDIR1": 2.0,
            "GRTDIR2": 3.0,
            "HEARTVAL": 5.0,
        }
        adults.append(adult)
        if has_child:
            children.append(
                {
                    "SERNUM": household_id,
                    "BENUNIT": 1,
                    "PERSON": 2,
                    "AGE80": 0,
                    "AGE": 8 + household_id % 8,
                    "SEX": 1 + household_id % 2,
                    "TOTHOURS": np.nan,
                    "HRPID": 0,
                    "UPERSON": 0,
                    "MARITAL": 2,
                    "FTED": 1,
                    "TYPEED2": 2,
                    "EDUCQUAL": 86,
                    "TRAIN": 9,
                    "EMAAMT": 0.0,
                    "CHEMAAMT": 1.0,
                    "FSMVAL": 3.0,
                    "FSFVVAL": 1.0,
                    "FSBVAL": 2.0,
                    "HEARTVAL": 4.0,
                }
            )
        accounts.extend(
            (
                {
                    "SERNUM": household_id,
                    "BENUNIT": 1,
                    "PERSON": 1,
                    "ACCOUNT": 21,
                    "ACCINT": 1.0,
                    "ACCTAX": 0,
                    "INVTAX": 0,
                },
                {
                    "SERNUM": household_id,
                    "BENUNIT": 1,
                    "PERSON": 1,
                    "ACCOUNT": 1,
                    "ACCINT": 2.0,
                    "ACCTAX": 1,
                    "INVTAX": 0,
                },
            )
        )
        benefits.extend(
            {
                "SERNUM": household_id,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": benefit,
                "VAR2": variant,
                "BENAMT": amount,
            }
            for benefit, variant, amount in (
                (14, 1, 2.0),
                (14, 2, 3.0),
                (16, 3, 4.0),
                (16, 4, 5.0),
                (5, 0, 1.0),
                (6, 0, 6.0),
                (3, 0, 7.0),
            )
        )
        jobs.append(
            {
                "SERNUM": household_id,
                "BENUNIT": 1,
                "PERSON": 1,
                "DEDUC1": 2.0,
                "SPNAMT": 3.0,
                "SALSAC": "1",
            }
        )
        pensions.append(
            {
                "SERNUM": household_id,
                "BENUNIT": 1,
                "PERSON": 1,
                "PENPAY": 10.0,
                "PTAMT": 2.0,
                "PTINC": 2,
                "POAMT": 3.0,
                "POINC": 2,
                "PENOTH": 0,
            }
        )
        penprov.append(
            {
                "SERNUM": household_id,
                "BENUNIT": 1,
                "PERSON": 1,
                "STEMPPEN": 5,
                "PENAMT": 4.0,
            }
        )
        oddjobs.append(
            {
                "SERNUM": household_id,
                "BENUNIT": 1,
                "PERSON": 1,
                "OJAMT": 4.0,
                "OJNOW": 1,
            }
        )
        maint.append(
            {
                "SERNUM": household_id,
                "BENUNIT": 1,
                "PERSON": 1,
                "MRUS": 2,
                "MRUAMT": 2.0,
                "MRAMT": 9.0,
            }
        )
        childcare.append(
            {
                "SERNUM": household_id,
                "BENUNIT": 1,
                "PERSON": 1,
                "CHAMT": 5.0,
                "COST": 1,
                "REGISTRD": 1,
            }
        )
        extchild.append({"SERNUM": household_id, "BENUNIT": 1, "NHHAMT": 2.0})
        mortgages.append(
            {
                "SERNUM": household_id,
                "RMORT": 1,
                "RMAMT": 120.0,
                "BORRAMT": 240.0,
                "MORTEND": 12.0,
            }
        )
    return {
        "accounts": pd.DataFrame(accounts),
        "adult": pd.DataFrame(adults),
        "benefits": pd.DataFrame(benefits),
        "benunit": pd.DataFrame(benunits),
        "child": pd.DataFrame(children),
        "chldcare": pd.DataFrame(childcare),
        "extchild": pd.DataFrame(extchild),
        "househol": pd.DataFrame(households),
        "job": pd.DataFrame(jobs),
        "maint": pd.DataFrame(maint),
        "mortgage": pd.DataFrame(mortgages),
        "oddjob": pd.DataFrame(oddjobs),
        "penprov": pd.DataFrame(penprov),
        "pension": pd.DataFrame(pensions),
    }


def _write_frs_raw(root: Path) -> dict[str, dict[str, object]]:
    tables = _frs_tables()
    if set(tables) != set(FRS_SPINE_TABLES):
        raise RuntimeError("Synthetic FRS table roster drifted from production.")
    artifacts: dict[str, dict[str, object]] = {}
    for table in sorted(tables):
        path = root / f"{table}.tab"
        _write_tab(path, tables[table])
        artifacts[table] = _artifact(path, table=table)
    return artifacts


def _was_donor() -> pd.DataFrame:
    rows = np.arange(_DONOR_ROWS, dtype=float)
    return pd.DataFrame(
        {
            "R8xshhwgt": 1.0 + rows % 7 / 10.0,
            "DVLUKValR8_sum": 10.0 + rows,
            "DVPropertyR8": 20_000.0 + rows * 500.0,
            "DVFESHARESR8_aggr": 1.0 + rows,
            "DVFShUKVR8_aggr": 3.0 + rows,
            "DVIISAVR8_aggR": 5.0 + rows,
            "DVCISAVR8_aggr": 7.0 + rows,
            "DVFCollVR8_aggr": 9.0 + rows,
            "totalpenr8_aggr": 100.0 + rows * 10.0,
            "dvvaldbt_scaper8_aggr": 40.0 + rows,
            "NumAdultR8": 1 + rows.astype(int) % 3,
            "NumCh18R8": rows.astype(int) % 3,
            "DVGIPPENR8_AGGR": 11.0 + rows,
            "DVGISER8_AGGR": 13.0 + rows,
            "DVGIINVR8_aggr": 15.0 + rows,
            "DVGIEMPR8_AGGR": 17.0 + rows,
            "HBedRmR8": 1 + rows.astype(int) % 5,
            "GORR8": np.take([8, 11, 12, 1], rows.astype(int) % 4),
            "DVPriRntR8": 1 + rows.astype(int) % 2,
            "CTAmtR8": 900.0 + rows * 10.0,
            "HFINWNTR8_Sum": -50.0 + rows * 4.0,
            "HFINWNTR8_exSLC_Sum": 20.0 + rows - rows % 5,
            "HMortGR8": np.where(
                np.isin(np.take([1, 2, 3, 4], rows.astype(int) % 4), [2, 3]),
                10_000.0 + rows * 100.0,
                0.0,
            ),
            "Ten1R8": np.take([1, 2, 3, 4], rows.astype(int) % 4),
            "HFINWR8_SUM": 30.0 + rows,
            "DVhvalueR8": 100_000.0 + rows * 2_000.0,
            "DVHseValR8_sum": 1_000.0 + rows * 50.0,
            "DVBlDValR8_sum": 3_000.0 + rows * 60.0,
            "DVTotinc_bhcR8": 20_000.0 + rows * 1_000.0,
            "DVSaValR8_aggr": 500.0 + rows * 20.0,
            "vcarnr8": rows.astype(int) % 4,
            "Tot_LosR8_aggr": 9_000.0 + rows * 10.0,
            "Tot_los_exc_SLCR8_aggr": 4_000.0 + rows * 5.0,
        }
    )


def _lcfs_donors() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = np.arange(_DONOR_ROWS, dtype=float)
    household: dict[str, object] = {
        "case": np.arange(1, _DONOR_ROWS + 1),
        "g018": 1 + rows.astype(int) % 3,
        "g019": rows.astype(int) % 3,
        "gorx": 1 + rows.astype(int) % 12,
        "p389p": 100.0 + rows * 10.0,
        "p344p": 150.0 + rows * 10.0,
        "weighta": 1.0 + rows % 7 / 10.0,
        "a122": 1 + rows.astype(int) % 8,
        "a121": 1 + rows.astype(int) % 8,
        "b226": 6.0 + rows % 3,
        "b489": 2.0 + rows % 5,
        "b490": 1.0 + rows % 4,
        "p537": 10.0 + rows,
    }
    for position, source in enumerate(
        (
            "p601",
            "p602",
            "p603",
            "p604",
            "p605",
            "p606",
            "p607",
            "p608",
            "p609",
            "p610",
            "p611",
            "p612",
            "c72211",
            "c72212",
            *BUS_FARE_LCFS_CODES,
        ),
        start=1,
    ):
        household[source] = position + rows / 10.0
    person = pd.DataFrame(
        {
            "case": np.arange(1, _DONOR_ROWS + 1),
            "b303p": 10.0 + rows,
            "b3262p": 1.0 + rows / 5.0,
            "p049p": 4.0 + rows / 4.0,
        }
    )
    return pd.DataFrame(household), person


def _etb_donor() -> pd.DataFrame:
    rows = np.arange(_DONOR_ROWS, dtype=float)
    return pd.DataFrame(
        {
            "year": np.where(rows % 2 == 0, 2023, 2024),
            "adults": 1 + rows.astype(int) % 3,
            "childs": rows.astype(int) % 3,
            "noretd": rows.astype(int) % 2,
            "disinc": 100.0 + rows * 10.0,
            "totvat": 10.0 + rows / 2.0,
            "expdis": 100.0 + rows * 2.0,
            "hhold_adj_weight": 1.0 + rows % 5 / 10.0,
            "educ": 10.0 + rows,
            "rail": 2.0 + rows / 5.0,
            "bussub": 1.0 + rows / 10.0,
            "primed": rows.astype(int) % 2,
            "secoed": (rows.astype(int) + 1) % 2,
            "furted": (rows.astype(int) // 2) % 2,
            "disliv": rows % 7,
            "pips": rows % 5,
        }
    )


def _spi_donor() -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for index in range(_DONOR_ROWS):
        row = {column: 0.0 for column in SPI_DONOR_REQUIRED_COLUMNS}
        row.update(
            {
                "SEX": float(1 + index % 2),
                "FACT": 1.0 + index % 5 / 10.0,
                "GORCODE": float((7, 10, 11, 1)[index % 4]),
                "AGERANGE": float(1 + index % 7),
                "PAY": 10_000.0 + index * 1_000.0,
                "EPB": float(index % 4) * 100.0,
                "EXPS": float(index % 3) * 50.0,
                "TAXTERM": float(index % 5) * 20.0,
                "PROFITS": float(index % 7) * 500.0,
                "CAPALL": float(index % 3) * 25.0,
                "LOSSBF": float(index % 4) * 10.0,
                "SRP": float(index % 6) * 100.0,
                "INCBBS": 100.0 + index * 5.0,
                "DIVIDENDS": 10.0 + index,
                "PENSION": float(index % 8) * 100.0,
                "INCPROP": float(index % 5) * 150.0,
                "OTHERINV": 5.0 + index / 2.0,
                "GIFTAID": 1.0 + index % 5,
                "GIFTINV": 1.0 + index % 3,
            }
        )
        employment = (
            max(row["PAY"] + row["EPB"] - row["EXPS"], 0.0)
            + row["INCPBEN"]
            + row["OSSBEN"]
            + row["TAXTERM"]
            + row["UBISJA"]
            + row["MOTHINC"]
        )
        self_employment = max(row["PROFITS"] - row["CAPALL"] - row["LOSSBF"], 0.0)
        row["TEI"] = (
            employment + row["OTHERINC"] + row["SRP"] + row["PENSION"] + self_employment
        )
        row["TII"] = row["OTHERINV"] + row["DIVIDENDS"] + row["INCPROP"] + row["INCBBS"]
        row["TI"] = row["TEI"] + row["TII"]
        rows.append(row)
    return pd.DataFrame(rows)


def _hmrc_income_targets(path: Path) -> HMRCIncomeTargetSet:
    upper_bounds = (*HMRC_SPI_INCOME_BAND_LOWER_BOUNDS[1:], None)
    targets = tuple(
        HMRCIncomeBandTargetRecord(
            name=f"synthetic/{component}/{measure}/{lower_bound}",
            component=component,
            measure=measure,
            unit="people" if measure == "count" else "GBP",
            value=1_000.0 if measure == "count" else 1_000_000.0,
            period=HMRC_SPI_BUILD_PERIOD,
            total_income_lower_bound=lower_bound,
            total_income_upper_bound=upper_bound,
        )
        for lower_bound, upper_bound in zip(
            HMRC_SPI_INCOME_BAND_LOWER_BOUNDS, upper_bounds, strict=True
        )
        for component in HMRC_SPI_INCOME_COMPONENTS
        for measure in ("count", "amount")
    )
    return HMRCIncomeTargetSet(
        source=HMRCIncomeSourceProvenance(
            local_path=path,
            sha256="0" * 64,
            publication_url="https://example.test/hmrc",
            ods_url="https://example.test/hmrc.ods",
            source_vintage="2023-24 synthetic",
            source_tax_year="2023-24",
            source_tax_year_start=2023,
            build_period=HMRC_SPI_BUILD_PERIOD,
            table_names=("Table_3_6", "Table_3_7"),
            size_bytes=0,
            mime_type="application/vnd.oasis.opendocument.spreadsheet",
        ),
        targets=targets,
    )


def _hmrc_target_payload(targets: HMRCIncomeTargetSet) -> dict[str, object]:
    return {
        "source": {
            **targets.source.__dict__,
            "local_path": str(targets.source.local_path),
        },
        "targets": [record.__dict__ for record in targets.targets],
    }


def _cgt_distribution() -> HMRCCapitalGainsJointDistribution:
    gain_bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    gain_uppers = (*gain_bounds[1:], None)
    cells: list[HMRCCapitalGainsCell] = []
    band_totals: list[HMRCCapitalGainsBandTotal] = []
    income_people = dict.fromkeys(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, 0.0)
    income_gains = dict.fromkeys(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, 0.0)
    for lower, upper in zip(gain_bounds, gain_uppers, strict=True):
        mean = lower * 2.0 if upper is None else lower + (upper - lower) * 0.4
        people = 100.0
        band_people = 0.0
        band_gains = 0.0
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
            gains = people * mean
            cells.append(HMRCCapitalGainsCell(lower, income_lower, people, gains))
            band_people += people
            band_gains += gains
            income_people[income_lower] += people
            income_gains[income_lower] += gains
        band_totals.append(HMRCCapitalGainsBandTotal(lower, band_people, band_gains))
    income_totals = tuple(
        HMRCCapitalGainsIncomeTotal(
            income_lower, income_people[income_lower], income_gains[income_lower]
        )
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    )
    return HMRCCapitalGainsJointDistribution(
        cells=tuple(cells),
        band_totals=tuple(band_totals),
        income_totals=income_totals,
        source=HMRCCapitalGainsSourceProvenance(
            local_path=Path("synthetic-cgt.ods"),
            sha256="synthetic",
            size_bytes=0,
            sheet_name="synthetic",
            source_vintage="2023-24",
            build_period="2024",
        ),
        total_individuals=sum(value.individuals or 0.0 for value in band_totals),
        total_gains=sum(value.gains for value in band_totals),
    )


def _cgt_distribution_payload(
    distribution: HMRCCapitalGainsJointDistribution,
) -> dict[str, object]:
    return {
        "cells": [value.__dict__ for value in distribution.cells],
        "band_totals": [value.__dict__ for value in distribution.band_totals],
        "income_totals": [value.__dict__ for value in distribution.income_totals],
        "source": {
            **distribution.source.__dict__,
            "local_path": str(distribution.source.local_path),
        },
        "total_individuals": distribution.total_individuals,
        "total_gains": distribution.total_gains,
    }


def _fixture_stages(
    frs_artifacts: Mapping[str, Mapping[str, object]],
) -> tuple[SourceStageSpec, ...]:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stages: list[SourceStageSpec] = []
    for committed in spec.sources.stages:
        if committed.stage in UK_SPINE_EXCLUSIONS:
            continue
        artifacts = [
            dict(frs_artifacts[str(artifact["table"])])
            if artifact.get("table") in frs_artifacts
            else dict(artifact)
            for artifact in committed.artifacts
        ]
        stage = SourceStageSpec.from_mapping(
            {**_stage_payload(committed), "artifacts": artifacts}
        )
        if stage.stage == "was_wealth":
            stage = _replace_operation(
                stage, "fit_weighted_qrf_chain", n_estimators=_QRF_ESTIMATORS
            )
        elif stage.stage == "lcfs_consumption":
            stage = _replace_operation(
                stage, "fit_weighted_qrf_chain", n_estimators=_QRF_ESTIMATORS
            )
            stage = _replace_operation(
                stage, "bridge_donor_column_via_qrf", n_estimators=_QRF_ESTIMATORS
            )
        elif stage.stage == "hmrc_spi_income_spine":
            stage = _replace_operation(
                stage,
                "fit_weighted_qrf_stage1",
                sample_size=_SPI_DONOR_SAMPLE_SIZE,
                n_estimators=_QRF_ESTIMATORS,
            )
            stage = _replace_operation(
                stage,
                "fit_weighted_qrf_stage2",
                n_estimators=_QRF_ESTIMATORS,
            )
        stages.append(stage)
    if len(stages) != UK_FIXTURE_STAGE_COUNT:
        raise RuntimeError(
            f"Expected {UK_FIXTURE_STAGE_COUNT} UK fixture stages, got {len(stages)}."
        )
    return tuple(stages)


def _normalization_markdown() -> str:
    lines = [
        "# UK spine parity string normalization",
        "",
        "The unchanged legacy transforms retain these 22 textual table columns as",
        "pandas `object`. The frozen graph dtype token `string` is specified by",
        'interface-freeze amendment 10 as pandas `StringDtype(storage="python")`.',
        "Before computing the legacy oracle's `uk_frame_content_identity` (live,",
        "in `legacy_oracle_identity`), exactly this audited surface is cast to that",
        "dtype. No values, row order, column order, weights, strata, mass records,",
        "or metadata change.",
        "",
    ]
    for entity, columns in _NORMALIZED_STRING_COLUMNS.items():
        lines.append(f"- `{entity}`: " + ", ".join(f"`{name}`" for name in columns))
    lines.append("")
    return "\n".join(lines)


def _normalize_legacy_strings(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    audited = {
        (entity, column)
        for entity, columns in _NORMALIZED_STRING_COLUMNS.items()
        for column in columns
    }
    legacy_object_strings = {
        (entity, column)
        for entity, table in tables.items()
        for column in table.columns
        if table[column].dtype == object
        and not table[column].dropna().empty
        and table[column].dropna().map(lambda value: isinstance(value, str)).all()
    }
    if legacy_object_strings != audited:
        raise RuntimeError(
            "Legacy object-string audit drifted: "
            f"expected {sorted(audited)}, observed {sorted(legacy_object_strings)}."
        )
    observed: list[tuple[str, str]] = []
    for entity, columns in _NORMALIZED_STRING_COLUMNS.items():
        for column in columns:
            if column not in tables[entity]:
                raise RuntimeError(
                    f"Audited normalization column {entity}.{column} is absent."
                )
            if tables[entity][column].dtype != object:
                raise RuntimeError(
                    f"Audited normalization column {entity}.{column} is "
                    f"{tables[entity][column].dtype}, not legacy object."
                )
            tables[entity][column] = tables[entity][column].astype(
                pd.StringDtype(storage="python")
            )
            observed.append((entity, column))
    expected_order = [
        (entity, column)
        for entity, columns in _NORMALIZED_STRING_COLUMNS.items()
        for column in columns
    ]
    if observed != expected_order or len(observed) != 22:
        raise RuntimeError("The legacy normalization audit is not exactly 22 cells.")
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata.copy(deep=True),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _write_root_source(root: Path, frame: Frame) -> None:
    graph = uk_spine_graph()
    declared = {
        (owned.entity, owned.column): owned.dtype
        for owned in graph.node("create_uk_frs").outputs
    }
    dtypes: dict[str, dict[str, str]] = {}
    for entity in frame.entities:
        table = frame.table(entity).copy(deep=True)
        dtypes[entity] = {}
        id_column = frame.schema.entity_id_column(entity)
        structural = {id_column}
        if entity == frame.schema.person_entity:
            structural.update(
                frame.schema.membership_column(group)
                for group in frame.schema.group_entities
            )
        for column in table.columns:
            dtype = declared.get((entity, column))
            if dtype is None and column not in structural:
                raise RuntimeError(
                    f"Root transform emitted undeclared cell {entity}.{column}."
                )
            dtypes[entity][column] = "int64" if dtype is None else dtype
        if entity == frame.schema.person_entity:
            table["__stratum__"] = frame.strata.astype(
                pd.StringDtype(storage="python")
            ).array
            dtypes[entity]["__stratum__"] = "string"
        _write_root_csv(root / f"{entity}.csv", table, dtypes[entity])
    _write_json(
        root / "schema.json",
        {
            "schema": {
                "person_entity": frame.schema.person_entity,
                "group_entities": list(frame.schema.group_entities),
                "links": [],
            },
            "dtypes": dtypes,
            "strata_column": "__stratum__",
            "weights": {
                "household": {"kind": frame.weights_for("household").kind.value}
            },
        },
    )
    _write_json(
        root / "weights.json",
        {
            "weights": {
                "household": {"values": frame.weights_for("household").values.tolist()}
            }
        },
    )


def _build_implementations(
    *,
    stages: Mapping[str, SourceStageSpec],
    raw_dir: Path,
    was: pd.DataFrame,
    lcfs_household: pd.DataFrame,
    lcfs_person: pd.DataFrame,
    etb: pd.DataFrame,
    spi_donor: pd.DataFrame,
    income_targets: HMRCIncomeTargetSet,
    cgt_distribution: HMRCCapitalGainsJointDistribution,
    cgt_parameters: UKCGTPolicyParameters,
) -> tuple[dict[str, object], dict[str, Frame]]:
    engine = PolicyEngineUKEngine()
    contract = load_uk_take_up_contract()
    root_capture: dict[str, Frame] = {}
    root_transform = UKFRSSpineStageTransform(raw_dir, stage=stages["frs_spine"])

    def capture_root(frame: Frame) -> Frame:
        result = root_transform(frame)
        # Snapshot the root: the next legacy stage (age_tail) rewrites the
        # person table in place, so an aliased capture would carry its ages.
        root_capture["frame"] = Frame(
            {
                entity: result.table(entity).copy(deep=True)
                for entity in result.entities
            },
            result.schema,
            {entity: result.weights_for(entity) for entity in result.weighted_entities},
            result.strata.copy(deep=True),
            mass_log=result.mass_log,
            metadata=dict(result.metadata),
        )
        return result

    implementations: dict[str, object] = {
        "frs_spine": capture_root,
        "frs_employment": UKFRSEmploymentStageTransform(
            raw_dir, stage=stages["frs_employment"]
        ),
        "frs_council_tax": UKFRSCouncilTaxStageTransform(
            raw_dir, stage=stages["frs_council_tax"]
        ),
        "frs_disability": UKFRSDisabilityStageTransform(stage=stages["frs_disability"]),
        "frs_education": UKFRSEducationStageTransform(
            raw_dir, stage=stages["frs_education"]
        ),
        "frs_legacy_proxies": UKFRSLegacyProxiesStageTransform(
            raw_dir,
            stage=stages["frs_legacy_proxies"],
            engine=engine,
        ),
        "frs_education_grant_split": UKFRSEducationGrantSplitStageTransform(
            stage=stages["frs_education_grant_split"], engine=engine
        ),
        "frs_take_up": UKFRSTakeUpStageTransform(
            contract=contract, stage=stages["frs_take_up"]
        ),
        "frs_person_draws": UKFRSPersonDrawsStageTransform(
            contract=contract, stage=stages["frs_person_draws"]
        ),
        "frs_household_draws": UKFRSHouseholdDrawsStageTransform(
            contract=contract, stage=stages["frs_household_draws"]
        ),
        "frs_brma": UKFRSBRMAStageTransform(stage=stages["frs_brma"], engine=engine),
        "was_wealth": UKWASWealthStageTransform(
            stage=stages["was_wealth"], engine=engine, donor=was
        ),
        "regional_property_uprating": UKRegionalPropertyUpratingStageTransform(
            stage=stages["regional_property_uprating"]
        ),
        "lcfs_consumption": UKLCFSConsumptionStageTransform(
            stage=stages["lcfs_consumption"],
            engine=engine,
            lcfs_household=lcfs_household,
            lcfs_person=lcfs_person,
            was_donor=was,
        ),
        "etb_vat": UKETBVATStageTransform(
            stage=stages["etb_vat"], engine=engine, donor=etb
        ),
        "etb_services": UKETBServicesStageTransform(
            stage=stages["etb_services"], engine=engine, donor=etb
        ),
        "frs_hmrc_spine_leaves": UKFRSHMRCSpineLeavesStageTransform(
            raw_dir, stage=stages["frs_hmrc_spine_leaves"]
        ),
        "spi_support_channel": UKSPISupportChannelStageTransform(
            stage=stages["spi_support_channel"],
            sample_fraction=_SPI_SAMPLE_FRACTION,
        ),
        "hmrc_spi_income_spine": UKSPIIncomeSpineStageTransform(
            raw_dir.parent / "spi_donor.csv",
            raw_dir.parent / "hmrc_income_targets.json",
            stage=stages["hmrc_spi_income_spine"],
            qrf_estimators=_QRF_ESTIMATORS,
            donor_sample_size=_SPI_DONOR_SAMPLE_SIZE,
            sampled_rung=True,
            donor_table=spi_donor,
            source_targets=income_targets,
        ),
        "uc_reporter_redraw": UKUCReporterRedrawStageTransform(
            stage=stages["uc_reporter_redraw"], engine=engine
        ),
        "uc_capital_coherence": UKUCCapitalCoherenceStageTransform(
            stage=stages["uc_capital_coherence"]
        ),
        "uc_deduction_attributes": UKUCDeductionAttributesStageTransform(
            stage=stages["uc_deduction_attributes"]
        ),
        "cgt_incidence_clone": UKCGTIncidenceCloneStageTransform(
            stage=stages["cgt_incidence_clone"]
        ),
        "cgt_band_donors": UKCGTBandDonorStageTransform(
            stage=stages["cgt_band_donors"]
        ),
        "hmrc_cgt_gains_spine": uk_cgt_spine_stage_transform(
            stages["hmrc_cgt_gains_spine"],
            "synthetic-cgt.ods",
            distribution=cgt_distribution,
            parameters=cgt_parameters,
        ),
        "salary_sacrifice": UKSalarySacrificeStageTransform(
            stage=stages["salary_sacrifice"]
        ),
        "student_loans": UKStudentLoansStageTransform(
            stage=stages["student_loans"], calibration_year=2025
        ),
        "age_tail": UKAgeTailStageTransform(stage=stages["age_tail"]),
    }
    return implementations, root_capture


def _run_legacy_plan(
    stages: Iterable[SourceStageSpec],
    implementations: Mapping[str, object],
) -> Frame:
    """Run the legacy 28-stage StagePlan oracle and return its final frame."""

    stages = tuple(stages)
    committed = load_country_spec("uk")
    fixture_spec = replace(
        committed,
        sources=SourceManifest(
            country="uk",
            version=1,
            policy="Hermetic real-transform H2 parity fixture.",
            stages=stages,
        ),
        geography_spine=None,
    )
    plan = country_stage_plan(fixture_spec, implementations)
    # Pandas 3 infers its transitional ``str`` dtype for text by default;
    # the legacy spine contract predates that switch and its physical text
    # surface is object. Preserve that legacy side of the comparison; the
    # one explicit interface normalization happens in
    # ``_normalize_legacy_strings``.
    with pd.option_context("future.infer_string", False):
        final, records = plan.run(uk_frs_spine_seed_frame())
    expected_stage_names = tuple(stage.stage for stage in stages)
    if tuple(record.stage for record in records) != expected_stage_names:
        raise RuntimeError(
            f"The legacy fixture StagePlan did not execute all {UK_FIXTURE_STAGE_COUNT} stages."
        )
    return final


def legacy_oracle_frame(fixture: Path) -> Frame:
    """The legacy oracle's final frame on the fixture, computed live.

    Rebuilds the same 28 transforms the graph's UK registry reconstructs from
    ``fixture/sources``, runs them through the legacy StagePlan in this
    process, root included, and applies the one string normalization the
    fixture documents. The content identity is a byte-exact fingerprint of
    every cell, and the root transform's household weights differ by one ulp
    between machines (2026-09-02: two of 135 fixture households on x86 versus
    the Mac that wrote the capture), so parity is asserted against this live
    frame, with the graph deriving its root the same way, never against a
    pinned string or the captured root.
    """

    from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs

    stages, implementations = fixture_stage_plan_inputs(fixture / "sources")
    return _normalize_legacy_strings(_run_legacy_plan(stages, implementations))


def legacy_oracle_identity(fixture: Path) -> str:
    """``uk_frame_content_identity`` of :func:`legacy_oracle_frame`."""

    return uk_frame_content_identity(legacy_oracle_frame(fixture))


def generate(output: Path) -> None:
    """Write all deterministic H2 artifacts beneath ``output``."""

    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    raw_dir = sources / "frs_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frs_artifacts = _write_frs_raw(raw_dir)
    stages = _fixture_stages(frs_artifacts)
    stage_map = {stage.stage: stage for stage in stages}

    was = _was_donor()
    lcfs_household, lcfs_person = _lcfs_donors()
    etb = _etb_donor()
    spi_donor = _spi_donor()
    _write_csv(sources / "was.csv", was)
    _write_csv(sources / "lcfs_household.csv", lcfs_household)
    _write_csv(sources / "lcfs_person.csv", lcfs_person)
    _write_csv(sources / "etb.csv", etb)
    _write_csv(sources / "spi_donor.csv", spi_donor)

    income_targets = _hmrc_income_targets(Path("synthetic-hmrc.ods"))
    cgt_distribution = _cgt_distribution()
    cgt_parameters = uk_cgt_policy_parameters("2024")
    _write_json(
        sources / "hmrc_income_targets.json", _hmrc_target_payload(income_targets)
    )
    _write_json(
        sources / "cgt_distribution.json",
        _cgt_distribution_payload(cgt_distribution),
    )

    implementations, root_capture = _build_implementations(
        stages=stage_map,
        raw_dir=raw_dir,
        was=was,
        lcfs_household=lcfs_household,
        lcfs_person=lcfs_person,
        etb=etb,
        spi_donor=spi_donor,
        income_targets=income_targets,
        cgt_distribution=cgt_distribution,
        cgt_parameters=cgt_parameters,
    )
    final = _run_legacy_plan(stages, implementations)
    try:
        root_frame = root_capture["frame"]
    except KeyError as error:
        raise RuntimeError("The FRS root transform did not execute.") from error
    _write_root_source(sources, root_frame)

    descriptor = {
        "schema_version": "uk-spine-parity-fixture.v1",
        "description": (
            "Data-only inputs for reconstructing the same 28 current UK stage "
            "transform classes used by the legacy StagePlan oracle."
        ),
        "stages": {stage.stage: _stage_payload(stage) for stage in stages},
        "config": {
            "spi_sample_fraction": _SPI_SAMPLE_FRACTION,
            "spi_donor_sample_size": _SPI_DONOR_SAMPLE_SIZE,
            "qrf_estimators": _QRF_ESTIMATORS,
            "student_loans_calibration_year": 2025,
        },
        "inputs": {
            "frs_raw": "frs_raw",
            "was": "was.csv",
            "lcfs_household": "lcfs_household.csv",
            "lcfs_person": "lcfs_person.csv",
            "etb": "etb.csv",
            "spi_donor": "spi_donor.csv",
            "hmrc_income_targets": "hmrc_income_targets.json",
            "cgt_distribution": "cgt_distribution.json",
        },
        "cgt_parameters": cgt_parameters.__dict__,
    }
    _write_json(sources / "fixture.json", descriptor)

    graph = uk_spine_graph()
    (output / "uk_spine.json").write_text(graph_to_json(graph) + "\n", encoding="utf-8")
    # Reported, never pinned: the identity is machine-specific (see
    # ``legacy_oracle_identity``), so the acceptance test computes it live.
    print(
        "legacy oracle identity on this machine: "
        + uk_frame_content_identity(_normalize_legacy_strings(final))
    )
    (output / "NORMALIZATION.md").write_text(
        _normalization_markdown(), encoding="utf-8"
    )
    (output / "PRODUCED_BY.txt").write_text(
        "tools/graph_uk_spine_fixture.py; current 28-transform legacy "
        "StagePlan oracle with parsed private-source seams. The acceptance "
        "test runs both sides from frs_raw in-process (root weights differ "
        "by one ulp between machines); the captured root tables serve the "
        "unbound registry's data-only CREATE path.\n",
        encoding="utf-8",
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    generate(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
