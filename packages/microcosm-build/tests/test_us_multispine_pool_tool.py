"""Small-fixture tests for the terminal US multispine pool build tool."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
import microcosm.build.us_runtime.multispine_pool as multispine_pool_module
import microcosm.build.us_runtime.post_transfer_calibration as post_transfer_calibration_runtime
import microcosm.build.us_runtime.stacked_spine as stacked_spine_module
from microcosm.build.gates import GateReport, GateResult
from microcosm.build.logbook import LOGBOOK_ROW_FIELDS, load_logbook_row
from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.spec_engine import LegacyPayloadMismatchError
from microcosm.build.us_runtime.acs_transfer import transfer_acs_inputs
from microcosm.build.us_runtime.acs_transfer_bank import (
    ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION,
)
from microcosm.build.us_runtime.multispine_pool import (
    MultispinePoolCheckpoint,
    PoolStageOutput,
)
from microcosm.build.us_runtime.operator_boundary import (
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
)
from microcosm.build.us_runtime.qbi_inputs import (
    US_QBI_OUTPUT_COLUMNS,
    bind_us_qbi_reconciliation_transition_authority,
    us_qbi_reconciliation_change_receipt,
    with_us_qbi_input_reconciliation,
)
from microcosm.build.us_runtime.stacked_spine import GapFillDirection
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
)
from microcosm.build.us_runtime.take_up_contract import (
    load_take_up_contract,
    take_up_contract_identity,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights, read_frame_table

_FIXTURE_SEED_PERSON_COLUMN = "takes_up_medicaid_if_eligible"


@pytest.fixture(autouse=True)
def _prime_worker_identity(prime_primary_qrf_worker_identity: None) -> None:
    """Share the real session attestation unless a test opts into live identity."""


@pytest.fixture(scope="module")
def pool_tool() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_multispine_pool.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_multispine_pool_fixture",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def release_tool() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release_pool_fixture",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_frame(
    *,
    measured_offset: float = 0.0,
    include_peridnum: bool = True,
) -> Frame:
    ids = np.asarray([1, 2], dtype=np.int64)
    person_data = {
        "person_id": ids,
        "person_household_id": ids,
        "person_tax_unit_id": ids,
        "person_spm_unit_id": ids,
        "person_family_id": ids,
        "person_marital_unit_id": ids,
        "A_AGE": np.asarray([30.0, 50.0]),
        "A_SEX": np.asarray([1, 2], dtype=np.int64),
        "source_year": np.asarray([2024, 2024], dtype=np.int64),
        "measured": np.asarray([1.0, 2.0]) + measured_offset,
    }
    if include_peridnum:
        person_data["PERIDNUM"] = np.asarray(["1", "2"], dtype=object)
    person = pd.DataFrame(person_data)
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([2.0, 2.0]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["fixture", "fixture"], dtype=object),
    )


def _many_household_source_frame(
    *,
    count: int = 100,
    measured_offset: float = 0.0,
    state_fips: str | None = None,
    puma: str | None = None,
) -> Frame:
    ids = np.arange(1, count + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "A_AGE": np.full(count, 40.0),
            "A_SEX": np.where(ids % 2, 1, 2).astype(np.int64),
            "source_year": np.full(count, 2024, dtype=np.int64),
            "measured": ids.astype(np.float64) + measured_offset,
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    if measured_offset:
        tables["household"]["TYPEHUGQ"] = 1
    if state_fips is not None:
        tables["household"]["state_fips"] = pd.Series(
            state_fips,
            index=tables["household"].index,
            dtype="string",
        )
    if puma is not None:
        tables["household"]["puma"] = pd.Series(
            puma,
            index=tables["household"].index,
            dtype="string",
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(count, 2.0), WeightKind.DESIGN)},
        pd.Series(["fixture"] * count, dtype=object),
    )


def _replace_person(
    frame: Frame,
    person: pd.DataFrame,
    *,
    preserve_metadata: bool = True,
) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata if preserve_metadata else None,
    )


def _fixture_qbi_stage_output(
    frame: Frame,
    receipt: Mapping[str, object],
) -> PoolStageOutput:
    """Attach a real, live-frame-bound QBI receipt to a tiny derive fixture."""

    person = frame.table("person").copy()
    if "age" not in person:
        person["age"] = pd.to_numeric(person["A_AGE"], errors="raise")
    if "SEMP" not in person:
        person["SEMP"] = 0.0
    if "self_employment_income_before_lsr" not in person:
        person["self_employment_income_before_lsr"] = 0.0
    if "non_qualified_dividend_income" not in person:
        person["non_qualified_dividend_income"] = 0.0
    for column in US_QBI_OUTPUT_COLUMNS:
        if column not in person:
            person[column] = 0.0
    before = _replace_person(frame, person)
    after = with_us_qbi_input_reconciliation(before)
    qbi_receipt = us_qbi_reconciliation_change_receipt(before, after)
    after = bind_us_qbi_reconciliation_transition_authority(after, qbi_receipt)
    return PoolStageOutput(
        after,
        {
            **dict(receipt),
            "qbi_input_reconciliation": qbi_receipt,
        },
        qbi_transition_authority_sha256=qbi_receipt["sha256"],
    )


def _with_fixture_pre_clone_strike_benefits(frame: Frame) -> Frame:
    """Fixture producer for one real pre-clone operator-owned target."""

    person = frame.table("person").copy()
    assert "strike_benefits" not in person
    channel = person[support_channel_column("person")].astype(str)
    person["strike_benefits"] = np.nan
    person.loc[channel.eq("asec"), "strike_benefits"] = 125.0
    return _replace_person(frame, person)


def _with_fixture_household_geography(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    household = tables["household"]
    household["puma"] = pd.Series("0600100", index=household.index, dtype="string")
    household["congressional_district_geoid"] = np.full(
        len(household), 601, dtype=np.int64
    )
    household["county_fips"] = pd.Series("06001", index=household.index, dtype="string")
    tables.update({link: frame.link(link) for link in frame.links})
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _fixture_geography_assignment_receipt(
    pool_tool: ModuleType,
    frame: Frame,
    *,
    gate: GateResult | None = None,
) -> dict[str, object]:
    household = frame.table("household")
    fixture_gate = gate or GateResult(
        name="us_puma_ladder",
        passed=True,
        details={"fixture": True},
    )
    return {
        "artifact_kind": "populace_us_stacked_household_geography_assignment",
        "schema_version": 1,
        "contract": pool_tool._stacked_geography_assignment_contract(),
        "pre_assignment_household_order": pool_tool._ordered_household_id_receipt(
            household
        ),
        "assigned_household_geography": (
            pool_tool._ordered_household_geography_receipt(household)
        ),
        "target_universe": (
            pool_tool._target_congressional_district_universe_receipt((601,))
        ),
        "output": {
            "household_rows": len(household),
            "positive_congressional_district_rows": len(household),
            "unique_congressional_district_values": 1,
        },
        "summary": {"applied": True, "household_rows": len(household)},
        "gate": GateReport((fixture_gate,)).to_manifest(),
    }


def _fixture_puma_ladder(pool_tool: ModuleType):
    puma = np.asarray([600_100], dtype=np.int64)
    population = np.asarray([100.0])
    return pool_tool.UsPumaLadder(
        puma=puma,
        puma_population=population,
        cd_overlap_puma=puma.copy(),
        cd_overlap_cd=np.asarray([601], dtype=np.int64),
        cd_overlap_population=population.copy(),
        county_overlap_puma=puma.copy(),
        county_overlap_county=np.asarray([6_001], dtype=np.int32),
        county_overlap_population=population.copy(),
        tract_overlap_puma=puma.copy(),
        tract_overlap_tract=np.asarray([6_001_000_100], dtype=np.int64),
        tract_overlap_population=population.copy(),
        metadata={
            "schema_version": 1,
            "kind": "us_puma_ladder",
            "puma_vintage": "2020_puma",
            "sampling_basis": "population",
            "layers": {
                "congressional_district": {"vintage": "119th_congress"},
                "county": {"vintage": "2020_census"},
                "tract": {"vintage": "2020_census"},
            },
        },
    )


def _semantic_string_columns(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in table.columns
        if isinstance(table[column].dtype, pd.StringDtype)
        or (
            pd.api.types.is_object_dtype(table[column].dtype)
            and pd.api.types.infer_dtype(table[column], skipna=True) == "string"
        )
    )


def _with_object_backed_strings(frame: Frame) -> Frame:
    tables = {}
    for entity in frame.entities:
        table = frame.table(entity).copy()
        for column in _semantic_string_columns(table):
            table[column] = table[column].astype(object)
        tables[entity] = table
    tables.update({link: frame.link(link) for link in frame.links})
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


class _MeanQRF:
    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(
        self,
        frame: Frame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: str,
    ) -> _MeanFitted:
        assert (
            weights == frame.resolve_weights(frame.column_entity(targets[0])).kind.value
        )
        table = frame.table(frame.column_entity(targets[0]))
        return _MeanFitted(
            {target: float(table[target].mean()) for target in targets},
            weights,
        )


class _MeanFitted:
    def __init__(self, means: dict[str, float], weight_kind: str) -> None:
        self.means = means
        self.weight_kind = weight_kind

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                target: np.full(len(frame), mean, dtype=np.float64)
                for target, mean in self.means.items()
            },
            index=frame.index,
        )


def _transfer_source_frame(targets: list[float]) -> Frame:
    frame = _source_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["fixture_transfer"] = np.asarray(targets, dtype=np.float64)
    tables["household"]["state_fips"] = np.asarray([6, 36], dtype=np.int64)
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _red_pool_result(
    pool_tool: ModuleType,
    tmp_path: Path,
    *,
    authenticated_qbi: bool = False,
):
    order: list[str] = []

    def stage(
        name: str,
        transform: Callable[[pd.DataFrame], None],
    ) -> Callable[[Frame], PoolStageOutput]:
        def apply(frame: Frame) -> PoolStageOutput:
            order.append(name)
            person = frame.table("person").copy()
            assert set(person[support_clone_index_column("person")].astype(int)) == {
                0,
                1,
            }
            assert set(person[support_channel_column("person")]) == {
                "asec",
                "acs",
            }
            transform(person)
            if name == "derive" and authenticated_qbi:
                return _fixture_qbi_stage_output(
                    _replace_person(frame, person),
                    {"fixture_stage": name},
                )
            receipt: dict[str, object] = {"fixture_stage": name}
            if name == "seed" and authenticated_qbi:
                receipt["programs"] = {
                    _FIXTURE_SEED_PERSON_COLUMN: {"entity": "person"},
                }
            return PoolStageOutput(
                _replace_person(frame, person),
                receipt,
            )

        return apply

    def transfer(person: pd.DataFrame) -> None:
        person["fixture_transfer"] = person["measured"]

    def derive(person: pd.DataFrame) -> None:
        person["fixture_derived"] = person["fixture_transfer"] + 1.0

    def seed(person: pd.DataFrame) -> None:
        column = _FIXTURE_SEED_PERSON_COLUMN if authenticated_qbi else "fixture_seed"
        person[column] = person["fixture_derived"] > 0.0

    def simulate(person: pd.DataFrame) -> None:
        channels = person[support_channel_column("person")]
        person["ssi"] = np.where(channels.eq("asec"), 1.0, 100.0)

    result = pool_tool.build_multispine_pool(
        _source_frame(),
        _source_frame(measured_offset=99.0),
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        impute=stage("impute", transfer),
        derive=stage("derive", derive),
        seed=stage("seed", seed),
        simulate=stage("simulate", simulate),
    )

    assert order == ["impute", "derive", "seed", "simulate"]
    assert not result.agreement_gate.passed
    assert not result.simulation_ready
    assert result.agreement_gate.name == "us_spine_agreement"
    assert result.agreement_gate.details["tolerances"] == {
        "incidence_ratio_bounds": [0.8, 1.25],
        "max_quantile_envelope_distance": 0.25,
        "max_categorical_total_variation_distance": 0.25,
    }
    assert "ssi" not in result.frame.table("person")
    return result


def _output_context(
    pool_tool: ModuleType,
    tmp_path: Path,
    *,
    authenticated_qbi: bool = True,
):
    result = _red_pool_result(
        pool_tool,
        tmp_path,
        authenticated_qbi=authenticated_qbi,
    )
    outputs = pool_tool._output_paths(tmp_path / "pool.h5")
    source_manifest = pool_tool.load_acs_source_manifest()
    verified_inputs = {}
    for index, role in enumerate(
        (
            "asec_raw_stage",
            "acs_household",
            "acs_person",
            "acs_rent_donor",
            "processed_puf",
            "puf_source_year",
        ),
        start=1,
    ):
        path = tmp_path / f"{role}.fixture"
        path.write_bytes(f"input-{index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        verified_inputs[role] = pool_tool._VerifiedInput(
            role=role,
            path=path,
            expected_sha256=digest,
            actual_sha256=digest,
            size_bytes=path.stat().st_size,
        )
    loaded = pool_tool._LoadedInputs(
        asec=_source_frame(),
        acs=_source_frame(measured_offset=99.0),
        acs_rent_donor=pd.DataFrame({"fixture": [1]}),
        puf_donor=pd.DataFrame({"RECID": [1]}),
        asec_raw_stage_checkpoint={"artifact": "fixture-raw-stage"},
        acs_build={"artifact": "fixture-unit-frame"},
        acs_native_inputs={"person": {"age": {"source": "fixture"}}},
        puf_donor_build={"artifact": "fixture-donor"},
    )
    return result, outputs, verified_inputs, source_manifest, loaded


def _checkpoint_fixture_store(
    pool_tool: ModuleType,
    root: Path,
    *,
    changed_role: str | None = None,
):
    verified_inputs = {}
    for index, role in enumerate(
        (
            "asec_raw_stage",
            "acs_household",
            "acs_person",
            "acs_rent_donor",
            "processed_puf",
            "puf_source_year",
        ),
        start=1,
    ):
        path = root.parent / f"{role}.checkpoint-fixture"
        if not path.exists():
            path.write_bytes(f"checkpoint-input-{index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual = "f" * 64 if role == changed_role else digest
        verified_inputs[role] = pool_tool._VerifiedInput(
            role=role,
            path=path,
            expected_sha256=actual,
            actual_sha256=actual,
            size_bytes=path.stat().st_size,
        )
    base_identity = pool_tool._pool_checkpoint_base_identity(
        verified_inputs,
        policyengine_us_version="fixture-engine-1",
    )
    return pool_tool._PoolStageCheckpointStore(
        root,
        base_identity=base_identity,
    )


def _checkpoint_fixture_input_receipts() -> dict[str, object]:
    return {
        "asec_raw_stage_checkpoint": {"artifact": "fixture-raw-stage"},
        "acs_pums_build": {"artifact": "fixture-unit-frame"},
        "acs_native_inputs": {"person": {"age": {"source": "fixture"}}},
        "puf_donor": {
            "rows": 0,
            "columns": [],
            "build_receipt": {"artifact": "fixture-donor"},
        },
    }


def _verified_inputs_fixture(pool_tool: ModuleType, root: Path):
    verified = {}
    for index, role in enumerate(
        (
            "asec_raw_stage",
            "acs_household",
            "acs_person",
            "acs_rent_donor",
            "processed_puf",
            "puf_source_year",
            "puma_ladder",
            "congressional_district_vintage_crosswalk",
        ),
        start=1,
    ):
        path = root / f"{role}.stacked-fixture"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"stacked-input-{index}".encode())
        digest = {
            "puma_ladder": pool_tool._STACKED_PUMA_LADDER_SHA256,
            "congressional_district_vintage_crosswalk": (
                pool_tool._STACKED_CD_CROSSWALK_SHA256
            ),
        }.get(role, hashlib.sha256(path.read_bytes()).hexdigest())
        verified[role] = pool_tool._VerifiedInput(
            role=role,
            path=path,
            expected_sha256=digest,
            actual_sha256=digest,
            size_bytes=path.stat().st_size,
        )
    return verified


def _run_checkpoint_fixture(
    pool_tool: ModuleType,
    tmp_path: Path,
    *,
    store,
    resume=None,
    target_bank_receipt: Mapping[str, object] | None = None,
    primary_qrf_manifest_path: Path | None = None,
    authenticated_qbi: bool = True,
    checkpoint_nullable_booleans: bool = False,
):
    order: list[str] = []

    def stage(
        name: str,
        transform: Callable[[pd.DataFrame], None],
    ) -> Callable[[Frame], PoolStageOutput]:
        def apply(frame: Frame) -> PoolStageOutput:
            order.append(name)
            person = frame.table("person").copy()
            transform(person)
            if name == "impute" and checkpoint_nullable_booleans:
                complete = np.resize(
                    np.asarray([True, False], dtype=np.bool_),
                    len(person),
                )
                missing = pd.array(complete, dtype="boolean")
                missing[1] = pd.NA
                person["is_female"] = pd.Series(
                    complete,
                    index=person.index,
                    dtype="boolean",
                )
                person["fixture_declared_boolean"] = pd.Series(
                    missing,
                    index=person.index,
                )
            receipt: dict[str, object] = {"fixture_stage": name}
            if name == "impute" and primary_qrf_manifest_path is not None:
                receipt = {
                    "primary_puf_qrf": {
                        "mode": "checkpoint_chain",
                        "checkpoint_manifest_path": str(
                            primary_qrf_manifest_path.resolve()
                        ),
                    }
                }
            if name == "impute" and target_bank_receipt is not None:
                receipt = {
                    "source_operator_chain": {"post_primary_completion": {}},
                    "primary_puf_qrf": {"fixture_manifest": True},
                    "puf_capital_gains_tail_transfer": {"fixture": True},
                    "acs_qrf_transfer": {
                        "target_families": {"person": {"fixture": ["target"]}},
                        "n_estimators": 100,
                        "max_targets_per_fit": 8,
                        "resolved_donor_channel": "puf_tax_detail",
                        "imputed_inputs": [],
                        "fit_records": [],
                        "deferred_inputs": [],
                        "target_bank": dict(target_bank_receipt),
                    },
                    "weights_audit": {"passed": True},
                }
            if name == "derive" and authenticated_qbi:
                return _fixture_qbi_stage_output(
                    _replace_person(frame, person),
                    receipt,
                )
            if name == "seed" and authenticated_qbi:
                receipt["programs"] = {
                    _FIXTURE_SEED_PERSON_COLUMN: {"entity": "person"},
                }
            return PoolStageOutput(_replace_person(frame, person), receipt)

        return apply

    result = pool_tool.build_multispine_pool(
        _source_frame() if resume is None else None,
        (
            _source_frame(measured_offset=99.0, include_peridnum=False)
            if resume is None
            else None
        ),
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        impute=stage(
            "impute",
            lambda person: person.__setitem__(
                "fixture_transfer",
                person["measured"],
            ),
        ),
        derive=stage(
            "derive",
            lambda person: person.__setitem__(
                "fixture_derived",
                person["fixture_transfer"] + 1.0,
            ),
        ),
        seed=stage(
            "seed",
            lambda person: person.__setitem__(
                (_FIXTURE_SEED_PERSON_COLUMN if authenticated_qbi else "fixture_seed"),
                person["fixture_derived"] > 0.0,
            ),
        ),
        simulate=stage(
            "simulate",
            lambda person: person.__setitem__(
                "ssi",
                np.where(
                    person[support_channel_column("person")].eq("asec"),
                    1.0,
                    100.0,
                ),
            ),
        ),
        checkpoint=store.write,
        resume=resume,
    )
    return result, order


def _run_production_impute_checkpoint_fixture(
    pool_tool: ModuleType,
    *,
    store,
    primary_qrf_checkpoint_dir: Path,
    acs_transfer_checkpoint_dir: Path,
    checkpoint_input_binding: Mapping[str, object],
):
    """Run the production impute closure while keeping later fixture stages tiny."""

    def stage(
        transform: Callable[[pd.DataFrame], None],
        *,
        reconcile_qbi: bool = False,
        seed_person_output: str | None = None,
    ) -> Callable[[Frame], PoolStageOutput]:
        def apply(frame: Frame) -> PoolStageOutput:
            person = frame.table("person").copy()
            transform(person)
            if reconcile_qbi:
                return _fixture_qbi_stage_output(
                    _replace_person(frame, person),
                    {"fixture_stage": True},
                )
            receipt: dict[str, object] = {"fixture_stage": True}
            if seed_person_output is not None:
                receipt["programs"] = {
                    seed_person_output: {"entity": "person"},
                }
            return PoolStageOutput(
                _replace_person(frame, person),
                receipt,
            )

        return apply

    return pool_tool.build_multispine_pool(
        _source_frame(),
        _source_frame(measured_offset=99.0),
        puf_donor=pd.DataFrame(),
        acs_rent_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=primary_qrf_checkpoint_dir,
        acs_transfer_checkpoint_dir=acs_transfer_checkpoint_dir,
        checkpoint_identity=store.base_identity,
        checkpoint_input_binding=checkpoint_input_binding,
        prepare_clone=stage(lambda _person: None),
        derive=stage(
            lambda person: person.__setitem__("fixture_derived", 1.0),
            reconcile_qbi=True,
        ),
        seed=stage(
            lambda person: person.__setitem__(_FIXTURE_SEED_PERSON_COLUMN, True),
            seed_person_output=_FIXTURE_SEED_PERSON_COLUMN,
        ),
        simulate=stage(
            lambda person: person.__setitem__(
                "ssi",
                np.where(
                    person[support_channel_column("person")].eq("asec"),
                    1.0,
                    100.0,
                ),
            ),
        ),
        checkpoint=store.write,
    )


def _stub_production_impute_kernels(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_bank_receipt: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Replace expensive kernels while retaining real routing and resume logic."""

    def initialize_primary_fixture(
        _frame: Frame,
        _donor: pd.DataFrame,
        checkpoint_dir: Path,
        **_kwargs,
    ) -> None:
        checkpoint_dir.mkdir(parents=True)
        pool_tool._atomic_write_json(
            pool_tool._primary_qrf_manifest_path(checkpoint_dir),
            {
                "artifact_kind": "fixture_primary_qrf_manifest",
                "target_order": ["fixture_target"],
            },
        )

    tail_receipt = {
        "tail_distribution_receipts": {
            "frame_after_stage": {
                "positive_mass_five_x_target_exceeded": True,
            }
        }
    }

    monkeypatch.setattr(
        pool_tool,
        "initialize_primary_puf_qrf_chain",
        initialize_primary_fixture,
    )
    monkeypatch.setattr(pool_tool, "run_primary_puf_qrf_chain", lambda *_args: None)
    monkeypatch.setattr(
        pool_tool,
        "finalize_primary_puf_qrf_chain",
        lambda frame, *_args, **_kwargs: (frame, "calibrated"),
    )
    monkeypatch.setattr(
        pool_tool,
        "transfer_puf_capital_gains_tail",
        lambda frame, *_args, **_kwargs: (frame, tail_receipt),
    )
    monkeypatch.setattr(
        pool_tool,
        "validate_puf_capital_gains_tail_manifest",
        lambda _receipt: None,
    )
    monkeypatch.setattr(
        pool_tool,
        "complete_multispine_source_inputs",
        lambda frame: SimpleNamespace(frame=frame, receipt={"fixture": True}),
    )
    monkeypatch.setattr(
        pool_tool,
        "pool_transfer_target_families",
        lambda: {"person": {"fixture": ("fixture_target",)}},
    )

    def transfer_fixture(frame: Frame, *_args, target_bank=None, **_kwargs):
        assert isinstance(target_bank, pool_tool.AcsTransferTargetBankStore)
        return SimpleNamespace(
            frame=frame,
            fit_records=(),
            resolved_donor_channel="fixture",
            imputed_inputs=(),
            deferred_inputs=(),
        )

    monkeypatch.setattr(pool_tool, "transfer_acs_inputs", transfer_fixture)
    if active_bank_receipt is not None:
        monkeypatch.setattr(
            pool_tool.AcsTransferTargetBankStore,
            "receipt",
            lambda _self: copy.deepcopy(active_bank_receipt["value"]),
        )
    return {
        "artifact_kind": "fixture_pool_input_binding",
        "schema_version": 1,
    }


def _target_bank_receipt_after_interruption(
    durable_target_index: int | None,
    *,
    total_targets: int = 9,
) -> dict[str, object]:
    targets: dict[str, object] = {}
    for index in range(total_targets):
        resumed = durable_target_index is not None and index <= durable_target_index
        source = "checkpoint" if resumed else "rebuilt"
        record: dict[str, object] = {
            "source": source,
            "descriptor": {
                "target_index": index,
                "total_targets": total_targets,
                "model_target": f"target_{index}",
            },
            "load_status": "resumed" if resumed else "missing",
            "path": f"/fixture/targets/{index:03d}__target_{index}.h5",
            "checkpoint_sha256": f"{index:064x}",
            "size_bytes": 1_000 + index,
        }
        if not resumed:
            record["write_status"] = "rebuilt"
            record["write_seconds"] = index + 0.25
        targets[str(index)] = record
    return {
        "artifact_kind": "populace_us_multispine_acs_transfer_target_bank_provenance",
        "schema_version": 1,
        "materializer_version": ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION,
        "root": "/fixture/acs-transfer",
        "identity": {"fixture": "identity"},
        "identity_sha256": "a" * 64,
        "targets": targets,
    }


def _seed_stale_green_outputs(outputs) -> None:
    outputs.pool_h5.write_bytes(b"stale green h5")
    outputs.agreement_diagnostics.write_text(
        json.dumps(
            {
                "simulation_ready": True,
                "publication_run_id": "stale-run",
            }
        ),
        encoding="utf-8",
    )
    outputs.manifest.write_text(
        json.dumps(
            {
                "status": "simulation_ready",
                "simulation_ready": True,
                "publication_run_id": "stale-run",
            }
        ),
        encoding="utf-8",
    )


def _assert_publication_tombstone(
    pool_tool: ModuleType,
    outputs,
    *,
    publication_run_id: str,
) -> None:
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest == {
        "agreement_diagnostics": {
            "path": str(outputs.agreement_diagnostics.resolve()),
            "publication_run_id": publication_run_id,
        },
        "artifact_kind": "populace_us_multispine_pool_manifest",
        "message": "publication in progress",
        "pool_h5": {
            "artifact_kind": "populace_us_multispine_input_pool",
            "path": str(outputs.pool_h5.resolve()),
            "publication_run_id": publication_run_id,
        },
        "publication_run_id": publication_run_id,
        "schema_version": pool_tool._LEGACY_POOL_MANIFEST_SCHEMA_VERSION,
        "simulation_ready": False,
        "status": "publication_in_progress",
    }
    with pytest.raises(ValueError, match="not simulation-ready"):
        pool_tool.load_simulation_ready_us_multispine_pool_manifest(outputs.manifest)


def _stacked_main_argv(
    tmp_path: Path,
    *,
    predecessor: str | None = None,
) -> list[str]:
    arguments: list[str] = []
    for option in (
        "asec-raw-stage-h5",
        "acs-household-zip",
        "acs-person-zip",
        "acs-rent-h5",
        "puf-h5",
        "puf-source-year-csv",
    ):
        arguments.extend([f"--{option}", str(tmp_path / option)])
        arguments.extend([f"--{option}-sha256", "1" * 64])
    arguments.extend(
        [
            "--puma-ladder",
            str(tmp_path / "puma-ladder"),
            "--puma-ladder-sha256",
            "39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8",
            "--congressional-district-vintage-crosswalk",
            str(tmp_path / "congressional-district-vintage-crosswalk"),
            "--congressional-district-vintage-crosswalk-sha256",
            "c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec",
        ]
    )
    arguments.extend(
        [
            "--sample-fraction",
            "0.01",
            "--sample-seed",
            "578",
            "--clone-attachment-fraction",
            "1.0",
            "--clone-attachment-seed",
            "579",
            "--out",
            str(tmp_path / "stacked-pool.h5"),
        ]
    )
    if predecessor is not None:
        arguments.extend(["--logbook-prev-row-digest", predecessor])
    return arguments


def _noncanonical_post_puf_authority_receipt() -> dict[str, object]:
    surface = {"person": {"model_required_boolean": ("is_pregnant",)}}
    test_authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
        post_puf_transfer_surface=surface,
    )
    return stacked_spine_module._authority_receipt(test_authority)


def _canonical_late_calibration_owner_receipt(
    spec: post_transfer_calibration_runtime.PostTransferCalibrationSpec,
    *,
    frame: Frame | None = None,
) -> dict[str, object]:
    if frame is not None:
        table = frame.table(spec.entity)
        channel = table[support_channel_column(spec.entity)].astype(str)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(spec.entity)],
            errors="raise",
        )
        reference = (channel.eq("asec") & clone_index.eq(0)).to_numpy(dtype=bool)
        recipient = (channel.eq("acs") & clone_index.eq(0)).to_numpy(dtype=bool)
        constrained = spec.special_constraint != "none"
        application = post_transfer_calibration_runtime.apply_post_transfer_calibration(
            frame,
            entity=spec.entity,
            family=spec.family,
            target=spec.target,
            reference_rows=reference,
            recipient_rows=recipient,
            mutable_rows=recipient,
            allowed_carrier_rows=recipient if constrained else None,
            addition_candidate_rows=recipient if constrained else None,
        )
        before_values = table[spec.target].to_numpy(copy=False)
        after_values = application.frame.table(spec.entity)[spec.target].to_numpy(
            copy=False
        )
        if not np.array_equal(
            before_values.view(np.uint64),
            after_values.view(np.uint64),
        ):
            raise AssertionError(
                f"Live calibration fixture unexpectedly changed {spec.key}."
            )
        calibration = application.receipt
        scope = calibration["scope"]
        constraint: dict[str, object] = {"constraint": spec.special_constraint}
        if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
            constraint.update(
                {
                    "qualifying_mutable_rows": scope["allowed_carrier_rows"],
                    "one_per_empty_tax_unit_addition_candidates": scope[
                        "addition_candidate_rows"
                    ],
                }
            )
        elif (
            spec.special_constraint
            == "weeks_requires_positive_unemployment_compensation"
        ):
            constraint["positive_unemployment_mutable_rows"] = scope[
                "allowed_carrier_rows"
            ]
        owner: dict[str, object] = {
            "stage": "late_transfer",
            "reference_selection": "asec_origin_clone_0",
            "recipient_selection": "acs_origin_clone_0",
            "mutable_selection": "recipient_null_before_nonnull_after",
            "reference_rows": scope["reference_rows"],
            "recipient_rows": scope["recipient_rows"],
            "mutable_rows": scope["mutable_rows"],
            "constraint": constraint,
            "context_binding": (
                stacked_spine_module._post_transfer_calibration_context_binding(
                    frame,
                    application.frame,
                    entity=spec.entity,
                    target=spec.target,
                    reference_rows=reference,
                    recipient_rows=recipient,
                    mutable_rows=recipient,
                    allowed_carrier_rows=recipient,
                    addition_candidate_rows=recipient,
                )
            ),
            "calibration": calibration,
        }
        if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
            owner["post_reconciliation"] = {"status": "verified_no_op"}
        return owner

    values = np.asarray(
        [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    )
    weights = np.asarray([2.0, 3.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    reference = np.asarray([True] * 5 + [False] * 5)
    recipient = ~reference
    constrained = spec.special_constraint != "none"
    calibration_result = (
        post_transfer_calibration_runtime.calibrate_post_transfer_values(
            values,
            weights,
            np.arange(1, len(values) + 1),
            spec=spec,
            reference_rows=reference,
            recipient_rows=recipient,
            mutable_rows=recipient,
            allowed_carrier_rows=recipient if constrained else None,
            addition_candidate_rows=recipient if constrained else None,
        )
    )
    calibration = calibration_result.receipt
    scope = calibration["scope"]
    constraint: dict[str, object] = {"constraint": spec.special_constraint}
    if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
        constraint.update(
            {
                "qualifying_mutable_rows": scope["allowed_carrier_rows"],
                "one_per_empty_tax_unit_addition_candidates": scope[
                    "addition_candidate_rows"
                ],
            }
        )
    elif spec.special_constraint == "weeks_requires_positive_unemployment_compensation":
        constraint["positive_unemployment_mutable_rows"] = scope["allowed_carrier_rows"]
    owner: dict[str, object] = {
        "stage": "late_transfer",
        "reference_selection": "asec_origin_clone_0",
        "recipient_selection": "acs_origin_clone_0",
        "mutable_selection": "recipient_null_before_nonnull_after",
        "reference_rows": scope["reference_rows"],
        "recipient_rows": scope["recipient_rows"],
        "mutable_rows": scope["mutable_rows"],
        "constraint": constraint,
        "context_binding": {
            "scope": dict(scope),
            "weights_sha256": calibration["weights"]["sha256"],
            "live_output": {
                "reference_rows": int(reference.sum()),
                "recipient_rows": int(recipient.sum()),
                "reference_entity_ids_sha256": (
                    stacked_spine_module._post_transfer_entity_ids_sha256(
                        np.arange(1, len(values) + 1)[reference]
                    )
                ),
                "recipient_entity_ids_sha256": (
                    stacked_spine_module._post_transfer_entity_ids_sha256(
                        np.arange(1, len(values) + 1)[recipient]
                    )
                ),
                "reference_output_values_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        calibration_result.values[reference],
                        boundary="synthetic reference calibration output",
                    )
                ),
                "recipient_output_values_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        calibration_result.values[recipient],
                        boundary="synthetic recipient calibration output",
                    )
                ),
                "reference_weights_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        weights[reference],
                        boundary="synthetic reference calibration weights",
                    )
                ),
                "recipient_weights_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        weights[recipient],
                        boundary="synthetic recipient calibration weights",
                    )
                ),
            },
        },
        "calibration": calibration,
    }
    if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
        owner["post_reconciliation"] = {"status": "verified_no_op"}
    return owner


def _canonical_pregnancy_structural_receipt() -> dict[str, object]:
    policy = acs_transfer_module.acs_transfer_execution_contract_identity(
        targets=("is_pregnant",),
        derive_schedule_d=False,
    )["structural_target_policies"]["is_pregnant"]
    return {
        "policy_sha256": policy["sha256"],
        "source_person_key": "person_source_id",
        "source_persons_checked": 1,
        "physical_rows_checked": 1,
        "clone_rows_checked": 0,
        "donor_rows_checked": 1,
        "qrf_draw_source_persons": 0,
        "qrf_draw_rows": 0,
        "qrf_fanout_rows": 0,
        "preexisting_value_fanout_rows": 0,
        "ineligible_rows_assigned_false": 0,
        "donor_preexisting_domain_violation_rows": 0,
        "recipient_preexisting_domain_violation_rows": 0,
        "preexisting_clone_disagreement_source_persons": 0,
        "inconsistent_eligibility_source_persons": 0,
        "maximum_clones_per_source_person": 1,
        "final_incomplete_rows": 0,
        "final_domain_violation_rows": 0,
        "final_clone_disagreement_source_persons": 0,
        "status": "verified",
    }


def _canonical_late_transfer_receipt(
    pool_tool: ModuleType,
    *,
    authority: Mapping[str, object] | None = None,
    frame: Frame | None = None,
) -> dict[str, object]:
    canonical_family = {
        (entity, target): family
        for entity, families in (
            pool_tool.CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    groups: dict[str, object] = {}
    targets: dict[str, object] = {}
    late_specs = {
        spec.key: spec
        for spec in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values()
        if spec.stage == "late_transfer"
    }
    policy_sha256 = (
        post_transfer_calibration_runtime.post_transfer_calibration_policy_identity()[
            "sha256"
        ]
    )
    for group in pool_tool.CANONICAL_US_LATE_TRANSFER_GROUPS:
        group_targets = {
            f"{group.entity}/{group.family}/{target}": {
                "authorized_null_rows": 0,
                "imputed_rows": 0,
                "unmodeled_rows": 0,
                "residual_null_rows": 0,
            }
            for target in group.targets
        }
        pregnancy_key = f"{group.entity}/{group.family}/is_pregnant"
        if pregnancy_key in group_targets:
            group_targets[pregnancy_key]["structural_policy"] = (
                _canonical_pregnancy_structural_receipt()
            )
        calibrated_keys = sorted(set(group_targets) & set(late_specs))
        for key in calibrated_keys:
            group_targets[key]["post_transfer_calibration"] = (
                _canonical_late_calibration_owner_receipt(
                    late_specs[key],
                    frame=frame,
                )
            )
        groups[group.name] = {
            "producer": group.name,
            "ordered_targets": list(group.targets),
            "targets": group_targets,
            "post_transfer_calibration": {
                "policy_sha256": policy_sha256,
                "target_count": len(calibrated_keys),
                "targets": calibrated_keys,
            },
        }
        for target in group.targets:
            targets[
                f"{group.entity}/{canonical_family[(group.entity, target)]}/{target}"
            ] = group_targets[f"{group.entity}/{group.family}/{target}"]
    return {
        "fixture": "post_puf_transfer",
        "authority": dict(
            pool_tool.stacked_spine_authority_receipt()
            if authority is None
            else authority
        ),
        "producer_schedule": pool_tool._json_ready(
            pool_tool.us_late_producer_schedule_receipt()
        ),
        "producer_execution_order": [
            producer
            for producer in stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE.order
            if producer != stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
        ],
        "groups": groups,
        "targets": targets,
        "completion": {
            "status": "complete",
            "group_count": 19,
            "target_count": 70,
            "residual_null_rows": 0,
        },
    }


def _canonical_late_dag_receipt(
    pool_tool: ModuleType,
    *,
    authority: Mapping[str, object] | None = None,
    output_frame_sha256: str = "f" * 64,
    frame: Frame | None = None,
) -> dict[str, object]:
    schedule = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE
    schedule_receipt = pool_tool._json_ready(
        pool_tool.us_late_producer_schedule_receipt()
    )
    source_order = [
        producer.removeprefix("source:")
        for producer in schedule.order
        if producer.startswith("source:")
    ]
    cps_source_evidence = {"fixture": "shared_cps_source_evidence"}
    source_receipts = {
        operator: {
            "phase": "post_clone",
            "operator_order": [operator],
            "suboperators": [{"operator": operator, "order_index": 0}],
            "cps_source_evidence": cps_source_evidence,
        }
        for operator in source_order
    }
    source_completion = {
        "phase": "post_clone",
        "operator_order": source_order,
        "suboperators": [
            {"operator": operator, "order_index": index}
            for index, operator in enumerate(source_order)
        ],
        "cps_source_evidence": cps_source_evidence,
        "deferred_transfer_inputs": {
            "inputs": {
                column: {}
                for column in (
                    "bank_account_assets",
                    "bond_assets",
                    "stock_assets",
                )
            }
        },
    }
    transfer = _canonical_late_transfer_receipt(
        pool_tool,
        authority=authority,
        frame=frame,
    )
    input_frame_sha256 = "e" * 64
    previous_sha256 = stacked_spine_module._late_execution_genesis_sha256(
        producer_schedule_sha256=schedule_receipt["payload_sha256"],
        input_frame_sha256=input_frame_sha256,
    )
    execution = []
    for index, producer_name in enumerate(schedule.order):
        contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
            producer_name
        ]
        declared_inputs = []
        if contract.kind == "acs_earnings_universe":
            available = (
                stacked_spine_module._late_acs_earnings_universe_resource_receipts()
            )
        elif contract.kind == "primary_puf":
            available: dict[str, object] = (
                stacked_spine_module.stacked_late_primary_resource_receipts(
                    pd.DataFrame({"fixture_donor": [1.0]}),
                    primary_qrf_checkpoint_identity_sha256="c" * 64,
                    clone_attachment_fraction=1.0,
                    clone_attachment_seed=578,
                    seed=0,
                    n_estimators=100,
                    fit_records_enabled=True,
                    tail_bound_diagnostics_enabled=True,
                )
            )
        elif contract.kind == "post_clone_source":
            available = stacked_spine_module._late_source_resource_receipts(
                producer_name=producer_name,
            )
        elif contract.kind == "source_finalizer":
            available = {
                f"person.@source_receipt:{operator}": (
                    stacked_spine_module._late_available_input_receipt(
                        producer=producer_name,
                        entity="person",
                        column=f"@source_receipt:{operator}",
                        rows=1,
                        binding={
                            "resource_kind": "source_operator_receipt",
                            "schema_version": 1,
                            "source_operator": operator,
                            "source_receipt_sha256": (
                                stacked_spine_module._canonical_sha256(
                                    source_receipts[operator]
                                )
                            ),
                        },
                    )
                )
                for operator in source_order
            }
            available.update(
                stacked_spine_module._late_source_finalizer_resource_receipts()
            )
        elif contract.kind == "late_transfer":
            group = next(
                group
                for group in pool_tool.CANONICAL_US_LATE_TRANSFER_GROUPS
                if group.name == producer_name
            )
            available = stacked_spine_module._late_transfer_resource_receipts(
                group_name=group.name,
                entity=group.entity,
                family=group.family,
                targets=group.targets,
                seed=0,
                n_estimators=100,
                max_targets_per_fit=(
                    stacked_spine_module.DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
                ),
                target_bank=None,
            )
        else:
            available = {}
        for item in contract.inputs:
            alternatives = []
            for alternative in item.alternatives:
                physical_evidence = []
                for column in alternative:
                    is_virtual = (
                        column.column.startswith("@")
                        and column.column != "@resolved_weight"
                        and column.entity != "frame"
                    )
                    key = f"{column.entity}.{column.column}"
                    resource_receipt = available.get(key) if is_virtual else None
                    present = not is_virtual or resource_receipt is not None
                    physical_evidence.append(
                        {
                            "entity": column.entity,
                            "column": column.column,
                            "value_kind": column.value_kind,
                            "required_scope": item.required_scope,
                            "scope_rows": 1,
                            "missing_rows": 0 if present else 1,
                            "invalid_rows": 0,
                            "status": "present" if present else "absent",
                            "content_sha256": (
                                stacked_spine_module._canonical_sha256(resource_receipt)
                                if resource_receipt is not None
                                else "a" * 64
                            ),
                            **(
                                {"weight_kind": "household_weight"}
                                if column.column == "@resolved_weight"
                                else {}
                            ),
                        }
                    )
                alternatives.append(physical_evidence)
            evidence = {"alternatives": alternatives}
            evidence["sha256"] = stacked_spine_module._canonical_sha256(evidence)
            declared_inputs.append(
                {
                    "entity": item.entity,
                    "column": item.column,
                    "required_scope": item.required_scope,
                    "producing_stage": item.producing_stage,
                    "unfilled_rows": 0,
                    "invalid_rows": 0,
                    "evidence": evidence,
                }
            )
        if contract.kind == "acs_earnings_universe":
            producer_receipt = {"fixture": "acs_earnings_universe"}
        elif contract.kind == "primary_puf":
            producer_receipt: Mapping[str, object] = {
                "fixture": "primary_puf",
                "primary_resource_receipts_sha256": (
                    stacked_spine_module._canonical_sha256(available)
                ),
            }
        elif contract.kind == "post_clone_source":
            producer_receipt = source_receipts[producer_name.removeprefix("source:")]
        elif contract.kind == "source_finalizer":
            producer_receipt = source_completion
        else:
            producer_receipt = transfer["groups"][producer_name]
        output_surface = [
            {
                "entity": output.entity,
                "column": output.column,
                "coverage_scope": output.coverage_scope,
                "status": "present",
                "content_sha256": (
                    stacked_spine_module._canonical_sha256(producer_receipt)
                    if output.column.startswith("@source_receipt:")
                    else "b" * 64
                ),
                **({} if output.entity == "frame" else {"scope_rows": 1}),
                **(
                    {"weight_kind": "household_weight"}
                    if output.column == "@resolved_weight"
                    else {}
                ),
            }
            for output in contract.outputs
        ]
        row: dict[str, object] = {
            "execution_index": index,
            "producer": producer_name,
            "kind": contract.kind,
            "declared_inputs": declared_inputs,
            "declared_absence_receipts": {},
            "available_input_receipts": available,
            "input_surface_sha256": stacked_spine_module._canonical_sha256(
                declared_inputs
            ),
            "output_surface": output_surface,
            "output_surface_sha256": stacked_spine_module._canonical_sha256(
                output_surface
            ),
            "producer_receipt": producer_receipt,
            "producer_receipt_sha256": stacked_spine_module._canonical_sha256(
                producer_receipt
            ),
            "previous_execution_sha256": previous_sha256,
            "status": "complete",
        }
        row["sha256"] = stacked_spine_module._canonical_sha256(row)
        previous_sha256 = row["sha256"]
        execution.append(row)
    receipt: dict[str, object] = {
        "version": stacked_spine_module.US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "producer_schedule": schedule_receipt,
        "input_frame_sha256": input_frame_sha256,
        "output_frame_sha256": output_frame_sha256,
        "execution_chain_sha256": previous_sha256,
        "execution": execution,
        "source_completion": source_completion,
        "post_puf_transfer": transfer,
    }
    receipt["sha256"] = stacked_spine_module._canonical_sha256(receipt)
    return receipt


def _canonical_late_impute_receipts(
    pool_tool: ModuleType,
    *,
    authority: Mapping[str, object] | None = None,
) -> dict[str, object]:
    dag = _canonical_late_dag_receipt(pool_tool, authority=authority)
    return {
        "source_operator_chain": {
            "late_dag_completion": dag["source_completion"],
        },
        "stacked_late_producer_dag": dag,
        "stacked_post_puf_transfer": dag["post_puf_transfer"],
    }


def _authorized_late_impute_fixture(
    pool_tool: ModuleType,
    frame: Frame,
    *,
    authority: Mapping[str, object] | None = None,
) -> tuple[Frame, dict[str, object], str]:
    """Bind one structurally signed synthetic DAG proof to a live fixture frame."""

    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    for entity, table in tables.items():
        if support_channel_column(entity) not in table:
            table[support_channel_column(entity)] = np.resize(
                np.asarray(["asec", "acs"], dtype=object),
                len(table),
            )
        if support_clone_index_column(entity) not in table:
            table[support_clone_index_column(entity)] = np.zeros(
                len(table), dtype=np.int64
            )
    for (
        spec
    ) in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values():
        if spec.stage == "late_transfer" and spec.target not in tables[spec.entity]:
            tables[spec.entity][spec.target] = np.ones(
                len(tables[spec.entity]), dtype=np.float64
            )
    person = tables["person"]
    person["unemployment_compensation"] = np.ones(len(person), dtype=np.float64)
    person["is_incapable_of_self_care"] = pd.Series(
        True,
        index=person.index,
        dtype="boolean",
    )
    person["tax_unit_role_input"] = pd.Series(
        "DEPENDENT",
        index=person.index,
        dtype="string",
    )
    tables.update({name: frame.link(name) for name in frame.links})
    frame = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    dag = _canonical_late_dag_receipt(
        pool_tool,
        authority=authority,
        output_frame_sha256=stacked_spine_module._late_frame_content_sha256(frame),
        frame=frame,
    )
    authorized, transition_authority_sha256 = (
        stacked_spine_module._bind_late_producer_transition_authority(frame, dag)
    )
    return (
        authorized,
        {
            "source_operator_chain": {
                "late_dag_completion": dag["source_completion"],
            },
            "stacked_late_producer_dag": dag,
            "stacked_post_puf_transfer": dag["post_puf_transfer"],
        },
        transition_authority_sha256,
    )


def _install_stacked_entrypoint_stubs(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    terminal: str,
    post_puf_authority: Mapping[str, object] | None = None,
    real_geography_assignment: bool = False,
) -> tuple[list[str], int]:
    order: list[str] = []
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    source_manifest = pool_tool.load_acs_source_manifest()
    puf_donor = pd.DataFrame({"fixture": np.arange(7)})
    loaded = pool_tool._LoadedInputs(
        asec=_many_household_source_frame(
            state_fips="06" if real_geography_assignment else None,
        ),
        acs=_many_household_source_frame(
            measured_offset=1_000.0,
            state_fips="06" if real_geography_assignment else None,
            puma="0600100" if real_geography_assignment else None,
        ),
        acs_rent_donor=pd.DataFrame({"fixture": [1.0]}),
        puf_donor=puf_donor,
        asec_raw_stage_checkpoint={"artifact": "fixture-raw-stage"},
        acs_build={"artifact": "fixture-acs-build"},
        acs_native_inputs={},
        puf_donor_build={"artifact": "fixture-puf-build"},
    )
    monkeypatch.setattr(
        pool_tool,
        "_verify_inputs",
        lambda _args, _outputs: (verified, source_manifest),
    )
    monkeypatch.setattr(
        pool_tool,
        "_load_inputs",
        lambda _args, *, acs_source_manifest: loaded,
    )
    monkeypatch.setattr(
        pool_tool,
        "load_acs_2022_rent_donor",
        lambda _path: loaded.acs_rent_donor,
    )
    monkeypatch.setattr(
        pool_tool,
        "_load_puf_donor",
        lambda _args: (loaded.puf_donor, loaded.puf_donor_build),
    )
    monkeypatch.setattr(pool_tool, "_git_code_pin", lambda: "a" * 40)
    crosswalk = pd.DataFrame(
        {
            "source_geography_id": ["5001700US0601"],
            "target_geography_id": ["5001900US0601"],
            "weight": [1.0],
        }
    )
    monkeypatch.setattr(
        pool_tool,
        "load_congressional_district_vintage_crosswalk",
        lambda _path: crosswalk,
    )
    fixture_ladder = (
        _fixture_puma_ladder(pool_tool) if real_geography_assignment else object()
    )
    monkeypatch.setattr(
        pool_tool,
        "load_us_puma_ladder",
        lambda _path: fixture_ladder,
    )
    fixture_puma_gate = GateResult(
        name="us_puma_ladder",
        passed=True,
        details={"fixture": True},
    )
    monkeypatch.setattr(
        pool_tool,
        "us_puma_ladder_gate",
        lambda *_args, **_kwargs: fixture_puma_gate,
    )

    real_stack = pool_tool.assemble_stacked_spine

    def stack(*args, **kwargs):
        order.append("stack")
        return real_stack(*args, **kwargs)

    monkeypatch.setattr(pool_tool, "assemble_stacked_spine", stack)

    real_assign_geography = pool_tool._assign_stacked_household_geography

    def assign_geography(frame: Frame, **kwargs: object):
        order.append("geography")
        if real_geography_assignment:
            return real_assign_geography(frame, **kwargs)
        assigned = _with_fixture_household_geography(frame)
        receipt = _fixture_geography_assignment_receipt(
            pool_tool,
            assigned,
            gate=fixture_puma_gate,
        )
        return assigned, receipt, (601,)

    monkeypatch.setattr(
        pool_tool,
        "_assign_stacked_household_geography",
        assign_geography,
    )

    real_build_stacked_pool = pool_tool.build_stacked_pool

    def build_stacked_pool(*args, **kwargs):
        order.append("build_stacked_pool")
        return real_build_stacked_pool(*args, **kwargs)

    monkeypatch.setattr(pool_tool, "build_stacked_pool", build_stacked_pool)

    def fixture_pre_clone_source_chain(
        frame: Frame,
        *,
        phase: str,
        operator_names: tuple[str, ...],
        operators: Mapping[str, object],
        **_kwargs: object,
    ) -> PoolStageOutput:
        order.append("prepare")
        assert phase == "pre_clone"
        assert operator_names
        assert set(operator_names) == set(operators)
        return PoolStageOutput(
            _with_fixture_pre_clone_strike_benefits(frame),
            {"fixture": "pre_clone_source_chain"},
        )

    monkeypatch.setattr(
        multispine_pool_module,
        "_run_source_operator_chain",
        fixture_pre_clone_source_chain,
    )
    directions = (
        GapFillDirection(
            name="asec_survey_to_acs",
            recipient_channel="acs",
            donor_channel="asec",
            target_families={
                "person": {
                    "source_operator_cps_carried": ("strike_benefits",),
                }
            },
        ),
    )
    monkeypatch.setattr(pool_tool, "stacked_gap_fill_plan", lambda: directions)

    def gap_fill(frame: Frame, **kwargs):
        order.append("gap")
        assert set(kwargs["target_banks"]) == {"asec_survey_to_acs"}
        counts = stacked_spine_module._verify_gap_fill_activation_authority(
            frame,
            direction=directions[0],
        )
        assert counts == {
            ("person", "strike_benefits"): {
                "authorized_null_rows": 1,
                "recipient_rows": 1,
                "donor_rows": 1,
            }
        }
        return SimpleNamespace(
            frame=frame,
            receipt={"fixture": "gap"},
            transfer_results={},
        )

    monkeypatch.setattr(pool_tool, "gap_fill_stacked_spine", gap_fill)
    monkeypatch.setattr(
        pool_tool,
        "weights_audit_gate",
        lambda _records: GateResult(name="fixture_weights", passed=True),
    )
    monkeypatch.setattr(
        pool_tool,
        "validate_puf_capital_gains_tail_manifest",
        lambda _manifest: None,
    )

    observed_primary_qrf_binding: dict[str, object] = {}

    def puf_pass(frame: Frame, donor: pd.DataFrame, **kwargs):
        order.append("puf")
        assert donor is puf_donor
        assert len(donor) == 7
        assert kwargs["clone_attachment_fraction"] == 1.0
        assert kwargs["clone_attachment_seed"] == 579
        primary_binding = kwargs["primary_qrf_input_binding"]
        stacked_spine_module._validate_stacked_late_primary_checkpoint_input_binding(
            primary_binding,
            boundary="tool wiring fixture",
        )
        observed_primary_qrf_binding.update(primary_binding)
        if terminal == "error":
            raise RuntimeError("fixture stacked error")
        checkpoint_dir = Path(kwargs["primary_qrf_checkpoint_dir"])
        pool_tool._atomic_write_json(
            checkpoint_dir / pool_tool.PRIMARY_QRF_MANIFEST_FILENAME,
            {"fixture": "primary-qrf"},
        )
        return SimpleNamespace(
            frame=frame,
            receipt={
                "primary_puf_qrf": {
                    "mode": "checkpoint_chain",
                    "resume_status": "initialized",
                },
                "puf_capital_gains_tail_transfer": {"fixture": "tail"},
                "tail_status": "applied",
                "primary_resource_receipts_sha256": (
                    stacked_spine_module._canonical_sha256(
                        primary_binding["primary_resource_receipts"]
                    )
                ),
            },
        )

    monkeypatch.setattr(pool_tool, "run_stacked_puf_pass", puf_pass)

    def late_producer_dag(frame: Frame, **kwargs: object):
        primary_puf_result = kwargs["primary_puf_producer"](frame)
        order.append("late_producer_dag")
        assert set(kwargs["primary_resource_receipts"]) == {
            "tax_unit.@puf_donor_tax_units",
            "tax_unit.@primary_qrf_checkpoint",
            "tax_unit.@primary_puf_execution_config",
        }
        assert (
            observed_primary_qrf_binding["primary_resource_receipts"]
            == kwargs["primary_resource_receipts"]
        )
        primary_config = kwargs["primary_resource_receipts"][
            "tax_unit.@primary_puf_execution_config"
        ]["binding"]
        assert primary_config["clone_attachment"] == {
            "fraction": 1.0,
            "seed": 579,
            "support_channels": ["asec", "puf_tax_detail"],
            "puf_clone_index": 1,
        }
        assert primary_config["qrf"]["seed"] == pool_tool.POOL_RANDOM_SEED
        assert (
            primary_config["qrf"]["n_estimators"] == pool_tool._PRIMARY_QRF_N_ESTIMATORS
        )
        target_banks = kwargs["target_banks"]
        assert isinstance(target_banks, Mapping)
        assert set(target_banks) == {
            group.name for group in pool_tool.CANONICAL_US_LATE_TRANSFER_GROUPS
        }
        schedule_sha256 = pool_tool.us_late_producer_schedule_receipt()[
            "payload_sha256"
        ]
        dag_sha256 = pool_tool.us_late_producer_schedule_receipt()["schedule_sha256"]
        for group in pool_tool.CANONICAL_US_LATE_TRANSFER_GROUPS:
            bank = target_banks[group.name]
            assert bank.root.parts[-3:] == (
                "late_producer_dag",
                group.entity,
                group.family,
            )
            assert bank._identity["late_producer_dag_sha256"] == dag_sha256
            assert bank._identity["late_producer_schedule_sha256"] == schedule_sha256
            assert bank._identity["late_producer"] == {
                "name": group.name,
                "entity": group.entity,
                "family": group.family,
                "ordered_targets": list(group.targets),
            }
        late_tables = {
            entity: primary_puf_result.frame.table(entity).copy(deep=True)
            for entity in primary_puf_result.frame.entities
        }
        for (
            spec
        ) in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values():
            if (
                spec.stage == "late_transfer"
                and spec.target not in late_tables[spec.entity]
            ):
                late_tables[spec.entity][spec.target] = np.ones(
                    len(late_tables[spec.entity]),
                    dtype=np.float64,
                )
        late_person = late_tables["person"]
        late_person["unemployment_compensation"] = np.ones(
            len(late_person),
            dtype=np.float64,
        )
        late_person["is_incapable_of_self_care"] = pd.Series(
            True,
            index=late_person.index,
            dtype="boolean",
        )
        late_person["tax_unit_role_input"] = pd.Series(
            "DEPENDENT",
            index=late_person.index,
            dtype="string",
        )
        late_tables.update(
            {
                name: primary_puf_result.frame.link(name)
                for name in primary_puf_result.frame.links
            }
        )
        late_frame = Frame(
            late_tables,
            primary_puf_result.frame.schema,
            {
                entity: primary_puf_result.frame.weights_for(entity)
                for entity in primary_puf_result.frame.weighted_entities
            },
            primary_puf_result.frame.strata,
            mass_log=primary_puf_result.frame.mass_log,
            metadata=primary_puf_result.frame.metadata,
        )
        dag_receipt = _canonical_late_dag_receipt(
            pool_tool,
            authority=post_puf_authority,
            output_frame_sha256=stacked_spine_module._late_frame_content_sha256(
                late_frame
            ),
            frame=late_frame,
        )
        authorized_frame, transition_authority_sha256 = (
            stacked_spine_module._bind_late_producer_transition_authority(
                late_frame,
                dag_receipt,
            )
        )
        return SimpleNamespace(
            frame=authorized_frame,
            receipt=dag_receipt,
            primary_puf_result=primary_puf_result,
            source_completion_receipt=dag_receipt["source_completion"],
            transfer_result=SimpleNamespace(fit_records=()),
            transition_authority_sha256=transition_authority_sha256,
        )

    monkeypatch.setattr(
        pool_tool,
        "run_stacked_late_producer_dag",
        late_producer_dag,
    )
    monkeypatch.setattr(
        pool_tool,
        "assert_stacked_tail_cells_preserved",
        lambda _frame, _manifest: {"passed": True},
    )

    def tail_prepare(frame: Frame):
        order.append("tail_prepare")
        return frame, {"fixture": "tail_prepare"}

    monkeypatch.setattr(pool_tool, "prepare_stacked_tail_derivation", tail_prepare)

    def identity_stage(name: str):
        def stage(frame: Frame):
            order.append(name)
            if name == "derive":
                return _fixture_qbi_stage_output(frame, {"fixture": name})
            return PoolStageOutput(frame, {"fixture": name})

        return stage

    monkeypatch.setattr(
        pool_tool,
        "derive_multispine_pool_inputs",
        identity_stage("derive"),
    )
    monkeypatch.setattr(
        pool_tool,
        "seed_multispine_pool_inputs",
        identity_stage("seed"),
    )

    def simulate(frame: Frame):
        order.append("simulate")
        person = frame.table("person").copy()
        person["ssi"] = 0.0
        return PoolStageOutput(_replace_person(frame, person), {"fixture": "simulate"})

    monkeypatch.setattr(
        pool_tool,
        "materialize_multispine_agreement_outputs",
        simulate,
    )

    def completeness(
        _frame: Frame,
        *,
        tail_manifest: Mapping[str, object],
    ) -> GateResult:
        order.append("completeness")
        assert tail_manifest == {"fixture": "tail"}
        return GateResult(name="fixture_completeness", passed=True)

    def battery(
        _frame: Frame,
        *,
        tail_manifest: Mapping[str, object],
    ) -> GateResult:
        order.append("battery")
        assert tail_manifest == {"fixture": "tail"}
        if terminal == "red":
            return GateResult(
                name="fixture_battery",
                passed=False,
                failures=("fixture terminal failure",),
            )
        return GateResult(name="fixture_battery", passed=True)

    monkeypatch.setattr(pool_tool, "stacked_completeness_gate", completeness)
    monkeypatch.setattr(pool_tool, "by_origin_battery", battery)
    real_publish = pool_tool._write_stacked_outputs

    def publish(*args, **kwargs):
        order.append("publish")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(pool_tool, "_write_stacked_outputs", publish)
    return order, len(puf_donor)


def test_stacked_operator_target_requires_preparation_before_activation_authority(
    pool_tool: ModuleType,
) -> None:
    stacked = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.01,
        sample_seed=578,
    ).frame
    direction = GapFillDirection(
        name="asec_survey_to_acs",
        recipient_channel="acs",
        donor_channel="asec",
        target_families={
            "person": {
                "source_operator_cps_carried": ("strike_benefits",),
            }
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            r"asec_survey_to_acs/person/source_operator_cps_carried/"
            r"strike_benefits: declared gap-fill target column is absent"
        ),
    ):
        stacked_spine_module._verify_gap_fill_activation_authority(
            stacked,
            direction=direction,
        )

    prepared = _with_fixture_pre_clone_strike_benefits(stacked)
    assert stacked_spine_module._verify_gap_fill_activation_authority(
        prepared,
        direction=direction,
    ) == {
        ("person", "strike_benefits"): {
            "authorized_null_rows": 1,
            "recipient_rows": 1,
            "donor_rows": 1,
        }
    }


@pytest.mark.parametrize(
    ("terminal", "expected_code", "disposition"),
    [
        ("success", 0, "iterating"),
        ("red", 1, "failed"),
        ("error", None, "failed"),
    ],
)
def test_stacked_tool_entrypoint_fixture_e2e_emits_one_logbook_row_at_every_terminal_state(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal: str,
    expected_code: int | None,
    disposition: str,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    """Exercise the real tool, stack assembly, orchestrator, and publication shell."""
    order, full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal=terminal,
    )
    if terminal == "error":
        with pytest.raises(RuntimeError, match="fixture stacked error"):
            pool_tool.main(_stacked_main_argv(tmp_path))
    else:
        assert pool_tool.main(_stacked_main_argv(tmp_path)) == expected_code

    rows = list((tmp_path / "logbook-spool").glob("*.json"))
    assert len(rows) == 1
    row = load_logbook_row(rows[0])
    assert frozenset(row.to_mapping()) == LOGBOOK_ROW_FIELDS
    assert row.disposition == disposition
    assert row.rung == "f001"
    assert row.seed == 578
    assert row.pipeline == "us-stacked-pool"
    assert len(row.input_pins_digest) == len(row.identity_digest) == 64
    assert "f001-s578-asec1-acs1" in row.build_id
    assert full_puf_rows == 7
    if terminal == "error":
        assert order == [
            "stack",
            "geography",
            "build_stacked_pool",
            "prepare",
            "gap",
            "puf",
        ]
        assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
        assert row.artifact_location is None
    else:
        assert order == [
            "stack",
            "geography",
            "build_stacked_pool",
            "prepare",
            "gap",
            "puf",
            "late_producer_dag",
            "tail_prepare",
            "derive",
            "seed",
            "simulate",
            "completeness",
            "battery",
            "publish",
        ]
        # Exported rows must never embed host-absolute paths; pytest tmp
        # directories live outside both the checkout and home on supported
        # platforms, so the reference lands on the stripped-absolute form.
        assert row.artifact_location == (
            "local://" + (tmp_path / "stacked-pool.h5").resolve().as_posix().lstrip("/")
        )
        expected_receipt_prefix = "local://" + (
            tmp_path / "logbook-receipts" / row.build_id
        ).resolve().as_posix().lstrip("/")
        assert all(
            verdict["receipt"].startswith(expected_receipt_prefix)
            for verdict in row.gate_verdicts.values()
        )
        manifest = json.loads(
            (tmp_path / "stacked-pool.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["schema_version"] == pool_tool.POOL_MANIFEST_SCHEMA_VERSION
        assert manifest["pipeline"] == "us-stacked-pool"
        assert manifest["pool_h5"]["materializer_version"] == (
            pool_tool.US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
        )
        with pd.HDFStore(manifest["pool_h5"]["path"], mode="r") as store:
            h5_metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
        assert h5_metadata["materializer_version"] == (
            pool_tool.US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
        )
        published_dag = manifest["stage_receipts"]["impute"][
            "stacked_late_producer_dag"
        ]
        published_transfer = published_dag["post_puf_transfer"]
        assert published_transfer["completion"] == {
            "status": "complete",
            "group_count": 19,
            "target_count": 70,
            "residual_null_rows": 0,
        }
        calibrated = [
            target["post_transfer_calibration"]
            for target in published_transfer["targets"].values()
            if "post_transfer_calibration" in target
        ]
        assert len(calibrated) == 7
        assert all(owner["context_binding"]["live_output"] for owner in calibrated)
        expected_late_authority_sha256 = (
            stacked_spine_module._late_producer_transition_authority_receipt(
                published_dag
            )["sha256"]
        )
        assert manifest["late_producer_transition_authority_sha256"] == (
            expected_late_authority_sha256
        )
        checkpoint_root = next(
            (tmp_path / "stacked-pool.checkpoints" / "stacked").iterdir()
        )
        for stage in ("transferred", "simulated"):
            checkpoint_path = checkpoint_root / f"{stage}.checkpoint.h5"
            checkpoint_manifest = json.loads(
                checkpoint_path.with_suffix(".manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            checkpoint_metadata = pool_tool.load_frame_checkpoint(
                checkpoint_path
            ).metadata
            assert (
                checkpoint_manifest["late_producer_transition_authority_sha256"]
                == expected_late_authority_sha256
            )
            assert (
                checkpoint_metadata["late_producer_transition_authority_sha256"]
                == expected_late_authority_sha256
            )
        assert manifest["operator_order"] == [
            "assemble_stacked_spine",
            "assign_us_puma_ladder",
            "prepare_multispine_source_inputs_for_clone",
            "gap_fill_stacked_spine",
            "run_stacked_late_producer_dag",
            "prepare_stacked_tail_derivation",
            "derive_multispine_pool_inputs",
            "seed_multispine_pool_inputs",
            "materialize_multispine_agreement_outputs",
            "stacked_completeness_gate",
            "by_origin_battery",
        ]
        assert manifest["sampling"] == {
            **manifest["sampling"],
            "sample_fraction": 0.01,
            "fraction_token": "f001",
            "sample_seed": 578,
            "realized_households": {"asec": 1, "acs": 1},
        }


def test_stacked_pool_fixed_h5_reaches_release_cd_vintage_preflight(
    pool_tool: ModuleType,
    release_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py", exc_type=ModuleNotFoundError)
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
        real_geography_assignment=True,
    )

    assert pool_tool.main(_stacked_main_argv(tmp_path)) == 0
    assert order.index("stack") < order.index("geography") < order.index("prepare")

    pool_h5 = tmp_path / "stacked-pool.h5"
    with h5py.File(pool_h5, mode="r") as h5:
        assert (
            release_tool._h5_attr_text(
                h5.attrs,
                release_tool.CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
            )
            == pool_tool._STACKED_CD_CROSSWALK_SHA256
        )
        assert (
            release_tool._h5_attr_text(
                h5.attrs,
                release_tool.CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
            )
            == pool_tool.CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
        )
    with pd.HDFStore(pool_h5, mode="r") as store:
        assert store.get_storer("household").format_type == "fixed"
        household = read_frame_table(store, "household")
    districts = pd.to_numeric(
        household["congressional_district_geoid"],
        errors="raise",
    )
    assert districts.gt(0).all()
    assert set(districts.astype(int)) == {601}

    release_tool._assert_cd_vintage_support_matches(
        pool_h5,
        {"sha256": pool_tool._STACKED_CD_CROSSWALK_SHA256},
    )
    preflight = release_tool._read_cd_vintage_support_provenance(pool_h5)
    assert preflight["household_congressional_district_geoid"] == {
        "exists": True,
        "table": "household",
        "column": "congressional_district_geoid",
        "rows": len(household),
        "positive_unique_count": 1,
    }


def test_stacked_config_authority_defaults_to_constants_without_loading_bundle(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = pool_tool._parser().parse_args(_stacked_main_argv(tmp_path))
    monkeypatch.setattr(
        pool_tool,
        "load_bundle",
        lambda _country: (_ for _ in ()).throw(
            AssertionError("the constants mode must not load a spec bundle")
        ),
    )

    assert args.config_authority == "constants"
    assert pool_tool._stacked_run_config(args) == {"config_authority": "constants"}


def test_constants_adapter_equals_live_constants_and_stays_out_of_identities(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    args = pool_tool._parser().parse_args(
        [
            *_stacked_main_argv(tmp_path),
            "--config-authority",
            "constants_adapter",
        ]
    )
    configured_identity = pool_tool._configured_stacked_identity(args)
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.01,
        sample_seed=578,
    )
    base_identity = pool_tool._stacked_checkpoint_base_identity(
        _verified_inputs_fixture(pool_tool, tmp_path / "pins"),
        stack_receipt=stack.receipt,
        sample_fraction=0.01,
        sample_seed=578,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=579,
        policyengine_us_version="fixture-engine",
    )
    checkpoint_identity = pool_tool._pool_checkpoint_stage_identity(
        base_identity,
        "assembled",
    )
    equality_call: dict[str, object] = {}
    real_assert_equal = pool_tool.assert_legacy_payload_equal

    def capture_equality(expected: object, actual: object) -> None:
        equality_call["expected"] = expected
        equality_call["actual"] = actual
        real_assert_equal(expected, actual)

    monkeypatch.setattr(
        pool_tool,
        "assert_legacy_payload_equal",
        capture_equality,
    )

    # This call performs the real bundle load, compilation, and field-complete
    # equality assertion against the live generation-0 constructors.
    run_config = pool_tool._stacked_run_config(args)
    assert (
        set(equality_call["expected"])
        == set(equality_call["actual"])
        == {
            "battery_contract",
            "gap_fill_plan",
            "gap_fill_producer_schedule_receipt",
            "late_producer_schedule_receipt",
            "overlap_ownership",
            "publication_release",
            "source_manifest",
            "spine_assembly",
            "spine_sampling",
            "stacked_authority_receipt",
            "stacked_checkpoint_static_components",
            "support_spine",
            "take_up_contract",
            "take_up_contract_identity",
        }
    )
    expected_gate = equality_call["expected"]
    assert isinstance(expected_gate, dict)
    assert expected_gate["publication_release"] == {
        "legacy_prefixes": ["populace-us-2024"],
        "rungs": ["f001", "f004", "f010", "f025", "f100"],
        "legacy_compiled_regexes": [pool_tool._STACKED_RELEASE_ID_PATTERN.pattern],
    }
    assert expected_gate["spine_sampling"] == {
        "channels": ["asec", "acs"],
        "fraction": {
            "default": 1.0,
            "rungs": [
                {
                    "fraction": fraction,
                    "token": token,
                    "percent_basis_points": basis_points,
                }
                for fraction, token, basis_points in (
                    (0.01, "f001", 100),
                    (0.04, "f004", 400),
                    (0.10, "f010", 1_000),
                    (0.25, "f025", 2_500),
                    (1.00, "f100", 10_000),
                )
            ],
        },
        "seed": {"default": 578},
        "exact_count_rule": "floor(fraction * eligible)",
    }
    assert expected_gate["spine_assembly"] == {
        "mass_anchor_channel": "asec",
        "household_mass_shares": {"asec": 0.5, "acs": 0.5},
    }
    assert run_config == {
        "config_authority": "constants_adapter",
        "spec_binding_status": "resolved",
        "spec_binding": {
            "attestation": "mirror-attested",
            "canonicalizer_version": 1,
            "country": "us",
            "schema_id": "country_spec",
            "schema_version": 1,
            "spec_sha256": "8b1546bc97b540afd525fe88f36a315de18b4d57bd9fab73035dacd8962e302e",
        },
    }

    def contains_spec_binding(value: object) -> bool:
        if isinstance(value, Mapping):
            return "spec_binding" in value or any(
                contains_spec_binding(item) for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_spec_binding(item) for item in value)
        return False

    assert not contains_spec_binding(stack.frame.metadata)
    assert not contains_spec_binding(configured_identity)
    assert not contains_spec_binding(base_identity)
    assert not contains_spec_binding(checkpoint_identity)


@pytest.mark.parametrize(
    ("mutation_path", "mutated_value", "expected_difference"),
    [
        (
            ("battery_contract", "gates", 1, "thresholds", "absolute"),
            0.5,
            "/battery_contract/gates/1/thresholds/absolute",
        ),
        (
            ("spine_assembly", "household_mass_shares", "asec"),
            0.49,
            "/spine_assembly/household_mass_shares/asec",
        ),
        (
            ("publication_release", "legacy_compiled_regexes", 0),
            "^mutated-release$",
            "/publication_release/legacy_compiled_regexes/0",
        ),
        (
            ("spine_sampling", "fraction", "default"),
            0.25,
            "/spine_sampling/fraction/default",
        ),
    ],
)
def test_constants_adapter_refuses_live_execution_surface_drift(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation_path: tuple[str | int, ...],
    mutated_value: object,
    expected_difference: str,
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    args = pool_tool._parser().parse_args(
        [
            *_stacked_main_argv(tmp_path),
            "--config-authority",
            "constants_adapter",
        ]
    )
    real_compile = pool_tool.compile_to_legacy_payload

    def compile_with_drift(resolved: object) -> dict[str, object]:
        payload = copy.deepcopy(real_compile(resolved))
        target: object = payload
        for token in mutation_path[:-1]:
            if isinstance(token, int):
                assert isinstance(target, list)
                target = target[token]
            else:
                assert isinstance(target, dict)
                target = target[token]
        leaf = mutation_path[-1]
        if isinstance(leaf, int):
            assert isinstance(target, list)
            target[leaf] = mutated_value
        else:
            assert isinstance(target, dict)
            target[leaf] = mutated_value
        return payload

    monkeypatch.setattr(pool_tool, "compile_to_legacy_payload", compile_with_drift)

    with pytest.raises(LegacyPayloadMismatchError) as failure:
        pool_tool._stacked_run_config(args)
    assert expected_difference in {
        difference.path for difference in failure.value.differences
    }


@pytest.mark.parametrize(
    ("failure_stage", "expected_status"),
    [
        ("preflight", "resolution_pending"),
        ("resolution", "resolution_failed"),
    ],
)
def test_constants_adapter_failure_receipt_preserves_requested_resolution_state(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
    expected_status: str,
) -> None:
    _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    if failure_stage == "preflight":
        monkeypatch.setattr(
            pool_tool,
            "_git_code_pin",
            lambda: (_ for _ in ()).throw(RuntimeError("fixture preflight failure")),
        )
        expected_error = "fixture preflight failure"
    else:
        monkeypatch.setattr(
            pool_tool,
            "_stacked_run_config",
            lambda _args: (_ for _ in ()).throw(
                RuntimeError("fixture adapter resolution failure")
            ),
        )
        expected_error = "fixture adapter resolution failure"

    with pytest.raises(RuntimeError, match=expected_error):
        pool_tool.main(
            [
                *_stacked_main_argv(tmp_path),
                "--config-authority",
                "constants_adapter",
            ]
        )

    error_path = next((tmp_path / "logbook-receipts").glob("*/error.json"))
    error_receipt = json.loads(error_path.read_text(encoding="utf-8"))
    assert error_receipt["run_config"] == {
        "config_authority": "constants_adapter",
        "spec_binding_status": expected_status,
    }
    assert "spec_binding" not in error_receipt["run_config"]
    assert "spec_sha256" not in json.dumps(error_receipt["run_config"])


def test_constants_adapter_post_resolution_failure_receipt_retains_binding(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="error",
    )
    binding = {
        "attestation": "mirror-attested",
        "canonicalizer_version": 1,
        "country": "us",
        "schema_id": "country_spec",
        "schema_version": 1,
        "spec_sha256": "f" * 64,
    }
    resolved_config = {
        "config_authority": "constants_adapter",
        "spec_binding_status": "resolved",
        "spec_binding": binding,
    }
    monkeypatch.setattr(
        pool_tool,
        "_stacked_run_config",
        lambda _args: resolved_config,
    )

    with pytest.raises(RuntimeError, match="fixture stacked error"):
        pool_tool.main(
            [
                *_stacked_main_argv(tmp_path),
                "--config-authority",
                "constants_adapter",
            ]
        )

    error_path = next((tmp_path / "logbook-receipts").glob("*/error.json"))
    error_receipt = json.loads(error_path.read_text(encoding="utf-8"))
    assert error_receipt["run_config"] == resolved_config


def test_constants_adapter_fixture_checkpoints_are_byte_identical_and_only_receipt_changes(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    fixed_binding = {
        "attestation": "mirror-attested",
        "canonicalizer_version": 1,
        "country": "us",
        "schema_id": "country_spec",
        "schema_version": 1,
        "spec_sha256": "5378bb9189aec96f50da22aac71e5bd2c3d919e9795f6ef2147e0bc9c739dd8e",
    }

    def run_fixture(root: Path, *, config_authority: str) -> dict[str, object]:
        root.mkdir()
        with monkeypatch.context() as patch:
            _install_stacked_entrypoint_stubs(
                pool_tool,
                patch,
                root,
                terminal="success",
            )
            patch.setattr(
                pool_tool,
                "_new_stacked_attempt_id",
                lambda **_kwargs: "populace-us-2024-stacked-attempt-fixture",
            )
            patch.setattr(
                pool_tool,
                "_new_stacked_release_id",
                lambda **_kwargs: (
                    "populace-us-2024-stacked-f001-s578-asec1-acs1-"
                    "20260817T000000Z-deadbeef"
                ),
            )
            patch.setattr(
                pool_tool,
                "_new_publication_run_id",
                lambda: "fixture-publication-run",
            )
            patch.setattr(
                pool_tool,
                "_stacked_run_config",
                lambda args: (
                    {"config_authority": "constants"}
                    if args.config_authority == "constants"
                    else {
                        "config_authority": "constants_adapter",
                        "spec_binding_status": "resolved",
                        "spec_binding": fixed_binding,
                    }
                ),
            )
            argv = _stacked_main_argv(root)
            if config_authority != "constants":
                argv.extend(["--config-authority", config_authority])
            assert pool_tool.main(argv) == 0

        checkpoint_files = sorted(
            (root / "stacked-pool.checkpoints").glob("stacked/*/*.checkpoint.h5")
        )
        assert [path.name for path in checkpoint_files] == [
            "assembled.checkpoint.h5",
            "simulated.checkpoint.h5",
            "transferred.checkpoint.h5",
        ]
        checkpoint_sha256 = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in checkpoint_files
        }
        receipt_path = next((root / "logbook-receipts").glob("*/terminal-gates.json"))
        return {
            "checkpoint_sha256": checkpoint_sha256,
            "receipt": json.loads(receipt_path.read_text(encoding="utf-8")),
        }

    constants = run_fixture(tmp_path / "constants", config_authority="constants")
    adapter = run_fixture(
        tmp_path / "constants-adapter",
        config_authority="constants_adapter",
    )

    assert constants["checkpoint_sha256"] == adapter["checkpoint_sha256"]
    constants_receipt = dict(constants["receipt"])
    adapter_receipt = dict(adapter["receipt"])
    assert constants_receipt.pop("run_config") == {"config_authority": "constants"}
    assert adapter_receipt.pop("run_config") == {
        "config_authority": "constants_adapter",
        "spec_binding_status": "resolved",
        "spec_binding": fixed_binding,
    }
    assert constants_receipt == adapter_receipt


def test_stacked_entrypoint_rejects_noncanonical_post_puf_transfer_receipt(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    noncanonical = _noncanonical_post_puf_authority_receipt()
    assert noncanonical["authority_form"] == "NON-CANONICAL"
    assert noncanonical["production_manifest_permitted"] is False
    order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
        post_puf_authority=noncanonical,
    )

    with pytest.raises(
        ValueError,
        match=(
            "stacked cold-build late-producer DAG: non-canonical stacked "
            "authority is forbidden"
        ),
    ):
        pool_tool.main(_stacked_main_argv(tmp_path))

    assert order == [
        "stack",
        "geography",
        "build_stacked_pool",
        "prepare",
        "gap",
        "puf",
        "late_producer_dag",
    ]
    assert not (tmp_path / "stacked-pool.h5").exists()
    assert not (tmp_path / "stacked-pool.manifest.json").exists()


def test_stacked_checkpoint_emission_propagates_and_authenticates_late_authority(
    pool_tool: ModuleType,
) -> None:
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool, _source_frame()
    )
    captured: list[MultispinePoolCheckpoint] = []

    pool_tool._emit_stacked_checkpoint(
        captured.append,
        stage="transferred",
        frame=authorized,
        assembly_receipt={},
        stage_receipts={
            "geography_assignment": {"fixture": True},
            "impute": impute,
        },
        late_producer_transition_authority_sha256=(transition_authority_sha256),
    )

    assert len(captured) == 1
    assert captured[0].late_producer_transition_authority_sha256 == (
        transition_authority_sha256
    )
    with pytest.raises(
        ValueError,
        match="differs from the independently carried late-producer",
    ):
        pool_tool._emit_stacked_checkpoint(
            captured.append,
            stage="transferred",
            frame=authorized,
            assembly_receipt={},
            stage_receipts={
                "geography_assignment": {"fixture": True},
                "impute": impute,
            },
            late_producer_transition_authority_sha256="0" * 64,
        )
    assert len(captured) == 1


def test_stacked_publication_rejects_noncanonical_receipt_before_any_write(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    noncanonical = _noncanonical_post_puf_authority_receipt()
    outputs = pool_tool._stacked_output_paths(tmp_path / "stacked-pool.h5")
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool,
        _source_frame(),
        authority=noncanonical,
    )
    result = SimpleNamespace(
        frame=authorized,
        stage_receipts={"impute": impute},
        late_producer_transition_authority_sha256=transition_authority_sha256,
    )

    with pytest.raises(
        ValueError,
        match=(
            "stacked publication entry: non-canonical stacked authority is forbidden"
        ),
    ):
        pool_tool._write_stacked_outputs(
            result,
            outputs=outputs,
            verified_inputs={},
            acs_source_manifest=pool_tool.load_acs_source_manifest(),
            input_receipts={},
            checkpoint_provenance={},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_stacked_publication_rejects_forged_late_transition_authority(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    authorized, impute, _transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool, _source_frame()
    )
    outputs = pool_tool._stacked_output_paths(tmp_path / "stacked-pool.h5")
    result = SimpleNamespace(
        frame=authorized,
        stage_receipts={"impute": impute},
        late_producer_transition_authority_sha256="0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="differs from the independently carried late-producer",
    ):
        pool_tool._write_stacked_outputs(
            result,
            outputs=outputs,
            verified_inputs={},
            acs_source_manifest=pool_tool.load_acs_source_manifest(),
            input_receipts={},
            checkpoint_provenance={},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_late_dag_validator_rejects_forged_execution_row(
    pool_tool: ModuleType,
) -> None:
    receipt = _canonical_late_dag_receipt(pool_tool)
    receipt["execution"][1]["producer"] = "source:forged"

    with pytest.raises(
        ValueError,
        match=r"execution row 1 is misbound",
    ):
        pool_tool.validate_stacked_late_producer_receipt(
            receipt,
            boundary="forged execution regression",
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    (
        (
            "stripped_group_summary",
            "stripped or misbound calibration summary evidence",
        ),
        ("stripped_target_evidence", "calibration evidence is absent"),
        ("malformed_target_evidence", "calibration receipt digest is invalid"),
        ("deleted_target_receipt", "target surface is non-canonical"),
        ("extra_target_receipt", "target surface is non-canonical"),
        ("stripped_owner_count", "reference_rows evidence is misbound"),
        ("stripped_live_output", "live-output binding is absent"),
    ),
)
def test_late_transfer_validator_rejects_stripped_calibration_evidence(
    pool_tool: ModuleType,
    mutation: str,
    error_match: str,
) -> None:
    receipt = _canonical_late_transfer_receipt(pool_tool)
    stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
        receipt,
        boundary="canonical calibration evidence control",
    )
    forged = copy.deepcopy(receipt)
    group_receipt = next(
        group
        for group in forged["groups"].values()
        if group["post_transfer_calibration"]["target_count"] > 0
    )
    if mutation == "stripped_group_summary":
        group_receipt.pop("post_transfer_calibration")
    else:
        target_key = group_receipt["post_transfer_calibration"]["targets"][0]
        if mutation == "deleted_target_receipt":
            group_receipt["targets"].pop(target_key)
        elif mutation == "extra_target_receipt":
            group_receipt["targets"]["person/forged/extra_target"] = {}
        elif mutation == "stripped_owner_count":
            group_receipt["targets"][target_key]["post_transfer_calibration"].pop(
                "reference_rows"
            )
        elif mutation == "stripped_live_output":
            group_receipt["targets"][target_key]["post_transfer_calibration"][
                "context_binding"
            ].pop("live_output")
        elif mutation == "stripped_target_evidence":
            target_receipt = group_receipt["targets"][target_key]
            target_receipt.pop("post_transfer_calibration")
        else:
            target_receipt = group_receipt["targets"][target_key]
            target_receipt["post_transfer_calibration"]["calibration"]["sha256"] = (
                "0" * 64
            )

    with pytest.raises(ValueError, match=error_match):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            forged,
            boundary=f"{mutation} regression",
        )


def test_late_transfer_validator_rejects_forged_pregnancy_policy(
    pool_tool: ModuleType,
) -> None:
    receipt = _canonical_late_transfer_receipt(pool_tool)
    stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
        receipt,
        boundary="canonical pregnancy policy control",
    )
    forged = copy.deepcopy(receipt)
    pregnancy = next(
        target_receipt
        for group in forged["groups"].values()
        for target_key, target_receipt in group["targets"].items()
        if target_key.endswith("/is_pregnant")
    )
    pregnancy["structural_policy"]["policy_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="pregnancy structural policy is invalid"):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            forged,
            boundary="forged pregnancy policy regression",
        )


@pytest.mark.parametrize(
    ("target", "constraint_column", "replacement", "error_match"),
    (
        (
            "weeks_unemployed",
            "unemployment_compensation",
            0.0,
            "positive weeks-unemployed carriers lack positive unemployment",
        ),
        (
            "pre_subsidy_care_expenses",
            "is_incapable_of_self_care",
            False,
            "live adult-care carriers violate qualifying-person",
        ),
    ),
)
def test_late_transfer_validator_recomputes_live_coupled_constraints(
    pool_tool: ModuleType,
    target: str,
    constraint_column: str,
    replacement: object,
    error_match: str,
) -> None:
    frame, impute, _transition = _authorized_late_impute_fixture(
        pool_tool,
        _source_frame(),
    )
    receipt = impute["stacked_post_puf_transfer"]
    stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
        receipt,
        boundary="live coupled-constraint control",
        frame=frame,
    )
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    person = tables["person"]
    recipient = person[support_channel_column("person")].astype(str).eq("acs")
    carrier = recipient & pd.to_numeric(person[target], errors="raise").gt(0.0)
    assert carrier.any()
    person.loc[person.index[carrier][0], constraint_column] = replacement
    tables.update({name: frame.link(name) for name in frame.links})
    corrupted = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )

    with pytest.raises(ValueError, match=error_match):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            receipt,
            boundary="live coupled-constraint mutation",
            frame=corrupted,
        )


def test_stacked_publication_rejects_forged_derived_order_before_any_write(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool, _source_frame()
    )
    impute["stacked_late_producer_dag"]["post_puf_transfer"][
        "producer_execution_order"
    ] = ["forged:wrong"]
    outputs = pool_tool._stacked_output_paths(tmp_path / "stacked-pool.h5")
    result = SimpleNamespace(
        frame=authorized,
        stage_receipts={"impute": impute},
        late_producer_transition_authority_sha256=transition_authority_sha256,
    )

    with pytest.raises(
        ValueError,
        match=r"execution order does not match the derived late-producer schedule",
    ):
        pool_tool._write_stacked_outputs(
            result,
            outputs=outputs,
            verified_inputs={},
            acs_source_manifest=pool_tool.load_acs_source_manifest(),
            input_receipts={},
            checkpoint_provenance={},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


@pytest.mark.parametrize(
    "failure",
    ("negative_seed", "oversized_seed", "invalid_output", "code_pin"),
)
def test_stacked_preflight_errors_emit_one_logbook_row(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    predecessor = "a" * 64
    arguments = _stacked_main_argv(tmp_path, predecessor=predecessor)
    if failure in {"negative_seed", "oversized_seed"}:
        seed_index = arguments.index("--sample-seed") + 1
        arguments[seed_index] = "-1" if failure == "negative_seed" else str(2**63)
        expected = "--sample-seed must be a non-negative signed 64-bit integer"
    elif failure == "invalid_output":
        output_index = arguments.index("--out") + 1
        arguments[output_index] = str(tmp_path / "not-an-h5.txt")
        expected = "--out must name an .h5 or .hdf5 file"
    else:
        monkeypatch.setattr(
            pool_tool,
            "_git_code_pin",
            lambda: (_ for _ in ()).throw(RuntimeError("fixture code pin failure")),
        )
        expected = "fixture code pin failure"

    with pytest.raises((RuntimeError, ValueError), match=expected):
        pool_tool.main(arguments)

    row_paths = list((tmp_path / "logbook-spool").glob("*.json"))
    assert len(row_paths) == 1
    row = load_logbook_row(row_paths[0])
    assert row.disposition == "failed"
    assert row.rung == "f001"
    assert row.prev_row_digest == predecessor
    assert row.code_pin == (
        "unresolved-local-git-code-pin" if failure == "code_pin" else "a" * 40
    )
    assert row.artifact_location is None
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    error_path = _receipt_file_from_reference(
        row.gate_verdicts["pipeline_error"]["receipt"]
    )
    assert error_path.is_file()
    assert error_path.parent == tmp_path / "logbook-receipts" / row.build_id
    assert order == []


def _receipt_file_from_reference(reference: str) -> Path:
    """Map one exported ``local://`` receipt reference back to a real file.

    Rows never embed host-absolute paths, so tests reconstruct the file
    location from the reference's anchor: ``~/`` means home, and the
    stripped-absolute fallback (the only form pytest tmp paths produce)
    re-roots at ``/``.
    """

    location = reference.split("#", maxsplit=1)[0]
    assert location.startswith("local://")
    tail = location.removeprefix("local://")
    assert not tail.startswith("/")
    if tail.startswith("~/"):
        return Path.home() / tail[2:]
    return Path("/") / tail


def test_publication_error_keeps_gate_receipts_and_does_not_claim_stale_h5(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    _order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    stale_h5 = tmp_path / "stacked-pool.h5"
    stale_h5.write_bytes(b"prior-build-artifact")

    def fail_publication(*_args, **_kwargs) -> None:
        raise RuntimeError("fixture publication failure")

    monkeypatch.setattr(pool_tool, "_write_stacked_outputs", fail_publication)

    with pytest.raises(RuntimeError, match="fixture publication failure"):
        pool_tool.main(_stacked_main_argv(tmp_path))

    row = load_logbook_row(next((tmp_path / "logbook-spool").glob("*.json")))
    assert row.artifact_location is None
    assert set(row.gate_verdicts) == {
        "fixture_completeness",
        "fixture_battery",
        "pipeline_error",
    }
    terminal_path = _receipt_file_from_reference(
        row.gate_verdicts["fixture_battery"]["receipt"]
    )
    assert terminal_path.is_file()
    assert stale_h5.read_bytes() == b"prior-build-artifact"


def test_logbook_gate_receipts_are_immutable_across_later_attempts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    _order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    assert pool_tool.main(_stacked_main_argv(tmp_path)) == 0
    first_row = load_logbook_row(next((tmp_path / "logbook-spool").glob("*.json")))
    first_receipt = _receipt_file_from_reference(
        first_row.gate_verdicts["fixture_battery"]["receipt"]
    )
    first_bytes = first_receipt.read_bytes()

    monkeypatch.setattr(
        pool_tool,
        "by_origin_battery",
        lambda _frame, *, tail_manifest: GateResult(
            name="fixture_battery",
            passed=False,
            failures=("later red verdict",),
        ),
    )
    assert (
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=first_row.row_digest))
        == 1
    )

    rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    second_row = next(
        row for row in rows if row.prev_row_digest == first_row.row_digest
    )
    second_receipt = _receipt_file_from_reference(
        second_row.gate_verdicts["fixture_battery"]["receipt"]
    )
    assert second_receipt != first_receipt
    assert first_receipt.read_bytes() == first_bytes
    assert (
        json.loads(first_bytes)["terminal_gates"]["gates"]["fixture_battery"]["passed"]
        is True
    )
    assert (
        json.loads(second_receipt.read_bytes())["terminal_gates"]["gates"][
            "fixture_battery"
        ]["passed"]
        is False
    )


def test_stacked_checkpoint_identity_binds_both_scale_controls_and_manifest(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    asec = _many_household_source_frame()
    acs = _many_household_source_frame(measured_offset=1_000.0)
    base_stack = pool_tool.assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=0.10,
        sample_seed=578,
    )

    def identity(
        *,
        stack=base_stack,
        sample_fraction: float = 0.10,
        sample_seed: int = 578,
        clone_fraction: float = 1.0,
        clone_seed: int = 578,
    ) -> dict[str, object]:
        return pool_tool._stacked_checkpoint_base_identity(
            verified,
            stack_receipt=stack.receipt,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
            clone_attachment_fraction=clone_fraction,
            clone_attachment_seed=clone_seed,
            policyengine_us_version="fixture-engine",
        )

    fraction_stack = pool_tool.assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=0.01,
        sample_seed=578,
    )
    seed_stack = pool_tool.assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=0.10,
        sample_seed=579,
    )
    mutated_receipt = copy.deepcopy(dict(base_stack.receipt))
    mutated_receipt["survey_samples"]["acs"]["selected_household_ids_sha256"] = "f" * 64
    mutated_stack = SimpleNamespace(receipt=mutated_receipt)
    identities = {
        "base": identity(),
        "sample_fraction": identity(
            stack=fraction_stack,
            sample_fraction=0.01,
        ),
        "sample_seed": identity(stack=seed_stack, sample_seed=579),
        "clone_fraction": identity(clone_fraction=0.5),
        "clone_seed": identity(clone_seed=579),
        "stack_manifest": identity(stack=mutated_stack),
    }
    producer_schedule = identities["base"]["pool_code"]["gap_fill_producer_schedule"]
    assert producer_schedule["status"] == "all_producers_precede_activation"
    assert producer_schedule["direction_count"] == 2
    assert producer_schedule["target_count"] == 48
    digests = {
        name: pool_tool._pool_checkpoint_identity_sha256(value)
        for name, value in identities.items()
    }
    assert len(set(digests.values())) == len(digests)

    bank_outputs = pool_tool._output_paths(
        tmp_path / "identity-pool.h5",
        checkpoint_root=tmp_path / "identity-banks",
    )
    base_bank_outputs = pool_tool._with_checkpoint_identity(
        bank_outputs,
        base_identity_sha256=digests["base"],
    )
    for base_bank in (
        base_bank_outputs.primary_qrf_checkpoint_dir,
        base_bank_outputs.acs_transfer_checkpoint_dir,
    ):
        base_bank.mkdir(parents=True)
        (base_bank / "stale-marker").write_text("must remain unopened\n")
    for name, changed_digest in digests.items():
        if name == "base":
            continue
        changed_outputs = pool_tool._with_checkpoint_identity(
            bank_outputs,
            base_identity_sha256=changed_digest,
        )
        for selected, stale in (
            (
                changed_outputs.primary_qrf_checkpoint_dir,
                base_bank_outputs.primary_qrf_checkpoint_dir,
            ),
            (
                changed_outputs.acs_transfer_checkpoint_dir,
                base_bank_outputs.acs_transfer_checkpoint_dir,
            ),
        ):
            assert selected != stale
            assert selected.name == changed_digest
            routing = pool_tool._identity_routed_bank_open_receipt(
                selected,
                current_base_identity_sha256=changed_digest,
            )
            assert routing["selected_path"] == str(selected.resolve())
            assert routing["identity_mismatches"] == [
                {
                    "load_status": "identity_mismatch",
                    "stale_base_identity_sha256": digests["base"],
                    "current_base_identity_sha256": changed_digest,
                    "disposition": "bypassed",
                    "path": str(stale.resolve()),
                }
            ]
            assert (stale / "stale-marker").read_text() == "must remain unopened\n"

    checkpoint_root = tmp_path / "identity-checkpoints"
    original_store = pool_tool._PoolStageCheckpointStore(
        checkpoint_root,
        base_identity=identities["base"],
    )
    original_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    original_store.write(
        pool_tool.MultispinePoolCheckpoint(
            stage="assembled",
            frame=base_stack.frame,
            assembly_receipt=base_stack.frame.metadata[
                pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
            ],
            stage_receipts={},
        )
    )
    for name, changed_identity in identities.items():
        if name == "base":
            continue
        changed_store = pool_tool._PoolStageCheckpointStore(
            checkpoint_root,
            base_identity=changed_identity,
        )
        assert changed_store.load_deepest() is None


def test_pool_checkpoint_identity_binds_late_producer_schedule(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")

    def identity() -> dict[str, object]:
        return pool_tool._pool_checkpoint_base_identity(
            verified,
            policyengine_us_version="fixture-engine",
        )

    current = identity()
    expected_schedule = pool_tool._json_ready(
        pool_tool.us_late_producer_schedule_receipt()
    )
    assert current["pool_code"]["late_producer_schedule"] == expected_schedule

    changed_schedule = copy.deepcopy(expected_schedule)
    changed_schedule["payload_sha256"] = "0" * 64
    monkeypatch.setattr(
        pool_tool,
        "us_late_producer_schedule_receipt",
        lambda: changed_schedule,
    )
    changed = identity()

    assert pool_tool._pool_checkpoint_identity_sha256(changed) != (
        pool_tool._pool_checkpoint_identity_sha256(current)
    )


def test_legacy_checkpoint_identity_excludes_stacked_late_producer_schedule(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    current = pool_tool._legacy_pool_checkpoint_base_identity(
        verified,
        policyengine_us_version="fixture-engine",
    )
    assert current["materializer_version"] == 3
    assert "late_producer_schedule" not in current["pool_code"]

    changed_schedule = pool_tool._json_ready(
        pool_tool.us_late_producer_schedule_receipt()
    )
    changed_schedule["payload_sha256"] = "0" * 64
    monkeypatch.setattr(
        pool_tool,
        "us_late_producer_schedule_receipt",
        lambda: changed_schedule,
    )
    changed = pool_tool._legacy_pool_checkpoint_base_identity(
        verified,
        policyengine_us_version="fixture-engine",
    )

    assert changed == current


def test_stacked_checkpoint_identity_binds_v13_semantic_contracts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    monkeypatch.setattr(
        pool_tool,
        "_policyengine_us_version",
        lambda: "fixture-engine",
    )
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.10,
        sample_seed=578,
    )

    def identity() -> dict[str, object]:
        return pool_tool._stacked_checkpoint_base_identity(
            verified,
            stack_receipt=stack.receipt,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            policyengine_us_version="fixture-engine",
        )

    current = identity()
    pool_code = current["pool_code"]
    assert current["materializer_version"] == 13
    assert current["stacked_authority"]["version"] == 12
    assert current["geography_assignment"] == (
        pool_tool._stacked_geography_assignment_contract()
    )
    assert current["geography_assignment"]["seed"] == {
        "site": "legacy_puma_ladder",
        "stream": "geography_legacy",
        "value_source": "run_request.build_model_seed",
        "value": 0,
    }
    assert pool_code["operator_order"] == [
        "assemble_stacked_spine",
        "assign_us_puma_ladder",
        "prepare_multispine_source_inputs_for_clone",
        "gap_fill_stacked_spine",
        "run_stacked_late_producer_dag",
        "prepare_stacked_tail_derivation",
        "derive_multispine_pool_inputs",
        "seed_multispine_pool_inputs",
        "materialize_multispine_agreement_outputs",
        "stacked_completeness_gate",
        "by_origin_battery",
    ]
    assert pool_code["late_producer_schedule"] == pool_tool._json_ready(
        pool_tool.us_late_producer_schedule_receipt()
    )
    resource_semantics = pool_code["late_producer_resource_semantics"]
    unsigned_resource_semantics = dict(resource_semantics)
    resource_semantics_sha256 = unsigned_resource_semantics.pop("sha256")
    assert resource_semantics_sha256 == stacked_spine_module._canonical_sha256(
        unsigned_resource_semantics
    )
    assert resource_semantics["producer_count"] == 38
    resource_rows = {
        row["producer"]: row["resources"] for row in resource_semantics["producers"]
    }
    assert list(resource_rows) == list(
        stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE.order
    )
    for (
        producer,
        contract,
    ) in stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY.items():
        assert set(resource_rows[producer]) == (
            stacked_spine_module._late_contract_available_input_keys(contract)
        )
    assert pool_code["primary_qrf_checkpoint_schema_version"] == 6
    assert pool_code["puf_capital_gains_tail_manifest_schema_version"] == 2
    assert pool_code["puf_capital_gains_tail_support_contract"] == (
        pool_tool.puf_capital_gains_tail_support_contract_identity()
    )
    assert pool_code["acs_pums_earnings_universe_contract"] == (
        pool_tool.acs_pums_earnings_universe_contract_identity()
    )
    assert pool_code["us_qbi_reconciliation_contract"] == (
        pool_tool.us_qbi_reconciliation_contract_identity()
    )
    assert pool_code["remaining_stage_input_manifest"] == (
        pool_tool.pool_remaining_stage_input_manifest_receipt()
    )

    with monkeypatch.context() as changed:
        changed.setattr(pool_tool, "PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION", 5)
        stale_qrf = identity()
    with monkeypatch.context() as changed:
        acs_contract = copy.deepcopy(
            pool_tool.acs_pums_earnings_universe_contract_identity()
        )
        acs_contract["minimum_age"] = 14
        acs_body = dict(acs_contract)
        acs_body.pop("sha256")
        acs_contract["sha256"] = hashlib.sha256(
            pool_tool._canonical_json_bytes(acs_body)
        ).hexdigest()
        changed.setattr(
            pool_tool,
            "acs_pums_earnings_universe_contract_identity",
            lambda: acs_contract,
        )
        stale_acs = identity()
    with monkeypatch.context() as changed:
        qbi_contract = copy.deepcopy(
            pool_tool.us_qbi_reconciliation_contract_identity()
        )
        qbi_contract["execution_scope"] = "recipient_subset"
        changed.setattr(
            pool_tool,
            "us_qbi_reconciliation_contract_identity",
            lambda: qbi_contract,
        )
        stale_qbi = identity()
    with monkeypatch.context() as changed:
        changed.setattr(pool_tool, "PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION", 1)
        stale_tail_schema = identity()
    with monkeypatch.context() as changed:
        remaining_manifest = copy.deepcopy(
            pool_tool.pool_remaining_stage_input_manifest_receipt()
        )
        remaining_manifest["manifest_sha256"] = "0" * 64
        changed.setattr(
            pool_tool,
            "pool_remaining_stage_input_manifest_receipt",
            lambda: remaining_manifest,
        )
        stale_remaining_manifest = identity()
    with monkeypatch.context() as changed:
        tail_contract = copy.deepcopy(
            pool_tool.puf_capital_gains_tail_support_contract_identity()
        )
        tail_contract["required_minimum"] = "one_recipient_per_status"
        changed.setattr(
            pool_tool,
            "puf_capital_gains_tail_support_contract_identity",
            lambda: tail_contract,
        )
        stale_tail_contract = identity()
    with monkeypatch.context() as changed:
        late_schedule = pool_tool._json_ready(
            pool_tool.us_late_producer_schedule_receipt()
        )
        late_schedule["payload_sha256"] = "0" * 64
        changed.setattr(
            pool_tool,
            "us_late_producer_schedule_receipt",
            lambda: late_schedule,
        )
        stale_late_schedule = identity()
    with monkeypatch.context() as changed:
        calibration_authority = copy.deepcopy(
            pool_tool.stacked_spine_authority_receipt()
        )
        calibration_authority["components"]["post_transfer_calibration"]["identity"][
            "scope"
        ]["reference"] = "forged_reference_scope"
        changed.setattr(
            pool_tool,
            "stacked_spine_authority_receipt",
            lambda: calibration_authority,
        )
        stale_post_transfer_calibration = identity()
    with monkeypatch.context() as changed:
        source_stage_binding = stacked_spine_module._late_source_stage_spec_binding

        def changed_source_stage_binding(
            operator: str,
            **kwargs: object,
        ) -> dict[str, object] | None:
            binding = source_stage_binding(operator, **kwargs)
            if operator != "with_us_adult_care_inputs" or binding is None:
                return binding
            mutated = copy.deepcopy(binding)
            mutated["asset_sha256"] = "0" * 64
            return mutated

        changed.setattr(
            stacked_spine_module,
            "_late_source_stage_spec_binding",
            changed_source_stage_binding,
        )
        stale_source_asset = identity()

    digests = {
        pool_tool._pool_checkpoint_identity_sha256(candidate)
        for candidate in (
            current,
            stale_qrf,
            stale_acs,
            stale_qbi,
            stale_tail_schema,
            stale_remaining_manifest,
            stale_tail_contract,
            stale_late_schedule,
            stale_post_transfer_calibration,
            stale_source_asset,
        )
    }
    assert len(digests) == 10

    # Positive control: discovery accepts the exact current semantic identity
    # under the same fixture engine version used to construct it.
    current_checkpoint_root = tmp_path / "current-semantic-checkpoints"
    current_store = pool_tool._PoolStageCheckpointStore(
        current_checkpoint_root,
        base_identity=current,
    )
    current_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    current_store.write(
        pool_tool.MultispinePoolCheckpoint(
            stage="assembled",
            frame=stack.frame,
            assembly_receipt=stack.frame.metadata[
                pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
            ],
            stage_receipts={},
        )
    )
    assert (
        pool_tool._discover_stacked_checkpoint_identity(
            current_checkpoint_root,
            verified_inputs=verified,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        == current
    )

    # A checkpoint produced by the current materializer with the prior QRF
    # schema is not merely identity-distinct: discovery must refuse it as stale.
    checkpoint_root = tmp_path / "mixed-qrf-version-checkpoints"
    stale_qrf_store = pool_tool._PoolStageCheckpointStore(
        checkpoint_root,
        base_identity=stale_qrf,
    )
    stale_qrf_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    stale_qrf_store.write(
        pool_tool.MultispinePoolCheckpoint(
            stage="assembled",
            frame=stack.frame,
            assembly_receipt=stack.frame.metadata[
                pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
            ],
            stage_receipts={},
        )
    )

    assert current["materializer_version"] == stale_qrf["materializer_version"] == 13
    assert stale_qrf["pool_code"]["primary_qrf_checkpoint_schema_version"] == 5
    assert (
        pool_tool._discover_stacked_checkpoint_identity(
            checkpoint_root,
            verified_inputs=verified,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        is None
    )
    assert "checkpoint base identity is stale" in capsys.readouterr().out

    # Resource semantics are equally resume-fatal: a checkpoint whose source
    # asset/config binding differs must never be selected under current code.
    resource_checkpoint_root = tmp_path / "mixed-resource-checkpoints"
    stale_resource_store = pool_tool._PoolStageCheckpointStore(
        resource_checkpoint_root,
        base_identity=stale_source_asset,
    )
    stale_resource_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    stale_resource_store.write(
        pool_tool.MultispinePoolCheckpoint(
            stage="assembled",
            frame=stack.frame,
            assembly_receipt=stack.frame.metadata[
                pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
            ],
            stage_receipts={},
        )
    )

    assert (
        pool_tool._discover_stacked_checkpoint_identity(
            resource_checkpoint_root,
            verified_inputs=verified,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        is None
    )
    assert "checkpoint base identity is stale" in capsys.readouterr().out


def test_pool_envelope_v7_preserves_stacked_bank_identity_but_rejects_v6(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.10,
        sample_seed=578,
    )

    def identity() -> dict[str, object]:
        return pool_tool._stacked_checkpoint_base_identity(
            verified,
            stack_receipt=stack.receipt,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            policyengine_us_version="fixture-engine",
        )

    current_identity = identity()
    current_digest = pool_tool._pool_checkpoint_identity_sha256(current_identity)
    checkpoint_root = tmp_path / "envelope-version-checkpoints"
    with monkeypatch.context() as legacy:
        legacy.setattr(
            pool_tool,
            "POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION",
            6,
        )
        assert identity() == current_identity
        legacy_store = pool_tool._PoolStageCheckpointStore(
            checkpoint_root,
            base_identity=current_identity,
        )
        assert legacy_store.base_identity_sha256 == current_digest
        legacy_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        legacy_store.write(
            pool_tool.MultispinePoolCheckpoint(
                stage="assembled",
                frame=stack.frame,
                assembly_receipt=stack.frame.metadata[
                    pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
                ],
                stage_receipts={},
            )
        )
    capsys.readouterr()

    assert pool_tool.POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION == 7
    assert identity() == current_identity
    current_store = pool_tool._PoolStageCheckpointStore(
        checkpoint_root,
        base_identity=current_identity,
    )
    assert current_store.base_identity_sha256 == current_digest
    assert current_store.load_deepest() is None
    assert "unsupported binding" in capsys.readouterr().out


def test_geography_assignment_receipt_rejects_divergent_clone_geography(
    pool_tool: ModuleType,
) -> None:
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(count=2),
        _many_household_source_frame(count=2, measured_offset=1_000.0),
        sample_fraction=1.0,
        sample_seed=578,
    )
    assigned = _with_fixture_household_geography(stack.frame)
    household = assigned.table("household")
    source_column = pool_tool.support_source_id_column("household")
    household.loc[household.index[1], source_column] = household.loc[
        household.index[0], source_column
    ]
    household.loc[household.index[1], "congressional_district_geoid"] = 602
    receipt = _fixture_geography_assignment_receipt(pool_tool, assigned)
    receipt["target_universe"] = (
        pool_tool._target_congressional_district_universe_receipt((601, 602))
    )
    receipt["output"]["unique_congressional_district_values"] = 2

    with pytest.raises(
        ValueError,
        match="cloned household support rows disagree.*congressional_district_geoid",
    ):
        pool_tool._validate_stacked_geography_assignment_receipt(
            assigned,
            receipt,
            target_districts=(601, 602),
            boundary="fixture clone divergence",
            require_exact_assembled_rows=False,
        )


def test_geography_assignment_receipt_rejects_coherent_valid_target_rewrite(
    pool_tool: ModuleType,
) -> None:
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(count=2),
        _many_household_source_frame(count=2, measured_offset=1_000.0),
        sample_fraction=1.0,
        sample_seed=578,
    )
    assigned = _with_fixture_household_geography(stack.frame)
    receipt = _fixture_geography_assignment_receipt(pool_tool, assigned)
    receipt["target_universe"] = (
        pool_tool._target_congressional_district_universe_receipt((601, 602))
    )
    assigned.table("household")["congressional_district_geoid"] = 602

    with pytest.raises(
        ValueError,
        match="ordered native household geography differs",
    ):
        pool_tool._validate_stacked_geography_assignment_receipt(
            assigned,
            receipt,
            target_districts=(601, 602),
            boundary="fixture coherent geography rewrite",
            require_exact_assembled_rows=False,
        )


@pytest.mark.parametrize(
    ("route", "stage_receipts"),
    (
        (
            "legacy",
            {"derive": {"qbi_input_reconciliation": {"fixture": "receipt"}}},
        ),
        (
            "stacked",
            {
                "derive": {
                    "pool_derivation": {
                        "qbi_input_reconciliation": {"fixture": "receipt"}
                    }
                }
            },
        ),
    ),
)
def test_qbi_receipt_route_resolution_is_exact(
    pool_tool: ModuleType,
    route: str,
    stage_receipts: Mapping[str, Mapping[str, object]],
) -> None:
    assert pool_tool._qbi_receipt_from_stage_receipts(
        stage_receipts,
        route=route,
        boundary="fixture QBI route",
    ) == {"fixture": "receipt"}


@pytest.mark.parametrize(
    ("route", "stage_receipts", "message"),
    (
        (
            "legacy",
            {
                "derive": {
                    "pool_derivation": {
                        "qbi_input_reconciliation": {"fixture": "stacked"}
                    }
                }
            },
            "legacy QBI receipt used the stacked derive route",
        ),
        (
            "stacked",
            {"derive": {"qbi_input_reconciliation": {"fixture": "legacy"}}},
            "stacked derive receipts have no pool_derivation object",
        ),
        (
            "stacked",
            {
                "derive": {
                    "qbi_input_reconciliation": {"fixture": "legacy"},
                    "pool_derivation": {
                        "qbi_input_reconciliation": {"fixture": "stacked"}
                    },
                }
            },
            "stacked QBI receipt also appears at the legacy route",
        ),
    ),
)
def test_qbi_receipt_route_resolution_rejects_wrong_or_ambiguous_paths(
    pool_tool: ModuleType,
    route: str,
    stage_receipts: Mapping[str, Mapping[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pool_tool._qbi_receipt_from_stage_receipts(
            stage_receipts,
            route=route,
            boundary="fixture QBI route",
        )


@pytest.mark.parametrize("legacy_version", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
def test_legacy_stacked_materializer_checkpoint_is_not_discovered(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    legacy_version: int,
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    monkeypatch.setattr(
        pool_tool,
        "_policyengine_us_version",
        lambda: "fixture-engine",
    )
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.10,
        sample_seed=578,
    )
    checkpoint_root = tmp_path / "stacked-materializer-checkpoints"

    with monkeypatch.context() as legacy:
        legacy.setattr(
            pool_tool,
            "_STACKED_CHECKPOINT_MATERIALIZER_VERSION",
            legacy_version,
        )
        legacy_identity = pool_tool._stacked_checkpoint_base_identity(
            verified,
            stack_receipt=stack.receipt,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            policyengine_us_version="fixture-engine",
        )
        legacy_store = pool_tool._PoolStageCheckpointStore(
            checkpoint_root,
            base_identity=legacy_identity,
        )
        legacy_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        legacy_store.write(
            pool_tool.MultispinePoolCheckpoint(
                stage="assembled",
                frame=stack.frame,
                assembly_receipt=stack.frame.metadata[
                    pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
                ],
                stage_receipts={},
            )
        )

    assert pool_tool._STACKED_CHECKPOINT_MATERIALIZER_VERSION == 13
    assert (
        pool_tool._discover_stacked_checkpoint_identity(
            checkpoint_root,
            verified_inputs=verified,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        is None
    )
    assert "checkpoint base identity is stale" in capsys.readouterr().out


def test_stacked_resume_rejects_noncanonical_post_puf_transfer_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.10,
        sample_seed=578,
    )
    assigned = _with_fixture_household_geography(stack.frame)
    geography_receipt = _fixture_geography_assignment_receipt(pool_tool, assigned)
    noncanonical = _noncanonical_post_puf_authority_receipt()
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool,
        assigned,
        authority=noncanonical,
    )
    resume = pool_tool.MultispinePoolCheckpoint(
        stage="transferred",
        frame=authorized,
        assembly_receipt=stack.frame.metadata[pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY],
        stage_receipts={
            "geography_assignment": geography_receipt,
            "impute": impute,
        },
        late_producer_transition_authority_sha256=transition_authority_sha256,
    )

    with pytest.raises(
        ValueError,
        match=(
            "stacked transferred checkpoint resume: non-canonical stacked "
            "authority is forbidden"
        ),
    ):
        pool_tool.build_stacked_pool(
            assigned,
            expected_stack_receipt=stack.receipt,
            geography_assignment_receipt=None,
            geography_target_districts=(601,),
            release_id=(
                "populace-us-2024-stacked-f010-s578-asec4-acs1-"
                "20260807T000000Z-deadbeef"
            ),
            puf_donor=None,
            acs_rent_donor=None,
            primary_qrf_checkpoint_dir=tmp_path / "primary-qrf",
            acs_transfer_checkpoint_dir=tmp_path / "acs-transfer",
            checkpoint_identity={},
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            resume=resume,
        )


def test_stacked_entrypoint_resumes_each_checkpoint_boundary(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    assert pool_tool.main(_stacked_main_argv(tmp_path)) == 0
    cold_order = list(order)
    first_row = load_logbook_row(next((tmp_path / "logbook-spool").glob("*.json")))

    order.clear()
    assert (
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=first_row.row_digest))
        == 0
    )
    simulated_resume_order = list(order)
    rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    second_row = next(
        row for row in rows if row.prev_row_digest == first_row.row_digest
    )

    checkpoint_root = next(
        (tmp_path / "stacked-pool.checkpoints" / "stacked").iterdir()
    )
    for suffix in (".h5", ".manifest.json"):
        (checkpoint_root / f"simulated.checkpoint{suffix}").unlink()
    order.clear()
    assert (
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=second_row.row_digest))
        == 0
    )
    transferred_resume_order = list(order)
    rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    third_row = next(
        row for row in rows if row.prev_row_digest == second_row.row_digest
    )

    for stage in ("transferred", "simulated"):
        for suffix in (".h5", ".manifest.json"):
            (checkpoint_root / f"{stage}.checkpoint{suffix}").unlink()
    order.clear()
    assert (
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=third_row.row_digest))
        == 0
    )
    assembled_resume_order = list(order)

    assert cold_order == [
        "stack",
        "geography",
        "build_stacked_pool",
        "prepare",
        "gap",
        "puf",
        "late_producer_dag",
        "tail_prepare",
        "derive",
        "seed",
        "simulate",
        "completeness",
        "battery",
        "publish",
    ]
    assert simulated_resume_order == [
        "build_stacked_pool",
        "completeness",
        "battery",
        "publish",
    ]
    assert transferred_resume_order == [
        "build_stacked_pool",
        "tail_prepare",
        "derive",
        "seed",
        "simulate",
        "completeness",
        "battery",
        "publish",
    ]
    assert assembled_resume_order == [
        "build_stacked_pool",
        "prepare",
        "gap",
        "puf",
        "late_producer_dag",
        "tail_prepare",
        "derive",
        "seed",
        "simulate",
        "completeness",
        "battery",
        "publish",
    ]
    final_rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    assert len(final_rows) == 4
    assert any(row.prev_row_digest == third_row.row_digest for row in final_rows)


def test_stacked_resume_error_uses_realized_stack_identity(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    _order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    assert pool_tool.main(_stacked_main_argv(tmp_path)) == 0
    first_row = load_logbook_row(next((tmp_path / "logbook-spool").glob("*.json")))

    checkpoint_root = next(
        (tmp_path / "stacked-pool.checkpoints" / "stacked").iterdir()
    )
    for stage in ("transferred", "simulated"):
        for suffix in (".h5", ".manifest.json"):
            (checkpoint_root / f"{stage}.checkpoint{suffix}").unlink()
    monkeypatch.setattr(
        pool_tool,
        "_load_puf_donor",
        lambda _args: (_ for _ in ()).throw(
            RuntimeError("fixture resumed donor failure")
        ),
    )

    with pytest.raises(RuntimeError, match="fixture resumed donor failure"):
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=first_row.row_digest))

    rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    failed_row = next(
        row for row in rows if row.prev_row_digest == first_row.row_digest
    )
    assert "f001-s578-asec1-acs1" in failed_row.build_id
    assert failed_row.identity_digest == first_row.identity_digest
    assert "checkpoint_loaded" in failed_row.phases_reached
    assert "resume_donors_loaded" not in failed_row.phases_reached
    assert failed_row.gate_verdicts["pipeline_error"]["verdict"] == "error"


@pytest.mark.parametrize(
    ("fraction", "token"),
    [(0.01, "f001"), (0.10, "f010"), (1.0, "f100")],
)
def test_stacked_release_id_carries_rung_seed_and_realized_counts(
    pool_tool: ModuleType,
    fraction: float,
    token: str,
) -> None:
    release_id = pool_tool._new_stacked_release_id(
        sample_fraction=fraction,
        sample_seed=578,
        realized_asec_households=123,
        realized_acs_households=456,
        timestamp=datetime(2026, 8, 5, 12, 34, 56, tzinfo=UTC),
        nonce="abcdef01",
    )

    assert release_id == (
        f"populace-us-2024-stacked-{token}-s578-asec123-acs456-"
        "20260805T123456Z-abcdef01"
    )


def test_legacy_two_spine_fixture_is_origin_main_byte_exact(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Keep this byte-level legacy-output golden on its pre-authentication
    # synthetic fixture. QBI authentication has independent generation,
    # checkpoint, resume, manifest, and publication tamper tests.
    monkeypatch.setattr(
        multispine_pool_module,
        "_validate_qbi_stage_receipt",
        lambda _frame, _stage_receipts, *, boundary, transition_authority_sha256: None,
    )
    monkeypatch.setattr(
        pool_tool,
        "_validate_qbi_stage_receipt",
        lambda _frame, _stage_receipts, *, route, boundary, transition_authority_sha256: (
            None
        ),
    )
    store = _checkpoint_fixture_store(pool_tool, tmp_path / "checkpoints")
    store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    result, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=store,
        authenticated_qbi=False,
    )
    artifact = tmp_path / "legacy-origin-main.canonical.h5"
    pool_tool.write_frame_checkpoint(artifact, result.frame)

    assert order == ["impute", "derive", "seed", "simulate"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        # Rebased once on the import-entry branch: frame checkpoints now
        # record string storage and NA markers explicitly (environment-
        # independent restores, needed once microcosm-build ships pyarrow),
        # which adds two metadata fields per StringDtype column. Verified
        # identical on CPython 3.13/3.14, Linux CI, and macOS.
        "12e937914d739f5bd9a1a59df7b6de7ae5458f06cdc6c3b93a09ddf4ee47ecbd"
    )


def test_legacy_entrypoint_publication_matches_origin_main_golden(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _unused_outputs, verified, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
        authenticated_qbi=False,
    )
    ready = replace(
        result,
        agreement_gate=GateResult("us_spine_agreement", True),
    )
    build_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fixture_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return ready

    def deterministic_fixture_h5(
        frame: Frame,
        path: Path,
        *,
        period: int,
        artifact_kind: str,
        publication_run_id: str,
    ) -> None:
        assert period == pool_tool.POOL_TIME_PERIOD
        assert artifact_kind == pool_tool.POOL_H5_ARTIFACT_KIND
        assert publication_run_id == "fixture-publication-run"
        # Pandas/PyTables embeds wall-clock metadata in its H5 bytes.  Keep the
        # real publication envelope but use Microcosm's timestamp-free Frame
        # serializer so this origin/main golden is stable across processes.
        pool_tool.write_frame_checkpoint(path, frame)

    monkeypatch.setattr(
        pool_tool,
        "_verify_inputs",
        lambda _args, _outputs: (verified, source_manifest),
    )
    monkeypatch.setattr(
        pool_tool,
        "_load_inputs",
        lambda _args, *, acs_source_manifest: loaded,
    )
    monkeypatch.setattr(pool_tool, "build_multispine_pool", fixture_build)
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: "fixture-publication-run",
    )
    monkeypatch.setattr(
        pool_tool,
        "_policyengine_us_version",
        lambda: "fixture-policyengine-us",
    )
    monkeypatch.setattr(
        pool_tool,
        "write_nullable_us_h5",
        deterministic_fixture_h5,
    )
    # This golden deliberately preserves the pre-authentication synthetic
    # output byte-for-byte. Dedicated QBI boundary tests below exercise the
    # authenticated production contract against real frame-bound receipts.
    monkeypatch.setattr(
        pool_tool,
        "_validate_qbi_stage_receipt",
        lambda _frame, _stage_receipts, *, route, boundary, transition_authority_sha256: (
            None
        ),
    )

    output = tmp_path / "legacy-pool.h5"
    checkpoint_root = tmp_path / "checkpoints"
    argv: list[str] = []
    for option in (
        "asec-raw-stage-h5",
        "acs-household-zip",
        "acs-person-zip",
        "acs-rent-h5",
        "puf-h5",
        "puf-source-year-csv",
    ):
        argv.extend([f"--{option}", str(tmp_path / option)])
        argv.extend([f"--{option}-sha256", "1" * 64])
    argv.extend(
        [
            "--checkpoint-root",
            str(checkpoint_root),
            "--out",
            str(output),
            "--legacy-two-spine",
        ]
    )

    assert pool_tool.main(argv) == 0
    assert len(build_calls) == 1
    positional, keywords = build_calls[0]
    assert positional == (loaded.asec, loaded.acs)
    assert keywords["puf_donor"] is loaded.puf_donor
    assert keywords["acs_rent_donor"] is loaded.acs_rent_donor
    assert keywords["source_native_inputs"] == {"acs": loaded.acs_native_inputs}
    assert keywords["resume"] is None
    assert callable(keywords["checkpoint"])
    checkpoint_store = keywords["checkpoint"].__self__
    assert checkpoint_store.base_identity["materializer_version"] == 3
    assert "late_producer_schedule" not in checkpoint_store.base_identity["pool_code"]

    outputs = pool_tool._output_paths(output, checkpoint_root=checkpoint_root)
    manifest = pool_tool._read_json_object(outputs.manifest)
    diagnostics = pool_tool._read_json_object(outputs.agreement_diagnostics)
    assert pool_tool.POOL_MANIFEST_SCHEMA_VERSION == 10
    assert pool_tool.POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION == 7
    assert manifest["schema_version"] == 4
    assert diagnostics["schema_version"] == 4
    assert "materializer_version" not in manifest["pool_h5"]
    assert manifest["stage_checkpoints"]["materializer_version"] == 3
    assert {
        receipt["materializer_version"]
        for receipt in manifest["stage_checkpoints"]["stages"].values()
    } == {3}
    manifest_bytes = outputs.manifest.read_bytes().replace(
        str(tmp_path.resolve()).encode(),
        b"$TMP",
    )
    actual = {
        "pool_h5": hashlib.sha256(outputs.pool_h5.read_bytes()).hexdigest(),
        "agreement": hashlib.sha256(
            outputs.agreement_diagnostics.read_bytes()
        ).hexdigest(),
        "manifest": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    # Generated by executing origin/main at 188c5d9c and this branch's explicit
    # legacy entrypoint against the same fixture path and publication run ID;
    # the pool tool remains unchanged through the e6be79a7 transplant base.
    assert actual == {
        # Rebased with the fixture golden above (explicit string-storage
        # checkpoint metadata).
        "pool_h5": "ced797ecdd44a638c2a3945f07ad612098a7095ca53a5f458699bca6d6e38b3e",
        "agreement": "f39f0d918bf7ee01dddb5517d8830b8adb541273c5be084307be91397caca3cb",
        # The PE-US 1.819.0 compatibility edits legitimately move the pool-code
        # checkpoint identities embedded in the otherwise legacy publication.
        "manifest": "63c6e6973079f0b793d5435113aaae66184564b70271f8af120fecdbb5015f63",
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_main_dispatches_only_to_the_selected_pipeline(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    calls: list[str] = []
    namespace = SimpleNamespace(
        legacy_two_spine=legacy,
        config_authority="constants",
    )
    parser = SimpleNamespace(parse_args=lambda _argv: namespace)
    monkeypatch.setattr(pool_tool, "_parser", lambda: parser)
    monkeypatch.setattr(
        pool_tool,
        "_main_legacy",
        lambda _args: calls.append("legacy") or 17,
    )
    monkeypatch.setattr(
        pool_tool,
        "_main_stacked",
        lambda _args: calls.append("stacked") or 23,
    )

    assert pool_tool.main(["fixture"]) == (17 if legacy else 23)
    assert calls == (["legacy"] if legacy else ["stacked"])


def test_main_refuses_constants_adapter_with_legacy_two_spine(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    namespace = SimpleNamespace(
        legacy_two_spine=True,
        config_authority="constants_adapter",
    )
    parser = SimpleNamespace(parse_args=lambda _argv: namespace)
    monkeypatch.setattr(pool_tool, "_parser", lambda: parser)
    monkeypatch.setattr(
        pool_tool,
        "_main_legacy",
        lambda _args: calls.append("legacy") or 17,
    )
    monkeypatch.setattr(
        pool_tool,
        "_main_stacked",
        lambda _args: calls.append("stacked") or 23,
    )

    with pytest.raises(
        ValueError,
        match="constants_adapter.*stacked pipeline",
    ):
        pool_tool.main(["fixture"])
    assert calls == []


def test_parser_exposes_eight_pinned_inputs_out_and_checkpoint_root(
    pool_tool: ModuleType,
) -> None:
    parser = pool_tool._parser()
    actions = {
        action.dest: action for action in parser._actions if action.dest != "help"
    }
    pairs = (
        ("asec_raw_stage_h5", "asec_raw_stage_h5_sha256"),
        ("acs_household_zip", "acs_household_zip_sha256"),
        ("acs_person_zip", "acs_person_zip_sha256"),
        ("acs_rent_h5", "acs_rent_h5_sha256"),
        ("puf_h5", "puf_h5_sha256"),
        ("puf_source_year_csv", "puf_source_year_csv_sha256"),
        ("puma_ladder", "puma_ladder_sha256"),
        (
            "congressional_district_vintage_crosswalk",
            "congressional_district_vintage_crosswalk_sha256",
        ),
    )
    expected_destinations = {destination for pair in pairs for destination in pair} | {
        "config_authority",
        "logbook_prev_row_digest",
        "clone_attachment_fraction",
        "clone_attachment_seed",
        "checkpoint_root",
        "legacy_two_spine",
        "out",
        "sample_fraction",
        "sample_seed",
    }

    assert set(actions) == expected_destinations
    assert all(
        actions[destination].required
        for destination in expected_destinations
        - {
            "checkpoint_root",
            "config_authority",
            "logbook_prev_row_digest",
            "clone_attachment_fraction",
            "clone_attachment_seed",
            "puma_ladder",
            "puma_ladder_sha256",
            "congressional_district_vintage_crosswalk",
            "congressional_district_vintage_crosswalk_sha256",
            "legacy_two_spine",
            "sample_fraction",
            "sample_seed",
        }
    )
    assert not actions["checkpoint_root"].required
    assert actions["out"].option_strings == ["--out"]
    assert actions["checkpoint_root"].option_strings == ["--checkpoint-root"]
    assert actions["checkpoint_root"].type is Path
    assert actions["sample_fraction"].default == 1.0
    assert actions["sample_seed"].default == 578
    assert actions["clone_attachment_fraction"].default == 1.0
    assert actions["clone_attachment_seed"].default == 578
    assert actions["legacy_two_spine"].default is False
    assert actions["config_authority"].default == "constants"
    assert tuple(actions["config_authority"].choices) == (
        "constants",
        "constants_adapter",
    )
    assert actions["logbook_prev_row_digest"].default is None
    for path_destination, sha_destination in pairs:
        assert actions[path_destination].type is Path
        assert actions[sha_destination].type is pool_tool._sha256_argument
        assert len(actions[path_destination].option_strings) == 1
        assert len(actions[sha_destination].option_strings) == 1

    option_names = {
        option for action in actions.values() for option in action.option_strings
    }
    assert not any(
        forbidden in option
        for option in option_names
        for forbidden in ("tolerance", "target", "per-target")
    )


@pytest.mark.parametrize(
    ("destination", "option"),
    [
        ("puma_ladder", "--puma-ladder"),
        ("puma_ladder_sha256", "--puma-ladder-sha256"),
        (
            "congressional_district_vintage_crosswalk",
            "--congressional-district-vintage-crosswalk",
        ),
        (
            "congressional_district_vintage_crosswalk_sha256",
            "--congressional-district-vintage-crosswalk-sha256",
        ),
    ],
)
def test_stacked_route_requires_each_pinned_geography_authority(
    pool_tool: ModuleType,
    tmp_path: Path,
    destination: str,
    option: str,
) -> None:
    args = SimpleNamespace(
        puma_ladder=tmp_path / "puma-ladder.npz",
        puma_ladder_sha256=pool_tool._STACKED_PUMA_LADDER_SHA256,
        congressional_district_vintage_crosswalk=tmp_path / "crosswalk.csv",
        congressional_district_vintage_crosswalk_sha256=(
            pool_tool._STACKED_CD_CROSSWALK_SHA256
        ),
    )
    setattr(args, destination, None)

    with pytest.raises(ValueError, match=option):
        pool_tool._require_stacked_geography_arguments(args)


@pytest.mark.parametrize(
    ("destination", "option"),
    [
        ("puma_ladder_sha256", "--puma-ladder-sha256"),
        (
            "congressional_district_vintage_crosswalk_sha256",
            "--congressional-district-vintage-crosswalk-sha256",
        ),
    ],
)
def test_stacked_route_rejects_noncanonical_geography_authority_pins(
    pool_tool: ModuleType,
    tmp_path: Path,
    destination: str,
    option: str,
) -> None:
    args = SimpleNamespace(
        puma_ladder=tmp_path / "puma-ladder.npz",
        puma_ladder_sha256=pool_tool._STACKED_PUMA_LADDER_SHA256,
        congressional_district_vintage_crosswalk=tmp_path / "crosswalk.csv",
        congressional_district_vintage_crosswalk_sha256=(
            pool_tool._STACKED_CD_CROSSWALK_SHA256
        ),
    )
    setattr(args, destination, "f" * 64)

    with pytest.raises(ValueError, match=option):
        pool_tool._require_stacked_geography_arguments(args)


def test_pool_tool_structurally_accepts_only_the_raw_stage_loader(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "load_asec_raw_stage_checkpoint" in imported
    assert "load_asec_raw_stage_checkpoint" in called
    assert "load_asec_pre_clone_checkpoint" not in imported
    assert "load_asec_pre_clone_checkpoint" not in called
    assert "pre_clone_enrichment" not in source


def test_checkpoint_root_defaults_alongside_out_and_accepts_override(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run" / "pool.h5"
    default = pool_tool._output_paths(output)

    assert default.checkpoint_root == output.with_suffix(".checkpoints")
    assert default.primary_qrf_checkpoint_dir == (
        output.with_suffix(".checkpoints") / "primary-qrf"
    )
    assert default.acs_transfer_checkpoint_dir == (
        output.with_suffix(".checkpoints") / "acs-transfer"
    )

    explicit_root = tmp_path / "persistent" / "pool-checkpoints"
    explicit = pool_tool._output_paths(
        output,
        checkpoint_root=explicit_root,
    )

    assert explicit.checkpoint_root == explicit_root
    assert explicit.primary_qrf_checkpoint_dir == explicit_root / "primary-qrf"
    assert explicit.acs_transfer_checkpoint_dir == explicit_root / "acs-transfer"

    identity_sha256 = "a" * 64
    bound = pool_tool._with_checkpoint_identity(
        explicit,
        base_identity_sha256=identity_sha256,
    )
    assert bound.primary_qrf_checkpoint_dir == (
        explicit_root / "primary-qrf" / identity_sha256
    )
    assert bound.acs_transfer_checkpoint_dir == (
        explicit_root / "acs-transfer" / identity_sha256
    )


def test_identity_routed_bank_sibling_scan_is_bounded_and_ignores_junk(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bank_root = tmp_path / "primary-qrf"
    bank_root.mkdir()
    current_digest = "f" * 64
    selected = bank_root / current_digest

    class _Entry:
        def __init__(self, name: str, *, directory: bool) -> None:
            self.name = name
            self._directory = directory

        def is_dir(self, *, follow_symlinks: bool) -> bool:
            assert follow_symlinks is False
            return self._directory

    entries = [
        _Entry(current_digest, directory=True),
        _Entry("not-a-digest", directory=True),
        _Entry("1" * 64, directory=False),
        _Entry("2" * 64, directory=False),
        _Entry("3" * 64, directory=True),
        *[_Entry(f"{index:064x}", directory=True) for index in range(10, 70)],
    ]

    class _Scandir:
        def __enter__(self):
            return iter(entries)

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(pool_tool.os, "scandir", lambda _root: _Scandir())

    receipt = pool_tool._identity_routed_bank_open_receipt(
        selected,
        current_base_identity_sha256=current_digest,
    )

    assert receipt["scan"] == {
        "limit": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
        "entries_examined": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
        "truncated": True,
    }
    stale_digests = {
        record["stale_base_identity_sha256"]
        for record in receipt["identity_mismatches"]
    }
    assert "3" * 64 in stale_digests
    assert current_digest not in stale_digests
    assert "1" * 64 not in stale_digests
    assert "2" * 64 not in stale_digests
    assert all(
        record["load_status"] == "identity_mismatch"
        and record["disposition"] == "bypassed"
        for record in receipt["identity_mismatches"]
    )


def test_checkpoint_root_allows_safe_input_colocation_on_persistent_volume(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "persistent-volume"
    outputs = pool_tool._output_paths(
        tmp_path / "published" / "pool.h5",
        checkpoint_root=checkpoint_root,
    )

    pool_tool._validate_checkpoint_path_layout(
        outputs,
        source_paths={checkpoint_root / "inputs" / "asec.h5"},
    )


@pytest.mark.parametrize(
    "outputs",
    (
        lambda pool_tool, tmp_path: pool_tool._output_paths(
            tmp_path / "pool.h5",
            checkpoint_root=tmp_path / "pool.h5",
        ),
        lambda pool_tool, tmp_path: pool_tool._output_paths(
            tmp_path / "checkpoints" / "assembled.checkpoint.h5",
            checkpoint_root=tmp_path / "checkpoints",
        ),
    ),
)
def test_checkpoint_root_rejects_publication_file_collisions(
    pool_tool: ModuleType,
    tmp_path: Path,
    outputs: Callable[[ModuleType, Path], object],
) -> None:
    with pytest.raises(
        ValueError,
        match="checkpoint paths collide with publication files",
    ):
        pool_tool._validate_checkpoint_path_layout(
            outputs(pool_tool, tmp_path),
            source_paths=set(),
        )


def test_stacked_nested_checkpoint_root_rejects_a_stage_publication_collision(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    configured_root = tmp_path / "checkpoints"
    stacked_root = configured_root / "stacked" / ("a" * 64)
    base_outputs = pool_tool._stacked_output_paths(
        stacked_root / "assembled.checkpoint.h5",
        checkpoint_root=configured_root,
    )
    nested_outputs = replace(
        base_outputs,
        checkpoint_root=stacked_root,
        primary_qrf_checkpoint_dir=stacked_root / "primary-qrf",
        acs_transfer_checkpoint_dir=stacked_root / "acs-transfer",
    )

    with pytest.raises(
        ValueError,
        match="checkpoint paths collide with publication files",
    ):
        pool_tool._validate_checkpoint_path_layout(
            nested_outputs,
            source_paths=set(),
        )


def test_atomic_json_fsyncs_parent_directory_after_rename(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(pool_tool.os, "fsync", tracked_fsync)
    monkeypatch.setattr(pool_tool.os, "replace", tracked_replace)
    output = tmp_path / "receipt.json"

    pool_tool._atomic_write_json(output, {"fixture": True})

    assert events == [
        ("fsync", "file"),
        ("replace", output.name),
        ("fsync", "directory"),
    ]


def test_json_ready_omits_empty_opt_in_transfer_pattern_regimes(
    pool_tool: ModuleType,
) -> None:
    pattern = acs_transfer_module.AcsTransferPattern(
        name="fixture",
        observed_optional_predictors=(),
        predictors=("age",),
        seed=1,
        weight_kind="source",
        donor_rows=2,
        recipient_rows=1,
    )

    legacy = pool_tool._json_ready(pattern)
    selected = pool_tool._json_ready(
        replace(
            pattern,
            target_regimes=(("fixture_target", "positive_only"),),
        )
    )

    assert isinstance(legacy, dict)
    assert "target_regimes" not in legacy
    assert isinstance(selected, dict)
    assert selected["target_regimes"] == [["fixture_target", "positive_only"]]


def test_pool_imputation_wires_post_clone_source_chain_after_primary_and_tail(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool._impute_pool)
    tree = ast.parse(source)
    expected = (
        "_initialize_or_resume_primary_qrf",
        "run_primary_puf_qrf_chain",
        "finalize_primary_puf_qrf_chain",
        "transfer_puf_capital_gains_tail",
        "complete_multispine_source_inputs",
        "pool_transfer_target_families",
        "AcsTransferTargetBankStore",
        "transfer_acs_inputs",
    )
    calls = sorted(
        (
            node.lineno,
            node.func.id,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected
    )

    assert tuple(name for _line, name in calls) == expected

    bank_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AcsTransferTargetBankStore"
    )
    identity_value = next(
        keyword.value for keyword in bank_call.keywords if keyword.arg == "identity"
    )
    assert isinstance(identity_value, ast.Call)
    assert isinstance(identity_value.func, ast.Name)
    assert identity_value.func.id == "_pool_checkpoint_stage_identity"
    assert isinstance(identity_value.args[0], ast.Name)
    assert identity_value.args[0].id == "checkpoint_identity"
    assert isinstance(identity_value.args[1], ast.Constant)
    assert identity_value.args[1].value == "transferred"

    transfer_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "transfer_acs_inputs"
    )
    target_bank_value = next(
        keyword.value
        for keyword in transfer_call.keywords
        if keyword.arg == "target_bank"
    )
    assert isinstance(target_bank_value, ast.Name)
    assert target_bank_value.id == "target_bank"


def test_pool_imputation_binds_and_publishes_acs_target_bank(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _source_frame()
    checkpoint_identity = {
        "artifact_kind": "fixture-pool-checkpoint-identity",
        "materializer_version": 7,
        "inputs": {"fixture": {"sha256": "b" * 64}},
    }
    base_identity_sha256 = pool_tool._pool_checkpoint_identity_sha256(
        checkpoint_identity
    )
    primary_qrf_dir = tmp_path / "primary-qrf" / base_identity_sha256
    primary_qrf_dir.mkdir(parents=True)
    pool_tool._atomic_write_json(
        pool_tool._primary_qrf_manifest_path(primary_qrf_dir),
        {"artifact_kind": "fixture-primary-manifest"},
    )
    pool_tool._atomic_write_json(
        pool_tool._primary_qrf_input_binding_path(primary_qrf_dir),
        {"artifact_kind": "fixture-primary-binding"},
    )
    acs_bank_dir = tmp_path / "acs-transfer" / base_identity_sha256
    observed: dict[str, object] = {}
    bank_receipt = {
        "artifact_kind": "fixture-acs-target-bank-receipt",
        "targets": {"0": {"source": "checkpoint"}},
    }

    class _RecordingBank:
        def __init__(self, root: Path, *, identity: object) -> None:
            observed["bank"] = self
            observed["root"] = root
            observed["identity"] = identity

        def receipt(self) -> dict[str, object]:
            return bank_receipt

    def fake_transfer(*args, target_bank=None, **kwargs):
        observed["transfer_target_bank"] = target_bank
        return SimpleNamespace(
            frame=args[0],
            fit_records=(),
            resolved_donor_channel="fixture",
            imputed_inputs=(),
            deferred_inputs=(),
        )

    tail_receipt = {
        "tail_distribution_receipts": {
            "frame_after_stage": {
                "positive_mass_five_x_target_exceeded": True,
            }
        }
    }
    monkeypatch.setattr(
        pool_tool,
        "_initialize_or_resume_primary_qrf",
        lambda *args, **kwargs: "resumed",
    )
    monkeypatch.setattr(pool_tool, "run_primary_puf_qrf_chain", lambda *args: None)
    monkeypatch.setattr(
        pool_tool,
        "finalize_primary_puf_qrf_chain",
        lambda *args, **kwargs: (frame, "calibrated"),
    )
    monkeypatch.setattr(
        pool_tool,
        "transfer_puf_capital_gains_tail",
        lambda *args, **kwargs: (frame, tail_receipt),
    )
    monkeypatch.setattr(
        pool_tool,
        "validate_puf_capital_gains_tail_manifest",
        lambda receipt: None,
    )
    monkeypatch.setattr(
        pool_tool,
        "complete_multispine_source_inputs",
        lambda input_frame: SimpleNamespace(frame=input_frame, receipt={}),
    )
    monkeypatch.setattr(
        pool_tool,
        "pool_transfer_target_families",
        lambda: {"person": {"fixture": ("fixture_target",)}},
    )
    monkeypatch.setattr(pool_tool, "AcsTransferTargetBankStore", _RecordingBank)
    monkeypatch.setattr(pool_tool, "transfer_acs_inputs", fake_transfer)

    result = pool_tool._impute_pool(
        frame,
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_bank_dir,
        checkpoint_identity=checkpoint_identity,
        checkpoint_input_binding={"artifact_kind": "fixture-input-binding"},
    )

    assert observed["root"] == acs_bank_dir
    assert observed["identity"] == pool_tool._pool_checkpoint_stage_identity(
        checkpoint_identity,
        "transferred",
    )
    assert observed["transfer_target_bank"] is observed["bank"]
    assert result.receipt["acs_qrf_transfer"]["target_bank"] == bank_receipt


def test_production_pool_wires_source_preparation_into_clone_stage(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool.build_multispine_pool)

    assert "prepare_multispine_source_inputs_for_clone" in source
    assert "prepare_clone=prepare_clone_operator" in source


def test_direct_pool_fixtures_are_operator_free_before_assembly() -> None:
    frame = _source_frame()
    for entities in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.values():
        for entity, columns in entities.items():
            assert set(frame.table(entity)).isdisjoint(columns)


@pytest.mark.parametrize(
    ("family", "entity", "column"),
    [
        (family, entity, sorted(columns)[0])
        for family, entities in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.items()
        for entity, columns in entities.items()
    ],
)
def test_each_operator_output_family_is_rejected_before_assembly(
    pool_tool: ModuleType,
    family: str,
    entity: str,
    column: str,
) -> None:
    frame = _source_frame()
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables[entity][column] = 0.0
    contaminated = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    with pytest.raises(
        ValueError,
        match=rf"fixture {family}.*{column}|fixture.*{column}",
    ):
        pool_tool.assert_operator_free_source_frame(
            contaminated,
            label=f"fixture {family}",
        )


def test_sha_mismatch_refuses_before_loading_or_writing(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_paths = {
        "asec-raw-stage-h5": tmp_path / "asec.h5",
        "acs-household-zip": tmp_path / "household.zip",
        "acs-person-zip": tmp_path / "person.zip",
        "acs-rent-h5": tmp_path / "rent.h5",
        "puf-h5": tmp_path / "puf.h5",
        "puf-source-year-csv": tmp_path / "puf.csv",
    }
    for path in source_paths.values():
        path.write_bytes(b"fixture input")

    called = {"load": False, "write": False}

    def unexpected_load(*_args, **_kwargs):
        called["load"] = True
        raise AssertionError("SHA mismatch must precede source-frame loading.")

    def unexpected_write(*_args, **_kwargs):
        called["write"] = True
        raise AssertionError("SHA mismatch must precede output writing.")

    monkeypatch.setattr(pool_tool, "_load_inputs", unexpected_load)
    monkeypatch.setattr(pool_tool, "_write_outputs", unexpected_write)
    output = tmp_path / "pool.h5"
    argv: list[str] = []
    for option, path in source_paths.items():
        argv.extend([f"--{option}", str(path)])
        digest = (
            pool_tool.ACS_2022_RENT_ARTIFACT_SHA256
            if option == "acs-rent-h5"
            else "0" * 64
        )
        argv.extend([f"--{option}-sha256", digest])
    argv.extend(
        [
            "--puma-ladder",
            str(tmp_path / "puma-ladder.npz"),
            "--puma-ladder-sha256",
            pool_tool._STACKED_PUMA_LADDER_SHA256,
            "--congressional-district-vintage-crosswalk",
            str(tmp_path / "congressional-district-vintage-crosswalk.csv"),
            "--congressional-district-vintage-crosswalk-sha256",
            pool_tool._STACKED_CD_CROSSWALK_SHA256,
        ]
    )
    argv.extend(["--out", str(output)])

    with pytest.raises(ValueError, match="ASEC raw-stage.*SHA-256 mismatch"):
        pool_tool.main(argv)

    assert called == {"load": False, "write": False}
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not output.with_suffix(".agreement.json").exists()


def test_primary_qrf_resume_refuses_a_changed_input_binding(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "primary-qrf"
    checkpoint_dir.mkdir()
    (checkpoint_dir / pool_tool.PRIMARY_QRF_MANIFEST_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    original = {
        "artifact_kind": "fixture_binding",
        "schema_version": 1,
        "inputs": {"processed_puf": {"sha256": "a" * 64}},
    }
    pool_tool._atomic_write_json(
        checkpoint_dir / pool_tool._PRIMARY_QRF_INPUT_BINDING_FILENAME,
        original,
    )
    changed = {
        **original,
        "inputs": {"processed_puf": {"sha256": "b" * 64}},
    }

    with pytest.raises(ValueError, match="refusing to reuse stale predictions"):
        pool_tool._initialize_or_resume_primary_qrf(
            _source_frame(),
            pd.DataFrame(),
            checkpoint_dir,
            input_binding=changed,
        )


@pytest.mark.parametrize(
    ("interruption_label", "durable_target_index"),
    (
        ("j0", 0),
        ("j1", 1),
        ("j2", 2),
        ("mid", 4),
        ("last", 8),
    ),
)
def test_production_transferred_checkpoint_bytes_ignore_bank_interruption_provenance(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interruption_label: str,
    durable_target_index: int,
) -> None:
    active_bank_receipt: dict[str, Mapping[str, object]] = {}
    checkpoint_input_binding = _stub_production_impute_kernels(
        pool_tool,
        monkeypatch,
        active_bank_receipt=active_bank_receipt,
    )

    canonical_acs_receipt_keys = {
        "target_families",
        "n_estimators",
        "max_targets_per_fit",
        "resolved_donor_channel",
        "imputed_inputs",
        "fit_records",
        "deferred_inputs",
    }
    canonical_primary_receipt_keys = {
        "checkpoint_manifest",
        "checkpoint_manifest_sha256",
        "input_binding",
        "input_binding_sha256",
        "n_estimators",
        "tail_bound_diagnostics",
    }
    canonical_impute_receipt_keys = {
        "source_operator_chain",
        "primary_puf_qrf",
        "puf_capital_gains_tail_transfer",
        "acs_qrf_transfer",
        "weights_audit",
    }
    cold_receipt = _target_bank_receipt_after_interruption(None)
    cold_store = _checkpoint_fixture_store(pool_tool, tmp_path / "cold-checkpoints")
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    primary_qrf_dir = tmp_path / "primary-qrf" / cold_store.base_identity_sha256
    acs_transfer_dir = tmp_path / "acs-transfer" / cold_store.base_identity_sha256
    active_bank_receipt["value"] = cold_receipt
    cold_result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=cold_store,
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_transfer_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )
    cold_bytes = cold_store.checkpoint_path("transferred").read_bytes()
    cold_sidecar = pool_tool._read_json_object(
        cold_store.checkpoint_receipts_path("transferred")
    )["operational_stage_receipts"]
    cold_primary_operational = cold_sidecar["impute"]["primary_puf_qrf"]
    assert cold_primary_operational["resume_status"] == "initialized"
    assert cold_primary_operational["identity_routing"]["identity_mismatches"] == []
    cold_target_bank = copy.deepcopy(
        cold_sidecar["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    assert cold_target_bank.pop("identity_routing")["identity_mismatches"] == []
    assert cold_target_bank == cold_receipt
    assert (
        cold_result.stage_receipts["impute"]["primary_puf_qrf"]["resume_status"]
        == "initialized"
    )

    resumed_receipt = _target_bank_receipt_after_interruption(durable_target_index)
    resumed_store = _checkpoint_fixture_store(
        pool_tool,
        tmp_path / f"{interruption_label}-checkpoints",
    )
    resumed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    assert resumed_store.base_identity_sha256 == cold_store.base_identity_sha256
    active_bank_receipt["value"] = resumed_receipt
    resumed_result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=resumed_store,
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_transfer_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )

    transferred_path = resumed_store.checkpoint_path("transferred")
    assert transferred_path.read_bytes() == cold_bytes
    metadata = pool_tool.load_frame_checkpoint(transferred_path).metadata
    canonical_receipts = metadata["stage_receipts"]
    assert set(canonical_receipts) == {"clone", "impute"}
    canonical_impute_receipt = canonical_receipts["impute"]
    assert set(canonical_impute_receipt) == canonical_impute_receipt_keys
    canonical_primary_receipt = canonical_impute_receipt["primary_puf_qrf"]
    assert set(canonical_primary_receipt) == canonical_primary_receipt_keys
    assert "resume_status" not in canonical_primary_receipt
    canonical_acs_receipt = canonical_impute_receipt["acs_qrf_transfer"]
    assert set(canonical_acs_receipt) == canonical_acs_receipt_keys
    assert "target_bank" not in canonical_acs_receipt

    receipts_path = resumed_store.checkpoint_receipts_path("transferred")
    sidecar = pool_tool._read_json_object(receipts_path)
    operational_impute = sidecar["operational_stage_receipts"]["impute"]
    assert operational_impute["primary_puf_qrf"]["resume_status"] == "resumed"
    assert (
        operational_impute["primary_puf_qrf"]["identity_routing"]["identity_mismatches"]
        == []
    )
    resumed_target_bank = copy.deepcopy(
        operational_impute["acs_qrf_transfer"]["target_bank"]
    )
    assert resumed_target_bank.pop("identity_routing")["identity_mismatches"] == []
    assert resumed_target_bank == resumed_receipt

    resumed_store.checkpoint_path("simulated").unlink()
    resumed_store.checkpoint_manifest_path("simulated").unlink()
    warm_store = _checkpoint_fixture_store(
        pool_tool,
        tmp_path / f"{interruption_label}-checkpoints",
    )
    checkpoint = warm_store.load_deepest()
    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert (
        checkpoint.stage_receipts["impute"]["primary_puf_qrf"]["resume_status"]
        == "resumed"
    )
    restored_target_bank = copy.deepcopy(
        checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    restored_target_bank.pop("identity_routing")
    assert restored_target_bank == resumed_receipt
    runtime_target_bank = copy.deepcopy(
        resumed_result.stage_receipts["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    runtime_target_bank.pop("identity_routing")
    assert runtime_target_bank == resumed_receipt
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == "loaded"
    assert receipts_provenance["path"] == str(receipts_path.resolve())


def test_transferred_checkpoint_bytes_ignore_stacked_qrf_manifest_location(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_bytes: list[bytes] = []
    manifest_locations: list[str] = []
    identity_digests: list[str] = []
    for label in ("volume-a", "volume-b"):
        (tmp_path / label).mkdir()
        store = _checkpoint_fixture_store(
            pool_tool,
            tmp_path / label / "checkpoints",
        )
        store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        manifest_path = tmp_path / label / "primary-qrf" / "manifest.json"
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=store,
            primary_qrf_manifest_path=manifest_path,
        )
        transferred_path = store.checkpoint_path("transferred")
        checkpoint_bytes.append(transferred_path.read_bytes())
        metadata = pool_tool.load_frame_checkpoint(transferred_path).metadata
        identity_digests.append(metadata["identity_sha256"])
        assert (
            "checkpoint_manifest_path"
            not in metadata["stage_receipts"]["impute"]["primary_puf_qrf"]
        )
        sidecar = pool_tool._read_json_object(
            store.checkpoint_receipts_path("transferred")
        )
        manifest_locations.append(
            sidecar["operational_stage_receipts"]["impute"]["primary_puf_qrf"][
                "checkpoint_manifest_path"
            ]
        )

    assert checkpoint_bytes[0] == checkpoint_bytes[1]
    assert identity_digests[0] == identity_digests[1]
    assert manifest_locations == [
        str((tmp_path / label / "primary-qrf" / "manifest.json").resolve())
        for label in ("volume-a", "volume-b")
    ]


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_operational_receipts_sidecar_damage_does_not_invalidate_checkpoint(
    pool_tool: ModuleType,
    tmp_path: Path,
    damage: str,
) -> None:
    root = tmp_path / f"{damage}-checkpoints"
    target_bank_receipt = _target_bank_receipt_after_interruption(2)
    cold_store = _checkpoint_fixture_store(pool_tool, root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
        target_bank_receipt=target_bank_receipt,
    )
    transferred_path = cold_store.checkpoint_path("transferred")
    canonical_bytes = transferred_path.read_bytes()
    expected_identity_sha256 = pool_tool.load_frame_checkpoint(
        transferred_path
    ).metadata["identity_sha256"]
    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    receipts_path = cold_store.checkpoint_receipts_path("transferred")
    if damage == "missing":
        receipts_path.unlink()
        expected_status = "missing"
    else:
        receipts_path.write_text("{not-json", encoding="utf-8")
        expected_status = "invalid_ignored"

    warm_store = _checkpoint_fixture_store(pool_tool, root)
    checkpoint = warm_store.load_deepest()

    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert transferred_path.read_bytes() == canonical_bytes
    assert (
        pool_tool.load_frame_checkpoint(transferred_path).metadata["identity_sha256"]
        == expected_identity_sha256
    )
    assert "target_bank" not in checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == expected_status


def test_same_identity_rewrite_cannot_reattach_stale_operational_receipts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "rewrite-checkpoints"
    receipt_a = _target_bank_receipt_after_interruption(0)
    cold_store = _checkpoint_fixture_store(pool_tool, root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
        target_bank_receipt=receipt_a,
    )
    receipts_path = cold_store.checkpoint_receipts_path("transferred")
    assert receipts_path.is_file()
    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    cold_store.checkpoint_receipts_path("simulated").unlink()

    real_atomic_write_json = pool_tool._atomic_write_json

    def interrupt_before_receipts_install(path: Path, payload: Mapping[str, object]):
        if Path(path) == receipts_path:
            assert not receipts_path.exists()
            raise RuntimeError("fixture crash before fresh receipts install")
        return real_atomic_write_json(path, payload)

    monkeypatch.setattr(
        pool_tool,
        "_atomic_write_json",
        interrupt_before_receipts_install,
    )
    rewrite_store = _checkpoint_fixture_store(pool_tool, root)
    rewrite_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    with pytest.raises(RuntimeError, match="before fresh receipts install"):
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=rewrite_store,
            target_bank_receipt=_target_bank_receipt_after_interruption(4),
        )
    assert not receipts_path.exists()

    warm_store = _checkpoint_fixture_store(pool_tool, root)
    checkpoint = warm_store.load_deepest()

    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert "target_bank" not in checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == "missing"


@pytest.mark.parametrize(
    ("resume_stage", "expected_order"),
    (
        ("assembled", ["impute", "derive", "seed", "simulate"]),
        ("transferred", ["derive", "seed", "simulate"]),
        ("simulated", []),
    ),
)
def test_pool_checkpoint_round_trip_resumes_each_boundary_byte_identically(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    resume_stage: str,
    expected_order: list[str],
) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    uninterrupted, cold_order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
    )
    assert cold_order == ["impute", "derive", "seed", "simulate"]
    cold_output = capsys.readouterr().out
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert cold_output.count(f"Rebuilt pool stage {stage!r}") == 1
    checkpoint_identity_sha256: dict[str, str] = {}
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        stored_metadata = pool_tool.load_frame_checkpoint(
            cold_store.checkpoint_path(stage)
        ).metadata
        checkpoint_identity_sha256[stage] = stored_metadata["identity_sha256"]
        assert "agreement_gate" not in stored_metadata
        assert "simulation_ready" not in stored_metadata
        assert "agreement" not in stored_metadata
        assert "agreement" not in stored_metadata["stage_receipts"]

    resume_index = pool_tool.POOL_CHECKPOINT_STAGE_ORDER.index(resume_stage)
    for deeper_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[resume_index + 1 :]:
        cold_store.checkpoint_path(deeper_stage).unlink()
        cold_store.checkpoint_manifest_path(deeper_stage).unlink()

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()
    assert resume is not None
    assert resume.stage == resume_stage
    resumed, warm_order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )

    assert warm_order == expected_order
    warm_output = capsys.readouterr().out
    cached_stages = pool_tool.POOL_CHECKPOINT_STAGE_ORDER[: resume_index + 1]
    assert (
        f"Resumed pool checkpoint {resume_stage!r} from " in warm_output
        and f"cached stages: {', '.join(cached_stages)}." in warm_output
    )
    for stage_index, stage in enumerate(pool_tool.POOL_CHECKPOINT_STAGE_ORDER):
        assert warm_output.count(f"Rebuilt pool stage {stage!r}") == (
            0 if stage_index <= resume_index else 1
        )
    assert resumed.assembly_receipt == uninterrupted.assembly_receipt
    assert resumed.provenance_counts == uninterrupted.provenance_counts
    assert resumed.stage_receipts == uninterrupted.stage_receipts
    assert pool_tool._json_ready(resumed.frame.metadata) == pool_tool._json_ready(
        uninterrupted.frame.metadata
    )
    for entity in uninterrupted.frame.entities:
        pd.testing.assert_frame_equal(
            resumed.frame.table(entity),
            uninterrupted.frame.table(entity),
            check_dtype=True,
            check_exact=True,
        )
        string_columns = _semantic_string_columns(uninterrupted.frame.table(entity))
        assert string_columns
        assert all(
            uninterrupted.frame.table(entity)[column].dtype == CANONICAL_STRING_DTYPE
            for column in string_columns
        )

    uninterrupted_pool_path = tmp_path / f"{resume_stage}.uninterrupted.pool.h5"
    resumed_pool_path = tmp_path / f"{resume_stage}.resumed.pool.h5"
    for path, result in (
        (uninterrupted_pool_path, uninterrupted),
        (resumed_pool_path, resumed),
    ):
        pool_tool.write_nullable_us_h5(
            result.frame,
            path,
            period=pool_tool.POOL_TIME_PERIOD,
            artifact_kind=pool_tool.POOL_H5_ARTIFACT_KIND,
            publication_run_id="fixture-publication",
        )
    with (
        pd.HDFStore(uninterrupted_pool_path, mode="r") as uninterrupted_store,
        pd.HDFStore(resumed_pool_path, mode="r") as resumed_store,
    ):
        assert resumed_store.keys() == uninterrupted_store.keys()
        for key in uninterrupted_store.keys():
            expected = uninterrupted_store[key]
            observed = resumed_store[key]
            if isinstance(expected, pd.DataFrame):
                pd.testing.assert_frame_equal(
                    observed,
                    expected,
                    check_dtype=True,
                    check_exact=True,
                )
            else:
                pd.testing.assert_series_equal(
                    observed,
                    expected,
                    check_dtype=True,
                    check_exact=True,
                )

    # PyTables' published container carries nondeterministic HDF metadata. The
    # deterministic Frame serializer gives the literal byte-level assertion
    # over the exact input-pool state after the production writer is checked.
    uninterrupted_canonical = tmp_path / f"{resume_stage}.uninterrupted.canonical.h5"
    resumed_canonical = tmp_path / f"{resume_stage}.resumed.canonical.h5"
    pool_tool.write_frame_checkpoint(uninterrupted_canonical, uninterrupted.frame)
    pool_tool.write_frame_checkpoint(resumed_canonical, resumed.frame)
    assert resumed_canonical.read_bytes() == uninterrupted_canonical.read_bytes()

    provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] == resume_stage
    assert provenance["base_identity_sha256"] == warm_store.base_identity_sha256
    assert (
        provenance["primary_qrf"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    assert (
        provenance["acs_transfer"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    assert provenance["acs_transfer"]["boundary_stage"] == "transferred"
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert (
            provenance["stages"][stage]["identity_sha256"]
            == checkpoint_identity_sha256[stage]
        )
    assert provenance["stages"][resume_stage]["source"] == "checkpoint"
    assert provenance["stages"][resume_stage]["resume_kind"] == "direct"
    for covered_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[:resume_index]:
        covered = provenance["stages"][covered_stage]
        assert covered["source"] == "checkpoint"
        assert covered["resume_kind"] == "covered_by_deeper_checkpoint"
        assert covered["source_checkpoint_stage"] == resume_stage
        assert covered["path"] == str(
            warm_store.checkpoint_path(resume_stage).resolve()
        )
        assert covered["nominal_stage_path"] == str(
            warm_store.checkpoint_path(covered_stage).resolve()
        )
    assert provenance["agreement"] == {
        "source": "always_fresh",
        "cached": False,
        "terminal_verdict_persisted": False,
    }


def test_pool_checkpoint_store_round_trips_nullable_boolean_families(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "nullable-boolean-checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())

    _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
        checkpoint_nullable_booleans=True,
    )

    h5py = pytest.importorskip("h5py")
    for stage, expected_schema in (
        ("assembled", 2),
        ("transferred", 3),
        ("simulated", 3),
    ):
        path = cold_store.checkpoint_path(stage)
        with h5py.File(path, mode="r") as h5:
            raw = np.asarray(h5["_populace_frame_checkpoint/metadata_json"]).tobytes()
        assert json.loads(raw)["schema_version"] == expected_schema
        manifest = pool_tool._read_json_object(
            cold_store.checkpoint_manifest_path(stage)
        )
        assert manifest["materializer_version"] == 7
        loaded = pool_tool.load_frame_checkpoint(path).frame
        if stage == "assembled":
            assert "fixture_declared_boolean" not in loaded.person
            continue
        assert loaded.person["is_female"].dtype == pd.BooleanDtype()
        assert loaded.person["fixture_declared_boolean"].dtype == pd.BooleanDtype()
        assert not loaded.person["is_female"].isna().any()
        assert loaded.person["fixture_declared_boolean"].isna().sum() == 1

    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resumed = warm_store.load_deepest()

    assert resumed is not None
    assert resumed.stage == "transferred"
    assert resumed.frame.person["is_female"].dtype == pd.BooleanDtype()
    assert resumed.frame.person["fixture_declared_boolean"].isna().sum() == 1


def test_simulated_v7_checkpoint_accepts_both_string_encodings_without_rewrite(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    """V7 authenticates both physical string encodings as one logical frame."""

    pytest.importorskip("h5py")
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    checkpoint_path = cold_store.checkpoint_path("simulated")
    loaded = pool_tool.load_frame_checkpoint(checkpoint_path)
    canonical_v2_bytes = checkpoint_path.read_bytes()
    canonical_identity = loaded.metadata["identity"]
    assert loaded.metadata["materializer_version"] == 7
    assert any(
        column["dtype"] == str(CANONICAL_STRING_DTYPE)
        for columns in loaded.metadata["frame_schema"]["entities"].values()
        for column in columns
    )

    canonical_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    canonical_resume = canonical_store.load_deepest()
    assert canonical_resume is not None
    assert canonical_resume.stage == "simulated"
    assert canonical_resume.simulation_frame is not None
    assert checkpoint_path.read_bytes() == canonical_v2_bytes

    legacy_frame = _with_object_backed_strings(loaded.frame)
    legacy_metadata = dict(loaded.metadata)
    legacy_metadata["frame_schema"] = pool_tool._frame_schema_payload(legacy_frame)
    pool_tool.write_frame_checkpoint(
        checkpoint_path,
        legacy_frame,
        metadata=legacy_metadata,
    )
    manifest_path = cold_store.checkpoint_manifest_path("simulated")
    manifest = pool_tool._read_json_object(manifest_path)
    manifest["checkpoint"]["sha256"] = pool_tool._file_sha256(checkpoint_path)
    manifest["checkpoint"]["size_bytes"] = checkpoint_path.stat().st_size
    manifest["frame_schema"] = legacy_metadata["frame_schema"]
    pool_tool._atomic_write_json(manifest_path, manifest)
    banked_v2_bytes = checkpoint_path.read_bytes()
    assert banked_v2_bytes != canonical_v2_bytes
    assert legacy_metadata["identity"] == canonical_identity
    assert legacy_metadata["materializer_version"] == 7
    assert any(
        column["dtype"] == "object"
        for columns in legacy_metadata["frame_schema"]["entities"].values()
        for column in columns
    )

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "simulated"
    assert resume.simulation_frame is not None
    assert checkpoint_path.read_bytes() == banked_v2_bytes
    assert (
        warm_store.provenance(
            primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        )["stages"]["simulated"]["load_status"]
        == "resumed"
    )
    for entity in US_SCHEMA.entities:
        canonical_table = canonical_resume.frame.table(entity)
        table = resume.frame.table(entity)
        pd.testing.assert_frame_equal(table, canonical_table, check_exact=True)
        pd.testing.assert_frame_equal(
            resume.simulation_frame.table(entity),
            canonical_resume.simulation_frame.table(entity),
            check_exact=True,
        )
        string_columns = _semantic_string_columns(table)
        assert string_columns
        assert all(
            table[column].dtype == CANONICAL_STRING_DTYPE for column in string_columns
        )


def test_resumed_checkpoint_provenance_is_published_in_final_manifest(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    _unused, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    base_identity = pool_tool._pool_checkpoint_base_identity(
        verified_inputs,
        policyengine_us_version="fixture-engine-1",
    )
    cold_store = pool_tool._PoolStageCheckpointStore(
        outputs.checkpoint_root,
        base_identity=base_identity,
    )
    cold_store.bind_input_receipts(pool_tool._loaded_input_receipts(loaded))
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    warm_store = pool_tool._PoolStageCheckpointStore(
        outputs.checkpoint_root,
        base_identity=base_identity,
    )
    resume = warm_store.load_deepest()
    assert resume is not None
    assert resume.stage == "simulated"
    resumed, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )
    assert order == []
    checkpoint_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=outputs.primary_qrf_checkpoint_dir,
    )

    pool_tool._write_outputs(
        resumed,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        input_receipts=warm_store.input_receipts,
        checkpoint_provenance=checkpoint_provenance,
    )

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["stage_checkpoints"] == checkpoint_provenance
    assert (
        manifest["stage_checkpoints"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        stage_manifest = pool_tool._read_json_object(
            warm_store.checkpoint_manifest_path(stage)
        )
        assert (
            manifest["stage_checkpoints"]["stages"][stage]["identity_sha256"]
            == stage_manifest["identity_sha256"]
        )


def test_pool_checkpoint_input_sha_mismatch_rebuilds_every_stage(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    changed_store = _checkpoint_fixture_store(
        pool_tool,
        checkpoint_root,
        changed_role="processed_puf",
    )
    assert changed_store.load_deepest() is None
    changed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=changed_store,
    )

    assert order == ["impute", "derive", "seed", "simulate"]
    assert not rebuilt.simulation_ready
    output = capsys.readouterr().out
    assert output.count("Ignored stale pool checkpoint") == 3
    provenance = changed_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] is None
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert provenance["stages"][stage]["source"] == "rebuilt"
        assert provenance["stages"][stage]["load_status"] == "identity_mismatch"


@pytest.mark.parametrize(
    "contract_field",
    (
        pytest.param("asserted_constraint", id="asserted_constraint_changed"),
        pytest.param(
            "inventory_built_against",
            id="inventory_built_against_changed",
        ),
    ),
)
def test_production_bank_routing_names_stale_contract_identity_siblings(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract_field: str,
) -> None:
    checkpoint_root = tmp_path / "production-routing-checkpoints"
    base_outputs = pool_tool._output_paths(
        tmp_path / "pool.h5",
        checkpoint_root=checkpoint_root,
    )
    contract = load_take_up_contract()
    current_contract_identity = take_up_contract_identity(contract)
    stale_contract = replace(
        contract,
        **{contract_field: f"{getattr(contract, contract_field)}-stale"},
    )
    stale_contract_identity = take_up_contract_identity(stale_contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: stale_contract_identity,
    )
    stale_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    stale_outputs = pool_tool._with_checkpoint_identity(
        base_outputs,
        base_identity_sha256=stale_store.base_identity_sha256,
    )
    stale_markers = []
    for bank_dir in (
        stale_outputs.primary_qrf_checkpoint_dir,
        stale_outputs.acs_transfer_checkpoint_dir,
    ):
        bank_dir.mkdir(parents=True)
        marker = bank_dir / "must-not-be-opened"
        marker.write_text("stale bank marker\n", encoding="utf-8")
        stale_markers.append(marker)

    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: current_contract_identity,
    )
    current_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    current_outputs = pool_tool._with_checkpoint_identity(
        base_outputs,
        base_identity_sha256=current_store.base_identity_sha256,
    )
    assert current_store.base_identity_sha256 != stale_store.base_identity_sha256
    assert (
        current_outputs.primary_qrf_checkpoint_dir
        != stale_outputs.primary_qrf_checkpoint_dir
    )
    assert (
        current_outputs.acs_transfer_checkpoint_dir
        != stale_outputs.acs_transfer_checkpoint_dir
    )

    checkpoint_input_binding = _stub_production_impute_kernels(
        pool_tool,
        monkeypatch,
    )
    current_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=current_store,
        primary_qrf_checkpoint_dir=current_outputs.primary_qrf_checkpoint_dir,
        acs_transfer_checkpoint_dir=current_outputs.acs_transfer_checkpoint_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )

    operational_impute = pool_tool._read_json_object(
        current_store.checkpoint_receipts_path("transferred")
    )["operational_stage_receipts"]["impute"]
    assert operational_impute["primary_puf_qrf"]["resume_status"] == "initialized"
    primary_routing = operational_impute["primary_puf_qrf"]["identity_routing"]
    target_bank = operational_impute["acs_qrf_transfer"]["target_bank"]
    acs_routing = target_bank["identity_routing"]
    assert target_bank["root"] == str(
        current_outputs.acs_transfer_checkpoint_dir.resolve()
    )
    assert target_bank["targets"] == {}

    for routing, bank_root, selected_path, stale_path in (
        (
            primary_routing,
            current_outputs.primary_qrf_checkpoint_dir.parent,
            current_outputs.primary_qrf_checkpoint_dir,
            stale_outputs.primary_qrf_checkpoint_dir,
        ),
        (
            acs_routing,
            current_outputs.acs_transfer_checkpoint_dir.parent,
            current_outputs.acs_transfer_checkpoint_dir,
            stale_outputs.acs_transfer_checkpoint_dir,
        ),
    ):
        assert routing["bank_root"] == str(bank_root.resolve())
        assert routing["selected_path"] == str(selected_path.resolve())
        assert (
            routing["current_base_identity_sha256"]
            == current_store.base_identity_sha256
        )
        assert routing["scan"] == {
            "limit": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
            "entries_examined": 1,
            "truncated": False,
        }
        assert routing["identity_mismatches"] == [
            {
                "load_status": "identity_mismatch",
                "stale_base_identity_sha256": stale_store.base_identity_sha256,
                "current_base_identity_sha256": current_store.base_identity_sha256,
                "disposition": "bypassed",
                "path": str(stale_path.resolve()),
            }
        ]
    assert all(
        marker.read_text(encoding="utf-8") == "stale bank marker\n"
        for marker in stale_markers
    )
    assert (
        result.stage_receipts["impute"]["primary_puf_qrf"]["identity_routing"]
        == primary_routing
    )


@pytest.mark.parametrize(
    "contract_field",
    (
        pytest.param("asserted_constraint", id="asserted_constraint_changed"),
        pytest.param(
            "inventory_built_against",
            id="inventory_built_against_changed",
        ),
    ),
)
def test_take_up_contract_identity_mutation_rebuilds_every_pool_boundary(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract_field: str,
) -> None:
    checkpoint_root = tmp_path / "take-up-contract-checkpoints"
    contract = load_take_up_contract()
    original_contract_identity = take_up_contract_identity(contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: original_contract_identity,
    )
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    assert (
        cold_store.base_identity["pool_code"]["take_up_contract"]
        == original_contract_identity
    )
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    changed_contract = replace(
        contract,
        **{contract_field: f"{getattr(contract, contract_field)}-changed"},
    )
    changed_contract_identity = take_up_contract_identity(changed_contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: changed_contract_identity,
    )
    changed_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)

    assert (
        changed_store.base_identity["pool_code"]["take_up_contract"]
        == changed_contract_identity
    )
    assert changed_store.base_identity_sha256 != cold_store.base_identity_sha256
    assert changed_store.load_deepest() is None
    changed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=changed_store,
    )

    assert order == ["impute", "derive", "seed", "simulate"]
    assert not rebuilt.simulation_ready
    provenance = changed_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] is None
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert provenance["stages"][stage]["source"] == "rebuilt"
        assert provenance["stages"][stage]["load_status"] == "identity_mismatch"


def test_tail_support_contract_identity_mutation_rebuilds_pool_checkpoints(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "tail-support-contract-checkpoints"
    original = pool_tool.puf_capital_gains_tail_support_contract_identity()
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    assert (
        cold_store.base_identity["pool_code"]["puf_capital_gains_tail_support_contract"]
        == original
    )
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    changed = copy.deepcopy(original)
    changed["insufficient_support_action"] = "silently_widen"
    monkeypatch.setattr(
        pool_tool,
        "puf_capital_gains_tail_support_contract_identity",
        lambda: changed,
    )
    changed_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)

    assert (
        changed_store.base_identity["pool_code"][
            "puf_capital_gains_tail_support_contract"
        ]
        == changed
    )
    assert changed_store.base_identity_sha256 != cold_store.base_identity_sha256
    assert changed_store.load_deepest() is None


@pytest.mark.parametrize("legacy_version", (1, 2, 3, 4, 5, 6))
def test_legacy_pool_materializer_artifacts_fail_closed_with_named_receipts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    legacy_version: int,
) -> None:
    checkpoint_root = tmp_path / "legacy-materializer-checkpoints"

    with monkeypatch.context() as legacy:
        legacy.setattr(
            pool_tool,
            "POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION",
            legacy_version,
        )
        legacy_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
        assert legacy_store.base_identity["materializer_version"] == legacy_version
        legacy_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        _run_checkpoint_fixture(pool_tool, tmp_path, store=legacy_store)
        for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
            metadata = pool_tool.load_frame_checkpoint(
                legacy_store.checkpoint_path(stage)
            ).metadata
            manifest = pool_tool._read_json_object(
                legacy_store.checkpoint_manifest_path(stage)
            )
            assert metadata["materializer_version"] == legacy_version
            assert metadata["identity"]["materializer_version"] == legacy_version
            assert manifest["materializer_version"] == legacy_version
            assert manifest["identity"]["materializer_version"] == legacy_version
    capsys.readouterr()

    assert pool_tool.POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION == 7
    current_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    assert current_store.base_identity["materializer_version"] == 7
    assert current_store.load_deepest() is None

    output = capsys.readouterr().out
    provenance = current_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert f"Ignored corrupt pool checkpoint '{stage}'" in output
        receipt = provenance["stages"][stage]
        assert receipt["source"] == "rebuilt"
        assert receipt["load_status"] == "invalid_rebuild"
        invalid = receipt["invalid_checkpoint"]
        assert invalid["reason"] == "checkpoint_validation_failed"
        assert invalid["message"] == (
            f"{stage} checkpoint manifest has an unsupported binding"
        )


@pytest.mark.parametrize(
    ("corrupt_stage", "expected_resume", "expected_order"),
    (
        (
            "assembled",
            None,
            ["impute", "derive", "seed", "simulate"],
        ),
        (
            "transferred",
            "assembled",
            ["impute", "derive", "seed", "simulate"],
        ),
        (
            "simulated",
            "transferred",
            ["derive", "seed", "simulate"],
        ),
    ),
)
def test_corrupt_pool_checkpoint_names_boundary_and_rebuilds(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    corrupt_stage: str,
    expected_resume: str | None,
    expected_order: list[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    baseline, _order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
    )
    capsys.readouterr()

    corrupt_index = pool_tool.POOL_CHECKPOINT_STAGE_ORDER.index(corrupt_stage)
    for deeper_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[corrupt_index + 1 :]:
        cold_store.checkpoint_path(deeper_stage).unlink()
        cold_store.checkpoint_manifest_path(deeper_stage).unlink()
    cold_store.checkpoint_path(corrupt_stage).write_bytes(b"corrupt checkpoint")

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()
    assert (None if resume is None else resume.stage) == expected_resume
    if resume is None:
        warm_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )

    assert order == expected_order
    output = capsys.readouterr().out
    assert f"Ignored corrupt pool checkpoint '{corrupt_stage}'" in output
    assert "SHA-256 mismatch" in output
    for entity in baseline.frame.entities:
        pd.testing.assert_frame_equal(
            rebuilt.frame.table(entity),
            baseline.frame.table(entity),
            check_dtype=True,
            check_exact=True,
        )
    provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    corrupt_receipt = provenance["stages"][corrupt_stage]
    assert corrupt_receipt["source"] == "rebuilt"
    assert corrupt_receipt["load_status"] == "invalid_rebuild"
    assert corrupt_receipt["invalid_checkpoint"]["reason"] == (
        "checkpoint_validation_failed"
    )


@pytest.mark.parametrize(
    ("drift_field", "expected_error"),
    (
        ("row_counts", "row_counts differs from its sidecar"),
        ("frame_schema", "frame_schema differs from its sidecar"),
    ),
)
def test_checkpoint_sidecar_shape_drift_names_failure_and_falls_back(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    drift_field: str,
    expected_error: str,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    transferred_manifest_path = cold_store.checkpoint_manifest_path("transferred")
    transferred_manifest = pool_tool._read_json_object(transferred_manifest_path)
    if drift_field == "row_counts":
        transferred_manifest["row_counts"]["person"] += 1
    else:
        transferred_manifest["frame_schema"]["entities"]["person"][0]["dtype"] = (
            "corrupt-dtype"
        )
    pool_tool._atomic_write_json(transferred_manifest_path, transferred_manifest)

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "assembled"
    output = capsys.readouterr().out
    assert "Ignored corrupt pool checkpoint 'transferred'" in output
    assert expected_error in output
    transferred_receipt = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )["stages"]["transferred"]
    assert transferred_receipt["load_status"] == "invalid_rebuild"
    assert transferred_receipt["invalid_checkpoint"]["reason"] == (
        "checkpoint_validation_failed"
    )


def test_invalid_deep_checkpoint_receipts_do_not_poison_valid_fallback(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    simulated_path = cold_store.checkpoint_path("simulated")
    simulated = pool_tool.load_frame_checkpoint(simulated_path)
    poisoned_metadata = dict(simulated.metadata)
    poisoned_metadata["input_receipts"] = {"poisoned": True}
    poisoned_metadata["simulation_output"] = None
    pool_tool.write_frame_checkpoint(
        simulated_path,
        simulated.frame,
        metadata=poisoned_metadata,
    )
    simulated_manifest_path = cold_store.checkpoint_manifest_path("simulated")
    simulated_manifest = pool_tool._read_json_object(simulated_manifest_path)
    simulated_manifest["checkpoint"]["sha256"] = pool_tool._file_sha256(simulated_path)
    simulated_manifest["checkpoint"]["size_bytes"] = simulated_path.stat().st_size
    pool_tool._atomic_write_json(simulated_manifest_path, simulated_manifest)

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "transferred"
    assert warm_store.input_receipts == _checkpoint_fixture_input_receipts()
    output = capsys.readouterr().out
    assert "Ignored corrupt pool checkpoint 'simulated'" in output
    assert "invalid SSI output binding" in output


def test_durable_checkpoint_write_rejects_forged_qbi_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    store.bind_input_receipts(_checkpoint_fixture_input_receipts())

    def forge_after_emission(checkpoint: MultispinePoolCheckpoint) -> None:
        if checkpoint.stage != "simulated":
            store.write(checkpoint)
            return
        receipts = copy.deepcopy(checkpoint.stage_receipts)
        receipts["derive"]["qbi_input_reconciliation"]["sha256"] = "0" * 64
        store.write(replace(checkpoint, stage_receipts=receipts))

    with pytest.raises(
        ValueError,
        match=(
            "pool simulated durable checkpoint write: QBI reconciliation "
            "receipt SHA-256"
        ),
    ):
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=SimpleNamespace(write=forge_after_emission),
        )

    assert not store.checkpoint_path("simulated").exists()


def test_durable_checkpoint_write_rejects_forged_qbi_transition_authority(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    store.bind_input_receipts(_checkpoint_fixture_input_receipts())

    def forge_after_emission(checkpoint: MultispinePoolCheckpoint) -> None:
        if checkpoint.stage != "simulated":
            store.write(checkpoint)
            return
        store.write(
            replace(
                checkpoint,
                qbi_transition_authority_sha256="0" * 64,
            )
        )

    with pytest.raises(
        ValueError,
        match="independently carried transition authority",
    ):
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=SimpleNamespace(write=forge_after_emission),
        )

    assert not store.checkpoint_path("simulated").exists()


def test_durable_checkpoint_load_rejects_forged_qbi_receipt_and_falls_back(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    simulated_path = cold_store.checkpoint_path("simulated")
    simulated = pool_tool.load_frame_checkpoint(simulated_path)
    poisoned_metadata = copy.deepcopy(simulated.metadata)
    poisoned_metadata["stage_receipts"]["derive"]["qbi_input_reconciliation"][
        "sha256"
    ] = "0" * 64
    pool_tool.write_frame_checkpoint(
        simulated_path,
        simulated.frame,
        metadata=poisoned_metadata,
    )
    simulated_manifest_path = cold_store.checkpoint_manifest_path("simulated")
    simulated_manifest = pool_tool._read_json_object(simulated_manifest_path)
    simulated_manifest["checkpoint"]["sha256"] = pool_tool._file_sha256(simulated_path)
    simulated_manifest["checkpoint"]["size_bytes"] = simulated_path.stat().st_size
    pool_tool._atomic_write_json(simulated_manifest_path, simulated_manifest)

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "transferred"
    output = capsys.readouterr().out
    assert "Ignored corrupt pool checkpoint 'simulated'" in output
    assert (
        "pool simulated durable checkpoint load: QBI reconciliation receipt SHA-256"
    ) in output


def test_synthetic_two_spine_path_reaches_fixed_red_terminal_gate(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    result = _red_pool_result(pool_tool, tmp_path)

    assert result.assembly_receipt["channels"] == ["asec", "acs"]
    assert result.assembly_receipt["native_row_counts"]["person"] == {
        "asec": 2,
        "acs": 2,
    }
    assert result.provenance_counts["person"] == {
        "rows": 8,
        "by_source_channel": {"asec": 4, "acs": 4},
        "by_clone_index": {"0": 4, "1": 4},
        "by_source_channel_and_clone_index": {
            "asec": {"0": 2, "1": 2},
            "acs": {"0": 2, "1": 2},
        },
    }
    assert result.stage_receipts == {
        stage: {"fixture_stage": stage}
        for stage in ("impute", "derive", "seed", "simulate")
    }
    assert any(
        "person/simulated_output/ssi/acs_vs_asec" in failure
        for failure in result.agreement_gate.failures
    )


def test_wired_path_uses_real_raw_preserving_transfer_before_gate(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    transfer_receipts = []

    def impute(frame: Frame) -> PoolStageOutput:
        person = frame.table("person").copy()
        person["age"] = pd.to_numeric(person["A_AGE"], errors="raise")
        person["is_female"] = pd.to_numeric(person["A_SEX"], errors="raise") == 2
        prepared = _replace_person(frame, person)
        transferred = transfer_acs_inputs(
            prepared,
            prepared,
            target_families={"person": {"fixture": ("fixture_transfer",)}},
            seed=0,
            n_estimators=3,
        )
        transfer_receipts.extend(transferred.imputed_inputs)
        return PoolStageOutput(
            transferred.frame,
            {"fit_records": list(transferred.fit_records)},
        )

    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def simulate(frame: Frame) -> PoolStageOutput:
        person = frame.table("person").copy()
        person["ssi"] = person["fixture_transfer"]
        return PoolStageOutput(_replace_person(frame, person))

    result = pool_tool.build_multispine_pool(
        _transfer_source_frame([10.0, 20.0]),
        _transfer_source_frame([np.nan, np.nan]),
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        impute=impute,
        derive=no_op,
        seed=no_op,
        simulate=simulate,
    )

    person = result.frame.table("person")
    channels = person[support_channel_column("person")]
    assert sorted(person.loc[channels.eq("asec"), "fixture_transfer"]) == [
        10.0,
        10.0,
        20.0,
        20.0,
    ]
    assert person.loc[channels.eq("acs"), "fixture_transfer"].tolist() == [
        15.0,
        15.0,
        15.0,
        15.0,
    ]
    assert sum(item.imputed_recipient_rows for item in transfer_receipts) == 4
    assert not result.agreement_gate.passed
    assert result.agreement_gate.name == "us_spine_agreement"
    assert "ssi" not in person


def test_manifest_and_publication_reject_forged_qbi_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    receipts = copy.deepcopy(result.stage_receipts)
    receipts["derive"]["qbi_input_reconciliation"]["sha256"] = "0" * 64
    forged = replace(result, stage_receipts=receipts)

    with pytest.raises(
        ValueError,
        match=("legacy production manifest: QBI reconciliation receipt SHA-256"),
    ):
        pool_tool._manifest_payload(
            result=forged,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            input_receipts={},
            checkpoint_provenance={},
            publication_run_id="forged-manifest",
        )

    with pytest.raises(
        ValueError,
        match="legacy publication entry: QBI reconciliation receipt SHA-256",
    ):
        pool_tool._write_outputs(
            forged,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_stacked_manifest_and_publication_reject_forged_qbi_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    legacy, _outputs, verified_inputs, source_manifest, _loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    receipt = copy.deepcopy(legacy.stage_receipts["derive"]["qbi_input_reconciliation"])
    receipt["sha256"] = "0" * 64
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool, legacy.frame
    )
    stacked = SimpleNamespace(
        frame=authorized,
        qbi_transition_authority_sha256=(legacy.qbi_transition_authority_sha256),
        late_producer_transition_authority_sha256=transition_authority_sha256,
        stage_receipts={
            "impute": impute,
            "derive": {"pool_derivation": {"qbi_input_reconciliation": receipt}},
        },
    )
    outputs = pool_tool._stacked_output_paths(tmp_path / "stacked-pool.h5")

    with pytest.raises(
        ValueError,
        match=("stacked production manifest: QBI reconciliation receipt SHA-256"),
    ):
        pool_tool._stacked_manifest_payload(
            result=stacked,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            input_receipts={},
            checkpoint_provenance={},
            publication_run_id="forged-stacked-manifest",
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    with pytest.raises(
        ValueError,
        match="stacked publication entry: QBI reconciliation receipt SHA-256",
    ):
        pool_tool._write_stacked_outputs(
            stacked,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            input_receipts={},
            checkpoint_provenance={},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_publication_rejects_mutated_qbi_output_with_regenerated_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    person = result.frame.table("person").copy()
    person.loc[person.index[0], "non_qualified_dividend_income"] = 100.0
    person.loc[person.index[0], "qualified_bdc_income"] = 50.0
    mutated_frame = _replace_person(result.frame, person)
    receipts = copy.deepcopy(result.stage_receipts)
    receipts["derive"]["qbi_input_reconciliation"] = (
        us_qbi_reconciliation_change_receipt(mutated_frame, mutated_frame)
    )
    forged = replace(
        result,
        frame=mutated_frame,
        stage_receipts=receipts,
    )

    with pytest.raises(
        ValueError,
        match=(
            "legacy production manifest: QBI receipt differs from the "
            "independently carried transition authority"
        ),
    ):
        pool_tool._manifest_payload(
            result=forged,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            input_receipts={},
            checkpoint_provenance={},
            publication_run_id="reissued-qbi-manifest",
        )

    with pytest.raises(
        ValueError,
        match=(
            "legacy publication entry: QBI receipt differs from the "
            "independently carried transition authority"
        ),
    ):
        pool_tool._write_outputs(
            forged,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_red_outputs_preserve_receipts_and_exclude_simulation_output(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )

    pool_tool._write_outputs(
        result,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    diagnostics = json.loads(outputs.agreement_diagnostics.read_text(encoding="utf-8"))
    expected_gate = GateReport((result.agreement_gate,)).to_manifest()

    assert manifest["status"] == "agreement_failed"
    assert manifest["simulation_ready"] is False
    assert manifest["calibration_applied"] is False
    assert manifest["calibration"]["applied"] is False
    assert manifest["assembly_receipt"] == result.assembly_receipt
    assert manifest["provenance_counts"] == result.provenance_counts
    assert manifest["stage_receipts"] == result.stage_receipts
    assert manifest["stage_checkpoints"]["enabled"] is False
    assert manifest["stage_checkpoints"]["agreement"] == {
        "source": "always_fresh",
        "cached": False,
        "terminal_verdict_persisted": False,
    }
    assert manifest["agreement_gate"] == expected_gate
    assert diagnostics["agreement_gate"] == expected_gate
    assert diagnostics["simulation_ready"] is False
    assert diagnostics["publication_run_id"] == manifest["publication_run_id"]
    assert manifest["provenance_pins"] == {
        role: pin.to_manifest() for role, pin in verified_inputs.items()
    }
    assert manifest["pool_h5"]["formula_outputs_persisted"] is False
    assert manifest["pool_h5"]["input_only"] is True
    assert manifest["pool_h5"]["publication_run_id"] == manifest["publication_run_id"]
    assert (
        manifest["pool_h5"]["sha256"]
        == hashlib.sha256(outputs.pool_h5.read_bytes()).hexdigest()
    )

    with pd.HDFStore(outputs.pool_h5, mode="r") as store:
        assert "ssi" not in store["person"].columns
        metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
    assert metadata["publication_run_id"] == manifest["publication_run_id"]
    assert "materializer_version" not in metadata


def test_ready_reader_binds_manifest_h5_and_diagnostics_to_one_run(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    ready = replace(
        result,
        agreement_gate=GateResult("us_spine_agreement", True),
    )
    pool_tool._write_outputs(
        ready,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )

    manifest = pool_tool.load_simulation_ready_us_multispine_pool_manifest(
        outputs.manifest
    )

    assert manifest["simulation_ready"] is True
    assert (
        manifest["agreement_diagnostics"]["publication_run_id"]
        == manifest["publication_run_id"]
    )


def test_ready_reader_rejects_manifest_h5_run_id_mismatch(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    ready = replace(
        result,
        agreement_gate=GateResult("us_spine_agreement", True),
    )
    pool_tool._write_outputs(
        ready,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    manifest["publication_run_id"] = "substituted-run"
    manifest["pool_h5"]["publication_run_id"] = "substituted-run"
    outputs.manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="H5.*run ID does not match"):
        pool_tool.load_simulation_ready_us_multispine_pool_manifest(outputs.manifest)


def test_h5_publication_failure_invalidates_stale_green_manifest(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "h5-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )

    def fail_h5(*_args, **_kwargs) -> None:
        _assert_publication_tombstone(
            pool_tool,
            outputs,
            publication_run_id=publication_run_id,
        )
        raise RuntimeError("injected H5 publication failure")

    monkeypatch.setattr(pool_tool, "write_nullable_us_h5", fail_h5)

    with pytest.raises(RuntimeError, match="injected H5 publication failure"):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    assert outputs.pool_h5.read_bytes() == b"stale green h5"


def test_diagnostics_publication_failure_keeps_pool_not_ready(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "diagnostics-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )
    temporary_diagnostics = pool_tool._publication_temporary_path(
        outputs.agreement_diagnostics,
        publication_run_id=publication_run_id,
    )
    atomic_write_json = pool_tool._atomic_write_json

    def fail_diagnostics(path, payload) -> None:
        if Path(path) == temporary_diagnostics:
            raise RuntimeError("injected diagnostics publication failure")
        atomic_write_json(path, payload)

    monkeypatch.setattr(pool_tool, "_atomic_write_json", fail_diagnostics)

    with pytest.raises(
        RuntimeError,
        match="injected diagnostics publication failure",
    ):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    assert outputs.pool_h5.read_bytes() == b"stale green h5"
    assert not pool_tool._publication_temporary_path(
        outputs.pool_h5,
        publication_run_id=publication_run_id,
    ).exists()


def test_final_manifest_failure_leaves_tombstone_as_readiness_authority(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "manifest-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )
    atomic_write_json = pool_tool._atomic_write_json

    def fail_final_manifest(path, payload) -> None:
        if (
            Path(path) == outputs.manifest
            and payload["status"] != "publication_in_progress"
        ):
            raise RuntimeError("injected final manifest publication failure")
        atomic_write_json(path, payload)

    monkeypatch.setattr(pool_tool, "_atomic_write_json", fail_final_manifest)

    with pytest.raises(
        RuntimeError,
        match="injected final manifest publication failure",
    ):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    diagnostics = json.loads(outputs.agreement_diagnostics.read_text(encoding="utf-8"))
    assert diagnostics["publication_run_id"] == publication_run_id
    with pd.HDFStore(outputs.pool_h5, mode="r") as store:
        metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
    assert metadata["publication_run_id"] == publication_run_id


def test_clone_safe_id_error_surfaces_unchanged_through_tool(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    asec = _source_frame()
    tables = {entity: asec.table(entity).copy() for entity in asec.entities}
    tables["person"].loc[0, "person_id"] = PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID + 1
    invalid = Frame(
        tables,
        asec.schema,
        {"household": asec.weights_for("household")},
        asec.strata,
    )

    def unreachable(_frame: Frame) -> PoolStageOutput:
        raise AssertionError("Assembly errors must precede every pool stage.")

    with pytest.raises(ValueError, match="Spine 'asec'.*clone-safe bound"):
        pool_tool.build_multispine_pool(
            invalid,
            _source_frame(),
            puf_donor=pd.DataFrame(),
            primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
            impute=unreachable,
            derive=unreachable,
            seed=unreachable,
            simulate=unreachable,
        )


def test_assembly_receipt_loss_surfaces_unchanged_through_tool(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def drop_receipt(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(
            _replace_person(
                frame,
                frame.table("person").copy(),
                preserve_metadata=False,
            )
        )

    with pytest.raises(
        ValueError,
        match="multispine pool derive output:.*no assembly manifest",
    ):
        pool_tool.build_multispine_pool(
            _source_frame(),
            _source_frame(),
            puf_donor=pd.DataFrame(),
            primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
            impute=no_op,
            derive=drop_receipt,
            seed=no_op,
            simulate=no_op,
        )


def test_local_artifact_reference_never_embeds_host_absolute_paths(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    """Exported row locations anchor to checkout, then home, never ``/``."""

    repo_file = Path(pool_tool.__file__).resolve()
    repo_reference = pool_tool._local_artifact_reference(repo_file)
    assert repo_reference == "local://tools/build_us_multispine_pool.py"

    home_path = Path.home() / "microcosm-test-unwritten" / "artifact.h5"
    home_reference = pool_tool._local_artifact_reference(home_path)
    assert home_reference == ("local://~/microcosm-test-unwritten/artifact.h5")

    outside_path = (tmp_path / "artifact.h5").resolve()
    outside_reference = pool_tool._local_artifact_reference(outside_path)
    assert outside_reference == f"local://{outside_path.as_posix().lstrip('/')}"

    for reference in (repo_reference, home_reference, outside_reference):
        assert not reference.startswith("local:///")
        assert "local://Users/" not in reference
