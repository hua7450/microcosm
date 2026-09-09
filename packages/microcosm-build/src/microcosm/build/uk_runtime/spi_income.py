"""Current-vintage SPI QRF stages for the UK national support channel."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.uk_runtime.frs_disability import (
    UKDWPDisabilityCategoryRates,
    UKDWPDisabilityFlagRates,
)
from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_HMRC_EMPLOYED_INCOME_COLUMN,
    SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_PAY_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
    SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UKSPISupportResult,
    support_channel_column,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

QRF: Any | None = None

SPI_DONOR_RELEASE = "spi_2022_23"
SPI_DONOR_FILENAME = "put2223uk.tab"
SPI_DONOR_VINTAGE = "2022-23"
SPI_DONOR_UKDS_STUDY = "SN 9422"
SPI_DONOR_DOI = "10.5255/UKDA-SN-9422-1"
SPI_DONOR_DOCUMENTATION_URL = (
    "https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/"
    "9422_put_2223_full_documentation.pdf"
)
SPI_DONOR_FIT_NAME = "uk_spi_2022_23_income"
FRS_ONLY_FIT_NAME = "uk_frs_only_spi_fill"
DEFAULT_SPI_DONOR_SAMPLE_SIZE = 100_000
# Recipient guard matching the existing constructed SPI age support below.
# SN 9422 Annex A labels the youngest band "Under 25"; it does not establish
# a survey age minimum. Keep observed FRS inputs below the constructed lower
# bound instead of extrapolating this coarsened donor age to younger children.
# FRS dependent-child status can extend to ages 16-19; this guard is an age
# support boundary, not a crosswalk to that household-composition definition.
SPI_MINIMUM_RECIPIENT_AGE = 16
SPI_DONOR_INCOME_YEAR = 2022
# Use the pinned engine's variable-specific indices. None explicitly retains
# nominal amounts where no same-scope indexed mapping has been reviewed;
# broader indexed benefit inputs do not establish a taxable SPI crosswalk.
SPI_INCOME_UPRATING_VARIABLES = {
    "self_employment_income": "self_employment_income",
    "savings_interest_income": "savings_interest_income",
    "dividend_income": "dividend_income",
    "private_pension_income": "private_pension_income",
    "property_income": "property_income",
    "other_investment_income": "other_investment_income",
    "gift_aid": None,
    "charitable_investment_gifts": None,
    "hmrc_spi_pay": "employment_income_before_lsr",
    "hmrc_spi_employment_benefits": "employment_income_before_lsr",
    "hmrc_spi_employment_expenses": "employment_income_before_lsr",
    "hmrc_spi_incapacity_benefit_income": None,
    "hmrc_spi_other_social_security_income": None,
    "hmrc_spi_taxable_termination_pay": "employment_income_before_lsr",
    "hmrc_spi_unemployment_benefit_income": None,
    "hmrc_spi_miscellaneous_employment_income": "employment_income_before_lsr",
    "hmrc_spi_other_income": "miscellaneous_income",
    "hmrc_spi_state_pension_income": None,
}
SPI_DONOR_SHA256 = "5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66"
SPI_DONOR_SIZE_BYTES = 141_323_762
_SPI_DONOR_VERIFICATION_TOKEN = object()
# The pinned donor's published TI field differs from TEI + TII by at most £5
# because the public-use fields are rounded. Synthetic draws never inherit
# that discrepancy: their accounting aggregates are derived after the draw.
SPI_TI_IDENTITY_ABS_TOLERANCE_GBP = 5.0
# Annex A gives the exact leaf formulas. The pinned PUT first rounds source
# fields, then averages composite records, then rounds remaining income fields
# to £5. The two max() operations do not commute with composite averaging, so
# source-leaf reconciliation has separate observed envelopes for the exact
# sha-pinned ordinary and composite records. Synthetic identities remain exact.
SPI_SOURCE_TEI_FORMULA = (
    "max(0, PAY + EPB - EXPS) + INCPBEN + OSSBEN + TAXTERM + UBISJA + "
    "MOTHINC + OTHERINC + SRP + PENSION + "
    "max(0, PROFITS - CAPALL - LOSSBF)"
)
SPI_SOURCE_TII_FORMULA = "OTHERINV + DIVIDENDS + INCPROP + INCBBS"
SPI_SOURCE_TI_FORMULA = "TEI + TII"
SPI_SOURCE_COMPOSITE_INDICATOR = "AGERANGE == -1"
SPI_SOURCE_LEAF_RECONCILIATION_ABS_TOLERANCE_GBP = {
    "ordinary": {"TEI": 15.0, "TII": 10.0, "TI": 20.0},
    "composite": {"TEI": 180.0, "TII": 10.0, "TI": 180.0},
}

SPI_POLICYENGINE_EMPLOYMENT_SOURCE_COLUMNS = ("PAY", "EPB", "TAXTERM")
SPI_POLICYENGINE_EMPLOYMENT_FORMULA = (
    "hmrc_spi_pay + hmrc_spi_employment_benefits + hmrc_spi_taxable_termination_pay"
)
SPI_HMRC_EMPLOYED_INCOME_SOURCE_COLUMN_MAP = {
    SPI_HMRC_PAY_COLUMN: ("PAY",),
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN: ("EPB",),
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN: ("EXPS",),
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN: ("INCPBEN",),
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN: ("OSSBEN",),
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN: ("TAXTERM",),
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN: ("UBISJA",),
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN: ("MOTHINC",),
}
SPI_HMRC_EMPLOYED_INCOME_FORMULA = (
    "max(0, hmrc_spi_pay + hmrc_spi_employment_benefits - "
    "hmrc_spi_employment_expenses) + hmrc_spi_incapacity_benefit_income + "
    "hmrc_spi_other_social_security_income + hmrc_spi_taxable_termination_pay + "
    "hmrc_spi_unemployment_benefit_income + "
    "hmrc_spi_miscellaneous_employment_income"
)
# The FRS instrument can source three full concepts and two explicitly named
# subsets. The remaining full SPI concepts are donor-only: they must be absent
# before the QRF rather than represented by zeroes or by the two partial leaves.
FRS_HMRC_AUXILIARY_SOURCE_COLUMNS = {
    FRS_HMRC_PAY_COLUMN: (FRS_HMRC_PAY_COLUMN,),
    FRS_HMRC_UBISJA_COLUMN: (FRS_HMRC_UBISJA_COLUMN,),
    FRS_HMRC_INCPBEN_COLUMN: (FRS_HMRC_INCPBEN_COLUMN,),
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: (
        FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    ),
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: (FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,),
}
FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS = (
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
)

SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS = {
    "incapacity_benefit_reported": (
        "Absent/all-default on the pinned enhanced-FRS export and certified "
        "Microcosm UK base; not a populated loader layer."
    ),
    "maternity_allowance_reported": (
        "Absent from the pinned enhanced-FRS export and certified Microcosm UK "
        "base; no training source can be materialized for this stage."
    ),
}

SPI_DONOR_REQUIRED_COLUMNS = (
    "SEX",
    "FACT",
    "GORCODE",
    "AGERANGE",
    "PAY",
    "EPB",
    "EXPS",
    "TAXTERM",
    "INCPBEN",
    "OSSBEN",
    "UBISJA",
    "MOTHINC",
    "OTHERINC",
    "PROFITS",
    "CAPALL",
    "LOSSBF",
    "SRP",
    "INCBBS",
    "DIVIDENDS",
    "PENSION",
    "INCPROP",
    "OTHERINV",
    "GIFTAID",
    "GIFTINV",
    "TEI",
    "TII",
    "TI",
)
SPI_INCOME_SOURCE_COLUMNS = {
    # Keep the PolicyEngine input on the pinned enhanced-FRS precedent. The
    # broader published HMRC employment measure is a separate auxiliary.
    "employment_income": SPI_POLICYENGINE_EMPLOYMENT_SOURCE_COLUMNS,
    "self_employment_income": ("PROFITS", "CAPALL", "LOSSBF"),
    "savings_interest_income": ("INCBBS",),
    "dividend_income": ("DIVIDENDS",),
    "private_pension_income": ("PENSION",),
    "property_income": ("INCPROP",),
    "other_investment_income": ("OTHERINV",),
    "gift_aid": ("GIFTAID",),
    "charitable_investment_gifts": ("GIFTINV",),
    **SPI_HMRC_EMPLOYED_INCOME_SOURCE_COLUMN_MAP,
    SPI_HMRC_OTHER_INCOME_COLUMN: ("OTHERINC",),
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN: ("SRP",),
}
SPI_QRF_SOURCE_COLUMNS = {
    output: sources
    for output, sources in SPI_INCOME_SOURCE_COLUMNS.items()
    if output != "employment_income"
}
SPI_DERIVED_POLICYENGINE_SOURCE_COLUMNS = {
    "employment_income": SPI_POLICYENGINE_EMPLOYMENT_SOURCE_COLUMNS,
}
_DIRECT_SPI_OUTPUT_SOURCE_COLUMNS = {
    output: sources
    for output, sources in SPI_INCOME_SOURCE_COLUMNS.items()
    if output not in {"employment_income", "self_employment_income"}
}
_SPI_AGE_RANGES = {
    -1: (16, 70),
    1: (16, 25),
    2: (25, 35),
    3: (35, 45),
    4: (45, 55),
    5: (55, 65),
    6: (65, 74),
    7: (74, 90),
}
_SPI_REGION_MAP = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
}


@dataclass(frozen=True)
class UKSPIIncomeImputationResult:
    """SPI-filled person table and auditable fit/source evidence."""

    person: pd.DataFrame
    fit_weight_records: tuple[FitWeightRecord, ...]
    donor_path: Path
    donor_sha256: str
    donor_size_bytes: int
    donor_rows: int
    stage2_training_rows: int
    spi_prediction_rows: int
    reviewed_absent_stage2_outputs: dict[str, str]
    pension_receipt_bridge: Mapping[str, object] | None = None
    income_uprating: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _SPIDonorFingerprint:
    """Stable-file identity binding a reviewed hash to the parsed donor bytes."""

    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class VerifiedSPIDonorIdentity:
    """Opaque proof that the local licensed donor matched the reviewed bytes."""

    path: Path
    sha256: str
    size_bytes: int
    fingerprint: _SPIDonorFingerprint
    _verification_token: object


def verify_spi_donor_identity(path: str | Path) -> VerifiedSPIDonorIdentity:
    """Verify the licensed donor before parsing and bind the proof to its file."""

    donor = Path(path).expanduser().resolve()
    if donor.name != SPI_DONOR_FILENAME:
        raise ValueError(
            f"Current SPI donor must be named {SPI_DONOR_FILENAME!r}, got "
            f"{donor.name!r}."
        )
    if not donor.is_file():
        raise FileNotFoundError(f"SPI 2022-23 donor not found: {donor}.")
    before = _spi_donor_fingerprint(donor)
    if before.size_bytes != SPI_DONOR_SIZE_BYTES:
        raise ValueError(
            "SPI 2022-23 donor size does not match the reviewed UKDS source "
            f"identity: expected {SPI_DONOR_SIZE_BYTES}, got {before.size_bytes}."
        )
    digest = _sha256(donor)
    after = _spi_donor_fingerprint(donor)
    if after != before:
        raise RuntimeError(
            "SPI 2022-23 donor changed while its reviewed identity was being verified."
        )
    if digest != SPI_DONOR_SHA256:
        raise ValueError(
            "SPI 2022-23 donor SHA-256 does not match the reviewed UKDS source "
            f"identity: expected {SPI_DONOR_SHA256}, got {digest}."
        )
    return VerifiedSPIDonorIdentity(
        path=donor,
        sha256=digest,
        size_bytes=before.size_bytes,
        fingerprint=before,
        _verification_token=_SPI_DONOR_VERIFICATION_TOKEN,
    )


def assert_frs_hmrc_auxiliary_crosswalk_available(person: pd.DataFrame) -> None:
    """Require the adjudicated FRS leaves and forbid unavailable full concepts."""

    required = tuple(FRS_HMRC_AUXILIARY_SOURCE_COLUMNS)
    missing = sorted(set(required) - set(person.columns))
    if missing:
        raise ValueError(
            "Certified UK FRS channel is missing adjudicated retained HMRC "
            f"leaf column(s): {missing}. Retain PAY, UBISJA, INCPBEN, "
            "ossben_identifiable_subset, and srp_regular_code5 from their "
            "reviewed raw FRS sources before the SPI QRF."
        )
    unavailable = sorted(
        set(FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS) & set(person.columns)
    )
    if unavailable:
        raise ValueError(
            "Pre-QRF FRS candidate must not materialize source-absent full "
            f"SPI concept column(s): {unavailable}. EPB, EXPS, TAXTERM, "
            "MOTHINC, OTHERINC, full OSSBEN, and full SRP cannot be zero-filled "
            "or inferred from their named subsets."
        )
    numeric = person[list(required)].apply(pd.to_numeric, errors="coerce")
    _require_finite_numeric(numeric, label="FRS HMRC retained leaves")
    if (numeric.to_numpy(dtype=float) < 0.0).any():
        raise ValueError("FRS HMRC retained leaves contain negative values.")


def impute_uk_spi_income_support(
    support: UKSPISupportResult,
    spi_tab_path: str | Path,
    *,
    seed: int = 42,
    n_estimators: int = 100,
    donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    build_period: int | str = 2023,
    verified_donor: VerifiedSPIDonorIdentity | None = None,
    donor_table: pd.DataFrame | None = None,
    initialize_frs_channel_columns: Mapping[str, float] | None = None,
    stage1_base_redraw_columns: Sequence[str] = (),
    condition_on_state_pension_receipt: bool = False,
    rebase_income_to_build_period: bool = False,
) -> UKSPIIncomeImputationResult:
    """Run strict SPI-income and FRS-only QRFs on rebuilt positive support."""

    if support.household_weight_kind is not WeightKind.IMPORTANCE:
        raise ValueError(
            "SPI income imputation requires rebuilt importance-weight support."
        )
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer.")
    if donor_sample_size is not None and (
        not isinstance(donor_sample_size, int) or donor_sample_size <= 0
    ):
        raise ValueError("donor_sample_size must be a positive integer or None.")

    donor_path = Path(spi_tab_path).expanduser().resolve()
    if donor_table is None:
        if verified_donor is None:
            # Keep the private helper call as the narrow test seam used by the
            # synthetic-donor unit tests. Production receives an opaque identity.
            verified_donor = _verify_spi_donor_identity(donor_path)
        elif verified_donor.path != donor_path:
            raise ValueError(
                "Verified SPI donor identity does not match the requested donor path."
            )
        if isinstance(verified_donor, VerifiedSPIDonorIdentity):
            _assert_verified_spi_donor_current(verified_donor)
        raw_donor = pd.read_csv(donor_path, delimiter="\t")
        if isinstance(verified_donor, VerifiedSPIDonorIdentity):
            _assert_verified_spi_donor_current(verified_donor)
    else:
        if verified_donor is not None:
            raise ValueError("donor_table and verified_donor are mutually exclusive.")
        if not isinstance(donor_table, pd.DataFrame):
            raise TypeError("donor_table must be a pandas DataFrame.")
        raw_donor = donor_table.copy(deep=True)
    donor = _prepare_spi_donor(raw_donor, seed=seed)
    donor_fit_weights = donor["FACT"].to_numpy(dtype=np.float64)
    if donor_sample_size is not None:
        donor = donor.sample(
            n=donor_sample_size,
            replace=True,
            weights="FACT",
            random_state=seed,
        ).reset_index(drop=True)
        # Mirroring the enhanced-FRS pipeline, the FACT-weighted bootstrap
        # itself represents the SPI design. Reapplying FACT to the sampled
        # rows would square the survey weights. Uniform typed DESIGN weights
        # keep the fit auditable without double-weighting the donor.
        donor_fit_weights = np.ones(len(donor), dtype=np.float64)

    person = support.person.copy()
    household = support.household
    person_channel = support_channel_column("person")
    household_channel = support_channel_column("household")
    _require_columns(person, (person_channel,), label="person support")
    _require_columns(
        household,
        (
            "household_id",
            "household_weight",
            household_channel,
            HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
        ),
        label="household support",
    )
    spi_household = household[household_channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    if (
        not spi_household.any()
        or not household.loc[spi_household, "household_weight"].gt(0.0).all()
    ):
        raise ValueError("SPI support rows must all carry positive effective mass.")
    spi_channel_people = person[person_channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    if not spi_channel_people.any():
        raise ValueError("SPI support has no person rows to impute.")
    _require_columns(person, ("age",), label="SPI recipient age domain")
    _require_finite_numeric(person[["age"]], label="SPI recipient ages")
    spi_people = spi_channel_people & person.age.ge(SPI_MINIMUM_RECIPIENT_AGE)
    if not spi_people.any():
        raise ValueError("SPI support has no recipients in the donor age domain.")
    base_people = person[person_channel] == BASE_FRS_SUPPORT_CHANNEL
    if not base_people.any():
        raise ValueError("SPI support has no FRS base rows.")
    if initialize_frs_channel_columns is not None:
        person = _initialize_frs_channel_columns(
            person,
            base_people=base_people,
            columns=initialize_frs_channel_columns,
        )
    base_redraw_columns = tuple(stage1_base_redraw_columns)
    unknown_redraw = sorted(
        set(base_redraw_columns) - set(SPI_INCOME_QRF_OUTPUT_COLUMNS)
    )
    if unknown_redraw:
        raise ValueError(
            "SPI stage-1 base redraw columns must be stage-1 QRF outputs; "
            f"unknown column(s): {unknown_redraw}."
        )
    person = _seed_frs_hmrc_auxiliary_leaves(person, spi_people=spi_people)

    recipient_predictors = _person_predictors(
        person.loc[spi_channel_people],
        household,
        income_predictors=(),
    )
    donor_predictors, encoded_recipient = _encode_predictor_pair(
        donor[["age", "gender", "region"]],
        recipient_predictors,
    )
    donor_frame = _person_fit_frame(
        predictors=donor_predictors,
        targets=donor[list(SPI_INCOME_QRF_OUTPUT_COLUMNS)],
        weights=donor_fit_weights,
        weight_kind=WeightKind.DESIGN,
    )
    qrf_cls = _qrf_class()
    stage1 = qrf_cls(n_estimators=n_estimators, seed=seed).fit(
        donor_frame,
        list(donor_predictors.columns),
        list(SPI_INCOME_QRF_OUTPUT_COLUMNS),
        weights="design",
    )
    stage1_draws = stage1.predict(encoded_recipient)
    _validate_predictions(
        stage1_draws,
        expected=SPI_INCOME_QRF_OUTPUT_COLUMNS,
        label="SPI stage-1",
    )
    # Hold the existing RNG stream fixed: the fitted forest is also used
    # later for the base-channel dividend redraw. Consume the legacy query
    # shape, then discard under-age draws before any assignment. This keeps
    # adult stage-1 draw pool and the base redraw stream identical.
    adult_positions = (
        person.loc[spi_channel_people, "age"].ge(SPI_MINIMUM_RECIPIENT_AGE).to_numpy()
    )
    stage1_draws = stage1_draws.iloc[np.flatnonzero(adult_positions)].copy()
    income_uprating = None
    uprating_factors = dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0)
    if rebase_income_to_build_period:
        uprating_factors, income_uprating = _spi_income_uprating_factors(
            int(build_period)
        )
        stage1_draws = stage1_draws.mul(pd.Series(uprating_factors), axis="columns")
    nonnegative_stage1 = [
        column
        for column in SPI_INCOME_QRF_OUTPUT_COLUMNS
        if column != SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN
    ]
    if (stage1_draws[nonnegative_stage1].to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("SPI stage-1 produced negative non-negative outputs.")
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        if column not in person:
            # These full concepts are unavailable on the FRS instrument, so
            # the FRS channel carries the adjudicated stage-time semantics:
            # zero, the value every consumer fills at read time. Shipping NaN
            # here was assessment-era honesty that made the artifact
            # unloadable — the engine's validate() refuses NaN inputs, and
            # the calibration seam's finiteness fence rightly refuses the
            # frame — while the auxiliary crosswalk guard above already
            # protects the QRF from mistaking the fill for measured data.
            person[column] = 0.0
        person.loc[spi_people, column] = stage1_draws[column].to_numpy()
    person = _derive_policyengine_employment_input(person, spi_people=spi_people)

    taxable_interest_draw = person.loc[spi_people, "savings_interest_income"].to_numpy(
        dtype=np.float64, copy=True
    )
    stage2_outputs, reviewed_absent = _stage2_outputs(person)
    base_household = household[household_channel] == BASE_FRS_SUPPORT_CHANNEL
    if "clone_index" in household:
        base_household &= household["clone_index"] == 0
    training_household_ids = set(household.loc[base_household, "household_id"])
    training_people = (person[person_channel] == BASE_FRS_SUPPORT_CHANNEL) & person[
        "person_household_id"
    ].isin(training_household_ids)
    if not training_people.any():
        raise ValueError("FRS-only stage has no canonical base training rows.")

    income_predictors = FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS
    train_predictors = _person_predictors(
        person.loc[training_people],
        household,
        income_predictors=income_predictors,
    )
    target_predictors = _person_predictors(
        person.loc[spi_people],
        household,
        income_predictors=income_predictors,
    )
    pension_bridge = None
    if condition_on_state_pension_receipt:
        # A receipt indicator is a predictive bridge, not an assertion that
        # FRS regular pension and the full taxable SPI SRP amount coincide.
        # Without it, stage 2 can attach a second, unrelated pension regime
        # to a stage-1 vector whose HMRC pension leaf is already determined.
        train_receipt = person.loc[training_people, "state_pension_reported"].gt(0)
        target_receipt = person.loc[
            spi_people, SPI_HMRC_STATE_PENSION_INCOME_COLUMN
        ].gt(0)
        train_predictors["state_pension_receipt"] = train_receipt.astype(float)
        target_predictors["state_pension_receipt"] = target_receipt.astype(float)
        pension_bridge = {
            "training_source": "state_pension_reported > 0",
            "recipient_source": "hmrc_spi_state_pension_income > 0",
            "training_positive_rows": int(train_receipt.sum()),
            "recipient_positive_rows": int(target_receipt.sum()),
        }
    encoded_train, encoded_target = _encode_predictor_pair(
        train_predictors,
        target_predictors,
    )
    training_targets = person.loc[training_people, list(stage2_outputs)].copy()
    _require_finite_numeric(training_targets, label="FRS-only training outputs")
    if (training_targets.to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("FRS-only training outputs must be non-negative.")
    person_weights = _person_household_weights(person, household)
    stage2_frame = _person_fit_frame(
        predictors=encoded_train,
        targets=training_targets,
        weights=person_weights.loc[training_people].to_numpy(dtype=np.float64),
        weight_kind=WeightKind.IMPORTANCE,
    )
    # Stage 2 deliberately derives a distinct RNG stream from the single
    # reviewed stage seed; changing this edits the derivation convention.
    stage2 = qrf_cls(n_estimators=n_estimators, seed=seed + 1).fit(
        stage2_frame,
        list(encoded_train.columns),
        list(stage2_outputs),
        weights="importance",
    )
    stage2_draws = stage2.predict(encoded_target)
    _validate_predictions(
        stage2_draws,
        expected=stage2_outputs,
        label="FRS-only stage-2",
    )
    if (stage2_draws.to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("FRS-only stage-2 produced negative non-negative outputs.")
    for column in stage2_outputs:
        person.loc[spi_people, column] = stage2_draws[column].to_numpy()

    if base_redraw_columns:
        base_predictors = _person_predictors(
            person.loc[base_people],
            household,
            income_predictors=(),
        )
        _, encoded_base = _encode_predictor_pair(
            donor[["age", "gender", "region"]],
            base_predictors,
        )
        base_draws = stage1.predict(encoded_base)
        _validate_predictions(
            base_draws,
            expected=SPI_INCOME_QRF_OUTPUT_COLUMNS,
            label="SPI stage-1 base redraw",
        )
        if (
            base_draws[list(base_redraw_columns)]
            .to_numpy(dtype=np.float64)
            .min(initial=0.0)
            < 0.0
        ):
            raise ValueError("SPI stage-1 base redraw produced negative outputs.")
        for column in base_redraw_columns:
            # This redraw also uses adult SPI donors. Preserve observed FRS
            # child dividends, while consuming the same base query stream.
            base_adults = (
                person.loc[base_people, "age"].ge(SPI_MINIMUM_RECIPIENT_AGE).to_numpy()
            )
            person.loc[
                base_people & person.age.ge(SPI_MINIMUM_RECIPIENT_AGE), column
            ] = base_draws[column].to_numpy()[base_adults] * uprating_factors[column]

    tax_free = person.loc[spi_people, "tax_free_savings_income"].to_numpy(
        dtype=np.float64
    )
    person.loc[spi_people, "savings_interest_income"] = taxable_interest_draw + tax_free
    person = derive_hmrc_income_auxiliaries(person, row_mask=spi_people)
    person = _refresh_disability_derived_inputs(person, spi_people=spi_people)
    return UKSPIIncomeImputationResult(
        person=person,
        fit_weight_records=(
            FitWeightRecord(SPI_DONOR_FIT_NAME, stage1.weight_kind),
            FitWeightRecord(FRS_ONLY_FIT_NAME, stage2.weight_kind),
        ),
        donor_path=donor_path,
        donor_sha256=(
            verified_donor.sha256
            if isinstance(verified_donor, VerifiedSPIDonorIdentity)
            else _sha256(donor_path)
        ),
        donor_size_bytes=(
            verified_donor.size_bytes
            if isinstance(verified_donor, VerifiedSPIDonorIdentity)
            else donor_path.stat().st_size
        ),
        donor_rows=len(donor),
        stage2_training_rows=int(training_people.sum()),
        spi_prediction_rows=int(spi_people.sum()),
        reviewed_absent_stage2_outputs=reviewed_absent,
        pension_receipt_bridge=pension_bridge,
        income_uprating=income_uprating,
    )


@cache
def _spi_income_uprating_factors(year: int):
    """Rebase SPI money flows before conditioning on build-year FRS incomes."""
    from policyengine_uk import CountryTaxBenefitSystem

    if year < SPI_DONOR_INCOME_YEAR:
        raise ValueError("SPI income cannot be rebased before its source year.")
    if set(SPI_INCOME_UPRATING_VARIABLES) != set(SPI_INCOME_QRF_OUTPUT_COLUMNS):
        raise ValueError("Every SPI income output needs an explicit uprating rule.")
    system = CountryTaxBenefitSystem()
    source = system.parameters(f"{SPI_DONOR_INCOME_YEAR}-01-01")
    target = system.parameters(f"{year}-01-01")
    factors = {}
    columns = {}
    for column, variable_name in SPI_INCOME_UPRATING_VARIABLES.items():
        path = None
        factor = 1.0
        if variable_name is not None:
            variable = system.variables.get(variable_name)
            if variable is None:
                raise ValueError(
                    f"Missing SPI uprating variable {variable_name!r} for {column}."
                )
            path = getattr(variable, "uprating", None)
            if not isinstance(path, str) or not path:
                raise ValueError(f"Unsupported SPI uprating index for {column}.")
            before, after = source, target
            try:
                for part in path.split("."):
                    before, after = getattr(before, part), getattr(after, part)
            except AttributeError as error:
                raise ValueError(
                    f"Missing SPI uprating parameter {path!r} for {column}."
                ) from error
            if not np.isfinite([before, after]).all() or before <= 0 or after <= 0:
                raise ValueError(f"SPI uprating index must be positive for {column}.")
            factor = float(after / before)
        factors[column] = factor
        columns[column] = {
            "variable": variable_name,
            "index": path,
            "factor": factor,
            "basis": "model_index" if path else "held_nominal",
        }
    return factors, {
        "from_period": SPI_DONOR_INCOME_YEAR,
        "to_period": year,
        "columns": columns,
    }


def _initialize_frs_channel_columns(
    person: pd.DataFrame,
    *,
    base_people: pd.Series,
    columns: Mapping[str, float],
) -> pd.DataFrame:
    if not isinstance(columns, Mapping):
        raise TypeError("initialize_frs_channel_columns must be a mapping.")
    if not base_people.index.equals(person.index):
        raise ValueError("FRS base mask must align to the person table.")
    result = person.copy()
    for column, value in columns.items():
        if not isinstance(column, str) or not column:
            raise ValueError("initialize_frs_channel_columns keys must be strings.")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(
                f"initialize_frs_channel_columns[{column!r}] must be finite."
            )
        if column not in result:
            # Belt-and-braces zero: every initialized column has its base
            # rows overwritten on the next line, but a NaN default here would
            # silently survive any future wiring change.
            result[column] = 0.0
        result.loc[base_people, column] = numeric
    return result


def _prepare_spi_donor(raw: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    _require_columns(raw, SPI_DONOR_REQUIRED_COLUMNS, label="SPI 2022-23 donor")
    numeric = pd.DataFrame(
        {
            column: pd.to_numeric(raw[column], errors="coerce")
            for column in SPI_DONOR_REQUIRED_COLUMNS
        }
    )
    _require_finite_numeric(numeric, label="SPI 2022-23 donor")
    if not (numeric["FACT"] > 0.0).all():
        raise ValueError("SPI 2022-23 FACT weights must be strictly positive.")
    sex = numeric["SEX"]
    if not sex.isin([0, 1, 2]).all():
        raise ValueError("SPI 2022-23 SEX must contain only documented codes 0/1/2.")
    age_codes = numeric["AGERANGE"].astype(int)
    unknown_age = sorted(set(age_codes) - set(_SPI_AGE_RANGES))
    if unknown_age:
        raise ValueError(f"SPI 2022-23 has unknown AGERANGE code(s): {unknown_age}.")
    rng = np.random.default_rng(seed)
    bounds = np.asarray([_SPI_AGE_RANGES[code] for code in age_codes])
    donor = pd.DataFrame(
        {
            "age": bounds[:, 0] + rng.random(len(raw)) * (bounds[:, 1] - bounds[:, 0]),
            "gender": np.select(
                (sex == 1, sex == 2),
                ("MALE", "FEMALE"),
                default="UNKNOWN",
            ),
            "region": numeric["GORCODE"]
            .astype(int)
            .map(_SPI_REGION_MAP)
            .fillna("UNKNOWN"),
            "FACT": numeric["FACT"],
            # PolicyEngine's persisted employment input follows the exact
            # enhanced-FRS precedent. It must not be widened to HMRC's
            # published employed-income measure.
            "employment_income": (numeric["PAY"] + numeric["EPB"] + numeric["TAXTERM"]),
            "self_employment_income": np.maximum(
                numeric["PROFITS"] - numeric["CAPALL"] - numeric["LOSSBF"],
                0.0,
            ),
        }
    )
    for output, sources in _DIRECT_SPI_OUTPUT_SOURCE_COLUMNS.items():
        donor[output] = numeric[list(sources)].sum(axis=1)
    _require_finite_numeric(
        donor[["age", "FACT", *SPI_INCOME_QRF_OUTPUT_COLUMNS]],
        label="SPI 2022-23 derived donor",
    )
    nonnegative_outputs = [
        column
        for column in SPI_INCOME_QRF_OUTPUT_COLUMNS
        if column != SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN
    ]
    if (donor[nonnegative_outputs].to_numpy(dtype=float) < 0.0).any():
        raise ValueError(
            "SPI 2022-23 non-negative donor outputs contain negative values."
        )

    # TI, TEI and TII remain source-validation fields only. The QRF draws the
    # documented leaves, and the synthetic aggregates are derived after every
    # draw so the accounting identity holds by construction.
    _validate_spi_source_leaf_reconciliation(numeric)
    ti_error = np.abs(numeric["TI"] - (numeric["TEI"] + numeric["TII"]))
    if (ti_error > SPI_TI_IDENTITY_ABS_TOLERANCE_GBP).any():
        worst = float(ti_error.max())
        raise ValueError(
            "SPI 2022-23 TI disagrees with the published TEI + TII identity; "
            f"worst absolute difference {worst:.6g} exceeds the reviewed "
            f"£{SPI_TI_IDENTITY_ABS_TOLERANCE_GBP:.0f} rounding tolerance."
        )
    for column in ("gift_aid", "charitable_investment_gifts"):
        if (donor[column] < 0.0).any():
            raise ValueError(f"SPI donor {column} must be non-negative.")
    return donor


def _validate_spi_source_leaf_reconciliation(numeric: pd.DataFrame) -> None:
    """Check Annex A leaf formulas within the pinned PUT anonymization envelope."""

    source_tei = (
        np.maximum(0.0, numeric["PAY"] + numeric["EPB"] - numeric["EXPS"])
        + numeric["INCPBEN"]
        + numeric["OSSBEN"]
        + numeric["TAXTERM"]
        + numeric["UBISJA"]
        + numeric["MOTHINC"]
        + numeric["OTHERINC"]
        + numeric["SRP"]
        + numeric["PENSION"]
        + np.maximum(
            0.0,
            numeric["PROFITS"] - numeric["CAPALL"] - numeric["LOSSBF"],
        )
    )
    source_tii = (
        numeric["OTHERINV"]
        + numeric["DIVIDENDS"]
        + numeric["INCPROP"]
        + numeric["INCBBS"]
    )
    derived = {"TEI": source_tei, "TII": source_tii, "TI": source_tei + source_tii}
    composite = numeric["AGERANGE"].eq(-1).to_numpy(dtype=bool)
    groups = {"ordinary": ~composite, "composite": composite}
    for group, mask in groups.items():
        if not mask.any():
            continue
        for field, tolerance in SPI_SOURCE_LEAF_RECONCILIATION_ABS_TOLERANCE_GBP[
            group
        ].items():
            error = np.abs(
                numeric.loc[mask, field].to_numpy(dtype=float)
                - derived[field].loc[mask].to_numpy(dtype=float)
            )
            if (error > tolerance).any():
                worst = float(error.max())
                raise ValueError(
                    "SPI 2022-23 source-leaf reconciliation failed for "
                    f"{group} {field}; worst absolute difference {worst:.6g} "
                    f"exceeds the reviewed £{tolerance:.0f} PUT anonymization "
                    "envelope."
                )


def _seed_frs_hmrc_auxiliary_leaves(
    person: pd.DataFrame,
    *,
    spi_people: pd.Series,
) -> pd.DataFrame:
    """Validate retained FRS leaves without manufacturing unavailable concepts."""

    if len(spi_people) != len(person):
        raise ValueError("SPI person mask must align to the person table.")
    assert_frs_hmrc_auxiliary_crosswalk_available(person)
    return person.copy()


def _derive_policyengine_employment_input(
    person: pd.DataFrame,
    *,
    spi_people: pd.Series,
) -> pd.DataFrame:
    """Set SPI employment_income to PAY + EPB + TAXTERM by construction."""

    required = (
        SPI_HMRC_PAY_COLUMN,
        SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
        SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    )
    _require_columns(person, required, label="PolicyEngine SPI employment crosswalk")
    leaves = person.loc[spi_people, list(required)].apply(
        pd.to_numeric, errors="coerce"
    )
    _require_finite_numeric(leaves, label="PolicyEngine SPI employment leaves")
    employment = leaves.sum(axis=1).to_numpy(dtype=float)
    if (employment < 0.0).any():
        raise ValueError("Derived PolicyEngine SPI employment_income is negative.")
    result = person.copy()
    if "employment_income" not in result:
        result["employment_income"] = 0.0
    result.loc[spi_people, "employment_income"] = employment
    return result


def derive_hmrc_income_auxiliaries(
    person: pd.DataFrame,
    *,
    row_mask: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Derive TEI, TII and TI exactly, optionally only on selected rows.

    The whole-frame default preserves the exact synthetic/calibration contract.
    A row mask is used by the mixed FRS/SPI stage: full concepts absent from the
    FRS instrument remain unavailable there, while SPI rows receive identities
    derived deterministically from their joint donor draws.
    """

    required = (
        *SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
        SPI_HMRC_OTHER_INCOME_COLUMN,
        SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
        "self_employment_income",
        "private_pension_income",
        "savings_interest_income",
        "tax_free_savings_income",
        "dividend_income",
        "property_income",
        "other_investment_income",
    )
    _require_columns(person, required, label="HMRC accounting identity")
    mask = _hmrc_auxiliary_row_mask(person, row_mask)
    numeric = person.loc[mask, list(required)].apply(pd.to_numeric, errors="coerce")
    _require_finite_numeric(numeric, label="HMRC accounting identity leaves")
    nonnegative = [
        column
        for column in required
        if column != SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN
    ]
    if (numeric[nonnegative].to_numpy(dtype=float) < 0.0).any():
        raise ValueError(
            "HMRC non-negative accounting identity leaves contain negative values."
        )

    taxable_interest = (
        numeric["savings_interest_income"] - numeric["tax_free_savings_income"]
    )
    if (taxable_interest < 0.0).any():
        raise ValueError(
            "HMRC accounting identity requires gross savings_interest_income "
            "to be at least tax_free_savings_income on every person row."
        )
    employed_income = (
        np.maximum(
            numeric[SPI_HMRC_PAY_COLUMN]
            + numeric[SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN]
            - numeric[SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN],
            0.0,
        )
        + numeric[SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN]
        + numeric[SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN]
        + numeric[SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN]
        + numeric[SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN]
        + numeric[SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN]
    ).to_numpy(dtype=float)
    total_earned = (
        employed_income
        + numeric[SPI_HMRC_OTHER_INCOME_COLUMN]
        + numeric[SPI_HMRC_STATE_PENSION_INCOME_COLUMN]
        + numeric["self_employment_income"]
        + numeric["private_pension_income"]
    ).to_numpy(dtype=float)
    total_investment = (
        taxable_interest
        + numeric["dividend_income"]
        + numeric["property_income"]
        + numeric["other_investment_income"]
    ).to_numpy(dtype=float)
    assessable = total_earned + total_investment

    result = person.copy()
    derived = {
        SPI_HMRC_EMPLOYED_INCOME_COLUMN: employed_income,
        SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN: total_earned,
        SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN: total_investment,
        "hmrc_spi_assessable_income": assessable,
    }
    if row_mask is None:
        for column, values in derived.items():
            result[column] = values
    else:
        # A prior full-frame auxiliary is not valid evidence for the partial
        # FRS channel. Clear every derived output to the stage-time zero,
        # then materialize SPI only — the mask-restricted arithmetic below
        # never reads the cleared rows, and the artifact must not ship NaN.
        for column, values in derived.items():
            result[column] = 0.0
            result.loc[mask, column] = values
    if not np.array_equal(
        result.loc[mask, "hmrc_spi_assessable_income"].to_numpy(dtype=float),
        result.loc[mask, SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN].to_numpy(dtype=float)
        + result.loc[mask, SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN].to_numpy(
            dtype=float
        ),
    ):
        raise RuntimeError("HMRC TI must equal derived TEI + TII exactly.")
    return result


def _hmrc_auxiliary_row_mask(
    person: pd.DataFrame,
    row_mask: pd.Series | np.ndarray | None,
) -> np.ndarray:
    """Return a strict row-aligned boolean mask for optional HMRC derivation."""

    if row_mask is None:
        return np.ones(len(person), dtype=bool)
    if isinstance(row_mask, pd.Series):
        if not row_mask.index.equals(person.index):
            raise ValueError("HMRC auxiliary row mask index must align to person.")
        if row_mask.isna().any() or not pd.api.types.is_bool_dtype(row_mask.dtype):
            raise ValueError("HMRC auxiliary row mask must contain only booleans.")
        mask = row_mask.to_numpy(dtype=bool)
    else:
        raw = np.asarray(row_mask)
        if raw.ndim != 1 or raw.dtype.kind != "b":
            raise ValueError("HMRC auxiliary row mask must be one-dimensional bool.")
        mask = raw.astype(bool, copy=False)
    if len(mask) != len(person):
        raise ValueError("HMRC auxiliary row mask must align to the person table.")
    return mask


def _person_predictors(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    income_predictors: tuple[str, ...],
) -> pd.DataFrame:
    required = ("person_household_id", "age", "gender", *income_predictors)
    _require_columns(person, required, label="SPI QRF person predictors")
    _require_columns(household, ("household_id", "region"), label="household")
    region = household.set_index("household_id")["region"]
    mapped_region = person["person_household_id"].map(region)
    if mapped_region.isna().any():
        raise ValueError("SPI QRF cannot map every person to a household region.")
    result = person[["age", "gender", *income_predictors]].copy()
    result["region"] = mapped_region.to_numpy()
    result = result[["age", "gender", "region", *income_predictors]]
    _require_finite_numeric(
        result[["age", *income_predictors]],
        label="SPI QRF numeric predictors",
    )
    if result[["gender", "region"]].isna().any().any():
        raise ValueError("SPI QRF categorical predictors contain missing values.")
    return result


def _encode_predictor_pair(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train.reset_index(drop=True), target.reset_index(drop=True)],
        ignore_index=True,
    )
    encoded = pd.get_dummies(
        combined,
        columns=["gender", "region"],
        drop_first=False,
        dtype=float,
    )
    encoded = encoded.reindex(sorted(encoded.columns), axis=1)
    train_encoded = encoded.iloc[: len(train)].copy()
    train_encoded.index = train.index
    target_encoded = encoded.iloc[len(train) :].copy()
    target_encoded.index = target.index
    _require_finite_numeric(train_encoded, label="encoded QRF training predictors")
    _require_finite_numeric(target_encoded, label="encoded QRF target predictors")
    return train_encoded, target_encoded


def _person_fit_frame(
    *,
    predictors: pd.DataFrame,
    targets: pd.DataFrame,
    weights: np.ndarray,
    weight_kind: WeightKind,
) -> Frame:
    if len(predictors) != len(targets) or len(predictors) != len(weights):
        raise ValueError("QRF predictors, targets, and weights must align.")
    ids = np.arange(1, len(predictors) + 1, dtype=np.int64)
    person = pd.concat(
        [predictors.reset_index(drop=True), targets.reset_index(drop=True)],
        axis=1,
    )
    person.insert(0, "person_household_id", ids)
    person.insert(0, "person_id", ids)
    household = pd.DataFrame({"household_id": ids})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray(weights), weight_kind)},
    )


def _stage2_outputs(person: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, str]]:
    outputs: list[str] = []
    reviewed: dict[str, str] = {}
    missing_unreviewed: list[str] = []
    for column in FRS_ONLY_SPI_FILL_PERSON_COLUMNS:
        if column in SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS:
            if column in person:
                values = pd.to_numeric(person[column], errors="coerce").to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Reviewed-absent FRS-only output {column!r} contains "
                        "non-finite values."
                    )
                if (values != 0.0).any():
                    raise ValueError(
                        f"Reviewed-absent FRS-only output {column!r} now carries "
                        "non-default source signal; update and review the source "
                        "manifest before adding it to the QRF surface."
                    )
            reviewed[column] = SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS[column]
        elif column in person:
            outputs.append(column)
        else:
            missing_unreviewed.append(column)
    if missing_unreviewed:
        raise ValueError(
            "FRS-only stage cannot silently narrow missing output(s): "
            f"{missing_unreviewed}."
        )
    if not outputs:
        raise ValueError("FRS-only stage has no materializable outputs.")
    return tuple(outputs), reviewed


def _person_household_weights(
    person: pd.DataFrame,
    household: pd.DataFrame,
) -> pd.Series:
    weights = household.set_index("household_id")["household_weight"]
    mapped = person["person_household_id"].map(weights)
    if mapped.isna().any() or not np.isfinite(mapped.to_numpy(dtype=float)).all():
        raise ValueError("Cannot resolve finite household weights for every person.")
    if not (mapped > 0.0).any():
        raise ValueError("Resolved person weights contain no positive mass.")
    return mapped.astype(float)


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected: tuple[str, ...],
    label: str,
) -> None:
    if tuple(predictions.columns) != tuple(expected):
        raise ValueError(
            f"{label} prediction surface mismatch: expected {list(expected)}, "
            f"got {list(predictions.columns)}."
        )
    _require_finite_numeric(predictions, label=f"{label} predictions")


def _require_finite_numeric(frame: pd.DataFrame, *, label: str) -> None:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite.all():
        non_finite_counts = {
            str(column): int((~finite[:, position]).sum())
            for position, column in enumerate(numeric.columns)
            if (~finite[:, position]).any()
        }
        raise ValueError(
            f"{label} must contain only finite numeric values; non-finite "
            f"counts by column: {non_finite_counts}."
        )


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing column(s): {missing}.")


def _qrf_class():
    if QRF is not None:
        return QRF
    return import_module("microcosm.fit").QRF


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spi_donor_fingerprint(path: Path) -> _SPIDonorFingerprint:
    stat = path.stat()
    return _SPIDonorFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _assert_verified_spi_donor_current(identity: VerifiedSPIDonorIdentity) -> None:
    if identity._verification_token is not _SPI_DONOR_VERIFICATION_TOKEN:
        raise ValueError(
            "SPI donor identity must come from verify_spi_donor_identity()."
        )
    if _spi_donor_fingerprint(identity.path) != identity.fingerprint:
        raise RuntimeError(
            "SPI 2022-23 donor changed after identity verification; refusing "
            "to parse bytes not bound to the reviewed SHA-256."
        )


def _verify_spi_donor_identity(path: Path) -> VerifiedSPIDonorIdentity:
    """Backward-compatible private seam for synthetic-donor unit tests."""

    return verify_spi_donor_identity(path)


def _refresh_disability_derived_inputs(
    person: pd.DataFrame,
    *,
    spi_people: pd.Series,
    category_rates: UKDWPDisabilityCategoryRates | None = None,
    flag_rates: UKDWPDisabilityFlagRates | None = None,
) -> pd.DataFrame:
    """Keep category/flag inputs coherent with stage-2 reported amounts.

    Re-derives the eight disability columns for the SPI-redrawn rows with the
    FRS stage's own derivation and, unless rates are injected, the FRS stage's
    declared year rule, so the two paths cannot disagree on the parameter
    tree or the week constant (uk-data#475, uk-data#476). The engine is only
    touched when rates are not supplied.
    """

    from microcosm.build.uk_runtime import frs_disability

    if category_rates is None or flag_rates is None:
        from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule

        year = resolve_uk_year_rule(frs_disability.YEAR_RULE)
        if category_rates is None:
            category_rates = frs_disability.uk_dwp_disability_category_rates(year)
        if flag_rates is None:
            flag_rates = frs_disability.uk_dwp_disability_flag_rates(year)
    derived = frs_disability.derive_frs_disability(
        person.loc[spi_people],
        category_rates=category_rates,
        flag_rates=flag_rates,
    )
    for column in frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS:
        default = "NONE" if column.endswith("_category") else False
        _assign_spi_values(
            person,
            spi_people,
            column,
            derived[column].to_numpy(),
            default=default,
        )
    return person


def _assign_spi_values(
    person: pd.DataFrame,
    spi_people: pd.Series,
    column: str,
    values: np.ndarray,
    *,
    default: object,
) -> None:
    if column not in person:
        person[column] = default
    elif isinstance(person[column].dtype, pd.CategoricalDtype):
        person[column] = person[column].astype(object)
    person.loc[spi_people, column] = values


__all__ = [
    "DEFAULT_SPI_DONOR_SAMPLE_SIZE",
    "FRS_HMRC_AUXILIARY_SOURCE_COLUMNS",
    "FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS",
    "FRS_ONLY_FIT_NAME",
    "SPI_DONOR_DOI",
    "SPI_DONOR_FILENAME",
    "SPI_DONOR_FIT_NAME",
    "SPI_DONOR_RELEASE",
    "SPI_DONOR_SHA256",
    "SPI_DONOR_SIZE_BYTES",
    "SPI_DONOR_UKDS_STUDY",
    "SPI_DONOR_VINTAGE",
    "SPI_DONOR_REQUIRED_COLUMNS",
    "SPI_INCOME_SOURCE_COLUMNS",
    "SPI_DERIVED_POLICYENGINE_SOURCE_COLUMNS",
    "SPI_DONOR_DOCUMENTATION_URL",
    "SPI_HMRC_EMPLOYED_INCOME_FORMULA",
    "SPI_HMRC_EMPLOYED_INCOME_SOURCE_COLUMN_MAP",
    "SPI_POLICYENGINE_EMPLOYMENT_FORMULA",
    "SPI_POLICYENGINE_EMPLOYMENT_SOURCE_COLUMNS",
    "SPI_QRF_SOURCE_COLUMNS",
    "SPI_SOURCE_COMPOSITE_INDICATOR",
    "SPI_SOURCE_LEAF_RECONCILIATION_ABS_TOLERANCE_GBP",
    "SPI_SOURCE_TEI_FORMULA",
    "SPI_SOURCE_TII_FORMULA",
    "SPI_SOURCE_TI_FORMULA",
    "SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS",
    "SPI_TI_IDENTITY_ABS_TOLERANCE_GBP",
    "UKSPIIncomeImputationResult",
    "VerifiedSPIDonorIdentity",
    "assert_frs_hmrc_auxiliary_crosswalk_available",
    "derive_hmrc_income_auxiliaries",
    "impute_uk_spi_income_support",
    "verify_spi_donor_identity",
]
