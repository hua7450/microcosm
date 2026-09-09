"""SPI income stages for the raw UK FRS spine build."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_RETAINED_LEAF_COLUMNS,
    FRS_HMRC_UBISJA_COLUMN,
    _materialize_source_leaves,
    _read_raw_frs_table,
)
from microcosm.build.uk_runtime.hmrc_income import (
    HMRCIncomeTargetSet,
    materialize_hmrc_spi_income_band_targets,
    verify_hmrc_spi_collated_ods,
)
from microcosm.build.uk_runtime.hmrc_replay import (
    HMRCReplayReport,
    build_conservative_hmrc_replay_report,
)
from microcosm.build.uk_runtime.hmrc_source_contract import (
    HMRC_DISTRIBUTIONAL_INPUTS,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.release_input_coverage import (
    DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE,
)
from microcosm.build.uk_runtime.spi_income import (
    DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    SPI_DONOR_INCOME_YEAR,
    SPI_INCOME_UPRATING_VARIABLES,
    SPI_MINIMUM_RECIPIENT_AGE,
    SPI_SOURCE_TI_FORMULA,
    SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    UKSPIIncomeImputationResult,
    assert_frs_hmrc_auxiliary_crosswalk_available,
    impute_uk_spi_income_support,
    verify_spi_donor_identity,
)
from microcosm.build.uk_runtime.spi_support import (
    DEFAULT_SPI_PRIOR_MASS_SHARE,
    DEFAULT_SPI_SUPPORT_HOUSEHOLDS,
    FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
    SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UKSPISupportResult,
    build_uk_spi_support_channel,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.uk_runtime.terminal_gates import (
    UKZeroWeightStratumDeclaration,
)
from microcosm.frame import Frame, WeightKind, engine_tables

UK_FRS_HMRC_SPINE_LEAVES_STAGE_NAME = "frs_hmrc_spine_leaves"
UK_HMRC_SPI_INCOME_SPINE_STAGE_NAME = "hmrc_spi_income_spine"
EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN = "employer_pension_contributions"
UK_HMRC_SPI_SPINE_REPLAY_REPORT_KIND = "uk_hmrc_spi_income_spine_208_fact_replay"

UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS = (
    *FRS_HMRC_RETAINED_LEAF_COLUMNS,
    EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
)
UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS = (
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    support_channel_column("person"),
    support_clone_index_column("person"),
    support_source_id_column("person"),
    support_channel_column("benunit"),
    support_clone_index_column("benunit"),
    support_source_id_column("benunit"),
    support_channel_column("household"),
    support_clone_index_column("household"),
    support_source_id_column("household"),
    "source_household_id",
    "source_year",
    "source_household_key",
)
UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS = (
    "other_investment_income",
    "gift_aid",
    "charitable_investment_gifts",
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    *SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
)
UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS = tuple(
    column
    for column in UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS
    if column != SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN
)
UK_SPI_INCOME_SPINE_REWRITE_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
    "employee_pension_contributions",
    EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
    "personal_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    "tax_free_savings_income",
    "universal_credit_reported",
    "pension_credit_reported",
    "child_benefit_reported",
    "housing_benefit_reported",
    "income_support_reported",
    "working_tax_credit_reported",
    "child_tax_credit_reported",
    "attendance_allowance_reported",
    "state_pension_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "pip_m_reported",
    "pip_dl_reported",
    "sda_reported",
    "carers_allowance_reported",
    "iidb_reported",
    "afcs_reported",
    "bsp_reported",
    "winter_fuel_allowance_reported",
    "council_tax_benefit_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
    FRS_HMRC_INCPBEN_COLUMN,
)

# Reviewed constants for every load-bearing manifest parameter the spine
# transforms consume. The drift asserts below fail closed on any manifest-only
# edit (adversarial-review finding on #717): a manifest change to these values
# requires a matching reviewed code change here. The #730/#684 two-arm rule
# applies to every declared parameter: it needs (1) a drift assert and (2) an
# executed-effect receipt, or an explicit absence statement. Seeded draws use
# twin-build determinism as their executed-effect receipt.
SPI_SPINE_STAGE1_PREDICTORS = ("age", "gender", "region")
SPI_SPINE_STAGE2_PREDICTORS = (
    "age",
    "gender",
    "region",
    *FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS,
    "state_pension_receipt",
)
SPI_SPINE_STAGE2_OUTPUT_COLUMNS = tuple(
    column
    for column in UK_SPI_INCOME_SPINE_REWRITE_COLUMNS
    if column not in FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS
    and column
    not in (FRS_HMRC_PAY_COLUMN, FRS_HMRC_UBISJA_COLUMN, FRS_HMRC_INCPBEN_COLUMN)
)
SPI_SPINE_FRS_CHANNEL_INITIALIZATION = {
    "gift_aid": 0.0,
    "charitable_investment_gifts": 0.0,
}
SPI_SPINE_BASE_REDRAW_COLUMNS = ("dividend_income",)
SPI_SPINE_SUPPORT_CHANNELS = {"base": "frs", "synthetic": SPI_SYNTHETIC_SUPPORT_CHANNEL}
SPI_SPINE_PRECLONE_GATE_NAME = "e7_spi_synthetic_preclone"
SPI_SPINE_EFFECTIVE_MASS_COLUMNS = ("gift_aid", "charitable_investment_gifts")


@dataclass(frozen=True)
class UKFRSHMRCSpineLeavesResult:
    frame: Frame
    source_signal_rows: dict[str, int]
    structural_zero_columns: tuple[str, ...]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_FRS_HMRC_SPINE_LEAVES_STAGE_NAME,
            "source_signal_rows": dict(self.source_signal_rows),
            "structural_zero_columns": list(self.structural_zero_columns),
        }


@dataclass(frozen=True)
class UKSPIIncomeSpineResult:
    frame: Frame
    support: UKSPISupportResult
    imputation: UKSPIIncomeImputationResult
    source_targets: HMRCIncomeTargetSet
    replay_report: HMRCReplayReport
    distributional_mass_shares: Mapping[str, float]
    post_draw_identity_rows: int

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_HMRC_SPI_INCOME_SPINE_STAGE_NAME,
            "source_vintages": {
                "spi_donor": "2022-23",
                "hmrc_surface": self.source_targets.source.source_vintage,
                "mapped_build_period": self.source_targets.source.build_period,
            },
            "spi_prior": {
                "households": self.support.n_spi_households,
                "mass_share": self.support.spi_prior_mass_share,
                "weight_kind": self.support.household_weight_kind.value,
                "mass_change_reason": self.support.mass_log[-1].reason,
            },
            "reviewed_absent_stage2_outputs": dict(
                self.imputation.reviewed_absent_stage2_outputs
            ),
            "recipient_minimum_age": SPI_MINIMUM_RECIPIENT_AGE,
            "pension_receipt_bridge": self.imputation.pension_receipt_bridge,
            "income_uprating": self.imputation.income_uprating,
            "targets": {
                "count": len(self.source_targets.targets),
                "classification": dict(self.replay_report.summary),
            },
            "effective_mass": {
                "minimum_nondefault_mass_share": (
                    DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
                ),
                "columns": dict(self.distributional_mass_shares),
            },
            "post_draw_identity": {
                "formula": SPI_SOURCE_TI_FORMULA,
                "rows_checked": self.post_draw_identity_rows,
                "exact": True,
            },
        }


@dataclass(frozen=True)
class UKFRSHMRCSpineLeavesStageTransform:
    frs_raw_dir: Path
    stage: SourceStageSpec
    # A #627 scale-ladder receipt build: the spine was subsampled after
    # frs_spine, so raw-tab person coverage legitimately exceeds the frame.
    sampled_rung: bool = False
    # Populated only by a live run; resume paths must re-run or skip evidence.
    last_result: UKFRSHMRCSpineLeavesResult | None = field(default=None, init=False)

    def __init__(
        self,
        frs_raw_dir: str | Path,
        *,
        stage: SourceStageSpec,
        sampled_rung: bool = False,
    ) -> None:
        object.__setattr__(self, "frs_raw_dir", Path(frs_raw_dir))
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "sampled_rung", bool(sampled_rung))
        object.__setattr__(self, "last_result", None)

    def __call__(self, frame: Frame) -> Frame:
        validate_uk_national_frame(frame)
        artifacts = _artifact_by_table(self.stage, expected=("adult", "benefits"))
        adult, adult_identity = _read_raw_frs_table(
            self.frs_raw_dir / str(artifacts["adult"]["locator"]),
            expected_filename="adult.tab",
            required_columns=("sernum", "person", "inearns"),
        )
        benefits, benefits_identity = _read_raw_frs_table(
            self.frs_raw_dir / str(artifacts["benefits"]["locator"]),
            expected_filename="benefits.tab",
            required_columns=("sernum", "person", "benefit", "benamt", "var2"),
        )
        _assert_identity_matches_artifact(adult_identity.evidence(), artifacts["adult"])
        _assert_identity_matches_artifact(
            benefits_identity.evidence(), artifacts["benefits"]
        )
        source_leaves = _materialize_source_leaves(adult, benefits)
        person = frame.table("person").copy()
        missing_ids = sorted(set(source_leaves.index) - set(person["person_id"]))
        if missing_ids and not self.sampled_rung:
            raise ValueError(
                "Raw FRS retained leaves contain person identity value(s) absent "
                f"from the raw spine: {missing_ids[:5]}."
            )
        if missing_ids:
            # A rung sample deliberately drops most source people; restrict
            # the raw surface to the survivors (the full-scale fence above
            # stays strict — mirrors frs_hmrc_leaves' sampled_rung posture).
            source_leaves = source_leaves.loc[
                source_leaves.index.isin(person["person_id"].to_numpy())
            ]
        aligned = source_leaves.reindex(person["person_id"], fill_value=0.0)
        for column in FRS_HMRC_RETAINED_LEAF_COLUMNS:
            person[column] = aligned[column].to_numpy(dtype=float)
        if "employee_pension_contributions" not in person:
            raise ValueError(
                "FRS HMRC spine leaves require employee_pension_contributions "
                "to derive employer pension contributions."
            )
        employee = pd.to_numeric(
            person["employee_pension_contributions"], errors="coerce"
        )
        if employee.isna().any() or (employee < 0.0).any():
            raise ValueError(
                "employee_pension_contributions must be finite and non-negative."
            )
        person[EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN] = 3.0 * employee.to_numpy(
            dtype=float
        )
        result_frame = uk_national_frame(
            person=person,
            benunit=frame.table("benunit"),
            household=engine_tables(frame)["household"],
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result_frame)
        source_signal_rows = {
            column: int((source_leaves[column] > 0.0).sum())
            for column in FRS_HMRC_RETAINED_LEAF_COLUMNS
        }
        result = UKFRSHMRCSpineLeavesResult(
            frame=result_frame,
            source_signal_rows=source_signal_rows,
            structural_zero_columns=tuple(
                column for column, rows in source_signal_rows.items() if rows == 0
            ),
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


@dataclass(frozen=True)
class UKSPISupportChannelStageTransform:
    stage: SourceStageSpec
    seed: int = 42
    # A #627 scale-ladder receipt build: scale the declared synthetic stack
    # by the survey-side sample fraction so the prior-mass pairing holds at
    # every rung (f100 keeps the declared count exactly).
    sample_fraction: float = 1.0
    sample_fraction_provider: Callable[[], float] | None = None
    # Populated only by a live run; resume paths must re-run or skip evidence.
    last_result: UKSPISupportResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        validate_uk_national_frame(frame)
        tables = engine_tables(frame)
        count, share, strata, declarations = _support_stage_parameters(
            self.stage,
            seed=self.seed,
        )
        sample_fraction = (
            self.sample_fraction_provider()
            if self.sample_fraction_provider is not None
            else self.sample_fraction
        )
        if sample_fraction != 1.0 and count is not None:
            count = max(1, int(round(count * sample_fraction)))
        result = build_uk_spi_support_channel(
            tables["person"],
            tables["benunit"],
            tables["household"],
            spi_household_count=count,
            seed=self.seed,
            source_year=int(uk_time_period(frame)),
            spi_prior_mass_share=share,
            strata_columns=strata,
            input_weight_kind=uk_household_weight_kind(frame),
            mass_log=frame.mass_log,
            zero_weight_declarations=declarations,
        )
        if result.household_weight_kind is not WeightKind.IMPORTANCE:
            got = (
                None
                if result.household_weight_kind is None
                else result.household_weight_kind.value
            )
            raise ValueError(
                "SPI support channel builder must return importance household "
                f"weights, got {got!r}."
            )
        result_frame = uk_national_frame(
            person=result.person,
            benunit=result.benunit,
            household=result.household,
            time_period=uk_time_period(frame),
            weight_kind=result.household_weight_kind,
            household_weights=result.household["household_weight"].to_numpy(
                dtype=float
            ),
            mass_log=result.mass_log,
        )
        validate_uk_national_frame(result_frame)
        object.__setattr__(self, "last_result", result)
        return result_frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {
            "evidence": {
                "stage": "spi_support_channel",
                "spi_households": self.last_result.n_spi_households,
                "spi_prior_mass_share": self.last_result.spi_prior_mass_share,
                "household_weight_kind": (
                    self.last_result.household_weight_kind.value
                    if self.last_result.household_weight_kind is not None
                    else None
                ),
            }
        }


@dataclass(frozen=True)
class UKSPIIncomeSpineStageTransform:
    spi_tab_path: Path
    hmrc_ods_path: Path
    stage: SourceStageSpec
    seed: int = 42
    qrf_estimators: int = 100
    donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE
    # A #627 scale-ladder receipt build: a sampled stack cannot carry mass in
    # ultra-sparse restored columns, so the effective-mass fence records the
    # gap instead of raising (f100 keeps the strict fence).
    sampled_rung: bool = False
    donor_table: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    source_targets: HMRCIncomeTargetSet | None = None
    # Populated only by a live run; resume paths must re-run or skip evidence.
    last_result: UKSPIIncomeSpineResult | None = field(default=None, init=False)

    def __init__(
        self,
        spi_tab_path: str | Path,
        hmrc_ods_path: str | Path,
        *,
        stage: SourceStageSpec,
        seed: int = 42,
        qrf_estimators: int = 100,
        donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE,
        sampled_rung: bool = False,
        donor_table: pd.DataFrame | None = None,
        source_targets: HMRCIncomeTargetSet | None = None,
    ) -> None:
        if donor_table is not None and not isinstance(donor_table, pd.DataFrame):
            raise TypeError("donor_table must be a pandas DataFrame.")
        if source_targets is not None and not isinstance(
            source_targets, HMRCIncomeTargetSet
        ):
            raise TypeError("source_targets must be an HMRCIncomeTargetSet.")
        object.__setattr__(self, "spi_tab_path", Path(spi_tab_path))
        object.__setattr__(self, "hmrc_ods_path", Path(hmrc_ods_path))
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "qrf_estimators", qrf_estimators)
        object.__setattr__(self, "donor_sample_size", donor_sample_size)
        object.__setattr__(self, "sampled_rung", bool(sampled_rung))
        object.__setattr__(
            self,
            "donor_table",
            None if donor_table is None else donor_table.copy(deep=True),
        )
        object.__setattr__(self, "source_targets", source_targets)
        object.__setattr__(self, "last_result", None)

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        if self.last_result is None:
            return ()
        return tuple(self.last_result.imputation.fit_weight_records)

    def __call__(self, frame: Frame) -> Frame:
        validate_uk_national_frame(frame)
        _assert_income_stage_parameters(
            self.stage,
            seed=self.seed,
            qrf_estimators=self.qrf_estimators,
            donor_sample_size=self.donor_sample_size,
        )
        donor_identity = (
            verify_spi_donor_identity(self.spi_tab_path)
            if self.donor_table is None
            else None
        )
        ods_identity = (
            verify_hmrc_spi_collated_ods(self.hmrc_ods_path)
            if self.source_targets is None
            else None
        )
        tables = engine_tables(frame)
        assert_frs_hmrc_auxiliary_crosswalk_available(tables["person"])
        support = _support_result_from_frame(frame, tables)
        stage1_op = _operation(self.stage, "fit_weighted_qrf_stage1")
        redraw_op = _operation(self.stage, "redraw_columns_from_fitted_qrf")
        imputation = impute_uk_spi_income_support(
            support,
            self.spi_tab_path,
            seed=self.seed,
            n_estimators=self.qrf_estimators,
            donor_sample_size=self.donor_sample_size,
            build_period=uk_time_period(frame),
            verified_donor=donor_identity,
            donor_table=self.donor_table,
            initialize_frs_channel_columns=stage1_op.parameters[
                "initialize_frs_channel_columns"
            ],
            stage1_base_redraw_columns=redraw_op.parameters["columns"],
            rebase_income_to_build_period=stage1_op.parameters[
                "rebase_income_to_build_period"
            ],
            condition_on_state_pension_receipt=(
                "state_pension_receipt"
                in _operation(self.stage, "fit_weighted_qrf_stage2").parameters[
                    "predictors"
                ]
            ),
        )
        result_frame = uk_national_frame(
            person=imputation.person,
            benunit=tables["benunit"],
            household=tables["household"],
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result_frame)
        source_targets = (
            materialize_hmrc_spi_income_band_targets(
                ods_identity,
                build_period=uk_time_period(frame),
            )
            if self.source_targets is None
            else self.source_targets
        )
        identity_rows = _assert_post_draw_identity(result_frame)
        distributional_mass_shares = _distributional_mass_shares(result_frame)
        insufficient = {
            name: share
            for name, share in distributional_mass_shares.items()
            if share < DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
        }
        if insufficient and not self.sampled_rung:
            raise RuntimeError(
                "Rebuilt SPI spine channel did not restore required "
                f"effective-mass coverage: {insufficient}."
            )
        # On a sampled rung the scaled stack cannot carry mass in
        # ultra-sparse restored columns (e.g. charitable_investment_gifts at
        # ~0.02 percent density); the gap is receipted via the replay
        # report's distributional_mass_shares — the full-scale fence above
        # stays strict.
        report = _build_spine_replay_report(
            source_targets=source_targets,
            support=support,
            imputation=imputation,
            frame=result_frame,
            distributional_mass_shares=distributional_mass_shares,
            identity_rows=identity_rows,
        )
        result = UKSPIIncomeSpineResult(
            frame=result_frame,
            support=support,
            imputation=imputation,
            source_targets=source_targets,
            replay_report=report,
            distributional_mass_shares=distributional_mass_shares,
            post_draw_identity_rows=identity_rows,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {
            "fit_weight_records": [
                {
                    "fit_name": record.fit_name,
                    "weight_kind": record.weight_kind,
                }
                for record in self.last_result.imputation.fit_weight_records
            ],
            "evidence": self.last_result.evidence(),
            "replay_payload": self.last_result.replay_report.to_payload(),
        }


def _support_result_from_frame(
    frame: Frame,
    tables: Mapping[str, pd.DataFrame],
) -> UKSPISupportResult:
    household = tables["household"]
    channel = support_channel_column("household")
    if channel not in household:
        raise ValueError("SPI income spine stage requires a support-channel frame.")
    spi_households = household[channel].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL)
    if not spi_households.any():
        raise ValueError("SPI income spine stage found no SPI support households.")
    return UKSPISupportResult(
        person=tables["person"],
        benunit=tables["benunit"],
        household=household,
        id_multiplier=1,
        spi_household_ids=tuple(household.loc[spi_households, "household_id"]),
        household_weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
        spi_prior_mass_share=DEFAULT_SPI_PRIOR_MASS_SHARE,
    )


def _assert_post_draw_identity(frame: Frame) -> int:
    """Require deterministic TEI + TII = TI on every rebuilt SPI draw."""

    person = frame.table("person")
    channel = support_channel_column("person")
    required = (
        channel,
        SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
        SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
        "hmrc_spi_assessable_income",
    )
    missing = sorted(set(required) - set(person.columns))
    if missing:
        raise RuntimeError(
            f"HMRC replay omitted post-draw identity column(s): {missing}."
        )
    spi = person[channel].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    if not spi.any():
        raise RuntimeError("HMRC replay contains no rebuilt SPI person draws.")
    numeric = person.loc[
        spi,
        [
            SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
            SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
            "hmrc_spi_assessable_income",
        ],
    ].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("HMRC post-draw identity contains non-finite values.")
    if not np.array_equal(
        numeric["hmrc_spi_assessable_income"].to_numpy(dtype=float),
        numeric[SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN].to_numpy(dtype=float)
        + numeric[SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN].to_numpy(dtype=float),
    ):
        raise RuntimeError("HMRC TI must equal deterministic TEI + TII exactly.")
    return int(spi.sum())


def _distributional_mass_shares(frame: Frame) -> dict[str, float]:
    """Audit charitable signal on strictly positive rebuilt-SPI mass."""

    person = frame.table("person")
    person_channel = support_channel_column("person")
    if person_channel not in person:
        raise RuntimeError(
            "Cannot audit HMRC distributional inputs without person support "
            "channel provenance."
        )
    spi_people = (
        person[person_channel].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    )
    if not spi_people.any():
        raise RuntimeError("Rebuilt HMRC family contains no SPI support people.")
    household = frame.table("household")
    household_weights = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"].to_numpy(),
    )
    mapped = pd.to_numeric(
        person["person_household_id"].map(household_weights),
        errors="coerce",
    ).to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(mapped).all() or (mapped < 0.0).any():
        raise RuntimeError(
            "Cannot audit HMRC distributional inputs without finite, "
            "non-negative person mass."
        )
    positive = mapped > 0.0
    total = float(mapped[positive].sum())
    if total <= 0.0:
        raise RuntimeError("HMRC distributional audit has no positive person mass.")
    shares: dict[str, float] = {}
    for column in HMRC_DISTRIBUTIONAL_INPUTS:
        if column not in person:
            raise RuntimeError(f"HMRC stage omitted distributional input {column!r}.")
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"HMRC distributional input {column!r} is non-finite.")
        shares[column] = (
            float(mapped[positive & spi_people & (values != 0.0)].sum()) / total
        )
    return shares


def _build_spine_replay_report(
    *,
    source_targets: HMRCIncomeTargetSet,
    support: UKSPISupportResult,
    imputation: UKSPIIncomeImputationResult,
    frame: Frame,
    distributional_mass_shares: Mapping[str, float],
    identity_rows: int,
) -> HMRCReplayReport:
    source_evidence = {
        "spi_donor": {
            "release": "2022-23",
            "sha256": imputation.donor_sha256,
            "size_bytes": imputation.donor_size_bytes,
            "rows_used": imputation.donor_rows,
        },
        "hmrc_surface": {
            "vintage": source_targets.source.source_vintage,
            "mapped_build_period": source_targets.source.build_period,
            "sha256": source_targets.source.sha256,
            "size_bytes": source_targets.source.size_bytes,
            "mime_type": source_targets.source.mime_type,
            "tables": list(source_targets.source.table_names),
        },
    }
    build_evidence = {
        "stage": UK_HMRC_SPI_INCOME_SPINE_STAGE_NAME,
        "output_weight_kind": uk_household_weight_kind(frame).value,
        "calibration_performed": False,
        "spi_prior_mass_share": support.spi_prior_mass_share,
        "spi_households": support.n_spi_households,
        "mass_change_reason": support.mass_log[-1].reason,
    }
    qrf_evidence = {
        "fits": {
            record.fit_name: {"weight_kind": record.weight_kind}
            for record in imputation.fit_weight_records
        },
        "donor_rows": imputation.donor_rows,
        "stage2_training_rows": imputation.stage2_training_rows,
        "spi_prediction_rows": imputation.spi_prediction_rows,
        "post_draw_identity": {
            "formula": SPI_SOURCE_TI_FORMULA,
            "rows_checked": identity_rows,
            "exact": True,
        },
    }
    effective_mass_evidence = {
        "minimum_nondefault_mass_share": DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE,
        "denominator": "all_person_effective_mass",
        "required_support_channel": SPI_SYNTHETIC_SUPPORT_CHANNEL,
        "columns": dict(distributional_mass_shares),
    }
    return build_conservative_hmrc_replay_report(
        source_targets,
        source_evidence=source_evidence,
        build_evidence=build_evidence,
        qrf_evidence=qrf_evidence,
        effective_mass_evidence=effective_mass_evidence,
        report_kind=UK_HMRC_SPI_SPINE_REPLAY_REPORT_KIND,
    )


def _support_stage_parameters(
    stage: SourceStageSpec,
    *,
    seed: int,
) -> tuple[int, float, tuple[str, ...], tuple[UKZeroWeightStratumDeclaration, ...]]:
    stack = _operation(stage, "stack_zero_weight_donors")
    gate = _operation(stage, "gate_zero_weight_strata")
    allocation = _operation(stage, "allocate_zero_weight_prior_mass")
    if stack.parameters.get("count") != DEFAULT_SPI_SUPPORT_HOUSEHOLDS:
        raise ValueError("SPI support manifest count drifted from the reviewed value.")
    if stack.parameters.get("seed") != seed:
        raise ValueError("SPI support manifest seed drifted from the reviewed value.")
    if stack.parameters.get("draw") != "uniform_without_replacement":
        raise ValueError("SPI support manifest must declare a uniform donor draw.")
    if stack.parameters.get("flag_column") != HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN:
        raise ValueError("SPI support manifest flag column drifted.")
    if dict(stack.parameters.get("channels", {})) != SPI_SPINE_SUPPORT_CHANNELS:
        raise ValueError("SPI support manifest channel labels drifted.")
    if allocation.parameters.get("share") != DEFAULT_SPI_PRIOR_MASS_SHARE:
        raise ValueError("SPI support prior-mass share drifted from 0.5.")
    if allocation.parameters.get("weight_kind_out") != WeightKind.IMPORTANCE.value:
        raise ValueError("SPI support allocation must advance to importance weights.")
    if allocation.parameters.get("conservation") != "exact_total":
        raise ValueError(
            "SPI support allocation must declare exact-total conservation."
        )
    strata = tuple(allocation.parameters.get("strata", ()))
    if strata != ("region",):
        raise ValueError("SPI support spine allocation must use strata ['region'].")
    declarations = _coerce_declarations(gate.parameters.get("declarations", ()))
    if len(declarations) != 1:
        raise ValueError("SPI support gate must declare exactly one pre-clone stratum.")
    declaration = declarations[0]
    if (
        declaration.name != SPI_SPINE_PRECLONE_GATE_NAME
        or dict(declaration.selector) != {HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True}
        or declaration.maximum_zero_weight_rows != DEFAULT_SPI_SUPPORT_HOUSEHOLDS
    ):
        raise ValueError(
            "SPI support pre-clone gate declaration drifted from the reviewed "
            "name/selector/maximum."
        )
    return (
        int(stack.parameters["count"]),
        float(allocation.parameters["share"]),
        strata,
        declarations,
    )


def _assert_income_stage_parameters(
    stage: SourceStageSpec,
    *,
    seed: int,
    qrf_estimators: int,
    donor_sample_size: int | None,
) -> None:
    stage1 = _operation(stage, "fit_weighted_qrf_stage1")
    stage2 = _operation(stage, "fit_weighted_qrf_stage2")
    redraw = _operation(stage, "redraw_columns_from_fitted_qrf")
    effective_mass = _operation(stage, "gate_distributional_effective_mass")
    if stage1.parameters.get("seed") != seed:
        raise ValueError("SPI income stage-1 seed drifted from the reviewed value.")
    if stage2.parameters.get("seed") != seed + 1:
        raise ValueError(
            "SPI income stage-2 seed drifted from the reviewed seed + 1 "
            "derivation convention."
        )
    if stage1.parameters.get("n_estimators") not in (None, qrf_estimators):
        raise ValueError("SPI income stage-1 estimator count drifted.")
    if stage2.parameters.get("n_estimators") not in (None, qrf_estimators):
        raise ValueError("SPI income stage-2 estimator count drifted.")
    if stage1.parameters.get("sample_size") != donor_sample_size:
        raise ValueError("SPI income donor sample size drifted.")
    if stage1.parameters.get("post_sample_fit_weight") != "uniform":
        raise ValueError("SPI income donor bootstrap must refit at uniform weight.")
    if tuple(stage1.parameters.get("predictors", ())) != SPI_SPINE_STAGE1_PREDICTORS:
        raise ValueError("SPI income stage-1 predictors drifted.")
    if stage1.parameters.get("recipient_minimum_age") != SPI_MINIMUM_RECIPIENT_AGE:
        raise ValueError("SPI income recipient age domain drifted.")
    if stage1.parameters.get("rebase_income_to_build_period") is not True:
        raise ValueError("SPI income build-year rebasing drifted.")
    if stage1.parameters.get("donor_income_period") != SPI_DONOR_INCOME_YEAR:
        raise ValueError("SPI donor income period drifted.")
    if (
        stage1.parameters.get("income_uprating_variables")
        != SPI_INCOME_UPRATING_VARIABLES
    ):
        raise ValueError("SPI income uprating-variable map drifted.")
    if tuple(stage1.parameters.get("outputs", ())) != SPI_INCOME_QRF_OUTPUT_COLUMNS:
        raise ValueError("SPI income stage-1 outputs drifted.")
    initialization = dict(stage1.parameters.get("initialize_frs_channel_columns", {}))
    if initialization != SPI_SPINE_FRS_CHANNEL_INITIALIZATION:
        raise ValueError("SPI income FRS-channel initialization map drifted.")
    if tuple(stage2.parameters.get("predictors", ())) != SPI_SPINE_STAGE2_PREDICTORS:
        raise ValueError("SPI income stage-2 predictors drifted.")
    if tuple(stage2.parameters.get("outputs", ())) != SPI_SPINE_STAGE2_OUTPUT_COLUMNS:
        raise ValueError("SPI income stage-2 outputs drifted.")
    reviewed_absent = stage2.parameters.get("reviewed_absent_outputs", {})
    if set(reviewed_absent) != set(SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS):
        raise ValueError("SPI income stage-2 reviewed-absent outputs drifted.")
    if redraw.parameters.get("fit") != "stage1":
        raise ValueError("SPI income base redraw must use the stage-1 fit.")
    if redraw.parameters.get("rows") != "base_support_channel":
        raise ValueError("SPI income base redraw must target the base support channel.")
    if tuple(redraw.parameters.get("columns", ())) != SPI_SPINE_BASE_REDRAW_COLUMNS:
        raise ValueError("SPI income base redraw columns drifted.")
    if (
        tuple(effective_mass.parameters.get("columns", ()))
        != SPI_SPINE_EFFECTIVE_MASS_COLUMNS
    ):
        raise ValueError("SPI income effective-mass columns drifted.")
    if (
        effective_mass.parameters.get("minimum_nondefault_mass_share")
        != DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
    ):
        raise ValueError("SPI income effective-mass floor drifted.")
    if effective_mass.parameters.get("fail_below_floor") is not True:
        raise ValueError("SPI income effective-mass gate must fail below the floor.")
    if (
        effective_mass.parameters.get("required_support_channel")
        != SPI_SYNTHETIC_SUPPORT_CHANNEL
    ):
        raise ValueError("SPI income effective-mass support channel drifted.")


def _operation(stage: SourceStageSpec, kind: str):
    matches = [operation for operation in stage.operations if operation.kind == kind]
    if len(matches) != 1:
        raise ValueError(
            f"Stage {stage.stage!r} must declare exactly one {kind!r} operation."
        )
    return matches[0]


def _artifact_by_table(
    stage: SourceStageSpec,
    *,
    expected: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    artifacts = {
        str(artifact["table"]): artifact
        for artifact in stage.artifacts
        if "table" in artifact
    }
    missing = sorted(set(expected) - set(artifacts))
    if missing:
        raise ValueError(
            f"Stage {stage.stage!r} is missing tab artifact(s): {missing}."
        )
    return artifacts


def _assert_identity_matches_artifact(
    evidence: Mapping[str, object],
    artifact: Mapping[str, Any],
) -> None:
    if evidence["sha256"] != artifact["sha256"]:
        raise ValueError(
            f"FRS tab {artifact['locator']!r} hashes to {evidence['sha256']}, "
            f"not the pinned {artifact['sha256']}."
        )
    if evidence["size_bytes"] != artifact["size_bytes"]:
        raise ValueError(
            f"FRS tab {artifact['locator']!r} is {evidence['size_bytes']} bytes, "
            f"not the pinned {artifact['size_bytes']}."
        )


def _coerce_declarations(
    values: object,
) -> tuple[UKZeroWeightStratumDeclaration, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise ValueError("zero-weight declarations must be a sequence.")
    declarations = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("zero-weight declarations must be mappings.")
        declarations.append(UKZeroWeightStratumDeclaration(**dict(value)))
    return tuple(declarations)


__all__ = [
    "EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN",
    "UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS",
    "UK_FRS_HMRC_SPINE_LEAVES_STAGE_NAME",
    "UK_HMRC_SPI_INCOME_SPINE_STAGE_NAME",
    "UK_HMRC_SPI_SPINE_REPLAY_REPORT_KIND",
    "UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS",
    "UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS",
    "UK_SPI_INCOME_SPINE_REWRITE_COLUMNS",
    "UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS",
    "UKFRSHMRCSpineLeavesResult",
    "UKFRSHMRCSpineLeavesStageTransform",
    "UKSPIIncomeSpineResult",
    "UKSPIIncomeSpineStageTransform",
    "UKSPISupportChannelStageTransform",
]
