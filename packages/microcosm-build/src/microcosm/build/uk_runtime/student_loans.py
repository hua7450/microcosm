"""UK student-loan cohort assignment and SLC liable-stock support top-ups."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.cgt_structure import (
    _assert_closed_world_operations,
)
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame, MassChangeRecord

PLAN_1_BEFORE = 2012
PLAN_5_FROM = 2023
PLAN_2_MIN_AGE = 21
PLAN_2_MAX_AGE = 55
PLAN_5_MIN_AGE = 18
PLAN_5_MAX_AGE = 25
PLAN_PRIORITY = ("PLAN_5", "PLAN_2")
STUDENT_LOAN_SEED = 42
YEAR_RULE = "calibration_year"
STUDENT_LOAN_ENUM_DOMAIN = ("NONE", "PLAN_1", "PLAN_2", "PLAN_5")
EXCLUDED_ENGLAND_REGIONS = ("SCOTLAND", "WALES", "NORTHERN_IRELAND")
PLAN_SALTS = {
    "PLAN_5": "student_loan_plan_5",
    "PLAN_2": "student_loan_plan_2",
}
PLAN_2025_STOCKS = {"PLAN_2": 8_940_000.0, "PLAN_5": 10_000.0}
STUDENT_LOANS_MASS_CHANGE_REASON = (
    "Student-loan plan assignment writes an enum column only; household rows "
    "and typed household weights pass through and total household mass is "
    "conserved."
)


def load_slc_liable_stocks() -> Mapping[str, Any]:
    """Load the pinned SLC Table 6a liable-stock series."""

    return json.loads(
        files("microcosm.build.uk")
        .joinpath("slc_liable_stocks.json")
        .read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class UKStudentLoanPlanReceipt:
    plan: str
    stock: float
    reported_count: float
    reported_england_count: float
    shortfall: float
    eligible_mass: float
    rate: float
    topped_up_rows: int
    topped_up_mass: float
    expected_topped_up_mass: float
    realization_deviation: float
    final_england_count: float

    def evidence(self) -> dict[str, object]:
        return {
            "stock": self.stock,
            "reported_count": self.reported_count,
            "reported_england_count": self.reported_england_count,
            "shortfall": self.shortfall,
            "eligible_mass": self.eligible_mass,
            "rate": self.rate,
            "topped_up_rows": self.topped_up_rows,
            "topped_up_mass": self.topped_up_mass,
            "expected_topped_up_mass": self.expected_topped_up_mass,
            "realization_deviation": self.realization_deviation,
            "final_england_count": self.final_england_count,
        }


@dataclass(frozen=True)
class UKStudentLoansResult:
    """Transformed frame plus a per-plan executed-effect receipt."""

    frame: Frame
    calibration_year: int
    plans: Mapping[str, UKStudentLoanPlanReceipt]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": "student_loans",
            "year_rule": YEAR_RULE,
            "calibration_year": self.calibration_year,
            "plans": {name: receipt.evidence() for name, receipt in self.plans.items()},
        }


@dataclass(frozen=True)
class UKStudentLoansStageTransform:
    """Whole-stage transform for student-loan plan support."""

    stage: SourceStageSpec
    stocks: Mapping[str, Any] | None = None
    calibration_year: int | None = None
    last_result: UKStudentLoansResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        resource = self.stocks or load_slc_liable_stocks()
        year = (
            self.calibration_year
            if self.calibration_year is not None
            else resolve_uk_year_rule(YEAR_RULE)
        )
        _assert_student_loans_stage_parameters(self.stage, stocks=resource, year=year)
        result = assign_student_loan_plans(frame, stocks=resource, year=year)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ("student_loan_plan",)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def assign_student_loan_plans(
    frame: Frame,
    *,
    stocks: Mapping[str, Any],
    year: int,
) -> UKStudentLoansResult:
    """Assign reported cohorts, then top up PLAN_5 before PLAN_2."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    required = {
        "person_id",
        "person_household_id",
        "age",
        "student_loan_repayments",
        "highest_education",
    }
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(f"Student-loan person columns missing: {missing}.")
    if "region" not in household.columns:
        raise ValueError("Student-loan assignment requires household region.")
    region_by_household = household.set_index("household_id")["region"]
    region = person["person_household_id"].map(region_by_household)
    if region.isna().any():
        raise ValueError("Student-loan people must all map to a household region.")
    region_names = np.asarray([_enum_name(value) for value in region], dtype=object)
    is_england = ~np.isin(region_names, EXCLUDED_ENGLAND_REGIONS)
    education = np.asarray(
        [_enum_name(value) for value in person["highest_education"]], dtype=object
    )
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float)
    repayments = pd.to_numeric(
        person["student_loan_repayments"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(age).all() or not np.isfinite(repayments).all():
        raise ValueError("Student-loan age and repayment inputs must be finite.")
    weights_by_household = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"],
    )
    person_weights = (
        person["person_household_id"].map(weights_by_household).to_numpy(dtype=float)
    )
    start_year = year - age + 18
    has_repayments = repayments > 0.0
    plan = np.full(len(person), "NONE", dtype=object)
    plan[has_repayments & (start_year < PLAN_1_BEFORE)] = "PLAN_1"
    plan[has_repayments & (start_year >= PLAN_5_FROM)] = "PLAN_5"
    plan[has_repayments & (plan == "NONE")] = "PLAN_2"
    reported_plan = plan.copy()
    receipts: dict[str, UKStudentLoanPlanReceipt] = {}
    for plan_name in PLAN_PRIORITY:
        stock = _stock(stocks, plan_name, year)
        current_england = float(person_weights[(plan == plan_name) & is_england].sum())
        shortfall = max(0.0, stock - current_england)
        eligible = (
            (plan == "NONE")
            & is_england
            & (education == "TERTIARY")
            & _plan_age_cohort_eligibility(plan_name, age=age, start_year=start_year)
        )
        eligible_mass = float(person_weights[eligible].sum())
        rate = min(1.0, shortfall / eligible_mass) if eligible_mass > 0.0 else 0.0
        draws = stable_identity_uniforms(
            person["person_id"].to_numpy(),
            seed=STUDENT_LOAN_SEED,
            salt=PLAN_SALTS[plan_name],
        )
        topped_up = eligible & (draws < rate)
        plan[topped_up] = plan_name
        receipts[plan_name] = UKStudentLoanPlanReceipt(
            plan=plan_name,
            stock=stock,
            reported_count=float(person_weights[reported_plan == plan_name].sum()),
            reported_england_count=float(
                person_weights[(reported_plan == plan_name) & is_england].sum()
            ),
            shortfall=shortfall,
            eligible_mass=eligible_mass,
            rate=rate,
            topped_up_rows=int(topped_up.sum()),
            topped_up_mass=float(person_weights[topped_up].sum()),
            expected_topped_up_mass=rate * eligible_mass,
            realization_deviation=(
                (float(person_weights[topped_up].sum()) - rate * eligible_mass)
                / (rate * eligible_mass)
                if rate * eligible_mass > 0.0
                else 0.0
            ),
            final_england_count=float(
                person_weights[(plan == plan_name) & is_england].sum()
            ),
        )
    unknown = sorted(set(plan) - set(STUDENT_LOAN_ENUM_DOMAIN))
    if unknown:
        raise ValueError(f"Student-loan assignment emitted unknown plan(s): {unknown}.")
    person["student_loan_plan"] = plan
    total = frame.weights_for("household").total
    mass_receipt = MassChangeRecord(
        entity="household",
        old_total=total,
        new_total=total,
        declared_factor=1.0,
        reason=STUDENT_LOANS_MASS_CHANGE_REASON,
    )
    result_frame = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=(*frame.mass_log, mass_receipt),
    )
    validate_uk_national_frame(result_frame)
    return UKStudentLoansResult(
        frame=result_frame,
        calibration_year=year,
        plans=receipts,
    )


def _plan_age_cohort_eligibility(
    plan: str,
    *,
    age: np.ndarray,
    start_year: np.ndarray,
) -> np.ndarray:
    if plan == "PLAN_5":
        return (
            (age >= PLAN_5_MIN_AGE)
            & (age <= PLAN_5_MAX_AGE)
            & (start_year >= PLAN_5_FROM)
        )
    if plan == "PLAN_2":
        return (
            (age >= PLAN_2_MIN_AGE)
            & (age <= PLAN_2_MAX_AGE)
            & (start_year >= PLAN_1_BEFORE)
            & (start_year < PLAN_5_FROM)
        )
    raise ValueError(f"Unsupported student-loan top-up plan {plan!r}.")


def _stock(stocks: Mapping[str, Any], plan: str, year: int) -> float:
    key = plan.lower()
    try:
        values = stocks["plans"][key]["liable"]
        value = values[str(year)]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"SLC liable-stock resource has no {key} value for {year}."
        ) from error
    result = float(value)
    if result < 0 or not np.isfinite(result):
        raise ValueError(f"SLC {key} liable stock for {year} is invalid: {value!r}.")
    return result


def _enum_name(value: object) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _assert_student_loans_stage_parameters(
    stage: SourceStageSpec,
    *,
    stocks: Mapping[str, Any],
    year: int,
) -> None:
    """Bind every stage parameter closed-world; per-plan receipts supply arm 2."""

    region_exclusions = list(EXCLUDED_ENGLAND_REGIONS)
    _assert_closed_world_operations(
        stage,
        (
            (
                "assign_student_loan_plan_cohorts",
                {
                    "year_rule": YEAR_RULE,
                    "start_year_formula": "year - age + 18",
                    "reported_repayment_test": "student_loan_repayments > 0",
                    "reported_country_gate": False,
                    "plan_1_before": PLAN_1_BEFORE,
                    "plan_5_from": PLAN_5_FROM,
                    "enum_domain": list(STUDENT_LOAN_ENUM_DOMAIN),
                    "plan_4_imputation": False,
                },
            ),
            (
                "top_up_to_stock",
                {
                    "plan": "PLAN_5",
                    "priority": 1,
                    "resource": "slc_liable_stocks.json",
                    "stock_series": "plan_5.liable",
                    "year_rule": YEAR_RULE,
                    "age_min": PLAN_5_MIN_AGE,
                    "age_max": PLAN_5_MAX_AGE,
                    "cohort_start_min": PLAN_5_FROM,
                    "eligible_region_exclusions": region_exclusions,
                    "highest_education": "TERTIARY",
                    "seed": STUDENT_LOAN_SEED,
                    "salt": PLAN_SALTS["PLAN_5"],
                },
            ),
            (
                "top_up_to_stock",
                {
                    "plan": "PLAN_2",
                    "priority": 2,
                    "resource": "slc_liable_stocks.json",
                    "stock_series": "plan_2.liable",
                    "year_rule": YEAR_RULE,
                    "age_min": PLAN_2_MIN_AGE,
                    "age_max": PLAN_2_MAX_AGE,
                    "cohort_start_min": PLAN_1_BEFORE,
                    "cohort_start_max_exclusive": PLAN_5_FROM,
                    "eligible_region_exclusions": region_exclusions,
                    "highest_education": "TERTIARY",
                    "seed": STUDENT_LOAN_SEED,
                    "salt": PLAN_SALTS["PLAN_2"],
                    "reason": STUDENT_LOANS_MASS_CHANGE_REASON,
                },
            ),
        ),
    )
    if year == 2025:
        for plan, expected_stock in PLAN_2025_STOCKS.items():
            if _stock(stocks, plan, year) != expected_stock:
                raise ValueError(
                    f"SLC {plan} liable stock for 2025 drifted from {expected_stock}."
                )
