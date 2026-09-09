"""Split aggregate FRS education grants into modelled capacities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
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
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

FRS_EDUCATION_GRANT_OUTPUT_COLUMNS = (
    "disabled_students_allowance_eligible_expenses",
)
FRS_EDUCATION_GRANT_REWRITES = ("education_grants",)
UK_EDUCATION_GRANT_CAPACITY_PREDICTORS = (
    "childcare_grant",
    "parents_learning_allowance",
    "adult_dependants_grant",
)
DISABLED_STUDENTS_ALLOWANCE_FIRST_MODELED_YEAR = 2025
DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES = (
    "maintenance_loan_in_england_system",
    "disabled_students_allowance_course_eligible",
    "disabled_students_allowance_has_qualifying_condition",
)
GRANT_CAPACITY_YEAR_RULE = "survey_year"
DSA_YEAR_RULE = "calibration_year"


@dataclass(frozen=True)
class UKDSAPolicy:
    """DSA maximum read at 1 January of the declared policy year."""

    maximum: float
    instant: str
    source: str


class UKFRSEducationGrantSplitStageTransform:
    """Whole-stage callable for FRS education grant splitting.

    ``PolicyEngineUKEngine.materialize`` labels the survey-year tables as a
    calibration-year dataset rather than uprating them, where uk-data lets
    the engine uprate the survey-year dataset and calculates at the policy
    year. The three DSA eligibility variables are non-monetary booleans;
    the licensed receipt for #862 measured them identical under both paths
    on every person (experiments/862-policy-year-rule-receipts.md).
    """

    def __init__(
        self,
        *,
        stage: SourceStageSpec,
        engine: object,
        policy: UKDSAPolicy | None = None,
    ) -> None:
        self.stage = stage
        self.engine = engine
        self.policy = policy

    def __call__(self, frame: Frame) -> Frame:
        _assert_frs_education_grant_stage_parameters(self.stage)
        assert_rules_engine_country(self.engine, "uk")
        survey_year = resolve_uk_year_rule(GRANT_CAPACITY_YEAR_RULE)
        policy_year = resolve_uk_year_rule(DSA_YEAR_RULE)
        capacities = self.engine.materialize(
            frame,
            UK_EDUCATION_GRANT_CAPACITY_PREDICTORS,
            survey_year,
        )
        if policy_year >= DISABLED_STUDENTS_ALLOWANCE_FIRST_MODELED_YEAR:
            capacities.update(
                self.engine.materialize(
                    frame,
                    DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES,
                    policy_year,
                )
            )
            policy = self.policy or uk_dsa_policy(policy_year)
        else:
            policy = self.policy or UKDSAPolicy(
                maximum=0.0,
                instant=f"{policy_year}-01-01",
                source="pre-2025 DSA not modelled",
            )
        return add_frs_education_grant_split(
            frame,
            capacities=capacities,
            policy=policy,
            policy_year=policy_year,
        )

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_EDUCATION_GRANT_OUTPUT_COLUMNS


def uk_dsa_policy(policy_year: int) -> UKDSAPolicy:
    """Read the DSA maximum at ``{policy_year}-01-01`` (uk-data#478)."""

    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "UK DSA parameters require `uv sync --all-packages --extra uk`."
        ) from exc

    parameters = ParameterNode(
        directory_path=str(Path(policyengine_uk.__file__).parent / "parameters")
    )
    instant = f"{policy_year}-01-01"
    return UKDSAPolicy(
        maximum=float(parameters.gov.dfe.disabled_students_allowance.maximum(instant)),
        instant=instant,
        source="policyengine-uk parameters "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


def add_frs_education_grant_split(
    frame: Frame,
    *,
    capacities: dict[str, np.ndarray],
    policy: UKDSAPolicy,
    policy_year: int,
) -> Frame:
    person = frame.table("person").copy()
    dsa_capacity = disabled_students_allowance_capacity(
        person,
        capacities=capacities,
        policy=policy,
        policy_year=policy_year,
    )
    grant_capacities = {
        name: capacities[name] for name in UK_EDUCATION_GRANT_CAPACITY_PREDICTORS
    }
    split = allocate_reported_education_grants(
        person["education_grants"], {**grant_capacities, "dsa": dsa_capacity}
    )
    person["education_grants"] = split["education_grants"]
    person["disabled_students_allowance_eligible_expenses"] = split["dsa"]
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result


def allocate_reported_education_grants(
    reported_grants, grant_capacities: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    reported = np.maximum(np.nan_to_num(np.asarray(reported_grants, dtype=float)), 0.0)
    capacities = {
        name: np.maximum(np.nan_to_num(np.asarray(value, dtype=float)), 0.0)
        for name, value in grant_capacities.items()
    }
    total_capacity = np.zeros_like(reported)
    for name, capacity in capacities.items():
        if capacity.shape != reported.shape:
            raise ValueError(
                f"{name} capacity has shape {capacity.shape}, expected {reported.shape}."
            )
        total_capacity += capacity
    fraction = np.divide(
        reported,
        total_capacity,
        out=np.zeros_like(reported),
        where=total_capacity > 0,
    )
    fraction = np.minimum(fraction, 1.0)
    result: dict[str, np.ndarray] = {}
    allocated = np.zeros_like(reported)
    for name, capacity in capacities.items():
        result[name] = capacity * fraction
        allocated += result[name]
    result["education_grants"] = np.maximum(reported - allocated, 0.0)
    return result


def disabled_students_allowance_capacity(
    person: pd.DataFrame,
    *,
    capacities: dict[str, np.ndarray],
    policy: UKDSAPolicy,
    policy_year: int,
) -> np.ndarray:
    if policy_year < DISABLED_STUDENTS_ALLOWANCE_FIRST_MODELED_YEAR:
        return np.zeros(len(person), dtype=float)
    eligible = np.ones(len(person), dtype=bool)
    for variable in DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES:
        eligible &= np.asarray(capacities[variable], dtype=bool)
    return np.where(eligible, policy.maximum, 0.0)


def _assert_frs_education_grant_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind all grant-split operations to the reviewed policy-year contract."""

    _assert_closed_world_operations(
        stage,
        (
            (
                "materialize_rules_engine_predictors",
                {
                    "predictors": list(UK_EDUCATION_GRANT_CAPACITY_PREDICTORS),
                    "consumed_only": True,
                    "year_rule": GRANT_CAPACITY_YEAR_RULE,
                },
            ),
            (
                "materialize_rules_engine_predictors",
                {
                    "predictors": list(
                        DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES
                    ),
                    "consumed_only": True,
                    "year_rule": DSA_YEAR_RULE,
                },
            ),
            (
                "derive",
                {
                    "scope": "proportional split of aggregate education_grants and DSA residual capacity",
                    "parameters": "DSA maximum from gov.dfe.disabled_students_allowance.maximum at the calibration year",
                    "year_rule": DSA_YEAR_RULE,
                },
            ),
        ),
    )
