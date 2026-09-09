from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.frs_education_grants import (
    DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES,
    UK_EDUCATION_GRANT_CAPACITY_PREDICTORS,
    UKDSAPolicy,
    UKFRSEducationGrantSplitStageTransform,
    _assert_frs_education_grant_stage_parameters,
    allocate_reported_education_grants,
    disabled_students_allowance_capacity,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
)


def test_grant_allocator_caps_fraction_and_keeps_residual() -> None:
    result = allocate_reported_education_grants(
        [60.0, 300.0],
        {
            "childcare_grant": np.array([100.0, 100.0]),
            "parents_learning_allowance": np.array([100.0, 0.0]),
            "adult_dependants_grant": np.array([0.0, 0.0]),
            "dsa": np.array([0.0, 50.0]),
        },
    )

    assert result["childcare_grant"].tolist() == [30.0, 100.0]
    assert result["parents_learning_allowance"].tolist() == [30.0, 0.0]
    assert result["dsa"].tolist() == [0.0, 50.0]
    assert result["education_grants"].tolist() == [0.0, 150.0]


def test_pre_2025_dsa_capacity_is_aligned_zero_vector() -> None:
    person = pd.DataFrame(index=[10, 20, 30])

    result = disabled_students_allowance_capacity(
        person,
        capacities={},
        policy=UKDSAPolicy(maximum=100.0, instant="2023-01-01", source="fixture"),
        policy_year=2023,
    )

    assert result.shape == (3,)
    assert result.tolist() == [0.0, 0.0, 0.0]


def test_2025_dsa_capacity_seeds_maximum_for_fully_eligible_people() -> None:
    person = pd.DataFrame(index=[10, 20, 30, 40])
    capacities = {
        "maintenance_loan_in_england_system": np.array([True, True, True, False]),
        "disabled_students_allowance_course_eligible": np.array(
            [True, True, False, True]
        ),
        "disabled_students_allowance_has_qualifying_condition": np.array(
            [True, False, True, True]
        ),
    }

    result = disabled_students_allowance_capacity(
        person,
        capacities=capacities,
        policy=UKDSAPolicy(maximum=100.0, instant="2025-01-01", source="fixture"),
        policy_year=2025,
    )

    assert result.tolist() == [100.0, 0.0, 0.0, 0.0]


def test_transform_materializes_survey_and_calibration_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = load_country_spec("uk").sources.stage_map()["frs_education_grant_split"]
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_benunit_id": [1],
                "person_household_id": [1],
                "education_grants": [100.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame({"household_id": [1]}),
        time_period="2024",
        household_weights=[1.0],
    )

    class _StubEngine:
        country = "uk"

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], int]] = []

        def materialize(self, frame, variables, period):
            self.calls.append((tuple(variables), period))
            if tuple(variables) == UK_EDUCATION_GRANT_CAPACITY_PREDICTORS:
                return {name: np.array([0.0]) for name in variables}
            return {name: np.array([True]) for name in variables}

    policy_years: list[int] = []

    def _policy(policy_year: int) -> UKDSAPolicy:
        policy_years.append(policy_year)
        return UKDSAPolicy(
            maximum=100.0,
            instant=f"{policy_year}-01-01",
            source="fixture",
        )

    monkeypatch.setattr(
        "microcosm.build.uk_runtime.frs_education_grants.uk_dsa_policy",
        _policy,
    )
    engine = _StubEngine()

    result = UKFRSEducationGrantSplitStageTransform(stage=stage, engine=engine)(frame)

    assert engine.calls == [
        (UK_EDUCATION_GRANT_CAPACITY_PREDICTORS, 2024),
        (DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES, 2025),
    ]
    assert policy_years == [2025]
    assert uk_time_period(result) == "2024"


@pytest.mark.parametrize("operation_index", [0, 1, 2])
def test_grant_stage_rejects_year_rule_drift(operation_index: int) -> None:
    stage = load_country_spec("uk").sources.stage_map()["frs_education_grant_split"]
    operations = list(stage.operations)
    operation = operations[operation_index]
    operations[operation_index] = SourceOperationSpec(
        operation.kind,
        {**operation.parameters, "year_rule": "__drift__"},
    )

    with pytest.raises(ValueError, match="declaration drifted"):
        _assert_frs_education_grant_stage_parameters(
            replace(stage, operations=tuple(operations))
        )
