from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.frs_disability import (
    UKDWPDisabilityCategoryRates,
    UKDWPDisabilityFlagRates,
    _assert_frs_disability_stage_parameters,
    derive_frs_disability,
    uk_dwp_disability_category_rates,
)
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR


def _category_rates() -> UKDWPDisabilityCategoryRates:
    return UKDWPDisabilityCategoryRates(
        aa_lower=10,
        aa_higher=20,
        dla_sc_lower=10,
        dla_sc_middle=20,
        dla_sc_higher=30,
        dla_m_lower=10,
        dla_m_higher=20,
        pip_m_standard=10,
        pip_m_enhanced=20,
        pip_dl_standard=10,
        pip_dl_enhanced=20,
        instant="2024-01-01",
        source="fixture",
    )


def _flags() -> UKDWPDisabilityFlagRates:
    return UKDWPDisabilityFlagRates(
        aa_higher=20,
        dla_sc_higher=30,
        pip_dl_enhanced=20,
        instant="2024-01-01",
        source="fixture",
    )


def test_disability_category_threshold_and_overwrite() -> None:
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                (20 - 1) * WEEKS_IN_YEAR,
                (20 - 1.01) * WEEKS_IN_YEAR,
            ],
            "dla_sc_reported": [0, 0],
            "dla_m_reported": [0, 0],
            "pip_m_reported": [0, 0],
            "pip_dl_reported": [0, 0],
        }
    )

    result = derive_frs_disability(
        person, category_rates=_category_rates(), flag_rates=_flags()
    )

    assert result["aa_category"].tolist() == ["HIGHER", "LOWER"]


def test_disability_flag_operator_asymmetry_and_afcs() -> None:
    boundary = (30 - 1) * WEEKS_IN_YEAR
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [0, 0],
            "dla_sc_reported": [boundary, 0],
            "dla_m_reported": [0, 0],
            "pip_m_reported": [0, 0],
            "pip_dl_reported": [0, 0],
            "sda_reported": [0, 0],
            "incapacity_benefit_reported": [0, 0],
            "iidb_reported": [0, 0],
            "afcs_reported": [0, 1],
            "esa_contrib_reported": [0, 0],
            "esa_income_reported": [0, 0],
        }
    )

    result = derive_frs_disability(
        person, category_rates=_category_rates(), flag_rates=_flags()
    )

    assert result["is_enhanced_disabled_for_benefits"].tolist() == [False, False]
    assert result["is_severely_disabled_for_benefits"].tolist() == [True, True]


@pytest.mark.requires_uk
def test_dwp_readers_share_fiscal_converted_tree() -> None:
    # The readers construct the real engine's parameter tree; the wheel gate
    # and the us-extra CI lane run without policyengine-uk, so skip there.
    from microcosm.build.uk_runtime.frs_disability import (
        uk_dwp_disability_flag_rates,
    )

    category_2024 = uk_dwp_disability_category_rates(2024)
    flags_2024 = uk_dwp_disability_flag_rates(2024)
    category_2023 = uk_dwp_disability_category_rates(2023)
    flags_2023 = uk_dwp_disability_flag_rates(2023)

    assert category_2024.instant == flags_2024.instant == "2024-01-01"
    assert np.isfinite(category_2024.aa_lower)
    assert category_2024.aa_higher == flags_2024.aa_higher == pytest.approx(108.55)
    assert category_2023.aa_higher == flags_2023.aa_higher == pytest.approx(101.75)


@pytest.mark.parametrize("operation_index", [0, 1])
def test_disability_stage_rejects_year_rule_drift(operation_index: int) -> None:
    stage = load_country_spec("uk").sources.stage_map()["frs_disability"]
    operations = list(stage.operations)
    operation = operations[operation_index]
    operations[operation_index] = SourceOperationSpec(
        operation.kind,
        {**operation.parameters, "year_rule": "calibration_year"},
    )

    with pytest.raises(ValueError, match="declaration drifted"):
        _assert_frs_disability_stage_parameters(
            replace(stage, operations=tuple(operations))
        )


def test_disability_stage_resolves_the_survey_year_not_the_frame_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Survey year and frame period are both 2024 on the current release, so a
    # frame stamped 2023 is the only way to see which one picks the rates.
    from microcosm.build.uk_runtime import frs_disability
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        uk_time_period,
    )

    stage = load_country_spec("uk").sources.stage_map()["frs_disability"]
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_benunit_id": [1],
                "person_household_id": [1],
                "attendance_allowance_reported": [20 * WEEKS_IN_YEAR],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame({"household_id": [1]}),
        time_period="2023",
        household_weights=[1.0],
    )
    years: list[int] = []

    def _category_reader(year: int) -> UKDWPDisabilityCategoryRates:
        years.append(year)
        return _category_rates()

    def _flag_reader(year: int) -> UKDWPDisabilityFlagRates:
        years.append(year)
        return _flags()

    monkeypatch.setattr(
        frs_disability, "uk_dwp_disability_category_rates", _category_reader
    )
    monkeypatch.setattr(frs_disability, "uk_dwp_disability_flag_rates", _flag_reader)

    result = frs_disability.UKFRSDisabilityStageTransform(stage=stage)(frame)

    assert years == [2024, 2024]
    assert uk_time_period(result) == "2023"
    assert result.table("person")["aa_category"].tolist() == ["HIGHER"]
