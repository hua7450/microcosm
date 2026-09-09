"""FRS disability benefit category and flag derivations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.cgt_structure import (
    _assert_closed_world_operations,
)
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame

UK_INTERNAL_DISABILITY_REPORTED_COLUMNS = (
    "attendance_allowance_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "pip_m_reported",
    "pip_dl_reported",
)
UK_DISABILITY_FLAG_REPORTED_COLUMNS = (
    *UK_INTERNAL_DISABILITY_REPORTED_COLUMNS,
    "sda_reported",
    "incapacity_benefit_reported",
    "iidb_reported",
    "afcs_reported",
    "esa_contrib_reported",
    "esa_income_reported",
)
FRS_DISABILITY_OUTPUT_COLUMNS = (
    "aa_category",
    "dla_sc_category",
    "dla_m_category",
    "pip_m_category",
    "pip_dl_category",
    "is_disabled_for_benefits",
    "is_enhanced_disabled_for_benefits",
    "is_severely_disabled_for_benefits",
)
YEAR_RULE = "survey_year"


@dataclass(frozen=True)
class UKDWPDisabilityCategoryRates:
    """Category thresholds from the fiscal-converted ``gov.dwp`` tree."""

    aa_lower: float
    aa_higher: float
    dla_sc_lower: float
    dla_sc_middle: float
    dla_sc_higher: float
    dla_m_lower: float
    dla_m_higher: float
    pip_m_standard: float
    pip_m_enhanced: float
    pip_dl_standard: float
    pip_dl_enhanced: float
    instant: str
    source: str


@dataclass(frozen=True)
class UKDWPDisabilityFlagRates:
    """Flag thresholds from the fiscal-converted ``gov.dwp`` tree."""

    aa_higher: float
    dla_sc_higher: float
    pip_dl_enhanced: float
    instant: str
    source: str


class UKFRSDisabilityStageTransform:
    """Whole-stage callable for FRS disability derivations."""

    def __init__(
        self,
        *,
        stage: SourceStageSpec,
        category_rates: UKDWPDisabilityCategoryRates | None = None,
        flag_rates: UKDWPDisabilityFlagRates | None = None,
    ) -> None:
        self.stage = stage
        self.category_rates = category_rates
        self.flag_rates = flag_rates

    def __call__(self, frame: Frame) -> Frame:
        _assert_frs_disability_stage_parameters(self.stage)
        year = resolve_uk_year_rule(YEAR_RULE)
        return add_frs_disability(
            frame,
            category_rates=self.category_rates
            or uk_dwp_disability_category_rates(year),
            flag_rates=self.flag_rates or uk_dwp_disability_flag_rates(year),
        )

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_DISABILITY_OUTPUT_COLUMNS


def uk_dwp_disability_category_rates(
    year: int,
) -> UKDWPDisabilityCategoryRates:
    """Read category rates from fiscal-converted ``gov.dwp`` at ``year``.

    The parameter instant is the fiscal-year label instant: policyengine-uk
    writes the April rates over the whole labelled year. This corrects the
    tree mismatch identified by uk-data#475 and uses the shared 365.25/7 week
    conversion identified by uk-data#476.
    """

    dwp = _uk_dwp_tree(year)
    import policyengine_uk

    instant = f"{year}-01-01"
    return UKDWPDisabilityCategoryRates(
        aa_lower=_parameter_value(dwp.attendance_allowance.lower, instant),
        aa_higher=_parameter_value(dwp.attendance_allowance.higher, instant),
        dla_sc_lower=_parameter_value(dwp.dla.self_care.lower, instant),
        dla_sc_middle=_parameter_value(dwp.dla.self_care.middle, instant),
        dla_sc_higher=_parameter_value(dwp.dla.self_care.higher, instant),
        dla_m_lower=_parameter_value(dwp.dla.mobility.lower, instant),
        dla_m_higher=_parameter_value(dwp.dla.mobility.higher, instant),
        pip_m_standard=_parameter_value(dwp.pip.mobility.standard, instant),
        pip_m_enhanced=_parameter_value(dwp.pip.mobility.enhanced, instant),
        pip_dl_standard=_parameter_value(dwp.pip.daily_living.standard, instant),
        pip_dl_enhanced=_parameter_value(dwp.pip.daily_living.enhanced, instant),
        instant=instant,
        source="policyengine-uk parameters (fiscal-converted gov.dwp tree) "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


def uk_dwp_disability_flag_rates(year: int) -> UKDWPDisabilityFlagRates:
    """Read flag rates from fiscal-converted ``gov.dwp`` at ``year``.

    The parameter instant is the fiscal-year label instant: policyengine-uk
    writes the April rates over the whole labelled year. This keeps the tree
    correction from uk-data#475 and the shared 365.25/7 week conversion from
    uk-data#476 aligned with the category derivation.
    """

    dwp = _uk_dwp_tree(year)
    import policyengine_uk

    instant = f"{year}-01-01"
    return UKDWPDisabilityFlagRates(
        aa_higher=_parameter_value(dwp.attendance_allowance.higher, instant),
        dla_sc_higher=_parameter_value(dwp.dla.self_care.higher, instant),
        pip_dl_enhanced=_parameter_value(dwp.pip.daily_living.enhanced, instant),
        instant=instant,
        source="policyengine-uk parameters (fiscal-converted gov.dwp tree) "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


@lru_cache
def _uk_dwp_tree(year: int):
    """Return the fiscal-converted DWP tree from one engine per year."""

    try:
        import policyengine_uk
    except ImportError as exc:
        raise ImportError(
            "UK DWP disability parameters require `uv sync --all-packages --extra uk`."
        ) from exc

    return policyengine_uk.CountryTaxBenefitSystem().parameters(year).gov.dwp


def _assert_frs_disability_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind both disability derivations to the reviewed policy-year contract."""

    _assert_closed_world_operations(
        stage,
        (
            (
                "derive",
                {
                    "parameters": "disability category thresholds from the fiscal-converted gov.dwp tree at the survey year",
                    "year_rule": YEAR_RULE,
                },
            ),
            (
                "derive",
                {
                    "parameters": "disability flags from the fiscal-converted gov.dwp tree at the survey year",
                    "year_rule": YEAR_RULE,
                },
            ),
        ),
    )


def add_frs_disability(
    frame: Frame,
    *,
    category_rates: UKDWPDisabilityCategoryRates,
    flag_rates: UKDWPDisabilityFlagRates,
) -> Frame:
    person = frame.table("person").copy()
    derived = derive_frs_disability(
        person, category_rates=category_rates, flag_rates=flag_rates
    )
    for column in FRS_DISABILITY_OUTPUT_COLUMNS:
        person[column] = derived[column].to_numpy()
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


def derive_frs_disability(
    person: pd.DataFrame,
    *,
    category_rates: UKDWPDisabilityCategoryRates,
    flag_rates: UKDWPDisabilityFlagRates,
) -> pd.DataFrame:
    values = pd.DataFrame(index=person.index)
    values["aa_category"] = _category(
        _amount(person, "attendance_allowance_reported"),
        (("LOWER", category_rates.aa_lower), ("HIGHER", category_rates.aa_higher)),
    )
    values["dla_sc_category"] = _category(
        _amount(person, "dla_sc_reported"),
        (
            ("LOWER", category_rates.dla_sc_lower),
            ("MIDDLE", category_rates.dla_sc_middle),
            ("HIGHER", category_rates.dla_sc_higher),
        ),
    )
    values["dla_m_category"] = _category(
        _amount(person, "dla_m_reported"),
        (("LOWER", category_rates.dla_m_lower), ("HIGHER", category_rates.dla_m_higher)),
    )
    values["pip_m_category"] = _category(
        _amount(person, "pip_m_reported"),
        (
            ("STANDARD", category_rates.pip_m_standard),
            ("ENHANCED", category_rates.pip_m_enhanced),
        ),
    )
    values["pip_dl_category"] = _category(
        _amount(person, "pip_dl_reported"),
        (
            ("STANDARD", category_rates.pip_dl_standard),
            ("ENHANCED", category_rates.pip_dl_enhanced),
        ),
    )
    total = sum(_amount(person, column) for column in UK_DISABILITY_FLAG_REPORTED_COLUMNS)
    dla_sc = _amount(person, "dla_sc_reported")
    aa = _amount(person, "attendance_allowance_reported")
    pip_dl = _amount(person, "pip_dl_reported")
    afcs = _amount(person, "afcs_reported")
    gap = WEEKS_IN_YEAR
    aa_higher = flag_rates.aa_higher * WEEKS_IN_YEAR - gap
    dla_sc_higher = flag_rates.dla_sc_higher * WEEKS_IN_YEAR - gap
    pip_dl_enhanced = flag_rates.pip_dl_enhanced * WEEKS_IN_YEAR - gap
    values["is_disabled_for_benefits"] = total > 0
    values["is_enhanced_disabled_for_benefits"] = (
        (aa >= aa_higher) | (dla_sc > dla_sc_higher) | (pip_dl >= pip_dl_enhanced)
    )
    values["is_severely_disabled_for_benefits"] = (
        (aa > 0) | (dla_sc >= dla_sc_higher) | (pip_dl >= pip_dl_enhanced) | (afcs > 0)
    )
    return values


def _amount(person: pd.DataFrame, column: str) -> pd.Series:
    if column not in person:
        return pd.Series(0.0, index=person.index)
    return pd.to_numeric(person[column], errors="coerce").fillna(0.0)


def _category(
    reported_amount: pd.Series, thresholds: tuple[tuple[str, float], ...]
) -> np.ndarray:
    weekly = reported_amount.to_numpy(dtype=float) / WEEKS_IN_YEAR
    category = np.full(len(weekly), "NONE", dtype=object)
    for name, weekly_rate in thresholds:
        category[weekly >= max(0.0, float(weekly_rate) - 1.0)] = name
    return category


def _parameter_value(value, instant: str) -> float:
    if callable(value):
        value = value(instant)
    return float(value)
