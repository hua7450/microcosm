"""UK source runtime seams for the shared source-stage manifest."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.source_runtime import (
    SourceOperationHandler,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.frame import Frame
from microcosm.frame.rules import materialize_rules_engine_predictors

__all__ = [
    "UK_NONNEGATIVE_OUTPUTS_BY_STAGE",
    "UK_NONNEGATIVE_SOURCE_OUTPUTS",
    "materialize_uk_rules_engine_predictors_from_manifest",
    "uk_source_operation_handlers",
    "uk_stage_implementations",
]


def _uk_nonnegative_outputs_by_stage() -> dict[str, tuple[str, ...]]:
    from microcosm.build.country_spec import load_country_spec

    spec = load_country_spec("uk")
    if spec.sources is None:
        return {}
    return {
        stage.stage: tuple(stage.nonnegative_outputs) for stage in spec.sources.stages
    }


UK_NONNEGATIVE_OUTPUTS_BY_STAGE = _uk_nonnegative_outputs_by_stage()
UK_NONNEGATIVE_SOURCE_OUTPUTS = tuple(
    dict.fromkeys(
        column
        for columns in UK_NONNEGATIVE_OUTPUTS_BY_STAGE.values()
        for column in columns
    )
)


def uk_stage_implementations(
    *,
    retained_leaves_transform: Callable[[Frame], Frame],
    hmrc_income_transform: Callable[[Frame], Frame],
    frs_spine_transform: Callable[[Frame], Frame] | None = None,
    frs_employment_transform: Callable[[Frame], Frame] | None = None,
    frs_council_tax_transform: Callable[[Frame], Frame] | None = None,
    frs_disability_transform: Callable[[Frame], Frame] | None = None,
    frs_education_transform: Callable[[Frame], Frame] | None = None,
    frs_legacy_proxies_transform: Callable[[Frame], Frame] | None = None,
    frs_education_grant_split_transform: Callable[[Frame], Frame] | None = None,
    frs_take_up_transform: Callable[[Frame], Frame] | None = None,
    frs_person_draws_transform: Callable[[Frame], Frame] | None = None,
    frs_household_draws_transform: Callable[[Frame], Frame] | None = None,
    frs_brma_transform: Callable[[Frame], Frame] | None = None,
    was_wealth_transform: Callable[[Frame], Frame] | None = None,
    regional_property_uprating_transform: Callable[[Frame], Frame] | None = None,
    lcfs_consumption_transform: Callable[[Frame], Frame] | None = None,
    etb_vat_transform: Callable[[Frame], Frame] | None = None,
    etb_services_transform: Callable[[Frame], Frame] | None = None,
    frs_hmrc_spine_leaves_transform: Callable[[Frame], Frame] | None = None,
    spi_support_channel_transform: Callable[[Frame], Frame] | None = None,
    hmrc_spi_income_spine_transform: Callable[[Frame], Frame] | None = None,
    uc_deduction_attributes_transform: Callable[[Frame], Frame] | None = None,
    cgt_incidence_clone_transform: Callable[[Frame], Frame] | None = None,
    cgt_band_donors_transform: Callable[[Frame], Frame] | None = None,
    hmrc_cgt_gains_spine_transform: Callable[[Frame], Frame] | None = None,
    salary_sacrifice_transform: Callable[[Frame], Frame] | None = None,
    student_loans_transform: Callable[[Frame], Frame] | None = None,
) -> dict[str, Callable[[Frame], Frame]]:
    """Return the whole-stage implementation map for the UK source plan."""

    implementations = {
        "frs_hmrc_retained_leaves": retained_leaves_transform,
        "hmrc_spi_income": hmrc_income_transform,
    }
    optional = {
        "frs_spine": frs_spine_transform,
        "frs_employment": frs_employment_transform,
        "frs_council_tax": frs_council_tax_transform,
        "frs_disability": frs_disability_transform,
        "frs_education": frs_education_transform,
        "frs_legacy_proxies": frs_legacy_proxies_transform,
        "frs_education_grant_split": frs_education_grant_split_transform,
        "frs_take_up": frs_take_up_transform,
        "frs_person_draws": frs_person_draws_transform,
        "frs_household_draws": frs_household_draws_transform,
        "frs_brma": frs_brma_transform,
        "was_wealth": was_wealth_transform,
        "regional_property_uprating": regional_property_uprating_transform,
        "lcfs_consumption": lcfs_consumption_transform,
        "etb_vat": etb_vat_transform,
        "etb_services": etb_services_transform,
        "frs_hmrc_spine_leaves": frs_hmrc_spine_leaves_transform,
        "spi_support_channel": spi_support_channel_transform,
        "hmrc_spi_income_spine": hmrc_spi_income_spine_transform,
        "uc_deduction_attributes": uc_deduction_attributes_transform,
        "cgt_incidence_clone": cgt_incidence_clone_transform,
        "cgt_band_donors": cgt_band_donors_transform,
        "hmrc_cgt_gains_spine": hmrc_cgt_gains_spine_transform,
        "salary_sacrifice": salary_sacrifice_transform,
        "student_loans": student_loans_transform,
    }
    implementations.update(
        {
            name: transform
            for name, transform in optional.items()
            if transform is not None
        }
    )
    return implementations


def uk_source_operation_handlers() -> Mapping[str, SourceOperationHandler]:
    """Return UK operation handlers available to declarative source runtimes."""

    return {
        "materialize_rules_engine_predictors": (
            materialize_uk_rules_engine_predictors_from_manifest
        )
    }


def materialize_uk_rules_engine_predictors_from_manifest(
    current: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> Frame:
    """Materialize declared rules-engine predictors onto a runtime Frame."""

    if current is not None:
        raise SourceRuntimeError(
            "materialize_rules_engine_predictors operates on the runtime Frame, "
            "not an intermediate source table."
        )
    frame = _extra(context, "frame", Frame)
    engine = _extra(context, "rules_engine", object)
    country = _require_uk_country(context)
    year_rule = operation.parameters.get("year_rule")
    if year_rule is not None:
        # A declared rule names the release year the predictors are evaluated
        # at and wins over the runtime period (#862).
        try:
            period = resolve_uk_year_rule(year_rule)
        except ValueError as error:
            raise SourceRuntimeError(str(error)) from error
    else:
        period = context.config.extra.get("period", context.config.target_year)
    if period is None:
        raise SourceRuntimeError(
            "materialize_rules_engine_predictors requires a period in the "
            "runtime context."
        )
    predictors = operation.parameters.get("predictors")
    if not isinstance(predictors, list) or not all(
        isinstance(name, str) and name for name in predictors
    ):
        raise SourceRuntimeError(
            "materialize_rules_engine_predictors requires a non-empty "
            "'predictors' list."
        )
    try:
        return materialize_rules_engine_predictors(
            frame,
            variables=tuple(predictors),
            period=period,
            engine=engine,
            country=country,
        )
    except ValueError as error:
        raise SourceRuntimeError(str(error)) from error


def _extra(
    context: SourceRuntimeContext,
    key: str,
    expected_type: type,
) -> Any:
    value = context.config.extra.get(key)
    if not isinstance(value, expected_type):
        raise SourceRuntimeError(
            f"materialize_rules_engine_predictors requires context.config.extra"
            f"[{key!r}] to be a {expected_type.__name__}."
        )
    return value


def _require_uk_country(context: SourceRuntimeContext) -> str:
    """Fail closed: this is the UK handler map, so the dataset country is UK.

    An absent ``country`` extra must not skip the engine assertion (the
    shared materializer only asserts when a country is supplied), and a
    context claiming another country must never reach a UK handler.
    """

    country = context.config.extra.get("country", "uk")
    if country != "uk":
        raise SourceRuntimeError(
            "materialize_rules_engine_predictors ran through the UK handler "
            f"map but the runtime context declares country {country!r}; UK "
            "handlers only serve country 'uk'."
        )
    return country
