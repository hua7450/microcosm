"""Batched terminal acceptance gates for UK release candidates.

The national builder evaluates this battery once, after every source stage and
immediately before writing its staging H5.  Every evaluator runs even when an
earlier evaluator fails, so one expensive build produces one complete named
failure report.  Evidence that does not exist yet is omitted: gates whose
evidence or reviewed thresholds are absent (the parity trio, the
weighted-integrity pair, the future delivered-take-up gates) are not
represented by placeholder passes.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import (
    GateResult,
    export_surface_gate,
)
from microcosm.build.gates import (
    target_surface_gate as _target_surface_gate,
)
from microcosm.build.uk_runtime.diagnostics import uk_weight_summary
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
    UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE,
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    UKReviewedExclusion,
    _expired_exclusion_failure,
    _premature_exclusion_failure,
    coerce_reviewed_exclusions,
    exclusion_evaluation_date,
    load_uk_reviewed_exclusion_register,
    uk_input_mass_parity_gate,
    uk_qrf_tail_concentration_gate,
)

__all__ = [
    "UK_ALLOWED_EXTRA_EXPORT_COLUMNS",
    "UK_CANDIDATE_DATASET_NAME",
    "UK_DEFAULT_ZERO_WEIGHT_STRATA",
    "UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS",
    "UK_MAX_TARGET_ABS_RELATIVE_ERROR",
    "UK_REFERENCE_DATASET_NAME",
    "UK_REVIEWED_EXPORT_EXCLUSIONS",
    "UKInputMassParityPolicy",
    "UKInputMassReference",
    "UKQRFTailConcentrationPolicy",
    "UKZeroWeightStratumDeclaration",
    "uk_default_degenerate_reviewed_exclusions",
    "uk_default_target_fit_reviewed_exclusions",
    "uk_degenerate_release_surface_gate",
    "uk_export_surface_gate",
    "uk_input_mass_parity_gate",
    "uk_qrf_tail_concentration_gate",
    "uk_target_fit_gate",
    "uk_target_surface_gate",
    "uk_weight_ess_gate",
    "uk_weight_ratio_gate",
    "uk_zero_weight_strata_gate",
]

UK_CANDIDATE_DATASET_NAME = "microcosm_uk_2024"
# The label names the pinned reference artifact exactly: the 2024-25 line's
# published enhanced_frs_2024_25.h5 (no separate "recalibrated" variant
# exists at this vintage; the June report strings keep their own label).
# After the swap, "reference" becomes the previous certified microcosm line
# once a second certified cut exists; that later increment flips this value.
UK_REFERENCE_DATASET_NAME = "enhanced_frs_2024_25"
UK_MAX_TARGET_ABS_RELATIVE_ERROR = 0.25


_SPI_FLAG = "household_is_spi_synthetic"
_CAPITAL_GAINS_FLAG = "household_is_capital_gains_clone"
_WEIGHT_COLUMN = "household_weight"


@dataclass(frozen=True)
class UKZeroWeightStratumDeclaration:
    """One reviewed zero-weight household stratum and its maximum size."""

    name: str
    selector: Mapping[str, object]
    maximum_zero_weight_rows: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("UK zero-weight stratum name must be non-empty.")
        if not isinstance(self.selector, Mapping) or not self.selector:
            raise ValueError(
                f"UK zero-weight stratum {self.name!r} needs a non-empty selector."
            )
        normalized: dict[str, object] = {}
        for raw_column, value in self.selector.items():
            column = str(raw_column)
            if not column:
                raise ValueError(
                    f"UK zero-weight stratum {self.name!r} has an empty selector "
                    "column."
                )
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"UK zero-weight stratum {self.name!r} selector {column!r} "
                    "must be finite."
                )
            if value is not None and not isinstance(value, str | bool | int | float):
                raise TypeError(
                    f"UK zero-weight stratum {self.name!r} selector {column!r} "
                    f"has unsupported value type {type(value).__name__}."
                )
            normalized[column] = value
        maximum = self.maximum_zero_weight_rows
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError(
                f"UK zero-weight stratum {self.name!r} maximum must be a "
                "non-negative integer."
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                f"UK zero-weight stratum {self.name!r} needs a reviewed reason."
            )
        object.__setattr__(self, "selector", dict(sorted(normalized.items())))


UK_DEFAULT_ZERO_WEIGHT_STRATA: tuple[UKZeroWeightStratumDeclaration, ...] = (
    UKZeroWeightStratumDeclaration(
        name="june_spi_synthetic_base",
        selector={_SPI_FLAG: True, _CAPITAL_GAINS_FLAG: False},
        maximum_zero_weight_rows=100_000,
        reason=(
            "The certified June FRS-derived artifact ships 100,000 zero-weight "
            "SPI-synthetic non-capital-gains rows."
        ),
    ),
    UKZeroWeightStratumDeclaration(
        name="june_spi_synthetic_capital_gains",
        selector={_SPI_FLAG: True, _CAPITAL_GAINS_FLAG: True},
        maximum_zero_weight_rows=100_000,
        reason=(
            "The certified June FRS-derived artifact ships 100,000 zero-weight "
            "SPI-synthetic capital-gains-clone rows."
        ),
    ),
)


# Reviewed candidate-only fields from the June UK prototype.  They are source
# provenance or genuine additional model inputs, not incumbent-surface losses.
UK_ALLOWED_EXTRA_EXPORT_COLUMNS: tuple[str, ...] = (
    "benunit.child_benefit_opts_out",
    "benunit.frs_benunit_capital",
    "benunit.uc_deduction_combination",
    "benunit.uc_deduction_random_draw",
    "benunit.uc_deduction_type_random_draw",
    "benunit.uc_latent_deduction_rate",
    "benunit.uc_reported_capital",
    "household.bus_fare_spending",
    "household.bus_subsidy_spending",
    "household.cash_isa",
    "household.clone_index",
    "household.constituency_code_oa",
    "household.consumer_debt",
    "household.electricity_consumption",
    "household.gas_consumption",
    "household.has_fuel_consumption",
    "household.household_is_capital_gains_clone",
    "household.household_is_cgt_band_donor",
    "household.household_is_spi_synthetic",
    "household.la_code_oa",
    "household.lsoa_code",
    "household.mortgage_debt",
    "household.msoa_code",
    "household.num_vehicles",
    "household.oa_code",
    "household.private_pension_wealth",
    "household.property_purchased",
    "household.rail_usage",
    "household.region_code_oa",
    "household.stocks_and_shares_isa",
    "person.aa_category",
    "person.a_and_e_visits",
    "person.admitted_patient_visits",
    "person.age_started_or_accepted_current_education_or_training",
    "person.attends_private_school_random_draw",
    "person.charitable_investment_gifts",
    "person.dla_m_category",
    "person.dla_sc_category",
    "person.esa_health_condition_proxy",
    "person.esa_support_group_proxy",
    "person.employment_sector",
    "person.gift_aid",
    "person.highest_education",
    "person.is_before_universal_credit_qualifying_young_person_terminal_date",
    "person.is_in_non_advanced_education",
    "person.is_parent",
    "person.is_uc_claimant",
    "person.legacy_jobseeker_proxy",
    "person.outpatient_visits",
    "person.pension_contributions_via_salary_sacrifice",
    "person.pip_dl_category",
    "person.pip_m_category",
    "person.receives_benefits_in_own_right",
    "person.salary_sacrifice_asked",
    "person.salary_sacrifice_reported",
    "person.sic_industry_division",
    "person.student_loan_balance",
    "person.student_loan_plan",
    "person.tax_free_childcare_spend_routed_share",
    "person.would_claim_marriage_allowance",
    "person.would_claim_scp",
)

UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS: tuple[str, ...] = (
    "person.attends_private_school",
    "person.is_higher_earner",
)

UK_REVIEWED_EXPORT_EXCLUSIONS: Mapping[str, str] = {
    "person.incapacity_benefit_reported": (
        "The enhanced FRS stores this legacy reported-benefit input as an "
        "all-zero layer; the candidate must drop dead zero layers."
    ),
}

_STRUCTURAL_COLUMNS: Mapping[str, frozenset[str]] = {
    "person": frozenset({"person_id", "person_household_id", "person_benunit_id"}),
    "benunit": frozenset({"benunit_id"}),
    "household": frozenset({"household_id", _WEIGHT_COLUMN}),
}


@functools.cache
def uk_default_degenerate_reviewed_exclusions() -> Mapping[str, UKReviewedExclusion]:
    """The committed degenerate-surface register (#630), loaded lazily."""

    return MappingProxyType(
        load_uk_reviewed_exclusion_register(
            None, resource=UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE
        )
    )


@functools.cache
def uk_default_target_fit_reviewed_exclusions() -> Mapping[str, UKReviewedExclusion]:
    """The committed target-fit deferral register (#796), loaded lazily."""

    return MappingProxyType(
        load_uk_reviewed_exclusion_register(
            None, resource=UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE
        )
    )


def _entity_tables(dataset: Any) -> tuple[tuple[str, pd.DataFrame], ...]:
    if isinstance(dataset, Mapping):
        raw = tuple((entity, dataset.get(entity)) for entity in _STRUCTURAL_COLUMNS)
    else:
        raw = tuple(
            (entity, getattr(dataset, entity, None)) for entity in _STRUCTURAL_COLUMNS
        )
    if any(not isinstance(table, pd.DataFrame) for _entity, table in raw):
        raise TypeError(
            "UK terminal gates require person, benunit, and household DataFrames."
        )
    return tuple(
        (entity, table) for entity, table in raw if isinstance(table, pd.DataFrame)
    )


def _reviewed_reasons(values: Mapping[str, str] | None) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("UK reviewed exclusions must be a mapping.")
    normalized = {str(name): str(reason) for name, reason in values.items()}
    missing = sorted(name for name, reason in normalized.items() if not reason.strip())
    if missing:
        raise ValueError(f"UK reviewed exclusions need reasons: {missing}.")
    return normalized


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return repr(value)


def _degenerate_kind(series: pd.Series) -> tuple[str, object | None] | None:
    missing = series.isna()
    if bool(missing.all()):
        return ("all_null", None)
    observed = series.loc[~missing]
    if (
        pd.api.types.is_numeric_dtype(observed.dtype)
        or pd.api.types.is_bool_dtype(observed.dtype)
    ) and bool((observed == 0).all()):
        return ("all_zero", 0)
    try:
        unique = pd.unique(observed)
    except (TypeError, ValueError):
        unique = np.asarray(list(dict.fromkeys(map(repr, observed))), dtype=object)
    if len(unique) == 1:
        return ("constant", _json_scalar(unique[0]))
    return None


def uk_degenerate_release_surface_gate(
    dataset: Any,
    *,
    reviewed_exclusions: Mapping[str, UKReviewedExclusion] | None = None,
    now: date | None = None,
) -> GateResult:
    """Reject every all-null, all-zero, or constant nonstructural column.

    ``reviewed_exclusions`` are schema-2 approval records (#610); an entry
    suppresses from its ``approved_on`` through its ``expires_on``, and any
    out-of-force entry fails the gate with correct-or-renew context — even
    when its column is absent or carries signal, so the register cannot rot
    silently at any column state (matching the input-mass and QRF wrappers).
    """

    evaluated_on = exclusion_evaluation_date(now)
    exclusions = coerce_reviewed_exclusions(
        reviewed_exclusions, label="UK degenerate-surface"
    )
    present: set[str] = set()
    live: dict[str, dict[str, object]] = {}
    excluded: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    checked = 0
    for entity, table in _entity_tables(dataset):
        structural = _STRUCTURAL_COLUMNS[entity]
        for column in table.columns:
            if column in structural:
                continue
            checked += 1
            name = f"{entity}.{column}"
            present.add(name)
            finding = _degenerate_kind(table[column])
            if finding is None:
                continue
            kind, value = finding
            detail = {"kind": kind, "value": value}
            record = exclusions.get(name)
            if (
                record is not None
                and not record.expired(evaluated_on)
                and not record.premature(evaluated_on)
            ):
                excluded[name] = {
                    **detail,
                    "reason": record.reason,
                    "approved_by": record.approved_by,
                    "adjudication": record.adjudication,
                    "expires_on": record.expires_on,
                }
                continue
            live[name] = detail
            degenerate_message = (
                f"{name}: persisted release column is {kind.replace('_', '-')}"
                + (f" at {value!r}" if kind == "constant" else "")
            )
            if record is not None and record.expired(evaluated_on):
                failures.append(
                    f"{degenerate_message}; its reviewed exclusion expired "
                    f"{record.expires_on} (approved_by {record.approved_by}, "
                    f"{record.adjudication}) — renew the adjudication or "
                    "remove the entry."
                )
            elif record is not None:
                failures.append(
                    f"{degenerate_message}; its reviewed exclusion takes force "
                    f"{record.approved_on} (approved_by {record.approved_by}, "
                    f"{record.adjudication}) — correct the receipt's "
                    "approved_on or wait for it."
                )
            else:
                failures.append(
                    f"{degenerate_message}; populate it with signal, drop it, "
                    "or record a reviewed exclusion."
                )

    expired = sorted(
        name for name, record in exclusions.items() if record.expired(evaluated_on)
    )
    premature = sorted(
        name for name, record in exclusions.items() if record.premature(evaluated_on)
    )
    # Stale probing covers in-force entries only: an out-of-force entry gets
    # receipt-context failures below, never "now carry signal; remove them."
    stale = sorted(
        name
        for name in exclusions
        if name in present
        and name not in excluded
        and name not in live
        and name not in expired
        and name not in premature
    )
    dormant = sorted(set(exclusions) - present)
    if stale:
        failures.append(
            "Stale reviewed degenerate-column exclusions now carry signal; remove "
            f"them: {stale}."
        )
    # Out-of-force entries whose column did not fail above (absent, or
    # present without a degenerate finding) must still fail the gate: the
    # register cannot rot just because its column moved. Live columns
    # already carry per-column receipt context, so only the remainder gets
    # the combined message (one failure per condition, never two per entry).
    unreported_expired = [name for name in expired if name not in live]
    if unreported_expired:
        failures.append(
            _expired_exclusion_failure(
                exclusions, unreported_expired, family="degenerate-column"
            )
        )
    unreported_premature = [name for name in premature if name not in live]
    if unreported_premature:
        failures.append(
            _premature_exclusion_failure(
                exclusions, unreported_premature, family="degenerate-column"
            )
        )
    by_kind = {
        kind: sorted(name for name, detail in live.items() if detail["kind"] == kind)
        for kind in ("all_null", "all_zero", "constant")
    }
    return GateResult(
        name="degenerate_release_surface",
        passed=not failures,
        failures=tuple(failures),
        details={
            "columns_checked": checked,
            "findings": dict(sorted(live.items())),
            "all_null_columns": by_kind["all_null"],
            "all_zero_columns": by_kind["all_zero"],
            "constant_columns": by_kind["constant"],
            "reviewed_exclusions": dict(sorted(excluded.items())),
            "stale_exclusions": stale,
            "dormant_exclusions": dormant,
            "expired_exclusions": expired,
            "premature_exclusions": premature,
            "exclusions_evaluated_on": evaluated_on.isoformat(),
        },
    )


def _household_weights(household: pd.DataFrame) -> np.ndarray:
    if _WEIGHT_COLUMN not in household:
        raise ValueError(f"UK household table is missing {_WEIGHT_COLUMN!r}.")
    weights = pd.to_numeric(household[_WEIGHT_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("UK household weights must be a non-empty vector.")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("UK household weights must be finite and non-negative.")
    return weights


def _selector_mask(
    household: pd.DataFrame,
    selector: Mapping[str, object],
) -> tuple[np.ndarray, list[str]]:
    missing = sorted(set(selector) - set(household.columns))
    if missing:
        return np.zeros(len(household), dtype=bool), missing
    mask = np.ones(len(household), dtype=bool)
    for column, expected in selector.items():
        values = household[column]
        matched = (
            values.isna() if expected is None else values.eq(expected).fillna(False)
        )
        mask &= matched.to_numpy(dtype=bool)
    return mask, []


def uk_zero_weight_strata_gate(
    household: pd.DataFrame,
    *,
    declarations: Sequence[UKZeroWeightStratumDeclaration] = (
        UK_DEFAULT_ZERO_WEIGHT_STRATA
    ),
) -> GateResult:
    """Reject zero-weight rows outside or beyond reviewed declarations."""

    if not isinstance(household, pd.DataFrame):
        raise TypeError("UK zero-weight strata gate requires a household DataFrame.")
    weights = _household_weights(household)
    materialized = tuple(declarations)
    if any(
        not isinstance(item, UKZeroWeightStratumDeclaration) for item in materialized
    ):
        raise TypeError(
            "UK zero-weight declarations must be UKZeroWeightStratumDeclaration "
            "instances."
        )
    names = [item.name for item in materialized]
    if len(names) != len(set(names)):
        raise ValueError("UK zero-weight stratum declaration names must be unique.")

    zero = weights == 0.0
    matches = np.zeros(len(household), dtype=np.int64)
    details: list[dict[str, object]] = []
    failures: list[str] = []
    for declaration in materialized:
        mask, missing = _selector_mask(household, declaration.selector)
        selected = zero & mask
        matches += selected.astype(np.int64)
        count = int(selected.sum())
        details.append(
            {
                "name": declaration.name,
                "selector": dict(declaration.selector),
                "maximum_zero_weight_rows": declaration.maximum_zero_weight_rows,
                "zero_weight_rows": count,
                "missing_selector_columns": missing,
                "reason": declaration.reason,
            }
        )
        if missing:
            failures.append(
                f"{declaration.name}: selector column(s) are missing from the "
                f"household release surface: {missing}."
            )
        if count > declaration.maximum_zero_weight_rows:
            failures.append(
                f"{declaration.name}: {count} zero-weight rows exceed the declared "
                f"maximum {declaration.maximum_zero_weight_rows}."
            )

    unmatched_positions = np.flatnonzero(zero & (matches == 0))
    ambiguous_positions = np.flatnonzero(zero & (matches > 1))
    if unmatched_positions.size:
        failures.append(
            f"{unmatched_positions.size} zero-weight household row(s) match no "
            "declared stratum."
        )
    if ambiguous_positions.size:
        failures.append(
            f"{ambiguous_positions.size} zero-weight household row(s) match more "
            "than one declared stratum."
        )
    id_values = (
        household["household_id"].tolist()
        if "household_id" in household
        else list(household.index)
    )
    return GateResult(
        name="zero_weight_strata",
        passed=not failures,
        failures=tuple(failures),
        details={
            "household_rows": len(household),
            "zero_weight_rows": int(zero.sum()),
            "declared_strata": details,
            "unmatched_zero_weight_rows": int(unmatched_positions.size),
            "unmatched_household_examples": [
                _json_scalar(id_values[index]) for index in unmatched_positions[:20]
            ],
            "ambiguous_zero_weight_rows": int(ambiguous_positions.size),
            "ambiguous_household_examples": [
                _json_scalar(id_values[index]) for index in ambiguous_positions[:20]
            ],
        },
    )


def uk_weight_ess_gate(
    weights: Sequence[float] | np.ndarray,
    *,
    minimum_ess_fraction: float,
) -> GateResult:
    """Require the shipped household weights to retain effective support."""

    minimum = float(minimum_ess_fraction)
    if not math.isfinite(minimum) or not 0.0 < minimum <= 1.0:
        raise ValueError("minimum_ess_fraction must be finite and in (0, 1].")
    summary = uk_weight_summary(weights)
    fraction = float(summary["ess_fraction"])
    if fraction < minimum:
        failures = (
            f"ESS fraction {fraction:.6g} is below the reviewed minimum {minimum:.6g}.",
        )
    else:
        failures = ()
    return GateResult(
        name="weight_ess",
        passed=not failures,
        failures=failures,
        details={**summary, "minimum_ess_fraction": minimum},
    )


def uk_weight_ratio_gate(
    weights: Sequence[float] | np.ndarray,
    *,
    maximum_max_to_median_ratio: float,
) -> GateResult:
    """Backstop a shipped-weight max/positive-median concentration blowout."""

    maximum = float(maximum_max_to_median_ratio)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "maximum_max_to_median_ratio must be finite and strictly positive."
        )
    summary = uk_weight_summary(weights)
    raw_ratio = summary["max_to_median_positive_weight"]
    failures: tuple[str, ...]
    if raw_ratio is None:
        failures = (
            "Max/positive-median weight ratio is undefined because the release "
            "has no positive median weight.",
        )
    else:
        ratio = float(raw_ratio)
        failures = (
            (
                f"Max/positive-median weight ratio {ratio!r} exceeds the "
                f"reviewed maximum {maximum!r}.",
            )
            if ratio > maximum
            else ()
        )
    return GateResult(
        name="weight_ratio",
        passed=not failures,
        failures=failures,
        details={**summary, "maximum_max_to_median_ratio": maximum},
    )


def _reviewed_export_exclusions(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    exclusions = dict(UK_REVIEWED_EXPORT_EXCLUSIONS)
    if overrides:
        exclusions.update(_reviewed_reasons(overrides))
    hard = sorted(set(exclusions) & set(UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS))
    if hard:
        raise ValueError(
            "UK export-surface reviewed exclusions cannot waive hard-required "
            f"reference columns: {hard}."
        )
    return exclusions


def uk_export_surface_gate(
    candidate_columns: Iterable[str],
    reference_columns: Iterable[str],
    *,
    allowed_extra_columns: Iterable[str] = UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Run the incumbent-compatible UK export-surface gate."""

    candidate = {str(name) for name in candidate_columns}
    reference = {str(name) for name in reference_columns}
    exclusions = _reviewed_export_exclusions(reviewed_exclusions)
    result = export_surface_gate(
        candidate,
        reference,
        candidate_name=UK_CANDIDATE_DATASET_NAME,
        reference_name=UK_REFERENCE_DATASET_NAME,
        allowed_extra_columns=allowed_extra_columns,
        reviewed_exclusions=exclusions,
    )
    forbidden = sorted(candidate & set(exclusions))
    failures = [*result.failures]
    if not candidate:
        failures.append("UK candidate export-surface evidence is empty.")
    if not reference:
        failures.append("Enhanced-FRS reference export-surface evidence is empty.")
    if forbidden:
        failures.append(
            f"{UK_CANDIDATE_DATASET_NAME}: exports {len(forbidden)} reviewed "
            f"reference-only column(s) that must be dropped: {forbidden[:20]}."
        )
    return GateResult(
        name=result.name,
        passed=not failures,
        failures=tuple(failures),
        details={**dict(result.details), "forbidden_candidate_columns": forbidden},
    )


def uk_target_surface_gate(
    candidate_targets: Iterable[str],
    reference_targets: Iterable[str],
) -> GateResult:
    """Require the UK candidate target surface to cover enhanced FRS."""

    candidate = {str(name) for name in candidate_targets}
    reference = {str(name) for name in reference_targets}
    result = _target_surface_gate(
        candidate,
        reference,
        candidate_name=UK_CANDIDATE_DATASET_NAME,
        reference_name=UK_REFERENCE_DATASET_NAME,
    )
    failures = [*result.failures]
    if not candidate:
        failures.append("UK candidate target-surface evidence is empty.")
    if not reference:
        failures.append("Enhanced-FRS reference target-surface evidence is empty.")
    return GateResult(
        name=result.name,
        passed=not failures,
        failures=tuple(failures),
        details=dict(result.details),
    )


def uk_target_fit_gate(
    target_relative_errors: Mapping[str, float],
    *,
    max_abs_relative_error: float = UK_MAX_TARGET_ABS_RELATIVE_ERROR,
    reviewed_exclusions: Mapping[str, UKReviewedExclusion] | None = None,
    now: date | None = None,
) -> GateResult:
    """Fail a UK artifact with severe shipped-weight target errors.

    ``reviewed_exclusions`` are schema-2 approval records (#610): each defers
    one named target's breach of the release bound through its approval
    window. The exclusion suppresses the release fence only — the target
    stays bound in calibration, so the solver keeps pulling toward it. An
    out-of-force entry fails the gate with correct-or-renew context, and an
    in-force entry whose target is back inside the bound is stale and fails
    (matching :func:`microcosm.build.gates.target_fit_gate`'s rot rule), so
    the register cannot outlive the defects it defers.
    """

    maximum = float(max_abs_relative_error)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("max_abs_relative_error must be finite and non-negative.")
    evaluated_on = exclusion_evaluation_date(now)
    exclusions = coerce_reviewed_exclusions(reviewed_exclusions, label="UK target-fit")
    errors = {str(name): float(error) for name, error in target_relative_errors.items()}
    nonfinite = sorted(
        name for name, error in errors.items() if not math.isfinite(error)
    )
    if nonfinite:
        raise ValueError(f"UK target relative errors must be finite: {nonfinite}.")
    breaching = {name: error for name, error in errors.items() if abs(error) > maximum}
    failing: dict[str, float] = {}
    excluded: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for name in sorted(breaching, key=lambda name: abs(breaching[name]), reverse=True):
        error = breaching[name]
        record = exclusions.get(name)
        breach_message = (
            f"{UK_CANDIDATE_DATASET_NAME}: {name} relative error "
            f"{error:+.1%} exceeds {maximum:.0%}."
        )
        if (
            record is not None
            and not record.expired(evaluated_on)
            and not record.premature(evaluated_on)
        ):
            excluded[name] = {
                "relative_error": error,
                "reason": record.reason,
                "approved_by": record.approved_by,
                "adjudication": record.adjudication,
                "expires_on": record.expires_on,
            }
            continue
        failing[name] = error
        if record is not None and record.expired(evaluated_on):
            failures.append(
                f"{breach_message[:-1]}; its reviewed exclusion expired "
                f"{record.expires_on} (approved_by {record.approved_by}, "
                f"{record.adjudication}) — renew the adjudication or remove "
                "the entry."
            )
        elif record is not None:
            failures.append(
                f"{breach_message[:-1]}; its reviewed exclusion takes force "
                f"{record.approved_on} (approved_by {record.approved_by}, "
                f"{record.adjudication}) — correct the receipt's approved_on "
                "or wait for it."
            )
        elif len(failing) <= 20:
            failures.append(breach_message)

    expired = sorted(
        name for name, record in exclusions.items() if record.expired(evaluated_on)
    )
    premature = sorted(
        name for name, record in exclusions.items() if record.premature(evaluated_on)
    )
    # Stale probing covers in-force entries only: an out-of-force entry gets
    # receipt-context failures instead, never "back inside the bound".
    stale = sorted(
        name
        for name in exclusions
        if name in errors
        and name not in breaching
        and name not in expired
        and name not in premature
    )
    dormant = sorted(name for name in exclusions if name not in errors)
    if stale:
        failures.append(
            "Stale reviewed target-fit exclusions are back inside the bound; "
            f"remove them: {stale}."
        )
    unreported_expired = [name for name in expired if name not in breaching]
    if unreported_expired:
        failures.append(
            _expired_exclusion_failure(
                exclusions, unreported_expired, family="target-fit"
            )
        )
    unreported_premature = [name for name in premature if name not in breaching]
    if unreported_premature:
        failures.append(
            _premature_exclusion_failure(
                exclusions, unreported_premature, family="target-fit"
            )
        )
    if not errors:
        failures.append("UK target-fit evidence is empty.")
    return GateResult(
        name="target_fit",
        passed=not failures,
        failures=tuple(failures),
        details={
            "candidate_name": UK_CANDIDATE_DATASET_NAME,
            "targets_checked": len(errors),
            "max_abs_relative_error": maximum,
            "reviewed_exclusions": dict(sorted(excluded.items())),
            "failing_targets": failing,
            "stale_exclusions": stale,
            "dormant_exclusions": dormant,
            "expired_exclusions": expired,
            "premature_exclusions": premature,
            "exclusions_evaluated_on": evaluated_on.isoformat(),
        },
    )


def _missing_fit_weight_evidence_gate() -> GateResult:
    return GateResult(
        name="weights_audit",
        passed=False,
        failures=(
            "A production fit stage ran but emitted no FitWeightRecord evidence; "
            "an absent audit is not a passing audit.",
        ),
        details={"fits_checked": 0, "evidence_missing": True},
    )
