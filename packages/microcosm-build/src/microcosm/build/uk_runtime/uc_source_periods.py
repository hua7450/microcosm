"""Declared observation-month coverage for UK UC calibration references.

The shared ``calendar_year_average`` operation averages available months. UK UC
contracts declare the intended source months separately so that a changed feed
cannot silently change that denominator. Explicit monthly-window operations can
cross calendar years and sum declared cells within each month. Their coverage
receipt describes source observations, not the model's annual observation date.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from microcosm.build.ledger_targets import (
    MONTHLY_WINDOW_OPERATIONS,
    LedgerTargetReference,
)
from microcosm.calibrate import TargetRegistry

EXPECTED_SOURCE_MONTHS = "uk_uc_expected_source_months"


def uc_source_month_metadata(
    months: object, *, value_operation: str = "calendar_year_average"
) -> dict[str, str]:
    """Validate and encode a contract's explicit, ordered monthly coverage."""

    valid = (
        isinstance(months, list)
        and bool(months)
        and all(
            isinstance(month, str)
            and re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", month)
            for month in months
        )
    )
    if not valid:
        raise ValueError("UK UC source_months must be a non-empty YYYY-MM list.")
    if months != sorted(set(months)) or (
        value_operation not in MONTHLY_WINDOW_OPERATIONS
        and len({month[:4] for month in months}) != 1
    ):
        raise ValueError(
            "UK UC source_months must be unique, ordered months in one calendar year."
        )
    return {EXPECTED_SOURCE_MONTHS: json.dumps(months, separators=(",", ":"))}


def validate_uc_source_month_coverage(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    facts: Iterable[Mapping[str, Any]],
) -> TargetRegistry:
    """Refuse a different resolved month set; record coverage without revaluing."""

    encoded = reference.metadata.get(EXPECTED_SOURCE_MONTHS)
    if encoded is None:
        return registry
    if reference.family != "dwp_universal_credit" or reference.value_operation not in {
        "calendar_year_average",
        *MONTHLY_WINDOW_OPERATIONS,
    }:
        raise ValueError(
            "UK UC source-month coverage requires a UC calendar_year_average or explicit monthly window reference."
        )
    try:
        expected = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "UK UC source-month coverage metadata is invalid JSON."
        ) from error
    uc_source_month_metadata(expected, value_operation=reference.value_operation)
    is_window = reference.value_operation in MONTHLY_WINDOW_OPERATIONS
    if is_window and expected != reference.ledger_selector.get("period_value"):
        raise ValueError(
            "UK UC source-month coverage metadata must equal the declared source window."
        )
    cells_per_month = (
        len(reference.value_operands)
        if reference.value_operation == "monthly_window_sum_average"
        else 1
    )

    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for fact in facts:
        lineage = fact.get("lineage")
        key = (
            fact.get("aggregate_fact_key")
            or fact.get("fact_key")
            or fact.get("legacy_fact_key")
            or fact.get("source_record_id")
            or (
                lineage.get("source_record_id")
                if isinstance(lineage, Mapping)
                else None
            )
        )
        if key:
            by_key.setdefault(str(key), []).append(fact)

    validated = []
    for spec in registry.specs:
        member_keys = json.loads(
            spec.metadata.get("ledger_member_fact_keys", "null")
        ) or [
            spec.metadata.get("ledger_fact_key")
            or spec.metadata.get("ledger_source_record_id")
        ]
        resolved = []
        for key in member_keys:
            matches = by_key.get(key, ())
            if len(matches) != 1:
                raise ValueError(
                    f"UK UC source-month coverage for {reference.name!r}: "
                    f"member {key!r} must identify exactly one fact."
                )
            period = matches[0].get("period", {})
            if period.get("type") != "month":
                raise ValueError(
                    f"UK UC source-month coverage for {reference.name!r}: "
                    "resolved member is not a monthly observation."
                )
            resolved.append(str(period.get("value", "")))
        resolved.sort()
        if resolved != sorted(expected * cells_per_month):
            raise ValueError(
                f"UK UC source-month coverage for {reference.name!r}: "
                f"expected exactly one fact per cell ({cells_per_month}) for each of {expected!r}; "
                f"resolved {resolved!r}."
            )
        validated.append(
            replace(
                spec,
                metadata={
                    **dict(spec.metadata),
                    "uk_uc_resolved_source_months": json.dumps(
                        expected, separators=(",", ":")
                    ),
                    "uk_uc_source_month_count": str(len(expected)),
                    "uk_uc_source_period_basis": (
                        "mean_of_declared_monthly_cell_sums"
                        if reference.value_operation == "monthly_window_sum_average"
                        else "mean_of_declared_months"
                    ),
                },
            )
        )
    return TargetRegistry(validated, country=registry.country)
