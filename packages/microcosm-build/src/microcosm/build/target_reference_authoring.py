"""Author Ledger target-reference resources from country contracts.

Country packages own target declarations and country-specific pinning policy.
This module owns the shared mechanics: candidate reference construction,
fan-out over native dimension values, compilation through the real Ledger
resolver, and membership reporting.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from microcosm.build.ledger_targets import (  # pyright: ignore[reportPrivateUsage]
    LedgerTargetReference,
    _assertion_allowed,
    _fact_matches_selector,
    _not_after_target_period,
    _period_key,
    _period_key_from_value,
    compile_ledger_target_references,
)

FanoutName = Callable[[Mapping[str, Any], Mapping[str, Any]], str | None]
GeographyPin = Mapping[str, str]


@dataclass(frozen=True)
class AuthoredTargetReferences:
    """Generated references and their compile-membership report."""

    references: tuple[dict[str, Any], ...]
    membership_report: dict[str, Any]

    @property
    def status_counts(self) -> Mapping[str, int]:
        """Membership status counts keyed by report status."""

        return self.membership_report["status_counts"]


@dataclass(frozen=True)
class TargetReferenceAuthoringConfig:
    """Country-supplied authoring policy."""

    target_period: int | str
    geography_pins: Mapping[str, GeographyPin] = field(default_factory=dict)
    fanout_name: FanoutName | None = None
    sum_target_ids: frozenset[str] = frozenset()
    value_operation_by_target_id: Mapping[str, str] = field(default_factory=dict)
    selector_pins_by_target_id: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    reference_metadata_by_target_id: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    signed_exclusions_by_target_id: Mapping[str, str] = field(default_factory=dict)
    binding_vocabulary: frozenset[str] = frozenset()
    source_fact_feed: str = ""


@dataclass(frozen=True)
class AreaSignedDeferral:
    """Signed area-level compile deferral for one contract target.

    Missing cells are the default use. ``defer_if_compiles`` is an explicit
    opt-in for a separate adjudication that prevents an available fact from
    binding; ordinary deferrals remain stale once their cell compiles.
    """

    target_id: str
    geography_level: str
    reason_id: str
    rationale: str
    area_ids: tuple[str, ...]
    defer_if_compiles: bool = False

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("AreaSignedDeferral.target_id must be non-empty.")
        if not self.geography_level:
            raise ValueError("AreaSignedDeferral.geography_level must be non-empty.")
        if not self.reason_id:
            raise ValueError("AreaSignedDeferral.reason_id must be non-empty.")
        if not self.rationale:
            raise ValueError("AreaSignedDeferral.rationale must be non-empty.")
        if not self.area_ids:
            raise ValueError("AreaSignedDeferral.area_ids must be non-empty.")
        object.__setattr__(
            self,
            "area_ids",
            tuple(dict.fromkeys(str(area_id) for area_id in self.area_ids)),
        )


@dataclass(frozen=True)
class AreaTargetReferenceAuthoringConfig:
    """Country-supplied authoring policy for area-grain targets."""

    target_period: int | str
    areas_by_geography_level: Mapping[str, Iterable[str]]
    area_signed_deferrals: tuple[AreaSignedDeferral, ...] = ()
    value_operation_by_target_id: Mapping[str, str] = field(default_factory=dict)
    selector_pins_by_target_id: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    reference_metadata_by_target_id: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    binding_vocabulary: frozenset[str] = frozenset()
    source_fact_feed: str = ""

    def normalized_areas(self) -> dict[str, tuple[str, ...]]:
        """Return area rosters with duplicate ids removed in declared order."""

        rosters: dict[str, tuple[str, ...]] = {}
        for level, area_ids in self.areas_by_geography_level.items():
            values = tuple(dict.fromkeys(str(area_id) for area_id in area_ids))
            if not values:
                raise ValueError(
                    f"Area roster for geography level {level!r} must be non-empty."
                )
            rosters[str(level)] = values
        return rosters


def author_target_references(
    contract: Mapping[str, Any],
    facts: Iterable[Mapping[str, Any]],
    config: TargetReferenceAuthoringConfig,
) -> AuthoredTargetReferences:
    """Build active target references and a membership report."""

    fact_rows = tuple(facts)
    facts_by_source = _facts_by_source(fact_rows)
    _validate_contract_bindings(contract, config.binding_vocabulary)
    active_rows: list[dict[str, Any]] = []
    target_entries: dict[str, Any] = {}
    geography_pin_report: dict[str, Any] = {}
    genuine_sum_residue: list[str] = []
    uprating_holds: list[dict[str, str]] = []

    for target in contract.get("targets", ()):
        target_id = str(target["target_id"])
        pin = config.geography_pins.get(target_id, {})
        geography_pin_report[target_id] = dict(pin)
        source_facts = _source_prefilter(fact_rows, facts_by_source, target)
        selector = _target_selector(target, pin, config)
        signed_exclusion = config.signed_exclusions_by_target_id.get(target_id)
        if signed_exclusion is not None:
            matched = [
                fact for fact in source_facts if _fact_matches_selector(fact, selector)
            ]
            eligible = [
                fact
                for fact in matched
                if _not_after_target_period(
                    _period_key(fact),
                    _period_key_from_value(config.target_period),
                )
            ]
            target_entries[target_id] = {
                "status": "signed_excluded",
                "candidates": [
                    {
                        "name": target_id,
                        "status": "signed_excluded",
                        "matched_fact_count_overall": len(matched),
                        "matched_fact_count_at_or_before_period": len(eligible),
                        "signed_rationale": signed_exclusion,
                    }
                ],
            }
            continue
        candidates = tuple(_candidate_rows(target, source_facts, pin, config))
        candidate_entries: list[dict[str, Any]] = []
        for row in candidates:
            reference = LedgerTargetReference(**row)
            matched = [
                fact
                for fact in source_facts
                if _fact_matches_selector(fact, row["ledger_selector"])
            ]
            eligible = [fact for fact in matched if _eligible_fact(reference, fact)]
            entry: dict[str, Any] = {
                "name": row["name"],
                "status": "",
                "matched_fact_count_overall": len(matched),
                "matched_fact_count_at_or_before_period": len(eligible),
            }
            if reference.period_match_policy == "source_window":
                entry["matched_fact_count_in_source_window"] = sum(
                    _assertion_allowed(reference, fact) for fact in matched
                )
            try:
                registry = compile_ledger_target_references(
                    matched,
                    [reference],
                    country=str(contract["country"]),
                )
            except ValueError as error:
                entry["status"] = _classify_deferral(error, matched, eligible)
                entry["error"] = _compact_compile_error(entry["status"])
            else:
                spec = registry.specs[0]
                resolved_period = spec.metadata.get("ledger_fact_period", "")
                _apply_uprating_hold(row, resolved_period, config.target_period)
                if "uprating_from_period" in row:
                    uprating_holds.append(
                        {
                            "name": row["name"],
                            "from": str(row["uprating_from_period"]),
                            "to": str(row["uprating_to_period"]),
                        }
                    )
                if row.get("value_operation") == "sum":
                    genuine_sum_residue.append(row["name"])
                active_rows.append(row)
                entry.update(
                    {
                        "status": "active",
                        "resolved_period": spec.period,
                        "resolved_value": spec.value,
                        "resolved_fact_period": resolved_period,
                        "resolved_fact_key": _json_safe_ledger_id(
                            spec.metadata.get("ledger_aggregate_fact_key", "")
                        ),
                    }
                )
            candidate_entries.append(entry)
        target_entries[target_id] = {
            "status": _target_status(candidate_entries),
            "candidates": candidate_entries,
        }

    status_counts = Counter(
        entry["status"]
        for target in target_entries.values()
        for entry in target["candidates"]
    )
    report = {
        "source_fact_feed": config.source_fact_feed,
        "target_period": config.target_period,
        "candidate_count": sum(
            len(target["candidates"]) for target in target_entries.values()
        ),
        "contract_target_count": len(tuple(contract.get("targets", ()))),
        "active_reference_count": len(active_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "geography_pins": geography_pin_report,
        "genuine_sum_residue": sorted(genuine_sum_residue),
        "uprating_holds": uprating_holds,
        "targets": target_entries,
    }
    return AuthoredTargetReferences(tuple(active_rows), report)


def author_area_target_references(
    contract: Mapping[str, Any],
    facts: Iterable[Mapping[str, Any]],
    config: AreaTargetReferenceAuthoringConfig,
) -> AuthoredTargetReferences:
    """Build area-grain target references over a declared geography roster."""

    fact_rows = tuple(facts)
    areas_by_level = config.normalized_areas()
    signed = _area_deferral_index(config, contract, areas_by_level)
    facts_by_area = _facts_by_geography_id(fact_rows)
    _validate_contract_bindings(contract, config.binding_vocabulary)

    active_rows: list[dict[str, Any]] = []
    target_entries: dict[str, Any] = {}
    uprating_holds: list[dict[str, str]] = []

    for target in contract.get("targets", ()):
        target_id = str(target["target_id"])
        level_entries: dict[str, Any] = {}
        for geography_level in target.get("geography_levels", ()):
            geography_level = str(geography_level)
            area_ids = areas_by_level.get(geography_level)
            if area_ids is None:
                # A declared level the roster lacks must refuse, not drop out:
                # skipping here yields no candidates, no references, and a
                # membership status of not_applicable -- a whole level could
                # leave the surface with the report agreeing nothing is wrong
                # (PR #795 round-3 review finding 2).
                raise ValueError(
                    f"target {target_id!r} declares geography level "
                    f"{geography_level!r}, which the area roster does not "
                    "carry."
                )
            candidates: list[dict[str, Any]] = []
            for area_id in area_ids:
                key = (target_id, geography_level, area_id)
                selector = _area_target_selector(
                    target,
                    geography_level=geography_level,
                    area_id=area_id,
                    config=config,
                )
                row = _area_reference_row(
                    target,
                    selector,
                    geography_level=geography_level,
                    area_id=area_id,
                    config=config,
                )
                source_facts = facts_by_area.get(area_id, ())
                matched = [
                    fact
                    for fact in source_facts
                    if _fact_matches_selector(fact, selector)
                ]
                reference = LedgerTargetReference(**row)
                eligible = [fact for fact in matched if _eligible_fact(reference, fact)]
                signed_deferral = signed.get(key)
                entry: dict[str, Any] = {
                    "name": row["name"],
                    "target_id": target_id,
                    "geography_level": geography_level,
                    "geography_id": area_id,
                    "status": "",
                    "matched_fact_count_overall": len(matched),
                    "matched_fact_count_at_or_before_period": len(eligible),
                }
                try:
                    registry = compile_ledger_target_references(
                        matched,
                        [reference],
                        country=str(contract["country"]),
                    )
                except ValueError as error:
                    status = _classify_area_deferral(error, matched, eligible)
                    entry["status"] = status
                    entry["error"] = _compact_compile_error(status)
                    if signed_deferral is None:
                        raise ValueError(
                            "Unsigned local target absence for "
                            f"{target_id!r} at {geography_level!r}/{area_id!r}: "
                            f"{status}."
                        ) from error
                    entry["signed_reason_id"] = signed_deferral.reason_id
                    entry["signed_rationale"] = signed_deferral.rationale
                else:
                    if (
                        signed_deferral is not None
                        and not signed_deferral.defer_if_compiles
                    ):
                        raise ValueError(
                            "Stale area signed deferral for "
                            f"{target_id!r} at {geography_level!r}/{area_id!r}: "
                            "the candidate now compiles."
                        )
                    if signed_deferral is not None:
                        entry.update(
                            {
                                "status": "signed_deferred",
                                "signed_reason_id": signed_deferral.reason_id,
                                "signed_rationale": signed_deferral.rationale,
                            }
                        )
                        candidates.append(entry)
                        continue
                    spec = registry.specs[0]
                    resolved_period = str(spec.metadata.get("ledger_fact_period", ""))
                    _apply_uprating_hold(row, resolved_period, config.target_period)
                    if "uprating_from_period" in row:
                        uprating_holds.append(
                            {
                                "name": row["name"],
                                "target_id": target_id,
                                "geography_level": geography_level,
                                "geography_id": area_id,
                                "from": str(row["uprating_from_period"]),
                                "to": str(row["uprating_to_period"]),
                            }
                        )
                    active_rows.append(row)
                    entry.update(
                        {
                            "status": "active",
                            "resolved_period": spec.period,
                            "resolved_value": spec.value,
                            "resolved_fact_period": resolved_period,
                            "resolved_fact_key": _json_safe_ledger_id(
                                spec.metadata.get("ledger_aggregate_fact_key", "")
                            ),
                        }
                    )
                candidates.append(entry)
            level_entries[geography_level] = {
                "status": _target_status(candidates),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        target_entries[target_id] = {
            "status": _target_status(
                [
                    entry
                    for level in level_entries.values()
                    for entry in level["candidates"]
                ]
            ),
            "geography_levels": level_entries,
        }

    status_counts = Counter(
        entry["status"]
        for target in target_entries.values()
        for level in target["geography_levels"].values()
        for entry in level["candidates"]
    )
    report = {
        "source_fact_feed": config.source_fact_feed,
        "target_period": config.target_period,
        "candidate_count": sum(
            level["candidate_count"]
            for target in target_entries.values()
            for level in target["geography_levels"].values()
        ),
        "contract_target_count": len(tuple(contract.get("targets", ()))),
        "active_reference_count": len(active_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "areas_by_geography_level": {
            level: list(area_ids) for level, area_ids in areas_by_level.items()
        },
        "signed_deferrals": [
            {
                "target_id": deferral.target_id,
                "geography_level": deferral.geography_level,
                "reason_id": deferral.reason_id,
                "rationale": deferral.rationale,
                "area_ids": list(deferral.area_ids),
                **({"defer_if_compiles": True} if deferral.defer_if_compiles else {}),
            }
            for deferral in config.area_signed_deferrals
        ],
        "uprating_holds": uprating_holds,
        "holds_by_target": dict(
            sorted(Counter(row["target_id"] for row in uprating_holds).items())
        ),
        "targets": target_entries,
    }
    return AuthoredTargetReferences(tuple(active_rows), report)


def target_references_resource(
    *,
    country: str,
    description: str,
    authored: AuthoredTargetReferences,
) -> dict[str, Any]:
    """Return a serialisable ``target_references.json`` resource."""

    return {
        "country": country,
        "description": description,
        "allowed_value_operations": [
            "identity",
            "sum",
            "difference",
            "calendar_year_average",
            "latest_plateau",
            "count_x_mean",
            *sorted(
                {
                    row["value_operation"]
                    for row in authored.references
                    if row.get("value_operation")
                    in {"monthly_window_average", "monthly_window_sum_average"}
                }
            ),
        ],
        "target_references": list(authored.references),
    }


def _validate_contract_bindings(
    contract: Mapping[str, Any],
    binding_vocabulary: frozenset[str],
) -> None:
    if not binding_vocabulary:
        return
    for target in contract.get("targets", ()):
        binding = target["bindings"]["policyengine"]
        unknown = sorted(set(binding) - binding_vocabulary)
        if unknown:
            raise ValueError(
                f"Contract target {target['target_id']!r} carries unknown "
                f"policyengine binding keys {unknown!r}."
            )


def _candidate_rows(
    target: Mapping[str, Any],
    facts: tuple[Mapping[str, Any], ...],
    geography_pin: GeographyPin,
    config: TargetReferenceAuthoringConfig,
) -> Iterable[dict[str, Any]]:
    selector = _target_selector(target, geography_pin, config)
    if "groupby_dimension" in selector:
        rows = _fanout_rows(target, facts, selector, config)
        if rows:
            yield from rows
            return
    yield _reference_row(target, selector, config)


def _target_selector(
    target: Mapping[str, Any],
    geography_pin: GeographyPin,
    config: TargetReferenceAuthoringConfig,
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    selector = {**dict(target["ledger_selector"]), **dict(geography_pin)}
    pins = config.selector_pins_by_target_id.get(target_id, {})
    for key, value in pins.items():
        if key == "dimension_values" and isinstance(value, Mapping):
            existing = selector.get("dimension_values")
            selector[key] = {
                **(dict(existing) if isinstance(existing, Mapping) else {}),
                **dict(value),
            }
            continue
        selector[key] = value
    return selector


def _area_target_selector(
    target: Mapping[str, Any],
    *,
    geography_level: str,
    area_id: str,
    config: AreaTargetReferenceAuthoringConfig,
) -> dict[str, Any]:
    selector = {
        **dict(target["ledger_selector"]),
        "geography_level": geography_level,
        "geography_id": area_id,
    }
    pins = config.selector_pins_by_target_id.get(str(target["target_id"]), {})
    for key, value in pins.items():
        if key == "dimension_values" and isinstance(value, Mapping):
            existing = selector.get("dimension_values")
            selector[key] = {
                **(dict(existing) if isinstance(existing, Mapping) else {}),
                **dict(value),
            }
            continue
        selector[key] = value
    return selector


def _area_reference_row(
    target: Mapping[str, Any],
    selector: Mapping[str, Any],
    *,
    geography_level: str,
    area_id: str,
    config: AreaTargetReferenceAuthoringConfig,
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    binding = target["bindings"]["policyengine"]
    row: dict[str, Any] = {
        "name": f"{target_id}@{area_id}",
        "ledger_selector": dict(selector),
        "entity": binding.get("from_entity") or binding.get("map_to") or "household",
        "measure": binding["metric_name"],
        "family": target["family"],
        "period": config.target_period,
        "metadata": {
            "contract_target_id": target_id,
            "measure_kind": "prepared_column",
            "geography_level": geography_level,
            "geography_id": area_id,
            **dict(config.reference_metadata_by_target_id.get(target_id, {})),
        },
    }
    assertion_policy = target.get("assertion_policy")
    if assertion_policy is not None:
        row["assertion_policy"] = assertion_policy
    value_operation = config.value_operation_by_target_id.get(target_id)
    if value_operation is not None and value_operation != "identity":
        row["value_operation"] = value_operation
    return row


def _fanout_rows(
    target: Mapping[str, Any],
    facts: tuple[Mapping[str, Any], ...],
    selector: Mapping[str, Any],
    config: TargetReferenceAuthoringConfig,
) -> tuple[dict[str, Any], ...]:
    dimension = str(selector["groupby_dimension"])
    matches = [fact for fact in facts if _fact_matches_selector(fact, selector)]
    existing_pins = selector.get("dimension_values") or {}
    pinned_dimension_names = (
        frozenset(str(key) for key in existing_pins)
        if isinstance(existing_pins, Mapping)
        else frozenset()
    )
    pins: dict[tuple[str, str], tuple[str, Any, Mapping[str, Any]]] = {}
    for fact in matches:
        dimension_name, dimension_value = _native_groupby_pin(
            fact,
            dimension,
            pinned_dimension_names=pinned_dimension_names,
        )
        if not dimension_name:
            continue
        key = (dimension_name, json.dumps(dimension_value, sort_keys=True))
        pins.setdefault(key, (dimension_name, dimension_value, fact))
    rows = []
    for _, (dimension_name, dimension_value, fact) in sorted(pins.items()):
        merged_dimension_values = {
            **(dict(existing_pins) if isinstance(existing_pins, Mapping) else {}),
            dimension_name: dimension_value,
        }
        pinned_selector = {
            **dict(selector),
            "dimension_values": merged_dimension_values,
        }
        name = (
            config.fanout_name(target, fact)
            if config.fanout_name is not None
            else _fallback_fanout_name(target, fact)
        )
        if not name:
            continue
        rows.append(_reference_row(target, pinned_selector, config, name=name))
    return tuple(rows)


def _facts_by_source(
    facts: tuple[Mapping[str, Any], ...],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for fact in facts:
        source = _source_name(fact)
        if source:
            buckets.setdefault(source, []).append(fact)
    return {key: tuple(value) for key, value in buckets.items()}


def _source_prefilter(
    facts: tuple[Mapping[str, Any], ...],
    facts_by_source: Mapping[str, tuple[Mapping[str, Any], ...]],
    target: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    source_name = target["ledger_selector"].get("source_name")
    if isinstance(source_name, str) and source_name:
        return facts_by_source.get(source_name, ())
    return facts


def _source_name(fact: Mapping[str, Any]) -> str:
    return str(
        fact.get("source", {}).get("source_name")
        or fact.get("observed_measure", {}).get("source_name")
        or ""
    )


def _facts_by_geography_id(
    facts: tuple[Mapping[str, Any], ...],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for fact in facts:
        area_id = str(fact.get("geography", {}).get("id") or "")
        if area_id:
            buckets.setdefault(area_id, []).append(fact)
    return {key: tuple(value) for key, value in buckets.items()}


def _area_deferral_index(
    config: AreaTargetReferenceAuthoringConfig,
    contract: Mapping[str, Any],
    areas_by_level: Mapping[str, tuple[str, ...]],
) -> dict[tuple[str, str, str], AreaSignedDeferral]:
    target_levels = {
        (str(target["target_id"]), str(level))
        for target in contract.get("targets", ())
        for level in target.get("geography_levels", ())
    }
    index: dict[tuple[str, str, str], AreaSignedDeferral] = {}
    for deferral in config.area_signed_deferrals:
        if (deferral.target_id, deferral.geography_level) not in target_levels:
            raise ValueError(
                "Area signed deferral references undeclared target/geography "
                f"{deferral.target_id!r}/{deferral.geography_level!r}."
            )
        roster = set(areas_by_level.get(deferral.geography_level, ()))
        outside = sorted(set(deferral.area_ids) - roster)
        if outside:
            raise ValueError(
                "Area signed deferral references area id(s) outside the roster "
                f"for {deferral.geography_level!r}: {outside!r}."
            )
        for area_id in deferral.area_ids:
            key = (deferral.target_id, deferral.geography_level, area_id)
            if key in index:
                raise ValueError(f"Duplicate area signed deferral for {key!r}.")
            index[key] = deferral
    return index


def _native_groupby_pin(
    fact: Mapping[str, Any],
    dimension: str,
    *,
    pinned_dimension_names: frozenset[str] = frozenset(),
) -> tuple[str, Any]:
    dimensions = fact.get("dimensions") or {}
    if not isinstance(dimensions, Mapping):
        return "", None
    candidates = (
        dimension,
        dimension.split(".")[-1],
        dimension.replace(".", "_"),
    )
    for candidate in candidates:
        if candidate in dimensions:
            if candidate in pinned_dimension_names:
                return "", None
            return candidate, dimensions[candidate]
    unpinned_dimensions = [
        (str(key), value)
        for key, value in dimensions.items()
        if str(key) not in pinned_dimension_names
    ]
    if len(unpinned_dimensions) == 1:
        return unpinned_dimensions[0]
    if len(dimensions) == 1:
        key, value = next(iter(dimensions.items()))
        if str(key) in pinned_dimension_names:
            return "", None
        return str(key), value
    return "", None


def _reference_row(
    target: Mapping[str, Any],
    selector: Mapping[str, Any],
    config: TargetReferenceAuthoringConfig,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    binding = target["bindings"]["policyengine"]
    row: dict[str, Any] = {
        "name": name or target_id,
        "ledger_selector": dict(selector),
        "entity": binding.get("from_entity") or binding.get("map_to") or "household",
        "measure": name or binding["metric_name"],
        "family": target["family"],
        "period": config.target_period,
        "metadata": {
            "contract_target_id": target_id,
            "measure_kind": "prepared_column",
            **dict(config.reference_metadata_by_target_id.get(target_id, {})),
        },
    }
    assertion_policy = target.get("assertion_policy")
    if assertion_policy is not None:
        row["assertion_policy"] = assertion_policy
    period_match_policy = target.get("period_match_policy")
    if period_match_policy is not None:
        row["period_match_policy"] = period_match_policy
    value_operation = config.value_operation_by_target_id.get(target_id)
    if value_operation is None and target_id in config.sum_target_ids:
        value_operation = "sum"
    if value_operation is not None and value_operation != "identity":
        row["value_operation"] = value_operation
    operands = target.get("value_operands")
    if operands is not None:
        row["value_operands"] = operands
    if target.get("expected_member_count") is not None:
        row["expected_member_count"] = target["expected_member_count"]
    dimension_values = selector.get("dimension_values")
    if value_operation == "sum" and isinstance(dimension_values, Mapping):
        member_count = 1
        has_list = False
        for value in dimension_values.values():
            if isinstance(value, list):
                has_list = True
                member_count *= len(value)
        if has_list:
            row["expected_member_count"] = member_count
    return row


def _fallback_fanout_name(target: Mapping[str, Any], fact: Mapping[str, Any]) -> str:
    value_id = str(fact.get("layout", {}).get("groupby_value_id") or "detail")
    return f"{target['target_id']}.{value_id}"


def _json_safe_ledger_id(value: str) -> str:
    return value.replace(".", "_").replace(":", "_")


def _compact_compile_error(status: str) -> str:
    return f"ledger_reference_compile_status_{status}"


def _eligible_fact(reference: LedgerTargetReference, fact: Mapping[str, Any]) -> bool:
    return _not_after_target_period(
        _period_key(fact),
        _period_key_from_value(reference.period),
    ) and _assertion_allowed(reference, fact)


def _classify_deferral(
    error: ValueError,
    matched: list[Mapping[str, Any]],
    eligible: list[Mapping[str, Any]],
) -> str:
    message = str(error)
    if "source_projection" in message:
        return "source_projection_policy"
    if not eligible or "at or before target period" in message:
        return "no_fact_at_or_before_period"
    if "matched multiple Ledger facts" in message:
        return "multi_fact"
    if "invalid value" in message or "missing value" in message:
        return "value_not_finite_or_missing"
    geography_ids = {
        str(fact.get("geography", {}).get("id", ""))
        for fact in eligible
        if fact.get("geography", {}).get("level") == "country"
    }
    if len(geography_ids) > 1:
        return "geography_ambiguous_at_country_level"
    if not matched:
        return "no_fact_at_or_before_period"
    return "signed_deferral_compile_error"


def _classify_area_deferral(
    error: ValueError,
    matched: list[Mapping[str, Any]],
    eligible: list[Mapping[str, Any]],
) -> str:
    if not matched:
        return "no_fact_for_area"
    return _classify_deferral(error, matched, eligible)


def _target_status(candidate_entries: list[dict[str, Any]]) -> str:
    if not candidate_entries:
        return "not_applicable"
    if any(entry["status"] == "active" for entry in candidate_entries):
        if all(entry["status"] == "active" for entry in candidate_entries):
            return "active"
        return "partially_active"
    statuses = sorted({entry["status"] for entry in candidate_entries})
    return statuses[0] if len(statuses) == 1 else "multiple_deferrals"


def _apply_uprating_hold(
    row: dict[str, Any],
    resolved_period: str,
    target_period: int | str,
) -> None:
    if not resolved_period or row.get("period_match_policy") == "source_window":
        return
    source_key = _period_key_from_value(resolved_period)
    target_key = _period_key_from_value(target_period)
    if source_key[0] and target_key[0] and source_key[1] < target_key[1]:
        row["uprating_from_period"] = resolved_period
        row["uprating_to_period"] = target_period


def is_finite_value(value: object) -> bool:
    """Return whether a fact value is finite when coerced to float."""

    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False
