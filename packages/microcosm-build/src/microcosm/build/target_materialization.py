"""Shared materialization of declarative Ledger target bindings."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from microcosm.calibrate import TargetRegistry

Provider = Callable[[Any, Mapping[str, Any], int | str], np.ndarray]

# Banded measures declare their slice in two places: the contract binding
# names the variable to band on (``groupby_variable``), and each compiled
# spec carries its own band's lower edge in Ledger filter metadata. Two
# encodings are in use — a numeric lower bound (HMRC SPI income bands) and a
# published range label (DWP award bands, in monthly units, hence
# ``band_period_factor``) — and both reduce to one lower edge, because no
# reference anywhere declares an upper bound. A band's upper edge is its
# sibling's lower edge within the same compiled contract target. Edges must
# never derive from an exclusion-pruned roster, or excluding a band silently
# widens its lower neighbour; only the compiled register's top band runs to
# infinity unless the binding declares a source-unit ``band_upper_bound``.
# Entity-count indicators: a value of one per record of the owning entity.
_COUNT_VALUE_VARIABLES = frozenset({"household_count", "person_count", "benunit_count"})

_BAND_LOWER_BOUND_SUFFIX = "_lower_bound"
_LEDGER_FILTER_PREFIX = "ledger_filter_"
_RANGE_LABEL = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(?:to|–|—|-)\s*(?:£\s*)?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OPEN_LABEL = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(?:or more|or over|and over|and above|\+)",
    re.IGNORECASE,
)
_MISSING_COLUMN = re.compile(r"^'(?:([a-z_0-9]+)\.)?([A-Za-z_0-9]+)'$")


class MeasureProvider(Protocol):
    """Provider of adapter-level variables needed for target materialization.

    ``knows`` answers whether the provider can compute a variable for a given
    entity. ``compute`` returns the values and a short receipted route. The
    returned array is injected into fresh probe adapters only; it is never
    written back to the source frame.
    """

    def knows(self, entity: str, variable: str) -> bool: ...

    def compute(self, entity: str, variable: str) -> tuple[np.ndarray, str]: ...


@dataclass(frozen=True)
class MaterializationSkip:
    """One target measure the interpreter could not prepare."""

    name: str
    measure: str
    reason: str


@dataclass(frozen=True)
class MeasureResolution:
    """Adapter-level variables resolved for target materialization."""

    measure_inputs: Mapping[tuple[str, str], np.ndarray]
    receipt: Mapping[str, Any]


class BandEdgeCoverageError(RuntimeError):
    """A supplied band-edge register does not cover a materialized spec.

    Deliberately not a :class:`ValueError`: the per-spec materialization loop
    converts ValueErrors into :class:`MaterializationSkip` entries, and a
    register that cannot bound a spec is a wrong-register problem for the
    whole run, not a per-spec data defect — it must refuse, never let the
    target quietly drop out of the solve (#792 review finding 2).
    """


class MeasureResolutionError(RuntimeError):
    """Raised when simulated measure resolution cannot bind the registry."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]):
        super().__init__(message)
        self.receipt = dict(receipt)


@dataclass(frozen=True)
class TargetMaterializationResult:
    """Prepared target columns and skipped-measure diagnostics."""

    adapter: Any
    registry: TargetRegistry
    skipped: tuple[MaterializationSkip, ...] = field(default_factory=tuple)

    def report(self) -> dict[str, object]:
        """Return a serialisable diagnostic report."""

        return {
            "prepared_count": len(self.registry) - len(self.skipped),
            "skipped_count": len(self.skipped),
            "skipped": [skip.__dict__ for skip in self.skipped],
        }


def assert_calibration_input_finite(frame: Any) -> None:
    """Refuse NaN-bearing float inputs before calibration materialization."""

    failures: list[str] = []
    for entity in frame.entities:
        table = frame.table(entity)
        for column in table.columns:
            series = table[column]
            if getattr(series.dtype, "kind", None) != "f":
                continue
            count = int(series.isna().sum())
            if count:
                failures.append(f"{entity}.{column}: {count}")
    if failures:
        raise ValueError(
            "Calibration input contains NaN float values: " + "; ".join(failures)
        )


def resolve_target_measures(
    adapter_factory: Callable[[], Any],
    registry: TargetRegistry,
    provider: MeasureProvider,
    *,
    period: int | str,
    max_rounds: int = 8,
    contract_targets: Mapping[str, Mapping[str, Any]] | None = None,
    band_edge_registry: TargetRegistry | None = None,
) -> MeasureResolution:
    """Resolve provider-computed inputs until target materialization binds.

    ``adapter_factory`` must return a fresh adapter over the same frame each
    round. Previously resolved values are injected into those probe adapters'
    table copies only, then the shared target-binding interpreter is retried.

    The source frame is never mutated — the guarantee rests on the adapter
    copying each entity table at construction (``UKFrameTargetAdapter``
    does), not on the later ``restore``. Any adapter passed here must copy;
    ``test_uk_national_calibration.py`` holds that property directly, on the
    source frame rather than on the restored output.
    """

    contract = _measure_resolution_contract(
        registry, provider, contract_targets=contract_targets
    )
    measure_inputs: dict[tuple[str, str], np.ndarray] = {}
    receipt: dict[str, Any] = {
        "rounds": [],
        "attached": {},
        "provider": _measure_provider_receipt(provider),
    }
    all_skips: list[dict[str, Any]] = []

    for round_index in range(max_rounds):
        probe = adapter_factory()
        _inject_measure_inputs(probe, measure_inputs)
        result = materialize_target_bindings(
            probe,
            registry,
            contract,
            period=period,
            band_edge_registry=band_edge_registry,
        )
        skipped = tuple(result.skipped)
        round_receipt = {
            "round": round_index,
            "provided_before": len(measure_inputs),
            "skipped": [skip.__dict__ for skip in skipped],
        }
        receipt["rounds"].append(round_receipt)
        all_skips.extend({**skip.__dict__, "round": round_index} for skip in skipped)
        if not skipped:
            return MeasureResolution(
                measure_inputs=MappingProxyType(dict(measure_inputs)),
                receipt=MappingProxyType(receipt),
            )

        provided_before = set(measure_inputs)
        provided_this_round: set[tuple[str, str]] = set()
        progressed = False
        for skip in skipped:
            reason = str(skip.reason)
            if "counterfactual_delta" in reason or "counterfactual delta" in reason:
                _raise_measure_resolution(
                    "counterfactual target measure cannot be resolved",
                    receipt,
                    all_skips,
                )
            match = _MISSING_COLUMN.match(reason)
            if match is None:
                _raise_measure_resolution(
                    f"unparseable target materialization skip reason: {reason}",
                    receipt,
                    all_skips,
                )
            entity, variable = match.group(1), match.group(2)
            if entity is None:
                entity = _provider_entity_for(provider, variable)
            if entity is None:
                _raise_measure_resolution(
                    f"target measure {variable!r} did not identify an entity",
                    receipt,
                    all_skips,
                )
            key = (entity, variable)
            if key in provided_this_round:
                continue
            if key in provided_before:
                _raise_measure_resolution(
                    f"{entity}.{variable} remained unmaterializable after injection",
                    receipt,
                    all_skips,
                )
            if not provider.knows(entity, variable):
                _raise_measure_resolution(
                    f"provider does not know {entity}.{variable}",
                    receipt,
                    all_skips,
                )
            try:
                values, route = provider.compute(entity, variable)
            except Exception as exc:  # noqa: BLE001 - receipt preserves cause
                _raise_measure_resolution(
                    f"provider failed computing {entity}.{variable}: {exc}",
                    receipt,
                    all_skips,
                )
            measure_inputs[key] = np.asarray(values)
            # Providers can learn the comparison contract while resolving
            # inputs (for example, UC claim-state/date proxies). Preserve the
            # post-computation receipt, not only the pre-resolution snapshot.
            receipt["provider"] = _measure_provider_receipt(provider)
            provided_this_round.add(key)
            receipt["attached"][f"{entity}.{variable}"] = route
            progressed = True
        if not progressed:
            _raise_measure_resolution(
                f"measure materialization made no progress on round {round_index}",
                receipt,
                all_skips,
            )

    _raise_measure_resolution(
        f"measure materialization did not converge after {max_rounds} rounds",
        receipt,
        all_skips,
    )


def _measure_resolution_contract(
    registry: TargetRegistry,
    provider: MeasureProvider,
    *,
    contract_targets: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Mapping[str, Any]]:
    if contract_targets is not None:
        return contract_targets
    for owner in (registry, provider):
        value = getattr(owner, "contract_targets", None)
        if value is not None:
            return value
    raise TypeError(
        "resolve_target_measures requires contract_targets, or a registry/provider "
        "with a contract_targets attribute."
    )


def _measure_provider_receipt(provider: MeasureProvider) -> Mapping[str, Any]:
    receipt = getattr(provider, "receipt", None)
    if callable(receipt):
        return dict(receipt())
    return {"class": type(provider).__name__}


def _provider_entity_for(provider: MeasureProvider, variable: str) -> str | None:
    entity_for = getattr(provider, "entity_for", None)
    if callable(entity_for):
        value = entity_for(variable)
        return None if value is None else str(value)
    return None


def _inject_measure_inputs(
    adapter: Any, measure_inputs: Mapping[tuple[str, str], np.ndarray]
) -> None:
    for (entity, variable), values in measure_inputs.items():
        adapter.tables[entity][variable] = values


def _raise_measure_resolution(
    message: str,
    receipt: dict[str, Any],
    skips: list[dict[str, Any]],
) -> None:
    receipt["skips"] = list(skips)
    raise MeasureResolutionError(message, receipt=receipt)


def _band_lower_edge(spec: Any, binding: Mapping[str, Any]) -> float | None:
    """This spec's band lower edge in model units, or None if unbanded.

    Reads the compiled spec's Ledger filter metadata: a ``*_lower_bound``
    entry when the publisher gives numeric edges, otherwise a range label
    such as ``"£500.01 to £600.00"`` or ``"5 or more"``. ``band_period_factor``
    converts published units to the model's (monthly award bands to annual).
    Non-band filters — ``ledger_filter_family_type = "Single, no children"``
    — carry no numeric range and are ignored here; the binding's own
    ``filters`` list applies those.
    """

    metadata = getattr(spec, "metadata", None) or {}
    factor = float(binding.get("band_period_factor", 1) or 1)
    numeric: list[tuple[str, float]] = []
    labelled: list[tuple[str, float]] = []
    for key, value in sorted(metadata.items()):
        if not key.startswith(_LEDGER_FILTER_PREFIX):
            continue
        if key.endswith(_BAND_LOWER_BOUND_SUFFIX):
            numeric.append((key, float(str(value).replace(",", ""))))
            continue
        match = _RANGE_LABEL.search(str(value)) or _OPEN_LABEL.search(str(value))
        if match is not None:
            labelled.append((key, float(match.group(1).replace(",", ""))))
    for candidates in (numeric, labelled):
        if not candidates:
            continue
        return _select_band_edge(spec, binding, candidates) * factor
    return None


def _select_band_edge(
    spec: Any,
    binding: Mapping[str, Any],
    candidates: Sequence[tuple[str, float]],
) -> float:
    """Pick the band edge belonging to this binding's groupby variable.

    A spec can carry several Ledger filters, and more than one of them can
    look like a band — an age filter alongside an income band, say. Slicing
    on the wrong variable's edges produces plausible-looking but wrong
    subpopulation totals, so the edge is bound to the declared banding
    dimension and an unresolvable tie refuses rather than picking the
    alphabetically-first key.
    """

    if len(candidates) == 1:
        return candidates[0][1]
    declared = binding.get("band_filter_dimension")
    wanted = {
        f"{_LEDGER_FILTER_PREFIX}{name}"
        for name in (declared, binding.get("groupby_variable"))
        if name
    }
    matched = [
        value
        for key, value in candidates
        if any(key == name or key.startswith(f"{name}_") for name in wanted)
    ]
    if len(matched) == 1:
        return matched[0]
    raise ValueError(
        f"banded measure {getattr(spec, 'measure', '?')!r} carries "
        f"{len(candidates)} band-like Ledger filters "
        f"({', '.join(key for key, _ in candidates)}) and "
        f"{'none' if not matched else 'several'} of them belong to "
        f"groupby_variable {binding.get('groupby_variable')!r}; declare "
        "band_filter_dimension on the binding to name the banding dimension."
    )


def _band_edges_by_group(
    registry: TargetRegistry,
    contract_targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[float]]:
    """Sorted band lower edges per contract target that declares bands.

    Grouped by contract target rather than by dimension: two measures can
    share a dimension while publishing different band sets, and taking the
    union of their edges would slice each measure on the other's boundaries.
    """

    edges: dict[str, set[float]] = {}
    for spec in registry.specs:
        contract_target_id = str(
            (getattr(spec, "metadata", None) or {}).get("contract_target_id", "")
        )
        target = contract_targets.get(contract_target_id)
        if target is None:
            continue
        binding = target.get("bindings", {}).get("policyengine", {})
        if not binding.get("groupby_variable"):
            continue
        lower = _band_lower_edge(spec, binding)
        if lower is not None:
            edges.setdefault(contract_target_id, set()).add(lower)
    return {key: sorted(values) for key, values in edges.items()}


def _band_bounds(
    spec: Any,
    binding: Mapping[str, Any],
    band_edges: Sequence[float],
) -> tuple[float, float]:
    """The half-open ``[lower, upper)`` this spec's band covers.

    Raises rather than returning an unsliced column: a banded measure whose
    band cannot be read would otherwise report the whole population as if it
    were one band, which is the failure this function exists to end.
    """

    lower = _band_lower_edge(spec, binding)
    if lower is None:
        raise ValueError(
            f"banded measure {getattr(spec, 'measure', '?')!r} declares "
            f"groupby_variable {binding['groupby_variable']!r} but its spec "
            "carries no readable band edge"
        )
    if lower not in band_edges:
        raise BandEdgeCoverageError(
            f"banded measure {getattr(spec, 'measure', '?')!r} carries band lower edge "
            f"{lower!r} that is absent from its contract target's band-edge set "
            f"{list(band_edges)!r}; the band-edge register does not cover this spec's "
            "band — pass the compiled (pre-exclusion) register this spec was pruned from."
        )
    upper = math.inf
    for edge in band_edges:
        if edge > lower:
            upper = edge
            break
    # A consumer may bind only the finite portion of a published histogram.
    # Declare its source-unit ceiling instead of allowing the last included
    # row to absorb a separately published, unbound top-coded category.
    if "band_upper_bound" in binding:
        inclusive = binding.get("band_upper_bound_inclusive")
        factor = float(binding.get("band_period_factor", 1))
        declared_upper = float(binding["band_upper_bound"]) * factor
        if (
            not isinstance(inclusive, bool)
            or not math.isfinite(factor)
            or factor <= 0
            or not math.isfinite(declared_upper)
            or declared_upper <= lower
        ):
            raise ValueError(
                "band_upper_bound requires a finite bound above the lower "
                "edge, a positive period factor and an explicit boolean "
                "band_upper_bound_inclusive"
            )
        if inclusive:
            declared_upper = np.nextafter(declared_upper, math.inf)
        upper = min(upper, declared_upper)
    elif "band_upper_bound_inclusive" in binding:
        raise ValueError("band_upper_bound_inclusive requires band_upper_bound")
    return lower, upper


def materialize_target_bindings(
    adapter: Any,
    registry: TargetRegistry,
    contract_targets: Mapping[str, Mapping[str, Any]],
    *,
    period: int | str,
    providers: Mapping[str, Provider] | None = None,
    band_edge_registry: TargetRegistry | None = None,
) -> TargetMaterializationResult:
    """Prepare measure columns declared by compiled Ledger target specs."""

    provider_registry = {**default_provider_registry(), **(providers or {})}
    band_edges = _band_edges_by_group(
        registry if band_edge_registry is None else band_edge_registry,
        contract_targets,
    )
    skipped: list[MaterializationSkip] = []
    for spec in registry.specs:
        if hasattr(adapter, "has_column") and adapter.has_column(
            spec.entity, spec.measure
        ):
            continue
        contract_target_id = spec.metadata.get("contract_target_id")
        target = contract_targets.get(str(contract_target_id))
        if target is None:
            skipped.append(
                MaterializationSkip(
                    name=spec.name,
                    measure=spec.measure,
                    reason="missing_contract_target",
                )
            )
            continue
        binding = target["bindings"]["policyengine"]
        kind = binding.get("kind")
        try:
            if kind:
                provider = provider_registry.get(str(kind))
                if provider is None:
                    raise ValueError(f"unsupported binding kind {kind!r}")
                values = provider(adapter, binding, period)
            else:
                band = None
                if binding.get("groupby_variable"):
                    band = _band_bounds(
                        spec,
                        binding,
                        band_edges.get(str(contract_target_id), ()),
                    )
                values = _prepared_column_values(
                    adapter, spec.entity, binding, band=band
                )
            adapter.set_column(spec.entity, spec.measure, values)
        except (KeyError, TypeError, ValueError) as error:
            skipped.append(
                MaterializationSkip(
                    name=spec.name,
                    measure=spec.measure,
                    reason=str(error),
                )
            )
    return TargetMaterializationResult(adapter, registry, tuple(skipped))


def default_provider_registry() -> dict[str, Provider]:
    """Return the generic provider engines keyed by binding kind."""

    return {
        "parameter_gated_threshold": parameter_gated_threshold,
        "baseline_flag_crosstab": baseline_flag_crosstab,
        "input_substitution_counterfactual": input_substitution_counterfactual,
    }


def parameter_gated_threshold(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    """Materialize an indicator/value gated by a period parameter."""

    gate = float(adapter.parameter(str(binding["gate_parameter"]), period))
    gated = _column(
        adapter, str(binding.get("from_entity") or "person"), binding["gated_variable"]
    )
    comparison = str(binding.get("gate_comparison") or ">")
    mask = _compare(gated, comparison, gate)
    value_variable = binding.get("value_variable")
    if value_variable and value_variable != "person_count":
        return np.where(
            mask,
            _column(
                adapter,
                str(binding.get("from_entity") or "person"),
                str(value_variable),
            ),
            0.0,
        )
    return mask.astype(float)


def baseline_flag_crosstab(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    """Materialize a baseline affected-flag crosstab column."""

    del period
    entity = str(binding.get("from_entity") or "household")
    flag = _column(adapter, entity, binding["affected_flag_variable"]).astype(bool)
    value_variable = str(binding.get("value_variable") or "")
    count_of = str(binding.get("count_of") or "")
    reduction = binding.get("value_reduction")
    if reduction:
        # A count declared over a member-level variable — the number of
        # children in the household, not whether it has any. Reading such a
        # boolean at the crosstab's own grain would collapse it to an
        # indicator and publish that against a count target, so the reduction
        # is declared and performed member-wise instead.
        values = _entity_reduction(adapter, reduction)
    elif value_variable and value_variable not in _COUNT_VALUE_VARIABLES:
        values = _column(adapter, entity, value_variable)
    elif count_of and count_of not in {"household", "person", "household_count"}:
        values = _column(adapter, entity, count_of)
    else:
        values = np.ones_like(flag, dtype=float)
    mask = flag
    for predicate in binding.get("filters", ()):
        mask &= _predicate_mask(adapter, entity, predicate)
    for predicate in binding.get("household_conditions", ()):
        mask &= _predicate_mask(adapter, entity, predicate)
    return np.where(mask, values, 0.0)


def input_substitution_counterfactual(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    """Materialize an output delta under a caller-owned counterfactual."""

    if hasattr(adapter, "counterfactual_delta"):
        return np.asarray(adapter.counterfactual_delta(binding, period), dtype=float)
    raise ValueError("adapter does not provide counterfactual_delta")


def _prepared_column_values(
    adapter: Any,
    entity: str,
    binding: Mapping[str, Any],
    *,
    band: tuple[float, float] | None = None,
) -> np.ndarray:
    entity = str(binding.get("from_entity") or entity)
    if "value_expression" in binding:
        values = _expression(adapter, entity, str(binding["value_expression"]))
    else:
        values = _column(adapter, entity, binding["value_variable"])
    mask = np.ones_like(values, dtype=bool)
    for predicate in binding.get("filters", ()):
        mask &= _predicate_mask(adapter, entity, predicate)
    for predicate in binding.get("household_conditions", ()):
        mask &= _predicate_mask(adapter, entity, predicate)
    if band is not None:
        lower, upper = band
        banded = _column(adapter, entity, binding["groupby_variable"]).astype(float)
        mask &= (banded >= lower) & (banded < upper)
    return np.where(mask, values, 0.0)


def _entity_reduction(adapter: Any, reduction: Mapping[str, Any]) -> np.ndarray:
    """Reduce a member-level variable to the target entity's grain.

    Declared as ``value_reduction`` on a binding whose value is a count over
    members. The adapter owns the linkage, the same way it owns
    ``household_condition``; an adapter that cannot reduce refuses rather than
    falling back to a same-grain read, which would silently substitute an
    indicator for a count.
    """

    if not hasattr(adapter, "entity_reduction"):
        raise ValueError(
            "binding declares value_reduction "
            f"{dict(reduction)!r} but the adapter cannot reduce member "
            "variables; refusing to substitute a same-grain read."
        )
    return np.asarray(adapter.entity_reduction(reduction), dtype=float)


def _predicate_mask(
    adapter: Any,
    default_entity: str,
    predicate: Mapping[str, Any],
) -> np.ndarray:
    if "reduce" in predicate and hasattr(adapter, "household_condition"):
        return np.asarray(adapter.household_condition(predicate), dtype=bool)
    entity = str(predicate.get("entity") or default_entity)
    variable = str(predicate.get("variable") or predicate.get("concept"))
    values = _column(adapter, entity, variable)
    if "operator" in predicate:
        operator, expected = str(predicate["operator"]), predicate["value"]
    else:
        operator, expected = "==", predicate["equals"]
    return _compare(values, operator, expected)


def _compare(values: np.ndarray, operator: str, expected: object) -> np.ndarray:
    if operator == "==":
        return values == expected
    if operator == "!=":
        return values != expected
    if operator == "in":
        return np.isin(values, expected)
    numeric = values.astype(float)
    threshold = float(expected)
    if operator == ">":
        return numeric > threshold
    if operator == ">=":
        return numeric >= threshold
    if operator == "<":
        return numeric < threshold
    if operator == "<=":
        return numeric <= threshold
    raise ValueError(f"unsupported predicate operator {operator!r}")


def _expression(adapter: Any, entity: str, expression: str) -> np.ndarray:
    parts = [part.strip() for part in expression.split("+")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"unsupported value_expression {expression!r}")
    total: np.ndarray | None = None
    for part in parts:
        values = _column(adapter, entity, part)
        total = values.copy() if total is None else total + values
    if total is None:
        raise ValueError(f"unsupported value_expression {expression!r}")
    return total


def _column(adapter: Any, entity: str, variable: object) -> np.ndarray:
    if hasattr(adapter, "column"):
        return np.asarray(adapter.column(entity, str(variable)))
    table = adapter.tables[entity]
    return np.asarray(table[str(variable)])
