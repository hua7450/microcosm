"""Ledger fact catalog adapters for Microcosm target profiles.

Ledger owns source-backed facts. Microcosm owns the active subset and the
model-variable mapping needed to compile a fact into a calibration target.
This module is intentionally duck-typed against the Ledger aggregate fact
schema so Microcosm can consume exported JSONL catalogs without importing the
Ledger implementation package at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from microcosm.calibrate import TargetRegistry, TargetSpec

SUPPORTED_LEDGER_AGGREGATIONS = frozenset(("sum",))
ALLOWED_ASSERTION_POLICIES = frozenset(("observed_only", "allow_source_projection"))
ALLOWED_PERIOD_MATCH_POLICIES = frozenset(
    ("latest_not_after", "exact", "source_window")
)
MONTHLY_WINDOW_OPERATIONS = frozenset(
    ("monthly_window_average", "monthly_window_sum_average")
)
ALLOWED_VALUE_OPERATIONS = frozenset(
    (
        "identity",
        "sum",
        "difference",
        "calendar_year_average",
        "latest_plateau",
        "count_x_mean",
        *MONTHLY_WINDOW_OPERATIONS,
    )
)
MULTI_FACT_VALUE_OPERATIONS = frozenset(
    (
        "sum",
        "difference",
        "calendar_year_average",
        "latest_plateau",
        "count_x_mean",
        *MONTHLY_WINDOW_OPERATIONS,
    )
)
EXACT_PERIOD_VALUE_OPERATIONS = frozenset(
    ("identity", "sum", "difference", "count_x_mean")
)
DEFAULT_HIERARCHY_MATCH_SPEC_FIELDS = ("entity", "period", "family", "filter")


@dataclass(frozen=True)
class LedgerTargetMapping:
    """How Microcosm maps Ledger facts to model-ready target rows.

    A Ledger fact is source truth; it is not automatically a Microcosm target.
    Microcosm must know which model column estimates the fact's measure, and
    optionally which model boolean column implements the fact's scoped
    population. Facts without a mapping are reported as unsupported.
    """

    measure_by_concept: Mapping[str, str] = field(default_factory=dict)
    measure_by_source_record_id: Mapping[str, str] = field(default_factory=dict)
    filter_by_source_record_id: Mapping[str, str] = field(default_factory=dict)
    filter_by_domain: Mapping[str, str] = field(default_factory=dict)
    signed_by_concept: frozenset[str] = frozenset()
    signed_by_source_record_id: frozenset[str] = frozenset()
    entity_by_ledger_entity: Mapping[str, str] = field(default_factory=dict)
    family_by_source_name: Mapping[str, str] = field(default_factory=dict)
    family_by_concept: Mapping[str, str] = field(default_factory=dict)
    default_family: str = "ledger"


@dataclass(frozen=True)
class LedgerTargetReference:
    """A Microcosm calibration target backed by one Ledger fact.

    Ledger owns the observed fact: value, source lineage, source dimensions, and
    period. Microcosm owns whether that fact is active for a build and how it
    maps into the model: entity, measure column, optional filter, calibration
    family, and release-gate metadata. Uprating fields are declarations for the
    materialization layer; this compiler records them in target metadata after a
    caller has resolved the appropriate Ledger fact/value.
    """

    name: str
    ledger_fact_key: str = ""
    ledger_source_record_id: str = ""
    ledger_selector: Mapping[str, object] = field(default_factory=dict)
    value_operation: str = "identity"
    value_operands: tuple[Mapping[str, object], ...] = ()
    expected_member_count: int | None = None
    entity: str = ""
    measure: str | None = None
    filter: str | None = None
    period: int | str | None = None
    source: str | None = None
    family: str = "ledger"
    signed: bool = False
    se: float | None = None
    tolerance: float | None = None
    notes: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    assertion_policy: str = "observed_only"
    period_match_policy: str = "latest_not_after"
    uprating_index: str | None = None
    uprating_from_period: int | str | None = None
    uprating_to_period: int | str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LedgerTargetReference.name must be non-empty.")
        if (
            not self.ledger_fact_key
            and not self.ledger_source_record_id
            and not self.ledger_selector
        ):
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: either ledger_fact_key "
                "or ledger_source_record_id or ledger_selector is required."
            )
        if self.value_operation not in ALLOWED_VALUE_OPERATIONS:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: unsupported value_operation "
                f"{self.value_operation!r}. Microcosm currently permits only "
                f"{sorted(ALLOWED_VALUE_OPERATIONS)!r} resolution from Ledger facts."
            )
        if not isinstance(self.ledger_selector, Mapping):
            raise TypeError(
                f"LedgerTargetReference {self.name!r}: ledger_selector must be a "
                f"mapping, got {type(self.ledger_selector).__name__}."
            )
        if self.expected_member_count is not None and (
            isinstance(self.expected_member_count, bool)
            or not isinstance(self.expected_member_count, int)
            or self.expected_member_count <= 0
        ):
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: expected_member_count "
                "must be a positive integer."
            )
        if self.value_operation == "difference":
            roles = [str(operand.get("role")) for operand in self.value_operands]
            if roles != ["minuend", "subtrahend"]:
                raise ValueError(
                    f"LedgerTargetReference {self.name!r}: difference requires "
                    "exactly ordered minuend/subtrahend operands."
                )
        if self.assertion_policy not in ALLOWED_ASSERTION_POLICIES:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: unsupported "
                f"assertion_policy {self.assertion_policy!r}."
            )
        if self.period_match_policy not in ALLOWED_PERIOD_MATCH_POLICIES:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: unsupported "
                f"period_match_policy {self.period_match_policy!r}; expected "
                f"one of {sorted(ALLOWED_PERIOD_MATCH_POLICIES)!r}."
            )
        is_window = self.value_operation in MONTHLY_WINDOW_OPERATIONS
        if is_window != (self.period_match_policy == "source_window"):
            raise ValueError(
                "A monthly window operation requires the paired source_window period policy."
            )
        if is_window:
            if self.period is None:
                raise ValueError(
                    "A monthly window requires an explicit model target period."
                )
            _declared_source_months(self)
            _monthly_window_operands(self)
        if self.period_match_policy == "exact" and self.period is None:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: period_match_policy="
                "'exact' requires an explicit target period."
            )
        if (
            self.period_match_policy == "exact"
            and self.value_operation not in EXACT_PERIOD_VALUE_OPERATIONS
        ):
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: period_match_policy="
                f"'exact' does not support value_operation {self.value_operation!r}; "
                "calendar_year_average and latest_plateau consume subperiod "
                "series whose selection semantics are not an exact scalar-period "
                "match. Use latest_not_after or add an explicit subperiod "
                "materialization contract before activating this reference."
            )
        if not self.entity:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: entity must be non-empty."
            )
        if not self.family:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: family must be non-empty."
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"LedgerTargetReference {self.name!r}: metadata must be a "
                f"mapping, got {type(self.metadata).__name__}."
            )
        metadata = {str(key): str(value) for key, value in self.metadata.items()}
        bad_metadata = sorted(
            key for key, value in metadata.items() if not key or not value
        )
        if bad_metadata:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: metadata keys and values "
                f"must be non-empty strings; bad keys {bad_metadata}."
            )
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class LedgerTargetParityReport:
    """Exact and calibration-effective parity between two target registries."""

    passed: bool
    failures: tuple[str, ...]
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class UnsupportedLedgerTarget:
    """A Ledger fact Microcosm deliberately did not activate."""

    reason: str
    identifier: str
    concept: str = ""
    source_record_id: str = ""


@dataclass(frozen=True)
class LedgerTargetSelection:
    """Selected Microcosm targets plus explicit unsupported Ledger facts."""

    specs: tuple[TargetSpec, ...]
    unsupported: tuple[UnsupportedLedgerTarget, ...]

    def to_registry(self, *, country: str) -> TargetRegistry:
        """Build a Microcosm target registry from the selected specs."""
        return TargetRegistry(self.specs, country=country)


@dataclass(frozen=True)
class LedgerTargetHierarchyRule:
    """Spec-declared parent/child reconciliation for Ledger-backed targets."""

    rule_id: str
    child_geography_level: str
    parent_geography_level: str
    parent_geography_id_template: str
    method: str = "scale_children_to_parent"
    match_spec_fields: tuple[str, ...] = DEFAULT_HIERARCHY_MATCH_SPEC_FIELDS
    match_metadata_keys: tuple[str, ...] = ()
    enabled_when: Mapping[str, str] = field(default_factory=dict)
    child_id_metadata_key: str = ""
    parent_key_metadata_key: str = ""
    expected_child_count_by_parent_key: Mapping[str, int] = field(default_factory=dict)
    on_incomplete: str = "fail"

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("LedgerTargetHierarchyRule.rule_id must be non-empty.")
        if self.method != "scale_children_to_parent":
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: unsupported "
                f"method {self.method!r}."
            )
        if not self.child_geography_level:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: "
                "child_geography_level is required."
            )
        if not self.parent_geography_level:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: "
                "parent_geography_level is required."
            )
        if not self.parent_geography_id_template:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: "
                "parent_geography_id.template is required."
            )
        object.__setattr__(
            self,
            "match_spec_fields",
            tuple(str(field_name) for field_name in self.match_spec_fields),
        )
        object.__setattr__(
            self,
            "match_metadata_keys",
            tuple(str(key) for key in self.match_metadata_keys),
        )
        object.__setattr__(
            self,
            "enabled_when",
            {
                str(key): _profile_scalar(value)
                for key, value in self.enabled_when.items()
            },
        )
        object.__setattr__(
            self,
            "expected_child_count_by_parent_key",
            {
                str(key): int(value)
                for key, value in self.expected_child_count_by_parent_key.items()
            },
        )
        if bool(self.expected_child_count_by_parent_key) != bool(
            self.child_id_metadata_key and self.parent_key_metadata_key
        ):
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: child "
                "completeness requires child_id_metadata_key, "
                "parent_key_metadata_key, and expected_child_count_by_parent_key."
            )
        bad_counts = sorted(
            key
            for key, value in self.expected_child_count_by_parent_key.items()
            if value <= 0
        )
        if bad_counts:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: expected child "
                f"counts must be positive; bad parent keys {bad_counts!r}."
            )
        if self.on_incomplete not in {"fail", "skip"}:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: unsupported "
                f"on_incomplete action {self.on_incomplete!r}."
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> LedgerTargetHierarchyRule:
        parent_geography_id = raw.get("parent_geography_id") or {}
        if not isinstance(parent_geography_id, Mapping):
            raise TypeError(
                "Ledger target hierarchy rule parent_geography_id must be a mapping."
            )
        match = raw.get("match") or {}
        if not isinstance(match, Mapping):
            raise TypeError("Ledger target hierarchy rule match must be a mapping.")
        enabled_when = raw.get("enabled_when") or {}
        if not isinstance(enabled_when, Mapping):
            raise TypeError(
                "Ledger target hierarchy rule enabled_when must be a mapping."
            )
        completeness = raw.get("child_completeness") or {}
        if not isinstance(completeness, Mapping):
            raise TypeError(
                "Ledger target hierarchy rule child_completeness must be a mapping."
            )
        expected_child_counts = (
            completeness.get("expected_child_count_by_parent_key") or {}
        )
        if not isinstance(expected_child_counts, Mapping):
            raise TypeError(
                "Ledger target hierarchy rule "
                "expected_child_count_by_parent_key must be a mapping."
            )
        return cls(
            rule_id=str(raw.get("id") or ""),
            method=str(raw.get("method") or "scale_children_to_parent"),
            child_geography_level=str(raw.get("child_geography_level") or ""),
            parent_geography_level=str(raw.get("parent_geography_level") or ""),
            parent_geography_id_template=str(parent_geography_id.get("template") or ""),
            match_spec_fields=tuple(
                str(field_name)
                for field_name in (
                    match.get("spec_fields") or DEFAULT_HIERARCHY_MATCH_SPEC_FIELDS
                )
            ),
            match_metadata_keys=tuple(
                str(key) for key in (match.get("metadata_keys") or ())
            ),
            enabled_when=enabled_when,
            child_id_metadata_key=str(completeness.get("child_id_metadata_key") or ""),
            parent_key_metadata_key=str(
                completeness.get("parent_key_metadata_key") or ""
            ),
            expected_child_count_by_parent_key=expected_child_counts,
            on_incomplete=str(completeness.get("on_incomplete") or "fail"),
        )


@dataclass(frozen=True)
class _LedgerFactIndex:
    facts: tuple[object, ...]
    by_identifier: Mapping[str, tuple[object, ...]]

    def lookup(self, identifier: str) -> tuple[object, ...]:
        return self.by_identifier.get(identifier, ())


def select_ledger_targets(
    facts: Iterable[object],
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> LedgerTargetSelection:
    """Select the Ledger facts Microcosm can target.

    Args:
        facts: Ledger ``AggregateFact`` objects or JSON-like mappings.
        mapping: Explicit Microcosm model-variable mapping.
        period: Optional period override for every resulting target.

    Returns:
        A selection with model-ready :class:`TargetSpec` rows and unsupported
        facts with reasons. Unsupported facts are a feature: the target catalog
        can be broader than Microcosm's current build surface.
    """

    specs: list[TargetSpec] = []
    unsupported: list[UnsupportedLedgerTarget] = []
    for fact in facts:
        try:
            specs.append(target_spec_from_ledger_fact(fact, mapping, period=period))
        except _UnsupportedFactError as exc:
            unsupported.append(exc.unsupported)
    return LedgerTargetSelection(tuple(specs), tuple(unsupported))


def select_ledger_targets_from_jsonl(
    path: str | Path,
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> LedgerTargetSelection:
    """Select Ledger facts from a JSON Lines consumer-contract file."""
    with Path(path).open() as file:
        return select_ledger_targets(
            _jsonl_rows(file),
            mapping,
            period=period,
        )


def compile_ledger_target_references(
    facts: Iterable[object],
    references: Iterable[LedgerTargetReference],
    *,
    country: str,
) -> TargetRegistry:
    """Resolve Microcosm target references against Ledger facts.

    This is the structured-registry path: each reference names the Ledger fact
    row it consumes, then overlays the model-facing calibration attributes that
    cannot live in the source database.
    """

    fact_index = _ledger_fact_index(facts)
    specs: list[TargetSpec] = []
    for reference in references:
        _require_executable_reference(reference)
        resolved = _resolve_reference_fact(reference, fact_index)
        specs.append(target_spec_from_ledger_reference(resolved, reference))
    return TargetRegistry(specs, country=country)


def _require_executable_reference(reference: LedgerTargetReference) -> None:
    """Refuse authoring placeholders until a scalar/fanout reference replaces them.

    Country packages may declare a future target surface before Chronicle has
    harvested and pinned the corresponding facts.  Those rows are useful schema
    evidence, but they are not executable selectors: in particular, resolving a
    multi-cell table through ``identity`` would either depend on accidental
    singleton input or become ambiguous as soon as the next cell arrived.
    """

    activation_status = reference.metadata.get("activation_status", "")
    if activation_status and activation_status != "active":
        raise ValueError(
            f"Ledger target reference {reference.name!r} is a non-executable "
            f"placeholder with activation_status={activation_status!r}. Replace "
            "it with harvested, cell-pinned references (or a reviewed scalar "
            "fact reference) before compilation."
        )


def apply_ledger_target_profile(
    registry: TargetRegistry,
    profile: Mapping[str, object] | None,
    *,
    context: Mapping[str, object] | None = None,
) -> TargetRegistry:
    """Apply spec-declared target-profile transformations.

    Ledger owns the raw facts. Microcosm target profiles can declare generic
    transformations needed to compile those facts into a coherent calibration
    surface, such as reconciling child-geography rows to an available parent.
    The declarations are data; this shared compiler owns the implementation.
    """

    if not profile:
        return registry
    schema_version = profile.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(
            "Ledger target profile schema_version must be an integer 1 or 2, "
            f"got {schema_version!r}."
        )
    if schema_version not in {1, 2}:
        raise ValueError(
            "Ledger target profile schema_version must be 1 or 2, got "
            f"{schema_version!r}."
        )
    result = registry
    for raw_rule in profile.get("hierarchy_reconciliations") or ():
        if not isinstance(raw_rule, Mapping):
            raise TypeError("Ledger hierarchy reconciliation entries must be mappings.")
        rule = LedgerTargetHierarchyRule.from_mapping(raw_rule)
        if _hierarchy_rule_enabled(rule, context or {}):
            result = reconcile_ledger_target_hierarchy(result, rule)
    return result


def reconcile_ledger_target_hierarchy(
    registry: TargetRegistry,
    rule: LedgerTargetHierarchyRule,
) -> TargetRegistry:
    """Scale child-geography target rows to their matching parent target row."""

    parent_by_key: dict[tuple[str, ...], TargetSpec] = {}
    child_groups: dict[tuple[str, ...], list[tuple[int, TargetSpec]]] = {}
    for index, spec in enumerate(registry.specs):
        geography_level = spec.metadata.get("ledger_geography_level", "")
        if geography_level == rule.parent_geography_level:
            key = _hierarchy_parent_key(spec, rule)
            existing = parent_by_key.get(key)
            if existing is not None:
                raise ValueError(
                    f"Ledger target hierarchy rule {rule.rule_id!r} found "
                    f"multiple parent targets for key {key!r}: "
                    f"{existing.name!r} and {spec.name!r}."
                )
            parent_by_key[key] = spec
        elif geography_level == rule.child_geography_level:
            parent_geography_id = _hierarchy_parent_geography_id(spec, rule)
            if not parent_geography_id:
                continue
            child_groups.setdefault(
                _hierarchy_match_key(spec, rule) + (parent_geography_id,),
                [],
            ).append((index, spec))

    replacements: dict[int, TargetSpec] = {}
    for key, children in child_groups.items():
        parent = parent_by_key.get(key)
        if parent is None:
            continue
        completeness = _hierarchy_completeness(parent, children, rule)
        if completeness is not None and not completeness["complete"]:
            if rule.on_incomplete == "skip":
                continue
            raise ValueError(
                f"Ledger target hierarchy rule {rule.rule_id!r} expected "
                f"{completeness['expected_child_count']} child target(s) for "
                f"parent key {completeness['parent_key']!r} and parent "
                f"{parent.name!r}, got {completeness['observed_child_count']}."
            )
        raw_sum = sum(spec.value for _, spec in children)
        parent_value = parent.value
        if raw_sum == 0 and parent_value != 0:
            child_names = [spec.name for _, spec in children]
            raise ValueError(
                f"Ledger target hierarchy rule {rule.rule_id!r} cannot scale "
                f"zero-valued children to nonzero parent {parent.name!r}; "
                f"children: {child_names!r}."
            )
        factor = 1.0 if raw_sum == 0 else parent_value / raw_sum
        if factor < 0:
            child_names = [spec.name for _, spec in children]
            raise ValueError(
                f"Ledger target hierarchy rule {rule.rule_id!r} cannot reconcile "
                f"opposite-signed child and parent targets for {parent.name!r}; "
                f"children: {child_names!r}."
            )
        coverage_ratio = 1.0 if parent_value == 0 else raw_sum / parent_value
        for index, spec in children:
            replacements[index] = replace(
                spec,
                value=spec.value * factor,
                metadata={
                    **dict(spec.metadata),
                    "hierarchy_reconciliation_rule": rule.rule_id,
                    "hierarchy_reconciliation_method": rule.method,
                    "hierarchy_raw_value": _format_float(spec.value),
                    "hierarchy_child_sum_raw": _format_float(raw_sum),
                    "hierarchy_parent_value": _format_float(parent_value),
                    "hierarchy_coverage_ratio": _format_float(coverage_ratio),
                    "hierarchy_reconciliation_factor": _format_float(factor),
                    "hierarchy_parent_target_name": parent.name,
                    "hierarchy_parent_target_period": str(parent.period),
                    "hierarchy_parent_geography_level": rule.parent_geography_level,
                    "hierarchy_parent_geography_id": parent.metadata.get(
                        "ledger_geography_id", ""
                    ),
                    **_hierarchy_completeness_metadata(completeness),
                },
            )
    if not replacements:
        return registry
    specs = [replacements.get(index, spec) for index, spec in enumerate(registry.specs)]
    return TargetRegistry(specs, country=registry.country)


def target_spec_from_ledger_reference(
    fact: object | tuple[object, ...],
    reference: LedgerTargetReference,
) -> TargetSpec:
    """Compile one resolved Ledger fact plus Microcosm mapping into a target."""

    _require_executable_reference(reference)
    facts = fact if isinstance(fact, tuple) else (fact,)
    if not facts:
        raise ValueError(f"Ledger target reference {reference.name!r} has no facts.")
    if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
        facts = _resolve_monthly_window_reference_facts(reference, list(facts))
    if len(facts) > 1 and reference.value_operation not in MULTI_FACT_VALUE_OPERATIONS:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: multiple Ledger facts "
            "require an aggregating value_operation, got "
            f"{reference.value_operation!r}."
        )
    numeric_values = []
    for member in facts:
        _validate_resolved_reference_fact(member, reference)
        numeric_values.append(_numeric_fact_value(member, reference))
        _validate_fact_aggregation(member, reference)

    if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
        numeric_value = sum(numeric_values) / len(_declared_source_months(reference))
    elif reference.value_operation == "calendar_year_average":
        numeric_value = sum(numeric_values) / len(numeric_values)
    elif reference.value_operation == "latest_plateau":
        numeric_value = numeric_values[-1]
    elif reference.value_operation == "count_x_mean":
        numeric_value = numeric_values[0] * numeric_values[1]
    elif reference.value_operation == "sum":
        numeric_value = sum(numeric_values)
    elif reference.value_operation == "difference":
        numeric_value = numeric_values[0] - numeric_values[1]
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: difference "
                f"produced invalid value {numeric_value!r}."
            )
    else:
        numeric_value = numeric_values[0]
    representative_fact = _value_representative_fact(
        facts,
        operation=reference.value_operation,
    )
    publication_metadata = {}
    if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
        # The window guard has proved these identities common to every member.
        publication_metadata = {
            key: value
            for key, value in {
                "ledger_source_release_key": _str_at(
                    representative_fact, "source_release_key"
                ),
                "ledger_source_sha256": _str_at(
                    representative_fact, "source", "source_sha256"
                ),
            }.items()
            if value
        }

    if not reference.measure:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: measure is required; "
            "count-like facts must be represented as sums of prepared indicator "
            "columns."
        )
    period = (
        reference.period
        if reference.period is not None
        else _at(representative_fact, "period", "value")
    )
    if period is None:
        raise ValueError(f"Ledger fact for {reference.name!r} is missing period.")

    return TargetSpec(
        name=reference.name,
        entity=reference.entity,
        measure=reference.measure,
        value=numeric_value,
        filter=reference.filter,
        period=period,
        se=reference.se,
        source=_source_citation(representative_fact),
        family=reference.family,
        signed=reference.signed,
        tolerance=reference.tolerance,
        notes=reference.notes,
        metadata={
            **_ledger_metadata(
                representative_fact,
                fact_key=_fact_key(representative_fact),
            ),
            **_multi_fact_reference_metadata(facts),
            **publication_metadata,
            "ledger_resolved_assertion": _fact_assertion(representative_fact),
            **_reference_metadata(reference),
        },
    )


def _value_representative_fact(
    facts: tuple[object, ...],
    *,
    operation: str,
) -> object:
    if len(facts) == 1 or operation not in MULTI_FACT_VALUE_OPERATIONS:
        return facts[0]
    return max(enumerate(facts), key=lambda item: (_period_key(item[1]), item[0]))[1]


def _numeric_fact_value(fact: object, reference: LedgerTargetReference) -> float:
    value = _at(fact, "value")
    if value is None:
        raise ValueError(f"Ledger fact for {reference.name!r} is missing value.")
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Ledger fact for {reference.name!r} has invalid value {value!r}."
        ) from exc
    if not math.isfinite(numeric_value):
        raise ValueError(
            f"Ledger fact for {reference.name!r} has invalid value {value!r}."
        )
    return numeric_value


def _validate_fact_aggregation(
    fact: object,
    reference: LedgerTargetReference,
) -> None:
    aggregation = _str_at(fact, "aggregation", "method")
    accepts_time_mean = (
        reference.metadata.get("fact_aggregation") == "time_mean"
        and aggregation == "mean"
    )
    accepts_count_x_mean = (
        reference.value_operation == "count_x_mean"
        and aggregation == "mean"
        and _count_mean_fact_role(fact) == "mean"
    )
    if (
        aggregation not in SUPPORTED_LEDGER_AGGREGATIONS
        and not accepts_time_mean
        and not accepts_count_x_mean
    ):
        raise ValueError(
            f"Ledger fact for {reference.name!r} has unsupported aggregation "
            f"{aggregation!r}; Microcosm targets must be compiled from sum "
            "facts, including counts represented as sums of prepared indicator "
            "columns. A mean fact is only accepted when the Microcosm mapping "
            "declares fact_aggregation=time_mean, asserting the mean is taken "
            "over time periods on a stock count (still a linear level), never "
            "over entities (a per-unit ratio is not calibratable)."
        )


def _multi_fact_reference_metadata(facts: tuple[object, ...]) -> dict[str, str]:
    if len(facts) == 1:
        return {}
    member_keys = tuple(_fact_key(fact) or _source_record_id(fact) for fact in facts)
    payload = json.dumps(member_keys, separators=(",", ":"))
    return {
        "ledger_member_fact_count": str(len(facts)),
        "ledger_member_aggregate_fact_keys": payload,
        "ledger_member_fact_keys": payload,
        "ledger_member_fact_digest": hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest(),
    }


def ledger_target_registry_parity_report(
    expected: TargetRegistry,
    actual: TargetRegistry,
) -> LedgerTargetParityReport:
    """Compare target registries before switching an active calibration path.

    The exact digest covers full :class:`TargetSpec` records. The calibration
    digest covers the fields consumed by solver compilation, so lineage-only
    registry additions can be reviewed without disguising calibration changes.
    """

    expected_exact = _registry_digest(expected)
    actual_exact = _registry_digest(actual)
    expected_calibration = _registry_digest(
        expected, payload_fn=_calibration_effective_spec_payload
    )
    actual_calibration = _registry_digest(
        actual, payload_fn=_calibration_effective_spec_payload
    )

    failures: list[str] = []
    if expected.country != actual.country:
        failures.append(
            f"country differs: expected {expected.country!r}, got {actual.country!r}."
        )
    if len(expected) != len(actual):
        failures.append(
            f"target count differs: expected {len(expected)}, got {len(actual)}."
        )
    if expected_exact != actual_exact:
        failures.append(
            f"full target registry digest differs: expected {expected_exact}, "
            f"got {actual_exact}."
        )
    if expected_calibration != actual_calibration:
        failures.append(
            "calibration-effective target digest differs: expected "
            f"{expected_calibration}, got {actual_calibration}."
        )
    if expected_exact != actual_exact or expected_calibration != actual_calibration:
        failures.extend(_first_spec_differences(expected, actual))

    return LedgerTargetParityReport(
        passed=not failures,
        failures=tuple(failures),
        details={
            "expected_count": len(expected),
            "actual_count": len(actual),
            "expected_version": expected.version,
            "actual_version": actual.version,
            "expected_full_digest": expected_exact,
            "actual_full_digest": actual_exact,
            "expected_calibration_digest": expected_calibration,
            "actual_calibration_digest": actual_calibration,
        },
    )


def target_spec_from_ledger_fact(
    fact: object,
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> TargetSpec:
    """Convert one supported Ledger fact to a Microcosm target spec.

    The target's ``name`` is the Ledger stable fact identifier when available,
    otherwise its source-record ID. Human-facing tooling should display the
    structured metadata fields, not this internal key.
    """

    source_record_id = _source_record_id(fact)
    fact_key = _fact_key(fact)
    identifier = fact_key or source_record_id
    if not identifier:
        raise _unsupported("missing_identifier", fact)

    concept = _primary_measure_concept(fact)
    if not concept:
        raise _unsupported("missing_measure_concept", fact)

    aggregation = _str_at(fact, "aggregation", "method")
    if aggregation not in SUPPORTED_LEDGER_AGGREGATIONS:
        raise _unsupported(f"unsupported_aggregation:{aggregation}", fact)

    measure = mapping.measure_by_source_record_id.get(source_record_id)
    if not measure:
        for candidate in _measure_concepts(fact):
            measure = mapping.measure_by_concept.get(candidate)
            if measure:
                break
    if not measure:
        raise _unsupported("missing_model_measure_mapping", fact)

    ledger_entity = _str_at(fact, "entity", "name")
    entity = mapping.entity_by_ledger_entity.get(ledger_entity, ledger_entity)
    if not entity:
        raise _unsupported("missing_entity", fact)

    value = _at(fact, "value")
    if value is None:
        raise _unsupported("missing_value", fact)
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _unsupported("invalid_value", fact) from exc
    if not math.isfinite(numeric_value):
        raise _unsupported("invalid_value", fact)

    target_period = period if period is not None else _at(fact, "period", "value")
    if target_period is None:
        raise _unsupported("missing_period", fact)

    source_name = _source_name(fact)
    family = (
        mapping.family_by_concept.get(concept)
        or mapping.family_by_source_name.get(source_name)
        or source_name
        or mapping.default_family
    )

    filter_column = mapping.filter_by_source_record_id.get(source_record_id)
    if filter_column is None and _requires_detail_filter_mapping(fact):
        raise _unsupported("missing_model_filter_mapping", fact)
    domain = _domain(fact)
    if filter_column is None and _requires_domain_filter_mapping(fact):
        filter_column = mapping.filter_by_domain.get(domain)
    if filter_column is None and _requires_filter_mapping(fact):
        raise _unsupported("missing_model_filter_mapping", fact)
    if numeric_value < 0 and not _allows_signed_target(fact, mapping):
        raise _unsupported("missing_signed_target_mapping", fact)

    try:
        return TargetSpec(
            name=identifier,
            entity=entity,
            measure=measure,
            value=numeric_value,
            filter=filter_column,
            period=target_period,
            source=_source_citation(fact),
            family=family,
            signed=numeric_value < 0,
            metadata=_ledger_metadata(fact, fact_key=fact_key),
        )
    except (TypeError, ValueError) as exc:
        raise _unsupported(f"invalid_target_spec:{type(exc).__name__}", fact) from exc


class _UnsupportedFactError(Exception):
    def __init__(self, unsupported: UnsupportedLedgerTarget) -> None:
        super().__init__(unsupported.reason)
        self.unsupported = unsupported


def _unsupported(reason: str, fact: object) -> _UnsupportedFactError:
    return _UnsupportedFactError(
        UnsupportedLedgerTarget(
            reason=reason,
            identifier=_fact_key(fact) or _source_record_id(fact),
            concept=_primary_measure_concept(fact),
            source_record_id=_source_record_id(fact),
        )
    )


def _ledger_fact_index(facts: Iterable[object]) -> _LedgerFactIndex:
    materialized_facts = tuple(facts)
    buckets: dict[str, list[object]] = {}
    for fact in materialized_facts:
        keys = (
            _str_at(fact, "aggregate_fact_key"),
            _str_at(fact, "semantic_fact_key"),
            _str_at(fact, "fact_key"),
            _str_at(fact, "legacy_fact_key"),
            _source_record_id(fact),
        )
        for key in dict.fromkeys(candidate for candidate in keys if candidate):
            bucket = buckets.setdefault(key, [])
            if not any(existing is fact for existing in bucket):
                bucket.append(fact)
    return _LedgerFactIndex(
        facts=materialized_facts,
        by_identifier={key: tuple(value) for key, value in buckets.items()},
    )


def _resolve_reference_fact(
    reference: LedgerTargetReference,
    fact_index: _LedgerFactIndex,
) -> object:
    identifiers = tuple(
        key
        for key in (reference.ledger_fact_key, reference.ledger_source_record_id)
        if key
    )
    matches_by_identifier = {
        key: fact_index.lookup(key) for key in identifiers if fact_index.lookup(key)
    }
    ambiguous = {
        key: matches
        for key, matches in matches_by_identifier.items()
        if len(matches) > 1
    }
    if ambiguous:
        examples = {
            key: [
                _fact_key(fact) or _source_record_id(fact) or f"fact[{index}]"
                for index, fact in enumerate(matches[:5])
            ]
            for key, matches in ambiguous.items()
        }
        raise ValueError(
            f"Ledger target reference {reference.name!r} matched multiple "
            f"Ledger facts for identifier(s): {examples!r}."
        )
    resolved = {key: matches[0] for key, matches in matches_by_identifier.items()}
    missing = tuple(key for key in identifiers if key not in resolved)
    if resolved and missing:
        raise ValueError(
            f"Ledger target reference {reference.name!r} matched only some Ledger "
            f"fact identifiers. Missing: {missing!r}."
        )
    if resolved:
        distinct_facts = {id(fact) for fact in resolved.values()}
        if len(distinct_facts) > 1:
            raise ValueError(
                f"Ledger target reference {reference.name!r} identifiers resolve "
                "to different Ledger facts."
            )
        return next(iter(resolved.values()))
    if reference.ledger_selector:
        matches = [
            fact
            for fact in fact_index.facts
            if _fact_matches_selector(fact, reference.ledger_selector)
        ]
        eligible_matches = _eligible_selector_matches(reference, matches)
        if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
            return _resolve_monthly_window_reference_facts(reference, eligible_matches)
        if reference.value_operation == "sum" and eligible_matches:
            return _resolve_sum_reference_facts(reference, eligible_matches)
        if reference.value_operation == "difference" and eligible_matches:
            return _resolve_difference_reference_facts(reference, eligible_matches)
        if reference.value_operation == "calendar_year_average" and eligible_matches:
            return _resolve_calendar_year_average_reference_facts(
                reference, eligible_matches
            )
        if reference.value_operation == "latest_plateau" and eligible_matches:
            return _resolve_latest_plateau_reference_facts(reference, eligible_matches)
        if reference.value_operation == "count_x_mean" and eligible_matches:
            return _resolve_count_x_mean_reference_facts(reference, eligible_matches)
        if len(eligible_matches) == 1:
            return eligible_matches[0]
        latest_match = (
            None
            if reference.period_match_policy == "exact"
            else _latest_period_selector_match(reference, eligible_matches)
        )
        if latest_match is not None:
            return latest_match
        if not matches:
            raise ValueError(
                f"Ledger target reference {reference.name!r} did not match a "
                f"Ledger fact selector: {dict(reference.ledger_selector)!r}."
            )
        if not eligible_matches:
            period_requirement = (
                "at exact target period"
                if reference.period_match_policy == "exact"
                else "at or before target period"
            )
            raise ValueError(
                f"Ledger target reference {reference.name!r} did not match a "
                f"Ledger fact {period_requirement} "
                f"{reference.period!r} for selector: "
                f"{dict(reference.ledger_selector)!r}."
            )
        identifiers = [
            _fact_key(fact) or _source_record_id(fact) or f"fact[{index}]"
            for index, fact in enumerate(eligible_matches)
        ]
        raise ValueError(
            f"Ledger target reference {reference.name!r} matched multiple "
            f"Ledger facts for selector {dict(reference.ledger_selector)!r}: "
            f"{identifiers!r}."
        )
    raise ValueError(
        f"Ledger target reference {reference.name!r} did not match a Ledger fact "
        f"identifier: {identifiers!r}."
    )


def _resolve_sum_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    partitions: dict[tuple[tuple[str, ...], tuple[int, int, str]], list[object]] = {}
    for fact in eligible_matches:
        key = (
            _selector_sum_partition_key(fact),
            _reference_period_partition_key(fact, reference),
        )
        partitions.setdefault(key, []).append(fact)
    if not partitions:
        raise ValueError(
            f"Ledger target reference {reference.name!r} did not match a Ledger "
            "fact partition."
        )
    latest_period = max(period_key for _, period_key in partitions)
    latest_matches = [
        fact
        for (_, period_key), facts in partitions.items()
        if period_key == latest_period
        for fact in facts
    ]
    if (
        reference.expected_member_count is not None
        and len(latest_matches) != reference.expected_member_count
    ):
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation=sum "
            f"expected {reference.expected_member_count} members at the latest "
            f"period but resolved {len(latest_matches)}; a declared member is missing."
        )
    return tuple(
        sorted(
            latest_matches,
            key=lambda fact: _fact_key(fact) or _source_record_id(fact),
        )
    )


def _resolve_difference_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    resolved: list[object] = []
    for operand in reference.value_operands:
        dimensions = operand.get("dimension_values")
        if not isinstance(dimensions, Mapping):
            raise ValueError(
                f"Ledger target reference {reference.name!r}: difference operand "
                "requires dimension_values."
            )
        shared_dimensions = reference.ledger_selector.get("dimension_values")
        selector = {
            **dict(reference.ledger_selector),
            **{
                str(key): value
                for key, value in operand.items()
                if key not in {"role", "dimension_values"}
            },
            "dimension_values": {
                **(
                    dict(shared_dimensions)
                    if isinstance(shared_dimensions, Mapping)
                    else {}
                ),
                **dict(dimensions),
            },
        }
        matches = [
            fact for fact in eligible_matches if _fact_matches_selector(fact, selector)
        ]
        match = _latest_period_selector_match(reference, matches)
        if match is None:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: difference operand "
                f"{operand.get('role')!r} did not resolve exactly once."
            )
        resolved.append(match)
    periods = {_period_key(fact) for fact in resolved}
    if len(periods) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: difference operands "
            "must resolve at the same latest period."
        )
    return tuple(resolved)


def _declared_source_months(reference: LedgerTargetReference) -> tuple[str, ...]:
    months = reference.ledger_selector.get("period_value")
    if (
        reference.ledger_selector.get("period_type") != "month"
        or not isinstance(months, list)
        or not months
        or not all(
            isinstance(month, str)
            and re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", month)
            for month in months
        )
        or months != sorted(set(months))
    ):
        raise ValueError(
            "A monthly window requires period_type=month and explicit unique ordered YYYY-MM period_value months."
        )
    return tuple(months)


def _monthly_window_operands(
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, object], ...]:
    if reference.value_operation == "monthly_window_average":
        if reference.value_operands:
            raise ValueError(
                "A single-series monthly window does not accept value_operands."
            )
        return ()
    dimensions = []
    for operand in reference.value_operands:
        if not isinstance(operand, Mapping) or set(operand) != {"dimension_values"}:
            raise ValueError(
                "A monthly window sum operand requires only dimension_values."
            )
        values = operand["dimension_values"]
        if (
            not isinstance(values, Mapping)
            or not values
            or any(
                not isinstance(key, str)
                or not key
                or value is None
                or isinstance(value, (Mapping, list, tuple, set))
                for key, value in values.items()
            )
        ):
            raise ValueError(
                "A monthly window sum operand requires complete scalar dimension_values."
            )
        dimensions.append(values)
    if not dimensions or any(set(value) != set(dimensions[0]) for value in dimensions):
        raise ValueError(
            "A monthly window sum requires operands with the same complete dimension keys."
        )
    encodings = [
        json.dumps(value, sort_keys=True, separators=(",", ":")) for value in dimensions
    ]
    if len(set(encodings)) != len(encodings):
        raise ValueError("A monthly window sum requires disjoint operand tuples.")
    return tuple(dimensions)


def _monthly_window_source_key(
    fact: object, dimensions: Mapping[str, object]
) -> tuple[str, ...]:
    """All window members must share a publication and statistical measure."""
    return (
        _source_name(fact),
        _str_at(fact, "source_release_key"),
        _str_at(fact, "source", "source_sha256"),
        _str_at(fact, "source", "vintage"),
        _str_at(fact, "source", "source_file"),
        _str_at(fact, "source", "source_table")
        or _str_at(fact, "observed_measure", "source_table"),
        _str_at(fact, "observed_measure", "source_measure_id")
        or _str_at(fact, "layout", "measure_id"),
        _source_measure_concept(fact),
        _str_at(fact, "observed_measure", "unit"),
        json.dumps(_at(fact, "entity"), sort_keys=True),
        json.dumps(_at(fact, "geography"), sort_keys=True),
        _normalized_record_set_id(_str_at(fact, "layout", "record_set_id")),
        _normalized_record_set_id(_str_at(fact, "layout", "record_set_spec_id")),
        _str_at(fact, "layout", "groupby_dimension"),
        _str_at(fact, "aggregation", "method"),
        _domain(fact),
        json.dumps(
            [
                constraint
                for constraint in _constraint_rows(fact)
                if _str_at(constraint, "variable") not in dimensions
            ],
            sort_keys=True,
        ),
    )


def _resolve_monthly_window_reference_facts(
    reference: LedgerTargetReference,
    matches: list[object],
) -> tuple[object, ...]:
    """Require a complete month-by-cell grid before averaging monthly totals.

    The same guard runs after direct-key resolution, so identifiers cannot
    bypass the explicit window, selector, or operand coverage contract.
    """
    months = _declared_source_months(reference)
    operands = _monthly_window_operands(reference)
    cell_count = len(operands) or 1
    expected_count = len(months) * cell_count
    prefix = f"Ledger target reference {reference.name!r}: monthly window"
    if reference.expected_member_count not in (None, expected_count):
        raise ValueError(
            f"{prefix} expected_member_count disagrees with its declared month-by-cell grid."
        )
    if len(matches) != expected_count or any(
        not _fact_matches_selector(fact, reference.ledger_selector) for fact in matches
    ):
        raise ValueError(
            f"{prefix} requires exactly {expected_count} selected facts, one per declared month and cell; resolved {len(matches)}."
        )
    grid: dict[tuple[str, int], object] = {}
    partitions: dict[int, set[tuple[str, ...]]] = {}
    for fact in matches:
        publication = _at(fact, "source_release_key")
        raw_hash = _at(fact, "source", "source_sha256")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (publication, raw_hash)
        ):
            raise ValueError(
                f"{prefix} requires nonempty source_release_key and "
                "source.source_sha256 on every member."
            )
        if operands:
            cells = [
                index
                for index, values in enumerate(operands)
                if set(_dimensions(fact)) == set(values)
                and _dimension_values_match(fact, values)
            ]
        else:
            cells = [0]
        if len(cells) != 1:
            raise ValueError(
                f"{prefix} requires every fact to match exactly one complete operand tuple."
            )
        cell = cells[0]
        key = (_str_at(fact, "period", "value"), cell)
        if key in grid:
            raise ValueError(f"{prefix} contains a duplicate month/cell observation.")
        grid[key] = fact
        partitions.setdefault(cell, set()).add(_selector_period_invariant_key(fact))
    if any(len(series) != 1 for series in partitions.values()):
        raise ValueError(
            f"{prefix} matched multiple semantic series for a declared cell."
        )
    dimension_cells = operands[0] if operands else {}
    if (
        len({_monthly_window_source_key(fact, dimension_cells) for fact in matches})
        != 1
    ):
        raise ValueError(
            f"{prefix} cells must share source publication, measure, entity and geography."
        )
    expected_grid = [(month, cell) for month in months for cell in range(cell_count)]
    if set(grid) != set(expected_grid):
        raise ValueError(f"{prefix} is missing a declared month/cell observation.")
    member_keys = [_fact_key(fact) or _source_record_id(fact) for fact in matches]
    if not all(member_keys) or len(set(member_keys)) != len(member_keys):
        raise ValueError(
            f"{prefix} requires distinct non-empty member fact identities."
        )
    return tuple(grid[key] for key in expected_grid)


def _resolve_calendar_year_average_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    monthly_matches = _monthly_operation_matches(
        reference, eligible_matches, operation="calendar_year_average"
    )
    target_year = _calendar_year_from_reference(reference)
    year_matches = [
        fact for fact in monthly_matches if _fact_year_month(fact)[0] == target_year
    ]
    if not year_matches:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            "'calendar_year_average' found no monthly Ledger facts in calendar "
            f"year {target_year}."
        )
    return _latest_series_partition(
        reference, year_matches, operation="calendar_year_average"
    )


def _resolve_latest_plateau_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    monthly_matches = _monthly_operation_matches(
        reference, eligible_matches, operation="latest_plateau"
    )
    partition = _latest_series_partition(
        reference, monthly_matches, operation="latest_plateau"
    )
    sorted_partition = tuple(sorted(partition, key=_fact_month_sort_key))
    plateau_value = _numeric_fact_value(sorted_partition[-1], reference)
    plateau = [sorted_partition[-1]]
    expected_month = _fact_month_index(sorted_partition[-1]) - 1
    for fact in reversed(sorted_partition[:-1]):
        if _fact_month_index(fact) != expected_month:
            break
        if _numeric_fact_value(fact, reference) != plateau_value:
            break
        plateau.append(fact)
        expected_month -= 1
    return tuple(reversed(plateau))


def _resolve_count_x_mean_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    partitions: dict[
        tuple[tuple[str, ...], tuple[int, int, str]], dict[str, list[object]]
    ] = {}
    for fact in eligible_matches:
        role = _count_mean_fact_role(fact)
        if not role:
            continue
        key = (
            _selector_count_mean_partition_key(fact),
            _reference_period_partition_key(fact, reference),
        )
        partitions.setdefault(key, {}).setdefault(role, []).append(fact)

    valid = {
        key: roles
        for key, roles in partitions.items()
        if len(roles.get("count", ())) == 1 and len(roles.get("mean", ())) == 1
    }
    if not valid:
        observed = {
            key: {role: len(facts) for role, facts in roles.items()}
            for key, roles in partitions.items()
        }
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            "'count_x_mean' requires exactly one count fact and one mean fact "
            f"in a shared selector partition; observed {observed!r}."
        )
    latest_period = max(period_key for _, period_key in valid)
    latest = [
        roles for (_, period_key), roles in valid.items() if period_key == latest_period
    ]
    if len(latest) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            "'count_x_mean' matched multiple count/mean selector partitions "
            "at the latest eligible period."
        )
    roles = latest[0]
    return (roles["count"][0], roles["mean"][0])


def _monthly_operation_matches(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
    *,
    operation: str,
) -> list[object]:
    non_month = [
        _at(fact, "period", "value")
        for fact in eligible_matches
        if _str_at(fact, "period", "type") != "month"
    ]
    if non_month:
        examples = tuple(str(period) for period in non_month[:5])
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            f"{operation!r} requires monthly Ledger facts; non-month period "
            f"values include {examples!r}."
        )
    return eligible_matches


def _calendar_year_from_reference(reference: LedgerTargetReference) -> int:
    period_key = _period_key_from_value(reference.period)
    if not period_key[0]:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            "'calendar_year_average' requires a parseable target period."
        )
    return period_key[1] // 100


def _latest_series_partition(
    reference: LedgerTargetReference,
    matches: list[object],
    *,
    operation: str,
) -> tuple[object, ...]:
    partitions: dict[tuple[str, ...], list[object]] = {}
    for fact in matches:
        partitions.setdefault(_selector_period_invariant_key(fact), []).append(fact)
    if not partitions:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            f"{operation!r} did not match a Ledger fact partition."
        )
    latest_period = max(_period_key(fact) for fact in matches)
    latest_partitions = [
        partition
        for partition in partitions.values()
        if max(_period_key(fact) for fact in partition) == latest_period
    ]
    if len(latest_partitions) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            f"{operation!r} matched multiple monthly Ledger series identities "
            "at the latest eligible period."
        )
    return tuple(sorted(latest_partitions[0], key=_fact_month_sort_key))


def _fact_month_sort_key(fact: object) -> tuple[int, str]:
    return (_fact_month_index(fact), _fact_key(fact) or _source_record_id(fact) or "")


def _fact_month_index(fact: object) -> int:
    year, month = _fact_year_month(fact)
    return year * 12 + month


def _fact_year_month(fact: object) -> tuple[int, int]:
    period_key = _period_key(fact)
    if not period_key[0]:
        raise ValueError("Ledger monthly period is not parseable.")
    year_month = period_key[1]
    year = year_month // 100
    month = year_month % 100
    if month < 1 or month > 12:
        raise ValueError("Ledger monthly period does not carry a month value.")
    return year, month


def _selector_sum_partition_key(fact: object) -> tuple[str, ...]:
    invariant = _selector_period_invariant_key(fact)
    return invariant[:8] + invariant[11:]


def _selector_count_mean_partition_key(fact: object) -> tuple[str, ...]:
    return (
        _source_name(fact),
        _str_at(fact, "geography", "level"),
        _str_at(fact, "geography", "id"),
        _str_at(fact, "entity", "name"),
        _normalized_record_set_id(_str_at(fact, "layout", "record_set_id")),
        _str_at(fact, "layout", "record_set_spec_id"),
        _str_at(fact, "layout", "groupby_dimension"),
        _normalized_period_bearing_id(_str_at(fact, "layout", "groupby_value_id")),
        json.dumps(_dimensions(fact), sort_keys=True, separators=(",", ":")),
        json.dumps(_constraint_rows(fact), sort_keys=True, separators=(",", ":")),
        _domain(fact),
    )


def _count_mean_fact_role(fact: object) -> str:
    measure_id = (
        _str_at(fact, "observed_measure", "source_measure_id")
        or _str_at(fact, "layout", "measure_id")
    ).lower()
    if measure_id.endswith("_count") or measure_id == "count":
        return "count"
    if measure_id.endswith("_mean") or measure_id == "mean":
        return "mean"
    return ""


def _eligible_selector_matches(
    reference: LedgerTargetReference,
    matches: list[object],
) -> list[object]:
    target_period_key = _period_key_from_value(reference.period)
    if reference.period_match_policy == "source_window":
        return [
            fact
            for fact in matches
            if _fact_matches_selector(fact, reference.ledger_selector)
            and _assertion_allowed(reference, fact)
        ]
    if reference.period_match_policy == "exact":
        return [
            fact
            for fact in matches
            if _exact_period_matches(fact, reference)
            and _assertion_allowed(reference, fact)
        ]
    return [
        fact
        for fact in matches
        if _not_after_target_period(_period_key(fact), target_period_key)
        and _assertion_allowed(reference, fact)
    ]


def _assertion_allowed(reference: LedgerTargetReference, fact: object) -> bool:
    assertion = _fact_assertion(fact)
    if assertion == "observation":
        return True
    if assertion == "source_projection":
        return reference.assertion_policy == "allow_source_projection"
    return False


def _validate_resolved_reference_fact(
    fact: object,
    reference: LedgerTargetReference,
) -> None:
    """Apply consumer policy after either identifier or selector resolution.

    Selector filtering still uses these predicates to choose an eligible row
    from a series.  This post-resolution check is the authority: exact keys and
    source-record identifiers must not bypass assertion, period, or declared
    geography-vintage policy.
    """

    _validate_reference_period(fact, reference)
    if not _assertion_allowed(reference, fact):
        raise ValueError(
            f"Ledger target reference {reference.name!r} assertion_policy="
            f"{reference.assertion_policy!r} does not allow resolved fact "
            f"assertion {_fact_assertion(fact)!r}."
        )
    vintage_pin = reference.ledger_selector.get("geography_vintage")
    if vintage_pin is not None and vintage_pin != "":
        if not _str_at(fact, "geography", "vintage") or not _fact_matches_selector(
            fact, {"geography_vintage": vintage_pin}
        ):
            raise ValueError(
                f"Ledger target reference {reference.name!r} requires geography "
                f"vintage {vintage_pin!r}, but resolved fact has vintage "
                f"{_at(fact, 'geography', 'vintage')!r}."
            )
    entity_pin = reference.ledger_selector.get("entity_name")
    if entity_pin is not None and _str_at(fact, "entity", "name") != entity_pin:
        raise ValueError(
            f"Ledger target reference {reference.name!r} requires entity_name "
            f"{entity_pin!r}, but resolved fact has entity "
            f"{_at(fact, 'entity', 'name')!r}."
        )


def _validate_reference_period(fact: object, reference: LedgerTargetReference) -> None:
    """Enforce the declared period consumer contract after either resolution path.

    An exact-period reference may consume either an observation at that period
    or a Ledger ``source_projection`` whose published fact period is that same
    target period. It may not silently substitute an older observation. The
    projection's source period belongs in Ledger lineage; the consumer still
    binds the projected fact to the period it estimates.
    """

    if reference.period_match_policy == "source_window":
        if _str_at(fact, "period", "type") != "month" or _str_at(
            fact, "period", "value"
        ) not in _declared_source_months(reference):
            raise ValueError("Resolved fact lies outside the declared monthly window.")
        return
    if reference.period_match_policy != "exact":
        if not _not_after_target_period(
            _period_key(fact), _period_key_from_value(reference.period)
        ):
            raise ValueError(
                f"Ledger target reference {reference.name!r} requires a fact "
                f"at or before target period {reference.period!r}, but resolved "
                f"fact period {_at(fact, 'period', 'value')!r}."
            )
        return
    if not _exact_period_matches(fact, reference):
        raise ValueError(
            f"Ledger target reference {reference.name!r} requires exact period "
            f"{reference.period!r}, but resolved fact period "
            f"{_at(fact, 'period', 'type')!r}:"
            f"{_at(fact, 'period', 'value')!r}. A period mismatch must resolve "
            "through a Ledger source_projection at the target period, never "
            "through a silently stale observation."
        )


def _fact_assertion(fact: object) -> str:
    return _str_at(fact, "assertion") or "observation"


def _latest_period_selector_match(
    reference: LedgerTargetReference,
    matches: list[object],
) -> object | None:
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    semantic_keys = {_selector_period_invariant_key(fact) for fact in matches}
    if len(semantic_keys) != 1:
        return None

    target_period_key = _period_key_from_value(reference.period)
    best: tuple[int, int, str] | None = None
    best_matches: list[object] = []
    for fact in matches:
        period_key = _period_key(fact)
        if best is None or _prefer_period_candidate(
            period_key,
            best,
            target_period_key=target_period_key,
        ):
            best = period_key
            best_matches = [fact]
        elif period_key == best:
            best_matches.append(fact)
    if len(best_matches) == 1:
        return best_matches[0]
    return None


def _selector_period_invariant_key(fact: object) -> tuple[str, ...]:
    return (
        _source_name(fact),
        _str_at(fact, "observed_measure", "source_measure_id")
        or _str_at(fact, "layout", "measure_id"),
        _source_measure_concept(fact),
        _str_at(fact, "geography", "level"),
        _str_at(fact, "geography", "id"),
        _str_at(fact, "entity", "name"),
        _str_at(fact, "aggregation", "method"),
        _normalized_record_set_id(_str_at(fact, "layout", "record_set_id")),
        _str_at(fact, "layout", "groupby_dimension"),
        _normalized_period_bearing_id(_str_at(fact, "layout", "groupby_value_id")),
        json.dumps(_dimensions(fact), sort_keys=True, separators=(",", ":")),
        json.dumps(_constraint_rows(fact), sort_keys=True, separators=(",", ":")),
        _domain(fact),
    )


def _normalized_record_set_id(record_set_id: str) -> str:
    if not record_set_id:
        return ""
    return ".".join(
        normalized
        for part in record_set_id.split(".")
        if (normalized := _normalized_record_set_part(part))
    )


def _normalized_record_set_part(value: str) -> str:
    if _is_period_token(value):
        return ""
    normalized = value.lower().replace("-", "_")
    pieces = [
        piece
        for piece in normalized.split("_")
        if piece and not _is_period_fragment(piece)
    ]
    return "_".join(pieces)


def _normalized_period_bearing_id(value: str) -> str:
    if _is_period_token(value):
        return ""
    return _normalized_record_set_part(value)


def _is_period_fragment(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    if _is_period_token(normalized):
        return True
    if normalized.startswith("year") and _is_period_token(normalized[4:]):
        return True
    month_names = (
        "jan",
        "january",
        "feb",
        "february",
        "mar",
        "march",
        "apr",
        "april",
        "may",
        "jun",
        "june",
        "jul",
        "july",
        "aug",
        "august",
        "sep",
        "sept",
        "september",
        "oct",
        "october",
        "nov",
        "november",
        "dec",
        "december",
    )
    return any(
        normalized.startswith(month)
        and normalized[len(month) :].isdigit()
        and len(normalized[len(month) :]) == 4
        for month in month_names
    )


def _is_period_token(value: str) -> bool:
    period_key = _period_key_from_value(value)
    # Source-table numbers such as table_1_2 are identity, not year tokens.
    return bool(period_key[0]) and 1000 <= period_key[1] // 100 <= 9999


def _comparable_period_value(fact: object) -> object:
    """The period value the surface compares a fact on.

    Every fiscal-year label on the surface is read as the opening year
    (OBR's and HMRC's ``2025`` is FY2025-26). DfT labels its reporting year
    by the March end year (``2025`` is YE March 2025, FY2024-25) and says so
    in the fact's own coverage dates, so a fiscal-year fact that carries
    ``period_coverage.start_date`` is keyed on that start year. Facts without
    coverage keep their label; the publisher's label is recorded beside the
    comparable value (``ledger_fact_period_label``) whenever they differ.
    """

    value = _at(fact, "period", "value")
    if _str_at(fact, "period", "type") != "fiscal_year":
        return value
    start = _str_at(fact, "period_coverage", "start_date")
    if len(start) < 4 or not start[:4].isdigit():
        return value
    label = str(value).strip()
    if not label.isdigit():
        return value
    return int(start[:4])


def _period_key(fact: object) -> tuple[int, int, str]:
    return _period_key_from_value(_comparable_period_value(fact))


def _exact_period_matches(fact: object, reference: LedgerTargetReference) -> bool:
    """Match exact periods by semantic value while retaining period-kind pins."""

    expected_value = reference.period
    actual_value = _comparable_period_value(fact)
    actual_type = _str_at(fact, "period", "type")
    selector_type = str(reference.ledger_selector.get("period_type", ""))
    expected_type_hint = period_type_hint(expected_value)
    if selector_type and expected_type_hint and selector_type != expected_type_hint:
        return False
    expected_type = selector_type or expected_type_hint
    if expected_type and actual_type != expected_type:
        return False
    actual_type_hint = period_type_hint(actual_value)
    if actual_type_hint and actual_type_hint != actual_type:
        return False
    return period_values_semantically_equal(
        expected_value,
        actual_value,
        declared_type=expected_type or actual_type,
    )


def _reference_period_partition_key(
    fact: object,
    reference: LedgerTargetReference,
) -> tuple[int, int, str]:
    period_key = (
        _normalize_period_value(
            _comparable_period_value(fact),
            declared_type=_str_at(fact, "period", "type"),
        )[0]
        if reference.period_match_policy == "exact"
        else _period_key(fact)
    )
    if reference.period_match_policy == "exact" and period_key[0]:
        return (period_key[0], period_key[1], "")
    return period_key


def period_values_semantically_equal(
    left: object, right: object, *, declared_type: str = ""
) -> bool:
    """Compare valid period spellings, using the declared type for untyped values.

    Opaque publisher labels retain literal equality. Malformed numeric period
    shapes do not, even when the same invalid spelling appears on both sides.
    """

    left_key, _, left_range_end, left_malformed = _normalize_period_value(
        left, declared_type=declared_type
    )
    right_key, _, right_range_end, right_malformed = _normalize_period_value(
        right, declared_type=declared_type
    )
    if left_malformed or right_malformed:
        return False
    if left_key[0] and right_key[0]:
        return left_key[:2] == right_key[:2] and left_range_end == right_range_end
    return left_key == right_key


def period_type_hint(value: object) -> str:
    """Return the period-kind prefix recognized by the shared normalizer."""

    return _normalize_period_value(value)[1]


def _period_key_from_value(value: object) -> tuple[int, int, str]:
    return _normalize_period_value(value)[0]


def _normalize_period_value(
    value: object,
    *,
    declared_type: str = "",
) -> tuple[tuple[int, int, str], str, int | None, bool]:
    """Normalize one scalar/typed period spelling once for all consumers.

    Annual aliases (``ty``, ``cy``, ``fy``, and ``ay``) and their long forms
    share one parser.  A two-part value under an annual prefix is an annual
    range (for example ``academic_year_2023_24``). A declared annual type also
    disambiguates untyped ``2003_04`` as a range rather than a monthly point.
    Annual ranges must end in the following year. Keep that explicit end year
    in the semantic identity: a range is not interchangeable with a scalar year.
    The final flag distinguishes malformed numeric shapes from opaque labels,
    so only the latter can fall back to literal equality.
    """

    label = "" if value is None else str(value)
    normalized = label.strip().lower().replace("-", "_")
    type_hint = ""
    long_prefixes = (
        ("tax_year", "tax_year"),
        ("calendar_year", "calendar_year"),
        ("fiscal_year", "fiscal_year"),
        ("academic_year", "academic_year"),
        ("month", "month"),
    )
    for prefix, period_type in long_prefixes:
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :].lstrip("_")
            if suffix[:1].isdigit():
                normalized = suffix
                type_hint = period_type
            break
    if not type_hint:
        aliases = {
            "ty": "tax_year",
            "cy": "calendar_year",
            "fy": "fiscal_year",
            "ay": "academic_year",
        }
        alias = normalized[:2]
        suffix = normalized[2:].lstrip("_")
        if alias in aliases and suffix[:1].isdigit():
            normalized = suffix
            type_hint = aliases[alias]

    annual_types = {"tax_year", "calendar_year", "fiscal_year", "academic_year"}
    if (
        not type_hint
        and declared_type in annual_types | {"month"}
        and normalized[:1].isdigit()
    ):
        type_hint = declared_type

    parts = normalized.split("_", maxsplit=1)
    if len(parts) == 2:
        year, suffix = parts
        if year.isdigit() and suffix.isdigit() and len(suffix) in {1, 2, 4}:
            annual_range = type_hint in annual_types or (
                not type_hint and (len(suffix) == 4 or int(suffix) > 12)
            )
            if annual_range:
                range_end = int(year) + 1
                expected_suffix = range_end if len(suffix) == 4 else range_end % 100
                if len(suffix) not in {2, 4} or int(suffix) != expected_suffix:
                    return (0, 0, label), type_hint, None, True
                return (1, int(year) * 100 + 99, label), type_hint, range_end, False
            if len(suffix) <= 2 and 1 <= int(suffix) <= 12:
                return (
                    (1, int(year) * 100 + int(suffix), label),
                    type_hint,
                    None,
                    False,
                )
        numeric_shape = year.isdigit() and all(
            part.isdigit() or not part for part in suffix.split("_")
        )
        return (0, 0, label), type_hint, None, bool(type_hint) or numeric_shape
    if type_hint == "month":
        return (0, 0, label), type_hint, None, True
    try:
        return (1, int(normalized) * 100 + 99, label), type_hint, None, False
    except ValueError:
        return (0, 0, label), type_hint, None, bool(type_hint)


def _prefer_period_candidate(
    candidate: tuple[int, int, str],
    current: tuple[int, int, str],
    *,
    target_period_key: tuple[int, int, str],
) -> bool:
    candidate_is_eligible = _not_after_target_period(candidate, target_period_key)
    current_is_eligible = _not_after_target_period(current, target_period_key)
    if candidate_is_eligible != current_is_eligible:
        return candidate_is_eligible
    return candidate > current


def _not_after_target_period(
    source_period_key: tuple[int, int, str],
    target_period_key: tuple[int, int, str],
) -> bool:
    if not source_period_key[0]:
        return False
    if not target_period_key[0]:
        return True
    return source_period_key[1] <= target_period_key[1]


def _reference_metadata(reference: LedgerTargetReference) -> dict[str, str]:
    metadata = dict(reference.metadata)
    metadata["ledger_value_operation"] = reference.value_operation
    metadata["ledger_assertion_policy"] = reference.assertion_policy
    metadata["ledger_period_match_policy"] = reference.period_match_policy
    if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
        months = _declared_source_months(reference)
        metadata["ledger_source_months"] = json.dumps(months, separators=(",", ":"))
        metadata["ledger_source_month_count"] = str(len(months))
        metadata["ledger_source_cell_count_per_month"] = str(
            len(reference.value_operands) or 1
        )
        metadata["ledger_value_formula"] = (
            "sum_of_declared_cells / declared_month_count"
        )
    if reference.value_operation == "difference":
        metadata["ledger_value_formula"] = "minuend - subtrahend"
    for key, value in sorted(reference.ledger_selector.items()):
        if isinstance(value, Mapping):
            continue
        if value is not None and value != "":
            metadata[f"ledger_selector_{key}"] = str(value)
    if reference.uprating_index is not None:
        metadata["uprating_index"] = str(reference.uprating_index)
    if reference.uprating_from_period is not None:
        metadata["uprating_from_period"] = str(reference.uprating_from_period)
    if reference.uprating_to_period is not None:
        metadata["uprating_to_period"] = str(reference.uprating_to_period)
    return metadata


def _at(obj: object, *path: str) -> Any:
    current = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _jsonl_rows(lines: Iterable[str]) -> Iterable[object]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid Ledger JSONL row {line_number}: {exc.msg}"
            ) from exc


def _str_at(obj: object, *path: str) -> str:
    value = _at(obj, *path)
    if value is None:
        return ""
    return str(value)


def _fact_key(fact: object) -> str:
    return (
        _str_at(fact, "aggregate_fact_key")
        or _str_at(fact, "fact_key")
        or _str_at(fact, "legacy_fact_key")
    )


def _source_record_id(fact: object) -> str:
    return _str_at(fact, "source_record_id") or _str_at(
        fact, "lineage", "source_record_id"
    )


def _fact_matches_selector(fact: object, selector: Mapping[str, object]) -> bool:
    """Return whether a consumer fact satisfies a structured reference selector.

    A list-valued scalar key matches by membership; an empty list is refused
    rather than treated as match-nothing, because it reads like match-anything.
    Note ``dimensions: []`` is NOT an empty membership list — it is the
    list-form dimensions selector's exact name-set match for the dimensionless
    total row (see :func:`_dimensions_match`).
    """

    for key, expected in selector.items():
        if key == "dimensions":
            if not _dimensions_match(fact, expected):
                return False
            continue
        if key == "dimension_values":
            if not _dimension_values_match(fact, expected):
                return False
            continue
        if expected is None or expected == "":
            continue
        candidates = _selector_candidates(fact, str(key))
        if isinstance(expected, (list, tuple)):
            if not expected:
                raise ValueError(
                    f"Ledger fact selector field {key!r} is an empty list: an "
                    "empty membership list matches nothing while reading like "
                    "match-anything. Pin at least one value or drop the field."
                )
            if not any(str(item) in candidates for item in expected):
                return False
            continue
        if str(expected) not in candidates:
            return False
    return True


def _selector_candidates(fact: object, key: str) -> tuple[str, ...]:
    if key == "aggregate_fact_key":
        return (_str_at(fact, "aggregate_fact_key"),)
    if key == "semantic_fact_key":
        return (_str_at(fact, "semantic_fact_key"),)
    if key == "legacy_fact_key":
        return (_str_at(fact, "legacy_fact_key"),)
    if key == "source_record_id":
        return (_source_record_id(fact),)
    if key == "source_name":
        return (_source_name(fact),)
    if key == "source_table":
        return (
            _str_at(fact, "source", "source_table"),
            _str_at(fact, "observed_measure", "source_table"),
        )
    if key == "source_measure_id":
        return (
            _str_at(fact, "observed_measure", "source_measure_id"),
            _str_at(fact, "layout", "measure_id"),
        )
    if key == "source_concept":
        return (
            _source_measure_concept(fact),
            _primary_measure_concept(fact),
        )
    if key == "period_type":
        return (_str_at(fact, "period", "type"),)
    if key == "period_value":
        return (_str_at(fact, "period", "value"),)
    if key == "geography_level":
        return (_str_at(fact, "geography", "level"),)
    if key == "geography_id":
        return (_str_at(fact, "geography", "id"),)
    if key == "geography_vintage":
        return (_str_at(fact, "geography", "vintage"),)
    if key == "entity_name":
        return (_str_at(fact, "entity", "name"),)
    if key in {"record_set_id", "layout_record_set_id"}:
        return (_str_at(fact, "layout", "record_set_id"),)
    if key in {"record_set_spec_id", "layout_record_set_spec_id"}:
        return (_str_at(fact, "layout", "record_set_spec_id"),)
    if key in {"groupby_dimension", "layout_groupby_dimension"}:
        return (_str_at(fact, "layout", "groupby_dimension"),)
    if key == "layout_groupby_value_id":
        return (_str_at(fact, "layout", "groupby_value_id"),)
    if key == "layout_measure_id":
        return (_str_at(fact, "layout", "measure_id"),)
    if key == "domain":
        return (_domain(fact),)
    if key == "assertion":
        return (_fact_assertion(fact),)
    raise ValueError(f"Unsupported Ledger fact selector field {key!r}.")


def _dimensions_match(fact: object, expected: object) -> bool:
    if isinstance(expected, list):
        expected_names = {str(name) for name in expected}
        dimension_names = {str(key) for key in _dimensions(fact)}
        return dimension_names == expected_names
    if not isinstance(expected, Mapping):
        raise ValueError(
            "Ledger fact selector field 'dimensions' must be a mapping or list."
        )
    dimensions = {str(key): str(value) for key, value in _dimensions(fact).items()}
    return all(
        dimensions.get(str(key)) == str(value) for key, value in expected.items()
    )


def _dimension_values_match(fact: object, expected: object) -> bool:
    """Match dimension-value pins with strict typed equality.

    A pin whose type disagrees with the fact's dimension value (float ``5.0``
    against int ``5``, ``"5"`` against ``5``) does not error — the fact simply
    fails to match, so under a resolves-or-defers activation rule the target
    quietly drops out of the active subset. When a target is unexpectedly
    inactive, check the authoring-run membership report before suspecting the
    feed. Empty pin lists are refused for the same reason as empty membership
    lists in :func:`_fact_matches_selector`.
    """

    if not isinstance(expected, Mapping):
        raise ValueError(
            "Ledger fact selector field 'dimension_values' must be a mapping."
        )
    dimensions = _dimensions(fact)
    for key, expected_value in expected.items():
        if isinstance(expected_value, list) and not expected_value:
            raise ValueError(
                f"dimension_values pin {key!r} is an empty list: an empty pin "
                "list matches nothing while reading like match-anything. Pin "
                "at least one value or drop the dimension."
            )
        if key not in dimensions:
            return False
        actual_value = dimensions[key]
        if isinstance(expected_value, list):
            if not any(
                _strict_scalar_equal(actual_value, candidate)
                for candidate in expected_value
            ):
                return False
            continue
        if not _strict_scalar_equal(actual_value, expected_value):
            return False
    return True


def _strict_scalar_equal(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


def _measure_concepts(fact: object) -> tuple[str, ...]:
    concepts = (
        _str_at(fact, "measure", "concept"),
        _str_at(fact, "concept_alignment", "canonical_concept"),
        _str_at(fact, "observed_measure", "source_concept"),
        _str_at(fact, "measure", "source_concept"),
    )
    return tuple(dict.fromkeys(concept for concept in concepts if concept))


def _primary_measure_concept(fact: object) -> str:
    concepts = _measure_concepts(fact)
    return concepts[0] if concepts else ""


def _source_measure_concept(fact: object) -> str:
    return (
        _str_at(fact, "observed_measure", "source_concept")
        or _str_at(fact, "measure", "source_concept")
        or _primary_measure_concept(fact)
    )


def _source_name(fact: object) -> str:
    return _str_at(fact, "source", "source_name") or _str_at(
        fact, "observed_measure", "source_name"
    )


def _dimensions(fact: object) -> Mapping[str, object]:
    filters = _at(fact, "filters")
    if isinstance(filters, Mapping):
        return filters
    dimensions = _at(fact, "dimensions")
    if isinstance(dimensions, Mapping):
        return dimensions
    return {}


def _domain(fact: object) -> str:
    return _str_at(fact, "domain") or _str_at(fact, "universe_constraints", "domain")


def _constraint_rows(fact: object) -> tuple[object, ...]:
    constraints = _at(fact, "constraints")
    if isinstance(constraints, list | tuple):
        return tuple(constraints)
    universe_constraints = _at(fact, "universe_constraints", "constraints")
    if isinstance(universe_constraints, list | tuple):
        return tuple(universe_constraints)
    return ()


def _requires_filter_mapping(fact: object) -> bool:
    return _requires_detail_filter_mapping(fact) or _requires_domain_filter_mapping(
        fact
    )


def _requires_detail_filter_mapping(fact: object) -> bool:
    if _constraint_rows(fact):
        return True
    return any(
        not _is_unscoped_dimension_value(value) for value in _dimensions(fact).values()
    )


def _requires_domain_filter_mapping(fact: object) -> bool:
    return not _is_unscoped_domain(_domain(fact))


def _is_unscoped_domain(domain: str) -> bool:
    normalized = domain.strip().lower().replace("_", " ")
    return normalized in {"", "all", "all households", "all persons", "population"}


def _is_unscoped_dimension_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    normalized = str(value).strip().lower().replace("_", " ")
    return normalized in {"", "all", "all returns", "total", "totals", "overall"}


def _allows_signed_target(fact: object, mapping: LedgerTargetMapping) -> bool:
    source_record_id = _source_record_id(fact)
    if source_record_id in mapping.signed_by_source_record_id:
        return True
    return any(
        concept in mapping.signed_by_concept for concept in _measure_concepts(fact)
    )


def _source_citation(fact: object) -> str:
    parts = [
        _source_name(fact),
        _str_at(fact, "source", "source_table")
        or _str_at(fact, "observed_measure", "source_table"),
        _str_at(fact, "source", "source_file"),
        _str_at(fact, "source", "vintage"),
        _str_at(fact, "source", "url"),
    ]
    citation = " | ".join(part for part in parts if part)
    return citation or "PolicyEngine Ledger source-backed fact"


def _ledger_metadata(fact: object, *, fact_key: str) -> dict[str, str]:
    metadata = {
        "ledger_source": "policyengine-ledger-data",
        "ledger_fact_key": fact_key,
        "ledger_source_record_id": _source_record_id(fact),
        "ledger_aggregate_fact_key": _str_at(fact, "aggregate_fact_key"),
        "ledger_semantic_fact_key": _str_at(fact, "semantic_fact_key"),
        "ledger_legacy_fact_key": _str_at(fact, "legacy_fact_key"),
        "ledger_observed_measure_key": _str_at(fact, "observed_measure_key"),
        "ledger_dimension_set_key": _str_at(fact, "dimension_set_key"),
        "ledger_universe_constraint_set_key": _str_at(
            fact, "universe_constraint_set_key"
        ),
        "ledger_measure_concept": _primary_measure_concept(fact),
        "ledger_measure_unit": _str_at(fact, "measure", "unit")
        or _str_at(fact, "observed_measure", "unit"),
        "ledger_source_concept": _source_measure_concept(fact),
        "ledger_concept_relation": _str_at(fact, "measure", "concept_relation")
        or _str_at(fact, "concept_alignment", "relation"),
        "ledger_concept_authority": _str_at(fact, "measure", "concept_authority")
        or _str_at(fact, "concept_alignment", "authority"),
        "ledger_legal_vintage": _str_at(fact, "measure", "legal_vintage")
        or _str_at(fact, "concept_alignment", "legal_vintage"),
        "ledger_period_type": _str_at(fact, "period", "type"),
        "ledger_fact_period": str(_comparable_period_value(fact)),
        # Stamped only when the publisher's label differs from the comparable
        # period (DfT's March-end fiscal labels); absent otherwise.
        "ledger_fact_period_label": (
            _str_at(fact, "period", "value")
            if str(_comparable_period_value(fact)) != _str_at(fact, "period", "value")
            else ""
        ),
        # Recorded only when the fact asserts it; legacy rows that omit the
        # field are not stamped (readers treat absence as
        # observation-by-default, same as the artifact loader).
        "ledger_assertion": _str_at(fact, "assertion"),
        "ledger_geography_level": _str_at(fact, "geography", "level"),
        "ledger_geography_id": _str_at(fact, "geography", "id"),
        "ledger_geography_name": _str_at(fact, "geography", "name"),
        "ledger_geography_vintage": _str_at(fact, "geography", "vintage"),
        "ledger_entity_name": _str_at(fact, "entity", "name"),
        "ledger_entity_role": _str_at(fact, "entity", "role"),
        "ledger_domain": _domain(fact),
        "ledger_layout_record_set_id": _str_at(fact, "layout", "record_set_id"),
        "ledger_layout_record_set_spec_id": _str_at(
            fact, "layout", "record_set_spec_id"
        ),
        "ledger_layout_groupby_dimension": _str_at(fact, "layout", "groupby_dimension"),
        "ledger_layout_groupby_value_id": _str_at(fact, "layout", "groupby_value_id"),
        "ledger_layout_measure_id": _str_at(fact, "layout", "measure_id"),
        "ledger_aggregation_method": _str_at(fact, "aggregation", "method"),
    }
    constraint_rows = _constraint_rows(fact)
    if constraint_rows:
        metadata["ledger_universe_constraint_count"] = str(len(constraint_rows))
    for key, value in sorted(_dimensions(fact).items()):
        if value is not None:
            metadata[f"ledger_filter_{key}"] = str(value)
    return {key: value for key, value in metadata.items() if value}


def _hierarchy_rule_enabled(
    rule: LedgerTargetHierarchyRule,
    context: Mapping[str, object],
) -> bool:
    for key, expected in rule.enabled_when.items():
        if _profile_scalar(context.get(key)) != expected:
            return False
    return True


def _hierarchy_parent_key(
    spec: TargetSpec,
    rule: LedgerTargetHierarchyRule,
) -> tuple[str, ...]:
    return _hierarchy_match_key(spec, rule) + (
        spec.metadata.get("ledger_geography_id", ""),
    )


def _hierarchy_match_key(
    spec: TargetSpec,
    rule: LedgerTargetHierarchyRule,
) -> tuple[str, ...]:
    values: list[str] = []
    for field_name in rule.match_spec_fields:
        if not hasattr(spec, field_name):
            raise ValueError(
                f"Ledger target hierarchy rule {rule.rule_id!r} references "
                f"unknown TargetSpec field {field_name!r}."
            )
        values.append(_profile_scalar(getattr(spec, field_name)))
    for key in rule.match_metadata_keys:
        values.append(spec.metadata.get(key, ""))
    return tuple(values)


def _hierarchy_parent_geography_id(
    spec: TargetSpec,
    rule: LedgerTargetHierarchyRule,
) -> str:
    values = {
        **{key: _profile_scalar(value) for key, value in spec.metadata.items()},
        "name": spec.name,
        "entity": spec.entity,
        "period": _profile_scalar(spec.period),
        "family": spec.family,
        "filter": spec.filter or "",
    }
    try:
        return rule.parent_geography_id_template.format(**values)
    except KeyError as exc:
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} cannot expand "
            f"parent geography template {rule.parent_geography_id_template!r}; "
            f"missing metadata key {exc.args[0]!r} on target {spec.name!r}."
        ) from exc


def _hierarchy_completeness(
    parent: TargetSpec,
    children: list[tuple[int, TargetSpec]],
    rule: LedgerTargetHierarchyRule,
) -> dict[str, object] | None:
    if not rule.expected_child_count_by_parent_key:
        return None
    parent_key = parent.metadata.get(rule.parent_key_metadata_key, "")
    if not parent_key:
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} cannot check child "
            f"completeness for parent {parent.name!r}: missing parent metadata "
            f"{rule.parent_key_metadata_key!r}."
        )
    expected_child_count = rule.expected_child_count_by_parent_key.get(parent_key)
    if expected_child_count is None:
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} has no expected child "
            f"count for parent key {parent_key!r}."
        )
    child_id_values = tuple(
        spec.metadata.get(rule.child_id_metadata_key, "") for _, spec in children
    )
    if "" in child_id_values:
        missing = [
            spec.name
            for _, spec in children
            if not spec.metadata.get(rule.child_id_metadata_key, "")
        ]
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} cannot check child "
            f"completeness for parent {parent.name!r}: missing child metadata "
            f"{rule.child_id_metadata_key!r} on targets {missing!r}."
        )
    child_ids = tuple(sorted(set(child_id_values)))
    if len(child_ids) != len(child_id_values):
        duplicate_child_ids = sorted(
            child_id
            for child_id in set(child_id_values)
            if child_id_values.count(child_id) > 1
        )
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} found duplicate "
            f"child ids for parent {parent.name!r}: {duplicate_child_ids!r}."
        )
    observed_child_count = len(child_ids)
    return {
        "parent_key": parent_key,
        "expected_child_count": expected_child_count,
        "observed_child_count": observed_child_count,
        "child_ids": ",".join(child_ids),
        "complete": observed_child_count == expected_child_count,
    }


def _hierarchy_completeness_metadata(
    completeness: Mapping[str, object] | None,
) -> dict[str, str]:
    if completeness is None:
        return {}
    return {
        "hierarchy_parent_key": _profile_scalar(completeness["parent_key"]),
        "hierarchy_expected_child_count": _profile_scalar(
            completeness["expected_child_count"]
        ),
        "hierarchy_observed_child_count": _profile_scalar(
            completeness["observed_child_count"]
        ),
        "hierarchy_child_ids": _profile_scalar(completeness["child_ids"]),
    }


def _profile_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _format_float(value: float) -> str:
    return f"{value:.15g}"


def _registry_digest(
    registry: TargetRegistry,
    *,
    payload_fn: Any = asdict,
) -> str:
    payload = {
        "country": registry.country,
        "specs": [payload_fn(spec) for spec in registry.specs],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _calibration_effective_spec_payload(spec: TargetSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "entity": spec.entity,
        "measure": spec.measure,
        "value": spec.value,
        "filter": spec.filter,
        "period": spec.period,
        "tolerance": spec.tolerance,
    }


def _first_spec_differences(
    expected: TargetRegistry,
    actual: TargetRegistry,
    *,
    limit: int = 5,
) -> tuple[str, ...]:
    expected_by_key = {spec.key: asdict(spec) for spec in expected.specs}
    actual_by_key = {spec.key: asdict(spec) for spec in actual.specs}
    failures: list[str] = []
    missing = sorted(set(expected_by_key) - set(actual_by_key))
    extra = sorted(set(actual_by_key) - set(expected_by_key))
    if missing:
        failures.append(f"missing target keys: {missing[:limit]!r}.")
    if extra:
        failures.append(f"extra target keys: {extra[:limit]!r}.")
    for key in sorted(set(expected_by_key) & set(actual_by_key)):
        if expected_by_key[key] != actual_by_key[key]:
            failures.append(f"target {key!r} differs.")
        if len(failures) >= limit:
            break
    return tuple(failures)
