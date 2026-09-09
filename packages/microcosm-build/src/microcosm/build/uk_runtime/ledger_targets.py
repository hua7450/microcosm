"""UK Ledger target-reference compilation and materialization helpers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources as importlib_resources
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.cross_grain import (
    CrossGrainBridge,
    CrossGrainRule,
    apply_cross_grain_reconciliation,
)
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    _fact_matches_selector,
    compile_ledger_target_references,
)
from microcosm.build.target_materialization import (
    TargetMaterializationResult,
    materialize_target_bindings,
)
from microcosm.build.uk_runtime.cgt_calibration import uk_cgt_annual_exempt_amount
from microcosm.build.uk_runtime.ladder_targets import (
    constituency_household_targets,
    local_authority_household_targets,
)
from microcosm.build.uk_runtime.local_target_census import family_for_metric
from microcosm.build.uk_runtime.local_targets import (
    AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL,
    area_groups_from_codes,
    load_uk_local_geography_contract,
    metric_names,
)
from microcosm.build.uk_runtime.uc_source_periods import (
    validate_uc_source_month_coverage,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import Frame


@dataclass(frozen=True)
class UKLedgerTargetCompilation:
    """Compiled UK Ledger target registry and unsupported-reference report."""

    registry: TargetRegistry
    unsupported: tuple[dict[str, str], ...]


LOCAL_REGISTRY_PARITY_FIXTURE_RESOURCE = "local_registry_parity_fixture_2025.json"
UK_LOCAL_TARGET_REFERENCE_MEMBERSHIP_RESOURCE = "local_target_reference_membership.json"
UK_POPULATION_TARGETS_RESOURCE = "uk_population_targets.json"
UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS = frozenset({"country", "region"})
_UK_LOCAL_FIXTURE_METRIC_ALIASES = {
    f"voa/council_tax/{band}": f"council_tax/band_{band.lower()}" for band in "ABCDEFGH"
}


def _uk_cross_grain_leg_of_area(area_code: str) -> str:
    # area_groups_from_codes maps code -> country group, so the single value is
    # this code's leg. It refuses an unknown prefix itself; the explicit miss
    # below keeps the refusal fail-closed rather than a bare StopIteration if
    # that mapping ever returns nothing for a code.
    leg = next(iter(area_groups_from_codes((area_code,)).values()), "")
    if not leg:
        raise ValueError(
            f"UK cross-grain area code {area_code!r} maps to no country leg."
        )
    return leg


UK_CROSS_GRAIN_GRAIN_PRECEDENCE = ("country", "constituency", "la")
UK_CROSS_GRAIN_BRIDGES = (
    CrossGrainBridge(
        bridge_id="national_household_composition_partition_vs_census_households",
        concept="uk.household.count",
        higher_target_ids=(
            "ons.household_composition.lone_households_under_65",
            "ons.household_composition.lone_households_over_65",
            "ons.household_composition.unrelated_adult_households",
            "ons.household_composition.couple_no_children_households",
            "ons.household_composition.couple_under_3_children_households",
            "ons.household_composition.couple_3_plus_children_households",
            "ons.household_composition.couple_non_dependent_children_only_households",
            "ons.household_composition.lone_parent_dependent_children_households",
            "ons.household_composition.lone_parent_non_dependent_children_households",
            "ons.household_composition.multi_family_households",
        ),
        lower_side="external:census_households/households",
    ),
    CrossGrainBridge(
        bridge_id="national_uc_caseload_vs_uc_households_by_area",
        concept="uk.benefit_unit.count",
        higher_target_ids=("dwp.uc.households",),
        lower_side="contract:dwp.uc.households_by_area",
    ),
    # The national ONS controls use inclusive integer-age bands (0--9), while
    # local targets use equivalent half-open encodings (0--10), so their
    # signatures cannot match directly. These bridges let the K02000001 UK
    # control rescale both constituency and local-authority bands over its
    # England/Wales/Scotland/Northern Ireland legs.
    CrossGrainBridge(
        bridge_id="national_age_0_9_vs_local_age_0_10",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_0_9_by_region",),
        lower_side="contract:ons.age.0_10",
    ),
    CrossGrainBridge(
        bridge_id="national_age_10_19_vs_local_age_10_20",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_10_19_by_region",),
        lower_side="contract:ons.age.10_20",
    ),
    CrossGrainBridge(
        bridge_id="national_age_20_29_vs_local_age_20_30",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_20_29_by_region",),
        lower_side="contract:ons.age.20_30",
    ),
    CrossGrainBridge(
        bridge_id="national_age_30_39_vs_local_age_30_40",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_30_39_by_region",),
        lower_side="contract:ons.age.30_40",
    ),
    CrossGrainBridge(
        bridge_id="national_age_40_49_vs_local_age_40_50",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_40_49_by_region",),
        lower_side="contract:ons.age.40_50",
    ),
    CrossGrainBridge(
        bridge_id="national_age_50_59_vs_local_age_50_60",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_50_59_by_region",),
        lower_side="contract:ons.age.50_60",
    ),
    CrossGrainBridge(
        bridge_id="national_age_60_69_vs_local_age_60_70",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_60_69_by_region",),
        lower_side="contract:ons.age.60_70",
    ),
    CrossGrainBridge(
        bridge_id="national_age_70_79_vs_local_age_70_80",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_70_79_by_region",),
        lower_side="contract:ons.age.70_80",
    ),
)
# A future move of these declarations into country-package spec JSON follows
# the country-owned specification direction established in microcosm#159.
UK_CROSS_GRAIN_RULE = CrossGrainRule(
    grain_precedence=UK_CROSS_GRAIN_GRAIN_PRECEDENCE,
    signature_fields=("concept", "entity", "map_to", "filters"),
    bridges=UK_CROSS_GRAIN_BRIDGES,
    leg_of_area=_uk_cross_grain_leg_of_area,
    parent_geography_legs={
        "K02000001": ("England", "Wales", "Scotland", "Northern Ireland"),
        "K03000001": ("England", "Wales", "Scotland"),
        "E92000001": ("England",),
        "W92000004": ("Wales",),
        "S92000003": ("Scotland",),
        "N92000002": ("Northern Ireland",),
    },
)


def align_uk_local_registry_parity_fixture(
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    """Align incumbent local metric names to the Microcosm contract ids.

    The incumbent fixture is extracted from the old runtime's metric columns
    (``hmrc/self_employment_income/amount@...``). The local Ledger registry is
    contract-named (``hmrc.self_employment_income.amount@...``). In-contract
    fixture rows are therefore renamed for comparison; out-of-contract metrics
    deliberately remain fixture-only so the signed receipt records the ruled
    exclusions.
    """

    metric_target_ids = _uk_local_metric_target_ids()
    rows: list[dict[str, Any]] = []
    for row in fixture.get("rows", ()):
        if not isinstance(row, Mapping):
            rows.append(row)
            continue
        updated = dict(row)
        metric = str(
            updated.get("metric") or str(updated.get("name", "")).split("@")[0]
        )
        contract_metric = _UK_LOCAL_FIXTURE_METRIC_ALIASES.get(metric, metric)
        target_id = metric_target_ids.get(contract_metric)
        if target_id is not None:
            geography_id = str(
                updated.get("geography_id")
                or str(updated.get("name", "")).split("@", 1)[1]
            )
            updated["name"] = f"{target_id}@{geography_id}"
            updated["contract_target_id"] = updated["name"]
            updated.setdefault("measure", metric)
        rows.append(updated)
    aligned = dict(fixture)
    aligned["rows"] = rows
    return aligned


def _uk_local_metric_target_ids() -> dict[str, str]:
    contract = load_uk_local_geography_contract()
    mapping: dict[str, str] = {}
    for target in contract.get("targets", ()):
        if not isinstance(target, Mapping):
            continue
        bindings = target.get("bindings")
        if not isinstance(bindings, Mapping):
            continue
        policyengine = bindings.get("policyengine")
        if not isinstance(policyengine, Mapping):
            continue
        metric_name = policyengine.get("metric_name")
        target_id = target.get("target_id")
        if not isinstance(metric_name, str) or not isinstance(target_id, str):
            continue
        existing = mapping.get(metric_name)
        if existing is not None and existing != target_id:
            raise ValueError(
                "UK local geography contract maps metric "
                f"{metric_name!r} to multiple target ids: "
                f"{existing!r} and {target_id!r}."
            )
        mapping[metric_name] = target_id
    return mapping


def compile_uk_target_registry(
    facts: Iterable[Mapping[str, Any]],
    *,
    target_period: int | str,
) -> UKLedgerTargetCompilation:
    """Compile packaged UK Ledger references against consumer fact rows."""

    fact_rows = tuple(facts)
    spec = load_country_spec("uk")
    compiled = []
    unsupported: list[dict[str, str]] = []
    for reference in spec.target_references:
        restamped = LedgerTargetReference(
            **{**reference.__dict__, "period": target_period}
        )
        candidate_facts = _candidate_facts_for_reference(fact_rows, restamped)
        try:
            registry = compile_ledger_target_references(
                candidate_facts,
                [restamped],
                country="uk",
            )
            registry = validate_uc_source_month_coverage(
                restamped, registry, candidate_facts
            )
        except ValueError as error:
            unsupported.append(
                {
                    "name": reference.name,
                    "period": target_period,
                    "reason": str(error),
                }
            )
        else:
            compiled.extend(registry.specs)
    return UKLedgerTargetCompilation(
        TargetRegistry(compiled, country="uk"),
        tuple(unsupported),
    )


def load_uk_local_area_crosswalk() -> dict[str, Any]:
    """The committed local-area crosswalk (roster + vintages per level)."""

    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text(encoding="utf-8")
    )


def load_uk_local_target_reference_membership() -> dict[str, Any]:
    """Load the committed local target membership and signed deferrals."""

    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_LOCAL_TARGET_REFERENCE_MEMBERSHIP_RESOURCE)
        .read_text(encoding="utf-8")
    )


def _uk_licensed_empty_legs_from_membership(
    membership: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    """Derive wholly deferred target legs from committed membership rosters."""

    areas_by_level = membership.get("areas_by_geography_level")
    if not isinstance(areas_by_level, Mapping):
        raise ValueError(
            "UK local target membership must expose areas_by_geography_level."
        )
    roster_by_level_leg: dict[tuple[str, str], set[str]] = {}
    for geography_level, raw_area_ids in areas_by_level.items():
        if not isinstance(raw_area_ids, (list, tuple)):
            raise ValueError(
                f"UK local target membership roster {geography_level!r} must be a list."
            )
        for raw_area_id in raw_area_ids:
            area_id = str(raw_area_id).strip()
            if not area_id:
                raise ValueError(
                    "UK local target membership rosters must not contain blank "
                    "area ids."
                )
            leg = _uk_cross_grain_leg_of_area(area_id)
            roster_by_level_leg.setdefault((str(geography_level), leg), set()).add(
                area_id
            )

    signed_deferrals = membership.get("signed_deferrals", ())
    if not isinstance(signed_deferrals, (list, tuple)):
        raise ValueError("UK local target membership signed_deferrals must be a list.")
    deferred_by_target_level_leg: dict[tuple[str, str, str], set[str]] = {}
    for deferral in signed_deferrals:
        if not isinstance(deferral, Mapping):
            raise ValueError(
                "UK local target membership signed deferrals must be mappings."
            )
        target_id = str(deferral.get("target_id", "")).strip()
        geography_level = str(deferral.get("geography_level", "")).strip()
        raw_area_ids = deferral.get("area_ids")
        if (
            not target_id
            or not geography_level
            or not isinstance(raw_area_ids, (list, tuple))
        ):
            raise ValueError(
                "UK local target membership signed deferrals must name a "
                "target_id, geography_level, and area_ids list."
            )
        for raw_area_id in raw_area_ids:
            area_id = str(raw_area_id).strip()
            leg = _uk_cross_grain_leg_of_area(area_id)
            roster = roster_by_level_leg.get((geography_level, leg), set())
            if area_id not in roster:
                raise ValueError(
                    "UK local target membership signed deferral area "
                    f"{area_id!r} is absent from the {geography_level!r} roster."
                )
            deferred_by_target_level_leg.setdefault(
                (target_id, geography_level, leg), set()
            ).add(area_id)

    licensed: dict[str, set[str]] = {}
    for (
        target_id,
        geography_level,
        leg,
    ), deferred in deferred_by_target_level_leg.items():
        roster = roster_by_level_leg[(geography_level, leg)]
        if roster and deferred == roster:
            licensed.setdefault(target_id, set()).add(leg)
    return {
        target_id: frozenset(sorted(legs))
        for target_id, legs in sorted(licensed.items())
    }


def compile_uk_local_target_registry(
    facts: Iterable[Mapping[str, Any]],
    *,
    target_period: int | str,
    crosswalk: Mapping[str, Any],
) -> UKLedgerTargetCompilation:
    """Compile packaged UK local-area Ledger references against fact rows."""

    fact_rows = tuple(facts)
    local_fact_buckets = _local_fact_buckets(fact_rows)
    spec = load_country_spec("uk")
    rosters = _local_crosswalk_rosters(crosswalk)
    compiled = []
    unsupported: list[dict[str, str]] = []
    for reference in spec.local_target_references:
        restamped = LedgerTargetReference(
            **{**reference.__dict__, "period": target_period}
        )
        _assert_local_reference_in_crosswalk(restamped, rosters)
        candidate_facts = _candidate_facts_for_reference(
            _local_candidate_fact_pool(
                fact_rows,
                local_fact_buckets,
                restamped,
            ),
            restamped,
        )
        _assert_local_fact_vintages(candidate_facts, restamped, rosters)
        try:
            registry = compile_ledger_target_references(
                candidate_facts,
                [restamped],
                country="uk",
            )
        except ValueError as error:
            unsupported.append(
                {
                    "name": reference.name,
                    "period": target_period,
                    "reason": str(error),
                }
            )
        else:
            compiled.extend(registry.specs)
    return UKLedgerTargetCompilation(
        TargetRegistry(compiled, country="uk"),
        tuple(unsupported),
    )


def _assert_local_fact_vintages(
    facts: tuple[Mapping[str, Any], ...],
    reference: LedgerTargetReference,
    rosters: Mapping[str, Mapping[str, Any]],
) -> None:
    """Refuse a matched fact whose boundary vintage is not the declared one.

    The crosswalk declares the boundary vintage per level (per code prefix at
    local-authority level, where the nations publish on different frames).
    Before this check the declaration was interpolated into error text only;
    now a fact on the wrong boundary set fails the compile, by name, instead
    of binding a value across vintages (PR #795 review, closing note).
    """

    selector = reference.ledger_selector
    level = str(selector.get("geography_level") or "")
    expected = rosters.get(level, {}).get("expected_vintage", "")
    if not expected:
        # A level the crosswalk declares no vintage for cannot be proven onto
        # any boundary frame (re-review finding 3: the silent escapes are the
        # cases the gate most needs to catch).
        raise ValueError(
            f"UK local target reference {reference.name!r} is at level "
            f"{level!r}, which declares no expected boundary vintage in the "
            "crosswalk."
        )
    for fact in facts:
        geography = fact.get("geography")
        if not isinstance(geography, Mapping):
            continue
        vintage = str(geography.get("vintage") or "")
        code = str(geography.get("id") or "")
        if isinstance(expected, Mapping):
            wanted = expected.get(code[:1])
            if wanted is None:
                raise ValueError(
                    f"UK local target reference {reference.name!r} matched a "
                    f"fact at {code!r}, whose prefix has no declared boundary "
                    f"vintage in the crosswalk for level {level!r}."
                )
        else:
            wanted = expected
        accepted = (
            {str(wanted)} if isinstance(wanted, str) else {str(v) for v in wanted}
        )
        if not vintage:
            raise ValueError(
                f"UK local target reference {reference.name!r} matched a fact "
                f"at {code!r} that declares no boundary vintage; the gate "
                "exists to prove the frame, and an unstamped fact is the case "
                "it most needs to catch."
            )
        if vintage not in accepted:
            raise ValueError(
                f"UK local target reference {reference.name!r} matched a fact "
                f"at {code!r} with boundary vintage {vintage!r}; the crosswalk "
                f"accepts {sorted(accepted)} for level {level!r}."
            )


def _candidate_facts_for_reference(
    facts: tuple[Mapping[str, Any], ...],
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, Any], ...]:
    if not reference.ledger_selector:
        return facts
    return tuple(
        fact
        for fact in facts
        if _fact_matches_selector(fact, reference.ledger_selector)
    )


def _local_fact_buckets(
    facts: tuple[Mapping[str, Any], ...],
) -> dict[tuple[str, str], tuple[Mapping[str, Any], ...]]:
    materialized: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for fact in facts:
        key = _local_fact_geography_key(fact)
        if key is not None:
            materialized.setdefault(key, []).append(fact)
    return {key: tuple(rows) for key, rows in materialized.items()}


def _local_candidate_fact_pool(
    facts: tuple[Mapping[str, Any], ...],
    buckets: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]],
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, Any], ...]:
    selector = reference.ledger_selector
    geography_level = selector.get("geography_level")
    geography_id = selector.get("geography_id")
    if not isinstance(geography_level, str) or not isinstance(geography_id, str):
        return facts
    return buckets.get((geography_level, geography_id), ())


def _local_fact_geography_key(fact: Mapping[str, Any]) -> tuple[str, str] | None:
    geography = fact.get("geography")
    if not isinstance(geography, Mapping):
        return None
    level = geography.get("level")
    geography_id = geography.get("id")
    if not isinstance(level, str) or not isinstance(geography_id, str):
        return None
    if not level or not geography_id:
        return None
    return (level, geography_id)


def _local_crosswalk_rosters(
    crosswalk: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    levels = crosswalk.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError("UK local area crosswalk must expose levels.")
    rosters: dict[str, dict[str, Any]] = {}
    for level, payload in levels.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"UK local area crosswalk level {level!r} is invalid.")
        area_ids = payload.get("area_ids")
        if not isinstance(area_ids, list) or not area_ids:
            raise ValueError(
                f"UK local area crosswalk level {level!r} must expose area_ids."
            )
        rosters[str(level)] = {
            "area_ids": frozenset(str(area_id) for area_id in area_ids),
            "expected_vintage": payload.get("expected_vintage", ""),
        }
    return rosters


def _assert_local_reference_in_crosswalk(
    reference: LedgerTargetReference,
    rosters: Mapping[str, Mapping[str, Any]],
) -> None:
    selector = reference.ledger_selector
    geography_level = str(selector.get("geography_level") or "")
    geography_id = str(selector.get("geography_id") or "")
    if not geography_level or not geography_id:
        raise ValueError(
            f"UK local target reference {reference.name!r} must pin "
            "geography_level and geography_id."
        )
    roster = rosters.get(geography_level)
    if roster is None:
        raise ValueError(
            f"UK local target reference {reference.name!r} uses unknown "
            f"geography level {geography_level!r}."
        )
    if geography_id not in roster["area_ids"]:
        raise ValueError(
            f"UK local target reference {reference.name!r} uses geography id "
            f"{geography_id!r} outside the {geography_level!r} roster for "
            f"expected vintage {roster['expected_vintage']!r}."
        )


def materialize_uk_ledger_targets(
    adapter: Any,
    registry: TargetRegistry,
    *,
    period: int | str,
    band_edge_registry: TargetRegistry | None = None,
) -> TargetMaterializationResult:
    """Materialize compiled UK Ledger target bindings on an adapter."""

    contract = _uk_contract_targets()
    return materialize_target_bindings(
        adapter,
        registry,
        contract,
        period=period,
        providers={
            "parameter_gated_threshold": _uk_parameter_gated_threshold,
            "baseline_flag_crosstab": _uk_baseline_flag_crosstab,
            "input_substitution_counterfactual": _uk_input_substitution,
        },
        band_edge_registry=band_edge_registry,
    )


#: Published-fact reductions rewritten to the internal reduction that carries
#: the same meaning on our frame. Facts keep the semantics of the source that
#: published them; translating those onto the model's own concepts is our job,
#: and a fact we cannot phrase internally gets translated and recorded, never
#: dropped.
#:
#: ``any_child_under`` (DWP Stat-Xplore, Scottish UC households with a child
#: under 1) names a dependent-child concept the model does not carry. There is
#: no ``is_child`` column because the model has no need of one: dependency is
#: derived from age where it is wanted. So "any child under N" is exactly "any
#: person aged under N" here, and the condition already supplies the age bound.
#: The rewrite is declared rather than aliased at the call site so it stays
#: greppable, testable, and visible in review.
UK_TRANSLATED_HOUSEHOLD_REDUCTIONS: Mapping[str, str] = {
    "any_child_under": "any",
}


class UKFrameTargetAdapter:
    """Frame-backed target materialization adapter for UK calibration stages."""

    def __init__(self, frame: Frame):
        self.frame = frame
        self.tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        self.link_tables = {name: frame.link(name).copy() for name in frame.links}

    def has_column(self, entity: str, variable: str) -> bool:
        return variable in self.tables[entity]

    def column(self, entity: str, variable: str) -> np.ndarray:
        table = self.tables[entity]
        count_variables = {
            "person_count",
            "household_count",
            "benunit_count",
            f"{entity}_count",
        }
        if variable in count_variables:
            return np.ones(len(table), dtype=float)
        if variable not in table:
            raise KeyError(f"{entity}.{variable}")
        return np.asarray(table[variable])

    def set_column(self, entity: str, variable: str, values: object) -> None:
        self.tables[entity][variable] = np.asarray(values)

    def parameter(self, parameter: str, period: int | str) -> float:
        if parameter == "gov.hmrc.cgt.annual_exempt_amount":
            return uk_cgt_annual_exempt_amount(period)
        raise KeyError(parameter)

    def counterfactual_delta(
        self,
        binding: Mapping[str, Any],
        period: int | str,
    ) -> np.ndarray:
        del period
        entity = str(binding.get("from_entity") or "person")
        metric_name = str(binding.get("metric_name") or "")
        if metric_name and metric_name in self.tables[entity]:
            return np.asarray(self.tables[entity][metric_name], dtype=float)
        raise ValueError(
            f"frame does not carry precomputed counterfactual delta {metric_name!r}"
        )

    def _household_ids_for(self, entity: str) -> Any:
        """The household each row of ``entity`` belongs to, row-aligned."""

        source = self.tables[entity]
        if entity == "household":
            return source["household_id"]
        if entity == "person":
            # People sit directly in a household; only group entities need the
            # membership lookup below, whose column would be the nonexistent
            # "person_person_id" here.
            return source["person_household_id"]
        people = self.tables["person"]
        entity_membership = f"person_{entity}_id"
        if entity_membership not in people:
            raise KeyError(entity_membership)
        group_to_household = (
            people[[entity_membership, "person_household_id"]]
            .drop_duplicates()
            .set_index(entity_membership)["person_household_id"]
        )
        if group_to_household.index.has_duplicates:
            raise ValueError(f"UK {entity} groups span multiple households.")
        return source[f"{entity}_id"].map(group_to_household)

    def entity_reduction(self, reduction: Mapping[str, Any]) -> np.ndarray:
        """Reduce a member-level variable to a household-grain value.

        The numeric sibling of :meth:`household_condition`: where that answers
        "does this household satisfy the predicate", this answers "what does
        this household's members sum to". A count target declared over a
        boolean member variable — the number of children, not whether the
        household has any — is only honest through this path; collapsing the
        boolean with ``any`` would publish an indicator against a count.
        """

        entity = str(reduction.get("entity") or "person")
        source = self.tables[entity]
        variable = str(reduction["variable"])
        if variable not in source:
            raise KeyError(f"{entity}.{variable}")
        household_ids = self._household_ids_for(entity)
        reduce = str(reduction.get("reduce") or "sum")
        values = source[variable]
        if reduce == "sum":
            aggregate = values.astype(float).groupby(household_ids).sum()
        elif reduce == "count":
            aggregate = values.groupby(household_ids).count()
        else:
            raise ValueError(f"Unsupported UK entity reduction {reduce!r}.")
        households = self.tables["household"]
        return np.asarray(
            households["household_id"].map(aggregate).fillna(0.0),
            dtype=float,
        )

    def household_condition(self, condition: Mapping[str, Any]) -> np.ndarray:
        """Evaluate a household predicate, optionally project it to its members.

        A benefit-unit target can share its dwelling's geography while keeping
        claimant/family filters at benefit-unit grain. ``map_to`` is explicit;
        legacy household conditions keep their household-aligned result.
        """
        entity = str(condition.get("entity") or "household")
        source = self.tables[entity]
        household_ids = self._household_ids_for(entity)

        published = str(condition["reduce"])
        reduce = UK_TRANSLATED_HOUSEHOLD_REDUCTIONS.get(published, published)
        variable = str(condition["variable"])
        if reduce == "any":
            matched = _compare_series(source[variable], condition)
            aggregate = matched.groupby(household_ids).any().astype(float)
            expected = {"operator": "==", "value": True}
        elif reduce == "sum":
            aggregate = source[variable].groupby(household_ids).sum()
            expected = condition
        elif reduce == "count":
            aggregate = source[variable].groupby(household_ids).count()
            expected = condition
        else:
            raise ValueError(
                f"Unsupported UK household reduction {published!r}."
                if published == reduce
                else f"Unsupported UK household reduction {published!r} "
                f"(translated to {reduce!r})."
            )

        households = self.tables["household"]
        ids = households["household_id"]
        matched = _compare_series(ids.map(aggregate).fillna(0.0), expected)
        map_to = str(condition.get("map_to") or "household")
        if map_to != "household":
            if map_to not in {"person", "benunit"}:
                raise ValueError(
                    f"Unsupported UK household condition map_to {map_to!r}."
                )
            matched = self._household_ids_for(map_to).map(
                pd.Series(matched.to_numpy(), index=ids)
            )
            if matched.isna().any():
                raise ValueError(
                    "UK household condition has unmatched household links."
                )
        return np.asarray(matched, dtype=bool)

    def to_frame(self) -> Frame:
        tables = {**self.tables, **self.link_tables}
        weights = {
            entity: self.frame.weights_for(entity)
            for entity in self.frame.weighted_entities
        }
        return Frame(
            tables,
            self.frame.schema,
            weights,
            self.frame.strata,
            mass_log=self.frame.mass_log,
            metadata=self.frame.metadata,
        )


@lru_cache(maxsize=2)
def _uk_contract_targets(
    *,
    national_only: bool = True,
) -> dict[str, Mapping[str, Any]]:
    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_POPULATION_TARGETS_RESOURCE)
        .read_text()
    )
    contract = json.loads(payload)
    _warn_on_undeclared_geography(contract["targets"])
    return {
        target["target_id"]: target
        for target in contract["targets"]
        if not national_only
        or set(target.get("geography_levels") or ())
        <= UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS
    }


def _spec_geography(spec: TargetSpec) -> tuple[str, str]:
    """Resolve one compiled target's local or Ledger geography spelling."""

    metadata = spec.metadata

    def spelling(prefix: str, label: str) -> tuple[str, str] | None:
        level_key = f"{prefix}geography_level"
        id_key = f"{prefix}geography_id"
        if level_key not in metadata and id_key not in metadata:
            return None
        level = str(metadata.get(level_key) or "").strip()
        geography_id = str(metadata.get(id_key) or "").strip()
        if not level or not geography_id:
            raise ValueError(
                f"UK target {spec.name!r} has blank {label} geography "
                f"(level={level!r}, id={geography_id!r})."
            )
        return level, geography_id

    local = spelling("", "local")
    ledger = spelling("ledger_", "ledger")
    if local is not None and ledger is not None and local != ledger:
        raise ValueError(
            f"UK target {spec.name!r} geography spellings disagree: "
            f"local={local!r}, ledger={ledger!r}."
        )
    resolved = local or ledger
    if resolved is None:
        raise ValueError(
            f"UK target {spec.name!r} names no geography under either the "
            "local or Ledger metadata spelling."
        )

    level, geography_id = resolved
    if local is None or level in UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS:
        contract_target_id = str(
            metadata.get("contract_target_id", spec.name.split("@", 1)[0])
        )
        contract = _uk_contract_targets(national_only=False).get(contract_target_id)
        if contract is None:
            raise ValueError(
                f"UK target {spec.name!r} references unknown contract target "
                f"{contract_target_id!r}."
            )
        declared_levels = tuple(
            str(value).strip()
            for value in contract.get("geography_levels") or ()
            if str(value).strip()
        )
        if level not in declared_levels:
            raise ValueError(
                f"UK target {spec.name!r} resolved national geography level "
                f"{level!r}, which disagrees with contract target "
                f"{contract_target_id!r} levels {list(declared_levels)!r}."
            )
    return level, geography_id


def apply_uk_cross_grain_reconciliation(
    local_frame: pd.DataFrame,
    bound_higher_targets: Iterable[str],
    *,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None = None,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the standing UK rule to a bound mixed-grain target surface.

    Increment #762 may extend the grains in the UK rowwise solve only through
    this front door, so detection, reconciliation, and the manifest receipt
    cannot be bypassed.
    """

    licences = (
        _uk_licensed_empty_legs_from_membership(
            load_uk_local_target_reference_membership()
        )
        if licensed_empty_legs is None
        else licensed_empty_legs
    )
    return apply_cross_grain_reconciliation(
        local_frame,
        bound_higher_targets,
        _uk_contract_targets(national_only=False),
        UK_CROSS_GRAIN_RULE,
        reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
        licensed_empty_legs=licences,
    )


#: The Ledger concept the A15 ladder uprating stands on: ONS "Families and
#: households in the UK", Table 5 all-households row, UK, calendar year.
UK_LEDGER_HOUSEHOLDS_TOTAL_CONCEPT = "ons.households_total"
UK_LEDGER_HOUSEHOLDS_TOTAL_GEOGRAPHY = "K02000001"


def uk_ledger_households_total(
    facts: Iterable[Mapping[str, Any]],
    *,
    period: int | str,
) -> dict[str, Any]:
    """Select the published UK household total for ``period`` from the Ledger.

    Exactly one fact must carry the ``ons.households_total`` concept at the
    UK country geography for the calendar year ``period`` with no
    dimensions; zero or several fail closed by name, so an artifact that
    lacks the vintage cannot silently bind the census-vintage ladder rows.
    """

    target_period = int(period)
    matches: list[Mapping[str, Any]] = []
    for fact in facts:
        alignment = fact.get("concept_alignment")
        if not isinstance(alignment, Mapping):
            continue
        if alignment.get("canonical_concept") != UK_LEDGER_HOUSEHOLDS_TOTAL_CONCEPT:
            continue
        geography = fact.get("geography")
        if not isinstance(geography, Mapping) or geography.get("id") != (
            UK_LEDGER_HOUSEHOLDS_TOTAL_GEOGRAPHY
        ):
            continue
        fact_period = fact.get("period")
        if not isinstance(fact_period, Mapping):
            continue
        if fact_period.get("type") != "calendar_year":
            continue
        try:
            if int(fact_period.get("value")) != target_period:
                continue
        except (TypeError, ValueError):
            continue
        if fact.get("dimensions"):
            continue
        matches.append(fact)
    if len(matches) != 1:
        raise ValueError(
            f"UK ladder household uprating needs exactly one Ledger fact for "
            f"{UK_LEDGER_HOUSEHOLDS_TOTAL_CONCEPT!r} at "
            f"{UK_LEDGER_HOUSEHOLDS_TOTAL_GEOGRAPHY} for calendar year "
            f"{target_period}; found {len(matches)}."
        )
    fact = matches[0]
    value = float(fact.get("value"))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"UK ladder household uprating reference must be a positive finite "
            f"count, got {fact.get('value')!r}."
        )
    lineage = fact.get("lineage")
    lineage = lineage if isinstance(lineage, Mapping) else {}
    return {
        "concept": UK_LEDGER_HOUSEHOLDS_TOTAL_CONCEPT,
        "geography_id": UK_LEDGER_HOUSEHOLDS_TOTAL_GEOGRAPHY,
        "period": target_period,
        "value": value,
        "semantic_fact_key": str(fact.get("semantic_fact_key", "")),
        "aggregate_fact_key": str(fact.get("aggregate_fact_key", "")),
        "source_record_id": str(lineage.get("source_record_id", "")),
    }


def uk_ladder_household_uprating(
    ladder: Any,
    households_reference: Mapping[str, Any],
    *,
    period: int | str,
) -> dict[str, Any]:
    """Derive the single national factor that moves the ladder's census
    household counts to the calibration period (microcosm#762 A15).

    The OA ladder's household counts are the 2021 (England, Wales, Northern
    Ireland) and 2022 (Scotland) census counts; the candidate calibrates at
    ``period``. One factor — the Ledger's published UK household total at
    ``period`` over the ladder's total — uprates every ladder household row
    while the ladder keeps its census shares for assignment and support.
    """

    households = np.asarray(ladder.households, dtype=np.float64)
    total = float(households.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("ladder household total must be positive and finite.")
    reference_value = float(households_reference["value"])
    if int(households_reference.get("period", period)) != int(period):
        raise ValueError(
            "UK ladder household uprating reference period "
            f"{households_reference.get('period')!r} is not the calibration "
            f"period {period!r}."
        )
    factor = reference_value / total
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"UK ladder household uprating factor is invalid: {factor!r}.")
    metadata = getattr(ladder, "metadata", {}) or {}
    return {
        "applied": True,
        "period": int(period),
        "factor": factor,
        "ladder_households_total": total,
        "ladder_oa_vintage": str(metadata.get("oa_vintage", "")),
        "reference": dict(households_reference),
        "adjudication": "microcosm#762 (A15, ruling 2026-09-03)",
        "reason": (
            "The OA ladder's household counts are census-vintage (2021; "
            "Scotland 2022); the candidate calibrates at the FRS release's "
            "calibration year, so every ladder household row is scaled by one "
            "national factor to the Ledger's published UK household total for "
            "that year. Assignment shares and support counts stay census-based."
        ),
    }


def _census_vintage_years(oa_vintage: Any) -> frozenset[int]:
    """The census years named by the ladder's ``oa_vintage`` metadata.

    ``"ew:2021_census;scotland:2022_census;ni:dz2021"`` names 2021 and 2022;
    a later ladder names its own years, so a hold from that vintage takes the
    factor without anyone editing a constant.
    """

    return frozenset(
        int(census_year or dz_year)
        for census_year, dz_year in re.findall(
            r"(?<!\d)(\d{4})_census\b|\bdz(\d{4})(?!\d)", str(oa_vintage or "")
        )
    )


def _is_census_vintage_hold(
    metadata: Mapping[str, Any],
    period: int | str,
    *,
    census_years: frozenset[int] | None = None,
) -> bool:
    """True for a compiled reference held from the ladder's census vintage to
    ``period``.

    The household factor is the growth from the ladder's census counts to the
    calibration period, so only a hold from one of those census years may take
    it; a hold from any other vintage (a 2023 estimate, say) keeps its value.
    """

    years = frozenset({2021, 2022}) if census_years is None else census_years
    from_period = metadata.get("uprating_from_period")
    to_period = metadata.get("uprating_to_period")
    if from_period is None or to_period is None:
        return False
    try:
        return int(from_period) in years and int(to_period) == int(period)
    except (TypeError, ValueError):
        return False


def uk_local_target_surface(
    local_registry: TargetRegistry,
    ladder: Any,
    *,
    bound_national_target_ids: Iterable[str],
    period: int | str,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None = None,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
    ladder_household_uprating: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble and reconcile the present-cell UK local target surface.

    ``ladder_household_uprating`` is the A15 receipt from
    :func:`uk_ladder_household_uprating`; when given, every ladder household
    row is scaled by its ``factor`` and the receipt rides the returned
    cross-grain receipt. Without it the ladder rows bind as published and
    the receipt says so.
    """

    if ladder_household_uprating is None:
        uprating_factor = 1.0
        uprating_receipt: dict[str, Any] = {
            "applied": False,
            "reason": (
                "no Ledger household reference supplied; ladder household rows "
                "bind at their census vintage."
            ),
        }
    elif not ladder_household_uprating.get("applied"):
        # A declined receipt (no Ledger facts on this path) rides through
        # unchanged so the manifest says why the rows bind as published.
        uprating_factor = 1.0
        uprating_receipt = dict(ladder_household_uprating)
        uprating_receipt["applied"] = False
    else:
        uprating_factor = float(ladder_household_uprating["factor"])
        if not np.isfinite(uprating_factor) or uprating_factor <= 0:
            raise ValueError(
                f"ladder household uprating factor is invalid: {uprating_factor!r}."
            )
        uprating_receipt = dict(ladder_household_uprating)
    # microcosm#762 A17 (ruling 2026-09-03): the census tenure cells are the
    # same household universe as the ladder rows split by tenure, carried
    # with identity holds from their census vintage; when the ladder rows
    # uprate, the held tenure cells uprate by the same national factor so the
    # partition keeps its published shares at the uprated level. A tenure
    # cell compiled from a fact at the calibration period carries no hold
    # and is never touched.
    tenure_uprated: dict[str, int] = {}
    tenure_holds: list[dict[str, Any]] = []

    level_to_area_type = {
        level: area_type
        for area_type, level in AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL.items()
    }
    target_id_to_metric = {
        target_id: metric for metric, target_id in _uk_local_metric_target_ids().items()
    }
    output_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    national_control_groups: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
    for spec in local_registry.specs:
        geography_level, geography_id = _spec_geography(spec)
        contract_target_id = str(
            spec.metadata.get("contract_target_id", spec.name.split("@", 1)[0])
        )
        if geography_level in level_to_area_type:
            area_type = level_to_area_type[geography_level]
            metric = target_id_to_metric.get(contract_target_id)
            if metric is None:
                raise ValueError(
                    "UK local target surface cannot map contract target "
                    f"{contract_target_id!r} to a PolicyEngine metric."
                )
            allowed = set(metric_names(area_type))
            if metric not in allowed:
                raise ValueError(
                    f"UK local target surface metric {metric!r} is not declared "
                    f"for area_type {area_type!r}."
                )
            output_position = len(output_rows)
            value = float(spec.value)
            if contract_target_id.startswith("ons.tenure."):
                from_period = spec.metadata.get("uprating_from_period")
                to_period = spec.metadata.get("uprating_to_period")
                vintage = uprating_receipt.get("ladder_oa_vintage")
                years = _census_vintage_years(vintage)
                attempted = from_period is not None or to_period is not None
                eligible = attempted and _is_census_vintage_hold(
                    spec.metadata, period, census_years=years
                )
                applied = bool(eligible and uprating_receipt.get("applied"))
                if applied:
                    value *= uprating_factor
                    tenure_uprated[str(from_period)] = (
                        tenure_uprated.get(str(from_period), 0) + 1
                    )
                    reason = "census_vintage_hold_uprated"
                elif not attempted:
                    reason = "no_identity_hold"
                elif not vintage:
                    reason = "missing_ladder_oa_vintage"
                elif not years:
                    reason = "non_census_ladder_vintage"
                elif not eligible:
                    reason = "hold_not_from_ladder_census_vintage_or_wrong_period"
                else:
                    reason = "ladder_household_uprating_not_applied"
                tenure_holds.append(
                    {
                        "target_name": spec.name,
                        "from_period": from_period,
                        "to_period": to_period,
                        "ladder_oa_vintage": vintage,
                        "attempted": attempted,
                        "eligible": eligible,
                        "applied": applied,
                        "skipped": not applied,
                        "reason": reason,
                    }
                )
            output_rows.append(
                {
                    "area_type": area_type,
                    "area_code": geography_id,
                    "metric": metric,
                    "value": value,
                    "target_name": spec.name,
                    "family": family_for_metric(metric),
                    "source_family": spec.family,
                    "source": spec.source,
                    "period": period,
                    "contract_target_id": contract_target_id,
                }
            )
            reconciliation_rows.append(
                {
                    "grain": area_type,
                    "geography_id": geography_id,
                    "target_id": f"contract:{contract_target_id}",
                    "value": value,
                    "_output_position": output_position,
                }
            )
        elif geography_level in UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS:
            value = float(spec.value)
            if not np.isfinite(value):
                raise ValueError(
                    f"UK national target cell {spec.name!r} has non-finite "
                    f"value {value!r}."
                )
            national_control_groups.setdefault(
                (contract_target_id, geography_id), []
            ).append((spec.name, geography_level, value))
        else:
            raise ValueError(
                f"UK target {spec.name!r} names unsupported geography_level "
                f"{geography_level!r}."
            )

    bridge_control_ids = {
        target_id
        for bridge in UK_CROSS_GRAIN_BRIDGES
        for target_id in bridge.higher_target_ids
    }
    fanout_target_ids = {
        target_id
        for (target_id, _), cells in national_control_groups.items()
        if len(cells) > 1 and target_id not in bridge_control_ids
    }
    fanout_targets_not_controls: list[dict[str, Any]] = []
    for (target_id, geography_id), cells in national_control_groups.items():
        cell_names = sorted(name for name, _, _ in cells)
        geography_levels = sorted({level for _, level, _ in cells})
        if len(geography_levels) != 1:
            raise ValueError(
                f"UK national target {target_id!r} at {geography_id!r} has "
                f"fan-out cells {cell_names} with mixed geography levels "
                f"{geography_levels}."
            )
        activated_sum = math.fsum(value for _, _, value in cells)
        if not math.isfinite(activated_sum):
            raise ValueError(
                f"UK national target {target_id!r} at {geography_id!r} has "
                f"fan-out cells {cell_names} with a non-finite summed total."
            )
        if target_id in bridge_control_ids and len(cells) > 1:
            raise ValueError(
                f"UK national target {target_id!r} is a cross-grain bridge control "
                f"but fans out into {len(cells)} cells {cell_names} at "
                f"{geography_id!r}; a bridge control must be one cell per geography."
            )
        if target_id in fanout_target_ids:
            if len(cells) == 1:
                # The target-id rule drops the target as a control at every
                # geography once it fans out at any; say so where it is a
                # single cell rather than dropping it silently.
                fanout_targets_not_controls.append(
                    {
                        "target_id": target_id,
                        "geography_id": geography_id,
                        "cells": 1,
                        "cell_names": cell_names,
                        "activated_sum": activated_sum,
                        "reason": (
                            "Single cell at this geography, but the target fans "
                            "out at another geography, so the target-id rule "
                            "drops it as a control everywhere."
                        ),
                    }
                )
            if len(cells) > 1:
                fanout_targets_not_controls.append(
                    {
                        "target_id": target_id,
                        "geography_id": geography_id,
                        "cells": len(cells),
                        "cell_names": cell_names,
                        "activated_sum": activated_sum,
                        "reason": (
                            "The activated cells are a band subset, so this "
                            "distribution is not a cross-grain control."
                        ),
                    }
                )
            continue
        reconciliation_rows.extend(
            {
                "grain": geography_level,
                "geography_id": geography_id,
                "target_id": target_id,
                "value": value,
                "_output_position": None,
            }
            for _, geography_level, value in cells
        )

    bound_control_ids = tuple(
        str(target_id)
        for target_id in bound_national_target_ids
        if str(target_id) not in fanout_target_ids
    )

    for area_type, targets in (
        ("constituency", constituency_household_targets(ladder)),
        ("la", local_authority_household_targets(ladder)),
    ):
        for row in targets.itertuples(index=False):
            output_position = len(output_rows)
            output_rows.append(
                {
                    "area_type": area_type,
                    "area_code": str(row.code),
                    "metric": "households",
                    "value": float(row.households) * uprating_factor,
                    "target_name": (
                        f"external:census_households/households@{row.code}"
                    ),
                    "family": "census_households",
                    "source": "UK OA geography ladder",
                    "period": period,
                    "contract_target_id": "external:census_households/households",
                }
            )
            reconciliation_rows.append(
                {
                    "grain": area_type,
                    "geography_id": str(row.code),
                    "target_id": "external:census_households/households",
                    "value": float(row.households) * uprating_factor,
                    "_output_position": output_position,
                }
            )

    reconciliation = pd.DataFrame(reconciliation_rows)
    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        reconciliation[["grain", "geography_id", "target_id", "value"]],
        bound_control_ids,
        reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
        licensed_empty_legs=licensed_empty_legs,
    )
    receipt["fanout_targets_not_controls"] = fanout_targets_not_controls
    uprating_receipt["tenure_cells"] = {
        "applied": bool(tenure_uprated),
        "cells": int(sum(tenure_uprated.values())),
        "total_cells": len(tenure_holds),
        "attempted_cells": sum(r["attempted"] for r in tenure_holds),
        "eligible_cells": sum(r["eligible"] for r in tenure_holds),
        "skipped_cells": sum(r["skipped"] for r in tenure_holds),
        "holds": tenure_holds,
        "by_census_vintage": dict(sorted(tenure_uprated.items())),
        "adjudication": "microcosm#762 (A17, ruling 2026-09-03)",
        "reason": (
            "census tenure cells (ONS Census 2021; Scotland's Census 2022) held "
            "to the calibration period are the ladder's household universe split "
            "by tenure; they take the ladder rows' national household factor so "
            "the published tenure shares hold at the uprated level."
        ),
    }
    receipt["ladder_household_uprating"] = uprating_receipt
    for position, value in enumerate(reconciled["value"].to_numpy(dtype=np.float64)):
        output_position = reconciliation.iloc[position]["_output_position"]
        if pd.notna(output_position):
            output_rows[int(output_position)]["value"] = float(value)
    return pd.DataFrame(output_rows), receipt


def _validate_uk_cross_grain_declarations() -> None:
    contract = _uk_contract_targets(national_only=False)
    matched_sides: dict[str, str] = {}
    unknown: list[str] = []
    for bridge in UK_CROSS_GRAIN_BRIDGES:
        for target_id in bridge.higher_target_ids:
            if target_id not in contract:
                unknown.append(target_id)
        lower_target_id = bridge.lower_side.removeprefix("contract:")
        if (
            bridge.lower_side.startswith("contract:")
            and lower_target_id not in contract
        ):
            unknown.append(lower_target_id)
        for side in (*bridge.higher_target_ids, bridge.lower_side):
            canonical = side.removeprefix("contract:")
            existing = matched_sides.get(canonical)
            if existing is not None:
                raise ValueError(
                    f"UK cross-grain target {canonical!r} is covered by both "
                    f"{existing!r} and {bridge.bridge_id!r}."
                )
            matched_sides[canonical] = bridge.bridge_id
    if unknown:
        raise ValueError(
            "UK cross-grain bridge target id(s) are absent from the committed "
            f"contract: {sorted(set(unknown))}."
        )


def _warn_on_undeclared_geography(targets) -> None:
    # Ruling on PR #795 review finding 3: an absent geography_levels reads as
    # national by doctrine (the empty set is a subset of the national levels),
    # and the contract test requires every committed target to declare the
    # field -- so this warning only ever fires on a hand-built contract, where
    # a silent default into the national surface is worth a loud note.
    undeclared = [
        str(target.get("target_id", "<unknown>"))
        for target in targets
        if not target.get("geography_levels")
    ]
    if undeclared:
        import warnings

        warnings.warn(
            "UK population contract target(s) declare no geography_levels and "
            f"default to the national surface: {undeclared[:5]}",
            stacklevel=3,
        )


def _uk_parameter_gated_threshold(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import parameter_gated_threshold

    return parameter_gated_threshold(adapter, binding, period)


def _uk_baseline_flag_crosstab(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import baseline_flag_crosstab

    return baseline_flag_crosstab(adapter, binding, period)


def _uk_input_substitution(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import input_substitution_counterfactual

    return input_substitution_counterfactual(adapter, binding, period)


def _compare_series(
    values: pd.Series,
    condition: Mapping[str, Any],
) -> pd.Series:
    operator = condition.get("operator")
    if operator is None:
        operator, expected = "==", condition["equals"]
    else:
        expected = condition["value"]
    if operator == "in":
        return values.isin(expected)
    operations = {
        "==": values.eq,
        "!=": values.ne,
        ">": values.gt,
        ">=": values.ge,
        "<": values.lt,
        "<=": values.le,
    }
    if operator not in operations:
        raise ValueError(f"Unsupported UK target condition operator {operator!r}.")
    return operations[operator](expected)


_validate_uk_cross_grain_declarations()
