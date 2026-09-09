"""Guarantees for the merged UK population calibration contract resource.

``uk/uk_population_targets.json`` is the consumer-side selection contract over
Chronicle facts (chronicle#166 ruling: contracts live in Microcosm; Chronicle
is facts-only). It carries the national and local-geography target rows in one
value-free resource, with separate scoped provenance blocks for the retired
national registry and local profile surfaces.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources as importlib_resources

from microcosm.build.uk_runtime.fiscal_targets import UK_CGT_TARGET_SPECS
from microcosm.build.uk_runtime.local_targets import (
    load_uk_local_geography_contract,
    metric_names,
    metric_names_from_target_profile,
)

FORBIDDEN_VALUE_KEYS = {"aggregation", "operation", "registry", "target_value", "value"}
FORBIDDEN_RUNTIME_KEYS = {
    "callable",
    "command",
    "execute",
    "executable",
    "function",
    "import",
    "imports",
    "module",
    "python_code",
    "runtime_code",
    "script",
    "solver",
}
BINDING_KINDS = {
    "input_substitution_counterfactual",
    "parameter_gated_threshold",
    "baseline_flag_crosstab",
}
PROJECTION_FAMILIES = {"obr", "slc_borrowers", "scotgov_social_security"}
NATIONAL_SELECTOR_KEYS = {
    "source_name",
    "source_concept",
    "source_measure_id",
    "groupby_dimension",
    "dimensions",
    "dimension_values",
    "entity_name",
    "period_type",
    "period_value",
    "layout_groupby_value_id",
}
LOCAL_SELECTOR_KEYS = {"source_name", "source_measure_id", "record_set_spec_id"}
POLICYENGINE_BINDING_KEYS = {
    "affected_flag_variable",
    "band",
    "band_filter_dimension",
    "band_period_factor",
    "band_upper_bound",
    "band_upper_bound_inclusive",
    "count_of",
    "filters",
    "folded_into",
    "from_entity",
    "gate_comparison",
    "gate_parameter",
    "gated_variable",
    "groupby_variable",
    "household_conditions",
    "kind",
    "value_reduction",
    "map_to",
    "metric_name",
    "notes",
    "output_delta",
    "output_variable",
    "reduce",
    "source_lines",
    "threshold_price_base_year",
    "value_expression",
    "value_variable",
    "zeroed_input",
}
LOCAL_POLICYENGINE_BINDING_KEYS = {
    "filters",
    "from_entity",
    "map_to",
    "metric_name",
    "value_expression",
    "value_variable",
}
ASSERTION_POLICIES = {"allow_source_projection"}
BINDING_REDUCERS = {"any"}
PREDICATE_REDUCERS = {"any", "any_child_under", "count", "sum"}
PREDICATE_ENTITIES = {"person", "benunit", "household"}
LOCAL_UC_RENAMES = {
    "dwp.universal_credit.households": "dwp.uc.households_by_area",
    "dwp.universal_credit.households.0_children": (
        "dwp.uc.households_by_area_children_0"
    ),
    "dwp.universal_credit.households.1_child": ("dwp.uc.households_by_area_children_1"),
    "dwp.universal_credit.households.2_children": (
        "dwp.uc.households_by_area_children_2"
    ),
    "dwp.universal_credit.households.3plus_children": (
        "dwp.uc.households_by_area_children_3plus"
    ),
}


def _load() -> dict:
    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text()
    )
    return json.loads(payload)


def _is_filter_predicate(node: Mapping) -> bool:
    return (
        "value" in node
        and "operator" in node
        and ("concept" in node or "variable" in node)
    )


def _carried_forbidden_keys(node, *, in_defaults: bool = False) -> set[str]:
    carried: set[str] = set()
    if isinstance(node, Mapping):
        forbidden = set(FORBIDDEN_VALUE_KEYS | FORBIDDEN_RUNTIME_KEYS)
        if _is_filter_predicate(node):
            forbidden.discard("value")
        if in_defaults:
            forbidden.discard("operation")
        carried.update(forbidden.intersection(node))
        for child in node.values():
            carried.update(_carried_forbidden_keys(child))
    elif isinstance(node, (list, tuple)):
        for child in node:
            carried.update(_carried_forbidden_keys(child))
    return carried


def test_uk_population_targets_is_registered_in_the_country_package() -> None:
    package = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("country_package.json")
        .read_text()
    )

    paths = [
        row["path"] if isinstance(row, dict) else row for row in package["resources"]
    ]
    assert "uk_population_targets.json" in paths
    assert "uk_national_targets.json" not in paths
    assert "uk_local_geography_targets.json" not in paths


def test_uk_population_targets_shape_order_and_registry_accounting() -> None:
    resource = _load()

    assert resource["country"] == "uk"
    assert resource["allowed_value_operations"] == [
        "identity",
        "sum",
        "difference",
        "calendar_year_average",
        "monthly_window_sum_average",
    ]
    assert resource["resolution_defaults"] == {
        "base_period_policy": "latest_not_after_build_base_period",
        "operation": "sum",
        "assertion_policy": "observed_only",
    }
    assert len(resource["targets"]) == 235

    target_ids = [target["target_id"] for target in resource["targets"]]
    registry_scope = resource["registry_parity"]["scope_target_ids"]
    profile_scope = resource["profile_parity"]["scope_target_ids"]
    assert len(registry_scope) == 202
    assert len(profile_scope) == 33
    assert target_ids[:202] == registry_scope
    assert target_ids[202:] == profile_scope

    parity = resource["registry_parity"]
    assert parity["pinned_ref"] == "12a1e028afeef08d8b2d74ee03fd9de3a78b2dd3"
    assert parity["pinned_version"] == "1.56.16"
    assert parity["mapped_rows"] + parity["excluded_rows"] == parity["registry_rows"]
    assert len(parity["mapped"]) == parity["mapped_rows"]
    assert len(parity["excluded"]) == parity["excluded_rows"]
    mapped_target_ids = set(parity["mapped"].values())
    unmapped_declarations = parity["unmapped_declarations"]
    assert set(mapped_target_ids).isdisjoint(unmapped_declarations)
    assert mapped_target_ids | set(unmapped_declarations) == set(registry_scope)
    assert all(reason for reason in unmapped_declarations.values())
    assert len(mapped_target_ids) == 184
    assert len(unmapped_declarations) == 18
    suppressed_ancestors = parity["suppressed_ancestors"]
    assert len(suppressed_ancestors) == 5
    assert set(suppressed_ancestors).isdisjoint(parity["mapped"])
    assert set(suppressed_ancestors).isdisjoint(parity["excluded"])
    assert {
        entry["contract_target_id"] for entry in suppressed_ancestors.values()
    } <= set(registry_scope)
    assert all(
        "410-Gone" in entry["rationale"] for entry in suppressed_ancestors.values()
    )


def test_uk_population_targets_profile_accounting_and_local_renames() -> None:
    resource = _load()

    parity = resource["profile_parity"]
    assert parity["source_profile_id"] == "uk_local_geography"
    assert parity["source_target_count"] == 25
    assert parity["contract_target_count"] == 33
    assert parity["corrected_rows"] == len(parity["corrected"]) == 25
    assert parity["activation_added_rows"] == len(parity["activation_additions"]) == 8

    targets = {target["target_id"]: target for target in resource["targets"]}
    corrected_ids = {entry["target_id"] for entry in parity["corrected"]}
    added_ids = {entry["target_id"] for entry in parity["activation_additions"]}
    assert corrected_ids | added_ids == set(parity["scope_target_ids"])
    assert corrected_ids.isdisjoint(added_ids)
    for entry in parity["corrected"]:
        target = targets[entry["target_id"]]
        assert entry["corrected_selector"] == target["ledger_selector"]
        assert entry["profile_selector"]
        assert entry["reason"]

    assert {
        entry["target_id"]: entry["renamed_from"]
        for entry in parity["corrected"]
        if "renamed_from" in entry
    } == {new: old for old, new in LOCAL_UC_RENAMES.items()}


def test_uk_population_targets_all_declare_geography_levels() -> None:
    """Ruling on PR #795 review finding 3: absent geography reads as national
    by doctrine, so the committed contract must never omit the field -- a
    non-national target that dropped it would silently leak into the national
    surface. The runtime warns on hand-built contracts; this test makes the
    committed one incapable of triggering that warning."""

    contract = _load()
    undeclared = [
        target["target_id"]
        for target in contract["targets"]
        if not target.get("geography_levels")
    ]
    assert undeclared == []


def test_uk_population_targets_accounting_blocks_partition_all_targets() -> None:
    resource = _load()
    target_ids = {target["target_id"] for target in resource["targets"]}
    registry_scope = set(resource["registry_parity"]["scope_target_ids"])
    profile_scope = set(resource["profile_parity"]["scope_target_ids"])

    assert registry_scope.isdisjoint(profile_scope)
    assert registry_scope | profile_scope == target_ids
    assert not (target_ids - registry_scope - profile_scope)


def test_uk_population_targets_split_terminal_sex_age_bands() -> None:
    resource = _load()
    parity = resource["registry_parity"]

    assert parity["mapped"]["ons/female_85_90"] == "ons.population.female_85_89"
    assert parity["mapped"]["ons/male_85_90"] == "ons.population.male_85_89"
    assert {
        "ons.population.female_90_plus",
        "ons.population.male_90_plus",
    } <= set(parity["unmapped_declarations"])
    assert (
        "single-age-90 share"
        in parity["accounting_notes"]["ons_terminal_sex_band_split"]
    )

    female_85_89 = _target_by_id(resource, "ons.population.female_85_89")
    female_90_plus = _target_by_id(resource, "ons.population.female_90_plus")
    male_85_89 = _target_by_id(resource, "ons.population.male_85_89")
    male_90_plus = _target_by_id(resource, "ons.population.male_90_plus")

    assert female_85_89["ledger_selector"]["dimension_values"] == {
        "sex": "female",
        "age": [85, 86, 87, 88, 89],
    }
    assert male_85_89["ledger_selector"]["dimension_values"] == {
        "sex": "male",
        "age": [85, 86, 87, 88, 89],
    }
    assert female_90_plus["ledger_selector"]["dimension_values"] == {
        "sex": "female",
        "age": ["90_plus"],
    }
    assert male_90_plus["ledger_selector"]["dimension_values"] == {
        "sex": "male",
        "age": ["90_plus"],
    }
    assert female_85_89["measurement"]["filters"] == [
        {"concept": "uk.demographics.gender", "equals": "female"},
        {"concept": "uk.demographics.age", "operator": ">=", "value": 85},
        {"concept": "uk.demographics.age", "operator": "<=", "value": 89},
    ]
    assert female_90_plus["measurement"]["filters"] == [
        {"concept": "uk.demographics.gender", "equals": "female"},
        {"concept": "uk.demographics.age", "operator": ">=", "value": 90},
    ]


def test_uk_population_targets_have_unique_target_ids() -> None:
    resource = _load()

    target_ids = [target["target_id"] for target in resource["targets"]]
    assert len(target_ids) == 235
    assert len(target_ids) == len(set(target_ids))


def test_uk_population_targets_are_value_free_and_hook_free() -> None:
    resource = _load()

    carried = _carried_forbidden_keys(
        {key: value for key, value in resource.items() if key != "resolution_defaults"}
    )
    carried |= _carried_forbidden_keys(
        resource["resolution_defaults"], in_defaults=True
    )
    assert not carried, sorted(carried)
    assert resource["resolution_defaults"]["operation"] == "sum"


def test_uk_population_targets_declare_selector_vocabularies_and_bindings() -> None:
    resource = _load()
    registry_scope = set(resource["registry_parity"]["scope_target_ids"])
    profile_scope = set(resource["profile_parity"]["scope_target_ids"])

    metric_names_seen: list[str] = []
    for target in resource["targets"]:
        target_id = target["target_id"]
        selector = target["ledger_selector"]
        assert selector, target_id
        binding = target["bindings"]["policyengine"]
        metric_names_seen.append(binding["metric_name"])

        if target_id in registry_scope:
            assert set(selector) <= NATIONAL_SELECTOR_KEYS, target_id
            assert "assertion" not in selector, target_id
            assert set(binding) <= POLICYENGINE_BINDING_KEYS, target_id
            kind = binding.get("kind")
            assert kind is None or kind in BINDING_KINDS, target_id
        elif target_id in profile_scope:
            assert set(selector) <= LOCAL_SELECTOR_KEYS, target_id
            assert "chronicle_selector" not in target
            assert set(target["bindings"]) == {"policyengine", "axiom"}
            assert set(binding) <= LOCAL_POLICYENGINE_BINDING_KEYS, target_id
        else:  # pragma: no cover - partition test proves this cannot happen.
            raise AssertionError(f"unscoped target {target_id}")

        if target["family"] in PROJECTION_FAMILIES:
            assert target.get("assertion_policy") == "allow_source_projection", (
                target_id
            )
        else:
            assert "assertion_policy" not in target, target_id

    assert len(metric_names_seen) == len(set(metric_names_seen))


def test_childcare_bus_observation_basis_and_entity_pins_are_closed_world() -> None:
    resource = _load()
    registry_scope = set(resource["registry_parity"]["scope_target_ids"])
    allowed_basis = {
        "annual_flow",
        "annual_unique_count",
        "january_stock",
        "fiscal_year_flow",
    }
    allowed_operations = set(resource["allowed_value_operations"])
    for target in resource["targets"]:
        operation = target.get("value_operation", "identity")
        assert operation in allowed_operations, target["target_id"]
        basis = target["measurement"].get("observation_basis")
        if basis is not None:
            assert basis in allowed_basis, target["target_id"]
        if target["target_id"] in registry_scope and target["measurement"][
            "concept"
        ].endswith(".amount"):
            assert "entity_name" in target["ledger_selector"], target["target_id"]

    childcare = [
        target
        for target in resource["targets"]
        if target["family"] == "dfe_funded_childcare"
    ]
    assert len(childcare) == 3
    assert all(
        not target["measurement"]["concept"].endswith(".amount") for target in childcare
    )


def test_diagnostic_only_cma_comparator_is_not_a_target() -> None:
    resource = _load()

    assert all(
        "cma" not in target["target_id"].lower() for target in resource["targets"]
    )
    assert all("role" not in target for target in resource["targets"])


def test_uk_population_targets_declare_chronicle_loader_guarantees() -> None:
    resource = _load()
    registry_scope = set(resource["registry_parity"]["scope_target_ids"])

    for target in resource["targets"]:
        if target["target_id"] not in registry_scope:
            continue
        selector = target["ledger_selector"]
        if "dimension_values" in selector:
            assert _valid_dimension_values(selector["dimension_values"]), target[
                "target_id"
            ]
        if "dimensions" in selector:
            dimensions = selector["dimensions"]
            assert isinstance(dimensions, list), target["target_id"]
            assert all(isinstance(name, str) and name for name in dimensions), target[
                "target_id"
            ]

        binding = target["bindings"]["policyengine"]
        kind = binding.get("kind")
        if kind == "input_substitution_counterfactual":
            assert {
                "zeroed_input",
                "folded_into",
                "output_variable",
                "output_delta",
            } <= set(binding), target["target_id"]
        elif kind == "parameter_gated_threshold":
            assert {
                "gate_parameter",
                "gated_variable",
                "gate_comparison",
            } <= set(binding), target["target_id"]
        elif kind == "baseline_flag_crosstab":
            assert "affected_flag_variable" in binding, target["target_id"]
            # The two-child-limit rebinding counts a real value_variable per
            # record; count_of survives as the legacy spelling the provider
            # still accepts. Either key must name the counted column.
            assert {"value_variable", "count_of"} & set(binding), target["target_id"]

        reduction = binding.get("value_reduction")
        if reduction is not None:
            assert set(reduction) == {"variable", "entity", "reduce"}, target[
                "target_id"
            ]
            assert reduction["reduce"] in PREDICATE_REDUCERS, target["target_id"]
            assert reduction["entity"] in PREDICATE_ENTITIES, target["target_id"]
            assert reduction["variable"] == binding.get("value_variable"), target[
                "target_id"
            ]

        if "reduce" in binding:
            assert binding["reduce"] in BINDING_REDUCERS, target["target_id"]
        for field in ("filters", "household_conditions"):
            for predicate in binding.get(field, ()):
                if "reduce" in predicate:
                    assert predicate["reduce"] in PREDICATE_REDUCERS, (
                        target["target_id"],
                        predicate,
                    )
                if "entity" in predicate:
                    assert predicate["entity"] in PREDICATE_ENTITIES, (
                        target["target_id"],
                        predicate,
                    )
                if "map_to" in predicate:
                    assert field == "household_conditions", target["target_id"]
                    assert predicate["map_to"] in PREDICATE_ENTITIES
                    assert predicate["map_to"] == binding.get(
                        "from_entity", "household"
                    )
                    assert "reduce" in predicate

        assertion_policy = target.get("assertion_policy")
        if assertion_policy is not None:
            assert assertion_policy in ASSERTION_POLICIES, target["target_id"]

    assert (
        _target_by_id(resource, "obr.esa")["bindings"]["policyengine"][
            "value_expression"
        ]
        == "esa_income + esa_contrib"
    )


def test_uk_population_targets_use_corrected_local_selector_vocabulary() -> None:
    resource = _load()
    targets = {target["target_id"]: target for target in resource["targets"]}

    assert targets["ons.age.0_10"]["ledger_selector"] == {
        "source_measure_id": "population",
        "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
    }
    assert "source_name" not in targets["ons.age.70_80"]["ledger_selector"]
    assert targets["ons.age.70_80"]["ledger_selector"]["record_set_spec_id"] == (
        "uk.local_geography.population.age_70_80.v1"
    )
    assert targets["dwp.uc.households_by_area_children_3plus"]["ledger_selector"] == {
        "source_name": "dwp",
        "source_measure_id": "universal_credit_households_by_children",
        "record_set_spec_id": "uk.local_geography.uc_households.children_3plus.v1",
    }
    assert (
        targets["dwp.uc.households_by_area_children_3plus"]["bindings"]["policyengine"][
            "metric_name"
        ]
        == "uc_hh_3plus_children"
    )
    assert targets["ons.tenure.private_rent"]["ledger_selector"] == {
        "source_measure_id": "households",
        "record_set_spec_id": "uk.local_geography.tenure.private_rent.v1",
    }
    assert targets["hmrc.employment_income.amount"]["ledger_selector"][
        "source_measure_id"
    ] == ["employment_income_count", "employment_income_mean"]
    assert "count_x_mean" in targets["hmrc.employment_income.amount"]["selector_note"]
    assert (
        targets["ons.equiv_net_income_bhc"]["ledger_selector"]["source_measure_id"]
        == "equivalised_net_income_before_housing_costs"
    )
    assert (
        targets["ons.equiv_housing_costs"]["bindings"]["policyengine"][
            "value_expression"
        ]
        == "equiv_hbai_household_net_income - equiv_hbai_household_net_income_ahc"
    )


def test_uk_population_targets_preserve_local_metric_ordering_contract() -> None:
    resource = load_uk_local_geography_contract()

    assert (
        metric_names_from_target_profile(resource, "constituency")
        == metric_names("constituency")[:-1]
    )
    assert metric_names_from_target_profile(resource, "la") == tuple(
        name for name in metric_names("la") if name != "households"
    )
    assert len(metric_names_from_target_profile(resource, "constituency")) == 17
    assert len(metric_names_from_target_profile(resource, "la")) == 29


def test_uk_population_uc_households_target_counts_benunits() -> None:
    resource = _load()
    target = _target_by_id(resource, "dwp.uc.households")

    assert target["measurement"] == {
        "entity": "benunit",
        "concept": "uk.benefit_unit.count",
        "source_months": [f"2025-{month:02}" for month in range(1, 13)],
        "filters": [
            {
                "concept": "uk.benefits.universal_credit.amount",
                "operator": ">",
                "value": 0,
            }
        ],
    }
    assert target["bindings"]["policyengine"]["value_variable"] == "benunit_count"
    assert target["bindings"]["policyengine"]["from_entity"] == "benunit"
    assert target["bindings"]["policyengine"]["filters"] == [
        {
            "variable": "universal_credit",
            "operator": ">",
            "value": 0,
        }
    ]
    assert target["ledger_selector"] == {
        "source_name": "dwp",
        "source_concept": "dwp.uc_households",
        "source_measure_id": "benefit_units",
        "groupby_dimension": "dwp.uc_family_type",
        "dimensions": ["family_type", "payment_indicator", "child_entitlement"],
        "dimension_values": {
            "payment_indicator": "Yes",
            "child_entitlement": ["No", "Yes"],
            "family_type": [
                "Single, no children",
                "Single, with children",
                "Couple, no children",
                "Couple, with children",
                "Unknown or missing family type",
            ],
        },
        "period_type": "month",
        "period_value": [f"2025-{month:02}" for month in range(1, 13)],
    }
    assert target["value_operation"] == "monthly_window_sum_average"
    assert target["period_match_policy"] == "source_window"
    assert len(target["value_operands"]) == 10
    assert "derived" in target["bindings"]["policyengine"]["notes"]
    assert "publisher grand Total" in target["bindings"]["policyengine"]["notes"]


def test_uk_uc_composition_and_disability_children_targets_are_rebound() -> None:
    resource = _load()
    composition_target_ids = {
        "dwp.uc.households_children_1",
        "dwp.uc.households_children_2",
        "dwp.uc.households_children_3",
        "dwp.uc.households_children_4",
        "dwp.uc.households_children_5_or_more",
        "dwp.uc.households_single_no_children",
        "dwp.uc.households_single_with_children",
        "dwp.uc.households_couple_no_children",
        "dwp.uc.households_couple_with_children",
    }
    allowed_filter_variables = {
        "universal_credit",
        "uc_calibration_child_count",
        "uc_calibration_administrative_family_type",
    }

    for target_id in composition_target_ids:
        binding = _target_by_id(resource, target_id)["bindings"]["policyengine"]
        assert binding["from_entity"] == "benunit"
        assert binding["value_variable"] == "benunit_count"
        assert all(
            condition["variable"] == "region" and condition["map_to"] == "benunit"
            for condition in binding.get("household_conditions", ())
        )
        assert "reduce" not in binding
        assert all(
            predicate["variable"] in allowed_filter_variables
            for predicate in binding["filters"]
        )

    for target_id, flag in {
        "dwp.uc.two_child_limit.children_claimant_pip": "uc_tcl_claimant_receives_pip",
        "dwp.uc.two_child_limit.children_disabled_child_element": (
            "uc_tcl_receives_disabled_child_element"
        ),
    }.items():
        binding = _target_by_id(resource, target_id)["bindings"]["policyengine"]
        assert binding["from_entity"] == "benunit"
        assert binding["value_variable"] == "uc_tcl_qualifying_child_count"
        assert "value_reduction" not in binding
        assert binding["kind"] == "baseline_flag_crosstab"
        assert binding["affected_flag_variable"] == "uc_tcl_affected_benunit_proxy"
        assert binding["filters"] == [{"variable": flag, "operator": ">", "value": 0}]


def test_paid_uc_targets_preserve_entitlement_margins_and_exact_calendar_months() -> (
    None
):
    """#252's paid margin is broader than its paid/child-entitled joint.

    No target may silently replace either entitlement status with Yes, or
    reconstruct the independently published child-count Total from detail cells.
    """
    targets = _load()["targets"]
    months = [f"2025-{month:02}" for month in range(1, 13)]
    paid_targets = [
        t
        for t in targets
        if t["target_id"] == "dwp.uc.households"
        or t["target_id"].startswith("dwp.uc.households_children_")
        or t["target_id"]
        in {
            "dwp.uc.households_single_no_children",
            "dwp.uc.households_single_with_children",
            "dwp.uc.households_couple_no_children",
            "dwp.uc.households_couple_with_children",
        }
    ]
    assert len(paid_targets) == 10
    for target in paid_targets:
        selector = target["ledger_selector"]
        assert target["measurement"]["source_months"] == months
        assert selector["period_type"] == "month"
        assert selector["period_value"] == months
        assert selector["dimension_values"]["payment_indicator"] == "Yes"
        assert not any(
            f["variable"] == "uc_calibration_child_entitlement"
            for f in target["bindings"]["policyengine"]["filters"]
        )
        if target["target_id"].startswith("dwp.uc.households_children_"):
            assert selector["source_measure_id"] == "total_benefit_units"
            assert selector["dimension_values"]["child_entitlement"] == "all"
            assert target["value_operation"] == "calendar_year_average"
            assert "value_operands" not in target
        else:
            assert selector["source_measure_id"] == "benefit_units"
            assert target["value_operation"] == "monthly_window_sum_average"
            assert target["period_match_policy"] == "source_window"
            terms = [o["dimension_values"] for o in target["value_operands"]]
            assert all(
                set(term) == {"family_type", "payment_indicator", "child_entitlement"}
                for term in terms
            )
            assert {term["child_entitlement"] for term in terms} == {"No", "Yes"}
            assert len(terms) == (
                10 if target["target_id"] == "dwp.uc.households" else 2
            )
            families = selector["dimension_values"]["family_type"]
            if isinstance(families, str):
                families = [families]
            assert {
                (
                    term["family_type"],
                    term["payment_indicator"],
                    term["child_entitlement"],
                )
                for term in terms
            } == {
                (family, "Yes", entitlement)
                for family in families
                for entitlement in ("No", "Yes")
            }


def test_uc_payment_bands_share_administrative_family_but_keep_source_window() -> None:
    targets = [
        t
        for t in _load()["targets"]
        if t["target_id"].startswith("dwp.uc.payment_distribution_")
    ]
    assert len(targets) == 4
    for target in targets:
        assert target["measurement"]["source_months"] == [
            f"2025-{month:02}" for month in range(4, 13)
        ]
        assert (
            target["bindings"]["policyengine"]["filters"][0]["variable"]
            == "uc_calibration_administrative_family_type"
        )
        assert target["bindings"]["policyengine"]["band_upper_bound"] == 2500


def test_paid_joint_diagnostics_do_not_add_active_targets() -> None:
    targets = _load()["targets"]
    assert len(targets) == 235
    assert not any(
        f.get("variable") == "uc_calibration_child_entitlement"
        for target in targets
        for f in target["bindings"]["policyengine"].get("filters", [])
    )


def test_uc_gb_bindings_match_the_committed_source_geography_and_finite_bands():
    """Source K03000001 is GB; DWP £2400.01–2500 is not its £2500.01+ row."""
    resource = _load()
    references = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("target_references.json")
        .read_text()
    )["target_references"]
    gb_ids = {
        row["metadata"]["contract_target_id"]
        for row in references
        if row["metadata"]["contract_target_id"].startswith("dwp.uc.")
        and row["ledger_selector"]["geography_id"] == "K03000001"
    }
    assert len(gb_ids) == 29
    for target_id in gb_ids:
        binding = _target_by_id(resource, target_id)["bindings"]["policyengine"]
        geographic = [
            condition
            for condition in binding["household_conditions"]
            if condition["variable"] == "region"
        ]
        assert len(geographic) == 1
        assert geographic[0]["entity"] == "household"
        assert geographic[0]["operator"] == "in"
        assert set(geographic[0]["value"]) == {
            "NORTH_EAST",
            "NORTH_WEST",
            "YORKSHIRE",
            "EAST_MIDLANDS",
            "WEST_MIDLANDS",
            "EAST_OF_ENGLAND",
            "LONDON",
            "SOUTH_EAST",
            "SOUTH_WEST",
            "WALES",
            "SCOTLAND",
        }
        assert geographic[0].get("map_to", "household") == binding.get(
            "from_entity", "household"
        )
        if target_id.startswith("dwp.uc.payment_distribution_"):
            assert binding["band_upper_bound"] == 2500
            assert binding["band_upper_bound_inclusive"] is True
            assert binding["band_period_factor"] == 12
    # The omitted source top-coded row is not introduced or merged into a
    # finite reference by these measurement-only changes.
    payment = [
        row for row in references if row["name"].startswith("dwp/uc_payment_dist/")
    ]
    assert len(payment) == 100
    assert all(
        "or over"
        not in row["ledger_selector"]["dimension_values"]["monthly_award_amount_bands"]
        for row in payment
    )


def test_uk_population_cgt_contract_names_match_runtime_specs() -> None:
    resource = _load()

    cgt_metric_names = sorted(
        target["bindings"]["policyengine"]["metric_name"]
        for target in resource["targets"]
        if target["target_id"].startswith("hmrc.cgt.")
    )
    mapped = resource["registry_parity"]["mapped"]

    assert UK_CGT_TARGET_SPECS == ()
    assert cgt_metric_names == [
        "hmrc/capital_gains_total",
        "hmrc/cgt_taxpayers",
    ]
    assert mapped["hmrc/capital_gains_total"] == "hmrc.cgt.gains_total"
    assert mapped["hmrc/cgt_taxpayers"] == "hmrc.cgt.taxpayers_total"


def _target_by_id(resource: Mapping, target_id: str) -> Mapping:
    for target in resource["targets"]:
        if target["target_id"] == target_id:
            return target
    raise AssertionError(f"missing target {target_id!r}")


def _valid_dimension_values(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not value:
        return False
    for key, expected in value.items():
        if not isinstance(key, str) or not key:
            return False
        if _is_dimension_scalar(expected):
            continue
        if (
            isinstance(expected, list)
            and expected
            and all(_is_dimension_scalar(item) for item in expected)
        ):
            continue
        return False
    return True


def _is_dimension_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool)
