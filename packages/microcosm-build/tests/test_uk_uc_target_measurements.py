"""Statistical UC claims and own children cannot cross benefit-unit boundaries."""

import json
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.ledger_targets import (
    UKFrameTargetAdapter,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    compute_uk_measure_input,
)
from microcosm.build.uk_runtime.uc_target_measurements import UC_TARGET_VARIABLES
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _fixture():
    # Own QYP ages, award, dwelling. Units 1, 3, 4 share a dwelling: two
    # affected claims and an unrelated claimant with a baby and PIP.
    scenarios = [
        ([19, 17, 15, 13, 11], 100, 0),  # five older qualifying children
        ([19, 10, 3], 100, 1),  # includes an older QYP; child gets PIP
        ([12, 10, 3], 0, 2),  # open-claim test input, no positive model award
        ([0], 100, 1),  # baby belongs only to this young claimant
        ([10, 8, 4], 100, 1),
        ([15, 14, 8], 100, 3),  # ambiguous 2017 birth cohort
        ([12, 10, 2], 100, 4),  # excepted third child, no element denied
    ]
    people, units = [], []
    for bu, (ages, award, hh) in enumerate(scenarios):
        units.append(
            dict(benunit_id=bu, dependent_children=len(ages), universal_credit=award)
        )
        for i, age in enumerate([19 if bu == 3 else 45, *ages]):
            people.append(
                dict(
                    person_id=len(people),
                    person_benunit_id=bu,
                    person_household_id=hh,
                    age=age,
                    birth_year=2025 - age,
                    is_benunit_head=i == 0,
                    is_parent=i == 0,
                    qyp=i > 0 or bu == 3,  # young claimant also QYP: exclude
                    pip=100 if (bu == 1 and i == 1) or (bu == 3 and i == 0) else 0,
                    disabled_element=100 if bu == 1 and i == 1 else 0,
                    denied_element=bu == 0 or (bu == 1 and i == 3),
                )
            )
    person = pd.DataFrame(people).iloc[::-1].reset_index(drop=True)
    frame = Frame(
        {
            "person": person,
            "benunit": pd.DataFrame(units),
            "household": pd.DataFrame({"household_id": range(5), "region": "SCOTLAND"}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.ones(5), WeightKind.DESIGN)},
    )
    columns = {
        "is_child_or_qualifying_young_person_for_universal_credit": "qyp",
        "uc_individual_disabled_child_element": "disabled_element",
        "uc_is_child_limit_affected": "denied_element",
    }

    def calculate(variable, year):
        assert year == 2025
        if variable == "universal_credit":
            return frame.table("benunit")[variable].to_numpy()
        return person[columns.get(variable, variable)].to_numpy()

    return frame, SimpleNamespace(calculate=calculate)


@pytest.mark.parametrize(
    "variable,expected",
    [
        ("uc_tcl_qualifying_child_count", [5, 3, 3, 1, 3, 3, 3]),
        ("uc_tcl_affected_child_count_proxy", [0, 1, 1, 0, 1, 1, 1]),
        ("uc_tcl_affected_benunit_proxy", [0, 1, 0, 0, 1, 1, 1]),
        ("uc_tcl_claimant_receives_pip", [0, 0, 0, 1, 0, 0, 0]),
        ("uc_tcl_receives_disabled_child_element", [0, 1, 0, 0, 0, 0, 0]),
        ("uc_calibration_has_child_under_one", [0, 0, 0, 1, 0, 0, 0]),
    ],
)
def test_uc_statistical_measures_preserve_own_children_and_claimants(
    variable, expected
):
    frame, sim = _fixture()
    result, route = compute_uk_measure_input(frame, sim, "benunit", variable, 2025)
    np.testing.assert_array_equal(result, expected)
    assert route == "uc_claim_statistical_proxy"
    with pytest.raises(KeyError, match="benunit-only"):
        compute_uk_measure_input(frame, sim, "household", variable, 2025)


def test_tcl_counts_include_exceptions_and_do_not_use_denied_element_flags():
    frame, sim = _fixture()
    before, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_tcl_affected_benunit_proxy", 2025
    )
    frame.table("person")["denied_element"] = ~frame.table("person")["denied_element"]
    after, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_tcl_affected_benunit_proxy", 2025
    )
    np.testing.assert_array_equal(before, after)
    assert before[6] and not before[0]


def test_tcl_receipt_explicitly_records_remaining_claim_date_and_period_proxies():
    frame, sim = _fixture()
    resolver = UKMeasureResolver.__new__(UKMeasureResolver)
    resolver.frame, resolver.simulation, resolver.year = frame, sim, 2025
    resolver._receipt = {"mode": "synthetic"}
    resolver._uc_tcl_measures_used = set()
    assert "uc_tcl_comparison_contract" not in resolver.receipt()
    for variable in UC_TARGET_VARIABLES:
        assert resolver.knows("benunit", variable)
        assert not resolver.knows("household", variable)
        assert resolver.entity_for(variable) == "benunit"
        resolver.compute("benunit", variable)
    contract = resolver.receipt()["uc_tcl_comparison_contract"]
    assert contract["source_birth_cutoff"] == "2017-04-06"
    assert contract["model_birth_cutoff_proxy"] == "model birth_year >= 2017"
    assert contract["model_claim_state_proxy"] == "universal_credit > 0"
    assert contract["status"] == "partial_alignment_with_explicit_proxies"
    assert contract["model_period"] == "2025"
    assert len(contract["computed_measures"]) == 5


def test_tcl_and_scottish_registry_sum_claims_without_borrowing_other_units_children():
    frame, sim = _fixture()
    adapter = UKFrameTargetAdapter(frame)
    for variable in UC_TARGET_VARIABLES:
        adapter.set_column(
            "benunit",
            variable,
            compute_uk_measure_input(frame, sim, "benunit", variable, 2025)[0],
        )
    refs = [
        r
        for r in load_country_spec("uk").target_references
        if r.name.startswith("dwp.uc.two_child_limit.")
        or r.name == "dwp.uc.scotland_households_child_under_1"
    ]
    assert len(refs) == 16
    registry = TargetRegistry(
        [
            TargetSpec(
                name=r.name,
                entity=r.entity,
                measure=r.measure,
                value=1,
                period=2025,
                source="synthetic",
                metadata=dict(r.metadata),
            )
            for r in refs
        ],
        country="uk",
    )
    result = materialize_uk_ledger_targets(adapter, registry, period=2025)
    assert not result.skipped
    problem = build_constraint_matrix(
        adapter.to_frame(), registry.to_target_set(), weight_entity="household"
    )
    assert not problem.skipped
    matrix = problem.matrix.toarray()
    by_name = dict(
        zip((name.rsplit("@", 1)[0] for name in problem.names), matrix, strict=True)
    )
    np.testing.assert_array_equal(
        by_name["dwp.uc.two_child_limit.households_affected"], [0, 2, 0, 1, 1]
    )
    np.testing.assert_array_equal(
        by_name["dwp.uc.two_child_limit.children_in_affected_households"],
        [0, 6, 0, 3, 3],
    )
    np.testing.assert_array_equal(
        by_name["dwp.uc.two_child_limit.children_affected"], [0, 2, 0, 1, 1]
    )
    np.testing.assert_array_equal(
        by_name["dwp.uc.two_child_limit.households_claimant_pip"], [0] * 5
    )
    np.testing.assert_array_equal(
        by_name["dwp.uc.two_child_limit.children_disabled_child_element"],
        [0, 3, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        by_name["dwp.uc.scotland_households_child_under_1"], [0, 1, 0, 0, 0]
    )


def test_tcl_contract_is_benefit_unit_native_and_geography_only_at_dwelling_grain():
    contract = json.loads(
        resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text()
    )
    targets = [
        t
        for t in contract["targets"]
        if t["target_id"].startswith("dwp.uc.two_child_limit.")
    ]
    assert len(targets) == 15
    for t in targets:
        b = t["bindings"]["policyengine"]
        assert t["measurement"]["entity"] == b["from_entity"] == "benunit"
        assert b["affected_flag_variable"] == "uc_tcl_affected_benunit_proxy"
        assert "value_reduction" not in b
        assert all(
            c["variable"] == "region" and c["map_to"] == "benunit"
            for c in b["household_conditions"]
        )


def test_resolution_receipt_retains_proxy_contract_discovered_during_computation():
    from microcosm.build.target_materialization import resolve_target_measures

    frame, sim = _fixture()
    resolver = UKMeasureResolver.__new__(UKMeasureResolver)
    resolver.frame, resolver.simulation, resolver.year = frame, sim, 2025
    resolver._receipt = {"mode": "synthetic"}
    resolver._uc_tcl_measures_used = set()
    contract = json.loads(
        resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text()
    )
    resolver.contract_targets = {t["target_id"]: t for t in contract["targets"]}
    name = "dwp.uc.two_child_limit.households_affected"
    registry = TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="benunit",
                measure="dwp/uc/two_child_limit/households_affected",
                value=1,
                period=2025,
                source="synthetic",
                metadata={"contract_target_id": name},
            )
        ],
        country="uk",
    )
    resolved = resolve_target_measures(
        lambda: UKFrameTargetAdapter(frame), registry, resolver, period=2025
    )
    receipt = resolved.receipt["provider"]["uc_tcl_comparison_contract"]
    assert receipt["computed_measures"] == ["uc_tcl_affected_benunit_proxy"]
    assert receipt["model_claim_state_proxy"] == "universal_credit > 0"
    assert receipt["status"] == "partial_alignment_with_explicit_proxies"
