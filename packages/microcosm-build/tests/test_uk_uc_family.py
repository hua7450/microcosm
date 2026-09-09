"""UC calibration family categories respect retained FRS relationships."""

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
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _fixture():
    # Single parent + 18-year-old QYP; unmarried parents + child;
    # childless couple including an 18-year-old partner; single claimant;
    # parent + reported 19-year-old who is not eligible for the child element.
    people = pd.DataFrame(
        {
            "person_id": np.arange(10),
            "person_benunit_id": [10, 10, 20, 20, 20, 30, 30, 40, 50, 50],
            "age": [49, 18, 35, 36, 8, 19, 18, 18, 45, 19],
            "is_benunit_head": [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            "is_parent": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        }
    )
    benunits = pd.DataFrame(
        {
            "benunit_id": [50, 30, 10, 40, 20],  # deliberately different order
            "dependent_children": [1, 0, 1, 0, 1],
            "is_married": [False] * 5,  # cohabiting couples must remain couples
        }
    )
    frame = SimpleNamespace(
        table=lambda entity: {"person": people, "benunit": benunits}[entity]
    )
    child_variable = "is_child_or_qualifying_young_person_for_universal_credit"
    sim = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(variables={}),
        calculate=lambda variable, year: (
            np.array([False, True, False, False, True, True, True, True, False, False])
            if variable == child_variable
            else None
        ),
    )
    return frame, sim


def test_uc_family_type_uses_relationships_and_recognises_older_children():
    frame, sim = _fixture()
    values, route = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_family_type", 2025
    )
    assert values.tolist() == [
        "LONE_PARENT",
        "COUPLE_NO_CHILDREN",
        "LONE_PARENT",
        "SINGLE",
        "COUPLE_WITH_CHILDREN",
    ]
    assert route == "frs_relationships_and_uc_child_status"


def test_uc_child_count_excludes_young_claimants_and_includes_reported_children():
    frame, sim = _fixture()
    values, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_child_count", 2025
    )
    assert values.tolist() == [1, 0, 1, 0, 1]


def test_uc_family_measure_refuses_missing_source_relationships():
    frame, sim = _fixture()
    frame.table("person").drop(columns="is_parent", inplace=True)
    with pytest.raises(KeyError, match="is_parent"):
        compute_uk_measure_input(
            frame, sim, "benunit", "uc_calibration_family_type", 2025
        )


def test_uc_family_measure_refuses_ambiguous_claimant_count():
    frame, sim = _fixture()
    frame.table("person").loc[4, "is_parent"] = 1
    with pytest.raises(ValueError, match="one or two.*claimants"):
        compute_uk_measure_input(
            frame, sim, "benunit", "uc_calibration_family_type", 2025
        )


def _administrative_fixture(allowances=None):
    frame, original = _fixture()
    # In benefit-unit order: lone parent; structural young couple receiving a
    # single allowance; lone parent; unavailable allowance; cohabiting parents.
    # The single-allowance couple is an explicit input scenario, not an assertion
    # that the engine reconstructs administrative partner eligibility.
    values = np.array([1200, 1200, 1200, 0, 1800], dtype=np.float32)
    if allowances is not None:
        values = np.asarray(allowances)

    def parameters(period):
        assert period == "2025"
        return SimpleNamespace(
            gov=SimpleNamespace(
                dwp=SimpleNamespace(
                    universal_credit=SimpleNamespace(
                        standard_allowance=SimpleNamespace(
                            amount=SimpleNamespace(SINGLE_YOUNG=80, SINGLE_OLD=100)
                        )
                    )
                )
            )
        )

    def calculate(variable, year):
        if variable == "uc_standard_allowance":
            return values
        if variable == "uc_child_element":
            # A reported 19-year-old need not attract a child element; a young
            # claimant qualifying as a QYP does not make their own claim entitled.
            return np.array([0, 0, 100, 0, 100])
        if variable == "universal_credit":
            return np.array([100, 100, 0, 0, 100])
        return original.calculate(variable, year)

    return frame, SimpleNamespace(
        calculate=calculate,
        tax_benefit_system=SimpleNamespace(parameters=parameters, variables={}),
    )


def test_administrative_family_uses_allowance_without_rewriting_claimant_roles():
    """DWP Family Type uses the allowance, including single-rate couples.

    https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Family%20Type.html
    """
    frame, sim = _administrative_fixture()
    before = frame.table("person").copy(deep=True)
    values, route = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_administrative_family_type", 2025
    )
    assert values.tolist() == [
        "LONE_PARENT",
        "SINGLE",
        "LONE_PARENT",
        "UNKNOWN",
        "COUPLE_WITH_CHILDREN",
    ]
    assert route == "uc_allowance_and_reported_child_proxy"
    structural, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_family_type", 2025
    )
    assert structural[1] == "COUPLE_NO_CHILDREN"
    pd.testing.assert_frame_equal(before, frame.table("person"))


def test_administrative_family_uses_strict_single_maximum_in_amount_precision():
    # Compare at the precision of the model amount, not a tolerance that can
    # relabel an actual amount above the maximum. Test both float widths.
    for dtype in (np.float32, np.float64):
        boundary = dtype(1200)
        above = np.nextafter(boundary, dtype(np.inf))
        frame, sim = _administrative_fixture(
            np.array([boundary, above, boundary, 0, 1800], dtype=dtype)
        )
        values, _ = compute_uk_measure_input(
            frame, sim, "benunit", "uc_calibration_administrative_family_type", 2025
        )
        assert values[0] == "LONE_PARENT"
        assert values[1] == "COUPLE_NO_CHILDREN"
        assert values[3] == "UNKNOWN"


@pytest.mark.parametrize("allowances", [[1200], [1200, 1200, np.nan, 0, 1800]])
def test_administrative_family_refuses_unusable_allowances(allowances):
    frame, sim = _administrative_fixture(allowances)
    with pytest.raises(ValueError, match="UC.*allowance"):
        compute_uk_measure_input(
            frame, sim, "benunit", "uc_calibration_administrative_family_type", 2025
        )


def test_child_entitlement_diagnostic_is_distinct_from_reported_children():
    frame, sim = _administrative_fixture()
    children, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_child_count", 2025
    )
    entitled, route = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_child_entitlement", 2025
    )
    assert children.tolist() == [1, 0, 1, 0, 1]
    assert entitled.tolist() == [False, False, True, False, True]
    assert (entitled & (sim.calculate("universal_credit", 2025) > 0)).tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert route == "uc_child_element_entitlement_proxy"


def test_paid_comparison_receipt_records_observation_and_family_limits():
    frame, sim = _administrative_fixture()
    resolver = UKMeasureResolver.__new__(UKMeasureResolver)
    resolver.frame, resolver.simulation, resolver.year = frame, sim, 2025
    resolver._receipt = {"mode": "synthetic"}
    resolver._uc_tcl_measures_used = set()
    assert "uc_paid_comparison_contract" not in resolver.receipt()
    for variable in (
        "uc_calibration_administrative_family_type",
        "uc_calibration_child_entitlement",
    ):
        assert resolver.knows("benunit", variable)
        assert not resolver.knows("household", variable)
        assert resolver.entity_for(variable) == "benunit"
        resolver.compute("benunit", variable)
    contract = resolver.receipt()["uc_paid_comparison_contract"]
    assert contract["model_payment_proxy"] == "universal_credit > 0"
    assert contract["model_child_entitlement_proxy"] == "uc_child_element > 0"
    assert contract["open_nil_claims_identified"] is False
    assert contract["model_period"] == "2025"
    assert contract["zero_standard_allowance_family"] == "UNKNOWN"
    assert "ineligible partner" in contract["family_limitations"]
    assert len(contract["computed_measures"]) == 2


def test_paid_diagnostic_masks_preserve_benunit_order_and_separate_joint_states():
    from microcosm.build.uk_runtime.measure_simulation import (
        compute_uc_paid_diagnostic_masks,
    )

    frame, sim = _administrative_fixture()
    before = {
        entity: frame.table(entity).copy(deep=True) for entity in ("person", "benunit")
    }
    masks = compute_uc_paid_diagnostic_masks(frame, sim, 2025)
    assert frame.table("benunit").benunit_id.tolist() == [50, 30, 10, 40, 20]
    assert all(mask.shape == (5,) and mask.dtype == bool for mask in masks.values())
    assert masks["paid.total"].tolist() == [True, True, False, False, True]
    assert masks["model.zero_award"].tolist() == [False, False, True, True, False]
    assert masks["paid.child_entitled"].tolist() == [False, False, False, False, True]
    assert masks["paid.family_measure_disagreement"].tolist() == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert masks["paid.child_count_measure_disagreement"].tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert masks["paid.children.1"].tolist() == [True, False, False, False, True]
    assert masks["paid_no_child_entitlement.family.LONE_PARENT"].tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert masks["paid_child_entitled.family.COUPLE_WITH_CHILDREN"].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert not masks["paid.family.UNKNOWN"].any()
    # Each partition independently conserves the paid mask; it does not create
    # a family-by-child-count joint from the separate publisher tables.
    for dimension, categories in {
        "family": [
            "SINGLE",
            "LONE_PARENT",
            "COUPLE_NO_CHILDREN",
            "COUPLE_WITH_CHILDREN",
            "UNKNOWN",
        ],
        "children": ["0", "1", "2", "3", "4", "5_or_more"],
    }.items():
        np.testing.assert_array_equal(
            sum(
                masks[f"paid.{dimension}.{category}"].astype(int)
                for category in categories
            ),
            masks["paid.total"].astype(int),
        )
        for category in categories:
            np.testing.assert_array_equal(
                masks[f"paid_child_entitled.{dimension}.{category}"]
                | masks[f"paid_no_child_entitlement.{dimension}.{category}"],
                masks[f"paid.{dimension}.{category}"],
            )
    for entity in before:
        pd.testing.assert_frame_equal(before[entity], frame.table(entity))
    assert not any("nil" in name or "open" in name for name in masks)


@pytest.mark.requires_uk
@pytest.mark.parametrize("year", [2024, 2025])
def test_released_engine_single_allowance_is_not_misclassified_by_float_precision(year):
    """Use the engine's processed year rates and real float32 allowance output."""
    from policyengine_uk import Simulation

    from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask

    frame, _ = _fixture()
    person, benunit = frame.table("person"), frame.table("benunit")
    claimants = frs_uc_claimant_mask(person, benunit)
    people = {
        str(int(row.person_id)): {
            "age": {str(year): int(row.age)},
            "is_uc_claimant": {str(year): bool(claimants[i])},
        }
        for i, row in enumerate(person.itertuples(index=False))
    }
    units = {
        str(int(bu)): {
            "members": [
                str(int(pid))
                for pid in person.loc[person.person_benunit_id == bu, "person_id"]
            ]
        }
        for bu in benunit.benunit_id
    }
    sim = Simulation(
        situation={
            "people": people,
            "benunits": units,
            "households": {
                key: {"members": unit["members"]} for key, unit in units.items()
            },
        }
    )
    values, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_administrative_family_type", year
    )
    assert values.tolist() == [
        "LONE_PARENT",
        "COUPLE_NO_CHILDREN",
        "LONE_PARENT",
        "SINGLE",
        "COUPLE_WITH_CHILDREN",
    ]


def test_full_uc_payment_registry_matches_independent_relationship_and_band_cases():
    """All active rows count benefit units, retaining edges of excluded bands."""
    # Expected labels are stated by scenario, never calculated by the resolver.
    # Each member is (age, head, parent, native UC child/QYP status).
    scenarios = [
        ("SINGLE", 0, [(18, True, False, True)]),
        ("LONE_PARENT", 1, [(49, True, True, False), (18, False, False, True)]),
        ("LONE_PARENT", 1, [(45, True, True, False), (19, False, False, False)]),
        ("SINGLE", 1, [(45, True, True, False), (20, False, False, False)]),
        ("LONE_PARENT", 1, [(45, True, True, False), (20, False, False, True)]),
        ("COUPLE_NO_CHILDREN", 0, [(19, True, False, True), (18, False, False, True)]),
        (
            "COUPLE_WITH_CHILDREN",
            1,
            [
                (35, True, True, False),
                (36, False, True, False),
                (8, False, False, True),
            ],
        ),
    ]
    # Match the published decimal lower edges, independently of _band_bounds.
    # DWP's last included band is £2,400.01–£2,500/month, inclusive:
    # https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Monthly%20Award%20Amount%20(bands).html
    # The separately published £2,500.01+ category remains unbound. It must
    # never be silently absorbed into the last finite reference.
    edges = [float(f"{100 * i}.01") * 12 for i in range(25)]
    awards = [-1.0, 0.0, 0.1, 30_000.0, np.nextafter(30_000.0, np.inf), 50_000.0]
    for edge in edges:
        awards.extend([np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)])
    people, benunits, expected_families, qualifying = [], [], [], []
    for family, dependent, members in scenarios:
        for award in awards:
            # Two identically eligible benefit units in a dwelling distinguish
            # summation from an any-member household indicator.
            household_id = len(benunits) // 2
            for _ in range(2):
                benunit_id = len(benunits)
                benunits.append(
                    {
                        "benunit_id": benunit_id,
                        "dependent_children": dependent,
                        "is_married": False,
                        "universal_credit": award,
                    }
                )
                expected_families.append(family)
                for age, head, parent, qyp in members:
                    people.append(
                        {
                            "person_id": len(people),
                            "person_benunit_id": benunit_id,
                            "person_household_id": household_id,
                            "age": age,
                            "is_benunit_head": head,
                            "is_parent": parent,
                        }
                    )
                    qualifying.append(qyp)
    # Reverse person-row order to expose positional aggregation assumptions;
    # Frame requires sorted group ids.
    bu = pd.DataFrame(benunits)
    hh = pd.DataFrame(
        {"household_id": np.arange(len(benunits) // 2), "region": "LONDON"}
    )
    frame = Frame(
        {
            "person": pd.DataFrame(people).iloc[::-1].reset_index(drop=True),
            "benunit": bu,
            "household": hh,
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.ones(len(hh)), WeightKind.DESIGN)},
    )
    allowance = np.array(
        [1800 if family.startswith("COUPLE") else 1200 for family in expected_families]
    )
    _, administrative_sim = _administrative_fixture()
    sim = SimpleNamespace(
        calculate=lambda variable, year: (
            allowance
            if variable == "uc_standard_allowance"
            else np.asarray(qualifying[::-1])
        ),
        tax_benefit_system=administrative_sim.tax_benefit_system,
    )
    family, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_family_type", 2025
    )
    np.testing.assert_array_equal(family, expected_families)
    adapter = UKFrameTargetAdapter(frame)
    adapter.set_column("benunit", "uc_calibration_family_type", family)
    administrative_family, _ = compute_uk_measure_input(
        frame, sim, "benunit", "uc_calibration_administrative_family_type", 2025
    )
    np.testing.assert_array_equal(administrative_family, expected_families)
    adapter.set_column(
        "benunit", "uc_calibration_administrative_family_type", administrative_family
    )

    refs = [
        r
        for r in load_country_spec("uk").target_references
        if r.name.startswith("dwp/uc_payment_dist/")
    ]
    excluded = {
        row["name"]
        for row in json.loads(
            resources.files("microcosm.build.uk")
            .joinpath("calibration_measure_exclusions.json")
            .read_text()
        )["exclusions"]
    }
    specs = [
        TargetSpec(
            name=r.name,
            entity=r.entity,
            measure=r.measure,
            value=1.0,
            period=2025,
            source="synthetic",
            metadata={
                **dict(r.metadata),
                **{
                    f"ledger_filter_{key}": value
                    for key, value in r.ledger_selector["dimension_values"].items()
                },
            },
        )
        for r in refs
    ]
    full_registry = TargetRegistry(specs, country="uk")
    active = [spec for spec in specs if spec.name not in excluded]
    assert len(specs) == 100
    assert len(active) == 84
    assert len({s.name for s in specs} & excluded) == 16
    registry = TargetRegistry(active, country="uk")
    result = materialize_uk_ledger_targets(
        adapter, registry, period=2025, band_edge_registry=full_registry
    )
    assert not result.skipped
    problem = build_constraint_matrix(
        adapter.to_frame(), registry.to_target_set(), weight_entity="household"
    )
    assert not problem.skipped
    assert len(problem.names) == 84
    assert not {f"{name}@2025" for name in excluded}.intersection(problem.names)
    expected_family = np.asarray(expected_families)[bu.benunit_id.to_numpy()]
    uc = bu.universal_credit.to_numpy()
    # Synthetic ids encode the independent fixture's two-benefit-unit dwelling.
    bu_household = bu.benunit_id.to_numpy() // 2
    for spec, name, row in zip(
        active, problem.names, problem.matrix.toarray(), strict=True
    ):
        family_name, band_name = spec.name.split("/")[-1].split("_annual_payment_")
        band_index = int(band_name.split("_to_")[0].replace("_", "")) // 1200
        lower = edges[band_index]
        upper = (
            edges[band_index + 1] if band_index < 24 else np.nextafter(30_000.0, np.inf)
        )
        mask = (
            (expected_family == family_name) & (uc > 0) & (uc >= lower) & (uc < upper)
        )
        expected = np.array(
            [mask[bu_household == hid].sum() for hid in hh.household_id]
        )
        np.testing.assert_array_equal(row, expected, err_msg=name)
        assert row.max() == 2, name
