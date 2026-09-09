"""FRS claimant roles must survive row reordering and older dependants."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask


def test_frs_roles_distinguish_partner_from_older_dependent():
    person = pd.DataFrame(
        {
            "person_benunit_id": [20, 10, 20, 10, 20, 30],
            "is_benunit_head": [False, True, True, False, False, True],
            "is_parent": [True, True, True, False, False, False],
            "age": [36, 42, 38, 19, 18, 17],
        },
        index=[8, 6, 4, 2, 0, 9],
    )
    benunit = pd.DataFrame(
        {"benunit_id": [30, 20, 10], "dependent_children": [0, 1, 1]}
    )
    np.testing.assert_array_equal(
        frs_uc_claimant_mask(person, benunit), [True, True, True, False, False, True]
    )


def test_childless_cohabiting_partners_are_both_claimants():
    person = pd.DataFrame(
        {
            "person_benunit_id": [7, 7],
            "is_benunit_head": [True, False],
            "is_parent": [False, False],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [7], "dependent_children": [0]})
    np.testing.assert_array_equal(frs_uc_claimant_mask(person, benunit), [True, True])


def test_couple_mask_uses_source_roles_and_returns_benunit_row_order():
    from microcosm.build.uk_runtime.uc_relationships import frs_uc_couple_mask

    person = pd.DataFrame(
        {
            "person_benunit_id": [20, 10, 20, 10, 30, 30],
            "is_benunit_head": [False, True, True, False, True, False],
            "is_parent": [True, True, True, False, False, False],
            "age": [36, 42, 38, 19, 26, 24],
        },
        index=[8, 6, 4, 2, 0, 9],
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [30, 20, 10],
            "dependent_children": [0, 1, 1],
            "is_married": [False, False, True],
        }
    )
    # Childless cohabitation; cohabiting parents; a lone married parent
    # whose 19-year-old dependent must not become their partner.
    np.testing.assert_array_equal(
        frs_uc_couple_mask(person, benunit), [True, True, False]
    )


@pytest.mark.parametrize("bad_role", [None, 2, "False"])
def test_invalid_roles_fail_instead_of_becoming_truthy(bad_role):
    person = pd.DataFrame(
        {"person_benunit_id": [7], "is_benunit_head": [bad_role], "is_parent": [False]}
    )
    benunit = pd.DataFrame({"benunit_id": [7], "dependent_children": [1]})
    with pytest.raises(ValueError, match="is_benunit_head"):
        frs_uc_claimant_mask(person, benunit)


def test_missing_membership_fails():
    person = pd.DataFrame(
        {"person_benunit_id": [9], "is_benunit_head": [True], "is_parent": [False]}
    )
    benunit = pd.DataFrame({"benunit_id": [7], "dependent_children": [0]})
    with pytest.raises(ValueError, match="membership"):
        frs_uc_claimant_mask(person, benunit)


def test_more_than_two_claimants_requires_source_reconciliation():
    person = pd.DataFrame(
        {
            "person_benunit_id": [7, 7, 7],
            "is_benunit_head": [True, False, False],
            "is_parent": [False, False, False],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [7], "dependent_children": [0]})
    with pytest.raises(ValueError, match="one or two"):
        frs_uc_claimant_mask(person, benunit)


@pytest.mark.requires_uk
@pytest.mark.parametrize("route", ["adapter", "direct_h5"])
def test_released_engine_retains_source_claimants_and_uc_awards(tmp_path, route):
    """Explicit FRS roles survive both model-loading paths without fallback.

    This tests recorded claim membership, including a 17-year-old partner;
    it does not add or establish eligibility for under-18 claimants.
    2025 monthly standard allowances are 400.14 single old, 497.55 couple
    young and 628.10 couple old; an eldest child born before 2017 adds 339.
    """
    from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        write_uk_national_frame,
    )
    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    person = pd.DataFrame(
        {
            "person_id": range(1, 10),
            "person_benunit_id": [10, 10, 20, 20, 30, 30, 40, 40, 40],
            "person_household_id": [1, 1, 2, 2, 3, 3, 4, 4, 4],
            "age": [40, 19, 24, 17, 40, 18, 40, 39, 19],
            "is_benunit_head": [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
            ],
            "is_parent": [True, False, False, False, True, False, True, True, False],
            "is_in_non_advanced_education": [
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                True,
            ],
            "age_started_or_accepted_current_education_or_training": [18] * 9,
            "is_before_universal_credit_qualifying_young_person_terminal_date": [True]
            * 9,
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20, 30, 40],
            "dependent_children": [1, 0, 1, 1],
            "would_claim_uc": [True] * 4,
            "uc_reported_capital": [0.0] * 4,
            "uc_latent_deduction_rate": [0.0] * 4,
            "uc_LCWRA_element": [0.0] * 4,
            "benefit_cap_reduction": [0.0] * 4,
        }
    )
    claimant = frs_uc_claimant_mask(person, benunit)
    person["is_uc_claimant"] = claimant
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=pd.DataFrame(
            {
                "household_id": [1, 2, 3, 4],
                "region": ["LONDON"] * 4,
                "council_tax": [0.0] * 4,
                "rent": [0.0] * 4,
                "tenure_type": ["OWNED_OUTRIGHT"] * 4,
            }
        ),
        time_period=2025,
        household_weights=np.ones(4),
    )
    variables = [
        "is_uc_claimant",
        "uc_standard_allowance",
        "uc_child_element",
        "universal_credit",
    ]
    if route == "adapter":
        values = PolicyEngineUKEngine().materialize(frame, variables, 2025)
    else:
        path = tmp_path / "claimants.h5"
        write_uk_national_frame(frame, path)
        resolver = UKMeasureResolver(
            simulation_source=path, scratch_dir=tmp_path, year=2025, frame=frame
        )
        assert "is_uc_claimant" in resolver.simulation.input_variables
        values = {
            name: np.asarray(resolver.simulation.calculate(name, 2025))
            for name in variables
        }
    np.testing.assert_array_equal(values["is_uc_claimant"], claimant)
    assert values["is_uc_claimant"].dtype.kind == "b"
    np.testing.assert_allclose(
        values["uc_standard_allowance"],
        [4801.68, 5970.60, 4801.68, 7537.20],
        atol=0.01,
        rtol=0,
    )
    np.testing.assert_allclose(
        values["uc_child_element"], [0, 0, 4068, 4068], atol=0.01, rtol=0
    )
    np.testing.assert_allclose(
        values["universal_credit"],
        [4801.68, 5970.60, 8869.68, 11605.20],
        atol=0.01,
        rtol=0,
    )
