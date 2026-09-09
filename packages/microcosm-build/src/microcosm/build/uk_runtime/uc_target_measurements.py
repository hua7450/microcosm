"""UC statistical measurements, separate from benefit entitlement formulas.

DWP's two-child-limit statistics include exceptions and count the claim's
own dependent children/qualifying young people. They do not count everyone
under 18 in the dwelling, or only children losing an element:
https://www.gov.uk/government/statistics/universal-credit-claimants-statistics-on-the-two-child-limit-policy-april-2025/background-information-and-methodology

The retained frame does not identify April open claims or exact birth dates.
The affected measures therefore explicitly remain positive-award/birth-year
proxies. Their receipts record those gaps; improved fit is not evidence that
the administrative population has been reconstructed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask

UC_TARGET_VARIABLES = frozenset(
    {
        "uc_tcl_qualifying_child_count",
        "uc_tcl_affected_child_count_proxy",
        "uc_tcl_affected_benunit_proxy",
        "uc_tcl_claimant_receives_pip",
        "uc_tcl_receives_disabled_child_element",
        "uc_calibration_has_child_under_one",
    }
)

UC_PAID_TARGET_VARIABLES = frozenset(
    {
        "uc_calibration_administrative_family_type",
        "uc_calibration_child_entitlement",
    }
)


def uc_paid_comparison_contract(year: int) -> dict[str, Any]:
    """Describe the paid-claim proxies without implying observed claim history."""
    return {
        "source_unit": "UC claim (single claimant or couple), Great Britain",
        "model_unit": "benunit",
        "source_payment_state": "Payment Indicator = Yes in each observation month",
        "model_payment_proxy": "universal_credit > 0",
        "model_child_entitlement_proxy": "uc_child_element > 0",
        "source_child_definition": "verified declared children or young people under 20",
        "model_child_definition_proxy": (
            "own benefit-unit nonclaimants under 20 or qualifying for a UC child element"
        ),
        "own_qualifying_child_comparison": (
            "own benefit-unit nonclaimants satisfying the model UC child/QYP rule; "
            "retained age and education inputs do not identify exact observation dates"
        ),
        "source_unknown_child_categories": (
            "retain in source reconciliation; no inferred model unknown-child cohort"
        ),
        "source_family_definition": (
            "standard allowance above the maximum single-person allowance means couple"
        ),
        "model_family_definition_proxy": (
            "model uc_standard_allowance compared with the model-year maximum single rate"
        ),
        "zero_standard_allowance_family": "UNKNOWN",
        "family_limitations": (
            "The model allowance does not establish administrative ineligible partner "
            "exceptions or missing declared-child records; structural claimant roles "
            "remain unchanged."
        ),
        "open_nil_claims_identified": False,
        "nil_claim_limitation": (
            "Zero model UC, reported receipt and would_claim_uc do not identify an "
            "administrative open claim with a calculated nil award."
        ),
        "amount_limitation": (
            "Annual model UC is not an observed monthly cash payment or claim trajectory."
        ),
        "model_period": str(year),
        "observation_period": "exact source months are declared on each target reference",
        "status": "partial_alignment_with_explicit_proxies",
        "source_metadata": {
            "family_type": "https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Family%20Type.html",
            "child_entitlement": "https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Child%20Entitlement.html",
            "payment_indicator": "https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Payment%20Indicator.html",
        },
    }


def _uc_benunit_values(
    frame: Any, simulation: Any, variable: str, year: int
) -> np.ndarray:
    raw = simulation.calculate(variable, year)
    values = np.asarray(raw.values if hasattr(raw, "values") else raw)
    if values.ndim != 1 or len(values) != len(frame.table("benunit")):
        raise ValueError(f"UC measurement {variable} must align with benunit rows.")
    if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
        raise ValueError(f"UC measurement {variable} must be finite numeric values.")
    if (values < 0).any():
        raise ValueError(f"UC measurement {variable} must be nonnegative.")
    return values


def uc_administrative_family_type(
    frame: Any, simulation: Any, year: int, child_count: np.ndarray
) -> np.ndarray:
    """Apply DWP's allowance-based family rule to the model's annual state.

    The child count remains an explicit reported-child proxy. This measurement
    never changes which members are claimants or how the model awards UC:
    https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Family%20Type.html
    """
    allowance = _uc_benunit_values(frame, simulation, "uc_standard_allowance", year)
    rates = simulation.tax_benefit_system.parameters(
        str(year)
    ).gov.dwp.universal_credit.standard_allowance.amount
    single_rates = np.asarray([rates.SINGLE_YOUNG, rates.SINGLE_OLD], dtype=float)
    if not np.isfinite(single_rates).all() or (single_rates <= 0).any():
        raise ValueError(
            "UC single standard allowance rates must be finite and positive."
        )
    # The model multiplies the monthly parameter by twelve. Cast the comparison
    # boundary to the amount's precision: a float32 single award must not become
    # a couple merely because its float64 parameter representation is smaller.
    amount_dtype = np.result_type(allowance.dtype, np.float32)
    single_maximum = np.asarray(single_rates.max() * 12, dtype=amount_dtype)
    if not np.isfinite(single_maximum):
        raise ValueError("UC annual single standard allowance maximum must be finite.")
    children = np.asarray(child_count)
    if (
        children.shape != allowance.shape
        or not np.isfinite(children).all()
        or (children < 0).any()
        or not np.equal(children, np.floor(children)).all()
    ):
        raise ValueError(
            "UC reported child counts must be nonnegative benunit integers."
        )
    couple = allowance > single_maximum
    has_children = children > 0
    return np.select(
        [allowance == 0, couple & has_children, couple, has_children],
        ["UNKNOWN", "COUPLE_WITH_CHILDREN", "COUPLE_NO_CHILDREN", "LONE_PARENT"],
        default="SINGLE",
    )


def uc_child_entitlement(frame: Any, simulation: Any, year: int) -> np.ndarray:
    """Measure the recorded model child element, separately from reported children."""
    return _uc_benunit_values(frame, simulation, "uc_child_element", year) > 0


def uc_tcl_comparison_contract(year: int) -> dict[str, Any]:
    """Expose the remaining source/model differences to run receipts."""
    return {
        "source_unit": "UC claim (single claimant or couple)",
        "model_unit": "benunit",
        "source_child_definition": "own dependent children or qualifying young people",
        "includes_exceptions": True,
        "source_claim_state": "open claim within source observation month",
        "model_claim_state_proxy": "universal_credit > 0",
        "source_birth_cutoff": "2017-04-06",
        "model_birth_cutoff_proxy": "model birth_year >= 2017",
        "birth_precision": (
            "model year minus retained age input; source-age observation "
            "date/advancement and exact birth date unavailable"
        ),
        "source_period": "April snapshot; retain matched fact's observation year",
        "model_period": str(year),
        "period_alignment": "annual model state; not an April claim-history replay",
        "status": "partial_alignment_with_explicit_proxies",
    }


def compute_uc_target_measure(
    frame: Any, simulation: Any, variable: str, year: int
) -> np.ndarray:
    """Return a statistical measurement in benefit-unit row order.

    All member reductions use the person's own benefit unit. Child elements and
    PIP on another unit in the same dwelling must not leak into these counts.
    """
    if variable not in UC_TARGET_VARIABLES:
        raise KeyError(variable)
    person, benunit = frame.table("person"), frame.table("benunit")
    members = person["person_benunit_id"].to_numpy()
    ids = benunit["benunit_id"].to_numpy()
    claimant = frs_uc_claimant_mask(person, benunit)

    def calculate(name: str, *, entity: str = "person") -> np.ndarray:
        raw = simulation.calculate(name, year)
        values = np.asarray(raw.values if hasattr(raw, "values") else raw)
        if values.ndim != 1 or len(values) != len(frame.table(entity)):
            raise ValueError(f"UC measurement {name} must align with {entity} rows.")
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError(f"UC measurement {name} must be finite numeric values.")
        return values

    def count(mask: np.ndarray) -> np.ndarray:
        return (
            pd.Series(mask.astype(float))
            .groupby(members, sort=False)
            .sum()
            .reindex(ids)
            .to_numpy()
        )

    if variable == "uc_calibration_has_child_under_one":
        age = calculate("age")
        return count(~claimant & (age >= 0) & (age < 1)) > 0
    if variable == "uc_tcl_claimant_receives_pip":
        return count(claimant & (calculate("pip") > 0)) > 0

    children = ~claimant & calculate(
        "is_child_or_qualifying_young_person_for_universal_credit"
    ).astype(bool)
    child_count = count(children)
    if variable == "uc_tcl_qualifying_child_count":
        return child_count
    if variable == "uc_tcl_receives_disabled_child_element":
        return (
            count(children & (calculate("uc_individual_disabled_child_element") > 0))
            > 0
        )

    birth_year = calculate("birth_year")
    if not np.equal(birth_year, np.floor(birth_year)).all():
        raise ValueError("UC birth-year proxy requires integer model birth years.")
    # The first two children are the oldest. This count is invariant to person
    # row order and ties, and deliberately includes excepted later children.
    post_cutoff_count = count(children & (birth_year >= 2017))
    affected_count = np.minimum(post_cutoff_count, np.maximum(child_count - 2, 0))
    if variable == "uc_tcl_affected_child_count_proxy":
        return affected_count
    return (affected_count > 0) & (calculate("universal_credit", entity="benunit") > 0)
