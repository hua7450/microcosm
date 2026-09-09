"""Relationship inputs shared by UC measurement and diagnostic model replays."""

from __future__ import annotations

import numpy as np
import pandas as pd

UC_COUPLE_DEFINITION = (
    "count(is_benunit_head OR is_parent OR (dependent_children == 0)) "
    "over benefit-unit members == 2"
)


def frs_uc_claimant_mask(person: pd.DataFrame, benunit: pd.DataFrame) -> np.ndarray:
    """Return FRS claimant/partner roles in person-row order.

    The FRS adult-file members of a unit with dependent children are retained
    as ``is_parent``; the head is retained separately. A childless FRS benefit
    unit consists of its claimant or couple. This source role does not assert
    eligibility, receipt of a couple standard allowance, or legal marriage.
    Dependent child-file members do not become partners on their 18th birthday.
    Membership is never changed here.

    Persist this mask as the Boolean ``is_uc_claimant`` input. It is
    authoritative over PolicyEngine-UK's calculator fallback: a 17-year-old
    partner or an older nonqualifying dependent cannot reliably be inferred
    from age and education alone. Reporter payment landing uses a separate
    established convention; this mask identifies every claimant or partner.
    """
    required_person = {"person_benunit_id", "is_benunit_head", "is_parent"}
    required_benunit = {"benunit_id", "dependent_children"}
    for table, required, label in (
        (person, required_person, "person"),
        (benunit, required_benunit, "benunit"),
    ):
        missing = required - set(table.columns)
        if missing:
            raise KeyError(f"UC relationship {label} inputs missing: {sorted(missing)}")
    ids = benunit["benunit_id"]
    members = person["person_benunit_id"]
    if ids.isna().any() or ids.duplicated().any() or members.isna().any():
        raise ValueError("UC relationship membership IDs must be present and unique.")
    if set(ids) != set(members):
        raise ValueError("UC relationship membership must cover every benefit unit.")
    for name in ("is_benunit_head", "is_parent"):
        values = person[name]
        if values.isna().any() or not values.isin([0, 1, False, True]).all():
            raise ValueError(f"{name} must contain complete boolean source roles.")
    dependent = pd.to_numeric(benunit["dependent_children"], errors="coerce")
    if (
        dependent.isna().any()
        or not np.isfinite(dependent).all()
        or (dependent < 0).any()
        or not np.equal(dependent, np.floor(dependent)).all()
    ):
        raise ValueError("dependent_children must contain nonnegative integer counts.")
    child_count = pd.Series(dependent.to_numpy(), index=ids).reindex(members).to_numpy()
    claimants = (
        person["is_benunit_head"].to_numpy(dtype=bool)
        | person["is_parent"].to_numpy(dtype=bool)
        | (child_count == 0)
    )
    counts = pd.Series(claimants).groupby(members.to_numpy()).sum().reindex(ids)
    if not counts.isin([1, 2]).all():
        raise ValueError("FRS UC roles require one or two claimants per benefit unit.")
    return claimants


def frs_uc_couple_mask(person: pd.DataFrame, benunit: pd.DataFrame) -> np.ndarray:
    """Return claimant/partner couple status in benefit-unit row order.

    UC treats cohabiting claimants as a couple regardless of legal marriage.
    Source roles, rather than an adult-age count, also keep an older dependent
    from becoming the lone claimant's partner.
    """
    claimants = frs_uc_claimant_mask(person, benunit)
    counts = (
        pd.Series(claimants)
        .groupby(person["person_benunit_id"].to_numpy(), sort=False)
        .sum()
        .reindex(benunit["benunit_id"])
    )
    return counts.eq(2).to_numpy(dtype=bool)
