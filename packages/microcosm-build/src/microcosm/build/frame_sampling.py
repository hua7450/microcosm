"""Seeded whole-unit household sampling for Frames — the scale-ladder core.

Promoted from ``us_runtime.stacked_spine`` (microcosm#627): the US stacked
pipeline and the UK national build draw their #624 rung samples through this
one module.  Country policy — what the sampling unit is, how the draw is
stratified, which units are always retained — arrives as parameters, never as
a fork.  The default call (no unit ids, no strata, no forced units) reproduces
the promoted US draw exactly: same RNG stream, same receipt fields, same
values for the same seed.

Two invariants every caller inherits:

- **Whole lineages.**  Selection happens at (or above) household grain and is
  materialized through :meth:`Frame.select` over the person membership mask,
  so every entity row of a selected household enters the sample together.
- **Full selection is a no-op.**  When the exact-count rule requests every
  unit, the input frame is returned unchanged and the RNG is never consulted —
  a ``fraction=1.0`` build is byte-invariant to the sampler's presence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from microcosm.frame import CONSERVE_MASS, Frame, MassChange

__all__ = [
    "EXACT_COUNT_RULE",
    "STRATIFIED_COUNT_RULE",
    "ids_sha256",
    "normalize_sampled_household_mass",
    "sample_frame_households",
    "validate_sample_fraction",
    "validate_sample_seed",
]

#: The deterministic per-group request rule, declared in every receipt.
EXACT_COUNT_RULE = "floor(fraction * eligible)"
STRATIFIED_COUNT_RULE = "one-per-stratum then largest-remainder allocation"

_MASS_RTOL = 1e-9


def validate_sample_fraction(
    fraction: float,
    *,
    label: str = "sample",
    boundary: str | None = None,
) -> None:
    """Reject anything but a finite fraction in ``(0, 1]``."""

    prefix = f"{boundary}: " if boundary else ""
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not np.isfinite(fraction)
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError(
            f"{prefix}{label} fraction must be a finite number in (0, 1]; "
            f"got {fraction!r}."
        )


def validate_sample_seed(seed: int, *, label: str = "sample") -> None:
    """Reject anything but a non-negative integer seed."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{label} seed must be a non-negative integer; got {seed!r}.")


def ids_sha256(ids: np.ndarray) -> str:
    """Canonical digest of an integer id inventory."""

    payload = json.dumps(
        [int(value) for value in np.asarray(ids).tolist()],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _aligned_int_array(
    values: pd.Series | np.ndarray,
    *,
    length: int,
    label: str,
    source_name: str,
) -> np.ndarray:
    array = np.asarray(values.to_numpy() if isinstance(values, pd.Series) else values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"{source_name} {label} must be integer-typed; got dtype "
            f"{array.dtype}. Fractional ids would silently collide in the "
            "selection digest."
        )
    if array.ndim != 1 or len(array) != length:
        raise ValueError(
            f"{source_name} {label} must be one-dimensional with one entry "
            f"per household row ({length}); got shape {array.shape}."
        )
    return array


def sample_frame_households(
    frame: Frame,
    *,
    fraction: float | None = None,
    count: int | None = None,
    seed: int,
    source_name: str,
    unit_ids: pd.Series | np.ndarray | None = None,
    unit_strata: pd.Series | np.ndarray | None = None,
    forced_unit_ids: Sequence[int] = (),
    unit_noun: str = "household",
    floor_context: str = "the sample",
) -> tuple[Frame, dict[str, object]]:
    """Draw one seeded, whole-unit household sample with a receipt.

    Fractional selection follows ``floor(fraction * eligible)`` within each
    stratum. Exact-count selection retains one unit per stratum and distributes
    the remaining request proportionally using deterministic largest
    remainders. Selection
    operates on the sorted unit-id inventory so equal frames produce equal
    samples regardless of incidental row order, and whole lineages enter the
    sample together via :meth:`Frame.select`.

    Args:
        frame: The source frame carrying typed household weights.
        fraction: Optional sampling fraction in ``(0, 1]``.
        count: Optional exact sampling-unit count. Exactly one of ``fraction``
            and ``count`` must be supplied.
        seed: Non-negative integer seed for the selection RNG.
        source_name: Label used in receipts-adjacent error messages.
        unit_ids: Optional per-household-row sampling-unit key.  When absent,
            each household is its own unit (the promoted US behavior).  When
            present, a drawn unit brings **all** its household rows — the
            clone-family contract.
        unit_strata: Optional per-household-row stratum label (strings).  The
            draw is proportional within each stratum; labels must be constant
            within a unit.  When absent, one implicit stratum covers every
            unit and the RNG stream matches the unstratified draw exactly.
        forced_unit_ids: Units always included in the selection, receipted,
            added after the draw so the RNG stream is unaffected.  Forced
            units are identity pins, not a sample: the floors-to-zero refusal
            deliberately counts only the proportional draw, so a fraction
            whose draw floors to zero fails closed even when forced retention
            alone would realize a non-empty frame.
        unit_noun: Noun used in the floors-to-zero refusal.
        floor_context: Requirement owner named in the floors-to-zero refusal.

    Returns:
        The sampled frame and a JSON-ready receipt.  The receipt always
        carries the promoted US fields; ``sampling_unit`` / ``strata`` /
        ``forced_unit_inclusions`` blocks appear only when the corresponding
        feature is exercised, so default-path receipts are unchanged.

    Raises:
        TypeError: If ``frame`` is not a Frame.
        ValueError: If the configuration or the realized selection violates
            the sampling contract (including a floor of zero units, which
            fails closed).
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"{source_name} must be a Frame, got {type(frame).__name__}.")
    if (fraction is None) == (count is None):
        raise ValueError(
            f"{source_name} requires exactly one of sample fraction or unit count."
        )
    if fraction is not None:
        validate_sample_fraction(fraction, label=f"{source_name} sample")
    if isinstance(count, bool) or (count is not None and (not isinstance(count, int) or count < 1)):
        raise ValueError(
            f"{source_name} sample unit count must be a positive integer; got {count!r}."
        )
    validate_sample_seed(seed, label=f"{source_name} sample")

    household = frame.table("household")
    household_ids = household["household_id"].to_numpy()
    eligible_households = int(len(household_ids))
    incoming_mass = float(frame.weights_for("household").total)
    person_entity = frame.schema.person_entity
    household_membership = frame.schema.membership_column("household")

    if unit_ids is None:
        unit_of_household = household_ids
    else:
        unit_of_household = _aligned_int_array(
            unit_ids,
            length=eligible_households,
            label="unit ids",
            source_name=source_name,
        )

    if unit_strata is None:
        stratum_of_household: np.ndarray | None = None
    else:
        strata_array = np.asarray(
            unit_strata.to_numpy()
            if isinstance(unit_strata, pd.Series)
            else unit_strata,
            dtype=object,
        )
        if strata_array.ndim != 1 or len(strata_array) != eligible_households:
            raise ValueError(
                f"{source_name} unit strata must be one-dimensional with one "
                f"entry per household row ({eligible_households}); got shape "
                f"{strata_array.shape}."
            )
        if pd.isna(strata_array).any():
            raise ValueError(
                f"{source_name} unit strata must not contain missing values."
            )
        stratum_of_household = np.asarray(
            [str(value) for value in strata_array], dtype=object
        )

    unit_table = pd.DataFrame({"unit": unit_of_household})
    if stratum_of_household is not None:
        unit_table["stratum"] = stratum_of_household
        deduped = unit_table.drop_duplicates()
        if deduped["unit"].duplicated().any():
            offenders = int(deduped["unit"].duplicated().sum())
            raise ValueError(
                f"{source_name} unit strata must be constant within each "
                f"sampling unit; {offenders} unit(s) span multiple strata."
            )
        stratum_of_unit = dict(
            zip(deduped["unit"].tolist(), deduped["stratum"].tolist(), strict=True)
        )
        group_keys = sorted(set(deduped["stratum"].tolist()))
        groups = {
            key: np.sort(deduped.loc[deduped["stratum"] == key, "unit"].to_numpy())
            for key in group_keys
        }
    else:
        groups = {None: np.sort(np.unique(unit_of_household))}
        group_keys = [None]

    eligible_units = int(sum(len(groups[key]) for key in group_keys))
    if count is not None:
        if count > eligible_units:
            raise ValueError(
                f"{source_name} sample unit count {count} exceeds the "
                f"eligible inventory of {eligible_units}."
            )
        if count < len(group_keys):
            raise ValueError(
                f"{source_name} sample unit count {count} cannot retain one "
                f"unit from each of {len(group_keys)} strata."
            )
        requested_by_group = _allocate_count_by_group(
            groups, group_keys=group_keys, count=count
        )
        exact_count_rule = STRATIFIED_COUNT_RULE
    else:
        assert fraction is not None
        requested_by_group = {
            key: int(math.floor(fraction * len(groups[key]))) for key in group_keys
        }
        exact_count_rule = EXACT_COUNT_RULE
    requested_units = int(sum(requested_by_group.values()))
    if requested_units < 1:
        raise ValueError(
            f"{source_name} sample fraction {fraction!r} floors to zero "
            f"{unit_noun}s ({EXACT_COUNT_RULE} with eligible={eligible_units}); "
            f"{floor_context} requires at least one sampled {unit_noun}."
        )

    forced_array = np.asarray(sorted(set(int(v) for v in forced_unit_ids)))
    if len(forced_array):
        inventory = np.concatenate([groups[key] for key in group_keys])
        missing_forced = np.setdiff1d(forced_array, inventory)
        if len(missing_forced):
            raise ValueError(
                f"{source_name} forced unit ids include {len(missing_forced)} "
                "value(s) absent from the unit inventory."
            )

    if requested_units == eligible_units:
        selected_units = np.sort(np.concatenate([groups[key] for key in group_keys]))
        force_added = np.asarray([], dtype=forced_array.dtype)
        sampled = frame
        selected_household_ids = np.sort(np.asarray(household_ids, copy=True))
    else:
        rng = np.random.default_rng(seed)
        drawn_groups: list[np.ndarray] = []
        for key in group_keys:
            group_units = groups[key]
            requested = requested_by_group[key]
            if requested == len(group_units):
                drawn_groups.append(group_units)
            elif requested > 0:
                drawn_groups.append(
                    np.sort(rng.choice(group_units, size=requested, replace=False))
                )
        drawn = (
            np.sort(np.concatenate(drawn_groups))
            if drawn_groups
            else np.asarray([], dtype=np.asarray(unit_of_household).dtype)
        )
        force_added = np.setdiff1d(forced_array, drawn)
        selected_units = np.union1d(drawn, forced_array) if len(forced_array) else drawn
        if unit_ids is None:
            selected_household_ids = selected_units
        else:
            member_mask = np.isin(unit_of_household, selected_units)
            selected_household_ids = np.sort(household_ids[member_mask])
        person_mask = (
            frame.table(person_entity)[household_membership]
            .isin(selected_household_ids)
            .to_numpy()
        )
        sampled = frame.select(person_mask)

    realized_ids = np.sort(sampled.table("household")["household_id"].to_numpy())
    if not np.array_equal(realized_ids, selected_household_ids):
        raise ValueError(
            f"{source_name} household sampling realized a different household "
            "set than it selected; whole-household selection failed."
        )

    receipt: dict[str, object] = {
        "fraction": float(fraction) if fraction is not None else None,
        "seed": int(seed),
        "eligible_household_count": eligible_households,
    }
    if unit_ids is None and unit_strata is None:
        # Units are households and there is one implicit stratum, so the
        # global floor is a household request — the promoted US receipt
        # field, kept in its promoted position. With strata the per-group
        # floors need not sum to floor(fraction * eligible), so emitting the
        # field would contradict the declared exact-count rule
        # (adversarial-review finding); the unit block carries the honest
        # counts instead.
        receipt["requested_household_count"] = requested_units
    receipt.update(
        {
            "realized_household_count": int(len(realized_ids)),
            "exact_count_rule": exact_count_rule,
            "selected_household_ids_sha256": ids_sha256(selected_household_ids),
            "incoming_household_mass": incoming_mass,
            "sampled_household_mass": float(sampled.weights_for("household").total),
        }
    )
    if unit_ids is not None or unit_strata is not None:
        receipt["sampling_unit"] = {
            "noun": unit_noun,
            "eligible_unit_count": eligible_units,
            "requested_unit_count": requested_units,
            "realized_unit_count": int(len(selected_units)),
            "selected_unit_ids_sha256": ids_sha256(selected_units),
        }
    if stratum_of_household is not None:
        realized_units_by_group: dict[str, int] = {key: 0 for key in group_keys}
        for unit in selected_units.tolist():
            realized_units_by_group[stratum_of_unit[unit]] += 1
        added_beyond_draw_by_group: dict[str, int] = {key: 0 for key in group_keys}
        for unit in force_added.tolist():
            added_beyond_draw_by_group[stratum_of_unit[unit]] += 1
        # realized = requested + added_beyond_draw, per group: forced
        # retention can push a stratum past its floor request, so the
        # receipt shows the arithmetic instead of leaving the excess to be
        # reconciled against the global forced_unit_inclusions block.
        receipt["strata"] = {
            str(key): {
                "eligible_units": int(len(groups[key])),
                "requested_units": requested_by_group[key],
                "added_beyond_draw": added_beyond_draw_by_group[key],
                "realized_units": realized_units_by_group[key],
            }
            for key in group_keys
        }
    if len(forced_array):
        receipt["forced_unit_inclusions"] = {
            "forced_unit_count": int(len(forced_array)),
            "added_beyond_draw_count": int(len(force_added)),
            "forced_unit_ids_sha256": ids_sha256(forced_array),
        }
    return sampled, receipt


def _allocate_count_by_group(
    groups: dict[object, np.ndarray],
    *,
    group_keys: list[object],
    count: int,
) -> dict[object, int]:
    """Allocate an exact request while retaining every declared stratum."""

    requested = {key: 1 for key in group_keys}
    remaining = count - len(group_keys)
    if remaining == 0:
        return requested
    capacities = {key: len(groups[key]) - 1 for key in group_keys}
    total_capacity = sum(capacities.values())
    raw = {
        key: remaining * capacities[key] / total_capacity if total_capacity else 0.0
        for key in group_keys
    }
    allocated = {key: int(math.floor(raw[key])) for key in group_keys}
    remainder = remaining - sum(allocated.values())
    order = sorted(
        group_keys,
        key=lambda key: (-(raw[key] - allocated[key]), str(key)),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    for key in group_keys:
        requested[key] += allocated[key]
    return requested


def normalize_sampled_household_mass(
    sampled: Frame,
    *,
    target_mass: float,
    source_name: str,
) -> tuple[Frame, float]:
    """Restore one sampled survey arm to its full-source design-weight mass."""

    weights = sampled.weights_for("household")
    sampled_mass = float(weights.total)
    if not np.isfinite(sampled_mass) or sampled_mass <= 0.0:
        raise ValueError(
            f"{source_name} sampled household mass must be positive and finite."
        )
    if not np.isfinite(target_mass) or float(target_mass) <= 0.0:
        raise ValueError(
            f"{source_name} target household mass must be positive and "
            f"finite; got {target_mass!r}."
        )
    factor = float(target_mass / sampled_mass)
    normalized_weights = weights.with_values(weights.values * factor, weights.kind)
    mass_policy: str | MassChange = (
        CONSERVE_MASS
        if np.isclose(factor, 1.0, rtol=_MASS_RTOL, atol=0.0)
        else MassChange(
            factor=factor,
            reason=(
                f"composition-preserving {source_name} survey sampling "
                "normalization to full-source household mass"
            ),
        )
    )
    normalized = sampled.with_weights(
        "household",
        normalized_weights,
        mass=mass_policy,
    )
    return normalized, factor
