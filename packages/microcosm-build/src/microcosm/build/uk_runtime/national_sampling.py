"""Scale-ladder sampling policy for the UK national build (#627).

The UK samples a mid-pipeline artifact: the certified compact already carries
geography/CG clone families (``clone_index`` reverses to the canonical
``clone_index == 0`` surface by ID arithmetic), SPI-synthetic and
capital-gains derivative rows that reverse **to the raw FRS surface** by
max-derived offsets (``max(canonical raw ids) + 1``), and the zero-weight SPI
dead channel whose rebuild enforces a per-stratum ``#base >= #dead`` quota.
A uniform household draw would break all three, so the UK policy over the
shared :mod:`microcosm.build.frame_sampling` core is:

- **The sampling unit is the source FRS family**: the raw canonical
  household plus every SPI/CG derivative and every geography clone that
  reverses onto it through both ID arithmetics.  A drawn family therefore
  keeps clone reversal *and* the SPI/CG source reversal intact — a derived
  row never survives without the raw row it reverses to.
- **The reversal constants are pinned by forced retention.**  The stage
  fence re-derives the clone multiplier (``10 ** len(str(max canonical
  id))``) and the SPI/CG offsets (``max(canonical raw id) + 1``,
  ``max(canonical pre-CG id) + 1``) from surviving data, so the families
  carrying each argmax id — canonical, raw, and pre-CG, household and
  person — are always retained and every constant is stable.
- **The draw is stratified by the raw canonical household's region.**
  Channel flags vary *within* a source family (raw + SPI + CG rows travel
  together), so they are not strata; the per-cell SPI quota is preserved
  structurally instead — a dead row's in-cell base source is in its own
  family — and a post-sample check re-asserts the full per-cell quota,
  failing closed on any irregular input.
- **Sampled mass is renormalized to the full-source household total** (the
  US stacked semantics #627 names), so the anchor population and the SPI
  50/50 mass-share allocation behave identically at every rung.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.build.frame_sampling import (
    normalize_sampled_household_mass,
    sample_frame_households,
    validate_sample_fraction,
    validate_sample_seed,
)
from microcosm.build.uk_runtime.national_frame import validate_uk_national_frame
from microcosm.build.uk_runtime.spi_support import (
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_REPLACEMENT_STRATA_COLUMNS,
)
from microcosm.frame import Frame

__all__ = [
    "UK_SAMPLE_RUNG_TOKENS",
    "UK_SAMPLE_SEED_DEFAULT",
    "sample_uk_national_frame",
    "sample_uk_spine_frame",
    "uk_spine_source_family_units",
    "uk_source_family_units",
]

#: The #624 scale-ladder rungs and their identity tokens, shared with the US
#: stacked pipeline (``LOGBOOK_RUNGS``): ~1% smoke, 10% dev, full scale.
UK_SAMPLE_RUNG_TOKENS: Mapping[float, str] = {
    0.01: "f001",
    0.10: "f010",
    1.00: "f100",
}

#: Default survey-sampling seed, matching the US stacked pipeline's. Distinct
#: from the build ``--seed`` (SPI replacement draw, donor bootstrap, QRF) so
#: dev-scale seed sweeps vary one draw at a time.
UK_SAMPLE_SEED_DEFAULT = 578

(
    _CLONE_INDEX_COLUMN,
    _CAPITAL_GAINS_CLONE_COLUMN,
    _REGION_COLUMN,
) = SPI_REPLACEMENT_STRATA_COLUMNS


def _required_household_columns() -> tuple[str, ...]:
    return (
        "household_id",
        HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
        *SPI_REPLACEMENT_STRATA_COLUMNS,
    )


def _strict_bool(values: pd.Series, *, label: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"UK sample: {label} contains missing values.")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if not numeric.isin([0, 1]).all():
            raise ValueError(
                f"UK sample: {label} must contain only boolean/0/1 values."
            )
        return numeric.to_numpy(dtype=float) != 0.0
    raise ValueError(f"UK sample: {label} must be a boolean column.")


def _int_column(values: pd.Series, *, label: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"UK sample: {label} contains missing values.")
    if not pd.api.types.is_integer_dtype(values.dtype):
        raise ValueError(
            f"UK sample: {label} must be integer-typed; got dtype {values.dtype}."
        )
    return values.to_numpy(dtype=np.int64)


def _str_column(values: pd.Series, *, label: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"UK sample: {label} contains missing values.")
    array = values.to_numpy(dtype=object)
    if not all(isinstance(value, str) and value for value in array):
        raise ValueError(
            f"UK sample: {label} must contain non-empty strings; numeric "
            "codes are refused because equal codes of different dtypes "
            "(1 vs 1.0) would silently split one stratum into two."
        )
    return array


_SPINE_SOURCE_HOUSEHOLD_ID_COLUMN = "source_household_id"
_SPINE_SUPPORT_CLONE_INDEX_COLUMN = "household_support_clone_index"
_SPINE_CGT_BAND_DONOR_COLUMN = "household_is_cgt_band_donor"


def uk_spine_source_family_units(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Return the UK spine's explicit source-family keys and raw-row strata.

    Every derivative row travels with the raw FRS household named by its
    ``source_household_id``. The raw household owns the family's region, so
    derivative flags and incidental derivative regions never split one
    source family across sampling strata.
    """

    household = frame.table("household")
    household_id_column = "household_id"
    lineage_columns = {
        _SPINE_SOURCE_HOUSEHOLD_ID_COLUMN,
        _SPINE_SUPPORT_CLONE_INDEX_COLUMN,
        HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
        _CAPITAL_GAINS_CLONE_COLUMN,
        _SPINE_CGT_BAND_DONOR_COLUMN,
    }
    required = {household_id_column, _REGION_COLUMN, *lineage_columns}
    missing = sorted(required - set(household.columns))
    present_lineage = lineage_columns & set(household.columns)
    if not present_lineage:
        raw_required = {household_id_column, _REGION_COLUMN}
        raw_missing = sorted(raw_required - set(household.columns))
        if raw_missing:
            raise ValueError(
                "UK spine sample requires household identity and region columns; "
                f"household is missing {raw_missing}."
            )
        household_ids = _int_column(
            household[household_id_column], label=household_id_column
        )
        regions = _str_column(household[_REGION_COLUMN], label=_REGION_COLUMN)
        return household_ids, np.asarray(
            [f"region={region}" for region in regions], dtype=object
        )
    if missing:
        raise ValueError(
            "UK spine sample has a partial lineage surface; "
            f"household is missing {missing}."
        )

    household_ids = _int_column(household["household_id"], label="household_id")
    units = _int_column(
        household[_SPINE_SOURCE_HOUSEHOLD_ID_COLUMN],
        label=_SPINE_SOURCE_HOUSEHOLD_ID_COLUMN,
    )
    support_clone = _int_column(
        household[_SPINE_SUPPORT_CLONE_INDEX_COLUMN],
        label=_SPINE_SUPPORT_CLONE_INDEX_COLUMN,
    )
    if (support_clone < 0).any():
        raise ValueError(
            "UK sample: household_support_clone_index must be non-negative."
        )
    spi = _strict_bool(
        household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    capital_gains = _strict_bool(
        household[_CAPITAL_GAINS_CLONE_COLUMN],
        label=_CAPITAL_GAINS_CLONE_COLUMN,
    )
    band_donor = _strict_bool(
        household[_SPINE_CGT_BAND_DONOR_COLUMN],
        label=_SPINE_CGT_BAND_DONOR_COLUMN,
    )
    regions = _str_column(household[_REGION_COLUMN], label=_REGION_COLUMN)

    raw = (support_clone == 0) & ~spi & ~capital_gains & ~band_donor
    source_rows = raw & (household_ids == units)
    source_keys = set(int(value) for value in units[source_rows])
    missing_sources = sorted(set(int(value) for value in units) - source_keys)
    if missing_sources:
        raise ValueError(
            "UK spine sample: source_household_id family key(s) lack a "
            "household_id-bearing raw row: "
            f"{missing_sources[:5]}."
        )

    raw_region_by_unit: dict[int, str] = {}
    multi_region: list[int] = []
    for unit in sorted(set(int(value) for value in units)):
        family_regions = set(str(value) for value in regions[raw & (units == unit)])
        if len(family_regions) > 1:
            multi_region.append(unit)
        else:
            raw_region_by_unit[unit] = next(iter(family_regions))
    if multi_region:
        raise ValueError(
            "UK spine sample: source family spans more than one raw-row "
            f"region for key(s) {multi_region[:5]}."
        )

    strata = np.asarray(
        [f"region={raw_region_by_unit[int(unit)]}" for unit in units],
        dtype=object,
    )
    return units, strata


def sample_uk_spine_frame(
    frame: Frame,
    *,
    fraction: float | None = None,
    source_households: int | None = None,
    seed: int,
) -> tuple[Frame, dict[str, object]]:
    """Sample complete explicit source families from a UK Microcosm spine."""

    if fraction is not None:
        validate_sample_fraction(fraction, label="UK spine sample")
    validate_sample_seed(seed, label="UK spine sample")
    validate_uk_national_frame(frame)

    household = frame.table("household")
    units, strata = uk_spine_source_family_units(frame)
    forced_source_families: tuple[int, ...] = ()
    if source_households is not None:
        unit_table = pd.DataFrame({"unit": units, "stratum": strata}).drop_duplicates()
        forced_source_families = tuple(
            sorted(
                {
                    int(value)
                    for _stratum, group in unit_table.groupby("stratum", sort=True)
                    for value in (group["unit"].min(), group["unit"].max())
                }
            )
        )
    pre_counts = pd.Series(units).value_counts(sort=False)
    full_mass = float(frame.weights_for("household").total)
    sampled, core_receipt = sample_frame_households(
        frame,
        fraction=fraction,
        count=source_households,
        seed=seed,
        source_name="UK spine",
        unit_ids=units,
        unit_strata=strata,
        forced_unit_ids=forced_source_families,
        unit_noun="source family",
        floor_context="the UK rowwise candidate",
    )
    normalized, factor = normalize_sampled_household_mass(
        sampled,
        target_mass=full_mass,
        source_name="UK spine",
    )

    post_units, _ = uk_spine_source_family_units(normalized)
    post_counts = pd.Series(post_units).value_counts(sort=False)
    incomplete = sorted(
        int(unit)
        for unit, count in post_counts.items()
        if int(count) != int(pre_counts.loc[unit])
    )
    if incomplete:
        raise ValueError(
            "UK spine sample retained incomplete source family key(s): "
            f"{incomplete[:5]}."
        )
    validate_uk_national_frame(normalized)
    result = {
        "fraction": float(fraction) if fraction is not None else None,
        "seed": int(seed),
        "rung_token": (
            UK_SAMPLE_RUNG_TOKENS.get(float(fraction))
            if fraction is not None
            else f"h{source_households:04d}"
        ),
        "pre_household_count": int(len(household)),
        "post_household_count": int(len(normalized.table("household"))),
        "pre_family_count": int(len(pre_counts)),
        "post_family_count": int(len(post_counts)),
        "normalization_factor": float(factor),
        "strata_count": int(len(np.unique(strata))),
        "receipt": dict(core_receipt),
    }
    if source_households is not None:
        unit_receipt = dict(core_receipt["sampling_unit"])
        forced = dict(core_receipt.get("forced_unit_inclusions", {}))
        result.update(
            {
                "sample_mode": "bounded_source_households",
                "requested_source_households": int(source_households),
                "eligible_source_families": int(unit_receipt["eligible_unit_count"]),
                "proportional_request": int(unit_receipt["requested_unit_count"]),
                "forced_additions": int(forced.get("added_beyond_draw_count", 0)),
                "realized_source_families": int(unit_receipt["realized_unit_count"]),
                "realized_household_rows": int(len(normalized.table("household"))),
            }
        )
        digest_payload = json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        result["receipt_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    return normalized, result


def uk_source_family_units(
    frame: Frame,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Per-household-row source-FRS-family key, mirroring the stage fence.

    Reproduces ``_resolve_candidate_lineage``'s two-layer arithmetic: the
    clone multiplier is ``10 ** max(1, len(str(canonical_max)))`` over the
    ``clone_index == 0`` household **and** person ids; the SPI and CG
    household offsets are ``max(canonical raw household id) + 1`` and
    ``max(canonical pre-CG household id) + 1``; and every row's family key is

    ``household_id - clone_index*multiplier - spi*spi_offset - cg*cg_offset``

    which must land on an existing canonical raw FRS household.  Fails closed
    before any draw on household-level rows the stage fence would refuse;
    person-level lineage defects are outside this pre-check and surface at
    stage one, where the fence stays strict.

    Returns:
        The per-household-row family keys, the forced family keys (the
        families carrying every argmax id the fence's constants are derived
        from — retaining them pins the multiplier's digit count and both
        offsets exactly), and the clone multiplier.
    """

    household = frame.table("household")
    missing = sorted(set(_required_household_columns()) - set(household.columns))
    if missing:
        raise ValueError(
            "UK sample requires the certified compact's lineage and channel "
            f"columns; household is missing {missing}."
        )
    person = frame.table("person")
    household_ids = _int_column(household["household_id"], label="household_id")
    clone_index = _int_column(household[_CLONE_INDEX_COLUMN], label=_CLONE_INDEX_COLUMN)
    if (clone_index < 0).any():
        raise ValueError("UK sample: clone_index must be non-negative.")
    spi_flag = _strict_bool(
        household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    cg_flag = _strict_bool(
        household[_CAPITAL_GAINS_CLONE_COLUMN],
        label=_CAPITAL_GAINS_CLONE_COLUMN,
    )
    canonical_households = clone_index == 0
    if not canonical_households.any():
        raise ValueError("UK sample requires clone_index=0 canonical rows.")

    person_ids = _int_column(person["person_id"], label="person_id")
    person_household = _int_column(
        person["person_household_id"], label="person_household_id"
    )
    household_position = pd.Series(np.arange(len(household_ids)), index=household_ids)
    person_household_position = household_position.reindex(person_household)
    if person_household_position.isna().any():
        raise ValueError(
            "UK sample: person_household_id cannot map every person to a household row."
        )
    person_positions = person_household_position.to_numpy(dtype=np.int64)
    person_clone_index = clone_index[person_positions]
    person_spi = spi_flag[person_positions]
    person_cg = cg_flag[person_positions]
    canonical_people = person_clone_index == 0
    if not canonical_people.any():
        raise ValueError("UK sample requires clone_index=0 person rows.")

    canonical_max = max(
        int(household_ids[canonical_households].max()),
        int(person_ids[canonical_people].max()),
    )
    multiplier = 10 ** max(1, len(str(canonical_max)))
    clone_reversed = household_ids - clone_index * multiplier
    if (clone_reversed <= 0).any():
        raise ValueError(
            "UK sample: clone reversal produced non-positive canonical ids; "
            "the input's clone lineage does not match the stage fence's "
            "arithmetic."
        )
    canonical_set = household_ids[canonical_households]
    unknown_clones = np.setdiff1d(np.unique(clone_reversed), canonical_set)
    if len(unknown_clones):
        raise ValueError(
            f"UK sample: {len(unknown_clones)} clone household(s) do not "
            "reverse to a clone_index=0 canonical row; refusing to sample an "
            "input the stage fence would reject."
        )

    raw_households = canonical_households & ~spi_flag & ~cg_flag
    pre_cg_households = canonical_households & ~cg_flag
    if not raw_households.any():
        raise ValueError("UK sample requires canonical raw FRS households.")
    spi_offset = int(household_ids[raw_households].max()) + 1
    cg_offset = int(household_ids[pre_cg_households].max()) + 1
    units = (
        clone_reversed
        - spi_flag.astype(np.int64) * spi_offset
        - cg_flag.astype(np.int64) * cg_offset
    )
    raw_set = frozenset(int(value) for value in household_ids[raw_households])
    if (units <= 0).any() or not set(int(v) for v in np.unique(units)) <= raw_set:
        bad = len(set(int(v) for v in np.unique(units)) - raw_set)
        raise ValueError(
            f"UK sample: {bad} household(s) do not reverse to the raw FRS "
            "surface; refusing to sample an input the stage fence would "
            "reject."
        )

    def _unit_of_max_household(mask: np.ndarray) -> int:
        position = int(np.argmax(np.where(mask, household_ids, -1)))
        return int(units[position])

    def _unit_of_max_person(mask: np.ndarray) -> int:
        position = int(np.argmax(np.where(mask, person_ids, -1)))
        return int(units[person_positions[position]])

    raw_people = canonical_people & ~person_spi & ~person_cg
    pre_cg_people = canonical_people & ~person_cg
    if not raw_people.any():
        raise ValueError("UK sample requires canonical raw FRS person rows.")
    forced = tuple(
        sorted(
            {
                # Clone-multiplier pins (digit count of the canonical max).
                _unit_of_max_household(canonical_households),
                _unit_of_max_person(canonical_people),
                # SPI-offset pins (max canonical raw id, exactly).
                _unit_of_max_household(raw_households),
                _unit_of_max_person(raw_people),
                # CG-offset pins (max canonical pre-CG id, exactly).
                _unit_of_max_household(pre_cg_households),
                _unit_of_max_person(pre_cg_people),
            }
        )
    )
    return units, forced, multiplier


def _assert_spi_replacement_quota(household: pd.DataFrame) -> None:
    """Re-assert the SPI rebuild's per-cell ``#base >= #dead`` on the sample.

    Source-family sampling preserves the inequality structurally when every
    dead row's source lives in its own cell; this check converts that
    assumption into a fail-closed runtime receipt, using the same stratum
    cells the rebuild's ``_sample_replacement_household_ids`` groups by.
    """

    synthetic = _strict_bool(
        household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    dead = household.loc[synthetic]
    if dead.empty:
        return
    strata = list(SPI_REPLACEMENT_STRATA_COLUMNS)
    base = household.loc[~synthetic]
    dead_counts = dead.groupby(strata, sort=True, dropna=False).size()
    base_counts = base.groupby(strata, sort=True, dropna=False).size()
    for key, quota in dead_counts.items():
        capacity = int(base_counts.get(key, 0))
        if capacity < int(quota):
            raise ValueError(
                "UK sample violates the SPI replacement quota for stratum "
                f"{key!r}: need {int(quota)} base household(s), found "
                f"{capacity}. Source-family sampling preserves this "
                "inequality when dead rows share their source's cell; an "
                "irregular structure in the input is the likely cause."
            )


def sample_uk_national_frame(
    frame: Frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[Frame, dict[str, object]]:
    """Draw one seeded UK rung sample and renormalize it to full mass.

    Runs on the loaded certified compact **before** provenance binding, so
    the certified-candidate fence attests the frame the stages actually
    consume.  The result satisfies :func:`validate_uk_national_frame`: the
    kernel mints the renormalization :class:`MassChangeRecord`, and the
    exported ``household_weight`` column is refreshed from the typed vector.

    Returns:
        The sampled frame and the shared sampling receipt extended with the
        normalization fields and the UK policy block.
    """

    validate_sample_fraction(fraction, label="UK sample")
    validate_sample_seed(seed, label="UK sample")
    validate_uk_national_frame(frame)

    household = frame.table("household")
    units, forced, multiplier = uk_source_family_units(frame)
    household_ids = _int_column(household["household_id"], label="household_id")
    region_values = _str_column(household[_REGION_COLUMN], label=_REGION_COLUMN)
    region_by_household = dict(
        zip(
            household_ids.tolist(),
            region_values.tolist(),
            strict=True,
        )
    )
    # The stratum is the raw canonical household's region: clone-0 quota
    # cells vary by family region, so proportionality must hold within each
    # canonical region. Channel flags vary within a source family and are
    # deliberately not strata — derived rows travel with their source.
    strata = np.asarray(
        [f"region={region_by_household[int(unit)]}" for unit in units],
        dtype=object,
    )
    sampled, receipt = sample_frame_households(
        frame,
        fraction=fraction,
        seed=seed,
        source_name="UK national",
        unit_ids=units,
        unit_strata=strata,
        forced_unit_ids=forced,
        unit_noun="source family",
        floor_context="the UK national build",
    )
    if sampled is not frame:
        full_mass = float(receipt["incoming_household_mass"])
        sampled, factor = normalize_sampled_household_mass(
            sampled, target_mass=full_mass, source_name="UK national"
        )
        receipt["normalization_factor"] = factor
        receipt["normalized_household_mass"] = float(
            sampled.weights_for("household").total
        )
    _assert_spi_replacement_quota(sampled.table("household"))
    validate_uk_national_frame(sampled)
    receipt["uk_policy"] = {
        "sampling_unit": "source_frs_family",
        "strata_columns": [
            f"{_REGION_COLUMN} (of the raw clone_index=0 canonical row)"
        ],
        "clone_multiplier": multiplier,
        "forced_retention": (
            "families carrying the argmax canonical, raw, and pre-CG "
            "household and person ids (clone-multiplier and SPI/CG offset "
            "pins)"
        ),
        "spi_replacement_quota_checked": True,
    }
    return sampled, receipt
