"""UK scale-ladder sampling policy (#627): source families, strata, quota."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.national_sampling import (
    sample_uk_national_frame,
    sample_uk_spine_frame,
    uk_source_family_units,
    uk_spine_source_family_units,
)
from microcosm.frame import Frame

_RAW = (
    {"id": 101, "region": "london", "weight": 10.0},
    {"id": 102, "region": "north", "weight": 10.0},
    {"id": 103, "region": "london", "weight": 10.0},
    {"id": 104, "region": "north", "weight": 10.0},
    {"id": 105, "region": "london", "weight": 10.0},
    {"id": 106, "region": "north", "weight": 10.0},
)


def _source_family_frame(
    *,
    spi_of: tuple[int, ...] = (103, 104),
    cg_of: tuple[int, ...] = (101,),
    spi_region_override: str | None = None,
    corrupt_spi_household: int | None = None,
) -> Frame:
    """Build the two-layer synthetic compact the sampler's arithmetic expects.

    Raw canonical households get one person each (``household_id * 1000 + 1``,
    satisfying the stage fence's ``person // 1000 == household`` invariant so
    these frames are fence-valid, not just sampler-valid);
    SPI/CG derivatives are constructed with the fence's max-derived offsets
    (``max(canonical raw id) + 1`` / ``max(canonical pre-CG id) + 1``); every
    canonical row then gets one geography clone at ``+ clone_multiplier``.
    """

    rows: list[dict[str, object]] = []
    for family in _RAW:
        rows.append(
            {
                "household_id": int(family["id"]),
                "person_id": int(family["id"]) * 1000 + 1,
                "region": family["region"],
                "spi": False,
                "cg": False,
                "weight": float(family["weight"]),
            }
        )
    raw_by_id = {row["household_id"]: row for row in rows}
    spi_household_offset = max(int(f["id"]) for f in _RAW) + 1
    spi_person_offset = max(int(f["id"]) * 1000 + 1 for f in _RAW) + 1
    for source_id in spi_of:
        source = raw_by_id[source_id]
        rows.append(
            {
                "household_id": source_id + spi_household_offset,
                "person_id": int(source["person_id"]) + spi_person_offset,
                "region": (
                    spi_region_override
                    if spi_region_override is not None
                    else source["region"]
                ),
                "spi": True,
                "cg": False,
                "weight": 0.0,
            }
        )
    cg_household_offset = max(int(r["household_id"]) for r in rows) + 1
    cg_person_offset = max(int(r["person_id"]) for r in rows) + 1
    for source_id in cg_of:
        source = raw_by_id[source_id]
        rows.append(
            {
                "household_id": source_id + cg_household_offset,
                "person_id": int(source["person_id"]) + cg_person_offset,
                "region": source["region"],
                "spi": False,
                "cg": True,
                "weight": float(source["weight"]),
            }
        )
    if corrupt_spi_household is not None:
        for row in rows:
            if row["spi"]:
                row["household_id"] = corrupt_spi_household
                break
    canonical_max = max(
        max(int(r["household_id"]) for r in rows),
        max(int(r["person_id"]) for r in rows),
    )
    multiplier = 10 ** max(1, len(str(canonical_max)))
    clone_rows = [
        {
            **row,
            "household_id": int(row["household_id"]) + multiplier,
            "person_id": int(row["person_id"]) + multiplier,
            "region": "scotland",
        }
        for row in rows
    ]
    all_rows = sorted(rows + clone_rows, key=lambda row: row["household_id"])
    household_ids = np.asarray(
        [row["household_id"] for row in all_rows], dtype=np.int64
    )
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "clone_index": [
                1 if int(row["household_id"]) > multiplier else 0 for row in all_rows
            ],
            "region": [row["region"] for row in all_rows],
            "household_is_spi_synthetic": [row["spi"] for row in all_rows],
            "household_is_capital_gains_clone": [row["cg"] for row in all_rows],
            "household_weight": [row["weight"] for row in all_rows],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.asarray(
                [row["person_id"] for row in all_rows], dtype=np.int64
            ),
            "person_benunit_id": household_ids + 1_000_000_000,
            "person_household_id": household_ids,
        }
    )
    benunit = pd.DataFrame({"benunit_id": household_ids + 1_000_000_000})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


def _spine_family_frame(
    *,
    families_per_region: int = 4,
    orphan_source: int | None = None,
    multi_region_family: bool = False,
    fractional_source: bool = False,
) -> Frame:
    """Build a tiny spine with explicit raw-family lineage and derivatives."""

    raw = [
        *[
            (household_id, "london")
            for household_id in range(1, 1 + families_per_region)
        ],
        *[
            (household_id, "north")
            for household_id in range(
                1 + families_per_region, 1 + 2 * families_per_region
            )
        ],
    ]
    north_source = 1 + families_per_region
    rows = [
        {
            "household_id": household_id,
            "source_household_id": household_id,
            "region": region,
            "support_clone": 0,
            "spi": False,
            "cgt": False,
            "band_donor": False,
            "weight": 10.0,
        }
        for household_id, region in raw
    ]
    rows.extend(
        [
            {
                "household_id": 10_001,
                "source_household_id": 1,
                "region": "london",
                "support_clone": 0,
                "spi": True,
                "cgt": False,
                "band_donor": False,
                "weight": 2.0,
            },
            {
                "household_id": 10_002,
                "source_household_id": 1,
                "region": "london",
                "support_clone": 1,
                "spi": False,
                "cgt": False,
                "band_donor": False,
                "weight": 2.0,
            },
            {
                "household_id": 20_001,
                "source_household_id": north_source,
                "region": "north",
                "support_clone": 0,
                "spi": False,
                "cgt": True,
                "band_donor": False,
                "weight": 2.0,
            },
            {
                "household_id": 20_002,
                "source_household_id": north_source,
                "region": "north",
                "support_clone": 0,
                "spi": False,
                "cgt": False,
                "band_donor": True,
                "weight": 2.0,
            },
        ]
    )
    if orphan_source is not None:
        rows[-1]["source_household_id"] = orphan_source
    if multi_region_family:
        rows.append(
            {
                "household_id": 30_001,
                "source_household_id": 1,
                "region": "north",
                "support_clone": 0,
                "spi": False,
                "cgt": False,
                "band_donor": False,
                "weight": 1.0,
            }
        )
    household = pd.DataFrame(
        {
            "household_id": [row["household_id"] for row in rows],
            "source_household_id": [row["source_household_id"] for row in rows],
            "source_household_key": [
                f"2023:{row['source_household_id']}" for row in rows
            ],
            "household_source_id": [row["source_household_id"] for row in rows],
            "household_support_clone_index": [row["support_clone"] for row in rows],
            "household_is_spi_synthetic": [row["spi"] for row in rows],
            "household_is_capital_gains_clone": [row["cgt"] for row in rows],
            "household_is_cgt_band_donor": [row["band_donor"] for row in rows],
            "region": [row["region"] for row in rows],
            "household_weight": [row["weight"] for row in rows],
        }
    )
    if fractional_source:
        household["source_household_id"] = household["source_household_id"].astype(
            float
        )
        household.loc[household.index[-1], "source_household_id"] += 0.5
    household_ids = household["household_id"].to_numpy(dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": household_ids + 1_000,
            "person_benunit_id": household_ids + 2_000,
            "person_household_id": household_ids,
        }
    )
    benunit = pd.DataFrame({"benunit_id": household_ids + 2_000})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


def _units_by_household(frame: Frame) -> dict[int, int]:
    units, _forced, _multiplier = uk_source_family_units(frame)
    household_ids = frame.table("household")["household_id"].to_numpy()
    return dict(zip(household_ids.tolist(), units.tolist(), strict=True))


def test_source_units_mirror_the_stage_fence_arithmetic() -> None:
    frame = _source_family_frame()
    units, forced, multiplier = uk_source_family_units(frame)
    mapping = _units_by_household(frame)
    # SPI derivatives reverse to their raw sources through max(raw)+1 = 107.
    assert mapping[210] == 103
    assert mapping[211] == 104
    # The CG derivative reverses through max(pre-CG)+1 = 212.
    assert mapping[313] == 101
    # Clones reverse through the multiplier first.
    assert mapping[multiplier + 210] == 103
    assert mapping[multiplier + 313] == 101
    assert set(units.tolist()) == {101, 102, 103, 104, 105, 106}
    # Forced pins: the argmax canonical ids live in the CG derivative of
    # family 101, the raw argmaxes in 106, the pre-CG argmaxes in 104.
    assert forced == (101, 104, 106)


def test_sampling_keeps_source_families_whole() -> None:
    frame = _source_family_frame()
    full_mapping = _units_by_household(frame)
    rows_by_unit: dict[int, set[int]] = {}
    for household_id, unit in full_mapping.items():
        rows_by_unit.setdefault(unit, set()).add(household_id)
    for seed in (0, 1, 7, 42):
        sampled, receipt = sample_uk_national_frame(frame, fraction=0.5, seed=seed)
        sampled_ids = set(sampled.table("household")["household_id"].tolist())
        sampled_units = {full_mapping[value] for value in sampled_ids}
        for unit in sampled_units:
            assert rows_by_unit[unit] <= sampled_ids
        # Forced retention keeps every pin family at every seed.
        assert {101, 104, 106} <= sampled_units
        assert receipt["uk_policy"]["sampling_unit"] == "source_frs_family"


def test_derived_rows_never_survive_without_their_source() -> None:
    """Regression for the smoke-rung failure: SPI/CG ids must keep reversing.

    The first credentialed 1% run failed stage one with "Candidate
    SPI/capital-gains person IDs do not reverse to the raw FRS surface"
    because clone-family units let a derivative survive without its raw
    source. Source-family units make that structurally impossible.
    """

    frame = _source_family_frame()
    full_mapping = _units_by_household(frame)
    for seed in (0, 3, 11):
        sampled, _receipt = sample_uk_national_frame(frame, fraction=0.5, seed=seed)
        household = sampled.table("household")
        sampled_ids = set(household["household_id"].tolist())
        derived = household.loc[
            household["household_is_spi_synthetic"]
            | household["household_is_capital_gains_clone"]
        ]
        for household_id in derived["household_id"].tolist():
            assert full_mapping[int(household_id)] in sampled_ids


def test_sampled_frame_is_normalized_refreshed_and_valid() -> None:
    frame = _source_family_frame()
    full_mass = float(frame.weights_for("household").total)
    sampled, receipt = sample_uk_national_frame(frame, fraction=0.5, seed=3)
    validate_uk_national_frame(sampled)
    weights = sampled.weights_for("household")
    assert float(weights.total) == pytest.approx(full_mass)
    assert "household_weight" not in sampled.table("household")
    record = sampled.mass_log[-1]
    assert record.entity == "household"
    assert "composition-preserving" in record.reason
    assert receipt["normalized_household_mass"] == pytest.approx(full_mass)


def test_numeric_region_codes_are_refused() -> None:
    """Region strata labels are formatted values, so an integer code in one
    build and a float code in another would silently split a stratum; the
    sampler accepts string regions only."""

    frame = _source_family_frame()
    household = frame.table("household").copy()
    household["region"] = np.arange(len(household)) % 3
    numeric = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2023",
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(ValueError, match="must contain non-empty strings"):
        sample_uk_national_frame(numeric, fraction=0.5, seed=1)


def test_strata_are_the_raw_canonical_regions() -> None:
    frame = _source_family_frame()
    _, receipt = sample_uk_national_frame(frame, fraction=0.5, seed=11)
    strata = receipt["strata"]
    assert set(strata) == {"region=london", "region=north"}
    assert strata["region=london"]["eligible_units"] == 3
    assert strata["region=london"]["requested_units"] == 1
    assert strata["region=north"]["eligible_units"] == 3
    assert strata["region=north"]["requested_units"] == 1


def test_spi_quota_violation_fails_closed() -> None:
    # An SPI derivative living in a region its source does not: the sampled
    # artifact carries a clone-0 wales cell with dead > base at any draw.
    frame = _source_family_frame(spi_region_override="wales")
    with pytest.raises(ValueError, match="SPI replacement quota"):
        sample_uk_national_frame(frame, fraction=0.5, seed=5)


def test_non_reversing_derivative_fails_closed() -> None:
    # Household 250 - 107 = 143 is not a raw canonical id.
    frame = _source_family_frame(corrupt_spi_household=250)
    with pytest.raises(ValueError, match="do not reverse to the raw FRS"):
        uk_source_family_units(frame)


def test_missing_lineage_columns_fail_closed() -> None:
    frame = _source_family_frame()
    household = frame.table("household").drop(columns=["clone_index"])
    stripped = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2023",
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(ValueError, match="missing \\['clone_index'\\]"):
        sample_uk_national_frame(stripped, fraction=0.5, seed=1)


def test_full_fraction_is_a_structural_no_op() -> None:
    frame = _source_family_frame()
    sampled, receipt = sample_uk_national_frame(frame, fraction=1.0, seed=42)
    assert sampled is frame
    assert "normalization_factor" not in receipt
    assert receipt["uk_policy"]["spi_replacement_quota_checked"] is True


def test_sampled_frames_pass_the_real_stage_fence() -> None:
    """The sampler's arithmetic is proven against the fence itself.

    The first credentialed rung run died because the sampler and
    ``_resolve_candidate_lineage`` disagreed; this test closes the coverage
    hole the adversarial review found by running the REAL fence over sampled
    frames: every draw must resolve with the full frame's exact multiplier
    and person-level SPI/CG offsets.
    """

    from microcosm.build.uk_runtime.frs_hmrc_leaves import (
        _resolve_candidate_lineage,
    )

    frame = _source_family_frame()
    full = _resolve_candidate_lineage(frame)
    for seed in (0, 3, 11, 42):
        sampled, _receipt = sample_uk_national_frame(frame, fraction=0.5, seed=seed)
        lineage = _resolve_candidate_lineage(sampled)
        assert lineage.clone_id_multiplier == full.clone_id_multiplier
        assert lineage.spi_person_id_offset == full.spi_person_id_offset
        assert (
            lineage.capital_gains_person_id_offset
            == full.capital_gains_person_id_offset
        )


def test_spine_source_units_use_raw_family_regions() -> None:
    frame = _spine_family_frame()
    units, strata = uk_spine_source_family_units(frame)
    household_ids = frame.table("household")["household_id"].to_numpy()
    mapping = dict(zip(household_ids.tolist(), units.tolist(), strict=True))
    assert mapping[10_001] == 1
    assert mapping[10_002] == 1
    assert mapping[20_001] == 5
    assert mapping[20_002] == 5
    assert set(strata[units == 1]) == {"region=london"}
    assert set(strata[units == 5]) == {"region=north"}


@pytest.mark.parametrize("fraction,rung", [(0.01, "f001"), (0.10, "f010")])
def test_spine_sampling_is_stratified_keeps_families_and_normalizes(
    fraction: float,
    rung: str,
) -> None:
    frame = _spine_family_frame(families_per_region=100)
    units, _ = uk_spine_source_family_units(frame)
    household_ids = frame.table("household")["household_id"].to_numpy()
    family_by_household = dict(zip(household_ids.tolist(), units.tolist(), strict=True))
    full_rows_by_family = {
        family: set(household_ids[units == family].tolist())
        for family in np.unique(units)
    }
    full_mass = float(frame.weights_for("household").total)

    sampled, receipt = sample_uk_spine_frame(frame, fraction=fraction, seed=19)

    sampled_ids = set(sampled.table("household")["household_id"].tolist())
    sampled_families = {family_by_household[value] for value in sampled_ids}
    for family in sampled_families:
        assert full_rows_by_family[family] <= sampled_ids
    assert float(sampled.weights_for("household").total) == pytest.approx(full_mass)
    assert receipt["pre_family_count"] == 200
    assert receipt["post_family_count"] == int(200 * fraction)
    assert receipt["strata_count"] == 2
    assert receipt["receipt"]["strata"]["region=london"]["requested_units"] == int(
        100 * fraction
    )
    assert receipt["receipt"]["strata"]["region=north"]["requested_units"] == int(
        100 * fraction
    )
    assert receipt["normalization_factor"] > 0.0
    assert receipt["rung_token"] == rung


def test_spine_sampling_missing_lineage_column_fails_closed() -> None:
    frame = _spine_family_frame()
    household = frame.table("household").drop(columns=["source_household_id"])
    stripped = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2023",
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(ValueError, match="source_household_id"):
        uk_spine_source_family_units(stripped)


def test_spine_sampling_fractional_source_fails_closed() -> None:
    with pytest.raises(ValueError, match="source_household_id must be integer"):
        uk_spine_source_family_units(_spine_family_frame(fractional_source=True))


def test_spine_sampling_orphan_derivative_fails_closed() -> None:
    with pytest.raises(ValueError, match="999"):
        uk_spine_source_family_units(_spine_family_frame(orphan_source=999))


def test_spine_sampling_multi_region_raw_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="spans more than one raw-row region.*1"):
        uk_spine_source_family_units(_spine_family_frame(multi_region_family=True))
