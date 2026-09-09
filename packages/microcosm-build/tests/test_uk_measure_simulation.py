from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    compute_uk_measure_input,
    load_uk_calibration_measure_exclusions,
)
from microcosm.calibrate import TargetRegistry, TargetSpec


class FrameStub:
    def __init__(self):
        self._tables = {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "person_benunit_id": [10, 10, 20],
                    "person_household_id": [100, 100, 200],
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": [10, 20]}),
            "household": pd.DataFrame({"household_id": [100, 200]}),
        }

    def table(self, entity):
        return self._tables[entity]


def _stub_value_type(values):
    """The policyengine ``value_type`` a real variable definition would carry."""

    return {"f": float, "i": int, "b": bool}.get(np.asarray(values).dtype.kind, str)


class SimulationStub:
    def __init__(self, values):
        self.values = values
        variables = {
            name: SimpleNamespace(
                entity=SimpleNamespace(key=entity),
                value_type=_stub_value_type(stub_values),
            )
            for name, (entity, stub_values) in values.items()
        }
        self.tax_benefit_system = SimpleNamespace(variables=variables)
        self.calls = []

    def calculate(self, variable, year, map_to=None):
        self.calls.append((variable, year, map_to))
        key = (variable, map_to)
        if key in self.values:
            return self.values[key][1]
        return self.values[variable][1]


@pytest.mark.parametrize("missing_variable", [True, False])
def test_direct_measure_resolver_refuses_dropped_uc_claimant_input(
    monkeypatch, tmp_path, missing_variable
):
    frame = FrameStub()
    frame.table("person")["is_uc_claimant"] = [True, False, True]
    simulation = SimulationStub(
        {"is_uc_claimant": ("person", np.array([True, False, True]))}
    )
    simulation.input_variables = []
    if missing_variable:
        simulation.tax_benefit_system.variables.clear()
    else:
        simulation.tax_benefit_system.variables[
            "is_uc_claimant"
        ].definition_period = "year"
    monkeypatch.setattr(
        measure_simulation,
        "_policyengine_uk_module",
        lambda: SimpleNamespace(__version__="2.94.0"),
    )
    with pytest.raises(ValueError, match="is_uc_claimant"):
        UKMeasureResolver(
            simulation_source=tmp_path / "synthetic.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
            microsimulation_factory=lambda **kwargs: simulation,
        )


def test_compute_uk_measure_input_native_route():
    sim = SimulationStub({"income_tax": ("person", np.array([1.0, 2.0, 3.0]))})

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "person", "income_tax", 2025
    )

    assert route == "native"
    assert values.tolist() == [1.0, 2.0, 3.0]


def test_compute_uk_measure_input_categorical_broadcast_group_to_person():
    sim = SimulationStub({"family_type": ("benunit", np.array(["couple", "single"]))})

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "person", "family_type", 2025
    )

    assert route == "categorical_broadcast_benunit_to_person"
    assert values.tolist() == ["couple", "couple", "single"]


def test_compute_uk_measure_input_bool_any_collapse_person_to_group():
    sim = SimulationStub({"is_disabled": ("person", np.array([False, True, False]))})

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "household", "is_disabled", 2025
    )

    assert route == "bool_any_collapse_person_to_household"
    assert values.tolist() == [1.0, 0.0]


def test_compute_uk_measure_input_numeric_map_to_entity():
    sim = SimulationStub(
        {
            "benefit": ("person", np.array([1.0, 2.0, 3.0])),
            ("benefit", "household"): ("household", np.array([3.0, 3.0])),
        }
    )

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "household", "benefit", 2025
    )

    assert route == "map_to_household"
    assert values.tolist() == [3.0, 3.0]
    assert sim.calls[-1] == ("benefit", 2025, "household")


def test_compute_uk_measure_input_refuses_unknown_and_length_mismatch():
    with pytest.raises(KeyError, match="no variable"):
        compute_uk_measure_input(
            FrameStub(), SimulationStub({}), "person", "missing", 2025
        )

    sim = SimulationStub({"income_tax": ("person", np.array([1.0]))})
    with pytest.raises(ValueError, match="produced 1 values"):
        compute_uk_measure_input(FrameStub(), sim, "person", "income_tax", 2025)


def _write_exclusions(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(**overrides) -> dict:
    entry = {
        "name": "a",
        "reason": "reviewed reason",
        "tracking": "microcosm#623",
        "approved_by": "juaristi22",
        "adjudication": "microcosm#757",
        "approved_on": "2026-08-25",
        "expires_on": "2026-11-25",
    }
    entry.update(overrides)
    return entry


def test_exclusion_loader_refusals(tmp_path: Path):
    base = {"schema_version": 2, "exclusions": [_entry()]}

    with pytest.raises(ValueError, match="schema_version"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "bad-version.json", {**base, "schema_version": 1}
            )
        )
    with pytest.raises(ValueError, match="unknown top-level"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(tmp_path / "unknown.json", {**base, "extra": True})
        )
    with pytest.raises(ValueError, match="empty reason"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "empty.json",
                {**base, "exclusions": [_entry(reason="")]},
            )
        )
    with pytest.raises(ValueError, match="empty approved_by"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "unapproved.json",
                {**base, "exclusions": [_entry(approved_by="")]},
            )
        )
    with pytest.raises(ValueError, match="canonical ISO"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "sloppy-date.json",
                {**base, "exclusions": [_entry(approved_on="2026-8-25")]},
            )
        )
    with pytest.raises(ValueError, match="after approved_on"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "inverted.json",
                {**base, "exclusions": [_entry(expires_on="2026-08-25")]},
            )
        )
    with pytest.raises(
        ValueError, match="unknown UK calibration measure exclusion field"
    ):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "extra-field.json",
                {**base, "exclusions": [_entry(sneaky="value")]},
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "duplicate.json",
                {**base, "exclusions": [_entry(), _entry(reason="two")]},
            )
        )


def test_exclusion_applier_returns_pruned_registry_and_receipt():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="drop", entity="person", measure="drop", value=1.0, source="test"
            ),
            TargetSpec(
                name="keep", entity="person", measure="keep", value=1.0, source="test"
            ),
        ],
        country="uk",
    )

    from datetime import date

    window = _entry(name="drop", reason="reviewed")
    pruned, receipt = apply_uk_calibration_measure_exclusions(
        registry, (window,), now=date(2026, 9, 1)
    )

    assert [spec.name for spec in pruned.specs] == ["keep"]
    assert receipt == {
        "drop": {
            "reason": "reviewed",
            "tracking": "microcosm#623",
            "approved_by": "juaristi22",
            "adjudication": "microcosm#757",
            "approved_on": "2026-08-25",
            "expires_on": "2026-11-25",
            "evaluated_on": "2026-09-01",
        }
    }
    with pytest.raises(ValueError, match="matched zero"):
        apply_uk_calibration_measure_exclusions(
            registry, (_entry(name="stale"),), now=date(2026, 9, 1)
        )
    # Outside the reviewed window the run refuses with correct-or-renew —
    # the narrowing neither lapses silently nor lives forever.
    with pytest.raises(ValueError, match="correct the underlying gap"):
        apply_uk_calibration_measure_exclusions(
            registry, (window,), now=date(2026, 11, 26)
        )
    with pytest.raises(ValueError, match="correct the underlying gap"):
        apply_uk_calibration_measure_exclusions(
            registry, (window,), now=date(2026, 8, 24)
        )


def test_exclusion_applier_warns_within_week_of_expiry():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="drop", entity="person", measure="drop", value=1.0, source="test"
            ),
        ],
        country="uk",
    )

    from datetime import date

    window = _entry(name="drop", reason="reviewed")
    # expires_on is 2026-11-25: five days out warns, mid-window is silent.
    with pytest.warns(UserWarning, match="within one week"):
        apply_uk_calibration_measure_exclusions(
            registry, (window,), now=date(2026, 11, 20)
        )

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        apply_uk_calibration_measure_exclusions(
            registry, (window,), now=date(2026, 9, 1)
        )


#: The adjudicated register census: one entry per class of the #757
#: ``uk_target_fit`` dispositions (issue comment 5427936411) plus the
#: standing salary-sacrifice adjudication, as revised by the #807
#: excluded-cell support re-measurement (spine-i and spine-j receipts):
#: the three mid-band "sparse cell" entries whose measured support was in
#: the hundreds are lifted. The counts are the record of what was signed;
#: a drifting count is a register change that must be re-adjudicated,
#: never absorbed.
_PACKAGED_EXCLUSION_CENSUS = {
    "hmrc.salary_sacrifice.": 5,
    "_1_000_000_to_inf": 11,
    "slc.": 5,
    "dwp/uc_payment_dist/": 16,
    "obr.universal_credit_": 2,
    "ons.household_composition.": 3,
    "obr.fuel_duties": 1,
    # microcosm#762 A16 (2026-09-03): the rows the spine cannot reach by
    # reweighting — savings interest, housing benefit, the two plan-2
    # borrower stocks and JSA claimants — windowed to one month. The two
    # ONS land rows first listed were withdrawn the same day (receipt R15):
    # their apparent 8.7x / 2.5x misses were an artefact of per-block engine
    # resolution, and single-block resolution puts them within 3 % and 9 %.
    "ons.savings_interest_income": 1,
    "obr.housing_benefit": 1,
    "dwp.jsa_claimants": 1,
}

_A16_UNREACHABLE_ROWS = (
    "ons.savings_interest_income",
    "obr.housing_benefit",
    "slc.borrowers.plan_2_liable",
    "slc.borrowers.plan_2_above_threshold",
    "dwp.jsa_claimants",
)


def test_packaged_exclusions_load():
    exclusions = load_uk_calibration_measure_exclusions()
    names = [entry["name"] for entry in exclusions]
    assert len(names) == len(set(names)) == 49

    for marker, expected in _PACKAGED_EXCLUSION_CENSUS.items():
        matched = [name for name in names if marker in name]
        assert len(matched) == expected, (marker, matched)
    # The sparse HMRC band cells are whatever remains: hmrc/ band cells
    # that are not the eleven 1m+ channel cells. After the #807 revision
    # only the three genuinely thin 500k-1m cells stay excluded.
    sparse = [
        name
        for name in names
        if name.startswith("hmrc/") and "_1_000_000_to_inf" not in name
    ]
    assert sorted(sparse) == [
        "hmrc/dividend_income_income_band_500_000_to_1_000_000",
        "hmrc/property_income_count_income_band_500_000_to_1_000_000",
        "hmrc/self_employment_income_count_income_band_500_000_to_1_000_000",
    ], sparse

    # The 2026-08-26 tranche carries the uk_target_fit disposition
    # adjudication and a uniform three-month window; the ONS composition
    # cells track the relationship-to-head successor issue.
    tranche = [e for e in exclusions if e["approved_on"] == "2026-08-26"]
    assert len(tranche) == 39
    for entry in tranche:
        assert "5427936411" in entry["adjudication"], entry["name"]
        assert entry["expires_on"] == "2026-11-26", entry["name"]
    for entry in exclusions:
        if entry["name"].startswith("ons.household_composition."):
            assert entry["tracking"] == "microcosm#791", entry["name"]

    # The 2026-09-03 tranche is #762's A16: five unreachable national rows,
    # a one-month window, each row tracked on its spine-defect issue.
    a16 = [e for e in exclusions if e["approved_on"] == "2026-09-03"]
    assert sorted(e["name"] for e in a16) == sorted(_A16_UNREACHABLE_ROWS)
    # Each row points at its own spine-defect issue (the October expiry follows
    # the pointer); #736 carries the tranche as a whole.
    a16_issues = {
        "ons.savings_interest_income": "microcosm#866",
        "obr.housing_benefit": "microcosm#867",
        "slc.borrowers.plan_2_liable": "microcosm#868",
        "slc.borrowers.plan_2_above_threshold": "microcosm#868",
        "dwp.jsa_claimants": "microcosm#869",
    }
    for entry in a16:
        assert entry["expires_on"] == "2026-10-03", entry["name"]
        assert entry["tracking"] == a16_issues[entry["name"]], entry["name"]
        assert "A16" in entry["adjudication"], entry["name"]

    # The lever targets are deliberately NOT excluded: the six UC
    # caseload / two-child-limit cells ride the would_claim_uc lever run.
    # Exclusions cannot move surviving cells because band edges are pinned
    # to the compiled register (#792); the non-excluded cells are expected
    # to pass at published band widths on the next seam run. The three
    # cells the #807 revision lifted (measured support in the hundreds)
    # are back on the calibrated surface with them.
    excluded = set(names)
    for riding in (
        "dwp.uc.households",
        "dwp.uc.households_single_no_children",
        "dwp.uc.two_child_limit.children_disabled_child_element",
        "ons.household_composition.couple_no_children_households",
        "hmrc/state_pension_income_band_40_000_to_50_000",
        "hmrc/state_pension_income_band_50_000_to_70_000",
        "hmrc/self_employment_income_income_band_50_000_to_70_000",
        "hmrc/private_pension_income_count_income_band_100_000_to_150_000",
    ):
        assert riding not in excluded, riding


def test_measure_resolver_direct_and_scratch_receipts(monkeypatch, tmp_path: Path):
    created = []

    class FakeMicrosimulation:
        def __init__(self, *, dataset):
            created.append(dataset)
            self.tax_benefit_system = SimpleNamespace(variables={})

    fake_policyengine_uk = SimpleNamespace(
        __version__="9.9.9", Microsimulation=FakeMicrosimulation
    )
    monkeypatch.setitem(sys.modules, "policyengine_uk", fake_policyengine_uk)
    writes = []
    monkeypatch.setattr(
        measure_simulation,
        "write_uk_national_frame",
        lambda frame, path: writes.append((frame, path)) or path,
    )

    frame = FrameStub()
    direct = UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=frame,
    )
    scratch = UKMeasureResolver(
        simulation_source=None,
        scratch_dir=tmp_path,
        year=2025,
        frame=frame,
    )

    assert direct.receipt()["mode"] == "direct_h5"
    assert direct.receipt()["policyengine_uk_version"] == "9.9.9"
    assert scratch.receipt()["mode"] == "scratch_frame_export"
    assert writes == [(frame, tmp_path / "simulation-input.h5")]
    assert created == [
        str(tmp_path / "input.h5"),
        str(tmp_path / "simulation-input.h5"),
    ]


def _resolver_over(sim, monkeypatch, tmp_path: Path) -> UKMeasureResolver:
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=lambda *, dataset: sim),
    )
    return UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=FrameStub(),
    )


def test_knows_answers_only_for_routes_compute_can_take(monkeypatch, tmp_path: Path):
    sim = SimulationStub(
        {
            "income_tax": ("person", np.array([1.0, 2.0, 3.0])),
            "is_disabled": ("person", np.array([False, True, False])),
            "family_type": ("benunit", np.array(["couple", "single"])),
            "benunit_income": ("benunit", np.array([1.0, 2.0])),
        }
    )
    resolver = _resolver_over(sim, monkeypatch, tmp_path)

    assert resolver.knows("person", "income_tax")
    assert resolver.knows("household", "income_tax")
    assert resolver.knows("household", "is_disabled")
    assert resolver.knows("person", "family_type")
    # Numeric group-to-group rides the map_to route…
    assert resolver.knows("household", "benunit_income")
    # …but a categorical one has no benunit-to-household mapping at all, and
    # the fence must say so rather than let compute raise mid-resolution.
    assert not resolver.knows("household", "family_type")
    assert not resolver.knows("person", "no_such_variable")
    assert not resolver.knows("firm", "income_tax")


def test_unknown_categorical_mapping_refuses_through_the_fence(
    monkeypatch, tmp_path: Path
):
    sim = SimulationStub({"family_type": ("benunit", np.array(["couple", "single"]))})
    resolver = _resolver_over(sim, monkeypatch, tmp_path)

    assert not resolver.knows("household", "family_type")
    with pytest.raises(KeyError, match="no categorical mapping"):
        resolver.compute("household", "family_type")


def test_exclusion_loader_requires_tracking(tmp_path: Path):
    # An exclusion narrows the calibrated target surface; the register has to
    # say where each narrowing is being resolved, not only why.
    with pytest.raises(ValueError, match="empty tracking"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "untracked.json",
                {"schema_version": 2, "exclusions": [_entry(tracking="")]},
            )
        )
