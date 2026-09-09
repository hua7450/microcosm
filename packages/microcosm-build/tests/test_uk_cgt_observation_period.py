"""Dated CGT measurements retain base-year rows through both calibration paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_calibration import UKNationalCalibrationStage
from microcosm.build.uk_runtime.national_doctrine import UKNationalSolveDoctrine
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind


def _frame():
    ids = np.arange(3, dtype=np.int64)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [40, 50, 60],
                "capital_gains": [3000.0, 5000.0, 20000.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [2.0, 3.0, 5.0],
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


def _registry():
    return TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="person",
                measure=measure,
                value=value,
                period=2025,
                source="synthetic observation-year integration fixture",
                family="hmrc_cgt",
                metadata={"contract_target_id": name, "ledger_fact_period": "2024"},
            )
            for name, measure, value in [
                ("hmrc.cgt.taxpayers_total", "hmrc/cgt_taxpayers", 16.0),
                ("hmrc.cgt.gains_total", "hmrc/capital_gains_total", 230000.0),
                ("hmrc.cgt.liability_total", "hmrc/cgt_liability", 32760.0),
            ]
        ],
        country="uk",
    )


class _Simulation:
    def __init__(self):
        self.calls = []
        self.tax_benefit_system = SimpleNamespace(
            variables={
                name: SimpleNamespace(
                    entity=SimpleNamespace(key="person"), value_type=float
                )
                for name in ("capital_gains", "capital_gains_tax")
            }
        )

    def calculate(self, variable, year):
        self.calls.append((variable, year))
        gains = np.array([3000.0, 5000.0, 20000.0]) * (1 if year == 2024 else 1.1)
        return (
            gains if variable == "capital_gains" else np.maximum(gains - 3000, 0) * 0.18
        )


def _assert_export(original, fitted, path):
    # Both source bytes and output period survive; fitted weights do not revert.
    assert fitted.metadata["time_period"] == "2024"
    for entity in original.entities:
        columns = [c for c in original.table(entity) if c != "household_weight"]
        pd.testing.assert_frame_equal(
            original.table(entity)[columns], fitted.table(entity)[columns]
        )
    assert not np.array_equal(
        original.weights_for("household").values, fitted.weights_for("household").values
    )
    write_uk_national_frame(fitted, path)
    reread, _ = load_uk_national_frame(path)
    assert reread.metadata["time_period"] == "2024"
    np.testing.assert_array_equal(
        reread.table("person").capital_gains, original.table("person").capital_gains
    )
    np.testing.assert_array_equal(
        reread.weights_for("household").values, fitted.weights_for("household").values
    )
    assert not any(
        str(c).startswith("cgt_2024_") or "/" in str(c) for c in reread.table("person")
    )


@pytest.mark.parametrize("route", ["national", "local"])
def test_dated_cgt_fit_restores_base2024_values_with_fitted_weights(
    monkeypatch, tmp_path, route
):
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    original = _frame()
    simulation = _Simulation()
    monkeypatch.setattr(
        measure_simulation,
        "_policyengine_uk_module",
        lambda: SimpleNamespace(__version__="test"),
    )

    def resolver_factory(**kwargs):
        kwargs.setdefault("simulation_source", None)
        return UKMeasureResolver(
            **kwargs, microsimulation_factory=lambda **_: simulation
        )

    registry = _registry()
    if route == "national":
        resolver = resolver_factory(
            frame=original, scratch_dir=tmp_path / "engine", year=2025
        )
        stage = UKNationalCalibrationStage(
            registry,
            band_edge_registry=registry,
            period=2025,
            doctrine=UKNationalSolveDoctrine(epochs=2),
            measure_resolver=resolver,
        )
        fitted = stage(original)
        receipt = stage.manifest["measure_resolution"]["provider"][
            "cgt_period_contract"
        ]
    else:
        from microcosm.build.uk_runtime.local_rowwise import (
            build_uk_rowwise_local_matrix,
            solve_uk_rowwise_weights_under_doctrine,
        )

        spec = importlib.util.spec_from_file_location(
            "cgt_rowwise_builder",
            Path(__file__).resolve().parents[3] / "tools/build_uk_rowwise_candidate.py",
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        monkeypatch.setattr(
            builder,
            "compute_household_metrics",
            lambda _sim, _area, *, period, household_ids: pd.DataFrame(
                {"households": np.ones(len(household_ids))}, index=household_ids
            ),
        )
        prepared, restore, national, metrics, engine_receipt = (
            builder._resolve_candidate_engine_surface(
                original,
                registry,
                period=2025,
                scratch_dir=tmp_path / "engine",
                resolver_factory=resolver_factory,
            )
        )
        receipt = engine_receipt["cgt_period_contract"]
        local = build_uk_rowwise_local_matrix(
            metrics["constituency"],
            pd.Series(["A", "A", "A"], index=[0, 1, 2]),
            pd.DataFrame({"code": ["A"], "households": [20.0]}),
        )
        result = solve_uk_rowwise_weights_under_doctrine(
            prepared,
            local,
            bound_families=["census_households/constituency", "national/hmrc_cgt"],
            national_rows=national,
            restore=restore,
            epochs=2,
        )
        fitted = result.frame
    assert all(spec.period == 2025 for spec in registry.specs)
    assert receipt["input_period"] == "2024"
    assert receipt["calibration_period"] == 2025
    assert receipt["bound_measurements"] == {
        "cgt_2024_gains": {
            "model_variable": "capital_gains",
            "measurement_period": 2024,
        },
        "cgt_2024_tax": {
            "model_variable": "capital_gains_tax",
            "measurement_period": 2024,
        },
    }
    assert set(simulation.calls) == {
        ("capital_gains", 2024),
        ("capital_gains_tax", 2024),
    }
    _assert_export(original, fitted, tmp_path / f"{route}.h5")


@pytest.mark.requires_uk
def test_actual_engine_resolves_observed_cgt_without_relabelling_source(tmp_path):
    frame = _frame()
    resolver = UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=2025
    )
    gains, route = resolver.compute("person", "cgt_2024_gains")
    tax, _ = resolver.compute("person", "cgt_2024_tax")
    np.testing.assert_array_equal(gains, frame.table("person").capital_gains)
    np.testing.assert_array_equal(
        tax, np.asarray(resolver.simulation.calculate("capital_gains_tax", 2024))
    )
    assert route == "engine_period:2024:capital_gains:native"
    assert (gains > 3000).tolist() == [False, True, True]
    assert tax[0] == 0 and np.all(tax[1:] > 0)
    # A separate forward calculation crosses the fixed AEA; it is not the fit.
    future, _ = resolver.compute("person", "cgt_2025_gains")
    assert future[0] > 3000
    assert frame.metadata["time_period"] == "2024"
    np.testing.assert_array_equal(
        frame.table("person").capital_gains, [3000, 5000, 20000]
    )
