"""Compile the full current national matrix for a pinned UC development replay.

This writes private diagnostic matrices and aggregate row estimates. It does not
export calibrated microdata or issue a release certificate. Reviewed exclusions
and the complete band-edge register are applied through the production route.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from diagnose_uk_uc_relationships import sha256, variable_overlay
from scipy import sparse

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.target_materialization import resolve_target_measures
from microcosm.build.uk_runtime.ledger_targets import (
    compile_uk_target_registry,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_calibration import (
    CalibrationFrameAdapter,
    drop_injected_measure_inputs,
    inject_measure_inputs,
)
from microcosm.build.uk_runtime.national_doctrine import uk_national_target_loss_weights
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask
from microcosm.calibrate import calibrate
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import WeightKind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--facts-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.replay_dir / "aggregate.json").read_text())
    inputs = {Path(k): v for k, v in receipt["input_sha256"].items()}
    for path, digest in inputs.items():
        assert sha256(path) == digest, path
    paths = {p.name: p for p in inputs}
    person, benunit, household = [
        pd.read_pickle(paths[f"{name}.pkl"])
        for name in ("person", "benunit", "household")
    ]
    inherited = np.load(paths["solve.npz"], allow_pickle=False)
    if receipt["explicit_frs_claimant_input"]:
        person["is_uc_claimant"] = frs_uc_claimant_mask(person, benunit)
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=receipt["source_year"],
        household_weights=inherited["initial_weights"],
        weight_kind=WeightKind.IMPORTANCE,
    )
    factory = None
    if receipt["model_overlay"]:
        from policyengine_uk import Microsimulation

        overlay_paths = []
        for item in receipt["model_overlay"]:
            path = Path(item["path"])
            assert sha256(path) == item["sha256"], path
            overlay_paths.append(path)
        scenario, _ = variable_overlay(overlay_paths)

        def factory(**kwargs):
            return Microsimulation(**kwargs, scenario=scenario)

    resolver = UKMeasureResolver(
        simulation_source=None,
        frame=frame,
        year=receipt["model_year"],
        scratch_dir=args.output_dir / "engine",
        microsimulation_factory=factory,
    )
    old = pd.read_pickle(args.replay_dir / "benefit_units_private.pkl")
    np.testing.assert_array_equal(
        resolver.compute("benunit", "universal_credit")[0], old.universal_credit
    )
    artifact = load_ledger_consumer_artifact(
        args.ledger_dir,
        expected_facts_sha256=args.facts_sha256,
        expected_manifest_sha256=args.manifest_sha256,
    )
    compilation = compile_uk_target_registry(
        artifact.facts, target_period=receipt["model_year"]
    )
    assert not compilation.unsupported, compilation.unsupported
    registry, exclusions = apply_uk_calibration_measure_exclusions(
        compilation.registry, load_uk_calibration_measure_exclusions(None)
    )
    print(f"Resolving {len(registry.specs)} active rows.", flush=True)
    resolution = resolve_target_measures(
        lambda: CalibrationFrameAdapter(frame),
        registry,
        resolver,
        period=receipt["model_year"],
        band_edge_registry=compilation.registry,
    )
    adapter = CalibrationFrameAdapter(frame)
    original_columns = {
        entity: set(table.columns) for entity, table in adapter.tables.items()
    }
    inject_measure_inputs(adapter, resolution.measure_inputs)
    materialized = materialize_uk_ledger_targets(
        adapter,
        registry,
        period=receipt["model_year"],
        band_edge_registry=compilation.registry,
    )
    assert not materialized.skipped, materialized.skipped
    drop_injected_measure_inputs(adapter, resolution.measure_inputs, original_columns)
    prepared = adapter.prepared_frame()
    targets = registry.to_target_set()
    problem = build_constraint_matrix(prepared, targets, weight_entity="household")
    assert not problem.skipped
    np.testing.assert_array_equal(problem.names, inherited["names"])
    np.testing.assert_array_equal(problem.target_vector, inherited["target_values"])
    np.testing.assert_array_equal(
        problem.initial_weights.values, inherited["initial_weights"]
    )
    coefficients = uk_national_target_loss_weights(
        [spec.family for spec in registry.specs], rule="family_equal"
    )
    sparse.save_npz(args.output_dir / "matrix.npz", problem.matrix)
    np.savez_compressed(
        args.output_dir / "inputs_private.npz",
        names=np.array(problem.names),
        target_values=problem.target_vector,
        initial_weights=problem.initial_weights.values,
        inherited_weights=inherited["weights"],
        target_loss_weights=coefficients,
        household_ids=household["household_id"].to_numpy(),
        target_families=np.array([spec.family for spec in registry.specs]),
    )
    row_estimates = pd.DataFrame(
        {
            "name": problem.names,
            "target": problem.target_vector,
            "prior": problem.matrix @ inherited["initial_weights"],
            "inherited_calibrated": problem.matrix @ inherited["weights"],
        }
    )
    run = None
    if args.epochs:
        print(f"Running diagnostic Adam for {args.epochs} epochs.", flush=True)
        result = calibrate(
            prepared,
            targets,
            weight_entity="household",
            epochs=args.epochs,
            learning_rate=0.02,
            mass="free",
            mass_reason="Development UC diagnosis on current measurement contract",
            max_weight_ratio=10,
            seed=0,
            l0_lambda=0,
            target_loss_cap=10.0,
            target_loss_weights=coefficients,
        )
        np.savez_compressed(
            args.output_dir / "weights_private.npz", weights=result.weights
        )
        row_estimates["new_calibrated"] = problem.matrix @ result.weights
        run = {
            "epochs": args.epochs,
            "learning_rate": 0.02,
            "seed": 0,
            "mass": "free",
            "max_weight_ratio": 10,
            "l0_lambda": 0,
            "target_loss_cap": 10.0,
            "target_weight_rule": "family_equal",
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
        }
    row_estimates.to_json(
        args.output_dir / "row_estimates.json", orient="records", indent=2
    )
    for path, digest in inputs.items():
        assert sha256(path) == digest, path
    (args.output_dir / "receipt.json").write_text(
        json.dumps(
            {
                "purpose": "Development matrix/solver diagnostic; no certified dataset export",
                "replay_receipt_sha256": sha256(args.replay_dir / "aggregate.json"),
                "tool_sha256": sha256(Path(__file__)),
                "target_contract_sha256": sha256(
                    Path(__file__).resolve().parents[1]
                    / "packages/microcosm-build/src/microcosm/build/uk/uk_population_targets.json"
                ),
                "matrix_sha256": sha256(args.output_dir / "matrix.npz"),
                "facts_sha256": args.facts_sha256,
                "manifest_sha256": args.manifest_sha256,
                "compiled_rows": len(compilation.registry.specs),
                "active_rows": len(problem.names),
                "exclusions": exclusions,
                "target_values_and_roster_unchanged": True,
                "measure_resolver": resolver.receipt(),
                "solver": run,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(problem.names), "solver": run}), flush=True)


if __name__ == "__main__":
    main()
