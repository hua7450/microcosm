"""Authenticate retained PR 879 evidence and add aggregate UC matrix audits.

This reads the original committed comparison receipts from an immutable git
revision, not a previous output. It preserves their numerical results and gates;
it does not run an engine, optimizer, or release. Local restricted artifacts are
required. Only allowlisted aggregate fields and content hashes enter the output.

The private composition snapshots were first hash-pinned during this review
repair. They are supplementary saved resolver output, not independently derived
families or a retrospectively authenticated pre-solve assertion. The historical
preflight, saved matrix, solve, and spine hashes already existed in committed
receipts. Independent relationship/boundary coverage lives in test_uk_uc_family.py.

    uv run --no-sync python experiments/build_879_corrected_measurement_receipt.py \
        --repo . --control /local/pub879-control --treatment /local/pub879-spi \
        --output /local/corrected-receipt.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SOURCE_COMMIT = "a20d8c31936d5d0d4d193891d2c11d74217a038f"
BASELINE_PATH = "experiments/receipts/uk-corrected-measurement-comparison.json"
ORIGIN_PATH = "experiments/receipts/uk-publication-spi-comparison.json"
BASELINE_SHA256 = "4e8bda6ba2eba3b7df56b56a565a6dd488c7d9c3739d46e4fb44bc45a41b6a93"
ORIGIN_SHA256 = "71502cc84f0cc8a896f32c5bceb7472839afa49c693ce79e2a86783f65ea52fa"
COMPOSITION_SHA256 = {
    "pub879-control": "08ab5b7507a857b6d9836c7895d7ee185acf583d96b5c27c6dc6eeb0bd197a53",
    "pub879-spi": "9f103aeffef98ec05e14dad28bb6c2fd7b5b564b40248015ced33982baa3998e",
}
FAMILIES = {"SINGLE", "LONE_PARENT", "COUPLE_NO_CHILDREN", "COUPLE_WITH_CHILDREN"}
BAND_FIELDS = (
    "name",
    "family",
    "active",
    "target",
    "lower_annual",
    "upper_annual",
    "support_benefit_units",
    "unique_source_benefit_units",
    "prior_estimate",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticate(path, expected):
    require(file_sha256(path) == expected, f"SHA-256 mismatch: {Path(path).name}")


def committed_json(repo, path, expected):
    raw = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=repo)
    require(
        hashlib.sha256(raw).hexdigest() == expected, f"Receipt hash mismatch: {path}"
    )
    return json.loads(raw)


def audit_saved_matrix(
    bands, composition, household_ids, names, matrix, initial_weights
):
    """Compare every active row with saved family/award counts, without IDs in output."""
    require(not composition.benunit_id.duplicated().any(), "Duplicate composition rows")
    require(len(set(household_ids)) == len(household_ids), "Duplicate household rows")
    require(
        set(composition.household_id) <= set(household_ids), "Unlinked benefit units"
    )
    require(set(composition.family) <= FAMILIES, "Unknown saved family")
    require(np.isfinite(composition.uc).all(), "Nonfinite saved awards")
    names = [str(name) for name in names]
    require(len(set(names)) == len(names), "Duplicate matrix names")
    require(matrix.shape == (len(names), len(household_ids)), "Matrix shape mismatch")
    require(len(initial_weights) == len(household_ids), "Initial weight shape mismatch")
    expected_names = {f"{band['name']}@2025" for band in bands if band["active"]}
    actual_names = {name for name in names if name.startswith("dwp/uc_payment_dist/")}
    require(actual_names == expected_names, "Active UC matrix roster mismatch")
    rows = []
    for band in bands:
        if not band["active"]:
            continue
        name = f"{band['name']}@2025"
        lower, upper = band["lower_annual"], band["upper_annual"]
        mask = (
            (composition.family == band["family"])
            & (composition.uc > 0)
            & (composition.uc >= lower)
        )
        if upper is not None:
            mask &= composition.uc < upper
        expected = (
            pd.Series(mask.to_numpy(dtype=float))
            .groupby(composition.household_id.to_numpy())
            .sum()
            .reindex(household_ids, fill_value=0)
            .to_numpy()
        )
        index = names.index(name)
        actual = matrix[index : index + 1].toarray().ravel()
        require(
            np.array_equal(actual, expected), f"Saved UC matrix row mismatch: {name}"
        )
        require(
            int(mask.sum()) == band["support_benefit_units"],
            f"Preflight support mismatch: {name}",
        )
        require(
            np.isclose(
                actual @ initial_weights, band["prior_estimate"], rtol=1e-12, atol=1e-8
            ),
            f"Preflight prior mismatch: {name}",
        )
        # A strict allowlist: no arbitrary preflight or private table columns.
        rows.append(
            {
                "name": name,
                "family": band["family"],
                "lower_annual": lower,
                "upper_annual": upper,
                "positive_payment_required": True,
                "exact_saved_matrix_match": True,
                "support_benefit_units": int(mask.sum()),
                "nonzero_households": int(np.count_nonzero(actual)),
            }
        )
    return rows


def add_spine_origin(run, origin, origin_key):
    source = origin["runs"][origin_key]
    digest = source["file_hashes"]["spine.h5"]
    require(
        set(run["input_identity"].values()) == {digest}, "Spine origin hash mismatch"
    )
    run["spine_origin_code_pin"] = source["code_pin"]
    run["spine_origin_receipt"] = {
        "path": ORIGIN_PATH,
        "git_revision": SOURCE_COMMIT,
        "sha256": ORIGIN_SHA256,
        "run_key": origin_key,
        "artifact": "spine.h5",
        "artifact_sha256": digest,
        "all_corrected_input_identity_hashes_match": True,
    }


def build_receipt(repo, directories):
    baseline = committed_json(repo, BASELINE_PATH, BASELINE_SHA256)
    origin = committed_json(repo, ORIGIN_PATH, ORIGIN_SHA256)
    receipt = copy.deepcopy(baseline)
    receipt["evidence_completion"] = {
        "date": "2026-09-07",
        "builder": "experiments/build_879_corrected_measurement_receipt.py",
        "baseline_receipt": {
            "path": BASELINE_PATH,
            "git_revision": SOURCE_COMMIT,
            "sha256": BASELINE_SHA256,
        },
        "calibration_rerun": False,
        "historical_results_and_gate_decisions_preserved": True,
        "historical_uc_check_scope": (
            "The run script asserted 84 active UC rows before solving using runtime-resolved "
            "families. The authenticated preflight records all 100 band inputs/supports, "
            "but did not serialize an individual exact-match assertion for UC rows."
        ),
        "saved_matrix_audit_scope": (
            "New exact row comparisons against retained composition snapshots. Snapshot hashes "
            "were first pinned during this review; family values remain saved resolver output. "
            "Independent relationship and full-register boundary tests are synthetic."
        ),
        "published_tail_boundary_correctness_claimed": False,
    }
    for label, origin_key in [
        ("pub879-control", "control"),
        ("pub879-spi", "treatment"),
    ]:
        directory = Path(directories[label])
        run = receipt["runs"][label]
        for name, digest in run["artifact_sha256"].items():
            authenticate(directory / name, digest)
        add_spine_origin(run, origin, origin_key)
        authenticate(directory / "spine.h5", run["input_identity"]["before"])
        snapshot_path = directory / "uc_composition_private.pkl"
        authenticate(snapshot_path, COMPOSITION_SHA256[label])
        preflight = json.loads((directory / "matrix_preflight.json").read_text())
        bands = [
            {field: row[field] for field in BAND_FIELDS}
            for row in preflight["payment_bands"]
        ]
        require(
            len(bands) == 100 and len({b["name"] for b in bands}) == 100,
            "Expected 100 unique UC bands",
        )
        require(sum(b["active"] for b in bands) == 84, "Expected 84 active UC bands")
        require(
            preflight["all_uc_awards_exactly_unchanged"] is True,
            "Historical award-invariance check missing",
        )
        require(
            preflight["voa"] == run["pre_solve_matrix_checks"]["voa"],
            "Historical VOA check mismatch",
        )
        for band in bands:
            published = next(
                row
                for row in receipt["uc_payment_band_support"]
                if row["name"] == band["name"]
            )
            for field in ("family", "active", "target", "lower_annual", "upper_annual"):
                require(
                    band[field] == published[field], f"Band input mismatch: {field}"
                )
            for field in (
                "support_benefit_units",
                "unique_source_benefit_units",
                "prior_estimate",
            ):
                require(
                    band[field] == published[origin_key][field],
                    f"Band aggregate mismatch: {field}",
                )
        # These pickle/PyTables inputs are loaded only after checking pinned hashes.
        composition = pd.read_pickle(snapshot_path)
        with pd.HDFStore(directory / "spine.h5", mode="r") as store:
            household_ids = store["household"].household_id.to_numpy()
        with np.load(directory / "solve.npz", allow_pickle=False) as solve:
            rows = audit_saved_matrix(
                bands,
                composition,
                household_ids,
                solve["names"],
                sparse.load_npz(directory / "matrix.npz"),
                solve["initial_weights"],
            )
        run["saved_uc_matrix_audit"] = {
            "evidence_type": "new_saved_matrix_audit",
            "audit_date": "2026-09-07",
            "preflight_sha256": run["artifact_sha256"]["matrix_preflight.json"],
            "matrix_sha256": run["artifact_sha256"]["matrix.npz"],
            "solve_sha256": run["artifact_sha256"]["solve.npz"],
            "composition_snapshot_sha256": COMPOSITION_SHA256[label],
            "composition_pin_scope": "first recorded during review repair; saved runtime resolver output",
            "spine_sha256": run["input_identity"]["before"],
            "full_band_count": 100,
            "active_rows_verified": len(rows),
            "excluded_band_count": 16,
            "rows": rows,
        }
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        args.repo, {"pub879-control": args.control, "pub879-spi": args.treatment}
    )
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        "Authenticated retained evidence; verified 84 UC rows per run; no calibration rerun."
    )


if __name__ == "__main__":
    main()
