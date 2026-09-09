"""Public evidence audits fail closed without optional country engines or data."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

_BUILDER = (
    Path(__file__).resolve().parents[3]
    / "experiments/build_879_corrected_measurement_receipt.py"
)
_spec = importlib.util.spec_from_file_location("corrected_receipt", _BUILDER)
receipt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(receipt)


def _audit_inputs():
    composition = pd.DataFrame(
        {
            "benunit_id": [1, 2, 3, 4, 5, 6],
            "household_id": [20, 10, 10, 20, 20, 10],
            "family": ["SINGLE"] * 5 + ["LONE_PARENT"],
            "uc": [0.0, 0.12, 1200.0, 1200.12, 50_000.0, 100.0],
            "private_unrequested_field": ["never publish"] * 6,
        }
    )
    names = [
        "dwp/uc_payment_dist/SINGLE_annual_payment_0_to_1_200@2025",
        "dwp/uc_payment_dist/SINGLE_annual_payment_1_200_to_2_400@2025",
    ]
    bands = [
        {
            "name": name.removesuffix("@2025"),
            "family": "SINGLE",
            "active": True,
            "lower_annual": lower,
            "upper_annual": upper,
            "support_benefit_units": 2,
            "prior_estimate": prior,
            "private_unrequested_field": "never publish",
        }
        for name, lower, upper, prior in zip(
            names, [0.12, 1200.12], [1200.12, None], [6.0, 10.0], strict=True
        )
    ]
    return (
        bands,
        composition,
        np.array([10, 20]),
        names,
        sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]]),
        np.array([3.0, 5.0]),
    )


def test_saved_audit_checks_exact_household_counts_and_allowlists_output():
    rows = receipt.audit_saved_matrix(*_audit_inputs())
    assert len(rows) == 2
    assert all(row["exact_saved_matrix_match"] for row in rows)
    assert all(row["support_benefit_units"] == 2 for row in rows)
    assert all(row["nonzero_households"] == 1 for row in rows)
    assert set(rows[0]) == {
        "name",
        "family",
        "lower_annual",
        "upper_annual",
        "positive_payment_required",
        "exact_saved_matrix_match",
        "support_benefit_units",
        "nonzero_households",
    }
    assert "never publish" not in json.dumps(rows)


def test_saved_audit_rejects_different_rows_with_equal_aggregate_support():
    args = list(_audit_inputs())
    args[4] = sparse.csr_matrix([[1.0, 1.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match="Saved UC matrix row mismatch"):
        receipt.audit_saved_matrix(*args)


@pytest.mark.parametrize(
    "field,value", [("support_benefit_units", 3), ("prior_estimate", 7.0)]
)
def test_saved_audit_rejects_preflight_aggregate_drift(field, value):
    args = _audit_inputs()
    args[0][0][field] = value
    with pytest.raises(ValueError, match="Preflight .* mismatch"):
        receipt.audit_saved_matrix(*args)


def test_saved_audit_rejects_excluded_row_in_matrix():
    args = _audit_inputs()
    args[0][1]["active"] = False
    with pytest.raises(ValueError, match="Active UC matrix roster mismatch"):
        receipt.audit_saved_matrix(*args)


def test_committed_receipt_is_authenticated_before_parsing(monkeypatch, tmp_path):
    raw = b'{"proof": "historical"}'
    monkeypatch.setattr(receipt.subprocess, "check_output", lambda *args, **kwargs: raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert receipt.committed_json(tmp_path, "receipt.json", digest) == {
        "proof": "historical"
    }
    with pytest.raises(ValueError, match="Receipt hash mismatch"):
        receipt.committed_json(tmp_path, "receipt.json", "0" * 64)


def test_spine_origin_requires_all_corrected_hashes_to_match():
    run = {
        "code_pin": "measurement",
        "input_identity": {"before": "a", "after": "a", "copied_after": "a"},
    }
    origin = {
        "runs": {
            "control": {"code_pin": "spine-origin", "file_hashes": {"spine.h5": "a"}}
        }
    }
    receipt.add_spine_origin(run, origin, "control")
    assert run["code_pin"] == "measurement"
    assert run["spine_origin_code_pin"] == "spine-origin"
    run["input_identity"]["copied_after"] = "b"
    with pytest.raises(ValueError, match="Spine origin hash mismatch"):
        receipt.add_spine_origin(run, origin, "control")


@pytest.mark.parametrize(
    "tampered",
    [
        "calibration_diagnostics.json",
        "calibration_gates.json",
        "matrix_preflight.json",
        "support_after_solve.json",
        "solve.npz",
        "matrix.npz",
        "spine.h5",
        "uc_composition_private.pkl",
    ],
)
def test_builder_authenticates_every_retained_input_before_private_loading(
    monkeypatch, tmp_path, tampered
):
    files = [
        "calibration_diagnostics.json",
        "calibration_gates.json",
        "matrix_preflight.json",
        "support_after_solve.json",
        "solve.npz",
        "matrix.npz",
    ]
    digest = hashlib.sha256(b"expected").hexdigest()
    for name in files + ["spine.h5", "uc_composition_private.pkl"]:
        (tmp_path / name).write_bytes(b"expected")
    (tmp_path / tampered).write_bytes(b"tampered")
    run = {
        "artifact_sha256": dict.fromkeys(files, digest),
        "input_identity": dict.fromkeys(["before", "after", "copied_after"], digest),
    }
    baseline = {"runs": {"pub879-control": run}}
    original = {
        "runs": {
            "control": {"code_pin": "original", "file_hashes": {"spine.h5": digest}}
        }
    }
    monkeypatch.setattr(
        receipt,
        "committed_json",
        lambda repo, path, expected: copy.deepcopy(
            baseline if path == receipt.BASELINE_PATH else original
        ),
    )
    monkeypatch.setitem(receipt.COMPOSITION_SHA256, "pub879-control", digest)

    def unexpected_load(*args, **kwargs):
        raise AssertionError(
            "private inputs must not be loaded after authentication fails"
        )

    monkeypatch.setattr(receipt.pd, "read_pickle", unexpected_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        receipt.build_receipt(tmp_path, {"pub879-control": tmp_path})
