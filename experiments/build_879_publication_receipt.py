"""Audit a matched SPI experiment and write aggregate-only local receipts.

Run with the pinned treatment venv. This helper never edits a repository or
exports a dataset. Its JSON is an experiment receipt, not a dashboard schema.

Example (all local paths are explicit; licensed inputs remain local)::

    uv run --no-sync python experiments/build_879_publication_receipt.py \
        --repo . --control /path/to/pub879-control --treatment /path/to/pub879-spi \
        --contract /path/to/publication-comparison-contract.json \
        --protected-manifest /path/to/publication-protected-pin-transition.json \
        --output /path/outside/repository/paired-receipt.json

The contract contains the reported comparison_contract fields plus
raw_source_locations_file (an acquisition manifest with repo/revision/files)
and chronicle_path (the canonical consumer artifact directory). The protected
manifest contains the reviewed protected_publication_files object. Their
portable metadata is bound by frozen canonical JSON hashes below, independently
of the generated output. Supplied files are checked against those approved pins,
the two fixed source commits and actual input bytes. Existing output is never
read, and passing a different contract cannot replace the selected pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

PUBLICATION_COMMIT = "d43b4203c6ebe10e062cb3ef3034e66731ea055d"
TREATMENT_COMMIT = "fc49b48200e1e15fe35bb4e15b948992bf03c28d"
# Reviewed experiment inputs, SHA-256 of sorted compact JSON with finite values.
# Local input locations are omitted from the portable comparison-contract pin.
CONTRACT_SHA256 = "07f0f618c7cb0606dbf4bc9841868334c976115fe8b2b17f8a654f3cd6f19283"
PROTECTED_SHA256 = "3e44dc75aca1658e4f78660dd5c0ec667e043c377662e30797828275a4955532"
OUTCOME_FIELDS = {
    "initial_estimate",
    "final_estimate",
    "relative_error",
    "within_tolerance",
    "final_capped_scaled_error",
    "final_loss_contribution",
}
NOMINAL = {
    "gift_aid",
    "charitable_investment_gifts",
    "hmrc_spi_incapacity_benefit_income",
    "hmrc_spi_other_social_security_income",
    "hmrc_spi_unemployment_benefit_income",
    "hmrc_spi_state_pension_income",
}
KEY_TARGETS = [
    "obr.income_tax@2025",
    "obr.state_pension@2025",
    "hmrc/state_pension_income_band_20_000_to_30_000@2025",
    "hmrc/state_pension_income_band_50_000_to_70_000@2025",
    "hmrc/private_pension_income_count_income_band_100_000_to_150_000@2025",
    "ons.savings_interest_income@2025",
    "dwp.uc.households@2025",
    "dwp.uc.households_single_with_children@2025",
    "dwp.uc.households_children_1@2025",
    "dwp.uc.households_children_2@2025",
    "dwp.uc.households_children_5_or_more@2025",
    "obr.capital_gains_tax@2025",
    "hmrc.cgt.gains_total@2025",
    "hmrc.cgt.taxpayers_total@2025",
    "hmrc.tfc.government_top_up@2025",
    "hmrc.tfc.children_with_used_accounts@2025",
    "dfe.funded_childcare.early_learning_2_year_olds@2025",
    "dfe.funded_childcare.working_parent_children_2_to_4@2025",
    "dfe.funded_childcare.universal_only_children@2025",
]


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def git_bytes(repo, ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=repo)


def read_run(path):
    path = Path(path).resolve()
    required = [
        "spine.h5",
        "spine.build.json",
        "spine.spine_gates.json",
        "calibration_diagnostics.json",
        "calibration_gates.json",
        "calibration_invocation.json",
        "uc_support.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete run {path.name}: {missing}")
    return {
        "path": path,
        "diagnostics": read(path / "calibration_diagnostics.json"),
        "build": read(path / "spine.build.json"),
        "gates": read(path / "calibration_gates.json"),
        "spine_gates": read(path / "spine.spine_gates.json"),
        "invocation": read(path / "calibration_invocation.json"),
        "uc_support": read(path / "uc_support.json"),
        "file_hashes": {name: file_sha(path / name) for name in required},
    }


def stable_target(row):
    return {key: value for key, value in row.items() if key not in OUTCOME_FIELDS}


def stable_resolution(resolution):
    provider = {
        key: value
        for key, value in resolution["provider"].items()
        if key != "source_path"
    }
    return {**resolution, "provider": provider}


def verify_run(run, repo, expected_code, contract, require_signature=True):
    d, build = run["diagnostics"], run["build"]
    assert d["build"]["code_pin"] == expected_code
    assert run["invocation"]["code_pin"] == expected_code
    for field in ["input_sha256_before", "input_sha256_after"]:
        assert run["invocation"][field] == run["file_hashes"]["spine.h5"]
    for field in ["input_sha256_before", "input_sha256_after"]:
        if field in run["uc_support"]:
            assert run["uc_support"][field] == run["file_hashes"]["spine.h5"]
    assert d["build"]["input_posture"]["sha256"] == run["file_hashes"]["spine.h5"]
    assert (
        d["build"]["spine_provenance"]["sidecar"]["sha256"]
        == run["file_hashes"]["spine.build.json"]
    )
    assert (
        d["build"]["spine_provenance"]["spine_gate_report"]["sha256"]
        == run["file_hashes"]["spine.spine_gates.json"]
    )
    assert (
        d["build"]["ledger"]["facts_sha256"] == contract["consumer_facts.jsonl_sha256"]
    )
    assert d["build"]["ledger"]["manifest_sha256"] == contract["manifest.json_sha256"]
    assert build["time_period"] == str(contract["build_period"])
    assert {row["period"] for row in d["targets"]} == {contract["calibration_year"]}
    lock = hashlib.sha256(git_bytes(repo, expected_code, "uv.lock")).hexdigest()
    assert lock == contract["microcosm_lock_sha256"]
    assert (
        d["build"]["runtime"]["policyengine-uk"]
        == build["rules_engine"]["version"]
        == "2.94.0"
    )
    assert d["target_surface"]["n_targets"] == len(d["targets"]) == 371
    assert d["build"]["register"] == {
        "country": "uk",
        "version": d["target_registry"]["version"],
        "compiled_count": 415,
        "excluded_count": 44,
        "calibrated_count": 371,
    }
    for key in ["method", "epochs", "learning_rate", "seed", "max_weight_ratio"]:
        assert d["options"][key] == contract["solver"][key]
    assert (
        d["build"]["doctrine"]["target_weight_rule"]
        == contract["solver"]["target_weight_rule"]
    )
    total = math.fsum(row["final_loss_contribution"] for row in d["targets"])
    assert math.isclose(total, d["final_loss"], rel_tol=1e-10, abs_tol=1e-12), (
        total,
        d["final_loss"],
    )
    attestation = run["gates"]["attestation"]
    if require_signature:
        assert attestation.get("signature") and not attestation.get("signing_error"), (
            "Selected run must have a valid producer signing step; archive unsigned attempts separately"
        )
    assert not run["gates"]["release_candidate"]
    if any(row["status"] == "failed" for row in run["gates"]["gates"].values()):
        assert not (run["path"] / "calibrated.h5").exists(), (
            "Gate-failed run unexpectedly exported calibrated.h5"
        )
    assert all(
        row["status"] != "failed" for row in run["spine_gates"]["gates"].values()
    )
    return {
        "code_pin": expected_code,
        "lock_sha256": lock,
        "target_binding_metadata_sha256": value_sha(
            [stable_target(row) for row in d["targets"]]
        ),
        "verified_loss_contribution_sum": total,
    }


def verify_raw_inputs(build, contract):
    source_manifest = read(contract["raw_source_locations_file"])
    inputs = Path(contract["raw_source_locations_file"]).parent
    paths = {
        Path(path).name: Path(path)
        for files in source_manifest["files"].values()
        for path in files
    }
    paths.update({path.name: path for path in inputs.glob("*.ods")})
    verified = {}
    for group in ["artifact_pins", "input_artifact_pins"]:
        for role, pin in build[group].items():
            candidate = pin.get("filename", pin.get("locator", role))
            name = Path(urlparse(candidate).path).name
            if name == "caller-supplied local input":
                name = role
            assert name in paths, (group, role, name)
            path = paths[name]
            if name not in verified:
                verified[name] = {
                    "sha256": file_sha(path),
                    "size_bytes": path.stat().st_size,
                }
            assert verified[name]["sha256"] == pin["sha256"], name
            assert verified[name]["size_bytes"] == pin["size_bytes"], name
    feed = Path(contract["chronicle_path"])
    assert (
        file_sha(feed / "consumer_facts.jsonl")
        == contract["consumer_facts.jsonl_sha256"]
    )
    assert file_sha(feed / "manifest.json") == contract["manifest.json_sha256"]
    return {
        "raw_acquisition_repository": source_manifest["repo"],
        "raw_acquisition_revision": source_manifest["revision"],
        "verified_files": verified,
    }


def source_quality(run_path):
    from microcosm.build.uk_runtime.national_frame import load_uk_national_frame

    path = Path(run_path) / "spine.h5"
    before = file_sha(path)
    frame, _ = load_uk_national_frame(path)
    p, b, h = (frame.table(entity) for entity in ["person", "benunit", "household"])
    hw = frame.weights_for("household").values
    pw = p.person_household_id.map(pd.Series(hw, index=h.household_id)).to_numpy()
    assert np.isfinite(pw).all()
    flows = [
        "employment_income",
        "self_employment_income",
        "private_pension_income",
        "dividend_income",
        "other_investment_income",
        "savings_interest_income",
        "tax_free_savings_income",
        "state_pension_reported",
        "hmrc_spi_state_pension_income",
    ]

    def aggregate(mask):
        count = int(mask.sum())
        if 0 < count < 3:
            return {"rows": "fewer_than_3", "suppressed": True}
        result = {"rows": count, "person_mass": float(pw[mask].sum()), "flows": {}}
        for column in flows:
            values = p[column].to_numpy(dtype=float)
            positive = mask & (values != 0)
            nonzero = int(positive.sum())
            result["flows"][column] = (
                {"suppressed": True, "nonzero_rows": "fewer_than_3"}
                if 0 < nonzero < 3
                else {
                    "nonzero_rows": nonzero,
                    "weighted_total": float(np.dot(pw[mask], values[mask])),
                }
            )
        return result

    channels = {}
    for channel in ["frs", "spi"]:
        selected = p.person_support_channel.eq(channel).to_numpy()
        channels[channel] = {
            name: aggregate(selected & mask)
            for name, mask in {
                "under_16": p.age.lt(16).to_numpy(),
                "ages_16_to_19": p.age.between(16, 19).to_numpy(),
                "age_16_plus": p.age.ge(16).to_numpy(),
            }.items()
        }
    selected = p.person_support_channel.eq("spi").to_numpy() & p.age.ge(16).to_numpy()
    mismatch = selected & (
        p.state_pension_reported.gt(0).to_numpy()
        != p.hmrc_spi_state_pension_income.gt(0).to_numpy()
    )
    n = int(mismatch.sum())
    pension = (
        {
            "mismatch_rows": "fewer_than_3",
            "suppressed": True,
            "interpretation": "predictive receipt proxy; not an entitlement or amount identity",
        }
        if 0 < n < 3
        else {
            "mismatch_rows": n,
            "mismatch_person_mass": float(pw[mismatch].sum()),
            "mismatch_share_of_spi_age_16_plus_mass": float(
                pw[mismatch].sum() / pw[selected].sum()
            ),
            "interpretation": "predictive receipt proxy; not an entitlement or amount identity",
        }
    )
    gains = p.capital_gains.to_numpy(dtype=float)
    cgt = {}
    for name, mask in {"all_positive": gains > 0, "at_least_5m": gains >= 5e6}.items():
        n = int(mask.sum())
        cgt[name] = (
            {"rows": "fewer_than_3", "suppressed": True}
            if 0 < n < 3
            else {
                "rows": n,
                "person_mass": float(pw[mask].sum()),
                "weighted_gains_total": float(np.dot(pw[mask], gains[mask])),
            }
        )
    order = np.argsort(h.household_id.to_numpy())
    identity = hashlib.sha256(
        np.asarray(h.household_id.to_numpy()[order], dtype="<i8").tobytes()
    ).hexdigest()
    weight_identity = hashlib.sha256(
        np.asarray(hw[order], dtype="<f8").tobytes()
    ).hexdigest()
    result = {
        "scope": "Final generated source spine at prior household weights; includes downstream copies; no licensed rows or IDs emitted",
        "spine_sha256": before,
        "row_counts": {"person": len(p), "benunit": len(b), "household": len(h)},
        "household_prior_weight_total": float(hw.sum()),
        "household_prior_weight_kind": frame.weights_for("household").kind.value,
        "household_id_set_sha256": identity,
        "prior_weights_by_sorted_household_id_sha256": weight_identity,
        "channels": channels,
        "spi_pension_receipt_proxy": pension,
        "cgt": cgt,
    }
    assert file_sha(path) == before, "Source-quality reader changed the H5 bytes"
    return result


def source_quality_comparison(a, b):
    return {
        "row_count_changes": {
            key: b["row_counts"][key] - a["row_counts"][key] for key in a["row_counts"]
        },
        "household_prior_weight_total_change": b["household_prior_weight_total"]
        - a["household_prior_weight_total"],
        "same_final_household_id_set": a["household_id_set_sha256"]
        == b["household_id_set_sha256"],
        "same_final_prior_weights_ordered_by_household_id": a[
            "prior_weights_by_sorted_household_id_sha256"
        ]
        == b["prior_weights_by_sorted_household_id_sha256"],
        "interpretation": "These are measured generated-support outcomes; input design-weight and sampling algorithms are held fixed.",
    }


def archived_attempts(run):
    result = []
    for directory in sorted(run["path"].iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        for gate_path in directory.rglob("calibration_gates.json"):
            if not (gate_path.parent / "calibration_diagnostics.json").exists():
                continue
            gates = read(gate_path)
            d = read(gate_path.parent / "calibration_diagnostics.json")
            numerical_keys = [key for key in d if key != "build"]
            selected = run["diagnostics"]
            numerical_equal = all(d[key] == selected[key] for key in numerical_keys)
            result.append(
                {
                    "directory": str(gate_path.parent.relative_to(run["path"])),
                    "selected_for_comparison": False,
                    "diagnostics_sha256": file_sha(
                        gate_path.parent / "calibration_diagnostics.json"
                    ),
                    "gates_sha256": file_sha(gate_path),
                    "code_pin": d["build"]["code_pin"],
                    "final_loss": d["final_loss"],
                    "all_diagnostics_except_build_equal_selected": numerical_equal,
                    "signing_error": gates["attestation"].get("signing_error"),
                    "reason": "Retained earlier attempt; not substituted for the selected correctly encoded developer-key run.",
                }
            )
    return result


def run_summary(run, verified, quality):
    d, gates, build = run["diagnostics"], run["gates"], run["build"]
    fields = [
        "initial_loss",
        "final_loss",
        "fraction_within_10pct",
        "effective_sample_size",
        "realized_max_weight_ratio",
        "top_1pct_weight_share",
        "n_records",
        "n_nonzero",
    ]
    spi = build["stage_evidence"]["hmrc_spi_income_spine"]
    return {
        "run_name": run["path"].name,
        **verified,
        "file_hashes": run["file_hashes"],
        "runtime": d["build"]["runtime"],
        "calibration_invocation": {
            key: value for key, value in run["invocation"].items() if key != "argv"
        },
        "metrics": {key: d[key] for key in fields},
        "calibrated_weight_summary": d["uk_diagnostics"]["weights"],
        "target_surface": d["target_surface"],
        "source_quality": quality,
        "spine_declared_counts": build["entity_row_counts"],
        "spine_prior_weight_total": build["household_weight_total"],
        "spine_weight_kind": build["household_weight_kind"],
        "spi_stage": {
            key: spi.get(key)
            for key in [
                "source_vintages",
                "spi_prior",
                "recipient_minimum_age",
                "income_uprating",
                "pension_receipt_bridge",
            ]
        },
        "uc_income_screen": run["uc_support"],
        "terminal_gates": gates["gates"],
        "spine_gate_statuses": {
            key: value["status"] for key, value in run["spine_gates"]["gates"].items()
        },
        "gate_evaluation_date": gates["gates"]["uk_target_fit"]["details"][
            "exclusions_evaluated_on"
        ],
        "attestation": {
            "signature_present": bool(gates["attestation"].get("signature")),
            "signing_error": gates["attestation"].get("signing_error"),
            "signature_algorithm": gates["attestation"].get("signature_algorithm"),
            "scope": "Producer emitted a local ephemeral developer-key signature; not a production certificate or independent cryptographic verification.",
        },
        "release_candidate": gates["release_candidate"],
        "shippable": gates["shippable"],
        "calibrated_h5_exported": (run["path"] / "calibrated.h5").exists(),
        "archived_attempts": archived_attempts(run),
    }


def paired_receipt(control, treatment, repo, contract, protected_manifest):
    assert contract["publication_base"] == PUBLICATION_COMMIT
    expected = {"control": PUBLICATION_COMMIT, "treatment": TREATMENT_COMMIT}
    checked = {
        "control": verify_run(control, repo, expected["control"], contract),
        "treatment": verify_run(treatment, repo, expected["treatment"], contract),
    }
    a, b = control["diagnostics"], treatment["diagnostics"]
    equalities = {}

    def same(name, left, right):
        assert left == right, f"Matched-comparison contract differs: {name}"
        equalities[name] = {"equal": True, "sha256": value_sha(left)}

    for key in ["options", "target_registry", "target_loss_basis"]:
        same(key, a[key], b[key])
    for key in [
        "runtime",
        "ledger",
        "doctrine",
        "doctrine_overrides",
        "measure_exclusions",
        "register",
    ]:
        same("calibration." + key, a["build"][key], b["build"][key])
    same(
        "measure_resolution_without_run_source_path",
        stable_resolution(a["build"]["measure_resolution"]),
        stable_resolution(b["build"]["measure_resolution"]),
    )
    for key in [
        "artifact_pins",
        "input_artifact_pins",
        "stage_artifact_pins",
        "resource_pins",
        "declared_seeds",
        "fit_weight_records",
        "sampling",
        "rules_engine",
        "time_period",
        "source_vintages",
        "stochastic_contract_sha256",
        "stages",
    ]:
        same("spine." + key, control["build"][key], treatment["build"][key])
    for key in ["names_sha256", "values_sha256"]:
        same(
            "target_surface." + key, a["target_surface"][key], b["target_surface"][key]
        )
    same(
        "every_target_registry_filter_measure_binding_and_loss_weight",
        [stable_target(row) for row in a["targets"]],
        [stable_target(row) for row in b["targets"]],
    )
    for key in [
        "policy_sha256",
        "gates_manifest_sha256",
        "release_candidate",
        "phases",
    ]:
        same("gates." + key, control["gates"][key], treatment["gates"][key])
    dates = [
        run["gates"]["gates"]["uk_target_fit"]["details"]["exclusions_evaluated_on"]
        for run in [control, treatment]
    ]
    same("gate_evaluation_date", *dates)
    protected = read(protected_manifest)
    assert value_sha(protected) == PROTECTED_SHA256
    assert (
        value_sha(
            {
                key: value
                for key, value in contract.items()
                if not key.endswith("_file") and key != "chronicle_path"
            }
        )
        == CONTRACT_SHA256
    )
    for row in protected["files"]:
        before = git_bytes(repo, expected["control"], row["path"])
        after = git_bytes(repo, expected["treatment"], row["path"])
        assert hashlib.sha256(before).hexdigest() == row["sha256"]
        assert hashlib.sha256(after).hexdigest() == row["current_sha256"]
    raw = verify_raw_inputs(control["build"], contract)
    rows, families = [], {}
    delta = a["final_loss"] - b["final_loss"]
    for x, y in zip(a["targets"], b["targets"], strict=True):
        change = x["final_loss_contribution"] - y["final_loss_contribution"]
        row = {
            "binding": stable_target(x),
            "control": {key: x.get(key) for key in sorted(OUTCOME_FIELDS)},
            "treatment": {key: y.get(key) for key in sorted(OUTCOME_FIELDS)},
            "loss_reduction": change,
            "share_of_net_loss_reduction": change / delta if delta else None,
        }
        rows.append(row)
        family = x["registry"]["family"]
        bucket = families.setdefault(
            family, {"target_count": 0, "control_loss": 0.0, "treatment_loss": 0.0}
        )
        bucket["target_count"] += 1
        bucket["control_loss"] += x["final_loss_contribution"]
        bucket["treatment_loss"] += y["final_loss_contribution"]
    for bucket in families.values():
        bucket["loss_reduction"] = bucket["control_loss"] - bucket["treatment_loss"]
        bucket["share_of_net_loss_reduction"] = (
            bucket["loss_reduction"] / delta if delta else None
        )
    quality = {
        "control": source_quality(control["path"]),
        "treatment": source_quality(treatment["path"]),
    }
    indexed = {row["binding"]["name"]: row for row in rows}
    spi = treatment["build"]["stage_evidence"]["hmrc_spi_income_spine"]
    uprating = spi["income_uprating"]
    assert uprating["from_period"] == 2022 and uprating["to_period"] == 2024
    assert {
        key
        for key, value in uprating["columns"].items()
        if value["basis"] == "held_nominal"
    } == NOMINAL
    assert (
        uprating["columns"]["other_investment_income"]["variable"]
        == "other_investment_income"
    )
    missing_key_targets = [key for key in KEY_TARGETS if key not in indexed]
    assert not missing_key_targets, missing_key_targets
    private = indexed[
        "hmrc/private_pension_income_count_income_band_100_000_to_150_000@2025"
    ]
    improved = [
        row["binding"]["name"]
        for row in rows
        if abs(row["treatment"]["relative_error"])
        < abs(row["control"]["relative_error"]) - 1e-12
    ]
    regressed = [
        row["binding"]["name"]
        for row in rows
        if abs(row["treatment"]["relative_error"])
        > abs(row["control"]["relative_error"]) + 1e-12
    ]
    within = {
        name: sum(abs(row[name]["relative_error"]) <= 0.1 for row in rows)
        for name in ["control", "treatment"]
    }
    regression_summary = {
        "comparison": "Absolute relative error; equality tolerance 1e-12",
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_count": len(rows) - len(improved) - len(regressed),
        "improved_targets": improved,
        "regressed_targets": regressed,
        "within_10pct_count": within,
        "crossed_outside_10pct": [
            row["binding"]["name"]
            for row in rows
            if abs(row["control"]["relative_error"])
            <= 0.1
            < abs(row["treatment"]["relative_error"])
        ],
        "crossed_inside_10pct": [
            row["binding"]["name"]
            for row in rows
            if abs(row["treatment"]["relative_error"])
            <= 0.1
            < abs(row["control"]["relative_error"])
        ],
    }
    return {
        "schema_version": 1,
        "receipt_kind": "uk_publication_spi_paired_experiment",
        "status": "Matched genuine runs; fit and release verdicts remain separate",
        "comparison_contract": {
            key: value
            for key, value in contract.items()
            if not key.endswith("_file") and key != "chronicle_path"
        },
        "contract_checks": equalities,
        "raw_source_verification": raw,
        "protected_publication_files": protected,
        "target_count": len(rows),
        "compiled_reference_count": a["build"]["register"]["compiled_count"],
        "regression_summary": regression_summary,
        "measure_exclusion_count": len(a["build"]["measure_exclusions"]),
        "shared_target_loss_basis": a["target_loss_basis"],
        "shared_measure_exclusions": a["build"]["measure_exclusions"],
        "runs": {
            "control": run_summary(control, checked["control"], quality["control"]),
            "treatment": run_summary(
                treatment, checked["treatment"], quality["treatment"]
            ),
        },
        "generated_support_comparison": source_quality_comparison(
            quality["control"], quality["treatment"]
        ),
        "objective_attribution": {
            "control_loss": a["final_loss"],
            "treatment_loss": b["final_loss"],
            "loss_reduction": delta,
            "relative_reduction": delta / a["final_loss"],
            "families": families,
        },
        "key_targets": {key: indexed[key] for key in KEY_TARGETS},
        "targets": rows,
        "decision_assessment": {
            "A22_A23": {
                "changed": False,
                "rates": {
                    "tax_free_childcare": 0.88,
                    "targeted_childcare": 0.597,
                    "extended_childcare": 0.6054,
                    "universal_childcare": 0.4539,
                },
                "interpretation": "National fit at these fixed rates does not re-estimate take-up or verify the historical eligible-base ceiling; no new ceiling-rate decision.",
            },
            "private_pension_deferral": {
                "changed": False,
                "control_relative_error": private["control"]["relative_error"],
                "treatment_relative_error": private["treatment"]["relative_error"],
                "control_stale": private["binding"]["name"]
                in control["gates"]["gates"]["uk_target_fit"]["details"][
                    "stale_exclusions"
                ],
                "treatment_stale": private["binding"]["name"]
                in treatment["gates"]["gates"]["uk_target_fit"]["details"][
                    "stale_exclusions"
                ],
                "interpretation": "Staleness is reported as a gate failure; this receipt does not remove or renew the protected entry.",
            },
        },
        "limitations": [
            "National-only single-seed comparison; no local-area accuracy claim.",
            "The age16 guard does not protect all qualifying FRS children aged16–19.",
            "Six SPI flows remain nominal and pension receipt is a proxy, not an entitlement identity.",
            "No calibrated file is exported if native terminal gates refuse.",
            "This is not the absent shared quality/dashboard schema and not a release certificate.",
        ],
    }


def main():
    if not __debug__:
        raise RuntimeError(
            "Run without Python optimization so contract assertions remain enabled"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--treatment", type=Path)
    parser.add_argument("--source-only", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--protected-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.resolve().is_relative_to(args.repo.resolve()), (
        "Keep repository clean: write only intermediate output outside it"
    )
    if args.source_only:
        result = source_quality(args.source_only)
    else:
        assert (
            args.control
            and args.treatment
            and args.contract
            and args.protected_manifest
        )
        contract = read(args.contract)
        result = paired_receipt(
            read_run(args.control),
            read_run(args.treatment),
            args.repo,
            contract,
            args.protected_manifest,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_sha(args.output),
                "target_count": result.get("target_count"),
                "status": "written",
            }
        )
    )


if __name__ == "__main__":
    main()
