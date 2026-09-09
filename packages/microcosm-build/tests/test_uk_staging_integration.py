"""Hermetic command-line integration coverage for UK staging smoke builds."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.staging_v2 import validate_v2_bundle
from microcosm.build.uk_runtime.graph import UK_SPINE_EXCLUSIONS

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "packages/microcosm-graph/tests/fixtures/parity/uk_spine/sources"
DRIVER = ROOT / "tools/build_uk_frs_spine.py"
WORKFLOW = ROOT / ".github/workflows/integration-tests.yml"
WORKFLOW_SCRIPT = ROOT / "tools/run_integration_tests.sh"
SEED = 42
RUN_ID = "ci-uk-smoke-full-s42"


def test_integration_workflow_runs_without_environment_approval() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = WORKFLOW_SCRIPT.read_text(encoding="utf-8")

    assert "name: Integration tests" in workflow
    assert "pull_request:" in workflow
    assert "paths:" not in workflow
    assert "environment:" not in workflow
    assert "head.repo.full_name" not in workflow
    assert "run: |" not in workflow
    assert "run: bash tools/run_integration_tests.sh" in workflow
    assert "uv sync --all-packages --locked --extra uk" in script
    assert "test_uk_staging_integration.py" in script
    assert "HF_STAGING_READ_TOKEN" in script


def _tracked_state() -> str:
    return subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def _command(tmp_path: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(DRIVER),
        "--synthetic-fixture-dir",
        str(FIXTURE),
        "--spine-h5",
        str(tmp_path / "uk-smoke.h5"),
        "--sample-fraction",
        "1.0",
        "--sample-seed",
        str(SEED),
        "--smoke",
        "--staging-local-only",
        "--staging-dir",
        str(tmp_path / "staging"),
        "--staging-run-id",
        RUN_ID,
        *extra,
    ]


@pytest.mark.requires_uk
def test_uk_staging_smoke_command_runs_every_spine_stage(tmp_path: Path) -> None:
    """Run all current spine transforms without licensed or remote inputs."""

    before = _tracked_state()
    started = time.perf_counter()
    result = subprocess.run(
        _command(tmp_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=12 * 60,
    )
    elapsed = time.perf_counter() - started
    assert result.returncode == 0, result.stderr
    assert _tracked_state() == before

    output = tmp_path / "uk-smoke.h5"
    sidecar = json.loads(output.with_suffix(".build.json").read_text())
    with h5py.File(output) as data:
        assert bool(data.attrs["populace_non_release"]) is True
        assert data.attrs["populace_release_posture"] == "smoke"
    assert sidecar["non_release"] is True
    assert sidecar["release_posture"] == "non_release_smoke"
    assert sidecar["sampling"] is None
    assert sidecar["synthetic_fixture"]["schema_version"] == (
        "uk-spine-parity-fixture.v1"
    )
    assert len(sidecar["synthetic_fixture"]["digest"]) == 64

    bundle = validate_v2_bundle(tmp_path / "staging", RUN_ID)
    manifest = bundle["run_manifest"]
    assert manifest["status"] == "completed"
    assert manifest["non_release"] is True
    assert manifest["sample"] == {"mode": "full"}
    assert manifest["delivery"]["mode"] == "local_only"
    assert manifest["delivery"]["upload_attempts"] == 0

    spec = load_country_spec("uk")
    expected = [
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    ]
    events = bundle["events"]
    for stage in expected:
        transitions = [
            event["status"] for event in events if event["stage_id"] == stage
        ]
        assert transitions == ["started", "completed"], stage
    stage_seconds = {
        event["stage_id"]: event["details"]["elapsed_seconds"]
        for event in events
        if event["status"] == "completed" and "elapsed_seconds" in event["details"]
    }
    assert set(stage_seconds) == set(expected)
    print(
        json.dumps(
            {
                "total_seconds": elapsed,
                "stage_seconds": stage_seconds,
                "household_rows": sidecar["entity_row_counts"]["household"],
            },
            sort_keys=True,
        )
    )


@pytest.mark.requires_uk
def test_synthetic_smoke_command_refuses_release_options(tmp_path: Path) -> None:
    result = subprocess.run(
        _command(tmp_path, "--release-candidate"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "non-release smoke builds refuse --release-candidate" in result.stderr
    assert not (tmp_path / "uk-smoke.h5").exists()
