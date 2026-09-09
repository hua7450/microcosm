import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from microcosm.build.staging_v2 import (
    CALIBRATION_PROGRESS_SCHEMA,
    EVENT_SCHEMA,
    LATEST_RUN_SCHEMA,
    PROGRESS_SCHEMA,
    RUN_INDEX_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    STAGING_CONTRACT_VERSION,
    StagingContentError,
    StagingContractError,
    StagingReadBackError,
    StagingTelemetryV2,
    disabled_staging_delivery,
    validate_staging_delivery,
    validate_v2_bundle,
    validate_v2_document,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "staging"


class Clock:
    def __init__(self) -> None:
        self.second = 0

    def __call__(self) -> str:
        value = f"2026-01-02T00:00:{self.second:02d}+00:00"
        self.second += 1
        return value


class MemoryApi:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, bytes] = {}

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
        assert repo_id == "policyengine/populace-uk-staging"
        assert repo_type == "dataset"
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def hf_hub_download(self, *, filename, repo_id, repo_type, **kwargs):
        if filename not in self.files:
            raise FileNotFoundError(filename)
        destination = self.root / "download" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[filename])
        return str(destination)


def _sample() -> dict:
    return {
        "mode": "bounded_source_households",
        "requested_source_households": 5,
        "eligible_source_families": 20,
        "proportional_request": 5,
        "forced_additions": 2,
        "realized_source_families": 7,
        "realized_household_rows": 9,
        "seed": 42,
        "receipt_sha256": "a" * 64,
    }


def _recorder(tmp_path, **kwargs) -> StagingTelemetryV2:
    defaults = {
        "run_id": "uk-smoke-5-42",
        "country_code": "GB",
        "operation_id": "uk_frs_spine",
        "pipeline_id": "uk_national_spine",
        "pipeline_version": "2026.09",
        "candidate_id": "uk-smoke-5-42",
        "local_dir": tmp_path / "telemetry",
        "release_id": None,
        "run_kind": "smoke",
        "delivery_mode": "local_only",
        "repo_id": None,
        "clock": Clock(),
    }
    defaults.update(kwargs)
    return StagingTelemetryV2(**defaults)


def test_version_2_local_bundle_has_explicit_schemas_and_ordered_events(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.set_sample(_sample())
    telemetry.stage("input_verification", event_status="completed", files=3)
    telemetry.stage("sampling", event_status="completed", source_families=7)
    telemetry.complete()

    bundle = telemetry.validate_local_bundle()

    assert bundle["run_manifest"]["schema_name"] == RUN_MANIFEST_SCHEMA
    assert bundle["progress"]["schema_name"] == PROGRESS_SCHEMA
    assert bundle["latest"]["schema_name"] == LATEST_RUN_SCHEMA
    assert bundle["index"]["schema_name"] == RUN_INDEX_SCHEMA
    assert {event["schema_name"] for event in bundle["events"]} == {EVENT_SCHEMA}
    assert [event["sequence"] for event in bundle["events"]] == [1, 2, 3, 4]
    assert bundle["progress"]["status"] == "completed"
    assert bundle["run_manifest"]["release_id"] is None
    assert bundle["run_manifest"]["non_release"] is True


def test_calibration_events_are_independently_versioned(tmp_path):
    telemetry = _recorder(tmp_path, run_kind="calibration")
    telemetry.calibration_progress(
        {
            "kind": "calibration_epoch",
            "epoch": 1,
            "epochs": 5,
            "phase": "solve",
            "loss": 1.25,
            "iteration": 4,
        }
    )

    bundle = validate_v2_bundle(telemetry.local_dir, telemetry.run_id)

    assert bundle["calibration_progress"]["schema_name"] == (
        CALIBRATION_PROGRESS_SCHEMA
    )
    assert bundle["calibration_progress"]["events"][0]["loss"] == 1.25
    assert bundle["events"][-1]["event_type"] == "calibration"


def test_typed_artifacts_accept_aggregate_json_and_refuse_population_data(tmp_path):
    telemetry = _recorder(tmp_path)
    diagnostic = tmp_path / "aggregate.json"
    diagnostic.write_text(json.dumps({"target_count": 3, "mean_loss": 0.25}))

    artifact = telemetry.add_artifact(
        "aggregate-diagnostics",
        diagnostic,
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )

    assert artifact["contract_relative_path"] == (
        "artifacts/aggregate-diagnostics.json"
    )
    assert len(artifact["sha256"]) == 64

    population = tmp_path / "population.h5"
    population.write_bytes(b"not really h5")
    with pytest.raises(StagingContentError, match="must be JSON"):
        telemetry.add_artifact(
            "population",
            population,
            artifact_kind="build_metadata",
            classification="non_row_level",
        )

    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"person_id": [1, 2]}))
    with pytest.raises(StagingContentError, match="row-level field"):
        telemetry.add_artifact(
            "rows",
            rows,
            artifact_kind="aggregate_diagnostics",
            classification="aggregate",
        )


def test_failure_uses_sanitized_contract_fields(tmp_path):
    telemetry = _recorder(tmp_path)
    telemetry.stage("input_verification")
    telemetry.fail(
        RuntimeError("/licensed/frs/adult.tab token=secret-value household 123"),
        local_diagnostic_reference="diagnostics/local-error.txt",
    )

    progress = telemetry.validate_local_bundle()["progress"]
    serialized = json.dumps(progress)

    assert progress["status"] == "failed"
    assert progress["failure"] == {
        "error_code": "BUILD_FAILED",
        "error_type": "RuntimeError",
        "message": "The build failed during input_verification.",
        "local_diagnostic_reference": "diagnostics/local-error.txt",
    }
    assert "licensed" not in serialized
    assert "secret-value" not in serialized
    assert "traceback" not in serialized.lower()


def test_delivery_validation_rejects_contradictions():
    local_only = {
        "contract_version": 2,
        "enabled": True,
        "mode": "local_only",
        "run_id": "run-a",
        "configured_repository": None,
        "upload_attempts": 0,
        "upload_successes": 0,
        "read_back": "not_requested",
        "last_error_code": None,
        "opt_out_reason": None,
    }
    assert validate_staging_delivery(local_only) == local_only
    assert disabled_staging_delivery("--no-staging")["enabled"] is False

    with pytest.raises(StagingContractError, match="cannot name a repository"):
        validate_staging_delivery(
            {**local_only, "configured_repository": "policyengine/example"}
        )
    with pytest.raises(StagingContractError, match="cannot exceed"):
        validate_staging_delivery(
            {**local_only, "upload_attempts": 0, "upload_successes": 1}
        )


def test_unknown_schema_version_is_incompatible():
    with pytest.raises(StagingContractError, match="Unsupported staging schema"):
        validate_v2_document(
            {
                "schema_name": RUN_MANIFEST_SCHEMA,
                "schema_version": 3,
            }
        )


def test_identifiers_and_paths_cannot_escape_contract_root(tmp_path):
    with pytest.raises(StagingContractError, match="run_id"):
        _recorder(tmp_path, run_id="../outside")
    with pytest.raises(StagingContractError, match="path_prefix"):
        _recorder(tmp_path, path_prefix="../runs")
    with pytest.raises(StagingContractError, match="Local-only"):
        _recorder(tmp_path, repo_id="policyengine/example")


def test_remote_failures_are_best_effort_and_preserve_repository(tmp_path, capsys):
    class FailingApi:
        def upload_file(self, **kwargs):
            raise RuntimeError("401 token=do-not-record")

    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=FailingApi(),
        upload_interval_seconds=0,
    )

    telemetry.stage("input_verification", force_upload=True)

    delivery = telemetry.delivery_summary
    assert delivery["configured_repository"] == "policyengine/populace-uk-staging"
    assert delivery["upload_attempts"] == 3
    assert delivery["upload_successes"] == 0
    assert delivery["last_error_code"] == "UPLOAD_FAILED"
    assert telemetry.validate_local_bundle()["progress"]["status"] == "running"
    assert "do-not-record" not in capsys.readouterr().err


def test_remote_read_back_validates_the_written_run(tmp_path):
    api = MemoryApi(tmp_path)
    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )
    telemetry.set_sample(_sample())
    telemetry.complete()

    telemetry.verify_remote()

    assert telemetry.delivery_summary["read_back"] == "passed"
    assert telemetry.uploads_succeeded > 0
    assert "runs/uk-smoke-5-42/run_manifest.json" in api.files
    assert "latest_staging.json" in api.files
    assert "runs.json" in api.files
    remote_manifest = json.loads(
        api.files["runs/uk-smoke-5-42/run_manifest.json"]
    )
    assert remote_manifest["delivery"]["read_back"] == "passed"


def test_remote_read_back_failure_is_explicit(tmp_path):
    class UploadOnlyApi(MemoryApi):
        def hf_hub_download(self, **kwargs):
            raise RuntimeError("unavailable")

    telemetry = _recorder(
        tmp_path,
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=UploadOnlyApi(tmp_path),
        upload_interval_seconds=0,
    )
    telemetry.complete()

    with pytest.raises(StagingReadBackError, match="read-back failed"):
        telemetry.verify_remote()
    assert telemetry.delivery_summary["read_back"] == "failed"


def test_local_run_index_merges_by_run_identifier(tmp_path):
    first = _recorder(tmp_path, run_id="first", candidate_id="first")
    first.complete()
    second = _recorder(tmp_path, run_id="second", candidate_id="second")
    second.complete()

    index = json.loads((second.local_dir / "runs.json").read_text())

    assert {run["run_id"] for run in index["runs"]} == {"first", "second"}
    assert index["schema_version"] == STAGING_CONTRACT_VERSION


def test_remote_run_index_is_merged_before_upload(tmp_path):
    api = MemoryApi(tmp_path)
    first = _recorder(
        tmp_path / "first",
        run_id="first",
        candidate_id="first",
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )
    first.complete()
    second = _recorder(
        tmp_path / "second",
        run_id="second",
        candidate_id="second",
        delivery_mode="local_and_remote",
        repo_id="policyengine/populace-uk-staging",
        api=api,
        upload_interval_seconds=0,
    )
    second.complete()

    index = json.loads(api.files["runs.json"])

    assert {run["run_id"] for run in index["runs"]} == {"first", "second"}


def test_bundle_validation_checks_reviewed_artifact_digest(tmp_path):
    telemetry = _recorder(tmp_path)
    diagnostic = tmp_path / "aggregate.json"
    diagnostic.write_text(json.dumps({"target_count": 3}))
    artifact = telemetry.add_artifact(
        "aggregate-diagnostics",
        diagnostic,
        artifact_kind="aggregate_diagnostics",
        classification="aggregate",
    )
    artifact_path = telemetry.run_dir / artifact["contract_relative_path"]
    artifact_path.write_text(json.dumps({"target_count": 4}))

    with pytest.raises(StagingContractError, match="wrong digest"):
        telemetry.validate_local_bundle()


def test_canonical_version_2_fixture_cases_validate():
    v2 = FIXTURE_ROOT / "v2"
    completed = validate_v2_bundle(
        v2 / "completed-spine", "uk-spine-v2-fixture"
    )
    calibration = validate_v2_bundle(
        v2 / "calibration", "uk-calibration-v2-fixture"
    )
    failed = validate_v2_bundle(v2 / "failed", "uk-failed-v2-fixture")
    cases = json.loads((v2 / "contract-cases.json").read_text())

    assert completed["progress"]["status"] == "completed"
    assert calibration["calibration_progress"]["events"][0]["phase"] == "solve"
    assert failed["progress"]["failure"]["message"] == (
        "The build failed during input_verification."
    )
    assert validate_staging_delivery(cases["delivery_success"])[
        "upload_successes"
    ] == 6
    assert validate_staging_delivery(cases["delivery_failure"])[
        "last_error_code"
    ] == "READ_BACK_FAILED"
    assert validate_staging_delivery(cases["deliberate_opt_out"])["enabled"] is False
    with pytest.raises(StagingContractError, match="Unsupported staging schema"):
        validate_v2_document(cases["unknown_version"])


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_contract_fixture_checksums_are_pinned(version):
    fixture_dir = FIXTURE_ROOT / version
    for line in (fixture_dir / "SHA256SUMS").read_text().splitlines():
        expected, relative_path = line.split("  ", 1)
        assert hashlib.sha256((fixture_dir / relative_path).read_bytes()).hexdigest() == (
            expected
        )


def test_canonical_version_2_fixtures_are_reproducible():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "generate_staging_contract_fixtures.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
