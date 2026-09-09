import json
from pathlib import Path

import microcosm.build.staging as staging_module
from microcosm.build.staging import StagingTelemetry

V1_FIXTURE = Path(__file__).parent / "fixtures" / "staging" / "v1"


class FakeApi:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str, str]] = []

    def upload_file(
        self,
        *,
        path_or_fileobj,
        path_in_repo,
        repo_id,
        repo_type,
    ):
        self.uploads.append((str(path_or_fileobj), path_in_repo, repo_id, repo_type))
        return None


def test_staging_telemetry_writes_local_progress(tmp_path) -> None:
    telemetry = StagingTelemetry(
        run_id="run-a",
        candidate_release_id="populace-us-2024-abc-20260618T000000Z",
        run_dir=tmp_path / "run-a",
    )

    telemetry.stage("calibrating", message="Calibration started.", n_targets=3)
    telemetry.calibration_progress(
        {"kind": "calibration_epoch", "epoch": 1, "epochs": 5, "loss": 12.5}
    )
    telemetry.complete()

    progress = json.loads((tmp_path / "run-a" / "progress.json").read_text())
    calibration = json.loads(
        (tmp_path / "run-a" / "calibration_progress.json").read_text()
    )
    events = (tmp_path / "run-a" / "events.ndjson").read_text().strip().splitlines()

    assert progress["status"] == "passed"
    assert progress["stage"] == "complete"
    assert calibration["events"][0]["loss"] == 12.5
    assert len(events) >= 3


def test_staging_telemetry_uploads_repo_paths(tmp_path) -> None:
    api = FakeApi()
    telemetry = StagingTelemetry(
        run_id="run-b",
        candidate_release_id="populace-us-2024-def-20260618T000000Z",
        run_dir=tmp_path / "run-b",
        repo_id="policyengine/populace-us-staging",
        api=api,
        upload_interval_seconds=0,
    )
    telemetry.stage("target_compilation", force_upload=True)

    uploaded_paths = {upload[1] for upload in api.uploads}
    assert "runs/run-b/progress.json" in uploaded_paths
    assert "runs/run-b/run_manifest.json" in uploaded_paths
    assert "latest_staging.json" in uploaded_paths
    assert "runs.json" in uploaded_paths


def test_runs_index_keeps_existing_runs(tmp_path) -> None:
    # An older run is already recorded in the repo's index.
    existing = tmp_path / "existing.json"
    existing.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": "2026-06-19T00:00:00+00:00",
                "runs": [
                    {
                        "run_id": "older",
                        "candidate_release_id": "populace-us-2024-old-20260619T000000Z",
                        "status": "running",
                        "stage": "write_calibration_npz",
                        "updated_at": "2026-06-19T00:00:00+00:00",
                    }
                ],
            }
        )
    )

    class FakeApiWithDownload(FakeApi):
        def hf_hub_download(self, *, repo_id, filename, repo_type, **kwargs):
            return str(existing)

    telemetry = StagingTelemetry(
        run_id="newer",
        candidate_release_id="populace-us-2024-new-20260620T000000Z",
        run_dir=tmp_path / "newer",
        repo_id="policyengine/populace-us-staging",
        api=FakeApiWithDownload(),
        upload_interval_seconds=0,
    )
    telemetry.complete()

    index = json.loads((tmp_path / "newer" / "runs.json").read_text())
    run_ids = [run["run_id"] for run in index["runs"]]
    # Both the pre-existing run and this run survive (no overwrite).
    assert run_ids == ["newer", "older"]


def test_upload_failures_never_raise_and_disable_after_three(tmp_path, capsys):
    class FailingApi:
        def upload_file(self, **kwargs):
            raise RuntimeError("401 Unauthorized")

        def hf_hub_download(self, **kwargs):
            raise RuntimeError("401 Unauthorized")

    telemetry = StagingTelemetry(
        run_id="run-1",
        candidate_release_id="run-1",
        run_dir=tmp_path / "run",
        repo_id="org/staging",
        api=FailingApi(),
        upload_interval_seconds=0.0,
    )
    # Staging telemetry is best-effort: failing uploads must never raise, and
    # after three consecutive failures uploads are disabled for the run.
    telemetry.stage("one", force_upload=True)
    telemetry.stage("two", force_upload=True)
    assert telemetry.repo_id is None
    err = capsys.readouterr().err
    assert "staging upload" in err
    assert "disabling staging uploads" in err
    # Nothing reached the repo, and the run says so. A configured destination
    # is not evidence of delivery.
    assert telemetry.uploads_succeeded == 0


def test_uploads_succeeded_counts_files_that_reached_the_repo(tmp_path):
    api = FakeApi()
    telemetry = StagingTelemetry(
        run_id="run-c",
        candidate_release_id="run-c",
        run_dir=tmp_path / "run-c",
        repo_id="org/staging",
        api=api,
        upload_interval_seconds=0.0,
    )

    telemetry.stage("target_compilation", force_upload=True)

    assert telemetry.uploads_succeeded == len(api.uploads)
    assert telemetry.uploads_succeeded > 0


def test_blank_path_prefix_falls_back_to_the_default(tmp_path):
    # A blank or slash-only prefix would put run files at the repo root, where
    # the dashboard's runs/<run_id> paths cannot find them.
    for blank in ("", "   ", "/", " / "):
        telemetry = StagingTelemetry(
            run_id="run-e",
            candidate_release_id="run-e",
            run_dir=tmp_path / "run-e",
            path_prefix=blank,
        )
        assert telemetry.path_prefix == "runs"
        assert telemetry.repo_run_prefix == "runs/run-e"


def test_path_prefix_is_trimmed_but_otherwise_respected(tmp_path):
    telemetry = StagingTelemetry(
        run_id="run-f",
        candidate_release_id="run-f",
        run_dir=tmp_path / "run-f",
        path_prefix="  /candidate-runs/  ",
    )
    assert telemetry.path_prefix == "candidate-runs"
    assert telemetry.repo_run_prefix == "candidate-runs/run-f"


def test_uploads_succeeded_is_zero_for_a_local_only_run(tmp_path):
    telemetry = StagingTelemetry(
        run_id="run-d",
        candidate_release_id="run-d",
        run_dir=tmp_path / "run-d",
    )

    telemetry.stage("target_compilation", force_upload=True)

    assert telemetry.uploads_succeeded == 0
    assert (tmp_path / "run-d" / "progress.json").is_file()


def test_us_version_1_bundle_matches_fixed_fixture(tmp_path, monkeypatch):
    timestamps = iter(
        [
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:02+00:00",
            "2026-01-01T00:00:03+00:00",
            "2026-01-01T00:00:04+00:00",
            "2026-01-01T00:00:05+00:00",
            "2026-01-01T00:00:06+00:00",
            "2026-01-01T00:00:07+00:00",
        ]
    )
    monkeypatch.setattr(staging_module, "_now", lambda: next(timestamps))

    class FixtureApi(FakeApi):
        def hf_hub_download(self, **kwargs):
            raise FileNotFoundError

    run_dir = tmp_path / "v1-us-fixture"
    telemetry = StagingTelemetry(
        run_id="v1-us-fixture",
        candidate_release_id="populace-us-2024-v1-fixture",
        run_dir=run_dir,
        repo_id="policyengine/populace-us-staging",
        api=FixtureApi(),
        upload_interval_seconds=float("inf"),
        started_at="2026-01-01T00:00:00+00:00",
    )
    telemetry.stage("calibrating", message="Calibration started.", n_targets=2)
    telemetry.calibration_progress(
        {"kind": "calibration_epoch", "epoch": 1, "epochs": 2, "loss": 1.5}
    )
    telemetry.complete()

    for filename in (
        "run_manifest.json",
        "progress.json",
        "calibration_progress.json",
        "latest_staging.json",
        "runs.json",
    ):
        assert json.loads((run_dir / filename).read_text()) == json.loads(
            (V1_FIXTURE / filename).read_text()
        )
    assert (run_dir / "events.ndjson").read_text().splitlines() == (
        V1_FIXTURE / "events.ndjson"
    ).read_text().splitlines()
