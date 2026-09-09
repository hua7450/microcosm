"""Generate or verify the canonical staging telemetry version 2 fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from microcosm.build.staging_v2 import StagingTelemetryV2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "packages" / "microcosm-build" / "tests" / "fixtures" / "staging" / "v2"
)


class FixtureClock:
    def __init__(self, day: int) -> None:
        self.day = day
        self.second = 0

    def __call__(self) -> str:
        timestamp = f"2026-01-{self.day:02d}T00:00:{self.second:02d}+00:00"
        self.second += 1
        return timestamp


def _completed_spine(root: Path) -> None:
    recorder = StagingTelemetryV2(
        run_id="uk-spine-v2-fixture",
        country_code="GB",
        operation_id="uk_frs_spine",
        pipeline_id="uk_national_spine",
        pipeline_version="2026.09",
        candidate_id="uk-spine-v2-fixture",
        local_dir=root / "completed-spine",
        run_kind="smoke",
        delivery_mode="local_only",
        repo_id=None,
        clock=FixtureClock(2),
    )
    recorder.set_sample({"mode": "full"})
    recorder.stage("input_verification", event_status="completed", input_count=3)
    recorder.stage("sampling", event_status="completed", household_rows=9)
    recorder.stage("construction", event_status="completed", produced_columns=12)
    recorder.stage("validation", event_status="completed", household_rows=9)
    recorder.stage("spine_h5_creation", event_status="completed", sha256="a" * 64)
    recorder.stage("sidecar_creation", event_status="completed", sha256="b" * 64)
    recorder.complete()
    recorder.validate_local_bundle()


def _calibration(root: Path) -> None:
    recorder = StagingTelemetryV2(
        run_id="uk-calibration-v2-fixture",
        country_code="GB",
        operation_id="uk_national_calibration",
        pipeline_id="uk_national_calibration",
        pipeline_version="2026.09",
        candidate_id="uk-calibration-candidate",
        release_id="populace-uk-2023-v2-fixture",
        local_dir=root / "calibration",
        run_kind="calibration",
        delivery_mode="local_only",
        repo_id=None,
        clock=FixtureClock(3),
    )
    recorder.set_sample({"mode": "full"})
    recorder.stage("target_compilation", event_status="completed", target_count=2)
    recorder.calibration_progress(
        {
            "kind": "calibration_epoch",
            "epoch": 1,
            "epochs": 3,
            "phase": "solve",
            "loss": 2.5,
            "iteration": 4,
        }
    )
    recorder.stage("diagnostics", event_status="completed", check_count=4)
    recorder.complete()
    recorder.validate_local_bundle()


def _failed(root: Path) -> None:
    recorder = StagingTelemetryV2(
        run_id="uk-failed-v2-fixture",
        country_code="GB",
        operation_id="uk_frs_spine",
        pipeline_id="uk_national_spine",
        pipeline_version="2026.09",
        candidate_id="uk-failed-v2-fixture",
        local_dir=root / "failed",
        run_kind="smoke",
        delivery_mode="local_only",
        repo_id=None,
        clock=FixtureClock(4),
    )
    recorder.stage("input_verification")
    recorder.fail(
        RuntimeError("/restricted/frs/adult.tab token=fixture-secret"),
        local_diagnostic_reference="diagnostics/operator-error.txt",
    )
    recorder.validate_local_bundle()


def _contract_cases(root: Path) -> None:
    cases = {
        "schema_name": "microcosm.staging.fixture-cases",
        "schema_version": 2,
        "delivery_success": {
            "contract_version": 2,
            "enabled": True,
            "mode": "local_and_remote",
            "run_id": "uk-delivery-success",
            "configured_repository": "policyengine/populace-uk-staging",
            "upload_attempts": 6,
            "upload_successes": 6,
            "read_back": "passed",
            "last_error_code": None,
            "opt_out_reason": None,
        },
        "delivery_failure": {
            "contract_version": 2,
            "enabled": True,
            "mode": "local_and_remote",
            "run_id": "uk-delivery-failure",
            "configured_repository": "policyengine/populace-uk-staging",
            "upload_attempts": 3,
            "upload_successes": 0,
            "read_back": "failed",
            "last_error_code": "READ_BACK_FAILED",
            "opt_out_reason": None,
        },
        "deliberate_opt_out": {
            "contract_version": 2,
            "enabled": False,
            "mode": "disabled",
            "run_id": None,
            "configured_repository": None,
            "upload_attempts": 0,
            "upload_successes": 0,
            "read_back": "not_requested",
            "last_error_code": None,
            "opt_out_reason": "--no-staging",
        },
        "unknown_version": {
            "schema_name": "microcosm.staging.run-manifest",
            "schema_version": 999,
        },
    }
    (root / "contract-cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n"
    )


def _generate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _completed_spine(root)
    _calibration(root)
    _failed(root)
    _contract_cases(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
        for path in files
    ]
    (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="microcosm-staging-fixtures-") as temporary:
        generated = Path(temporary) / "v2"
        _generate(generated)
        if args.check:
            if _tree(generated) != _tree(args.output):
                raise SystemExit("staging version 2 fixtures are not reproducible")
            return 0
        if args.output.exists():
            shutil.rmtree(args.output)
        shutil.copytree(generated, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
