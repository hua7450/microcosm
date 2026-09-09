"""Calibrate an existing UK national spine H5 without replaying source stages.

This driver is intentionally thin: it verifies pinned inputs, compiles the
Ledger target registry, applies the reviewed measure-exclusion register, records
any explicit doctrine overrides, and delegates the calibration/gate/logbook
work to :func:`microcosm.build.uk_runtime.calibration_run.run_uk_calibration`.

Signed deviation for v1: no sampling rungs and no checkpointing. This seam runs
full-scale only; scale ladders and resumable source-stage checkpoints belong to
the spine build lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.staging_cli import (
    add_uk_staging_arguments,
    validate_uk_staging_arguments,
)
from microcosm.build.staging_v2 import (
    StagingTelemetryV2,
    disabled_staging_delivery,
)
from microcosm.build.uk_runtime.calibration_run import (
    UKCalibrationRunPaths,
    run_uk_calibration,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry
from microcosm.build.uk_runtime.local_target_census import _LEDGER_FACT_FEED_PIN
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_doctrine import uk_doctrine_with_overrides
from microcosm.build.uk_runtime.release_identity import UK_NATIONAL_RELEASE_ID
from microcosm.calibrate import TargetRegistry

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_UK_RELEASE_ID = re.compile(
    r"populace-uk-[1-9][0-9]*-[a-z0-9_]+-k[1-9][0-9]*"
)
_UK_JUNE_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_input_paths(args)
    telemetry = _create_staging_telemetry(args)
    try:
        if telemetry is not None:
            telemetry.set_sample(_full_sample())
            telemetry.stage("target_compilation", event_status="started")
        artifact = load_ledger_consumer_artifact(
            args.ledger_facts,
            expected_facts_sha256=args.ledger_facts_sha256,
            expected_manifest_sha256=args.ledger_manifest_sha256,
        )
        _check_committed_ledger_feed_pin(
            artifact.facts_sha256,
            allow_unpinned_feed=args.allow_unpinned_feed,
        )
        calibration_year = load_uk_frs_release().calibration_year
        compilation = compile_uk_target_registry(
            artifact.facts, target_period=calibration_year
        )
        if compilation.unsupported:
            raise SystemExit(
                f"{len(compilation.unsupported)} target references failed to compile"
            )
        _compare_frozen_register(args.register_json, compilation.registry)
        exclusions = load_uk_calibration_measure_exclusions(args.measure_exclusions)
        registry, exclusion_receipt = apply_uk_calibration_measure_exclusions(
            compilation.registry, exclusions
        )
        if telemetry is not None:
            telemetry.stage(
                "target_compilation",
                event_status="completed",
                compiled_target_count=len(compilation.registry.specs),
                active_target_count=len(registry.specs),
            )
        overrides = {
            key: value
            for key, value in {
                "epochs": args.epochs,
                "target_weight_rule": args.target_weight_rule,
                "learning_rate": args.learning_rate,
                "target_loss_cap": args.target_loss_cap,
            }.items()
            if value is not None
        }
        doctrine, doctrine_overrides = uk_doctrine_with_overrides(**overrides)
        paths = UKCalibrationRunPaths(
            input_h5=args.input_h5,
            staging_h5=args.staging_h5,
            diagnostics_json=args.diagnostics_json,
            build_record_json=args.build_record_json,
            terminal_gate_json=args.terminal_gate_json,
        )
        # The resolver reads the input H5 to build its simulation, so the pin is
        # verified here first — no bytes are consumed before they match the CLI sha.
        measured_input_sha = _sha256_file(args.input_h5)
        if measured_input_sha != args.input_sha256:
            raise SystemExit(
                "error: --input-h5 sha mismatch: "
                f"measured {measured_input_sha}, pinned {args.input_sha256}"
            )
        resolver = UKMeasureResolver(
            simulation_source=args.input_h5,
            scratch_dir=args.staging_h5.parent,
            year=calibration_year,
            frame=None,
        )

        def event_callback(stage_id: str, status: str, details) -> None:
            if telemetry is not None:
                telemetry.stage(stage_id, event_status=status, **dict(details))

        result = run_uk_calibration(
            paths=paths,
            input_sha256=args.input_sha256,
            ledger_artifact=artifact,
            register_registry=registry,
            band_edge_registry=compilation.registry,
            calibration_year=calibration_year,
            exclusion_receipt=exclusion_receipt,
            doctrine=doctrine,
            doctrine_overrides=doctrine_overrides,
            measure_resolver=resolver,
            source_pins={
                "input_h5": {
                    "sha256": args.input_sha256,
                    "size_bytes": args.input_h5.stat().st_size,
                },
                "ledger_facts": _ledger_facts_pin(artifact),
            },
            run_config_extra={
                "calibration_year": calibration_year,
                "allow_unpinned_feed": args.allow_unpinned_feed,
            },
            release_id=args.release_id,
            logbook_prev_row_digest=args.logbook_prev_row_digest,
            progress_callback=(
                telemetry.calibration_progress if telemetry is not None else None
            ),
            event_callback=event_callback if telemetry is not None else None,
            staging_delivery=_staging_delivery(args, telemetry),
        )
        build_record_sha256 = result.build_record_sha256
        build_record = dict(result.build_record)
        if telemetry is not None:
            telemetry.complete(message="UK calibration staging run completed.")
            if args.staging_read_back:
                telemetry.verify_remote()
            telemetry.validate_local_bundle()
            build_record["staging_delivery"] = telemetry.delivery_summary
            _write_json(args.build_record_json, build_record)
            build_record_sha256 = _sha256_file(args.build_record_json)
        summary = {
            "staging_h5_sha256": result.staging_sha256,
            "diagnostics_sha256": result.diagnostics_sha256,
            "terminal_gate_sha256": result.terminal_gate_sha256,
            "build_record_sha256": build_record_sha256,
            "gate_verdicts": build_record["gate_summary"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        if telemetry is not None and telemetry.status == "running":
            try:
                telemetry.fail(error)
                telemetry.validate_local_bundle()
            except Exception:
                pass
        raise


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True, type=_sha256)
    parser.add_argument("--ledger-facts", required=True, type=Path)
    parser.add_argument("--ledger-facts-sha256", required=True, type=_sha256)
    parser.add_argument("--ledger-manifest-sha256", required=True, type=_sha256)
    parser.add_argument(
        "--allow-unpinned-feed",
        action="store_true",
        help=(
            "Allow a Chronicle fact feed whose content hash differs from the "
            "committed UK feed pin; the override is recorded in the run manifest."
        ),
    )
    parser.add_argument("--staging-h5", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--build-record-json", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--terminal-gate-json", type=Path)
    parser.add_argument("--register-json", type=Path)
    parser.add_argument("--measure-exclusions", type=Path)
    parser.add_argument("--release-candidate", action="store_true")
    parser.add_argument("--logbook-prev-row-digest", type=_sha256)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--target-weight-rule")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--target-loss-cap", type=float)
    add_uk_staging_arguments(parser)
    args = parser.parse_args(argv)
    args.terminal_gate_json = args.terminal_gate_json or args.staging_h5.with_suffix(
        ".terminal_gates.json"
    )
    if args.release_candidate:
        # The seam refuses release-candidate posture outright (the #757
        # release-cut audit, issue comment 5413502559): its scoped battery
        # covers 6 of the declared entries and must never sign a
        # shippability claim, and its evidence_absent gaps already refuse
        # upstream. A candidate's verdict comes only from the release-cut
        # certification producer (tools/certify_uk_release_cut.py).
        parser.error(
            "--release-candidate is refused on the calibration seam: the "
            "seam's scoped battery cannot sign shippability; run the "
            "release-cut certification producer instead"
        )
    if (
        _CANONICAL_UK_RELEASE_ID.fullmatch(args.release_id)
        or args.release_id == _UK_JUNE_RELEASE_ID
        or args.release_id == UK_NATIONAL_RELEASE_ID
    ):
        parser.error(
            "canonical UK release ids belong to the release-cut "
            "certification producer; the seam runs under a staging or dev "
            "release id"
        )
    _validate_distinct_paths(
        {
            "--input-h5": args.input_h5,
            "--ledger-facts": args.ledger_facts,
            "--staging-h5": args.staging_h5,
            "--diagnostics-json": args.diagnostics_json,
            "--build-record-json": args.build_record_json,
            "--terminal-gate-json": args.terminal_gate_json,
            **({"--register-json": args.register_json} if args.register_json else {}),
            **(
                {"--measure-exclusions": args.measure_exclusions}
                if args.measure_exclusions
                else {}
            ),
        }
    )
    validate_uk_staging_arguments(parser, args)
    return args


def _validate_input_paths(args: argparse.Namespace) -> None:
    if not args.input_h5.is_file():
        raise SystemExit(f"error: --input-h5 does not exist: {args.input_h5}")
    if not args.ledger_facts.exists():
        raise SystemExit(f"error: --ledger-facts does not exist: {args.ledger_facts}")


def _create_staging_telemetry(args: argparse.Namespace) -> StagingTelemetryV2 | None:
    if args.no_staging:
        return None
    local_only = args.staging_local_only
    return StagingTelemetryV2(
        run_id=args.staging_run_id or f"{args.release_id}-calibration",
        country_code="GB",
        operation_id="uk_national_calibration",
        pipeline_id="uk-frs-calibration",
        pipeline_version=metadata.version("microcosm-build"),
        candidate_id=args.staging_candidate_id or args.release_id,
        local_dir=args.staging_dir or args.staging_h5.parent / "staging",
        run_kind="calibration",
        delivery_mode="local_only" if local_only else "local_and_remote",
        repo_id=None if local_only else args.staging_repo_id,
        path_prefix=args.staging_prefix,
        upload_interval_seconds=args.staging_upload_interval_seconds,
    )


def _full_sample() -> dict[str, object]:
    return {
        "mode": "full",
        "requested_source_households": None,
        "eligible_source_families": None,
        "proportional_request": None,
        "forced_additions": 0,
        "realized_source_families": None,
        "realized_household_rows": None,
        "seed": None,
        "receipt_sha256": None,
    }


def _staging_delivery(
    args: argparse.Namespace, telemetry: StagingTelemetryV2 | None
) -> dict[str, object]:
    if telemetry is None:
        return disabled_staging_delivery("--no-staging")
    return telemetry.delivery_summary


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_committed_ledger_feed_pin(
    facts_sha256: str,
    *,
    allow_unpinned_feed: bool,
) -> None:
    committed = _LEDGER_FACT_FEED_PIN["facts_sha256"]
    if facts_sha256 != committed and not allow_unpinned_feed:
        raise SystemExit(
            "error: Chronicle consumer facts differ from the committed UK feed pin: "
            f"loaded {facts_sha256}, committed {committed}; pass "
            "--allow-unpinned-feed only for an explicitly reviewed diagnostic run"
        )


def _validate_distinct_paths(paths: dict[str, Path]) -> None:
    resolved = {label: path.expanduser().resolve() for label, path in paths.items()}
    for (left_label, left), (right_label, right) in combinations(resolved.items(), 2):
        if _paths_alias(left, right):
            raise SystemExit(
                f"error: {left_label} and {right_label} must be distinct paths "
                f"({left} aliases {right})"
            )


def _paths_alias(left: Path, right: Path) -> bool:
    if str(left).casefold() == str(right).casefold():
        return True
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except FileNotFoundError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def _compare_frozen_register(path: Path | None, registry: Any) -> None:
    """Refuse when the compiled register is not the frozen scoring surface.

    Both sides are compared on ``TargetRegistry.version`` — the registry's own
    content hash — through the validating loader, so the same artifact serves
    this check and ``tools/score_uk_national_candidate.py``, and incidental
    formatting cannot make two identical registers look different.
    """

    if path is None:
        return
    try:
        frozen = TargetRegistry.from_json(path)
    except ValueError as error:
        raise SystemExit(
            f"error: frozen scoring register is unusable: {error}"
        ) from error
    if frozen.version != registry.version:
        raise SystemExit(
            "re-derived register differs from the frozen scoring register: "
            f"{registry.version} vs {frozen.version}"
        )


def _ledger_facts_pin(artifact: Any) -> dict[str, object]:
    facts_path = (
        artifact.path / "consumer_facts.jsonl"
        if artifact.path.is_dir()
        else artifact.path
    )
    return {"sha256": artifact.facts_sha256, "size_bytes": facts_path.stat().st_size}


if __name__ == "__main__":
    raise SystemExit(main())
