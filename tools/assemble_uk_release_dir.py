"""Assemble a certified UK national candidate into a publishable release.

This derivative packaging step closes the certification/build-record identity
join, mints the calibration weight bundle, writes the two release manifests,
copies the signed evidence byte-for-byte, and validates the finished release
directory before printing the inspect-lane publication command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from microcosm.build.staging_v2 import (
    StagingContractError,
    validate_staging_delivery,
)
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.release_identity import UK_NATIONAL_RELEASE_ID
from microcosm.data.contract import validate_release_dir

_ATTEMPT_PREFIX = "uk-frs-calibration-attempt-"
_ATTEMPT_SUFFIX = re.compile(r"(?P<timestamp>\d{8}T\d{6}Z)-(?P<uuid>[0-9a-f]{8})")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_REPO_ID = "policyengine/populace-uk-private"
_RUNTIME_PACKAGES = (
    "python",
    "policyengine-core",
    "policyengine-uk",
    "microcosm-data",
    "microcosm-build",
    "microcosm-calibrate",
)


def _refuse_non_release_smoke_spine(spine_h5: Path) -> None:
    """Reject bounded smoke output before any release files are created."""

    sidecar_path = spine_h5.with_suffix(".build.json")
    if sidecar_path.is_file():
        sidecar = _load_json(sidecar_path, label="spine build sidecar")
        if sidecar.get("non_release") is True or sidecar.get(
            "release_posture"
        ) == "non_release_smoke":
            raise SystemExit(
                "error: release assembly refuses a non-release smoke spine sidecar"
            )
    if not spine_h5.is_file():
        return
    try:
        import h5py

        with h5py.File(spine_h5, mode="r") as file:
            non_release = bool(file.attrs.get("populace_non_release", False))
            posture = file.attrs.get("populace_release_posture", "")
            if isinstance(posture, bytes):
                posture = posture.decode("utf-8")
    except OSError:
        return
    if non_release or posture == "smoke":
        raise SystemExit("error: release assembly refuses a non-release smoke H5")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = _assemble(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _assemble(args: argparse.Namespace) -> dict[str, object]:
    _refuse_non_release_smoke_spine(args.spine_h5)
    certification_bytes = args.certification_json.read_bytes()
    certification = _load_json_bytes(
        certification_bytes, label="--certification-json"
    )
    build_record = _load_json(args.build_record_json, label="--build-record-json")
    raw_staging_delivery = build_record.get("staging_delivery")
    if not isinstance(raw_staging_delivery, Mapping):
        raise SystemExit(
            "error: build record is missing valid staging-delivery evidence"
        )
    try:
        staging_delivery = validate_staging_delivery(raw_staging_delivery)
    except StagingContractError as error:
        raise SystemExit(f"error: invalid staging-delivery evidence: {error}") from error
    diagnostics_bytes = args.diagnostics_json.read_bytes()
    diagnostics = _load_json_bytes(diagnostics_bytes, label="--diagnostics-json")

    spine_report_path = args.spine_h5.with_suffix(".spine_gates.json")
    inputs = {
        "candidate": args.candidate_h5,
        "certification": args.certification_json,
        "diagnostics": args.diagnostics_json,
        "calibration seam report": args.seam_gate_report,
        "release-cut report": args.release_cut_gate_json,
        "spine report": spine_report_path,
        "score receipt": args.score_receipt,
        "spine H5": args.spine_h5,
    }
    measured = {name: _sha256(path) for name, path in inputs.items()}

    candidate = _mapping(certification.get("candidate"), "certification.candidate")
    artifacts = _mapping(build_record.get("artifacts"), "build_record.artifacts")
    staging_h5 = _mapping(
        artifacts.get("staging_h5"), "build_record.artifacts.staging_h5"
    )
    diagnostics_artifact = _mapping(
        artifacts.get("diagnostics_json"),
        "build_record.artifacts.diagnostics_json",
    )
    terminal_artifact = _mapping(
        artifacts.get("terminal_gate_json"),
        "build_record.artifacts.terminal_gate_json",
    )
    parts = _mapping(certification.get("parts"), "certification.parts")
    spine_part = _mapping(parts.get("spine"), "certification.parts.spine")
    seam_part = _mapping(
        parts.get("calibration_seam"), "certification.parts.calibration_seam"
    )
    release_cut_part = _mapping(
        parts.get("release_cut"), "certification.parts.release_cut"
    )
    score_binding = _mapping(
        certification.get("score_receipt"), "certification.score_receipt"
    )
    input_posture = _mapping(
        build_record.get("input_posture"), "build_record.input_posture"
    )

    _require_equal(
        "candidate bytes vs certification.candidate.sha256",
        measured["candidate"],
        candidate.get("sha256"),
    )
    _require_equal(
        "candidate bytes vs build_record.artifacts.staging_h5.sha256",
        measured["candidate"],
        staging_h5.get("sha256"),
    )
    _require_equal(
        "diagnostics bytes vs certification.diagnostics_sha256",
        measured["diagnostics"],
        certification.get("diagnostics_sha256"),
    )
    _require_equal(
        "diagnostics bytes vs build_record.artifacts.diagnostics_json.sha256",
        measured["diagnostics"],
        diagnostics_artifact.get("sha256"),
    )
    _require_equal(
        "calibration seam report vs build_record.artifacts.terminal_gate_json.sha256",
        measured["calibration seam report"],
        terminal_artifact.get("sha256"),
    )
    _require_equal(
        "calibration seam report vs certification.parts.calibration_seam.sha256",
        measured["calibration seam report"],
        seam_part.get("sha256"),
    )
    _require_equal(
        "release-cut report vs certification.parts.release_cut.sha256",
        measured["release-cut report"],
        release_cut_part.get("sha256"),
    )
    _require_equal(
        "spine report vs certification.parts.spine.sha256",
        measured["spine report"],
        spine_part.get("sha256"),
    )
    _require_equal(
        "score receipt vs certification.score_receipt.sha256",
        measured["score receipt"],
        score_binding.get("sha256"),
    )
    _require_equal(
        "spine H5 vs build_record.input_posture.sha256",
        measured["spine H5"],
        input_posture.get("sha256"),
    )
    _require_equal(
        "certification.release_id",
        certification.get("release_id"),
        UK_NATIONAL_RELEASE_ID,
    )
    if certification.get("shippable") is not True:
        raise SystemExit(
            "error: certification.shippable must be true before release assembly"
        )

    # Release identity comes only from the signed diagnostics build block
    # (its bytes are pinned by certification.diagnostics_sha256). The
    # separately supplied build record is convenience input and must agree
    # with the authenticated fields, never substitute for them.
    build_block = _mapping(diagnostics.get("build"), "diagnostics.build")
    attempt_id = build_block.get("build_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise SystemExit(
            "error: diagnostics.build.build_id must be a non-empty string"
        )
    _require_equal(
        "build_record.build_id vs signed diagnostics.build.build_id",
        build_record.get("build_id"),
        attempt_id,
    )
    signed_input_posture = _mapping(
        build_block.get("input_posture"), "diagnostics.build.input_posture"
    )
    _require_equal(
        "spine H5 vs signed diagnostics.build.input_posture.sha256",
        measured["spine H5"],
        signed_input_posture.get("sha256"),
    )
    cut_tag = _cut_tag(attempt_id, args.cut_tag)
    runtime = _runtime_versions(build_block, dict(args.runtime_version))
    code_pin = _diagnostics_code_pin(diagnostics)
    candidate_filename = candidate.get("filename")
    if not isinstance(candidate_filename, str) or not candidate_filename:
        raise SystemExit("error: certification.candidate.filename must be non-empty")
    dataset_key = Path(candidate_filename).stem
    calibration_filename = f"{dataset_key}_calibration.npz"
    # Nothing is written in place: assembly stages into a private directory,
    # validates there, and only then atomically renames into empty
    # destinations — a late failure can never leave a plausible partial
    # release or corrupt a previous assembly.
    destination = args.out_dir / UK_NATIONAL_RELEASE_ID
    if destination.exists() or destination.is_symlink():
        raise SystemExit(
            f"error: {destination} already exists; remove the previous "
            "assembly before re-assembling"
        )
    npz_destination = args.candidate_h5.parent / calibration_filename
    if npz_destination.exists() or npz_destination.is_symlink():
        raise SystemExit(
            f"error: {npz_destination} already exists; remove it before "
            "re-assembling"
        )
    staging_parent = args.out_dir / f".assemble-{uuid.uuid4().hex}"
    # The NPZ stages beside its own destination, not under staging_parent:
    # its finalizing rename must never cross a filesystem boundary, where
    # Path.rename raises and the staging cleanup would delete the artifact
    # out from under an already-renamed release directory.
    # The temp name must keep the .npz suffix: np.savez appends it to any
    # other filename, silently writing beside the path the code then hashes.
    calibration_path = npz_destination.with_name(
        f".{calibration_filename}.{uuid.uuid4().hex}.tmp.npz"
    )
    target_surface = _mapping(
        diagnostics.get("target_surface"), "diagnostics.target_surface"
    )
    target_registry = _mapping(
        diagnostics.get("target_registry"), "diagnostics.target_registry"
    )
    candidate_frame, _candidate_provenance = load_uk_national_frame(
        args.candidate_h5
    )
    spine_frame, _spine_provenance = load_uk_national_frame(args.spine_h5)
    # The NPZ pairs the two weight vectors row by row, so the household axes
    # must be identical — same ids, same order — before the pairing is
    # digested and shipped as authoritative. A digest over mis-paired bytes
    # authenticates the bytes, not the pairing.
    candidate_ids = candidate_frame.table("household")["household_id"].to_numpy()
    spine_ids = spine_frame.table("household")["household_id"].to_numpy()
    if candidate_ids.shape != spine_ids.shape or not np.array_equal(
        candidate_ids, spine_ids
    ):
        detail = (
            f"{candidate_ids.shape[0]} vs {spine_ids.shape[0]} household rows"
            if candidate_ids.shape != spine_ids.shape
            else "household ids differ"
        )
        raise SystemExit(
            f"error: candidate and spine household axes are misaligned "
            f"({detail}); the calibration NPZ would pair weights across "
            "different households"
        )
    household_weight = np.asarray(
        candidate_frame.weights_for("household").values, dtype=np.float64
    )
    initial_household_weight = np.asarray(
        spine_frame.weights_for("household").values, dtype=np.float64
    )

    staging_parent.mkdir(parents=True)
    try:
        return _stage_and_finalize(
            args=args,
            staging_parent=staging_parent,
            destination=destination,
            npz_destination=npz_destination,
            calibration_path=calibration_path,
            calibration_filename=calibration_filename,
            household_weight=household_weight,
            initial_household_weight=initial_household_weight,
            measured=measured,
            candidate_filename=candidate_filename,
            dataset_key=dataset_key,
            cut_tag=cut_tag,
            attempt_id=attempt_id,
            runtime=runtime,
            code_pin=code_pin,
            target_surface=target_surface,
            target_registry=target_registry,
            spine_part=spine_part,
            seam_part=seam_part,
            release_cut_part=release_cut_part,
            spine_report_path=spine_report_path,
            staging_delivery=staging_delivery,
        )
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
        calibration_path.unlink(missing_ok=True)


def _stage_and_finalize(
    *,
    args: argparse.Namespace,
    staging_parent: Path,
    destination: Path,
    npz_destination: Path,
    calibration_path: Path,
    calibration_filename: str,
    household_weight: np.ndarray,
    initial_household_weight: np.ndarray,
    measured: Mapping[str, str],
    candidate_filename: str,
    dataset_key: str,
    cut_tag: str,
    attempt_id: str,
    runtime: Mapping[str, str],
    code_pin: str,
    target_surface: Mapping[str, object],
    target_registry: Mapping[str, object],
    spine_part: Mapping[str, object],
    seam_part: Mapping[str, object],
    release_cut_part: Mapping[str, object],
    spine_report_path: Path,
    staging_delivery: Mapping[str, object],
) -> dict[str, object]:
    np.savez(
        calibration_path,
        household_weight=household_weight,
        initial_household_weight=initial_household_weight,
    )
    calibration_sha = _sha256(calibration_path)

    created_at = datetime.now(UTC).isoformat()
    release_dir = staging_parent / UK_NATIONAL_RELEASE_ID
    release_dir.mkdir(parents=True)
    build_manifest = {
        "build_id": UK_NATIONAL_RELEASE_ID,
        "code": {
            "repository": "PolicyEngine/microcosm",
            "git_commit": code_pin,
            "git_dirty": False,
        },
        "build_sha": code_pin[:7],
        "runtime": runtime,
        "dataset": {
            "filename": candidate_filename,
            "sha256": measured["candidate"],
        },
        "calibration": {
            "filename": calibration_filename,
            "sha256": calibration_sha,
            "target_surface": {
                "sha256": target_surface.get("sha256"),
                "n_targets": target_surface.get("n_targets"),
            },
            "target_registry": {
                "version": target_registry.get("version"),
                "n_specs": target_registry.get("n_specs"),
            },
        },
        # Signed sources only: the certification's part statuses. The build
        # record's gate summary is unsigned convenience input and deliberately
        # does not reach the manifest.
        "gates": {
            "spine": spine_part.get("statuses"),
            "calibration_seam": seam_part.get("statuses"),
            "release_cut": release_cut_part.get("statuses"),
        },
        "attempt_id": attempt_id,
        "cut_tag": cut_tag,
        "created_at": created_at,
        "staging": dict(staging_delivery),
    }
    build_manifest_path = release_dir / "build_manifest.json"
    _write_json(build_manifest_path, build_manifest)

    evidence = {
        "calibration_diagnostics": (
            args.diagnostics_json,
            "calibration_diagnostics.json",
        ),
        "release_certification": (
            args.certification_json,
            "release_certification.json",
        ),
        "terminal_gates": (args.seam_gate_report, "terminal_gates.json"),
        "release_cut_gates": (
            args.release_cut_gate_json,
            "release_cut_gates.json",
        ),
        "spine_gates": (spine_report_path, "spine_gates.json"),
        "score_vs_enhanced_frs": (
            args.score_receipt,
            "score_vs_enhanced_frs.json",
        ),
    }

    def artifact(kind: str, path: str, sha256: str) -> dict[str, str]:
        return {
            "kind": kind,
            "path": path,
            "repo_id": _REPO_ID,
            "revision": cut_tag,
            "sha256": sha256,
        }

    release_manifest = {
        "schema_version": 1,
        "data_package": {
            "name": "microcosm-data",
            "version": runtime["microcosm-data"],
        },
        "default_datasets": {"national": dataset_key},
        "build": {
            "build_id": UK_NATIONAL_RELEASE_ID,
            "built_at": created_at,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": runtime["policyengine-core"],
            },
            "built_with_model_package": {
                "name": "policyengine-uk",
                "version": runtime["policyengine-uk"],
            },
            "attempt_id": attempt_id,
            "cut_tag": cut_tag,
        },
        "compatible_core_packages": [
            {
                "name": "policyengine-core",
                "specifier": f"=={runtime['policyengine-core']}",
            }
        ],
        "compatible_model_packages": [
            {
                "name": "policyengine-uk",
                "specifier": f"=={runtime['policyengine-uk']}",
            }
        ],
        "artifacts": {
            dataset_key: artifact(
                "microdata", candidate_filename, measured["candidate"]
            ),
            f"{dataset_key}_calibration": artifact(
                "calibration", calibration_filename, calibration_sha
            ),
            **{
                key: artifact("diagnostics", filename, measured[name])
                for key, (_source, filename), name in (
                    (
                        "calibration_diagnostics",
                        evidence["calibration_diagnostics"],
                        "diagnostics",
                    ),
                    (
                        "release_certification",
                        evidence["release_certification"],
                        "certification",
                    ),
                    (
                        "terminal_gates",
                        evidence["terminal_gates"],
                        "calibration seam report",
                    ),
                    (
                        "release_cut_gates",
                        evidence["release_cut_gates"],
                        "release-cut report",
                    ),
                    (
                        "spine_gates",
                        evidence["spine_gates"],
                        "spine report",
                    ),
                    (
                        "score_vs_enhanced_frs",
                        evidence["score_vs_enhanced_frs"],
                        "score receipt",
                    ),
                )
            },
        },
        "description": (
            "UK national calibration pipeline release assembled from attempt "
            f"{attempt_id} at immutable cut {cut_tag}."
        ),
        "publisher_labels": {
            "obr": "Office for Budget Responsibility",
            "hmrc": "HM Revenue and Customs",
            "ons": "Office for National Statistics",
            "dwp": "Department for Work and Pensions",
        },
    }
    release_manifest_path = release_dir / "release_manifest.json"
    _write_json(release_manifest_path, release_manifest)

    evidence_summary: dict[str, dict[str, str]] = {}
    for key, (source, filename) in evidence.items():
        copy_path = release_dir / filename
        shutil.copyfile(source, copy_path)
        evidence_summary[key] = {
            "path": str(destination / filename),
            "sha256": _sha256(copy_path),
        }

    validate_release_dir(release_dir)

    # Only a directory that already validated moves into place, and only into
    # the destinations proven empty before staging began. Both renames stay
    # inside their own directory tree (the NPZ staged beside its
    # destination), so neither can fail on a filesystem boundary. The NPZ
    # lands first and the release directory is the commit point: a failure
    # between the two rolls the NPZ back rather than leaving a published
    # directory without its artifact.
    calibration_path.rename(npz_destination)
    try:
        release_dir.rename(destination)
    except BaseException:
        npz_destination.unlink(missing_ok=True)
        raise

    publish_command = shlex.join(
        [
            "uv",
            "run",
            "python",
            "-m",
            "microcosm.data.publish_cli",
            str(destination),
            "--repo-id",
            _REPO_ID,
            "--artifact-root",
            str(args.candidate_h5.parent),
            "--no-latest",
            "--tag-name",
            cut_tag,
        ]
    )
    return {
        "release_id": UK_NATIONAL_RELEASE_ID,
        "release_dir": str(destination),
        "cut_tag": cut_tag,
        "dataset": {
            "path": str(args.candidate_h5),
            "sha256": measured["candidate"],
        },
        "calibration": {
            "path": str(npz_destination),
            "sha256": calibration_sha,
        },
        "build_manifest": {
            "path": str(destination / "build_manifest.json"),
            "sha256": _sha256(destination / "build_manifest.json"),
        },
        "release_manifest": {
            "path": str(destination / "release_manifest.json"),
            "sha256": _sha256(destination / "release_manifest.json"),
        },
        "evidence": evidence_summary,
        "publish_command": publish_command,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-h5", required=True, type=Path)
    parser.add_argument("--spine-h5", required=True, type=Path)
    parser.add_argument("--certification-json", type=Path)
    parser.add_argument("--build-record-json", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--seam-gate-report", required=True, type=Path)
    parser.add_argument("--release-cut-gate-json", type=Path)
    parser.add_argument("--score-receipt", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("releases"))
    parser.add_argument("--cut-tag")
    parser.add_argument(
        "--runtime-version",
        action="append",
        default=[],
        type=_runtime_version,
        metavar="PACKAGE=VERSION",
    )
    args = parser.parse_args(argv)
    args.certification_json = (
        args.certification_json
        or args.candidate_h5.with_suffix(".release_certification.json")
    )
    args.release_cut_gate_json = (
        args.release_cut_gate_json
        or args.candidate_h5.with_suffix(".release_cut_gates.json")
    )
    distinct = {
        "--candidate-h5": args.candidate_h5,
        "--spine-h5": args.spine_h5,
        "--certification-json": args.certification_json,
        "--build-record-json": args.build_record_json,
        "--diagnostics-json": args.diagnostics_json,
        "--seam-gate-report": args.seam_gate_report,
        "--release-cut-gate-json": args.release_cut_gate_json,
        "--score-receipt": args.score_receipt,
    }
    resolved: dict[Path, str] = {}
    for flag, path in distinct.items():
        canonical = path.resolve()
        if canonical in resolved:
            parser.error(f"{flag} aliases {resolved[canonical]}: {path}")
        resolved[canonical] = flag
    return args


def _runtime_version(value: str) -> tuple[str, str]:
    package, separator, package_version = value.partition("=")
    if not separator or package not in _RUNTIME_PACKAGES or not package_version:
        raise argparse.ArgumentTypeError(
            "expected PACKAGE=VERSION for one of " + ", ".join(_RUNTIME_PACKAGES)
        )
    return package, package_version


def _runtime_versions(
    build_block: Mapping[str, object], overrides: dict[str, str]
) -> dict[str, str]:
    """Runtime pins from the signed diagnostics build block, and only there.

    The seam captures the calibrating environment's versions at solve time
    (``diagnostics.build.runtime``), signed with the diagnostics bytes. An
    override may re-assert an authenticated value but never replace it: a
    manifest pin that contradicts the signed provenance is an invented
    environment, which is exactly what the release contract cannot detect.
    """

    signed_runtime = _mapping(
        build_block.get("runtime"), "diagnostics.build.runtime"
    )
    runtime: dict[str, str] = {}
    for package in _RUNTIME_PACKAGES:
        value = signed_runtime.get(package)
        if not isinstance(value, str) or not value or value == "unavailable":
            raise SystemExit(
                f"error: diagnostics.build.runtime.{package!r} is missing or "
                "unresolved; the candidate must be re-cut by a seam that "
                "records runtime provenance"
            )
        override = overrides.get(package)
        if override is not None and override != value:
            raise SystemExit(
                f"error: --runtime-version {package}={override} contradicts "
                f"the signed provenance {value}"
            )
        runtime[package] = value
    return runtime


def _diagnostics_code_pin(diagnostics: Mapping[str, object]) -> str:
    build = _mapping(diagnostics.get("build"), "diagnostics.build")
    code_pin = build.get("code_pin")
    if not isinstance(code_pin, str) or not _GIT_COMMIT.fullmatch(code_pin):
        raise SystemExit(
            "error: diagnostics.build.code_pin must be a 40-character lowercase "
            "git commit"
        )
    return code_pin


def _cut_tag(attempt_id: str, override: str | None) -> str:
    prefix = UK_NATIONAL_RELEASE_ID + "-"
    if override is not None:
        # The override may re-tag a cut but never leave the grammar the
        # contract validates: prefix plus an attempt-shaped suffix.
        if not override.startswith(prefix) or not _ATTEMPT_SUFFIX.fullmatch(
            override[len(prefix) :]
        ):
            raise SystemExit(
                f"error: --cut-tag must be "
                f"{prefix}<YYYYMMDDTHHMMSSZ>-<uuid8>"
            )
        return override
    if not attempt_id.startswith(_ATTEMPT_PREFIX):
        raise SystemExit(
            f"error: build_record.build_id must start with {_ATTEMPT_PREFIX!r}"
        )
    suffix = attempt_id.removeprefix(_ATTEMPT_PREFIX)
    if not _ATTEMPT_SUFFIX.fullmatch(suffix):
        raise SystemExit(
            "error: build_record.build_id must end with "
            "<YYYYMMDDTHHMMSSZ>-<uuid8>"
        )
    return f"{UK_NATIONAL_RELEASE_ID}-{suffix}"


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    return _load_json_bytes(path.read_bytes(), label=label)


def _load_json_bytes(raw: bytes, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {label} is not valid JSON: {error}") from error
    return _mapping(payload, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"error: {label} must be a JSON object")
    return value


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"error: {label} mismatch: {actual!r} != {expected!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
