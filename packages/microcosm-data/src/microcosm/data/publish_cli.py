"""CLI for contract-gated Microcosm release publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from microcosm.data.release import publish_release


def _non_release_artifact(release_dir: Path) -> bool:
    """Return whether the manifest explicitly identifies non-release output."""

    path = release_dir / "build_manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and (
        manifest.get("non_release") is True
        or manifest.get("release_posture") == "non_release_smoke"
    )


def _staging_undelivered(release_dir: Path) -> bool:
    """True if a build that should have staged has nothing to show for it.

    The gate is scoped by the presence of the ``staging`` key. Builders with
    no staging path are untouched, while an explicit ``enabled: false`` is a
    legitimate opt-out. A present but empty block, or an enabled run with no
    successful upload, records intended staging that never arrived.
    """
    path = release_dir / "build_manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(manifest, dict) or "staging" not in manifest:
        return False
    staging = manifest["staging"]
    if not isinstance(staging, dict) or not staging:
        return True
    if "contract_version" in staging:
        return _version_2_staging_undelivered(staging)
    if staging.get("enabled") is False:
        return not bool(staging.get("reason"))
    return not staging.get("uploads_succeeded")


def _version_2_staging_undelivered(staging: dict[str, object]) -> bool:
    """Validate publication-relevant version 2 delivery semantics."""

    required = {
        "contract_version",
        "enabled",
        "mode",
        "run_id",
        "configured_repository",
        "upload_attempts",
        "upload_successes",
        "read_back",
        "last_error_code",
        "opt_out_reason",
    }
    if set(staging) != required or staging.get("contract_version") != 2:
        return True
    enabled = staging.get("enabled")
    mode = staging.get("mode")
    run_id = staging.get("run_id")
    repository = staging.get("configured_repository")
    attempts = staging.get("upload_attempts")
    successes = staging.get("upload_successes")
    read_back = staging.get("read_back")
    reason = staging.get("opt_out_reason")
    if (
        not isinstance(enabled, bool)
        or isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 0
        or isinstance(successes, bool)
        or not isinstance(successes, int)
        or successes < 0
        or successes > attempts
        or read_back not in {"not_requested", "passed", "failed"}
    ):
        return True
    if not enabled:
        return not (
            mode == "disabled"
            and run_id is None
            and repository is None
            and attempts == 0
            and successes == 0
            and isinstance(reason, str)
            and bool(reason.strip())
        )
    if reason is not None or not isinstance(run_id, str) or not run_id:
        return True
    if mode == "local_only":
        return True
    if mode != "local_and_remote" or not isinstance(repository, str) or not repository:
        return True
    return successes == 0


def _reform_validation_skipped(release_dir: Path) -> bool:
    """True if the release carries a reform_validation.json that was built with
    out-of-sample reforms skipped (``out_of_sample_simulated`` false).

    Such a release publishes blank out-of-sample (OBBBA) rows on the dashboard,
    indistinguishable from a real result — so publishing one is refused unless
    explicitly allowed. Absent/unreadable file or a true flag → not skipped.
    """
    path = release_dir / "reform_validation.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return payload.get("out_of_sample_simulated") is False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", help="Local releases/<release_id> directory.")
    parser.add_argument(
        "--repo-id",
        default="policyengine/populace-us",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--artifact-root",
        help=(
            "Directory holding root artifacts named by release_manifest.json, "
            "for example populace_us_2024.h5."
        ),
    )
    parser.add_argument(
        "--create-tag",
        action="store_true",
        default=True,
        help=(
            "Create a Hugging Face tag for the immutable release before any "
            "main-branch update."
        ),
    )
    parser.add_argument(
        "--no-create-tag",
        action="store_false",
        dest="create_tag",
        help=(
            "Skip creating a Hugging Face tag. Refused when the release "
            "manifest pins artifacts to the release id."
        ),
    )
    parser.add_argument(
        "--tag-name",
        help="Optional tag name override; defaults to the release id.",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Additional release-dir file to include in the immutable release.",
    )
    parser.add_argument(
        "--updated-at",
        help="Optional latest.json timestamp override for reproducible tests.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help=(
            "Publish as a non-default release: upload files and create the "
            "immutable tag, but never touch latest.json (the default pointer)."
        ),
    )
    parser.add_argument(
        "--tag-only",
        action="store_true",
        help=(
            "Publish only the immutable tag revision, without any main-branch "
            "commit. Requires --no-latest and tag creation; used by exact-k "
            "ladder candidates."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-reform-validation",
        action="store_true",
        help=(
            "Publish even if reform_validation.json was built with out-of-sample "
            "reforms skipped (out_of_sample_simulated false). Off by default so a "
            "release never silently ships blank OBBBA validation."
        ),
    )
    parser.add_argument(
        "--allow-missing-staging",
        action="store_true",
        help=(
            "Publish even if the build recorded no staging telemetry, or "
            "recorded staging that never uploaded anything. Off by default so "
            "a release that never appeared on the staging dashboard is not "
            "shipped unnoticed. A declared --no-staging build publishes "
            "without this flag."
        ),
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help=(
            "Publish at the EVIDENCE tier (microcosm#506): validate against the "
            "evidence release contract (which requires a non-empty "
            "known_failures block) and move latest-evidence.json instead of "
            "latest.json. Structurally never touches the certified pointer; a "
            "certified-shape release is refused under this flag."
        ),
    )
    args = parser.parse_args(argv)

    if args.tag_only and not args.no_latest:
        parser.error("--tag-only requires --no-latest.")
    if args.tag_only and not args.create_tag:
        parser.error("--tag-only requires tag creation; remove --no-create-tag.")

    if _non_release_artifact(Path(args.release_dir)):
        print(
            "refusing to publish: build_manifest.json identifies this as "
            "non-release smoke output.",
            file=sys.stderr,
        )
        return 1

    if not args.allow_incomplete_reform_validation and _reform_validation_skipped(
        Path(args.release_dir)
    ):
        print(
            "refusing to publish: reform_validation.json has "
            "out_of_sample_simulated=false (built with "
            "--skip-out-of-sample-reforms), so the dashboard would show blank "
            "out-of-sample reforms. Rebuild without skipping, or pass "
            "--allow-incomplete-reform-validation to publish anyway.",
            file=sys.stderr,
        )
        return 1

    if not args.allow_missing_staging and _staging_undelivered(Path(args.release_dir)):
        print(
            "refusing to publish: this release's build_manifest records no "
            "delivered staging telemetry, so the build never appeared on the "
            "staging dashboard and there is no pre-publication review of it. "
            "Either the staging destination was lost mid-build (uploads "
            "self-disable after repeated failures — check the build machine's "
            "Hugging Face write token), or the manifest predates staging "
            "provenance. Rebuild with staging reaching its repo, or pass "
            "--allow-missing-staging to publish anyway. A build that declared "
            "--no-staging publishes without the flag.",
            file=sys.stderr,
        )
        return 1

    pointer = publish_release(
        Path(args.release_dir),
        args.repo_id,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        create_tag=args.create_tag,
        tag_name=args.tag_name,
        extra_files=tuple(args.extra_file),
        updated_at=args.updated_at,
        update_latest=not args.no_latest,
        tag_only=args.tag_only,
        evidence=args.evidence,
    )
    print(json.dumps(pointer, indent=2))

    # The Slack release alert now fires inside publish_release (coupled to the
    # promotion, so every publish path announces the release), warning loudly if
    # the webhook is unset. Nothing to do here.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
