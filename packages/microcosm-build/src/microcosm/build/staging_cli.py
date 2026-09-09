"""Shared command-line options for UK staging telemetry version 2."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from microcosm.build.staging_v2 import (
    DEFAULT_STAGING_PREFIX,
    DEFAULT_UK_STAGING_REPO,
)

__all__ = ["add_uk_staging_arguments", "validate_uk_staging_arguments"]


def _environment_default(name: str, fallback: str) -> str:
    return os.environ[name] if name in os.environ else fallback


def add_uk_staging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common version 2 staging controls to a UK build parser."""

    parser.add_argument(
        "--staging-dir",
        type=Path,
        help="Local root for version 2 staging files; defaults beside the output.",
    )
    parser.add_argument(
        "--staging-repo-id",
        default=_environment_default(
            "POPULACE_UK_STAGING_REPO_ID", DEFAULT_UK_STAGING_REPO
        ),
        help=(
            "Access-controlled Hugging Face dataset repository for best-effort "
            "telemetry delivery."
        ),
    )
    parser.add_argument(
        "--staging-prefix",
        default=_environment_default(
            "POPULACE_UK_STAGING_PREFIX", DEFAULT_STAGING_PREFIX
        ),
        help="Repository-relative prefix for immutable run directories.",
    )
    parser.add_argument(
        "--staging-run-id",
        help="Override the staging run identifier.",
    )
    parser.add_argument(
        "--staging-candidate-id",
        help="Override the staging candidate identifier.",
    )
    parser.add_argument(
        "--staging-upload-interval-seconds",
        type=float,
        default=30.0,
        help="Minimum interval between best-effort progress uploads.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--staging-local-only",
        action="store_true",
        help="Write and validate staging files locally without a remote client.",
    )
    mode.add_argument(
        "--no-staging",
        action="store_true",
        help="Deliberately disable staging telemetry for this build.",
    )
    parser.add_argument(
        "--staging-read-back",
        action="store_true",
        help="After final upload, authenticate and validate the remote core files.",
    )


def validate_uk_staging_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject contradictory or incomplete UK staging configuration."""

    interval = args.staging_upload_interval_seconds
    if not math.isfinite(interval) or interval < 0.0:
        parser.error("--staging-upload-interval-seconds must be finite and non-negative.")
    if args.staging_read_back and (args.staging_local_only or args.no_staging):
        parser.error("--staging-read-back requires remote staging.")
    if not args.staging_local_only and not args.no_staging:
        if not isinstance(args.staging_repo_id, str) or not args.staging_repo_id.strip():
            parser.error("remote staging requires a non-empty --staging-repo-id.")
