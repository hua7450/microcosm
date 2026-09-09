#!/usr/bin/env python3
"""Create or audit the externally read-only GitHub staging environment."""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

DEFAULT_REPOSITORY = "PolicyEngine/microcosm"
DEFAULT_REVIEWERS = ("anth-volk", "MaxGhenis")
ALLOWED_SECRET_NAMES = {"HF_STAGING_READ_TOKEN"}


def _gh(*arguments: str, input_payload: object | None = None) -> Any:
    result = subprocess.run(
        ["gh", "api", *arguments],
        input=(None if input_payload is None else json.dumps(input_payload)),
        text=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def _reviewer(login: str) -> dict[str, object]:
    payload = _gh(f"users/{login}")
    return {"type": "User", "id": int(payload["id"])}


def _environment(repository: str) -> dict[str, object] | None:
    result = subprocess.run(
        ["gh", "api", f"repos/{repository}/environments/staging"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _secret_names(repository: str) -> list[str]:
    payload = _gh(f"repos/{repository}/environments/staging/secrets")
    return sorted(str(row["name"]) for row in payload.get("secrets", []))


def _audit(repository: str, reviewers: tuple[str, ...]) -> dict[str, object]:
    environment = _environment(repository)
    if environment is None:
        return {
            "repository": repository,
            "environment": "staging",
            "exists": False,
        }
    configured_reviewers = sorted(
        str(row["reviewer"]["login"])
        for row in environment.get("protection_rules", [])
        if row.get("type") == "required_reviewers"
        for row in row.get("reviewers", [])
        if row.get("type") == "User"
    )
    secrets = _secret_names(repository)
    unexpected_secrets = sorted(set(secrets) - ALLOWED_SECRET_NAMES)
    if unexpected_secrets:
        raise RuntimeError(
            "The staging environment contains unapproved secret name(s): "
            f"{unexpected_secrets}. Remove them before enabling the workflow."
        )
    missing_reviewers = sorted(set(reviewers) - set(configured_reviewers))
    if missing_reviewers:
        raise RuntimeError(
            f"The staging environment is missing reviewer(s): {missing_reviewers}."
        )
    return {
        "repository": repository,
        "environment": "staging",
        "exists": True,
        "required_reviewers": configured_reviewers,
        "prevent_self_review": any(
            row.get("prevent_self_review") is True
            for row in environment.get("protection_rules", [])
            if row.get("type") == "required_reviewers"
        ),
        "secret_names": secrets,
        "external_writer_secret_present": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument(
        "--reviewer",
        action="append",
        dest="reviewers",
        help="Required reviewer login; repeat for multiple users.",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    reviewers = tuple(args.reviewers or DEFAULT_REVIEWERS)
    if not reviewers:
        parser.error("at least one required reviewer is required")

    if args.apply:
        _gh(
            "--method",
            "PUT",
            f"repos/{args.repository}/environments/staging",
            "--input",
            "-",
            input_payload={
                "wait_timer": 0,
                "prevent_self_review": True,
                "reviewers": [_reviewer(login) for login in reviewers],
            },
        )
    result = _audit(args.repository, reviewers)
    if args.apply and result.get("exists") is not True:
        raise RuntimeError("GitHub did not return the configured staging environment.")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
