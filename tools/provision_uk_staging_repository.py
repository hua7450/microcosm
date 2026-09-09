#!/usr/bin/env python3
"""Provision or audit the access-controlled UK staging telemetry repository."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from microcosm.build.staging_v2 import DEFAULT_UK_STAGING_REPO


def _settings(api: HfApi, repo_id: str) -> dict[str, object]:
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    return {
        "repo_id": repo_id,
        "private": bool(info.private),
        "gated": info.gated,
    }


def _require_private_manual(settings: dict[str, object]) -> None:
    if settings["private"] is not True or settings["gated"] != "manual":
        raise RuntimeError(
            "UK staging must remain private with individual manual access approval; "
            f"observed private={settings['private']!r}, gated={settings['gated']!r}."
        )


def _apply(api: HfApi, repo_id: str) -> dict[str, object]:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    try:
        api.update_repo_settings(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            gated="manual",
        )
    except BaseException:
        # Configuration is fail-closed: no recovery path ever selects public
        # visibility, even if the manual-approval update fails.
        try:
            api.update_repo_settings(
                repo_id=repo_id,
                repo_type="dataset",
                private=True,
            )
        except BaseException:
            pass
        raise
    settings = _settings(api, repo_id)
    _require_private_manual(settings)
    return settings


def _verify_access(api: HfApi, repo_id: str) -> dict[str, object]:
    anonymous_refused = False
    try:
        HfApi(token=False).repo_info(repo_id=repo_id, repo_type="dataset")
    except HfHubHTTPError:
        anonymous_refused = True
    if not anonymous_refused:
        raise RuntimeError(
            "Anonymous access unexpectedly reached the private repository."
        )
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    return {
        "anonymous_access_refused": True,
        "approved_access_succeeded": bool(info.private),
    }


def _verify_write(api: HfApi, repo_id: str) -> dict[str, object]:
    path_in_repo = "verification/operator-write-probe.json"
    payload = {
        "schema_name": "microcosm.staging.operator-write-probe",
        "schema_version": 1,
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    with tempfile.TemporaryDirectory(prefix="microcosm-uk-staging-probe-") as temp:
        source = Path(temp) / "operator-write-probe.json"
        source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=source,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Verify UK staging operator write access",
        )
        api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Remove UK staging operator write probe",
        )
    return {"write_probe_succeeded": True, "write_probe_removed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_UK_STAGING_REPO)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify-access", action="store_true")
    parser.add_argument("--verify-write", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_write and not args.verify_access:
        parser.error("--verify-write requires --verify-access.")

    api = HfApi()
    result = _apply(api, args.repo_id) if args.apply else _settings(api, args.repo_id)
    _require_private_manual(result)
    if args.verify_access:
        result.update(_verify_access(api, args.repo_id))
    if args.verify_write:
        result.update(_verify_write(api, args.repo_id))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
