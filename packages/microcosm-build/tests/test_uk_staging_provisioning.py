"""Idempotent external-resource setup stays private and externally read-only."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeHuggingFaceApi:
    def __init__(self, *, fail_manual: bool = False) -> None:
        self.fail_manual = fail_manual
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.private = False
        self.gated: str | bool = False

    def create_repo(self, **kwargs) -> None:
        self.calls.append(("create_repo", kwargs))
        self.private = kwargs["private"] is True

    def update_repo_settings(self, **kwargs) -> None:
        self.calls.append(("update_repo_settings", kwargs))
        if kwargs.get("gated") == "manual" and self.fail_manual:
            raise RuntimeError("manual approval unavailable")
        if "private" in kwargs:
            self.private = kwargs["private"] is True
        if "gated" in kwargs:
            self.gated = kwargs["gated"]

    def upload_file(self, **kwargs) -> None:
        self.calls.append(("upload_file", kwargs))

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=self.private, gated=self.gated)


def test_hugging_face_setup_is_idempotent_and_private() -> None:
    tool = _load_tool("provision_uk_staging_repository")
    api = _FakeHuggingFaceApi()

    first = tool._apply(api, "policyengine/populace-uk-staging")
    second = tool._apply(api, "policyengine/populace-uk-staging")

    assert (
        first
        == second
        == {
            "repo_id": "policyengine/populace-uk-staging",
            "private": True,
            "gated": "manual",
        }
    )
    creates = [payload for name, payload in api.calls if name == "create_repo"]
    assert len(creates) == 2
    assert all(
        payload["private"] is True and payload["exist_ok"] is True
        for payload in creates
    )


def test_hugging_face_setup_failure_never_selects_public_visibility() -> None:
    tool = _load_tool("provision_uk_staging_repository")
    api = _FakeHuggingFaceApi(fail_manual=True)

    with pytest.raises(RuntimeError, match="manual approval unavailable"):
        tool._apply(api, "policyengine/populace-uk-staging")

    settings_updates = [
        payload for name, payload in api.calls if name == "update_repo_settings"
    ]
    assert settings_updates[-1] == {
        "repo_id": "policyengine/populace-uk-staging",
        "repo_type": "dataset",
        "private": True,
    }
    assert all(payload.get("private") is not False for payload in settings_updates)
    assert api.private is True


def test_github_environment_audit_allows_only_optional_read_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool("configure_github_staging_environment")
    environment = {
        "protection_rules": [
            {
                "type": "required_reviewers",
                "prevent_self_review": True,
                "reviewers": [
                    {"type": "User", "reviewer": {"login": "anth-volk"}},
                    {"type": "User", "reviewer": {"login": "MaxGhenis"}},
                ],
            }
        ]
    }
    monkeypatch.setattr(tool, "_environment", lambda repository: environment)
    monkeypatch.setattr(
        tool, "_secret_names", lambda repository: ["HF_STAGING_READ_TOKEN"]
    )

    result = tool._audit("PolicyEngine/microcosm", ("anth-volk", "MaxGhenis"))

    assert result["prevent_self_review"] is True
    assert result["external_writer_secret_present"] is False
    assert result["secret_names"] == ["HF_STAGING_READ_TOKEN"]


def test_github_environment_audit_rejects_writer_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool("configure_github_staging_environment")
    monkeypatch.setattr(
        tool, "_environment", lambda repository: {"protection_rules": []}
    )
    monkeypatch.setattr(
        tool, "_secret_names", lambda repository: ["HF_STAGING_WRITE_TOKEN"]
    )

    with pytest.raises(RuntimeError, match="unapproved secret"):
        tool._audit("PolicyEngine/microcosm", ())
