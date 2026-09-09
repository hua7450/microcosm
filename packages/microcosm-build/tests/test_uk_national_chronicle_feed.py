from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import pytest


def test_national_feed_records_the_complete_merged_source_artifact():
    from microcosm.build.uk_runtime.national_chronicle_feed import (
        load_uk_national_chronicle_feed,
    )

    pin = load_uk_national_chronicle_feed()
    resource = files("microcosm.build.uk").joinpath("national_chronicle_feed.json")
    raw = resource.read_bytes()
    assert pin.source_commit == "ec7169b5db40b9f54117c80f70f14efc1dd0fedd"
    assert pin.source_repo == "PolicyEngine/chronicle"
    assert pin.fact_row_count == 131450
    assert pin.facts_sha256 == (
        "4a50ee9568a01bbb57f73d927084ed6b4b9e52249b51a2338455874ae6e382b5"
    )
    assert pin.manifest_sha256 == (
        "a95d0ee9f87f36947eaecdb3de29cf81a91e47ccaa822fed42da677eedca877f"
    )
    assert pin.artifact_schema_version == "policyengine_ledger.consumer_artifact.v2"
    assert pin.consumer_fact_schema_sha256 == (
        "76ac268e626c86146cee51193e0cbecbb197ddbf3bf410156fe7da7c0edae3ad"
    )
    assert pin.resource_sha256 == hashlib.sha256(raw).hexdigest()
    assert pin.resource_size_bytes == len(raw)
    assert pin.to_dict()["source_commit"] == pin.source_commit


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("facts_sha256", "not-a-digest"),
        ("manifest_sha256", "A" * 64),
        ("consumer_fact_schema_sha256", "a" * 63),
        ("source_commit", "ec7169b5"),
        ("fact_row_count", True),
        ("country", "us"),
    ],
)
def test_national_feed_rejects_malformed_identity(
    monkeypatch, tmp_path, field, bad_value
):
    from microcosm.build.uk_runtime import national_chronicle_feed

    raw = json.loads(national_chronicle_feed._feed_path().read_text())
    raw[field] = bad_value
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(national_chronicle_feed, "_feed_path", lambda: path)

    with pytest.raises(ValueError, match=field):
        national_chronicle_feed.load_uk_national_chronicle_feed()
