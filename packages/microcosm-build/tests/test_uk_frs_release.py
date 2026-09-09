from __future__ import annotations

import copy
import json
from importlib.resources import files

import pytest

from microcosm.build.source_manifest import SourceManifest
from microcosm.build.uk_runtime import frs_spine
from microcosm.build.uk_runtime.frs_release import (
    load_uk_frs_release,
    resolve_uk_year_rule,
)
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract


def _source_manifest() -> SourceManifest:
    raw = json.loads(
        files("microcosm.build.uk").joinpath("source_stages.json").read_text()
    )
    return SourceManifest.from_mapping(raw)


def _reload_with(monkeypatch, mutated: dict) -> None:
    load_uk_frs_release.cache_clear()
    payload = json.dumps(mutated)

    class _FakePath:
        def read_text(self, *args, **kwargs):
            return payload

    monkeypatch.setattr(
        "microcosm.build.uk_runtime.frs_release._release_path",
        lambda: _FakePath(),
    )


def test_uk_frs_release_loads_and_controls_runtime_period() -> None:
    release = load_uk_frs_release()

    assert release.country == "uk"
    assert release.name == "frs_2024_25"
    assert release.survey_year == 2024
    assert release.base_year == 2024
    assert release.calibration_year == 2025
    assert frs_spine.TIME_PERIOD == release.time_period == str(release.base_year)
    assert release.vintage == "2024_25"
    assert release.ukds_study_number == 9563
    assert release.doi == "10.5255/UKDA-SN-9563-1"


def test_uk_year_rules_resolve_from_release() -> None:
    assert resolve_uk_year_rule("survey_year") == 2024
    assert resolve_uk_year_rule("calibration_year") == 2025
    with pytest.raises(ValueError, match="Unknown UK year_rule"):
        resolve_uk_year_rule("base_year")


def test_release_lockstep_with_source_manifest_and_take_up_contract() -> None:
    release = load_uk_frs_release()
    manifest = _source_manifest()

    for stage in manifest.stages:
        for artifact in stage.artifacts:
            if artifact.get("role") != "frs_table":
                continue
            assert artifact["vintage"] == release.vintage
            assert artifact["ukds_study_number"] == release.ukds_study_number
            assert artifact["doi"] == release.doi
            assert artifact["tax_year_start"] == release.base_year
    assert load_uk_take_up_contract().build_year == release.base_year
    assert "SN 9563" in manifest.stage_map()["frs_spine"].source


def test_release_validation_refuses_bad_years_and_hashes(monkeypatch) -> None:
    raw = json.loads(
        files("microcosm.build.uk").joinpath("frs_release.json").read_text()
    )

    mutated = copy.deepcopy(raw)
    mutated["time_period"] = "2023"
    _reload_with(monkeypatch, mutated)
    with pytest.raises(ValueError, match="time_period"):
        load_uk_frs_release()

    mutated = copy.deepcopy(raw)
    mutated["ukds_tab_zip"]["sha256"] = "ABC"
    _reload_with(monkeypatch, mutated)
    with pytest.raises(ValueError, match="lowercase hex"):
        load_uk_frs_release()

    mutated = copy.deepcopy(raw)
    mutated["vintage"] = "2023_24"
    _reload_with(monkeypatch, mutated)
    with pytest.raises(ValueError, match="survey_year"):
        load_uk_frs_release()
