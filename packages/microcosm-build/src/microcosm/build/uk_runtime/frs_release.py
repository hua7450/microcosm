"""Frozen UK FRS raw-vintage release loader."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

__all__ = [
    "UK_YEAR_RULES",
    "UKFRSRelease",
    "load_uk_frs_release",
    "resolve_uk_year_rule",
]

_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VINTAGE = re.compile(r"^\d{4}_\d{2}$")
UK_YEAR_RULES = ("survey_year", "calibration_year")


@dataclass(frozen=True)
class UKFRSRelease:
    """Checked-in identity for the UK raw FRS release used by the build."""

    version: int
    country: str
    policy: str
    name: str
    survey_year: int
    base_year: int
    calibration_year: int
    time_period: str
    vintage: str
    ukds_study_number: int
    doi: str
    ukds_tab_zip: Mapping[str, Any]
    acquisition: Mapping[str, Any]
    notes: str


def _release_path():
    return files("microcosm.build.uk").joinpath("frs_release.json")


@lru_cache(maxsize=1)
def load_uk_frs_release() -> UKFRSRelease:
    """Load and validate ``build/uk/frs_release.json``."""

    raw = json.loads(_release_path().read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("UK FRS release must contain a JSON object.")
    release = UKFRSRelease(
        version=_int(raw, "version"),
        country=_str(raw, "country"),
        policy=_str(raw, "policy"),
        name=_str(raw, "name"),
        survey_year=_int(raw, "survey_year"),
        base_year=_int(raw, "base_year"),
        calibration_year=_int(raw, "calibration_year"),
        time_period=_str(raw, "time_period"),
        vintage=_str(raw, "vintage"),
        ukds_study_number=_int(raw, "ukds_study_number"),
        doi=_str(raw, "doi"),
        ukds_tab_zip=_mapping(raw, "ukds_tab_zip"),
        acquisition=_mapping(raw, "acquisition"),
        notes=_str(raw, "notes"),
    )
    _validate(release)
    return release


def resolve_uk_year_rule(
    rule: str,
    *,
    release: UKFRSRelease | None = None,
) -> int:
    """Map a declared year_rule to its release year; reject any other string."""

    if rule not in UK_YEAR_RULES:
        raise ValueError(
            f"Unknown UK year_rule {rule!r}; expected one of {UK_YEAR_RULES}."
        )
    release = release or load_uk_frs_release()
    return int(getattr(release, rule))


def _int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"UK FRS release requires integer {key}.")
    return value


def _str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"UK FRS release requires non-empty string {key}.")
    return value


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"UK FRS release requires object {key}.")
    return dict(value)


def _validate(release: UKFRSRelease) -> None:
    if release.country != "uk":
        raise ValueError("UK FRS release country must be 'uk'.")
    if not (release.survey_year <= release.base_year < release.calibration_year):
        raise ValueError(
            "UK FRS release years must satisfy survey_year <= base_year < "
            "calibration_year."
        )
    if release.time_period != str(release.base_year):
        raise ValueError("UK FRS release time_period must equal base_year.")
    if not _VINTAGE.fullmatch(release.vintage):
        raise ValueError("UK FRS release vintage must match YYYY_YY.")
    if not release.vintage.startswith(str(release.survey_year)):
        raise ValueError("UK FRS release vintage must start with survey_year.")
    _validate_sha256(release.ukds_tab_zip, "ukds_tab_zip.sha256", "sha256")
    _validate_sha256(release.acquisition, "acquisition.zip_sha256", "zip_sha256")


def _validate_sha256(raw: Mapping[str, Any], label: str, key: str) -> None:
    value = raw.get(key)
    if not isinstance(value, str) or not _LOWER_SHA256.fullmatch(value):
        raise ValueError(f"UK FRS release {label} must be 64 lowercase hex chars.")
