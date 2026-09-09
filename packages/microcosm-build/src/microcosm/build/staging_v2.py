"""Version 2 staging telemetry contract and storage.

Version 1 remains implemented by :mod:`microcosm.build.staging`.  This module
defines a separate repository-file contract for country-neutral staging data
and applies the additional content restrictions required for UK source data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker

STAGING_CONTRACT_VERSION = 2
DEFAULT_UK_STAGING_REPO = "policyengine/populace-uk-staging"
DEFAULT_STAGING_PREFIX = "runs"
LATEST_STAGING_POINTER = "latest_staging.json"
RUNS_INDEX = "runs.json"

RUN_INDEX_SCHEMA = "microcosm.staging.run-index"
LATEST_RUN_SCHEMA = "microcosm.staging.latest-run"
RUN_MANIFEST_SCHEMA = "microcosm.staging.run-manifest"
PROGRESS_SCHEMA = "microcosm.staging.progress"
CALIBRATION_PROGRESS_SCHEMA = "microcosm.staging.calibration-progress"
EVENT_SCHEMA = "microcosm.staging.event"

DeliveryMode = Literal["local_and_remote", "local_only", "disabled"]
LifecycleStatus = Literal["running", "completed", "failed"]

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_NAMES = {
    RUN_INDEX_SCHEMA,
    LATEST_RUN_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    PROGRESS_SCHEMA,
    CALIBRATION_PROGRESS_SCHEMA,
    EVENT_SCHEMA,
}
_ARTIFACT_KINDS = {
    "aggregate_diagnostics",
    "build_metadata",
    "sampling_receipt",
    "validation_summary",
}
_ARTIFACT_CLASSIFICATIONS = {"aggregate", "non_row_level"}
_PROHIBITED_SUFFIXES = {
    ".7z",
    ".arrow",
    ".csv",
    ".feather",
    ".gz",
    ".h5",
    ".hdf",
    ".hdf5",
    ".npz",
    ".parquet",
    ".rar",
    ".tab",
    ".tar",
    ".tsv",
    ".xz",
    ".zip",
}
_PROHIBITED_CONTENT_KEYS = {
    "benunit_id",
    "household_id",
    "person_id",
    "raw_record",
    "raw_records",
    "row_id",
    "source_value",
    "source_values",
}
_MAX_REMOTE_FILE_BYTES = 5 * 1024 * 1024


class StagingContractError(ValueError):
    """A version 2 record or bundle violates the published contract."""


class StagingContentError(StagingContractError):
    """A file is not permitted in the UK staging repository."""


class StagingReadBackError(StagingContractError):
    """Authenticated verification could not read back the uploaded run."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _ndjson_bytes(events: list[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for event in events
        )
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise StagingContractError(
            f"{label} must match {_SAFE_ID.pattern!r}; received {value!r}."
        )
    return value


def _safe_relative_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StagingContractError(f"{label} must be a non-empty relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise StagingContractError(f"{label} must not escape its contract root.")
    if any(not part or part in {"/", "\\"} for part in path.parts):
        raise StagingContractError(f"{label} contains an invalid path segment.")
    return path.as_posix()


def _schema_identity(name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_name": {"const": name},
            "schema_version": {"const": STAGING_CONTRACT_VERSION},
        },
        "required": ["schema_name", "schema_version"],
    }


_SAFE_ID_SCHEMA = {"type": "string", "pattern": _SAFE_ID.pattern}
_NULLABLE_SAFE_ID_SCHEMA = {"anyOf": [_SAFE_ID_SCHEMA, {"type": "null"}]}
_TIMESTAMP_SCHEMA = {"type": "string", "format": "date-time"}
_SHA256_SCHEMA = {"type": "string", "pattern": _SHA256.pattern}

_SAMPLE_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "mode": {"enum": ["full", "bounded_source_households"]},
        "requested_source_households": {"type": ["integer", "null"], "minimum": 1},
        "eligible_source_families": {"type": ["integer", "null"], "minimum": 0},
        "proportional_request": {"type": ["integer", "null"], "minimum": 0},
        "forced_additions": {"type": "integer", "minimum": 0},
        "realized_source_families": {"type": ["integer", "null"], "minimum": 0},
        "realized_household_rows": {"type": ["integer", "null"], "minimum": 0},
        "seed": {"type": ["integer", "null"]},
        "receipt_sha256": {"anyOf": [_SHA256_SCHEMA, {"type": "null"}]},
    },
    "required": [
        "mode",
        "requested_source_households",
        "eligible_source_families",
        "proportional_request",
        "forced_additions",
        "realized_source_families",
        "realized_household_rows",
        "seed",
        "receipt_sha256",
    ],
    "additionalProperties": False,
}

_DELIVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contract_version": {"const": STAGING_CONTRACT_VERSION},
        "enabled": {"type": "boolean"},
        "mode": {"enum": ["local_and_remote", "local_only", "disabled"]},
        "run_id": _NULLABLE_SAFE_ID_SCHEMA,
        "configured_repository": {"type": ["string", "null"], "minLength": 1},
        "upload_attempts": {"type": "integer", "minimum": 0},
        "upload_successes": {"type": "integer", "minimum": 0},
        "read_back": {"enum": ["not_requested", "passed", "failed"]},
        "last_error_code": {"type": ["string", "null"], "pattern": "^[A-Z0-9_]+$"},
        "opt_out_reason": {"type": ["string", "null"], "minLength": 1},
    },
    "required": [
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
    ],
    "additionalProperties": False,
}

_ARTIFACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "logical_name": _SAFE_ID_SCHEMA,
        "artifact_kind": {"enum": sorted(_ARTIFACT_KINDS)},
        "contract_relative_path": {"type": "string", "minLength": 1},
        "media_type": {"const": "application/json"},
        "sha256": _SHA256_SCHEMA,
        "classification": {"enum": sorted(_ARTIFACT_CLASSIFICATIONS)},
    },
    "required": [
        "logical_name",
        "artifact_kind",
        "contract_relative_path",
        "media_type",
        "sha256",
        "classification",
    ],
    "additionalProperties": False,
}

_FAILURE_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "error_code": {"type": "string", "pattern": "^[A-Z0-9_]+$"},
        "error_type": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]*$"},
        "message": {"type": "string", "minLength": 1, "maxLength": 500},
        "local_diagnostic_reference": {"type": ["string", "null"], "maxLength": 200},
    },
    "required": [
        "error_code",
        "error_type",
        "message",
        "local_diagnostic_reference",
    ],
    "additionalProperties": False,
}

_PIPELINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"id": _SAFE_ID_SCHEMA, "version": {"type": "string", "minLength": 1}},
    "required": ["id", "version"],
    "additionalProperties": False,
}

_RUN_FIELDS: dict[str, Any] = {
    "run_id": _SAFE_ID_SCHEMA,
    "country_code": {"type": "string", "pattern": "^[A-Z]{2}$"},
    "operation_id": _SAFE_ID_SCHEMA,
    "pipeline": _PIPELINE_SCHEMA,
    "candidate_id": _SAFE_ID_SCHEMA,
    "release_id": _NULLABLE_SAFE_ID_SCHEMA,
    "run_kind": _SAFE_ID_SCHEMA,
    "non_release": {"type": "boolean"},
    "started_at": _TIMESTAMP_SCHEMA,
    "updated_at": _TIMESTAMP_SCHEMA,
    "status": {"enum": ["running", "completed", "failed"]},
    "current_stage": _SAFE_ID_SCHEMA,
}
_RUN_REQUIRED = list(_RUN_FIELDS)


def _object_schema(
    name: str,
    properties: Mapping[str, Any],
    required: list[str],
) -> dict[str, Any]:
    identity = _schema_identity(name)
    return {
        "type": "object",
        "properties": {**identity["properties"], **properties},
        "required": [*identity["required"], *required],
        "additionalProperties": False,
    }


SCHEMAS: dict[str, dict[str, Any]] = {
    RUN_MANIFEST_SCHEMA: _object_schema(
        RUN_MANIFEST_SCHEMA,
        {
            **_RUN_FIELDS,
            "sample": _SAMPLE_SCHEMA,
            "delivery": _DELIVERY_SCHEMA,
            "artifacts": {"type": "array", "items": _ARTIFACT_SCHEMA},
            "failure": _FAILURE_SCHEMA,
            "paths": {
                "type": "object",
                "properties": {
                    "progress": {"type": "string"},
                    "events": {"type": "string"},
                    "calibration_progress": {"type": ["string", "null"]},
                },
                "required": ["progress", "events", "calibration_progress"],
                "additionalProperties": False,
            },
        },
        [*_RUN_REQUIRED, "sample", "delivery", "artifacts", "failure", "paths"],
    ),
    PROGRESS_SCHEMA: _object_schema(
        PROGRESS_SCHEMA,
        {
            **_RUN_FIELDS,
            "sample": _SAMPLE_SCHEMA,
            "delivery": _DELIVERY_SCHEMA,
            "message": {"type": ["string", "null"], "maxLength": 500},
            "details": {"type": "object"},
            "failure": _FAILURE_SCHEMA,
        },
        [
            *_RUN_REQUIRED,
            "sample",
            "delivery",
            "message",
            "details",
            "failure",
        ],
    ),
    CALIBRATION_PROGRESS_SCHEMA: _object_schema(
        CALIBRATION_PROGRESS_SCHEMA,
        {
            "run_id": _SAFE_ID_SCHEMA,
            "candidate_id": _SAFE_ID_SCHEMA,
            "updated_at": _TIMESTAMP_SCHEMA,
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp": _TIMESTAMP_SCHEMA,
                        "epoch": {"type": ["integer", "null"], "minimum": 0},
                        "epochs": {"type": ["integer", "null"], "minimum": 0},
                        "phase": {"type": ["string", "null"]},
                        "loss": {"type": ["number", "null"]},
                        "iteration": {"type": ["integer", "null"], "minimum": 0},
                        "budget_search": {"type": ["integer", "null"], "minimum": 0},
                        "budget_iteration": {"type": ["integer", "null"], "minimum": 0},
                        "budget_iters": {"type": ["integer", "null"], "minimum": 0},
                        "l0_lambda": {"type": ["number", "null"]},
                    },
                    "required": [
                        "timestamp",
                        "epoch",
                        "epochs",
                        "phase",
                        "loss",
                        "iteration",
                        "budget_search",
                        "budget_iteration",
                        "budget_iters",
                        "l0_lambda",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        ["run_id", "candidate_id", "updated_at", "events"],
    ),
    EVENT_SCHEMA: _object_schema(
        EVENT_SCHEMA,
        {
            "sequence": {"type": "integer", "minimum": 1},
            "timestamp": _TIMESTAMP_SCHEMA,
            "event_type": {"enum": ["stage", "calibration"]},
            "run_id": _SAFE_ID_SCHEMA,
            "stage_id": _SAFE_ID_SCHEMA,
            "status": {"enum": ["started", "completed", "failed", "progress"]},
            "message": {"type": ["string", "null"], "maxLength": 500},
            "details": {"type": "object"},
        },
        [
            "sequence",
            "timestamp",
            "event_type",
            "run_id",
            "stage_id",
            "status",
            "message",
            "details",
        ],
    ),
    LATEST_RUN_SCHEMA: _object_schema(
        LATEST_RUN_SCHEMA,
        {
            "run_id": _SAFE_ID_SCHEMA,
            "candidate_id": _SAFE_ID_SCHEMA,
            "release_id": _NULLABLE_SAFE_ID_SCHEMA,
            "updated_at": _TIMESTAMP_SCHEMA,
            "paths": {
                "type": "object",
                "properties": {
                    "run_manifest": {"type": "string"},
                    "progress": {"type": "string"},
                    "calibration_progress": {"type": ["string", "null"]},
                    "events": {"type": "string"},
                },
                "required": [
                    "run_manifest",
                    "progress",
                    "calibration_progress",
                    "events",
                ],
                "additionalProperties": False,
            },
        },
        ["run_id", "candidate_id", "release_id", "updated_at", "paths"],
    ),
    RUN_INDEX_SCHEMA: _object_schema(
        RUN_INDEX_SCHEMA,
        {
            "updated_at": _TIMESTAMP_SCHEMA,
            "runs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "run_id": _SAFE_ID_SCHEMA,
                        "candidate_id": _SAFE_ID_SCHEMA,
                        "release_id": _NULLABLE_SAFE_ID_SCHEMA,
                        "country_code": {"type": "string", "pattern": "^[A-Z]{2}$"},
                        "run_kind": _SAFE_ID_SCHEMA,
                        "non_release": {"type": "boolean"},
                        "status": {"enum": ["running", "completed", "failed"]},
                        "stage": _SAFE_ID_SCHEMA,
                        "started_at": _TIMESTAMP_SCHEMA,
                        "updated_at": _TIMESTAMP_SCHEMA,
                        "progress_path": {"type": "string"},
                        "run_manifest_path": {"type": "string"},
                    },
                    "required": [
                        "run_id",
                        "candidate_id",
                        "release_id",
                        "country_code",
                        "run_kind",
                        "non_release",
                        "status",
                        "stage",
                        "started_at",
                        "updated_at",
                        "progress_path",
                        "run_manifest_path",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        ["updated_at", "runs"],
    ),
}


def validate_staging_delivery(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a version 2 delivery summary."""

    normalized = _jsonable(payload)
    validator = Draft202012Validator(_DELIVERY_SCHEMA)
    errors = sorted(validator.iter_errors(normalized), key=lambda error: list(error.path))
    if errors:
        raise StagingContractError(errors[0].message)
    enabled = normalized["enabled"]
    mode = normalized["mode"]
    repository = normalized["configured_repository"]
    run_id = normalized["run_id"]
    reason = normalized["opt_out_reason"]
    if normalized["upload_successes"] > normalized["upload_attempts"]:
        raise StagingContractError("upload_successes cannot exceed upload_attempts.")
    if enabled:
        if mode == "disabled" or run_id is None or reason is not None:
            raise StagingContractError("Enabled staging has contradictory delivery fields.")
        if mode == "local_and_remote" and not repository:
            raise StagingContractError("Remote staging requires a configured repository.")
        if mode == "local_only" and repository is not None:
            raise StagingContractError("Local-only staging cannot name a repository.")
    elif (
        mode != "disabled"
        or run_id is not None
        or repository is not None
        or not reason
        or normalized["upload_attempts"]
        or normalized["upload_successes"]
    ):
        raise StagingContractError("Disabled staging has contradictory delivery fields.")
    return normalized


def disabled_staging_delivery(reason: str) -> dict[str, Any]:
    """Return explicit version 2 evidence for a deliberate staging opt-out."""

    if not isinstance(reason, str) or not reason.strip():
        raise StagingContractError("A deliberate staging opt-out requires a reason.")
    return validate_staging_delivery(
        {
            "contract_version": STAGING_CONTRACT_VERSION,
            "enabled": False,
            "mode": "disabled",
            "run_id": None,
            "configured_repository": None,
            "upload_attempts": 0,
            "upload_successes": 0,
            "read_back": "not_requested",
            "last_error_code": None,
            "opt_out_reason": reason.strip(),
        }
    )


def validate_v2_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one independently identified version 2 document."""

    normalized = _jsonable(payload)
    name = normalized.get("schema_name")
    version = normalized.get("schema_version")
    if name not in _SCHEMA_NAMES or version != STAGING_CONTRACT_VERSION:
        raise StagingContractError(
            f"Unsupported staging schema identity: {name!r} version {version!r}."
        )
    validator = Draft202012Validator(SCHEMAS[name], format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(normalized), key=lambda error: list(error.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].absolute_path)
        label = f" at {path}" if path else ""
        raise StagingContractError(f"Invalid {name}{label}: {errors[0].message}")
    if name in {RUN_MANIFEST_SCHEMA, PROGRESS_SCHEMA}:
        if normalized["non_release"] != (normalized["release_id"] is None):
            raise StagingContractError(
                "non_release must be true exactly when release_id is absent."
            )
        validate_staging_delivery(normalized["delivery"])
        if normalized["status"] == "failed" and normalized["failure"] is None:
            raise StagingContractError("Failed runs require sanitized failure data.")
        if normalized["status"] != "failed" and normalized["failure"] is not None:
            raise StagingContractError("Non-failed runs cannot contain failure data.")
    if name == RUN_INDEX_SCHEMA:
        run_ids = [row["run_id"] for row in normalized["runs"]]
        if len(run_ids) != len(set(run_ids)):
            raise StagingContractError("Run index contains duplicate run identifiers.")
    return normalized


@dataclass(frozen=True)
class UKStagingContentPolicy:
    """Validate every remote file before a transport operation."""

    max_file_bytes: int = _MAX_REMOTE_FILE_BYTES

    def validate_payload(self, payload: Any) -> None:
        self._reject_sensitive_keys(payload)

    def validate_remote_file(self, path: str, data: bytes) -> None:
        safe_path = _safe_relative_path(path, label="remote path")
        suffix = PurePosixPath(safe_path).suffix.lower()
        if suffix in _PROHIBITED_SUFFIXES:
            raise StagingContentError(f"Prohibited staging file type: {suffix}.")
        if suffix not in {".json", ".ndjson"}:
            raise StagingContentError(f"Unapproved staging file type: {suffix or '<none>'}.")
        if len(data) > self.max_file_bytes:
            raise StagingContentError(
                f"Staging file exceeds the {self.max_file_bytes}-byte limit."
            )
        try:
            if suffix == ".ndjson":
                values = [json.loads(line) for line in data.decode().splitlines() if line]
            else:
                values = json.loads(data)
        except (UnicodeDecodeError, ValueError) as exc:
            raise StagingContentError("Staging files must contain valid UTF-8 JSON.") from exc
        self.validate_payload(values)

    def validate_artifact(
        self,
        *,
        logical_name: str,
        artifact_kind: str,
        source: Path,
        classification: str,
    ) -> bytes:
        _safe_identifier(logical_name, label="artifact logical name")
        if artifact_kind not in _ARTIFACT_KINDS:
            raise StagingContentError(f"Unapproved artifact kind: {artifact_kind!r}.")
        if classification not in _ARTIFACT_CLASSIFICATIONS:
            raise StagingContentError(
                f"Unapproved artifact classification: {classification!r}."
            )
        if source.suffix.lower() != ".json":
            raise StagingContentError("Reviewed staging artifacts must be JSON.")
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise StagingContentError("Reviewed staging artifact is not readable.") from exc
        self.validate_remote_file(f"artifacts/{logical_name}.json", data)
        return data

    def _reject_sensitive_keys(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).strip().lower()
                if normalized in _PROHIBITED_CONTENT_KEYS:
                    raise StagingContentError(
                        f"Prohibited row-level field in staging content: {key!r}."
                    )
                self._reject_sensitive_keys(item)
        elif isinstance(value, list):
            for item in value:
                self._reject_sensitive_keys(item)


@dataclass
class HuggingFaceDatasetStorage:
    """Small transport wrapper for one access-controlled dataset repository."""

    repo_id: str
    api: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id.strip():
            raise StagingContractError("Remote staging requires a repository identifier.")
        self.repo_id = self.repo_id.strip()

    def _api(self) -> Any:
        if self.api is None:
            from huggingface_hub import HfApi

            self.api = HfApi()
        return self.api

    def upload(self, local_path: Path, path_in_repo: str) -> None:
        self._api().upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

    def download(self, path_in_repo: str) -> bytes:
        api = self._api()
        download = getattr(api, "hf_hub_download", None)
        if download is None:
            from huggingface_hub import hf_hub_download as download

        local = download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            force_download=True,
        )
        return Path(local).read_bytes()


class StagingTelemetryV2:
    """Record, validate, persist, and optionally upload telemetry version 2."""

    def __init__(
        self,
        *,
        run_id: str,
        country_code: str,
        operation_id: str,
        pipeline_id: str,
        pipeline_version: str,
        candidate_id: str,
        local_dir: Path | str,
        release_id: str | None = None,
        run_kind: str = "build",
        delivery_mode: DeliveryMode = "local_and_remote",
        repo_id: str | None = DEFAULT_UK_STAGING_REPO,
        path_prefix: str = DEFAULT_STAGING_PREFIX,
        upload_interval_seconds: float = 30.0,
        api: Any = None,
        clock: Callable[[], str] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        content_policy: UKStagingContentPolicy | None = None,
    ) -> None:
        self.run_id = _safe_identifier(run_id, label="run_id")
        self.candidate_id = _safe_identifier(candidate_id, label="candidate_id")
        self.release_id = (
            _safe_identifier(release_id, label="release_id") if release_id else None
        )
        self.operation_id = _safe_identifier(operation_id, label="operation_id")
        self.pipeline_id = _safe_identifier(pipeline_id, label="pipeline_id")
        self.run_kind = _safe_identifier(run_kind, label="run_kind")
        if not re.fullmatch(r"[A-Z]{2}", country_code):
            raise StagingContractError("country_code must be ISO 3166-1 alpha-2.")
        if not isinstance(pipeline_version, str) or not pipeline_version.strip():
            raise StagingContractError("pipeline_version must be non-empty.")
        if delivery_mode == "disabled":
            raise StagingContractError(
                "Do not construct telemetry for disabled staging; record an opt-out."
            )
        if delivery_mode == "local_and_remote" and not (repo_id or "").strip():
            raise StagingContractError("Remote staging requires a repository identifier.")
        if delivery_mode == "local_only" and repo_id not in {None, ""}:
            raise StagingContractError("Local-only staging cannot name a repository.")
        self.country_code = country_code
        self.pipeline_version = pipeline_version.strip()
        self.delivery_mode = delivery_mode
        self.repo_id = repo_id.strip() if isinstance(repo_id, str) and repo_id.strip() else None
        self.path_prefix = _safe_relative_path(path_prefix, label="path_prefix")
        self.local_dir = Path(local_dir)
        self.run_dir = self.local_dir / self.path_prefix / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.upload_interval_seconds = max(0.0, float(upload_interval_seconds))
        self._clock = clock
        self._monotonic = monotonic
        self._content_policy = content_policy or UKStagingContentPolicy()
        self._transport = (
            HuggingFaceDatasetStorage(self.repo_id, api=api)
            if self.delivery_mode == "local_and_remote" and self.repo_id
            else None
        )
        self.started_at = self._clock()
        self.updated_at = self.started_at
        self.status: LifecycleStatus = "running"
        self.current_stage = "created"
        self.message: str | None = "Staging run created."
        self.details: dict[str, Any] = {}
        self.sample: dict[str, Any] | None = None
        self.failure: dict[str, Any] | None = None
        self._artifacts: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._calibration_events: list[dict[str, Any]] = []
        self._last_upload_at = 0.0
        self._consecutive_upload_failures = 0
        self._remote_disabled = False
        self._delivery = {
            "contract_version": STAGING_CONTRACT_VERSION,
            "enabled": True,
            "mode": self.delivery_mode,
            "run_id": self.run_id,
            "configured_repository": self.repo_id,
            "upload_attempts": 0,
            "upload_successes": 0,
            "read_back": "not_requested",
            "last_error_code": None,
            "opt_out_reason": None,
        }
        self._merge_remote_index()
        self._append_event(
            stage_id="created",
            status="started",
            message=self.message,
            details={},
            timestamp=self.started_at,
        )
        self._persist_bundle()

    def _merge_remote_index(self) -> None:
        """Seed the local advisory index from the remote copy when available."""

        if self._transport is None:
            return
        try:
            remote = validate_v2_document(
                json.loads(self._transport.download(RUNS_INDEX))
            )
            if remote["schema_name"] != RUN_INDEX_SCHEMA:
                return
        except Exception:
            return
        path = self.local_dir / RUNS_INDEX
        local_runs: list[dict[str, Any]] = []
        if path.exists():
            try:
                local = validate_v2_document(json.loads(path.read_text()))
                if local["schema_name"] == RUN_INDEX_SCHEMA:
                    local_runs = list(local["runs"])
            except (OSError, ValueError, StagingContractError):
                local_runs = []
        by_run = {row["run_id"]: row for row in remote["runs"]}
        for row in local_runs:
            incumbent = by_run.get(row["run_id"])
            if incumbent is None or str(row["updated_at"]) >= str(
                incumbent["updated_at"]
            ):
                by_run[row["run_id"]] = row
        merged = {
            "schema_name": RUN_INDEX_SCHEMA,
            "schema_version": STAGING_CONTRACT_VERSION,
            "updated_at": max(
                [str(remote["updated_at"])]
                + [str(row["updated_at"]) for row in local_runs]
            ),
            "runs": sorted(
                by_run.values(),
                key=lambda row: str(row.get("updated_at") or row.get("started_at")),
                reverse=True,
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json_bytes(validate_v2_document(merged)))

    @property
    def repo_run_prefix(self) -> str:
        return f"{self.path_prefix}/{self.run_id}"

    @property
    def uploads_succeeded(self) -> int:
        return int(self._delivery["upload_successes"])

    @property
    def delivery_summary(self) -> dict[str, Any]:
        return validate_staging_delivery(dict(self._delivery))

    def set_sample(self, sample: Mapping[str, Any]) -> None:
        normalized = _jsonable(sample)
        validator = Draft202012Validator(_SAMPLE_SCHEMA)
        errors = sorted(validator.iter_errors(normalized), key=lambda error: list(error.path))
        if errors:
            raise StagingContractError(f"Invalid sampling evidence: {errors[0].message}")
        if normalized["mode"] == "bounded_source_households":
            required = (
                "requested_source_households",
                "eligible_source_families",
                "proportional_request",
                "realized_source_families",
                "realized_household_rows",
                "seed",
                "receipt_sha256",
            )
            if any(normalized[field] is None for field in required):
                raise StagingContractError(
                    "Bounded samples require requested, realized, seed, and receipt data."
                )
        self.sample = normalized
        self._persist_bundle()

    def stage(
        self,
        stage_id: str,
        *,
        event_status: Literal["started", "completed", "progress"] = "started",
        message: str | None = None,
        force_upload: bool = False,
        **details: Any,
    ) -> None:
        if self.status != "running":
            raise StagingContractError("Cannot add a stage to a finished staging run.")
        stage_id = _safe_identifier(stage_id, label="stage_id")
        safe_details = _jsonable(details)
        self._content_policy.validate_payload(safe_details)
        self.current_stage = stage_id
        self.updated_at = self._clock()
        self.message = message
        self.details = safe_details
        self._append_event(
            stage_id=stage_id,
            status=event_status,
            message=message,
            details=safe_details,
            timestamp=self.updated_at,
        )
        self._persist_bundle()
        self._maybe_upload(force=force_upload)

    def calibration_progress(self, event: Mapping[str, Any]) -> None:
        if event.get("kind") != "calibration_epoch":
            return
        if self.status != "running":
            raise StagingContractError("Cannot add calibration data to a finished run.")
        timestamp = self._clock()
        row = {
            "timestamp": timestamp,
            "epoch": _optional_int(event.get("epoch")),
            "epochs": _optional_int(event.get("epochs")),
            "phase": _optional_string(event.get("phase")),
            "loss": _optional_float(event.get("loss")),
            "iteration": _optional_int(event.get("iteration")),
            "budget_search": _optional_int(event.get("budget_search")),
            "budget_iteration": _optional_int(event.get("budget_iteration")),
            "budget_iters": _optional_int(event.get("budget_iters")),
            "l0_lambda": _optional_float(event.get("l0_lambda")),
        }
        self._calibration_events.append(row)
        self.current_stage = "calibrating"
        self.updated_at = timestamp
        self.details = {key: value for key, value in row.items() if key != "timestamp"}
        self._append_event(
            stage_id="calibrating",
            status="progress",
            message=None,
            details=self.details,
            timestamp=timestamp,
            event_type="calibration",
        )
        self._persist_bundle()
        self._maybe_upload()

    def add_artifact(
        self,
        logical_name: str,
        source: Path | str,
        *,
        artifact_kind: str,
        classification: str,
        force_upload: bool = True,
    ) -> dict[str, Any]:
        source_path = Path(source)
        data = self._content_policy.validate_artifact(
            logical_name=logical_name,
            artifact_kind=artifact_kind,
            source=source_path,
            classification=classification,
        )
        logical_name = _safe_identifier(logical_name, label="artifact logical name")
        relative_path = f"artifacts/{logical_name}.json"
        destination = self.run_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        artifact = {
            "logical_name": logical_name,
            "artifact_kind": artifact_kind,
            "contract_relative_path": relative_path,
            "media_type": "application/json",
            "sha256": _sha256_bytes(data),
            "classification": classification,
        }
        self._artifacts = [
            existing
            for existing in self._artifacts
            if existing["logical_name"] != logical_name
        ]
        self._artifacts.append(artifact)
        self._artifacts.sort(key=lambda item: item["logical_name"])
        self._persist_bundle()
        self._maybe_upload(force=force_upload)
        return dict(artifact)

    def fail(
        self,
        error: BaseException,
        *,
        error_code: str = "BUILD_FAILED",
        local_diagnostic_reference: str | None = None,
    ) -> None:
        error_code = error_code.strip().upper()
        if not re.fullmatch(r"[A-Z0-9_]+", error_code):
            raise StagingContractError("error_code must contain only A-Z, 0-9, and _.")
        error_type = type(error).__name__
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", error_type):
            error_type = "BuildError"
        stage = self.current_stage
        self.status = "failed"
        self.current_stage = "failed"
        self.updated_at = self._clock()
        self.message = f"The build failed during {stage}."
        self.details = {}
        self.failure = {
            "error_code": error_code,
            "error_type": error_type,
            "message": self.message,
            "local_diagnostic_reference": _safe_local_reference(
                local_diagnostic_reference
            ),
        }
        self._append_event(
            stage_id="failed",
            status="failed",
            message=self.message,
            details={"error_code": error_code, "error_type": error_type},
            timestamp=self.updated_at,
        )
        self._persist_bundle()
        self._maybe_upload(force=True)

    def complete(self, *, message: str = "Staging run completed.") -> None:
        if self.status != "running":
            raise StagingContractError("Cannot complete a finished staging run.")
        self.status = "completed"
        self.current_stage = "complete"
        self.updated_at = self._clock()
        self.message = message
        self.details = {}
        self._append_event(
            stage_id="complete",
            status="completed",
            message=message,
            details={},
            timestamp=self.updated_at,
        )
        self._persist_bundle()
        self._maybe_upload(force=True)

    def verify_remote(self) -> None:
        if self._transport is None:
            raise StagingReadBackError("Remote read-back requires remote staging mode.")
        self._maybe_upload(force=True)
        expected = {
            f"{self.repo_run_prefix}/run_manifest.json": RUN_MANIFEST_SCHEMA,
            f"{self.repo_run_prefix}/progress.json": PROGRESS_SCHEMA,
            LATEST_STAGING_POINTER: LATEST_RUN_SCHEMA,
            RUNS_INDEX: RUN_INDEX_SCHEMA,
        }
        try:
            documents = {
                path: validate_v2_document(json.loads(self._transport.download(path)))
                for path in expected
            }
            for path, schema_name in expected.items():
                if documents[path]["schema_name"] != schema_name:
                    raise StagingReadBackError(
                        f"Remote file {path} has the wrong schema identity."
                    )
            manifest = documents[f"{self.repo_run_prefix}/run_manifest.json"]
            progress = documents[f"{self.repo_run_prefix}/progress.json"]
            latest = documents[LATEST_STAGING_POINTER]
            index = documents[RUNS_INDEX]
            if {manifest["run_id"], progress["run_id"], latest["run_id"]} != {
                self.run_id
            }:
                raise StagingReadBackError("Remote run identifiers do not match.")
            if not any(row["run_id"] == self.run_id for row in index["runs"]):
                raise StagingReadBackError("Remote run index does not contain this run.")
        except Exception as exc:
            self._delivery["read_back"] = "failed"
            self._delivery["last_error_code"] = "READ_BACK_FAILED"
            self._persist_bundle()
            if isinstance(exc, StagingReadBackError):
                raise
            raise StagingReadBackError("Authenticated staging read-back failed.") from exc
        self._delivery["read_back"] = "passed"
        self._delivery["last_error_code"] = None
        self._persist_bundle()
        # Make the verified state visible to authenticated consumers. This is
        # best effort because read-back has already established that the core
        # run files are present and valid.
        self._maybe_upload(force=True)

    def validate_local_bundle(self) -> dict[str, Any]:
        return validate_v2_bundle(self.local_dir, self.run_id, path_prefix=self.path_prefix)

    def _append_event(
        self,
        *,
        stage_id: str,
        status: str,
        message: str | None,
        details: Mapping[str, Any],
        timestamp: str,
        event_type: str = "stage",
    ) -> None:
        event = {
            "schema_name": EVENT_SCHEMA,
            "schema_version": STAGING_CONTRACT_VERSION,
            "sequence": len(self._events) + 1,
            "timestamp": timestamp,
            "event_type": event_type,
            "run_id": self.run_id,
            "stage_id": stage_id,
            "status": status,
            "message": message,
            "details": _jsonable(details),
        }
        self._content_policy.validate_payload(event["details"])
        self._events.append(validate_v2_document(event))

    def _run_fields(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "country_code": self.country_code,
            "operation_id": self.operation_id,
            "pipeline": {"id": self.pipeline_id, "version": self.pipeline_version},
            "candidate_id": self.candidate_id,
            "release_id": self.release_id,
            "run_kind": self.run_kind,
            "non_release": self.release_id is None,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "current_stage": self.current_stage,
        }

    def _manifest(self) -> dict[str, Any]:
        calibration_path = (
            f"{self.repo_run_prefix}/calibration_progress.json"
            if self._calibration_events
            else None
        )
        return {
            "schema_name": RUN_MANIFEST_SCHEMA,
            "schema_version": STAGING_CONTRACT_VERSION,
            **self._run_fields(),
            "sample": self.sample,
            "delivery": self.delivery_summary,
            "artifacts": list(self._artifacts),
            "failure": self.failure,
            "paths": {
                "progress": f"{self.repo_run_prefix}/progress.json",
                "events": f"{self.repo_run_prefix}/events.ndjson",
                "calibration_progress": calibration_path,
            },
        }

    def _progress(self) -> dict[str, Any]:
        return {
            "schema_name": PROGRESS_SCHEMA,
            "schema_version": STAGING_CONTRACT_VERSION,
            **self._run_fields(),
            "sample": self.sample,
            "delivery": self.delivery_summary,
            "message": self.message,
            "details": self.details,
            "failure": self.failure,
        }

    def _calibration_progress(self) -> dict[str, Any]:
        return {
            "schema_name": CALIBRATION_PROGRESS_SCHEMA,
            "schema_version": STAGING_CONTRACT_VERSION,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "updated_at": self.updated_at,
            "events": list(self._calibration_events),
        }

    def _latest(self) -> dict[str, Any]:
        return {
            "schema_name": LATEST_RUN_SCHEMA,
            "schema_version": STAGING_CONTRACT_VERSION,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "release_id": self.release_id,
            "updated_at": self.updated_at,
            "paths": {
                "run_manifest": f"{self.repo_run_prefix}/run_manifest.json",
                "progress": f"{self.repo_run_prefix}/progress.json",
                "calibration_progress": (
                    f"{self.repo_run_prefix}/calibration_progress.json"
                    if self._calibration_events
                    else None
                ),
                "events": f"{self.repo_run_prefix}/events.ndjson",
            },
        }

    def _summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "release_id": self.release_id,
            "country_code": self.country_code,
            "run_kind": self.run_kind,
            "non_release": self.release_id is None,
            "status": self.status,
            "stage": self.current_stage,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "progress_path": f"{self.repo_run_prefix}/progress.json",
            "run_manifest_path": f"{self.repo_run_prefix}/run_manifest.json",
        }

    def _index(self) -> dict[str, Any]:
        existing: list[dict[str, Any]] = []
        path = self.local_dir / RUNS_INDEX
        if path.exists():
            try:
                payload = validate_v2_document(json.loads(path.read_text()))
                existing = list(payload["runs"])
            except (OSError, ValueError, StagingContractError):
                existing = []
        runs = [row for row in existing if row["run_id"] != self.run_id]
        runs.append(self._summary())
        runs.sort(
            key=lambda row: str(row.get("updated_at") or row.get("started_at")),
            reverse=True,
        )
        return {
            "schema_name": RUN_INDEX_SCHEMA,
            "schema_version": STAGING_CONTRACT_VERSION,
            "updated_at": self.updated_at,
            "runs": runs,
        }

    def _persist_bundle(self) -> None:
        documents = {
            self.run_dir / "run_manifest.json": self._manifest(),
            self.run_dir / "progress.json": self._progress(),
            self.local_dir / LATEST_STAGING_POINTER: self._latest(),
            self.local_dir / RUNS_INDEX: self._index(),
        }
        if self._calibration_events:
            documents[self.run_dir / "calibration_progress.json"] = (
                self._calibration_progress()
            )
        for path, payload in documents.items():
            validate_v2_document(payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_json_bytes(payload))
        (self.run_dir / "events.ndjson").write_bytes(_ndjson_bytes(self._events))

    def _upload_paths(self) -> list[tuple[Path, str]]:
        paths = [
            (self.run_dir / "run_manifest.json", f"{self.repo_run_prefix}/run_manifest.json"),
            (self.run_dir / "progress.json", f"{self.repo_run_prefix}/progress.json"),
            (self.run_dir / "events.ndjson", f"{self.repo_run_prefix}/events.ndjson"),
        ]
        calibration = self.run_dir / "calibration_progress.json"
        if calibration.exists():
            paths.append(
                (calibration, f"{self.repo_run_prefix}/calibration_progress.json")
            )
        for artifact in self._artifacts:
            relative = artifact["contract_relative_path"]
            paths.append((self.run_dir / relative, f"{self.repo_run_prefix}/{relative}"))
        paths.extend(
            [
                (self.local_dir / LATEST_STAGING_POINTER, LATEST_STAGING_POINTER),
                (self.local_dir / RUNS_INDEX, RUNS_INDEX),
            ]
        )
        return paths

    def _maybe_upload(self, *, force: bool = False) -> None:
        if self._transport is None or self._remote_disabled:
            return
        now = self._monotonic()
        if not force and now - self._last_upload_at < self.upload_interval_seconds:
            return
        self._last_upload_at = now
        for local_path, remote_path in self._upload_paths():
            data = local_path.read_bytes()
            self._content_policy.validate_remote_file(remote_path, data)
            self._delivery["upload_attempts"] += 1
            try:
                self._transport.upload(local_path, remote_path)
            except Exception:
                self._consecutive_upload_failures += 1
                self._delivery["last_error_code"] = "UPLOAD_FAILED"
                print(
                    f"warning: staging upload failed for {remote_path}",
                    file=sys.stderr,
                )
                if self._consecutive_upload_failures >= 3:
                    self._remote_disabled = True
                    print(
                        "warning: pausing remote staging writes after three "
                        "consecutive failures; local telemetry continues.",
                        file=sys.stderr,
                    )
                    break
            else:
                self._consecutive_upload_failures = 0
                self._delivery["upload_successes"] += 1
                self._delivery["last_error_code"] = None
        self._persist_bundle()


def validate_v2_bundle(
    local_dir: Path | str,
    run_id: str,
    *,
    path_prefix: str = DEFAULT_STAGING_PREFIX,
) -> dict[str, Any]:
    """Validate a complete locally stored version 2 bundle."""

    run_id = _safe_identifier(run_id, label="run_id")
    prefix = _safe_relative_path(path_prefix, label="path_prefix")
    root = Path(local_dir)
    run_dir = root / prefix / run_id
    documents: dict[str, Any] = {}
    required = {
        "run_manifest": (run_dir / "run_manifest.json", RUN_MANIFEST_SCHEMA),
        "progress": (run_dir / "progress.json", PROGRESS_SCHEMA),
        "latest": (root / LATEST_STAGING_POINTER, LATEST_RUN_SCHEMA),
        "index": (root / RUNS_INDEX, RUN_INDEX_SCHEMA),
    }
    for label, (path, schema_name) in required.items():
        try:
            payload = validate_v2_document(json.loads(path.read_text()))
        except OSError as exc:
            raise StagingContractError(f"Missing required staging file: {path}.") from exc
        if payload["schema_name"] != schema_name:
            raise StagingContractError(f"{path} has the wrong schema identity.")
        documents[label] = payload
    event_path = run_dir / "events.ndjson"
    try:
        events = [
            validate_v2_document(json.loads(line))
            for line in event_path.read_text().splitlines()
            if line
        ]
    except OSError as exc:
        raise StagingContractError(f"Missing required staging file: {event_path}.") from exc
    if not events:
        raise StagingContractError("A staging run must contain at least one event.")
    sequences = [event["sequence"] for event in events]
    if sequences != list(range(1, len(events) + 1)):
        raise StagingContractError("Staging event sequence is not contiguous and ordered.")
    if any(event["run_id"] != run_id for event in events):
        raise StagingContractError("Staging events identify a different run.")
    for label in ("run_manifest", "progress", "latest"):
        if documents[label]["run_id"] != run_id:
            raise StagingContractError(f"{label} identifies a different run.")
    if not any(row["run_id"] == run_id for row in documents["index"]["runs"]):
        raise StagingContractError("Run index does not contain the validated run.")
    manifest = documents["run_manifest"]
    expected_run_paths = {
        "progress": f"{prefix}/{run_id}/progress.json",
        "events": f"{prefix}/{run_id}/events.ndjson",
    }
    for field, expected in expected_run_paths.items():
        if manifest["paths"][field] != expected:
            raise StagingContractError(
                f"Run manifest {field} path does not identify the validated run."
            )
    expected_latest_paths = {
        "run_manifest": f"{prefix}/{run_id}/run_manifest.json",
        "progress": expected_run_paths["progress"],
        "events": expected_run_paths["events"],
    }
    for field, expected in expected_latest_paths.items():
        if documents["latest"]["paths"][field] != expected:
            raise StagingContractError(
                f"Latest-run {field} path does not identify the validated run."
            )
    for field in _RUN_REQUIRED:
        if documents["run_manifest"][field] != documents["progress"][field]:
            raise StagingContractError(f"Bundle disagrees on {field}.")
    for field in ("candidate_id", "release_id"):
        values = {documents[label][field] for label in ("run_manifest", "latest")}
        if len(values) != 1:
            raise StagingContractError(f"Bundle disagrees on {field}.")
    calibration_path = manifest["paths"]["calibration_progress"]
    if documents["latest"]["paths"]["calibration_progress"] != calibration_path:
        raise StagingContractError("Bundle disagrees on calibration progress path.")
    if calibration_path is not None:
        expected_calibration = f"{prefix}/{run_id}/calibration_progress.json"
        if (
            _safe_relative_path(calibration_path, label="calibration progress path")
            != expected_calibration
        ):
            raise StagingContractError(
                "Calibration progress path does not identify the validated run."
            )
        try:
            calibration = validate_v2_document(
                json.loads((root / calibration_path).read_text())
            )
        except OSError as exc:
            raise StagingContractError(
                f"Missing calibration progress file: {root / calibration_path}."
            ) from exc
        if calibration["schema_name"] != CALIBRATION_PROGRESS_SCHEMA:
            raise StagingContractError("Calibration progress has the wrong schema identity.")
        if calibration["run_id"] != run_id:
            raise StagingContractError("Calibration progress identifies a different run.")
        if calibration["candidate_id"] != manifest["candidate_id"]:
            raise StagingContractError("Bundle disagrees on candidate_id.")
        documents["calibration_progress"] = calibration
    policy = UKStagingContentPolicy()
    artifact_names: set[str] = set()
    artifact_paths: set[str] = set()
    for artifact in manifest["artifacts"]:
        logical_name = artifact["logical_name"]
        relative = _safe_relative_path(
            artifact["contract_relative_path"], label="artifact path"
        )
        expected_relative = f"artifacts/{logical_name}.json"
        if relative != expected_relative:
            raise StagingContractError(
                f"Artifact {logical_name!r} path must be {expected_relative!r}."
            )
        if logical_name in artifact_names or relative in artifact_paths:
            raise StagingContractError("Run manifest contains a duplicate artifact.")
        artifact_names.add(logical_name)
        artifact_paths.add(relative)
        artifact_path = run_dir / relative
        try:
            data = artifact_path.read_bytes()
        except OSError as exc:
            raise StagingContractError(
                f"Missing reviewed staging artifact: {artifact_path}."
            ) from exc
        if _sha256_bytes(data) != artifact["sha256"]:
            raise StagingContractError(
                f"Reviewed staging artifact {logical_name!r} has the wrong digest."
            )
        policy.validate_remote_file(f"{prefix}/{run_id}/{relative}", data)
    documents["events"] = events
    return documents


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise StagingContractError("Calibration integer fields cannot be booleans.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StagingContractError("Calibration integer field is invalid.") from exc
    if parsed < 0:
        raise StagingContractError("Calibration integer fields cannot be negative.")
    return parsed


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StagingContractError("Calibration numeric field is invalid.") from exc
    return parsed if math.isfinite(parsed) else None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _safe_local_reference(value: str | None) -> str | None:
    if value is None:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    parsed = path.as_posix()
    return parsed[:200] if parsed else None
