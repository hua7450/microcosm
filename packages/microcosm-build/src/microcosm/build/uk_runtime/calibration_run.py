"""Production UK national calibration seam orchestration."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import platform
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from microcosm.build.country_spec import GatesManifest, load_country_spec
from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryRun,
    gate_signing_key_env,
)
from microcosm.build.gate_battery import (
    _canonical_json_bytes as gate_battery_canonical_json_bytes,
)
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
    write_error_receipt,
)
from microcosm.build.staging_v2 import validate_staging_delivery
from microcosm.build.target_materialization import assert_calibration_input_finite
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.diagnostics import (
    uk_target_geography_levels,
    write_uk_calibration_diagnostics,
)
from microcosm.build.uk_runtime.etb_services import (
    UK_NHS_SPENDING_COMPONENT_COLUMNS,
)
from microcosm.build.uk_runtime.national_calibration import UKNationalCalibrationStage
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_household_weight_kind,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry
from microcosm.frame import Frame

_REPOSITORY = Path(__file__).resolve().parents[6]
# The FRS line's spine, staging, imputation and calibration stages share one
# hash chain (logbook/README.md): the dataset token names the base data, not
# the build mechanism, so calibration derives the ratified `uk/frs` scope.
_PIPELINE = "uk-frs-calibration"


@dataclass(frozen=True)
class UKCalibrationRunPaths:
    input_h5: Path
    staging_h5: Path
    diagnostics_json: Path
    build_record_json: Path
    terminal_gate_json: Path


@dataclass(frozen=True)
class UKCalibrationRunResult:
    frame: Frame
    diagnostics_sha256: str
    staging_sha256: str
    build_record_sha256: str
    terminal_gate_sha256: str
    logbook_spool: Path
    gate_report: Mapping[str, object]
    build_record: Mapping[str, object]


UK_CALIBRATION_GATE_SCOPE = (
    "uk_target_fit",
    "uk_weight_ratio",
    "uk_weight_ess",
    "uk_zero_weight_strata",
    "uk_aggregate_admin",
    "uk_calibration_reference_coverage",
)

UK_LOCAL_GATE_SCOPE = (
    "uk_local_geography_ladder_post_calibration",
    "uk_local_area_support",
    "uk_local_target_fit",
    "uk_local_per_family_fit",
    "uk_local_weight_ratio",
    "uk_local_weight_ess",
)

UK_SPINE_GATE_SCOPE = (
    "uk_stage_was_wealth_support",
    "uk_stage_uc_deduction_attributes",
    "uk_stage_lcfs_consumption_support",
    "uk_stage_etb_vat_support",
    "uk_stage_etb_services_support",
    "uk_stage_frs_hmrc_spine_leaves_signal",
    "uk_stage_spi_support_channel_mass",
    "uk_stage_hmrc_spi_income_spine_identity",
    "uk_stage_cgt_incidence_clone_mass",
    "uk_stage_cgt_band_donors_support",
    "uk_stage_hmrc_cgt_gains_spine_summary",
    "uk_stage_salary_sacrifice_realization",
    "uk_stage_student_loans_realization",
    "uk_stage_age_tail_targets",
    # Weight-independent, and its column exists from frs_brma onward, so the
    # spine checks it at the assembled boundary instead of the release end.
    "uk_brma_enum_domain",
)

UK_NATIONAL_GATE_SCOPE = (
    "uk_release_input_coverage_manifest_current",
    "uk_release_family_build_stages",
    "uk_ledger_compile_parity_production_2023",
    "uk_ledger_compile_parity_incumbent_2025",
    # The local-surface compile gates live with the same owner as the national
    # parity pair: the release-cut certification producer
    # (tools/certify_uk_release_cut.py) compiles and supplies both registry
    # artifacts — never the seam.
    "uk_ledger_compile_parity_local_incumbent_2025",
    "uk_target_surface_local_default_2025",
    "uk_release_input_coverage",
    "uk_degenerate_release_surface",
    "uk_nonnegative_columns",
    "uk_uc_capital_coherence",
    "uk_uc_deduction_combination_enum_domain",
    "uk_support",
    "uk_aggregate_admin",
    "uk_export_surface",
    "uk_take_up_signal",
    "uk_student_loan_plan_enum_domain",
    "uk_target_surface",
    "uk_input_mass_parity",
    "uk_qrf_tail_concentration",
    "uk_weights_audit",
)

_SWAP_ACCEPTANCE_GATE_IDS = frozenset({"uk_export_surface", "uk_target_surface"})

#: Gate ids two batteries both own, on purpose. A release certification unions
#: the scoped reports, so a gate appearing twice has to be a declared duplicate
#: the union can reconcile — never an accident that silently double-counts.
#:
#: `uk_aggregate_admin` measures the same admin anchors on two different frames:
#: the seam checks them on the frame it just calibrated, the national terminal
#: battery re-checks them on the release frame. Both are real checks, so both
#: keep the gate.
UK_SHARED_GATE_IDS = frozenset({"uk_aggregate_admin"})


def _scope_exclusions() -> dict[str, str]:
    full = {entry.id for entry in load_country_spec("uk").gates.gates}
    spine = set(UK_SPINE_GATE_SCOPE)
    national = set(UK_NATIONAL_GATE_SCOPE)
    calibration = set(UK_CALIBRATION_GATE_SCOPE)
    local = set(UK_LOCAL_GATE_SCOPE)
    # Closed-world means both halves: every gate owned by someone (below), and
    # no gate owned twice without saying so. Coverage alone would let an
    # accidental overlap through, and the certification union is exactly where
    # that would be paid for.
    for left_name, left, right_name, right in (
        ("calibration", calibration, "spine", spine),
        ("calibration", calibration, "national", national),
        ("calibration", calibration, "local", local),
        ("spine", spine, "national", national),
        ("spine", spine, "local", local),
        ("national", national, "local", local),
    ):
        undeclared = (left & right) - UK_SHARED_GATE_IDS
        if undeclared:
            raise RuntimeError(
                f"UK gate scopes {left_name} and {right_name} both claim "
                f"{sorted(undeclared)} without declaring them in "
                "UK_SHARED_GATE_IDS."
            )
    classified = calibration | spine | national | local
    rationales: dict[str, str] = {}
    for gate_id in sorted(full - set(UK_CALIBRATION_GATE_SCOPE)):
        if gate_id in spine:
            reason = (
                "spine-construction gate; owned by the spine build's scoped battery."
            )
        elif gate_id in national:
            reason = (
                "owned by the release-cut certification producer; runner lands "
                "with the certification, June runner retired"
            )
        elif gate_id in local:
            reason = (
                "local-candidate gate; owned by the rowwise candidate's "
                "scoped battery and excluded from national certification "
                "until microcosm#146."
            )
        elif "parity" in gate_id or gate_id in _SWAP_ACCEPTANCE_GATE_IDS:
            reason = "swap-acceptance evidence; produced by the swap lane, not the calibration seam."
        else:
            reason = "outside the calibration seam's reviewed gate scope."
        rationales[gate_id] = reason
    if classified | set(rationales) != full:
        raise RuntimeError("UK gate scope does not classify every gate id.")
    return rationales


UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS = _scope_exclusions()


def uk_local_gate_scope_exclusions() -> dict[str, str]:
    """Classify every declared entry the local-candidate battery does not run."""

    full = {entry.id for entry in load_country_spec("uk").gates.gates}
    local = set(UK_LOCAL_GATE_SCOPE)
    exclusions: dict[str, str] = {}
    for gate_id in sorted(full - local):
        if gate_id in UK_SPINE_GATE_SCOPE:
            reason = "spine-construction gate; owned by the spine build."
        elif gate_id in UK_CALIBRATION_GATE_SCOPE:
            reason = "national calibration-seam gate; outside the local candidate."
        elif gate_id in UK_NATIONAL_GATE_SCOPE:
            reason = "national release-cut gate; outside the local candidate."
        else:  # pragma: no cover - _scope_exclusions enforces closed-world ownership
            raise RuntimeError(f"UK gate {gate_id!r} belongs to no declared scope.")
        exclusions[gate_id] = reason
    if local | set(exclusions) != full:
        raise RuntimeError("UK local gate scope does not classify every gate id.")
    return exclusions


def run_uk_calibration(
    *,
    paths: UKCalibrationRunPaths,
    input_sha256: str,
    ledger_artifact: Any,
    register_registry: TargetRegistry,
    band_edge_registry: TargetRegistry,
    calibration_year: int,
    exclusion_receipt: Mapping[str, Mapping[str, str]],
    doctrine: Any,
    doctrine_overrides: Mapping[str, Mapping[str, object]],
    measure_resolver: object | None,
    source_pins: Mapping[str, Mapping[str, object]],
    run_config_extra: Mapping[str, object],
    release_id: str,
    logbook_prev_row_digest: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    event_callback: (Callable[[str, str, Mapping[str, object]], None] | None) = None,
    staging_delivery: Mapping[str, object] | None = None,
    exact_k: int | None = None,
    exact_k_pi_hi: float | None = None,
    exact_k_seed: int | None = None,
) -> UKCalibrationRunResult:
    """Run the UK national calibration seam and write its sidecars."""

    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    # Pure-argument validation precedes every environment probe: an
    # incoherent register/receipt/band-edge triple must refuse identically
    # whether or not a git checkout or Logbook chain is reachable.
    _validate_band_edge_registry(
        register_registry=register_registry,
        band_edge_registry=band_edge_registry,
        exclusion_receipt=exclusion_receipt,
    )
    edge_registry = band_edge_registry
    code_pin = git_code_pin(_REPOSITORY)
    # Predecessor configuration is validated before anything is written: a
    # disagreeing chain must refuse with no artifact on disk, not after a
    # staged H5, diagnostics and a signed gate report already exist.
    predecessor = resolve_predecessor(logbook_prev_row_digest)
    run_config = {
        "pipeline": _PIPELINE,
        "release_id": release_id,
        "register_sha256": register_registry.version,
        "calibration_year": int(calibration_year),
        "doctrine": _doctrine_payload(doctrine),
        "doctrine_overrides": dict(doctrine_overrides),
        # The caller verifies the feed's facts and manifest digests; sealing
        # the verified identity into run_config carries it through the
        # identity digest, the build record and the Logbook row, so the run
        # says which Ledger artifact it was measured against.
        "ledger": _ledger_provenance(ledger_artifact),
        "exact_k": (
            None
            if exact_k is None
            else {
                "k": exact_k,
                "pi_hi": exact_k_pi_hi,
                "seed": exact_k_seed,
            }
        ),
        **dict(run_config_extra),
    }
    run_config["band_edge_register_sha256"] = edge_registry.version
    state = AttemptState(
        # Attempts are distinct rows even when they re-run one release: both
        # the local chain and the store refuse a repeated build id.
        build_id=_new_calibration_attempt_id(timestamp=started_ts),
        identity_digest=hashlib.sha256(canonical_json_bytes(run_config)).hexdigest(),
        input_pins_digest=role_pins_digest(source_pins),
        phases_reached=["attempt_started"],
        gate_verdicts={},
    )
    spool_dir = paths.staging_h5.parent / "logbook-spool"
    try:
        return _run_uk_calibration_attempt(
            paths=paths,
            input_sha256=input_sha256,
            ledger_artifact=ledger_artifact,
            register_registry=register_registry,
            band_edge_registry=edge_registry,
            calibration_year=calibration_year,
            exclusion_receipt=exclusion_receipt,
            doctrine=doctrine,
            doctrine_overrides=doctrine_overrides,
            measure_resolver=measure_resolver,
            source_pins=source_pins,
            release_id=release_id,
            state=state,
            run_config=run_config,
            code_pin=code_pin,
            started_at=started_at,
            started_ts=started_ts,
            predecessor=predecessor,
            spool_dir=spool_dir,
            progress_callback=progress_callback,
            event_callback=event_callback,
            staging_delivery=staging_delivery,
            exact_k=exact_k,
            exact_k_pi_hi=exact_k_pi_hi,
            exact_k_seed=exact_k_seed,
        )
    except BaseException as error:
        # Every terminal disposition records a row — successful, failed, or
        # refused (logbook/README.md). A refusal that left no row would be a
        # silent gap in the chain the run is supposed to evidence.
        _record_failed_attempt(
            error=error,
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            seed=(
                exact_k_seed if exact_k is not None else getattr(doctrine, "seed", None)
            ),
            code_pin=code_pin,
            predecessor=predecessor,
            receipt_base_dir=paths.staging_h5.parent,
            spool_dir=spool_dir,
        )
        raise


def _new_calibration_attempt_id(*, timestamp: datetime) -> str:
    instant = timestamp.astimezone(UTC)
    return (
        "uk-frs-calibration-attempt-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


def _validate_band_edge_registry(
    *,
    register_registry: TargetRegistry,
    band_edge_registry: TargetRegistry,
    exclusion_receipt: Mapping[str, Mapping[str, str]],
) -> None:
    """Require the band-edge register to reconstitute the compiled roster.

    The reconciliation always runs — an empty receipt is a claim that nothing
    was pruned, so the two rosters must then be name-identical; it is never
    permission to skip the check. Receipt keys are spec names by the
    applier's construction: ``apply_uk_calibration_measure_exclusions``
    builds the receipt from the matched ``spec.name`` set and raises on any
    exclusion matching zero registry specs, so ``pruned + receipt keys ==
    compiled`` is an exact identity, not a heuristic.
    """

    register_names = _registry_spec_names(register_registry)
    edge_names = _registry_spec_names(band_edge_registry)
    excluded_names = {str(name) for name in exclusion_receipt}
    if not register_names <= edge_names:
        missing = sorted(register_names - edge_names)
        raise ValueError(
            "band-edge register must include every materialized registry spec; "
            f"missing={missing}."
        )
    extra = edge_names - register_names
    if extra != excluded_names:
        raise ValueError(
            "band-edge register extra spec names must match the measure "
            f"exclusion receipt; extra={sorted(extra)}, "
            f"receipt={sorted(excluded_names)}."
        )


def _registry_spec_names(registry: TargetRegistry) -> set[str]:
    return {str(spec.name) for spec in registry.specs}


def _record_failed_attempt(
    *,
    error: BaseException,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    seed: int | None,
    code_pin: str,
    predecessor: str | None,
    receipt_base_dir: Path,
    spool_dir: Path,
) -> None:
    if state.spool_path is not None:
        return
    error_path = write_error_receipt(
        error_receipt_path(receipt_base_dir, build_id=state.build_id),
        state=state,
        pipeline=_PIPELINE,
        error=error,
    )
    apply_error_verdict(
        state,
        f"{local_artifact_reference(error_path, repository_hint=_REPOSITORY)}"
        "#/error_type",
    )
    record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung="f100",
        seed=seed,
        code_pin=code_pin,
        # An operator interrupt is a discarded attempt, not a failed one; the
        # row says which so the chain reads honestly.
        disposition=("discarded" if isinstance(error, KeyboardInterrupt) else "failed"),
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _run_uk_calibration_attempt(
    *,
    paths: UKCalibrationRunPaths,
    input_sha256: str,
    ledger_artifact: Any,
    register_registry: TargetRegistry,
    band_edge_registry: TargetRegistry,
    calibration_year: int,
    exclusion_receipt: Mapping[str, Mapping[str, str]],
    doctrine: Any,
    doctrine_overrides: Mapping[str, Mapping[str, object]],
    measure_resolver: object | None,
    source_pins: Mapping[str, Mapping[str, object]],
    release_id: str,
    state: AttemptState,
    run_config: Mapping[str, object],
    code_pin: str,
    started_at: float,
    started_ts: datetime,
    predecessor: str | None,
    spool_dir: Path,
    progress_callback: Callable[[dict[str, object]], None] | None,
    event_callback: Callable[[str, str, Mapping[str, object]], None] | None,
    staging_delivery: Mapping[str, object] | None,
    exact_k: int | None,
    exact_k_pi_hi: float | None,
    exact_k_seed: int | None,
) -> UKCalibrationRunResult:
    _notify_run_event(event_callback, "input_loading", "started")
    measured_input_sha = _sha256_file(paths.input_h5)
    if measured_input_sha != input_sha256:
        raise ValueError(
            "input H5 sha mismatch: "
            f"measured {measured_input_sha}, pinned {input_sha256}"
        )
    append_phase(state, "input_sha_verified")
    frame, _provenance = load_uk_national_frame(paths.input_h5)
    append_phase(state, "input_loaded")
    spine_sidecar_path = paths.input_h5.with_suffix(".build.json")
    spine_sidecar = load_bound_spine_sidecar(spine_sidecar_path, frame)
    append_phase(state, "input_sidecar_bound")
    assert_calibration_input_finite(frame)
    append_phase(state, "input_finite")
    _notify_run_event(
        event_callback,
        "input_loading",
        "completed",
        entity_row_counts={
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
    )

    stage = UKNationalCalibrationStage(
        register_registry,
        # The declared calibration year the register was compiled at — the
        # stage validates it; there is deliberately no fallback to the input
        # frame's base-year time_period or any ambient default.
        period=calibration_year,
        doctrine=doctrine,
        measure_resolver=measure_resolver,
        band_edge_registry=band_edge_registry,
        progress_callback=progress_callback,
        stage_callback=event_callback,
        exact_k=exact_k,
        exact_k_pi_hi=exact_k_pi_hi,
        exact_k_seed=exact_k_seed,
    )
    _notify_run_event(event_callback, "calibration", "started")
    calibrated = stage(frame)
    append_phase(state, "national_calibration_solved")
    _notify_run_event(
        event_callback,
        "calibration",
        "completed",
        target_count=len(stage.registry.specs),
        entity_row_counts={
            entity: int(len(calibrated.table(entity))) for entity in calibrated.entities
        },
    )

    build_block = {
        "build_id": state.build_id,
        "code_pin": code_pin,
        # Captured at solve time and signed with the diagnostics bytes, so a
        # release assembler can only ever pin the environment that actually
        # calibrated the candidate — never an invented one.
        "runtime": _runtime_provenance(),
        "source_pins": dict(source_pins),
        "ledger": run_config["ledger"],
        "input_posture": {
            "tier": "staging_candidate",
            "sha256": measured_input_sha,
            "size_bytes": paths.input_h5.stat().st_size,
        },
        "doctrine": _doctrine_payload(doctrine),
        "doctrine_overrides": dict(doctrine_overrides),
        "measure_exclusions": dict(exclusion_receipt),
        "measure_resolution": (
            stage.manifest.get("measure_resolution")
            if isinstance(stage.manifest, Mapping)
            else None
        ),
        "register": _register_census(register_registry, exclusion_receipt),
        "spine_provenance": spine_provenance_from_sidecar(
            spine_sidecar_path,
            spine_sidecar,
        ),
        "score_vs_enhanced_frs": None,
        "exact_k_ladder": (
            stage.manifest.get("exact_k_ladder")
            if isinstance(stage.manifest, Mapping)
            else None
        ),
    }
    _notify_run_event(event_callback, "diagnostics", "started")
    write_uk_calibration_diagnostics(
        stage.solve_result,
        paths.diagnostics_json,
        calibrated,
        target_geography_levels=uk_target_geography_levels(stage.registry),
        target_registry=stage.registry,
        build=build_block,
    )
    diagnostics_sha = _sha256_file(paths.diagnostics_json)
    append_phase(state, "diagnostics_written")
    _notify_run_event(
        event_callback,
        "diagnostics",
        "completed",
        target_count=len(stage.diagnostics),
    )

    _notify_run_event(event_callback, "release_check_evaluation", "started")
    gate_report = _run_calibration_gate_battery(
        calibrated,
        stage,
        paths.terminal_gate_json,
        release_id=release_id,
        diagnostics_sha256=diagnostics_sha,
    )
    append_phase(state, "calibration_gates_evaluated")
    _notify_run_event(
        event_callback,
        "release_check_evaluation",
        "completed",
        check_count=len(gate_report["gates"]),
    )
    for gate_id, payload in gate_report["gates"].items():
        state.gate_verdicts[gate_id] = {
            "verdict": payload["status"],
            "receipt": f"local://{paths.terminal_gate_json.name}#/gates/{gate_id}",
        }

    _notify_run_event(event_callback, "candidate_h5_creation", "started")
    write_uk_national_frame(calibrated, paths.staging_h5)
    staging_sha = _sha256_file(paths.staging_h5)
    append_phase(state, "staging_h5_written")
    _notify_run_event(
        event_callback,
        "candidate_h5_creation",
        "completed",
        size_bytes=paths.staging_h5.stat().st_size,
    )

    _notify_run_event(event_callback, "build_record_creation", "started")
    record = {
        "schema_version": 1,
        "pipeline": _PIPELINE,
        "build_id": state.build_id,
        "run_config": run_config,
        "source_pins": dict(source_pins),
        "role_pins_digest": role_pins_digest(source_pins),
        "input_posture": build_block["input_posture"],
        "spine_provenance": build_block["spine_provenance"],
        "register": build_block["register"],
        "calibration": stage.manifest,
        "gate_summary": _gate_summary(gate_report),
        # No shippability claim lives here: the calibration-scoped battery
        # covers 6 of the declared gate entries. The release verdict is the
        # release-cut certification's, produced over this record.
        "certification": {
            "expected_artifact": str(
                paths.staging_h5.with_suffix(".release_certification.json")
            ),
            "producer": "tools/certify_uk_release_cut.py",
        },
        "artifacts": {
            "staging_h5": {"path": str(paths.staging_h5), "sha256": staging_sha},
            "diagnostics_json": {
                "path": str(paths.diagnostics_json),
                "sha256": diagnostics_sha,
            },
            "terminal_gate_json": {
                "path": str(paths.terminal_gate_json),
                "sha256": _sha256_file(paths.terminal_gate_json),
            },
        },
    }
    if staging_delivery is not None:
        record["staging_delivery"] = validate_staging_delivery(staging_delivery)
    _write_json(paths.build_record_json, record)
    build_record_sha = _sha256_file(paths.build_record_json)
    append_phase(state, "build_record_written")
    _notify_run_event(
        event_callback,
        "build_record_creation",
        "completed",
        size_bytes=paths.build_record_json.stat().st_size,
    )
    state.artifact_location = local_artifact_reference(
        paths.staging_h5, repository_hint=_REPOSITORY
    )
    spool = record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung="f100",
        seed=(exact_k_seed if exact_k is not None else getattr(doctrine, "seed", None)),
        code_pin=code_pin,
        disposition="iterating",
        predecessor=predecessor,
        spool_dir=spool_dir,
    )
    return UKCalibrationRunResult(
        frame=calibrated,
        diagnostics_sha256=diagnostics_sha,
        staging_sha256=staging_sha,
        build_record_sha256=build_record_sha,
        terminal_gate_sha256=_sha256_file(paths.terminal_gate_json),
        logbook_spool=spool,
        gate_report=gate_report,
        build_record=record,
    )


def _notify_run_event(
    callback: Callable[[str, str, Mapping[str, object]], None] | None,
    stage_id: str,
    status: str,
    **details: object,
) -> None:
    if callback is not None:
        callback(stage_id, status, details)


def _run_calibration_gate_battery(
    frame: Frame,
    stage: UKNationalCalibrationStage,
    path: Path,
    *,
    release_id: str,
    diagnostics_sha256: str,
) -> dict[str, object]:
    manifest = _calibration_gate_manifest()
    admin_totals, admin_receipt = uk_aggregate_admin_totals(frame, manifest)
    artifacts = {
        "national_calibration": stage.manifest,
        "parity_evidence": SimpleNamespace(
            target_relative_errors={
                str(row["name"]): float(row["relative_error"])
                for row in stage.diagnostics
            }
        ),
        "aggregate_admin": admin_totals,
        # The target-fit deferral register is evaluated against the run
        # clock (schema-2 approval windows); the seam supplies today's date
        # exactly as the rowwise candidate build supplies its start date.
        "exclusions_evaluated_on": datetime.now(UTC).date(),
    }
    battery = GateBatteryRun(
        manifest,
        release_id=release_id,
        # The seam never runs release-candidate posture: its scoped battery
        # covers 6 of the declared entries and must never sign a
        # shippability claim (the #757 release-cut audit). Shippability
        # comes only from the release-cut certification.
        report_path=path,
        release_candidate=False,
        registry=UK_GATE_REGISTRY,
        release_evidence={"calibration_diagnostics_sha256": diagnostics_sha256},
    )
    battery.run_phase("terminal", EvidenceContext(frame=frame, artifacts=artifacts))
    battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
    payload = battery.report_payload()
    finalize_uk_scoped_gate_report(
        payload,
        posture="calibration_seam",
        scope_exclusions=dict(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS),
        aggregate_admin_measurement=admin_receipt,
    )
    _write_json(path, payload)
    return payload


def load_bound_spine_sidecar(path: Path, frame: Frame) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"input H5 build sidecar absent: {path}")
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"input H5 build sidecar is invalid JSON: {path}") from exc
    if not isinstance(sidecar, dict):
        raise ValueError(f"input H5 build sidecar must be a JSON object: {path}")
    _assert_spine_sidecar_binds_frame(sidecar, frame)
    _assert_spine_gate_report_passed(_spine_gate_report_path(path), sidecar)
    return sidecar


def _assert_spine_sidecar_binds_frame(
    sidecar: Mapping[str, object],
    frame: Frame,
) -> None:
    expected_counts = sidecar.get("entity_row_counts")
    actual_counts = {entity: int(len(frame.table(entity))) for entity in frame.entities}
    if expected_counts != actual_counts:
        raise ValueError(
            "input H5 build sidecar row-count mismatch: "
            f"sidecar {expected_counts!r}, frame {actual_counts!r}"
        )
    expected_kind = sidecar.get("household_weight_kind")
    actual_kind = uk_household_weight_kind(frame).value
    if expected_kind != actual_kind:
        raise ValueError(
            "input H5 build sidecar household_weight_kind mismatch: "
            f"sidecar {expected_kind!r}, frame {actual_kind!r}"
        )
    expected_total = sidecar.get("household_weight_total")
    actual_total = float(frame.weights_for("household").values.sum())
    if not isinstance(expected_total, int | float) or not np.isclose(
        float(expected_total), actual_total
    ):
        raise ValueError(
            "input H5 build sidecar household_weight_total mismatch: "
            f"sidecar {expected_total!r}, frame {actual_total!r}"
        )


def _spine_gate_report_path(sidecar_path: Path) -> Path:
    if sidecar_path.name.endswith(".build.json"):
        stem = sidecar_path.name[: -len(".build.json")]
        return sidecar_path.with_name(f"{stem}.spine_gates.json")
    return sidecar_path.with_suffix(".spine_gates.json")


def _assert_spine_gate_report_passed(
    report_path: Path,
    sidecar: Mapping[str, object],
) -> None:
    bypass = sidecar.get("spine_gate_bypass")
    if bypass is not None:
        if not isinstance(bypass, Mapping) or bypass.get("reviewed") is not True:
            raise ValueError("spine_gate_bypass must be a reviewed bypass object.")
        if not str(bypass.get("reason", "")).strip():
            raise ValueError("spine_gate_bypass needs a non-empty reason.")
        return
    if not report_path.is_file():
        raise ValueError(f"input H5 spine gate report absent: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"input H5 spine gate report is invalid JSON: {report_path}"
        ) from exc
    if not isinstance(report, Mapping):
        raise ValueError(
            f"input H5 spine gate report must be a JSON object: {report_path}"
        )
    if report.get("blocked_at_phase") is not None:
        raise ValueError(
            "input H5 spine gate report blocked_at_phase must be null; "
            f"got {report.get('blocked_at_phase')!r}."
        )
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("input H5 spine gate report is missing gates.")
    failing = sorted(
        f"{gate_id}:{payload.get('status')}"
        for gate_id, payload in gates.items()
        if isinstance(payload, Mapping)
        and payload.get("criticality") == "release_blocking"
        and payload.get("status") != "passed"
    )
    if failing:
        raise ValueError(
            "input H5 spine gate report has non-passing release-blocking "
            f"entries: {failing}."
        )


def spine_provenance_from_sidecar(
    path: Path,
    sidecar: Mapping[str, object],
) -> dict[str, object]:
    report_path = _spine_gate_report_path(path)
    if report_path.is_file():
        spine_gate_report: Mapping[str, object] = {
            "path": str(report_path),
            "sha256": _sha256_file(report_path),
        }
    elif isinstance(sidecar.get("spine_gate_bypass"), Mapping):
        spine_gate_report = {"bypass": dict(sidecar["spine_gate_bypass"])}
    else:
        spine_gate_report = {}
    return {
        "sidecar": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "schema_version": sidecar.get("schema_version"),
            "pipeline": sidecar.get("pipeline"),
        },
        "spine_gate_report": dict(spine_gate_report),
        "stages": list(sidecar.get("stages", ())),
        "stage_records": list(sidecar.get("stage_records", ())),
        "stage_evidence": dict(sidecar.get("stage_evidence", {})),
        "artifact_pins": dict(sidecar.get("artifact_pins", {})),
        "input_artifact_pins": dict(sidecar.get("input_artifact_pins", {})),
        "resource_pins": dict(sidecar.get("resource_pins", {})),
        "stage_artifact_pins": dict(sidecar.get("stage_artifact_pins", {})),
        "declared_seeds": dict(sidecar.get("declared_seeds", {})),
        "rules_engine": dict(sidecar.get("rules_engine", {})),
        "source_vintages": dict(sidecar.get("source_vintages", {})),
        "stochastic_contract_sha256": sidecar.get("stochastic_contract_sha256"),
    }


def _calibration_gate_manifest() -> GatesManifest:
    return uk_scoped_gate_manifest(
        UK_CALIBRATION_GATE_SCOPE,
        phases=("terminal",),
        policy_suffix="calibration_seam_scope",
    )


def uk_scoped_gate_manifest(
    scope: tuple[str, ...] | frozenset[str],
    *,
    phases: tuple[str, ...],
    policy_suffix: str,
    source: GatesManifest | None = None,
) -> GatesManifest:
    """Filter the declared UK gate spec to one battery's scope.

    The one scope-filtering implementation every scoped producer shares
    (spine build, calibration seam, release cut). ``source`` lets a caller
    that already holds a loaded spec — or a hermetic test that stubs one —
    supply it; the default is the committed package spec.
    """

    if source is None:
        source = load_country_spec("uk").gates
    entries = tuple(entry for entry in source.gates if entry.id in scope)
    missing = sorted(set(scope) - {entry.id for entry in entries})
    if missing:
        raise RuntimeError(f"UK gate scope names undeclared gate id(s): {missing}.")
    return GatesManifest(
        country=source.country,
        version=source.version,
        policy=f"{source.policy}; {policy_suffix}",
        phases=phases,
        gates=entries,
    )


#: Admin anchors published against a concept the frame carries only in parts.
#: The anchor keeps the publisher's shape (one NHS budget line); our stages
#: carry the spend split by point of delivery. Composing is the translation,
#: declared here and recorded per anchor in the measurement receipt — never a
#: reason to drop the anchor or to let it measure a silent zero.
UK_DERIVED_ADMIN_ANCHOR_MEASURES: Mapping[str, tuple[str, ...]] = {
    "nhs_spending": UK_NHS_SPENDING_COMPONENT_COLUMNS,
}


def uk_aggregate_admin_totals(
    frame: Frame, manifest: GatesManifest
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Measure every declared admin anchor, fail-loud on absent evidence.

    The anchor value's magnitude tells its statistic — NEED per-household
    means are hundreds of pounds, the NHS anchor is a national total — the
    same reviewed convention the first armed-run receipts used. Non-household
    entities carry their household's weight through the person linkage.
    """

    anchors = []
    for entry in manifest.gates:
        if entry.id == "uk_aggregate_admin":
            anchors = list(entry.parameters.get("anchors", ()))
            break
    household_weights = np.asarray(frame.weights_for("household").values, dtype=float)
    totals: dict[str, float] = {}
    receipt: list[dict[str, object]] = []
    for anchor in anchors:
        entity = str(anchor.get("entity", "household"))
        name = str(anchor.get("name", anchor.get("measure")))
        measure = str(anchor.get("measure", anchor.get("name")))
        table = frame.table(entity)
        composed_from: tuple[str, ...] = ()
        if measure not in table:
            composed_from = UK_DERIVED_ADMIN_ANCHOR_MEASURES.get(measure, ())
            missing = [column for column in composed_from if column not in table]
            if not composed_from or missing:
                raise ValueError(
                    f"aggregate_admin anchor {name!r} needs {entity}.{measure}, "
                    "which the calibrated frame does not carry"
                    + (
                        f" (declared as the sum of {list(composed_from)}, "
                        f"missing {missing})"
                        if composed_from
                        else ""
                    )
                    + "; refusing to fabricate a measured value."
                )
        if entity == "household":
            weights = household_weights
        elif entity == "person":
            # `household_id` is a group-table id column, which the Frame
            # kernel validates unique before any frame exists, and the CGT
            # clone stage offsets its clones' ids for exactly that reason —
            # so this lookup cannot silently drop a duplicate key.
            person = frame.table("person")
            lookup = dict(
                zip(
                    frame.table("household")["household_id"].to_numpy(),
                    household_weights,
                    strict=True,
                )
            )
            weights = np.asarray(
                [lookup[key] for key in person["person_household_id"].to_numpy()],
                dtype=float,
            )
        else:
            raise ValueError(
                f"aggregate_admin anchor {name!r} declares entity {entity!r}; "
                "the calibration seam measures household and person anchors "
                "only."
            )
        values = (
            table[list(composed_from)].to_numpy(dtype=float).sum(axis=1)
            if composed_from
            else table[measure].to_numpy(dtype=float)
        )
        total = float(np.dot(values, weights))
        carriers = values != 0
        carrier_weight = float(weights[carriers].sum())
        mean_carriers = (
            float(np.dot(values[carriers], weights[carriers]) / carrier_weight)
            if carrier_weight
            else float("nan")
        )
        declared = float(anchor["value"])
        measured = mean_carriers if abs(declared) < 1e6 else total
        totals[name] = measured
        receipt.append(
            {
                "anchor": name,
                "entity": entity,
                "measure": measure,
                "measured": measured,
                "weighted_total": total,
                "weighted_mean_carriers": mean_carriers,
                "statistic_convention": "assessed_by_anchor_magnitude",
                "composed_from": list(composed_from),
            }
        )
    return totals, receipt


#: The manifest fields the UK run seals into ``run_config``. Narrower than
#: the artifact's whole manifest on purpose: the identity digest should name
#: *which* published artifact was compiled, not re-hash its manifest.
_LEDGER_MANIFEST_IDENTITY_FIELDS = (
    "artifact_id",
    "profile",
    "schema_version",
    "generated_at",
)

#: Epoch witnesses :meth:`LedgerConsumerArtifact.provenance` supplies, split
#: by shape. Named here so the delegation below cannot quietly drop one: a
#: run that compiled a chronicle-era or mixed-epoch feed has to say so in its
#: own evidence, not only in the loader's return value
#: (PolicyEngine/chronicle#143).
_LEDGER_EPOCH_WITNESS_SCALARS = ("schema_epoch",)
_LEDGER_EPOCH_WITNESS_LISTS = (
    "fact_key_epochs",
    "undeclared_fact_key_domains",
    "fact_schema_versions",
)


def _ledger_provenance(artifact: Any) -> dict[str, object]:
    """The verified identity of the Chronicle consumer feed this run compiled.

    Delegates to :meth:`microcosm.build.ledger_artifact.LedgerConsumerArtifact.provenance`
    rather than rebuilding the block field by field. Rebuilding it is how the
    epoch witnesses went missing here in the first place: the loader learned
    which era resolved the targets and the UK run kept reporting only the
    hashes. Anything the shared block gains, this block gains.

    The whole block is sealed into ``run_config`` and so into the run's
    identity digest: the epoch witnesses enter it beside the hashes, which is
    why this delegation is a change to the digest of an otherwise identical
    configuration, once. Only the manifest sub-block is UK-shaped, and it
    stays narrow so the digest names the artifact rather than re-hashing its
    whole manifest.

    A bare ``consumer_facts.jsonl`` feed carries no manifest, so its
    Chronicle-side provenance is recorded as absent rather than invented. So
    is a stand-in that predates the shared block: the fields it cannot supply
    are recorded ``None``/empty rather than fabricated.
    """

    shared = getattr(artifact, "provenance", None)
    block: Mapping[str, Any] = shared() if callable(shared) else {}
    provenance: dict[str, object] = {
        field: block.get(field, getattr(artifact, field, None))
        for field in ("facts_sha256", "fact_row_count", "manifest_sha256")
    }
    for field in _LEDGER_EPOCH_WITNESS_SCALARS:
        provenance[field] = block.get(field)
    for field in _LEDGER_EPOCH_WITNESS_LISTS:
        provenance[field] = list(block.get(field) or ())
    manifest = getattr(artifact, "manifest", None)
    if isinstance(manifest, Mapping):
        provenance["manifest"] = {
            key: manifest.get(key)
            for key in _LEDGER_MANIFEST_IDENTITY_FIELDS
            if manifest.get(key) is not None
        }
    return provenance


#: Packages whose versions the seam pins into the signed diagnostics build
#: block. The release assembler treats these as the only legitimate source
#: for manifest runtime/compatibility pins.
_RUNTIME_PROVENANCE_PACKAGES = (
    "policyengine-core",
    "policyengine-uk",
    "microcosm-frame",
    "microcosm-calibrate",
    "microcosm-build",
    "microcosm-data",
)


def runtime_provenance() -> dict[str, str]:
    """The calibrating environment's package versions, for the build block."""

    runtime = {"python": platform.python_version()}
    for package in _RUNTIME_PROVENANCE_PACKAGES:
        try:
            runtime[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            runtime[package] = "unavailable"
    return runtime


# Compatibility aliases for the established internal call sites.
_load_bound_spine_sidecar = load_bound_spine_sidecar
_spine_provenance_from_sidecar = spine_provenance_from_sidecar
_runtime_provenance = runtime_provenance


def _register_census(
    registry: TargetRegistry, exclusions: Mapping[str, Mapping[str, str]]
) -> dict[str, object]:
    return {
        "country": registry.country,
        "version": registry.version,
        "compiled_count": len(registry.specs) + len(exclusions),
        "excluded_count": len(exclusions),
        "calibrated_count": len(registry.specs),
    }


def _doctrine_payload(doctrine: Any) -> dict[str, object]:
    return {
        key: getattr(doctrine, key)
        for key in (
            "epochs",
            "learning_rate",
            "max_weight_ratio",
            "seed",
            "target_loss_cap",
            "scale_rule",
            "target_weight_rule",
            "mass_rule",
            "l0_lambda",
        )
        if hasattr(doctrine, key)
    }


def _gate_summary(report: Mapping[str, object]) -> dict[str, object]:
    gates = report.get("gates", {})
    if not isinstance(gates, Mapping):
        return {}
    return {
        gate_id: payload.get("status")
        for gate_id, payload in gates.items()
        if isinstance(payload, Mapping)
    }


def finalize_uk_scoped_gate_report(
    payload: dict[str, object],
    *,
    posture: str,
    scope_exclusions: Mapping[str, str],
    aggregate_admin_measurement: object,
) -> None:
    """Graft the scoped-report trio onto a battery payload and re-sign it.

    Every scoped UK producer (the calibration seam, the release-cut
    certification) declares its posture, the rationale for each gate it
    does not run, and its admin-anchor measurement receipt, then signs the
    augmented bytes. One implementation, shared, so the parts the
    certification composes over cannot drift apart in shape.
    """

    payload["posture"] = posture
    payload["scope_exclusions"] = dict(scope_exclusions)
    payload["aggregate_admin_measurement"] = aggregate_admin_measurement
    resign_uk_gate_report(payload)


def resign_uk_gate_report(payload: dict[str, object]) -> None:
    attestation = payload.get("attestation")
    if not isinstance(attestation, dict):
        raise RuntimeError("gate report has no attestation block.")
    env = gate_signing_key_env("uk")
    encoded = os.environ.get(env)
    if not encoded:
        raise RuntimeError(
            f"{env} must be set; unsigned full-scale calibration runs refuse to stage."
        )
    try:
        key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"{env} must be valid base64.") from exc
    if len(key) != 32:
        raise RuntimeError(f"{env} must decode to exactly 32 bytes.")
    attestation.pop("signing_error", None)
    attestation["signing_key_sha256"] = hashlib.sha256(key).hexdigest()
    attestation["signature"] = None
    # Sign with the gate battery's canonical form — the one the battery used
    # for the original signature and the one every verifier (the data
    # contract's terminal and dense checks) recomputes. The Logbook's
    # canonical form renders integral floats without the ".0" (50.0 -> 50),
    # so a report re-signed with it cannot be authenticated wherever a gate
    # detail carries an integral float (microcosm#762 R17: the local
    # battery's ESS floor 50.0).
    attestation["signature"] = hmac.new(
        key, gate_battery_canonical_json_bytes(payload), hashlib.sha256
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
