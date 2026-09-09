"""Build a joint local, ladder, and national UK rowwise candidate.

Pinned Ledger facts supply the local and national registries. The command
samples before cloning, resolves both local grains and national measures on the
cloned frame, and calibrates every row in one doctrine solve. A dry run compiles
the registries and reports analytical matrix/support evidence without running
the policy engine, solving, or writing output files.

For pre-#762 synthetic fixtures, omitting the Ledger arguments retains the
adjudicated constituency-household compatibility path.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryBlockedError,
    GateBatteryRun,
)
from microcosm.build.gates import GateResult
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    atomic_write_json,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    preflight_digest,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
    sha256_argument,
    write_error_receipt,
)
from microcosm.build.target_materialization import resolve_target_measures
from microcosm.build.uk_runtime import (
    UK_GATE_REGISTRY,
    UK_LOCAL_CLONE_COUNT,
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_SOLVE_EPOCHS,
    UK_LOCAL_TARGET_LOSS_CAP,
    CalibrationFrameAdapter,
    UKLadderRowwiseDatasetResult,
    UkOaLadder,
    UKRowwiseDoctrineSolve,
    UKRowwiseLocalMatrix,
    UKRowwiseNationalRows,
    apply_uk_cross_grain_reconciliation,
    build_uk_rowwise_local_matrix,
    build_uk_rowwise_local_surface_matrix,
    clone_uk_dataset_with_ladder_geography,
    compile_uk_local_target_registry,
    compile_uk_target_registry,
    compute_household_metrics,
    constituency_household_targets,
    drop_injected_measure_inputs,
    inject_measure_inputs,
    ladder_clone_index_column,
    ladder_target_provenance,
    load_bound_spine_sidecar,
    load_uk_local_area_crosswalk,
    load_uk_national_frame,
    load_uk_oa_ladder,
    local_target_census,
    materialize_uk_ledger_targets,
    require_adjudicated_uk_local_binding,
    rotated_uk_local_holdout,
    runtime_provenance,
    solve_uk_rowwise_weights_under_doctrine,
    spine_provenance_from_sidecar,
    uk_household_weight_kind,
    uk_ladder_area_support_summary,
    uk_ladder_household_uprating,
    uk_ledger_households_total,
    uk_local_doctrine_with_overrides,
    uk_local_target_surface,
    uk_support_limited_misses,
    uk_time_period,
    write_uk_calibration_diagnostics,
    write_uk_rowwise_dataset,
)
from microcosm.build.uk_runtime.calibration_run import (
    UK_LOCAL_GATE_SCOPE,
    finalize_uk_scoped_gate_report,
    uk_local_gate_scope_exclusions,
    uk_scoped_gate_manifest,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import _spec_geography
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_RUNG_TOKENS,
    UK_SAMPLE_SEED_DEFAULT,
    sample_uk_spine_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import Frame, MassChangeRecord

BOUND_TARGET_FAMILIES = ("census_households/constituency",)
BOUND_NATIONAL_TARGETS: tuple[str, ...] = ()
CANDIDATE_FILENAME_TEMPLATE = "microcosm_uk_{calibration_year}_local.h5"
LOCAL_GATE_REPORT_FILENAME_TEMPLATE = (
    "microcosm_uk_{calibration_year}_local.local_gates.json"
)
MANIFEST_FILENAME = "rowwise_candidate_manifest.json"
SOLVE_DIAGNOSTICS_FILENAME = "solve_diagnostics.csv"
CALIBRATION_DIAGNOSTICS_FILENAME = "calibration_diagnostics.json"
AREA_SUPPORT_FILENAME = "area_support_summary.csv"
PAST_CAP_FILENAME = "past_cap_census.json"
LOCAL_REGISTRY_FILENAME = "local_target_registry.json"

_CONSERVE_MASS = False
_TARGET_RECORDS: int | None = None
_L0_LAMBDA = 0.0
_BUDGET_ITERS = 10
_UK_CANDIDATE_PIPELINE = "uk-local-candidate"
_LOCAL_GATE_POLICY_SUFFIX = "local_candidate"
_REPOSITORY = Path(__file__).resolve().parents[1]
_PAST_CAP_COUNT_KEYS = (
    "n_targets",
    "past_at_init",
    "past_at_final",
    "escaped",
    "frozen",
    "pushed_out",
)


class _LadderAssignment:
    """A clone paired in memory with the exact ladder object that produced it."""

    def __init__(
        self,
        result: UKLadderRowwiseDatasetResult,
        ladder: UkOaLadder,
    ) -> None:
        self.result = result
        self.ladder = ladder


def _new_candidate_build_id(
    *, seed: int, timestamp: datetime, rung: str = "f100"
) -> str:
    instant = timestamp.astimezone(UTC)
    return (
        f"uk-local-candidate-{rung}-s{seed}-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


def _candidate_clone_counts_argument(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not value.strip() or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "candidate clone counts must be a non-empty comma list of positive integers"
        )
    try:
        counts = [int(part.strip()) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate clone counts must be a comma list of positive integers"
        ) from error
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError(
            "candidate clone counts must all be positive integers"
        )
    return tuple(sorted(set(counts)))


def _sample_candidate_frame(
    frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Sample spine families below f100; keep the full rung untouched."""

    pre_count = len(frame.table("household"))
    if fraction == 1.0:
        return frame, {
            "fraction": 1.0,
            "seed": int(seed),
            "rung_token": UK_SAMPLE_RUNG_TOKENS[fraction],
            "sampled": False,
            "pre_household_count": int(pre_count),
            "post_household_count": int(pre_count),
        }

    sampled, receipt = sample_uk_spine_frame(
        frame,
        fraction=fraction,
        seed=seed,
    )
    return sampled, {"sampled": True, **receipt}


def _resolve_candidate_engine_surface(
    frame,
    national_registry,
    *,
    period: int,
    scratch_dir: Path,
    band_edge_registry=None,
    resolver_factory=UKMeasureResolver,
    blocks: int = 1,
) -> tuple[Any, Any, UKRowwiseNationalRows, dict[str, pd.DataFrame], dict[str, Any]]:
    """Resolve national inputs and local metrics on the cloned frame.

    ``blocks=1`` uses one scratch-mode engine for the whole clone.  The
    reviewed escape hatch ``blocks=K`` resolves each clone index separately,
    then rejoins every entity-level prepared column by its stable entity id so
    the full-frame target materialization and single solve retain frame order.
    """

    household = frame.table("household")
    if blocks < 1:
        raise ValueError("engine resolution blocks must be positive.")
    if blocks == 1:
        block_frames = [(None, frame)]
    else:
        clone_column = ladder_clone_index_column("household")
        if clone_column not in household.columns:
            raise ValueError(f"per-clone engine resolution requires {clone_column}.")
        clone_indices = tuple(sorted(household[clone_column].unique().tolist()))
        if len(clone_indices) != blocks:
            raise ValueError(
                "engine resolution blocks must match the realized clone indices: "
                f"requested {blocks}, found {clone_indices}."
            )
        person = frame.table("person")
        block_frames = []
        for clone_index in clone_indices:
            household_ids = set(
                household.loc[
                    household[clone_column] == clone_index,
                    "household_id",
                ].tolist()
            )
            person_mask = person["person_household_id"].isin(household_ids)
            block = frame.select(person_mask)
            # The block carries a K-th of the cloned mass while its log still
            # ends on the full-clone record, and the scratch export validates
            # the chain. Declare the subset explicitly: old = the cloned
            # total, new = the block total, reason naming the block. The block
            # frame is engine scratch and is discarded after resolution.
            block_weights = block.weights_for("household")
            full_total = float(frame.weights_for("household").total)
            block_total = float(block_weights.total)
            subset_record = MassChangeRecord(
                entity="household",
                old_total=full_total,
                new_total=block_total,
                declared_factor=block_total / full_total,
                reason=(
                    f"engine resolution block {clone_index} of {blocks}: "
                    "scratch subset of the cloned frame for measure "
                    "resolution only, discarded after resolution"
                ),
            )
            block = Frame(
                {
                    **{name: block.table(name) for name in block.entities},
                    **{name: block.link(name) for name in block.links},
                },
                block.schema,
                {
                    entity: block.weights_for(entity)
                    for entity in block.weighted_entities
                },
                block.strata,
                mass_log=(*block.mass_log, subset_record),
                metadata=block.metadata,
            )
            block_frames.append((clone_index, block))

    measure_parts: dict[tuple[str, str], list[pd.Series]] = {}
    metric_parts: dict[str, list[pd.DataFrame]] = {
        "constituency": [],
        "la": [],
    }
    resolver_receipts: list[Mapping[str, Any]] = []
    national_input_keys: set[tuple[str, str]] | None = None
    for clone_index, block_frame in block_frames:
        block_scratch = (
            scratch_dir if clone_index is None else scratch_dir / f"clone-{clone_index}"
        )
        resolver = resolver_factory(
            simulation_source=None,
            scratch_dir=block_scratch,
            year=period,
            frame=block_frame,
        )
        resolution = resolve_target_measures(
            lambda block_frame=block_frame: CalibrationFrameAdapter(block_frame),
            national_registry,
            resolver,
            period=period,
        )
        keys = set(resolution.measure_inputs)
        if national_input_keys is None:
            national_input_keys = keys
        elif keys != national_input_keys:
            raise RuntimeError(
                "per-clone engine resolution returned inconsistent national inputs."
            )
        for (entity, variable), values in resolution.measure_inputs.items():
            entity_table = block_frame.table(entity)
            entity_id = f"{entity}_id"
            measure_parts.setdefault((entity, variable), []).append(
                pd.Series(
                    np.asarray(values),
                    index=entity_table[entity_id].tolist(),
                )
            )
        block_household_ids = block_frame.table("household")["household_id"].tolist()
        for area_type in metric_parts:
            metric_parts[area_type].append(
                compute_household_metrics(
                    resolver.simulation,
                    area_type,
                    period=period,
                    household_ids=block_household_ids,
                )
            )
        resolver_receipts.append(resolver.receipt())
        del resolver
        simulation_input = block_scratch / "simulation-input.h5"
        simulation_input.unlink(missing_ok=True)
        try:
            block_scratch.rmdir()
        except OSError:
            pass

    measure_inputs: dict[tuple[str, str], np.ndarray] = {}
    for (entity, variable), parts in measure_parts.items():
        combined = pd.concat(parts)
        if combined.index.has_duplicates:
            raise RuntimeError(
                f"per-clone engine resolution duplicated {entity} ids for {variable}."
            )
        ordered_ids = frame.table(entity)[f"{entity}_id"]
        ordered = combined.reindex(ordered_ids.tolist())
        if ordered.isna().any():
            raise RuntimeError(
                f"per-clone engine resolution missed {entity} rows for {variable}."
            )
        measure_inputs[(entity, variable)] = ordered.to_numpy()

    full_household_ids = household["household_id"].tolist()
    local_metrics = {}
    for area_type, parts in metric_parts.items():
        combined = pd.concat(parts)
        if combined.index.has_duplicates:
            raise RuntimeError(
                f"per-clone engine resolution duplicated {area_type} household ids."
            )
        ordered = combined.reindex(full_household_ids)
        if ordered.isna().any().any():
            raise RuntimeError(
                f"per-clone engine resolution missed {area_type} household rows."
            )
        local_metrics[area_type] = ordered

    adapter = CalibrationFrameAdapter(frame)
    # Injected engine inputs are scratch state for materialization only:
    # they must be dropped before the prepared frame is assembled, or the
    # flattening rule refuses columns that now exist on two entities
    # (region, esa_* on the live spine). Same lifecycle as the national stage.
    original_columns = {
        entity: set(table.columns) for entity, table in adapter.tables.items()
    }
    inject_measure_inputs(adapter, measure_inputs)
    materialized = materialize_uk_ledger_targets(
        adapter,
        national_registry,
        period=period,
        band_edge_registry=(
            national_registry if band_edge_registry is None else band_edge_registry
        ),
    )
    if materialized.skipped:
        raise RuntimeError(
            "candidate national target materialization skipped row(s): "
            f"{[skip.__dict__ for skip in materialized.skipped]}."
        )
    modes = {receipt.get("mode") for receipt in resolver_receipts}
    versions = {receipt.get("policyengine_uk_version") for receipt in resolver_receipts}
    if len(modes) != 1 or len(versions) != 1:
        raise RuntimeError("per-clone engine resolver provenance is inconsistent.")
    cgt_period_contract = resolver_receipts[0].get("cgt_period_contract")
    if any(
        block_receipt.get("cgt_period_contract") != cgt_period_contract
        for block_receipt in resolver_receipts[1:]
    ):
        raise RuntimeError("per-clone CGT period contract is inconsistent.")
    receipt = {
        "mode": next(iter(modes)),
        "engine_version": next(iter(versions)),
        "households": len(frame.table("household")),
        "persons": len(frame.table("person")),
        "benunits": len(frame.table("benunit")),
        "national_inputs": len(measure_inputs),
        "local_metrics": {
            area_type: len(metrics.columns)
            for area_type, metrics in local_metrics.items()
        },
        "blocks": blocks,
    }
    if cgt_period_contract is not None:
        receipt["cgt_period_contract"] = cgt_period_contract
    if blocks > 1:
        receipt["deviation"] = "per_clone_block_engine_resolution"
        present = sorted(
            column
            for column in UK_BLOCK_SENSITIVE_MEASURE_COLUMNS
            if column in measure_inputs
        )
        receipt["block_sensitivity"] = {
            "known_population_normalised_measures": list(
                UK_BLOCK_SENSITIVE_MEASURE_COLUMNS
            ),
            "present_in_this_run": present,
            "caveat": (
                "per-block engine resolution mis-measures population-normalised "
                "formulas (each block reproduces a national aggregate); rows "
                "on these measures are not evidence for adjudication from this "
                "run. Resolve in a single block before ruling on them."
            ),
        }
    try:
        scratch_dir.rmdir()
    except OSError:
        pass
    drop_injected_measure_inputs(adapter, measure_inputs, original_columns)
    national_rows = UKRowwiseNationalRows(
        targets=national_registry.to_target_set(),
        registry=national_registry,
        families=tuple(sorted({spec.family for spec in national_registry.specs})),
    )
    return (
        adapter.prepared_frame(),
        adapter.restore,
        national_rows,
        local_metrics,
        receipt,
    )


def _pin_from_artifact(info: Mapping[str, Any]) -> dict[str, object]:
    return {
        "sha256": str(info["sha256"]),
        "size_bytes": int(info["bytes"]),
    }


def _candidate_identity_digest(
    *,
    pins: dict[str, dict[str, object]],
    args: argparse.Namespace,
    source_year: int,
) -> str:
    payload = {
        "build_kind": "uk_rowwise_calibrated_candidate",
        "inputs": pins,
        "parameters": _parameters(args, source_year=source_year),
        "source_year": source_year,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _record_candidate_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    seed: int,
    code_pin: str,
    disposition: str,
    predecessor: str | None,
    spool_dir: Path,
    rung: str = "f100",
) -> Path:
    return record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_UK_CANDIDATE_PIPELINE,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        disposition=disposition,
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _record_candidate_error(
    *,
    error: BaseException,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    seed: int,
    code_pin: str,
    predecessor: str | None,
    base_dir: Path,
    spool_dir: Path,
    rung: str = "f100",
) -> None:
    error_path = write_error_receipt(
        error_receipt_path(base_dir, build_id=state.build_id),
        state=state,
        pipeline=_UK_CANDIDATE_PIPELINE,
        error=error,
    )
    apply_error_verdict(
        state,
        f"{local_artifact_reference(error_path, repository_hint=_REPOSITORY)}#/error_type",
    )
    _record_candidate_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        seed=seed,
        code_pin=code_pin,
        disposition="failed",
        predecessor=predecessor,
        spool_dir=spool_dir,
        rung=rung,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="National Microcosm UK staging H5.",
    )
    parser.add_argument(
        "--input-sha256",
        type=sha256_argument,
        help="Pinned SHA-256 of --input-h5 (required for the joint registry path).",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        required=True,
        help="Full-UK OA geography ladder NPZ.",
    )
    parser.add_argument(
        "--ladder-sha256",
        type=sha256_argument,
        help="Pinned SHA-256 of --ladder (required for the joint registry path).",
    )
    parser.add_argument("--ledger-facts", type=Path)
    parser.add_argument("--ledger-facts-sha256", type=sha256_argument)
    parser.add_argument("--ledger-manifest-sha256", type=sha256_argument)
    parser.add_argument("--measure-exclusions", type=Path)
    parser.add_argument("--register-json", type=Path)
    parser.add_argument(
        "--target-weight-rule",
        choices=("uniform", "grain_equal"),
        default=UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule,
    )
    parser.add_argument("--release-candidate", action="store_true")
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the candidate H5 and evidence sidecars.",
    )
    parser.add_argument("--n-clones", type=int, default=UK_LOCAL_CLONE_COUNT)
    parser.add_argument(
        "--candidate-clone-counts",
        type=_candidate_clone_counts_argument,
        help="Dry-run only comma-separated candidate clone counts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Spine sampling rung: 0.01, 0.10, or 1.0.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=UK_SAMPLE_SEED_DEFAULT,
    )
    parser.add_argument(
        "--engine-blocks",
        type=int,
        default=1,
        help="Resolve one engine or one block per clone (must equal --n-clones).",
    )
    parser.add_argument(
        "--source-year",
        type=int,
        help="Survey year recorded for lineage (calibration uses the FRS release year).",
    )
    parser.add_argument("--source-lineage-modulus", type=int)
    parser.add_argument("--epochs", type=int, default=UK_LOCAL_SOLVE_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument(
        "--expected-constituency-vintage",
        default="2024_pcon",
        help="Constituency vintage required from the ladder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the fenced clone/matrix plan without solving or writing any file."
        ),
    )
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help=(
            "Optional current Logbook chain head. If omitted, "
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST is used, then genesis null."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the rowwise candidate build."""

    args = _parse_args(argv)
    _validate_cli_args(args)
    if args.candidate_clone_counts is not None and not args.dry_run:
        raise ValueError("--candidate-clone-counts is valid only with --dry-run.")
    if _CONSERVE_MASS:
        raise NotImplementedError(
            "the candidate manifest's calibration_mass_change block reads "
            "the kernel's free-mass record; a conserve-mass doctrine run "
            "appends no record and needs its own reviewed manifest shape "
            "before this constant may flip."
        )
    if args.dry_run:
        # Dry runs plan without solving or writing and record no Logbook
        # row on any path, so they need no chain configuration.
        return _run_candidate(args, attempt=None)
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    digest = preflight_digest(_UK_CANDIDATE_PIPELINE)
    state = AttemptState(
        build_id=_new_candidate_build_id(
            seed=args.seed,
            timestamp=started_ts,
            rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        ),
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            }
        },
    )
    return _run_candidate(
        args,
        attempt={
            "state": state,
            "started_at": started_at,
            "started_ts": started_ts,
            "code_pin": "unresolved-local-git-code-pin",
            # Logbook chain configuration is validated before any terminal
            # work: a malformed or conflicting head refuses the run with no
            # row and no side effects (#666 adversarial-review finding).
            "predecessor": resolve_predecessor(args.logbook_prev_row_digest),
        },
    )


def _run_candidate(
    args: argparse.Namespace,
    *,
    attempt: dict[str, object] | None,
) -> int:
    """Build the candidate, recording every non-dry terminal outcome.

    The recording envelope opens before input verification so that setup
    failures — unreadable inputs, frame or ladder load errors, clone and
    target-binding refusals — still spool a failed row (#666
    adversarial-review finding). Dry runs pass ``attempt=None`` and record
    nothing.
    """

    out_dir = args.out.expanduser().resolve()
    try:
        input_h5 = _require_file(args.input_h5, label="--input-h5")
        ladder_path = _require_file(args.ladder, label="--ladder")
        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(f"--out must be a directory path, got {out_dir}.")

        input_artifact = _artifact_info(input_h5)
        ladder_artifact = _artifact_info(ladder_path)
        _verify_requested_pin("--input-h5", input_artifact, requested=args.input_sha256)
        _verify_requested_pin("--ladder", ladder_artifact, requested=args.ladder_sha256)
        pins = {
            "dataset": _pin_from_artifact(input_artifact),
            "ladder": _pin_from_artifact(ladder_artifact),
        }
        state: AttemptState | None = None
        if attempt is not None:
            unpacked_state = attempt["state"]
            assert isinstance(unpacked_state, AttemptState)
            state = unpacked_state
            attempt["code_pin"] = git_code_pin(_REPOSITORY)
            state.input_pins_digest = role_pins_digest(pins)
            append_phase(state, "configured")
            append_phase(state, "inputs_pinned")
        national_frame, _national_provenance = load_uk_national_frame(input_h5)
        calibration_year = int(load_uk_frs_release().calibration_year)
        args._calibration_year = calibration_year
        if args.ledger_facts is not None:
            spine_sidecar_path = input_h5.with_suffix(".build.json")
            spine_sidecar = load_bound_spine_sidecar(
                spine_sidecar_path,
                national_frame,
            )
            args._spine_provenance = spine_provenance_from_sidecar(
                spine_sidecar_path,
                spine_sidecar,
            )
        else:
            args._spine_provenance = {}
        national_frame, sampling = _sample_candidate_frame(
            national_frame,
            fraction=args.sample_fraction,
            seed=args.sample_seed,
        )
        args._sampling_receipt = sampling
        source_year = _source_year(
            args.source_year,
            time_period=uk_time_period(national_frame),
        )
        if state is not None:
            # The identity digest waits on the frame-derived source year;
            # earlier failures record with the preflight placeholder.
            state.identity_digest = _candidate_identity_digest(
                pins=pins,
                args=args,
                source_year=source_year,
            )
        output_paths = _output_paths(
            out_dir,
            source_year=source_year,
            calibration_year=calibration_year,
        )
        _validate_output_paths(
            output_paths,
            input_h5=input_h5,
            ladder_path=ladder_path,
        )
        ladder = load_uk_oa_ladder(ladder_path)
        target_provenance = ladder_target_provenance(ladder)
        joint_inputs = _load_joint_target_inputs(args)
        if joint_inputs is not None:
            # microcosm#762 A15: the ladder's census household counts bind at
            # the calibration year through one national factor from the
            # Ledger's published UK household total (fail-closed by name on a
            # real Ledger artifact; a synthetic artifact without facts binds
            # the rows as published and says so, which the release posture
            # refuses).
            facts = getattr(joint_inputs.get("artifact"), "facts", None)
            if facts is None:
                joint_inputs["ladder_household_uprating"] = {
                    "applied": False,
                    "reason": (
                        "the joint target inputs carry no Ledger facts; ladder "
                        "household rows bind at their census vintage."
                    ),
                }
            else:
                joint_inputs["ladder_household_uprating"] = (
                    uk_ladder_household_uprating(
                        ladder,
                        uk_ledger_households_total(
                            facts, period=joint_inputs["calibration_year"]
                        ),
                        period=joint_inputs["calibration_year"],
                    )
                )
            if args.release_candidate and not joint_inputs[
                "ladder_household_uprating"
            ].get("applied"):
                raise SystemExit(
                    "error: --release-candidate requires the A15 ladder household "
                    "uprating: "
                    + str(joint_inputs["ladder_household_uprating"].get("reason"))
                )
        doctrine, doctrine_override = uk_local_doctrine_with_overrides(
            UK_LOCAL_SOLVE_DOCTRINE,
            (
                {}
                if args.target_weight_rule == UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule
                else {"target_weight_rule": args.target_weight_rule}
            ),
        )
        args._doctrine_override_receipt = doctrine_override
        if doctrine.target_weight_rule != args.target_weight_rule:
            raise RuntimeError(
                "local doctrine override did not bind the requested rule."
            )

        print("cloning through the ladder route...", file=sys.stderr, flush=True)
        assignment = _clone_with_ladder_binding(
            national_frame,
            ladder,
            n_clones=args.n_clones,
            seed=args.seed,
            source_year=source_year,
            expected_constituency_vintage=args.expected_constituency_vintage,
            source_lineage_modulus=args.source_lineage_modulus,
        )
        clone = assignment.result
        if state is not None:
            append_phase(state, "cloned")

        if joint_inputs is not None and args.dry_run:
            plan = _joint_dry_run_plan(
                args,
                clone=clone,
                sampled_spine=national_frame,
                ladder=ladder,
                joint_inputs=joint_inputs,
                source_year=source_year,
                input_artifact=input_artifact,
                ladder_artifact=ladder_artifact,
                target_provenance=target_provenance,
            )
            _assert_artifacts_unchanged(
                input_h5=input_h5,
                input_artifact=input_artifact,
                ladder_path=ladder_path,
                ladder_artifact=ladder_artifact,
            )
            print(_json_text(plan), end="")
            return 0

        if joint_inputs is None:
            print("binding census household targets...", file=sys.stderr, flush=True)
            household, problem, cross_grain = _build_bound_problem(
                assignment,
                target_ladder=ladder,
            )
            solve_frame = clone.frame
            restore = None
            national_rows = None
            bound_families = BOUND_TARGET_FAMILIES
            measure_resolution: Mapping[str, Any] = {}
            args._rung_surface = {
                "fraction": float(args.sample_fraction),
                "dropped_cells": 0,
                "dropped_by_grain": {},
                "dropped_by_family": {},
            }
        else:
            print("resolving joint local and national surface...", file=sys.stderr)
            (
                solve_frame,
                restore,
                national_rows,
                local_metrics,
                measure_resolution,
            ) = _resolve_candidate_engine_surface(
                clone.frame,
                joint_inputs["national_registry"],
                period=joint_inputs["calibration_year"],
                scratch_dir=out_dir.parent
                / f".{out_dir.name}.candidate-engine-scratch",
                band_edge_registry=joint_inputs["band_edge_registry"],
                blocks=args.engine_blocks,
            )
            (
                household,
                problem,
                cross_grain,
                bound_families,
                rung_surface,
            ) = _build_joint_problem(
                assignment,
                target_ladder=ladder,
                local_registry=joint_inputs["local_registry"],
                national_registry=joint_inputs["national_registry"],
                local_metrics=local_metrics,
                period=joint_inputs["calibration_year"],
                sample_fraction=args.sample_fraction,
                reviewed_unbound_higher_targets=joint_inputs[
                    "reviewed_unbound_higher_targets"
                ],
                ladder_household_uprating=joint_inputs.get("ladder_household_uprating"),
            )
            args._rung_surface = rung_surface
        args._bound_families = tuple(bound_families)
        args._joint_inputs_receipt = joint_inputs
        args._measure_resolution = dict(measure_resolution)
        if state is not None:
            append_phase(state, "targets_bound")

        if args.dry_run:
            binding_adjudications = require_adjudicated_uk_local_binding(
                bound_families,
                problem.target_frame,
            )
            _assert_artifacts_unchanged(
                input_h5=input_h5,
                input_artifact=input_artifact,
                ladder_path=ladder_path,
                ladder_artifact=ladder_artifact,
            )
            plan = _dry_run_plan(
                args,
                clone=clone,
                problem=problem,
                source_year=source_year,
                input_artifact=input_artifact,
                ladder_artifact=ladder_artifact,
                target_provenance=target_provenance,
                binding_adjudications=binding_adjudications,
                cross_grain=cross_grain,
            )
            print(_json_text(plan), end="")
            return 0

        assert attempt is not None
        assert state is not None
        started_at = attempt["started_at"]
        started_ts = attempt["started_ts"]
        assert isinstance(started_at, float)
        assert isinstance(started_ts, datetime)
        predecessor = attempt["predecessor"]
        assert predecessor is None or isinstance(predecessor, str)
        code_pin = str(attempt["code_pin"])

        print(
            f"solving {problem.matrix.shape[0]} targets x "
            f"{problem.matrix.shape[1]} households under the doctrine...",
            file=sys.stderr,
            flush=True,
        )
        solve = solve_uk_rowwise_weights_under_doctrine(
            solve_frame,
            problem,
            bound_families=bound_families,
            national_rows=national_rows,
            target_weight_rule=args.target_weight_rule,
            restore=restore,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            conserve_mass=_CONSERVE_MASS,
            target_records=_TARGET_RECORDS,
            l0_lambda=_L0_LAMBDA,
            budget_iters=_BUDGET_ITERS,
            seed=args.seed,
        )
        _validate_solve_result(solve, problem=problem)
        append_phase(state, "solved")

        # The kernel minted the calibration mass record inside calibrate() (the
        # CALIBRATED kind transition is enforced there too); the record names
        # the bound families via the doctrine's mass reason.
        calibration_record = solve.frame.mass_log[-1]
        if "calibration" not in calibration_record.reason:
            raise ValueError(
                "calibrated frame's latest mass record is not the calibration "
                f"record: {calibration_record.reason!r}."
            )
        support = _candidate_area_support(
            solve.frame.table("household"),
            ladder,
            weights=solve.weights,
        )
        _validate_support_summary(support)
        local_diagnostics = _local_gate_diagnostics(solve.diagnostics)
        target_registry, target_geography_levels = _local_diagnostics_registry(
            solve,
            problem,
            national_registry=(
                None if joint_inputs is None else joint_inputs["national_registry"]
            ),
        )
        try:
            gate_report, candidate_gate = _run_local_gate_battery(
                frame=solve.frame,
                support=support,
                diagnostics=local_diagnostics,
                report_path=output_paths["local_gates"],
                release_id=state.build_id,
                evaluated_on=started_ts.date(),
                enforce_only=(
                    None
                    if args.sample_fraction == 1.0
                    else ("uk_local_geography_ladder_post_calibration",)
                ),
                # The battery attests the posture it ran under: a release
                # candidate blocks on absent evidence and can be shippable.
                release_candidate=bool(args.release_candidate),
            )
        except GateBatteryBlockedError:
            # Write-then-block, extended to the whole evidence bundle: a
            # release-blocking failure at f100 still writes the diagnostics,
            # manifest and artifact (marked unreleasable) so the block can be
            # reviewed; only a non-passing ladder verdict is structural and
            # re-raises. The Logbook row records the attempt as failed.
            gate_report = json.loads(
                output_paths["local_gates"].read_text(encoding="utf-8")
            )
            _apply_gate_verdicts(state, gate_report, output_paths["local_gates"])
            ladder_entry = gate_report["gates"].get(
                "uk_local_geography_ladder_post_calibration", {}
            )
            if ladder_entry.get("status") != "passed":
                raise
            candidate_gate = clone.gate
            blocked_failures, diagnostic_failures = _gate_failures_by_criticality(
                gate_report
            )
            if not blocked_failures:
                # The battery blocked, yet the persisted report names no
                # failed release-blocking entry: the report and the error
                # disagree, which is structural.
                raise
            unenforced_failures = []
        else:
            _apply_gate_verdicts(state, gate_report, output_paths["local_gates"])
            # Nothing blocked. Release-blocking entries can still hold a
            # failure here: below f100 only the ladder gate is enforced, and
            # a dev build tolerates absent evidence. Those lines are reported
            # as not enforced, never as a block.
            unenforced_failures, diagnostic_failures = _gate_failures_by_criticality(
                gate_report
            )
            blocked_failures = []
        args._gate_report = gate_report
        args._blocked_failures = blocked_failures
        args._diagnostic_failures = diagnostic_failures
        args._unenforced_release_failures = (
            [] if blocked_failures else unenforced_failures
        )
        append_phase(
            state, "candidate_gated" if not blocked_failures else "candidate_blocked"
        )

        if args.skip_holdout:
            rotated_holdout = {"skipped": True}
        else:
            rotated_holdout = rotated_uk_local_holdout(
                solve_frame,
                problem,
                bound_families=bound_families,
                national_rows=national_rows,
                target_weight_rule=args.target_weight_rule,
                restore=restore,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                conserve_mass=_CONSERVE_MASS,
                target_records=_TARGET_RECORDS,
                l0_lambda=_L0_LAMBDA,
                budget_iters=_BUDGET_ITERS,
                solve_seed=args.seed,
            )
        args._rotated_holdout = rotated_holdout

        candidate = dataclasses.replace(
            clone,
            frame=solve.frame,
            gate=candidate_gate,
            output_path=None,
        )
        support_by_grain = {
            ("la" if grain == "local_authority" else str(grain)): rows.reset_index(
                drop=True
            )
            for grain, rows in support.groupby("geography_level", sort=True)
        }
        args._support_limited_misses = uk_support_limited_misses(
            solve.diagnostics,
            support_by_grain,
            max_abs_relative_error=0.25,
        )
        _assert_artifacts_unchanged(
            input_h5=input_h5,
            input_artifact=input_artifact,
            ladder_path=ladder_path,
            ladder_artifact=ladder_artifact,
        )

        manifest = _write_output_bundle(
            args,
            candidate=candidate,
            clone=clone,
            problem=problem,
            solve=solve,
            local_diagnostics=local_diagnostics,
            target_registry=target_registry,
            target_geography_levels=target_geography_levels,
            rotated_holdout=rotated_holdout,
            support=support,
            calibration_record=calibration_record,
            source_year=source_year,
            output_paths=output_paths,
            input_artifact=input_artifact,
            ladder_artifact=ladder_artifact,
            target_provenance=target_provenance,
            cross_grain=cross_grain,
        )
        append_phase(state, "published")
        state.artifact_location = local_artifact_reference(
            output_paths["dataset"],
            repository_hint=_REPOSITORY,
        )
        spool_path = _record_candidate_attempt(
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            seed=args.seed,
            code_pin=code_pin,
            disposition="failed" if blocked_failures else "iterating",
            predecessor=predecessor,
            spool_dir=out_dir / "logbook-spool",
            rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        )
        print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
        print(_json_text(manifest), end="")
        if blocked_failures:
            print(
                "Gate battery blocked the artifact at f100; evidence bundle "
                f"written, artifact unreleasable: {blocked_failures[:5]}",
                file=sys.stderr,
            )
            return 1
        return 0
    except Exception as error:
        if attempt is None:
            # Dry runs record no row on any path, including failures.
            raise
        failed_state = attempt["state"]
        assert isinstance(failed_state, AttemptState)
        failed_started_at = attempt["started_at"]
        failed_started_ts = attempt["started_ts"]
        assert isinstance(failed_started_at, float)
        assert isinstance(failed_started_ts, datetime)
        failed_predecessor = attempt["predecessor"]
        assert failed_predecessor is None or isinstance(failed_predecessor, str)
        _record_candidate_error(
            error=error,
            state=failed_state,
            started_at=failed_started_at,
            started_ts=failed_started_ts,
            seed=args.seed,
            code_pin=str(attempt["code_pin"]),
            predecessor=failed_predecessor,
            base_dir=out_dir,
            spool_dir=out_dir / "logbook-spool",
            rung=UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        )
        raise


def _clone_with_ladder_binding(
    dataset: Any,
    ladder: UkOaLadder,
    *,
    n_clones: int,
    seed: int,
    source_year: int,
    expected_constituency_vintage: str | None,
    source_lineage_modulus: int | None,
) -> _LadderAssignment:
    clone = clone_uk_dataset_with_ladder_geography(
        dataset,
        ladder,
        n_clones=n_clones,
        seed=seed,
        source_year=source_year,
        expected_constituency_vintage=expected_constituency_vintage,
        source_lineage_modulus=source_lineage_modulus,
    )
    return _LadderAssignment(clone, ladder)


def _verify_requested_pin(
    label: str,
    artifact: Mapping[str, Any],
    *,
    requested: str | None,
) -> None:
    if requested is None:
        artifact["pin_verified"] = False
        return
    measured = str(artifact["sha256"])
    if measured != requested:
        raise SystemExit(
            f"error: {label} sha mismatch: measured {measured}, pinned {requested}"
        )
    artifact["pin_verified"] = True


def _load_joint_target_inputs(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.ledger_facts is None:
        return None
    artifact = load_ledger_consumer_artifact(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
    )
    calibration_year = int(load_uk_frs_release().calibration_year)
    national_compilation = compile_uk_target_registry(
        artifact.facts, target_period=calibration_year
    )
    if national_compilation.unsupported:
        raise SystemExit(
            f"{len(national_compilation.unsupported)} national target references "
            "failed to compile"
        )
    local_compilation = compile_uk_local_target_registry(
        artifact.facts,
        target_period=calibration_year,
        crosswalk=load_uk_local_area_crosswalk(),
    )
    if local_compilation.unsupported:
        raise SystemExit(
            f"{len(local_compilation.unsupported)} local target references "
            "failed to compile"
        )
    exclusions = load_uk_calibration_measure_exclusions(args.measure_exclusions)
    national_registry, exclusion_receipt = apply_uk_calibration_measure_exclusions(
        national_compilation.registry, exclusions
    )
    national_specs_by_name = {
        spec.name: spec for spec in national_compilation.registry.specs
    }
    reviewed_unbound_higher_targets = {
        str(
            national_specs_by_name[name].metadata.get(
                "contract_target_id", national_specs_by_name[name].name
            )
        ): record
        for name, record in exclusion_receipt.items()
    }
    if args.register_json is not None:
        try:
            frozen = TargetRegistry.from_json(args.register_json)
        except ValueError as error:
            raise SystemExit(
                f"error: frozen scoring register is unusable: {error}"
            ) from error
        if frozen.version != national_registry.version:
            raise SystemExit(
                "re-derived register differs from the frozen scoring register: "
                f"{national_registry.version} vs {frozen.version}"
            )
    return {
        "artifact": artifact,
        "calibration_year": calibration_year,
        "national_registry": national_registry,
        "band_edge_registry": national_compilation.registry,
        "local_registry": local_compilation.registry,
        "measure_exclusions": exclusion_receipt,
        "reviewed_unbound_higher_targets": reviewed_unbound_higher_targets,
    }


def _national_contract_target_ids(registry: TargetRegistry) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(spec.metadata.get("contract_target_id", spec.name))
                for spec in registry.specs
            }
        )
    )


def _joint_surface_registry(
    local_registry: TargetRegistry,
    national_registry: TargetRegistry,
) -> TargetRegistry:
    """Put national controls beside local cells for cross-grain reconciliation."""

    return TargetRegistry(
        [*local_registry.specs, *national_registry.specs],
        country="uk",
    )


def _build_joint_problem(
    assignment: _LadderAssignment,
    *,
    target_ladder: UkOaLadder,
    local_registry: TargetRegistry,
    national_registry: TargetRegistry,
    local_metrics: Mapping[str, pd.DataFrame],
    period: int,
    sample_fraction: float,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]],
    ladder_household_uprating: Mapping[str, Any] | None = None,
) -> tuple[
    pd.DataFrame,
    UKRowwiseLocalMatrix,
    dict[str, Any],
    tuple[str, ...],
    dict[str, Any],
]:
    if assignment.ladder is not target_ladder:
        raise ValueError(
            "assignment and targets must come from the same loaded UK OA ladder object."
        )
    household = assignment.result.frame.table("household").reset_index(drop=True)
    household_index = pd.Index(household["household_id"], name="household_id")
    metrics = {
        grain: frame.set_axis(household_index, axis="index")
        for grain, frame in local_metrics.items()
    }
    assigned = {
        "constituency": pd.Series(
            household["constituency_code"].astype(str).to_numpy(),
            index=household_index,
        ),
        "la": pd.Series(
            household["local_authority_code"].astype(str).to_numpy(),
            index=household_index,
        ),
    }
    national_ids = _national_contract_target_ids(national_registry)
    surface, cross_grain = uk_local_target_surface(
        _joint_surface_registry(local_registry, national_registry),
        target_ladder,
        bound_national_target_ids=national_ids,
        period=period,
        reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
        ladder_household_uprating=ladder_household_uprating,
    )
    covered = {
        grain: set(values.astype(str).tolist()) for grain, values in assigned.items()
    }
    covered_mask = pd.Series(
        [
            str(row.area_code) in covered[str(row.area_type)]
            for row in surface.itertuples(index=False)
        ],
        index=surface.index,
        dtype=bool,
    )
    dropped = surface.loc[~covered_mask]
    if sample_fraction < 1.0:
        surface = surface.loc[covered_mask].reset_index(drop=True)
    # Below f100 a covered area can still carry a nonzero cell with no metric
    # support in the sample (no self-employed household among three drawn
    # rows). The builder refuses such a cell at every rung; at development
    # rungs the cell is dropped here and receipted instead. f100 stays strict.
    unreachable = surface.iloc[0:0]
    if sample_fraction < 1.0 and len(surface):
        nonzero_by_grain = {
            grain: (metrics[grain] != 0).groupby(assigned[grain]).sum()
            for grain in metrics
        }
        unreachable_mask = pd.Series(
            [
                float(row.value) != 0.0
                and str(row.metric) in nonzero_by_grain[str(row.area_type)].columns
                and str(row.area_code) in nonzero_by_grain[str(row.area_type)].index
                and int(
                    nonzero_by_grain[str(row.area_type)].loc[
                        str(row.area_code), str(row.metric)
                    ]
                )
                == 0
                for row in surface.itertuples(index=False)
            ],
            index=surface.index,
            dtype=bool,
        )
        unreachable = surface.loc[unreachable_mask]
        surface = surface.loc[~unreachable_mask].reset_index(drop=True)
    rung_surface = {
        "dropped_unreachable_cells": int(len(unreachable)),
        "dropped_unreachable_by_grain": {
            str(key): int(value)
            for key, value in unreachable.groupby("area_type").size().items()
        },
        "dropped_unreachable_by_family": {
            str(key): int(value)
            for key, value in unreachable.groupby("family").size().items()
        },
        "fraction": float(sample_fraction),
        "dropped_cells": int(len(dropped) if sample_fraction < 1.0 else 0),
        "dropped_by_grain": (
            {
                str(key): int(value)
                for key, value in dropped.groupby("area_type").size().items()
            }
            if sample_fraction < 1.0
            else {}
        ),
        "dropped_by_family": (
            {
                str(key): int(value)
                for key, value in dropped.groupby("family").size().items()
            }
            if sample_fraction < 1.0
            else {}
        ),
    }
    rosters = {
        "constituency": tuple(map(str, np.unique(target_ladder.constituency_code))),
        "la": tuple(map(str, np.unique(target_ladder.local_authority_code))),
    }
    problem = build_uk_rowwise_local_surface_matrix(
        metrics,
        assigned,
        surface,
        area_codes_by_grain=rosters,
        require_every_assigned_area_covered=(sample_fraction == 1.0),
    )
    local_bound = tuple(
        sorted(
            {
                f"{row.family}/{row.area_type}"
                for row in surface[["family", "area_type"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
        )
    )
    national_bound = tuple(
        f"national/{family}"
        for family in sorted({spec.family for spec in national_registry.specs})
    )
    return (
        household,
        problem,
        cross_grain,
        (*local_bound, *national_bound),
        rung_surface,
    )


def _joint_dry_run_plan(
    args: argparse.Namespace,
    *,
    clone: UKLadderRowwiseDatasetResult,
    sampled_spine: Any,
    ladder: UkOaLadder,
    joint_inputs: Mapping[str, Any],
    source_year: int,
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    national_registry = joint_inputs["national_registry"]
    surface, cross_grain = uk_local_target_surface(
        _joint_surface_registry(
            joint_inputs["local_registry"],
            national_registry,
        ),
        ladder,
        bound_national_target_ids=_national_contract_target_ids(national_registry),
        period=joint_inputs["calibration_year"],
        reviewed_unbound_higher_targets=joint_inputs["reviewed_unbound_higher_targets"],
        ladder_household_uprating=joint_inputs.get("ladder_household_uprating"),
    )
    household = clone.frame.table("household")
    covered = {
        "constituency": set(household["constituency_code"].astype(str)),
        "la": set(household["local_authority_code"].astype(str)),
    }
    covered_mask = pd.Series(
        [
            str(row.area_code) in covered[str(row.area_type)]
            for row in surface.itertuples(index=False)
        ],
        index=surface.index,
        dtype=bool,
    )
    dropped = surface.loc[~covered_mask]
    active_surface = (
        surface.loc[covered_mask].reset_index(drop=True)
        if args.sample_fraction < 1.0
        else surface
    )
    household_count = len(clone.frame.table("household"))
    clone_support: dict[str, object] = {}
    for clone_count in args.candidate_clone_counts or (args.n_clones,):
        candidate = (
            clone
            if clone_count == args.n_clones
            else _clone_with_ladder_binding(
                sampled_spine,
                ladder,
                n_clones=clone_count,
                seed=args.seed,
                source_year=source_year,
                expected_constituency_vintage=args.expected_constituency_vintage,
                source_lineage_modulus=args.source_lineage_modulus,
            ).result
        )
        # The typed frame weights are the authority; the persisted
        # household_weight column is an export artefact the loaded spine
        # does not carry, so attach them the way the real support path does.
        candidate_household = candidate.frame.table("household").copy()
        candidate_household["household_weight"] = np.asarray(
            candidate.frame.weights_for("household").values, dtype=np.float64
        )
        summaries = uk_ladder_area_support_summary(candidate_household, ladder)
        clone_support[str(clone_count)] = {
            grain: {
                "minimum_rows": int(rows["nonzero_households"].min()),
                "minimum_effective_sample_size": float(
                    rows["effective_sample_size"].min()
                ),
                "minimum_distinct_sources": int(
                    rows["nonzero_source_households"].min()
                ),
            }
            for grain, rows in summaries.items()
        }
    return {
        "schema_version": 2,
        "build_kind": "uk_rowwise_calibrated_candidate_plan",
        "dry_run": True,
        "survey_year": source_year,
        "calibration_year": joint_inputs["calibration_year"],
        "identity": {
            "spine": dict(input_artifact),
            "ladder": dict(ladder_artifact),
            "ledger": joint_inputs["artifact"].provenance(),
        },
        "sampling": dict(args._sampling_receipt),
        "rung_surface": {
            "rung": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
            "fraction": args.sample_fraction,
            "dropped_cells": int(len(dropped) if args.sample_fraction < 1.0 else 0),
            "dropped_by_grain": (
                {
                    str(key): int(value)
                    for key, value in dropped.groupby("area_type").size().items()
                }
                if args.sample_fraction < 1.0
                else {}
            ),
            "dropped_by_family": (
                {
                    str(key): int(value)
                    for key, value in dropped.groupby("family").size().items()
                }
                if args.sample_fraction < 1.0
                else {}
            ),
            "unreachable_check": "deferred_to_build",
        },
        "vintages": _local_vintage_census(joint_inputs["local_registry"]),
        "cross_grain": cross_grain,
        "matrix": {
            "rows": int(len(active_surface) + len(national_registry.specs)),
            "columns": household_count,
            "local_rows": len(active_surface),
            "national_rows": len(national_registry.specs),
        },
        "candidate_clone_counts": list(args.candidate_clone_counts or (args.n_clones,)),
        "candidate_clone_support": clone_support,
        "releasable": False,
        "engine": "not_run",
        "ladder_target_provenance": dict(target_provenance),
    }


def _local_vintage_census(registry: TargetRegistry) -> list[dict[str, object]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for spec in registry.specs:
        resolved = str(spec.metadata.get("ledger_fact_period", ""))
        target = str(spec.period)
        if not resolved or resolved == target:
            continue
        level, _ = _spec_geography(spec)
        key = (spec.family, level, resolved, target)
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "family": family,
            "geography_level": level,
            "resolved_period": resolved,
            "target_period": target,
            "cells": cells,
        }
        for (family, level, resolved, target), cells in sorted(counts.items())
    ]


def _build_bound_problem(
    assignment: _LadderAssignment,
    *,
    target_ladder: UkOaLadder,
) -> tuple[pd.DataFrame, UKRowwiseLocalMatrix, dict[str, Any]]:
    """Bind the one target family, refusing separately loaded ladders."""

    if assignment.ladder is not target_ladder:
        raise ValueError(
            "assignment and targets must come from the same loaded UK OA ladder object."
        )
    clone = assignment.result
    household = clone.frame.table("household").reset_index(drop=True)
    household_index = pd.Index(
        household["household_id"],
        name="household_id",
    )
    metrics = pd.DataFrame(
        {"households": np.ones(len(household), dtype=np.float64)},
        index=household_index,
    )
    assigned = pd.Series(
        household["constituency_code"].astype(str).to_numpy(),
        index=household_index,
        name="constituency_code",
    )
    targets = constituency_household_targets(target_ladder)
    local_surface = pd.DataFrame(
        {
            "grain": "constituency",
            "geography_id": targets["code"].astype(str),
            "target_id": "external:census_households/households",
            "value": targets["households"].to_numpy(dtype=np.float64),
        }
    )
    reconciled_surface, cross_grain = apply_uk_cross_grain_reconciliation(
        local_surface,
        BOUND_NATIONAL_TARGETS,
    )
    targets = targets.copy()
    targets["households"] = reconciled_surface["value"].to_numpy(dtype=np.float64)
    problem = build_uk_rowwise_local_matrix(
        metrics,
        assigned,
        targets,
        area_type="constituency",
        code_column="code",
    )
    return (
        household,
        problem,
        {
            "bound_national_targets": list(BOUND_NATIONAL_TARGETS),
            **cross_grain,
        },
    )


def _candidate_area_support(
    household: pd.DataFrame,
    ladder: UkOaLadder,
    *,
    weights: np.ndarray,
) -> pd.DataFrame:
    weighted_household = household.copy()
    weighted_household["household_weight"] = np.asarray(weights, dtype=np.float64)
    summaries = uk_ladder_area_support_summary(weighted_household, ladder)
    return pd.concat(
        (
            summaries["constituency"].assign(geography_level="constituency"),
            summaries["la"].assign(geography_level="local_authority"),
        ),
        ignore_index=True,
    )[
        [
            "geography_level",
            "area_code",
            "assigned_households",
            "nonzero_households",
            "nonzero_source_households",
            "weight_sum",
            "max_weight",
            "effective_sample_size",
        ]
    ]


def _local_gate_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    result = diagnostics.copy()
    required = {"family", "area_type", "area_code", "metric"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"local diagnostics are missing binding columns {missing}.")
    if result[list(required)].isna().any().any():
        raise ValueError("local diagnostics contain unclassified binding rows.")
    return result


def _local_diagnostics_registry(
    solve: UKRowwiseDoctrineSolve,
    problem: UKRowwiseLocalMatrix,
    *,
    national_registry: TargetRegistry | None = None,
) -> tuple[TargetRegistry, dict[str, str]]:
    targets = tuple(solve.calibration_result.problem.targets)
    expected = len(problem.target_frame) + (
        0 if national_registry is None else len(national_registry.specs)
    )
    if len(targets) != expected:
        raise RuntimeError(
            "candidate diagnostics registry is not aligned to the solve."
        )
    specs: list[TargetSpec] = []
    geography: dict[str, str] = {}
    for target, row in zip(
        targets[: len(problem.target_frame)],
        problem.target_frame.itertuples(index=False),
        strict=True,
    ):
        metric = str(row.metric)
        family = (
            str(row.family)
            if "family" in problem.target_frame.columns
            else local_target_census.family_for_metric(metric)
        )
        spec = TargetSpec(
            name=str(target.name),
            entity=str(target.entity),
            value=float(target.value),
            measure=f"rowwise_metric:{metric}",
            filter=f"rowwise_area:{row.area_code}",
            period=target.period,
            source=str(target.source),
            family=family,
            metadata={key: str(value) for key, value in target.metadata.items()},
        )
        specs.append(spec)
        geography[spec.to_target().row_name] = str(row.area_type)
    if national_registry is not None:
        specs.extend(national_registry.specs)
        for spec in national_registry.specs:
            level, _ = _spec_geography(spec)
            geography[spec.to_target().row_name] = level
    return TargetRegistry(specs, country="uk"), geography


#: Measure columns whose policyengine-uk formulas normalise by a population
#: total (a fixed national aggregate allocated by each household's share of
#: total weighted corporate wealth, or a term scaled by a weight sum). Under
#: ``--engine-blocks K`` the engine sees one clone block at a time, so each
#: block reproduces the whole aggregate and the column comes out K× (receipt
#: R15 in ``experiments/762-uk-rowwise-candidate-receipts.md``: corporate
#: land value ×15.000 at K=15). Evidence from a per-block run must not
#: adjudicate these rows; the release posture is single-block.
UK_BLOCK_SENSITIVE_MEASURE_COLUMNS = (
    "ons/corporate_land_value",
    "ons/land_value",
    "slc/student_loan_repayment/england",
)


def _run_local_gate_battery(
    *,
    frame: Any,
    support: pd.DataFrame,
    diagnostics: pd.DataFrame,
    report_path: Path,
    release_id: str,
    evaluated_on: date,
    enforce_only: tuple[str, ...] | None = None,
    release_candidate: bool = False,
) -> tuple[dict[str, object], GateResult]:
    manifest = uk_scoped_gate_manifest(
        UK_LOCAL_GATE_SCOPE,
        phases=("terminal",),
        policy_suffix=_LOCAL_GATE_POLICY_SUFFIX,
    )
    battery = GateBatteryRun(
        manifest,
        release_id=release_id,
        report_path=report_path,
        release_candidate=release_candidate,
        registry=UK_GATE_REGISTRY,
    )
    phase = battery.run_phase(
        "terminal",
        EvidenceContext(
            frame=frame,
            artifacts={
                "uk_area_support_summary": support,
                "local_target_diagnostics": diagnostics,
                # The run's own clock, not the wall clock: the same artifact
                # must reproduce the same exclusion verdicts.
                "exclusions_evaluated_on": evaluated_on,
            },
        ),
    )
    if enforce_only is None:
        try:
            battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
        except GateBatteryBlockedError:
            payload = battery.report_payload()
            finalize_uk_scoped_gate_report(
                payload,
                posture="local_candidate",
                scope_exclusions=uk_local_gate_scope_exclusions(),
                aggregate_admin_measurement=None,
            )
            atomic_write_json(report_path, payload)
            raise
    else:
        unknown = sorted(set(enforce_only) - set(UK_LOCAL_GATE_SCOPE))
        if unknown:
            raise ValueError(f"enforce_only names unknown local gates: {unknown}.")
        selected_blocking = [
            outcome
            for outcome in phase.blocking_outcomes(release_candidate=release_candidate)
            if outcome.entry.id in enforce_only
        ]
        if selected_blocking:
            payload = battery.report_payload()
            finalize_uk_scoped_gate_report(
                payload,
                posture="local_candidate",
                scope_exclusions=uk_local_gate_scope_exclusions(),
                aggregate_admin_measurement=None,
            )
            atomic_write_json(report_path, payload)
            failures = [
                failure
                for outcome in selected_blocking
                if outcome.result is not None
                for failure in outcome.result.failures
            ]
            raise GateBatteryBlockedError("terminal", failures, report_path)
    payload = battery.report_payload()
    finalize_uk_scoped_gate_report(
        payload,
        posture="local_candidate",
        scope_exclusions=uk_local_gate_scope_exclusions(),
        aggregate_admin_measurement=None,
    )
    atomic_write_json(report_path, payload)
    ladder = next(
        outcome
        for outcome in phase.outcomes
        if outcome.entry.id == "uk_local_geography_ladder_post_calibration"
    )
    if ladder.result is None or not ladder.result.passed:
        raise RuntimeError(
            "a non-passing local geography-ladder result escaped battery enforcement."
        )
    return payload, ladder.result


def _gate_failures_by_criticality(
    gate_report: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Split a persisted battery report's failure lines by criticality.

    Returns ``(release_blocking, diagnostic)``, each entry-prefixed like
    :class:`GateBatteryBlockedError`'s lines. Only ``failed`` and
    ``evidence_absent`` entries are failures; ``not_applicable`` and
    ``unreached`` entries are not.
    """

    blocking: list[str] = []
    diagnostic: list[str] = []
    gates = gate_report.get("gates", {})
    if not isinstance(gates, Mapping):
        return blocking, diagnostic
    for gate_id, payload in gates.items():
        if not isinstance(payload, Mapping):
            continue
        status = payload.get("status")
        if status not in {"failed", "evidence_absent"}:
            continue
        lines = [f"[{gate_id}] {line}" for line in payload.get("failures") or ()]
        if not lines:
            lines = [f"[{gate_id}] {payload.get('reason') or status}"]
        bucket = blocking if _is_release_blocking(payload) else diagnostic
        bucket.extend(lines)
    return blocking, diagnostic


def _is_release_blocking(payload: Mapping[str, Any]) -> bool:
    """Fail-closed criticality read.

    Only an entry that explicitly declares ``criticality: diagnostic`` is
    exempt from vetoing the release; a missing or unknown criticality is
    treated as release-blocking, so partial schema drift on one persisted
    entry cannot drop a failed gate out of both the blocking list and
    ``all_gates_passed``.
    """

    return payload.get("criticality") != "diagnostic"


def _release_verdict(
    *,
    sample_fraction: float,
    engine_blocks: int,
    release_blocking_gates_passed: bool,
) -> tuple[bool, dict[str, bool]]:
    """``releasable`` needs the full rung, a single-block engine resolution and
    every release-blocking gate passed.

    Per-block engine resolution mis-measures population-normalised formulas
    (each block reproduces a national aggregate: the ×K land-value artefact
    behind the #736 erratum), so a run resolved in more than one block is
    diagnostic-only whatever its gates say. The posture is written beside the
    verdict so a reader sees which leg failed.
    """

    posture = {
        "full_rung": float(sample_fraction) == 1.0,
        "single_block_engine": int(engine_blocks) == 1,
        "release_blocking_gates_passed": bool(release_blocking_gates_passed),
    }
    return all(posture.values()), posture


def _apply_gate_verdicts(
    state: AttemptState,
    report: Mapping[str, object],
    report_path: Path,
) -> None:
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(UK_LOCAL_GATE_SCOPE):
        raise RuntimeError("local gate report does not cover the declared scope.")
    receipt = local_artifact_reference(report_path, repository_hint=_REPOSITORY)
    state.gate_verdicts = {
        gate_id: {
            "verdict": str(payload["status"]),
            "receipt": f"{receipt}#/gates/{gate_id}",
        }
        for gate_id, payload in gates.items()
        if isinstance(payload, Mapping)
    }
    if set(state.gate_verdicts) != set(UK_LOCAL_GATE_SCOPE):
        raise RuntimeError("local gate verdicts are malformed.")


def _dry_run_plan(
    args: argparse.Namespace,
    *,
    clone: UKLadderRowwiseDatasetResult,
    problem: UKRowwiseLocalMatrix,
    source_year: int,
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    binding_adjudications: Mapping[str, Any],
    cross_grain: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "build_kind": "uk_rowwise_calibrated_candidate_plan",
        "dry_run": True,
        "candidate_scope": "adjudicated_partial",
        "bound_target_families": list(args._bound_families),
        "binding_adjudications": dict(binding_adjudications),
        "cross_grain": dict(cross_grain),
        "ladder_target_provenance": dict(target_provenance),
        "inputs": {
            "dataset": dict(input_artifact),
            "ladder": dict(ladder_artifact),
        },
        "sampling": dict(args._sampling_receipt),
        "survey_year": source_year,
        "calibration_year": (
            args._joint_inputs_receipt["calibration_year"]
            if args._joint_inputs_receipt is not None
            else source_year
        ),
        "rung_surface": {
            "rung": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
            "fraction": args.sample_fraction,
            "unreachable_check": "completed",
        },
        "releasable": args.sample_fraction == 1.0 and args.engine_blocks == 1,
        "parameters": _parameters(args, source_year=source_year),
        "shapes": {
            "person": list(clone.frame.table("person").shape),
            "benunit": list(clone.frame.table("benunit").shape),
            "household": list(clone.frame.table("household").shape),
            "local_matrix": list(problem.matrix.shape),
        },
        "target_count": int(len(problem.targets)),
        "gate": _gate_payload(clone.gate, phase="post_clone"),
    }


def _write_output_bundle(
    args: argparse.Namespace,
    *,
    candidate: UKLadderRowwiseDatasetResult,
    clone: UKLadderRowwiseDatasetResult,
    problem: UKRowwiseLocalMatrix,
    solve: UKRowwiseDoctrineSolve,
    local_diagnostics: pd.DataFrame,
    target_registry: TargetRegistry,
    target_geography_levels: Mapping[str, str],
    rotated_holdout: Mapping[str, object],
    support: pd.DataFrame,
    calibration_record: MassChangeRecord,
    source_year: int,
    output_paths: Mapping[str, Path],
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    cross_grain: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage the complete bundle, then publish atomically per file."""

    out_dir = output_paths["manifest"].parent
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{out_dir.name}.rowwise-candidate.",
            dir=out_dir.parent,
        )
    )
    try:
        staged = {key: staging_dir / path.name for key, path in output_paths.items()}
        print(
            f"staging candidate for {output_paths['dataset']}...",
            file=sys.stderr,
            flush=True,
        )
        write_uk_rowwise_dataset(candidate, staged["dataset"])
        solve.diagnostics.to_csv(staged["diagnostics"], index=False)
        support = support.copy()
        support["support_below_floor"] = (
            (support["assigned_households"] < 50)
            | (support["effective_sample_size"] < 50.0)
            | (support["nonzero_source_households"] < 50)
        )
        support.to_csv(staged["support"], index=False)
        staged["past_cap"].write_text(_json_text(dict(solve.past_cap_census or {})))
        local_registry = _local_output_registry(
            problem,
            period=(
                args._joint_inputs_receipt["calibration_year"]
                if args._joint_inputs_receipt is not None
                else source_year
            ),
        )
        local_registry.to_json(staged["local_registry"])
        write_uk_calibration_diagnostics(
            solve.calibration_result,
            staged["calibration_diagnostics"],
            solve.frame,
            target_geography_levels=target_geography_levels,
            target_registry=target_registry,
            local_area_support=support,
            rotated_holdout=rotated_holdout,
            build={
                "build_kind": "uk_rowwise_calibrated_candidate",
                "candidate_scope": "adjudicated_partial",
            },
        )
        calibration_diagnostics = json.loads(
            staged["calibration_diagnostics"].read_text(encoding="utf-8")
        )
        args._weakest_areas_by_fit = calibration_diagnostics["uk_diagnostics"][
            "weakest_areas_by_fit"
        ]

        outputs = {
            "dataset": _artifact_info(
                staged["dataset"],
                reported_path=output_paths["dataset"],
            ),
            "solve_diagnostics": _artifact_info(
                staged["diagnostics"],
                reported_path=output_paths["diagnostics"],
            ),
            "area_support_summary": _artifact_info(
                staged["support"],
                reported_path=output_paths["support"],
            ),
            "past_cap_census": _artifact_info(
                staged["past_cap"],
                reported_path=output_paths["past_cap"],
            ),
            "calibration_diagnostics": _artifact_info(
                staged["calibration_diagnostics"],
                reported_path=output_paths["calibration_diagnostics"],
            ),
            "local_gate_report": _artifact_info(output_paths["local_gates"]),
            "local_target_registry": _artifact_info(
                staged["local_registry"],
                reported_path=output_paths["local_registry"],
            ),
        }
        manifest = _manifest(
            args,
            candidate=candidate,
            clone=clone,
            problem=problem,
            solve=solve,
            support=support,
            calibration_record=calibration_record,
            source_year=source_year,
            input_artifact=input_artifact,
            ladder_artifact=ladder_artifact,
            target_provenance=target_provenance,
            cross_grain=cross_grain,
            calibration_diagnostics=calibration_diagnostics,
            outputs=outputs,
        )
        staged["manifest"].write_text(_json_text(manifest))
        _publish_staged_files(staged, output_paths)
        return manifest
    finally:
        shutil.rmtree(staging_dir)


def _manifest(
    args: argparse.Namespace,
    *,
    candidate: UKLadderRowwiseDatasetResult,
    clone: UKLadderRowwiseDatasetResult,
    problem: UKRowwiseLocalMatrix,
    solve: UKRowwiseDoctrineSolve,
    support: pd.DataFrame,
    calibration_record: MassChangeRecord,
    source_year: int,
    input_artifact: Mapping[str, Any],
    ladder_artifact: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    cross_grain: Mapping[str, Any],
    calibration_diagnostics: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    abs_errors = solve.diagnostics["abs_relative_error"].to_numpy(dtype=np.float64)
    old_total = float(calibration_record.old_total)
    new_total = float(calibration_record.new_total)
    past_cap = dict(solve.past_cap_census or {})
    gate_rows = args._gate_report.get("gates", {})
    # ``releasable`` follows the battery's own doctrine: release-blocking
    # entries decide, diagnostic entries (target_fit, weight_ratio, ...) are
    # reported but never veto. ``failing_gate_ids`` still lists every
    # non-passing entry of either criticality.
    release_gate_rows = {
        gate_id: payload
        for gate_id, payload in gate_rows.items()
        if isinstance(payload, Mapping) and _is_release_blocking(payload)
    }
    all_gates_passed = bool(release_gate_rows) and all(
        payload.get("status") == "passed" for payload in release_gate_rows.values()
    )
    releasable, release_posture = _release_verdict(
        sample_fraction=args.sample_fraction,
        engine_blocks=args.engine_blocks,
        release_blocking_gates_passed=all_gates_passed,
    )
    area_gate = gate_rows.get("uk_local_area_support", {})
    area_exclusion_details = (
        area_gate.get("details", {}) if isinstance(area_gate, Mapping) else {}
    )
    ladder_rows = int(
        problem.target_frame["target_name"]
        .astype(str)
        .str.startswith("external:census_households/households@")
        .sum()
    )
    local_rows = int(len(problem.target_frame) - ladder_rows)
    sample_stage = (
        []
        if args.sample_fraction == 1.0
        else [
            {
                "stage": "sample",
                "kind": uk_household_weight_kind(clone.frame).value,
            }
        ]
    )
    return {
        "schema_version": 2,
        "build_kind": "uk_rowwise_calibrated_candidate",
        "candidate_scope": "adjudicated_partial",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "bound_target_families": list(args._bound_families),
        "binding_adjudications": dict(solve.binding_adjudications),
        "cross_grain": dict(cross_grain),
        "ladder_target_provenance": dict(target_provenance),
        "parameters": _parameters(args, source_year=source_year),
        "inputs": {
            "dataset": dict(input_artifact),
            "ladder": dict(ladder_artifact),
        },
        "identity": {
            "spine": {
                **dict(input_artifact),
                "spine_provenance": dict(args._spine_provenance),
            },
            "ladder": {
                **dict(ladder_artifact),
                "layer_vintages": dict(target_provenance),
                "matches_local_area_crosswalk_pin": True,
            },
            **(
                {"ledger": args._joint_inputs_receipt["artifact"].provenance()}
                if args._joint_inputs_receipt is not None
                else {}
            ),
            "code": {"git_commit": _git_commit(), "git_dirty": _git_dirty()},
            "runtime": runtime_provenance(),
            "sampling": dict(args._sampling_receipt),
            "survey_year": source_year,
            "calibration_year": (
                args._joint_inputs_receipt["calibration_year"]
                if args._joint_inputs_receipt is not None
                else source_year
            ),
        },
        "sampling": dict(args._sampling_receipt),
        "rung_surface": {
            **dict(args._rung_surface),
            "rung": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
            "fraction": args.sample_fraction,
            "unreachable_check": "completed",
        },
        "outputs": dict(outputs),
        "geography": {
            "constituencies_assigned": int(
                support.loc[
                    support["geography_level"] == "constituency",
                    "area_code",
                ].nunique()
            ),
            "local_authorities_assigned": int(
                support.loc[
                    support["geography_level"] == "local_authority",
                    "area_code",
                ].nunique()
            ),
            "missing_geography_rows": 0,
            "ladder_gate": _gate_payload(candidate.gate, phase="post_calibration"),
        },
        "gate": _gate_payload(candidate.gate, phase="post_calibration"),
        "weights": {
            "household_weight_kind": uk_household_weight_kind(candidate.frame).value,
            "household_weight_kind_chain": [
                {
                    "stage": "staging",
                    "kind": uk_household_weight_kind(clone.frame).value,
                },
                *sample_stage,
                {
                    "stage": "ladder_clone",
                    "kind": uk_household_weight_kind(clone.frame).value,
                },
                {
                    "stage": "rowwise_calibration",
                    "kind": uk_household_weight_kind(candidate.frame).value,
                },
            ],
            "mass_log_records_before_calibration": len(clone.frame.mass_log),
            "mass_log_records": len(candidate.frame.mass_log),
            "calibration_mass_change": {
                "entity": str(calibration_record.entity),
                "old_total": old_total,
                "new_total": new_total,
                "relative_shift": (new_total - old_total) / old_total,
                "declared_factor": calibration_record.declared_factor,
                "reason": str(calibration_record.reason),
            },
            "abs_delta": abs(new_total - old_total),
            "declared_stretch_bound": float(UK_LOCAL_MAX_WEIGHT_RATIO),
            "realized_max_weight_ratio_vs_design": float(
                np.max(
                    np.divide(
                        np.asarray(solve.weights, dtype=np.float64),
                        np.asarray(clone.frame.weights_for("household").values),
                    )
                )
            ),
        },
        "solve": {
            "n_targets": int(len(problem.targets) + len(solve.national_diagnostics)),
            "n_targets_by_kind": {
                "local": local_rows,
                "ladder": ladder_rows,
                "national": int(len(solve.national_diagnostics)),
            },
            "n_households": int(problem.n_households),
            "initial_loss": float(solve.initial_loss),
            "final_loss": float(solve.final_loss),
            "max_abs_relative_error": float(abs_errors.max()),
            "median_abs_relative_error": float(np.median(abs_errors)),
            "n_nonzero": int(solve.n_nonzero),
            "past_cap": {key: int(past_cap[key]) for key in _PAST_CAP_COUNT_KEYS},
            "loss_shape": "capped_relative_error",
            "target_weight_rule": args.target_weight_rule,
            "target_weight_rule_override": dict(args._doctrine_override_receipt),
            "measure_resolution": dict(args._measure_resolution),
            "cross_grain": dict(cross_grain),
            "binding_adjudications": dict(solve.binding_adjudications),
            "area_support_exclusions": {
                "resource": "local_area_support_exclusions.json",
                "entries_stood_on": sorted(
                    area_exclusion_details.get("reviewed_exclusions", {})
                ),
                "stale": list(area_exclusion_details.get("stale_exclusions", [])),
                "unknown": list(area_exclusion_details.get("unknown_exclusions", [])),
            },
            "past_cap_by_kind": {
                "local": dict(solve.past_cap_census or {}),
                "national": dict(solve.national_past_cap_census or {}),
                "all": dict(solve.all_past_cap_census or {}),
            },
        },
        "diagnostics": {
            "schema_version": calibration_diagnostics["schema_version"],
            "target_registry": calibration_diagnostics["target_registry"],
            "weakest_families": calibration_diagnostics["uk_diagnostics"][
                "weakest_families"
            ],
            "weakest_areas_by_fit": calibration_diagnostics["uk_diagnostics"][
                "weakest_areas_by_fit"
            ],
            "rotated_holdout": calibration_diagnostics["uk_diagnostics"][
                "rotated_holdout"
            ],
        },
        "support": {
            "min_assigned_households": int(support["assigned_households"].min()),
            "min_nonzero_households": int(support["nonzero_households"].min()),
            "min_effective_sample_size": float(support["effective_sample_size"].min()),
            "by_geography_level": {
                str(level): {
                    "min_assigned_households": int(rows["assigned_households"].min()),
                    "min_nonzero_households": int(rows["nonzero_households"].min()),
                    "min_effective_sample_size": float(
                        rows["effective_sample_size"].min()
                    ),
                    "min_nonzero_source_households": int(
                        rows["nonzero_source_households"].min()
                    ),
                }
                for level, rows in support.groupby("geography_level", sort=True)
            },
        },
        "fit": {
            "local_by_family": _fit_by_family(solve.diagnostics),
            "national_by_family": _fit_by_family(solve.national_diagnostics),
            "weakest_families": sorted(
                [
                    *_fit_by_family(solve.diagnostics),
                    *_fit_by_family(solve.national_diagnostics),
                ],
                key=lambda row: (
                    -float(row["worst_abs_relative_error"]),
                    row["family"],
                ),
            )[:10],
            "weakest_areas_by_fit": dict(args._weakest_areas_by_fit),
            "support_limited_misses": dict(args._support_limited_misses),
            "rotated_holdout": dict(args._rotated_holdout),
        },
        "vintages": (
            _local_vintage_census(args._joint_inputs_receipt["local_registry"])
            if args._joint_inputs_receipt is not None
            else []
        ),
        "failing_gate_ids": sorted(
            gate_id
            for gate_id, payload in gate_rows.items()
            if not isinstance(payload, Mapping) or payload.get("status") != "passed"
        ),
        "releasable": releasable,
        "release_posture": release_posture,
        "ladder_household_uprating": dict(
            cross_grain.get("ladder_household_uprating")
            or {"applied": False, "reason": "no cross-grain receipt"}
        ),
        # The reviewed measure exclusions the national compile stood on
        # (name -> register record), so the narrowing is in the evidence.
        "measure_exclusions": {
            str(name): dict(record)
            for name, record in sorted(
                (
                    (getattr(args, "_joint_inputs_receipt", None) or {}).get(
                        "measure_exclusions"
                    )
                    or {}
                ).items()
            )
        },
        "blocked_at_f100": bool(getattr(args, "_blocked_failures", [])),
        "blocking_failures": list(getattr(args, "_blocked_failures", [])),
        "diagnostic_failures": list(getattr(args, "_diagnostic_failures", [])),
        "release_gate_failures_not_enforced": list(
            getattr(args, "_unenforced_release_failures", [])
        ),
    }


def _parameters(args: argparse.Namespace, *, source_year: int) -> dict[str, Any]:
    return {
        "n_clones": int(args.n_clones),
        "seed": int(args.seed),
        "source_year": source_year,
        "source_lineage_modulus": args.source_lineage_modulus,
        "sample_fraction": float(args.sample_fraction),
        "sample_seed": int(args.sample_seed),
        "engine_blocks": int(args.engine_blocks),
        "target_weight_rule": args.target_weight_rule,
        "release_candidate": bool(args.release_candidate),
        "skip_holdout": bool(args.skip_holdout),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "expected_constituency_vintage": str(args.expected_constituency_vintage),
        "doctrine": _doctrine_bounds(),
        "solve_options": {
            "conserve_mass": _CONSERVE_MASS,
            "target_records": _TARGET_RECORDS,
            "l0_lambda": _L0_LAMBDA,
            "budget_iters": _BUDGET_ITERS,
        },
    }


def _fit_by_family(diagnostics: pd.DataFrame) -> list[dict[str, object]]:
    if diagnostics.empty:
        return []
    rows = []
    for family, group in diagnostics.groupby("family", sort=True):
        errors = group["abs_relative_error"].to_numpy(dtype=np.float64)
        worst_index = int(np.argmax(errors))
        worst = group.iloc[worst_index]
        rows.append(
            {
                "family": str(family),
                "n_targets": len(group),
                "share_within_10pct": float((errors <= 0.10).mean()),
                "share_within_25pct": float((errors <= 0.25).mean()),
                "worst_abs_relative_error": float(errors[worst_index]),
                "worst_cell": str(
                    worst.get("target_name", worst.get("name", "unknown"))
                ),
            }
        )
    return rows


def _local_output_registry(
    problem: UKRowwiseLocalMatrix,
    *,
    period: int,
) -> TargetRegistry:
    specs = []
    for row in problem.target_frame.itertuples(index=False):
        payload = row._asdict()
        specs.append(
            TargetSpec(
                name=str(payload["target_name"]),
                entity="household",
                value=float(payload["value"]),
                measure=str(payload["metric"]),
                period=int(payload.get("period", period)),
                source=str(payload.get("source", "uk_rowwise_local_surface")),
                family=str(payload["family"]),
                metadata={
                    "area_type": str(payload["area_type"]),
                    "area_code": str(payload["area_code"]),
                    "metric": str(payload["metric"]),
                },
            )
        )
    return TargetRegistry(specs, country="uk")


def _doctrine_bounds() -> dict[str, Any]:
    return {
        "target_loss_cap": float(UK_LOCAL_TARGET_LOSS_CAP),
        "max_weight_ratio": float(UK_LOCAL_MAX_WEIGHT_RATIO),
        "scale_rule": UK_LOCAL_SOLVE_DOCTRINE.scale_rule,
        "target_weight_rule": UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule,
        "solve_epochs": int(UK_LOCAL_SOLVE_EPOCHS),
        "clone_count": int(UK_LOCAL_CLONE_COUNT),
    }


def _gate_payload(gate: GateResult, *, phase: str) -> dict[str, Any]:
    return {
        "name": str(gate.name),
        "passed": bool(gate.passed),
        "failures": list(gate.failures),
        "details": dict(gate.details),
        "phase": phase,
    }


def _validate_solve_result(
    solve: UKRowwiseDoctrineSolve,
    *,
    problem: UKRowwiseLocalMatrix,
) -> None:
    if solve.past_cap_census is None:
        raise RuntimeError(
            "doctrine solve returned no past-cap census; refusing candidate."
        )
    if len(solve.weights) != problem.n_households:
        raise RuntimeError(
            "doctrine solve returned a weight vector with the wrong length."
        )
    weights = np.asarray(solve.weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise RuntimeError(
            "doctrine solve returned non-finite or negative household weights."
        )
    if not np.isfinite([solve.initial_loss, solve.final_loss]).all():
        raise RuntimeError("doctrine solve returned a non-finite loss.")
    errors = solve.diagnostics["abs_relative_error"].to_numpy(dtype=np.float64)
    if len(errors) != len(problem.targets) or not np.isfinite(errors).all():
        raise RuntimeError(
            "doctrine solve returned incomplete or non-finite diagnostics."
        )
    missing_counts = sorted(set(_PAST_CAP_COUNT_KEYS) - set(solve.past_cap_census))
    if missing_counts:
        raise RuntimeError(
            f"past-cap census is missing count field(s): {missing_counts}."
        )


def _validate_support_summary(support: pd.DataFrame) -> None:
    required = {
        "geography_level",
        "area_code",
        "assigned_households",
        "nonzero_households",
        "nonzero_source_households",
        "effective_sample_size",
    }
    missing = sorted(required - set(support.columns))
    if missing or support.empty:
        raise RuntimeError(
            f"area support summary is empty or missing required columns: {missing}."
        )
    numeric = sorted(required - {"geography_level", "area_code"})
    values = support[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("area support summary contains invalid values.")


def _validate_cli_args(args: argparse.Namespace) -> None:
    ledger_values = (
        args.ledger_facts,
        args.ledger_facts_sha256,
        args.ledger_manifest_sha256,
    )
    if any(value is not None for value in ledger_values) and not all(
        value is not None for value in ledger_values
    ):
        raise ValueError(
            "--ledger-facts, --ledger-facts-sha256, and "
            "--ledger-manifest-sha256 must be supplied together."
        )
    if args.ledger_facts is not None and (
        args.input_sha256 is None or args.ladder_sha256 is None
    ):
        raise ValueError(
            "the joint registry path requires --input-sha256 and --ladder-sha256."
        )
    if args.release_candidate:
        required_release = {
            "--input-sha256": args.input_sha256,
            "--ladder-sha256": args.ladder_sha256,
            "--ledger-facts": args.ledger_facts,
            "--ledger-facts-sha256": args.ledger_facts_sha256,
            "--ledger-manifest-sha256": args.ledger_manifest_sha256,
        }
        missing_release = [
            name for name, value in required_release.items() if value is None
        ]
        if missing_release:
            raise ValueError(
                "--release-candidate requires pinned joint inputs: "
                + ", ".join(missing_release)
            )
        refused = []
        if args.target_weight_rule != UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule:
            refused.append("--target-weight-rule")
        if args.epochs != UK_LOCAL_SOLVE_EPOCHS:
            refused.append(f"--epochs != doctrine {UK_LOCAL_SOLVE_EPOCHS}")
        if args.n_clones != UK_LOCAL_CLONE_COUNT:
            refused.append(f"--n-clones != doctrine {UK_LOCAL_CLONE_COUNT}")
        if args.measure_exclusions is not None:
            refused.append("--measure-exclusions")
        if args.skip_holdout:
            refused.append("--skip-holdout")
        if args.engine_blocks > 1:
            refused.append("--engine-blocks > 1")
        if args.sample_fraction != 1.0:
            refused.append("--sample-fraction != 1.0")
        if refused:
            raise ValueError(
                "--release-candidate refuses non-release settings: "
                + ", ".join(refused)
            )
    if args.n_clones <= 0:
        raise ValueError("--n-clones must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.sample_fraction not in UK_SAMPLE_RUNG_TOKENS:
        raise ValueError(
            "--sample-fraction must be one of "
            f"{sorted(UK_SAMPLE_RUNG_TOKENS)}, got {args.sample_fraction!r}."
        )
    if args.sample_seed < 0:
        raise ValueError("--sample-seed must be non-negative.")
    if args.engine_blocks <= 0:
        raise ValueError("--engine-blocks must be positive.")
    if args.engine_blocks > 1 and args.engine_blocks != args.n_clones:
        raise ValueError("--engine-blocks greater than one must equal --n-clones.")
    if args.source_year is not None and args.source_year <= 0:
        raise ValueError("--source-year must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive and finite.")
    if not str(args.expected_constituency_vintage).strip():
        raise ValueError("--expected-constituency-vintage must be non-empty.")


def _output_paths(
    out_dir: Path,
    *,
    source_year: int,
    calibration_year: int,
) -> dict[str, Path]:
    dataset = out_dir / CANDIDATE_FILENAME_TEMPLATE.format(
        calibration_year=calibration_year
    )
    return {
        "dataset": dataset,
        "manifest": out_dir / MANIFEST_FILENAME,
        "diagnostics": out_dir / SOLVE_DIAGNOSTICS_FILENAME,
        "support": out_dir / AREA_SUPPORT_FILENAME,
        "past_cap": out_dir / PAST_CAP_FILENAME,
        "calibration_diagnostics": out_dir / CALIBRATION_DIAGNOSTICS_FILENAME,
        "local_gates": out_dir
        / LOCAL_GATE_REPORT_FILENAME_TEMPLATE.format(calibration_year=calibration_year),
        "local_registry": out_dir / LOCAL_REGISTRY_FILENAME,
    }


def _validate_output_paths(
    output_paths: Mapping[str, Path],
    *,
    input_h5: Path,
    ladder_path: Path,
) -> None:
    resolved = {name: path.resolve() for name, path in output_paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("candidate output paths must be distinct.")
    protected = {input_h5.resolve(), ladder_path.resolve()}
    collisions = sorted(str(path) for path in resolved.values() if path in protected)
    if collisions:
        raise ValueError(
            "candidate outputs must differ from --input-h5 and --ladder; "
            f"collision(s): {collisions}."
        )
    existing = sorted(str(path) for path in resolved.values() if path.exists())
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing candidate artifact(s): {existing}."
        )


def _publish_staged_files(
    staged: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> None:
    out_dir = output_paths["manifest"].parent
    created_out_dir = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    publish_order = (
        "dataset",
        "diagnostics",
        "support",
        "past_cap",
        "calibration_diagnostics",
        "local_registry",
        "manifest",
    )
    published: list[Path] = []
    succeeded = False
    try:
        for key in publish_order:
            destination = output_paths[key]
            if destination.exists():
                raise FileExistsError(
                    "candidate output appeared during publication; refusing "
                    f"to overwrite {destination}."
                )
            staged[key].replace(destination)
            published.append(destination)
        succeeded = True
    finally:
        if not succeeded:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            if created_out_dir:
                try:
                    out_dir.rmdir()
                except OSError:
                    pass


def _assert_artifacts_unchanged(
    *,
    input_h5: Path,
    input_artifact: Mapping[str, Any],
    ladder_path: Path,
    ladder_artifact: Mapping[str, Any],
) -> None:
    for label, path, before in (
        ("input H5", input_h5, input_artifact),
        ("ladder", ladder_path, ladder_artifact),
    ):
        after = _artifact_info(path)
        if after["sha256"] != before["sha256"] or after["bytes"] != before["bytes"]:
            raise RuntimeError(
                f"{label} changed during the candidate build; refusing to "
                "bind mixed source bytes."
            )


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} artifact not found: {resolved}.")
    return resolved


def _source_year(requested: int | None, *, time_period: str) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--source-year must be positive.")
        return requested
    prefix = str(time_period).strip()[:4]
    if len(prefix) != 4 or not prefix.isdigit():
        raise ValueError(
            "Could not infer source year from input H5 time_period; pass --source-year."
        )
    return int(prefix)


def _artifact_info(
    path: Path,
    *,
    reported_path: Path | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str((reported_path or path).resolve()),
        "sha256": digest.hexdigest(),
        "bytes": int(path.stat().st_size),
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_dirty() -> bool | None:
    """Measured, not asserted: tracked modifications in the working tree.

    ``None`` when git cannot answer (no repository), so a downstream
    assembler records the pin as unmeasured rather than clean.
    """

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _json_text(payload: Any) -> str:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
