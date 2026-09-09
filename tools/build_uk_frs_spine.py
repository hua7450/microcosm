"""Build the raw UK FRS spine Frame from pinned local tabs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from microcosm.build.country_spec import (
    GatesManifest,
    load_country_spec,
)
from microcosm.build.frame_sampling import (
    normalize_sampled_household_mass,
    sample_frame_households,
)
from microcosm.build.gate_battery import BlockingMode, EvidenceContext, GateBatteryRun
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
from microcosm.build.plan import StageRecord
from microcosm.build.staging_cli import (
    add_uk_staging_arguments,
    validate_uk_staging_arguments,
)
from microcosm.build.staging_v2 import (
    StagingTelemetryV2,
    disabled_staging_delivery,
)
from microcosm.build.uk_runtime.age_tail import UKAgeTailStageTransform
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.calibration_run import (
    UK_SPINE_GATE_SCOPE,
    uk_scoped_gate_manifest,
)
from microcosm.build.uk_runtime.cgt_imputation import uk_cgt_spine_stage_transform
from microcosm.build.uk_runtime.cgt_structure import (
    UKCGTBandDonorStageTransform,
    UKCGTIncidenceCloneStageTransform,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.etb_services import UKETBServicesStageTransform
from microcosm.build.uk_runtime.etb_vat import UKETBVATStageTransform
from microcosm.build.uk_runtime.frs_brma import UKFRSBRMAStageTransform
from microcosm.build.uk_runtime.frs_council_tax import UKFRSCouncilTaxStageTransform
from microcosm.build.uk_runtime.frs_disability import UKFRSDisabilityStageTransform
from microcosm.build.uk_runtime.frs_education import UKFRSEducationStageTransform
from microcosm.build.uk_runtime.frs_education_grants import (
    FRS_EDUCATION_GRANT_REWRITES,
    UKFRSEducationGrantSplitStageTransform,
)
from microcosm.build.uk_runtime.frs_employment import UKFRSEmploymentStageTransform
from microcosm.build.uk_runtime.frs_household_draws import (
    UKFRSHouseholdDrawsStageTransform,
)
from microcosm.build.uk_runtime.frs_legacy_proxies import (
    UKFRSLegacyProxiesStageTransform,
)
from microcosm.build.uk_runtime.frs_person_draws import UKFRSPersonDrawsStageTransform
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.frs_spine import (
    UKFRSSpineStageTransform,
    uk_frs_spine_seed_frame,
)
from microcosm.build.uk_runtime.frs_take_up import UKFRSTakeUpStageTransform
from microcosm.build.uk_runtime.graph import (
    UK_SPINE_EXCLUSIONS,
    uk_registry,
    uk_spine_graph,
)
from microcosm.build.uk_runtime.hmrc_replay import write_hmrc_replay_report
from microcosm.build.uk_runtime.lcfs_consumption import (
    UKLCFSConsumptionStageTransform,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_RUNG_TOKENS,
    UK_SAMPLE_SEED_DEFAULT,
)
from microcosm.build.uk_runtime.regional_uprating import (
    UKRegionalPropertyUpratingStageTransform,
)
from microcosm.build.uk_runtime.salary_sacrifice import UKSalarySacrificeStageTransform
from microcosm.build.uk_runtime.spi_spine import (
    UKFRSHMRCSpineLeavesStageTransform,
    UKSPIIncomeSpineStageTransform,
    UKSPISupportChannelStageTransform,
)
from microcosm.build.uk_runtime.student_loans import UKStudentLoansStageTransform
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from microcosm.build.uk_runtime.uc_capital_coherence import (
    UKUCCapitalCoherenceStageTransform,
)
from microcosm.build.uk_runtime.uc_deduction_attributes import (
    UKUCDeductionAttributesStageTransform,
)
from microcosm.build.uk_runtime.uc_reporter_redraw import (
    UKUCReporterRedrawStageTransform,
)
from microcosm.build.uk_runtime.was_wealth import UKWASWealthStageTransform
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from microcosm.graph import ContentStore, compile_graph, run_graph

_PIPELINE = "uk-frs-spine"
_REPOSITORY = Path(__file__).resolve().parents[1]
_RUNG_NAMED_EDGE_SIGNATURE = "The least populated classes in y have only 1 member"
_RUNG_ABORT_EXIT_CODE = 3
#: The last stage of the assembled checkpoint: everything through the base
#: FRS mapping and the stochastic draws. A name, not an index — a position
#: standing in for a key is correct only while two independently-maintained
#: orderings happen to agree (the uk-data#468 class).
UK_SPINE_ASSEMBLED_FINAL_STAGE = "frs_brma"


def _uk_spine_stage_names(spec) -> tuple[str, ...]:
    """Derive the runnable manifest stages from graph ownership edges."""

    if spec.sources is None:
        raise ValueError("UK country spec has no source stages.")
    declared = {
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    }
    compiled = compile_graph(uk_spine_graph(spec))
    ordered = tuple(node_id for node_id in compiled.order if node_id in declared)
    if set(ordered) != declared:
        raise ValueError(
            "UK spine graph and manifest stage roster disagree: "
            f"graph={list(ordered)!r}, manifest={sorted(declared)!r}."
        )
    return ordered


def _rung_sample_fraction(value: str) -> float:
    """CLI rung policy (#624) over the permissive library validator."""

    try:
        fraction = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"sample fraction must be a number; got {value!r}."
        ) from error
    if fraction not in UK_SAMPLE_RUNG_TOKENS:
        raise argparse.ArgumentTypeError(
            "sample fraction must be one of 0.01, 0.10, or 1.0 (the #624 rungs)."
        )
    return fraction


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic UK FRS spine from pinned raw tabs. Every "
            "stochastic stage draws identity-keyed from seeds declared in the "
            "manifest, so two runs from the same inputs are payload-identical."
        )
    )
    parser.add_argument(
        "--frs-raw-dir",
        type=Path,
        help="Directory containing the 14 licensed FRS 2024-25 tab files.",
    )
    parser.add_argument(
        "--spine-h5",
        type=Path,
        required=True,
        help="Output H5 path for the raw FRS spine Frame.",
    )
    parser.add_argument(
        "--spi-tab",
        type=Path,
        help="Pinned local SPI 2022-23 put2223uk.tab path.",
    )
    parser.add_argument(
        "--hmrc-ods",
        type=Path,
        help="Pinned local HMRC collated ODS path.",
    )
    parser.add_argument(
        "--cgt-ods",
        type=Path,
        help="Pinned local HMRC Capital Gains Tax Table 3 ODS path.",
    )
    parser.add_argument(
        "--synthetic-fixture-dir",
        type=Path,
        help=(
            "Data-only UK spine fixture source for local-only integration testing. "
            "Requires --smoke and --staging-local-only and cannot be combined with "
            "licensed input options."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Optional directory for a copy of the completed spine checkpoint.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=_rung_sample_fraction,
        default=1.0,
        help=(
            "Scale-ladder rung (#624): 0.01 smoke, 0.10 dev, or 1.0 full. "
            "Below 1.0 the raw FRS spine is sampled immediately after ingest, "
            "renormalized to full household mass, and treated as a receipt."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Mark the output as non-release and stop after spine and telemetry "
            "verification. Combine with --sample-fraction 0.01 for a small "
            "licensed-data run, or use the complete synthetic fixture in tests."
        ),
    )
    parser.add_argument(
        "--release-candidate",
        action="store_true",
        help=(
            "Evaluate the spine battery at release-candidate strictness: "
            "evidence_absent gaps block instead of being tolerated. Explicit "
            "by design - a full-scale developer build is not a release "
            "candidate unless the caller says so."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=UK_SAMPLE_SEED_DEFAULT,
        help=f"Raw FRS spine sampling seed (default: {UK_SAMPLE_SEED_DEFAULT}).",
    )
    parser.add_argument(
        "--was-tab",
        type=Path,
        help="Caller-supplied private WAS round-8 household tab for was_wealth.",
    )
    parser.add_argument(
        "--lcfs-hh-tab",
        type=Path,
        help="Caller-supplied private LCFS 2023-24 household tab for lcfs_consumption.",
    )
    parser.add_argument(
        "--lcfs-person-tab",
        type=Path,
        help="Caller-supplied private LCFS 2023-24 person tab for lcfs_consumption.",
    )
    parser.add_argument(
        "--etb-tab",
        type=Path,
        help="Caller-supplied private ETB 1977-2024 household tab for ETB stages.",
    )
    parser.add_argument(
        "--emit-nonzero-shares",
        type=Path,
        help="Optional JSON path for unweighted per-produced-column nonzero shares.",
    )
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help="Optional current Logbook chain head.",
    )
    add_uk_staging_arguments(parser)
    args = parser.parse_args(argv)
    if args.sample_seed < 0:
        parser.error("sample seed must be a non-negative integer.")
    if args.smoke and args.release_candidate:
        parser.error("non-release smoke builds refuse --release-candidate.")
    if _is_sampled(args) and args.checkpoint_dir is not None:
        parser.error(
            "sampled spine builds refuse --checkpoint-dir; sampled artifacts "
            "cannot be reused as full-scale checkpoints."
        )
    production_inputs = {
        "--frs-raw-dir": args.frs_raw_dir,
        "--spi-tab": args.spi_tab,
        "--hmrc-ods": args.hmrc_ods,
    }
    if args.synthetic_fixture_dir is None:
        missing = [flag for flag, value in production_inputs.items() if value is None]
        if missing:
            parser.error(f"production builds require {', '.join(missing)}.")
    else:
        supplied = [
            flag for flag, value in production_inputs.items() if value is not None
        ]
        supplied.extend(
            flag
            for flag, value in (
                ("--cgt-ods", args.cgt_ods),
                ("--was-tab", args.was_tab),
                ("--lcfs-hh-tab", args.lcfs_hh_tab),
                ("--lcfs-person-tab", args.lcfs_person_tab),
                ("--etb-tab", args.etb_tab),
            )
            if value is not None
        )
        if supplied:
            parser.error(
                "--synthetic-fixture-dir cannot be combined with licensed input "
                f"options: {', '.join(supplied)}."
            )
        if not args.smoke or not args.staging_local_only:
            parser.error(
                "--synthetic-fixture-dir requires --smoke and --staging-local-only."
            )
    validate_uk_staging_arguments(parser, args)
    return args


def _is_sampled(args: argparse.Namespace) -> bool:
    return args.sample_fraction != 1.0


def _sample_token(args: argparse.Namespace) -> str:
    return UK_SAMPLE_RUNG_TOKENS[args.sample_fraction]


def _validate_args(args: argparse.Namespace) -> None:
    if args.synthetic_fixture_dir is not None:
        if not args.synthetic_fixture_dir.is_dir():
            raise ValueError(
                "--synthetic-fixture-dir must be an existing directory: "
                f"{args.synthetic_fixture_dir}"
            )
        if not (args.synthetic_fixture_dir / "fixture.json").is_file():
            raise ValueError(
                "--synthetic-fixture-dir must contain fixture.json: "
                f"{args.synthetic_fixture_dir}"
            )
    elif args.frs_raw_dir is None or not args.frs_raw_dir.is_dir():
        raise ValueError(
            f"--frs-raw-dir must be an existing directory: {args.frs_raw_dir}"
        )
    if args.spine_h5.suffix != ".h5":
        raise ValueError("--spine-h5 must end with '.h5'.")
    if args.synthetic_fixture_dir is None:
        if args.spi_tab is None or not args.spi_tab.is_file():
            raise ValueError(f"--spi-tab must be an existing file: {args.spi_tab}")
        if args.spi_tab.name != "put2223uk.tab":
            raise ValueError("--spi-tab must name put2223uk.tab.")
        if args.hmrc_ods is None or not args.hmrc_ods.is_file():
            raise ValueError(f"--hmrc-ods must be an existing file: {args.hmrc_ods}")
        if args.hmrc_ods.suffix.lower() != ".ods":
            raise ValueError("--hmrc-ods must end with '.ods'.")
    if args.cgt_ods is not None:
        if not args.cgt_ods.is_file():
            raise ValueError(f"--cgt-ods must be an existing file: {args.cgt_ods}")
        if args.cgt_ods.suffix.lower() != ".ods":
            raise ValueError("--cgt-ods must end with '.ods'.")
    paths = {
        "spine_h5": args.spine_h5,
        "build_sidecar": args.spine_h5.with_suffix(".build.json"),
        "hmrc_replay_sidecar": args.spine_h5.with_suffix(".hmrc_replay.json"),
    }
    if args.emit_nonzero_shares is not None:
        paths["emit_nonzero_shares"] = args.emit_nonzero_shares
    resolved: dict[Path, str] = {}
    for label, path in paths.items():
        target = Path(path).expanduser().resolve()
        other = resolved.get(target)
        if other is not None:
            raise ValueError(f"{label} path collides with {other}: {target}.")
        resolved[target] = label


def _synthetic_fixture_evidence(source: Path | None) -> dict[str, object] | None:
    """Bind a non-release integration run to every file in its fixture."""

    if source is None:
        return None
    root = source.resolve()
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.resolve().relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    descriptor = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    return {
        "schema_version": descriptor.get("schema_version"),
        "file_count": len(files),
        "digest": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _synthetic_graph_sources(source: Path) -> dict[str, Path]:
    """Resolve every split graph source to one reviewed fixture input."""

    root = source.resolve()
    descriptor = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    inputs = descriptor.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Synthetic fixture inputs must be an object.")
    names = {
        "was": "was",
        "lcfs_household": "lcfs_household",
        "lcfs_person": "lcfs_person",
        "etb": "etb",
        "spi": "spi_donor",
        "hmrc_income": "hmrc_income_targets",
        "hmrc_cgt": "cgt_distribution",
    }
    resolved = {"frs": root}
    for role, name in names.items():
        relative = inputs.get(name)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Synthetic fixture inputs.{name} must be a path.")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Synthetic fixture inputs.{name} escapes the fixture directory."
            ) from exc
        if not path.exists():
            raise ValueError(f"Synthetic fixture input {name!r} does not exist.")
        resolved[role] = path
    return resolved


def _synthetic_fixture_input(source: Path, name: str) -> Path:
    """Resolve one named fixture input without allowing path traversal."""

    root = source.resolve()
    descriptor = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    inputs = descriptor.get("inputs")
    relative = inputs.get(name) if isinstance(inputs, dict) else None
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Synthetic fixture inputs.{name} must be a path.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Synthetic fixture inputs.{name} escapes the fixture directory."
        ) from exc
    if not path.exists():
        raise ValueError(f"Synthetic fixture input {name!r} does not exist.")
    return path


def _artifact_pins(stages) -> dict[str, dict[str, object]]:
    pins = {}
    for stage in stages:
        for artifact in stage.artifacts:
            key = artifact.get("table", artifact.get("filename"))
            if key is None:
                continue
            key = str(key)
            pin = {
                "locator": str(artifact["locator"]),
                "sha256": str(artifact["sha256"]),
                "size_bytes": int(artifact["size_bytes"]),
            }
            if key in pins and pins[key] != pin:
                raise ValueError(
                    f"UK source artifact {key!r} has inconsistent pins across stages."
                )
            pins[key] = pin
    return dict(sorted(pins.items()))


def _stage_artifact_pins(stage) -> dict[str, dict[str, object]]:
    return {
        str(artifact.get("table", artifact.get("filename"))): {
            "locator": str(artifact["locator"]),
            "sha256": str(artifact["sha256"]),
            "size_bytes": int(artifact["size_bytes"]),
        }
        for artifact in stage.artifacts
        if "table" in artifact or "filename" in artifact
    }


def _resource_pins(stages, spec) -> dict[str, str]:
    """Country-package resources the selected stages declare as inputs.

    Non-tab artifacts reference committed resources by filename; their bytes
    are hashed by load_country_spec, so the pin is the spec's recorded sha.
    """

    pins: dict[str, str] = {}
    for stage in stages:
        for artifact in stage.artifacts:
            if "resource" not in artifact:
                continue
            resource = str(artifact["resource"])
            sha256 = spec.resource_hashes.get(resource)
            if sha256 is None:
                raise ValueError(
                    f"stage {stage.stage!r} declares resource artifact "
                    f"{resource!r} which is not a declared country-package "
                    "resource."
                )
            pins[resource] = str(sha256)
    return dict(sorted(pins.items()))


def _input_artifact_pins(stages) -> dict[str, dict[str, object]]:
    """Caller-supplied private input artifacts, pinned by role.

    Non-table, non-resource artifacts (the SPI donor tab and the HMRC ODS)
    carry their own sha256/size pins in the manifest. Binding them here puts
    the pins in the build sidecar and the Logbook input-pins digest, so two
    runs with different high-impact source inputs can never share build-side
    provenance (adversarial-review finding on #717).
    """

    pins: dict[str, dict[str, object]] = {}
    for stage in stages:
        for artifact in stage.artifacts:
            if "table" in artifact or "resource" in artifact:
                continue
            if "sha256" not in artifact:
                continue
            role = str(artifact.get("role") or artifact.get("filename") or "")
            if not role:
                raise ValueError(
                    f"stage {stage.stage!r} declares a pinned input artifact "
                    "without a role or filename."
                )
            pin = {
                "filename": str(
                    artifact.get("filename") or artifact.get("locator") or ""
                ),
                "kind": str(artifact.get("kind", "")),
                "sha256": str(artifact["sha256"]),
                "size_bytes": int(artifact["size_bytes"]),
            }
            if role in pins and pins[role] != pin:
                raise ValueError(
                    f"input artifact role {role!r} has inconsistent pins across stages."
                )
            pins[role] = pin
    return dict(sorted(pins.items()))


def _role_pins(pins: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        table: {
            "sha256": str(pin["sha256"]),
            "size_bytes": int(pin["size_bytes"]),
        }
        for table, pin in pins.items()
    }


def _entity_row_counts(frame) -> dict[str, int]:
    return {entity: int(len(frame.table(entity))) for entity in frame.entities}


def _rules_engine() -> PolicyEngineUKEngine:
    try:
        import policyengine_uk  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "build_uk_frs_spine requires the microcosm-build 'uk' extra "
            "(policyengine-uk). Run: uv sync --all-packages --extra uk"
        ) from exc
    return PolicyEngineUKEngine()


def _rules_engine_provenance() -> dict[str, str]:
    try:
        version = metadata.version("policyengine-uk")
    except metadata.PackageNotFoundError:
        return {"package": "policyengine-uk", "version": "unavailable"}
    return {"package": "policyengine-uk", "version": version}


def _declared_seeds(stages) -> dict[str, dict[str, int]]:
    declared: dict[str, dict[str, int]] = {}
    for stage in stages:
        stage_seeds: dict[str, int] = {}
        for operation in stage.operations:
            output = operation.parameters.get("output")
            seed = operation.parameters.get("seed")
            if seed is None:
                seed = operation.parameters.get("seed_base")
            if isinstance(output, str) and isinstance(seed, int):
                stage_seeds[output] = seed
            elif isinstance(seed, int):
                if operation.kind == "stack_zero_weight_donors":
                    stage_seeds["stack_zero_weight_donors"] = seed
                elif operation.kind == "strict_read_private_table":
                    stage_seeds["donor_bootstrap"] = seed
                elif operation.kind == "fit_weighted_qrf_stage1":
                    stage_seeds["stage1"] = seed
                elif operation.kind == "fit_weighted_qrf_stage2":
                    stage_seeds["stage2"] = seed
                elif operation.kind == "bridge_donor_column_via_qrf":
                    stage_seeds["bridge_donor_column_via_qrf"] = seed
                elif operation.kind == "assign_binary_from_rate":
                    target = operation.parameters.get("target")
                    if isinstance(target, str):
                        stage_seeds[target] = seed
                    else:
                        stage_seeds["assign_binary_from_rate"] = seed
                elif operation.kind == "fit_weighted_qrf_chain":
                    stage_seeds[stage.stage] = seed
                elif operation.kind == "fit_weighted_qrf":
                    stage_seeds[stage.stage] = seed
                elif operation.kind == "draw_capital_gains_prior_from_banded_quantiles":
                    stage_seeds[str(operation.parameters["salt"])] = seed
                elif operation.kind == "stack_band_donor_households":
                    stage_seeds["stack_band_donor_households"] = seed
                elif operation.kind == "within_band_draws":
                    stage_seeds["within_band_draws"] = seed
                elif operation.kind == "convert_donors_to_target_stock":
                    stage_seeds[str(operation.parameters["salt"])] = seed
                elif operation.kind == "top_up_to_stock":
                    stage_seeds[str(operation.parameters["salt"])] = seed
        if stage_seeds:
            declared[stage.stage] = stage_seeds
    return declared


def _result_evidence(result: object) -> object:
    if isinstance(result, dict):
        return result
    evidence = getattr(result, "evidence", None)
    if callable(evidence):
        return evidence()
    return None


def _collect_stage_evidence(
    *,
    stage_names: Sequence[str],
    implementations: Mapping[str, object],
) -> dict[str, object]:
    evidence_by_stage: dict[str, object] = {}
    for stage_name in stage_names:
        implementation = implementations.get(stage_name)
        if implementation is None:
            continue
        metadata = None
        metadata_hook = getattr(implementation, "checkpoint_metadata", None)
        if callable(metadata_hook):
            metadata = dict(metadata_hook())
            evidence = metadata.get("evidence", metadata)
        else:
            evidence = _result_evidence(getattr(implementation, "last_result", None))
        if evidence is not None:
            evidence_by_stage[stage_name] = evidence
    return evidence_by_stage


def _collect_fit_weight_records(
    *,
    stage_names: Sequence[str],
    implementations: Mapping[str, object],
) -> dict[str, list[dict[str, str]]]:
    """Persist each fitting stage's resolved weight kinds into the sidecar.

    The terminal weights audit (``uk_weights_audit``) consumes
    :class:`FitWeightRecord` evidence that only exists on live stage
    objects; the release-cut certification producer runs in a later
    process, so the sidecar carries the records across the run boundary.
    Duck-typed like ``stage_evidence``: every stage whose transform exposes
    ``fit_weight_records`` contributes, in stage order. A fitting stage
    whose records are missing, unreadable, or empty records an empty list —
    the audit binding fails an empty record set, so the gap stays visible
    rather than vanishing from the sidecar.
    """

    records_by_stage: dict[str, list[dict[str, str]]] = {}
    for stage_name in stage_names:
        implementation = implementations.get(stage_name)
        if implementation is None:
            continue
        # Detect the hook without evaluating it: a raising property must
        # count as a fitting stage with unreadable records, not vanish.
        exposes_records = getattr(
            type(implementation), "fit_weight_records", None
        ) is not None or "fit_weight_records" in getattr(implementation, "__dict__", {})
        if not exposes_records:
            continue
        try:
            records = tuple(implementation.fit_weight_records or ())
        except Exception:  # noqa: BLE001 - unreadable records fail the audit
            records_by_stage[stage_name] = []
            continue
        records_by_stage[stage_name] = [
            {
                "fit_name": str(record.fit_name),
                "weight_kind": str(record.weight_kind),
            }
            for record in records
        ]
    return records_by_stage


def _build_sidecar(
    *,
    frame,
    stages,
    records,
    artifact_pins,
    resource_pins: dict[str, str],
    input_artifact_pins: dict[str, dict[str, object]],
    hmrc_replay: dict[str, object],
    stochastic_contract_sha256: str,
    frs_vintage: str,
    sampling: dict[str, object] | None,
    non_release: bool = False,
    release_posture: str = "development",
    synthetic_fixture: Mapping[str, object] | None = None,
    staging_delivery: Mapping[str, object] | None = None,
    spine_gate_report: dict[str, object] | None = None,
) -> dict[str, object]:
    household_weight = frame.weights_for("household")
    return {
        "schema_version": 2,
        "pipeline": _PIPELINE,
        "uk_frame_content_identity": uk_frame_content_identity(frame),
        "stages": [stage.stage for stage in stages],
        "time_period": str(frame.metadata["time_period"]),
        "household_weight_kind": uk_household_weight_kind(frame).value,
        "household_weight_total": float(household_weight.values.sum()),
        "entity_row_counts": _entity_row_counts(frame),
        "artifact_pins": artifact_pins,
        "resource_pins": resource_pins,
        "input_artifact_pins": input_artifact_pins,
        "hmrc_replay": hmrc_replay,
        "stage_artifact_pins": {
            stage.stage: _stage_artifact_pins(stage) for stage in stages
        },
        "stage_records": [
            {
                "stage": record.stage,
                "produced": list(record.produced),
                "nonzero_share": dict(record.nonzero_share),
                "seconds": record.seconds,
            }
            for record in records
        ],
        "operations": {
            stage.stage: [operation.kind for operation in stage.operations]
            for stage in stages
        },
        "declared_seeds": _declared_seeds(stages),
        "source_vintages": {"frs": frs_vintage},
        "sampling": sampling,
        "non_release": non_release,
        "release_posture": release_posture,
        "synthetic_fixture": (
            None if synthetic_fixture is None else dict(synthetic_fixture)
        ),
        "staging_delivery": dict(staging_delivery or {}),
        "spine_gate_report": spine_gate_report,
        "stochastic_contract_sha256": stochastic_contract_sha256,
        "rules_engine": _rules_engine_provenance(),
    }


def _mark_non_release_h5(path: Path, *, build_id: str) -> None:
    """Persist machine-readable refusal evidence on a bounded smoke H5."""

    import h5py

    with h5py.File(path, mode="r+") as file:
        file.attrs["populace_non_release"] = True
        file.attrs["populace_release_posture"] = "smoke"
        file.attrs["populace_smoke_build_id"] = build_id


def _nonzero_shares(frame, columns: list[str]) -> dict[str, float]:
    shares: dict[str, float] = {}
    for column in columns:
        for entity in frame.entities:
            table = frame.table(entity)
            if column not in table.columns:
                continue
            values = table[column]
            if values.dtype == object:
                shares[column] = float(values.astype(str).ne("").mean())
            else:
                shares[column] = float((values != 0).mean())
            break
    return shares


def _series_nonzero_share(values) -> float:
    if values.dtype == object or str(values.dtype).startswith("string"):
        return float(values.fillna("").astype(str).ne("").mean())
    return float((values != 0).mean())


def _graph_stage_records(
    *,
    manifest,
    store: ContentStore,
    stages,
    frame,
) -> tuple[StageRecord, ...]:
    """Project immediate node artifacts onto the legacy record schema.

    Entity ids and memberships are executor-carried context, not owned cells,
    so the root node exposes no artifact for them although ``frs_spine``
    declares them as outputs.  Their share is read from the final population
    instead, which is what the legacy plan recorded (identity columns are
    never zero, so the value is 1.0 on every vintage).
    """

    structural = _structural_columns(frame)
    records: list[StageRecord] = []
    for stage in stages:
        output_node = (
            f"{stage.stage}.owned"
            if f"{stage.stage}.owned" in manifest.nodes
            else stage.stage
        )
        output_receipt = manifest.nodes[output_node]
        shares: dict[str, float] = {}
        for column in stage.outputs:
            matches = [
                (coordinate, key)
                for coordinate, key in output_receipt.artifacts.items()
                if coordinate[1] == column
            ]
            if not matches and column in structural:
                shares[column] = _nonzero_shares(frame, [column])[column]
                continue
            if len(matches) != 1:
                raise RuntimeError(
                    f"graph stage {stage.stage!r} exposes {len(matches)} artifacts "
                    f"for declared output {column!r}."
                )
            shares[column] = _series_nonzero_share(store.load_column(matches[0][1]))
        execution_node = "create_uk_frs" if stage.stage == "frs_spine" else stage.stage
        records.append(
            StageRecord(
                stage=stage.stage,
                produced=stage.outputs,
                donor_survey=stage.survey,
                nonzero_share=shares,
                seconds=manifest.nodes[execution_node].wall_time,
            )
        )
    return tuple(records)


def _structural_columns(frame) -> frozenset[str]:
    """Entity id and membership columns the executor carries outside owned cells."""

    schema = frame.schema
    columns = {schema.entity_id_column(entity) for entity in frame.entities}
    columns.update(schema.membership_column(group) for group in schema.group_entities)
    return frozenset(columns)


def _new_build_id(timestamp: datetime) -> str:
    return f"uk-frs-spine-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"


def _record_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    code_pin: str,
    disposition: str,
    predecessor: str | None,
    rung: str,
    spool_dir: Path,
) -> Path:
    return record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung=rung,
        seed=None,
        code_pin=code_pin,
        disposition=disposition,
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _sample_spine_frame(
    frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[object, dict[str, object] | None]:
    if fraction == 1.0:
        return frame, None
    household_weight = frame.weights_for("household")
    pre_households = int(len(frame.table("household")))
    sampled, receipt = sample_frame_households(
        frame,
        fraction=fraction,
        seed=seed,
        source_name="UK FRS spine",
    )
    normalized, factor = normalize_sampled_household_mass(
        sampled,
        target_mass=float(household_weight.total),
        source_name="UK FRS spine",
    )
    return normalized, {
        "fraction": float(fraction),
        "seed": int(seed),
        "rung_token": UK_SAMPLE_RUNG_TOKENS[fraction],
        "pre_household_count": pre_households,
        "post_household_count": int(len(normalized.table("household"))),
        "normalization_factor": float(factor),
        "receipt": dict(receipt),
    }


class _SampledGraphRootTransform:
    """CREATE-stage adapter applying the declared sampling rung at ingest."""

    def __init__(
        self,
        transform,
        *,
        fraction: float,
        seed: int,
    ) -> None:
        self.transform = transform
        self.fraction = fraction
        self.seed = seed
        self.sampling: dict[str, object] | None = None

    def _sample(self, assembled):
        sampled, self.sampling = _sample_spine_frame(
            assembled,
            fraction=self.fraction,
            seed=self.seed,
        )
        # Graph populations use row positions as their internal alignment
        # index. Frame sampling preserves source DataFrame indexes by design,
        # so normalize those indexes at this adapter boundary.
        tables = {
            entity: sampled.table(entity).reset_index(drop=True)
            for entity in sampled.entities
        }
        tables.update(
            {name: sampled.link(name).reset_index(drop=True) for name in sampled.links}
        )
        return Frame(
            tables,
            sampled.schema,
            {
                entity: sampled.weights_for(entity)
                for entity in sampled.weighted_entities
            },
            sampled.strata.reset_index(drop=True),
            mass_log=sampled.mass_log,
            metadata=sampled.metadata,
        )

    def effective_fraction(self) -> float:
        """Return the configured input sampling fraction."""

        return float(self.fraction)

    def __call__(self, frame):
        return self._sample(self.transform(frame))

    def run_with_sources(self, frame, sources):
        runner = getattr(self.transform, "run_with_sources", None)
        assembled = (
            runner(frame, sources) if callable(runner) else self.transform(frame)
        )
        return self._sample(assembled)

    def checkpoint_metadata(self) -> dict[str, object]:
        hook = getattr(self.transform, "checkpoint_metadata", None)
        if not callable(hook):
            raise RuntimeError("FRS root transform exposes no checkpoint metadata.")
        return dict(hook())


class _ObservedGraphTransform:
    """Report aggregate lifecycle data around one real graph transform."""

    def __init__(
        self,
        transform,
        *,
        stage_id: str,
        produced_column_count: int,
        telemetry: StagingTelemetryV2,
    ):
        self.transform = transform
        self.stage_id = stage_id
        self.produced_column_count = produced_column_count
        self.telemetry = telemetry

    def _run(self, runner, frame, *args):
        started = time.perf_counter()
        self.telemetry.stage(
            self.stage_id,
            event_status="started",
            entity_row_counts=_entity_row_counts(frame),
            produced_column_count=0,
        )
        result = runner(frame, *args)
        self.telemetry.stage(
            self.stage_id,
            event_status="completed",
            elapsed_seconds=time.perf_counter() - started,
            entity_row_counts=_entity_row_counts(result),
            produced_column_count=self.produced_column_count,
        )
        return result

    def __call__(self, frame):
        return self._run(self.transform, frame)

    def run_with_sources(self, frame, sources):
        runner = getattr(self.transform, "run_with_sources", None)
        if callable(runner):
            return self._run(runner, frame, sources)
        return self._run(self.transform, frame)

    def __getattr__(self, name: str):
        return getattr(self.transform, name)


class _GraphSourceTransform:
    """Build a file-reading stage from only the node's declared source paths."""

    def __init__(self, factory) -> None:
        self.factory = factory
        self.transform = None

    def run_with_sources(self, frame, sources):
        self.transform = self.factory(sources)
        result = self.transform(frame)
        if hasattr(self.transform, "fit_weight_records"):
            self.fit_weight_records = self.transform.fit_weight_records
        return result

    def __getattr__(self, name: str):
        transform = self.__dict__.get("transform")
        if transform is None:
            raise AttributeError(name)
        return getattr(transform, name)


def _run_plan_with_spine_sampling(
    plan,
    *,
    sample_fraction: float,
    sample_seed: int,
    spine_battery: GateBatteryRun | None = None,
    stage_evidence_provider=None,
    gate_artifacts: Mapping[str, object] | None = None,
) -> tuple[object, tuple[object, ...], dict[str, object] | None]:
    if not plan.stages or plan.stages[0].name != "frs_spine":
        frame, records = plan.run(uk_frs_spine_seed_frame())
        return frame, records, None

    from microcosm.build.plan import StagePlan

    spine_frame, spine_records = StagePlan(plan.stages[:1]).run(
        uk_frs_spine_seed_frame()
    )
    spine_frame, sampling = _sample_spine_frame(
        spine_frame,
        fraction=sample_fraction,
        seed=sample_seed,
    )
    if len(plan.stages) == 1:
        return spine_frame, spine_records, sampling
    names = tuple(stage.name for stage in plan.stages)
    if UK_SPINE_ASSEMBLED_FINAL_STAGE in names:
        assembled_end = names.index(UK_SPINE_ASSEMBLED_FINAL_STAGE) + 1
    elif spine_battery is not None:
        raise RuntimeError(
            "spine battery is armed but the declared assembled-boundary stage "
            f"{UK_SPINE_ASSEMBLED_FINAL_STAGE!r} is not in the plan; a stage "
            "plan change must move the boundary declaration with it."
        )
    else:
        assembled_end = len(plan.stages)
    frame, assembled_records = StagePlan(plan.stages[1:assembled_end]).run(spine_frame)
    # Each boundary offers only the stages that have actually run: asking a
    # later stage for checkpoint evidence would (correctly) raise, and the
    # first licensed battery run did exactly that at the assembled boundary.
    executed = tuple(stage.name for stage in plan.stages[:assembled_end])
    if spine_battery is not None:
        _run_spine_gate_phase(
            spine_battery,
            "assembled",
            frame=frame,
            stage_evidence=(
                stage_evidence_provider(executed)
                if stage_evidence_provider is not None
                else {}
            ),
            gate_artifacts=gate_artifacts,
        )
    if assembled_end == len(plan.stages):
        return frame, (*spine_records, *assembled_records), sampling
    frame, tail_records = StagePlan(plan.stages[assembled_end:]).run(frame)
    executed = tuple(stage.name for stage in plan.stages)
    if spine_battery is not None:
        _run_spine_gate_phase(
            spine_battery,
            "transferred",
            frame=frame,
            stage_evidence=(
                stage_evidence_provider(executed)
                if stage_evidence_provider is not None
                else {}
            ),
            gate_artifacts=gate_artifacts,
        )
    return frame, (*spine_records, *assembled_records, *tail_records), sampling


def _run_spine_gate_phase(
    battery: GateBatteryRun,
    phase: str,
    *,
    frame,
    stage_evidence: Mapping[str, object],
    gate_artifacts: Mapping[str, object] | None = None,
) -> None:
    artifacts: dict[str, object] = {"stage_evidence": dict(stage_evidence)}
    # The enum-domain gate resolves its domain from the live rules engine,
    # exactly as the national terminal battery supplied it.
    artifacts.update(dict(gate_artifacts or {}))
    battery.run_phase(
        phase,
        EvidenceContext(frame=frame, artifacts=artifacts),
    )
    battery.enforce(phase, mode=BlockingMode.BLOCKS_ARTIFACT)


def _spine_gate_report_path(spine_h5: Path) -> Path:
    return spine_h5.with_suffix(".spine_gates.json")


def _spine_gate_manifest_from_spec(spec) -> GatesManifest | None:
    """The spine build's scoped battery manifest, from the shared helper.

    A spec without a gates block leaves the battery unarmed (``None``),
    exactly as before; when armed, the filtering runs through the one
    scope-filtering implementation every scoped producer shares. The
    driver passes the spec it already loaded, which is also the hermetic
    tests' stub point. Digests are identical to the previous local copy
    because entries, phases, and the policy suffix are unchanged.
    """

    source = getattr(spec, "gates", None)
    if source is None:
        return None
    return uk_scoped_gate_manifest(
        UK_SPINE_GATE_SCOPE,
        phases=("assembled", "transferred"),
        policy_suffix="spine_build_scope",
        source=source,
    )


def _rung_abort_receipt(
    args: argparse.Namespace,
    *,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "uk_frs_spine_rung_abort_receipt",
        "build_kind": "uk_frs_spine",
        "sampling": {
            "sample_fraction": float(args.sample_fraction),
            "sample_seed": int(args.sample_seed),
            "rung_token": _sample_token(args),
        },
        "named_edge": "spine_split_singleton_class",
        "stage": "frs_spine",
        "error": str(error),
        "disposition": "aborted_with_receipt",
        "remedy": (
            "Re-roll --sample-seed; accepted dev-scale statistical edge. "
            "The computation is never altered to avoid it."
        ),
    }


def _exception_chain_contains(error: BaseException, text: str) -> bool:
    """Match a named rung edge through graph execution wrappers."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if text in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _create_staging_telemetry(
    args: argparse.Namespace, *, state: AttemptState
) -> StagingTelemetryV2 | None:
    if args.no_staging:
        return None
    local_dir = args.staging_dir or args.spine_h5.parent / "staging"
    local_only = args.staging_local_only
    return StagingTelemetryV2(
        run_id=args.staging_run_id or state.build_id,
        country_code="GB",
        operation_id="uk_frs_spine",
        pipeline_id=_PIPELINE,
        pipeline_version=metadata.version("microcosm-build"),
        candidate_id=args.staging_candidate_id or state.build_id,
        local_dir=local_dir,
        run_kind="smoke" if args.smoke else "spine",
        delivery_mode="local_only" if local_only else "local_and_remote",
        repo_id=None if local_only else args.staging_repo_id,
        path_prefix=args.staging_prefix,
        upload_interval_seconds=args.staging_upload_interval_seconds,
    )


def _telemetry_sample(
    args: argparse.Namespace, sampling: Mapping[str, object] | None
) -> dict[str, object] | None:
    if args.sample_fraction == 1.0:
        return {"mode": "full"}
    return None


def _staging_delivery(
    args: argparse.Namespace, telemetry: StagingTelemetryV2 | None
) -> dict[str, object]:
    if telemetry is None:
        return disabled_staging_delivery("--no-staging")
    return telemetry.delivery_summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rung = UK_SAMPLE_RUNG_TOKENS[args.sample_fraction]
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    predecessor = resolve_predecessor(args.logbook_prev_row_digest)
    digest = preflight_digest(_PIPELINE)
    state = AttemptState(
        build_id=_new_build_id(started_ts),
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-spine-receipt",
            }
        },
    )
    code_pin = "unresolved-local-git-code-pin"
    spool_dir = args.spine_h5.parent / "logbook-spool"
    telemetry: StagingTelemetryV2 | None = None
    try:
        _validate_args(args)
        # A crash between the H5 write and the sidecar writes must never
        # leave a stale sidecar beside a fresh H5 (adversarial-review
        # finding on #717): clear every output up front, and treat the
        # build sidecar - written last, binding the replay hash - as the
        # marker that the bundle is complete.
        stale_outputs = [
            args.spine_h5,
            args.spine_h5.with_suffix(".build.json"),
            args.spine_h5.with_suffix(".hmrc_replay.json"),
            _spine_gate_report_path(args.spine_h5),
            args.spine_h5.with_suffix(".rung_abort.json"),
        ]
        if args.emit_nonzero_shares is not None:
            stale_outputs.append(args.emit_nonzero_shares)
        for stale in stale_outputs:
            stale.unlink(missing_ok=True)
        telemetry = _create_staging_telemetry(args, state=state)
        if telemetry is not None:
            initial_sample = _telemetry_sample(args, None)
            if initial_sample is not None:
                telemetry.set_sample(initial_sample)
            telemetry.stage(
                "configuration",
                event_status="completed",
                smoke=args.smoke,
                sample_mode=("fraction" if args.sample_fraction != 1.0 else "full"),
            )
        code_pin = git_code_pin(_REPOSITORY)
        append_phase(state, "configured")
        spec = load_country_spec("uk")
        if spec.sources is None:
            raise ValueError("UK country spec has no source stages.")
        stages_by_name = spec.sources.stage_map()
        graph = uk_spine_graph(
            spec,
            source_mode="split",
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
        )
        compiled_graph = compile_graph(graph)
        stage_names = _uk_spine_stage_names(spec)
        if (
            args.synthetic_fixture_dir is None
            and "hmrc_cgt_gains_spine" in stage_names
            and args.cgt_ods is None
        ):
            raise ValueError(
                "--cgt-ods is required when hmrc_cgt_gains_spine is scheduled."
            )
        if (
            args.synthetic_fixture_dir is None
            and "was_wealth" in stage_names
            and args.was_tab is None
        ):
            raise ValueError(
                "--was-tab is required when the was_wealth stage is scheduled."
            )
        if args.synthetic_fixture_dir is None and "lcfs_consumption" in stage_names:
            missing_lcfs = [
                flag
                for flag, value in (
                    ("--lcfs-hh-tab", args.lcfs_hh_tab),
                    ("--lcfs-person-tab", args.lcfs_person_tab),
                    ("--was-tab", args.was_tab),
                )
                if value is None
            ]
            if missing_lcfs:
                raise ValueError(
                    "lcfs_consumption requires caller-supplied private inputs: "
                    f"{', '.join(missing_lcfs)}."
                )
        if (
            args.synthetic_fixture_dir is None
            and ("etb_vat" in stage_names or "etb_services" in stage_names)
            and args.etb_tab is None
        ):
            raise ValueError(
                "--etb-tab is required when etb_vat or etb_services is scheduled."
            )
        stages = [stages_by_name[name] for name in stage_names]
        artifact_pins = _artifact_pins(stages)
        resource_pins = _resource_pins(stages, spec)
        input_artifact_pins = _input_artifact_pins(stages)
        overlapping_pin_roles = set(artifact_pins) & set(input_artifact_pins)
        if overlapping_pin_roles:
            raise ValueError(
                "input artifact roles collide with FRS tab names: "
                f"{sorted(overlapping_pin_roles)}."
            )
        state.input_pins_digest = role_pins_digest(
            _role_pins({**artifact_pins, **input_artifact_pins})
        )
        synthetic_fixture = _synthetic_fixture_evidence(args.synthetic_fixture_dir)
        run_config = {
            "pipeline": _PIPELINE,
            "stages": list(stage_names),
            "artifact_pins_digest": state.input_pins_digest,
            "spine_h5": str(args.spine_h5),
            "synthetic_fixture": synthetic_fixture,
        }
        state.identity_digest = hashlib.sha256(
            canonical_json_bytes(run_config)
        ).hexdigest()
        append_phase(state, "inputs_pinned")
        if telemetry is not None:
            telemetry.stage(
                "input_verification",
                event_status="completed",
                stage_count=len(stage_names),
                input_artifact_count=len(artifact_pins) + len(input_artifact_pins),
            )
        engine = _rules_engine()
        stochastic_contract = load_uk_take_up_contract()
        frs_release = load_uk_frs_release()
        hmrc_spine_transform = _GraphSourceTransform(
            lambda sources: UKSPIIncomeSpineStageTransform(
                sources["spi"],
                sources["hmrc_income"],
                stage=stages_by_name["hmrc_spi_income_spine"],
                sampled_rung=_is_sampled(args),
            )
        )
        implementations = {
            "frs_spine": _GraphSourceTransform(
                lambda sources: UKFRSSpineStageTransform(
                    sources["frs"],
                    stage=stages_by_name["frs_spine"],
                )
            ),
            "frs_employment": _GraphSourceTransform(
                lambda sources: UKFRSEmploymentStageTransform(
                    sources["frs"],
                    stage=stages_by_name["frs_employment"],
                )
            ),
            "frs_council_tax": _GraphSourceTransform(
                lambda sources: UKFRSCouncilTaxStageTransform(
                    sources["frs"],
                    stage=stages_by_name["frs_council_tax"],
                )
            ),
            "frs_disability": UKFRSDisabilityStageTransform(
                stage=stages_by_name["frs_disability"],
            ),
            "frs_education": _GraphSourceTransform(
                lambda sources: UKFRSEducationStageTransform(
                    sources["frs"],
                    stage=stages_by_name["frs_education"],
                )
            ),
            "frs_legacy_proxies": _GraphSourceTransform(
                lambda sources: UKFRSLegacyProxiesStageTransform(
                    sources["frs"],
                    stage=stages_by_name["frs_legacy_proxies"],
                    engine=engine,
                )
            ),
            "frs_education_grant_split": (
                UKFRSEducationGrantSplitStageTransform(
                    stage=stages_by_name["frs_education_grant_split"],
                    engine=engine,
                )
            ),
            "frs_take_up": UKFRSTakeUpStageTransform(
                contract=stochastic_contract,
                stage=stages_by_name["frs_take_up"],
            ),
            "frs_person_draws": UKFRSPersonDrawsStageTransform(
                contract=stochastic_contract,
                stage=stages_by_name["frs_person_draws"],
            ),
            "frs_household_draws": UKFRSHouseholdDrawsStageTransform(
                contract=stochastic_contract,
                stage=stages_by_name["frs_household_draws"],
            ),
            "frs_brma": UKFRSBRMAStageTransform(
                stage=stages_by_name["frs_brma"],
                engine=engine,
            ),
        }
        if "was_wealth" in stage_names:
            implementations["was_wealth"] = _GraphSourceTransform(
                lambda sources: UKWASWealthStageTransform(
                    stage=stages_by_name["was_wealth"],
                    engine=engine,
                    was_tab_path=sources["was"],
                )
            )
        if "regional_property_uprating" in stage_names:
            implementations["regional_property_uprating"] = (
                UKRegionalPropertyUpratingStageTransform(
                    stage=stages_by_name["regional_property_uprating"],
                )
            )
        if "lcfs_consumption" in stage_names:
            implementations["lcfs_consumption"] = _GraphSourceTransform(
                lambda sources: UKLCFSConsumptionStageTransform(
                    stage=stages_by_name["lcfs_consumption"],
                    engine=engine,
                    lcfs_hh_tab_path=sources["lcfs_household"],
                    lcfs_person_tab_path=sources["lcfs_person"],
                    was_tab_path=sources["was"],
                )
            )
        if "etb_vat" in stage_names:
            implementations["etb_vat"] = _GraphSourceTransform(
                lambda sources: UKETBVATStageTransform(
                    stage=stages_by_name["etb_vat"],
                    engine=engine,
                    etb_tab_path=sources["etb"],
                )
            )
        if "etb_services" in stage_names:
            implementations["etb_services"] = _GraphSourceTransform(
                lambda sources: UKETBServicesStageTransform(
                    stage=stages_by_name["etb_services"],
                    engine=engine,
                    etb_tab_path=sources["etb"],
                )
            )
        implementations["frs_hmrc_spine_leaves"] = _GraphSourceTransform(
            lambda sources: UKFRSHMRCSpineLeavesStageTransform(
                sources["frs"],
                stage=stages_by_name["frs_hmrc_spine_leaves"],
                sampled_rung=_is_sampled(args),
            )
        )
        implementations["spi_support_channel"] = UKSPISupportChannelStageTransform(
            stage=stages_by_name["spi_support_channel"],
            sample_fraction=args.sample_fraction,
        )
        implementations["hmrc_spi_income_spine"] = hmrc_spine_transform
        if "uc_reporter_redraw" in stage_names:
            implementations["uc_reporter_redraw"] = UKUCReporterRedrawStageTransform(
                stage=stages_by_name["uc_reporter_redraw"],
                engine=engine,
            )
        if "uc_capital_coherence" in stage_names:
            implementations["uc_capital_coherence"] = (
                UKUCCapitalCoherenceStageTransform(
                    stage=stages_by_name["uc_capital_coherence"]
                )
            )
        if "uc_deduction_attributes" in stage_names:
            implementations["uc_deduction_attributes"] = (
                UKUCDeductionAttributesStageTransform(
                    stage=stages_by_name["uc_deduction_attributes"]
                )
            )
        if "cgt_incidence_clone" in stage_names:
            implementations["cgt_incidence_clone"] = UKCGTIncidenceCloneStageTransform(
                stage=stages_by_name["cgt_incidence_clone"]
            )
        if "cgt_band_donors" in stage_names:
            implementations["cgt_band_donors"] = UKCGTBandDonorStageTransform(
                stage=stages_by_name["cgt_band_donors"]
            )
        if "hmrc_cgt_gains_spine" in stage_names:
            implementations["hmrc_cgt_gains_spine"] = _GraphSourceTransform(
                lambda sources: uk_cgt_spine_stage_transform(
                    stages_by_name["hmrc_cgt_gains_spine"],
                    sources["hmrc_cgt"],
                )
            )
        if "salary_sacrifice" in stage_names:
            implementations["salary_sacrifice"] = UKSalarySacrificeStageTransform(
                stage=stages_by_name["salary_sacrifice"]
            )
        if "student_loans" in stage_names:
            implementations["student_loans"] = UKStudentLoansStageTransform(
                stage=stages_by_name["student_loans"],
                calibration_year=frs_release.calibration_year,
            )
        if "age_tail" in stage_names:
            implementations["age_tail"] = UKAgeTailStageTransform(
                stage=stages_by_name["age_tail"]
            )
        if args.synthetic_fixture_dir is not None:
            from microcosm.build.uk_runtime.graph_kernels import (
                fixture_stage_plan_inputs,
            )

            fixture_stages, fixture_implementations = fixture_stage_plan_inputs(
                args.synthetic_fixture_dir
            )
            fixture_stage_names = tuple(stage.stage for stage in fixture_stages)
            if fixture_stage_names != tuple(stage_names):
                raise ValueError(
                    "Synthetic fixture stage order differs from the current UK spine: "
                    f"fixture={fixture_stage_names!r}, current={tuple(stage_names)!r}."
                )
            implementations = dict(fixture_implementations)
            fixture_by_name = {stage.stage: stage for stage in fixture_stages}
            implementations["frs_hmrc_spine_leaves"] = (
                UKFRSHMRCSpineLeavesStageTransform(
                    _synthetic_fixture_input(args.synthetic_fixture_dir, "frs_raw"),
                    stage=fixture_by_name["frs_hmrc_spine_leaves"],
                    sampled_rung=True,
                )
            )
            hmrc_spine_transform = implementations["hmrc_spi_income_spine"]
        sampled_root = _SampledGraphRootTransform(
            implementations["frs_spine"],
            fraction=args.sample_fraction,
            seed=args.sample_seed,
        )
        implementations["frs_spine"] = sampled_root
        if telemetry is not None:
            implementations = {
                stage_id: _ObservedGraphTransform(
                    transform,
                    stage_id=stage_id,
                    produced_column_count=len(stages_by_name[stage_id].outputs),
                    telemetry=telemetry,
                )
                for stage_id, transform in implementations.items()
            }
        spine_gate_path = _spine_gate_report_path(args.spine_h5)
        spine_gate_manifest = _spine_gate_manifest_from_spec(spec)
        spine_battery = (
            GateBatteryRun(
                spine_gate_manifest,
                release_id=state.build_id,
                report_path=spine_gate_path,
                release_candidate=args.release_candidate,
                registry=UK_GATE_REGISTRY,
            )
            if spine_gate_manifest is not None
            else None
        )
        checkpoint_root = (
            args.checkpoint_dir
            if args.checkpoint_dir is not None
            else args.spine_h5.parent / f".{args.spine_h5.stem}.checkpoints"
        )
        if args.synthetic_fixture_dir is not None:
            graph_sources = _synthetic_graph_sources(args.synthetic_fixture_dir)
        else:
            graph_sources = {"frs": args.frs_raw_dir}
            if "was_wealth" in stage_names or "lcfs_consumption" in stage_names:
                graph_sources["was"] = args.was_tab
            if "lcfs_consumption" in stage_names:
                graph_sources["lcfs_household"] = args.lcfs_hh_tab
                graph_sources["lcfs_person"] = args.lcfs_person_tab
            if "etb_vat" in stage_names or "etb_services" in stage_names:
                graph_sources["etb"] = args.etb_tab
            if "hmrc_spi_income_spine" in stage_names:
                graph_sources["spi"] = args.spi_tab
                graph_sources["hmrc_income"] = args.hmrc_ods
            if "hmrc_cgt_gains_spine" in stage_names:
                graph_sources["hmrc_cgt"] = args.cgt_ods
        graph_store = ContentStore(checkpoint_root / "node-graph")
        graph_manifest = run_graph(
            compiled_graph,
            sources=graph_sources,
            store=graph_store,
            kernels=uk_registry(implementations, graph=graph),
            resume="forbid",
            decisions=(),
        )
        final_version = compiled_graph.versions[compiled_graph.order[-1]]
        frame = graph_manifest.population(final_version)
        records = _graph_stage_records(
            manifest=graph_manifest,
            store=graph_store,
            stages=stages,
            frame=frame,
        )
        sampling = sampled_root.sampling
        if telemetry is not None:
            sample = _telemetry_sample(args, sampling)
            if sample is not None:
                telemetry.set_sample(sample)
            telemetry.stage(
                "sampling",
                event_status="completed",
                realized_household_rows=(
                    len(frame.table("household"))
                    if sampling is None
                    else sampling.get(
                        "realized_household_rows",
                        sampling.get("post_household_count"),
                    )
                ),
            )
            telemetry.stage("validation", event_status="started")
        if spine_battery is not None:
            if UK_SPINE_ASSEMBLED_FINAL_STAGE not in stage_names:
                raise RuntimeError(
                    "spine battery is armed but the graph has no assembled "
                    f"boundary stage {UK_SPINE_ASSEMBLED_FINAL_STAGE!r}."
                )
            assembled_index = stage_names.index(UK_SPINE_ASSEMBLED_FINAL_STAGE) + 1
            assembled_version = compiled_graph.versions[UK_SPINE_ASSEMBLED_FINAL_STAGE]
            _run_spine_gate_phase(
                spine_battery,
                "assembled",
                frame=graph_manifest.population(assembled_version),
                stage_evidence=_collect_stage_evidence(
                    stage_names=stage_names[:assembled_index],
                    implementations=implementations,
                ),
                gate_artifacts={"rules_engine": engine},
            )
            if assembled_index < len(stage_names):
                _run_spine_gate_phase(
                    spine_battery,
                    "transferred",
                    frame=frame,
                    stage_evidence=_collect_stage_evidence(
                        stage_names=stage_names,
                        implementations=implementations,
                    ),
                    gate_artifacts={"rules_engine": engine},
                )
        if spine_battery is not None:
            append_phase(state, "spine_gates_evaluated")
        if telemetry is not None:
            telemetry.stage(
                "validation",
                event_status="completed",
                entity_row_counts=_entity_row_counts(frame),
            )
        append_phase(state, "spine_built")
        if telemetry is not None:
            telemetry.stage("spine_h5_creation", event_status="started")
        output = write_uk_national_frame(frame, args.spine_h5)
        if args.smoke:
            _mark_non_release_h5(output, build_id=state.build_id)
        if telemetry is not None:
            telemetry.stage(
                "spine_h5_creation",
                event_status="completed",
                size_bytes=output.stat().st_size,
            )
        append_phase(state, "spine_written")
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            write_uk_national_frame(frame, args.checkpoint_dir / "frs_spine.h5")
            append_phase(state, "checkpoint_written")
        sidecar_path = output.with_suffix(".build.json")
        replay_sidecar_path = output.with_suffix(".hmrc_replay.json")
        if hmrc_spine_transform.last_result is None:
            raise RuntimeError("HMRC SPI spine stage did not record replay evidence.")
        write_hmrc_replay_report(
            hmrc_spine_transform.last_result.replay_report,
            replay_sidecar_path,
        )
        append_phase(state, "hmrc_replay_sidecar_written")
        replay_bytes = replay_sidecar_path.read_bytes()
        replay_binding = {
            "filename": replay_sidecar_path.name,
            "report_kind": str(json.loads(replay_bytes).get("report_kind", "")),
            "sha256": hashlib.sha256(replay_bytes).hexdigest(),
        }
        if telemetry is not None:
            telemetry.stage("sidecar_creation", event_status="started")
        sidecar = _build_sidecar(
            frame=frame,
            stages=stages,
            records=records,
            artifact_pins=artifact_pins,
            resource_pins=resource_pins,
            input_artifact_pins=input_artifact_pins,
            hmrc_replay=replay_binding,
            stochastic_contract_sha256=stochastic_contract.resource_sha256,
            frs_vintage=frs_release.vintage,
            sampling=sampling,
            non_release=args.smoke,
            release_posture=(
                "non_release_smoke"
                if args.smoke
                else "release_candidate"
                if args.release_candidate
                else "development"
            ),
            synthetic_fixture=synthetic_fixture,
            staging_delivery=_staging_delivery(args, telemetry),
            spine_gate_report=(
                {
                    "path": str(spine_gate_path),
                    "sha256": hashlib.sha256(spine_gate_path.read_bytes()).hexdigest(),
                }
                if spine_gate_path.is_file()
                else None
            ),
        )
        stage_evidence = _collect_stage_evidence(
            stage_names=stage_names,
            implementations=implementations,
        )
        if stage_evidence:
            sidecar["stage_evidence"] = stage_evidence
        fit_weight_records = _collect_fit_weight_records(
            stage_names=stage_names,
            implementations=implementations,
        )
        if fit_weight_records:
            sidecar["fit_weight_records"] = fit_weight_records
        atomic_write_json(sidecar_path, sidecar)
        if telemetry is not None:
            telemetry.stage(
                "sidecar_creation",
                event_status="completed",
                size_bytes=sidecar_path.stat().st_size,
            )
        append_phase(state, "build_sidecar_written")
        if args.emit_nonzero_shares is not None:
            final_columns = list(
                dict.fromkeys(
                    [column for record in records for column in record.produced]
                    + list(FRS_EDUCATION_GRANT_REWRITES)
                )
            )
            atomic_write_json(
                args.emit_nonzero_shares,
                {
                    "stages": {
                        record.stage: dict(record.nonzero_share) for record in records
                    },
                    "final": _nonzero_shares(frame, final_columns),
                },
            )
            append_phase(state, "nonzero_shares_written")
        if telemetry is not None:
            telemetry.complete(
                message=(
                    "Non-release smoke verification completed."
                    if args.smoke
                    else "UK spine staging run completed."
                )
            )
            if args.staging_read_back:
                telemetry.verify_remote()
            telemetry.validate_local_bundle()
            sidecar["staging_delivery"] = telemetry.delivery_summary
            atomic_write_json(sidecar_path, sidecar)
        state.artifact_location = local_artifact_reference(
            output,
            repository_hint=_REPOSITORY,
        )
        state.gate_verdicts = {
            "pipeline": {
                "verdict": "passed",
                "receipt": local_artifact_reference(
                    sidecar_path, repository_hint=_REPOSITORY
                ),
            }
        }
        if spine_gate_path.is_file():
            gate_payload = json.loads(spine_gate_path.read_text(encoding="utf-8"))
            for gate_id, payload in gate_payload.get("gates", {}).items():
                state.gate_verdicts[str(gate_id)] = {
                    "verdict": str(payload.get("status")),
                    "receipt": (
                        f"{local_artifact_reference(spine_gate_path, repository_hint=_REPOSITORY)}"
                        f"#/gates/{gate_id}"
                    ),
                }
        spool_path = _record_attempt(
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            code_pin=code_pin,
            disposition="iterating",
            predecessor=predecessor,
            rung=rung,
            spool_dir=spool_dir,
        )
        print(f"Wrote FRS spine H5: {output}", file=sys.stderr)
        print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
        return 0
    except Exception as error:
        if telemetry is not None and telemetry.status == "running":
            try:
                telemetry.fail(error)
                telemetry.validate_local_bundle()
            except Exception:
                pass
        if _is_sampled(args) and _exception_chain_contains(
            error, _RUNG_NAMED_EDGE_SIGNATURE
        ):
            rung_abort_path = args.spine_h5.with_suffix(".rung_abort.json")
            receipt = _rung_abort_receipt(args, error=error)
            atomic_write_json(rung_abort_path, receipt)
            state.gate_verdicts = {
                "uk_frs_spine_rung_abort": {
                    "verdict": "aborted",
                    "receipt": (
                        f"{local_artifact_reference(rung_abort_path, repository_hint=_REPOSITORY)}"
                        "#/named_edge"
                    ),
                }
            }
            append_phase(state, "rung_aborted")
            _record_attempt(
                state=state,
                started_at=started_at,
                started_ts=started_ts,
                code_pin=code_pin,
                disposition="discarded",
                predecessor=predecessor,
                rung=rung,
                spool_dir=spool_dir,
            )
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return _RUNG_ABORT_EXIT_CODE
        try:
            receipt_path = write_error_receipt(
                error_receipt_path(args.spine_h5.parent, build_id=state.build_id),
                state=state,
                pipeline=_PIPELINE,
                error=error,
            )
            apply_error_verdict(
                state,
                local_artifact_reference(receipt_path, repository_hint=_REPOSITORY),
            )
            _record_attempt(
                state=state,
                started_at=started_at,
                started_ts=started_ts,
                code_pin=code_pin,
                disposition="failed",
                predecessor=predecessor,
                rung=rung,
                spool_dir=spool_dir,
            )
        except Exception:
            pass
        print(f"UK FRS spine build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
