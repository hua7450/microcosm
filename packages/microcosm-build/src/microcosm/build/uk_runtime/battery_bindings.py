"""UK bindings for the country-agnostic gate battery (microcosm#611).

The UK's gates are declared in ``uk/gates.json`` under the shared
country-neutral vocabulary; this module supplies the other half of the
contract — the :class:`~microcosm.build.gate_battery.GateBinding`
implementations that know where each gate's evidence lives in an
:class:`~microcosm.build.gate_battery.EvidenceContext` and how declared
spec parameters become gate arguments. The gate *comparisons* stay in
:mod:`microcosm.build.uk_runtime.terminal_gates` and
:mod:`microcosm.build.uk_runtime.weighted_integrity`, unchanged: a
binding adapts evidence, it never re-implements a verdict, so the
migration onto the battery executor is behaviour-preserving by
construction (the differential test pins it).

Two legacy evaluators mint UK-flavored result names
(``uk_release_input_coverage``, ``qrf_tail_concentration``); their
bindings re-mint the verdict under the declared neutral name, because the
battery executor fails a gate closed when the returned name disagrees
with the declared one. Any *other* unexpected name passes through
untouched so that check keeps biting.

The evidence surface handed to the legacy gate modules (the three entity
tables plus period, weight-kind, and mass-log metadata) lives in
:mod:`microcosm.build.uk_runtime.national_frame` as the single copy shared
with the legacy report evaluator; the Frame-typed gates (input-mass parity,
QRF tail concentration) read the frame directly and skip it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from importlib.resources import files
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gate_battery import (
    DEFAULT_REGISTRY,
    EvidenceContext,
    GateBinding,
)
from microcosm.build.gates import (
    GateResult,
    aggregate_admin_gate,
    area_support_gate,
    column_implication_gate,
    enum_domain_gate,
    ledger_compile_parity_gate,
    nonnegative_columns_gate,
    per_family_fit_gate,
    support_gate,
    target_surface_gate,
    weights_audit_gate,
)
from microcosm.build.uk_runtime.frs_take_up import uk_take_up_signal_gate
from microcosm.build.uk_runtime.geography_ladder import uk_geography_ladder_gate
from microcosm.build.uk_runtime.ledger_targets import (
    LOCAL_REGISTRY_PARITY_FIXTURE_RESOURCE,
    align_uk_local_registry_parity_fixture,
)
from microcosm.build.uk_runtime.local_targets import (
    load_uk_local_geography_contract,
    metric_names,
)
from microcosm.build.uk_runtime.national_frame import _uk_gate_surface
from microcosm.build.uk_runtime.release_input_coverage import (
    assert_uk_release_input_coverage_build_stages,
    assert_uk_release_input_coverage_manifest_current,
    uk_release_input_coverage_gate,
)
from microcosm.build.uk_runtime.source_runtime import UK_NONNEGATIVE_OUTPUTS_BY_STAGE
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.build.uk_runtime.terminal_gates import (
    UKZeroWeightStratumDeclaration,
    _household_weights,
    _missing_fit_weight_evidence_gate,
    uk_default_degenerate_reviewed_exclusions,
    uk_default_target_fit_reviewed_exclusions,
    uk_degenerate_release_surface_gate,
    uk_export_surface_gate,
    uk_target_fit_gate,
    uk_target_surface_gate,
    uk_weight_ess_gate,
    uk_weight_ratio_gate,
    uk_zero_weight_strata_gate,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
    UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    UK_INPUT_MASS_REFERENCE_REGISTRY,
    UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
    UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE,
    UKInputMassParityPolicy,
    UKQRFTailConcentrationPolicy,
    UKReviewedExclusion,
    _input_mass_reference_evidence_sha256,
    coerce_input_mass_reference_registry,
    coerce_reviewed_exclusions,
    load_uk_reviewed_exclusion_register,
    uk_default_input_mass_reviewed_exclusions,
    uk_default_qrf_tail_reviewed_exclusions,
    uk_input_mass_parity_gate,
    uk_input_mass_totals,
    uk_qrf_tail_concentration_columns,
    uk_qrf_tail_concentration_gate,
)
from microcosm.calibrate.registry import TargetSpec

__all__ = [
    "UK_GATE_REGISTRY",
    "UKGateBinding",
]


@dataclass(frozen=True)
class UKGateBinding:
    """A :class:`GateBinding` over a UK runtime evaluator.

    Attributes:
        name: The declared country-neutral gate name the battery expects
            back.
        evaluator: Runs the legacy UK gate over ``(context, parameters)``.
            Declared parameters it does not consume pass through to the
            gate call, so an unknown parameter raises and fails closed —
            the same discipline as the battery's ``FunctionBinding``.
        parameter_keys: The spec parameter keys the evaluator can route —
            checked against every declared entry before any gate runs
            (``validate_gate_parameters``), so a typo'd parameter is
            refused at arm time instead of surfacing as a mid-battery
            evaluator error. Fail-closed empty by default.
        artifact_keys: Context artifact keys the gate needs; missing keys
            resolve to ``evidence_absent`` before evaluation.
        artifact_selector: Optional parameter-dependent override of
            ``artifact_keys`` for bindings whose evidence artifact follows a
            declared parameter.
        needs_frame: Whether the gate reads the phase frame.
        frame_predicate: Optional parameter-dependent override of
            ``needs_frame`` for bindings that serve more than one declared
            entry (the coverage gate runs frameless at preflight).
        legacy_name: The UK-flavored name the legacy evaluator mints, when
            it differs from ``name``; the verdict is re-minted under the
            declared name. Any other unexpected name passes through so the
            executor's name check fails it closed.
        evidence: Optional hook producing the entry's ``evidence_sha256``
            payload; ``None`` records no evidence hash.
    """

    name: str
    evaluator: Callable[[EvidenceContext, Mapping[str, Any]], GateResult]
    parameter_keys: frozenset[str] = frozenset()
    artifact_keys: frozenset[str] = frozenset()
    artifact_selector: Callable[[Mapping[str, Any]], frozenset[str]] | None = None
    needs_frame: bool = True
    frame_predicate: Callable[[Mapping[str, Any]], bool] | None = None
    legacy_name: str | None = None
    evidence: Callable[[EvidenceContext, Mapping[str, Any]], object] | None = None

    def required_artifacts(self, parameters: Mapping[str, Any]) -> frozenset[str]:
        if self.artifact_selector is not None:
            return self.artifact_selector(parameters)
        return self.artifact_keys

    def requires_frame(self, parameters: Mapping[str, Any]) -> bool:
        if self.frame_predicate is not None:
            return self.frame_predicate(parameters)
        return self.needs_frame

    def evaluate(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> GateResult:
        result = self.evaluator(context, parameters)
        if self.legacy_name is not None and result.name == self.legacy_name:
            result = replace(result, name=self.name)
        return result

    def evidence_payload(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> object | None:
        if self.evidence is None:
            return None
        return self.evidence(context, parameters)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def _evaluate_release_input_coverage(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    engine = context.artifacts["coverage_engine"]
    manifest = context.artifacts.get("coverage_manifest")
    check = parameters.get("check")
    if check == "manifest_current":
        assert_uk_release_input_coverage_manifest_current(
            engine=engine, manifest=manifest
        )
        return GateResult(
            name="release_input_coverage",
            passed=True,
            details={"check": "manifest_current"},
        )
    if check is not None:
        raise ValueError(
            f"unknown release_input_coverage check {check!r}; the declared "
            "preflight mode is 'manifest_current'."
        )
    return uk_release_input_coverage_gate(
        _uk_gate_surface(context.frame), engine, manifest=manifest
    )


def _coverage_requires_frame(parameters: Mapping[str, Any]) -> bool:
    return parameters.get("check") != "manifest_current"


def _evaluate_source_coverage(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    stage_names = tuple(str(name) for name in context.artifacts["build_stage_names"])
    assert_uk_release_input_coverage_build_stages(
        stage_names, manifest=context.artifacts.get("coverage_manifest")
    )
    return GateResult(
        name="source_coverage",
        passed=True,
        details={"stage_names": stage_names},
    )


def _evaluate_calibration_reference_coverage(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    if parameters:
        raise ValueError("calibration_reference_coverage takes no parameters.")
    evidence = context.artifacts["national_calibration"]
    declared = int(evidence["activated_reference_count"])
    resolved = int(evidence["resolved_reference_count"])
    matrix = int(evidence["matrix_target_count"])
    passed = declared == resolved == matrix
    return GateResult(
        name="calibration_reference_coverage",
        passed=passed,
        failures=()
        if passed
        else (
            f"Activated/resolved/matrix target counts differ: {declared}/{resolved}/{matrix}.",
        ),
        details={"activated": declared, "resolved": resolved, "matrix": matrix},
    )


def _evaluate_stage_health(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    stage = str(parameters["stage"])
    evidence_by_stage = context.artifacts["stage_evidence"]
    if not isinstance(evidence_by_stage, Mapping):
        raise TypeError("stage_evidence must map stage names to receipt payloads.")
    receipt = evidence_by_stage.get(stage)
    if not isinstance(receipt, Mapping):
        raise ValueError(f"stage_evidence has no receipt object for {stage!r}.")
    return uk_stage_health_gate(
        evidence=receipt,
        stage=stage,
        check=str(parameters["check"]),
        parameters=parameters,
    )


def _stage_health_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    stage = str(parameters["stage"])
    evidence_by_stage = context.artifacts["stage_evidence"]
    if not isinstance(evidence_by_stage, Mapping):
        raise TypeError("stage_evidence must map stage names to receipt payloads.")
    return {stage: evidence_by_stage.get(stage)}


def _evaluate_nonnegative_columns(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    if parameters:
        raise ValueError(
            "uk_nonnegative_columns takes no parameters; the checked columns "
            "come from source_stages.json nonnegative_outputs."
        )
    # The required set is every nonnegative output declared by a stage the
    # build actually scheduled — absence of a scheduled stage's declared
    # column is a failure (the shared gate's missing-column path), while
    # unscheduled stages' columns are not demanded. Never pre-filter to
    # present columns: that would let a frame missing every declared output
    # pass with columns_required=0.
    stage_names = tuple(context.artifacts["build_stage_names"])
    required = tuple(
        dict.fromkeys(
            column
            for stage in stage_names
            for column in UK_NONNEGATIVE_OUTPUTS_BY_STAGE.get(str(stage), ())
        )
    )
    column_values: dict[str, Any] = {}
    for entity in context.frame.entities:
        table = context.frame.table(entity)
        for column in table.columns:
            column_values.setdefault(str(column), table[column])
    return nonnegative_columns_gate(
        column_values,
        required,
    )


def _evaluate_column_implication(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    """Bind a person signal to a benunit flag and its same-source carrier.

    The capital checks (sentinel floor, sentinel parity, same-source
    equality, nonfinite) are deliberately re-derived at the terminal frame
    even though ``cohere_uc_capital`` establishes them at stage time: they
    guard the pipeline BETWEEN that stage and the terminal boundary — CGT
    cloning and donor stacking, salary sacrifice, student loans, age tail,
    and assembly — any of which could rebuild an entity table and corrupt
    one column without the other. Against the producer itself they cannot
    fail; that is not their job (adversarial-review finding 3).
    """

    frame = context.frame
    assert frame is not None  # GateBinding enforces frame evidence first.
    source_entity = str(parameters["numeric_entity"])
    target_entity = str(parameters["boolean_entity"])
    source = frame.table(source_entity)
    target = frame.table(target_entity)
    numeric_column = str(parameters["numeric_column"])
    source_group_column = str(parameters["numeric_group_column"])
    target_id_column = str(parameters["boolean_id_column"])
    boolean_column = str(parameters["boolean_column"])
    threshold = float(parameters.get("threshold", 0.0))

    required_source = {numeric_column, source_group_column}
    missing_source = sorted(required_source - set(source.columns))
    required_target = {target_id_column, boolean_column}
    missing_target = sorted(required_target - set(target.columns))
    if missing_source or missing_target:
        raise ValueError(
            "column_implication evidence is missing columns: "
            f"{source_entity}={missing_source}, {target_entity}={missing_target}."
        )
    if target[target_id_column].duplicated().any():
        raise ValueError(f"{target_entity}.{target_id_column} must be unique.")

    if threshold < 0:
        raise ValueError(
            f"column_implication amount threshold must be nonnegative, got {threshold!r}."
        )
    numeric = pd.to_numeric(source[numeric_column], errors="coerce")
    positive_ids = set(source.loc[numeric > threshold, source_group_column].tolist())
    aggregated = target[target_id_column].isin(positive_ids).to_numpy(dtype=np.int8)
    # The configured threshold applies to the person-level amounts above; the
    # aggregated evidence is a 0/1 indicator, so the primitive's threshold is
    # pinned at 0.0 regardless — reusing the amount threshold there would make
    # any configuration >= 1 vacuously green (adversarial-review finding 2).
    result = column_implication_gate(
        aggregated,
        target[boolean_column],
        numeric_column=f"{source_entity}.{numeric_column} aggregated to {target_entity}",
        boolean_column=f"{target_entity}.{boolean_column}",
        threshold=0.0,
    )

    capital_column = str(parameters["capital_column"])
    carrier_column = str(parameters["carrier_column"])
    sentinel = float(parameters.get("sentinel", -1.0))
    missing_capital = sorted({capital_column, carrier_column} - set(target.columns))
    if missing_capital:
        raise ValueError(
            f"column_implication {target_entity} capital evidence is missing "
            f"columns {missing_capital}."
        )
    capital = pd.to_numeric(target[capital_column], errors="coerce").to_numpy(
        dtype=float
    )
    carrier = pd.to_numeric(target[carrier_column], errors="coerce").to_numpy(
        dtype=float
    )
    nonfinite = ~np.isfinite(capital) | ~np.isfinite(carrier)
    # The -1 contract reserves exactly one negative: a value is either the
    # sentinel or nonnegative. A bare floor (`< sentinel`) would admit the
    # open interval between them — the one region the contract does not
    # define (adversarial-review verification residual 1). Sentinel equality
    # is exact (#833): every producer writes the -1.0 literal or a
    # nonnegative amount, and a tolerance band would silently reclassify a
    # corrupted near-sentinel value as a declared absence.
    out_of_domain = np.isfinite(capital) & ~((capital == sentinel) | (capital >= 0.0))
    carrier_out_of_domain = np.isfinite(carrier) & ~(
        (carrier == sentinel) | (carrier >= 0.0)
    )
    sentinel_mismatch = (capital == sentinel) != (carrier == sentinel)
    same_source_mismatch = (
        np.isfinite(capital) & np.isfinite(carrier) & (capital != carrier)
    )

    failures = list(result.failures)
    if nonfinite.any():
        failures.append(
            f"{target_entity}.{capital_column}/{carrier_column}: "
            f"{int(nonfinite.sum())} row(s) have non-finite carrier evidence."
        )
    if out_of_domain.any():
        failures.append(
            f"{target_entity}.{capital_column}: {int(out_of_domain.sum())} "
            f"value(s) outside the declared domain (exactly the {sentinel:g} "
            "sentinel or >= 0)."
        )
    if carrier_out_of_domain.any():
        failures.append(
            f"{target_entity}.{carrier_column}: "
            f"{int(carrier_out_of_domain.sum())} value(s) outside the declared "
            f"domain (exactly the {sentinel:g} sentinel or >= 0)."
        )
    if sentinel_mismatch.any():
        failures.append(
            f"{target_entity}.{capital_column}: sentinel {sentinel:g} is allowed "
            f"only where {carrier_column} has the same sentinel; "
            f"{int(sentinel_mismatch.sum())} mismatch(es)."
        )
    if same_source_mismatch.any():
        failures.append(
            f"{target_entity}.{capital_column} must equal {carrier_column}; "
            f"{int(same_source_mismatch.sum())} mismatch(es)."
        )
    return GateResult(
        name="column_implication",
        passed=not failures,
        failures=tuple(failures),
        details={
            **dict(result.details),
            "amount_threshold": threshold,
            "capital_column": f"{target_entity}.{capital_column}",
            "carrier_column": f"{target_entity}.{carrier_column}",
            "sentinel": sentinel,
            "capital_domain_violation_count": int(out_of_domain.sum()),
            "carrier_domain_violation_count": int(carrier_out_of_domain.sum()),
            "sentinel_mismatch_count": int(sentinel_mismatch.sum()),
            "same_source_mismatch_count": int(same_source_mismatch.sum()),
            "nonfinite_capital_count": int(nonfinite.sum()),
        },
    )


def _evaluate_take_up_signal(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    return uk_take_up_signal_gate(context.frame, **dict(parameters))


def _evaluate_enum_domain(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    columns = tuple(parameters.get("columns", ()))
    if len(columns) != 1 or not isinstance(columns[0], str):
        raise ValueError("UK enum_domain gates must declare exactly one column.")
    column = columns[0]
    domain = context.artifacts.get(f"{column}_enum_domain")
    if domain is None:
        engine = context.artifacts["rules_engine"]
        variable = engine._variable(column)
        domain = getattr(variable, "possible_values", None)
    if domain is None:
        raise ValueError(f"{column} enum domain could not be resolved from evidence.")
    matches = [
        context.frame.table(entity)[column]
        for entity in context.frame.entities
        if column in context.frame.table(entity).columns
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{column} must occur on exactly one frame entity; found {len(matches)}."
        )
    return enum_domain_gate({column: matches[0]}, {column: domain})


def _evaluate_support(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    resource_names = parameters.get("support_bounds_resources")
    if resource_names is None:
        single = parameters.get("support_bounds_resource")
        resource_names = [single] if isinstance(single, str) else None
    if not isinstance(resource_names, (list, tuple)) or not all(
        isinstance(name, str) for name in resource_names
    ):
        raise ValueError(
            "uk_support must declare support_bounds_resources as a list of "
            "support-bound resource filenames."
        )
    allowed = {
        "was_wealth_support_bounds.json",
        "lcfs_consumption_support_bounds.json",
        "etb_vat_support_bounds.json",
        "etb_services_support_bounds.json",
        "uc_deduction_support_bounds.json",
    }
    if set(resource_names) - allowed:
        raise ValueError(
            "uk_support declared unknown support-bound resource(s): "
            f"{sorted(set(resource_names) - allowed)}."
        )
    donor_ranges: dict[str, tuple[float, float]] = {}
    for resource_name in resource_names:
        resource = json.loads(
            files("microcosm.build.uk").joinpath(str(resource_name)).read_text()
        )
        raw_bounds = resource.get("bounds")
        if not isinstance(raw_bounds, Mapping):
            raise ValueError(f"{resource_name} is missing bounds.")
        for column, bounds in raw_bounds.items():
            donor_ranges[str(column)] = (float(bounds[0]), float(bounds[1]))
    values: dict[str, np.ndarray] = {}
    for entity in context.frame.entities:
        table = context.frame.table(entity)
        for column in donor_ranges:
            if column in table.columns:
                values.setdefault(column, table[column].to_numpy())
    return support_gate(values, donor_ranges)


def _evaluate_aggregate_admin(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    aggregate_artifact = context.artifacts["aggregate_admin"]
    if not isinstance(aggregate_artifact, Mapping):
        raise ValueError("aggregate_admin artifact must be a mapping.")
    anchors_payload = parameters.get("anchors")
    if not isinstance(anchors_payload, (list, tuple)):
        raise ValueError("aggregate_admin requires an anchors list.")
    anchors = tuple(
        TargetSpec(
            name=str(anchor["name"]),
            entity=str(anchor.get("entity", "household")),
            value=float(anchor["value"]),
            measure=str(anchor.get("measure", anchor["name"])),
            period=str(anchor.get("period", "2023")),
            source=str(anchor["source"]),
            family=str(anchor.get("family", "uk_admin")),
            tolerance=(
                None if anchor.get("tolerance") is None else float(anchor["tolerance"])
            ),
        )
        for anchor in anchors_payload
    )
    return aggregate_admin_gate(
        {str(key): float(value) for key, value in aggregate_artifact.items()},
        anchors,
        default_rtol=float(parameters.get("default_rtol", 0.5)),
    )


def _stage_names_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    return {
        "stage_names": [str(name) for name in context.artifacts["build_stage_names"]]
    }


def _exclusion_clock(context: EvidenceContext) -> date:
    """The one expiry clock every exclusion-consuming gate shares.

    Exclusion receipts carry approval and expiry dates, and the release
    contract requires every gate in one report to evaluate them on the same
    date. A per-gate default could straddle midnight, so the clock is a
    required artifact and anything but a plain ``date`` is refused.
    """

    clock = context.artifacts["exclusions_evaluated_on"]
    if isinstance(clock, datetime) or not isinstance(clock, date):
        raise ValueError(
            "exclusions_evaluated_on must be a datetime.date, got "
            f"{type(clock).__name__}; expiry must be evaluated on one "
            "shared clock."
        )
    return clock


def _resolve_degenerate_exclusions(
    context: EvidenceContext,
) -> tuple[Mapping[str, UKReviewedExclusion], str]:
    """The exclusion records the degenerate gate runs, and their source.

    The committed register is the reviewed policy of record (#630/#610);
    a supplied ``reviewed_degenerate_exclusions`` artifact is the loud
    review-time override — the evidence hook digests whichever resolved,
    so an overridden run self-describes in the signed report. The label
    follows the *content*, not the artifact's presence: records identical
    to the committed register are the committed policy whichever route
    delivered them, so an override cannot masquerade as a deviation (or a
    caller re-supplying the register as a false one).
    """

    committed = uk_default_degenerate_reviewed_exclusions()
    override = context.artifacts.get("reviewed_degenerate_exclusions")
    if override is None:
        return committed, "committed"
    resolved = coerce_reviewed_exclusions(
        override, label="UK degenerate-surface policy"
    )
    committed_payload = {
        name: record.policy_payload() for name, record in committed.items()
    }
    resolved_payload = {
        name: record.policy_payload() for name, record in resolved.items()
    }
    if resolved_payload == committed_payload:
        return committed, "committed"
    return resolved, "override"


def _exclusion_payload(
    records: Mapping[str, UKReviewedExclusion],
) -> dict[str, dict[str, str]]:
    return {name: record.policy_payload() for name, record in sorted(records.items())}


def _validate_input_mass_exclusion_references(
    register: Mapping[str, Mapping[str, UKReviewedExclusion]],
    *,
    label: str,
) -> None:
    unknown = sorted(set(register) - set(UK_INPUT_MASS_REFERENCE_REGISTRY))
    if unknown:
        raise ValueError(
            f"{label} names unknown input-mass reference(s) {unknown}; known "
            f"references are {sorted(UK_INPUT_MASS_REFERENCE_REGISTRY)}."
        )


def _resolve_input_mass_exclusions(
    context: EvidenceContext, *, reference: str
) -> tuple[Mapping[str, UKReviewedExclusion], str]:
    """Active-reference input-mass exclusions and their content source."""

    committed = uk_default_input_mass_reviewed_exclusions()
    _validate_input_mass_exclusion_references(
        committed, label="committed input-mass exclusion register"
    )
    committed_active = dict(committed.get(reference, {}))
    override = context.artifacts.get("reviewed_input_mass_exclusions")
    if override is None:
        return MappingProxyType(committed_active), "committed"
    if not isinstance(override, Mapping):
        raise TypeError("reviewed_input_mass_exclusions must be a mapping.")
    resolved: dict[str, Mapping[str, UKReviewedExclusion]] = {}
    for name, exclusions in override.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(
                "reviewed_input_mass_exclusions reference names must be "
                f"non-empty trimmed strings; got {name!r}."
            )
        resolved[name] = MappingProxyType(
            coerce_reviewed_exclusions(
                exclusions, label=f"UK input-mass override reference {name!r}"
            )
        )
    _validate_input_mass_exclusion_references(
        resolved, label="input-mass exclusion override"
    )
    active = dict(resolved.get(reference, {}))
    if _exclusion_payload(active) == _exclusion_payload(committed_active):
        return MappingProxyType(committed_active), "committed"
    return MappingProxyType(active), "override"


def _resolve_qrf_tail_exclusions(
    context: EvidenceContext,
) -> tuple[Mapping[str, UKReviewedExclusion], str]:
    """QRF tail exclusions and their content source."""

    committed = uk_default_qrf_tail_reviewed_exclusions()
    override = context.artifacts.get("reviewed_qrf_tail_exclusions")
    if override is None:
        return committed, "committed"
    resolved = coerce_reviewed_exclusions(override, label="UK QRF tail policy")
    if _exclusion_payload(resolved) == _exclusion_payload(committed):
        return committed, "committed"
    return MappingProxyType(resolved), "override"


def _evaluate_degenerate_release_surface(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    register = kwargs.pop("reviewed_exclusions_resource", None)
    if register != UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE:
        raise ValueError(
            f"uk/gates.json names exclusion register {register!r} but the "
            f"runtime loads {UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE!r}."
        )
    resolved, _source = _resolve_degenerate_exclusions(context)
    return uk_degenerate_release_surface_gate(
        _uk_gate_surface(context.frame),
        reviewed_exclusions=resolved,
        now=_exclusion_clock(context),
        **kwargs,
    )


def _degenerate_exclusions_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    resolved, source = _resolve_degenerate_exclusions(context)
    # ``exclusions_policy`` answers "which register content governed this
    # run" — deliberately distinct from the build record's
    # ``degenerate_exclusions_override_supplied``, which answers "did the
    # operator invoke the override path". The two can honestly disagree
    # (a review file byte-identical to the committed register), so they
    # carry different names.
    return {
        "exclusions_policy": source,
        "reviewed_exclusions": {
            name: record.policy_payload() for name, record in sorted(resolved.items())
        },
    }


def _evaluate_zero_weight_strata(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    declared = kwargs.pop("declarations", None)
    if declared is None:
        raise ValueError(
            "zero_weight_strata requires declared strata; an undeclared "
            "zero-weight surface is not a reviewed one."
        )
    declaration_keys = {"name", "selector", "maximum_zero_weight_rows", "reason"}
    for entry in declared:
        unknown = sorted(set(entry) - declaration_keys)
        if unknown:
            raise ValueError(
                f"zero_weight_strata declaration has unknown keys {unknown}; "
                f"allowed: {sorted(declaration_keys)}. A silently ignored key "
                "would sit in the policy hash while binding nothing."
            )
    declarations = tuple(
        UKZeroWeightStratumDeclaration(
            name=entry["name"],
            selector=entry["selector"],
            maximum_zero_weight_rows=entry["maximum_zero_weight_rows"],
            reason=entry["reason"],
        )
        for entry in declared
    )
    return uk_zero_weight_strata_gate(
        _uk_gate_surface(context.frame).household,
        declarations=declarations,
        **kwargs,
    )


def _evaluate_weight_ess(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    weights = _household_weights(_uk_gate_surface(context.frame).household)
    return uk_weight_ess_gate(weights, **dict(parameters))


def _evaluate_weight_ratio(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    weights = _household_weights(_uk_gate_surface(context.frame).household)
    return uk_weight_ratio_gate(weights, **dict(parameters))


def _evaluate_geography_ladder(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    household = _uk_gate_surface(context.frame).household
    weights = _household_weights(household)
    return uk_geography_ladder_gate(household, weights, **dict(parameters))


def _local_area_roster(
    resource: str, levels: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    payload = json.loads(files("microcosm.build.uk").joinpath(resource).read_text())
    declared = payload.get("levels")
    if not isinstance(declared, Mapping):
        raise ValueError(f"{resource} must expose a levels object.")
    roster: dict[str, tuple[str, ...]] = {}
    for level in levels:
        row = declared.get(level)
        if not isinstance(row, Mapping):
            raise ValueError(f"{resource} must expose level {level!r}.")
        area_ids = row.get("area_ids")
        if not isinstance(area_ids, (list, tuple)):
            raise ValueError(f"{resource} level {level!r} must expose area_ids.")
        roster[level] = tuple(str(area_id) for area_id in area_ids)
    return roster


def _evaluate_area_support(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    resource = str(kwargs.pop("crosswalk_resource"))
    exclusions_resource = str(kwargs.pop("exclusions_resource"))
    records = load_uk_reviewed_exclusion_register(
        None,
        resource=exclusions_resource,
    )
    clock = _exclusion_clock(context)
    invalid = {
        key: record
        for key, record in records.items()
        if record.expired(clock) or record.premature(clock)
    }
    if invalid:
        return GateResult(
            name="area_support",
            passed=False,
            failures=tuple(
                f"{key}: reviewed exclusion is outside its approval window "
                f"{record.approved_on}..{record.expires_on} at {clock.isoformat()}."
                for key, record in sorted(invalid.items())
            ),
            details={"invalid_reviewed_exclusions": sorted(invalid)},
        )
    levels = tuple(str(level) for level in kwargs["geography_levels"])
    return area_support_gate(
        context.artifacts["uk_area_support_summary"],
        area_roster=_local_area_roster(resource, levels),
        reviewed_exclusions=_exclusion_payload(records),
        **kwargs,
    )


def _local_target_error_items(evidence: object) -> tuple[tuple[str, float], ...]:
    if hasattr(evidence, "to_dict"):
        rows = evidence.to_dict(orient="records")
    elif isinstance(evidence, (list, tuple)):
        rows = evidence
    else:
        raise TypeError(
            "local_target_diagnostics must be a DataFrame or a sequence of rows."
        )
    items: list[tuple[str, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"local target diagnostic row {index} must be a mapping.")
        family = str(row.get("family", "")).strip()
        area_code = str(row.get("area_code", "")).strip()
        metric = str(row.get("metric", "")).strip()
        if not family or not area_code or not metric:
            raise ValueError(
                f"local target diagnostic row {index} must name family, "
                "area_code, and metric."
            )
        error = float(row["relative_error"])
        if not np.isfinite(error):
            raise ValueError(
                f"local target diagnostic {family}/{area_code}/{metric} has "
                "a non-finite relative_error."
            )
        items.append((f"{family}/{area_code}/{metric}", error))
    if not items:
        raise ValueError("local_target_diagnostics must not be empty.")
    return tuple(items)


def _evaluate_per_family_fit(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    items = _local_target_error_items(context.artifacts["local_target_diagnostics"])
    return per_family_fit_gate(
        (name for name, _ in items),
        (error for _, error in items),
        **dict(parameters),
    )


def _evaluate_weights_audit(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    # The shared binding would pass a present-but-empty record tuple to the
    # gate, which audits it vacuously; the legacy battery's guard is that a
    # fit stage which emitted nothing is a failed audit, not a passing one.
    records = tuple(context.artifacts["fit_weight_records"])
    if not records:
        return _missing_fit_weight_evidence_gate()
    return weights_audit_gate(records, **dict(parameters))


def _evaluate_export_surface(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    parity = context.artifacts["parity_evidence"]
    return uk_export_surface_gate(
        parity.candidate_columns,
        parity.reference_columns,
        **dict(parameters),
    )


def _evaluate_target_surface(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    if parameters.get("expected") == "local_default_surface":
        return _evaluate_local_default_target_surface(context, parameters)
    # National path: declared parameters forward to the gate unchanged, so an
    # entry that later declares reviewed exclusions keeps working and an
    # unknown parameter still fails closed inside the gate call (PR #795
    # review finding 4 restored the forward the local branch had dropped).
    parity = context.artifacts["parity_evidence"]
    return uk_target_surface_gate(
        parity.candidate_targets,
        parity.reference_targets,
        **dict(parameters),
    )


def _evaluate_local_default_target_surface(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    registry_artifact = _ledger_compile_parity_registry_artifact(parameters)
    if registry_artifact != "uk_ledger_compiled_local_registries":
        raise ValueError(
            "local_default_surface target_surface requires "
            "registry_artifact='uk_ledger_compiled_local_registries'."
        )
    target_period = parameters["target_period"]
    registry = _ledger_compile_parity_registry(
        context,
        target_period,
        registry_artifact=registry_artifact,
    )
    crosswalk_resource = str(parameters["crosswalk_resource"])
    membership_resource = str(parameters["membership_resource"])
    reviewed = dict(_local_default_reviewed_exclusions(membership_resource))
    reviewed.update(_ladder_derived_households_exclusions(crosswalk_resource))
    return target_surface_gate(
        _local_default_candidate_surface(registry),
        _local_default_expected_surface(crosswalk_resource),
        candidate_name="UK local compiled Ledger surface",
        reference_name="UK local default metric surface",
        reviewed_exclusions=reviewed,
    )


_LADDER_DERIVED_HOUSEHOLDS_RATIONALE = (
    "census_households is ladder-derived: the households column binds from the "
    "OA-ladder artifact's census household counts (the ladder sha is its "
    "provenance), never from Chronicle facts, so no ledger reference exists by "
    "design. See uk_local_target_census.json (source status pinned_in_ladder) "
    "and microcosm#542, which bound the family from the ladder."
)


def _ladder_derived_households_exclusions(crosswalk_resource: str) -> dict[str, str]:
    crosswalk = json.loads(
        files("microcosm.build.uk").joinpath(crosswalk_resource).read_text()
    )
    levels = crosswalk.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError(f"{crosswalk_resource} must expose levels.")
    reviewed: dict[str, str] = {}
    for geography_level in ("constituency", "local_authority"):
        level = levels.get(geography_level)
        if not isinstance(level, Mapping):
            raise ValueError(f"{crosswalk_resource} must expose {geography_level!r}.")
        for area_id in level.get("area_ids", ()):
            reviewed[f"households@{area_id}"] = _LADDER_DERIVED_HOUSEHOLDS_RATIONALE
    return reviewed


def _target_surface_required_artifacts(
    parameters: Mapping[str, Any],
) -> frozenset[str]:
    if parameters.get("expected") == "local_default_surface":
        return frozenset({_ledger_compile_parity_registry_artifact(parameters)})
    return frozenset({"parity_evidence"})


def _target_surface_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object | None:
    if parameters.get("expected") != "local_default_surface":
        return None
    registry_artifact = _ledger_compile_parity_registry_artifact(parameters)
    target_period = parameters["target_period"]
    registry = _ledger_compile_parity_registry(
        context,
        target_period,
        registry_artifact=registry_artifact,
    )
    crosswalk_resource = str(parameters["crosswalk_resource"])
    membership_resource = str(parameters["membership_resource"])
    crosswalk_text = (
        files("microcosm.build.uk").joinpath(crosswalk_resource).read_text()
    )
    membership_text = (
        files("microcosm.build.uk").joinpath(membership_resource).read_text()
    )
    candidate_targets = sorted(_local_default_candidate_surface(registry))
    reference_targets = sorted(_local_default_expected_surface(crosswalk_resource))
    reviewed_exclusions = _local_default_reviewed_exclusions(membership_resource)
    return {
        "expected": "local_default_surface",
        "registry_artifact": registry_artifact,
        "registry_count": len(registry) if hasattr(registry, "__len__") else None,
        "target_period": target_period,
        "crosswalk_resource": crosswalk_resource,
        "crosswalk_sha256": hashlib.sha256(crosswalk_text.encode("utf-8")).hexdigest(),
        "membership_resource": membership_resource,
        "membership_sha256": hashlib.sha256(
            membership_text.encode("utf-8")
        ).hexdigest(),
        "candidate_targets": candidate_targets,
        "reference_targets": reference_targets,
        "reviewed_exclusions": {
            name: reviewed_exclusions[name] for name in sorted(reviewed_exclusions)
        },
    }


def _local_default_candidate_surface(registry: object) -> frozenset[str]:
    targets: set[str] = set()
    for spec in registry:
        name = str(spec.name)
        if "@" not in name:
            raise ValueError(
                f"UK local compiled target {name!r} is missing the @ area suffix."
            )
        area_id = name.split("@", 1)[1]
        measure = str(spec.measure)
        if not measure:
            raise ValueError(f"UK local compiled target {name!r} has no measure.")
        targets.add(f"{measure}@{area_id}")
    return frozenset(targets)


def _local_default_expected_surface(crosswalk_resource: str) -> frozenset[str]:
    crosswalk = json.loads(
        files("microcosm.build.uk").joinpath(crosswalk_resource).read_text()
    )
    levels = crosswalk.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError(f"{crosswalk_resource} must expose levels.")
    expected: set[str] = set()
    for area_type, geography_level in (
        ("constituency", "constituency"),
        ("la", "local_authority"),
    ):
        level = levels.get(geography_level)
        if not isinstance(level, Mapping):
            raise ValueError(f"{crosswalk_resource} must expose {geography_level!r}.")
        area_ids = level.get("area_ids")
        if not isinstance(area_ids, (list, tuple)):
            raise ValueError(
                f"{crosswalk_resource} level {geography_level!r} must expose area_ids."
            )
        for metric_name in metric_names(area_type):
            expected.update(f"{metric_name}@{area_id}" for area_id in area_ids)
    return frozenset(expected)


def _local_default_reviewed_exclusions(
    membership_resource: str,
) -> dict[str, str]:
    membership = json.loads(
        files("microcosm.build.uk").joinpath(membership_resource).read_text()
    )
    metric_by_target_id = _local_metric_by_target_id()
    reviewed: dict[str, str] = {}
    for deferral in membership.get("signed_deferrals", ()):
        if not isinstance(deferral, Mapping):
            continue
        target_id = str(deferral.get("target_id", ""))
        metric_name = metric_by_target_id.get(target_id)
        if metric_name is None:
            continue
        rationale = str(deferral.get("rationale", ""))
        for area_id in deferral.get("area_ids", ()):
            reviewed[f"{metric_name}@{area_id}"] = rationale
    return reviewed


def _local_metric_by_target_id() -> dict[str, str]:
    contract = load_uk_local_geography_contract()
    mapping: dict[str, str] = {}
    for target in contract.get("targets", ()):
        if not isinstance(target, Mapping):
            continue
        target_id = target.get("target_id")
        bindings = target.get("bindings")
        if not isinstance(target_id, str) or not isinstance(bindings, Mapping):
            continue
        policyengine = bindings.get("policyengine")
        if not isinstance(policyengine, Mapping):
            continue
        metric_name = policyengine.get("metric_name")
        if isinstance(metric_name, str):
            mapping[target_id] = metric_name
    return mapping


def _resolve_target_fit_exclusions(
    context: EvidenceContext,
) -> tuple[Mapping[str, UKReviewedExclusion], str]:
    """Target-fit deferral exclusions and their content source."""

    committed = uk_default_target_fit_reviewed_exclusions()
    override = context.artifacts.get("reviewed_target_fit_exclusions")
    if override is None:
        return committed, "committed"
    resolved = coerce_reviewed_exclusions(override, label="UK target-fit policy")
    if _exclusion_payload(resolved) == _exclusion_payload(committed):
        return committed, "committed"
    return MappingProxyType(resolved), "override"


def _evaluate_target_fit(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    if kwargs.pop("surface", None) == "local_candidate":
        errors = dict(
            _local_target_error_items(context.artifacts["local_target_diagnostics"])
        )
        return uk_target_fit_gate(errors, **kwargs)
    register = kwargs.pop("reviewed_exclusions_resource", None)
    if register != UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE:
        raise ValueError(
            f"uk/gates.json names exclusion register {register!r} but the "
            f"runtime loads {UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE!r}."
        )
    exclusions, _source = _resolve_target_fit_exclusions(context)
    parity = context.artifacts["parity_evidence"]
    return uk_target_fit_gate(
        parity.target_relative_errors,
        reviewed_exclusions=exclusions,
        now=_exclusion_clock(context),
        **kwargs,
    )


def _target_fit_required_artifacts(
    parameters: Mapping[str, Any],
) -> frozenset[str]:
    if parameters.get("surface") == "local_candidate":
        return frozenset({"local_target_diagnostics"})
    return frozenset({"parity_evidence", "exclusions_evaluated_on"})


def _evaluate_input_mass_parity(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    reference_name = kwargs.pop("reference", None)
    declared_registry = kwargs.pop("reference_registry", None)
    register = kwargs.pop("reviewed_exclusions_resource", None)
    relative_tolerance = kwargs.pop("relative_tolerance", None)
    minimum_reference_total = kwargs.pop("minimum_reference_total", None)
    coerced_registry = coerce_input_mass_reference_registry(
        declared_registry, label="uk/gates.json input_mass_parity"
    )
    if coerced_registry != dict(UK_INPUT_MASS_REFERENCE_REGISTRY):
        raise ValueError(
            "uk/gates.json declares input-mass reference_registry that does "
            "not match the runtime registry; the declared pins and enforced "
            "pins must move together."
        )
    if reference_name not in coerced_registry:
        raise ValueError(
            f"uk/gates.json declares input-mass reference {reference_name!r} "
            f"but known references are {sorted(coerced_registry)}."
        )
    if register != UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE:
        raise ValueError(
            f"uk/gates.json names exclusion register {register!r} but the "
            f"runtime loads {UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE!r}."
        )
    exclusions, _source = _resolve_input_mass_exclusions(
        context, reference=reference_name
    )
    policy = UKInputMassParityPolicy(
        relative_tolerance=relative_tolerance,
        minimum_reference_total=minimum_reference_total,
        reviewed_exclusions=exclusions,
    )
    return uk_input_mass_parity_gate(
        uk_input_mass_totals(context.frame),
        context.artifacts["input_mass_reference"],
        descriptor=coerced_registry[reference_name],
        policy=policy,
        now=_exclusion_clock(context),
        **kwargs,
    )


def _input_mass_reference_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    reference_name = str(parameters["reference"])
    reference = context.artifacts["input_mass_reference"]
    exclusions, source = _resolve_input_mass_exclusions(
        context, reference=reference_name
    )
    return {
        "reference": reference_name,
        "reference_evidence_sha256": _input_mass_reference_evidence_sha256(reference),
        "exclusions_policy": source,
        "reviewed_exclusions": _exclusion_payload(exclusions),
    }


def _evaluate_tail_concentration(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    register = kwargs.pop("reviewed_exclusions_resource", None)
    top_k = kwargs.pop("top_k", None)
    max_top_share = kwargs.pop("max_top_share", None)
    min_nonzero_records = kwargs.pop("min_nonzero_records", None)
    if register != UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE:
        raise ValueError(
            f"uk/gates.json names exclusion register {register!r} but the "
            f"runtime loads {UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE!r}."
        )
    exclusions, _source = _resolve_qrf_tail_exclusions(context)
    policy = UKQRFTailConcentrationPolicy(
        top_k=top_k,
        max_top_share=max_top_share,
        min_nonzero_records=min_nonzero_records,
        reviewed_exclusions=exclusions,
    )
    values, weights, surface = uk_qrf_tail_concentration_columns(context.frame)
    return uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=policy,
        surface=surface,
        now=_exclusion_clock(context),
        **kwargs,
    )


def _evaluate_ledger_compile_parity(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    fixture_resource = str(parameters["fixture_resource"])
    fixture = _load_ledger_compile_parity_fixture(fixture_resource)
    signed_differences = parameters.get("signed_differences", ())
    signed_resource = parameters.get("signed_differences_resource")
    if signed_resource is not None:
        signed_differences = json.loads(
            files("microcosm.build.uk").joinpath(str(signed_resource)).read_text()
        )["differences"]
    target_period = parameters["target_period"]
    return ledger_compile_parity_gate(
        _ledger_compile_parity_registry(
            context,
            target_period,
            registry_artifact=_ledger_compile_parity_registry_artifact(parameters),
        ),
        fixture,
        signed_differences=signed_differences,
    )


def _load_ledger_compile_parity_fixture(fixture_resource: str) -> dict[str, Any]:
    fixture = json.loads(
        files("microcosm.build.uk").joinpath(fixture_resource).read_text()
    )
    if fixture_resource == LOCAL_REGISTRY_PARITY_FIXTURE_RESOURCE:
        return align_uk_local_registry_parity_fixture(fixture)
    return fixture


def _ledger_compile_parity_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    target_period = parameters["target_period"]
    registry_artifact = _ledger_compile_parity_registry_artifact(parameters)
    registry = _ledger_compile_parity_registry(
        context,
        target_period,
        registry_artifact=registry_artifact,
    )
    fixture_resource = str(parameters["fixture_resource"])
    fixture_text = files("microcosm.build.uk").joinpath(fixture_resource).read_text()
    return {
        "registry_artifact": registry_artifact,
        "registry_version": getattr(registry, "version", None),
        "registry_count": len(registry) if hasattr(registry, "__len__") else None,
        "fixture_resource": fixture_resource,
        "fixture_sha256": hashlib.sha256(fixture_text.encode("utf-8")).hexdigest(),
        "target_period": target_period,
        "signed_differences": parameters.get("signed_differences", ()),
        "signed_differences_resource": parameters.get("signed_differences_resource"),
    }


def _ledger_compile_parity_registry(
    context: EvidenceContext,
    target_period: object,
    *,
    registry_artifact: str = "uk_ledger_compiled_registries",
) -> object:
    registries = context.artifacts[registry_artifact]
    if not isinstance(registries, Mapping):
        raise TypeError(f"{registry_artifact} must map target periods to registries.")
    if target_period in registries:
        return registries[target_period]
    target_period_key = str(target_period)
    if target_period_key in registries:
        return registries[target_period_key]
    raise KeyError(
        f"UK Ledger compile parity artifact {registry_artifact!r} has no registry "
        f"for target period {target_period!r}; available periods: "
        f"{sorted(map(str, registries))}."
    )


_LEDGER_COMPILE_PARITY_REGISTRY_ARTIFACTS = frozenset(
    {
        "uk_ledger_compiled_registries",
        "uk_ledger_compiled_local_registries",
    }
)


def _ledger_compile_parity_registry_artifact(
    parameters: Mapping[str, Any],
) -> str:
    artifact = str(parameters.get("registry_artifact", "uk_ledger_compiled_registries"))
    if artifact not in _LEDGER_COMPILE_PARITY_REGISTRY_ARTIFACTS:
        raise ValueError(
            "ledger_compile_parity registry_artifact must be one of "
            f"{sorted(_LEDGER_COMPILE_PARITY_REGISTRY_ARTIFACTS)}, got "
            f"{artifact!r}."
        )
    return artifact


def _ledger_compile_parity_required_artifacts(
    parameters: Mapping[str, Any],
) -> frozenset[str]:
    return frozenset({_ledger_compile_parity_registry_artifact(parameters)})


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

#: The UK consumer registry: the shared defaults plus a binding for every
#: gate ``uk/gates.json`` declares. ``weights_audit`` overrides the shared
#: binding to keep the legacy guard the shared one lacks — a fit stage
#: that emitted an empty record set is a failed audit, not a vacuous pass.
#: ``input_mass_parity`` and ``tail_concentration`` override the generic
#: data-in-artifacts bindings with UK ones that derive candidate evidence
#: from the frame and hold the supplied reference to the spec-declared pin
#: (the microcosm#327 rule: a parity gate's reference is a declared input,
#: never implicit module state).
UK_GATE_REGISTRY: Mapping[str, GateBinding] = {
    **DEFAULT_REGISTRY,
    "weights_audit": UKGateBinding(
        name="weights_audit",
        evaluator=_evaluate_weights_audit,
        parameter_keys=frozenset({"allowed_unweighted"}),
        artifact_keys=frozenset({"fit_weight_records"}),
        needs_frame=False,
    ),
    "release_input_coverage": UKGateBinding(
        name="release_input_coverage",
        evaluator=_evaluate_release_input_coverage,
        parameter_keys=frozenset({"check"}),
        artifact_keys=frozenset({"coverage_engine"}),
        frame_predicate=_coverage_requires_frame,
        legacy_name="uk_release_input_coverage",
    ),
    "source_coverage": UKGateBinding(
        name="source_coverage",
        evaluator=_evaluate_source_coverage,
        artifact_keys=frozenset({"build_stage_names"}),
        needs_frame=False,
        evidence=_stage_names_evidence,
    ),
    "calibration_reference_coverage": UKGateBinding(
        name="calibration_reference_coverage",
        evaluator=_evaluate_calibration_reference_coverage,
        artifact_keys=frozenset({"national_calibration"}),
        needs_frame=False,
    ),
    "stage_health": UKGateBinding(
        name="stage_health",
        evaluator=_evaluate_stage_health,
        parameter_keys=frozenset(
            {
                "stage",
                "check",
                "columns",
                "exempt_columns",
                "max_clipped_low_rows_by_column",
                "max_clipped_high_rows_by_column",
                "target",
                "maximum_abs_realization_deviation",
                "allow_cap_bound",
                "stocks",
                "maximum_relative_mass_imbalance",
                "spi_prior_mass_share",
                "absolute_tolerance",
                "household_weight_kind",
                "minimum_spi_households",
                "minimum_identity_rows",
                "minimum_target_count",
                "minimum_signal_rows",
                "structural_zero_columns",
                "maximum_relative_deviation",
                "support_bounds_resource",
                "minimum_band_rows",
            }
        ),
        artifact_keys=frozenset({"stage_evidence"}),
        needs_frame=False,
        evidence=_stage_health_evidence,
    ),
    "nonnegative_columns": UKGateBinding(
        name="nonnegative_columns",
        evaluator=_evaluate_nonnegative_columns,
        artifact_keys=frozenset({"build_stage_names"}),
    ),
    "column_implication": UKGateBinding(
        name="column_implication",
        evaluator=_evaluate_column_implication,
        parameter_keys=frozenset(
            {
                "numeric_entity",
                "numeric_column",
                "numeric_group_column",
                "boolean_entity",
                "boolean_id_column",
                "boolean_column",
                "threshold",
                "capital_column",
                "carrier_column",
                "sentinel",
            }
        ),
    ),
    "take_up_signal": UKGateBinding(
        name="take_up_signal",
        evaluator=_evaluate_take_up_signal,
        parameter_keys=frozenset({"maximum_share_deviation"}),
    ),
    "enum_domain": UKGateBinding(
        name="enum_domain",
        evaluator=_evaluate_enum_domain,
        parameter_keys=frozenset({"columns"}),
    ),
    "support": UKGateBinding(
        name="support",
        evaluator=_evaluate_support,
        parameter_keys=frozenset(
            {"support_bounds_resource", "support_bounds_resources"}
        ),
    ),
    "aggregate_admin": UKGateBinding(
        name="aggregate_admin",
        evaluator=_evaluate_aggregate_admin,
        parameter_keys=frozenset({"anchors", "default_rtol"}),
        artifact_keys=frozenset({"aggregate_admin"}),
        needs_frame=False,
        legacy_name="aggregate_vs_admin",
    ),
    "degenerate_release_surface": UKGateBinding(
        name="degenerate_release_surface",
        evaluator=_evaluate_degenerate_release_surface,
        parameter_keys=frozenset({"reviewed_exclusions_resource"}),
        artifact_keys=frozenset({"exclusions_evaluated_on"}),
        evidence=_degenerate_exclusions_evidence,
    ),
    "zero_weight_strata": UKGateBinding(
        name="zero_weight_strata",
        evaluator=_evaluate_zero_weight_strata,
        parameter_keys=frozenset({"declarations"}),
    ),
    "weight_ess": UKGateBinding(
        name="weight_ess",
        evaluator=_evaluate_weight_ess,
        parameter_keys=frozenset({"minimum_ess_fraction"}),
    ),
    "weight_ratio": UKGateBinding(
        name="weight_ratio",
        evaluator=_evaluate_weight_ratio,
        parameter_keys=frozenset({"maximum_max_to_median_ratio"}),
    ),
    "spine_agreement": UKGateBinding(
        name="spine_agreement",
        evaluator=_evaluate_geography_ladder,
        legacy_name="uk_geography_ladder",
    ),
    "area_support": UKGateBinding(
        name="area_support",
        evaluator=_evaluate_area_support,
        parameter_keys=frozenset(
            {
                "crosswalk_resource",
                "geography_levels",
                "minimum_rows",
                "minimum_effective_sample_size",
                "minimum_distinct_sources",
                "exclusions_resource",
            }
        ),
        artifact_keys=frozenset({"uk_area_support_summary", "exclusions_evaluated_on"}),
        needs_frame=False,
    ),
    "per_family_fit": UKGateBinding(
        name="per_family_fit",
        evaluator=_evaluate_per_family_fit,
        parameter_keys=frozenset(
            {
                "within",
                "min_family_share",
                "hard_within",
                "min_hard_family_share",
                "min_family_size",
            }
        ),
        artifact_keys=frozenset({"local_target_diagnostics"}),
        needs_frame=False,
    ),
    "export_surface": UKGateBinding(
        name="export_surface",
        evaluator=_evaluate_export_surface,
        parameter_keys=frozenset({"allowed_extra_columns", "reviewed_exclusions"}),
        artifact_keys=frozenset({"parity_evidence"}),
        needs_frame=False,
    ),
    "target_surface": UKGateBinding(
        name="target_surface",
        evaluator=_evaluate_target_surface,
        parameter_keys=frozenset(
            {
                "expected",
                "registry_artifact",
                "target_period",
                "crosswalk_resource",
                "membership_resource",
            }
        ),
        artifact_selector=_target_surface_required_artifacts,
        needs_frame=False,
        evidence=_target_surface_evidence,
    ),
    "target_fit": UKGateBinding(
        name="target_fit",
        evaluator=_evaluate_target_fit,
        parameter_keys=frozenset(
            {"max_abs_relative_error", "reviewed_exclusions_resource", "surface"}
        ),
        artifact_selector=_target_fit_required_artifacts,
        needs_frame=False,
    ),
    "input_mass_parity": UKGateBinding(
        name="input_mass_parity",
        evaluator=_evaluate_input_mass_parity,
        parameter_keys=frozenset(
            {
                "reference",
                "reference_registry",
                "reviewed_exclusions_resource",
                "relative_tolerance",
                "minimum_reference_total",
                "candidate_name",
            }
        ),
        artifact_keys=frozenset({"input_mass_reference", "exclusions_evaluated_on"}),
        evidence=_input_mass_reference_evidence,
    ),
    "tail_concentration": UKGateBinding(
        name="tail_concentration",
        evaluator=_evaluate_tail_concentration,
        parameter_keys=frozenset(
            {
                "reviewed_exclusions_resource",
                "top_k",
                "max_top_share",
                "min_nonzero_records",
            }
        ),
        artifact_keys=frozenset({"exclusions_evaluated_on"}),
        legacy_name="qrf_tail_concentration",
    ),
    "ledger_compile_parity": UKGateBinding(
        name="ledger_compile_parity",
        evaluator=_evaluate_ledger_compile_parity,
        parameter_keys=frozenset(
            {
                "fixture_resource",
                "signed_differences",
                "signed_differences_resource",
                "target_period",
                "registry_artifact",
            }
        ),
        artifact_selector=_ledger_compile_parity_required_artifacts,
        needs_frame=False,
        evidence=_ledger_compile_parity_evidence,
    ),
}
