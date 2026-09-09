"""microcosm.build: stage plans and acceptance gates for dataset builds.

A build is a typed :class:`~microcosm.build.plan.StagePlan` over a
:class:`~microcosm.frame.Frame` — each stage declares what it consumes, what it
produces, and (for imputations) which donor survey it draws from; the executor
materializes stages in order and fails loudly on any donor it cannot load (no
silent fallbacks, per the charter). The product of a build is only publishable
once it passes the :mod:`microcosm.build.gates` suite: parity, support,
aggregate-vs-admin, per-family fit — plus the :mod:`microcosm.build.holdout`
rotation for scoring.

Importing this shard asserts compatibility with the installed
:mod:`microcosm.frame` kernel — the constellation mechanism from DESIGN.md.
"""

from microcosm.frame import __version__ as _frame_version

#: The microcosm-frame series this shard is built against (pre-1.0: minor-level
#: compatibility; see the identical gate in microcosm-calibrate).
_REQUIRED_FRAME_SERIES = (0, 1)


def _assert_frame_compatible(version: str, required: tuple[int, int]) -> None:
    """Raise unless the installed microcosm-frame is the expected series."""
    parts = version.split(".")
    try:
        installed = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):  # pragma: no cover - defensive
        raise ImportError(
            f"microcosm-build cannot parse microcosm-frame version "
            f"{version!r}; expected a {required[0]}.{required[1]}.x kernel."
        ) from None

    if required[0] == 0:
        compatible = installed == required
        expected = f"{required[0]}.{required[1]}.x"
    else:
        compatible = installed[0] == required[0]
        expected = f"{required[0]}.x"

    if not compatible:
        raise ImportError(
            f"microcosm-build requires microcosm-frame {expected}, but "
            f"{version} is installed. Install the matching constellation "
            "(the workspace releases the shards in lockstep): upgrade or pin "
            f"microcosm-frame to {expected}."
        )


_assert_frame_compatible(_frame_version, _REQUIRED_FRAME_SERIES)

from microcosm.build.chronicle_epoch import (  # noqa: E402 - after the compat gate
    ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS,
    ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS,
    CHRONICLE_EPOCH,
    DECLARED_IDENTITIES,
    DECLARED_IDENTITY_EPOCHS,
    EPOCHS,
    LEDGER_EPOCH,
    UNDECLARED,
    fact_key_epoch,
    fact_key_epoch_label,
    feed_fact_key_epochs,
    feed_undeclared_fact_key_domains,
    is_chronicle_fact_key,
    parse_fact_key,
)
from microcosm.build.country_spec import (  # noqa: E402 - after the compat gate
    CountryResourceRow,
    CountrySpec,
    GateSelectionSpec,
    GatesManifest,
    GeographySpineManifest,
    GeographySpineSpec,
    ReleaseContractManifest,
    ResolvedCountrySpec,
    country_stage_plan,
    load_country_spec,
)
from microcosm.build.gate_battery import (  # noqa: E402 - after the compat gate
    BlockingMode,
    EvidenceContext,
    FunctionBinding,
    GateBatteryBlockedError,
    GateBatteryRun,
    GateBinding,
    GateOutcome,
    GatePhaseReport,
    GateStatus,
    evaluate_phase,
    validate_gate_parameters,
)
from microcosm.build.gates import (  # noqa: E402 - after the compat gate
    FitWeightRecord,
    GateReport,
    GateResult,
    TargetCoverageRequirement,
    TargetFitRequirement,
    aggregate_admin_gate,
    area_support_gate,
    column_implication_gate,
    default_valued_columns_gate,
    enum_domain_gate,
    export_surface_gate,
    exported_nonzero_gate,
    formula_owned_export_gate,
    input_column_coverage_gate,
    input_mass_parity_gate,
    ledger_compile_parity_gate,
    ledger_compile_parity_signed_differences,
    macro_realism_gate,
    nonconstant_columns_gate,
    nonnegative_columns_gate,
    parity_gate,
    per_family_fit_gate,
    relative_error_loss,
    source_coverage_gate,
    source_stage_input_coverage_gate,
    support_gate,
    tail_concentration_gate,
    target_fit_gate,
    target_profile_coverage_gate,
    target_surface_gate,
    weights_audit_gate,
)
from microcosm.build.holdout import (  # noqa: E402 - after the compat gate
    rotated_folds,
    summarize_rotations,
)
from microcosm.build.ledger_artifact import (  # noqa: E402 - after the compat gate
    LedgerConsumerArtifact,
    add_ledger_artifact_args,
    load_ledger_consumer_artifact,
    resolve_ledger_artifact,
)
from microcosm.build.ledger_targets import (  # noqa: E402 - after the compat gate
    LedgerTargetMapping,
    LedgerTargetSelection,
    UnsupportedLedgerTarget,
    apply_ledger_target_profile,
    select_ledger_targets,
    select_ledger_targets_from_jsonl,
    target_spec_from_ledger_fact,
)

# Only ``logbook_env_names`` is re-exported here. The reader function is
# spelled ``logbook_env`` — the same name as its module — so binding it on
# the package would make ``import microcosm.build.logbook_env as env`` hand
# back the function instead of the module. Import the function from its
# module: ``from microcosm.build.logbook_env import logbook_env``.
from microcosm.build.logbook_env import (  # noqa: E402 - after the compat gate
    logbook_env_names,
)
from microcosm.build.monetary_profile import (  # noqa: E402 - after compat gate
    MonetaryTargetContract,
    MonetaryTargetProfile,
)
from microcosm.build.monetary_targets import (  # noqa: E402 - after compat gate
    MonetaryBasis,
    PreparedMonetaryMeasure,
    bind_monetary_target,
    prepare_monetary_measure,
)
from microcosm.build.plan import (  # noqa: E402 - after the compat gate
    DonorSpec,
    Stage,
    StageObservation,
    StagePlan,
    StageRecord,
)
from microcosm.build.source_runtime import (  # noqa: E402 - after the compat gate
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    UnsupportedSourceOperationError,
    run_source_stage,
)
from microcosm.build.staging import (  # noqa: E402 - after the compat gate
    DEFAULT_STAGING_PREFIX,
    LATEST_STAGING_POINTER,
    RUNS_INDEX,
    STAGING_SCHEMA_VERSION,
    StagingTelemetry,
)

__version__ = "0.1.0"

__all__ = [
    "BlockingMode",
    "CountrySpec",
    "CountryResourceRow",
    "DonorSpec",
    "DEFAULT_STAGING_PREFIX",
    "EvidenceContext",
    "FitWeightRecord",
    "FunctionBinding",
    "GateBatteryBlockedError",
    "GateBatteryRun",
    "GateBinding",
    "GateOutcome",
    "GatePhaseReport",
    "GateReport",
    "GateResult",
    "GateSelectionSpec",
    "GateStatus",
    "GatesManifest",
    "evaluate_phase",
    "validate_gate_parameters",
    "GeographySpineManifest",
    "GeographySpineSpec",
    "ReleaseContractManifest",
    "ResolvedCountrySpec",
    "Stage",
    "StageObservation",
    "StagePlan",
    "StageRecord",
    "SourceRuntimeConfig",
    "SourceRuntimeContext",
    "SourceRuntimeError",
    "LATEST_STAGING_POINTER",
    "RUNS_INDEX",
    "STAGING_SCHEMA_VERSION",
    "StagingTelemetry",
    "TargetCoverageRequirement",
    "TargetFitRequirement",
    "ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS",
    "ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS",
    "CHRONICLE_EPOCH",
    "DECLARED_IDENTITIES",
    "DECLARED_IDENTITY_EPOCHS",
    "EPOCHS",
    "LEDGER_EPOCH",
    "UNDECLARED",
    "LedgerConsumerArtifact",
    "LedgerTargetMapping",
    "LedgerTargetSelection",
    "MonetaryBasis",
    "MonetaryTargetContract",
    "MonetaryTargetProfile",
    "PreparedMonetaryMeasure",
    "add_ledger_artifact_args",
    "logbook_env_names",
    "fact_key_epoch",
    "fact_key_epoch_label",
    "feed_fact_key_epochs",
    "feed_undeclared_fact_key_domains",
    "is_chronicle_fact_key",
    "parse_fact_key",
    "aggregate_admin_gate",
    "area_support_gate",
    "column_implication_gate",
    "apply_ledger_target_profile",
    "bind_monetary_target",
    "default_valued_columns_gate",
    "enum_domain_gate",
    "export_surface_gate",
    "exported_nonzero_gate",
    "formula_owned_export_gate",
    "input_column_coverage_gate",
    "input_mass_parity_gate",
    "ledger_compile_parity_gate",
    "ledger_compile_parity_signed_differences",
    "load_ledger_consumer_artifact",
    "macro_realism_gate",
    "nonconstant_columns_gate",
    "nonnegative_columns_gate",
    "parity_gate",
    "prepare_monetary_measure",
    "per_family_fit_gate",
    "relative_error_loss",
    "resolve_ledger_artifact",
    "rotated_folds",
    "source_coverage_gate",
    "source_stage_input_coverage_gate",
    "select_ledger_targets",
    "select_ledger_targets_from_jsonl",
    "summarize_rotations",
    "support_gate",
    "tail_concentration_gate",
    "target_fit_gate",
    "target_profile_coverage_gate",
    "target_spec_from_ledger_fact",
    "target_surface_gate",
    "weights_audit_gate",
    "UnsupportedLedgerTarget",
    "UnsupportedSourceOperationError",
    "country_stage_plan",
    "load_country_spec",
    "run_source_stage",
    "__version__",
]
