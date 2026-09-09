"""Structure-exact coverage of the generation-0 US authority inventory.

This module is deliberately independent of the constants-era runtime.  It
compares the resolved bundle with the compiled IR and the pure legacy adapter
payload.  Counts are reported as useful diagnostics, but a check passes only
when the complete ordered object, its content digest, or both agree with the
reviewed generation-0 contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from .canonical import canonical_json_bytes, sha256_json
from .compiler_ir import CompiledSpecIR, compile_spec
from .imputation_semantics import derive_primary_effective_predictor_tuples
from .legacy_adapter import compile_to_legacy_payload
from .model import FrozenValue, ResolvedSpec, thaw_json
from .stacked_authority_semantics import project_stacked_checkpoint_base_identity
from .typed_closure import compile_producer_outputs

EXPECTED_ADAPTER_SURFACES = frozenset(
    {
        "battery_contract",
        "calibration_contract",
        "calibration_tail_contracts",
        "imputation",
        "publication_release",
        "source_manifest",
        "spine_assembly",
        "spine_sampling",
        "stacked_authority_receipt",
        "stacked_checkpoint_static_components",
        "support_spine",
        "take_up_contract",
        "take_up_contract_identity",
    }
)

EXPECTED_AUTHORITY_COMPONENTS = (
    "gap_fill_plan",
    "post_puf_transfer_surface",
    "declared_surface",
    "metric_registry",
    "joint_metric_registry",
    "support_profile",
    "puf_capital_gains_tail_support_contract",
    "late_producer_schedule",
    "post_transfer_calibration",
)

EXPECTED_CHECKPOINT_TOP_LEVEL = frozenset(
    {
        "artifact_kind",
        "materializer_version",
        "model_seed",
        "period",
        "pipeline",
        "policyengine_us_version",
        "pool_code",
        "geography_assignment",
        "schema_version",
        "stacked_authority",
    }
)

EXPECTED_FULL_CHECKPOINT_TOP_LEVEL = EXPECTED_CHECKPOINT_TOP_LEVEL | {
    "clone_attachment",
    "inputs",
    "sampling",
}

EXPECTED_CHECKPOINT_POOL_CODE = frozenset(
    {
        "acs_pums_earnings_universe_contract",
        "acs_transfer_max_targets_per_fit",
        "acs_transfer_n_estimators",
        "derive_operator_order",
        "gap_fill_producer_schedule",
        "late_producer_resource_semantics",
        "late_producer_schedule",
        "operator_order",
        "post_clone_source_operator_order",
        "pre_clone_source_operator_order",
        "primary_qrf_checkpoint_schema_version",
        "primary_qrf_n_estimators",
        "primary_qrf_target_order",
        "puf_capital_gains_tail_manifest_schema_version",
        "puf_capital_gains_tail_support_contract",
        "remaining_stage_input_manifest",
        "simulation_household_batch_size",
        "take_up_contract",
        "us_qbi_reconciliation_contract",
    }
)

EXPECTED_PROGRAM_IDS = (
    "snap",
    "tanf",
    "eitc",
    "medicaid",
    "chip",
    "basic_health_program",
    "medicare",
    "ssi",
    "dc_ptc",
    "head_start",
    "early_head_start",
    "housing_assistance",
    "wic",
    "aca",
    "ca_premium_subsidy",
    "co_premium_assistance",
    "nm_premium_assistance",
)

EXPECTED_TAKE_UP_MECHANISMS: Mapping[str, tuple[str, tuple[str, ...], str]] = {
    "snap": (
        "modeled",
        ("probability_seed", "assignment", "count_calibration"),
        "out_of_scope",
    ),
    "tanf": ("modeled", ("probability_seed",), "seed"),
    "eitc": ("modeled", ("probability_seed",), "seed"),
    "medicaid": (
        "modeled",
        ("assignment", "count_calibration"),
        "count_calibrated",
    ),
    "chip": ("engine", ("engine_default",), "rate_unsourced"),
    "basic_health_program": (
        "engine",
        ("engine_default",),
        "rate_unsourced",
    ),
    "medicare": ("measured", ("measured_map",), "out_of_scope"),
    "ssi": (
        "modeled",
        ("probability_seed", "delivery_gate"),
        "count_calibrated",
    ),
    "dc_ptc": ("engine", ("engine_default",), "rate_unsourced"),
    "head_start": ("transferred", ("imputed_transfer",), "out_of_scope"),
    "early_head_start": ("engine", ("engine_default",), "rate_unsourced"),
    "housing_assistance": (
        "mixed",
        ("measured_map", "imputed_transfer"),
        "out_of_scope",
    ),
    "wic": ("modeled", ("probability_seed",), "out_of_scope"),
    "aca": (
        "modeled",
        (
            "measured_map",
            "assignment",
            "count_calibration",
            "count_calibration",
            "assignment",
            "assignment",
            "count_calibration",
        ),
        "out_of_scope",
    ),
    "ca_premium_subsidy": ("engine", ("engine_default",), "rate_unsourced"),
    "co_premium_assistance": (
        "engine",
        ("engine_default",),
        "rate_unsourced",
    ),
    "nm_premium_assistance": (
        "engine",
        ("engine_default",),
        "rate_unsourced",
    ),
}

EXPECTED_POST_CLONE_OPERATORS = (
    "with_us_prior_year_income_inputs",
    "with_us_medicare_take_up_input",
    "with_us_pregnancy_inputs",
    "with_us_wic_claim_input",
    "impute_us_housing_assistance_to_puf_support",
    "with_us_child_support_inputs",
    "with_us_disability_benefits",
    "with_us_workers_compensation",
    "with_us_weeks_unemployed",
    "with_us_childcare_inputs",
    "with_us_adult_care_inputs",
    "with_us_energy_subsidy_input",
    "with_us_retirement_contribution_inputs",
    "with_us_retirement_distribution_inputs",
    "with_us_immigration_inputs",
    "with_us_education_inputs",
)

EXPECTED_PRODUCER_ORDER = (
    "acs_pums_earnings_universe",
    "primary_puf_qrf",
    "source:impute_us_housing_assistance_to_puf_support",
    "source:with_us_child_support_inputs",
    "source:with_us_childcare_inputs",
    "source:with_us_disability_benefits",
    "source:with_us_energy_subsidy_input",
    "source:with_us_immigration_inputs",
    "source:with_us_medicare_take_up_input",
    "source:with_us_pregnancy_inputs",
    "source:with_us_prior_year_income_inputs",
    "source:with_us_retirement_contribution_inputs",
    "source:with_us_retirement_distribution_inputs",
    "source:with_us_weeks_unemployed",
    "source:with_us_workers_compensation",
    "transfer:person/puf_tax_itemization__batch_1",
    "transfer:person/puf_tax_itemization__batch_4",
    "transfer:person/puf_tax_itemization__batch_5",
    "transfer:tax_unit/puf_tax_itemization",
    "source:with_us_adult_care_inputs",
    "source:with_us_wic_claim_input",
    "transfer:person/model_required_boolean",
    "transfer:person/puf_tax_itemization__batch_2",
    "transfer:person/puf_tax_itemization__batch_3",
    "transfer:person/source_operator_child_support",
    "transfer:person/source_operator_disability_benefits",
    "transfer:person/source_operator_immigration",
    "transfer:person/source_operator_medicare_take_up",
    "transfer:person/source_operator_retirement_contributions",
    "transfer:person/source_operator_retirement_distributions",
    "transfer:person/source_operator_weeks_unemployed",
    "transfer:person/source_operator_workers_compensation",
    "transfer:spm_unit/source_operator_energy_subsidy",
    "source:with_us_education_inputs",
    "transfer:person/adult_care",
    "transfer:person/source_operator_wic_claim",
    "source_finalizer",
    "transfer:person/source_operator_education_inputs",
)

EXPECTED_SEED_STREAMS = (
    "build_model",
    "calibration",
    "exact_k_selection",
    "geography_legacy",
    "puf_archived_disaggregation",
    "puf_clone_attachment",
    "puf_live_disaggregation",
    "qrf_fit_draw",
    "sampling_acs",
    "sampling_asec",
    "sipp_training_cap",
    "ssi_model",
    "ssi_weighted_replacement",
    "stable_entity_draw",
)

EXPECTED_SEED_GROUPS: Mapping[str, tuple[str, ...]] = {
    "run_request_578": (
        "survey_sample_asec",
        "survey_sample_acs",
        "puf_clone_attachment",
    ),
    "puf_archived_42_vs_live_build_seed": (
        "puf_archived_aggregate_disaggregation",
        "puf_live_aggregate_disaggregation",
    ),
    "ssi_fixed_contracts": (
        "ssi_weighted_replacement_training",
        "ssi_archived_qrf_model",
    ),
    "stable_x31_training_caps": (
        "sipp_vehicle_training_cap",
        "sipp_financial_asset_training_cap",
        "acs_rent_archived_training_cap",
    ),
    "tips_fixed_cap": ("sipp_tip_training_cap",),
    "scf_composite_and_models": (
        "scf_household_source_selector",
        "scf_financial_asset_qrf_model",
        "scf_net_worth_qrf_model",
        "scf_auto_loan_qrf_model",
    ),
    "acs_transfer_derivations": (
        "acs_transfer_family_seed",
        "acs_transfer_pattern_seed",
    ),
    "qrf_shared_rng_contracts": (
        "sipp_vehicle_qrf_model",
        "sipp_financial_asset_qrf_models",
        "primary_qrf_fit_draw",
        "acs_qrf_fit_draw",
    ),
    "blake2b_stable_draws": (
        "source_aca_assignment",
        "source_count_calibration",
        "source_joint_count_calibration",
        "snap_take_up_assignment",
        "pregnancy_assignment",
        "wic_claim_assignment",
        "snap_discretionary_exemption_assignment",
        "immigration_ead_workers_assignment",
        "immigration_ead_students_assignment",
        "ssi_take_up_assignment",
        "medicaid_take_up_assignment",
        "snap_state_take_up_assignment",
        "tanf_take_up_assignment",
        "eitc_take_up_assignment",
    ),
    "other_direct_stochastic_sites": (
        "adult_care_weighted_prefix_assignment",
        "capital_gains_tail_random_rank",
        "torch_calibration_reseed",
        "exact_k_pcg64_selection",
    ),
    "build_seeded_5000_caps": (
        "prior_year_income_training_cap",
        "childcare_training_cap",
        "retirement_contributions_training_cap",
        "disability_benefits_training_cap",
        "housing_inputs_training_cap",
        "workers_compensation_training_cap",
        "retirement_distributions_training_cap",
        "child_support_training_cap",
        "energy_subsidy_training_cap",
        "other_health_insurance_training_cap",
        "weeks_unemployed_training_cap",
    ),
    "legacy_geography_draws": (
        "legacy_geography_ladder",
        "legacy_puma_ladder",
        "legacy_congressional_district_assignment",
    ),
}

EXPECTED_RUNGS = (
    (0.01, 100, "f001"),
    (0.04, 400, "f004"),
    (0.10, 1000, "f010"),
    (0.25, 2500, "f025"),
    (1.00, 10000, "f100"),
)
EXPECTED_RELEASE_REGEX = (
    r"^microcosm-us-2024-stacked-f(?:001|004|010|025|100)-s[0-9]+-"
    r"asec[0-9]+-acs[0-9]+-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
)
EXPECTED_LEGACY_RELEASE_REGEX = EXPECTED_RELEASE_REGEX.replace(
    "microcosm-us-2024", "populace-us-2024"
)

EXPECTED_HASHES = {
    "acs_group_predictors": "a927bb7ecf3e84f54c93583ab79318654514ac546aefafba67da5285615fbd60",
    "acs_person_predictors": "878c788a6f037d7aca12b3586ea034eff04f3034ffa11935a736493042551f25",
    "authority": "9d4a9672a0f03039b1fe874b9fe21ed575be0d29f14afc396d03cdf5c809bdd2",
    "early_families": "4aa9f736fd76e83955477ad1667e58f48f264783f05bdc7f0102cd32d61323bd",
    "full_checkpoint": "7176664c34039def5f43281a7735f792fe43de4ef0a6e7ca18f53d812b412d15",
    "gap_fill_schedule": "1c31f9868f7884347cc19cf1ff65da43f950b9114941a715bab168246db414a7",
    "graph_nodes": "40cd51ffdfe2e9d9d08d48c08e8ded9de1e4b134783bab05c4abc6ad5c72ca1e",
    "geography_assignment": "f49425ca8734ac559c73cf44f6458d86d3162a48956b98a27e6e758959361585",
    "late_families": "d91f9ff0eb52f43e7b6eed3d5c58c37abe1620c3a11021da15dae9c10e16d382",
    "late_resource_semantics": "b2b7dbd64db211088e85c94ad6ca1b942cb5e45683eb8c34ba9b664b5de64624",
    "late_schedule": "e59c019d3d454eac99ac0ac209b6c5b6faaf9bdfcaeee18c36a25be19bf7da2f",
    "ownership": "5f64f0aac49e2313177564f71876bffc8c81b3ded4df701e70930e60e9c98356",
    "primary_tuples": "987b501c695e31f45521c4a178528f75ab3df22c09bc407b182213b2de99ee57",
    "seed_map": "c94a5af8eb24866156b8cd4b77b93dc497b7df5b025204fe1022f80b1d8446ea",
    "seed_protocol": "1c53f1d9b3e185a41181fbe861f9b08d56c83880710fd1e18d7a4ceb5d6a354e",
    "source_manifest": "cd5ba8924d64da5425ee14cca82a774e3f4b2bb5aabe06df291cc3cc457287a9",
    "take_up": "fa186daea0f8dd641cc470e41d1a2953f887d45282ec990201298f47bedf8d4d",
    "tail": "ac92829c88a1a4fb6460d61190918d5d99c6c377fc8dd8f62f02b332d09bf59c",
}

INVENTORY_REPORT_SCHEMA_VERSION = 1

EXPECTED_INVENTORY_ITEMS = frozenset(
    {
        "acs_group_predictors_exact",
        "acs_person_predictors_exact",
        "capital_gains_tail_contract_exact",
        "conditional_ownership_matrix_exact",
        "early_gap_fill_plan_exact",
        "early_transfer_surface_exact",
        "gap_fill_schedule_receipt_exact",
        "stacked_geography_assignment_exact",
        "itemization_declared_splits_exact",
        "late_schedule_receipt_exact",
        "late_split_ledger_exact",
        "legacy_adapter_surfaces_exact",
        "post_clone_operator_order_exact",
        "primary_predictor_tuples_exact",
        "producer_dag_order_edges_waves_exact",
        "producer_inputs_exact",
        "producer_outputs_exact",
        "producer_receipt_transition_contract_exact",
        "producer_registry_exact",
        "producer_resource_semantics_exact",
        "producer_virtual_resources_exact",
        "qrf_model_parameters_explicit",
        "release_line_and_regex_exact",
        "release_rungs_exact",
        "seed_inventory_groups_exhaustive",
        "seed_owner_rows_exact",
        "seed_protocol_and_owner_map_digests_exact",
        "seed_protocol_header_streams_exact",
        "seed_site_definitions_exact",
        "seed_site_owner_bindings_exact",
        "source_stage_manifest_exact",
        "stacked_authority_components_exact",
        "stacked_authority_identity_exact",
        "stacked_checkpoint_base_identity_exact",
        "stacked_checkpoint_pool_code_exact",
        "stacked_checkpoint_top_level_exact",
        "take_up_identity_exact",
        "take_up_legacy_contract_exact",
        "take_up_pipeline_steps_exact",
        "take_up_program_mechanisms_exact",
        "take_up_program_order_exact",
    }
)

EXPECTED_INVENTORY_COUNTS: Mapping[str, int] = {
    "adapter_surfaces": 13,
    "authority_components": 9,
    "early_families": 13,
    "early_targets": 48,
    "itemization_batches": 5,
    "itemization_targets": 37,
    "late_groups": 19,
    "late_targets": 70,
    "ownership_rows": 18,
    "primary_effective_predictor_tuples": 65,
    "primary_families": 1,
    "primary_targets": 65,
    "producer_authored_outputs": 92,
    "producer_compiled_outputs": 227,
    "producer_inputs": 2_744,
    "producer_nodes": 38,
    "producer_virtual_resources": 75,
    "release_rungs": 5,
    "resolved_references": 334,
    "seed_owner_bindings": 112,
    "seed_owner_rows": 54,
    "seed_sites": 53,
    "seed_streams": 14,
    "source_operators": 16,
    "source_stages": 37,
    "stacked_checkpoint_full_components": 13,
    "stacked_checkpoint_pool_code_components": 19,
    "stacked_checkpoint_static_components": 10,
    "tail_control_fields": 934,
    "take_up_pipeline_steps": 28,
    "take_up_programs": 17,
    "typed_artifacts": 84,
    "typed_columns": 176,
    "typed_entities": 8,
    "typed_scopes": 7,
}


class InventoryCoverageError(AssertionError):
    """The generated inventory omits or changes a required F0 contract."""


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InventoryCoverageError(f"{location}: object required")
    return value


def _array(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise InventoryCoverageError(f"{location}: array required")
    return value


def _wire(value: FrozenValue) -> object:
    return thaw_json(value)


#: Subtrees that describe the machine a build ran on, never the spec.
_OPERATIONAL_KEYS = frozenset({"worker_execution", "audit_aliases"})


def _without_operational_bindings(value: object) -> object:
    """Strip execution-profile subtrees before semantic inventory digesting.

    ``worker_execution`` is the primary-QRF worker's execution binding: its
    semantic identity hashes the interpreter binary, ABI, ``pyvenv.cfg``,
    the installed distributions' RECORD files, and the resolved fit
    controls (CPU count included), and its audit aliases spell the
    interpreter path. All of that authenticates an environment for replay
    and legitimately differs between macOS and the Linux CI runners; none
    of it is spec evidence. Inventory digests are therefore computed over
    the receipt with the whole subtree removed, as they were before the
    binding became portable, so the pinned digests hold on every platform.
    The live generation-0 receipt itself is unchanged.
    """

    if isinstance(value, Mapping):
        drop_self_hash = "producers" in value and "sha256" in value
        return {
            key: _without_operational_bindings(item)
            for key, item in value.items()
            if key not in _OPERATIONAL_KEYS and not (drop_self_hash and key == "sha256")
        }
    if isinstance(value, list):
        return [_without_operational_bindings(item) for item in value]
    return value


def _operational_free_sha256(value: object) -> str:
    """Digest with operational subtrees and receipt self-hashes removed.

    The receipt's embedded ``sha256`` is computed over the unpruned
    content, so it re-imports the interpreter-path instability; a
    self-hash is identified as a ``sha256`` key sitting beside the
    ``producers`` array it summarizes. Input content pins (``sha256``
    beside a locator) are semantic and stay.
    """

    return sha256_json(_without_operational_bindings(value))


def _json_equal(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _terminal_count(value: object) -> int:
    if isinstance(value, Mapping):
        return (
            1 if not value else sum(_terminal_count(child) for child in value.values())
        )
    if isinstance(value, (list, tuple)):
        return 1 if not value else sum(_terminal_count(child) for child in value)
    return 1


def _bundle_home_matches(
    domains: Mapping[str, object],
    pointer: str,
) -> tuple[object, ...]:
    """Resolve one reviewed bundle-home pointer, including path wildcards."""

    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return ()
    tokens = [
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer.removeprefix("/").split("/")
        if token
    ]
    matches: tuple[object, ...] = (domains,)
    for token in tokens:
        selected: list[object] = []
        for value in matches:
            if token == "*":
                if isinstance(value, Mapping):
                    selected.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    selected.extend(value)
                continue
            if isinstance(value, Mapping):
                if token in value:
                    selected.append(value[token])
                continue
            if isinstance(value, (list, tuple)):
                try:
                    index = int(token)
                except ValueError:
                    continue
                if 0 <= index < len(value):
                    selected.append(value[index])
        matches = tuple(selected)
        if not matches:
            break
    return matches


def _take_up_steps(program: Mapping[str, object]) -> tuple[str, ...]:
    result = [
        str(_mapping(value, "take_up pipeline step")["kind"])
        for value in _array(program.get("pipeline", []), "take_up pipeline")
    ]
    for segment_value in _array(program.get("segments", []), "take_up segments"):
        segment = _mapping(segment_value, "take_up segment")
        result.extend(
            str(_mapping(value, "take_up segment step")["kind"])
            for value in _array(segment.get("pipeline", []), "take_up segment pipeline")
        )
    return tuple(result)


def _check_item(
    *,
    clauses: Mapping[str, bool],
    bundle_homes: Sequence[str],
    bundle_home_match_counts: Mapping[str, int],
    consumers: Sequence[str],
    observed: Mapping[str, object],
    expected: Mapping[str, object] | str,
) -> dict[str, object]:
    failures = [name for name, passed in clauses.items() if not passed]
    return {
        "status": "covered" if not failures else "missing",
        "bundle_homes": list(bundle_homes),
        "bundle_home_match_counts": dict(bundle_home_match_counts),
        "compiler_consumers": list(consumers),
        "observed": dict(observed),
        "expected": dict(expected) if isinstance(expected, Mapping) else expected,
        "failures": failures,
    }


def build_inventory_coverage(
    spec: ResolvedSpec,
    *,
    compiled: CompiledSpecIR | None = None,
    legacy_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the fail-closed, structure-exact D1 inventory report."""

    if not isinstance(spec, ResolvedSpec):
        raise TypeError("build_inventory_coverage requires a ResolvedSpec")
    if spec.country != "us":
        raise InventoryCoverageError("the generation-0 inventory is US-specific")
    compiled = compile_spec(spec) if compiled is None else compiled
    expected_legacy = compile_to_legacy_payload(spec)
    legacy = (
        expected_legacy if legacy_payload is None else deepcopy(dict(legacy_payload))
    )
    domains = {
        resource.descriptor.kind.value: resource.domain.to_wire()
        for resource in spec.resources
        if resource.descriptor.kind.value not in {"legacy_json", "schema"}
    }
    imputation = _mapping(domains["imputation"], "imputation")
    sources = _mapping(domains["sources"], "sources")
    geography = _mapping(domains["geography"], "geography")
    spine = _mapping(domains["spine"], "spine")
    take_up = _mapping(domains["take_up"], "take_up")
    calibration = _mapping(domains["calibration"], "calibration")
    publication = _mapping(domains["publication"], "publication")
    graph = _mapping(imputation["producer_graph"], "imputation/producer_graph")
    source_nodes = [
        _mapping(value, "producer node")
        for value in _array(graph["nodes"], "producer graph nodes")
    ]
    compiled_by_id = {node.id: node for node in compiled.producer_graph.nodes}
    expected_outputs = compile_producer_outputs(domains)
    actual_imputation = _mapping(legacy.get("imputation"), "legacy/imputation")
    expected_imputation = _mapping(
        expected_legacy["imputation"], "expected legacy/imputation"
    )
    authority = _mapping(legacy.get("stacked_authority_receipt"), "stacked authority")
    expected_authority = _mapping(
        expected_legacy["stacked_authority_receipt"], "expected stacked authority"
    )
    static = _mapping(
        legacy.get("stacked_checkpoint_static_components"), "checkpoint static"
    )
    expected_static = _mapping(
        expected_legacy["stacked_checkpoint_static_components"],
        "expected checkpoint static",
    )
    pool_code = _mapping(static.get("pool_code"), "checkpoint pool_code")
    geography_assignment = _mapping(
        static.get("geography_assignment"),
        "checkpoint geography assignment",
    )
    expected_geography_assignment = _mapping(
        expected_static.get("geography_assignment"),
        "expected checkpoint geography assignment",
    )

    family_rows = [
        _mapping(value, "imputation family")
        for value in _array(imputation["families"], "imputation families")
    ]
    early = [row for row in family_rows if row.get("stage") == "gap_fill_stacked_spine"]
    primary = [row for row in family_rows if row.get("stage") == "primary_puf_qrf"]
    late = [row for row in family_rows if row.get("stage") == "late_producer_dag"]
    itemization = [
        row
        for row in late
        if str(row.get("id", "")).startswith("late/person/puf_tax_itemization__batch_")
    ]
    primary_tuples = derive_primary_effective_predictor_tuples(imputation)

    items: dict[str, dict[str, object]] = {}

    def add(
        name: str,
        *,
        clauses: Mapping[str, bool],
        homes: Sequence[str],
        consumers: Sequence[str],
        observed: Mapping[str, object],
        expected: Mapping[str, object] | str,
    ) -> None:
        home_match_counts = {
            home: len(_bundle_home_matches(domains, home)) for home in homes
        }
        home_clauses = {
            f"bundle home matched zero values: {home}": count > 0
            for home, count in home_match_counts.items()
        }
        items[name] = _check_item(
            clauses={**clauses, **home_clauses},
            bundle_homes=homes,
            bundle_home_match_counts=home_match_counts,
            consumers=consumers,
            observed=observed,
            expected=expected,
        )

    add(
        "legacy_adapter_surfaces_exact",
        clauses={
            "adapter surface names differ": set(legacy) == EXPECTED_ADAPTER_SURFACES
        },
        homes=("/bundle",),
        consumers=("legacy_adapter.compile_to_legacy_payload",),
        observed={"names": sorted(legacy)},
        expected={"names": sorted(EXPECTED_ADAPTER_SURFACES)},
    )

    early_targets = sum(len(_array(row["targets"], "early targets")) for row in early)
    add(
        "early_gap_fill_plan_exact",
        clauses={
            "early family count differs": len(early) == 13,
            "early target count differs": early_targets == 48,
            "early family content digest differs": sha256_json(early)
            == EXPECTED_HASHES["early_families"],
            "compiled gap-fill plan differs": _json_equal(
                actual_imputation.get("gap_fill_plan"),
                expected_imputation.get("gap_fill_plan"),
            ),
        },
        homes=("/imputation/families", "/imputation/gap_fill_schedule"),
        consumers=("legacy_adapter.imputation.gap_fill_plan",),
        observed={
            "families": len(early),
            "targets": early_targets,
            "families_sha256": sha256_json(early),
        },
        expected={
            "families": 13,
            "targets": 48,
            "sha256": EXPECTED_HASHES["early_families"],
        },
    )
    add(
        "early_transfer_surface_exact",
        clauses={
            "post-PUF authority component differs": _json_equal(
                _mapping(authority.get("components"), "authority components").get(
                    "post_puf_transfer_surface"
                ),
                _mapping(
                    expected_authority.get("components"),
                    "expected authority components",
                ).get("post_puf_transfer_surface"),
            ),
            "gap-fill schedule differs": _json_equal(
                actual_imputation.get("gap_fill_producer_schedule_receipt"),
                expected_imputation.get("gap_fill_producer_schedule_receipt"),
            ),
        },
        homes=("/imputation/families", "/imputation/producer_graph/nodes"),
        consumers=(
            "legacy_adapter.imputation.gap_fill_producer_schedule_receipt",
            "legacy_adapter.stacked_authority_receipt.post_puf_transfer_surface",
        ),
        observed={"families_sha256": sha256_json(early)},
        expected="complete ordered early family and transfer authority projections",
    )

    blocks = _mapping(imputation["predictor_blocks"], "predictor blocks")
    person_block_ids = (
        "acs_person_required",
        "acs_person_optional_income",
        "acs_person_optional_structure",
        "acs_person_housing_required",
    )
    group_block_ids = ("acs_group_required", "acs_group_optional")
    person_blocks = {key: blocks[key] for key in person_block_ids}
    group_blocks = {key: blocks[key] for key in group_block_ids}
    actual_transfer = actual_imputation.get("transfer_execution_contract_identities")
    expected_transfer = expected_imputation.get(
        "transfer_execution_contract_identities"
    )
    add(
        "acs_person_predictors_exact",
        clauses={
            "person predictor block digest differs": sha256_json(person_blocks)
            == EXPECTED_HASHES["acs_person_predictors"],
            "transfer identities differ": _json_equal(
                actual_transfer, expected_transfer
            ),
        },
        homes=tuple(f"/imputation/predictor_blocks/{key}" for key in person_block_ids),
        consumers=("legacy_adapter.imputation.transfer_execution_contract_identities",),
        observed={
            "sha256": sha256_json(person_blocks),
            "block_ids": list(person_blocks),
        },
        expected={
            "sha256": EXPECTED_HASHES["acs_person_predictors"],
            "block_ids": list(person_block_ids),
        },
    )
    add(
        "acs_group_predictors_exact",
        clauses={
            "group predictor block digest differs": sha256_json(group_blocks)
            == EXPECTED_HASHES["acs_group_predictors"],
            "transfer identities differ": _json_equal(
                actual_transfer, expected_transfer
            ),
        },
        homes=tuple(f"/imputation/predictor_blocks/{key}" for key in group_block_ids),
        consumers=("legacy_adapter.imputation.transfer_execution_contract_identities",),
        observed={"sha256": sha256_json(group_blocks), "block_ids": list(group_blocks)},
        expected={
            "sha256": EXPECTED_HASHES["acs_group_predictors"],
            "block_ids": list(group_block_ids),
        },
    )

    primary_target_order = [str(row["target"]) for row in primary_tuples]
    add(
        "primary_predictor_tuples_exact",
        clauses={
            "primary family is not unique": len(primary) == 1,
            "effective tuple count differs": len(primary_tuples) == 65,
            "effective tuple digest differs": sha256_json(primary_tuples)
            == EXPECTED_HASHES["primary_tuples"],
            "legacy target order differs": _mapping(
                actual_imputation.get("primary_qrf"), "primary QRF"
            ).get("target_order")
            == primary_target_order,
        },
        homes=("/imputation/families", "/imputation/predictor_blocks/puf_tax_detail"),
        consumers=("compiler_ir.node_slices", "legacy_adapter.imputation.primary_qrf"),
        observed={"tuples": len(primary_tuples), "sha256": sha256_json(primary_tuples)},
        expected={"tuples": 65, "sha256": EXPECTED_HASHES["primary_tuples"]},
    )
    qrf_params = _mapping(
        _mapping(
            _mapping(imputation["models"], "models")["regime_gated_qrf"], "QRF model"
        )["params"],
        "QRF params",
    )
    add(
        "qrf_model_parameters_explicit",
        clauses={
            "QRF parameters differ": _json_equal(
                qrf_params,
                {"n_estimators": 100, "max_samples_leaf": None, "zero_atol": 1e-6},
            ),
            "primary estimator identity differs": pool_code.get(
                "primary_qrf_n_estimators"
            )
            == 100,
            "ACS estimator identity differs": pool_code.get("acs_transfer_n_estimators")
            == 100,
        },
        homes=("/imputation/models/regime_gated_qrf/params",),
        consumers=(
            "compiler_ir.node_slices",
            "legacy_adapter.stacked_checkpoint_static_components",
        ),
        observed={"params": dict(qrf_params)},
        expected={"n_estimators": 100, "max_samples_leaf": None, "zero_atol": 1e-6},
    )

    late_targets = sum(len(_array(row["targets"], "late targets")) for row in late)
    late_schedule = _mapping(
        actual_imputation.get("late_producer_schedule_receipt"), "late schedule"
    )
    expected_late_schedule = _mapping(
        expected_imputation.get("late_producer_schedule_receipt"),
        "expected late schedule",
    )
    add(
        "late_split_ledger_exact",
        clauses={
            "late group count differs": len(late) == 19,
            "late target count differs": late_targets == 70,
            "late family digest differs": sha256_json(late)
            == EXPECTED_HASHES["late_families"],
            "legacy late schedule differs": _json_equal(
                late_schedule, expected_late_schedule
            ),
        },
        homes=("/imputation/families", "/imputation/chaining/split_after"),
        consumers=("legacy_adapter.imputation.late_producer_schedule_receipt",),
        observed={
            "groups": len(late),
            "targets": late_targets,
            "sha256": sha256_json(late),
        },
        expected={
            "groups": 19,
            "targets": 70,
            "sha256": EXPECTED_HASHES["late_families"],
        },
    )
    itemization_sizes = [
        len(_array(row["targets"], "itemization targets")) for row in itemization
    ]
    add(
        "itemization_declared_splits_exact",
        clauses={
            "itemization batch ids differ": [row["id"] for row in itemization]
            == [
                f"late/person/puf_tax_itemization__batch_{index}"
                for index in range(1, 6)
            ],
            "itemization batch sizes differ": itemization_sizes == [8, 8, 8, 8, 5],
            "itemization target count differs": sum(itemization_sizes) == 37,
            "legacy late schedule differs": _json_equal(
                late_schedule, expected_late_schedule
            ),
        },
        homes=("/imputation/families", "/imputation/chaining/split_after"),
        consumers=("legacy_adapter.imputation.late_producer_schedule_receipt",),
        observed={
            "ids": [row["id"] for row in itemization],
            "batch_sizes": itemization_sizes,
        },
        expected={"batches": 5, "targets": 37, "batch_sizes": [8, 8, 8, 8, 5]},
    )

    source_node_by_id = {str(row["id"]): row for row in source_nodes}
    node_sources_exact = set(source_node_by_id) == set(compiled_by_id) and all(
        _json_equal(_wire(compiled_by_id[node_id].source), source_node)
        for node_id, source_node in source_node_by_id.items()
    )
    add(
        "producer_registry_exact",
        clauses={
            "producer ids or source objects differ": node_sources_exact,
            "producer count differs": len(source_nodes) == 38,
            "producer source digest differs": sha256_json(source_nodes)
            == EXPECTED_HASHES["graph_nodes"],
        },
        homes=("/imputation/producer_graph/nodes",),
        consumers=("compiler_ir.producer_graph.nodes", "compiler_ir.node_slices"),
        observed={"nodes": len(source_nodes), "sha256": sha256_json(source_nodes)},
        expected={"nodes": 38, "sha256": EXPECTED_HASHES["graph_nodes"]},
    )
    graph_relation = {
        "edges": [list(edge) for edge in compiled.producer_graph.edges],
        "waves": [list(wave) for wave in compiled.producer_graph.waves],
        "order": list(compiled.producer_graph.order),
    }
    legacy_graph_relation = {
        "edges": late_schedule.get("edges"),
        "waves": late_schedule.get("waves"),
        "order": late_schedule.get("order"),
    }
    add(
        "producer_dag_order_edges_waves_exact",
        clauses={
            "producer order differs": compiled.producer_graph.order
            == EXPECTED_PRODUCER_ORDER,
            "edge count differs": len(compiled.producer_graph.edges) == 71,
            "wave count differs": len(compiled.producer_graph.waves) == 6,
            "stage DAG differs": (
                compiled.stage_dag.edges == compiled.producer_graph.edges
                and compiled.stage_dag.waves == compiled.producer_graph.waves
                and compiled.stage_dag.order == compiled.producer_graph.order
            ),
            "legacy schedule graph differs": _json_equal(
                graph_relation, legacy_graph_relation
            ),
            "schedule digest differs": compiled.producer_graph.schedule_sha256
            == EXPECTED_HASHES["late_schedule"],
            "legacy schedule digest differs": late_schedule.get("schedule_sha256")
            == EXPECTED_HASHES["late_schedule"],
        },
        homes=(
            "/imputation/producer_graph/nodes",
            "/imputation/producer_graph/external_stages",
        ),
        consumers=(
            "compiler_ir.stage_dag",
            "legacy_adapter.imputation.late_producer_schedule_receipt",
        ),
        observed={
            "nodes": len(compiled.producer_graph.order),
            "edges": len(compiled.producer_graph.edges),
            "waves": len(compiled.producer_graph.waves),
            "schedule_sha256": compiled.producer_graph.schedule_sha256,
        },
        expected={
            "nodes": 38,
            "edges": 71,
            "waves": 6,
            "schedule_sha256": EXPECTED_HASHES["late_schedule"],
        },
    )

    inputs_exact = node_sources_exact and all(
        _json_equal(
            [_wire(value) for value in compiled_by_id[node_id].inputs],
            source_node.get("inputs", []),
        )
        for node_id, source_node in source_node_by_id.items()
    )
    input_count = sum(
        len(_array(row.get("inputs", []), "producer inputs")) for row in source_nodes
    )
    add(
        "producer_inputs_exact",
        clauses={
            "producer input rows differ": inputs_exact,
            "input row count differs": input_count == 2744,
        },
        homes=("/imputation/producer_graph/nodes/*/inputs",),
        consumers=(
            "compiler_ir.producer_graph.nodes.inputs",
            "compiler_ir.node_slices",
        ),
        observed={"rows": input_count},
        expected={"rows": 2744, "relation": "source rows preserved exactly"},
    )
    outputs_exact = set(expected_outputs) == set(compiled_by_id) and all(
        _json_equal(
            [_wire(value) for value in compiled_by_id[node_id].outputs],
            expected_outputs[node_id],
        )
        for node_id in expected_outputs
    )
    authored_output_count = sum(
        len(_array(row.get("outputs", []), "producer outputs")) for row in source_nodes
    )
    add(
        "producer_outputs_exact",
        clauses={
            "compiled outputs differ from typed closure": outputs_exact,
            "authored output count differs": authored_output_count == 92,
            "compiled output count differs": compiled.producer_graph.compiled_output_count
            == 227,
        },
        homes=(
            "/imputation/producer_graph/nodes/*/outputs",
            "/imputation/families/*/targets",
        ),
        consumers=(
            "typed_closure.compile_producer_outputs",
            "compiler_ir.producer_graph.nodes.outputs",
        ),
        observed={
            "authored_rows": authored_output_count,
            "compiled_rows": compiled.producer_graph.compiled_output_count,
        },
        expected={
            "authored_rows": 92,
            "compiled_rows": 227,
            "relation": "typed closure exact",
        },
    )
    virtual_count = sum(
        len(_array(row.get("virtual_resources", []), "virtual resources"))
        for row in source_nodes
    )
    virtual_exact = node_sources_exact and all(
        _mapping(_wire(compiled_by_id[node_id].source), "compiled node source").get(
            "virtual_resources", []
        )
        == source_node.get("virtual_resources", [])
        for node_id, source_node in source_node_by_id.items()
    )
    add(
        "producer_virtual_resources_exact",
        clauses={
            "virtual resources differ": virtual_exact,
            "virtual resource count differs": virtual_count == 75,
        },
        homes=("/imputation/producer_graph/nodes/*/virtual_resources",),
        consumers=(
            "compiler_ir.node_slices",
            "legacy_adapter.imputation.late_producer_resource_semantics",
        ),
        observed={"rows": virtual_count},
        expected={"rows": 75, "relation": "source node slices preserve full bindings"},
    )
    resource_semantics = _mapping(
        actual_imputation.get("late_producer_resource_semantics"), "resource semantics"
    )
    add(
        "producer_resource_semantics_exact",
        clauses={
            "resource semantics object differs": _json_equal(
                resource_semantics,
                expected_imputation.get("late_producer_resource_semantics"),
            ),
            "resource semantics digest differs": _operational_free_sha256(
                resource_semantics
            )
            == EXPECTED_HASHES["late_resource_semantics"],
        },
        homes=(
            "/imputation/producer_graph/resource_semantics",
            "/imputation/producer_graph/nodes/*/virtual_resources",
        ),
        consumers=("legacy_adapter.imputation.late_producer_resource_semantics",),
        observed={
            "sha256": _operational_free_sha256(resource_semantics),
            "producer_count": resource_semantics.get("producer_count"),
        },
        expected={
            "sha256": EXPECTED_HASHES["late_resource_semantics"],
            "producer_count": 38,
        },
    )
    execution_contract = graph.get("execution_receipt_contract")
    add(
        "producer_receipt_transition_contract_exact",
        clauses={
            "execution receipt contract differs": _json_equal(
                late_schedule.get("execution_receipt_contract"), execution_contract
            ),
            "legacy schedule differs from pure projection": _json_equal(
                late_schedule, expected_late_schedule
            ),
            "transition authority is absent": isinstance(execution_contract, Mapping)
            and "transition_authority" in execution_contract,
        },
        homes=("/imputation/producer_graph/execution_receipt_contract",),
        consumers=("legacy_adapter.imputation.late_producer_schedule_receipt",),
        observed={"sha256": sha256_json(execution_contract)},
        expected="complete execution-row and transition-authority object",
    )

    source_ownership = _array(graph["ownership_matrix"], "ownership matrix")
    compiled_ownership = list(compiled.producer_graph.ownership_matrix)
    overlap = _mapping(actual_imputation.get("overlap_ownership"), "overlap ownership")
    cell_keys = [
        (row["entity"], row["target"], row["origin"], row["clone_index"])
        for value in source_ownership
        for row in [_mapping(value, "ownership row")]
    ]
    final_owner_exact = all(
        [
            action["producer"]
            for value in _array(row["producer_actions"], "ownership actions")
            for action in [_mapping(value, "ownership action")]
            if action["owns_final"]
        ]
        == [row["final_owner"]]
        for value in source_ownership
        for row in [_mapping(value, "ownership row")]
    )
    add(
        "conditional_ownership_matrix_exact",
        clauses={
            "ownership source and IR differ": _json_equal(
                source_ownership, compiled_ownership
            ),
            "ownership row count differs": len(source_ownership) == 18,
            "ownership cells are not unique": len(set(cell_keys)) == 18,
            "final ownership is ambiguous": final_owner_exact,
            "legacy ownership differs": _json_equal(
                overlap, expected_imputation.get("overlap_ownership")
            ),
            "ownership digest differs": overlap.get("sha256")
            == EXPECTED_HASHES["ownership"],
        },
        homes=(
            "/imputation/producer_graph/ownership_contract",
            "/imputation/producer_graph/ownership_matrix",
        ),
        consumers=(
            "compiler_ir.producer_graph.ownership_matrix",
            "legacy_adapter.imputation.overlap_ownership",
        ),
        observed={"rows": len(source_ownership), "sha256": overlap.get("sha256")},
        expected={"rows": 18, "sha256": EXPECTED_HASHES["ownership"]},
    )

    source_manifest = legacy.get("source_manifest")
    expected_source_manifest = expected_legacy["source_manifest"]
    add(
        "source_stage_manifest_exact",
        clauses={
            "source manifest differs from projection": _json_equal(
                source_manifest, expected_source_manifest
            ),
            "source manifest digest differs": sha256_json(source_manifest)
            == EXPECTED_HASHES["source_manifest"],
            "stage count differs": len(_array(sources["stages"], "source stages"))
            == 37,
        },
        homes=("/sources/stage_manifest", "/sources/stages"),
        consumers=(
            "legacy_adapter.source_manifest",
            "legacy_adapter.imputation.late_producer_resource_semantics",
        ),
        observed={
            "stages": len(_array(sources["stages"], "source stages")),
            "sha256": sha256_json(source_manifest),
        },
        expected={"stages": 37, "sha256": EXPECTED_HASHES["source_manifest"]},
    )
    pipeline = _mapping(spine["pipeline_contract"], "spine pipeline contract")
    operator_order = tuple(
        str(value)
        for value in _array(
            pipeline["post_clone_source_operator_order"], "post-clone operators"
        )
    )
    source_node_ids = {
        node_id.removeprefix("source:")
        for node_id in source_node_by_id
        if node_id.startswith("source:")
    }
    add(
        "post_clone_operator_order_exact",
        clauses={
            "authored operator order differs": operator_order
            == EXPECTED_POST_CLONE_OPERATORS,
            "operator nodes differ": source_node_ids
            == set(EXPECTED_POST_CLONE_OPERATORS),
            "checkpoint operator order differs": pool_code.get(
                "post_clone_source_operator_order"
            )
            == list(EXPECTED_POST_CLONE_OPERATORS),
        },
        homes=(
            "/spine/pipeline_contract/post_clone_source_operator_order",
            "/imputation/producer_graph/nodes",
        ),
        consumers=(
            "compiler_ir.producer_graph",
            "legacy_adapter.stacked_checkpoint_static_components",
        ),
        observed={"operators": list(operator_order)},
        expected={"operators": list(EXPECTED_POST_CLONE_OPERATORS)},
    )

    programs = [
        _mapping(value, "take-up program")
        for value in _array(take_up["programs"], "take-up programs")
    ]
    program_ids = tuple(str(row["id"]) for row in programs)
    actual_take_up = _mapping(legacy.get("take_up_contract"), "legacy take-up")
    expected_take_up = _mapping(expected_legacy["take_up_contract"], "expected take-up")
    legacy_programs = _array(actual_take_up.get("programs"), "legacy take-up programs")
    legacy_by_variable = {
        str(_mapping(value, "legacy take-up program")["variable"]): _mapping(
            value, "legacy take-up program"
        )
        for value in legacy_programs
    }
    add(
        "take_up_program_order_exact",
        clauses={
            "program order differs": program_ids == EXPECTED_PROGRAM_IDS,
            "legacy program count differs": len(legacy_programs) == 17,
            "legacy variable order differs": [
                _mapping(value, "legacy program")["variable"]
                for value in legacy_programs
            ]
            == [row["variable"] for row in programs],
        },
        homes=("/take_up/programs",),
        consumers=("legacy_adapter.take_up_contract.programs",),
        observed={"program_ids": list(program_ids)},
        expected={"program_ids": list(EXPECTED_PROGRAM_IDS)},
    )
    step_map = {str(row["id"]): _take_up_steps(row) for row in programs}
    add(
        "take_up_pipeline_steps_exact",
        clauses={
            "typed pipeline step kinds differ": all(
                step_map.get(program_id) == contract[1]
                for program_id, contract in EXPECTED_TAKE_UP_MECHANISMS.items()
            ),
            "pipeline step count differs": sum(
                len(value) for value in step_map.values()
            )
            == 28,
            "legacy contract differs from typed projection": _json_equal(
                actual_take_up, expected_take_up
            ),
        },
        homes=("/take_up/programs/*/pipeline", "/take_up/programs/*/segments"),
        consumers=("legacy_adapter.take_up_contract",),
        observed={
            "steps": sum(len(value) for value in step_map.values()),
            "by_program": {key: list(value) for key, value in step_map.items()},
        },
        expected={
            "steps": 28,
            "by_program": {
                key: list(value[1])
                for key, value in EXPECTED_TAKE_UP_MECHANISMS.items()
            },
        },
    )
    mechanism_rows: dict[str, object] = {}
    mechanisms_exact = True
    for row in programs:
        program_id = str(row["id"])
        variable = str(row["variable"])
        legacy_row = legacy_by_variable.get(variable, {})
        observed_mechanism = (
            str(row["ownership"]),
            _take_up_steps(row),
            legacy_row.get("populace_treatment"),
        )
        mechanism_rows[program_id] = [
            observed_mechanism[0],
            list(observed_mechanism[1]),
            observed_mechanism[2],
        ]
        mechanisms_exact &= (
            observed_mechanism == EXPECTED_TAKE_UP_MECHANISMS[program_id]
        )
    add(
        "take_up_program_mechanisms_exact",
        clauses={
            "ownership, typed steps, or legacy treatment differ": mechanisms_exact
        },
        homes=("/take_up/programs",),
        consumers=(
            "take_up_semantics.validate_take_up_semantics",
            "legacy_adapter.take_up_contract",
        ),
        observed={"programs": mechanism_rows},
        expected={
            "programs": {
                key: [value[0], list(value[1]), value[2]]
                for key, value in EXPECTED_TAKE_UP_MECHANISMS.items()
            }
        },
    )
    add(
        "take_up_legacy_contract_exact",
        clauses={
            "legacy take-up contract differs": _json_equal(
                actual_take_up, expected_take_up
            ),
            "legacy take-up digest differs": sha256_json(actual_take_up)
            == EXPECTED_HASHES["take_up"],
        },
        homes=(
            "/take_up/programs",
            "/take_up/legacy_contract_metadata",
            "/take_up/scope_registry",
        ),
        consumers=("legacy_adapter.take_up_contract",),
        observed={"sha256": sha256_json(actual_take_up)},
        expected={"sha256": EXPECTED_HASHES["take_up"]},
    )
    take_up_identity = legacy.get("take_up_contract_identity")
    add(
        "take_up_identity_exact",
        clauses={
            "take-up identity differs": _json_equal(
                take_up_identity, expected_legacy["take_up_contract_identity"]
            ),
            "take-up identity does not bind raw contract": _mapping(
                take_up_identity, "take-up identity"
            ).get("resource_sha256")
            == EXPECTED_HASHES["take_up"],
        },
        homes=("/take_up",),
        consumers=(
            "legacy_adapter.take_up_contract_identity",
            "legacy_adapter.stacked_checkpoint_static_components",
        ),
        observed={
            "resource_sha256": _mapping(take_up_identity, "take-up identity").get(
                "resource_sha256"
            )
        },
        expected={"resource_sha256": EXPECTED_HASHES["take_up"]},
    )

    support_role = next(
        _mapping(value, "support role")
        for value in _array(spine["support_roles"], "support roles")
        if _mapping(value, "support role").get("id") == "puf_tax_detail"
    )
    tail_support = _mapping(support_role["tail_support"], "tail support")
    primary_node = source_node_by_id["primary_puf_qrf"]
    primary_resource = next(
        _mapping(value, "primary virtual resource")
        for value in _array(
            primary_node["virtual_resources"], "primary virtual resources"
        )
        if _mapping(value, "primary virtual resource").get("id")
        == "tax_unit.@primary_puf_execution_config"
    )
    capital_gains = _mapping(
        _mapping(primary_resource["binding"], "primary binding")["capital_gains_tail"],
        "capital-gains tail",
    )
    calibration_tails = _mapping(calibration["tail_contracts"], "calibration tails")
    tail_bundle = {
        "spine": tail_support,
        "calibration": calibration_tails,
        "legacy": legacy.get("calibration_tail_contracts"),
    }
    tail_count = (
        _terminal_count(capital_gains)
        + _terminal_count(tail_support)
        + _terminal_count(calibration_tails)
    )
    authority_components = _mapping(authority.get("components"), "authority components")
    add(
        "capital_gains_tail_contract_exact",
        clauses={
            "tail terminal field count differs": tail_count == 934,
            "resolved calibration tails differ": _json_equal(
                legacy.get("calibration_tail_contracts"),
                expected_legacy["calibration_tail_contracts"],
            ),
            "tail authority identity differs": _mapping(
                authority_components.get("puf_capital_gains_tail_support_contract"),
                "tail authority component",
            ).get("identity")
            == tail_support.get("legacy_contract"),
            "checkpoint tail identity differs": pool_code.get(
                "puf_capital_gains_tail_support_contract"
            )
            == tail_support.get("legacy_contract"),
            "tail object digest differs": sha256_json(tail_bundle)
            == EXPECTED_HASHES["tail"],
        },
        homes=(
            "/spine/support_roles",
            "/imputation/producer_graph/nodes",
            "/calibration/tail_contracts",
        ),
        consumers=(
            "legacy_adapter.calibration_tail_contracts",
            "legacy_adapter.stacked_authority_receipt",
            "legacy_adapter.stacked_checkpoint_static_components",
        ),
        observed={"terminal_fields": tail_count, "sha256": sha256_json(tail_bundle)},
        expected={"terminal_fields": 934, "sha256": EXPECTED_HASHES["tail"]},
    )

    add(
        "stacked_authority_components_exact",
        clauses={
            "stacked authority differs from pure projection": _json_equal(
                authority, expected_authority
            ),
            "authority component order differs": tuple(authority_components)
            == EXPECTED_AUTHORITY_COMPONENTS,
            "component digest validation failed": all(
                _mapping(value, "authority component").get("digest_matches_declared")
                is True
                for value in authority_components.values()
            ),
        },
        homes=("/battery", "/imputation", "/spine"),
        consumers=("legacy_adapter.stacked_authority_receipt",),
        observed={"component_names": list(authority_components)},
        expected={"component_names": list(EXPECTED_AUTHORITY_COMPONENTS)},
    )
    add(
        "stacked_authority_identity_exact",
        clauses={
            "stacked authority SHA differs": authority.get("sha256")
            == EXPECTED_HASHES["authority"],
            "canonical authority is not valid": authority.get("canonical") is True
            and authority.get("integrity_valid") is True,
        },
        homes=("/battery/authority_binding",),
        consumers=(
            "legacy_adapter.stacked_authority_receipt",
            "legacy_adapter.stacked_checkpoint_static_components",
        ),
        observed={"sha256": authority.get("sha256")},
        expected={"sha256": EXPECTED_HASHES["authority"]},
    )
    gap_schedule = _mapping(
        actual_imputation.get("gap_fill_producer_schedule_receipt"), "gap schedule"
    )
    add(
        "gap_fill_schedule_receipt_exact",
        clauses={
            "gap-fill schedule differs": _json_equal(
                gap_schedule,
                expected_imputation.get("gap_fill_producer_schedule_receipt"),
            ),
            "gap-fill schedule digest differs": gap_schedule.get("sha256")
            == EXPECTED_HASHES["gap_fill_schedule"],
            "checkpoint gap-fill schedule differs": _json_equal(
                pool_code.get("gap_fill_producer_schedule"), gap_schedule
            ),
        },
        homes=("/imputation/gap_fill_schedule", "/imputation/families"),
        consumers=(
            "legacy_adapter.imputation.gap_fill_producer_schedule_receipt",
            "legacy_adapter.stacked_checkpoint_static_components",
        ),
        observed={"sha256": gap_schedule.get("sha256")},
        expected={"sha256": EXPECTED_HASHES["gap_fill_schedule"]},
    )
    add(
        "late_schedule_receipt_exact",
        clauses={
            "late schedule differs": _json_equal(late_schedule, expected_late_schedule),
            "late schedule semantic digest differs": late_schedule.get(
                "schedule_sha256"
            )
            == EXPECTED_HASHES["late_schedule"],
            "checkpoint late schedule differs": _json_equal(
                pool_code.get("late_producer_schedule"), late_schedule
            ),
        },
        homes=("/imputation/producer_graph", "/imputation/families"),
        consumers=(
            "legacy_adapter.imputation.late_producer_schedule_receipt",
            "legacy_adapter.stacked_checkpoint_static_components",
        ),
        observed={"schedule_sha256": late_schedule.get("schedule_sha256")},
        expected={"schedule_sha256": EXPECTED_HASHES["late_schedule"]},
    )

    checkpoint_input_pins = {
        "zeta": {"sha256": "b" * 64, "size_bytes": 23},
        "alpha": {"sha256": "a" * 64, "size_bytes": 17},
    }
    checkpoint_stack_receipt = {
        "sample_seed": 578,
        "sample_fraction": 0.25,
        "survey_samples": {
            "asec": {"realized_household_count": 3},
            "acs": {"realized_household_count": 5},
        },
        "unicode_probe": "Caf\u00e9",
    }
    full_checkpoint = project_stacked_checkpoint_base_identity(
        spec,
        input_pins=checkpoint_input_pins,
        stack_receipt=checkpoint_stack_receipt,
        sample_fraction=0.25,
        sample_seed=578,
        clone_attachment_fraction=0.4,
        clone_attachment_seed=991,
    )
    full_pool_code = _mapping(full_checkpoint.get("pool_code"), "full pool code")
    full_resource_semantics = _mapping(
        full_pool_code.get("late_producer_resource_semantics"),
        "full checkpoint late resource semantics",
    )
    primary_resource_rows = [
        _mapping(value, "full checkpoint resource-semantics producer")
        for value in _array(
            full_resource_semantics.get("producers"),
            "full checkpoint resource-semantics producers",
        )
        if _mapping(value, "full checkpoint resource-semantics producer").get(
            "producer"
        )
        == "primary_puf_qrf"
    ]
    primary_resource_row = (
        primary_resource_rows[0] if len(primary_resource_rows) == 1 else {}
    )
    primary_resources = _mapping(
        primary_resource_row.get("resources", {}),
        "full checkpoint primary resources",
    )
    primary_execution = _mapping(
        primary_resources.get("tax_unit.@primary_puf_execution_config", {}),
        "full checkpoint primary execution resource",
    )
    primary_binding = _mapping(
        primary_execution.get("binding", {}),
        "full checkpoint primary execution binding",
    )
    resolved_attachment = _mapping(
        primary_binding.get("clone_attachment", {}),
        "full checkpoint resolved clone attachment",
    )
    add(
        "stacked_checkpoint_base_identity_exact",
        clauses={
            "full checkpoint identity digest differs": _operational_free_sha256(
                full_checkpoint
            )
            == EXPECTED_HASHES["full_checkpoint"],
            "full checkpoint top-level fields differ": set(full_checkpoint)
            == EXPECTED_FULL_CHECKPOINT_TOP_LEVEL,
            "full checkpoint input pins differ": full_checkpoint.get("inputs")
            == {
                "alpha": {"sha256": "a" * 64, "size_bytes": 17},
                "zeta": {"sha256": "b" * 64, "size_bytes": 23},
            },
            "full checkpoint sampling vector differs": full_checkpoint.get("sampling")
            == {
                "sample_fraction": 0.25,
                "fraction_token": "f025",
                "sample_seed": 578,
                "stack_manifest_sha256": sha256_json(checkpoint_stack_receipt),
                "stack_manifest": checkpoint_stack_receipt,
            },
            "full checkpoint clone attachment differs": full_checkpoint.get(
                "clone_attachment"
            )
            == {"fraction": 0.4, "seed": 991},
            "full checkpoint resolved attachment differs": (
                resolved_attachment.get("fraction") == 0.4
                and resolved_attachment.get("seed") == 991
            ),
            "full checkpoint pool-code fields differ": set(full_pool_code)
            == EXPECTED_CHECKPOINT_POOL_CODE,
        },
        homes=(
            "/bundle/dataset_run",
            "/spine/pipeline_contract",
            "/spine/support_roles",
            "/imputation",
            "/publication/release/rung_fractions",
        ),
        consumers=(
            "stacked_authority_semantics.project_stacked_checkpoint_base_identity",
        ),
        observed={
            "sha256": _operational_free_sha256(full_checkpoint),
            "field_names": sorted(full_checkpoint),
            "input_roles": list(_mapping(full_checkpoint["inputs"], "full inputs")),
            "fraction_token": _mapping(
                full_checkpoint["sampling"], "full sampling"
            ).get("fraction_token"),
        },
        expected={
            "sha256": EXPECTED_HASHES["full_checkpoint"],
            "field_names": sorted(EXPECTED_FULL_CHECKPOINT_TOP_LEVEL),
            "input_roles": ["alpha", "zeta"],
            "fraction_token": "f025",
        },
    )

    add(
        "stacked_checkpoint_top_level_exact",
        clauses={
            "checkpoint static object differs": _json_equal(static, expected_static),
            "checkpoint top-level fields differ": set(static)
            == EXPECTED_CHECKPOINT_TOP_LEVEL,
            "stacked authority is not embedded exactly": _json_equal(
                static.get("stacked_authority"), authority
            ),
        },
        homes=("/bundle/dataset_run", "/spine/pipeline_contract", "/imputation"),
        consumers=("legacy_adapter.stacked_checkpoint_static_components",),
        observed={"field_names": sorted(static)},
        expected={"field_names": sorted(EXPECTED_CHECKPOINT_TOP_LEVEL)},
    )
    geography_authorities = _mapping(
        geography_assignment.get("authorities"),
        "checkpoint geography authorities",
    )
    geography_seed = _mapping(
        geography_assignment.get("seed"),
        "checkpoint geography seed",
    )
    add(
        "stacked_geography_assignment_exact",
        clauses={
            "checkpoint geography assignment differs": _json_equal(
                geography_assignment,
                expected_geography_assignment,
            ),
            "geography declaration is not embedded exactly": _json_equal(
                geography_assignment.get("declaration"),
                geography.get("assignment"),
            ),
            "geography assignment authorities differ": set(geography_authorities)
            == {
                "puma_ladder",
                "congressional_district_vintage_crosswalk",
            },
            "geography assignment seed differs": geography_seed
            == {
                "site": "legacy_puma_ladder",
                "stream": "geography_legacy",
                "value_source": "run_request.build_model_seed",
                "value": 0,
            },
            "geography assignment digest differs": _operational_free_sha256(
                geography_assignment
            )
            == EXPECTED_HASHES["geography_assignment"],
        },
        homes=(
            "/geography/assignment",
            "/sources/sources",
            "/vintages/records",
            "/spine/seed_site_bindings",
        ),
        consumers=(
            "legacy_adapter.stacked_checkpoint_static_components.geography_assignment",
        ),
        observed={
            "sha256": _operational_free_sha256(geography_assignment),
            "authority_roles": sorted(geography_authorities),
            "target_vintage": _mapping(
                geography_authorities.get("congressional_district_vintage_crosswalk"),
                "checkpoint geography crosswalk authority",
            ).get("target_vintage"),
        },
        expected={
            "sha256": EXPECTED_HASHES["geography_assignment"],
            "authority_roles": [
                "congressional_district_vintage_crosswalk",
                "puma_ladder",
            ],
            "target_vintage": "119th_congress",
        },
    )
    add(
        "stacked_checkpoint_pool_code_exact",
        clauses={
            "checkpoint pool-code fields differ": set(pool_code)
            == EXPECTED_CHECKPOINT_POOL_CODE,
            "checkpoint pool-code object differs": _json_equal(
                pool_code, _mapping(expected_static["pool_code"], "expected pool code")
            ),
        },
        homes=("/spine/pipeline_contract", "/imputation", "/take_up"),
        consumers=("legacy_adapter.stacked_checkpoint_static_components.pool_code",),
        observed={
            "field_names": sorted(pool_code),
            "sha256": _operational_free_sha256(pool_code),
        },
        expected={"field_names": sorted(EXPECTED_CHECKPOINT_POOL_CODE)},
    )

    protocol = spec.seed_protocol
    protocol_sites = {site.id: site.to_wire() for site in protocol.sites}
    compiled_sites = {site.id: site for site in compiled.seed_stream_map.sites}
    add(
        "seed_protocol_header_streams_exact",
        clauses={
            "seed protocol id differs": compiled.seed_stream_map.protocol_id
            == "legacy-v1",
            "seed implementation id differs": compiled.seed_stream_map.implementation_id
            == protocol.implementation_id,
            "seed implementation digest differs": compiled.seed_stream_map.implementation_sha256
            == protocol.implementation_sha256,
            "seed stream index differs": tuple(sorted(protocol.streams))
            == EXPECTED_SEED_STREAMS,
        },
        homes=("/bundle/seed_protocol",),
        consumers=("compiler_ir.seed_stream_map",),
        observed={
            "protocol": compiled.seed_stream_map.protocol_id,
            "streams": sorted(protocol.streams),
            "implementation_sha256": protocol.implementation_sha256,
        },
        expected={
            "protocol": "legacy-v1",
            "streams": list(EXPECTED_SEED_STREAMS),
            "implementation_sha256": EXPECTED_HASHES["seed_protocol"],
        },
    )
    site_definitions_exact = set(compiled_sites) == set(protocol_sites) and all(
        compiled_sites[site_id].stream
        == str(protocol_sites[site_id]["stream"]).removeprefix("stream:")
        and _json_equal(
            _wire(compiled_sites[site_id].contract),
            {
                key: value
                for key, value in protocol_sites[site_id].items()
                if key not in {"id", "stream"}
            },
        )
        for site_id in protocol_sites
    )
    add(
        "seed_site_definitions_exact",
        clauses={
            "seed site definitions differ": site_definitions_exact,
            "seed site count differs": len(protocol_sites) == 53,
        },
        homes=("/bundle/seed_protocol",),
        consumers=("compiler_ir.seed_stream_map.sites", "compiler_ir.node_slices"),
        observed={
            "sites": len(protocol_sites),
            "sha256": sha256_json([site.to_wire() for site in protocol.sites]),
        },
        expected={"sites": 53, "relation": "all site fields preserved exactly"},
    )
    binding_by_site = {binding.site: binding for binding in spec.seed_site_bindings}
    binding_exact = set(binding_by_site) == set(compiled_sites) and all(
        list(compiled_sites[site_id].owners)
        == [(owner.kind.value, owner.id) for owner in binding_by_site[site_id].owners]
        for site_id in compiled_sites
    )
    add(
        "seed_site_owner_bindings_exact",
        clauses={
            "site-owner bindings differ": binding_exact,
            "owner binding count differs": sum(
                len(site.owners) for site in compiled_sites.values()
            )
            == 112,
            "one or more sites have no owner": all(
                site.owners for site in compiled_sites.values()
            ),
        },
        homes=("/spine/seed_site_bindings",),
        consumers=("compiler_ir.seed_stream_map.sites.owners",),
        observed={
            "bindings": sum(len(site.owners) for site in compiled_sites.values())
        },
        expected={"bindings": 112, "coverage": "all 53 sites"},
    )
    expected_owner_sites: dict[tuple[str, str], list[str]] = {}
    for site in protocol.sites:
        binding = binding_by_site[site.id]
        for owner in binding.owners:
            expected_owner_sites.setdefault((owner.kind.value, owner.id), []).append(
                site.id
            )
    expected_owner_rows = []
    for (kind, owner_id), site_ids in sorted(expected_owner_sites.items()):
        streams = list(
            dict.fromkeys(protocol_sites[site_id]["stream"] for site_id in site_ids)
        )
        expected_owner_rows.append(
            {"kind": kind, "id": owner_id, "sites": site_ids, "streams": streams}
        )
    compiled_owner_rows = [owner.to_wire() for owner in compiled.seed_stream_map.owners]
    add(
        "seed_owner_rows_exact",
        clauses={
            "seed owner rows differ": _json_equal(
                compiled_owner_rows, expected_owner_rows
            ),
            "seed owner row count differs": len(compiled_owner_rows) == 54,
        },
        homes=("/spine/seed_site_bindings", "/bundle/seed_protocol"),
        consumers=("compiler_ir.seed_stream_map.owners",),
        observed={"owner_rows": len(compiled_owner_rows)},
        expected={
            "owner_rows": 54,
            "relation": "site-order-preserving owner aggregation",
        },
    )
    grouped_ids = [
        site_id for group in EXPECTED_SEED_GROUPS.values() for site_id in group
    ]
    groups_disjoint = len(grouped_ids) == len(set(grouped_ids))
    add(
        "seed_inventory_groups_exhaustive",
        clauses={
            "seed groups overlap": groups_disjoint,
            "seed groups do not cover the protocol exactly": set(grouped_ids)
            == set(protocol_sites),
            "seed group cardinality differs": len(grouped_ids) == 53,
        },
        homes=("/bundle/seed_protocol", "/spine/seed_site_bindings"),
        consumers=("compiler_ir.seed_stream_map",),
        observed={
            "groups": {key: list(value) for key, value in EXPECTED_SEED_GROUPS.items()},
            "sites": len(grouped_ids),
        },
        expected={
            "groups": len(EXPECTED_SEED_GROUPS),
            "sites": 53,
            "partition": "disjoint and exhaustive",
        },
    )
    add(
        "seed_protocol_and_owner_map_digests_exact",
        clauses={
            "seed protocol content digest differs": protocol.implementation_sha256
            == EXPECTED_HASHES["seed_protocol"],
            "compiled seed map digest differs": sha256_json(
                compiled.seed_stream_map.to_wire()
            )
            == EXPECTED_HASHES["seed_map"],
        },
        homes=("/bundle/seed_protocol", "/spine/seed_site_bindings"),
        consumers=("compiler_ir.seed_stream_map", "compiler_ir.node_slices"),
        observed={
            "protocol_sha256": protocol.implementation_sha256,
            "map_sha256": sha256_json(compiled.seed_stream_map.to_wire()),
        },
        expected={
            "protocol_sha256": EXPECTED_HASHES["seed_protocol"],
            "map_sha256": EXPECTED_HASHES["seed_map"],
        },
    )

    release = _mapping(publication["release"], "publication release")
    rung_rows = [
        _mapping(value, "release rung")
        for value in _array(release["rung_fractions"], "release rungs")
    ]
    observed_rungs = tuple(
        (float(row["fraction"]), int(row["percent_basis_points"]), str(row["token"]))
        for row in rung_rows
    )
    adapter_release = _mapping(legacy.get("publication_release"), "publication adapter")
    add(
        "release_rungs_exact",
        clauses={
            "release rungs differ": observed_rungs == EXPECTED_RUNGS,
            "adapter rung order differs": adapter_release.get("rungs")
            == [row[2] for row in EXPECTED_RUNGS],
            "publication projection differs": _json_equal(
                adapter_release, expected_legacy["publication_release"]
            ),
        },
        homes=("/publication/release/rung_fractions",),
        consumers=(
            "legacy_adapter.publication_release",
            "legacy_adapter.spine_sampling",
        ),
        observed={"rungs": [list(row) for row in observed_rungs]},
        expected={"rungs": [list(row) for row in EXPECTED_RUNGS]},
    )
    line = _mapping(release["line"], "publication release line")
    add(
        "release_line_and_regex_exact",
        clauses={
            "release line differs": line.get("value") == "microcosm-us-2024",
            "legacy prefix differs": line.get("legacy_prefixes")
            == ["populace-us-2024"],
            "compiled release regex differs": adapter_release.get("compiled_regex")
            == EXPECTED_RELEASE_REGEX,
            "legacy reader regex differs": adapter_release.get(
                "legacy_compiled_regexes"
            )
            == [EXPECTED_LEGACY_RELEASE_REGEX],
        },
        homes=("/publication/release",),
        consumers=("legacy_adapter.publication_release",),
        observed={
            "line": line.get("value"),
            "legacy_prefixes": line.get("legacy_prefixes"),
            "compiled_regex": adapter_release.get("compiled_regex"),
            "legacy_regexes": adapter_release.get("legacy_compiled_regexes"),
        },
        expected={
            "line": "microcosm-us-2024",
            "legacy_prefix": "populace-us-2024",
            "compiled_regex": EXPECTED_RELEASE_REGEX,
            "legacy_regex": EXPECTED_LEGACY_RELEASE_REGEX,
        },
    )

    counts = {
        "adapter_surfaces": len(legacy),
        "authority_components": len(authority_components),
        "early_families": len(early),
        "early_targets": early_targets,
        "itemization_batches": len(itemization),
        "itemization_targets": sum(itemization_sizes),
        "late_groups": len(late),
        "late_targets": late_targets,
        "ownership_rows": len(compiled.producer_graph.ownership_matrix),
        "primary_effective_predictor_tuples": len(primary_tuples),
        "primary_families": len(primary),
        "primary_targets": sum(
            len(_array(row["targets"], "primary targets")) for row in primary
        ),
        "producer_authored_outputs": authored_output_count,
        "producer_compiled_outputs": compiled.producer_graph.compiled_output_count,
        "producer_inputs": input_count,
        "producer_nodes": len(source_nodes),
        "producer_virtual_resources": virtual_count,
        "release_rungs": len(rung_rows),
        "resolved_references": len(spec.references),
        "seed_owner_bindings": sum(
            len(site.owners) for site in compiled.seed_stream_map.sites
        ),
        "seed_owner_rows": len(compiled.seed_stream_map.owners),
        "seed_sites": len(compiled.seed_stream_map.sites),
        "seed_streams": len(protocol.streams),
        "source_operators": len(operator_order),
        "source_stages": len(_array(sources["stages"], "source stages")),
        "stacked_checkpoint_pool_code_components": len(pool_code),
        "stacked_checkpoint_full_components": len(full_checkpoint),
        "stacked_checkpoint_static_components": len(static),
        "tail_control_fields": tail_count,
        "take_up_pipeline_steps": sum(len(value) for value in step_map.values()),
        "take_up_programs": len(programs),
        "typed_artifacts": len(spec.artifacts),
        "typed_columns": len(spec.columns),
        "typed_entities": len(spec.entities),
        "typed_scopes": len(spec.scopes),
    }
    missing = sorted(
        name for name, item in items.items() if item["status"] != "covered"
    )
    return {
        "report_schema_version": INVENTORY_REPORT_SCHEMA_VERSION,
        "country": spec.country,
        "spec_binding": spec.spec_binding.to_wire(),
        "compiler_ir_abi": compiled.compiler_ir_abi.to_wire(),
        "required_item_count": len(items),
        "covered_item_count": len(items) - len(missing),
        "missing_item_count": len(missing),
        "missing_items": missing,
        "items": items,
        "counts": counts,
    }


def assert_inventory_coverage_complete(report: Mapping[str, object]) -> None:
    """Recompute and validate every summary instead of trusting pass labels."""

    failures: list[str] = []
    expected_top_level = {
        "report_schema_version",
        "country",
        "spec_binding",
        "compiler_ir_abi",
        "required_item_count",
        "covered_item_count",
        "missing_item_count",
        "missing_items",
        "items",
        "counts",
    }
    if set(report) != expected_top_level:
        failures.append("inventory top-level fields differ")
    if report.get("report_schema_version") != INVENTORY_REPORT_SCHEMA_VERSION:
        failures.append(
            "inventory report schema version differs: "
            f"{report.get('report_schema_version')!r}"
        )
    if report.get("country") != "us":
        failures.append(f"inventory country differs: {report.get('country')!r}")

    binding = report.get("spec_binding")
    if not isinstance(binding, Mapping):
        failures.append("inventory spec_binding is missing")
    else:
        expected_binding_fields = {
            "attestation",
            "canonicalizer_version",
            "country",
            "schema_id",
            "schema_version",
            "spec_sha256",
        }
        if set(binding) != expected_binding_fields:
            failures.append("inventory spec_binding fields differ")
        if (
            binding.get("attestation") != "mirror-attested"
            or binding.get("canonicalizer_version") != 1
            or binding.get("country") != "us"
            or binding.get("schema_id") != "country_spec"
            or binding.get("schema_version") != 1
        ):
            failures.append("inventory spec_binding contract differs")
        spec_sha256 = binding.get("spec_sha256")
        if (
            not isinstance(spec_sha256, str)
            or len(spec_sha256) != 64
            or any(character not in "0123456789abcdef" for character in spec_sha256)
        ):
            failures.append("inventory spec_binding SHA-256 is invalid")

    compiler_abi = report.get("compiler_ir_abi")
    if not isinstance(compiler_abi, Mapping):
        failures.append("inventory compiler_ir_abi is missing")
    else:
        if (
            set(compiler_abi) != {"version", "sha256"}
            or compiler_abi.get("version") != 1
        ):
            failures.append("inventory compiler IR ABI contract differs")
        abi_sha256 = compiler_abi.get("sha256")
        if (
            not isinstance(abi_sha256, str)
            or len(abi_sha256) != 64
            or any(character not in "0123456789abcdef" for character in abi_sha256)
        ):
            failures.append("inventory compiler IR ABI SHA-256 is invalid")

    items = report.get("items")
    if not isinstance(items, Mapping):
        failures.append("inventory items are missing")
        items = {}
    if set(items) != EXPECTED_INVENTORY_ITEMS:
        failures.append(
            "inventory item registry differs: "
            f"expected {len(EXPECTED_INVENTORY_ITEMS)}, got {len(items)}"
        )

    recomputed_missing: list[str] = []
    expected_item_fields = {
        "status",
        "bundle_homes",
        "bundle_home_match_counts",
        "compiler_consumers",
        "observed",
        "expected",
        "failures",
    }
    for raw_name, raw_item in sorted(items.items(), key=lambda row: str(row[0])):
        name = str(raw_name)
        if not isinstance(raw_item, Mapping):
            failures.append(f"{name}: inventory item is not an object")
            recomputed_missing.append(name)
            continue
        if set(raw_item) != expected_item_fields:
            failures.append(f"{name}: inventory item fields differ")
        item_failures = raw_item.get("failures")
        if not isinstance(item_failures, list) or any(
            not isinstance(value, str) for value in item_failures
        ):
            failures.append(f"{name}: failures must be a string array")
            item_failures = ["malformed failures"]
        recomputed_status = "missing" if item_failures else "covered"
        if raw_item.get("status") != recomputed_status:
            failures.append(f"{name}: status does not match its failed clauses")
        if recomputed_status == "missing":
            recomputed_missing.append(name)

        homes = raw_item.get("bundle_homes")
        match_counts = raw_item.get("bundle_home_match_counts")
        if (
            not isinstance(homes, list)
            or not homes
            or any(
                not isinstance(home, str) or not home.startswith("/") for home in homes
            )
        ):
            failures.append(f"{name}: bundle_homes are malformed or empty")
            homes = []
        if not isinstance(match_counts, Mapping) or set(match_counts) != set(homes):
            failures.append(f"{name}: bundle-home match evidence is incomplete")
        elif any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in match_counts.values()
        ):
            failures.append(f"{name}: a bundle home matched zero values")

        consumers = raw_item.get("compiler_consumers")
        if (
            not isinstance(consumers, list)
            or not consumers
            or any(not isinstance(value, str) or not value for value in consumers)
        ):
            failures.append(f"{name}: compiler_consumers are malformed or empty")
        if not isinstance(raw_item.get("observed"), Mapping):
            failures.append(f"{name}: observed evidence is not an object")
        if not isinstance(raw_item.get("expected"), (Mapping, str)):
            failures.append(f"{name}: expected evidence is malformed")

    recomputed_missing.sort()
    recomputed_required = len(items)
    recomputed_covered = recomputed_required - len(recomputed_missing)
    if recomputed_missing:
        failures.append(
            "inventory has missing required items: " + ", ".join(recomputed_missing)
        )
    expected_summaries = {
        "required_item_count": recomputed_required,
        "covered_item_count": recomputed_covered,
        "missing_item_count": len(recomputed_missing),
        "missing_items": recomputed_missing,
    }
    for field, expected in expected_summaries.items():
        if report.get(field) != expected:
            failures.append(
                f"inventory {field} differs: expected {expected!r}, "
                f"got {report.get(field)!r}"
            )

    if report.get("counts") != EXPECTED_INVENTORY_COUNTS:
        failures.append("inventory diagnostic counts differ from the reviewed vector")

    if failures:
        item_details = [
            f"{name}: {items[name].get('failures', [])!r}"
            for name in recomputed_missing
            if isinstance(items.get(name), Mapping)
        ]
        raise InventoryCoverageError(
            "spec-engine inventory coverage failed:\n- "
            + "\n- ".join([*failures, *item_details])
        )


__all__ = [
    "InventoryCoverageError",
    "assert_inventory_coverage_complete",
    "build_inventory_coverage",
]
