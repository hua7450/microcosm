"""The release artifact contract: what a published release MUST contain.

The releases already on the Hub disagree with each other — one carries no
``build_manifest.json`` at all, and two different ``release_manifest.json``
schemas coexist (an unversioned early shape next to ``schema_version: 1``).
A consumer iterating ``releases/`` therefore cannot trust the listing, and
every consumer ends up re-implementing its own defensive filter. The charter
makes "stage manifests are load-bearing" a binding process rule; the release
directory is the most public manifest of all, so its contract lives here,
with the producer — not in every consumer.

:func:`validate_release_dir` is the single gate: it checks a local release
directory against the contract and raises :class:`ReleaseContractError`
naming **every** failure at once (a publisher should see the full repair
list, not play whack-a-mole one failure per run). Publishing code calls it
before any byte reaches the Hub.

Two publication tiers share this module (microcosm#506). The **certified**
tier is :func:`validate_release_dir`, unchanged. The **evidence** tier is
:func:`validate_evidence_release_dir`, a sibling — not a relaxation — for
the best-available artifact when terminal gates failed: the same required
files, the same shape and provenance checks, plus a mandatory non-empty
``known_failures`` block carrying every recorded gate failure verbatim with
an owner issue. The tiers are structurally mutually exclusive: an evidence
manifest declares :data:`EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION`, which
the certified contract rejects, and a certified manifest carries no
``known_failures``, which the evidence contract requires. (Distinct from
the UK terminal-gate *evidence receipts* checked below: those attest how a
certified verdict was reached; the evidence *tier* publishes an artifact
whose verdicts failed.)
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from microcosm.data.denied_pools import denied_pool_publication_for
from microcosm.data.us_critical_targets import (
    US_CRITICAL_TARGET_FIT_REQUIREMENTS as _US_CRITICAL_TARGET_FIT_REQUIREMENTS,
)
from microcosm.data.us_critical_targets import (
    US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR as _US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR,
)
from microcosm.data.us_critical_targets import (
    is_congressional_district_target,
)

__all__ = [
    "EVIDENCE_RELEASE_ID_SEGMENT",
    "EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION",
    "LOCAL_AREA_REQUIRED_RELEASE_FILES",
    "NATIONAL_DEFAULT_DATASET_ROLE",
    "NON_DEFAULT_LOCAL_AREA_DATASET_ROLE",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "REQUIRED_RELEASE_FILES",
    "US_SOURCE_COVERAGE_DIAGNOSTICS_FILE",
    "ReleaseContractError",
    "release_dataset_role",
    "required_release_files",
    "validate_evidence_release_dir",
    "validate_release_dir",
]

#: The release-manifest schema this library reads and writes. Bump it with the
#: schema, and keep :func:`validate_release_dir` rejecting drift loudly — the
#: unversioned 1abddeb-era manifest is exactly the silence this guards against.
RELEASE_MANIFEST_SCHEMA_VERSION = 1

#: The release-manifest schema marker for EVIDENCE-tier releases
#: (microcosm#506). Deliberately a distinct value, not a superset flag on the
#: certified schema: the certified contract rejects any manifest carrying it,
#: so an evidence artifact can never be mistaken for (or promoted as) a
#: certified one, no matter which gates happened to fail.
EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION = "1-evidence"

#: Evidence release ids must carry this segment (e.g.
#: ``populace-us-2024-evidence-<sha>-<date>``) so the tier is visible in the
#: artifact name itself — in Hub tags, download paths, and logs.
EVIDENCE_RELEASE_ID_SEGMENT = "-evidence-"

#: Files a release directory must contain to count as published. A release
#: missing any of these is invisible to :func:`validate_release_dir`-respecting
#: publishers, by design.
REQUIRED_RELEASE_FILES = (
    "build_manifest.json",
    "release_manifest.json",
    "calibration_diagnostics.json",
)

# Dataset-role classes (microcosm#398). The national default keeps the full
# historical contract; a non-default local-area artifact gets its own
# contract and can never move the latest.json pointer.
NATIONAL_DEFAULT_DATASET_ROLE = "national_default"
NON_DEFAULT_LOCAL_AREA_DATASET_ROLE = "non_default_local_area"

#: Files a non-default local-area release directory must carry. The source
#: coverage entry matches :data:`US_SOURCE_COVERAGE_DIAGNOSTICS_FILE`.
LOCAL_AREA_REQUIRED_RELEASE_FILES = (
    "build_manifest.json",
    "release_manifest.json",
    "calibration_diagnostics.json",
    "gate_summary.json",
    "us_source_coverage.json",
    "sha256sums.txt",
)

#: Provenance-chain keys the local-area source coverage must carry (the
#: local product's analog of the national fiscal_target_sources map).
LOCAL_AREA_SOURCE_COVERAGE_KEYS = (
    "acs_sources",
    "geography_ladder",
    "transfer_coverage",
    "donor_release",
)

# Lockstep with microcosm.calibrate.diagnostics.CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
# (schema 6 = final per-target loss attribution plus warning-only degradation).
# microcosm-data cannot import
# microcosm-calibrate (dependency direction), so the builder test suite pins the
# two constants equal — see test_calibration_diagnostics_schema_lockstep.
CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 6
US_SOURCE_COVERAGE_DIAGNOSTICS_FILE = "us_source_coverage.json"
SOURCE_COVERAGE_DIAGNOSTICS_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UK_EXACT_K_RELEASE_ID_RE = re.compile(
    r"^populace-uk-(?P<year>[1-9][0-9]*)-(?P<tier>.+)-k(?P<record_count>[1-9][0-9]*)$"
)
_UK_JUNE_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
_UK_LEGACY_RELEASE_IDS = frozenset({_UK_JUNE_RELEASE_ID})
_UK_RELEASE_TIERS = frozenset({"frs", "cps-transfer"})
_UK_DIAGNOSTICS_SCHEMA_VERSION = 1
_UK_TERMINAL_GATE_REPORT_FILE = "terminal_gates.json"
_UK_TERMINAL_GATE_SCHEMA_VERSION = 3
_UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION = 5
_UK_TERMINAL_GATE_PRODUCER = (
    "microcosm.build.uk_runtime.terminal_gates.uk_terminal_gate_report"
)
_UK_TERMINAL_GATE_SIGNATURE_ALGORITHM = "hmac-sha256"
_UK_TERMINAL_GATE_SIGNING_KEY_ENV = "POPULACE_UK_TERMINAL_GATE_SIGNING_KEY"
# Lockstep with
# microcosm.build.uk_runtime.terminal_gates.UK_TERMINAL_GATE_POLICY_SHA256.  The
# data shard deliberately does not depend on the build shard: publication must
# independently pin the reviewed gate policy rather than trust a producer's
# self-description.  The pinned digest is the certified *default* policy; the
# increment-4 weighted-integrity slots (#609) are unarmed in it, so a release
# that arms them with measured thresholds requires a reviewed move of this pin
# alongside the committed threshold constants — a threshold outside this hash
# is not attested.
_UK_TERMINAL_GATE_POLICY_SHA256 = (
    "ae93bd10a02362a523eb077bcbd32b362cef31f0447acbc40537df696e30c757"
)
# The published June release attests the pre-#630 policy (no degenerate
# reviewed exclusions). Its report is immutable, so the superseded digest
# stays pinned for exactly the grandfathered release ids — a vintage-aware
# pin, never a loosening: every release still matches exactly one reviewed
# policy digest. Today the terminal-report checker only runs for exact-k
# release ids, so this branch is defensive; it becomes load-bearing the day
# grandfathered reports are checked.
_UK_TERMINAL_GATE_POLICY_SHA256_LEGACY = (
    "74c9cd474d76e2b8d4ca5b298c19fc6348ac1a90746594afc8a81283a0398b68"
)
_UK_ALWAYS_APPLICABLE_GATE_NAMES = (
    "uk_release_input_coverage",
    "degenerate_release_surface",
    "zero_weight_strata",
    "weight_ess",
    "weight_ratio",
)
_UK_TERMINAL_EVIDENCE_GATE_NAMES = {
    "hmrc_spi_income": ("weights_audit",),
    "release_parity": ("export_surface", "target_surface", "target_fit"),
    "input_mass_parity": ("input_mass_parity",),
    "qrf_tail_concentration": ("qrf_tail_concentration",),
}
_UK_TERMINAL_EVIDENCE_STAGES = frozenset(
    {"release_dataset", *_UK_TERMINAL_EVIDENCE_GATE_NAMES}
)
_UK_WEIGHT_SUMMARY_FIELDS = (
    "n_records",
    "positive_weight_records",
    "zero_weight_records",
    "total_weight",
    "effective_sample_size",
    "ess_fraction",
    "median_positive_weight",
    "max_weight",
    "max_to_median_positive_weight",
    "top_1pct_weight_share",
)
_UK_MIN_ESS_FRACTION = 0.01
_UK_MAX_TO_MEDIAN_WEIGHT_RATIO = 1_151.2542195939373
_UK_MAX_TARGET_ABS_RELATIVE_ERROR = 0.25
# Spec-armed weighted-integrity thresholds (uk/gates.json parameters;
# input-mass pair from microcosm#630, QRF tail pair re-armed from the #686
# L3 baselines by microcosm#757 B4): passing reports must carry exactly the
# committed values,
# so a re-signed report cannot loosen a fence the spec armed. Held in
# lockstep with the committed spec by the build-shard sync tests.
_UK_INPUT_MASS_RELATIVE_TOLERANCE = 4.521811483823806
_UK_INPUT_MASS_MINIMUM_REFERENCE_TOTAL = 0.0
_UK_QRF_TAIL_TOP_K = 100
_UK_QRF_TAIL_MAX_TOP_SHARE = 0.9994670564654868
_UK_QRF_TAIL_MIN_NONZERO_RECORDS = 104
# Independent publication pin for the active reviewed reference source. The data
# shard cannot import the build shard, so keep this in lockstep with
# uk/gates.json reference_registry["efrs-post-calibration"].identity.
_UK_INPUT_MASS_ACTIVE_REFERENCE = "efrs-post-calibration"
_UK_INPUT_MASS_REFERENCE_IDENTITY = {
    "filename": "enhanced_frs_2024_25.h5",
    "revision": "a9e52499b6a6cca100a5ce4f36ca27b2e8a213df",
    "sha256": "e433e532b17bd8ce76030156285816e33d44e93edabd2204adbef71d19a68712",
    "vintage": "2024_25",
}
# Independent publication pin for the canonical
# {"reference": {"identity": ..., "totals": ...}} evidence emitted from the
# reviewed 131-column enhanced-FRS reference. The totals remain uncommitted
# under the UKDS EUL; keep this in lockstep with
# UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256 in the build shard.
_UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256 = (
    "fd41cb5f6cf6c4ef812320f21d1942173d49ce6f8725b21fbc9d9ca5423d298c"
)
_UK_TERMINAL_GATE_DETAIL_FIELDS = {
    "uk_release_input_coverage": frozenset(
        {
            "required_columns",
            "present_columns",
            "missing",
            "degenerate_required",
            "reviewed_exclusions",
            "stale_exclusions",
            "dormant_exclusions",
        }
    ),
    "degenerate_release_surface": frozenset(
        {
            "columns_checked",
            "findings",
            "all_null_columns",
            "all_zero_columns",
            "constant_columns",
            "reviewed_exclusions",
            "stale_exclusions",
            "dormant_exclusions",
            "expired_exclusions",
            "premature_exclusions",
            "exclusions_evaluated_on",
        }
    ),
    "zero_weight_strata": frozenset(
        {
            "household_rows",
            "zero_weight_rows",
            "declared_strata",
            "unmatched_zero_weight_rows",
            "unmatched_household_examples",
            "ambiguous_zero_weight_rows",
            "ambiguous_household_examples",
        }
    ),
    "weight_ess": frozenset({*_UK_WEIGHT_SUMMARY_FIELDS, "minimum_ess_fraction"}),
    "weight_ratio": frozenset(
        {*_UK_WEIGHT_SUMMARY_FIELDS, "maximum_max_to_median_ratio"}
    ),
    "weights_audit": frozenset(
        {
            "fits_checked",
            "resolved_weight_kinds",
            "unweighted_fits",
            "allowed_unweighted",
            "unused_allowed_unweighted",
        }
    ),
    "export_surface": frozenset(
        {
            "candidate_columns",
            "reference_columns",
            "missing_reference_columns",
            "unexpected_candidate_columns",
            "forbidden_candidate_columns",
        }
    ),
    "target_surface": frozenset(
        {
            "candidate_targets",
            "reference_targets",
            "extra_candidate_targets",
            "missing_reference_targets",
        }
    ),
    "target_fit": frozenset(
        {
            "targets_checked",
            "max_abs_relative_error",
            "failing_targets",
            "reviewed_exclusions",
            "stale_exclusions",
            "dormant_exclusions",
            "expired_exclusions",
            "premature_exclusions",
            "exclusions_evaluated_on",
        }
    ),
    "aggregate_vs_admin": frozenset({"anchors_checked"}),
    "input_mass_parity": frozenset(
        {
            "candidate_name",
            "reference_name",
            "relative_tolerance",
            "minimum_reference_total",
            "columns_checked",
            "columns_below_reference_floor",
            "candidate_only_columns",
            "worst_drifts",
            "reviewed_exclusions",
            "unused_reviewed_exclusions",
            "stale_exclusions",
            "dormant_exclusions",
            "expired_exclusions",
            "premature_exclusions",
            "exclusions_evaluated_on",
            "reference",
            "reference_scope_note",
            "reference_identity",
        }
    ),
    "qrf_tail_concentration": frozenset(
        {
            "columns_checked",
            "top_k",
            "max_top_share",
            "min_nonzero_records",
            "top_share",
            "carrier_counts",
            "thin_columns",
            "reviewed_exclusions",
            "stale_exclusions",
            "dormant_exclusions",
            "expired_exclusions",
            "premature_exclusions",
            "exclusions_evaluated_on",
            "surface",
        }
    ),
    "support": frozenset({"columns_checked"}),
}
_UK_TARGET_GEOGRAPHY_LEVELS = frozenset(
    {"national", "region", "country", "local_authority", "constituency"}
)

# ---------------------------------------------------------------------------
# Schema-4 gate-battery verification. Every constant here mirrors the shared
# executor (microcosm.build.gate_battery) or the committed UK spec by name;
# the data shard deliberately does not import the build shard, and the
# build-shard sync tests hold each mirror in lockstep.
# ---------------------------------------------------------------------------
_UK_GATE_BATTERY_SCHEMA_VERSION = 4
_UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION = 6
_UK_GATE_BATTERY_PRODUCER = "microcosm.build.gate_battery"
# gate_signing_key_env("uk") in the build shard; the legacy POPULACE variable
# stays with the schema-3 path above.
_UK_GATE_BATTERY_SIGNING_KEY_ENV = "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY"
_UK_GATE_BATTERY_PHASES = ("preflight", "assembled", "transferred", "terminal")
_UK_GATE_BATTERY_STATUSES = frozenset(
    {"passed", "failed", "not_applicable", "evidence_absent", "unreached"}
)
_UK_GATE_BATTERY_SHIPPABLE_STATUSES = frozenset({"passed", "not_applicable"})
# Vintage pins over the committed uk/gates.json: the manifest digest covers
# phase order and notes, the policy digest the reviewed thresholds, and the
# fingerprint derives from the manifest digest. Editing the spec moves all
# three here in the same reviewed change.
_UK_GATE_BATTERY_POLICY_SHA256 = (
    "8deb610d25fced25e27be58989311efa217f49e6d644f37268cf0f7efd122193"
)
_UK_GATE_BATTERY_GATES_MANIFEST_SHA256 = (
    "cd90537532e1800a76d4414a7d7454f5dedb4dfd574aca2dfca4ec1e1fe6cff3"
)
_UK_GATE_BATTERY_SPEC_FINGERPRINT = (
    "8b28c7d23b9e8ebc9e7436abd7309e59f0ef14fb582506f874147a5c1c4f44ef"
)
#: Spec entry id -> the legacy gate name whose observable detail checks
#: apply unchanged (the battery re-keys the report by entry id; the gate
#: implementations and their detail schemas are the same code).
_UK_GATE_BATTERY_ENTRY_LEGACY_NAMES = {
    "uk_release_input_coverage": "uk_release_input_coverage",
    "uk_degenerate_release_surface": "degenerate_release_surface",
    "uk_zero_weight_strata": "zero_weight_strata",
    "uk_weight_ess": "weight_ess",
    "uk_weight_ratio": "weight_ratio",
    "uk_weights_audit": "weights_audit",
    "uk_nonnegative_columns": "nonnegative_columns",
    "uk_uc_capital_coherence": "column_implication",
    "uk_support": "support",
    "uk_aggregate_admin": "aggregate_vs_admin",
    "uk_export_surface": "export_surface",
    "uk_take_up_signal": "take_up_signal",
    "uk_brma_enum_domain": "enum_domain",
    "uk_uc_deduction_combination_enum_domain": "enum_domain",
    "uk_student_loan_plan_enum_domain": "enum_domain",
    "uk_target_surface": "target_surface",
    "uk_target_fit": "target_fit",
    "uk_input_mass_parity": "input_mass_parity",
    "uk_qrf_tail_concentration": "qrf_tail_concentration",
}
#: Spec entry id -> (gate, phase), mirrored per entry so a report cannot
#: relabel an entry's identity.
_UK_GATE_BATTERY_ENTRY_GATES = {
    "uk_release_input_coverage_manifest_current": (
        "release_input_coverage",
        "preflight",
    ),
    "uk_release_family_build_stages": ("source_coverage", "preflight"),
    "uk_ledger_compile_parity_production_2023": (
        "ledger_compile_parity",
        "preflight",
    ),
    "uk_ledger_compile_parity_incumbent_2025": (
        "ledger_compile_parity",
        "preflight",
    ),
    "uk_stage_was_wealth_support": ("stage_health", "transferred"),
    "uk_stage_uc_deduction_attributes": ("stage_health", "transferred"),
    "uk_stage_lcfs_consumption_support": ("stage_health", "transferred"),
    "uk_stage_etb_vat_support": ("stage_health", "transferred"),
    "uk_stage_etb_services_support": ("stage_health", "transferred"),
    "uk_stage_frs_hmrc_spine_leaves_signal": (
        "stage_health",
        "transferred",
    ),
    "uk_stage_spi_support_channel_mass": ("stage_health", "transferred"),
    "uk_stage_hmrc_spi_income_spine_identity": (
        "stage_health",
        "transferred",
    ),
    "uk_stage_cgt_incidence_clone_mass": ("stage_health", "transferred"),
    "uk_stage_cgt_band_donors_support": ("stage_health", "transferred"),
    "uk_stage_hmrc_cgt_gains_spine_summary": (
        "stage_health",
        "transferred",
    ),
    "uk_stage_salary_sacrifice_realization": (
        "stage_health",
        "transferred",
    ),
    "uk_stage_student_loans_realization": ("stage_health", "transferred"),
    "uk_stage_age_tail_targets": ("stage_health", "assembled"),
    "uk_ledger_compile_parity_local_incumbent_2025": (
        "ledger_compile_parity",
        "preflight",
    ),
    "uk_target_surface_local_default_2025": ("target_surface", "preflight"),
    "uk_release_input_coverage": ("release_input_coverage", "terminal"),
    "uk_degenerate_release_surface": ("degenerate_release_surface", "terminal"),
    "uk_zero_weight_strata": ("zero_weight_strata", "terminal"),
    "uk_weight_ess": ("weight_ess", "terminal"),
    "uk_weight_ratio": ("weight_ratio", "terminal"),
    "uk_weights_audit": ("weights_audit", "terminal"),
    "uk_nonnegative_columns": ("nonnegative_columns", "terminal"),
    "uk_uc_capital_coherence": ("column_implication", "terminal"),
    "uk_support": ("support", "terminal"),
    "uk_aggregate_admin": ("aggregate_admin", "terminal"),
    "uk_export_surface": ("export_surface", "terminal"),
    "uk_take_up_signal": ("take_up_signal", "terminal"),
    "uk_brma_enum_domain": ("enum_domain", "assembled"),
    "uk_uc_deduction_combination_enum_domain": ("enum_domain", "terminal"),
    "uk_student_loan_plan_enum_domain": ("enum_domain", "terminal"),
    "uk_calibration_reference_coverage": (
        "calibration_reference_coverage",
        "terminal",
    ),
    "uk_target_surface": ("target_surface", "terminal"),
    "uk_target_fit": ("target_fit", "terminal"),
    "uk_input_mass_parity": ("input_mass_parity", "terminal"),
    "uk_qrf_tail_concentration": ("tail_concentration", "terminal"),
    "uk_local_geography_ladder_post_calibration": (
        "spine_agreement",
        "terminal",
    ),
    "uk_local_area_support": ("area_support", "terminal"),
    "uk_local_target_fit": ("target_fit", "terminal"),
    "uk_local_per_family_fit": ("per_family_fit", "terminal"),
    "uk_local_weight_ratio": ("weight_ratio", "terminal"),
    "uk_local_weight_ess": ("weight_ess", "terminal"),
}
_UK_GATE_BATTERY_ENTRY_IDS = frozenset(_UK_GATE_BATTERY_ENTRY_GATES)
_UK_GATE_BATTERY_DIAGNOSTIC_IDS = frozenset()
#: The entries whose bindings contribute an evidence digest; their keys are
#: the only ones a schema-4 ``evidence_sha256`` may carry, and each appears
#: exactly when its entry evaluated.
_UK_GATE_BATTERY_EVIDENCE_IDS = frozenset(
    {
        "uk_release_family_build_stages",
        "uk_ledger_compile_parity_production_2023",
        "uk_ledger_compile_parity_incumbent_2025",
        "uk_ledger_compile_parity_local_incumbent_2025",
        "uk_target_surface_local_default_2025",
        "uk_degenerate_release_surface",
        "uk_input_mass_parity",
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
    }
)
# The input-mass binding's evidence payload wraps the reviewed reference
# digest as {"reference_evidence_sha256": ...} before the executor's
# canonical hash; this pins the wrapped digest so the entry's evidence line
# still binds the enhanced-FRS incumbent totals.
_UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256 = (
    "c9211cbb923e13f4850b834b5bdb1ff1de87fe9237c332b5de63f01ed417aa2d"
)
# The degenerate binding's evidence payload digests the resolved exclusion
# records; for a release that must be the committed register, so its digest
# is a vintage pin that moves with every reviewed register edit (the same
# tripwire as the policy digest).
_UK_GATE_BATTERY_DEGENERATE_EVIDENCE_SHA256 = (
    "6f0243bcda09dad26945376230c44ec3cf55d4e417c3a25e29bae8c59bc1a69d"
)


# --- UK release certification (microcosm#757 item B5) ----------------------
# The multi-part certification the release-cut producer composes: the spine
# build's battery report, the calibration seam's battery report, and the
# release-cut battery report union to the full declared gate-entry set with
# no gap and no overlap beyond the declared shared id. The data shard cannot
# import the build shard, so the part scopes, phases, and scoped-manifest
# digests are hand-mirrored here and held in lockstep by the build-shard
# sync tests (test_gate_battery_contract_pins). The phase and digest checks
# that _check_uk_gate_battery_report applies to one unfiltered report apply
# here per-certification (the 5413502559 audit's nine refusal points).
_UK_RELEASE_CERTIFICATION_FILE = "release_certification.json"
_UK_RELEASE_CUT_GATE_REPORT_FILE = "release_cut_gates.json"
# The certified national line's constant release id (ruling 2026-08-27):
# ordering and run identity live in the Logbook and versioning, so the id
# stays fixed across cuts. Mirrored from
# microcosm.build.uk_runtime.release_identity.UK_NATIONAL_RELEASE_ID (the
# data shard cannot import the build shard); lockstep-tested.
_UK_NATIONAL_RELEASE_ID = "microcosm-uk-2024-25-national"
# The dense joint national + local line (microcosm#762 A18, ruling
# 2026-09-03): the same constant-id approach, published on the inspect lane
# under the non-default local-area role. Mirrored from
# microcosm.build.uk_runtime.release_identity.UK_DENSE_RELEASE_ID; lockstep-
# tested in test_gate_battery_contract_pins.py.
_UK_DENSE_RELEASE_ID = "microcosm-uk-2024-25-dense"
_UK_DENSE_CUT_TAG_RE = re.compile(
    re.escape(_UK_DENSE_RELEASE_ID) + r"-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}"
)
_UK_DENSE_GATE_REPORT_FILE = "uk_local_gates.json"
_UK_DENSE_SCORE_RECEIPT_FILE = "score_vs_incumbent.json"
_UK_DENSE_SOURCE_COVERAGE_FILE = "uk_source_coverage.json"
_UK_DENSE_NAMESPACE = "uk_dense"
_UK_DENSE_REQUIRED_RELEASE_FILES = (
    "build_manifest.json",
    "release_manifest.json",
    "calibration_diagnostics.json",
    "gate_summary.json",
    _UK_DENSE_SOURCE_COVERAGE_FILE,
    _UK_DENSE_GATE_REPORT_FILE,
    _UK_DENSE_SCORE_RECEIPT_FILE,
    "incumbent_surface_evaluation.json",
    "rowwise_candidate_manifest.json",
    "incumbent_manifest.json",
    "source_calibration_diagnostics.json",
    "sha256sums.txt",
)
# The local battery scope (microcosm.build.uk_runtime.calibration_run
# .UK_LOCAL_GATE_SCOPE) and its scoped spec digests, mirrored and
# lockstep-tested against the live scoped manifest.
_UK_DENSE_GATE_ENTRY_IDS = frozenset(
    {
        "uk_local_area_support",
        "uk_local_geography_ladder_post_calibration",
        "uk_local_per_family_fit",
        "uk_local_target_fit",
        "uk_local_weight_ess",
        "uk_local_weight_ratio",
    }
)
_UK_DENSE_RELEASE_BLOCKING_IDS = _UK_DENSE_GATE_ENTRY_IDS
_UK_DENSE_GATE_PHASES = ("terminal",)
_UK_DENSE_GATE_DIGESTS = {
    "gates_manifest_sha256": "c900204bbc59bb58d5cf0fb0e5c703236f82618c0ebc737279e49774e6641604",
    "policy_sha256": "ad63a0aa05ac127d45a42d82c5a0e376f23cf0f708c7b6d25958479b3483bbeb",
    "spec_fingerprint": "33ca1741241fe936d5148bf170b4d040d640df070857822c7d490382778243f0",
}
_UK_DENSE_SOURCE_COVERAGE_KEYS = (
    "spine",
    "ledger_artifact",
    "geography_ladder",
    "incumbent",
    "doctrine",
    "measure_exclusions",
    "signed_deferrals",
    "holdout",
    "uprating",
)
# The per-cut tag grammar the assembler mints from the calibration attempt id
# (tools/assemble_uk_release_dir.py). The contract validates the same shape so
# a hand-edited or stale revision cannot claim a cut the attempt chain never
# produced.
_UK_NATIONAL_REVISION_SUFFIX_RE = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")
# The canonical release-dir filenames of the evidence the certification signs
# (tools/assemble_uk_release_dir.py copies each byte-for-byte). Validation
# binds every local file to its signed digest: a certification whose evidence
# was removed or rewritten must refuse, not validate around the gap.
_UK_CERTIFICATION_PART_EVIDENCE_FILES: Mapping[str, str] = {
    "spine": "spine_gates.json",
    "calibration_seam": "terminal_gates.json",
    "release_cut": "release_cut_gates.json",
}
_UK_CERTIFICATION_SCORE_RECEIPT_FILE = "score_vs_enhanced_frs.json"
_UK_RELEASE_CERTIFICATION_SCHEMA_VERSION = 1
_UK_RELEASE_CERTIFICATION_KIND = "uk_release_certification"
_UK_CERTIFICATION_SHARED_GATE_IDS = frozenset({"uk_aggregate_admin"})
_UK_CERTIFICATION_EXCLUDED_GATE_IDS = frozenset(
    {
        "uk_local_geography_ladder_post_calibration",
        "uk_local_area_support",
        "uk_local_target_fit",
        "uk_local_per_family_fit",
        "uk_local_weight_ratio",
        "uk_local_weight_ess",
    }
)
_UK_CERTIFICATION_PART_PHASES: Mapping[str, tuple[str, ...]] = {
    "spine": ("assembled", "transferred"),
    "calibration_seam": ("terminal",),
    "release_cut": ("preflight", "terminal"),
}
_UK_CERTIFICATION_PART_SCOPES: Mapping[str, frozenset[str]] = {
    "spine": frozenset(
        {
            "uk_brma_enum_domain",
            "uk_stage_age_tail_targets",
            "uk_stage_cgt_band_donors_support",
            "uk_stage_cgt_incidence_clone_mass",
            "uk_stage_etb_services_support",
            "uk_stage_etb_vat_support",
            "uk_stage_frs_hmrc_spine_leaves_signal",
            "uk_stage_hmrc_cgt_gains_spine_summary",
            "uk_stage_hmrc_spi_income_spine_identity",
            "uk_stage_lcfs_consumption_support",
            "uk_stage_salary_sacrifice_realization",
            "uk_stage_spi_support_channel_mass",
            "uk_stage_student_loans_realization",
            "uk_stage_uc_deduction_attributes",
            "uk_stage_was_wealth_support",
        }
    ),
    "calibration_seam": frozenset(
        {
            "uk_aggregate_admin",
            "uk_calibration_reference_coverage",
            "uk_target_fit",
            "uk_weight_ess",
            "uk_weight_ratio",
            "uk_zero_weight_strata",
        }
    ),
    "release_cut": frozenset(
        {
            "uk_aggregate_admin",
            "uk_degenerate_release_surface",
            "uk_export_surface",
            "uk_input_mass_parity",
            "uk_ledger_compile_parity_incumbent_2025",
            "uk_ledger_compile_parity_local_incumbent_2025",
            "uk_target_surface_local_default_2025",
            "uk_ledger_compile_parity_production_2023",
            "uk_nonnegative_columns",
            "uk_uc_capital_coherence",
            "uk_uc_deduction_combination_enum_domain",
            "uk_qrf_tail_concentration",
            "uk_release_family_build_stages",
            "uk_release_input_coverage",
            "uk_release_input_coverage_manifest_current",
            "uk_student_loan_plan_enum_domain",
            "uk_support",
            "uk_take_up_signal",
            "uk_target_surface",
            "uk_weights_audit",
        }
    ),
}
_UK_CERTIFICATION_PART_DIGESTS: Mapping[str, Mapping[str, str]] = {
    "spine": {
        "gates_manifest_sha256": (
            "70a47a57039a236ae67df7d019fc7f8d43cbffab14a47d00a1e3e28dbc83a5a3"
        ),
        "policy_sha256": (
            "21e6b68e013cfc33a5bf96e141c42fd6fb23a34a237ae71a9e5f806a958552c8"
        ),
    },
    "calibration_seam": {
        "gates_manifest_sha256": (
            "cb0d7ce17c0cd3cf70bd432d4ba85efce3fa837ebf3caba5f5cf0b5545c5dc61"
        ),
        "policy_sha256": (
            "290b1ad240bf4f6412dcaa87c77283dad79d88c817b10b2f402736378fd3d63d"
        ),
    },
    "release_cut": {
        "gates_manifest_sha256": (
            "79a1bd35f18ced387d988a1285aa26afb9404c635f69f32204aa020559d4a0e3"
        ),
        "policy_sha256": (
            "e9cdea8c7dad2ee582a6b71df0d5f25eed142f52bd352e29683a4977f6ac69d9"
        ),
    },
}
_UK_CERTIFICATION_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "country",
        "release_id",
        "candidate",
        "parts",
        "spec",
        "doctrine",
        "diagnostics_sha256",
        "score_receipt",
        "exclusions_evaluated_on",
        "shippable",
        "attestation",
    }
)


def required_release_files(release_id: str) -> tuple[str, ...]:
    """Files required for a release id's country-specific contract."""
    if release_id.startswith("populace-us-"):
        return (*REQUIRED_RELEASE_FILES, US_SOURCE_COVERAGE_DIAGNOSTICS_FILE)
    if _is_uk_exact_k_release_id(release_id):
        return (*REQUIRED_RELEASE_FILES, _UK_TERMINAL_GATE_REPORT_FILE)
    if release_id == _UK_NATIONAL_RELEASE_ID:
        # The certified national line ships its composed verdict: the
        # shippability claim lives only in the certification, so a national
        # release without one is refused at the required-files layer, not
        # just when part artifacts happen to be present.
        return (*REQUIRED_RELEASE_FILES, _UK_RELEASE_CERTIFICATION_FILE)
    return REQUIRED_RELEASE_FILES


class ReleaseContractError(ValueError):
    """A release directory violates the release contract.

    Attributes:
        failures: Every contract violation found, each a self-contained
            human-readable sentence naming the file and field at fault.
    """

    def __init__(self, release_dir: Path, failures: list[str]) -> None:
        self.failures = list(failures)
        bullet_list = "\n".join(f"  - {failure}" for failure in self.failures)
        super().__init__(
            f"Release directory {release_dir} violates the release contract "
            f"({len(self.failures)} failure(s)):\n{bullet_list}"
        )


def _load_json(path: Path, failures: list[str]) -> Mapping | None:
    try:
        loaded = json.loads(
            path.read_text(),
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        failures.append(f"{path.name} is not valid JSON: {exc}.")
        return None
    if not isinstance(loaded, Mapping):
        failures.append(
            f"{path.name} must be a JSON object, got {type(loaded).__name__}."
        )
        return None
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    """Hash JSON exactly as the UK gate aggregator's attestation does."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _uk_release_key_from_env(env_var: str, failures: list[str]) -> bytes | None:
    """Load an out-of-band trust root used to authenticate UK reports."""

    encoded = os.environ.get(env_var)
    if not encoded:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} verification requires "
            f"{env_var} to contain the release key."
        )
        return None
    try:
        key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        failures.append(
            f"{env_var} must be valid base64 to verify {_UK_TERMINAL_GATE_REPORT_FILE}."
        )
        return None
    if len(key) != 32:
        failures.append(
            f"{env_var} must decode to exactly 32 bytes "
            f"to verify {_UK_TERMINAL_GATE_REPORT_FILE}."
        )
        return None
    return key


def _uk_terminal_verification_key(failures: list[str]) -> bytes | None:
    """The legacy aggregator's trust root (schema-3 reports)."""

    return _uk_release_key_from_env(_UK_TERMINAL_GATE_SIGNING_KEY_ENV, failures)


def _uk_gate_battery_verification_key(failures: list[str]) -> bytes | None:
    """The shared executor's trust root (schema-4 reports)."""

    return _uk_release_key_from_env(_UK_GATE_BATTERY_SIGNING_KEY_ENV, failures)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def _check_sha256_field(
    *,
    filename: str,
    owner: str,
    value: object,
    failures: list[str],
) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        failures.append(f"{filename} {owner} must be a 64-character lowercase sha256.")


def _check_target_surface_ref(
    surface: object,
    *,
    filename: str,
    owner: str,
    failures: list[str],
) -> None:
    if not isinstance(surface, Mapping):
        failures.append(f"{filename} is missing {owner} target_surface object.")
        return
    _check_sha256_field(
        filename=filename,
        owner=f"{owner}.target_surface.sha256",
        value=surface.get("sha256"),
        failures=failures,
    )
    n_targets = surface.get("n_targets")
    if not isinstance(n_targets, int) or n_targets <= 0:
        failures.append(f"{filename} {owner}.target_surface.n_targets must be > 0.")


def _check_target_registry_ref(
    registry: object,
    *,
    filename: str,
    owner: str,
    failures: list[str],
) -> None:
    if not isinstance(registry, Mapping):
        failures.append(f"{filename} is missing {owner} target_registry object.")
        return
    if not registry.get("version"):
        failures.append(f"{filename} {owner}.target_registry.version is required.")
    n_specs = registry.get("n_specs")
    if not isinstance(n_specs, int) or n_specs <= 0:
        failures.append(f"{filename} {owner}.target_registry.n_specs must be > 0.")


def _check_base_pool_not_denied(manifest: Mapping, failures: list[str]) -> None:
    """A release assembled from a denied pool may not be published, however it
    reached the release directory (microcosm#856)."""

    base_pool = manifest.get("base_pool")
    if not isinstance(base_pool, Mapping):
        return
    match = denied_pool_publication_for(
        publication_run_id=base_pool.get("publication_run_id"),
        manifest_sha256=base_pool.get("manifest_sha256"),
        pool_h5_sha256=base_pool.get("pool_h5_sha256"),
        content_identity_sha256=base_pool.get("content_identity_sha256"),
    )
    if match is not None:
        run_id, denied, how = match
        failures.append(
            f"build_manifest.json base_pool is denied publication {run_id!r} "
            f"(matched by {how}; release_id {denied.release_id!r}) and cannot be "
            f"published. Reason: {denied.reason}. Reference: {denied.reference}."
        )


def _check_build_manifest(
    manifest: Mapping, release_id: str, failures: list[str]
) -> None:
    _check_base_pool_not_denied(manifest, failures)
    build_id = manifest.get("build_id")
    if not build_id:
        failures.append("build_manifest.json is missing 'build_id'.")
    elif build_id != release_id:
        failures.append(
            f"build_manifest.json 'build_id' is {build_id!r} but the release "
            f"directory is named {release_id!r}; the directory name IS the "
            f"build id."
        )
    code = manifest.get("code")
    if not isinstance(code, Mapping):
        failures.append(
            "build_manifest.json is missing the 'code' object (repository, "
            "git_commit, git_dirty)."
        )
    else:
        if not code.get("repository"):
            failures.append("build_manifest.json 'code.repository' is required.")
        git_commit = code.get("git_commit")
        if not isinstance(git_commit, str) or not _GIT_COMMIT_RE.fullmatch(git_commit):
            failures.append(
                "build_manifest.json 'code.git_commit' must be a full "
                "40-character lowercase git commit."
            )
        if code.get("git_dirty") is not False:
            failures.append(
                "build_manifest.json 'code.git_dirty' must be false for a "
                "publishable release."
            )
        build_sha = manifest.get("build_sha")
        if isinstance(build_sha, str) and isinstance(git_commit, str):
            if not git_commit.startswith(build_sha):
                failures.append(
                    "build_manifest.json 'build_sha' must be a prefix of "
                    "'code.git_commit'."
                )
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        failures.append(
            "build_manifest.json is missing the 'runtime' object "
            "(Python and package versions used for target materialization)."
        )
    else:
        required_packages = ["python", "policyengine-core"]
        model_package = _expected_model_package(release_id)
        if model_package is not None:
            required_packages.append(model_package)
        for package in required_packages:
            value = runtime.get(package)
            if not value or value in {"not-installed", "unknown"}:
                failures.append(
                    f"build_manifest.json 'runtime.{package}' must be a resolved "
                    "version, not missing or unknown."
                )
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        failures.append("build_manifest.json is missing the 'dataset' object.")
    else:
        if not dataset.get("filename"):
            failures.append("build_manifest.json 'dataset' is missing 'filename'.")
        _check_sha256_field(
            filename="build_manifest.json",
            owner="'dataset.sha256'",
            value=dataset.get("sha256"),
            failures=failures,
        )
    calibration = manifest.get("calibration")
    if not isinstance(calibration, Mapping):
        failures.append("build_manifest.json is missing the 'calibration' object.")
    else:
        if not calibration.get("filename"):
            failures.append("build_manifest.json 'calibration.filename' is required.")
        _check_sha256_field(
            filename="build_manifest.json",
            owner="'calibration.sha256'",
            value=calibration.get("sha256"),
            failures=failures,
        )
        _check_target_surface_ref(
            calibration.get("target_surface"),
            filename="build_manifest.json",
            owner="'calibration'",
            failures=failures,
        )
        _check_target_registry_ref(
            calibration.get("target_registry"),
            filename="build_manifest.json",
            owner="'calibration'",
            failures=failures,
        )
    gates = manifest.get("gates")
    if not isinstance(gates, Mapping):
        failures.append(
            "build_manifest.json is missing the 'gates' object (the "
            "acceptance-gate verdicts are the point of the manifest)."
        )
    if _is_uk_exact_k_release_id(release_id):
        _check_uk_terminal_build_manifest(manifest, failures)
    _check_uk_exact_k_manifest_fields(
        manifest,
        release_id,
        filename="build_manifest.json",
        count_fields=("n_records",),
        failures=failures,
    )


def _check_uk_terminal_build_manifest(
    manifest: Mapping,
    failures: list[str],
) -> None:
    """Require exact-k UK builds to bind gate evidence and the report bytes."""

    evidence = manifest.get("terminal_gate_evidence")
    if not isinstance(evidence, Mapping):
        failures.append(
            "build_manifest.json canonical UK releases require a "
            "'terminal_gate_evidence' object."
        )
    else:
        # Two evidence vocabularies, never mixed: the legacy aggregator keys
        # by evidence stage (release_dataset always present); the gate
        # battery keys by evidence-bearing spec entry id. The report checker
        # for the matching schema holds the manifest and the report equal.
        stages = set(map(str, evidence))
        if stages and stages <= _UK_GATE_BATTERY_EVIDENCE_IDS:
            pass  # battery vocabulary; membership is entry-conditional
        else:
            missing = sorted({"release_dataset"} - stages)
            unexpected = sorted(
                str(stage)
                for stage in stages
                - _UK_TERMINAL_EVIDENCE_STAGES
                - _UK_GATE_BATTERY_EVIDENCE_IDS
            )
            mixed = sorted(stages & _UK_GATE_BATTERY_EVIDENCE_IDS)
            if missing:
                failures.append(
                    "build_manifest.json terminal_gate_evidence is missing "
                    f"always-applicable stage(s): {missing}."
                )
            if unexpected:
                failures.append(
                    "build_manifest.json terminal_gate_evidence has unknown "
                    f"stage(s): {unexpected}."
                )
            if mixed:
                legacy_seen = sorted(stages & _UK_TERMINAL_EVIDENCE_STAGES)
                failures.append(
                    "build_manifest.json terminal_gate_evidence mixes the "
                    f"legacy stage vocabulary {legacy_seen} with battery "
                    f"entry ids {mixed}; one build produces one report under "
                    "one schema, so a manifest never straddles the two."
                )
        for stage, digest in evidence.items():
            _check_sha256_field(
                filename="build_manifest.json",
                owner=f"terminal_gate_evidence[{stage!r}]",
                value=digest,
                failures=failures,
            )

    gates = manifest.get("gates")
    terminal = gates.get("uk_terminal") if isinstance(gates, Mapping) else None
    if not isinstance(terminal, Mapping):
        failures.append(
            "build_manifest.json canonical UK gates must include an "
            "'uk_terminal' report pointer."
        )
        return
    if terminal.get("passed") is not True:
        failures.append("build_manifest.json gates.uk_terminal.passed must be true.")
    if terminal.get("path") != _UK_TERMINAL_GATE_REPORT_FILE:
        failures.append(
            "build_manifest.json gates.uk_terminal.path must be "
            f"{_UK_TERMINAL_GATE_REPORT_FILE!r}."
        )
    _check_sha256_field(
        filename="build_manifest.json",
        owner="gates.uk_terminal.sha256",
        value=terminal.get("sha256"),
        failures=failures,
    )


def _check_release_manifest(
    manifest: Mapping,
    release_id: str,
    failures: list[str],
    *,
    expected_schema_version: object = RELEASE_MANIFEST_SCHEMA_VERSION,
) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version is None:
        failures.append(
            "release_manifest.json has no 'schema_version'; unversioned "
            "manifests (the 1abddeb-era shape) are not publishable."
        )
    elif schema_version != expected_schema_version:
        failures.append(
            f"release_manifest.json 'schema_version' is {schema_version!r}; "
            f"this library publishes version "
            f"{expected_schema_version!r}."
        )
    build = manifest.get("build")
    if not isinstance(build, Mapping) or not build.get("build_id"):
        failures.append("release_manifest.json is missing 'build.build_id'.")
    elif build["build_id"] != release_id:
        failures.append(
            f"release_manifest.json 'build.build_id' is "
            f"{build['build_id']!r} but the release directory is named "
            f"{release_id!r}."
        )
    _check_uk_release_identity(manifest, release_id, failures)
    if isinstance(build, Mapping):
        built_with_core_package = build.get("built_with_core_package")
        built_with_model_package = build.get("built_with_model_package")
        _check_release_manifest_package(
            built_with_core_package,
            field="build.built_with_core_package",
            expected_name="policyengine-core",
            failures=failures,
        )
        _check_release_manifest_package(
            built_with_model_package,
            field="build.built_with_model_package",
            expected_name=_expected_model_package(release_id),
            failures=failures,
        )
        _check_compatible_package_entries(
            manifest.get("compatible_core_packages"),
            field="compatible_core_packages",
            expected_name="policyengine-core",
            built_with_package=built_with_core_package,
            failures=failures,
        )
        _check_compatible_package_entries(
            manifest.get("compatible_model_packages"),
            field="compatible_model_packages",
            expected_name=_expected_model_package(release_id),
            built_with_package=built_with_model_package,
            failures=failures,
        )
    data_package = manifest.get("data_package")
    _check_release_manifest_package(
        data_package,
        field="data_package",
        # Legacy releases were recorded under the pre-rename package name.
        expected_name=("microcosm-data", "populace-data"),
        failures=failures,
    )
    default_datasets = manifest.get("default_datasets")
    if not isinstance(default_datasets, Mapping):
        failures.append(
            "release_manifest.json is missing the 'default_datasets' object."
        )
    elif not default_datasets.get("national"):
        failures.append(
            "release_manifest.json 'default_datasets.national' is required."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        failures.append(
            "release_manifest.json must declare a non-empty 'artifacts' mapping."
        )
    else:
        diagnostics_artifact = artifacts.get("calibration_diagnostics")
        if not isinstance(diagnostics_artifact, Mapping):
            failures.append(
                "release_manifest.json artifacts must include "
                "'calibration_diagnostics'."
            )
        elif diagnostics_artifact.get("path") != "calibration_diagnostics.json":
            failures.append(
                "release_manifest.json artifact 'calibration_diagnostics' "
                "must point to calibration_diagnostics.json."
            )
        for key, entry in artifacts.items():
            if not isinstance(entry, Mapping):
                failures.append(
                    f"release_manifest.json artifact {key!r} must be an object."
                )
                continue
            for field in ("kind", "path", "repo_id", "revision", "sha256"):
                if not entry.get(field):
                    failures.append(
                        f"release_manifest.json artifact {key!r} is missing {field!r}."
                    )
            revision = entry.get("revision")
            revision_matches_release = revision == release_id or (
                release_id == _UK_NATIONAL_RELEASE_ID
                and isinstance(revision, str)
                and revision.startswith(release_id + "-")
                and _UK_NATIONAL_REVISION_SUFFIX_RE.fullmatch(
                    revision[len(release_id) + 1 :]
                )
                is not None
            )
            # A present-but-non-string revision must fail here rather than
            # slide past the isinstance guard: publish collects only string
            # revisions, so a numeric revision would otherwise vanish into an
            # empty pin set and publish under a dangling tag.
            if revision and (
                not isinstance(revision, str) or not revision_matches_release
            ):
                expected = (
                    f"the release id {release_id!r} or a "
                    f"'{release_id}-<YYYYMMDDTHHMMSSZ>-<uuid8>' per-cut tag"
                    if release_id == _UK_NATIONAL_RELEASE_ID
                    else f"the release id {release_id!r}"
                )
                failures.append(
                    f"release_manifest.json artifact {key!r} revision is "
                    f"{revision!r}, expected {expected}."
                )
            if isinstance(entry, Mapping):
                _check_sha256_field(
                    filename="release_manifest.json",
                    owner=f"artifact {key!r}.sha256",
                    value=entry.get("sha256"),
                    failures=failures,
                )
        # One release pins one revision: individually grammar-valid revisions
        # from two different cuts must refuse here, not later at publish.
        distinct_revisions = sorted(
            {
                entry.get("revision")
                for entry in artifacts.values()
                if isinstance(entry, Mapping)
                and isinstance(entry.get("revision"), str)
                and entry.get("revision")
            }
        )
        if len(distinct_revisions) > 1:
            failures.append(
                "release_manifest.json artifacts pin more than one revision "
                f"({distinct_revisions}); every artifact must pin the same "
                "release revision."
            )
        if release_id.startswith("populace-us-"):
            _check_us_release_has_no_split_microdata_artifacts(
                artifacts,
                failures=failures,
            )
        if (
            release_id.startswith("populace-us-")
            and _artifact_by_path(manifest, US_SOURCE_COVERAGE_DIAGNOSTICS_FILE) is None
        ):
            failures.append(
                "release_manifest.json artifacts must include "
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE!r} for US releases."
            )
        if _is_uk_exact_k_release_id(release_id):
            terminal_artifact = _artifact_by_path(
                manifest, _UK_TERMINAL_GATE_REPORT_FILE
            )
            if terminal_artifact is None:
                failures.append(
                    "release_manifest.json artifacts must include "
                    f"{_UK_TERMINAL_GATE_REPORT_FILE!r} for canonical UK "
                    "releases."
                )
            elif terminal_artifact.get("kind") != "diagnostics":
                failures.append(
                    "release_manifest.json terminal gate report artifact "
                    "must have kind 'diagnostics'."
                )
        if isinstance(default_datasets, Mapping):
            national = default_datasets.get("national")
            if isinstance(national, str) and national not in artifacts:
                failures.append(
                    "release_manifest.json 'default_datasets.national' "
                    f"points to {national!r}, which is not an artifact key."
                )
            elif isinstance(national, str):
                default_artifact = artifacts.get(national)
                if (
                    isinstance(default_artifact, Mapping)
                    and default_artifact.get("kind") != "microdata"
                ):
                    failures.append(
                        "release_manifest.json 'default_datasets.national' "
                        f"points to artifact {national!r}, whose kind is "
                        f"{default_artifact.get('kind')!r}, not 'microdata'."
                    )


def _check_uk_release_identity(
    manifest: Mapping,
    release_id: str,
    failures: list[str],
) -> None:
    """Enforce canonical UK identity while narrowly grandfathering old ids."""

    if not release_id.startswith("populace-uk-"):
        return

    match = _UK_EXACT_K_RELEASE_ID_RE.fullmatch(release_id)
    if match is None:
        if release_id not in _UK_LEGACY_RELEASE_IDS:
            failures.append(
                "release_manifest.json UK release id is neither canonical "
                "'populace-uk-<year>-<tier>-k<N>' nor the grandfathered June "
                f"hash/timestamp id: {release_id!r}."
            )
            return
        # This one known artifact predates the tier contract and does not need
        # a re-cut. Its recorded construction is FRS-derived, so an added tier
        # may only make that lineage explicit; it cannot relabel the artifact.
        legacy_tier = manifest.get("tier")
        if legacy_tier not in (None, "frs"):
            failures.append(
                "release_manifest.json grandfathered UK 'tier', when present, "
                f"must be 'frs' for the known FRS lineage, got {legacy_tier!r}."
            )
        return

    release_id_tier = match.group("tier")
    if release_id_tier not in _UK_RELEASE_TIERS:
        failures.append(
            "release_manifest.json exact-k UK release id has unratified tier "
            f"{release_id_tier!r}; expected one of {sorted(_UK_RELEASE_TIERS)}."
        )

    manifest_tier = manifest.get("tier")
    if manifest_tier is None:
        failures.append(
            "release_manifest.json exact-k UK releases require top-level 'tier'."
        )
    elif not isinstance(manifest_tier, str) or manifest_tier not in _UK_RELEASE_TIERS:
        failures.append(
            "release_manifest.json exact-k UK 'tier' must be one of "
            f"{sorted(_UK_RELEASE_TIERS)}, got {manifest_tier!r}."
        )
    if isinstance(manifest_tier, str) and manifest_tier != release_id_tier:
        failures.append(
            f"release_manifest.json top-level 'tier' is {manifest_tier!r} but "
            f"the release id names tier {release_id_tier!r}."
        )
    _check_uk_exact_k_manifest_fields(
        manifest,
        release_id,
        filename="release_manifest.json",
        count_fields=("record_count", "n_records"),
        failures=failures,
    )


def _is_uk_exact_k_release_id(release_id: str) -> bool:
    return _UK_EXACT_K_RELEASE_ID_RE.fullmatch(release_id) is not None


def _check_uk_exact_k_manifest_fields(
    manifest: Mapping,
    release_id: str,
    *,
    filename: str,
    count_fields: tuple[str, ...],
    failures: list[str],
) -> None:
    """Bind canonical UK manifest identity fields to the exact-k id."""

    match = _UK_EXACT_K_RELEASE_ID_RE.fullmatch(release_id)
    if match is None:
        return
    expected_year = int(match.group("year"))
    expected_records = int(match.group("record_count"))
    if manifest.get("country") != "uk":
        failures.append(
            f"{filename} canonical UK 'country' must be 'uk', got "
            f"{manifest.get('country')!r}."
        )
    year = manifest.get("year")
    if isinstance(year, bool) or not isinstance(year, int) or year != expected_year:
        failures.append(
            f"{filename} canonical UK 'year' must equal release-id year "
            f"{expected_year}, got {year!r}."
        )
    for field in count_fields:
        value = manifest.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != expected_records
        ):
            failures.append(
                f"{filename} canonical UK {field!r} must equal release-id "
                f"record count {expected_records}, got {value!r}."
            )


def _check_uk_exact_k_diagnostics_identity(
    diagnostics: Mapping,
    release_id: str,
    failures: list[str],
) -> None:
    """Reconcile the exact-k id with every shipped-record diagnostic."""

    match = _UK_EXACT_K_RELEASE_ID_RE.fullmatch(release_id)
    if match is None:
        return
    expected = int(match.group("record_count"))
    observations: list[tuple[str, object]] = [
        ("top-level n_records", diagnostics.get("n_records")),
    ]
    uk = diagnostics.get("uk_diagnostics")
    if isinstance(uk, Mapping):
        weights = uk.get("weights")
        if isinstance(weights, Mapping):
            observations.append(
                ("uk_diagnostics.weights.n_records", weights.get("n_records"))
            )
    target_surface = diagnostics.get("target_surface")
    if isinstance(target_surface, Mapping) and "n_records" in target_surface:
        observations.append(("target_surface.n_records", target_surface["n_records"]))
    for field, value in observations:
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            failures.append(
                "calibration_diagnostics.json canonical UK "
                f"{field} must equal release-id record count {expected}, got "
                f"{value!r}."
            )


def _check_us_release_has_no_split_microdata_artifacts(
    artifacts: Mapping, *, failures: list[str]
) -> None:
    split_keys = {
        key
        for key in artifacts
        if isinstance(key, str)
        and (key.startswith("states/") or key.startswith("districts/"))
    }
    if not split_keys:
        return
    failures.append(
        "release_manifest.json US releases must publish a single national "
        "microdata artifact; split state/district microdata artifacts are no "
        f"longer part of the release contract: {_sample_values(sorted(split_keys))}"
    )


def _sample_values(values: list[object], limit: int = 10) -> list[object]:
    sample = list(values[:limit])
    if len(values) > limit:
        sample.append(f"... +{len(values) - limit} more")
    return sample


def _check_release_manifest_package(
    package: object,
    *,
    field: str,
    expected_name: str | tuple[str, ...] | None,
    failures: list[str],
) -> None:
    if not isinstance(package, Mapping):
        failures.append(f"release_manifest.json is missing the '{field}' object.")
        return
    name = package.get("name")
    expected_names = (
        (expected_name,) if isinstance(expected_name, str) else expected_name
    )
    if expected_names is not None and name not in expected_names:
        rendered = " or ".join(repr(n) for n in expected_names)
        failures.append(f"release_manifest.json '{field}.name' must be {rendered}.")
    elif not name:
        failures.append(f"release_manifest.json '{field}.name' is required.")
    version = package.get("version")
    if not version:
        failures.append(f"release_manifest.json '{field}.version' is required.")
    elif version in {"not-installed", "unknown"}:
        failures.append(
            f"release_manifest.json '{field}.version' must be resolved, "
            f"not {version!r}."
        )


def _check_compatible_package_entries(
    entries: object,
    *,
    field: str,
    expected_name: str | None,
    built_with_package: object,
    failures: list[str],
) -> None:
    if expected_name is None:
        return
    if not isinstance(entries, list) or not entries:
        failures.append(
            f"release_manifest.json must declare a non-empty '{field}' list."
        )
        return

    matching_specifiers: list[str] = []
    for index, entry in enumerate(entries):
        owner = f"release_manifest.json {field}[{index}]"
        if not isinstance(entry, Mapping):
            failures.append(f"{owner} must be an object.")
            continue
        name = entry.get("name")
        specifier = entry.get("specifier")
        if not isinstance(name, str) or not name:
            failures.append(f"{owner}.name is required.")
        if not isinstance(specifier, str) or not specifier.strip():
            failures.append(f"{owner}.specifier is required.")
            continue
        try:
            SpecifierSet(specifier)
        except InvalidSpecifier:
            failures.append(
                f"{owner}.specifier {specifier!r} is not a valid PEP 440 specifier."
            )
            continue
        if name == expected_name:
            matching_specifiers.append(specifier)

    if not matching_specifiers:
        failures.append(
            f"release_manifest.json '{field}' must include {expected_name!r}."
        )
        return

    built_version = (
        built_with_package.get("version")
        if isinstance(built_with_package, Mapping)
        and built_with_package.get("name") == expected_name
        else None
    )
    if not isinstance(built_version, str) or not built_version:
        return
    try:
        version = Version(built_version)
    except InvalidVersion:
        built_field = (
            "build.built_with_core_package"
            if field == "compatible_core_packages"
            else "build.built_with_model_package"
        )
        failures.append(
            "release_manifest.json "
            f"{built_field}.version {built_version!r} is not a valid PEP 440 "
            "version."
        )
        return
    if not any(version in SpecifierSet(specifier) for specifier in matching_specifiers):
        failures.append(
            f"release_manifest.json '{field}' must include the built "
            f"{expected_name} version {built_version!r}."
        )


def _expected_model_package(release_id: str) -> str | None:
    if release_id.startswith("populace-us-"):
        return "policyengine-us"
    if release_id == _UK_NATIONAL_RELEASE_ID:
        return "policyengine-uk"
    if release_id.startswith("populace-uk-"):
        return "policyengine-uk"
    return None


def _check_local_artifact_hashes(
    release_dir: Path,
    release_manifest: Mapping | None,
    failures: list[str],
) -> None:
    if release_manifest is None:
        return
    artifacts = release_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return
    for key, entry in artifacts.items():
        if not isinstance(entry, Mapping):
            continue
        path = entry.get("path")
        expected_sha = entry.get("sha256")
        if not isinstance(path, str) or not isinstance(expected_sha, str):
            continue
        local = release_dir / path
        if not local.is_file():
            continue
        observed_sha = _sha256(local)
        if observed_sha != expected_sha:
            failures.append(
                f"release_manifest.json artifact {key!r} declares sha256 "
                f"{expected_sha} for local file {path!r}, but observed "
                f"{observed_sha}."
            )


def _artifact_by_path(release_manifest: Mapping, path: str) -> Mapping | None:
    artifacts = release_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    for artifact in artifacts.values():
        if isinstance(artifact, Mapping) and artifact.get("path") == path:
            return artifact
    return None


def _check_uk_terminal_gate_links(
    *,
    build_manifest: Mapping | None,
    release_manifest: Mapping | None,
    report_sha256: str,
    failures: list[str],
) -> None:
    """Cross-link both manifests to the exact terminal-report bytes."""

    build_terminal: object = None
    if build_manifest is not None:
        gates = build_manifest.get("gates")
        if isinstance(gates, Mapping):
            build_terminal = gates.get("uk_terminal")
    build_sha: object = None
    if isinstance(build_terminal, Mapping):
        build_sha = build_terminal.get("sha256")
        if build_sha != report_sha256:
            failures.append(
                "build_manifest.json gates.uk_terminal.sha256 must match the "
                f"local {_UK_TERMINAL_GATE_REPORT_FILE} bytes."
            )

    release_artifact: Mapping | None = None
    if release_manifest is not None:
        release_artifact = _artifact_by_path(
            release_manifest, _UK_TERMINAL_GATE_REPORT_FILE
        )
    if release_artifact is not None:
        release_sha = release_artifact.get("sha256")
        if release_sha != report_sha256:
            failures.append(
                "release_manifest.json terminal gate report artifact sha256 "
                f"must match the local {_UK_TERMINAL_GATE_REPORT_FILE} bytes."
            )
        if (
            isinstance(build_sha, str)
            and isinstance(release_sha, str)
            and (build_sha != release_sha)
        ):
            failures.append(
                "build_manifest.json and release_manifest.json terminal gate "
                "report sha256 values must match."
            )


def _uk_terminal_observable_matches(observed: object, expected: object) -> bool:
    """Compare JSON observables without letting booleans masquerade as counts."""

    if isinstance(observed, bool) or isinstance(expected, bool):
        return type(observed) is type(expected) and observed == expected
    if isinstance(observed, int | float) and isinstance(expected, int | float):
        return (
            math.isfinite(float(observed))
            and math.isfinite(float(expected))
            and (
                math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
    return observed == expected


def _uk_terminal_gate_details(
    gates: Mapping,
    name: str,
) -> Mapping | None:
    gate = gates.get(name)
    if not isinstance(gate, Mapping):
        return None
    details = gate.get("details")
    return details if isinstance(details, Mapping) else None


def _check_uk_terminal_gate_observables(
    gates: Mapping,
    *,
    calibration_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    """Bind passing gate detail schemas to the accepted release diagnostics."""

    for name, required_fields in _UK_TERMINAL_GATE_DETAIL_FIELDS.items():
        if name not in gates:
            continue
        details = _uk_terminal_gate_details(gates, name)
        if details is None:
            continue
        missing = sorted(required_fields - set(details))
        if missing:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} gate {name!r}.details is not "
                f"the honest aggregator detail schema; missing {missing}."
            )

    coverage = _uk_terminal_gate_details(gates, "uk_release_input_coverage")
    if coverage is not None:
        required = coverage.get("required_columns")
        present = coverage.get("present_columns")
        if (
            isinstance(required, bool)
            or not isinstance(required, int)
            or required <= 0
            or isinstance(present, bool)
            or not isinstance(present, int)
            or present < required
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} input-coverage details must "
                "record positive required_columns covered by present_columns."
            )
        for field in ("missing", "degenerate_required", "stale_exclusions"):
            if coverage.get(field) != []:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing input coverage "
                    f"requires details.{field} to be an empty list."
                )

    degenerate = _uk_terminal_gate_details(gates, "degenerate_release_surface")
    if degenerate is not None:
        checked = degenerate.get("columns_checked")
        if isinstance(checked, bool) or not isinstance(checked, int) or checked <= 0:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} degenerate-surface details "
                "must record a positive columns_checked count."
            )
        for field, empty in (
            ("findings", {}),
            ("all_null_columns", []),
            ("all_zero_columns", []),
            ("constant_columns", []),
            ("stale_exclusions", []),
            # Schema-2 exclusions (#610): an out-of-force approval must
            # never ride a published report. Presence is enforced by the
            # required detail schema above, so the checks read strictly.
            ("expired_exclusions", []),
            ("premature_exclusions", []),
        ):
            if degenerate.get(field) != empty:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing degenerate-surface "
                    f"gate requires details.{field} to be {empty!r}."
                )

    # One report evaluates every exclusion register on one date (the
    # aggregator threads a single clock). A hand-composed collection mixing
    # evaluation dates is not the aggregator's output.
    evaluation_dates = {
        name: details["exclusions_evaluated_on"]
        for name in (
            "degenerate_release_surface",
            "input_mass_parity",
            "qrf_tail_concentration",
        )
        if (details := _uk_terminal_gate_details(gates, name)) is not None
        and "exclusions_evaluated_on" in details
    }
    if len(set(evaluation_dates.values())) > 1:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} exclusion-consuming gates must "
            "share one details.exclusions_evaluated_on date; got "
            f"{dict(sorted(evaluation_dates.items()))}."
        )

    diagnostic_weights: Mapping | None = None
    if calibration_diagnostics is not None:
        uk = calibration_diagnostics.get("uk_diagnostics")
        if isinstance(uk, Mapping) and isinstance(uk.get("weights"), Mapping):
            diagnostic_weights = uk["weights"]

    for name in ("weight_ess", "weight_ratio"):
        details = _uk_terminal_gate_details(gates, name)
        if details is None or diagnostic_weights is None:
            continue
        for field in _UK_WEIGHT_SUMMARY_FIELDS:
            if field not in details or field not in diagnostic_weights:
                continue
            if not _uk_terminal_observable_matches(
                details[field], diagnostic_weights[field]
            ):
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} gate {name!r}.details.{field} "
                    "must match calibration_diagnostics.json "
                    f"uk_diagnostics.weights.{field}."
                )

    ess = _uk_terminal_gate_details(gates, "weight_ess")
    if ess is not None and not _uk_terminal_observable_matches(
        ess.get("minimum_ess_fraction"), _UK_MIN_ESS_FRACTION
    ):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} weight_ess.details."
            f"minimum_ess_fraction must equal {_UK_MIN_ESS_FRACTION}."
        )
    ratio = _uk_terminal_gate_details(gates, "weight_ratio")
    if ratio is not None and not _uk_terminal_observable_matches(
        ratio.get("maximum_max_to_median_ratio"),
        _UK_MAX_TO_MEDIAN_WEIGHT_RATIO,
    ):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} weight_ratio.details."
            "maximum_max_to_median_ratio must equal the certified June bound "
            f"{_UK_MAX_TO_MEDIAN_WEIGHT_RATIO}."
        )

    zero = _uk_terminal_gate_details(gates, "zero_weight_strata")
    if zero is not None and diagnostic_weights is not None:
        for terminal_field, diagnostic_field in (
            ("household_rows", "n_records"),
            ("zero_weight_rows", "zero_weight_records"),
        ):
            if not _uk_terminal_observable_matches(
                zero.get(terminal_field), diagnostic_weights.get(diagnostic_field)
            ):
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} zero_weight_strata.details."
                    f"{terminal_field} must match calibration_diagnostics.json "
                    f"uk_diagnostics.weights.{diagnostic_field}."
                )
        for field in ("unmatched_zero_weight_rows", "ambiguous_zero_weight_rows"):
            if zero.get(field) != 0:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing zero-weight gate "
                    f"requires details.{field} to be zero."
                )
        declared = zero.get("declared_strata")
        if not isinstance(declared, list) or not declared:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} zero_weight_strata.details."
                "declared_strata must be a non-empty list."
            )
        elif all(isinstance(row, Mapping) for row in declared):
            declared_zero = sum(
                row.get("zero_weight_rows", 0)
                for row in declared
                if isinstance(row.get("zero_weight_rows"), int)
                and not isinstance(row.get("zero_weight_rows"), bool)
            )
            if declared_zero != zero.get("zero_weight_rows"):
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} declared zero-weight "
                    "strata must reconcile to details.zero_weight_rows."
                )

    weights_audit = _uk_terminal_gate_details(gates, "weights_audit")
    if weights_audit is not None:
        fits = weights_audit.get("fits_checked")
        kinds = weights_audit.get("resolved_weight_kinds")
        if (
            isinstance(fits, bool)
            or not isinstance(fits, int)
            or fits <= 0
            or not isinstance(kinds, Mapping)
            or len(kinds) != fits
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing weights_audit details "
                "must enumerate at least one resolved production fit."
            )
        if weights_audit.get("unweighted_fits") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing weights_audit details "
                "must not contain unweighted fits."
            )

    input_mass = _uk_terminal_gate_details(gates, "input_mass_parity")
    if input_mass is not None:
        if input_mass.get("reference") != _UK_INPUT_MASS_ACTIVE_REFERENCE:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} input_mass_parity.details."
                f"reference must equal {_UK_INPUT_MASS_ACTIVE_REFERENCE!r}."
            )
        scope_note = input_mass.get("reference_scope_note")
        if not isinstance(scope_note, str) or not scope_note.strip():
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} input_mass_parity.details."
                "reference_scope_note must be a non-empty string."
            )
        identity = input_mass.get("reference_identity")
        if identity != _UK_INPUT_MASS_REFERENCE_IDENTITY:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} input_mass_parity.details."
                "reference_identity must match the active reviewed "
                f"efrs-post-calibration reference {_UK_INPUT_MASS_REFERENCE_IDENTITY}."
            )
        if not _uk_terminal_observable_matches(
            input_mass.get("relative_tolerance"),
            _UK_INPUT_MASS_RELATIVE_TOLERANCE,
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} input_mass_parity.details."
                "relative_tolerance must equal the committed spec value "
                f"{_UK_INPUT_MASS_RELATIVE_TOLERANCE}."
            )
        if not _uk_terminal_observable_matches(
            input_mass.get("minimum_reference_total"),
            _UK_INPUT_MASS_MINIMUM_REFERENCE_TOTAL,
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} input_mass_parity.details."
                "minimum_reference_total must equal the committed spec value "
                f"{_UK_INPUT_MASS_MINIMUM_REFERENCE_TOTAL}."
            )
        if input_mass.get("stale_exclusions") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing input-mass parity "
                "requires details.stale_exclusions to be an empty list."
            )
        if input_mass.get("expired_exclusions") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing input-mass parity "
                "requires details.expired_exclusions to be an empty list."
            )
        if input_mass.get("premature_exclusions") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing input-mass parity "
                "requires details.premature_exclusions to be an empty list."
            )

    qrf_tail = _uk_terminal_gate_details(gates, "qrf_tail_concentration")
    if qrf_tail is not None:
        columns_checked = qrf_tail.get("columns_checked")
        if (
            isinstance(columns_checked, bool)
            or not isinstance(columns_checked, int)
            or columns_checked <= 0
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.columns_checked to be positive."
            )
        top_k = qrf_tail.get("top_k")
        valid_top_k = (
            not isinstance(top_k, bool) and isinstance(top_k, int) and top_k >= 1
        )
        if not valid_top_k:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.top_k to be a positive non-boolean integer."
            )
        elif top_k != _UK_QRF_TAIL_TOP_K:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                f"requires details.top_k to equal the committed spec value "
                f"{_UK_QRF_TAIL_TOP_K}."
            )
        max_top_share = qrf_tail.get("max_top_share")
        valid_max_top_share = (
            not isinstance(max_top_share, bool)
            and isinstance(max_top_share, int | float)
            and 0.0 < max_top_share < 1.0
            and math.isfinite(max_top_share)
        )
        if not valid_max_top_share:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.max_top_share to be a finite non-boolean "
                "number in (0, 1)."
            )
        elif not _uk_terminal_observable_matches(
            max_top_share, _UK_QRF_TAIL_MAX_TOP_SHARE
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                f"requires details.max_top_share to equal the committed spec "
                f"value {_UK_QRF_TAIL_MAX_TOP_SHARE}."
            )
        min_nonzero_records = qrf_tail.get("min_nonzero_records")
        valid_min_nonzero_records_type = not isinstance(
            min_nonzero_records, bool
        ) and isinstance(min_nonzero_records, int)
        valid_min_nonzero_records = (
            valid_min_nonzero_records_type
            and valid_top_k
            and min_nonzero_records > top_k
        )
        if not valid_min_nonzero_records:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.min_nonzero_records to be a non-boolean integer "
                "greater than details.top_k."
            )
        elif min_nonzero_records != _UK_QRF_TAIL_MIN_NONZERO_RECORDS:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                f"requires details.min_nonzero_records to equal the committed "
                f"spec value {_UK_QRF_TAIL_MIN_NONZERO_RECORDS}."
            )
        top_share = qrf_tail.get("top_share")
        carrier_counts = qrf_tail.get("carrier_counts")
        thin_columns = qrf_tail.get("thin_columns")
        if (
            not isinstance(top_share, Mapping)
            or not isinstance(carrier_counts, Mapping)
            or set(top_share) != set(carrier_counts)
            or len(top_share) != columns_checked
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "must reconcile details.columns_checked, top_share, and "
                "carrier_counts."
            )
        valid_top_shares = isinstance(top_share, Mapping) and all(
            not isinstance(share, bool)
            and isinstance(share, int | float)
            and 0.0 <= share <= 1.0
            and math.isfinite(share)
            for share in top_share.values()
        )
        if not valid_top_shares:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.top_share values to be finite non-boolean "
                "numbers in [0, 1]."
            )
        valid_carrier_counts = isinstance(carrier_counts, Mapping) and all(
            not isinstance(count, bool)
            and isinstance(count, int)
            and (not valid_min_nonzero_records_type or count >= min_nonzero_records)
            for count in carrier_counts.values()
        )
        if not valid_carrier_counts:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.carrier_counts values to be non-boolean integers "
                "at least details.min_nonzero_records."
            )
        valid_thin_counts = isinstance(thin_columns, Mapping) and all(
            not isinstance(count, bool)
            and isinstance(count, int)
            and count >= 0
            and (not valid_min_nonzero_records_type or count < min_nonzero_records)
            for count in thin_columns.values()
        )
        if not valid_thin_counts:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.thin_columns values to be non-boolean integers "
                "in [0, details.min_nonzero_records)."
            )
        reviewed_exclusions = qrf_tail.get("reviewed_exclusions")
        valid_reviewed_exclusions = isinstance(reviewed_exclusions, Mapping) and all(
            isinstance(name, str)
            and bool(name.strip())
            and isinstance(reason, str)
            and bool(reason.strip())
            for name, reason in reviewed_exclusions.items()
        )
        if not valid_reviewed_exclusions:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.reviewed_exclusions to map non-empty column "
                "names to non-empty string reasons."
            )
        elif valid_top_shares and valid_max_top_share:
            high_share_columns = {
                name for name, share in top_share.items() if share > max_top_share
            }
            if high_share_columns != set(reviewed_exclusions):
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                    "requires columns above details.max_top_share to match "
                    "details.reviewed_exclusions exactly."
                )
        if qrf_tail.get("expired_exclusions") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.expired_exclusions to be an empty list."
            )
        if qrf_tail.get("premature_exclusions") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.premature_exclusions to be an empty list."
            )
        surface = qrf_tail.get("surface")
        if not isinstance(surface, Mapping):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.surface to be an object."
            )
        else:
            for field in ("absent_columns", "non_numeric_columns"):
                if surface.get(field) != []:
                    failures.append(
                        f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail "
                        f"concentration requires details.surface.{field} to "
                        "be empty."
                    )
            declared_count = surface.get("declared_qrf_outputs")
            classified = {
                field: surface.get(field)
                for field in (
                    "checked_columns",
                    "absent_columns",
                    "non_numeric_columns",
                )
            }
            if (
                isinstance(declared_count, bool)
                or not isinstance(declared_count, int)
                or declared_count <= 0
                or not isinstance(thin_columns, Mapping)
                or any(
                    not isinstance(names, list)
                    or any(not isinstance(name, str) or not name for name in names)
                    for names in classified.values()
                )
            ):
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail "
                    "concentration must carry a positive declared output "
                    "count, thin-columns object, and three column-name lists."
                )
            elif isinstance(top_share, Mapping):
                checked = set(classified["checked_columns"])
                absent = set(classified["absent_columns"])
                non_numeric = set(classified["non_numeric_columns"])
                all_lists = [
                    *classified["checked_columns"],
                    *classified["absent_columns"],
                    *classified["non_numeric_columns"],
                ]
                accounted = checked | absent | non_numeric
                gate_accounted = set(top_share) | set(thin_columns)
                if (
                    len(all_lists) != len(accounted)
                    or declared_count != len(accounted)
                    or set(top_share) & set(thin_columns)
                    or accounted != gate_accounted
                    or checked != gate_accounted - absent - non_numeric
                ):
                    failures.append(
                        f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail "
                        "concentration must reconcile declared, checked, "
                        "absent, nonnumeric, checked-tail, and thin outputs."
                    )
        if qrf_tail.get("stale_exclusions") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing QRF tail concentration "
                "requires details.stale_exclusions to be an empty list."
            )

    export = _uk_terminal_gate_details(gates, "export_surface")
    if export is not None:
        for field in ("candidate_columns", "reference_columns"):
            value = export.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} export_surface.details."
                    f"{field} must be a positive integer."
                )
        for field in (
            "missing_reference_columns",
            "unexpected_candidate_columns",
            "forbidden_candidate_columns",
        ):
            if export.get(field) != []:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} passing export surface "
                    f"requires details.{field} to be an empty list."
                )

    target_count = None
    if calibration_diagnostics is not None and isinstance(
        calibration_diagnostics.get("targets"), list
    ):
        target_count = len(calibration_diagnostics["targets"])
    target_surface = _uk_terminal_gate_details(gates, "target_surface")
    if target_surface is not None:
        if target_surface.get("candidate_targets") != target_count:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} target_surface.details."
                "candidate_targets must match len(calibration diagnostics targets)."
            )
        reference_targets = target_surface.get("reference_targets")
        if (
            isinstance(reference_targets, bool)
            or not isinstance(reference_targets, int)
            or reference_targets <= 0
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} target_surface.details."
                "reference_targets must be a positive integer."
            )
        if target_surface.get("missing_reference_targets") != []:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing target surface requires "
                "details.missing_reference_targets to be an empty list."
            )

    target_fit = _uk_terminal_gate_details(gates, "target_fit")
    if target_fit is not None:
        if target_fit.get("targets_checked") != target_count:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} target_fit.details."
                "targets_checked must match len(calibration diagnostics targets)."
            )
        if not _uk_terminal_observable_matches(
            target_fit.get("max_abs_relative_error"),
            _UK_MAX_TARGET_ABS_RELATIVE_ERROR,
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} target_fit.details."
                "max_abs_relative_error must equal "
                f"{_UK_MAX_TARGET_ABS_RELATIVE_ERROR}."
            )
        if target_fit.get("failing_targets") != {}:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} passing target fit requires "
                "details.failing_targets to be an empty object."
            )


def _check_uk_terminal_gate_report(
    report: Mapping,
    *,
    release_id: str,
    calibration_diagnostics_sha256: str | None,
    build_manifest: Mapping | None,
    calibration_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    """Independently verify the exact-k UK terminal-gate attestation.

    The gate producer is not imported into microcosm-data.  This verifier pins
    its producer and policy identities, derives the only permissible gate set
    from build-manifest evidence stages, recomputes the evidence/result
    bindings, and verifies the complete report with the out-of-band release
    key. A caller therefore cannot promote a hand-composed collection of
    passing ``GateResult`` objects as the canonical terminal verdict.
    """

    required_report_fields = {
        "schema_version",
        "enforced",
        "passed",
        "gates",
        "attestation",
    }
    if set(report) != required_report_fields:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} must contain exactly "
            f"{sorted(required_report_fields)}, got {sorted(map(str, report))}."
        )
    if report.get("schema_version") != _UK_TERMINAL_GATE_SCHEMA_VERSION:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} schema_version must be "
            f"{_UK_TERMINAL_GATE_SCHEMA_VERSION}."
        )
    if report.get("enforced") is not True:
        failures.append(f"{_UK_TERMINAL_GATE_REPORT_FILE} enforced must be true.")
    if report.get("passed") is not True:
        failures.append(f"{_UK_TERMINAL_GATE_REPORT_FILE} passed must be true.")

    build_evidence: Mapping = {}
    if build_manifest is not None and isinstance(
        build_manifest.get("terminal_gate_evidence"), Mapping
    ):
        build_evidence = build_manifest["terminal_gate_evidence"]
    expected_gate_names = list(_UK_ALWAYS_APPLICABLE_GATE_NAMES)
    for stage, stage_gate_names in _UK_TERMINAL_EVIDENCE_GATE_NAMES.items():
        if stage in build_evidence:
            expected_gate_names.extend(stage_gate_names)
    expected_gate_set = set(expected_gate_names)

    gates = report.get("gates")
    valid_gates: Mapping = {}
    if not isinstance(gates, Mapping):
        failures.append(f"{_UK_TERMINAL_GATE_REPORT_FILE} gates must be an object.")
    else:
        valid_gates = gates
        actual_gate_set = set(gates)
        if actual_gate_set != expected_gate_set:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} evaluated gate membership "
                "must be derived from build_manifest.json evidence stages; "
                f"expected {expected_gate_names}, got {sorted(map(str, gates))}."
            )
        for name, gate in gates.items():
            owner = f"{_UK_TERMINAL_GATE_REPORT_FILE} gate {name!r}"
            if not isinstance(gate, Mapping):
                failures.append(f"{owner} must be an object.")
                continue
            if set(gate) != {"passed", "failures", "details"}:
                failures.append(
                    f"{owner} must contain exactly passed, failures, and details."
                )
            if gate.get("passed") is not True:
                failures.append(f"{owner}.passed must be true.")
            if gate.get("failures") != []:
                failures.append(f"{owner}.failures must be an empty list.")
            if not isinstance(gate.get("details"), Mapping):
                failures.append(f"{owner}.details must be an object.")

    _check_uk_terminal_gate_observables(
        valid_gates,
        calibration_diagnostics=calibration_diagnostics,
        failures=failures,
    )

    attestation = report.get("attestation")
    if not isinstance(attestation, Mapping):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation must be an object."
        )
        return
    unsigned_fields = {
        "schema_version",
        "producer",
        "release_id",
        "calibration_diagnostics_sha256",
        "policy_sha256",
        "evaluated_gates",
        "evidence_sha256",
        "gate_results_sha256",
        "signature_algorithm",
        "signing_key_sha256",
    }
    required_attestation_fields = {*unsigned_fields, "signature"}
    if set(attestation) != required_attestation_fields:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation must contain exactly "
            f"{sorted(required_attestation_fields)}."
        )
    if (
        attestation.get("schema_version")
        != _UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION
    ):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.schema_version must be "
            f"{_UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION}."
        )
    if attestation.get("producer") != _UK_TERMINAL_GATE_PRODUCER:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.producer must name the "
            "honest UK terminal gate aggregator."
        )
    if attestation.get("release_id") != release_id:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.release_id must match "
            f"the release being validated; expected {release_id!r}, got "
            f"{attestation.get('release_id')!r}."
        )
    _check_sha256_field(
        filename=_UK_TERMINAL_GATE_REPORT_FILE,
        owner="attestation.calibration_diagnostics_sha256",
        value=attestation.get("calibration_diagnostics_sha256"),
        failures=failures,
    )
    if (
        calibration_diagnostics_sha256 is not None
        and attestation.get("calibration_diagnostics_sha256")
        != calibration_diagnostics_sha256
    ):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation."
            "calibration_diagnostics_sha256 must match the local "
            "calibration_diagnostics.json bytes."
        )
    expected_policy_sha256 = (
        _UK_TERMINAL_GATE_POLICY_SHA256_LEGACY
        if release_id in _UK_LEGACY_RELEASE_IDS
        else _UK_TERMINAL_GATE_POLICY_SHA256
    )
    if attestation.get("policy_sha256") != expected_policy_sha256:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.policy_sha256 does "
            "not match the certified UK gate policy for this release vintage."
        )
    if attestation.get("signature_algorithm") != _UK_TERMINAL_GATE_SIGNATURE_ALGORITHM:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.signature_algorithm "
            f"must be {_UK_TERMINAL_GATE_SIGNATURE_ALGORITHM!r}."
        )
    _check_sha256_field(
        filename=_UK_TERMINAL_GATE_REPORT_FILE,
        owner="attestation.signing_key_sha256",
        value=attestation.get("signing_key_sha256"),
        failures=failures,
    )
    _check_sha256_field(
        filename=_UK_TERMINAL_GATE_REPORT_FILE,
        owner="attestation.signature",
        value=attestation.get("signature"),
        failures=failures,
    )

    evaluated_gates = attestation.get("evaluated_gates")
    if evaluated_gates != expected_gate_names:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.evaluated_gates must "
            "exactly follow build-manifest evidence membership; "
            f"expected {expected_gate_names}, got {evaluated_gates!r}."
        )

    evidence = attestation.get("evidence_sha256")
    if not isinstance(evidence, Mapping):
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.evidence_sha256 must "
            "be an object."
        )
    else:
        for stage, digest in evidence.items():
            _check_sha256_field(
                filename=_UK_TERMINAL_GATE_REPORT_FILE,
                owner=f"attestation.evidence_sha256[{stage!r}]",
                value=digest,
                failures=failures,
            )
        if dict(evidence) != dict(build_evidence):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.evidence_sha256 "
                "must exactly match build_manifest.json terminal_gate_evidence."
            )
        if (
            "input_mass_parity" in build_evidence
            and evidence.get("input_mass_parity")
            != _UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} input_mass_parity evidence "
                "digest must bind the reviewed enhanced-FRS incumbent totals."
            )
        diagnostic_weights: Mapping | None = None
        if calibration_diagnostics is not None:
            uk = calibration_diagnostics.get("uk_diagnostics")
            if isinstance(uk, Mapping) and isinstance(uk.get("weights"), Mapping):
                diagnostic_weights = uk["weights"]
        if diagnostic_weights is not None:
            expected_release_dataset_sha = _canonical_sha256(
                {
                    "weights": {
                        field: diagnostic_weights.get(field)
                        for field in _UK_WEIGHT_SUMMARY_FIELDS
                    }
                }
            )
            if evidence.get("release_dataset") != expected_release_dataset_sha:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} release_dataset evidence "
                    "digest must bind calibration_diagnostics.json shipped-weight "
                    "observables."
                )

    expected_gate_results_sha = _canonical_sha256(valid_gates)
    if attestation.get("gate_results_sha256") != expected_gate_results_sha:
        failures.append(
            f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.gate_results_sha256 "
            "does not match gates."
        )

    verification_key = _uk_terminal_verification_key(failures)
    if verification_key is not None:
        expected_key_sha256 = hashlib.sha256(verification_key).hexdigest()
        if attestation.get("signing_key_sha256") != expected_key_sha256:
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.signing_key_sha256 "
                "does not identify the trusted release key."
            )
        unsigned_attestation = {
            field: attestation.get(field) for field in unsigned_fields
        }
        unsigned_report = {
            "schema_version": report.get("schema_version"),
            "enforced": report.get("enforced"),
            "passed": report.get("passed"),
            "gates": valid_gates,
            "attestation": unsigned_attestation,
        }
        expected_signature = hmac.new(
            verification_key,
            _canonical_json_bytes(unsigned_report),
            hashlib.sha256,
        ).hexdigest()
        signature = attestation.get("signature")
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, expected_signature
        ):
            failures.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} attestation.signature does "
                "not authenticate the complete report with the trusted release key."
            )


def _check_uk_gate_battery_report(
    report: Mapping,
    *,
    release_id: str,
    calibration_diagnostics_sha256: str | None,
    build_manifest: Mapping | None,
    calibration_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    """Independently verify an exact-k schema-4 gate-battery report.

    The battery producer is not imported into microcosm-data. This verifier
    pins the producer and the reviewed spec identities (policy digest,
    manifest digest, fingerprint, entry membership), recomputes shippability
    from the recorded outcomes instead of trusting the ``shippable`` flag,
    re-applies the legacy observable detail checks through a projection onto
    the shared gate implementations' names, and authenticates the complete
    report with the executor's out-of-band release key.
    """

    file = _UK_TERMINAL_GATE_REPORT_FILE
    required_report_fields = {
        "schema_version",
        "country",
        "release_id",
        "release_candidate",
        "spec_fingerprint",
        "gates_manifest_sha256",
        "phases",
        "phases_evaluated",
        "blocked_at_phase",
        "shippable",
        "gates",
        "policy_sha256",
        "release_evidence",
        "evidence_sha256",
        "attestation",
    }
    if set(report) != required_report_fields:
        failures.append(
            f"{file} (schema 4) must contain exactly "
            f"{sorted(required_report_fields)}, got {sorted(map(str, report))}."
        )
    schema_version = report.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != _UK_GATE_BATTERY_SCHEMA_VERSION
    ):
        failures.append(
            f"{file} schema_version must be the integer "
            f"{_UK_GATE_BATTERY_SCHEMA_VERSION}."
        )
    if report.get("country") != "uk":
        failures.append(f"{file} country must be 'uk'.")
    if report.get("release_id") != release_id:
        failures.append(
            f"{file} release_id must match the release being validated; "
            f"expected {release_id!r}, got {report.get('release_id')!r}."
        )
    if report.get("release_candidate") is not True:
        failures.append(
            f"{file} release_candidate must be true: a report produced off "
            "the candidate posture excused absent evidence and cannot be "
            "promoted into a release."
        )
    if report.get("blocked_at_phase") is not None:
        failures.append(f"{file} blocked_at_phase must be null for a release.")
    if list(report.get("phases") or ()) != list(_UK_GATE_BATTERY_PHASES):
        failures.append(f"{file} phases must be {list(_UK_GATE_BATTERY_PHASES)}.")
    if list(report.get("phases_evaluated") or ()) != list(_UK_GATE_BATTERY_PHASES):
        failures.append(f"{file} phases_evaluated must cover every declared phase.")
    if report.get("shippable") is not True:
        failures.append(f"{file} shippable must be true.")

    if report.get("policy_sha256") != _UK_GATE_BATTERY_POLICY_SHA256:
        failures.append(
            f"{file} policy_sha256 does not match the certified UK gate "
            "policy for this spec vintage."
        )
    if report.get("gates_manifest_sha256") != _UK_GATE_BATTERY_GATES_MANIFEST_SHA256:
        failures.append(
            f"{file} gates_manifest_sha256 does not match the committed "
            "uk/gates.json for this spec vintage."
        )
    if report.get("spec_fingerprint") != _UK_GATE_BATTERY_SPEC_FINGERPRINT:
        failures.append(
            f"{file} spec_fingerprint does not match the committed UK spec "
            "for this vintage."
        )

    gates = report.get("gates")
    valid_gates: dict[str, Mapping] = {}
    if not isinstance(gates, Mapping):
        failures.append(f"{file} gates must be an object keyed by entry id.")
        gates = {}
    if set(map(str, gates)) != set(_UK_GATE_BATTERY_ENTRY_IDS):
        failures.append(
            f"{file} gates must contain exactly the declared UK entry ids; "
            f"expected {sorted(_UK_GATE_BATTERY_ENTRY_IDS)}, got "
            f"{sorted(map(str, gates))}."
        )
    required_entry_fields = {
        "gate",
        "phase",
        "criticality",
        "status",
        "failures",
        "details",
        "reason",
    }
    for entry_id, outcome in gates.items():
        owner = f"{file} gates[{entry_id!r}]"
        if not isinstance(outcome, Mapping):
            failures.append(f"{owner} must be an object.")
            continue
        if set(outcome) != required_entry_fields:
            failures.append(
                f"{owner} must contain exactly {sorted(required_entry_fields)}."
            )
            continue
        status = outcome.get("status")
        if status not in _UK_GATE_BATTERY_STATUSES:
            failures.append(f"{owner}.status {status!r} is outside the taxonomy.")
            continue
        if status == "unreached":
            failures.append(
                f"{owner} is unreached, which contradicts a complete, "
                "unblocked evaluation."
            )
        if status == "not_applicable":
            failures.append(
                f"{owner} claims not_applicable, but no entry in this spec "
                "vintage declares an excuse."
            )
        pinned = _UK_GATE_BATTERY_ENTRY_GATES.get(str(entry_id))
        if pinned is not None:
            pinned_gate, pinned_phase = pinned
            if outcome.get("gate") != pinned_gate:
                failures.append(
                    f"{owner}.gate must be {pinned_gate!r} per the committed "
                    f"spec, got {outcome.get('gate')!r}."
                )
            if outcome.get("phase") != pinned_phase:
                failures.append(
                    f"{owner}.phase must be {pinned_phase!r} per the committed "
                    f"spec, got {outcome.get('phase')!r}."
                )
        # Criticality is pinned per entry against the committed spec, so a
        # relabel in either direction is a failure and cannot dodge the
        # shippability recompute below.  `unreached`, `not_applicable` and
        # any status outside the taxonomy are already refused above for every
        # entry, diagnostic ones included; what the diagnostic label buys is
        # only that `failed`/`evidence_absent` do not block, which is the
        # declared posture for the four local fit gates until microcosm#762
        # arms them.
        expected_criticality = (
            "diagnostic"
            if entry_id in _UK_GATE_BATTERY_DIAGNOSTIC_IDS
            else "release_blocking"
        )
        criticality = outcome.get("criticality")
        if criticality != expected_criticality:
            failures.append(
                f"{owner}.criticality must be {expected_criticality!r} per the "
                f"committed spec, got {criticality!r}."
            )
        elif (
            criticality == "release_blocking"
            and status not in _UK_GATE_BATTERY_SHIPPABLE_STATUSES
        ):
            # Shippability is recomputed here, per entry, instead of
            # trusting the report's own shippable flag.
            failures.append(
                f"{owner} is release-blocking with status {status!r}; the "
                "release cannot ship it."
            )
        details = outcome.get("details")
        failures_list = outcome.get("failures")
        if not isinstance(details, Mapping):
            failures.append(f"{owner}.details must be an object.")
        if not isinstance(failures_list, list):
            failures.append(f"{owner}.failures must be a list.")
            continue
        if not isinstance(details, Mapping):
            continue

        # Mirror the producer-side GateResult/GateOutcome envelope here: the
        # data shard cannot import those build-side classes, but a signed
        # release report must still preserve their projected invariants.
        reason = outcome.get("reason")
        if status == "passed":
            if failures_list != []:
                failures.append(f"{owner} passed entries cannot carry failure text.")
            if reason is not None:
                failures.append(f"{owner} passed entries cannot carry a reason.")
        elif status == "failed":
            if not failures_list or any(
                not isinstance(item, str) or not item.strip() for item in failures_list
            ):
                failures.append(
                    f"{owner} failed entries must carry non-empty failure text."
                )
            if reason is not None:
                failures.append(f"{owner} failed entries cannot carry a reason.")
        elif status in {"not_applicable", "evidence_absent"}:
            if failures_list != []:
                failures.append(f"{owner} {status} entries cannot carry failure text.")
            if dict(details) != {}:
                failures.append(f"{owner} {status} entries cannot carry details.")
            if not isinstance(reason, str) or not reason.strip():
                failures.append(
                    f"{owner} {status} entries must carry a non-empty string reason."
                )
        valid_gates[str(entry_id)] = outcome

    preflight_coverage = valid_gates.get("uk_release_input_coverage_manifest_current")
    if preflight_coverage is not None:
        if preflight_coverage.get("status") != "passed":
            failures.append(f"{file} the manifest-currency preflight must have passed.")
        elif dict(preflight_coverage.get("details", {})) != {
            "check": "manifest_current"
        }:
            failures.append(
                f"{file} the manifest-currency preflight details must be "
                "exactly {'check': 'manifest_current'}."
            )
    roster = valid_gates.get("uk_release_family_build_stages")
    if roster is not None and set(roster.get("details", {})) != {"stage_names"}:
        failures.append(
            f"{file} the build-stage roster details must carry exactly 'stage_names'."
        )

    # The observable detail schemas are the same gate implementations the
    # legacy report carried, re-keyed by entry id; project the evaluated
    # entries back onto the legacy names and reuse the checks verbatim.
    projected = {
        _UK_GATE_BATTERY_ENTRY_LEGACY_NAMES[entry_id]: {
            "passed": outcome.get("status") == "passed",
            "failures": list(outcome.get("failures", ())),
            "details": dict(outcome.get("details", {})),
        }
        for entry_id, outcome in valid_gates.items()
        if entry_id in _UK_GATE_BATTERY_ENTRY_LEGACY_NAMES
        and outcome.get("status") in ("passed", "failed")
    }
    _check_uk_terminal_gate_observables(
        projected,
        calibration_diagnostics=calibration_diagnostics,
        failures=failures,
    )

    release_evidence = report.get("release_evidence")
    if not isinstance(release_evidence, Mapping) or set(release_evidence) != {
        "calibration_diagnostics_sha256"
    }:
        failures.append(
            f"{file} release_evidence must carry exactly "
            "'calibration_diagnostics_sha256'."
        )
    else:
        _check_sha256_field(
            filename=file,
            owner="release_evidence.calibration_diagnostics_sha256",
            value=release_evidence.get("calibration_diagnostics_sha256"),
            failures=failures,
        )
        if (
            calibration_diagnostics_sha256 is not None
            and release_evidence.get("calibration_diagnostics_sha256")
            != calibration_diagnostics_sha256
        ):
            failures.append(
                f"{file} release_evidence.calibration_diagnostics_sha256 "
                "must match the local calibration_diagnostics.json bytes."
            )

    evidence = report.get("evidence_sha256")
    if not isinstance(evidence, Mapping):
        failures.append(f"{file} evidence_sha256 must be an object.")
        evidence = {}
    unexpected_evidence = sorted(
        str(key) for key in set(evidence) - _UK_GATE_BATTERY_EVIDENCE_IDS
    )
    if unexpected_evidence:
        failures.append(
            f"{file} evidence_sha256 has keys outside the evidence-bearing "
            f"entries: {unexpected_evidence}."
        )
    for entry_id in sorted(_UK_GATE_BATTERY_EVIDENCE_IDS):
        evaluated = valid_gates.get(entry_id, {}).get("status") in (
            "passed",
            "failed",
        )
        if evaluated and entry_id not in evidence:
            failures.append(
                f"{file} evidence_sha256 is missing the evaluated "
                f"evidence-bearing entry {entry_id!r}."
            )
        if not evaluated and entry_id in evidence:
            failures.append(
                f"{file} evidence_sha256 carries {entry_id!r} although the "
                "entry did not evaluate."
            )
    for entry_id, digest in evidence.items():
        _check_sha256_field(
            filename=file,
            owner=f"evidence_sha256[{entry_id!r}]",
            value=digest,
            failures=failures,
        )
    if (
        "uk_input_mass_parity" in evidence
        and evidence.get("uk_input_mass_parity")
        != _UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256
    ):
        failures.append(
            f"{file} uk_input_mass_parity evidence digest must bind the "
            "reviewed enhanced-FRS incumbent totals."
        )
    if (
        "uk_degenerate_release_surface" in evidence
        and evidence.get("uk_degenerate_release_surface")
        != _UK_GATE_BATTERY_DEGENERATE_EVIDENCE_SHA256
    ):
        failures.append(
            f"{file} uk_degenerate_release_surface evidence digest must bind "
            "the committed exclusion register; an overridden register is "
            "never releasable."
        )
    if build_manifest is not None:
        build_evidence = build_manifest.get("terminal_gate_evidence")
        if isinstance(build_evidence, Mapping) and dict(build_evidence) != dict(
            evidence
        ):
            failures.append(
                f"{file} evidence_sha256 must exactly match "
                "build_manifest.json terminal_gate_evidence."
            )

    attestation = report.get("attestation")
    if not isinstance(attestation, Mapping):
        failures.append(f"{file} attestation must be an object.")
        return
    required_attestation_fields = {
        "schema_version",
        "producer",
        "country",
        "release_id",
        "release_candidate",
        "spec_fingerprint",
        "gates_manifest_sha256",
        "policy_sha256",
        "phases",
        "phases_evaluated",
        "blocked_at_phase",
        "release_evidence",
        "evidence_sha256",
        "gate_outcomes_sha256",
        "signature_algorithm",
        "signing_key_sha256",
        "signature",
    }
    if set(attestation) != required_attestation_fields:
        # signing_error is deliberately outside the set: an unsigned report
        # records the hole there and can never verify as a release.
        failures.append(
            f"{file} attestation must contain exactly "
            f"{sorted(required_attestation_fields)}, got "
            f"{sorted(map(str, attestation))}."
        )
    attestation_schema = attestation.get("schema_version")
    if (
        type(attestation_schema) is not int
        or attestation_schema != _UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION
    ):
        failures.append(
            f"{file} attestation.schema_version must be the integer "
            f"{_UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION}."
        )
    if attestation.get("producer") != _UK_GATE_BATTERY_PRODUCER:
        failures.append(
            f"{file} attestation.producer must name the shared gate-battery executor."
        )
    for field in (
        "country",
        "release_id",
        "release_candidate",
        "spec_fingerprint",
        "gates_manifest_sha256",
        "policy_sha256",
        "phases",
        "phases_evaluated",
        "blocked_at_phase",
        "release_evidence",
        "evidence_sha256",
    ):
        if attestation.get(field) != report.get(field):
            failures.append(
                f"{file} attestation.{field} must equal the report body's {field}."
            )
    if attestation.get("signature_algorithm") != _UK_TERMINAL_GATE_SIGNATURE_ALGORITHM:
        failures.append(
            f"{file} attestation.signature_algorithm must be "
            f"{_UK_TERMINAL_GATE_SIGNATURE_ALGORITHM!r}."
        )
    expected_gate_outcomes_sha = _canonical_sha256(
        {str(entry_id): outcome for entry_id, outcome in gates.items()}
    )
    if attestation.get("gate_outcomes_sha256") != expected_gate_outcomes_sha:
        failures.append(
            f"{file} attestation.gate_outcomes_sha256 does not match gates."
        )

    verification_key = _uk_gate_battery_verification_key(failures)
    if verification_key is not None:
        expected_key_sha256 = hashlib.sha256(verification_key).hexdigest()
        if attestation.get("signing_key_sha256") != expected_key_sha256:
            failures.append(
                f"{file} attestation.signing_key_sha256 does not identify "
                "the trusted release key."
            )
        unsigned_report = dict(report)
        unsigned_report["attestation"] = {
            **{str(key): value for key, value in attestation.items()},
            "signature": None,
        }
        expected_signature = hmac.new(
            verification_key,
            _canonical_json_bytes(unsigned_report),
            hashlib.sha256,
        ).hexdigest()
        signature = attestation.get("signature")
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, expected_signature
        ):
            failures.append(
                f"{file} attestation.signature does not authenticate the "
                "complete report with the trusted release key."
            )


def _check_uk_release_certification(
    certification: Mapping,
    *,
    release_id: str,
    calibration_diagnostics_sha256: str | None,
    failures: list[str],
) -> None:
    """Validate the multi-part release certification (microcosm#757 B5).

    The certification is the only artifact that may carry a UK shippability
    verdict: its parts must union to the full declared gate-entry set with
    no gap and no overlap beyond the declared shared id, each part pinned to
    the committed spec's scoped digests, with every release-blocking entry
    passed and the whole document signed by the release key. Shippability is
    recomputed from the parts, never read off the flag.
    """

    file = _UK_RELEASE_CERTIFICATION_FILE
    actual_fields = set(certification)
    if actual_fields != _UK_CERTIFICATION_REQUIRED_FIELDS:
        missing = sorted(_UK_CERTIFICATION_REQUIRED_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - _UK_CERTIFICATION_REQUIRED_FIELDS)
        failures.append(
            f"{file} must carry exactly the certification fields; "
            f"missing {missing}, unexpected {unexpected}."
        )
        return
    schema = certification.get("schema_version")
    if type(schema) is not int or schema != _UK_RELEASE_CERTIFICATION_SCHEMA_VERSION:
        failures.append(
            f"{file} schema_version must be the integer "
            f"{_UK_RELEASE_CERTIFICATION_SCHEMA_VERSION}, got {schema!r}."
        )
        return
    if certification.get("kind") != _UK_RELEASE_CERTIFICATION_KIND:
        failures.append(
            f"{file} kind must be {_UK_RELEASE_CERTIFICATION_KIND!r}, got "
            f"{certification.get('kind')!r}."
        )
    if certification.get("country") != "uk":
        failures.append(f"{file} country must be 'uk'.")
    if certification.get("release_id") != release_id:
        failures.append(
            f"{file} release_id {certification.get('release_id')!r} does not "
            f"match the release under validation ({release_id!r})."
        )

    parts = certification.get("parts")
    if not isinstance(parts, Mapping) or set(parts) != set(
        _UK_CERTIFICATION_PART_SCOPES
    ):
        failures.append(
            f"{file} parts must be exactly "
            f"{sorted(_UK_CERTIFICATION_PART_SCOPES)}, got "
            f"{sorted(parts) if isinstance(parts, Mapping) else parts!r}."
        )
        return
    all_passed = True
    for part_name, expected_scope in _UK_CERTIFICATION_PART_SCOPES.items():
        part = parts[part_name]
        if not isinstance(part, Mapping):
            failures.append(f"{file} parts.{part_name} must be an object.")
            all_passed = False
            continue
        if sorted(part.get("entry_ids", ())) != sorted(expected_scope):
            failures.append(
                f"{file} parts.{part_name}.entry_ids must equal the declared "
                f"{part_name} scope."
            )
            all_passed = False
        expected_phases = list(_UK_CERTIFICATION_PART_PHASES[part_name])
        if list(part.get("phases", ())) != expected_phases:
            failures.append(
                f"{file} parts.{part_name}.phases must be {expected_phases}, "
                f"got {part.get('phases')!r}."
            )
        for field, expected_digest in _UK_CERTIFICATION_PART_DIGESTS[part_name].items():
            if part.get(field) != expected_digest:
                failures.append(
                    f"{file} parts.{part_name}.{field} does not match the "
                    "committed spec's scoped manifest digest."
                )
        statuses = part.get("statuses")
        expected_statuses = {"passed": len(expected_scope)}
        if statuses != expected_statuses:
            failures.append(
                f"{file} parts.{part_name}.statuses must be "
                f"{expected_statuses}, got {statuses!r}: shippability is "
                "recomputed from the parts, and only a fully-passed part "
                "certifies."
            )
            all_passed = False
        part_sha = part.get("sha256")
        if not isinstance(part_sha, str) or not _SHA256_RE.fullmatch(part_sha):
            failures.append(
                f"{file} parts.{part_name}.sha256 must be a sha256 hex digest."
            )

    # The union and overlap are properties of the mirrored scopes; assert
    # them against the full entry-id mirror so the three constants cannot
    # drift apart silently.
    union: dict[str, int] = {}
    for scope in _UK_CERTIFICATION_PART_SCOPES.values():
        for gate_id in scope:
            union[gate_id] = union.get(gate_id, 0) + 1
    if set(union) | set(_UK_CERTIFICATION_EXCLUDED_GATE_IDS) != (
        _UK_GATE_BATTERY_ENTRY_IDS
    ):
        failures.append(
            f"{file} mirrored part scopes plus certification exclusions do not "
            "cover the declared gate-entry set."
        )
    overlap = sorted(
        gate_id
        for gate_id, count in union.items()
        if count > 1 and gate_id not in _UK_CERTIFICATION_SHARED_GATE_IDS
    )
    if overlap:
        failures.append(
            f"{file} mirrored part scopes overlap beyond the declared shared "
            f"ids: {overlap}."
        )

    spec = certification.get("spec")
    if not isinstance(spec, Mapping):
        failures.append(f"{file} spec must be an object.")
    else:
        for field, expected in (
            ("gates_manifest_sha256", _UK_GATE_BATTERY_GATES_MANIFEST_SHA256),
            ("policy_sha256", _UK_GATE_BATTERY_POLICY_SHA256),
            ("spec_fingerprint", _UK_GATE_BATTERY_SPEC_FINGERPRINT),
        ):
            if spec.get(field) != expected:
                failures.append(
                    f"{file} spec.{field} does not match the committed "
                    "full-manifest pin."
                )
        if spec.get("declared_entry_count") != len(_UK_GATE_BATTERY_ENTRY_IDS):
            failures.append(
                f"{file} spec.declared_entry_count must be "
                f"{len(_UK_GATE_BATTERY_ENTRY_IDS)}."
            )
        if list(spec.get("declared_phases", ())) != list(_UK_GATE_BATTERY_PHASES):
            failures.append(
                f"{file} spec.declared_phases must be {list(_UK_GATE_BATTERY_PHASES)}."
            )
        if list(spec.get("shared_gate_ids", ())) != sorted(
            _UK_CERTIFICATION_SHARED_GATE_IDS
        ):
            failures.append(
                f"{file} spec.shared_gate_ids must be "
                f"{sorted(_UK_CERTIFICATION_SHARED_GATE_IDS)}."
            )
        if list(spec.get("certification_excluded_gate_ids", ())) != sorted(
            _UK_CERTIFICATION_EXCLUDED_GATE_IDS
        ):
            failures.append(
                f"{file} spec.certification_excluded_gate_ids must be "
                f"{sorted(_UK_CERTIFICATION_EXCLUDED_GATE_IDS)}."
            )

    if (
        calibration_diagnostics_sha256 is not None
        and certification.get("diagnostics_sha256") != calibration_diagnostics_sha256
    ):
        failures.append(
            f"{file} diagnostics_sha256 does not match the release's "
            "calibration_diagnostics.json bytes."
        )

    if certification.get("shippable") is not True or not all_passed:
        failures.append(
            f"{file} does not certify a shippable candidate: shippable must "
            "be true and every part fully passed."
        )

    attestation = certification.get("attestation")
    if not isinstance(attestation, Mapping):
        failures.append(f"{file} attestation must be an object.")
        return
    verification_key = _uk_gate_battery_verification_key(failures)
    if verification_key is None:
        return
    expected_key_sha256 = hashlib.sha256(verification_key).hexdigest()
    if attestation.get("signing_key_sha256") != expected_key_sha256:
        failures.append(
            f"{file} attestation.signing_key_sha256 does not identify the "
            "trusted release key."
        )
    unsigned = dict(certification)
    unsigned["attestation"] = {
        **{str(key): value for key, value in attestation.items()},
        "signature": None,
    }
    expected_signature = hmac.new(
        verification_key,
        _canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()
    signature = attestation.get("signature")
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature, expected_signature
    ):
        failures.append(
            f"{file} attestation.signature does not authenticate the "
            "complete certification with the trusted release key."
        )


def _check_uk_certification_evidence_binding(
    certification: Mapping,
    release_dir: Path,
    failures: list[str],
) -> None:
    """Bind every signed evidence digest to the local file's actual bytes.

    The certification signs the part-report and score-receipt digests; a
    release directory whose copies were removed or rewritten must refuse
    here, not validate on the digest fields alone.
    """

    parts = certification.get("parts")
    parts = parts if isinstance(parts, Mapping) else {}
    bindings: list[tuple[str, str, object]] = []
    for part, filename in _UK_CERTIFICATION_PART_EVIDENCE_FILES.items():
        part_payload = parts.get(part)
        signed_sha = (
            part_payload.get("sha256") if isinstance(part_payload, Mapping) else None
        )
        bindings.append((f"parts.{part}", filename, signed_sha))
    score_receipt = certification.get("score_receipt")
    bindings.append(
        (
            "score_receipt",
            _UK_CERTIFICATION_SCORE_RECEIPT_FILE,
            score_receipt.get("sha256") if isinstance(score_receipt, Mapping) else None,
        )
    )
    for owner, filename, signed_sha in bindings:
        path = release_dir / filename
        if not path.is_file():
            failures.append(
                f"{_UK_RELEASE_CERTIFICATION_FILE} signs {owner} but the "
                f"release directory is missing {filename!r}."
            )
            continue
        if not isinstance(signed_sha, str) or not _SHA256_RE.fullmatch(signed_sha):
            # Refuse, never skip: the parts block is shape-checked elsewhere
            # but score_receipt.sha256 is not, and a malformed digest must
            # not leave its evidence file unbound.
            failures.append(
                f"{_UK_RELEASE_CERTIFICATION_FILE} {owner}.sha256 is not a "
                f"sha256 digest; {filename!r} cannot be bound."
            )
            continue
        if _sha256(path) != signed_sha:
            failures.append(
                f"{filename} does not match the certification's signed {owner}.sha256."
            )


def _check_calibration_diagnostics(
    diagnostics: Mapping,
    failures: list[str],
    *,
    grandfathered_uk_june: bool = False,
) -> None:
    """Validate shared diagnostics, with one byte-lineage-scoped exemption.

    The release ``populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z``
    predates diagnostics schema 5. Its real schema-2 rows use the legacy
    ``aggregation`` selector instead of modern ``measure``/``filter`` selector
    objects. Only that exact release id may use this path; every other release
    remains on the modern contract.
    """

    schema_version = diagnostics.get("schema_version")
    if schema_version is None:
        failures.append("calibration_diagnostics.json is missing 'schema_version'.")
    elif grandfathered_uk_june and schema_version != 2:
        failures.append(
            "calibration_diagnostics.json grandfathered June UK release "
            f"requires legacy schema version 2, got {schema_version!r}."
        )
    elif (
        not grandfathered_uk_june
        and schema_version != CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
    ):
        failures.append(
            f"calibration_diagnostics.json 'schema_version' is {schema_version!r}; "
            f"this library publishes version "
            f"{CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION}."
        )

    expected_sections = {
        "target_surface": Mapping,
        "target_registry": Mapping,
        "targets": list,
        "loss_trajectory": list,
        "skipped": list,
        "options": Mapping,
    }
    for section, expected_type in expected_sections.items():
        value = diagnostics.get(section)
        if not isinstance(value, expected_type):
            failures.append(
                f"calibration_diagnostics.json is missing a {section!r} "
                f"{expected_type.__name__}."
            )

    _check_target_surface_ref(
        diagnostics.get("target_surface"),
        filename="calibration_diagnostics.json",
        owner="top-level",
        failures=failures,
    )
    _check_target_registry_ref(
        diagnostics.get("target_registry"),
        filename="calibration_diagnostics.json",
        owner="top-level",
        failures=failures,
    )

    targets = diagnostics.get("targets")
    if isinstance(targets, list):
        surface = diagnostics.get("target_surface")
        if isinstance(surface, Mapping) and surface.get("n_targets") != len(targets):
            failures.append(
                "calibration_diagnostics.json target_surface.n_targets must "
                "equal len(targets)."
            )
        for index, target in enumerate(targets):
            if not isinstance(target, Mapping):
                failures.append(
                    f"calibration_diagnostics.json target row {index} must be an object."
                )
                continue
            required_fields = (
                "name",
                "target_name",
                "period",
                "entity",
                "target",
                "compiled_target",
                "initial_estimate",
                "final_estimate",
                "relative_error",
                *(("aggregation",) if grandfathered_uk_june else ("measure", "filter")),
            )
            for field in required_fields:
                if field not in target:
                    failures.append(
                        "calibration_diagnostics.json target row "
                        f"{index} is missing {field!r}."
                    )
            if not target.get("source"):
                failures.append(
                    "calibration_diagnostics.json target row "
                    f"{index} is missing non-empty 'source'."
                )
            if not grandfathered_uk_june:
                if not isinstance(target.get("measure"), Mapping):
                    failures.append(
                        "calibration_diagnostics.json target row "
                        f"{index} is missing 'measure' selector object."
                    )
                target_filter = target.get("filter")
                if target_filter is not None and not isinstance(target_filter, Mapping):
                    failures.append(
                        "calibration_diagnostics.json target row "
                        f"{index} has non-null 'filter' that is not a selector object."
                    )
            if not isinstance(target.get("metadata"), Mapping):
                failures.append(
                    "calibration_diagnostics.json target row "
                    f"{index} is missing 'metadata' object."
                )


def _uk_non_negative_int(
    value: object,
    *,
    field: str,
    failures: list[str],
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        failures.append(
            f"calibration_diagnostics.json {field} must be a non-negative integer, "
            f"got {value!r}."
        )
        return None
    return value


def _uk_finite_number(
    value: object,
    *,
    field: str,
    failures: list[str],
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        failures.append(
            f"calibration_diagnostics.json {field} must be a finite number, "
            f"got {value!r}."
        )
        return None
    number = float(value)
    if not math.isfinite(number):
        failures.append(
            f"calibration_diagnostics.json {field} must be finite, got {value!r}."
        )
        return None
    if minimum is not None and number < minimum:
        failures.append(
            f"calibration_diagnostics.json {field} must be at least {minimum}, "
            f"got {number}."
        )
    if maximum is not None and number > maximum:
        failures.append(
            f"calibration_diagnostics.json {field} must be at most {maximum}, "
            f"got {number}."
        )
    return number


def _check_uk_calibration_diagnostics(
    diagnostics: Mapping,
    failures: list[str],
) -> None:
    """Require the versioned UK release diagnostics on canonical exact-k ids."""

    uk = diagnostics.get("uk_diagnostics")
    if not isinstance(uk, Mapping):
        failures.append(
            "calibration_diagnostics.json canonical UK releases require a "
            "'uk_diagnostics' object."
        )
        return
    if uk.get("schema_version") != _UK_DIAGNOSTICS_SCHEMA_VERSION:
        failures.append(
            "calibration_diagnostics.json 'uk_diagnostics.schema_version' is "
            f"{uk.get('schema_version')!r}; expected "
            f"{_UK_DIAGNOSTICS_SCHEMA_VERSION}."
        )

    weights = uk.get("weights")
    if not isinstance(weights, Mapping):
        failures.append(
            "calibration_diagnostics.json 'uk_diagnostics.weights' must be an object."
        )
        weights = {}
    n_records = _uk_non_negative_int(
        weights.get("n_records"),
        field="uk_diagnostics.weights.n_records",
        failures=failures,
    )
    if n_records == 0:
        failures.append(
            "calibration_diagnostics.json UK release diagnostics require at "
            "least one weight record."
        )
    positive_records = _uk_non_negative_int(
        weights.get("positive_weight_records"),
        field="uk_diagnostics.weights.positive_weight_records",
        failures=failures,
    )
    zero_records = _uk_non_negative_int(
        weights.get("zero_weight_records"),
        field="uk_diagnostics.weights.zero_weight_records",
        failures=failures,
    )
    _uk_finite_number(
        weights.get("total_weight"),
        field="uk_diagnostics.weights.total_weight",
        failures=failures,
        minimum=0.0,
    )
    max_weight = _uk_finite_number(
        weights.get("max_weight"),
        field="uk_diagnostics.weights.max_weight",
        failures=failures,
        minimum=0.0,
    )
    ess = _uk_finite_number(
        weights.get("effective_sample_size"),
        field="uk_diagnostics.weights.effective_sample_size",
        failures=failures,
        minimum=0.0,
        maximum=float(n_records) if n_records is not None else None,
    )
    ess_fraction = _uk_finite_number(
        weights.get("ess_fraction"),
        field="uk_diagnostics.weights.ess_fraction",
        failures=failures,
        minimum=0.0,
        maximum=1.0,
    )
    top_share = _uk_finite_number(
        weights.get("top_1pct_weight_share"),
        field="uk_diagnostics.weights.top_1pct_weight_share",
        failures=failures,
        minimum=0.0,
        maximum=1.0,
    )
    if (
        n_records is not None
        and n_records > 0
        and ess is not None
        and ess_fraction is not None
        and not math.isclose(
            ess_fraction,
            ess / n_records,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        failures.append(
            "calibration_diagnostics.json UK ess_fraction must equal "
            "effective_sample_size/n_records."
        )
    shared_n_records = diagnostics.get("n_records")
    if not isinstance(shared_n_records, int) or isinstance(shared_n_records, bool):
        failures.append(
            "calibration_diagnostics.json canonical UK releases require numeric "
            "top-level n_records."
        )
    elif n_records is not None and shared_n_records != n_records:
        failures.append(
            "calibration_diagnostics.json UK weights.n_records must match "
            "top-level n_records."
        )
    target_surface = diagnostics.get("target_surface")
    if (
        isinstance(target_surface, Mapping)
        and "n_records" in target_surface
        and n_records is not None
        and target_surface.get("n_records") != n_records
    ):
        failures.append(
            "calibration_diagnostics.json UK weights.n_records must match "
            "target_surface.n_records."
        )
    for field, observed in (
        ("effective_sample_size", ess),
        ("top_1pct_weight_share", top_share),
    ):
        shared = diagnostics.get(field)
        if isinstance(shared, bool) or not isinstance(shared, int | float):
            failures.append(
                "calibration_diagnostics.json canonical UK releases require "
                f"numeric top-level {field}."
            )
        elif observed is not None and not math.isclose(
            float(shared),
            observed,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            failures.append(
                f"calibration_diagnostics.json UK weights.{field} must match "
                f"top-level {field}."
            )
    ratio_fields = ("median_positive_weight", "max_to_median_positive_weight")
    for field in ratio_fields:
        if field not in weights:
            failures.append(
                "calibration_diagnostics.json UK weight diagnostics require "
                f"uk_diagnostics.weights.{field}."
            )
    median = weights.get("median_positive_weight")
    ratio = weights.get("max_to_median_positive_weight")
    if positive_records == 0:
        if median is not None or ratio is not None:
            failures.append(
                "calibration_diagnostics.json UK all-zero weights require null "
                "median_positive_weight and max_to_median_positive_weight."
            )
    elif positive_records is not None:
        valid_median = _uk_finite_number(
            median,
            field="uk_diagnostics.weights.median_positive_weight",
            failures=failures,
            minimum=0.0,
        )
        valid_ratio = _uk_finite_number(
            ratio,
            field="uk_diagnostics.weights.max_to_median_positive_weight",
            failures=failures,
            minimum=1.0,
        )
        if valid_median is not None and valid_median <= 0.0:
            failures.append(
                "calibration_diagnostics.json UK positive weights require a "
                "strictly positive median_positive_weight."
            )
        if (
            valid_median is not None
            and valid_median > 0.0
            and max_weight is not None
            and valid_ratio is not None
            and not math.isclose(
                valid_ratio,
                max_weight / valid_median,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            failures.append(
                "calibration_diagnostics.json UK "
                "max_to_median_positive_weight must equal "
                "max_weight/median_positive_weight."
            )
    if (
        n_records is not None
        and positive_records is not None
        and zero_records is not None
        and positive_records + zero_records != n_records
    ):
        failures.append(
            "calibration_diagnostics.json UK positive- and zero-weight record "
            "counts must sum to weights.n_records."
        )

    strata = uk.get("zero_weight_rows_by_stratum")
    if not isinstance(strata, list) or not strata:
        failures.append(
            "calibration_diagnostics.json "
            "'uk_diagnostics.zero_weight_rows_by_stratum' must be a non-empty list."
        )
        strata = []
    stratum_rows = 0
    stratum_positive = 0
    stratum_zero = 0
    for index, row in enumerate(strata):
        if not isinstance(row, Mapping):
            failures.append(
                "calibration_diagnostics.json UK zero-weight stratum row "
                f"{index} must be an object."
            )
            continue
        if not isinstance(row.get("stratum"), Mapping) or not row["stratum"]:
            failures.append(
                "calibration_diagnostics.json UK zero-weight stratum row "
                f"{index} needs a non-empty 'stratum' object."
            )
        rows = _uk_non_negative_int(
            row.get("rows"),
            field=f"uk_diagnostics.zero_weight_rows_by_stratum[{index}].rows",
            failures=failures,
        )
        positive = _uk_non_negative_int(
            row.get("positive_weight_rows"),
            field=(
                "uk_diagnostics.zero_weight_rows_by_stratum"
                f"[{index}].positive_weight_rows"
            ),
            failures=failures,
        )
        zero = _uk_non_negative_int(
            row.get("zero_weight_rows"),
            field=(
                f"uk_diagnostics.zero_weight_rows_by_stratum[{index}].zero_weight_rows"
            ),
            failures=failures,
        )
        _uk_finite_number(
            row.get("weight_sum"),
            field=(f"uk_diagnostics.zero_weight_rows_by_stratum[{index}].weight_sum"),
            failures=failures,
            minimum=0.0,
        )
        if rows is None or positive is None or zero is None:
            continue
        if positive + zero != rows:
            failures.append(
                "calibration_diagnostics.json UK zero-weight stratum row "
                f"{index} counts do not reconcile."
            )
        stratum_rows += rows
        stratum_positive += positive
        stratum_zero += zero
    if n_records is not None and strata and stratum_rows != n_records:
        failures.append(
            "calibration_diagnostics.json UK stratum rows do not reconcile to "
            "weights.n_records."
        )
    if positive_records is not None and strata and stratum_positive != positive_records:
        failures.append(
            "calibration_diagnostics.json UK positive stratum rows do not "
            "reconcile to weights.positive_weight_records."
        )
    if zero_records is not None and strata and stratum_zero != zero_records:
        failures.append(
            "calibration_diagnostics.json UK zero-weight stratum rows do not "
            "reconcile to weights.zero_weight_records."
        )

    rates = uk.get("target_pass_rates_by_geography_level")
    if not isinstance(rates, list):
        failures.append(
            "calibration_diagnostics.json "
            "'uk_diagnostics.target_pass_rates_by_geography_level' must be a list."
        )
        rates = []
    seen_levels: set[str] = set()
    total_targets = total_scored = total_skipped = 0
    for index, row in enumerate(rates):
        if not isinstance(row, Mapping):
            failures.append(
                "calibration_diagnostics.json UK geography pass-rate row "
                f"{index} must be an object."
            )
            continue
        level = row.get("geography_level")
        if not isinstance(level, str) or level not in _UK_TARGET_GEOGRAPHY_LEVELS:
            failures.append(
                "calibration_diagnostics.json UK geography pass-rate row "
                f"{index} has unknown level {level!r}."
            )
        elif level in seen_levels:
            failures.append(
                "calibration_diagnostics.json UK geography pass-rate level "
                f"{level!r} appears more than once."
            )
        else:
            seen_levels.add(level)
        counts = [
            _uk_non_negative_int(
                row.get(field),
                field=(
                    "uk_diagnostics.target_pass_rates_by_geography_level"
                    f"[{index}].{field}"
                ),
                failures=failures,
            )
            for field in ("n_targets", "n_scored", "n_skipped", "n_within_10pct")
        ]
        if any(value is None for value in counts):
            continue
        n_targets, n_scored, n_skipped, n_within = counts
        assert n_targets is not None
        assert n_scored is not None
        assert n_skipped is not None
        assert n_within is not None
        if n_scored + n_skipped != n_targets or n_within > n_scored:
            failures.append(
                "calibration_diagnostics.json UK geography pass-rate row "
                f"{index} counts do not reconcile."
            )
        pass_rate = row.get("pass_rate")
        if n_targets == 0:
            if pass_rate is not None:
                failures.append(
                    "calibration_diagnostics.json UK empty geography pass-rate "
                    f"row {index} must use null pass_rate."
                )
        else:
            observed_rate = _uk_finite_number(
                pass_rate,
                field=(
                    "uk_diagnostics.target_pass_rates_by_geography_level"
                    f"[{index}].pass_rate"
                ),
                failures=failures,
                minimum=0.0,
                maximum=1.0,
            )
            expected_rate = n_within / n_targets
            if observed_rate is not None and not math.isclose(
                observed_rate,
                expected_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                failures.append(
                    "calibration_diagnostics.json UK geography pass-rate row "
                    f"{index} pass_rate does not match n_within_10pct/n_targets."
                )
        total_targets += n_targets
        total_scored += n_scored
        total_skipped += n_skipped
    missing_levels = sorted(_UK_TARGET_GEOGRAPHY_LEVELS - seen_levels)
    if missing_levels:
        failures.append(
            "calibration_diagnostics.json UK geography pass rates are missing "
            f"level(s): {missing_levels}."
        )
    registry = diagnostics.get("target_registry")
    if isinstance(registry, Mapping):
        if registry.get("country") != "uk":
            failures.append(
                "calibration_diagnostics.json canonical UK releases require "
                "target_registry.country == 'uk'."
            )
        if isinstance(registry.get("n_specs"), int) and (
            total_targets != registry["n_specs"]
        ):
            failures.append(
                "calibration_diagnostics.json UK geography target counts do not "
                "reconcile to target_registry.n_specs."
            )
    targets = diagnostics.get("targets")
    if isinstance(targets, list) and total_scored != len(targets):
        failures.append(
            "calibration_diagnostics.json UK geography scored counts do not "
            "reconcile to len(targets)."
        )
    skipped = diagnostics.get("skipped")
    if isinstance(skipped, list) and total_skipped != len(skipped):
        failures.append(
            "calibration_diagnostics.json UK geography skipped counts do not "
            "reconcile to len(skipped)."
        )


def _check_us_critical_target_fit(diagnostics: Mapping, failures: list[str]) -> None:
    targets = diagnostics.get("targets")
    if not isinstance(targets, list):
        return
    incumbent_targets = _incumbent_critical_targets(diagnostics)
    for requirement in _US_CRITICAL_TARGET_FIT_REQUIREMENTS:
        matches = [
            target
            for target in targets
            if isinstance(target, Mapping)
            and not _is_congressional_district_layout_target(target)
            and requirement.matches(
                name=str(target.get("name") or ""),
                family=_target_registry_family(target),
                target_role=(
                    str(target["metadata"].get("target_role") or "")
                    if isinstance(target.get("metadata"), Mapping)
                    else ""
                ),
            )
        ]
        if not matches:
            failures.append(
                "calibration_diagnostics.json is missing required US critical "
                f"target {requirement.requirement_id!r} "
                f"({requirement.label})."
            )
            continue
        for target in matches:
            relative_error = target.get("relative_error")
            computed_relative_error = _target_relative_error(target, failures)
            if computed_relative_error is None:
                continue
            if not isinstance(relative_error, int | float):
                failures.append(
                    "calibration_diagnostics.json critical target "
                    f"{target.get('name')!r} has non-numeric relative_error "
                    f"{relative_error!r}."
                )
            elif not math.isclose(
                float(relative_error),
                computed_relative_error,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                failures.append(
                    "calibration_diagnostics.json critical target "
                    f"{target.get('name')!r} has stale relative_error "
                    f"{relative_error!r}; computed "
                    f"{computed_relative_error:.6g} from target and "
                    "final_estimate."
                )
            max_abs = float(requirement.max_abs_relative_error)
            if abs(computed_relative_error) > max_abs:
                incumbent_relative_error = _incumbent_relative_error_for_target(
                    target,
                    incumbent_targets.get(str(target.get("name"))),
                    failures,
                )
                improved_over_incumbent = incumbent_relative_error is not None and abs(
                    computed_relative_error
                ) < abs(incumbent_relative_error)
                if (
                    requirement.allow_incumbent_improvement
                    and improved_over_incumbent
                    and abs(computed_relative_error)
                    <= _US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR
                ):
                    continue
                failures.append(
                    "calibration_diagnostics.json critical target "
                    f"{target.get('name')!r} ({requirement.label}) has "
                    f"relative_error={computed_relative_error:.6g}, exceeding "
                    f"{max_abs:.6g}; target={target.get('target')!r}, "
                    f"final_estimate={target.get('final_estimate')!r}"
                    + (
                        "."
                        if incumbent_relative_error is None
                        else (
                            "; incumbent_relative_error="
                            f"{incumbent_relative_error:.6g}; "
                            "improvement_hard_stop="
                            f"{_US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR:.6g}."
                        )
                    )
                )


def _incumbent_critical_targets(diagnostics: Mapping) -> Mapping:
    build = diagnostics.get("build")
    if not isinstance(build, Mapping):
        return {}
    incumbent = build.get("incumbent_diagnostics")
    if not isinstance(incumbent, Mapping):
        return {}
    targets = incumbent.get("critical_targets")
    if not isinstance(targets, Mapping):
        return {}
    return targets


def _incumbent_relative_error_for_target(
    target: Mapping,
    incumbent: object,
    failures: list[str],
) -> float | None:
    if not isinstance(incumbent, Mapping):
        return None
    current_target = target.get("target")
    incumbent_target = incumbent.get("target")
    incumbent_final = incumbent.get("final_estimate")
    if not isinstance(current_target, int | float) or not isinstance(
        incumbent_target, int | float
    ):
        return None
    if not isinstance(incumbent_final, int | float):
        return None
    current_target = float(current_target)
    incumbent_target = float(incumbent_target)
    incumbent_final = float(incumbent_final)
    if not (
        math.isfinite(current_target)
        and math.isfinite(incumbent_target)
        and math.isfinite(incumbent_final)
    ):
        return None
    if not math.isclose(
        incumbent_target,
        current_target,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        failures.append(
            "calibration_diagnostics.json critical target "
            f"{target.get('name')!r} has incumbent target "
            f"{incumbent_target!r}, which does not match current target "
            f"{current_target!r}."
        )
        return None
    if current_target == 0.0:
        return incumbent_final - current_target
    return (incumbent_final - current_target) / current_target


def _target_relative_error(target: Mapping, failures: list[str]) -> float | None:
    target_value = target.get("target")
    final_estimate = target.get("final_estimate")
    if not isinstance(target_value, int | float) or not isinstance(
        final_estimate, int | float
    ):
        failures.append(
            "calibration_diagnostics.json critical target "
            f"{target.get('name')!r} has non-numeric target/final_estimate: "
            f"target={target_value!r}, final_estimate={final_estimate!r}."
        )
        return None
    target_value = float(target_value)
    final_estimate = float(final_estimate)
    if not math.isfinite(target_value) or not math.isfinite(final_estimate):
        failures.append(
            "calibration_diagnostics.json critical target "
            f"{target.get('name')!r} has non-finite target/final_estimate: "
            f"target={target_value!r}, final_estimate={final_estimate!r}."
        )
        return None
    if target_value == 0.0:
        return final_estimate - target_value
    return (final_estimate - target_value) / target_value


def _target_registry_family(target: Mapping) -> str:
    registry = target.get("registry")
    if isinstance(registry, Mapping):
        family = registry.get("family")
        if family is not None:
            return str(family)
    return ""


def _is_congressional_district_layout_target(target: Mapping) -> bool:
    return is_congressional_district_target(
        target.get("name"),
        target.get("metadata"),
    )


def _check_source_coverage_diagnostics(
    diagnostics: Mapping,
    failures: list[str],
    *,
    require_gate_passed: bool = True,
) -> None:
    schema_version = diagnostics.get("schema_version")
    if schema_version is None:
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing 'schema_version'."
        )
    elif schema_version != SOURCE_COVERAGE_DIAGNOSTICS_SCHEMA_VERSION:
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} 'schema_version' is "
            f"{schema_version!r}; this library publishes version "
            f"{SOURCE_COVERAGE_DIAGNOSTICS_SCHEMA_VERSION}."
        )
    if diagnostics.get("classification") != "release_gate":
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} must declare "
            "'classification'='release_gate'."
        )

    source_contract = diagnostics.get("source_contract")
    if not isinstance(source_contract, Mapping):
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing the "
            "'source_contract' object."
        )
    else:
        if source_contract.get("name") != "us_source_coverage":
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} source_contract.name must "
                "be 'us_source_coverage'."
            )
        ledger_commit = source_contract.get("ledger_commit")
        if not isinstance(ledger_commit, str) or len(ledger_commit) != 40:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} source_contract.ledger_commit "
                "must be a 40-character commit hash."
            )

    gate = diagnostics.get("gate")
    if not isinstance(gate, Mapping):
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing the 'gate' object."
        )
    else:
        if gate.get("name") != "us_source_coverage":
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.name must be "
                "'us_source_coverage'."
            )
        if require_gate_passed and gate.get("passed") is not True:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.passed must be true."
            )
        gate_failures = gate.get("failures")
        if not isinstance(gate_failures, list):
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.failures must be a list."
            )
        elif gate_failures and require_gate_passed:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.failures must be empty."
            )

    expected_sections = {
        "coverage_summary": Mapping,
        "hard_target_families": Mapping,
        "validation_only_families": Mapping,
        "source_gap_families": Mapping,
        "fiscal_target_sources": Mapping,
        "missing_hard_targets": list,
        "reviewed_exclusions": Mapping,
        "validation_only_activated": list,
    }
    for section, expected_type in expected_sections.items():
        value = diagnostics.get(section)
        if not isinstance(value, expected_type):
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing a "
                f"{section!r} {expected_type.__name__}."
            )

    reviewed = diagnostics.get("reviewed_exclusions")
    if isinstance(reviewed, Mapping):
        bad_reviewed = sorted(
            str(alias)
            for alias, reason in reviewed.items()
            if not isinstance(reason, str) or not reason.strip()
        )
        if bad_reviewed:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} reviewed_exclusions "
                f"need non-empty string reasons for {bad_reviewed}."
            )

    fiscal_sources = diagnostics.get("fiscal_target_sources")
    if isinstance(fiscal_sources, Mapping):
        for family, source in fiscal_sources.items():
            if not isinstance(source, Mapping):
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                    f"fiscal_target_sources[{family!r}] must be an object."
                )
                continue
            target_count = source.get("target_count")
            if not isinstance(target_count, int) or target_count <= 0:
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                    f"fiscal_target_sources[{family!r}].target_count must be > 0."
                )
            sources = source.get("sources")
            if (
                not isinstance(sources, list)
                or not sources
                or any(not isinstance(item, str) or not item for item in sources)
            ):
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                    f"fiscal_target_sources[{family!r}].sources must be a "
                    "non-empty list of strings."
                )
            reference_urls = source.get("reference_urls")
            if not isinstance(reference_urls, list) or any(
                not isinstance(item, str) or not item for item in reference_urls
            ):
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                    f"fiscal_target_sources[{family!r}].reference_urls must "
                    "be a list of strings."
                )


def _validate_local_area_release_dir(release_dir: Path, release_id: str) -> None:
    """The non-default local-area release contract (microcosm#398).

    The artifact is calibrated to a local surface (population marginals +
    state administrative families) by design, so the national critical-target
    checks do not apply. What must hold instead: the declared gates all
    passed, the per-target diagnostics are complete and consistent, the
    source coverage carries the local provenance chain, a
    reviewed-limitations register exists, the manifest claims no default
    slot, and every artifact is pinned to the immutable release id.
    """

    failures: list[str] = []
    for filename in LOCAL_AREA_REQUIRED_RELEASE_FILES:
        if not (release_dir / filename).is_file():
            failures.append(f"required file {filename!r} is missing.")

    release_manifest: Mapping | None = None
    manifest_path = release_dir / "release_manifest.json"
    if manifest_path.is_file():
        release_manifest = _load_json(manifest_path, failures)
    if release_manifest is not None:
        _check_local_area_release_manifest(release_manifest, release_id, failures)

    build_manifest_path = release_dir / "build_manifest.json"
    if build_manifest_path.is_file():
        build_manifest = _load_json(build_manifest_path, failures)
        if build_manifest is not None:
            build_id = build_manifest.get("build_id")
            if build_id != release_id:
                failures.append(
                    f"build_manifest.json 'build_id' is {build_id!r} but the "
                    f"release directory is named {release_id!r}."
                )

    gate_path = release_dir / "gate_summary.json"
    if gate_path.is_file():
        gate_summary = _load_json(gate_path, failures)
        if gate_summary is not None:
            _check_local_area_gates(gate_summary, failures)

    diagnostics_path = release_dir / "calibration_diagnostics.json"
    if diagnostics_path.is_file():
        diagnostics = _load_json(diagnostics_path, failures)
        if diagnostics is not None:
            _check_local_area_calibration_diagnostics(diagnostics, failures)
            if _is_uk_exact_k_release_id(release_id):
                _check_uk_calibration_diagnostics(diagnostics, failures)
                _check_uk_exact_k_diagnostics_identity(
                    diagnostics, release_id, failures
                )

    coverage_path = release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE
    if coverage_path.is_file():
        coverage = _load_json(coverage_path, failures)
        if coverage is not None:
            for key in LOCAL_AREA_SOURCE_COVERAGE_KEYS:
                if coverage.get(key) is None:
                    failures.append(
                        f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} is missing "
                        f"the local-area provenance key {key!r} (or it is "
                        "null)."
                    )
            donor = coverage.get("donor_release")
            if isinstance(donor, Mapping) and not donor.get("release_id"):
                failures.append(
                    f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} donor_release "
                    "must pin a non-empty release_id (stage with "
                    "--donor-release-manifest)."
                )

    _check_local_area_checksum_ledger(release_dir, release_manifest, failures)
    _check_local_artifact_hashes(release_dir, release_manifest, failures)

    if failures:
        raise ReleaseContractError(release_dir, failures)


def _validate_uk_dense_release_dir(release_dir: Path, release_id: str) -> None:
    """The dense joint UK line's contract (microcosm#762 A18).

    The non-default local-area role with UK evidence in place of the US
    coverage file: the signed local gate-battery report verified with the
    executor's key (release posture attested, every release-blocking entry
    passed, the scoped spec digests pinned), the head-to-head score against
    the incumbent, and a source-coverage receipt naming the spine, the Ledger
    artifact, the geography ladder, the incumbent extraction, the doctrine,
    the reviewed exclusions and signed deferrals, the holdout and the vintage
    uprating. Artifacts pin to the release id or to one of its immutable
    per-cut tags.
    """

    failures: list[str] = []
    for filename in _UK_DENSE_REQUIRED_RELEASE_FILES:
        if not (release_dir / filename).is_file():
            failures.append(f"required file {filename!r} is missing.")
    release_manifest: Mapping | None = None
    manifest_path = release_dir / "release_manifest.json"
    if manifest_path.is_file():
        release_manifest = _load_json(manifest_path, failures)
    if release_manifest is not None:
        _check_local_area_release_manifest(
            release_manifest,
            release_id,
            failures,
            revision_ok=lambda revision: (
                revision == release_id
                or _UK_DENSE_CUT_TAG_RE.fullmatch(revision) is not None
            ),
        )
        namespace = release_manifest.get("namespace")
        if namespace != _UK_DENSE_NAMESPACE:
            failures.append(
                "release_manifest.json namespace must be "
                f"{_UK_DENSE_NAMESPACE!r}, got {namespace!r}."
            )
    build_manifest_path = release_dir / "build_manifest.json"
    attempt_id: str | None = None
    if build_manifest_path.is_file():
        build_manifest = _load_json(build_manifest_path, failures)
        if build_manifest is not None:
            code = build_manifest.get("code")
            if not isinstance(code, Mapping) or code.get("git_dirty") is not False:
                failures.append(
                    "build_manifest.json code.git_dirty must be false for release."
                )
            candidate_attempt = build_manifest.get("attempt_id")
            if isinstance(candidate_attempt, str) and candidate_attempt:
                attempt_id = candidate_attempt
            else:
                failures.append(
                    "build_manifest.json must carry the calibration 'attempt_id' "
                    "the signed gate report is bound to."
                )
        if build_manifest is not None and build_manifest.get("build_id") != release_id:
            failures.append(
                "build_manifest.json 'build_id' is "
                f"{build_manifest.get('build_id')!r} but the release directory "
                f"is named {release_id!r}."
            )
    gate_path = release_dir / "gate_summary.json"
    if gate_path.is_file():
        gate_summary = _load_json(gate_path, failures)
        if gate_summary is not None:
            _check_local_area_gates(gate_summary, failures)
    diagnostics_path = release_dir / "calibration_diagnostics.json"
    if diagnostics_path.is_file():
        diagnostics = _load_json(diagnostics_path, failures)
        if diagnostics is not None:
            _check_local_area_calibration_diagnostics(diagnostics, failures)
    report_path = release_dir / _UK_DENSE_GATE_REPORT_FILE
    if report_path.is_file():
        report = _load_json(report_path, failures)
        if report is not None:
            _check_uk_dense_gate_report(
                report, failures=failures, attempt_id=attempt_id
            )
            if (
                _artifact_by_path(release_manifest or {}, _UK_DENSE_GATE_REPORT_FILE)
                is None
            ):
                failures.append(
                    "release_manifest.json must declare the signed gate report "
                    f"{_UK_DENSE_GATE_REPORT_FILE!r} as an artifact."
                )
    coverage_path = release_dir / _UK_DENSE_SOURCE_COVERAGE_FILE
    if coverage_path.is_file():
        coverage = _load_json(coverage_path, failures)
        if coverage is not None:
            _check_uk_dense_source_coverage(coverage, failures)
            _check_uk_measure_exclusions(coverage.get("measure_exclusions"), failures)
    _check_uk_dense_surface_files(release_dir, release_manifest or {}, failures)
    _check_local_area_checksum_ledger(
        release_dir,
        release_manifest,
        failures,
        required_files=_UK_DENSE_REQUIRED_RELEASE_FILES,
    )
    _check_local_artifact_hashes(release_dir, release_manifest, failures)
    if failures:
        raise ReleaseContractError(release_dir, failures)


def _check_uk_dense_gate_report(
    report: Mapping, *, failures: list[str], attempt_id: str | None = None
) -> None:
    """Verify the signed local gate-battery report the dense line ships.

    Same producer and signing dance as the exact-k terminal report, scoped to
    the local battery: the six local entries, one terminal phase, the local
    scoped digests, release posture attested, every release-blocking entry
    passed, ``gate_outcomes_sha256`` recomputed, and the HMAC over the
    canonical report re-nulled at the signature slot.
    """

    file = _UK_DENSE_GATE_REPORT_FILE
    core_fields = {
        "schema_version",
        "country",
        "release_id",
        "release_candidate",
        "spec_fingerprint",
        "gates_manifest_sha256",
        "phases",
        "phases_evaluated",
        "blocked_at_phase",
        "shippable",
        "gates",
        "policy_sha256",
        "release_evidence",
        "evidence_sha256",
        "attestation",
    }
    missing = sorted(core_fields - set(map(str, report)))
    if missing:
        failures.append(f"{file} is missing report field(s) {missing}.")
        return
    if attempt_id is not None and report.get("release_id") != attempt_id:
        # Bind the signed report to the run this directory was cut from: a
        # stale but validly signed shippable report dropped into a
        # hand-assembled directory must not pass on its own signature.
        failures.append(
            f"{file} release_id {report.get('release_id')!r} is not the build's "
            f"attempt id {attempt_id!r}; the signed report does not belong to "
            "this run."
        )
    if report.get("schema_version") != _UK_GATE_BATTERY_SCHEMA_VERSION:
        failures.append(
            f"{file} schema_version must be {_UK_GATE_BATTERY_SCHEMA_VERSION}."
        )
    if report.get("country") != "uk":
        failures.append(f"{file} country must be 'uk'.")
    if report.get("release_candidate") is not True:
        failures.append(
            f"{file} release_candidate must be true: the battery ran in the "
            "dev posture."
        )
    if report.get("shippable") is not True:
        failures.append(f"{file} shippable must be true.")
    if report.get("blocked_at_phase") is not None:
        failures.append(f"{file} blocked_at_phase must be null.")
    if list(report.get("phases") or ()) != list(_UK_DENSE_GATE_PHASES):
        failures.append(f"{file} phases must be {list(_UK_DENSE_GATE_PHASES)}.")
    if list(report.get("phases_evaluated") or ()) != list(_UK_DENSE_GATE_PHASES):
        failures.append(
            f"{file} phases_evaluated must be {list(_UK_DENSE_GATE_PHASES)}."
        )
    for field, expected in _UK_DENSE_GATE_DIGESTS.items():
        if report.get(field) != expected:
            failures.append(
                f"{file} {field} does not match the reviewed local gate spec."
            )
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        failures.append(f"{file} gates must be an object.")
        return
    if set(map(str, gates)) != set(_UK_DENSE_GATE_ENTRY_IDS):
        failures.append(
            f"{file} gates must contain exactly the local battery entries "
            f"{sorted(_UK_DENSE_GATE_ENTRY_IDS)}, got {sorted(map(str, gates))}."
        )
    for entry_id, entry in gates.items():
        if not isinstance(entry, Mapping):
            failures.append(f"{file} gate {entry_id!r} must be an object.")
            continue
        status = entry.get("status")
        if status not in _UK_GATE_BATTERY_STATUSES:
            failures.append(f"{file} gate {entry_id!r} has unknown status {status!r}.")
        expected_criticality = (
            "release_blocking"
            if entry_id in _UK_DENSE_RELEASE_BLOCKING_IDS
            else "diagnostic"
        )
        if entry.get("criticality") != expected_criticality:
            failures.append(
                f"{file} gate {entry_id!r} criticality must be "
                f"{expected_criticality!r}."
            )
        if (
            entry_id in _UK_DENSE_RELEASE_BLOCKING_IDS
            and status not in _UK_GATE_BATTERY_SHIPPABLE_STATUSES
        ):
            failures.append(
                f"{file} release-blocking gate {entry_id!r} did not pass ({status!r})."
            )
    attestation = report.get("attestation")
    if not isinstance(attestation, Mapping):
        failures.append(f"{file} attestation must be an object.")
        return
    if attestation.get("schema_version") != _UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION:
        failures.append(
            f"{file} attestation.schema_version must be "
            f"{_UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION}."
        )
    if attestation.get("producer") != _UK_GATE_BATTERY_PRODUCER:
        failures.append(
            f"{file} attestation.producer must name the gate-battery executor."
        )
    if attestation.get("signature_algorithm") != _UK_TERMINAL_GATE_SIGNATURE_ALGORITHM:
        failures.append(
            f"{file} attestation.signature_algorithm must be "
            f"{_UK_TERMINAL_GATE_SIGNATURE_ALGORITHM!r}."
        )
    for field in (
        "release_id",
        "release_candidate",
        "spec_fingerprint",
        "gates_manifest_sha256",
        "policy_sha256",
    ):
        if attestation.get(field) != report.get(field):
            failures.append(f"{file} attestation.{field} disagrees with the report.")
    if attestation.get("gate_outcomes_sha256") != _canonical_sha256(gates):
        failures.append(
            f"{file} attestation.gate_outcomes_sha256 does not match gates."
        )
    if "signing_error" in attestation:
        failures.append(
            f"{file} attestation records a signing error; the report is unsigned."
        )
    verification_key = _uk_gate_battery_verification_key(failures)
    if verification_key is None:
        return
    if (
        attestation.get("signing_key_sha256")
        != hashlib.sha256(verification_key).hexdigest()
    ):
        failures.append(
            f"{file} attestation.signing_key_sha256 does not identify the trusted key."
        )
    unsigned_report = dict(report)
    unsigned_report["attestation"] = {**attestation, "signature": None}
    expected_signature = hmac.new(
        verification_key, _canonical_json_bytes(unsigned_report), hashlib.sha256
    ).hexdigest()
    signature = attestation.get("signature")
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature, expected_signature
    ):
        failures.append(
            f"{file} attestation.signature does not authenticate the report "
            "with the trusted key."
        )


def _check_uk_dense_source_coverage(coverage: Mapping, failures: list[str]) -> None:
    file = _UK_DENSE_SOURCE_COVERAGE_FILE
    for key in _UK_DENSE_SOURCE_COVERAGE_KEYS:
        value = coverage.get(key)
        if not isinstance(value, Mapping) or not value:
            failures.append(f"{file} is missing the coverage object {key!r}.")
    for key, sha_field in (
        ("spine", "sha256"),
        ("geography_ladder", "sha256"),
        ("ledger_artifact", "facts_sha256"),
        ("ledger_artifact", "manifest_sha256"),
    ):
        block = coverage.get(key)
        if isinstance(block, Mapping):
            digest = block.get(sha_field)
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                failures.append(f"{file} {key}.{sha_field} must be a sha256 digest.")
    incumbent = coverage.get("incumbent")
    if isinstance(incumbent, Mapping) and not incumbent.get("snapshot"):
        failures.append(f"{file} incumbent must name the private-repo snapshot.")
    holdout = coverage.get("holdout")
    if isinstance(holdout, Mapping):
        value = holdout.get("mean_holdout_loss")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(
                f"{file} holdout.mean_holdout_loss must be a finite number."
            )


def _check_local_area_checksum_ledger(
    release_dir: Path,
    release_manifest: Mapping | None,
    failures: list[str],
    *,
    required_files: tuple[str, ...] = LOCAL_AREA_REQUIRED_RELEASE_FILES,
) -> None:
    """Validate sha256sums.txt as a real ledger, not a presence token.

    Every required bundle file (and the manifest-declared artifacts,
    including the root H5 by name) must appear with a valid digest; entries
    naming files present in the directory must hash-match; unsafe or
    duplicate names are rejected.
    """

    ledger_path = release_dir / "sha256sums.txt"
    if not ledger_path.is_file():
        return
    entries: dict[str, str] = {}
    for line_number, line in enumerate(ledger_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]):
            failures.append(
                f"sha256sums.txt line {line_number} is not '<sha256>  <filename>'."
            )
            continue
        digest, name = parts[0], parts[1].strip()
        if "/" in name or name.startswith(".."):
            failures.append(
                f"sha256sums.txt line {line_number} names an unsafe path {name!r}."
            )
            continue
        if name in entries:
            failures.append(f"sha256sums.txt lists {name!r} more than once.")
            continue
        entries[name] = digest

    for filename in required_files:
        if filename == "sha256sums.txt":
            continue
        if filename not in entries:
            failures.append(
                f"sha256sums.txt does not cover required file {filename!r}."
            )
    if isinstance(release_manifest, Mapping):
        artifacts = release_manifest.get("artifacts")
        if isinstance(artifacts, Mapping):
            for key, entry in artifacts.items():
                if not isinstance(entry, Mapping):
                    continue
                path = entry.get("path")
                declared = entry.get("sha256")
                if not isinstance(path, str) or not isinstance(declared, str):
                    continue
                if path not in entries:
                    failures.append(
                        f"sha256sums.txt does not cover manifest artifact "
                        f"{key!r} ({path!r})."
                    )
                elif entries[path] != declared:
                    failures.append(
                        f"sha256sums.txt digest for {path!r} disagrees with "
                        "release_manifest.json."
                    )
    for name, digest in entries.items():
        local = release_dir / name
        if local.is_file() and _sha256(local) != digest:
            failures.append(
                f"sha256sums.txt digest for {name!r} does not match the file's bytes."
            )


def _check_local_area_release_manifest(
    manifest: Mapping,
    release_id: str,
    failures: list[str],
    *,
    revision_ok: Callable[[str], bool] | None = None,
) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
        failures.append(
            f"release_manifest.json 'schema_version' is {schema_version!r}; "
            f"this library publishes version {RELEASE_MANIFEST_SCHEMA_VERSION}."
        )
    build = manifest.get("build")
    if not isinstance(build, Mapping) or not build.get("build_id"):
        failures.append("release_manifest.json is missing 'build.build_id'.")
    elif build["build_id"] != release_id:
        failures.append(
            f"release_manifest.json 'build.build_id' is "
            f"{build['build_id']!r} but the release directory is named "
            f"{release_id!r}."
        )
    _check_uk_release_identity(manifest, release_id, failures)
    _check_release_manifest_package(
        manifest.get("data_package"),
        field="data_package",
        # Legacy releases were recorded under the pre-rename package name.
        expected_name=("microcosm-data", "populace-data"),
        failures=failures,
    )
    default_datasets = manifest.get("default_datasets")
    if default_datasets != {}:
        failures.append(
            "release_manifest.json 'default_datasets' must be an empty "
            "object for a non-default local-area release; it claims no "
            f"default slot (got {default_datasets!r})."
        )
    if manifest.get("is_default") is not False:
        failures.append(
            "release_manifest.json 'is_default' must be false for a "
            "non-default local-area release."
        )
    namespace = manifest.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        failures.append(
            "release_manifest.json must declare a non-empty 'namespace' for "
            "a non-default local-area release."
        )
    limitations = manifest.get("reviewed_limitations")
    if not isinstance(limitations, list):
        failures.append(
            "release_manifest.json must carry a 'reviewed_limitations' list "
            "(explicitly empty if none) for a non-default local-area release."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        failures.append(
            "release_manifest.json must declare a non-empty 'artifacts' map."
        )
        return
    microdata = [
        name
        for name, entry in artifacts.items()
        if isinstance(entry, Mapping) and entry.get("kind") == "microdata"
    ]
    if len(microdata) != 1:
        failures.append(
            "release_manifest.json must declare exactly one microdata "
            f"artifact; found {sorted(microdata)!r}."
        )
    for name, entry in artifacts.items():
        if not isinstance(entry, Mapping):
            failures.append(
                f"release_manifest.json artifact {name!r} must be an object."
            )
            continue
        for field in ("kind", "path", "repo_id", "revision", "sha256"):
            if not entry.get(field):
                failures.append(
                    f"release_manifest.json artifact {name!r} is missing {field!r}."
                )
        revision = entry.get("revision")
        pinned = (
            revision == release_id
            if revision_ok is None
            else bool(revision_ok(str(revision)))
        )
        if revision is not None and not pinned:
            failures.append(
                f"release_manifest.json artifact {name!r} revision "
                f"{revision!r} is not pinned to the release id "
                f"{release_id!r}; non-default artifacts are discoverable by "
                "their immutable tag only."
            )
        sha = entry.get("sha256")
        if isinstance(sha, str) and not _SHA256_RE.fullmatch(sha):
            failures.append(
                f"release_manifest.json artifact {name!r} sha256 {sha!r} is "
                "not a 64-hex digest."
            )


def _check_local_area_gates(gate_summary: Mapping, failures: list[str]) -> None:
    gates = gate_summary.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        failures.append("gate_summary.json must carry a non-empty 'gates' object.")
        return
    for name, gate in gates.items():
        if not isinstance(gate, Mapping) or "passed" not in gate:
            failures.append(
                f"gate_summary.json gate {name!r} must be an object with a "
                "'passed' flag."
            )
            continue
        if gate["passed"] is not True:
            failures.append(
                f"gate_summary.json gate {name!r} did not pass; a "
                "non-default local-area release still publishes only green "
                "gates."
            )


def _check_local_area_calibration_diagnostics(
    diagnostics: Mapping, failures: list[str]
) -> None:
    n_targets = diagnostics.get("n_targets")
    targets = diagnostics.get("targets")
    if not isinstance(n_targets, int) or n_targets <= 0:
        failures.append(
            "calibration_diagnostics.json must carry a positive integer 'n_targets'."
        )
    if not isinstance(targets, list) or not targets:
        failures.append(
            "calibration_diagnostics.json must carry a non-empty 'targets' list."
        )
        return
    if isinstance(n_targets, int) and len(targets) != n_targets:
        failures.append(
            f"calibration_diagnostics.json 'targets' has {len(targets)} "
            f"rows but 'n_targets' is {n_targets}."
        )
    required_fields = (
        "target",
        "compiled_target",
        "initial_estimate",
        "final_estimate",
    )
    seen_names: set[str] = set()
    for index, row in enumerate(targets):
        if not isinstance(row, Mapping):
            failures.append(
                f"calibration_diagnostics.json target row {index} must be an object."
            )
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name:
            failures.append(
                f"calibration_diagnostics.json target row {index} is missing "
                "a non-empty 'name'."
            )
        elif name in seen_names:
            failures.append(
                f"calibration_diagnostics.json target name {name!r} appears "
                "more than once."
            )
        else:
            seen_names.add(name)
        for field in required_fields:
            value = row.get(field)
            if field not in row:
                failures.append(
                    f"calibration_diagnostics.json target row {index} is "
                    f"missing {field!r}."
                )
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(
                    f"calibration_diagnostics.json target row {index} field "
                    f"{field!r} must be a finite number."
                )
    households = diagnostics.get("households")
    if not isinstance(households, int) or households <= 0:
        failures.append(
            "calibration_diagnostics.json must carry a positive integer 'households'."
        )
    for field in ("final_loss", "fraction_within_10pct"):
        value = diagnostics.get(field)
        if field not in diagnostics:
            failures.append(f"calibration_diagnostics.json is missing {field!r}.")
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(
                f"calibration_diagnostics.json {field!r} must be a finite number."
            )


def release_dataset_role(release_dir: Path | str) -> str:
    """The release's declared dataset role (default: national default).

    Read from ``release_manifest.json``'s ``dataset_role``. An absent field
    or unreadable manifest is the national default — the pre-role-class
    shape — so every historical release keeps its meaning; the publish path
    separately refuses pointer updates for any non-default role.
    """

    manifest_path = Path(release_dir) / "release_manifest.json"
    if not manifest_path.is_file():
        return NATIONAL_DEFAULT_DATASET_ROLE
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return NATIONAL_DEFAULT_DATASET_ROLE
    if not isinstance(manifest, Mapping):
        return NATIONAL_DEFAULT_DATASET_ROLE
    role = manifest.get("dataset_role", NATIONAL_DEFAULT_DATASET_ROLE)
    return role if isinstance(role, str) and role else NATIONAL_DEFAULT_DATASET_ROLE


def validate_release_dir(release_dir: Path | str) -> None:
    """Check a local release directory against its dataset-role contract.

    The directory name is the build id (``populace-us-2024-<sha>-<date>``)
    and both manifests must agree with the directory about which build this
    is. Which checks apply is keyed off ``release_manifest.json``'s
    ``dataset_role`` (microcosm#398):

    - ``national_default`` (or absent — every pre-role release): the full
      national contract, unchanged — :data:`REQUIRED_RELEASE_FILES`, the
      US critical-target set, national source-coverage enumeration.
    - ``non_default_local_area``: the local-area contract — gates object
      with every gate passed, per-target diagnostics shape, provenance-chain
      source coverage, a reviewed-limitations register, an empty
      ``default_datasets`` map, and artifacts pinned to the release id. The
      national critical-target set deliberately does not apply: the artifact
      is calibrated to a local surface by design.

    TODO(#578 H5 household-count reconciliation): when the first modern UK
    exact-k release is actually cut, bind these manifest/diagnostic counts to
    the household row count read from its shipped H5. There is deliberately no
    stub H5 check before that release artifact exists.

    Args:
        release_dir: The local ``releases/<build_id>`` directory about to be
            published.

    Raises:
        ReleaseContractError: Naming every violation found — missing files,
            unparseable or unversioned manifests, schema drift, and build-id
            mismatches between the manifests and the directory name.
    """
    release_dir = Path(release_dir)
    release_id = release_dir.name
    failures: list[str] = []

    if not release_dir.is_dir():
        raise ReleaseContractError(release_dir, [f"{release_dir} is not a directory."])

    # Role dispatch is strict about a PRESENT dataset_role: only an absent
    # field gets the legacy national semantics; an explicit null, boolean,
    # number, empty string, or unknown value is a contract error rather
    # than a silent fallback.
    role: str = NATIONAL_DEFAULT_DATASET_ROLE
    manifest_probe_path = release_dir / "release_manifest.json"
    if manifest_probe_path.is_file():
        try:
            manifest_probe = json.loads(manifest_probe_path.read_text())
        except (OSError, ValueError):
            manifest_probe = None
        if isinstance(manifest_probe, Mapping) and "dataset_role" in manifest_probe:
            declared_role = manifest_probe["dataset_role"]
            if declared_role not in (
                NATIONAL_DEFAULT_DATASET_ROLE,
                NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
            ):
                raise ReleaseContractError(
                    release_dir,
                    [
                        "release_manifest.json declares unknown dataset_role "
                        f"{declared_role!r}; known roles are "
                        f"{NATIONAL_DEFAULT_DATASET_ROLE!r} and "
                        f"{NON_DEFAULT_LOCAL_AREA_DATASET_ROLE!r}."
                    ],
                )
            role = declared_role
    if role == NON_DEFAULT_LOCAL_AREA_DATASET_ROLE:
        if release_id == _UK_DENSE_RELEASE_ID:
            _validate_uk_dense_release_dir(release_dir, release_id)
            return
        _validate_local_area_release_dir(release_dir, release_id)
        return

    build_manifest: Mapping | None = None
    release_manifest: Mapping | None = None
    calibration_diagnostics: Mapping | None = None
    calibration_diagnostics_sha256: str | None = None
    source_coverage_diagnostics: Mapping | None = None

    for filename in required_release_files(release_id):
        if not (release_dir / filename).is_file():
            failures.append(f"required file {filename!r} is missing.")

    build_manifest_path = release_dir / "build_manifest.json"
    if build_manifest_path.is_file():
        manifest = _load_json(build_manifest_path, failures)
        if manifest is not None:
            build_manifest = manifest
            _check_build_manifest(manifest, release_id, failures)

    release_manifest_path = release_dir / "release_manifest.json"
    if release_manifest_path.is_file():
        manifest = _load_json(release_manifest_path, failures)
        if manifest is not None:
            release_manifest = manifest
            _check_release_manifest(manifest, release_id, failures)

    calibration_diagnostics_path = release_dir / "calibration_diagnostics.json"
    if calibration_diagnostics_path.is_file():
        calibration_diagnostics_sha256 = _sha256(calibration_diagnostics_path)
        diagnostics = _load_json(calibration_diagnostics_path, failures)
        if diagnostics is not None:
            calibration_diagnostics = diagnostics
            _check_calibration_diagnostics(
                diagnostics,
                failures,
                grandfathered_uk_june=release_id == _UK_JUNE_RELEASE_ID,
            )
            if (
                _is_uk_exact_k_release_id(release_id)
                or release_id == _UK_NATIONAL_RELEASE_ID
            ):
                _check_uk_calibration_diagnostics(diagnostics, failures)
            if _is_uk_exact_k_release_id(release_id):
                _check_uk_exact_k_diagnostics_identity(
                    diagnostics, release_id, failures
                )
            if release_id.startswith("populace-us-"):
                _check_us_critical_target_fit(diagnostics, failures)

    terminal_gate_path = release_dir / _UK_TERMINAL_GATE_REPORT_FILE
    if _is_uk_exact_k_release_id(release_id) and terminal_gate_path.is_file():
        terminal_gate_sha256 = _sha256(terminal_gate_path)
        _check_uk_terminal_gate_links(
            build_manifest=build_manifest,
            release_manifest=release_manifest,
            report_sha256=terminal_gate_sha256,
            failures=failures,
        )
        terminal_gate_report = _load_json(terminal_gate_path, failures)
        if terminal_gate_report is not None:
            # Vintage dispatch on the report's own schema: 3 is the legacy
            # aggregator, 4 the shared gate battery. Anything else is not a
            # UK terminal report.
            report_schema = terminal_gate_report.get("schema_version")
            if type(report_schema) is not int:
                # A float 4.0 or string "4" must not route as a vintage.
                report_schema = None
            if report_schema == _UK_GATE_BATTERY_SCHEMA_VERSION:
                _check_uk_gate_battery_report(
                    terminal_gate_report,
                    release_id=release_id,
                    calibration_diagnostics_sha256=calibration_diagnostics_sha256,
                    build_manifest=build_manifest,
                    calibration_diagnostics=calibration_diagnostics,
                    failures=failures,
                )
            elif report_schema == _UK_TERMINAL_GATE_SCHEMA_VERSION:
                _check_uk_terminal_gate_report(
                    terminal_gate_report,
                    release_id=release_id,
                    calibration_diagnostics_sha256=calibration_diagnostics_sha256,
                    build_manifest=build_manifest,
                    calibration_diagnostics=calibration_diagnostics,
                    failures=failures,
                )
            else:
                failures.append(
                    f"{_UK_TERMINAL_GATE_REPORT_FILE} schema_version must be "
                    f"{_UK_TERMINAL_GATE_SCHEMA_VERSION} (legacy aggregator) "
                    f"or {_UK_GATE_BATTERY_SCHEMA_VERSION} (gate battery), "
                    f"got {report_schema!r}."
                )

    certification_path = release_dir / _UK_RELEASE_CERTIFICATION_FILE
    # A release that ships any national-line part must ship the composed
    # verdict: the shippability claim lives only in the certification, so a
    # directory carrying a release-cut report, or a calibration-seam-scoped
    # terminal report, without release_certification.json is refused rather
    # than validating clean by omission. (The id-keyed required-files rule
    # lands with the national publication integration, once the canonical
    # national release-id form exists.)
    national_line_parts = []
    if (release_dir / _UK_RELEASE_CUT_GATE_REPORT_FILE).is_file():
        national_line_parts.append(_UK_RELEASE_CUT_GATE_REPORT_FILE)
    seam_terminal_path = release_dir / _UK_TERMINAL_GATE_REPORT_FILE
    if seam_terminal_path.is_file():
        try:
            probe = json.loads(seam_terminal_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            probe = None
        if isinstance(probe, Mapping) and probe.get("posture") == "calibration_seam":
            national_line_parts.append(
                f"{_UK_TERMINAL_GATE_REPORT_FILE} (posture calibration_seam)"
            )
    if national_line_parts and not certification_path.is_file():
        failures.append(
            f"{_UK_RELEASE_CERTIFICATION_FILE} is missing while national-line "
            f"gate artifacts are present ({', '.join(national_line_parts)}); "
            "a candidate's shippability verdict comes only from the "
            "certification, so its omission cannot validate clean."
        )
    if certification_path.is_file():
        certification = _load_json(certification_path, failures)
        if certification is not None:
            _check_uk_release_certification(
                certification,
                release_id=release_id,
                calibration_diagnostics_sha256=calibration_diagnostics_sha256,
                failures=failures,
            )
            _check_uk_certification_evidence_binding(
                certification,
                release_dir,
                failures,
            )

    _check_cross_manifest_consistency(
        build_manifest,
        release_manifest,
        calibration_diagnostics,
        failures,
    )
    _check_local_artifact_hashes(release_dir, release_manifest, failures)

    source_coverage_path = release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE
    if release_id.startswith("populace-us-") and source_coverage_path.is_file():
        diagnostics = _load_json(source_coverage_path, failures)
        if diagnostics is not None:
            source_coverage_diagnostics = diagnostics
            _check_source_coverage_diagnostics(diagnostics, failures)

    _check_us_fiscal_source_consistency(
        calibration_diagnostics, source_coverage_diagnostics, failures
    )

    if failures:
        raise ReleaseContractError(release_dir, failures)


#: Owner refs in ``known_failures`` must point at a tracked issue — a
#: ``#NNN`` shorthand (optionally repo-qualified) or a GitHub issue/PR URL.
#: A name or a prose excuse is not an owner: the evidence tier ships a
#: failure only when somewhere is accountable for fixing it.
_ISSUE_REF_RE = re.compile(r"#\d+|github\.com/\S+/(?:issues|pull)/\d+")


#: First quoted token in a contract-recomputed critical-fit failure — the
#: fallback binding token when the failure names no diagnostics target.
_QUOTED_TOKEN_RE = re.compile(r"'([^']+)'")


def _token_appears_delimited(token: str, text: str) -> bool:
    """True if ``token`` appears in ``text`` as a whole name, not merely as a
    substring of a longer one (``ctc_amount@2024`` must not be satisfied by
    an entry about ``actc_amount@2024``)."""
    return (
        re.search(
            rf"(?<![A-Za-z0-9_.]){re.escape(token)}(?![A-Za-z0-9_])",
            text,
        )
        is not None
    )


def _recorded_failure_subset_binding(
    recorded: object,
    *,
    recorded_texts: list[str],
    source: str,
    failures: list[str],
    verbatim: bool,
) -> None:
    """Require every recorded failure string to ride into ``known_failures``.

    ``verbatim`` demands exact-string membership; otherwise substring
    containment suffices (the builder records some families with a
    gate-family prefix around the raw string). Fail-closed on shape: a
    record that is not a list of strings is itself a violation — absence
    must never silently disable the binding.
    """
    if not isinstance(recorded, list) or any(
        not isinstance(failure, str) for failure in recorded
    ):
        failures.append(
            f"{source} must be a list of strings for an evidence release; "
            "the known_failures binding cannot be verified otherwise."
        )
        return
    combined = "\n".join(recorded_texts)
    if verbatim:
        missing = [failure for failure in recorded if failure not in recorded_texts]
        requirement = "verbatim"
    else:
        missing = [failure for failure in recorded if failure not in combined]
        requirement = "within an entry"
    if missing:
        failures.append(
            f"release_manifest.json known_failures must carry every {source} "
            f"entry {requirement}; missing {_sample_values(missing)}."
        )


def _check_evidence_known_failures_binding(
    *,
    release_manifest: Mapping | None,
    build_manifest: Mapping | None,
    calibration_diagnostics: Mapping | None,
    source_coverage_diagnostics: Mapping | None,
    recomputed_critical_failures: list[str],
    failures: list[str],
) -> None:
    """Bind ``known_failures`` to what the artifact itself records.

    The evidence tier's honesty cannot rest on trusting the manifest author:
    everything a local validator can recompute or read back must be
    acknowledged, and every locally checkable record is REQUIRED to be
    present and well-shaped (fail-closed — deleting a record must never
    disable its binding). A hand-edited manifest that softens or drops a
    recorded failure fails here:

    - ``build_manifest.json`` ``gates.calibration.failures`` — verbatim
      membership in ``known_failures``;
    - ``calibration_diagnostics.json`` ``build.release_gates.failures``
      (the run's merged terminal record) — verbatim membership;
    - ``us_source_coverage.json`` ``gate.failures`` — containment (the
      builder records these with a gate-family prefix);
    - every critical-target breach recomputed from the diagnostics (the
      same check the certified contract enforces as a hard refusal) —
      acknowledged by delimited name in the record.

    The converse direction is deliberately open: ``known_failures`` may
    carry entries beyond what is locally recomputable, so the tier can
    over-disclose but never under-disclose. The binding guarantees each
    recorded failure is NAMED with an owner — prose sentiment around it is
    for the human review the #490 register pattern already requires.
    """
    if release_manifest is None:
        return
    known_failures = release_manifest.get("known_failures")
    if not isinstance(known_failures, list):
        return  # the shape failure is already recorded
    recorded_texts = [
        entry.get("failure")
        for entry in known_failures
        if isinstance(entry, Mapping) and isinstance(entry.get("failure"), str)
    ]
    combined = "\n".join(recorded_texts)
    if build_manifest is not None:
        gates = build_manifest.get("gates")
        calibration = gates.get("calibration") if isinstance(gates, Mapping) else None
        _recorded_failure_subset_binding(
            calibration.get("failures") if isinstance(calibration, Mapping) else None,
            recorded_texts=recorded_texts,
            source="build_manifest.json gates.calibration.failures",
            failures=failures,
            verbatim=True,
        )
    if calibration_diagnostics is not None:
        build = calibration_diagnostics.get("build")
        release_gates = (
            build.get("release_gates") if isinstance(build, Mapping) else None
        )
        _recorded_failure_subset_binding(
            (
                release_gates.get("failures")
                if isinstance(release_gates, Mapping)
                else None
            ),
            recorded_texts=recorded_texts,
            source="calibration_diagnostics.json build.release_gates.failures",
            failures=failures,
            verbatim=True,
        )
    if source_coverage_diagnostics is not None:
        gate = source_coverage_diagnostics.get("gate")
        _recorded_failure_subset_binding(
            gate.get("failures") if isinstance(gate, Mapping) else None,
            recorded_texts=recorded_texts,
            source=f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} gate.failures",
            failures=failures,
            verbatim=False,
        )
    diagnostic_target_names = _diagnostic_target_names(calibration_diagnostics)
    for recomputed in recomputed_critical_failures:
        tokens = [
            name
            for name in diagnostic_target_names
            if _token_appears_delimited(name, recomputed)
        ]
        if not tokens:
            match = _QUOTED_TOKEN_RE.search(recomputed)
            tokens = [match.group(1)] if match else []
        unacknowledged = [
            token for token in tokens if not _token_appears_delimited(token, combined)
        ]
        if not tokens or unacknowledged:
            failures.append(
                "release_manifest.json known_failures must acknowledge the "
                "critical-target breach naming "
                f"{unacknowledged or [recomputed]}; the evidence tier "
                "records failures, it never hides them."
            )


def _diagnostic_target_names(calibration_diagnostics: Mapping | None) -> list[str]:
    if calibration_diagnostics is None:
        return []
    targets = calibration_diagnostics.get("targets")
    if not isinstance(targets, list):
        return []
    return [
        str(target["name"])
        for target in targets
        if isinstance(target, Mapping) and target.get("name")
    ]


def _check_evidence_release_manifest(manifest: Mapping, failures: list[str]) -> None:
    """Evidence-only manifest requirements: the tier marker and the honest
    non-empty ``known_failures`` record."""
    if manifest.get("tier") != "evidence":
        failures.append(
            "release_manifest.json 'tier' must be 'evidence' for an "
            "evidence-tier release."
        )
    known_failures = manifest.get("known_failures")
    if not isinstance(known_failures, list) or not known_failures:
        failures.append(
            "release_manifest.json must declare a non-empty 'known_failures' "
            "list; the evidence tier exists to carry recorded gate failures "
            "honestly, never to hide them (an all-green artifact belongs on "
            "the certified path)."
        )
        return
    for index, entry in enumerate(known_failures):
        owner_prefix = f"release_manifest.json known_failures[{index}]"
        if not isinstance(entry, Mapping):
            failures.append(
                f"{owner_prefix} must be an object with 'failure' and 'owner'."
            )
            continue
        failure_text = entry.get("failure")
        if not isinstance(failure_text, str) or not failure_text.strip():
            failures.append(
                f"{owner_prefix}.failure must be the recorded gate-failure "
                "string, verbatim and non-empty."
            )
        owner = entry.get("owner")
        if not isinstance(owner, str) or not _ISSUE_REF_RE.search(owner):
            failures.append(
                f"{owner_prefix}.owner must carry an issue reference "
                "(e.g. 'PolicyEngine/microcosm#487' or an issue URL)."
            )


def validate_evidence_release_dir(release_dir: Path | str) -> None:
    """Check a local EVIDENCE-tier release directory against its contract.

    The sibling of :func:`validate_release_dir` for the best-available
    artifact when terminal gates failed (microcosm#506): the same required
    files and the same shape, provenance, and cross-manifest checks, with the
    gate *verdict* requirements replaced by a recording requirement — the
    release manifest must carry a non-empty ``known_failures`` block naming
    every recorded gate failure verbatim, each with an owner issue.

    This is a different output contract, not a bypass of the certified one:

    - the release manifest must declare
      :data:`EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION` and ``tier:
      "evidence"``, which the certified contract structurally rejects;
    - the release id must carry the :data:`EVIDENCE_RELEASE_ID_SEGMENT`, so
      the tier is visible in every tag and download path;
    - critical-target fit and gate-passed requirements are not enforced —
      their failures are exactly what ``known_failures`` records — but
      everything that makes the artifact *auditable* (required files, build
      provenance with a clean git commit, artifact hashes, cross-manifest
      agreement) is enforced unchanged.

    Scope (microcosm#506, "dense first"): the tier exists for the US
    national artifact. Non-default local-area releases and UK exact-k
    releases are refused outright — each has its own certification lane
    (microcosm#398, microcosm#611) and no evidence-tier semantics have been
    adjudicated for them.

    Args:
        release_dir: The local ``releases/<build_id>`` directory about to be
            published at the evidence tier.

    Raises:
        ReleaseContractError: Naming every violation found.
    """
    release_dir = Path(release_dir)
    release_id = release_dir.name
    failures: list[str] = []

    if not release_dir.is_dir():
        raise ReleaseContractError(release_dir, [f"{release_dir} is not a directory."])

    if EVIDENCE_RELEASE_ID_SEGMENT not in release_id:
        failures.append(
            f"evidence release ids must carry the "
            f"{EVIDENCE_RELEASE_ID_SEGMENT!r} segment; {release_id!r} does "
            "not name its tier."
        )

    if not release_id.startswith("populace-us-"):
        # Without the US prefix every country-specific requirement above the
        # generic three files silently deactivates — an out-of-scope id must
        # not buy a weaker contract.
        failures.append(
            "the evidence tier is scoped to US national releases "
            f"(microcosm#506); release id {release_id!r} is out of scope."
        )

    if _is_uk_exact_k_release_id(release_id):
        raise ReleaseContractError(
            release_dir,
            [
                "UK exact-k releases have no evidence-tier contract; the "
                "gate-battery lane (microcosm#611) owns their verdicts and "
                "microcosm#506 scoped the evidence tier to the US national "
                "artifact."
            ],
        )

    manifest_probe_path = release_dir / "release_manifest.json"
    if manifest_probe_path.is_file():
        try:
            manifest_probe = json.loads(manifest_probe_path.read_text())
        except (OSError, ValueError):
            manifest_probe = None
        if isinstance(manifest_probe, Mapping) and "dataset_role" in manifest_probe:
            declared_role = manifest_probe["dataset_role"]
            if declared_role != NATIONAL_DEFAULT_DATASET_ROLE:
                raise ReleaseContractError(
                    release_dir,
                    [
                        "the evidence tier supports only "
                        f"{NATIONAL_DEFAULT_DATASET_ROLE!r} releases "
                        f"(microcosm#506); dataset_role {declared_role!r} "
                        "has its own contract and no evidence-tier "
                        "semantics."
                    ],
                )

    build_manifest: Mapping | None = None
    release_manifest: Mapping | None = None
    calibration_diagnostics: Mapping | None = None
    source_coverage_diagnostics: Mapping | None = None

    for filename in required_release_files(release_id):
        if not (release_dir / filename).is_file():
            failures.append(f"required file {filename!r} is missing.")

    build_manifest_path = release_dir / "build_manifest.json"
    if build_manifest_path.is_file():
        manifest = _load_json(build_manifest_path, failures)
        if manifest is not None:
            build_manifest = manifest
            _check_build_manifest(manifest, release_id, failures)

    release_manifest_path = release_dir / "release_manifest.json"
    if release_manifest_path.is_file():
        manifest = _load_json(release_manifest_path, failures)
        if manifest is not None:
            release_manifest = manifest
            _check_release_manifest(
                manifest,
                release_id,
                failures,
                expected_schema_version=EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION,
            )
            _check_evidence_release_manifest(manifest, failures)

    recomputed_critical_failures: list[str] = []
    calibration_diagnostics_path = release_dir / "calibration_diagnostics.json"
    if calibration_diagnostics_path.is_file():
        diagnostics = _load_json(calibration_diagnostics_path, failures)
        if diagnostics is not None:
            calibration_diagnostics = diagnostics
            _check_calibration_diagnostics(diagnostics, failures)
            # Critical-fit breaches are permitted at the evidence tier — but
            # never silently. Recompute the certified verdicts into a scratch
            # list and require each breach to be acknowledged in
            # known_failures (binding check below).
            if release_id.startswith("populace-us-"):
                _check_us_critical_target_fit(diagnostics, recomputed_critical_failures)

    _check_cross_manifest_consistency(
        build_manifest,
        release_manifest,
        calibration_diagnostics,
        failures,
    )
    _check_local_artifact_hashes(release_dir, release_manifest, failures)

    source_coverage_path = release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE
    if release_id.startswith("populace-us-") and source_coverage_path.is_file():
        diagnostics = _load_json(source_coverage_path, failures)
        if diagnostics is not None:
            source_coverage_diagnostics = diagnostics
            _check_source_coverage_diagnostics(
                diagnostics,
                failures,
                require_gate_passed=False,
            )

    _check_evidence_known_failures_binding(
        release_manifest=release_manifest,
        build_manifest=build_manifest,
        calibration_diagnostics=calibration_diagnostics,
        source_coverage_diagnostics=source_coverage_diagnostics,
        recomputed_critical_failures=recomputed_critical_failures,
        failures=failures,
    )

    _check_us_fiscal_source_consistency(
        calibration_diagnostics, source_coverage_diagnostics, failures
    )

    if failures:
        raise ReleaseContractError(release_dir, failures)


def _check_us_fiscal_source_consistency(
    calibration_diagnostics: Mapping | None,
    source_coverage_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    if calibration_diagnostics is None or source_coverage_diagnostics is None:
        return
    targets = calibration_diagnostics.get("targets")
    fiscal_sources = source_coverage_diagnostics.get("fiscal_target_sources")
    if not isinstance(targets, list) or not isinstance(fiscal_sources, Mapping):
        return
    calibrated_family_counts: dict[str, int] = {}
    for family in (
        registry.get("family")
        for target in targets
        if isinstance(target, Mapping)
        for registry in (target.get("registry"),)
        if isinstance(registry, Mapping) and registry.get("family")
    ):
        calibrated_family_counts[str(family)] = (
            calibrated_family_counts.get(str(family), 0) + 1
        )
    calibrated_families = set(calibrated_family_counts)
    missing = sorted(
        str(family) for family in calibrated_families - fiscal_sources.keys()
    )
    if missing:
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} fiscal_target_sources must "
            f"cover every calibrated target family; missing {missing}."
        )
    unexpected = sorted(
        str(family) for family in fiscal_sources.keys() - calibrated_families
    )
    if unexpected:
        failures.append(
            f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} fiscal_target_sources must "
            f"only describe calibrated target families; unexpected {unexpected}."
        )
    for family, expected_count in sorted(calibrated_family_counts.items()):
        source = fiscal_sources.get(family)
        if not isinstance(source, Mapping):
            continue
        target_count = source.get("target_count")
        if target_count != expected_count:
            failures.append(
                f"{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE} "
                f"fiscal_target_sources[{family!r}].target_count is "
                f"{target_count!r} but calibration_diagnostics.json has "
                f"{expected_count} calibrated target(s) for that family."
            )


def _check_cross_manifest_consistency(
    build_manifest: Mapping | None,
    release_manifest: Mapping | None,
    calibration_diagnostics: Mapping | None,
    failures: list[str],
) -> None:
    """Fields duplicated across files must agree exactly."""
    if build_manifest is not None and calibration_diagnostics is not None:
        calibration = build_manifest.get("calibration")
        build_surface = (
            calibration.get("target_surface")
            if isinstance(calibration, Mapping)
            else None
        )
        diagnostics_surface = calibration_diagnostics.get("target_surface")
        if isinstance(build_surface, Mapping) and isinstance(
            diagnostics_surface, Mapping
        ):
            for field in ("sha256", "n_targets"):
                if build_surface.get(field) != diagnostics_surface.get(field):
                    failures.append(
                        "build_manifest.json calibration.target_surface."
                        f"{field} must match calibration_diagnostics.json "
                        f"target_surface.{field}."
                    )
        build_registry = (
            calibration.get("target_registry")
            if isinstance(calibration, Mapping)
            else None
        )
        diagnostics_registry = calibration_diagnostics.get("target_registry")
        if isinstance(build_registry, Mapping) and isinstance(
            diagnostics_registry, Mapping
        ):
            for field in ("version", "n_specs"):
                if build_registry.get(field) != diagnostics_registry.get(field):
                    failures.append(
                        "build_manifest.json calibration.target_registry."
                        f"{field} must match calibration_diagnostics.json "
                        f"target_registry.{field}."
                    )

    if build_manifest is not None and release_manifest is not None:
        dataset = build_manifest.get("dataset")
        if isinstance(dataset, Mapping):
            _check_root_artifact_matches_build_manifest(
                release_manifest,
                path=dataset.get("filename"),
                sha256=dataset.get("sha256"),
                description="dataset",
                failures=failures,
            )
            _check_default_dataset_matches_build_manifest(
                release_manifest,
                path=dataset.get("filename"),
                sha256=dataset.get("sha256"),
                failures=failures,
            )
        calibration = build_manifest.get("calibration")
        if isinstance(calibration, Mapping):
            _check_root_artifact_matches_build_manifest(
                release_manifest,
                path=calibration.get("filename"),
                sha256=calibration.get("sha256"),
                description="calibration",
                failures=failures,
            )

    if release_manifest is not None and calibration_diagnostics is not None:
        artifacts = release_manifest.get("artifacts")
        if isinstance(artifacts, Mapping):
            diagnostics_artifact = artifacts.get("calibration_diagnostics")
            if isinstance(
                diagnostics_artifact, Mapping
            ) and not diagnostics_artifact.get("sha256"):
                failures.append(
                    "release_manifest.json artifact 'calibration_diagnostics' "
                    "must record the diagnostics sha256."
                )


def _check_default_dataset_matches_build_manifest(
    release_manifest: Mapping,
    *,
    path: object,
    sha256: object,
    failures: list[str],
) -> None:
    default_datasets = release_manifest.get("default_datasets")
    artifacts = release_manifest.get("artifacts")
    if not isinstance(default_datasets, Mapping) or not isinstance(artifacts, Mapping):
        return
    default_key = default_datasets.get("national")
    if not isinstance(default_key, str):
        return
    default_artifact = artifacts.get(default_key)
    if not isinstance(default_artifact, Mapping):
        return
    if default_artifact.get("path") != path:
        failures.append(
            "release_manifest.json 'default_datasets.national' must point to "
            "the dataset root artifact declared by build_manifest.json."
        )
    if default_artifact.get("sha256") != sha256:
        failures.append(
            "release_manifest.json default dataset artifact must have sha256 "
            "matching build_manifest.json."
        )


def _check_root_artifact_matches_build_manifest(
    release_manifest: Mapping,
    *,
    path: object,
    sha256: object,
    description: str,
    failures: list[str],
) -> None:
    if not isinstance(path, str) or not path:
        return
    artifact = _artifact_by_path(release_manifest, path)
    if artifact is None:
        failures.append(
            f"release_manifest.json artifacts must include the {description} "
            f"root artifact {path!r} declared by build_manifest.json."
        )
        return
    if artifact.get("sha256") != sha256:
        failures.append(
            f"release_manifest.json artifact for {description} root artifact "
            f"{path!r} must have sha256 matching build_manifest.json."
        )


def _check_uk_measure_exclusions(
    exclusions: object, failures: list[str], *, today: date | None = None
) -> None:
    """Validate time-limited measure approvals using the current validation date.

    The optional clock is for tests. Production assembly and upload
    validation call this without a date and never trust a saved assembly date.
    This does not govern separate support or binding adjudication registers.
    """
    today = date.today() if today is None else today
    if not isinstance(exclusions, Mapping) or not exclusions:
        failures.append("measure_exclusions must carry time-limited approval records.")
        return
    for name, record in exclusions.items():
        label = f"measure exclusion {name!r}"
        if not isinstance(record, Mapping):
            failures.append(f"{label} must be an approval record.")
            continue
        for field in ("reason", "tracking", "approved_by", "adjudication"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                failures.append(f"{label} requires {field} provenance.")
        dates = {}
        for field in ("approved_on", "expires_on"):
            value = record.get(field)
            try:
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None
                ):
                    raise ValueError
                dates[field] = date.fromisoformat(value)
            except ValueError:
                failures.append(f"{label} requires valid ISO {field}, got {value!r}.")
        if len(dates) == 2:
            if dates["approved_on"] > dates["expires_on"]:
                failures.append(f"{label} approved_on is after expires_on.")
            if today < dates["approved_on"]:
                failures.append(f"{label} approved_on is in the future.")
            if today > dates["expires_on"]:
                failures.append(f"{label} expired {record['expires_on']}.")


# Mirrors the existing local target-fit/per-family limits, lockstep-tested.
_UK_DENSE_SURFACE_LIMITS = {
    "max_abs_relative_error": 0.25,
    "within": 0.1,
    "min_family_share": 0.5,
    "hard_within": 0.25,
    "min_hard_family_share": 0.5,
    "min_family_size": 5,
}
_UK_DENSE_SURFACE_INVENTORIES = {
    "national": {
        "rows": 637,
        "sha256": "2ac67154d96a45900ba96f1fdf934d144560557d4afe692f2c71151792e0dffb",
    },
    "local": {
        "rows": 23545,
        "sha256": "1d783e5933ca632250a9e38f8ddd00223f8be93e4878973b148fb72620cf1e23",
    },
}
_UK_DENSE_SURFACE_FILE = "incumbent_surface_evaluation.json"


def _uk_incumbent_row_identity(row: Mapping, grain: str) -> dict:
    fields = (
        ("incumbent_name", "incumbent_target", "family")
        if grain == "national"
        else (
            "incumbent_name",
            "incumbent_target",
            "area_type",
            "geography_id",
            "incumbent_metric",
        )
    )
    return {field: row.get(field) for field in fields}


def uk_incumbent_surface_assessment(payload: Mapping) -> dict:
    """Recompute unchanged absolute quality limits over the full incumbent surface.

    National comparisons use incumbent target values, not invented realized
    incumbent estimates. Local comparisons additionally require the incumbent's
    measured estimates. Deferred and unmeasurable rows never disappear.
    """
    failures = []
    grains = {}
    limits = _UK_DENSE_SURFACE_LIMITS
    for grain in ("national", "local"):
        rows = payload.get(f"{grain}_rows")
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(r, Mapping) for r in rows)
        ):
            failures.append(f"{grain} rows must be a complete nonempty row list.")
            continue
        identities = sorted(
            (_uk_incumbent_row_identity(r, grain) for r in rows),
            key=lambda r: str(r["incumbent_name"]),
        )
        inventory = {"rows": len(rows), "sha256": _canonical_sha256(identities)}
        if inventory != _UK_DENSE_SURFACE_INVENTORIES.get(grain):
            failures.append(
                f"{grain} rows do not match the pinned incumbent target inventory."
            )
        names = [r.get("incumbent_name") for r in rows]
        if any(not isinstance(n, str) or not n for n in names) or len(
            set(names)
        ) != len(names):
            failures.append(f"{grain} row names must be nonempty and unique.")
        measured = 0
        errors = []
        families = {}
        for row in rows:
            name = row.get("incumbent_name")
            family = (
                row.get("family")
                if grain == "national"
                else row.get("incumbent_metric")
            )
            if not isinstance(family, str) or not family:
                failures.append(f"{grain}/{name}: family is missing.")
                continue
            bucket = families.setdefault(
                family, {"rows": 0, "measured": 0, "within_10": 0, "within_25": 0}
            )
            bucket["rows"] += 1
            if not isinstance(row.get("status"), str) or not row["status"]:
                failures.append(f"{grain}/{name}: measurement status is missing.")
            values = ("incumbent_target", "candidate_estimate") + (
                ("incumbent_estimate",) if grain == "local" else ()
            )
            invalid = [
                k
                for k in values
                if isinstance(row.get(k), bool)
                or not isinstance(row.get(k), int | float)
                or not math.isfinite(row[k])
            ]
            if invalid:
                failures.append(
                    f"{grain}/{name}: missing/non-finite required measurement {invalid}; status={row.get('status')!r}, reason={row.get('skip_reason') or row.get('status_detail')!r}."
                )
                continue
            target = row["incumbent_target"]
            error = (
                abs(row["candidate_estimate"] - target) / abs(target)
                if target
                else abs(row["candidate_estimate"])
            )
            errors.append(error)
            measured += 1
            bucket["measured"] += 1
            bucket["within_10"] += error <= limits["within"]
            bucket["within_25"] += error <= limits["hard_within"]
            if error > limits["max_abs_relative_error"]:
                failures.append(
                    f"{grain}/{name}: absolute relative error {error:.8g} > {limits['max_abs_relative_error']}."
                )
        for family, bucket in sorted(families.items()):
            # Full family denominator: missing measurements cannot improve fit.
            for field, threshold in (("within_25", "min_hard_family_share"),):
                if (
                    bucket["rows"] >= limits["min_family_size"]
                    and bucket[field] / bucket["rows"] < limits[threshold]
                ):
                    failures.append(
                        f"{grain}/{family}: {field} share {bucket[field] / bucket['rows']:.8g} < {limits[threshold]}."
                    )
        grains[grain] = {
            "rows": len(rows),
            "measured": measured,
            "maximum_abs_relative_error": max(errors) if errors else None,
            "families": dict(sorted(families.items())),
        }
    return {
        "limits": dict(limits),
        "comparison_basis": {
            "national": "candidate_vs_pinned_incumbent_targets",
            "local": "candidate_and_incumbent_estimates_vs_pinned_incumbent_targets",
        },
        "grains": grains,
        "passed": not failures,
        "failures": failures,
    }


def _check_uk_incumbent_surface_evaluation(
    payload: Mapping, failures: list[str], *, expected_identity: Mapping
) -> None:
    label = _UK_DENSE_SURFACE_FILE
    if (
        payload.get("schema_version") != 2
        or payload.get("kind") != "uk_incumbent_surface_evaluation"
    ):
        failures.append(
            f"{label} requires schema 2 with authenticated input identities."
        )
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        failures.append(f"{label} requires identity.")
    else:
        for key, expected in expected_identity.items():
            actual = identity.get(key)
            if key.endswith("sha256") and (
                not isinstance(actual, str)
                or re.fullmatch(r"[0-9a-f]{64}", actual) is None
            ):
                failures.append(f"{label} identity.{key} must be a SHA256 digest.")
            if expected is None or actual != expected:
                failures.append(
                    f"{label} identity.{key} does not match its source artifact."
                )
    resolution = payload.get("measure_resolution")
    if (
        not isinstance(resolution, Mapping)
        or type(resolution.get("blocks")) is not int
        or resolution.get("blocks") != 1
    ):
        failures.append(f"{label} requires single-block engine measurement.")
    assessment = uk_incumbent_surface_assessment(payload)
    if payload.get("summary") != assessment:
        failures.append(f"{label} summary does not match its measured rows.")
    failures.extend(f"{label}: {failure}" for failure in assessment["failures"])


def _check_uk_dense_surface_files(
    release_dir: Path, release_manifest: Mapping, failures: list[str]
) -> None:
    names = (
        _UK_DENSE_SURFACE_FILE,
        "rowwise_candidate_manifest.json",
        "incumbent_manifest.json",
        "source_calibration_diagnostics.json",
    )
    loaded = {}
    hashes = {}
    for name in names:
        path = release_dir / name
        if not path.is_file():
            continue
        loaded[name] = _load_json(path, failures)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if _artifact_by_path(release_manifest, name) is None:
            failures.append(
                f"release_manifest.json must declare {name!r} as an artifact."
            )
    if len(loaded) != len(names) or any(v is None for v in loaded.values()):
        return
    candidate = loaded["rowwise_candidate_manifest.json"]
    incumbent = loaded["incumbent_manifest.json"]
    try:
        ledger = candidate["identity"]["ledger"]
        dataset = _artifact_by_path(release_manifest, "microcosm_uk_2025_dense.h5")
        expected = {
            "candidate_dataset_sha256": dataset["sha256"],
            "candidate_manifest_sha256": hashes["rowwise_candidate_manifest.json"],
            "candidate_diagnostics_sha256": hashes[
                "source_calibration_diagnostics.json"
            ],
            "ledger_facts_sha256": ledger["facts_sha256"],
            "ledger_manifest_sha256": ledger["manifest_sha256"],
            "incumbent_manifest_sha256": hashes["incumbent_manifest.json"],
            "incumbent_metrics_sha256": incumbent["outputs"]["metrics"]["sha256"],
            "incumbent_weights_sha256": incumbent["outputs"]["weights"]["sha256"],
        }
        for key, identity_key in (
            ("dataset", "candidate_dataset_sha256"),
            ("calibration_diagnostics", "candidate_diagnostics_sha256"),
        ):
            if candidate["outputs"][key]["sha256"] != expected[identity_key]:
                failures.append(
                    f"rowwise_candidate_manifest.json {key} hash does not match released evidence."
                )
        build = _load_json(release_dir / "build_manifest.json", failures)
        report = _load_json(release_dir / _UK_DENSE_GATE_REPORT_FILE, failures)
        if (
            candidate["outputs"]["local_gate_report"]["sha256"]
            != hashlib.sha256(
                (release_dir / _UK_DENSE_GATE_REPORT_FILE).read_bytes()
            ).hexdigest()
        ):
            failures.append(
                "rowwise_candidate_manifest.json gate report does not match released attempt."
            )
        if build and candidate["identity"]["code"].get("git_dirty") is not False:
            failures.append(
                "rowwise_candidate_manifest.json identity.code.git_dirty must be false."
            )
        if build and build.get("code", {}).get("git_commit") != candidate["identity"][
            "code"
        ].get("git_commit"):
            failures.append(
                "build_manifest.json code does not match the evaluated candidate manifest."
            )
        _check_uk_measure_exclusions(candidate.get("measure_exclusions"), failures)
        coverage = _load_json(release_dir / _UK_DENSE_SOURCE_COVERAGE_FILE, failures)
        if coverage:
            if coverage.get("measure_exclusions") != candidate.get(
                "measure_exclusions"
            ):
                failures.append(
                    "uk_source_coverage.json measure_exclusions do not match the original candidate approvals."
                )
            for key in ("facts_sha256", "manifest_sha256"):
                if coverage.get("ledger_artifact", {}).get(key) != ledger[key]:
                    failures.append(
                        f"uk_source_coverage.json Ledger {key} does not match the evaluated source."
                    )
            if coverage.get("incumbent", {}).get("snapshot") != incumbent.get("inputs"):
                failures.append(
                    "uk_source_coverage.json incumbent snapshot does not match the evaluated source."
                )
        shipped_diagnostics = _load_json(
            release_dir / "calibration_diagnostics.json", failures
        )
        original_diagnostics = loaded["source_calibration_diagnostics.json"]
        if shipped_diagnostics:
            if (
                shipped_diagnostics.get("source_diagnostics_sha256")
                != expected["candidate_diagnostics_sha256"]
            ):
                failures.append(
                    "calibration_diagnostics.json source hash does not match the evaluated source."
                )
            if any(
                shipped_diagnostics.get(key) != value
                for key, value in original_diagnostics.items()
            ):
                failures.append(
                    "calibration_diagnostics.json changed original evaluated diagnostic values."
                )
        score = _load_json(release_dir / _UK_DENSE_SCORE_RECEIPT_FILE, failures)
        for key, identity_key in (
            ("candidate_diagnostics", "candidate_diagnostics_sha256"),
            ("incumbent_household_metrics", "incumbent_metrics_sha256"),
            ("incumbent_wide_weights", "incumbent_weights_sha256"),
        ):
            if (
                not score
                or score.get("artifacts", {}).get(key, {}).get("sha256")
                != expected[identity_key]
            ):
                failures.append(
                    f"score_vs_incumbent.json {key} does not match the evaluated source."
                )
        if build and report and report.get("release_id") != build.get("attempt_id"):
            failures.append(
                "incumbent evaluation source manifest is not bound to released attempt."
            )
    except (KeyError, TypeError) as error:
        failures.append(f"incumbent surface source identities are incomplete: {error}.")
        return
    _check_uk_incumbent_surface_evaluation(
        loaded[_UK_DENSE_SURFACE_FILE], failures, expected_identity=expected
    )
