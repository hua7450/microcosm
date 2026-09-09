"""The release contract: every published release looks the same, loudly.

These are behavioral tests against the failure modes already observed on the
Hub: a release with no build manifest at all (1abddeb), and two coexisting
release-manifest schemas (an unversioned early shape next to
``schema_version: 1``). A valid release passes silently; every broken release
fails with each violation named.
"""

import base64
import hashlib
import hmac
import json
import shutil
from pathlib import Path

import pytest

from microcosm.data import (
    EVIDENCE_RELEASE_ID_SEGMENT,
    EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION,
    RELEASE_MANIFEST_SCHEMA_VERSION,
    US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    ReleaseContractError,
    contract,
    required_release_files,
    validate_evidence_release_dir,
    validate_release_dir,
)

RELEASE_ID = "populace-us-2024-9f1260b-20260611"
UK_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
UK_EXACT_K_RELEASE_ID = "populace-uk-2023-frs-k535080"
UK_NATIONAL_RELEASE_ID = "microcosm-uk-2024-25-national"
UK_NATIONAL_CUT_TAG = f"{UK_NATIONAL_RELEASE_ID}-20260828T101112Z-1a2b3c4d"
UK_RECORD_COUNT = 535_080
UK_TERMINAL_GATE_REPORT_FILE = "terminal_gates.json"
UK_TERMINAL_GATE_PRODUCER = (
    "microcosm.build.uk_runtime.terminal_gates.uk_terminal_gate_report"
)
# The current reviewed policy digest (with the #630 source_year degenerate
# exclusion in the committed register); the legacy digest is what the
# grandfathered June release attests.
UK_TERMINAL_GATE_POLICY_SHA256 = (
    "ae93bd10a02362a523eb077bcbd32b362cef31f0447acbc40537df696e30c757"
)
UK_TERMINAL_GATE_POLICY_SHA256_LEGACY = (
    "74c9cd474d76e2b8d4ca5b298c19fc6348ac1a90746594afc8a81283a0398b68"
)
UK_TERMINAL_GATE_SIGNATURE_ALGORITHM = "hmac-sha256"
UK_TERMINAL_GATE_SIGNING_KEY_ENV = "POPULACE_UK_TERMINAL_GATE_SIGNING_KEY"
TEST_UK_TERMINAL_GATE_SIGNING_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="
TEST_UK_TERMINAL_GATE_SIGNING_KEY_BYTES = base64.b64decode(
    TEST_UK_TERMINAL_GATE_SIGNING_KEY
)
FORGED_UK_TERMINAL_GATE_SIGNING_KEY_BYTES = base64.b64decode(
    "ZmVkY2JhOTg3NjU0MzIxMGZlZGNiYTk4NzY1NDMyMTA="
)
UK_ALWAYS_APPLICABLE_GATE_NAMES = (
    "uk_release_input_coverage",
    "degenerate_release_surface",
    "zero_weight_strata",
    "weight_ess",
    "weight_ratio",
)
UK_EVIDENCE_GATE_NAMES = {
    "hmrc_spi_income": ("weights_audit",),
    "release_parity": ("export_surface", "target_surface", "target_fit"),
    "input_mass_parity": ("input_mass_parity",),
    "qrf_tail_concentration": ("qrf_tail_concentration",),
}
UK_INPUT_MASS_REFERENCE_IDENTITY = {
    "filename": "enhanced_frs_2024_25.h5",
    "revision": "a9e52499b6a6cca100a5ce4f36ca27b2e8a213df",
    "sha256": "e433e532b17bd8ce76030156285816e33d44e93edabd2204adbef71d19a68712",
    "vintage": "2024_25",
}
UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256 = (
    "fd41cb5f6cf6c4ef812320f21d1942173d49ce6f8725b21fbc9d9ca5423d298c"
)
UK_INPUT_MASS_ACTIVE_REFERENCE = "efrs-post-calibration"
UK_INPUT_MASS_REFERENCE_SCOPE_NOTE = (
    "Channel-blind post-calibration enhanced-FRS production incumbent, pinned "
    "to the 2024-25 line; structurally lacks the SPI clone channel, so "
    "SPI-channel-exclusive columns are comparable only through per-reference "
    "reviewed exclusions."
)
UK_INPUT_MASS_REVIEWED_EXCLUSIONS = {
    "charitable_investment_gifts": {
        "reason": (
            "SPI-channel-exclusive column on a channel-blind reference: the "
            "efrs-post-calibration incumbent structurally lacks the SPI clone "
            "channel, so its reference mass is survey-side scraps while the "
            "staged candidate's mass is the admin-captured SPI channel "
            "functioning as designed (microcosm#630 case 2). Compared "
            "normally against any future channel-aware reference."
        ),
        "approved_by": "juaristi22",
        "adjudication": "microcosm#630",
        "approved_on": "2026-08-20",
        "expires_on": "2027-02-20",
    },
    "owned_land": {
        "reason": "Sparse heavy-tailed WAS donor column (0.7 percent weighted nonzero share) whose weighted total is dominated by a handful of large farm/estate records: the spine-e stability receipt (data/ukds/acceptance/757-swap/owned_land_stability_receipt_spine_e.json) measures a 53.8 percent national and 96.7 percent West Midlands swing between adjacent seeds on the 25-stage candidate \u2014 the realization-variance class the archived incumbent data repo records at uk-data#448 (4.6x Wales swing across releases), reproduced from the E5 instrument's method. Register parity at this grain stays not meaningful; the one-month expiry keeps the end-of-workstream revisit registered on microcosm#145 live (winsorised donor or separate land imputation are the candidate remedies).",
        "approved_by": "juaristi22",
        "adjudication": "microcosm#714",
        "approved_on": "2026-08-26",
        "expires_on": "2026-09-26",
    },
}
GIT_COMMIT = "5fa48f07436a806ad75ff76fd22cfb8613bddbe0"
DATASET_SHA = "d" * 64
CALIBRATION_SHA = "a" * 64
DIAGNOSTICS_SHA = "c" * 64
SOURCE_COVERAGE_SHA = "9" * 64
TARGET_SURFACE_SHA = "e" * 64
REGISTRY_VERSION = "registryabc123"
TARGET_COUNT = 20
UK_JUNE_FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "uk_june_2023" / UK_RELEASE_ID
)


@pytest.fixture(autouse=True)
def _trusted_terminal_gate_signing_key(monkeypatch) -> None:
    monkeypatch.setenv(
        UK_TERMINAL_GATE_SIGNING_KEY_ENV,
        TEST_UK_TERMINAL_GATE_SIGNING_KEY,
    )


# ---------------------------------------------------------------------------
# Schema-4 gate-battery mirrors. Local copies in the schema-3 style; the
# lockstep test below holds them equal to the contract module's pins, and the
# build-shard sync tests hold the contract module equal to the producer.
# ---------------------------------------------------------------------------
UK_GATE_BATTERY_PRODUCER = "microcosm.build.gate_battery"
UK_GATE_BATTERY_SIGNING_KEY_ENV = "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY"
UK_GATE_BATTERY_POLICY_SHA256 = (
    "8deb610d25fced25e27be58989311efa217f49e6d644f37268cf0f7efd122193"
)
UK_GATE_BATTERY_GATES_MANIFEST_SHA256 = (
    "cd90537532e1800a76d4414a7d7454f5dedb4dfd574aca2dfca4ec1e1fe6cff3"
)
UK_GATE_BATTERY_SPEC_FINGERPRINT = (
    "8b28c7d23b9e8ebc9e7436abd7309e59f0ef14fb582506f874147a5c1c4f44ef"
)
UK_GATE_BATTERY_DEGENERATE_EVIDENCE_SHA256 = (
    "6f0243bcda09dad26945376230c44ec3cf55d4e417c3a25e29bae8c59bc1a69d"
)
UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256 = (
    "c9211cbb923e13f4850b834b5bdb1ff1de87fe9237c332b5de63f01ed417aa2d"
)
#: Spec entry id -> (neutral gate name, phase, legacy detail-schema name).
UK_GATE_BATTERY_ENTRIES = {
    "uk_release_input_coverage_manifest_current": (
        "release_input_coverage",
        "preflight",
        None,
    ),
    "uk_release_family_build_stages": ("source_coverage", "preflight", None),
    "uk_ledger_compile_parity_production_2023": (
        "ledger_compile_parity",
        "preflight",
        None,
    ),
    "uk_ledger_compile_parity_incumbent_2025": (
        "ledger_compile_parity",
        "preflight",
        None,
    ),
    "uk_stage_was_wealth_support": ("stage_health", "transferred", None),
    "uk_stage_uc_deduction_attributes": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_lcfs_consumption_support": ("stage_health", "transferred", None),
    "uk_stage_etb_vat_support": ("stage_health", "transferred", None),
    "uk_stage_etb_services_support": ("stage_health", "transferred", None),
    "uk_stage_frs_hmrc_spine_leaves_signal": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_spi_support_channel_mass": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_hmrc_spi_income_spine_identity": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_cgt_incidence_clone_mass": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_cgt_band_donors_support": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_hmrc_cgt_gains_spine_summary": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_salary_sacrifice_realization": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_student_loans_realization": (
        "stage_health",
        "transferred",
        None,
    ),
    "uk_stage_age_tail_targets": ("stage_health", "assembled", None),
    "uk_ledger_compile_parity_local_incumbent_2025": (
        "ledger_compile_parity",
        "preflight",
        None,
    ),
    "uk_target_surface_local_default_2025": (
        "target_surface",
        "preflight",
        None,
    ),
    "uk_release_input_coverage": (
        "release_input_coverage",
        "terminal",
        "uk_release_input_coverage",
    ),
    "uk_degenerate_release_surface": (
        "degenerate_release_surface",
        "terminal",
        "degenerate_release_surface",
    ),
    "uk_zero_weight_strata": ("zero_weight_strata", "terminal", "zero_weight_strata"),
    "uk_weight_ess": ("weight_ess", "terminal", "weight_ess"),
    "uk_weight_ratio": ("weight_ratio", "terminal", "weight_ratio"),
    "uk_weights_audit": ("weights_audit", "terminal", "weights_audit"),
    "uk_nonnegative_columns": (
        "nonnegative_columns",
        "terminal",
        "nonnegative_columns",
    ),
    "uk_uc_capital_coherence": (
        "column_implication",
        "terminal",
        "column_implication",
    ),
    "uk_support": ("support", "terminal", "support"),
    "uk_aggregate_admin": ("aggregate_admin", "terminal", "aggregate_vs_admin"),
    "uk_export_surface": ("export_surface", "terminal", "export_surface"),
    "uk_take_up_signal": ("take_up_signal", "terminal", "take_up_signal"),
    "uk_brma_enum_domain": ("enum_domain", "assembled", "enum_domain"),
    "uk_uc_deduction_combination_enum_domain": (
        "enum_domain",
        "terminal",
        "enum_domain",
    ),
    "uk_student_loan_plan_enum_domain": (
        "enum_domain",
        "terminal",
        "enum_domain",
    ),
    "uk_calibration_reference_coverage": (
        "calibration_reference_coverage",
        "terminal",
        None,
    ),
    "uk_target_surface": ("target_surface", "terminal", "target_surface"),
    "uk_target_fit": ("target_fit", "terminal", "target_fit"),
    "uk_input_mass_parity": ("input_mass_parity", "terminal", "input_mass_parity"),
    "uk_qrf_tail_concentration": (
        "tail_concentration",
        "terminal",
        "qrf_tail_concentration",
    ),
    "uk_local_geography_ladder_post_calibration": (
        "spine_agreement",
        "terminal",
        None,
    ),
    "uk_local_area_support": ("area_support", "terminal", None),
    "uk_local_target_fit": ("target_fit", "terminal", None),
    "uk_local_per_family_fit": ("per_family_fit", "terminal", None),
    "uk_local_weight_ratio": ("weight_ratio", "terminal", None),
    "uk_local_weight_ess": ("weight_ess", "terminal", None),
}


@pytest.fixture(autouse=True)
def _trusted_gate_battery_signing_key(monkeypatch) -> None:
    monkeypatch.setenv(
        UK_GATE_BATTERY_SIGNING_KEY_ENV,
        TEST_UK_TERMINAL_GATE_SIGNING_KEY,
    )


DEDUCTION_CRITICAL_TARGETS = (
    (
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount",
        1_000_000_000_000.0,
        1_020_000_000_000.0,
        "itemized_deduction_total",
    ),
    (
        "irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_amount",
        120_000_000_000.0,
        121_000_000_000.0,
        "salt_deduction_total",
    ),
    (
        "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount",
        80_000_000_000.0,
        69_000_000_000.0,
        "medical_expense_deduction_total",
    ),
    # microcosm#511: the Table 2.1 mortgage amount row is name-registered (its
    # production target_role is the generic soi_fiscal_distribution).
    (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_amount@2024",
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_amount",
        186_310_104_604.0,
        199_110_000_000.0,
        "soi_fiscal_distribution",
    ),
)


def _model_package(release_id: str) -> tuple[str, str]:
    if release_id.startswith("populace-uk-") or release_id == UK_NATIONAL_RELEASE_ID:
        return ("policyengine-uk", "2.89.0")
    return ("policyengine-us", "1.729.0")


def _build_manifest(release_id: str = RELEASE_ID) -> dict:
    model_package, model_version = _model_package(release_id)
    manifest = {
        "build_id": release_id,
        "builder": "microcosm",
        "build_sha": GIT_COMMIT[:7],
        "code": {
            "repository": "PolicyEngine/microcosm",
            "git_commit": GIT_COMMIT,
            "git_dirty": False,
        },
        "runtime": {
            "python": "3.14.0",
            "policyengine-core": "3.19.0",
            model_package: model_version,
        },
        "dataset": {"filename": "populace_us_2024.h5", "sha256": DATASET_SHA},
        "calibration": {
            "filename": "populace_us_2024_calibration.npz",
            "sha256": CALIBRATION_SHA,
            "target_surface": {"sha256": TARGET_SURFACE_SHA, "n_targets": TARGET_COUNT},
            "target_registry": {"version": REGISTRY_VERSION, "n_specs": TARGET_COUNT},
        },
        "gates": {"exported_nonzero": {"passed": True}},
    }
    if "-k" in release_id and release_id.startswith("populace-uk-"):
        manifest.update(country="uk", year=2023, n_records=UK_RECORD_COUNT)
    return manifest


def _release_manifest(
    release_id: str = RELEASE_ID,
    *,
    diagnostics_sha: str = DIAGNOSTICS_SHA,
    source_coverage_sha: str = SOURCE_COVERAGE_SHA,
    terminal_gate_sha: str | None = None,
) -> dict:
    model_package, model_version = _model_package(release_id)
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release_id,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.19.0",
            },
            "built_with_model_package": {
                "name": model_package,
                "version": model_version,
            },
        },
        "compatible_core_packages": [
            {"name": "policyengine-core", "specifier": "==3.19.0"}
        ],
        "compatible_model_packages": [
            {"name": model_package, "specifier": f"=={model_version}"}
        ],
        "artifacts": {
            "populace_us_2024": {
                "kind": "microdata",
                "path": "populace_us_2024.h5",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": DATASET_SHA,
            },
            "populace_us_2024_calibration": {
                "kind": "calibration",
                "path": "populace_us_2024_calibration.npz",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": CALIBRATION_SHA,
            },
            "calibration_diagnostics": {
                "kind": "diagnostics",
                "path": "calibration_diagnostics.json",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": diagnostics_sha,
            },
        },
    }
    if release_id.startswith("populace-us-"):
        manifest["artifacts"]["us_source_coverage"] = {
            "kind": "diagnostics",
            "path": US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
            "repo_id": "policyengine/populace-us",
            "revision": release_id,
            "sha256": source_coverage_sha,
        }
    if terminal_gate_sha is not None:
        manifest["artifacts"]["terminal_gates"] = {
            "kind": "diagnostics",
            "path": UK_TERMINAL_GATE_REPORT_FILE,
            "repo_id": "policyengine/populace-uk-private",
            "revision": release_id,
            "sha256": terminal_gate_sha,
        }
    return manifest


def _calibration_diagnostics() -> dict:
    return {
        "schema_version": 6,
        "weight_entity": "household",
        "options": {"epochs": 120},
        "target_surface": {
            "schema_version": 1,
            "weight_entity": "household",
            "n_targets": TARGET_COUNT,
            "n_records": 2,
            "constraint_matrix": {"rows": 1, "columns": 2, "nnz": 2},
            "sha256": TARGET_SURFACE_SHA,
            "names_sha256": "b" * 64,
            "values_sha256": "f" * 64,
        },
        "target_registry": {
            "country": "us",
            "version": REGISTRY_VERSION,
            "n_specs": TARGET_COUNT,
        },
        "loss_trajectory": [1.0, 0.5],
        "skipped": [],
        "targets": [
            _target_row(
                "population@2024",
                target_name="population",
                target=1.0,
                initial_estimate=0.8,
                final_estimate=1.0,
                relative_error=0.0,
                family="cbo",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all."
                "income_tax_liability_amount@2024",
                target_name=(
                    "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount"
                ),
                target=2_105_345_646_000.0,
                initial_estimate=2_000_000_000_000.0,
                final_estimate=2_067_762_165_736.424,
                relative_error=-0.0178514536722185,
                family="irs_soi",
                target_role="federal_income_tax_total",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all."
                "income_tax_liability_returns@2024",
                target_name=(
                    "irs_soi.ty2022.historic_table_2.us.all."
                    "income_tax_liability_returns"
                ),
                target=113_562_590.0,
                initial_estimate=105_421_734.40619682,
                final_estimate=105_437_267.69738781,
                relative_error=-0.07154928663226319,
                family="irs_soi",
            ),
            _target_row(
                "ssa_supplement.cy2024.oasdi_ssi_payments."
                "social_security_benefits.payment_amount@2024",
                target_name=(
                    "ssa_supplement.cy2024.oasdi_ssi_payments."
                    "social_security_benefits.payment_amount"
                ),
                target=1_471_195_000_000.0,
                initial_estimate=1_541_646_703_291.2527,
                final_estimate=1_541_540_768_722.367,
                relative_error=0.047815394099604024,
                family="ssa",
                target_role="social_security_total",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024",
                target_name="irs_soi.ty2022.historic_table_2.us.all.ctc_amount",
                target=82_863_353_000.0,
                initial_estimate=132_000_000_000.0,
                final_estimate=90_000_000_000.0,
                relative_error=(90_000_000_000.0 - 82_863_353_000.0) / 82_863_353_000.0,
                family="irs_soi",
                target_role="ctc_total",
            ),
            *additional_critical_credit_rows(),
            *deduction_critical_target_rows(),
            # The SOI Table 1.4 national dollar blanket (microcosm#462) needs
            # at least one Table 1.4 dollar row on the surface, within its
            # 25% blocking tolerance (the live Build M wages row).
            _target_row(
                "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024",
                target_name="irs_soi.ty2023.table_1_4.all.wages_salaries_amount",
                target=10_773_360_188_645.0,
                initial_estimate=10_500_000_000_000.0,
                final_estimate=10_774_383_029_502.0,
                relative_error=(10_774_383_029_502.0 - 10_773_360_188_645.0)
                / 10_773_360_188_645.0,
                family="irs_soi",
            ),
        ],
    }


def additional_critical_credit_rows() -> list[dict]:
    rows = [
        (
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims@2024",
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims",
            38_068_980.0,
            36_607_400.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.actc_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.actc_amount",
            33_858_000_000.0,
            33_501_200_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.actc_claims@2024",
            "irs_soi.ty2022.historic_table_2.us.all.actc_claims",
            17_691_400.0,
            17_434_500.0,
        ),
        (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount@2024",
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount",
            69_041_649_000.0,
            58_954_970_066.74941,
        ),
        (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns@2024",
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns",
            23_837_149.0,
            23_349_300.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount",
            53_910_190_000.0,
            56_821_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns@2024",
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns",
            7_841_370.0,
            8_385_450.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount",
            455_904_900_000.0,
            454_551_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns@2024",
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns",
            24_475_100.0,
            24_472_900.0,
        ),
        # microcosm#511: paired count row for the registered Table 2.1
        # mortgage amount target (O-1 landed +2.45%).
        (
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "home_mortgage_interest_returns@2024",
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "home_mortgage_interest_returns",
            11_644_348.0,
            11_929_445.0,
        ),
    ]
    return [
        _target_row(
            name,
            target_name=target_name,
            target=target,
            initial_estimate=target,
            final_estimate=final,
            relative_error=(final - target) / target,
            family="irs_soi",
        )
        for name, target_name, target, final in rows
    ]


def deduction_critical_target_rows() -> list[dict]:
    return [
        _target_row(
            name,
            target_name=target_name,
            target=target,
            initial_estimate=target * 1.5,
            final_estimate=final,
            relative_error=(final - target) / target,
            family="irs_soi",
            target_role=target_role,
        )
        for name, target_name, target, final, target_role in DEDUCTION_CRITICAL_TARGETS
    ]


def _target_row(
    name: str,
    *,
    target_name: str,
    target: float,
    initial_estimate: float,
    final_estimate: float,
    relative_error: float,
    family: str,
    target_role: str | None = None,
) -> dict:
    metadata = {"target_role": target_role} if target_role else {}
    return {
        "name": name,
        "target_name": target_name,
        "period": 2024,
        "entity": "household",
        "measure": {"kind": "column", "name": "household_count"},
        "filter": None,
        "source": "Fixture admin target",
        "metadata": metadata,
        "target": target,
        "compiled_target": target,
        "initial_estimate": initial_estimate,
        "final_estimate": final_estimate,
        "relative_error": relative_error,
        "within_tolerance": None,
        "registry": {"family": family},
    }


def _source_coverage_diagnostics() -> dict:
    return {
        "schema_version": 1,
        "classification": "release_gate",
        "source_contract": {
            "name": "us_source_coverage",
            "ledger_commit": "5fa48f07436a806ad75ff76fd22cfb8613bddbe0",
        },
        "gate": {
            "name": "us_source_coverage",
            "passed": True,
            "failures": [],
        },
        "coverage_summary": {
            "hard_target": {
                "families": 9,
                "package_aliases": 38,
                "covered_package_aliases": 38,
                "missing_package_aliases": 0,
                "reviewed_excluded_package_aliases": 0,
            },
            "validation_only": {"families": 6, "activated_families": 0},
            "source_gap": {"families": 6, "missing_source_packages": 11},
        },
        "hard_target_families": {"population_age_sex": {}},
        "validation_only_families": {"census_cps_spm": {}},
        "source_gap_families": {"usda_wic": {}},
        "active_target_aliases": ["census-pep-2024-national-age-sex"],
        "active_target_families": [],
        "missing_hard_targets": [],
        "reviewed_exclusions": {},
        "validation_only_activated": [],
        "fiscal_target_sources": {
            "cbo": {
                "label": "Congressional Budget Office revenue projections",
                "target_count": 1,
                "sources": ["Census PEP 2024"],
                "reference_urls": ["https://example.test/source"],
            },
            "irs_soi": {
                "label": "IRS Statistics of Income",
                "target_count": 18,
                "sources": ["IRS SOI Historic Table 2"],
                "reference_urls": ["https://example.test/soi"],
            },
            "ssa": {
                "label": "Social Security Administration",
                "target_count": 1,
                "sources": ["SSA Annual Statistical Supplement"],
                "reference_urls": ["https://example.test/ssa"],
            },
        },
    }


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    """A complete, contract-valid release directory."""
    directory = tmp_path / "releases" / RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(json.dumps(_build_manifest()))
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics())
    )
    (directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps(_source_coverage_diagnostics())
    )
    diagnostics_sha = _sha256(directory / "calibration_diagnostics.json")
    source_coverage_sha = _sha256(directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE)
    (directory / "release_manifest.json").write_text(
        json.dumps(
            _release_manifest(
                diagnostics_sha=diagnostics_sha,
                source_coverage_sha=source_coverage_sha,
            )
        )
    )
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _terminal_gate_signature(key: bytes, payload: object) -> str:
    return hmac.new(key, _canonical_json_bytes(payload), hashlib.sha256).hexdigest()


def _terminal_weight_summary() -> dict:
    return {
        "n_records": UK_RECORD_COUNT,
        "positive_weight_records": 335_080,
        "zero_weight_records": 200_000,
        "total_weight": 335_080.0,
        "effective_sample_size": 335_080.0,
        "ess_fraction": 335_080 / UK_RECORD_COUNT,
        "median_positive_weight": 1.0,
        "max_weight": 1.0,
        "max_to_median_positive_weight": 1.0,
        "top_1pct_weight_share": 0.01,
    }


def _terminal_gate_details(name: str) -> dict:
    if name == "uk_release_input_coverage":
        return {
            "required_columns": 1,
            "present_columns": 1,
            "missing": [],
            "degenerate_required": [],
            "reviewed_exclusions": {},
            "stale_exclusions": [],
            "dormant_exclusions": [],
        }
    if name == "degenerate_release_surface":
        return {
            "columns_checked": 1,
            "findings": {},
            "all_null_columns": [],
            "all_zero_columns": [],
            "constant_columns": [],
            "reviewed_exclusions": {},
            "stale_exclusions": [],
            "dormant_exclusions": [],
            "expired_exclusions": [],
            "premature_exclusions": [],
            "exclusions_evaluated_on": "2026-08-11",
        }
    if name == "zero_weight_strata":
        return {
            "household_rows": UK_RECORD_COUNT,
            "zero_weight_rows": 200_000,
            "declared_strata": [
                {
                    "name": "june_spi_synthetic_base",
                    "zero_weight_rows": 100_000,
                },
                {
                    "name": "june_spi_synthetic_capital_gains",
                    "zero_weight_rows": 100_000,
                },
            ],
            "unmatched_zero_weight_rows": 0,
            "unmatched_household_examples": [],
            "ambiguous_zero_weight_rows": 0,
            "ambiguous_household_examples": [],
        }
    if name == "weight_ess":
        return {**_terminal_weight_summary(), "minimum_ess_fraction": 0.01}
    if name == "weight_ratio":
        return {
            **_terminal_weight_summary(),
            "maximum_max_to_median_ratio": 1_151.2542195939373,
        }
    if name == "weights_audit":
        return {
            "fits_checked": 1,
            "resolved_weight_kinds": {"spi_qrf": "importance"},
            "unweighted_fits": [],
            "allowed_unweighted": {},
            "unused_allowed_unweighted": [],
        }
    if name == "nonnegative_columns":
        return {
            "columns_checked": 1,
            "negative_counts": {},
            "minima": {"employment_income": 1.0},
            "reviewed_exclusions": {},
            "unused_reviewed_exclusions": [],
            "atol": 0.0,
            "chunk_size": 1_000_000,
        }
    if name == "support":
        return {"columns_checked": 13}
    if name == "aggregate_vs_admin":
        return {"anchors_checked": 3}
    if name == "export_surface":
        return {
            "candidate_columns": 1,
            "reference_columns": 1,
            "missing_reference_columns": [],
            "unexpected_candidate_columns": [],
            "forbidden_candidate_columns": [],
        }
    if name == "target_surface":
        return {
            "candidate_targets": TARGET_COUNT,
            "reference_targets": TARGET_COUNT,
            "extra_candidate_targets": [],
            "missing_reference_targets": [],
        }
    if name == "target_fit":
        return {
            "targets_checked": TARGET_COUNT,
            "max_abs_relative_error": 0.25,
            "failing_targets": {},
            "reviewed_exclusions": {},
            "stale_exclusions": [],
            "dormant_exclusions": [],
            "expired_exclusions": [],
            "premature_exclusions": [],
            "exclusions_evaluated_on": "2026-08-30",
        }
    if name == "input_mass_parity":
        return {
            "candidate_name": "uk_release_candidate",
            "reference_name": UK_INPUT_MASS_REFERENCE_IDENTITY["filename"],
            "relative_tolerance": 4.521811483823806,
            "minimum_reference_total": 0.0,
            "columns_checked": 1,
            "columns_below_reference_floor": 0,
            "candidate_only_columns": [],
            "worst_drifts": {"employment_income": 0.0},
            "reviewed_exclusions": {},
            "unused_reviewed_exclusions": [],
            "stale_exclusions": [],
            "dormant_exclusions": [],
            "expired_exclusions": [],
            "premature_exclusions": [],
            "exclusions_evaluated_on": "2026-08-11",
            "reference": UK_INPUT_MASS_ACTIVE_REFERENCE,
            "reference_scope_note": UK_INPUT_MASS_REFERENCE_SCOPE_NOTE,
            "reference_identity": dict(UK_INPUT_MASS_REFERENCE_IDENTITY),
        }
    if name == "qrf_tail_concentration":
        return {
            "columns_checked": 1,
            "top_k": 100,
            "max_top_share": 0.9994670564654868,
            "min_nonzero_records": 104,
            "top_share": {"self_employment_income": 0.5},
            "carrier_counts": {"self_employment_income": 274},
            "thin_columns": {},
            "reviewed_exclusions": {},
            "stale_exclusions": [],
            "dormant_exclusions": [],
            "expired_exclusions": [],
            "premature_exclusions": [],
            "exclusions_evaluated_on": "2026-08-11",
            "surface": {
                "declared_qrf_outputs": 1,
                "checked_columns": ["self_employment_income"],
                "absent_columns": [],
                "non_numeric_columns": [],
                "density_filter": "none: every declared output is checked (#609)",
            },
        }
    if name == "take_up_signal":
        # Mirrors uk_take_up_signal_gate's per-column detail block.
        return {
            "benunit.would_claim_uc": {
                "weighted_share": 0.55,
                "target": 0.55,
                "absolute_deviation": 0.0,
                "unique_count": 2,
            }
        }
    if name == "enum_domain":
        # Mirrors enum_domain_gate's detail block.
        return {
            "columns_checked": 1,
            "invalid_counts": {},
            "invalid_examples": {},
            "allowed_values": {"brma": ["CENTRAL_LONDON"]},
        }
    if name == "column_implication":
        # Mirrors _evaluate_column_implication's composite detail block.
        return {
            "numeric_column": (
                "person.universal_credit_reported aggregated to benunit"
            ),
            "boolean_column": "benunit.would_claim_uc",
            "threshold": 0.0,
            "rows_checked": 1,
            "implicated_rows": 1,
            "violation_count": 0,
            "nonfinite_count": 0,
            "capital_column": "benunit.uc_reported_capital",
            "carrier_column": "benunit.frs_benunit_capital",
            "sentinel": -1.0,
            "capital_domain_violation_count": 0,
            "carrier_domain_violation_count": 0,
            "sentinel_mismatch_count": 0,
            "same_source_mismatch_count": 0,
            "nonfinite_capital_count": 0,
        }
    raise AssertionError(f"No terminal fixture details for {name!r}")


def _terminal_gate_payload(
    *,
    release_id: str,
    calibration_diagnostics_sha256: str,
    evidence_stages: tuple[str, ...] = ("hmrc_spi_income", "release_parity"),
    signing_key: bytes = TEST_UK_TERMINAL_GATE_SIGNING_KEY_BYTES,
) -> tuple[dict, dict[str, str]]:
    gate_names = list(UK_ALWAYS_APPLICABLE_GATE_NAMES)
    for stage, names in UK_EVIDENCE_GATE_NAMES.items():
        if stage in evidence_stages:
            gate_names.extend(names)
    gates = {
        name: {
            "passed": True,
            "failures": [],
            "details": _terminal_gate_details(name),
        }
        for name in gate_names
    }
    evidence = {
        "release_dataset": _canonical_sha256({"weights": _terminal_weight_summary()}),
        **{
            stage: (
                UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256
                if stage == "input_mass_parity"
                else _canonical_sha256({"fixture_evidence_stage": stage})
            )
            for stage in evidence_stages
        },
    }
    unsigned_attestation = {
        "schema_version": 5,
        "producer": UK_TERMINAL_GATE_PRODUCER,
        "release_id": release_id,
        "calibration_diagnostics_sha256": calibration_diagnostics_sha256,
        "policy_sha256": UK_TERMINAL_GATE_POLICY_SHA256,
        "evaluated_gates": gate_names,
        "evidence_sha256": evidence,
        "gate_results_sha256": _canonical_sha256(gates),
        "signature_algorithm": UK_TERMINAL_GATE_SIGNATURE_ALGORITHM,
        "signing_key_sha256": hashlib.sha256(signing_key).hexdigest(),
    }
    payload = {
        "schema_version": 3,
        "enforced": True,
        "passed": True,
        "gates": gates,
        "attestation": unsigned_attestation,
    }
    payload["attestation"]["signature"] = _terminal_gate_signature(signing_key, payload)
    return payload, evidence


def _refresh_terminal_gate_attestation(
    payload: dict,
    *,
    signing_key: bytes = TEST_UK_TERMINAL_GATE_SIGNING_KEY_BYTES,
) -> None:
    attestation = payload["attestation"]
    ratio = payload["gates"].get("weight_ratio")
    if isinstance(ratio, dict) and isinstance(ratio.get("details"), dict):
        details = ratio["details"]
        fields = tuple(_terminal_weight_summary())
        if all(field in details for field in fields):
            attestation["evidence_sha256"]["release_dataset"] = _canonical_sha256(
                {"weights": {field: details[field] for field in fields}}
            )
    attestation["gate_results_sha256"] = _canonical_sha256(payload["gates"])
    attestation["signature_algorithm"] = UK_TERMINAL_GATE_SIGNATURE_ALGORITHM
    attestation["signing_key_sha256"] = hashlib.sha256(signing_key).hexdigest()
    unsigned_attestation = {
        key: value for key, value in attestation.items() if key != "signature"
    }
    attestation["signature"] = _terminal_gate_signature(
        signing_key,
        {
            "schema_version": payload["schema_version"],
            "enforced": payload["enforced"],
            "passed": payload["passed"],
            "gates": payload["gates"],
            "attestation": unsigned_attestation,
        },
    )


def _write_json_and_refresh_manifest_hash(
    release_dir: Path,
    *,
    filename: str,
    artifact_key: str,
    payload: dict,
) -> None:
    (release_dir / filename).write_text(json.dumps(payload))
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][artifact_key]["sha256"] = _sha256(release_dir / filename)
    manifest_path.write_text(json.dumps(manifest))


def _write_terminal_and_refresh_manifest_hashes(
    release_dir: Path,
    payload: dict,
) -> None:
    terminal_path = release_dir / UK_TERMINAL_GATE_REPORT_FILE
    terminal_path.write_text(json.dumps(payload))
    _refresh_terminal_manifest_hashes(release_dir)


def _refresh_terminal_manifest_hashes(release_dir: Path) -> None:
    terminal_path = release_dir / UK_TERMINAL_GATE_REPORT_FILE
    digest = _sha256(terminal_path)

    build_path = release_dir / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["gates"]["uk_terminal"]["sha256"] = digest
    build_path.write_text(json.dumps(build))

    release_path = release_dir / "release_manifest.json"
    release = json.loads(release_path.read_text())
    release["artifacts"]["terminal_gates"]["sha256"] = digest
    release_path.write_text(json.dumps(release))


def _resign_gate_battery(
    payload: dict,
    *,
    signing_key: bytes = TEST_UK_TERMINAL_GATE_SIGNING_KEY_BYTES,
) -> None:
    """Rebuild the schema-4 attestation from the body and sign, as the
    producer does: the signature covers the canonical payload with the
    signature slot nulled."""

    attestation = {
        "schema_version": 6,
        "producer": UK_GATE_BATTERY_PRODUCER,
        "country": payload["country"],
        "release_id": payload["release_id"],
        "release_candidate": payload["release_candidate"],
        "spec_fingerprint": payload["spec_fingerprint"],
        "gates_manifest_sha256": payload["gates_manifest_sha256"],
        "policy_sha256": payload["policy_sha256"],
        "phases": payload["phases"],
        "phases_evaluated": payload["phases_evaluated"],
        "blocked_at_phase": payload["blocked_at_phase"],
        "release_evidence": payload["release_evidence"],
        "evidence_sha256": payload["evidence_sha256"],
        "gate_outcomes_sha256": _canonical_sha256(payload["gates"]),
        "signature_algorithm": UK_TERMINAL_GATE_SIGNATURE_ALGORITHM,
        "signing_key_sha256": hashlib.sha256(signing_key).hexdigest(),
        "signature": None,
    }
    payload["attestation"] = attestation
    attestation["signature"] = hmac.new(
        signing_key, _canonical_json_bytes(payload), hashlib.sha256
    ).hexdigest()


def _gate_battery_payload(
    *,
    release_id: str,
    calibration_diagnostics_sha256: str,
    signing_key: bytes = TEST_UK_TERMINAL_GATE_SIGNING_KEY_BYTES,
) -> tuple[dict, dict[str, str]]:
    """A fully-armed, all-passing, signed schema-4 battery report."""

    stage_names = [
        "frs_spine",
        "frs_employment",
        "frs_council_tax",
        "frs_disability",
        "frs_education",
        "frs_legacy_proxies",
        "frs_education_grant_split",
        "frs_take_up",
        "frs_person_draws",
        "frs_household_draws",
        "frs_brma",
        "was_wealth",
        "regional_property_uprating",
        "lcfs_consumption",
        "etb_vat",
        "etb_services",
        "frs_hmrc_spine_leaves",
        "spi_support_channel",
        "hmrc_spi_income_spine",
        "uc_capital_coherence",
        "uc_deduction_attributes",
        "cgt_incidence_clone",
        "cgt_band_donors",
        "hmrc_cgt_gains_spine",
        "salary_sacrifice",
        "student_loans",
        "age_tail",
    ]
    stage_health_stages = {
        "uk_stage_was_wealth_support": "was_wealth",
        "uk_stage_uc_deduction_attributes": "uc_deduction_attributes",
        "uk_stage_lcfs_consumption_support": "lcfs_consumption",
        "uk_stage_etb_vat_support": "etb_vat",
        "uk_stage_etb_services_support": "etb_services",
        "uk_stage_frs_hmrc_spine_leaves_signal": "frs_hmrc_spine_leaves",
        "uk_stage_spi_support_channel_mass": "spi_support_channel",
        "uk_stage_hmrc_spi_income_spine_identity": "hmrc_spi_income_spine",
        "uk_stage_cgt_incidence_clone_mass": "cgt_incidence_clone",
        "uk_stage_cgt_band_donors_support": "cgt_band_donors",
        "uk_stage_hmrc_cgt_gains_spine_summary": "hmrc_cgt_gains_spine",
        "uk_stage_salary_sacrifice_realization": "salary_sacrifice",
        "uk_stage_student_loans_realization": "student_loans",
        "uk_stage_age_tail_targets": "age_tail",
    }
    gates: dict[str, dict] = {}
    for entry_id, (gate, phase, detail_name) in UK_GATE_BATTERY_ENTRIES.items():
        if entry_id == "uk_release_input_coverage_manifest_current":
            details: dict = {"check": "manifest_current"}
        elif entry_id == "uk_release_family_build_stages":
            details = {"stage_names": list(stage_names)}
        elif entry_id == "uk_ledger_compile_parity_production_2023":
            details = {
                "fixture": "production_2023",
                "expected_count": 149,
                "actual_count": 149,
                "signed_difference_count": 0,
            }
        elif entry_id == "uk_ledger_compile_parity_incumbent_2025":
            details = {
                "fixture": "incumbent_2025",
                "expected_count": 636,
                "actual_count": 636,
                "signed_difference_count": 0,
            }
        elif entry_id == "uk_ledger_compile_parity_local_incumbent_2025":
            details = {
                "fixture": "incumbent_local_2025",
                "expected_count": 23_545,
                "actual_count": 17_077,
                "signed_difference_count": 23_837,
            }
        elif entry_id == "uk_target_surface_local_default_2025":
            details = {
                "candidate_name": "UK local compiled Ledger surface",
                "reference_name": "UK local default metric surface",
                "candidate_targets": 17_077,
                "reference_targets": 19_642,
                "extra_candidate_targets": [],
                "missing_reference_targets": [],
                "reviewed_exclusions": {},
                "unused_reviewed_exclusions": [],
            }
        elif entry_id == "uk_calibration_reference_coverage":
            details = {"activated": 388, "resolved": 388, "matrix": 388}
        elif entry_id.startswith("uk_local_"):
            # Local candidate gates are explicitly excluded from national
            # certification; this full-report fixture needs only their
            # authenticated envelope, not candidate-only evidence details.
            details = {}
        elif gate == "stage_health":
            details = {
                "stage": stage_health_stages[entry_id],
                "check": "fixture",
            }
        else:
            details = _terminal_gate_details(detail_name)
        gates[entry_id] = {
            "gate": gate,
            "phase": phase,
            # PR #870 review: every entry is release-blocking, the local fit and
            # weight gates included.
            "criticality": "release_blocking",
            "status": "passed",
            "failures": [],
            "details": details,
            "reason": None,
        }
    evidence = {
        "uk_release_family_build_stages": _canonical_sha256(
            {"stage_names": list(stage_names)}
        ),
        "uk_ledger_compile_parity_production_2023": _canonical_sha256(
            {
                "fixture_resource": "parity_fixture_production_2023.json",
                "registry_count": 149,
                "registry_version": "fixture",
                "signed_differences": [],
            }
        ),
        "uk_ledger_compile_parity_incumbent_2025": _canonical_sha256(
            {
                "fixture_resource": "registry_parity_fixture_2025.json",
                "registry_count": 636,
                "registry_version": "fixture",
                "signed_differences": [],
            }
        ),
        "uk_ledger_compile_parity_local_incumbent_2025": _canonical_sha256(
            {
                "fixture_resource": "local_registry_parity_fixture_2025.json",
                "registry_artifact": "uk_ledger_compiled_local_registries",
                "registry_count": 17_077,
                "registry_version": "fixture",
                "signed_differences": [],
                "target_period": 2025,
            }
        ),
        "uk_target_surface_local_default_2025": _canonical_sha256(
            {
                "candidate_targets": 17_077,
                "crosswalk_resource": "local_area_crosswalk.json",
                "expected": "local_default_surface",
                "membership_resource": "local_target_reference_membership.json",
                "reference_targets": 19_642,
                "registry_artifact": "uk_ledger_compiled_local_registries",
                "registry_count": 17_077,
                "reviewed_exclusions": 1_554,
                "target_period": 2025,
            }
        ),
        "uk_degenerate_release_surface": UK_GATE_BATTERY_DEGENERATE_EVIDENCE_SHA256,
        "uk_input_mass_parity": UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256,
    }
    for entry_id, stage in stage_health_stages.items():
        evidence[entry_id] = _canonical_sha256({stage: {"stage": stage}})
    payload = {
        "schema_version": 4,
        "country": "uk",
        "release_id": release_id,
        "release_candidate": True,
        "spec_fingerprint": UK_GATE_BATTERY_SPEC_FINGERPRINT,
        "gates_manifest_sha256": UK_GATE_BATTERY_GATES_MANIFEST_SHA256,
        "phases": ["preflight", "assembled", "transferred", "terminal"],
        "phases_evaluated": ["preflight", "assembled", "transferred", "terminal"],
        "blocked_at_phase": None,
        "shippable": True,
        "gates": gates,
        "policy_sha256": UK_GATE_BATTERY_POLICY_SHA256,
        "release_evidence": {
            "calibration_diagnostics_sha256": calibration_diagnostics_sha256
        },
        "evidence_sha256": evidence,
    }
    _resign_gate_battery(payload, signing_key=signing_key)
    return payload, evidence


def _upgrade_release_to_gate_battery(directory: Path) -> dict:
    """Swap a fixture release's schema-3 report for a valid schema-4 one."""

    payload, evidence = _gate_battery_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
    )
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = evidence
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)
    return payload


def _write_battery_release(tmp_path: Path) -> tuple[Path, dict]:
    directory = _write_uk_release_dir(tmp_path, UK_EXACT_K_RELEASE_ID, tier="frs")
    payload = _upgrade_release_to_gate_battery(directory)
    return directory, payload


def _rewrite_battery_report(
    directory: Path, payload: dict, *, resign: bool = True
) -> None:
    if resign:
        _resign_gate_battery(payload)
    _write_terminal_and_refresh_manifest_hashes(directory, payload)


def _battery_failures(directory: Path) -> str:
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)
    return "\n".join(excinfo.value.failures)


def _write_uk_release_dir(
    tmp_path: Path,
    release_id: str,
    *,
    tier: str | None = None,
) -> Path:
    directory = tmp_path / "releases" / release_id
    directory.mkdir(parents=True)
    build_manifest = _build_manifest(release_id)
    diagnostics = _calibration_diagnostics()
    if "-k" in release_id:
        diagnostics["target_registry"]["country"] = "uk"
        diagnostics["target_surface"]["n_records"] = UK_RECORD_COUNT
        diagnostics["n_records"] = UK_RECORD_COUNT
        diagnostics["effective_sample_size"] = 335_080.0
        diagnostics["top_1pct_weight_share"] = 0.01
        diagnostics["uk_diagnostics"] = {
            "schema_version": 1,
            "weights": {
                "n_records": UK_RECORD_COUNT,
                "positive_weight_records": 335_080,
                "zero_weight_records": 200_000,
                "total_weight": 335_080.0,
                "effective_sample_size": 335_080.0,
                "ess_fraction": 335_080 / UK_RECORD_COUNT,
                "median_positive_weight": 1.0,
                "max_weight": 1.0,
                "max_to_median_positive_weight": 1.0,
                "top_1pct_weight_share": 0.01,
            },
            "zero_weight_rows_by_stratum": [
                {
                    "stratum": {"household_is_spi_synthetic": False},
                    "rows": 335_080,
                    "positive_weight_rows": 335_080,
                    "zero_weight_rows": 0,
                    "weight_sum": 335_080.0,
                },
                {
                    "stratum": {"household_is_spi_synthetic": True},
                    "rows": 200_000,
                    "positive_weight_rows": 0,
                    "zero_weight_rows": 200_000,
                    "weight_sum": 0.0,
                },
            ],
            "target_pass_rates_by_geography_level": [
                {
                    "geography_level": level,
                    "n_targets": TARGET_COUNT if level == "national" else 0,
                    "n_scored": TARGET_COUNT if level == "national" else 0,
                    "n_skipped": 0,
                    "n_within_10pct": TARGET_COUNT if level == "national" else 0,
                    "pass_rate": 1.0 if level == "national" else None,
                }
                for level in (
                    "national",
                    "region",
                    "country",
                    "local_authority",
                    "constituency",
                )
            ],
        }
    diagnostics_path = directory / "calibration_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))
    terminal_gate_sha: str | None = None
    if release_id.startswith("populace-uk-") and release_id.endswith(
        f"-k{UK_RECORD_COUNT}"
    ):
        terminal_payload, terminal_evidence = _terminal_gate_payload(
            release_id=release_id,
            calibration_diagnostics_sha256=_sha256(diagnostics_path),
        )
        terminal_path = directory / UK_TERMINAL_GATE_REPORT_FILE
        terminal_path.write_text(json.dumps(terminal_payload))
        terminal_gate_sha = _sha256(terminal_path)
        build_manifest["terminal_gate_evidence"] = terminal_evidence
        build_manifest["gates"]["uk_terminal"] = {
            "passed": True,
            "path": UK_TERMINAL_GATE_REPORT_FILE,
            "sha256": terminal_gate_sha,
        }
    (directory / "build_manifest.json").write_text(json.dumps(build_manifest))
    manifest = _release_manifest(
        release_id,
        diagnostics_sha=_sha256(directory / "calibration_diagnostics.json"),
        terminal_gate_sha=terminal_gate_sha,
    )
    if "-k" in release_id:
        manifest.update(
            country="uk",
            year=2023,
            record_count=UK_RECORD_COUNT,
            n_records=UK_RECORD_COUNT,
        )
    if tier is not None:
        manifest["tier"] = tier
    (directory / "release_manifest.json").write_text(json.dumps(manifest))
    return directory


def _copy_real_uk_june_release(tmp_path: Path) -> Path:
    """Copy the semantic-real June JSONs sourced at df82567.

    The committed fixtures retain all 149 targets from
    ``policyengine-uk-data@df82567f598990b476cf0c26fe8f9bc7a06ddde1``.
    Only JSON whitespace is trimmed; the release manifest diagnostics digest
    is refreshed for those minified bytes. Original source hashes were
    build ``630b05bc...``, diagnostics ``80b98127...``, and release
    ``687c5c19...``.
    """

    directory = tmp_path / "releases" / UK_RELEASE_ID
    shutil.copytree(UK_JUNE_FIXTURE_DIR, directory)
    return directory


def _write_uk_national_release_dir(tmp_path: Path) -> Path:
    """Build the constant-id national fixture from the green exact-k shape."""

    directory = _write_uk_release_dir(tmp_path, UK_EXACT_K_RELEASE_ID, tier="frs")
    national = directory.with_name(UK_NATIONAL_RELEASE_ID)
    directory.replace(national)

    build_path = national / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["build_id"] = UK_NATIONAL_RELEASE_ID
    build_path.write_text(json.dumps(build))

    release_path = national / "release_manifest.json"
    release = json.loads(release_path.read_text())
    release["build"]["build_id"] = UK_NATIONAL_RELEASE_ID
    for artifact in release["artifacts"].values():
        artifact["revision"] = UK_NATIONAL_CUT_TAG

    # The certification signs the evidence copies' actual bytes: the seam
    # report already sits in the directory (terminal_gates.json from the
    # exact-k shape); the other three are written here so every signed digest
    # binds to a real local file.
    (national / "spine_gates.json").write_text(json.dumps({"fixture": "spine"}))
    (national / "release_cut_gates.json").write_text(
        json.dumps({"fixture": "release_cut"})
    )
    (national / "score_vs_enhanced_frs.json").write_text(
        json.dumps({"fixture": "score"})
    )
    certification = _green_uk_certification(
        TEST_UK_TERMINAL_GATE_SIGNING_KEY_BYTES,
        release_id=UK_NATIONAL_RELEASE_ID,
        diagnostics_sha256=_sha256(national / "calibration_diagnostics.json"),
        part_shas={
            "spine": _sha256(national / "spine_gates.json"),
            "calibration_seam": _sha256(national / "terminal_gates.json"),
            "release_cut": _sha256(national / "release_cut_gates.json"),
        },
        score_receipt_sha256=_sha256(national / "score_vs_enhanced_frs.json"),
    )
    certification_path = national / "release_certification.json"
    certification_path.write_text(json.dumps(certification))
    release["artifacts"]["release_certification"] = {
        "kind": "diagnostics",
        "path": "release_certification.json",
        "repo_id": "policyengine/populace-uk-private",
        "revision": UK_NATIONAL_CUT_TAG,
        "sha256": _sha256(certification_path),
    }
    release_path.write_text(json.dumps(release))
    return national


def _rewrite_exact_k_fixture_to_two_records(directory: Path) -> None:
    """Reproduce Sol's internally consistent k535080/n_records=2 probe."""

    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["n_records"] = 2
    build_path.write_text(json.dumps(build))

    diagnostics = json.loads((directory / "calibration_diagnostics.json").read_text())
    diagnostics["n_records"] = 2
    diagnostics["target_surface"]["n_records"] = 2
    diagnostics["effective_sample_size"] = 1.0
    diagnostics["top_1pct_weight_share"] = 1.0
    weights = diagnostics["uk_diagnostics"]["weights"]
    weights.update(
        n_records=2,
        positive_weight_records=1,
        zero_weight_records=1,
        total_weight=1.0,
        effective_sample_size=1.0,
        ess_fraction=0.5,
        top_1pct_weight_share=1.0,
    )
    positive, zero = diagnostics["uk_diagnostics"]["zero_weight_rows_by_stratum"]
    positive.update(rows=1, positive_weight_rows=1, zero_weight_rows=0, weight_sum=1.0)
    zero.update(rows=1, positive_weight_rows=0, zero_weight_rows=1, weight_sum=0.0)
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    release_path = directory / "release_manifest.json"
    release = json.loads(release_path.read_text())
    release["record_count"] = 2
    release["n_records"] = 2
    release_path.write_text(json.dumps(release))


def _split_microdata_artifact_entry(release_id: str, key: str) -> dict:
    return {
        "kind": (
            "state_microdata"
            if key.startswith("states/")
            else "congressional_district_microdata"
        ),
        "path": f"{key}.h5",
        "repo_id": "policyengine/populace-us",
        "revision": release_id,
        "sha256": "7" * 64,
    }


def test_a_complete_release_passes(release_dir: Path) -> None:
    validate_release_dir(release_dir)


def test_us_release_rejects_bad_critical_target_fit(release_dir: Path) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all."
        "income_tax_liability_amount@2024"
    )
    target["final_estimate"] = 735_173_331_468.564
    target["relative_error"] = -0.6508063496056629
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "federal income tax liability amount" in failures
    assert "relative_error=-0.650806" in failures


def test_us_release_recomputes_critical_target_fit(release_dir: Path) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all."
        "income_tax_liability_amount@2024"
    )
    target["final_estimate"] = 735_173_331_468.564
    target["relative_error"] = 0.0
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "stale relative_error" in failures
    assert "relative_error=-0.650806" in failures


def test_us_release_rejects_bad_ctc_fit(release_dir: Path) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    )
    ctc_target = 82_863_353_000.0
    ctc_final = 132_511_000_000.0
    target["final_estimate"] = ctc_final
    target["relative_error"] = (ctc_final - ctc_target) / ctc_target
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "Child Tax Credit amount" in failures
    assert "relative_error=0.599151" in failures


@pytest.mark.parametrize(
    "deduction",
    DEDUCTION_CRITICAL_TARGETS,
    ids=lambda row: row[4],
)
def test_us_release_rejects_bad_deduction_fit(
    release_dir: Path, deduction: tuple
) -> None:
    diagnostics = _calibration_diagnostics()
    deduction_name, _, deduction_target, _, target_role = deduction
    target = next(
        row for row in diagnostics["targets"] if row["name"] == deduction_name
    )
    bad_final = deduction_target * 1.5
    target["final_estimate"] = bad_final
    target["relative_error"] = (bad_final - deduction_target) / deduction_target
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert deduction_name in failures
    expected_cap = {
        "salt_deduction_total": 0.1,
        # 2026-07-22 adjudication: relaxed to the 0.25 broad-fit bound while
        # the #462 loss-contract alignment lands (see the register comment).
        "medical_expense_deduction_total": 0.25,
        # microcosm#511: interim 0.20 while the donor-side E19200 concept
        # carve (microcosm#515) lands (see the register comment).
        "soi_fiscal_distribution": 0.2,
    }.get(target_role, 0.15)
    assert f"exceeding {expected_cap}" in failures


def test_us_release_ignores_congressional_district_layout_critical_fit(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    salt_target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all."
        "limited_state_local_taxes_amount@2024"
    )
    cd_layout_target = dict(salt_target)
    cd_layout_target["name"] = (
        "irs_soi.ty2023.congressional_district_2022.all_returns.us."
        "limited_state_local_taxes_amount@2024"
    )
    cd_layout_target["target_name"] = cd_layout_target["name"].removesuffix("@2024")
    cd_layout_target["target"] = 250_437_565_000.0
    cd_layout_target["compiled_target"] = 250_437_565_000.0
    cd_layout_target["final_estimate"] = 130_722_333_208.88704
    cd_layout_target["relative_error"] = (
        cd_layout_target["final_estimate"] - cd_layout_target["target"]
    ) / cd_layout_target["target"]
    cd_layout_target["metadata"] = {
        **salt_target["metadata"],
        "ledger_source_record_id": cd_layout_target["target_name"],
        "ledger_layout_groupby_dimension": "irs_soi.congressional_district",
        "ledger_layout_groupby_value_id": "us",
        "target_role": "salt_deduction_total",
    }
    diagnostics["targets"].append(cd_layout_target)
    diagnostics["target_surface"]["n_targets"] += 1
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    source_coverage = json.loads(
        (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).read_text()
    )
    source_coverage["fiscal_target_sources"]["irs_soi"]["target_count"] += 1
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=source_coverage,
    )
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    build_manifest["calibration"]["target_surface"]["n_targets"] += 1
    (release_dir / "build_manifest.json").write_text(json.dumps(build_manifest))

    validate_release_dir(release_dir)


@pytest.mark.parametrize(
    "deduction",
    DEDUCTION_CRITICAL_TARGETS,
    ids=lambda row: row[4],
)
def test_us_release_rejects_deduction_improvement_past_absolute_gate(
    release_dir: Path, deduction: tuple
) -> None:
    diagnostics = _calibration_diagnostics()
    deduction_name, _, deduction_target, _, target_role = deduction
    target = next(
        row for row in diagnostics["targets"] if row["name"] == deduction_name
    )
    # Past each row's own absolute cap (medical sits at the adjudicated 0.25
    # bound, 2026-07-22; mortgage at the interim 0.20, microcosm#511): even
    # improving on the incumbent never passes it.
    overshoot = (
        1.30
        if target_role in {"medical_expense_deduction_total", "soi_fiscal_distribution"}
        else 1.20
    )
    current_final = deduction_target * overshoot
    target["final_estimate"] = current_final
    target["relative_error"] = (current_final - deduction_target) / deduction_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": deduction_target,
                    "final_estimate": deduction_target * 3.0,
                    "relative_error": 2.0,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert deduction_name in failures
    expected_rel = (
        "relative_error=0.3"
        if target_role in {"medical_expense_deduction_total", "soi_fiscal_distribution"}
        else "relative_error=0.2"
    )
    assert expected_rel in failures


def test_us_release_allows_bad_ctc_fit_when_it_improves_incumbent(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    )
    ctc_target = 82_863_353_000.0
    ctc_final = 99_282_300_000.0
    target["final_estimate"] = ctc_final
    target["relative_error"] = (ctc_final - ctc_target) / ctc_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": ctc_target,
                    "final_estimate": 134_904_000_000.0,
                    "relative_error": (134_904_000_000.0 - ctc_target) / ctc_target,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    validate_release_dir(release_dir)


def test_us_release_rejects_incumbent_improvement_past_hard_stop(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    )
    ctc_target = 82_863_353_000.0
    ctc_final = ctc_target * 1.26
    target["final_estimate"] = ctc_final
    target["relative_error"] = (ctc_final - ctc_target) / ctc_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": ctc_target,
                    "final_estimate": 134_904_000_000.0,
                    "relative_error": (134_904_000_000.0 - ctc_target) / ctc_target,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "Child Tax Credit amount" in failures
    assert "relative_error=0.26" in failures
    assert "improvement_hard_stop=0.25" in failures


def test_us_release_rejects_mortgage_amount_improvement_inside_hard_stop(
    release_dir: Path,
) -> None:
    # microcosm#511: the interim 0.20 mortgage cap is unconditional. A miss in
    # the 0.20-0.25 band with an improving incumbent is exactly where the
    # incumbent-improvement escape would fire if the register entry ever
    # regressed to allow_incumbent_improvement=True, so pin that band.
    mortgage_name = (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_amount@2024"
    )
    diagnostics = _calibration_diagnostics()
    target = next(row for row in diagnostics["targets"] if row["name"] == mortgage_name)
    mortgage_target = 186_310_104_604.0
    mortgage_final = mortgage_target * 1.225
    target["final_estimate"] = mortgage_final
    target["relative_error"] = (mortgage_final - mortgage_target) / mortgage_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": mortgage_target,
                    # The certified O-1 shipped state: worse than the new
                    # +22.5%, so this is a genuine improvement.
                    "final_estimate": 241_268_995_041.0,
                    "relative_error": (241_268_995_041.0 - mortgage_target)
                    / mortgage_target,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert mortgage_name in failures
    assert "home mortgage interest deduction amount" in failures
    assert "relative_error=0.225" in failures
    assert "exceeding 0.2" in failures


def test_us_release_rejects_bad_mortgage_returns_fit(release_dir: Path) -> None:
    # microcosm#511: the paired returns row carries the standard 0.15 cap.
    returns_name = (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_returns@2024"
    )
    diagnostics = _calibration_diagnostics()
    target = next(row for row in diagnostics["targets"] if row["name"] == returns_name)
    returns_target = 11_644_348.0
    returns_final = returns_target * 1.2
    target["final_estimate"] = returns_final
    target["relative_error"] = (returns_final - returns_target) / returns_target
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert returns_name in failures
    assert "home mortgage interest deduction returns" in failures
    assert "exceeding 0.15" in failures


def test_us_release_requires_mortgage_returns_row(release_dir: Path) -> None:
    # microcosm#511: the returns requirement is required-present on its own,
    # not just as a rider on the amount row.
    returns_name = (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_returns@2024"
    )
    diagnostics = _calibration_diagnostics()
    diagnostics["targets"] = [
        row for row in diagnostics["targets"] if row["name"] != returns_name
    ]
    diagnostics["target_surface"]["n_targets"] = len(diagnostics["targets"])
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    build_manifest = _build_manifest()
    build_manifest["calibration"]["target_surface"]["n_targets"] = len(
        diagnostics["targets"]
    )
    build_manifest["calibration"]["target_registry"]["n_specs"] = len(
        diagnostics["targets"]
    )
    (release_dir / "build_manifest.json").write_text(json.dumps(build_manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "home_mortgage_interest_returns" in failures
    assert "home mortgage interest deduction returns" in failures


def test_us_release_rejects_incumbent_improvement_with_mismatched_target(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    )
    ctc_target = 82_863_353_000.0
    ctc_final = 99_282_300_000.0
    target["final_estimate"] = ctc_final
    target["relative_error"] = (ctc_final - ctc_target) / ctc_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": ctc_target + 1_000_000_000.0,
                    "final_estimate": 134_904_000_000.0,
                    "relative_error": 0.0,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "does not match current target" in failures
    assert "Child Tax Credit amount" in failures


def test_us_release_requires_critical_targets(release_dir: Path) -> None:
    diagnostics = _calibration_diagnostics()
    diagnostics["targets"] = [
        row
        for row in diagnostics["targets"]
        if row["name"] != "ssa_supplement.cy2024.oasdi_ssi_payments."
        "social_security_benefits.payment_amount@2024"
    ]
    diagnostics["target_surface"]["n_targets"] = len(diagnostics["targets"])
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    source_coverage = _source_coverage_diagnostics()
    source_coverage["fiscal_target_sources"].pop("ssa")
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=source_coverage,
    )
    build_manifest = _build_manifest()
    build_manifest["calibration"]["target_surface"]["n_targets"] = len(
        diagnostics["targets"]
    )
    (release_dir / "build_manifest.json").write_text(json.dumps(build_manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "social_security_benefits" in failures


@pytest.mark.parametrize(
    "deduction",
    DEDUCTION_CRITICAL_TARGETS,
    ids=lambda row: row[4],
)
def test_us_release_requires_direct_deduction_targets(
    release_dir: Path, deduction: tuple
) -> None:
    diagnostics = _calibration_diagnostics()
    deduction_name, _, _, _, target_role = deduction
    diagnostics["targets"] = [
        row for row in diagnostics["targets"] if row["name"] != deduction_name
    ]
    diagnostics["target_surface"]["n_targets"] = len(diagnostics["targets"])
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    build_manifest = _build_manifest()
    build_manifest["calibration"]["target_surface"]["n_targets"] = len(
        diagnostics["targets"]
    )
    build_manifest["calibration"]["target_registry"]["n_specs"] = len(
        diagnostics["targets"]
    )
    (release_dir / "build_manifest.json").write_text(json.dumps(build_manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    expected_requirement = {
        # microcosm#511: the mortgage row's production role is the generic
        # soi_fiscal_distribution; its requirement id is name-derived.
        "soi_fiscal_distribution": "home_mortgage_interest_amount",
    }.get(target_role, target_role.replace("_total", "_amount"))
    assert expected_requirement in failures


@pytest.mark.parametrize("filename", required_release_files(RELEASE_ID))
def test_each_required_file_is_named_when_missing(
    release_dir: Path, filename: str
) -> None:
    (release_dir / filename).unlink()
    with pytest.raises(ReleaseContractError, match=filename):
        validate_release_dir(release_dir)


def test_real_june_release_validates_with_legacy_schema_and_selector_shapes(
    tmp_path: Path,
) -> None:
    directory = _copy_real_uk_june_release(tmp_path)
    diagnostics = json.loads((directory / "calibration_diagnostics.json").read_text())

    validate_release_dir(directory)
    assert diagnostics["schema_version"] == 2
    assert len(diagnostics["targets"]) == 149
    assert all("aggregation" in row for row in diagnostics["targets"])
    assert all("measure" not in row for row in diagnostics["targets"])
    assert all("filter" not in row for row in diagnostics["targets"])
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE not in required_release_files(
        UK_RELEASE_ID
    )


def test_uk_national_release_dir_validates(tmp_path: Path) -> None:
    validate_release_dir(_write_uk_national_release_dir(tmp_path))


def test_uk_national_release_requires_uk_diagnostics(tmp_path: Path) -> None:
    directory = _write_uk_national_release_dir(tmp_path)
    diagnostics_path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics.pop("uk_diagnostics")
    diagnostics_path.write_text(json.dumps(diagnostics))

    with pytest.raises(ReleaseContractError, match="require a 'uk_diagnostics'"):
        validate_release_dir(directory)


def test_uk_national_release_requires_policyengine_uk_runtime(
    tmp_path: Path,
) -> None:
    directory = _write_uk_national_release_dir(tmp_path)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["runtime"].pop("policyengine-uk")
    build_path.write_text(json.dumps(build))

    with pytest.raises(ReleaseContractError, match="runtime.policyengine-uk"):
        validate_release_dir(directory)


def test_uk_national_release_requires_policyengine_uk_model_pin(
    tmp_path: Path,
) -> None:
    directory = _write_uk_national_release_dir(tmp_path)
    release_path = directory / "release_manifest.json"
    release = json.loads(release_path.read_text())
    release["build"]["built_with_model_package"]["name"] = "policyengine-us"
    release_path.write_text(json.dumps(release))

    with pytest.raises(
        ReleaseContractError,
        match="built_with_model_package.name.*policyengine-uk",
    ):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    "revision",
    [
        "main",
        UK_NATIONAL_RELEASE_ID + "-",
        # Prefixed but outside the attempt-derived cut-tag grammar: the
        # contract validates the same <YYYYMMDDTHHMMSSZ>-<uuid8> shape the
        # assembler mints, so a hand-edited suffix cannot claim a cut.
        UK_NATIONAL_RELEASE_ID + "-hotfix",
        UK_NATIONAL_RELEASE_ID + "-20260828t101112Z-1a2b3c4d",
        UK_NATIONAL_RELEASE_ID + "-20260828T101112Z-1A2B3C4D",
        # Present but non-string: must fail loudly rather than vanish from
        # the publish layer's string-only revision collection.
        123,
    ],
)
def test_uk_national_release_rejects_invalid_artifact_revisions(
    tmp_path: Path,
    revision: str,
) -> None:
    directory = _write_uk_national_release_dir(tmp_path)
    release_path = directory / "release_manifest.json"
    release = json.loads(release_path.read_text())
    for artifact in release["artifacts"].values():
        artifact["revision"] = revision
    release_path.write_text(json.dumps(release))

    with pytest.raises(ReleaseContractError, match="revision"):
        validate_release_dir(directory)


def test_uk_national_release_rejects_mixed_cut_revisions(tmp_path: Path) -> None:
    # Two individually grammar-valid cut tags are still two cuts: the
    # contract refuses the mixture, not just publish.
    directory = _write_uk_national_release_dir(tmp_path)
    release_path = directory / "release_manifest.json"
    release = json.loads(release_path.read_text())
    first_key = next(iter(release["artifacts"]))
    release["artifacts"][first_key]["revision"] = (
        UK_NATIONAL_RELEASE_ID + "-20260901T000000Z-deadbeef"
    )
    release_path.write_text(json.dumps(release))

    with pytest.raises(ReleaseContractError, match="more than one revision"):
        validate_release_dir(directory)


def test_uk_national_release_binds_signed_evidence_bytes(tmp_path: Path) -> None:
    # Rewriting a copied part report must refuse: the certification signs the
    # evidence bytes, not just the digest fields' shapes.
    directory = _write_uk_national_release_dir(tmp_path)
    (directory / "release_cut_gates.json").write_text(
        json.dumps({"fixture": "tampered"})
    )

    with pytest.raises(
        ReleaseContractError, match="does not match the certification's signed"
    ):
        validate_release_dir(directory)


def test_uk_national_release_requires_signed_evidence_files(tmp_path: Path) -> None:
    directory = _write_uk_national_release_dir(tmp_path)
    (directory / "terminal_gates.json").unlink()

    with pytest.raises(ReleaseContractError, match="missing 'terminal_gates.json'"):
        validate_release_dir(directory)


def test_uk_national_release_refuses_unbindable_score_digest(
    tmp_path: Path,
) -> None:
    # score_receipt.sha256 is the one signed digest outside the shape-checked
    # parts block: a malformed value must refuse the binding, never skip it.
    directory = _write_uk_national_release_dir(tmp_path)
    certification = _green_uk_certification(
        TEST_UK_TERMINAL_GATE_SIGNING_KEY_BYTES,
        release_id=UK_NATIONAL_RELEASE_ID,
        diagnostics_sha256=_sha256(directory / "calibration_diagnostics.json"),
        part_shas={
            "spine": _sha256(directory / "spine_gates.json"),
            "calibration_seam": _sha256(directory / "terminal_gates.json"),
            "release_cut": _sha256(directory / "release_cut_gates.json"),
        },
        score_receipt_sha256="not-a-digest",
    )
    certification_path = directory / "release_certification.json"
    certification_path.write_text(json.dumps(certification))
    release_path = directory / "release_manifest.json"
    release = json.loads(release_path.read_text())
    release["artifacts"]["release_certification"]["sha256"] = _sha256(
        certification_path
    )
    release_path.write_text(json.dumps(release))

    with pytest.raises(ReleaseContractError, match=r"score_receipt\.sha256 is not a"):
        validate_release_dir(directory)


def test_us_release_rejects_dash_suffixed_artifact_revision(
    release_dir: Path,
) -> None:
    release_path = release_dir / "release_manifest.json"
    release = json.loads(release_path.read_text())
    release["artifacts"]["populace_us_2024"]["revision"] = RELEASE_ID + "-cut"
    release_path.write_text(json.dumps(release))

    with pytest.raises(ReleaseContractError, match="revision"):
        validate_release_dir(release_dir)


@pytest.mark.parametrize(
    ("tier", "accepted"),
    [(None, True), ("frs", True), ("cps-transfer", False)],
)
def test_grandfathered_june_release_is_bound_to_its_frs_lineage(
    tmp_path: Path,
    tier: str | None,
    accepted: bool,
) -> None:
    directory = _copy_real_uk_june_release(tmp_path)
    manifest_path = directory / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if tier is not None:
        manifest["tier"] = tier
        manifest_path.write_text(json.dumps(manifest))

    if accepted:
        validate_release_dir(directory)
    else:
        with pytest.raises(ReleaseContractError, match="known FRS lineage"):
            validate_release_dir(directory)


def test_legacy_diagnostics_exemption_is_scoped_to_the_exact_june_id(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    diagnostics_path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["schema_version"] = 2
    for row in diagnostics["targets"]:
        row["aggregation"] = "weighted_sum"
        row.pop("measure")
        row.pop("filter")
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match="publishes version 6"):
        validate_release_dir(directory)


def test_exact_k_uk_release_requires_and_accepts_ratified_tier(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )

    validate_release_dir(directory)


def test_exact_k_uk_required_files_include_terminal_report() -> None:
    assert UK_TERMINAL_GATE_REPORT_FILE in required_release_files(UK_EXACT_K_RELEASE_ID)
    assert UK_TERMINAL_GATE_REPORT_FILE not in required_release_files(UK_RELEASE_ID)


def test_exact_k_uk_release_rejects_missing_terminal_report(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    (directory / UK_TERMINAL_GATE_REPORT_FILE).unlink()

    with pytest.raises(
        ReleaseContractError, match="required file 'terminal_gates.json'"
    ):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_sol_exported_nonzero_only_gates(
    tmp_path: Path,
) -> None:
    """Sol's ``gates={exported_nonzero}`` publication bypass must stay shut."""

    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["gates"] = {"exported_nonzero": {"passed": True}}
    build_path.write_text(json.dumps(build))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    assert "gates must include an 'uk_terminal' report pointer" in "\n".join(
        excinfo.value.failures
    )


def test_exact_k_uk_release_rejects_sol_hand_composed_passing_parity_trio(
    tmp_path: Path,
) -> None:
    """Even rehashed ``enforced:true`` raw parity gates are not an attestation."""

    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / UK_TERMINAL_GATE_REPORT_FILE
    payload = json.loads(path.read_text())
    parity_names = ["export_surface", "target_surface", "target_fit"]
    payload["gates"] = {
        name: {"passed": True, "failures": [], "details": {}} for name in parity_names
    }
    payload["passed"] = True
    payload["enforced"] = True
    payload["attestation"]["evaluated_gates"] = parity_names
    _refresh_terminal_gate_attestation(payload)
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "evaluated gate membership" in failures
    assert "attestation.evaluated_gates" in failures


def test_exact_k_uk_release_rejects_complete_public_hash_forgery(
    tmp_path: Path,
) -> None:
    """All expected names plus reproducible public hashes are not evidence."""

    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / UK_TERMINAL_GATE_REPORT_FILE
    payload = json.loads(path.read_text())
    assert set(payload["gates"]) == {
        *UK_ALWAYS_APPLICABLE_GATE_NAMES,
        "weights_audit",
        "export_surface",
        "target_surface",
        "target_fit",
    }
    for gate in payload["gates"].values():
        gate["details"] = {}
    _refresh_terminal_gate_attestation(payload)
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "honest aggregator detail schema" in failures


def test_exact_k_uk_release_rejects_complete_report_signed_by_untrusted_key(
    tmp_path: Path,
) -> None:
    """Honest-shaped JSON and public digests cannot impersonate the signer."""

    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload, evidence = _terminal_gate_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
        signing_key=FORGED_UK_TERMINAL_GATE_SIGNING_KEY_BYTES,
    )
    build = json.loads((directory / "build_manifest.json").read_text())
    assert build["terminal_gate_evidence"] == evidence
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "does not identify the trusted release key" in failures
    assert "does not authenticate the complete report" in failures


def test_exact_k_uk_release_requires_out_of_band_verification_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    monkeypatch.delenv(UK_TERMINAL_GATE_SIGNING_KEY_ENV)

    with pytest.raises(ReleaseContractError, match="verification requires"):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    ("encoded_key", "match"),
    [
        ("not-base64!", "must be valid base64"),
        (base64.b64encode(b"x" * 31).decode(), "exactly 32 bytes"),
    ],
)
def test_exact_k_uk_verifier_rejects_malformed_or_wrong_length_key(
    tmp_path: Path,
    monkeypatch,
    encoded_key: str,
    match: str,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    monkeypatch.setenv(UK_TERMINAL_GATE_SIGNING_KEY_ENV, encoded_key)

    with pytest.raises(ReleaseContractError, match=match):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_unchanged_cross_id_report_bytes(
    tmp_path: Path,
) -> None:
    sibling_release_id = "populace-uk-2023-cps-transfer-k535080"
    source = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    sibling = _write_uk_release_dir(
        tmp_path,
        sibling_release_id,
        tier="cps-transfer",
    )
    validate_release_dir(source)
    validate_release_dir(sibling)
    source_report = source / UK_TERMINAL_GATE_REPORT_FILE
    sibling_report = sibling / UK_TERMINAL_GATE_REPORT_FILE
    authentic_bytes = source_report.read_bytes()
    authentic_sha256 = hashlib.sha256(authentic_bytes).hexdigest()

    sibling_report.write_bytes(authentic_bytes)
    _refresh_terminal_manifest_hashes(sibling)

    assert sibling_report.read_bytes() == authentic_bytes
    assert _sha256(sibling_report) == authentic_sha256
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(sibling)

    failures = "\n".join(excinfo.value.failures)
    assert "attestation.release_id must match the release being validated" in failures
    assert "attestation.signature does not authenticate" not in failures


def test_exact_k_uk_attestation_binds_exact_calibration_diagnostics_bytes(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    diagnostics_path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["options"]["epochs"] += 1
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(
        ReleaseContractError,
        match="attestation.calibration_diagnostics_sha256",
    ):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_aggregator_report_transplant(
    tmp_path: Path,
) -> None:
    """A genuine four-row gate report cannot attest the k535080 fixture."""

    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / UK_TERMINAL_GATE_REPORT_FILE
    payload = json.loads(path.read_text())
    four_row_summary = {
        "n_records": 4,
        "positive_weight_records": 4,
        "zero_weight_records": 0,
        "total_weight": 4.0,
        "effective_sample_size": 4.0,
        "ess_fraction": 1.0,
        "median_positive_weight": 1.0,
        "max_weight": 1.0,
        "max_to_median_positive_weight": 1.0,
        "top_1pct_weight_share": 0.25,
    }
    for name in ("weight_ess", "weight_ratio"):
        payload["gates"][name]["details"].update(four_row_summary)
    payload["gates"]["zero_weight_strata"]["details"].update(
        household_rows=4,
        zero_weight_rows=0,
        declared_strata=[
            {"name": "june_spi_synthetic_base", "zero_weight_rows": 0},
            {
                "name": "june_spi_synthetic_capital_gains",
                "zero_weight_rows": 0,
            },
        ],
    )
    _refresh_terminal_gate_attestation(payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = dict(payload["attestation"]["evidence_sha256"])
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "weight_ess'.details.n_records must match" in failures
    assert "weight_ratio'.details.n_records must match" in failures
    assert "zero_weight_strata.details.household_rows must match" in failures


def test_exact_k_uk_release_rejects_rehashed_weight_observable_mutation(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / UK_TERMINAL_GATE_REPORT_FILE
    payload = json.loads(path.read_text())
    for name in ("weight_ess", "weight_ratio"):
        payload["gates"][name]["details"]["total_weight"] = 1.0
    _refresh_terminal_gate_attestation(payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = dict(payload["attestation"]["evidence_sha256"])
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "weight_ess'.details.total_weight must match" in failures
    assert "weight_ratio'.details.total_weight must match" in failures
    assert "release_dataset evidence digest" in failures


@pytest.mark.parametrize(
    "evidence_stages",
    [
        (),
        ("hmrc_spi_income",),
        ("release_parity",),
        ("hmrc_spi_income", "release_parity"),
        ("input_mass_parity",),
        ("qrf_tail_concentration",),
        ("input_mass_parity", "qrf_tail_concentration"),
        (
            "hmrc_spi_income",
            "release_parity",
            "input_mass_parity",
            "qrf_tail_concentration",
        ),
    ],
)
def test_exact_k_uk_terminal_membership_tracks_build_evidence_stages(
    tmp_path: Path,
    evidence_stages: tuple[str, ...],
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload, evidence = _terminal_gate_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
        evidence_stages=evidence_stages,
    )
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = evidence
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    validate_release_dir(directory)


def test_exact_k_uk_terminal_rejects_unreviewed_input_mass_identity(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload, evidence = _terminal_gate_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
        evidence_stages=("input_mass_parity",),
    )
    payload["gates"]["input_mass_parity"]["details"]["reference_identity"] = {
        "filename": "caller-selected.h5",
        "revision": "caller-selected",
        "sha256": "b" * 64,
        "vintage": "caller-selected",
    }
    _refresh_terminal_gate_attestation(payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = evidence
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "reference_identity must match the active reviewed" in failures


def test_exact_k_uk_terminal_rejects_substituted_input_mass_totals(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload, _evidence = _terminal_gate_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
        evidence_stages=("input_mass_parity",),
    )
    substituted = _canonical_sha256(
        {
            "reference": {
                "identity": dict(UK_INPUT_MASS_REFERENCE_IDENTITY),
                "totals": {"employment_income": 1.0},
            }
        }
    )
    payload["attestation"]["evidence_sha256"]["input_mass_parity"] = substituted
    _refresh_terminal_gate_attestation(payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = dict(payload["attestation"]["evidence_sha256"])
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "must bind the reviewed enhanced-FRS incumbent totals" in failures


def test_exact_k_uk_terminal_rejects_vacuous_qrf_surface(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload, evidence = _terminal_gate_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
        evidence_stages=("qrf_tail_concentration",),
    )
    details = payload["gates"]["qrf_tail_concentration"]["details"]
    details["columns_checked"] = 0
    details["top_share"] = {}
    details["carrier_counts"] = {}
    details["surface"]["checked_columns"] = []
    details["surface"]["absent_columns"] = ["declared_but_absent"]
    _refresh_terminal_gate_attestation(payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = evidence
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "requires details.columns_checked to be positive" in failures
    assert "requires details.surface.absent_columns to be empty" in failures


def test_exact_k_uk_terminal_rejects_partially_omitted_qrf_surface(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload, _evidence = _terminal_gate_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
        evidence_stages=("qrf_tail_concentration",),
    )
    payload["gates"]["qrf_tail_concentration"]["details"]["surface"][
        "declared_qrf_outputs"
    ] = 2
    _refresh_terminal_gate_attestation(payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = dict(payload["attestation"]["evidence_sha256"])
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "must reconcile declared, checked, absent, nonnumeric" in failures


def test_exact_k_uk_terminal_rejects_evidence_gate_without_its_stage(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload, evidence = _terminal_gate_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
        evidence_stages=(),
    )
    payload["gates"]["weights_audit"] = {
        "passed": True,
        "failures": [],
        "details": _terminal_gate_details("weights_audit"),
    }
    payload["attestation"]["evaluated_gates"].append("weights_audit")
    _refresh_terminal_gate_attestation(payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = evidence
    build_path.write_text(json.dumps(build))
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "evaluated gate membership" in failures
    assert "attestation.evaluated_gates" in failures
    assert "attestation.signature does not authenticate" not in failures


def test_exact_k_uk_terminal_report_rejects_non_aggregator_producer(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / UK_TERMINAL_GATE_REPORT_FILE
    payload = json.loads(path.read_text())
    payload["attestation"]["producer"] = "sol.hand_composed_gate_report"
    _refresh_terminal_gate_attestation(payload)
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(
        ReleaseContractError, match="honest UK terminal gate aggregator"
    ):
        validate_release_dir(directory)


def test_uk_terminal_policy_pins_are_in_lockstep_with_the_contract() -> None:
    """Both reviewed digests — current and grandfathered — match the
    contract module's private literals, so a typo in either constant is
    detectable even while the legacy branch stays defensively unreachable
    (the June id is not an exact-k id, so its report is never checked)."""

    from microcosm.data import contract as contract_module

    assert (
        contract_module._UK_TERMINAL_GATE_POLICY_SHA256
        == UK_TERMINAL_GATE_POLICY_SHA256
    )
    assert (
        contract_module._UK_TERMINAL_GATE_POLICY_SHA256_LEGACY
        == UK_TERMINAL_GATE_POLICY_SHA256_LEGACY
    )


def test_exact_k_uk_terminal_report_rejects_uncertified_policy(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / UK_TERMINAL_GATE_REPORT_FILE
    payload = json.loads(path.read_text())
    payload["attestation"]["policy_sha256"] = "0" * 64
    _refresh_terminal_gate_attestation(payload)
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError, match="certified UK gate policy"):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    ("failed_field", "match"),
    [
        ("report", "terminal_gates.json passed must be true"),
        ("gate", "gate 'weight_ratio'.passed must be true"),
    ],
)
def test_exact_k_uk_terminal_report_requires_green_verdicts(
    tmp_path: Path,
    failed_field: str,
    match: str,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / UK_TERMINAL_GATE_REPORT_FILE
    payload = json.loads(path.read_text())
    if failed_field == "report":
        payload["passed"] = False
    else:
        payload["gates"]["weight_ratio"].update(
            passed=False,
            failures=["reviewed ratio exceeded"],
        )
    _refresh_terminal_gate_attestation(payload)
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError, match=match):
        validate_release_dir(directory)


def _assert_no_resign_terminal_mutation_rejected(
    directory: Path,
    payload: dict,
    *,
    named_failures: tuple[str, ...],
) -> None:
    original_signature = payload["attestation"]["signature"]
    _write_terminal_and_refresh_manifest_hashes(directory, payload)
    persisted = json.loads(
        (directory / UK_TERMINAL_GATE_REPORT_FILE).read_text(encoding="utf-8")
    )
    assert persisted["attestation"]["signature"] == original_signature

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "attestation.signature does not authenticate" in failures
    for named_failure in named_failures:
        assert named_failure in failures


def test_exact_k_uk_terminal_no_resign_altered_gate_verdict_fails_authentication(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload = json.loads(
        (directory / UK_TERMINAL_GATE_REPORT_FILE).read_text(encoding="utf-8")
    )
    payload["gates"]["weight_ratio"]["passed"] = False

    _assert_no_resign_terminal_mutation_rejected(
        directory,
        payload,
        named_failures=(
            "gate 'weight_ratio'.passed must be true",
            "gate_results_sha256 does not match gates",
        ),
    )


def test_exact_k_uk_terminal_no_resign_removed_gate_fails_authentication(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload = json.loads(
        (directory / UK_TERMINAL_GATE_REPORT_FILE).read_text(encoding="utf-8")
    )
    del payload["gates"]["weight_ratio"]

    _assert_no_resign_terminal_mutation_rejected(
        directory,
        payload,
        named_failures=(
            "evaluated gate membership",
            "gate_results_sha256 does not match gates",
        ),
    )


def test_exact_k_uk_terminal_no_resign_altered_policy_fails_authentication(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload = json.loads(
        (directory / UK_TERMINAL_GATE_REPORT_FILE).read_text(encoding="utf-8")
    )
    payload["attestation"]["policy_sha256"] = "0" * 64

    _assert_no_resign_terminal_mutation_rejected(
        directory,
        payload,
        named_failures=("does not match the certified UK gate policy",),
    )


def test_exact_k_uk_terminal_no_resign_substituted_evidence_fails_authentication(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload = json.loads(
        (directory / UK_TERMINAL_GATE_REPORT_FILE).read_text(encoding="utf-8")
    )
    payload["attestation"]["evidence_sha256"]["hmrc_spi_income"] = "0" * 64

    _assert_no_resign_terminal_mutation_rejected(
        directory,
        payload,
        named_failures=("must exactly match build_manifest.json",),
    )


def test_exact_k_uk_terminal_evidence_digest_must_match_build_manifest(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"]["hmrc_spi_income"] = "0" * 64
    build_path.write_text(json.dumps(build))

    with pytest.raises(ReleaseContractError, match="must exactly match build_manifest"):
        validate_release_dir(directory)


def test_exact_k_uk_terminal_report_recomputes_attestation_digests(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / UK_TERMINAL_GATE_REPORT_FILE
    payload = json.loads(path.read_text())
    payload["gates"]["weight_ratio"]["details"] = {"tampered": True}
    _write_terminal_and_refresh_manifest_hashes(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "gate_results_sha256 does not match gates" in failures
    assert "attestation.signature does not authenticate" in failures


def test_exact_k_uk_terminal_report_hashes_cross_link_both_manifests(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["gates"]["uk_terminal"]["sha256"] = "0" * 64
    build_path.write_text(json.dumps(build))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "gates.uk_terminal.sha256 must match the local" in failures
    assert "terminal gate report sha256 values must match" in failures


def test_exact_k_uk_release_rejects_sol_k535080_n_records_two_probe(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    _rewrite_exact_k_fixture_to_two_records(directory)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    for expected in (
        "build_manifest.json canonical UK 'n_records'",
        "release_manifest.json canonical UK 'record_count'",
        "release_manifest.json canonical UK 'n_records'",
        "canonical UK top-level n_records",
        "canonical UK uk_diagnostics.weights.n_records",
        "canonical UK target_surface.n_records",
    ):
        assert expected in failures
    assert failures.count("release-id record count 535080") >= 6


def test_exact_k_uk_release_rejects_sol_identity_field_probe(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    _rewrite_exact_k_fixture_to_two_records(directory)
    manifest_path = directory / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(country="xx", year=1900)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "canonical UK 'country' must be 'uk', got 'xx'" in failures
    assert "release-id year 2023, got 1900" in failures
    assert "release-id record count 535080" in failures


@pytest.mark.parametrize("tier", ["public", "true", "full"])
def test_exact_k_uk_release_rejects_unratified_tier(
    tmp_path: Path,
    tier: str,
) -> None:
    release_id = f"populace-uk-2023-{tier}-k535080"
    directory = _write_uk_release_dir(tmp_path, release_id, tier=tier)

    with pytest.raises(ReleaseContractError, match="unratified tier"):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_missing_manifest_tier(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(tmp_path, UK_EXACT_K_RELEASE_ID)

    with pytest.raises(ReleaseContractError, match="top-level 'tier'"):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_tier_mismatch(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="cps-transfer",
    )

    with pytest.raises(ReleaseContractError, match="release id names tier 'frs'"):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    "release_id",
    [
        "populace-uk-2023-frs-k0",
        "populace-uk-2023-public-k535080-extra",
        "populace-uk-2023-frs",
        "populace-uk-2099-deadbee-20990101",
    ],
)
def test_malformed_uk_release_ids_are_not_grandfathered(
    tmp_path: Path,
    release_id: str,
) -> None:
    directory = _write_uk_release_dir(tmp_path, release_id, tier="frs")

    with pytest.raises(ReleaseContractError, match="neither canonical"):
        validate_release_dir(directory)


def test_exact_k_uk_release_requires_standard_diagnostics(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    diagnostics = _calibration_diagnostics()
    diagnostics["target_registry"]["country"] = "uk"
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match="require a 'uk_diagnostics'"):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("zero_reconciliation", "zero-weight stratum rows do not reconcile"),
        ("missing_geography", "missing level"),
    ],
)
def test_exact_k_uk_release_diagnostics_fail_closed(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(path.read_text())
    uk = diagnostics["uk_diagnostics"]
    if mutation == "zero_reconciliation":
        uk["weights"]["zero_weight_records"] = 0
    else:
        uk["target_pass_rates_by_geography_level"].pop()
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match=match):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    "field",
    ["median_positive_weight", "max_to_median_positive_weight"],
)
def test_exact_k_uk_release_requires_weight_ratio_fields(
    tmp_path: Path,
    field: str,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(path.read_text())
    diagnostics["uk_diagnostics"]["weights"].pop(field)
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match=field):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_impossible_or_inconsistent_ess(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(path.read_text())
    diagnostics["uk_diagnostics"]["weights"]["effective_sample_size"] = 999.0
    diagnostics["uk_diagnostics"]["weights"]["ess_fraction"] = 0.25
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match="effective_sample_size"):
        validate_release_dir(directory)


def test_exact_k_uk_release_recomputes_max_to_positive_median_ratio(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(path.read_text())
    weights = diagnostics["uk_diagnostics"]["weights"]
    assert weights["max_weight"] == weights["median_positive_weight"] == 1.0
    weights["max_to_median_positive_weight"] = 999.0
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(
        ReleaseContractError,
        match="must equal max_weight/median_positive_weight",
    ):
        validate_release_dir(directory)


def test_the_1abddeb_shape_is_rejected(release_dir: Path) -> None:
    """The regression: a release with only an unversioned release manifest."""
    (release_dir / "build_manifest.json").unlink()
    (release_dir / "calibration_diagnostics.json").unlink()
    (release_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "release_id": RELEASE_ID,
                "country_id": "us",
                "artifacts": {},
                "validation": {},
            }
        )
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "build_manifest.json" in failures
    assert "schema_version" in failures


def test_schema_drift_is_rejected_by_version(release_dir: Path) -> None:
    manifest = _release_manifest()
    manifest["schema_version"] = RELEASE_MANIFEST_SCHEMA_VERSION + 1
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="schema_version"):
        validate_release_dir(release_dir)


def test_build_id_mismatch_names_both_ids(release_dir: Path) -> None:
    (release_dir / "build_manifest.json").write_text(
        json.dumps(_build_manifest("populace-us-2024-other-20260101"))
    )
    with pytest.raises(ReleaseContractError, match="populace-us-2024-other"):
        validate_release_dir(release_dir)


def test_release_manifest_build_id_must_match_directory(
    release_dir: Path,
) -> None:
    manifest = _release_manifest("populace-us-2024-other-20260101")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="build.build_id"):
        validate_release_dir(release_dir)


def test_artifact_entries_must_carry_provenance(release_dir: Path) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"].pop("sha256")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="sha256"):
        validate_release_dir(release_dir)


def test_release_manifest_must_list_calibration_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"].pop("calibration_diagnostics")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        validate_release_dir(release_dir)


def test_release_manifest_must_list_us_source_coverage_for_us_release(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"].pop("us_source_coverage")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures


def test_release_manifest_rejects_us_split_microdata_artifacts(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"]["states/CA"] = _split_microdata_artifact_entry(
        RELEASE_ID,
        "states/CA",
    )
    manifest["artifacts"]["districts/AK-01"] = _split_microdata_artifact_entry(
        RELEASE_ID,
        "districts/AK-01",
    )
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "single national microdata artifact" in failures
    assert "states/CA" in failures
    assert "districts/AK-01" in failures


def test_release_manifest_local_calibration_diagnostics_hash_must_match(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"]["calibration_diagnostics"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'calibration_diagnostics' declares sha256" in failures
    assert "calibration_diagnostics.json" in failures


def test_release_manifest_local_us_source_coverage_hash_must_match(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"]["us_source_coverage"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'us_source_coverage' declares sha256" in failures
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures


def test_build_manifest_requires_clean_git_commit(release_dir: Path) -> None:
    manifest = _build_manifest()
    manifest["code"]["git_dirty"] = True
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "code.git_dirty" in failures


def test_target_surface_hash_must_match_between_manifest_and_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _build_manifest()
    manifest["calibration"]["target_surface"]["sha256"] = "1" * 64
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "target_surface.sha256 must match" in failures


def test_target_registry_version_must_match_between_manifest_and_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _build_manifest()
    manifest["calibration"]["target_registry"]["version"] = "other"
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "target_registry.version must match" in failures


def test_unparseable_manifest_is_a_named_failure(release_dir: Path) -> None:
    (release_dir / "build_manifest.json").write_text("{not json")
    with pytest.raises(ReleaseContractError, match="not valid JSON"):
        validate_release_dir(release_dir)


def test_malformed_calibration_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text(
        json.dumps({"schema_version": 1})
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "calibration_diagnostics.json" in failures
    assert "targets" in failures


def test_malformed_us_source_coverage_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps({"schema_version": 1})
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures
    assert "coverage_summary" in failures


def test_us_source_coverage_rejects_legacy_commit_contract(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["source_contract"].pop("ledger_commit")
    payload["source_contract"]["".join(("ar", "ch", "_commit"))] = GIT_COMMIT
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "source_contract.ledger_commit" in failures


def test_failed_us_source_coverage_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["gate"] = {
        "name": "us_source_coverage",
        "passed": False,
        "failures": ["social_security_ssi/ssa-ssi-table-7b1-2024 missing"],
    }
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "gate.passed must be true" in failures
    assert "gate.failures must be empty" in failures


def test_us_source_coverage_reviewed_exclusions_need_reasons(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["reviewed_exclusions"] = {"ssa-ssi-table-7b1-2024": ""}
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "reviewed_exclusions need non-empty string reasons" in failures


def test_us_source_coverage_requires_fiscal_target_sources(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    del payload["fiscal_target_sources"]
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources" in failures


def test_us_source_coverage_must_cover_calibrated_families(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"] = {}
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources must cover every calibrated target family" in failures


def test_us_source_coverage_must_not_claim_uncalibrated_families(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"]["jct"] = {
        "label": "Joint Committee on Taxation",
        "target_count": 1,
        "sources": ["JCT tax expenditures"],
        "reference_urls": ["https://example.test/jct"],
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=payload,
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "unexpected ['jct']" in failures


def test_us_source_coverage_target_counts_match_calibration(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"]["cbo"]["target_count"] = 2
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=payload,
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources['cbo'].target_count is 2" in failures
    assert "has 1 calibrated target(s)" in failures


def test_build_manifest_requires_runtime_versions(release_dir: Path) -> None:
    manifest = _build_manifest()
    del manifest["runtime"]
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "runtime" in failures


def test_build_manifest_rejects_unknown_runtime_versions(release_dir: Path) -> None:
    manifest = _build_manifest()
    manifest["runtime"]["policyengine-us"] = "not-installed"
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "runtime.policyengine-us" in failures


def test_release_manifest_must_include_dataset_root_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    del manifest["artifacts"]["populace_us_2024"]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "dataset root artifact" in failures


def test_release_manifest_requires_policyengine_certification_shape(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    del manifest["data_package"]
    del manifest["default_datasets"]
    del manifest["build"]["built_with_model_package"]
    del manifest["compatible_model_packages"]
    del manifest["artifacts"]["populace_us_2024"]["revision"]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "data_package" in failures
    assert "default_datasets" in failures
    assert "build.built_with_model_package" in failures
    assert "compatible_model_packages" in failures
    assert "artifact 'populace_us_2024' is missing 'revision'" in failures


def test_release_manifest_rejects_unresolved_package_versions(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["data_package"]["version"] = "not-installed"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "data_package.version" in failures
    assert "not-installed" in failures


def test_release_manifest_requires_compatible_core_package(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    del manifest["compatible_core_packages"]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "compatible_core_packages" in failures


def test_release_manifest_compatible_specifier_must_be_valid(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["compatible_model_packages"][0]["specifier"] = "not a specifier"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "valid PEP 440 specifier" in failures


def test_release_manifest_compatible_model_package_must_cover_build_version(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["compatible_model_packages"] = [
        {"name": "policyengine-us", "specifier": "==1.728.0"}
    ]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "built policyengine-us version '1.729.0'" in failures


def test_release_manifest_compatible_core_package_must_cover_build_version(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["compatible_core_packages"] = [
        {"name": "policyengine-core", "specifier": "==3.18.0"}
    ]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "built policyengine-core version '3.19.0'" in failures


def test_release_manifest_default_dataset_must_name_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["default_datasets"]["national"] = "missing_dataset"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default_datasets.national" in failures
    assert "missing_dataset" in failures


def test_release_manifest_default_dataset_must_be_microdata_root_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["default_datasets"]["national"] = "calibration_diagnostics"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default_datasets.national" in failures
    assert "not 'microdata'" in failures
    assert "dataset root artifact" in failures


def test_release_manifest_default_dataset_hash_must_match_build_manifest(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default dataset artifact" in failures
    assert "matching build_manifest.json" in failures


def test_release_manifest_artifact_revisions_must_pin_release_tag(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["revision"] = "main"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'populace_us_2024' revision" in failures
    assert RELEASE_ID in failures


def test_release_manifest_root_artifact_hashes_match_build_manifest(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "sha256 matching build_manifest.json" in failures


def test_all_failures_reported_at_once(release_dir: Path) -> None:
    """A publisher sees the full repair list, not one failure per run."""
    (release_dir / "calibration_diagnostics.json").unlink()
    manifest = _release_manifest()
    del manifest["schema_version"]
    manifest["artifacts"] = {}
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    assert len(excinfo.value.failures) >= 3


def test_a_missing_directory_is_a_contract_error(tmp_path: Path) -> None:
    with pytest.raises(ReleaseContractError, match="is not a directory"):
        validate_release_dir(tmp_path / "releases" / "nope")


def test_us_release_rejects_table_1_4_national_dollar_breach(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024"
    )
    # Replay the live Build M defect (microcosm#462) onto the fixture's Table
    # 1.4 row: the capital-gain-distributions dollar row shipped at +634.8%
    # relative error, recorded in the release's own diagnostics.
    target["name"] = (
        "irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount@2024"
    )
    target["target_name"] = (
        "irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount"
    )
    target["target"] = 10_155_465_319.0
    target["compiled_target"] = 10_155_465_319.0
    target["initial_estimate"] = 10_155_465_319.0
    target["final_estimate"] = 74_617_447_202.0
    target["relative_error"] = (74_617_447_202.0 - 10_155_465_319.0) / 10_155_465_319.0
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "SOI Pub 1304 Table 1.4 national dollar rows" in failures
    assert "capital_gain_distributions_amount" in failures
    assert "relative_error=6.3475" in failures


def test_us_release_requires_table_1_4_national_dollar_rows(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024"
    )
    # Rename the only Table 1.4 dollar row out of the class (keeping the row
    # count intact): a diagnostics surface with no national Table 1.4 dollar
    # row must not certify — a dropped or renamed feed family gates nothing.
    target["name"] = "irs_soi.ty2023.table_1_9.all.wages_salaries_amount@2024"
    target["target_name"] = "irs_soi.ty2023.table_1_9.all.wages_salaries_amount"
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "soi_table_1_4_national_dollar_rows" in failures


def test_us_release_table_1_4_returns_rows_are_outside_the_dollar_blanket(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row for row in diagnostics["targets"] if row["name"] == "population@2024"
    )
    # The live Build M estate-trust net-loss RETURNS row landed at +495.9%; a
    # count row is a distinct defect class the dollar blanket must not gate.
    target["name"] = "irs_soi.ty2023.table_1_4.all.estate_trust_net_loss_returns@2024"
    target["target_name"] = "irs_soi.ty2023.table_1_4.all.estate_trust_net_loss_returns"
    target["target"] = 36_592.0
    target["compiled_target"] = 36_592.0
    target["initial_estimate"] = 36_592.0
    target["final_estimate"] = 218_052.0
    target["relative_error"] = (218_052.0 - 36_592.0) / 36_592.0
    target["registry"] = {"family": "irs_soi"}
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    source_coverage = json.loads(
        (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).read_text()
    )
    del source_coverage["fiscal_target_sources"]["cbo"]
    source_coverage["fiscal_target_sources"]["irs_soi"]["target_count"] += 1
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=source_coverage,
    )

    validate_release_dir(release_dir)


def _resignable_qrf_release(tmp_path: Path) -> tuple[Path, dict]:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    payload, evidence = _terminal_gate_payload(
        release_id=UK_EXACT_K_RELEASE_ID,
        calibration_diagnostics_sha256=_sha256(
            directory / "calibration_diagnostics.json"
        ),
        evidence_stages=("qrf_tail_concentration",),
    )
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = evidence
    build_path.write_text(json.dumps(build))
    return directory, payload


def _write_resigned_qrf_release(directory: Path, payload: dict) -> None:
    _refresh_terminal_gate_attestation(payload)
    _write_terminal_and_refresh_manifest_hashes(directory, payload)


def test_exact_k_uk_terminal_rejects_unexcluded_high_qrf_share(
    tmp_path: Path,
) -> None:
    directory, payload = _resignable_qrf_release(tmp_path)
    details = payload["gates"]["qrf_tail_concentration"]["details"]
    details["top_share"]["self_employment_income"] = 1.0
    _write_resigned_qrf_release(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "columns above details.max_top_share" in failures
    assert "attestation.signature does not authenticate" not in failures


def test_exact_k_uk_terminal_rejects_qrf_thin_count_at_minimum(
    tmp_path: Path,
) -> None:
    directory, payload = _resignable_qrf_release(tmp_path)
    details = payload["gates"]["qrf_tail_concentration"]["details"]
    details["thin_columns"]["invented_thin_output"] = details["min_nonzero_records"]
    details["surface"]["declared_qrf_outputs"] = 2
    details["surface"]["checked_columns"].append("invented_thin_output")
    _write_resigned_qrf_release(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "details.thin_columns values" in failures
    assert "must reconcile declared, checked, absent, nonnumeric" not in failures
    assert "attestation.signature does not authenticate" not in failures


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("top_k", True, "details.top_k"),
        ("top_k", 0, "details.top_k"),
        ("max_top_share", True, "details.max_top_share"),
        ("max_top_share", 1.0, "details.max_top_share"),
        ("min_nonzero_records", True, "details.min_nonzero_records"),
        ("min_nonzero_records", 1, "details.min_nonzero_records"),
    ],
)
def test_exact_k_uk_terminal_rejects_invalid_qrf_thresholds(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    directory, payload = _resignable_qrf_release(tmp_path)
    payload["gates"]["qrf_tail_concentration"]["details"][field] = value
    _write_resigned_qrf_release(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert match in failures
    assert "attestation.signature does not authenticate" not in failures


def test_exact_k_uk_terminal_rejects_boolean_qrf_carrier_count(
    tmp_path: Path,
) -> None:
    directory, payload = _resignable_qrf_release(tmp_path)
    details = payload["gates"]["qrf_tail_concentration"]["details"]
    details["carrier_counts"]["self_employment_income"] = True
    _write_resigned_qrf_release(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert "details.carrier_counts values" in failures
    assert "attestation.signature does not authenticate" not in failures


def test_exact_k_uk_terminal_rejects_mixed_exclusion_evaluation_dates(
    tmp_path: Path,
) -> None:
    """One report evaluates every register on one date; a hand-composed
    collection mixing evaluation dates is not the aggregator's output
    (adversarial-review finding: expiry enforcement was otherwise
    invisible to the contract)."""

    directory, payload = _resignable_qrf_release(tmp_path)
    degenerate = payload["gates"]["degenerate_release_surface"]["details"]
    degenerate["exclusions_evaluated_on"] = "2020-01-01"
    _write_resigned_qrf_release(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)
    failures = "\n".join(excinfo.value.failures)
    assert "share one details.exclusions_evaluated_on" in failures


def test_exact_k_uk_terminal_requires_the_exclusion_receipt_fields(
    tmp_path: Path,
) -> None:
    """A key-signed report cannot simply omit expired_exclusions to dodge
    the empty-list expectation: the fields are part of the required detail
    schema (adversarial-review finding — the checks previously defaulted
    absent fields to empty)."""

    directory, payload = _resignable_qrf_release(tmp_path)
    degenerate = payload["gates"]["degenerate_release_surface"]["details"]
    del degenerate["expired_exclusions"]
    _write_resigned_qrf_release(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)
    failures = "\n".join(excinfo.value.failures)
    assert "honest aggregator detail schema" in failures
    assert "expired_exclusions" in failures


def test_exact_k_uk_terminal_accepts_high_qrf_share_with_live_exclusion(
    tmp_path: Path,
) -> None:
    directory, payload = _resignable_qrf_release(tmp_path)
    details = payload["gates"]["qrf_tail_concentration"]["details"]
    details["top_share"]["self_employment_income"] = 1.0
    details["reviewed_exclusions"] = {
        "self_employment_income": "Reviewed concentrated QRF instrument."
    }
    _write_resigned_qrf_release(directory, payload)

    validate_release_dir(directory)


@pytest.mark.parametrize(
    ("observable", "value", "match"),
    [
        ("top_share", True, "details.top_share values"),
        ("top_share", -0.1, "details.top_share values"),
        ("carrier_counts", 1, "details.carrier_counts values"),
        ("thin_columns", True, "details.thin_columns values"),
        ("thin_columns", -1, "details.thin_columns values"),
    ],
)
def test_exact_k_uk_terminal_rejects_invalid_qrf_observable_values(
    tmp_path: Path,
    observable: str,
    value: object,
    match: str,
) -> None:
    directory, payload = _resignable_qrf_release(tmp_path)
    details = payload["gates"]["qrf_tail_concentration"]["details"]
    column = "self_employment_income"
    if observable == "thin_columns":
        column = "invented_thin_output"
        details["surface"]["declared_qrf_outputs"] = 2
        details["surface"]["checked_columns"].append(column)
    details[observable][column] = value
    _write_resigned_qrf_release(directory, payload)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(directory)

    failures = "\n".join(excinfo.value.failures)
    assert match in failures
    assert "attestation.signature does not authenticate" not in failures


# ---------------------------------------------------------------------------
# Schema-4 gate-battery reports (the #611 executor).
# ---------------------------------------------------------------------------


def test_uk_gate_battery_pins_are_in_lockstep_with_the_contract() -> None:
    from microcosm.data import contract as contract_module

    assert contract_module._UK_GATE_BATTERY_PRODUCER == UK_GATE_BATTERY_PRODUCER
    assert (
        contract_module._UK_GATE_BATTERY_SIGNING_KEY_ENV
        == UK_GATE_BATTERY_SIGNING_KEY_ENV
    )
    assert (
        contract_module._UK_GATE_BATTERY_POLICY_SHA256 == UK_GATE_BATTERY_POLICY_SHA256
    )
    assert (
        contract_module._UK_GATE_BATTERY_GATES_MANIFEST_SHA256
        == UK_GATE_BATTERY_GATES_MANIFEST_SHA256
    )
    assert (
        contract_module._UK_GATE_BATTERY_SPEC_FINGERPRINT
        == UK_GATE_BATTERY_SPEC_FINGERPRINT
    )
    assert contract_module._UK_GATE_BATTERY_ENTRY_IDS == frozenset(
        UK_GATE_BATTERY_ENTRIES
    )
    assert contract_module._UK_GATE_BATTERY_ENTRY_GATES == {
        entry_id: (gate, phase)
        for entry_id, (gate, phase, _detail) in UK_GATE_BATTERY_ENTRIES.items()
    }
    assert (
        contract_module._UK_GATE_BATTERY_DEGENERATE_EVIDENCE_SHA256
        == UK_GATE_BATTERY_DEGENERATE_EVIDENCE_SHA256
    )
    assert (
        contract_module._UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256
        == UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256
    )
    assert UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256 == _canonical_sha256(
        {
            "reference": UK_INPUT_MASS_ACTIVE_REFERENCE,
            "reference_evidence_sha256": UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256,
            "exclusions_policy": "committed",
            "reviewed_exclusions": UK_INPUT_MASS_REVIEWED_EXCLUSIONS,
        }
    )


def test_exact_k_uk_gate_battery_report_validates_end_to_end(tmp_path: Path) -> None:
    directory, _payload = _write_battery_release(tmp_path)

    validate_release_dir(directory)


def test_exact_k_uk_gate_battery_rejects_unknown_report_schema(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    payload["schema_version"] = 5
    _rewrite_battery_report(directory, payload, resign=False)

    assert "schema_version must be 3" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_a_non_candidate_report(
    tmp_path: Path,
) -> None:
    # A report produced off the candidate posture excused absent evidence;
    # promoting it into a release dir must fail even when honestly signed.
    directory, payload = _write_battery_release(tmp_path)
    payload["release_candidate"] = False
    _rewrite_battery_report(directory, payload)

    assert "release_candidate must be true" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_loosened_input_mass_tolerance(
    tmp_path: Path,
) -> None:
    # The tolerance is committed spec, not report self-description: an
    # honestly re-signed report claiming a looser fence still refuses.
    directory, payload = _write_battery_release(tmp_path)
    details = payload["gates"]["uk_input_mass_parity"]["details"]
    details["relative_tolerance"] = 50.0
    _rewrite_battery_report(directory, payload)

    assert "relative_tolerance must equal the committed spec value" in (
        _battery_failures(directory)
    )


def test_exact_k_uk_gate_battery_rejects_loosened_qrf_thresholds(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    details = payload["gates"]["uk_qrf_tail_concentration"]["details"]
    details["max_top_share"] = 0.999999
    _rewrite_battery_report(directory, payload)

    assert "max_top_share to equal the committed spec value" in (
        _battery_failures(directory)
    )


def test_exact_k_uk_gate_battery_rejects_shrunken_qrf_tail(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    details = payload["gates"]["uk_qrf_tail_concentration"]["details"]
    details["top_k"] = 5
    details["min_nonzero_records"] = 6
    _rewrite_battery_report(directory, payload)

    failures = _battery_failures(directory)
    assert "top_k to equal the committed spec value" in failures
    assert "min_nonzero_records to equal the committed spec value" in failures


def test_exact_k_uk_gate_battery_rejects_a_blocked_report(tmp_path: Path) -> None:
    directory, payload = _write_battery_release(tmp_path)
    payload["blocked_at_phase"] = "terminal"
    _rewrite_battery_report(directory, payload)

    assert "blocked_at_phase must be null" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_recomputes_shippability(tmp_path: Path) -> None:
    # shippable: true is asserted but never trusted — a failed blocking
    # entry inside an honestly re-signed report still refuses.
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_weight_ratio"]
    entry["status"] = "failed"
    entry["failures"] = ["seeded ratio failure"]
    _rewrite_battery_report(directory, payload)

    failures = _battery_failures(directory)
    assert "release-blocking with status 'failed'" in failures


def test_exact_k_uk_gate_battery_status_checks_cover_diagnostic_entries(
    tmp_path: Path,
) -> None:
    # The diagnostic label exempts an entry from the shippability recompute
    # and nothing else: a diagnostic gate that compared nothing, or that
    # carries a status outside the taxonomy, is still refused.
    for status, expected in (
        ("not_applicable", "claims not_applicable"),
        ("unreached", "is unreached"),
        ("error", "outside the taxonomy"),
    ):
        directory, payload = _write_battery_release(tmp_path / status)
        payload["gates"]["uk_local_target_fit"]["status"] = status
        _rewrite_battery_report(directory, payload)

        assert expected in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_relabelled_diagnostic_criticality(
    tmp_path: Path,
) -> None:
    # Criticality is pinned per entry, so a blocking gate cannot be relabelled
    # diagnostic to dodge the recompute, nor the reverse.
    directory, payload = _write_battery_release(tmp_path)
    payload["gates"]["uk_local_area_support"]["criticality"] = "diagnostic"
    _rewrite_battery_report(directory, payload)

    assert "criticality must be 'release_blocking'" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_passed_entry_with_failures(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_weight_ratio"]
    entry["status"] = "failed"
    entry["failures"] = ["seeded ratio failure"]
    entry["status"] = "passed"
    _rewrite_battery_report(directory, payload)

    failures = _battery_failures(directory)
    assert "passed entries cannot carry failure text" in failures


def test_exact_k_uk_gate_battery_rejects_passed_entry_with_reason(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_weight_ratio"]
    entry["reason"] = "seeded caveat"
    _rewrite_battery_report(directory, payload)

    failures = _battery_failures(directory)
    assert "passed entries cannot carry a reason" in failures


def test_exact_k_uk_gate_battery_rejects_absent_evidence_without_reason(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_input_mass_parity"]
    entry["status"] = "evidence_absent"
    entry["details"] = {}
    entry["reason"] = None
    del payload["evidence_sha256"]["uk_input_mass_parity"]
    _rewrite_battery_report(directory, payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = dict(payload["evidence_sha256"])
    build_path.write_text(json.dumps(build))
    _refresh_terminal_manifest_hashes(directory)

    failures = _battery_failures(directory)
    assert "evidence_absent entries must carry a non-empty string reason" in failures


def test_exact_k_uk_gate_battery_rejects_failed_entry_without_failures(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_weight_ratio"]
    entry["status"] = "failed"
    entry["failures"] = []
    _rewrite_battery_report(directory, payload)

    failures = _battery_failures(directory)
    assert "failed entries must carry non-empty failure text" in failures


def test_exact_k_uk_gate_battery_reports_details_and_failures_type_errors(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_weight_ratio"]
    entry["details"] = []
    entry["failures"] = "seeded failure"
    _rewrite_battery_report(directory, payload)

    failures = _battery_failures(directory)
    assert ".details must be an object" in failures
    assert ".failures must be a list" in failures


def test_exact_k_uk_gate_battery_rejects_excused_absent_evidence(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_input_mass_parity"]
    entry["status"] = "evidence_absent"
    entry["details"] = {}
    entry["reason"] = "missing evidence: input_mass_reference"
    del payload["evidence_sha256"]["uk_input_mass_parity"]
    _rewrite_battery_report(directory, payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = dict(payload["evidence_sha256"])
    build_path.write_text(json.dumps(build))
    _refresh_terminal_manifest_hashes(directory)

    failures = _battery_failures(directory)
    assert "release-blocking with status 'evidence_absent'" in failures


def test_exact_k_uk_gate_battery_rejects_a_missing_entry(tmp_path: Path) -> None:
    directory, payload = _write_battery_release(tmp_path)
    del payload["gates"]["uk_weight_ess"]
    _rewrite_battery_report(directory, payload)

    assert "exactly the declared UK entry ids" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_an_uncertified_policy(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    payload["policy_sha256"] = "b" * 64
    _rewrite_battery_report(directory, payload)

    assert "certified UK gate policy" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_a_moved_manifest_digest(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    payload["gates_manifest_sha256"] = "b" * 64
    _rewrite_battery_report(directory, payload)

    assert "committed uk/gates.json" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_a_moved_spec_fingerprint(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    payload["spec_fingerprint"] = "b" * 64
    _rewrite_battery_report(directory, payload)

    assert "spec_fingerprint does not match" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_a_broken_diagnostics_link(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    payload["release_evidence"] = {"calibration_diagnostics_sha256": "b" * 64}
    _rewrite_battery_report(directory, payload)

    assert (
        "release_evidence.calibration_diagnostics_sha256 must match"
        in _battery_failures(directory)
    )


def test_exact_k_uk_gate_battery_rejects_a_mixed_exclusion_clock(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    details = payload["gates"]["uk_degenerate_release_surface"]["details"]
    details["exclusions_evaluated_on"] = "2026-01-01"
    _rewrite_battery_report(directory, payload)

    assert "exclusion-consuming gates must" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_no_resign_tamper_fails_authentication(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    payload["gates"]["uk_weight_ratio"]["details"]["max_weight"] = 1.0e9
    _rewrite_battery_report(directory, payload, resign=False)

    failures = _battery_failures(directory)
    assert "attestation.signature does not authenticate" in failures
    assert "attestation.gate_outcomes_sha256 does not match" in failures


def test_exact_k_uk_gate_battery_rejects_a_forged_key(tmp_path: Path) -> None:
    directory, payload = _write_battery_release(tmp_path)
    _resign_gate_battery(payload, signing_key=FORGED_UK_TERMINAL_GATE_SIGNING_KEY_BYTES)
    _rewrite_battery_report(directory, payload, resign=False)

    failures = _battery_failures(directory)
    assert "signing_key_sha256 does not identify the trusted release key" in failures


def test_exact_k_uk_gate_battery_requires_the_executor_key_env(
    tmp_path: Path, monkeypatch
) -> None:
    directory, _payload = _write_battery_release(tmp_path)
    monkeypatch.delenv(UK_GATE_BATTERY_SIGNING_KEY_ENV)

    assert UK_GATE_BATTERY_SIGNING_KEY_ENV in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_a_recorded_signing_error(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    _resign_gate_battery(payload)
    payload["attestation"]["signing_error"] = "key was absent at build time"
    _rewrite_battery_report(directory, payload, resign=False)

    assert "attestation must contain exactly" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_an_unpinned_input_mass_evidence(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    payload["evidence_sha256"]["uk_input_mass_parity"] = "b" * 64
    _rewrite_battery_report(directory, payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = dict(payload["evidence_sha256"])
    build_path.write_text(json.dumps(build))
    _refresh_terminal_manifest_hashes(directory)

    assert "bind the reviewed enhanced-FRS" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_mixed_manifest_evidence_vocabulary(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = {
        "release_dataset": "a" * 64,
        **dict(payload["evidence_sha256"]),
    }
    build_path.write_text(json.dumps(build))

    failures = _battery_failures(directory)
    assert "mixes the legacy stage vocabulary" in failures


def test_exact_k_uk_gate_battery_rejects_a_diagnostic_relabel(tmp_path: Path) -> None:
    # Every entry in this vintage is release_blocking; a failed entry
    # relabeled diagnostic would dodge the shippability recompute.
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_weight_ratio"]
    entry["status"] = "failed"
    entry["failures"] = ["seeded ratio failure"]
    entry["criticality"] = "diagnostic"
    _rewrite_battery_report(directory, payload)

    failures = _battery_failures(directory)
    assert "criticality must be 'release_blocking'" in failures


def test_exact_k_uk_gate_battery_rejects_a_not_applicable_excuse(
    tmp_path: Path,
) -> None:
    # No entry in this spec vintage declares an excuse, so not_applicable
    # cannot skip a gate's observables or drop its evidence requirement.
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_qrf_tail_concentration"]
    entry["status"] = "not_applicable"
    entry["details"] = {}
    entry["reason"] = "declared excused"
    _rewrite_battery_report(directory, payload)

    assert "no entry in this spec vintage declares an excuse" in _battery_failures(
        directory
    )


def test_exact_k_uk_gate_battery_pins_each_entrys_gate_and_phase(
    tmp_path: Path,
) -> None:
    directory, payload = _write_battery_release(tmp_path)
    entry = payload["gates"]["uk_weight_ratio"]
    entry["gate"] = "export_surface"
    entry["phase"] = "preflight"
    _rewrite_battery_report(directory, payload)

    failures = _battery_failures(directory)
    assert "gate must be 'weight_ratio'" in failures
    assert "phase must be 'terminal'" in failures


def test_exact_k_uk_gate_battery_rejects_a_float_schema_version(
    tmp_path: Path,
) -> None:
    # 4.0 == 4 in Python; the dispatch and the checker must not let a
    # non-integer vintage route as one.
    directory, payload = _write_battery_release(tmp_path)
    payload["schema_version"] = 4.0
    _rewrite_battery_report(directory, payload, resign=False)

    assert "schema_version must be 3" in _battery_failures(directory)


def test_exact_k_uk_gate_battery_rejects_an_overridden_exclusion_register(
    tmp_path: Path,
) -> None:
    # The committed register is the policy of record: an overridden
    # register's evidence digest is never releasable, however honestly
    # the report self-describes it.
    directory, payload = _write_battery_release(tmp_path)
    payload["evidence_sha256"]["uk_degenerate_release_surface"] = "b" * 64
    _rewrite_battery_report(directory, payload)
    build_path = directory / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["terminal_gate_evidence"] = dict(payload["evidence_sha256"])
    build_path.write_text(json.dumps(build))
    _refresh_terminal_manifest_hashes(directory)

    assert "bind the committed exclusion register" in _battery_failures(directory)


# ---------------------------------------------------------------------------
# Evidence-tier release contract (microcosm#506)
# ---------------------------------------------------------------------------
#
# The evidence contract is a SIBLING of the certified one, not a relaxation:
# same required files, plus a mandatory non-empty known_failures block naming
# every recorded gate failure verbatim with an owner issue. The two tiers must
# never be confusable — an evidence manifest fails certified validation
# structurally (distinct schema marker), and a certified manifest fails
# evidence validation (no tier, no known_failures).

EVIDENCE_RELEASE_ID = "populace-us-2024-evidence-9f1260b-20260611"


def _known_failures() -> list[dict]:
    return [
        {
            "failure": (
                "SOI Table 1.4 national dollar fit failed: target "
                "'irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount"
                "@2024' has relative_error=-0.302, exceeding 0.25."
            ),
            "owner": "PolicyEngine/microcosm#487",
        },
        {
            "failure": (
                "QRF tail concentration failed: 7 sparse QRF-imputed columns "
                "concentrate past the top-k weighted-mass share bound."
            ),
            "owner": "PolicyEngine/microcosm#481",
        },
    ]


def _evidence_release_manifest(
    *,
    diagnostics_sha: str,
    source_coverage_sha: str,
    known_failures: list[dict] | None = None,
) -> dict:
    manifest = _release_manifest(
        EVIDENCE_RELEASE_ID,
        diagnostics_sha=diagnostics_sha,
        source_coverage_sha=source_coverage_sha,
    )
    manifest["schema_version"] = EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION
    manifest["tier"] = "evidence"
    manifest["known_failures"] = (
        _known_failures() if known_failures is None else known_failures
    )
    return manifest


def _evidence_build_manifest() -> dict:
    """The build manifest an --evidence-release run writes: the battery
    verdict records the same failure strings known_failures carries."""
    manifest = _build_manifest(EVIDENCE_RELEASE_ID)
    manifest["gates"]["calibration"] = {
        "passed": False,
        "failures": [entry["failure"] for entry in _known_failures()],
    }
    return manifest


def _evidence_calibration_diagnostics() -> dict:
    """Diagnostics as the evidence builder writes them: the merged terminal
    record rides in build.release_gates."""
    diagnostics = _calibration_diagnostics()
    diagnostics["build"] = {
        "release_gates": {
            "passed": False,
            "failures": [entry["failure"] for entry in _known_failures()],
        }
    }
    return diagnostics


@pytest.fixture
def evidence_release_dir(tmp_path: Path) -> Path:
    """A complete, evidence-contract-valid release directory."""
    directory = tmp_path / "releases" / EVIDENCE_RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(
        json.dumps(_evidence_build_manifest())
    )
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_evidence_calibration_diagnostics())
    )
    (directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps(_source_coverage_diagnostics())
    )
    (directory / "release_manifest.json").write_text(
        json.dumps(
            _evidence_release_manifest(
                diagnostics_sha=_sha256(directory / "calibration_diagnostics.json"),
                source_coverage_sha=_sha256(
                    directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE
                ),
            )
        )
    )
    return directory


def _rewrite_evidence_manifest(evidence_release_dir: Path, mutate) -> None:
    manifest_path = evidence_release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))


def test_a_complete_evidence_release_passes(evidence_release_dir: Path) -> None:
    validate_evidence_release_dir(evidence_release_dir)


def test_evidence_release_fails_the_certified_contract(
    evidence_release_dir: Path,
) -> None:
    """The structural guarantee: whatever gates failed, an evidence manifest
    can never certify — its schema marker alone refuses the certified tier."""
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert f"'schema_version' is {EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION!r}" in (
        failures
    )


def test_certified_release_fails_the_evidence_contract(release_dir: Path) -> None:
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "known_failures" in failures
    assert "tier" in failures
    assert EVIDENCE_RELEASE_ID_SEGMENT in failures


def test_evidence_release_requires_the_id_segment(tmp_path: Path) -> None:
    """An evidence-shaped manifest under a certified-shaped id is refused: the
    tier must be visible in the release id itself."""
    directory = tmp_path / "releases" / RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(json.dumps(_build_manifest()))
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics())
    )
    (directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps(_source_coverage_diagnostics())
    )
    manifest = _release_manifest(
        diagnostics_sha=_sha256(directory / "calibration_diagnostics.json"),
        source_coverage_sha=_sha256(directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE),
    )
    manifest["schema_version"] = EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION
    manifest["tier"] = "evidence"
    manifest["known_failures"] = _known_failures()
    (directory / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(directory)
    failures = "\n".join(excinfo.value.failures)
    assert EVIDENCE_RELEASE_ID_SEGMENT in failures


def test_evidence_release_refuses_uk_exact_k_ids(tmp_path: Path) -> None:
    """UK exact-k verdicts belong to the gate-battery lane (microcosm#611);
    the evidence tier is scoped to the US national artifact."""
    directory = tmp_path / "releases" / "populace-uk-2023-evidence-frs-k535080"
    directory.mkdir(parents=True)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(directory)
    assert "no evidence-tier contract" in "\n".join(excinfo.value.failures)


def test_evidence_release_refuses_non_default_roles(
    evidence_release_dir: Path,
) -> None:
    """Local-area releases have their own contract (microcosm#398) and no
    adjudicated evidence-tier semantics."""
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(dataset_role="non_default_local_area"),
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "evidence tier supports only" in failures


def test_evidence_release_rejects_empty_known_failures(
    evidence_release_dir: Path,
) -> None:
    """The tier exists to carry failures honestly; an empty block is invalid."""
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(known_failures=[]),
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "non-empty 'known_failures'" in failures


def test_evidence_release_rejects_missing_known_failures(
    evidence_release_dir: Path,
) -> None:
    def _drop(manifest: dict) -> None:
        del manifest["known_failures"]

    _rewrite_evidence_manifest(evidence_release_dir, _drop)
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    assert "non-empty 'known_failures'" in "\n".join(excinfo.value.failures)


def test_evidence_known_failures_require_verbatim_failure_text(
    evidence_release_dir: Path,
) -> None:
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            known_failures=[{"failure": "", "owner": "PolicyEngine/microcosm#487"}]
        ),
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "known_failures[0].failure" in failures


def test_evidence_known_failures_require_an_owner_issue_ref(
    evidence_release_dir: Path,
) -> None:
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            known_failures=[
                {"failure": "QRF tail concentration failed: ...", "owner": "Max"}
            ]
        ),
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "known_failures[0].owner" in failures
    assert "issue reference" in failures


def test_evidence_known_failures_accept_issue_url_owners(
    evidence_release_dir: Path,
) -> None:
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            known_failures=[
                {
                    "failure": entry["failure"],
                    "owner": "https://github.com/PolicyEngine/microcosm/issues/481",
                }
                for entry in _known_failures()
            ]
        ),
    )
    validate_evidence_release_dir(evidence_release_dir)


def test_evidence_release_requires_the_evidence_tier_field(
    evidence_release_dir: Path,
) -> None:
    def _drop(manifest: dict) -> None:
        del manifest["tier"]

    _rewrite_evidence_manifest(evidence_release_dir, _drop)
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    assert "'tier'" in "\n".join(excinfo.value.failures)


def test_evidence_release_rejects_certified_schema_version(
    evidence_release_dir: Path,
) -> None:
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            schema_version=RELEASE_MANIFEST_SCHEMA_VERSION
        ),
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert f"{EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION!r}" in failures


BREACHED_CRITICAL_TARGET = (
    "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount@2024"
)


def _breach_critical_target(
    evidence_release_dir: Path, *, name: str = BREACHED_CRITICAL_TARGET
) -> None:
    """Push a critical row far past its blocking tolerance (the same breach
    the certified suite uses to prove a hard refusal), keeping the fixture's
    recorded build.release_gates block intact."""
    diagnostics = json.loads(
        (evidence_release_dir / "calibration_diagnostics.json").read_text()
    )
    target = next(row for row in diagnostics["targets"] if row["name"] == name)
    target["final_estimate"] = target["target"] * 0.35
    target["relative_error"] = -0.65
    _write_json_and_refresh_manifest_hash(
        evidence_release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )


def test_evidence_release_requires_breaches_to_be_acknowledged(
    evidence_release_dir: Path,
) -> None:
    """A critical breach the known_failures record does not name is refused:
    the tier records failures, it never hides them."""
    _breach_critical_target(evidence_release_dir)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "must acknowledge the critical-target breach" in failures
    assert BREACHED_CRITICAL_TARGET in failures


def test_evidence_release_tolerates_acknowledged_critical_breaches(
    evidence_release_dir: Path,
) -> None:
    """The tier's point: the Build N-class breach ships as evidence once
    known_failures names it, instead of blocking the artifact — while the
    certified contract still refuses the same directory."""
    _breach_critical_target(evidence_release_dir)
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            known_failures=[
                *manifest["known_failures"],
                {
                    "failure": (
                        "Fiscal critical target "
                        f"'{BREACHED_CRITICAL_TARGET}' breached its blocking "
                        "tolerance with relative_error=-0.6508."
                    ),
                    "owner": "PolicyEngine/microcosm#487",
                },
            ]
        ),
    )

    validate_evidence_release_dir(evidence_release_dir)

    with pytest.raises(ReleaseContractError):
        validate_release_dir(evidence_release_dir)


def test_evidence_release_requires_recorded_gate_failures_verbatim(
    evidence_release_dir: Path,
) -> None:
    """Every failure the build manifest records must ride into known_failures
    unmodified — a softened or dropped copy is refused."""
    recorded = (
        "SOI Table 1.4 national dollar fit failed: target "
        "'irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount@2024' "
        "has relative_error=-0.302, exceeding 0.25."
    )
    build_manifest = _build_manifest(EVIDENCE_RELEASE_ID)
    build_manifest["gates"]["calibration"] = {"passed": False, "failures": [recorded]}
    (evidence_release_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest)
    )

    # The default fixture's first entry IS that verbatim string, so the
    # binding holds as-is.
    validate_evidence_release_dir(evidence_release_dir)

    # Softening one character of the recorded string breaks the binding.
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            known_failures=[
                {
                    "failure": recorded.replace("-0.302", "-0.03"),
                    "owner": "PolicyEngine/microcosm#487",
                }
            ]
        ),
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "gates.calibration.failures entry verbatim" in failures


def test_evidence_release_scope_requires_the_us_prefix(tmp_path: Path) -> None:
    """A generic id with the segment must not buy a weaker contract by
    deactivating the US-specific requirements."""
    directory = tmp_path / "releases" / "acme-evidence-build"
    directory.mkdir(parents=True)

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(directory)
    assert "out of scope" in "\n".join(excinfo.value.failures)


def test_evidence_release_binds_failed_source_coverage_gate(
    evidence_release_dir: Path,
) -> None:
    """A failed coverage gate no longer blocks the evidence tier, but each of
    its recorded failures must ride into known_failures (the builder records
    them with a gate-family prefix) — unacknowledged, the release refuses."""
    payload = _source_coverage_diagnostics()
    payload["gate"] = {
        "name": "us_source_coverage",
        "passed": False,
        "failures": ["social_security_ssi/ssa-ssi-table-7b1-2024 missing"],
    }
    _write_json_and_refresh_manifest_hash(
        evidence_release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=payload,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    assert "gate.failures entry within an entry" in "\n".join(excinfo.value.failures)

    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            known_failures=[
                *manifest["known_failures"],
                {
                    "failure": (
                        "Source coverage failed: "
                        "social_security_ssi/ssa-ssi-table-7b1-2024 missing"
                    ),
                    "owner": "PolicyEngine/microcosm#470",
                },
            ]
        ),
    )
    validate_evidence_release_dir(evidence_release_dir)


def test_evidence_release_requires_the_battery_record_to_exist(
    evidence_release_dir: Path,
) -> None:
    """Deleting a locally checkable record must fail CLOSED, not silently
    disable its binding (sol round-2 finding)."""
    build_manifest = _evidence_build_manifest()
    del build_manifest["gates"]["calibration"]
    (evidence_release_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest)
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "gates.calibration.failures must be a list of strings" in failures


def test_evidence_release_requires_the_terminal_record_to_exist(
    evidence_release_dir: Path,
) -> None:
    diagnostics = _evidence_calibration_diagnostics()
    del diagnostics["build"]
    _write_json_and_refresh_manifest_hash(
        evidence_release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "build.release_gates.failures must be a list of strings" in failures


def test_evidence_breach_acknowledgment_rejects_substring_collisions(
    evidence_release_dir: Path,
) -> None:
    """An entry naming actc_amount must not satisfy a ctc_amount breach: the
    acknowledgment match is name-delimited, not substring (sol round-2)."""
    ctc_name = "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    actc_name = "irs_soi.ty2022.historic_table_2.us.all.actc_amount@2024"
    _breach_critical_target(evidence_release_dir, name=ctc_name)
    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            known_failures=[
                *manifest["known_failures"],
                {
                    "failure": f"Fiscal critical target '{actc_name}' drifted.",
                    "owner": "PolicyEngine/microcosm#487",
                },
            ]
        ),
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "must acknowledge the critical-target breach" in failures

    _rewrite_evidence_manifest(
        evidence_release_dir,
        lambda manifest: manifest.update(
            known_failures=[
                *manifest["known_failures"],
                {
                    "failure": f"Fiscal critical target '{ctc_name}' breached.",
                    "owner": "PolicyEngine/microcosm#487",
                },
            ]
        ),
    )
    validate_evidence_release_dir(evidence_release_dir)


def test_evidence_release_still_enforces_required_files(
    evidence_release_dir: Path,
) -> None:
    (evidence_release_dir / "calibration_diagnostics.json").unlink()
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "required file 'calibration_diagnostics.json' is missing." in failures


def test_evidence_release_still_enforces_artifact_hashes(
    evidence_release_dir: Path,
) -> None:
    (evidence_release_dir / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics() | {"options": {"epochs": 121}})
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "declares sha256" in failures


def test_evidence_release_still_enforces_build_id_match(
    evidence_release_dir: Path,
) -> None:
    build_manifest = _build_manifest(EVIDENCE_RELEASE_ID)
    build_manifest["build_id"] = "populace-us-2024-evidence-other-20260611"
    (evidence_release_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest)
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    assert "the directory name IS the" in "\n".join(excinfo.value.failures)


def test_evidence_release_still_enforces_dirty_git_refusal(
    evidence_release_dir: Path,
) -> None:
    """Evidence tier relaxes gate verdicts, never provenance."""
    build_manifest = _build_manifest(EVIDENCE_RELEASE_ID)
    build_manifest["code"]["git_dirty"] = True
    (evidence_release_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest)
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_evidence_release_dir(evidence_release_dir)
    assert "'code.git_dirty' must be false" in "\n".join(excinfo.value.failures)


def test_breach_acknowledgment_matching_is_name_delimited() -> None:
    from microcosm.data import contract as contract_module

    assert not contract_module._token_appears_delimited(
        "ctc_amount@2024", "an entry about actc_amount@2024 only"
    )
    assert contract_module._token_appears_delimited(
        "ctc_amount@2024", "the ctc_amount@2024 row breached"
    )
    # A dotted-name suffix is not the name.
    assert not contract_module._token_appears_delimited(
        "b.c@2024", "this names a.b.c@2024"
    )


# --- UK release certification (microcosm#757 B5) ---------------------------


def _green_uk_certification(
    key: bytes,
    *,
    release_id: str = "uk-757-first-certified-cut",
    diagnostics_sha256: str = "c" * 64,
    part_shas: dict[str, str] | None = None,
    score_receipt_sha256: str = "d" * 64,
) -> dict:
    parts = {}
    for part_name, scope in contract._UK_CERTIFICATION_PART_SCOPES.items():
        parts[part_name] = {
            "path": f"{part_name}.json",
            "sha256": (part_shas or {}).get(part_name, "a" * 64),
            "release_id": release_id,
            "phases": list(contract._UK_CERTIFICATION_PART_PHASES[part_name]),
            "entry_ids": sorted(scope),
            "gates_manifest_sha256": contract._UK_CERTIFICATION_PART_DIGESTS[part_name][
                "gates_manifest_sha256"
            ],
            "policy_sha256": contract._UK_CERTIFICATION_PART_DIGESTS[part_name][
                "policy_sha256"
            ],
            "statuses": {"passed": len(scope)},
        }
    certification = {
        "schema_version": 1,
        "kind": "uk_release_certification",
        "country": "uk",
        "release_id": release_id,
        "candidate": {
            "name": "microcosm_uk_2024",
            "filename": "microcosm_uk_2024.h5",
            "sha256": "b" * 64,
            "size_bytes": 1,
        },
        "parts": parts,
        "spec": {
            "gates_manifest_sha256": contract._UK_GATE_BATTERY_GATES_MANIFEST_SHA256,
            "policy_sha256": contract._UK_GATE_BATTERY_POLICY_SHA256,
            "spec_fingerprint": contract._UK_GATE_BATTERY_SPEC_FINGERPRINT,
            "declared_entry_count": len(contract._UK_GATE_BATTERY_ENTRY_IDS),
            "declared_phases": list(contract._UK_GATE_BATTERY_PHASES),
            "shared_gate_ids": sorted(contract._UK_CERTIFICATION_SHARED_GATE_IDS),
            "certification_excluded_gate_ids": sorted(
                contract._UK_CERTIFICATION_EXCLUDED_GATE_IDS
            ),
        },
        "doctrine": {"payload": {"epochs": 1500}, "overrides": {}},
        "diagnostics_sha256": diagnostics_sha256,
        "score_receipt": {
            "filename": "score_vs_enhanced_frs.json",
            "sha256": score_receipt_sha256,
        },
        "exclusions_evaluated_on": "2026-08-27",
        "shippable": True,
    }
    attestation = {
        "producer": "microcosm.build.uk_runtime.release_certification",
        "signature_algorithm": "hmac-sha256",
        "signing_key_sha256": hashlib.sha256(key).hexdigest(),
        "signature": None,
    }
    certification["attestation"] = attestation
    attestation["signature"] = hmac.new(
        key, contract._canonical_json_bytes(certification), hashlib.sha256
    ).hexdigest()
    return certification


def _certification_failures(certification, monkeypatch, key: bytes) -> list[str]:
    monkeypatch.setenv(
        UK_GATE_BATTERY_SIGNING_KEY_ENV, base64.b64encode(key).decode("ascii")
    )
    failures: list[str] = []
    contract._check_uk_release_certification(
        certification,
        release_id="uk-757-first-certified-cut",
        calibration_diagnostics_sha256="c" * 64,
        failures=failures,
    )
    return failures


def test_uk_release_certification_green(monkeypatch) -> None:
    key = bytes(range(32))
    certification = _green_uk_certification(key)
    assert _certification_failures(certification, monkeypatch, key) == []


def test_uk_release_certification_refusals(monkeypatch) -> None:
    key = bytes(range(32))

    certification = _green_uk_certification(key)
    certification["shippable"] = False
    assert any(
        "does not certify a shippable candidate" in line
        for line in _certification_failures(certification, monkeypatch, key)
    )

    certification = _green_uk_certification(key)
    certification["parts"]["release_cut"]["statuses"] = {"passed": 15, "failed": 1}
    failures = _certification_failures(certification, monkeypatch, key)
    assert any("statuses" in line for line in failures)
    assert any("does not certify a shippable" in line for line in failures)

    certification = _green_uk_certification(key)
    certification["parts"]["spine"]["entry_ids"] = sorted(
        set(certification["parts"]["spine"]["entry_ids"]) - {"uk_brma_enum_domain"}
    )
    assert any(
        "entry_ids" in line
        for line in _certification_failures(certification, monkeypatch, key)
    )

    certification = _green_uk_certification(key)
    certification["parts"]["calibration_seam"]["gates_manifest_sha256"] = "e" * 64
    assert any(
        "scoped manifest digest" in line
        for line in _certification_failures(certification, monkeypatch, key)
    )

    certification = _green_uk_certification(key)
    certification["diagnostics_sha256"] = "f" * 64
    assert any(
        "diagnostics_sha256" in line
        for line in _certification_failures(certification, monkeypatch, key)
    )

    # A tampered field breaks the signature: the flag flip is caught both as
    # a verdict refusal and as a signature failure.
    certification = _green_uk_certification(key)
    certification["release_id"] = "uk-757-first-certified-cut"
    certification["doctrine"] = {"payload": {}, "overrides": {}}
    assert any(
        "signature does not authenticate" in line
        for line in _certification_failures(certification, monkeypatch, key)
    )

    certification = _green_uk_certification(key)
    del certification["score_receipt"]
    assert any(
        "exactly the certification fields" in line
        for line in _certification_failures(certification, monkeypatch, key)
    )


def test_national_line_artifacts_require_the_certification(tmp_path) -> None:
    # A release that ships any national-line gate part without the composed
    # certification must refuse: the shippability verdict lives only in the
    # certification, so its omission cannot validate clean (green-by-absence).
    release_dir = tmp_path / "uk-757-first-certified-cut"
    release_dir.mkdir()
    (release_dir / "release_cut_gates.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ReleaseContractError) as caught:
        validate_release_dir(release_dir)
    assert any(
        "release_certification.json is missing while national-line" in line
        for line in caught.value.failures
    )

    # A calibration-seam-scoped terminal report is a national-line part too.
    seam_dir = tmp_path / "uk-757-seam-only"
    seam_dir.mkdir()
    (seam_dir / "terminal_gates.json").write_text(
        '{"posture": "calibration_seam"}', encoding="utf-8"
    )
    with pytest.raises(ReleaseContractError) as caught:
        validate_release_dir(seam_dir)
    assert any(
        "release_certification.json is missing while national-line" in line
        for line in caught.value.failures
    )

    # With the certification present the omission failure clears (the file's
    # own validation and the base required-files failures still apply).
    (release_dir / "release_certification.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReleaseContractError) as caught:
        validate_release_dir(release_dir)
    assert not any(
        "is missing while national-line" in line for line in caught.value.failures
    )
    assert any(
        "must carry exactly the certification fields" in line
        for line in caught.value.failures
    )


def test_national_release_id_requires_the_certification() -> None:
    from microcosm.data.contract import required_release_files

    required = required_release_files("microcosm-uk-2024-25-national")
    assert "release_certification.json" in required
    assert "release_certification.json" not in required_release_files(
        "dev-757-rebind-proof"
    )


def test_a_release_built_from_a_denied_pool_cannot_be_published(monkeypatch) -> None:
    """The publisher never loads an H5, so it must consult the deny-list itself."""
    from microcosm.data import contract as contract_module
    from microcosm.data import denied_pools

    denied = denied_pools.DeniedPoolPublication(
        manifest_sha256="0" * 64,
        pool_h5_sha256="1" * 64,
        content_identity_sha256="2" * 64,
        release_id="fixture-release",
        reason="fixture pool is excluded",
        reference="microcosm#856; fixture-plan-gate",
    )
    monkeypatch.setattr(
        denied_pools, "DENIED_POOL_PUBLICATIONS", {"denied-run": denied}, raising=False
    )
    for identity in (
        {"publication_run_id": "denied-run"},
        {"publication_run_id": "other", "manifest_sha256": "0" * 64},
        {"publication_run_id": "other", "pool_h5_sha256": "1" * 64},
        {"publication_run_id": "other", "content_identity_sha256": "2" * 64},
    ):
        manifest = _build_manifest()
        manifest["base_pool"] = {"status": "gate_failed", **identity}
        failures: list[str] = []
        contract_module._check_build_manifest(manifest, RELEASE_ID, failures)
        assert any("denied publication 'denied-run'" in f for f in failures), identity
    manifest = _build_manifest()
    manifest["base_pool"] = {"status": "simulation_ready", "publication_run_id": "fine"}
    failures = []
    contract_module._check_build_manifest(manifest, RELEASE_ID, failures)
    assert not any("denied" in f for f in failures)
