from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.logbook import load_spool_rows
from microcosm.build.source_manifest import SourceManifest, SourceStageSpec
from microcosm.build.staging_v2 import validate_v2_bundle
from microcosm.build.uk_runtime import (
    frs_disability,
    frs_education_grants,
    frs_legacy_proxies,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.frs_spine import (
    FRS_SPINE_TABLES,
    REGION_MAP,
    UC_CAPITAL_UNAVAILABLE,
    WEEKS_IN_YEAR,
    UKFRSSpineStageTransform,
    build_uk_frs_spine_frame,
    scottish_water_and_sewerage_weekly,
    uk_frs_spine_seed_frame,
)
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame, WeightKind, engine_tables

_TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "build_uk_frs_spine.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_uk_frs_spine", _TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tab(root: Path, table: str, rows: list[dict[str, object]]) -> None:
    path = root / f"{table}.tab"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _fixture_tables() -> dict[str, list[dict[str, object]]]:
    household_2 = {
        "SERNUM": 2,
        "GROSS4": 20.0,
        "GVTREGNO": 1,
        "PTENTYP2": 5,
        "TYPEACC": 1,
        "BEDROOM6": 3,
        "CTANNUAL": 1000.0,
        "CTBAND": 4,
        "CTREBAMT": 2.0,
        "ADULTH": 1,
        # CWATAMT/CSEWAMT are retired in FRS 2024-25: the headers survive but
        # carry no data at all, so the fixture leaves them blank exactly as the
        # real tab does. CWATAMT1/CSEWAMT1 are their Scotland-only successors.
        "CSEWAMT": "",
        "CWATAMTD": 0.0,
        "CWATAMT1": "",
        "CSEWAMT1": "",
        "WATSEWRT": 3.0,
        "NIRATLIA": 4.0,
        "RT2REBAM": 0.0,
        "HHRENT": 5.0,
        "SUBRENT": 6.0,
        "TENTYP2": 5,
        "MORTINT": 7.0,
        "STRUINS": 8.0,
        **{f"CHRGAMT{i}": float(i) for i in range(1, 10)},
    }
    household_1 = {
        **household_2,
        "SERNUM": 1,
        "GROSS4": 10.0,
        "GVTREGNO": 12,
        "PTENTYP2": 6,
        "TYPEACC": 4,
        "BEDROOM6": 2,
        "CTANNUAL": -1.0,
        "CTBAND": 2,
        "CTREBAMT": 1.0,
        "CSEWAMT": "",
        "CWATAMTD": 3.0,
        "CWATAMT1": 4.0,
        "CSEWAMT1": 5.0,
        "WATSEWRT": 99.0,
        "NIRATLIA": -1.0,
        "RT2REBAM": 5.0,
        "HHRENT": 6.0,
        "SUBRENT": 0.0,
        "MORTINT": 8.0,
        "STRUINS": 9.0,
    }
    adult_1 = {
        "SERNUM": 1,
        "BENUNIT": 1,
        "PERSON": 1,
        "AGE80": 0,
        "AGE": 40,
        "SEX": 1,
        "TOTHOURS": 40,
        "HRPID": 1,
        "UPERSON": 1,
        "MARITAL": 1,
        "EMPSTATI": 5,
        "MJOBSECT": 1,
        "SIC": 84,
        "FTED": 2,
        "TYPEED2": 0,
        "EDUCQUAL": 17,
        "TRAIN": 10,
        "EMAAMT": 0.0,
        "CHEMAAMT": 0.0,
        "INEARNS": 10.0,
        "SEINCAM2": 3.0,
        "MNTUS1": 2,
        "MNTUSAM1": 1.0,
        "MNTAMT1": 9.0,
        "MNTAMT2": 2.0,
        "CVPAY": 1.0,
        "ROYYR1": 2.0,
        "ROYYR2": 3.0,
        "ROYYR3": 4.0,
        "ROYYR4": 5.0,
        "ALLPAY2": 6.0,
        "ALLPAY3": 7.0,
        "ALLPAY4": 8.0,
        "CHAMTERN": 9.0,
        "CHAMTTST": 10.0,
        "APAMT": 11.0,
        "APDAMT": 12.0,
        "PAREAMT": 13.0,
        "REDAMT": 100.0,
        "SLREPAMT": 2.0,
        "SSPADJ": 1.0,
        "SMPADJ": 0.5,
        "TUBORR": 500.0,
        "ACCSSAMT": 1.0,
        "GRTDIR1": 2.0,
        "GRTDIR2": 3.0,
        # heartval is on the adult tape too; the three school columns are not.
        "HEARTVAL": 5.0,
    }
    adult_2 = {**adult_1, "SERNUM": 2, "PERSON": 1, "SEX": 2, "HRPID": 1}
    child_1 = {
        "SERNUM": 1,
        "BENUNIT": 1,
        "PERSON": 2,
        "AGE80": 0,
        "AGE": 8,
        "SEX": 2,
        "TOTHOURS": np.nan,
        "HRPID": 0,
        "UPERSON": 0,
        "MARITAL": 2,
        "FTED": 1,
        "TYPEED2": 2,
        "EDUCQUAL": 86,
        "TRAIN": 9,
        "EMAAMT": 0.0,
        "CHEMAAMT": 1.0,
        "FSMVAL": 3.0,
        "FSFVVAL": 1.0,
        "FSBVAL": 2.0,
        "HEARTVAL": 4.0,
    }
    return {
        "adult": [adult_2, adult_1],
        "child": [child_1],
        "benunit": [
            {
                "SERNUM": 2,
                "BENUNIT": 1,
                "FAMTYPB2": 5,
                "DEPCHLDB": 0,
                "TOTCAPB4": 222.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "FAMTYPB2": 7,
                "DEPCHLDB": 1,
                "TOTCAPB4": 111.0,
            },
        ],
        "househol": [household_2, household_1],
        "pension": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "PENPAY": 10.0,
                "PTAMT": 2.0,
                "PTINC": 2,
                "POAMT": 3.0,
                "POINC": 2,
                "PENOTH": 0,
            }
        ],
        "oddjob": [{"SERNUM": 1, "BENUNIT": 1, "PERSON": 1, "OJAMT": 4.0, "OJNOW": 1}],
        "accounts": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "ACCOUNT": 21,
                "ACCINT": 1.0,
                "ACCTAX": 0,
                "INVTAX": 0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "ACCOUNT": 1,
                "ACCINT": 2.0,
                "ACCTAX": 1,
                "INVTAX": 0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "ACCOUNT": 7,
                "ACCINT": 3.0,
                "ACCTAX": 0,
                "INVTAX": 0,
            },
        ],
        "job": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "DEDUC1": 2.0,
                "SPNAMT": 3.0,
                "SALSAC": "1",
            }
        ],
        "benefits": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 14,
                "VAR2": 1,
                "BENAMT": 2.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 14,
                "VAR2": 2,
                "BENAMT": 3.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 16,
                "VAR2": 3,
                "BENAMT": 4.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 16,
                "VAR2": 4,
                "BENAMT": 5.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 6,
                "VAR2": 0,
                "BENAMT": 6.0,
            },
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "BENEFIT": 3,
                "VAR2": 0,
                "BENAMT": 7.0,
            },
        ],
        "maint": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "MRUS": 2,
                "MRUAMT": 2.0,
                "MRAMT": 9.0,
            }
        ],
        "penprov": [
            {"SERNUM": 1, "BENUNIT": 1, "PERSON": 1, "STEMPPEN": 5, "PENAMT": 4.0},
            {"SERNUM": 1, "BENUNIT": 1, "PERSON": 1, "STEMPPEN": 6, "PENAMT": 100.0},
        ],
        "chldcare": [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "CHAMT": 5.0,
                "COST": 1,
                "REGISTRD": 1,
            }
        ],
        "extchild": [{"SERNUM": 1, "BENUNIT": 1, "NHHAMT": 2.0}],
        "mortgage": [
            {"SERNUM": 1, "RMORT": 1, "RMAMT": 120.0, "BORRAMT": 240.0, "MORTEND": 12.0}
        ],
    }


def _write_fixture(
    root: Path, tables: dict[str, list[dict[str, object]]] | None = None
) -> SourceStageSpec:
    root.mkdir(exist_ok=True)
    for table, rows in (tables or _fixture_tables()).items():
        _write_tab(root, table, rows)
    artifacts = []
    for table in FRS_SPINE_TABLES:
        path = root / f"{table}.tab"
        artifacts.append(
            {
                "role": "frs_table",
                "table": table,
                "kind": "licensed_microdata",
                "format": "tab",
                "vintage": "2024_25",
                "locator": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "runtime_sha256_required": True,
            }
        )
    return SourceStageSpec.from_mapping(
        {
            "stage": "frs_spine",
            "survey": "Synthetic FRS",
            "source": "local fabricated rows",
            "grain": "household",
            "artifacts": artifacts,
            "operations": [{"kind": "read_tables"}],
            "outputs": ["employment_income"],
        }
    )


def _manifest_stage() -> SourceStageSpec:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()["frs_spine"]


def _synthetic_spec(stage: SourceStageSpec) -> SimpleNamespace:
    def source_stage(
        name: str,
        *,
        tables: tuple[str, ...] = (),
        outputs: tuple[str, ...],
        operations: list[dict[str, object]] | None = None,
        nonnegative_outputs: tuple[str, ...] = (),
        rewrites: tuple[str, ...] = (),
        grain: str = "person",
        extra_artifacts: tuple[dict[str, object], ...] = (),
    ) -> SourceStageSpec:
        artifacts = [
            artifact for artifact in stage.artifacts if artifact.get("table") in tables
        ]
        artifacts.extend(extra_artifacts)
        payload = {
            "stage": name,
            "survey": "Synthetic FRS",
            "source": "local fabricated rows",
            "grain": grain,
            "artifacts": artifacts,
            "operations": operations or [{"kind": "derive"}],
            "outputs": list(outputs),
            "nonnegative_outputs": list(nonnegative_outputs),
        }
        if rewrites:
            payload["rewrites"] = list(rewrites)
        return SourceStageSpec.from_mapping(payload)

    return SimpleNamespace(
        country="uk",
        sources=SourceManifest(
            country="uk",
            version=1,
            policy="Synthetic FRS spine spec.",
            stages=(
                stage,
                source_stage(
                    "frs_employment",
                    tables=("adult",),
                    operations=[{"kind": "read_tables"}, {"kind": "map_coded_amounts"}],
                    outputs=(
                        "employment_status",
                        "employment_sector",
                        "sic_industry_division",
                    ),
                    nonnegative_outputs=("sic_industry_division",),
                ),
                source_stage(
                    "frs_council_tax",
                    tables=("househol",),
                    operations=[{"kind": "read_tables"}, {"kind": "impute_cell_means"}],
                    outputs=("council_tax",),
                    nonnegative_outputs=("council_tax",),
                ),
                source_stage(
                    "frs_disability",
                    operations=[
                        {
                            "kind": "derive",
                            "parameters": "disability category thresholds from the fiscal-converted gov.dwp tree at the survey year",
                            "year_rule": "survey_year",
                        },
                        {
                            "kind": "derive",
                            "parameters": "disability flags from the fiscal-converted gov.dwp tree at the survey year",
                            "year_rule": "survey_year",
                        },
                    ],
                    outputs=(
                        "aa_category",
                        "dla_sc_category",
                        "dla_m_category",
                        "pip_m_category",
                        "pip_dl_category",
                        "is_disabled_for_benefits",
                        "is_enhanced_disabled_for_benefits",
                        "is_severely_disabled_for_benefits",
                    ),
                ),
                source_stage(
                    "frs_education",
                    tables=("adult", "child"),
                    operations=[{"kind": "read_tables"}, {"kind": "derive"}],
                    outputs=(
                        "current_education",
                        "highest_education",
                        "is_in_non_advanced_education",
                        "is_in_approved_training",
                        "age_started_or_accepted_current_education_or_training",
                        "is_before_universal_credit_qualifying_young_person_terminal_date",
                        "adult_ema",
                        "child_ema",
                        "receives_benefits_in_own_right",
                    ),
                    nonnegative_outputs=(
                        "adult_ema",
                        "child_ema",
                        "age_started_or_accepted_current_education_or_training",
                    ),
                ),
                source_stage(
                    "frs_legacy_proxies",
                    tables=("adult",),
                    operations=[
                        {"kind": "read_tables"},
                        {
                            "kind": "materialize_rules_engine_predictors",
                            "predictors": ["state_pension_age"],
                        },
                        {"kind": "derive"},
                    ],
                    outputs=(
                        "legacy_jobseeker_proxy",
                        "esa_health_condition_proxy",
                        "esa_support_group_proxy",
                    ),
                ),
                source_stage(
                    "frs_education_grant_split",
                    operations=[
                        {
                            "kind": "materialize_rules_engine_predictors",
                            "predictors": [
                                "childcare_grant",
                                "parents_learning_allowance",
                                "adult_dependants_grant",
                            ],
                            "consumed_only": True,
                            "year_rule": "survey_year",
                        },
                        {
                            "kind": "materialize_rules_engine_predictors",
                            "predictors": [
                                "maintenance_loan_in_england_system",
                                "disabled_students_allowance_course_eligible",
                                "disabled_students_allowance_has_qualifying_condition",
                            ],
                            "consumed_only": True,
                            "year_rule": "calibration_year",
                        },
                        {
                            "kind": "derive",
                            "scope": "proportional split of aggregate education_grants and DSA residual capacity",
                            "parameters": "DSA maximum from gov.dfe.disabled_students_allowance.maximum at the calibration year",
                            "year_rule": "calibration_year",
                        },
                    ],
                    outputs=("disabled_students_allowance_eligible_expenses",),
                    rewrites=("education_grants",),
                    nonnegative_outputs=(
                        "disabled_students_allowance_eligible_expenses",
                    ),
                ),
                source_stage(
                    "frs_take_up",
                    grain="benunit",
                    operations=[
                        {"kind": "aggregate_person_to_benunit"},
                        {
                            "kind": "assign_binary_with_anchored_residual",
                            "output": "would_claim_child_benefit",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "child_benefit_opts_out",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_with_anchored_residual",
                            "output": "would_claim_pc",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_with_anchored_residual",
                            "output": "would_claim_uc",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "would_claim_tfc",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "would_claim_extended_childcare",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "would_claim_universal_childcare",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "would_claim_targeted_childcare",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_clipped_normal",
                            "output": "maximum_extended_childcare_hours_usage",
                            "seed": 0,
                        },
                    ],
                    outputs=(
                        "would_claim_child_benefit",
                        "child_benefit_opts_out",
                        "would_claim_pc",
                        "would_claim_uc",
                        "would_claim_tfc",
                        "would_claim_extended_childcare",
                        "would_claim_universal_childcare",
                        "would_claim_targeted_childcare",
                        "maximum_extended_childcare_hours_usage",
                    ),
                    nonnegative_outputs=("maximum_extended_childcare_hours_usage",),
                ),
                source_stage(
                    "frs_person_draws",
                    operations=[
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "would_claim_marriage_allowance",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_banded_rates",
                            "output": "would_claim_scp",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_uniform_draw",
                            "output": "attends_private_school_random_draw",
                            "seed": 0,
                        },
                    ],
                    outputs=(
                        "would_claim_marriage_allowance",
                        "would_claim_scp",
                        "attends_private_school_random_draw",
                    ),
                ),
                source_stage(
                    "frs_household_draws",
                    grain="household",
                    operations=[
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "household_owns_tv",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "would_evade_tv_licence_fee",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "main_residential_property_purchased_is_first_home",
                            "seed": 0,
                        },
                        {
                            "kind": "assign_binary_from_rate",
                            "output": "property_purchased",
                            "seed": 0,
                        },
                    ],
                    outputs=(
                        "household_owns_tv",
                        "would_evade_tv_licence_fee",
                        "main_residential_property_purchased_is_first_home",
                        "property_purchased",
                    ),
                ),
                source_stage(
                    "frs_brma",
                    grain="household",
                    operations=[
                        {"kind": "materialize_rules_engine_predictors"},
                        {
                            "kind": "sample_categorical_from_count_table",
                            "output": "brma",
                            "seed": 0,
                        },
                    ],
                    outputs=("brma",),
                    # Mirrors the real manifest's non-tab resource artifact so
                    # the driver's pin split is exercised without licensed data.
                    extra_artifacts=(
                        {
                            "role": "count_resource",
                            "resource": "brma_rent_counts.json",
                            "kind": "public_aggregated_counts",
                            "format": "json",
                        },
                    ),
                ),
                source_stage(
                    "frs_hmrc_spine_leaves",
                    tables=("adult", "benefits"),
                    operations=[
                        {"kind": "retain_adjudicated_frs_hmrc_leaves"},
                        {
                            "kind": "derive",
                            "output": "employer_pension_contributions",
                        },
                    ],
                    outputs=(
                        "hmrc_spi_pay",
                        "hmrc_spi_unemployment_benefit_income",
                        "hmrc_spi_incapacity_benefit_income",
                        "ossben_identifiable_subset",
                        "srp_regular_code5",
                        "employer_pension_contributions",
                    ),
                ),
                source_stage(
                    "spi_support_channel",
                    grain="household",
                    operations=[
                        {
                            "kind": "stack_zero_weight_donors",
                            "count": 10000,
                            "seed": 42,
                            "draw": "uniform_without_replacement",
                        },
                        {
                            "kind": "gate_zero_weight_strata",
                            "declarations": [
                                {
                                    "name": "e7_spi_synthetic_preclone",
                                    "selector": {"household_is_spi_synthetic": True},
                                    "maximum_zero_weight_rows": 10000,
                                    "reason": "synthetic driver fixture",
                                }
                            ],
                        },
                        {
                            "kind": "allocate_zero_weight_prior_mass",
                            "share": 0.5,
                            "strata": ["region"],
                        },
                    ],
                    outputs=(
                        "household_is_spi_synthetic",
                        "person_support_channel",
                        "person_support_clone_index",
                        "person_source_id",
                        "benunit_support_channel",
                        "benunit_support_clone_index",
                        "benunit_source_id",
                        "household_support_channel",
                        "household_support_clone_index",
                        "household_source_id",
                        "source_household_id",
                        "source_year",
                        "source_household_key",
                    ),
                ),
                source_stage(
                    "hmrc_spi_income_spine",
                    operations=[
                        {"kind": "verify_pinned_hmrc_source_pair"},
                        {"kind": "strict_read_private_table", "seed": 42},
                        {
                            "kind": "fit_weighted_qrf_stage1",
                            "seed": 42,
                            "sample_size": 100000,
                            "initialize_frs_channel_columns": {
                                "gift_aid": 0.0,
                                "charitable_investment_gifts": 0.0,
                            },
                        },
                        {"kind": "fit_weighted_qrf_stage2", "seed": 43},
                        {
                            "kind": "redraw_columns_from_fitted_qrf",
                            "fit": "stage1",
                            "columns": ["dividend_income"],
                            "rows": "base_support_channel",
                        },
                    ],
                    outputs=(
                        "other_investment_income",
                        "gift_aid",
                        "charitable_investment_gifts",
                        "hmrc_spi_employment_benefits",
                        "hmrc_spi_employment_expenses",
                        "hmrc_spi_other_social_security_income",
                        "hmrc_spi_taxable_termination_pay",
                        "hmrc_spi_miscellaneous_employment_income",
                        "hmrc_spi_other_income",
                        "hmrc_spi_state_pension_income",
                        "hmrc_spi_employed_income",
                        "hmrc_spi_total_earned_income",
                        "hmrc_spi_total_investment_income",
                        "hmrc_spi_assessable_income",
                    ),
                ),
            ),
        ),
        geography_spine=None,
        resource_hashes={"brma_rent_counts.json": "f" * 64},
    )


class _FakeUKEngine:
    country = "uk"

    def materialize(self, frame, variables, period):
        person_count = len(frame.table("person"))
        values = {}
        for variable in variables:
            if variable == "state_pension_age":
                values[variable] = np.full(person_count, 66.0)
            elif variable == "LHA_category":
                values[variable] = np.array(["A"] * len(frame.table("benunit")))
            else:
                values[variable] = np.zeros(person_count)
        return values


def test_manifest_stage_and_runtime_agree_on_artifacts_and_operations() -> None:
    stage = _manifest_stage()

    assert len(stage.artifacts) == 14
    assert [artifact["table"] for artifact in stage.artifacts] == sorted(
        FRS_SPINE_TABLES
    )
    assert {operation.kind for operation in stage.operations} == {
        "read_tables",
        "replace_sentinels",
        "assemble_group_entities",
        "map_columns",
        "map_coded_amounts",
        "annualize_periodic_amounts",
    }
    assert set(stage.outputs) == set(UKFRSSpineStageTransform.output_columns())


def test_full_roster_artifact_pins_are_coherent() -> None:
    # Regression armor for the #723 licensed-run finding: the WAS round-8 tab
    # was declared by two stages (was_wealth, lcfs_consumption) with different
    # locator strings, so `_artifact_pins()` refused the full-roster driver
    # run — a break only a licensed build could see. Collecting pins over the
    # complete manifest keeps the same-artifact-same-pin invariant CI-visible.
    tool = _load_tool()
    spec = load_country_spec("uk")
    stages = list(spec.sources.stage_map().values())
    pins = tool._artifact_pins(stages)
    assert pins, "full-roster pin collection must not be empty"


def test_builds_structural_frame_from_pinned_tabs(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)

    frame = build_uk_frs_spine_frame(tmp_path, stage=stage)

    validate_uk_national_frame(frame)
    assert uk_time_period(frame) == "2024"
    assert frame.weights_for("household").kind is WeightKind.DESIGN
    np.testing.assert_array_equal(
        frame.weights_for("household").values,
        np.array([10.0, 20.0]),
    )
    assert frame.table("household")["household_id"].tolist() == [1, 2]
    assert frame.table("person")["person_id"].tolist() == [1001, 1002, 2001]
    assert not frame.table("person").isna().any().any()
    assert not frame.table("benunit").isna().any().any()
    assert not frame.table("household").isna().any().any()


def test_root_stage_ignores_seed_frame_content(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    plan = country_stage_plan(
        load_country_spec("uk"),
        {"frs_spine": UKFRSSpineStageTransform(tmp_path, stage=stage)},
        stage_names=("frs_spine",),
    )

    frame, records = plan.run(uk_frs_spine_seed_frame())

    assert frame.table("household")["household_id"].tolist() == [1, 2]
    assert records[0].stage == "frs_spine"


def test_root_stage_reports_capital_sentinel_mapping_count(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    transform = UKFRSSpineStageTransform(tmp_path, stage=stage)

    transform(uk_frs_spine_seed_frame())

    assert transform.checkpoint_metadata()["evidence"]["frs_benunit_capital"] == {
        "unavailable_sentinel": UC_CAPITAL_UNAVAILABLE,
        "mapped_rows": 0,
    }


def test_benunit_capital_maps_unavailable_raw_values_to_named_sentinel(
    tmp_path: Path,
) -> None:
    tables = _fixture_tables()
    tables["benunit"][0]["TOTCAPB4"] = ""
    tables["benunit"][1]["TOTCAPB4"] = -7
    stage = _write_fixture(tmp_path, tables)
    transform = UKFRSSpineStageTransform(tmp_path, stage=stage)

    frame = transform(uk_frs_spine_seed_frame())

    assert frame.table("benunit")["frs_benunit_capital"].tolist() == [
        UC_CAPITAL_UNAVAILABLE,
        UC_CAPITAL_UNAVAILABLE,
    ]
    assert (
        transform.checkpoint_metadata()["evidence"]["frs_benunit_capital"][
            "mapped_rows"
        ]
        == 2
    )


def test_direct_person_mapping_values_are_ported(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)

    person = build_uk_frs_spine_frame(tmp_path, stage=stage).table("person")
    adult = person.loc[person["person_id"] == 1001].iloc[0]

    assert adult["age"] == 40
    assert adult["gender"] == "MALE"
    assert adult["marital_status"] == "MARRIED"
    assert adult["is_household_head"]
    assert adult["is_benunit_head"]
    assert adult["is_parent"]
    assert adult["hours_worked"] == pytest.approx(40 * WEEKS_IN_YEAR)
    assert adult["employment_income"] == pytest.approx(10 * WEEKS_IN_YEAR)
    assert adult["self_employment_income"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["private_pension_income"] == pytest.approx(15 * WEEKS_IN_YEAR)
    assert adult["tax_free_savings_income"] == pytest.approx(1 * WEEKS_IN_YEAR)
    assert adult["savings_interest_income"] == pytest.approx(3.5 * WEEKS_IN_YEAR)
    assert adult["dividend_income"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["property_income"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["maintenance_income"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["miscellaneous_income"] == pytest.approx(41 * WEEKS_IN_YEAR)
    assert adult["private_transfer_income"] == pytest.approx(57 * WEEKS_IN_YEAR)
    assert adult["lump_sum_income"] == pytest.approx(100)
    assert adult["student_loan_repayments"] == pytest.approx(2 * WEEKS_IN_YEAR)
    assert adult["statutory_sick_pay"] == pytest.approx(WEEKS_IN_YEAR)
    assert adult["statutory_maternity_pay"] == pytest.approx(0.5 * WEEKS_IN_YEAR)
    assert adult["student_loans"] == pytest.approx(500)
    assert adult["access_fund"] == pytest.approx(WEEKS_IN_YEAR)
    assert adult["education_grants"] == pytest.approx(5)
    assert adult["council_tax_benefit_reported"] == pytest.approx(WEEKS_IN_YEAR)
    assert adult["maintenance_expenses"] == pytest.approx(2 * WEEKS_IN_YEAR)
    assert adult["childcare_expenses"] == pytest.approx(5 * WEEKS_IN_YEAR)
    assert adult["personal_pension_contributions"] == pytest.approx(
        95.2 * WEEKS_IN_YEAR
    )
    assert adult["employee_pension_contributions"] == pytest.approx(2 * WEEKS_IN_YEAR)
    assert adult["pension_contributions_via_salary_sacrifice"] == pytest.approx(
        3 * WEEKS_IN_YEAR
    )
    assert adult["salary_sacrifice_reported"] == 1
    assert adult["salary_sacrifice_asked"] == 1


@pytest.mark.parametrize("couple_has_children", [False, True])
def test_uc_claimant_input_uses_frs_membership_not_age_or_marriage(
    tmp_path: Path, couple_has_children: bool
) -> None:
    tables = _fixture_tables()
    # A lone parent with a 19-year-old child remains a single claimant.
    tables["child"][0]["AGE"] = 19
    # A young cohabiting partner is a claimant even below the generic adult age.
    tables["adult"].append(
        {
            **tables["adult"][0],
            "PERSON": 2,
            "UPERSON": 2,
            "HRPID": 0,
            "AGE": 17,
            "MARITAL": 2,
        }
    )
    tables["benunit"][0]["FAMTYPB2"] = 6
    if couple_has_children:
        tables["benunit"][0]["DEPCHLDB"] = 1
        tables["child"].append(
            {**tables["child"][0], "SERNUM": 2, "PERSON": 3, "AGE": 18}
        )
    stage = _write_fixture(tmp_path, tables)
    frame = build_uk_frs_spine_frame(tmp_path, stage=stage)
    person = frame.table("person").set_index("person_id")
    expected = {1001: True, 1002: False, 2001: True, 2002: True}
    if couple_has_children:
        expected[2003] = False
    assert person["is_uc_claimant"].to_dict() == expected
    assert person["is_uc_claimant"].dtype == bool
    assert not frame.table("benunit").set_index("benunit_id").loc[201, "is_married"]


def test_benefit_code_splits_are_ported(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)

    adult = (
        build_uk_frs_spine_frame(tmp_path, stage=stage)
        .table("person")
        .loc[lambda frame: frame["person_id"] == 1001]
        .iloc[0]
    )

    assert adult["child_benefit_reported"] == pytest.approx(7 * WEEKS_IN_YEAR)
    assert adult["jsa_contrib_reported"] == pytest.approx(2 * WEEKS_IN_YEAR)
    assert adult["jsa_income_reported"] == pytest.approx(3 * WEEKS_IN_YEAR)
    assert adult["esa_contrib_reported"] == pytest.approx(4 * WEEKS_IN_YEAR)
    assert adult["esa_income_reported"] == pytest.approx(5 * WEEKS_IN_YEAR)
    assert adult["bsp_reported"] == pytest.approx(6 * WEEKS_IN_YEAR)


def test_household_and_benunit_mapping_values_are_ported(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)

    frame = build_uk_frs_spine_frame(tmp_path, stage=stage)
    household = frame.table("household").set_index("household_id")
    benunit = frame.table("benunit").set_index("benunit_id")

    # GVTREGNO=12 is Scotland in the skip-3 FRS coding — consistent with the
    # Scottish water-charge treatment this same household receives below.
    assert household.loc[1, "region"] == "SCOTLAND"
    assert household.loc[1, "tenure_type"] == "OWNED_WITH_MORTGAGE"
    assert household.loc[1, "accommodation_type"] == "FLAT"
    assert household.loc[1, "num_bedrooms"] == 2
    assert household.loc[1, "council_tax_reported"] == 0
    assert household.loc[1, "council_tax_band"] == "B"
    assert household.loc[1, "council_tax_rebate"] == pytest.approx(WEEKS_IN_YEAR)
    assert household.loc[1, "council_tax_single_adult_raw"] == 1
    # Scotland: CWATAMTD 3 (after discount) + CSEWAMT1 5 (gross) discounted at
    # this household's own observed factor CWATAMTD/CWATAMT1 = 3/4, so
    # 3 + 5 * 0.75 = 6.75. WATSEWRT is not asked in Scotland and is ignored.
    assert household.loc[1, "water_and_sewerage_charges"] == pytest.approx(
        6.75 * WEEKS_IN_YEAR
    )
    assert household.loc[1, "domestic_rates"] == pytest.approx(5 * WEEKS_IN_YEAR)
    assert household.loc[1, "rent"] == pytest.approx(6 * WEEKS_IN_YEAR)
    assert household.loc[1, "subrent"] == 0
    assert household.loc[1, "mortgage_interest_repayment"] == pytest.approx(
        8 * WEEKS_IN_YEAR
    )
    assert household.loc[1, "mortgage_capital_repayment"] == pytest.approx(10)
    assert household.loc[1, "structural_insurance_payments"] == pytest.approx(
        9 * WEEKS_IN_YEAR
    )
    assert household.loc[1, "housing_service_charges"] == pytest.approx(
        45 * WEEKS_IN_YEAR
    )
    assert household.loc[1, "external_child_payments"] == pytest.approx(
        2 * WEEKS_IN_YEAR
    )
    assert benunit.loc[101, "is_married"]
    assert benunit.loc[101, "dependent_children"] == 1
    assert benunit.loc[101, "frs_benunit_capital"] == 111.0
    assert benunit.loc[201, "frs_benunit_capital"] == 222.0


def test_region_code_map_covers_all_twelve_regions() -> None:
    # FRS GVTREGNO skip-3 coding: no code 3 (retired Merseyside), Scotland
    # is 12 (the water-charge branch reads the same code), Northern Ireland
    # is 13. Verified against the 2023-24 tabs in the #692 review: zero
    # code-3 households, 1,844 code-13 households.
    assert 3 not in REGION_MAP
    assert REGION_MAP == {
        1: "NORTH_EAST",
        2: "NORTH_WEST",
        4: "YORKSHIRE",
        5: "EAST_MIDLANDS",
        6: "WEST_MIDLANDS",
        7: "EAST_OF_ENGLAND",
        8: "LONDON",
        9: "SOUTH_EAST",
        10: "SOUTH_WEST",
        11: "WALES",
        12: "SCOTLAND",
        13: "NORTHERN_IRELAND",
    }


def test_shuffled_household_fixture_produces_identical_output(tmp_path: Path) -> None:
    sorted_dir = tmp_path / "sorted"
    shuffled_dir = tmp_path / "shuffled"
    sorted_tables = _fixture_tables()
    sorted_tables["househol"] = sorted(
        sorted_tables["househol"], key=lambda row: int(row["SERNUM"])
    )
    shuffled_tables = _fixture_tables()

    sorted_stage = _write_fixture(sorted_dir, sorted_tables)
    shuffled_stage = _write_fixture(shuffled_dir, shuffled_tables)

    sorted_frame = build_uk_frs_spine_frame(sorted_dir, stage=sorted_stage)
    shuffled_frame = build_uk_frs_spine_frame(shuffled_dir, stage=shuffled_stage)

    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            sorted_frame.table(entity).reset_index(drop=True),
            shuffled_frame.table(entity).reset_index(drop=True),
        )


def _stub_policy_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the driver run engine-hermetic.

    The us-extra CI lane and the wheel-gate venv have no policyengine-uk,
    so the driver tests stub the three deferred parameter readers and then
    block the engine import outright — a regression to eager engine use
    fails loudly in every environment, not just engine-less ones.
    """

    monkeypatch.setattr(
        frs_disability,
        "uk_dwp_disability_category_rates",
        lambda period: frs_disability.UKDWPDisabilityCategoryRates(
            aa_lower=68.1,
            aa_higher=101.75,
            dla_sc_lower=26.9,
            dla_sc_middle=68.1,
            dla_sc_higher=101.75,
            dla_m_lower=26.9,
            dla_m_higher=71.0,
            pip_m_standard=26.9,
            pip_m_enhanced=71.0,
            pip_dl_standard=68.1,
            pip_dl_enhanced=101.75,
            instant=f"{period}-01-01",
            source="test stub",
        ),
    )
    monkeypatch.setattr(
        frs_disability,
        "uk_dwp_disability_flag_rates",
        lambda period: frs_disability.UKDWPDisabilityFlagRates(
            aa_higher=101.75,
            dla_sc_higher=101.75,
            pip_dl_enhanced=101.75,
            instant=f"{period}-01-01",
            source="test stub",
        ),
    )
    monkeypatch.setattr(
        frs_legacy_proxies,
        "uk_legacy_jsa_policy",
        lambda period: frs_legacy_proxies.UKLegacyJSAPolicy(
            max_weekly_hours_single=16.0,
            instant=f"{period}-01-01",
            source="test stub",
        ),
    )
    monkeypatch.setattr(
        frs_education_grants,
        "uk_dsa_policy",
        lambda period: frs_education_grants.UKDSAPolicy(
            maximum=0.0,
            instant=f"{period}-01-01",
            source="test stub",
        ),
    )
    monkeypatch.setitem(sys.modules, "policyengine_uk", None)


def _patch_spi_spine_driver_runtime(
    tool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    spi_tab = tmp_path / "put2223uk.tab"
    hmrc_ods = tmp_path / "Collated_Tables_3_1_to_3_11_2324.ods"
    spi_tab.write_text("synthetic\\n", encoding="utf-8")
    hmrc_ods.write_text("synthetic\\n", encoding="utf-8")

    class _FakeStageTransform:
        def __init__(self, *args, stage, **kwargs) -> None:
            self.stage = stage
            self.last_result = None

        def __call__(self, frame: Frame) -> Frame:
            tables = engine_tables(frame)
            person = tables["person"].copy()
            benunit = tables["benunit"].copy()
            household = tables["household"].copy()
            for column in self.stage.outputs:
                if column.startswith("household_") or column.startswith("source_"):
                    household[column] = _fake_value(column, len(household))
                elif column.startswith("benunit_"):
                    benunit[column] = _fake_value(column, len(benunit))
                else:
                    person[column] = _fake_value(column, len(person))
            result = uk_national_frame(
                person=person,
                benunit=benunit,
                household=household,
                time_period=uk_time_period(frame),
                weight_kind=(
                    WeightKind.IMPORTANCE
                    if self.stage.stage == "spi_support_channel"
                    else uk_household_weight_kind(frame)
                ),
                household_weights=frame.weights_for("household").values,
                mass_log=frame.mass_log,
            )
            self.last_result = SimpleNamespace(replay_report={"report_kind": "fake"})
            return result

    def _write_fake_replay(report, path):
        output = Path(path)
        output.write_text(
            json.dumps({"report_kind": "fake_spine_replay"}) + "\n",
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr(tool, "UKFRSHMRCSpineLeavesStageTransform", _FakeStageTransform)
    monkeypatch.setattr(tool, "UKSPISupportChannelStageTransform", _FakeStageTransform)
    monkeypatch.setattr(tool, "UKSPIIncomeSpineStageTransform", _FakeStageTransform)
    monkeypatch.setattr(tool, "write_hmrc_replay_report", _write_fake_replay)
    return spi_tab, hmrc_ods


def _fake_value(column: str, rows: int):
    if column.endswith("_support_channel"):
        return ["frs"] * rows
    if column == "source_household_key":
        return [f"2024:{index + 1}" for index in range(rows)]
    if column == "household_is_spi_synthetic":
        return [False] * rows
    return np.arange(1, rows + 1, dtype=float)


def test_driver_writes_spine_h5_sidecars_and_logbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The driver writes the spine H5 through the shared writer, which needs
    # pytables — an extra the packaging-gate wheels venv deliberately lacks.
    pytest.importorskip("tables")
    raw_dir = tmp_path / "raw"
    stage = _write_fixture(raw_dir)
    output = tmp_path / "spine.h5"
    shares = tmp_path / "shares.json"
    tool = _load_tool()
    monkeypatch.setattr(
        tool, "load_country_spec", lambda country: _synthetic_spec(stage)
    )
    monkeypatch.setattr(tool, "_rules_engine", lambda: _FakeUKEngine())
    _stub_policy_readers(monkeypatch)
    spi_tab, hmrc_ods = _patch_spi_spine_driver_runtime(tool, monkeypatch, tmp_path)
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)

    # Stale sidecars from an earlier interrupted run must never survive
    # beside a fresh H5 (adversarial-review finding on #717).
    stale_replay = output.with_suffix(".hmrc_replay.json")
    stale_build = output.with_suffix(".build.json")
    stale_replay.write_text('{"report_kind": "stale_leftover"}')
    stale_build.write_text('{"stale": true}')

    assert (
        tool.main(
            [
                "--frs-raw-dir",
                str(raw_dir),
                "--spine-h5",
                str(output),
                "--spi-tab",
                str(spi_tab),
                "--hmrc-ods",
                str(hmrc_ods),
                "--emit-nonzero-shares",
                str(shares),
                "--no-staging",
            ]
        )
        == 0
    )

    frame, _ = load_uk_national_frame(output)
    assert len(frame.table("person")) == 3
    assert len(frame.table("benunit")) == 2
    assert len(frame.table("household")) == 2
    assert {
        "would_claim_child_benefit",
        "child_benefit_opts_out",
        "would_claim_pc",
        "would_claim_uc",
        "would_claim_tfc",
        "would_claim_extended_childcare",
        "would_claim_universal_childcare",
        "would_claim_targeted_childcare",
        "maximum_extended_childcare_hours_usage",
    } <= set(frame.table("benunit"))
    assert {
        "would_claim_marriage_allowance",
        "would_claim_scp",
        "attends_private_school_random_draw",
    } <= set(frame.table("person"))
    assert {
        "household_owns_tv",
        "would_evade_tv_licence_fee",
        "main_residential_property_purchased_is_first_home",
        "property_purchased",
        "brma",
    } <= set(frame.table("household"))
    sidecar = json.loads(output.with_suffix(".build.json").read_text())
    assert sidecar["pipeline"] == "uk-frs-spine"
    assert sidecar["schema_version"] == 2
    assert sidecar["stages"] == list(tool._uk_spine_stage_names(_synthetic_spec(stage)))
    assert sidecar["uk_frame_content_identity"] == uk_frame_content_identity(frame)
    assert sidecar["entity_row_counts"] == {
        "person": 3,
        "benunit": 2,
        "household": 2,
    }
    assert sidecar["household_weight_total"] == 30.0
    assert set(sidecar["artifact_pins"]) == set(FRS_SPINE_TABLES)
    assert sidecar["declared_seeds"]["frs_take_up"]["would_claim_child_benefit"] == 0
    assert sidecar["declared_seeds"]["frs_brma"] == {"brma": 0}
    assert sidecar["declared_seeds"]["spi_support_channel"] == {
        "stack_zero_weight_donors": 42
    }
    assert sidecar["declared_seeds"]["hmrc_spi_income_spine"] == {
        "donor_bootstrap": 42,
        "stage1": 42,
        "stage2": 43,
    }
    assert sidecar["source_vintages"] == {"frs": "2024_25"}
    assert sidecar["sampling"] is None
    replay_bytes = output.with_suffix(".hmrc_replay.json").read_bytes()
    assert json.loads(replay_bytes) == {"report_kind": "fake_spine_replay"}
    # The synthetic spec declares no non-table pinned artifacts, so the pin
    # map is present but empty; the replay binding must match the file on
    # disk byte-for-byte.
    assert sidecar["input_artifact_pins"] == {}
    assert sidecar["hmrc_replay"] == {
        "filename": output.with_suffix(".hmrc_replay.json").name,
        "report_kind": "fake_spine_replay",
        "sha256": hashlib.sha256(replay_bytes).hexdigest(),
    }
    assert len(sidecar["stochastic_contract_sha256"]) == 64
    assert sidecar["resource_pins"] == {"brma_rent_counts.json": "f" * 64}
    # Resolve the expected version the way the driver does, so the assertion
    # holds in the engine-hermetic lane too: the real version where
    # policyengine-uk is installed, the documented fallback where it is not.
    # Either way this still fails on the "unknown" the U1 pin fix removed.
    try:
        expected_engine_version = metadata.version("policyengine-uk")
    except metadata.PackageNotFoundError:
        expected_engine_version = "unavailable"
    assert sidecar["rules_engine"]["version"] == expected_engine_version
    share_payload = json.loads(shares.read_text())
    assert share_payload["stages"]["frs_spine"]["employment_income"] == pytest.approx(
        2 / 3
    )
    assert "education_grants" in share_payload["final"]
    rows = load_spool_rows(tmp_path / "logbook-spool")
    assert len(rows) == 1
    assert rows[0].pipeline == "uk-frs-spine"
    assert rows[0].disposition == "iterating"


def test_driver_writes_payload_identical_h5s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The driver writes the spine H5 through the shared writer, which needs
    # pytables — an extra the packaging-gate wheels venv deliberately lacks.
    pytest.importorskip("tables")
    raw_dir = tmp_path / "raw"
    stage = _write_fixture(raw_dir)
    output = tmp_path / "spine.h5"
    tool = _load_tool()
    monkeypatch.setattr(
        tool, "load_country_spec", lambda country: _synthetic_spec(stage)
    )
    monkeypatch.setattr(tool, "_rules_engine", lambda: _FakeUKEngine())
    _stub_policy_readers(monkeypatch)
    spi_tab, hmrc_ods = _patch_spi_spine_driver_runtime(tool, monkeypatch, tmp_path)
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)

    assert (
        tool.main(
            [
                "--frs-raw-dir",
                str(raw_dir),
                "--spine-h5",
                str(output),
                "--spi-tab",
                str(spi_tab),
                "--hmrc-ods",
                str(hmrc_ods),
                "--no-staging",
            ]
        )
        == 0
    )
    first_frame, _ = load_uk_national_frame(output)
    assert (
        tool.main(
            [
                "--frs-raw-dir",
                str(raw_dir),
                "--spine-h5",
                str(output),
                "--spi-tab",
                str(spi_tab),
                "--hmrc-ods",
                str(hmrc_ods),
                "--no-staging",
            ]
        )
        == 0
    )
    second_frame, _ = load_uk_national_frame(output)

    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            first_frame.table(entity).reset_index(drop=True),
            second_frame.table(entity).reset_index(drop=True),
        )


def test_driver_refuses_missing_spi_tab(tmp_path: Path) -> None:
    tool = _load_tool()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    hmrc_ods = tmp_path / "Collated_Tables_3_1_to_3_11_2324.ods"
    hmrc_ods.write_text("synthetic\n", encoding="utf-8")
    args = tool._parse_args(
        [
            "--frs-raw-dir",
            str(raw_dir),
            "--spine-h5",
            str(tmp_path / "spine.h5"),
            "--spi-tab",
            str(tmp_path / "put2223uk.tab"),
            "--hmrc-ods",
            str(hmrc_ods),
        ]
    )

    with pytest.raises(ValueError, match="--spi-tab must be an existing file"):
        tool._validate_args(args)


def test_driver_refuses_misnamed_spi_tab(tmp_path: Path) -> None:
    tool = _load_tool()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    spi_tab = tmp_path / "spi.tab"
    spi_tab.write_text("synthetic\n", encoding="utf-8")
    hmrc_ods = tmp_path / "Collated_Tables_3_1_to_3_11_2324.ods"
    hmrc_ods.write_text("synthetic\n", encoding="utf-8")
    args = tool._parse_args(
        [
            "--frs-raw-dir",
            str(raw_dir),
            "--spine-h5",
            str(tmp_path / "spine.h5"),
            "--spi-tab",
            str(spi_tab),
            "--hmrc-ods",
            str(hmrc_ods),
        ]
    )

    with pytest.raises(ValueError, match="--spi-tab must name put2223uk.tab"):
        tool._validate_args(args)


def test_driver_refuses_missing_hmrc_ods(tmp_path: Path) -> None:
    tool = _load_tool()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    spi_tab = tmp_path / "put2223uk.tab"
    spi_tab.write_text("synthetic\n", encoding="utf-8")
    args = tool._parse_args(
        [
            "--frs-raw-dir",
            str(raw_dir),
            "--spine-h5",
            str(tmp_path / "spine.h5"),
            "--spi-tab",
            str(spi_tab),
            "--hmrc-ods",
            str(tmp_path / "hmrc.ods"),
        ]
    )

    with pytest.raises(ValueError, match="--hmrc-ods must be an existing file"):
        tool._validate_args(args)


def test_driver_refuses_misnamed_hmrc_ods(tmp_path: Path) -> None:
    tool = _load_tool()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    spi_tab = tmp_path / "put2223uk.tab"
    spi_tab.write_text("synthetic\n", encoding="utf-8")
    hmrc_ods = tmp_path / "hmrc.txt"
    hmrc_ods.write_text("synthetic\n", encoding="utf-8")
    args = tool._parse_args(
        [
            "--frs-raw-dir",
            str(raw_dir),
            "--spine-h5",
            str(tmp_path / "spine.h5"),
            "--spi-tab",
            str(spi_tab),
            "--hmrc-ods",
            str(hmrc_ods),
        ]
    )

    with pytest.raises(ValueError, match="--hmrc-ods must end with '.ods'"):
        tool._validate_args(args)


def test_driver_rejects_non_rung_sample_fraction(tmp_path: Path) -> None:
    tool = _load_tool()

    with pytest.raises(SystemExit):
        tool._parse_args(
            [
                "--frs-raw-dir",
                str(tmp_path),
                "--spine-h5",
                str(tmp_path / "spine.h5"),
                "--spi-tab",
                str(tmp_path / "put2223uk.tab"),
                "--hmrc-ods",
                str(tmp_path / "hmrc.ods"),
                "--sample-fraction",
                "0.5",
            ]
        )


def test_driver_derives_rung_tokens_from_sample_fraction() -> None:
    tool = _load_tool()

    assert tool.UK_SAMPLE_RUNG_TOKENS[tool._rung_sample_fraction("0.01")] == "f001"
    assert tool.UK_SAMPLE_RUNG_TOKENS[tool._rung_sample_fraction("0.10")] == "f010"
    assert tool.UK_SAMPLE_RUNG_TOKENS[tool._rung_sample_fraction("1.0")] == "f100"


def test_driver_accepts_full_fixture_smoke_posture(tmp_path: Path) -> None:
    tool = _load_tool()

    args = tool._parse_args(
        [
            "--frs-raw-dir",
            str(tmp_path),
            "--spine-h5",
            str(tmp_path / "spine.h5"),
            "--spi-tab",
            str(tmp_path / "put2223uk.tab"),
            "--hmrc-ods",
            str(tmp_path / "hmrc.ods"),
            "--sample-seed",
            "41",
            "--smoke",
        ]
    )

    assert args.sample_fraction == 1.0
    assert tool._sample_token(args) == "f100"
    assert (
        tool._new_build_id(datetime(2026, 9, 8, tzinfo=UTC))
        == "uk-frs-spine-20260908T000000Z"
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--smoke", "--release-candidate"],
        [
            "--sample-fraction",
            "0.10",
            "--checkpoint-dir",
            "checkpoints",
        ],
    ],
)
def test_driver_refuses_incompatible_smoke_or_sampling_options(
    tmp_path: Path, extra: list[str]
) -> None:
    tool = _load_tool()

    with pytest.raises(SystemExit):
        tool._parse_args(
            [
                "--frs-raw-dir",
                str(tmp_path),
                "--spine-h5",
                str(tmp_path / "spine.h5"),
                "--spi-tab",
                str(tmp_path / "put2223uk.tab"),
                "--hmrc-ods",
                str(tmp_path / "hmrc.ods"),
                *extra,
            ]
        )


def test_driver_exposes_all_staging_modes_and_rejects_empty_remote(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    base = [
        "--frs-raw-dir",
        str(tmp_path),
        "--spine-h5",
        str(tmp_path / "spine.h5"),
        "--spi-tab",
        str(tmp_path / "put2223uk.tab"),
        "--hmrc-ods",
        str(tmp_path / "hmrc.ods"),
    ]

    remote = tool._parse_args(base)
    local = tool._parse_args([*base, "--staging-local-only"])
    disabled = tool._parse_args([*base, "--no-staging"])

    assert remote.staging_repo_id == "policyengine/populace-uk-staging"
    assert not remote.staging_local_only and not remote.no_staging
    assert local.staging_local_only and not local.no_staging
    assert disabled.no_staging and not disabled.staging_local_only
    with pytest.raises(SystemExit):
        tool._parse_args([*base, "--staging-repo-id", ""])
    with pytest.raises(SystemExit):
        tool._parse_args([*base, "--staging-local-only", "--staging-read-back"])


def test_driver_refuses_checkpoint_dir_on_sampled_rung(tmp_path: Path) -> None:
    tool = _load_tool()

    with pytest.raises(SystemExit):
        tool._parse_args(
            [
                "--frs-raw-dir",
                str(tmp_path),
                "--spine-h5",
                str(tmp_path / "spine.h5"),
                "--spi-tab",
                str(tmp_path / "put2223uk.tab"),
                "--hmrc-ods",
                str(tmp_path / "hmrc.ods"),
                "--sample-fraction",
                "0.10",
                "--checkpoint-dir",
                str(tmp_path / "checkpoints"),
            ]
        )


def test_driver_records_sampled_spine_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tables")
    raw_dir = tmp_path / "raw"
    stage = _write_fixture(raw_dir)
    output = tmp_path / "spine.h5"
    tool = _load_tool()
    monkeypatch.setattr(
        tool, "load_country_spec", lambda country: _synthetic_spec(stage)
    )
    monkeypatch.setattr(tool, "_rules_engine", lambda: _FakeUKEngine())
    _stub_policy_readers(monkeypatch)
    spi_tab, hmrc_ods = _patch_spi_spine_driver_runtime(tool, monkeypatch, tmp_path)
    monkeypatch.setattr(
        tool,
        "sample_frame_households",
        lambda frame, **kwargs: (
            frame,
            {"eligible_households": len(frame.table("household"))},
        ),
    )
    monkeypatch.setattr(
        tool,
        "normalize_sampled_household_mass",
        lambda frame, **kwargs: (frame, 1.0),
    )

    assert (
        tool.main(
            [
                "--frs-raw-dir",
                str(raw_dir),
                "--spine-h5",
                str(output),
                "--spi-tab",
                str(spi_tab),
                "--hmrc-ods",
                str(hmrc_ods),
                "--sample-fraction",
                "0.10",
                "--sample-seed",
                "999",
                "--no-staging",
            ]
        )
        == 0
    )

    sidecar = json.loads(output.with_suffix(".build.json").read_text())
    assert sidecar["sampling"] == {
        "fraction": 0.1,
        "seed": 999,
        "rung_token": "f010",
        "pre_household_count": 2,
        "post_household_count": 2,
        "normalization_factor": 1.0,
        "receipt": {"eligible_households": 2},
    }
    rows = load_spool_rows(tmp_path / "logbook-spool")
    assert rows[0].rung == "f010"


def test_driver_marks_full_fixture_smoke_outputs_non_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5py = pytest.importorskip("h5py")
    raw_dir = tmp_path / "raw"
    stage = _write_fixture(raw_dir)
    output = tmp_path / "smoke.h5"
    tool = _load_tool()
    monkeypatch.setattr(
        tool, "load_country_spec", lambda country: _synthetic_spec(stage)
    )
    monkeypatch.setattr(tool, "_rules_engine", lambda: _FakeUKEngine())
    _stub_policy_readers(monkeypatch)
    spi_tab, hmrc_ods = _patch_spi_spine_driver_runtime(tool, monkeypatch, tmp_path)
    assert (
        tool.main(
            [
                "--frs-raw-dir",
                str(raw_dir),
                "--spine-h5",
                str(output),
                "--spi-tab",
                str(spi_tab),
                "--hmrc-ods",
                str(hmrc_ods),
                "--sample-seed",
                "41",
                "--smoke",
                "--staging-local-only",
                "--staging-run-id",
                "full-smoke-test",
                "--staging-dir",
                str(tmp_path / "staging"),
            ]
        )
        == 0
    )

    sidecar = json.loads(output.with_suffix(".build.json").read_text())
    assert sidecar["non_release"] is True
    assert sidecar["release_posture"] == "non_release_smoke"
    assert sidecar["sampling"] is None
    bundle = validate_v2_bundle(tmp_path / "staging", "full-smoke-test")
    assert bundle["run_manifest"]["non_release"] is True
    assert bundle["run_manifest"]["sample"] == {"mode": "full"}
    assert sidecar["staging_delivery"] == bundle["run_manifest"]["delivery"]
    event_pairs = [(event["stage_id"], event["status"]) for event in bundle["events"]]
    assert ("frs_spine", "started") in event_pairs
    assert ("frs_spine", "completed") in event_pairs
    assert ("sampling", "completed") in event_pairs
    assert ("spine_h5_creation", "completed") in event_pairs
    assert ("sidecar_creation", "completed") in event_pairs
    assert event_pairs[-1] == ("complete", "completed")
    with h5py.File(output, mode="r") as file:
        assert bool(file.attrs["populace_non_release"]) is True
        assert file.attrs["populace_release_posture"] == "smoke"
        assert file.attrs["populace_smoke_build_id"].startswith("uk-frs-spine-")
    rows = load_spool_rows(tmp_path / "logbook-spool")
    assert rows[0].rung == "f100"


def test_driver_records_sanitized_failed_staging_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    stage = _write_fixture(raw_dir)
    output = tmp_path / "failed.h5"
    staging_dir = tmp_path / "staging"
    tool = _load_tool()
    monkeypatch.setattr(
        tool, "load_country_spec", lambda country: _synthetic_spec(stage)
    )
    monkeypatch.setattr(tool, "_rules_engine", lambda: _FakeUKEngine())
    _stub_policy_readers(monkeypatch)
    spi_tab, hmrc_ods = _patch_spi_spine_driver_runtime(tool, monkeypatch, tmp_path)
    secret = "operator-secret-value"

    def _fail_graph(*args, **kwargs):
        raise RuntimeError(f"{tmp_path}/adult.tab token={secret}")

    monkeypatch.setattr(tool, "run_graph", _fail_graph)

    assert (
        tool.main(
            [
                "--frs-raw-dir",
                str(raw_dir),
                "--spine-h5",
                str(output),
                "--spi-tab",
                str(spi_tab),
                "--hmrc-ods",
                str(hmrc_ods),
                "--staging-local-only",
                "--staging-dir",
                str(staging_dir),
                "--staging-run-id",
                "failed-staging-test",
            ]
        )
        == 1
    )

    bundle = validate_v2_bundle(staging_dir, "failed-staging-test")
    manifest = bundle["run_manifest"]
    serialized = json.dumps(bundle, default=str)
    assert manifest["status"] == "failed"
    assert manifest["failure"]["error_code"] == "BUILD_FAILED"
    assert secret not in serialized
    assert str(tmp_path) not in serialized


def test_driver_sampled_named_edge_aborts_with_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    stage = _write_fixture(raw_dir)
    output = tmp_path / "spine.h5"
    tool = _load_tool()
    monkeypatch.setattr(
        tool, "load_country_spec", lambda country: _synthetic_spec(stage)
    )
    monkeypatch.setattr(tool, "_rules_engine", lambda: _FakeUKEngine())
    _stub_policy_readers(monkeypatch)
    spi_tab, hmrc_ods = _patch_spi_spine_driver_runtime(tool, monkeypatch, tmp_path)

    def _raise_named_edge(*args, **kwargs):
        raise ValueError(tool._RUNG_NAMED_EDGE_SIGNATURE)

    monkeypatch.setattr(tool, "sample_frame_households", _raise_named_edge)

    assert (
        tool.main(
            [
                "--frs-raw-dir",
                str(raw_dir),
                "--spine-h5",
                str(output),
                "--spi-tab",
                str(spi_tab),
                "--hmrc-ods",
                str(hmrc_ods),
                "--sample-fraction",
                "0.10",
                "--no-staging",
            ]
        )
        == tool._RUNG_ABORT_EXIT_CODE
    )

    receipt = json.loads(output.with_suffix(".rung_abort.json").read_text())
    assert receipt["named_edge"] == "spine_split_singleton_class"
    assert receipt["sampling"]["rung_token"] == "f010"
    assert not output.exists()
    rows = load_spool_rows(tmp_path / "logbook-spool")
    assert rows[0].disposition == "discarded"
    assert rows[0].rung == "f010"


def test_refuses_missing_tab(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    (tmp_path / "adult.tab").unlink()

    with pytest.raises(FileNotFoundError, match="adult.tab"):
        build_uk_frs_spine_frame(tmp_path, stage=stage)


def test_refuses_size_mismatch(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    with (tmp_path / "adult.tab").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="not the pinned"):
        build_uk_frs_spine_frame(tmp_path, stage=stage)


def test_refuses_sha_mismatch(tmp_path: Path) -> None:
    stage = _write_fixture(tmp_path)
    adult_path = tmp_path / "adult.tab"
    payload = adult_path.read_bytes()
    replacement = b"2" if payload[:1] != b"2" else b"1"
    adult_path.write_bytes(replacement + payload[1:])
    assert adult_path.stat().st_size == next(
        int(artifact["size_bytes"])
        for artifact in stage.artifacts
        if artifact["table"] == "adult"
    )

    with pytest.raises(ValueError, match="hashes to"):
        build_uk_frs_spine_frame(tmp_path, stage=stage)


def test_refuses_nan_in_produced_weight_column(tmp_path: Path) -> None:
    tables = _fixture_tables()
    tables["househol"][0]["GROSS4"] = ""
    stage = _write_fixture(tmp_path, tables)

    with pytest.raises(ValueError, match="produced NaN"):
        build_uk_frs_spine_frame(tmp_path, stage=stage)


def test_input_artifact_pins_bind_spi_donor_and_ods() -> None:
    tool = _load_tool()
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stage_map = spec.sources.stage_map()
    stages = [stage_map[name] for name in tool._uk_spine_stage_names(spec)]

    pins = tool._input_artifact_pins(stages)

    assert set(pins) == {
        "cgt_published_fact_surface",
        "etb_household_tab",
        "lcfs_household_tab",
        "lcfs_person_tab",
        "published_fact_surface",
        "qrf_donor",
        "was_bridge_donor",
        "was_qrf_donor",
    }
    for pin in pins.values():
        assert len(str(pin["sha256"])) == 64
        assert int(pin["size_bytes"]) > 0
        assert str(pin["filename"])
    declared = {
        str(artifact["role"]): str(artifact["sha256"])
        for stage_name in (
            "was_wealth",
            "lcfs_consumption",
            "etb_vat",
            "etb_services",
            "hmrc_spi_income_spine",
            "hmrc_cgt_gains_spine",
        )
        for artifact in stage_map[stage_name].artifacts
        if "table" not in artifact
        and "resource" not in artifact
        and "sha256" in artifact
    }
    assert {role: pin["sha256"] for role, pin in pins.items()} == declared


def test_e8_manifest_seeds_all_reach_the_build_sidecar_harvester() -> None:
    tool = _load_tool()
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stages = spec.sources.stage_map()

    declared = tool._declared_seeds(
        [stages[name] for name in tool._uk_spine_stage_names(spec)]
    )

    assert declared["cgt_incidence_clone"] == {"cgt_prior_amount": 0}
    assert declared["uc_capital_coherence"] == {"frs_benunit_capital": 0}
    assert declared["uc_deduction_attributes"] == {
        "uc_deduction_random_draw": 0,
        "uc_deduction_type_random_draw": 0,
    }
    assert declared["cgt_band_donors"] == {"stack_band_donor_households": 1}
    assert declared["hmrc_cgt_gains_spine"] == {"within_band_draws": 552}
    assert declared["salary_sacrifice"] == {
        "salary_sacrifice": 42,
        "salary_sacrifice_conversion": 2024,
    }
    assert declared["student_loans"] == {
        "student_loan_plan_5": 42,
        "student_loan_plan_2": 42,
    }


def test_spine_sidecar_collects_stage_evidence_by_duck_type() -> None:
    tool = _load_tool()

    class _EvidenceResult:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def evidence(self) -> dict[str, object]:
            return self.payload

    class _CheckpointStage:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def checkpoint_metadata(self) -> dict[str, object]:
            return {"evidence": self.payload}

    e8_payloads = {
        "cgt_incidence_clone": {"stage": "cgt_incidence_clone", "rows": 1},
        "cgt_band_donors": {"stage": "cgt_band_donors", "rows": 2},
        "salary_sacrifice": {"stage": "salary_sacrifice", "rows": 3},
        "student_loans": {"stage": "student_loans", "rows": 4},
        "age_tail": {"stage": "age_tail", "rows": 5},
    }
    spi_payloads = {
        "frs_hmrc_spine_leaves": {
            "stage": "frs_hmrc_spine_leaves",
            "source_signal_rows": {"employment_income": 2},
        },
        "spi_support_channel": {
            "stage": "spi_support_channel",
            "spi_households": 7,
        },
        "hmrc_spi_income_spine": {
            "stage": "hmrc_spi_income_spine",
            "targets": {"count": 8},
        },
    }
    new_payload = {"stage": "future_stage", "rows": 9}
    implementations = {
        "frs_spine": SimpleNamespace(),
        "frs_hmrc_spine_leaves": _CheckpointStage(
            spi_payloads["frs_hmrc_spine_leaves"]
        ),
        "spi_support_channel": _CheckpointStage(spi_payloads["spi_support_channel"]),
        "hmrc_spi_income_spine": _CheckpointStage(
            spi_payloads["hmrc_spi_income_spine"]
        ),
        "cgt_incidence_clone": _CheckpointStage(e8_payloads["cgt_incidence_clone"]),
        "cgt_band_donors": _CheckpointStage(e8_payloads["cgt_band_donors"]),
        "salary_sacrifice": _CheckpointStage(e8_payloads["salary_sacrifice"]),
        "student_loans": SimpleNamespace(
            last_result=_EvidenceResult(e8_payloads["student_loans"])
        ),
        "age_tail": SimpleNamespace(last_result=e8_payloads["age_tail"]),
        "future_stage": _CheckpointStage(new_payload),
    }

    evidence = tool._collect_stage_evidence(
        stage_names=(
            "frs_spine",
            "frs_hmrc_spine_leaves",
            "spi_support_channel",
            "hmrc_spi_income_spine",
            "cgt_incidence_clone",
            "cgt_band_donors",
            "salary_sacrifice",
            "student_loans",
            "age_tail",
            "future_stage",
        ),
        implementations=implementations,
    )

    assert evidence == {
        **spi_payloads,
        **e8_payloads,
        "future_stage": new_payload,
    }
    assert list(evidence) == [
        "frs_hmrc_spine_leaves",
        "spi_support_channel",
        "hmrc_spi_income_spine",
        "cgt_incidence_clone",
        "cgt_band_donors",
        "salary_sacrifice",
        "student_loans",
        "age_tail",
        "future_stage",
    ]
    assert "frs_spine" not in evidence


def test_collect_fit_weight_records_is_duck_typed_and_fail_visible():
    tool = _load_tool()

    class _Record:
        def __init__(self, fit_name, weight_kind):
            self.fit_name = fit_name
            self.weight_kind = weight_kind

    class _Broken:
        @property
        def fit_weight_records(self):
            raise RuntimeError("records unreadable")

    implementations = {
        "frs_spine": SimpleNamespace(),
        "was_wealth": SimpleNamespace(
            fit_weight_records=(_Record("uk_was_2018_20_wealth:savings", "design"),)
        ),
        "etb_vat": SimpleNamespace(fit_weight_records=()),
        "lcfs_consumption": _Broken(),
    }
    records = tool._collect_fit_weight_records(
        stage_names=("frs_spine", "was_wealth", "etb_vat", "lcfs_consumption"),
        implementations=implementations,
    )
    # Stages without the hook contribute nothing; a fitting stage with no or
    # unreadable records persists an empty list, so the release-cut weights
    # audit fails visibly instead of the gap vanishing from the sidecar.
    assert records == {
        "was_wealth": [
            {"fit_name": "uk_was_2018_20_wealth:savings", "weight_kind": "design"}
        ],
        "etb_vat": [],
        "lcfs_consumption": [],
    }
    assert "frs_spine" not in records


class TestScottishWaterAndSewerage:
    """The FRS 2024-25 cell retirement, at the three shapes the tab presents.

    CWATAMT/CSEWAMT survive as headers in this vintage but carry no data, so a
    fixture that supplies them (as the pre-#686 one did) never exercises what
    the real tab does. Each case below is a real domain on the 2024-25 tab.
    """

    @staticmethod
    def _frame(**columns: object) -> pd.DataFrame:
        return pd.DataFrame({name: [value] for name, value in columns.items()})

    def test_discount_factor_carries_to_the_gross_sewerage_cell(self) -> None:
        # 1,641 of 1,684 Scottish households: a positive gross water bill, so
        # the household's own discount factor is observable and applies to the
        # sewerage side of the same bill.
        frame = self._frame(CSEWAMT="", CWATAMTD=3.0, CWATAMT1=4.0, CSEWAMT1=5.0)
        frame.columns = [c.lower() for c in frame.columns]
        assert scottish_water_and_sewerage_weekly(frame).iloc[0] == pytest.approx(6.75)

    def test_undiscounted_household_keeps_the_gross_sewerage_charge(self) -> None:
        frame = self._frame(CWATAMTD=4.0, CWATAMT1=4.0, CSEWAMT1=5.0)
        frame.columns = [c.lower() for c in frame.columns]
        assert scottish_water_and_sewerage_weekly(frame).iloc[0] == pytest.approx(9.0)

    def test_recorded_water_without_a_gross_bill_cell_is_not_scaled(self) -> None:
        # 22 Scottish households carry a recorded CWATAMTD with CWATAMT1 == 0;
        # their CSEWAMT1 is zero too, so the fallback factor cannot move them.
        frame = self._frame(CWATAMTD=3.0, CWATAMT1=0.0, CSEWAMT1=0.0)
        frame.columns = [c.lower() for c in frame.columns]
        assert scottish_water_and_sewerage_weekly(frame).iloc[0] == pytest.approx(3.0)

    def test_sewerage_without_an_observable_discount_is_refused(self) -> None:
        # The domain claim in the docstring — that a household with no gross
        # water bill also carries no gross sewerage — is what makes the 1.0
        # fallback safe. A vintage refresh that breaks it must refuse at build
        # time rather than silently pay sewerage at gross, which would flow
        # into council_tax through the netting.
        frame = self._frame(CWATAMTD=3.0, CWATAMT1=0.0, CSEWAMT1=5.0)
        frame.columns = [c.lower() for c in frame.columns]
        with pytest.raises(ValueError, match="no discount factor is observable"):
            scottish_water_and_sewerage_weekly(frame)

    def test_household_without_council_tax_cells_is_zero(self) -> None:
        # 21 Scottish households carry no council-tax cells at all.
        frame = self._frame(CWATAMTD="", CWATAMT1="", CSEWAMT1="")
        frame.columns = [c.lower() for c in frame.columns]
        assert scottish_water_and_sewerage_weekly(frame).iloc[0] == pytest.approx(0.0)

    def test_retired_cells_cannot_reintroduce_the_incumbent_zeroing(self) -> None:
        # The incumbent adds CSEWAMT before filling, so an all-blank CSEWAMT
        # propagates NaN and zeroes every Scottish household. The successor
        # cells must decide the answer on their own.
        blank = self._frame(CSEWAMT="", CWATAMTD=3.0, CWATAMT1=4.0, CSEWAMT1=5.0)
        blank.columns = [c.lower() for c in blank.columns]
        absent = self._frame(CWATAMTD=3.0, CWATAMT1=4.0, CSEWAMT1=5.0)
        absent.columns = [c.lower() for c in absent.columns]
        result = scottish_water_and_sewerage_weekly(blank).iloc[0]
        assert result == pytest.approx(
            scottish_water_and_sewerage_weekly(absent).iloc[0]
        )
        assert result > 0


def test_in_kind_benefits_map_from_the_raw_person_tapes(tmp_path: Path) -> None:
    """The four in-kind benefit columns, ported at #686.

    They were absent from the spine while the incumbent mapped them straight
    off the person tapes, so the parity screen reported them as columns the
    candidate did not produce. Each is a plain weeklyised amount.
    """

    stage = _write_fixture(tmp_path)

    frame = build_uk_frs_spine_frame(tmp_path, stage=stage)
    person = frame.table("person").set_index("person_id")

    children = person.loc[person["age"] < 16]
    assert len(children) == 1
    child = children.iloc[0]
    assert child["free_school_meals"] == pytest.approx(3.0 * WEEKS_IN_YEAR)
    assert child["free_school_fruit_veg"] == pytest.approx(1.0 * WEEKS_IN_YEAR)
    assert child["free_school_breakfasts"] == pytest.approx(2.0 * WEEKS_IN_YEAR)
    assert child["healthy_start_vouchers"] == pytest.approx(4.0 * WEEKS_IN_YEAR)

    # heartval is on the adult tape as well; the school columns are child-only
    # and must read as zero for adults rather than propagating NaN.
    adults = person.loc[person["age"] > 16]
    assert (adults["healthy_start_vouchers"] > 0).all()
    for column in (
        "free_school_meals",
        "free_school_fruit_veg",
        "free_school_breakfasts",
    ):
        assert (adults[column] == 0).all()
        assert person[column].notna().all()


def test_boundary_evidence_asks_only_the_stages_that_have_run() -> None:
    """The first licensed battery run failed at the assembled boundary because
    the evidence provider consulted all 27 implementations, and an un-run
    stage's checkpoint hook (correctly) refuses. Each boundary must offer only
    its executed prefix — an un-run stage being consulted is the regression.
    """

    tool = _load_tool()

    class _RefusesUntilRun:
        def __init__(self) -> None:
            self.ran = False

        def checkpoint_metadata(self) -> dict[str, object]:
            if not self.ran:
                raise RuntimeError(
                    "checkpoint metadata requires a completed stage run."
                )
            return {"evidence": {"stage": "late_stage", "ok": True}}

    late = _RefusesUntilRun()
    implementations = {"early_stage": SimpleNamespace(), "late_stage": late}

    # The assembled-boundary call: only the executed prefix is offered, so the
    # un-run late stage is never consulted and nothing raises.
    assembled = tool._collect_stage_evidence(
        stage_names=("early_stage",), implementations=implementations
    )
    assert assembled == {}

    late.ran = True
    transferred = tool._collect_stage_evidence(
        stage_names=("early_stage", "late_stage"), implementations=implementations
    )
    assert transferred == {"late_stage": {"stage": "late_stage", "ok": True}}
