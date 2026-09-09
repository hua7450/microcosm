from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.source_manifest import (
    FORBIDDEN_SOURCE_DEPENDENCIES,
    SourceManifest,
)
from microcosm.build.uk_runtime.graph import UK_SPINE_EXCLUSIONS, uk_spine_graph
from microcosm.frame import Frame
from microcosm.graph import compile_graph

ROOT = Path(__file__).resolve().parents[3]
UK_PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build/uk"
FROZEN_SOURCE_STAGES = UK_PACKAGE / "hmrc_income_source_stages.json"
CANONICAL_SOURCE_STAGES = UK_PACKAGE / "source_stages.json"
E3_STAGE_NAMES = [
    "frs_employment",
    "frs_council_tax",
    "frs_disability",
    "frs_education",
    "frs_legacy_proxies",
    "frs_education_grant_split",
]
POST_FRS_SPINE_STAGE_NAMES = [
    "age_tail",
]
E4_STAGE_NAMES = [
    "frs_take_up",
    "frs_person_draws",
    "frs_household_draws",
    "frs_brma",
]
E5_STAGE_NAMES = [
    "was_wealth",
    "regional_property_uprating",
]
E6_STAGE_NAMES = [
    "lcfs_consumption",
    "etb_vat",
    "etb_services",
]
E7_STAGE_NAMES = [
    "frs_hmrc_spine_leaves",
    "spi_support_channel",
    "hmrc_spi_income_spine",
]
UC_REPORTER_REDRAW_STAGE_NAMES = [
    "uc_reporter_redraw",
]
UC_COHERENCE_STAGE_NAMES = [
    "uc_capital_coherence",
]
E9_STAGE_NAMES = [
    "uc_deduction_attributes",
]
E8_STAGE_NAMES = [
    "cgt_incidence_clone",
    "cgt_band_donors",
    "hmrc_cgt_gains_spine",
    "salary_sacrifice",
    "student_loans",
]
UK_SOURCE_STAGE_NAMES = [
    "frs_spine",
    *POST_FRS_SPINE_STAGE_NAMES,
    *E3_STAGE_NAMES,
    *E4_STAGE_NAMES,
    *E5_STAGE_NAMES,
    *E6_STAGE_NAMES,
    *E7_STAGE_NAMES,
    *UC_REPORTER_REDRAW_STAGE_NAMES,
    *UC_COHERENCE_STAGE_NAMES,
    *E9_STAGE_NAMES,
    *E8_STAGE_NAMES,
    "frs_hmrc_retained_leaves",
    "hmrc_spi_income",
]
FROZEN_SOURCE_STAGES_SHA256 = (
    "c0341af7166ae3a85a3c1164e7d9e880c4b4aec122f1a8fa90c73b46c596e1ea"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(frame: Frame) -> Frame:
    return frame


def _uk_graph_stage_names(spec) -> list[str]:
    manifest_stages = {
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    }
    return [
        node_id
        for node_id in compile_graph(uk_spine_graph(spec)).order
        if node_id in manifest_stages
    ]


def _assert_no_forbidden_dependency(value: object) -> None:
    text = json.dumps(value, sort_keys=True).lower()
    for dependency in FORBIDDEN_SOURCE_DEPENDENCIES:
        assert dependency not in text


def _expected_reviewed_source() -> str:
    return (
        "PolicyEngine licensed UKDS mirror (private Hugging Face repository), "
        "spi_2022_23.zip"
    )


def _rephrase_stage2_predictor_note(value: str) -> str:
    return value.replace(
        "policyengine-" + "uk-data frs_only.py",
        "the incumbent UK data build's frs_only.py",
    )


class TestUKSourceStagesManifest:
    def test_source_stages_json_loads_as_shared_manifest(self) -> None:
        manifest = SourceManifest.from_mapping(_load_json(CANONICAL_SOURCE_STAGES))

        assert manifest.country == "uk"
        assert manifest.version == 1
        assert [stage.stage for stage in manifest.stages] == UK_SOURCE_STAGE_NAMES

    def test_country_spec_declares_uk_source_stages(self) -> None:
        spec = load_country_spec("uk")

        assert spec.sources is not None
        assert [stage.stage for stage in spec.sources.stages] == UK_SOURCE_STAGE_NAMES

    def test_e6_block_sits_between_e5_and_e7(self) -> None:
        canonical = _load_json(CANONICAL_SOURCE_STAGES)
        names = [stage["stage"] for stage in canonical["stages"]]

        assert (
            names[
                names.index("regional_property_uprating") + 1 : names.index(
                    "frs_hmrc_spine_leaves"
                )
            ]
            == E6_STAGE_NAMES
        )

    def test_e7_block_sits_between_e6_and_e8(self) -> None:
        canonical = _load_json(CANONICAL_SOURCE_STAGES)
        names = [stage["stage"] for stage in canonical["stages"]]

        assert names[
            names.index("etb_services") + 1 : names.index("cgt_incidence_clone")
        ] == [
            *E7_STAGE_NAMES,
            *UC_REPORTER_REDRAW_STAGE_NAMES,
            *UC_COHERENCE_STAGE_NAMES,
            *E9_STAGE_NAMES,
        ]

    def test_age_tail_runs_immediately_after_frs_spine(self) -> None:
        canonical = _load_json(CANONICAL_SOURCE_STAGES)
        spine = [stage["stage"] for stage in canonical["stages"][:-2]]

        assert spine[1] == "age_tail"

    def test_age_tail_position_owns_the_only_later_age_rewrite_guard(self) -> None:
        manifest = SourceManifest.from_mapping(_load_json(CANONICAL_SOURCE_STAGES))
        stages = list(manifest.stages)
        names = [stage.stage for stage in stages]
        age_tail_index = names.index("age_tail")

        assert age_tail_index == names.index("frs_spine") + 1
        for stage in stages[age_tail_index + 1 :]:
            assert "age" not in stage.outputs, stage.stage
            assert "age" not in stage.rewrites, stage.stage

    def test_e8_block_is_final_and_the_certified_pair_stays_last(self) -> None:
        # The E8 stages stay contiguous at the end of the spine, while the
        # certified pair stays at [-2:] (the frozen-copy lockstep test reads
        # them from there). age_tail is now the post-frs_spine block, before
        # every stage that conditions on age.
        canonical = _load_json(CANONICAL_SOURCE_STAGES)
        names = [stage["stage"] for stage in canonical["stages"]]

        assert names[-2:] == ["frs_hmrc_retained_leaves", "hmrc_spi_income"]
        spine = names[:-2]
        start = spine.index(E8_STAGE_NAMES[0])
        assert spine[start : start + len(E8_STAGE_NAMES)] == E8_STAGE_NAMES
        assert spine[start + len(E8_STAGE_NAMES) :] == []

    def test_copy_is_lockstep_with_frozen_original_except_citation_rewrites(
        self,
    ) -> None:
        frozen = _load_json(FROZEN_SOURCE_STAGES)
        canonical = _load_json(CANONICAL_SOURCE_STAGES)
        frozen_stage = frozen["stages"][0]
        stage1, stage2 = canonical["stages"][-2:]

        expected_operations = copy.deepcopy(frozen_stage["operations"])
        predictor_note = expected_operations[6]["reviewed_absent_predictors"][
            "other_investment_income"
        ]
        expected_operations[6]["reviewed_absent_predictors"][
            "other_investment_income"
        ] = _rephrase_stage2_predictor_note(predictor_note)
        # FRS retained leaves now come from the FRS 2024-25 spine while the
        # frozen HMRC fact surface stays byte-pinned.
        expected_operations[1]["source_vintage"] = "2024-25"
        expected_operations[1]["mapped_build_period"] = 2024
        # Signed period re-map (#723) for materialized HMRC SPI facts.
        expected_operations[7]["mapped_build_period"] = 2024
        expected_operations[7]["period_mapping"] = "latest_published_tax_year"

        assert stage1["operations"] + stage2["operations"] == expected_operations
        _assert_no_forbidden_dependency(
            stage2["operations"][4]["reviewed_absent_predictors"][
                "other_investment_income"
            ]
        )

        expected_artifacts = copy.deepcopy(frozen_stage["artifacts"])
        expected_artifacts[0]["reviewed_source"] = _expected_reviewed_source()
        # Signed period re-map (#723): the ODS source surface remains the
        # frozen 2023-24 file, but the canonical manifest declares that it is
        # replayed against build period 2024.
        expected_artifacts[1]["mapped_build_period"] = 2024
        expected_artifacts[1]["period_mapping"] = "latest_published_tax_year"
        # Declared output-name correction (licensed-data acceptance finding):
        # the frozen original listed the SPI concept "state_pension", but the
        # stage writes the auxiliary column SPI_HMRC_STATE_PENSION_INCOME_COLUMN
        # ("hmrc_spi_state_pension_income") — the model input state_pension is
        # formula-owned and never a frame column here. Outputs became
        # load-bearing when country_stage_plan compiled them into
        # StagePlan.produces, so the copy declares the persisted truth. The
        # operation payloads keep the concept name unchanged.
        expected_outputs = [
            "hmrc_spi_state_pension_income" if name == "state_pension" else name
            for name in frozen_stage["outputs"]
        ]
        assert stage2["outputs"] == expected_outputs
        assert stage2["grain"] == frozen_stage["grain"]
        assert stage2["artifacts"] == expected_artifacts
        _assert_no_forbidden_dependency(stage2["artifacts"])
        _assert_no_forbidden_dependency(stage2["notes"])

    def test_frozen_original_bytes_are_pinned(self) -> None:
        digest = hashlib.sha256(FROZEN_SOURCE_STAGES.read_bytes()).hexdigest()

        assert digest == FROZEN_SOURCE_STAGES_SHA256

    def test_country_stage_plan_assembles_two_certified_uk_national_stages(
        self,
    ) -> None:
        spec = load_country_spec("uk")
        plan = country_stage_plan(
            spec,
            {
                "frs_hmrc_retained_leaves": _identity,
                "hmrc_spi_income": _identity,
            },
            stage_names=("frs_hmrc_retained_leaves", "hmrc_spi_income"),
        )

        assert [stage.name for stage in plan.stages] == [
            "frs_hmrc_retained_leaves",
            "hmrc_spi_income",
        ]

    def test_country_stage_plan_assembles_spine_plan(self) -> None:
        spec = load_country_spec("uk")
        implementations = {name: _identity for name in UK_SOURCE_STAGE_NAMES}
        graph_stage_names = _uk_graph_stage_names(spec)
        plan = country_stage_plan(
            spec,
            implementations,
            stage_names=tuple(graph_stage_names),
        )

        assert [stage.name for stage in plan.stages] == graph_stage_names

    @pytest.mark.parametrize(
        "implementations, match",
        [
            ({"frs_hmrc_retained_leaves": _identity}, "missing"),
            (
                {
                    "frs_spine": _identity,
                    "frs_employment": _identity,
                    "frs_council_tax": _identity,
                    "frs_disability": _identity,
                    "frs_education": _identity,
                    "frs_legacy_proxies": _identity,
                    "frs_education_grant_split": _identity,
                    "frs_take_up": _identity,
                    "frs_person_draws": _identity,
                    "frs_household_draws": _identity,
                    "frs_brma": _identity,
                    "was_wealth": _identity,
                    "regional_property_uprating": _identity,
                    "lcfs_consumption": _identity,
                    "etb_vat": _identity,
                    "etb_services": _identity,
                    "frs_hmrc_spine_leaves": _identity,
                    "spi_support_channel": _identity,
                    "hmrc_spi_income_spine": _identity,
                    "uc_reporter_redraw": _identity,
                    "uc_capital_coherence": _identity,
                    "uc_deduction_attributes": _identity,
                    "cgt_incidence_clone": _identity,
                    "cgt_band_donors": _identity,
                    "hmrc_cgt_gains_spine": _identity,
                    "salary_sacrifice": _identity,
                    "student_loans": _identity,
                    "age_tail": _identity,
                    "frs_hmrc_retained_leaves": _identity,
                    "hmrc_spi_income": _identity,
                    "hmrc_spi_income_fallback": _identity,
                },
                "Unknown stage implementation",
            ),
        ],
    )
    def test_country_stage_plan_refuses_missing_or_unknown_uk_stage(
        self,
        implementations,
        match: str,
    ) -> None:
        spec = load_country_spec("uk")

        with pytest.raises(ValueError, match=match):
            country_stage_plan(spec, implementations)


class TestDeclaredOutputsAreWrittenColumns:
    """Declared outputs must name columns the stages actually write.

    Outputs are load-bearing (``country_stage_plan`` compiles them into
    ``StagePlan.produces``), and the licensed-data acceptance for this
    migration caught a declared output that was an SPI *concept* rather
    than a persisted column — harmless while nothing read the field,
    refused at full rung once it did. This pins every declared output to
    a named runtime written-column constant so the class cannot recur
    without a licensed build to find it (microcosm#690 review).
    """

    def test_stage1_outputs_are_exactly_the_retained_leaf_columns(self) -> None:
        from microcosm.build.uk_runtime.frs_hmrc_leaves import (
            FRS_HMRC_RETAINED_LEAF_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        stage1 = stages["frs_hmrc_retained_leaves"]
        assert stage1.outputs == tuple(FRS_HMRC_RETAINED_LEAF_COLUMNS)

    def test_e3_outputs_are_backed_by_runtime_written_columns(self) -> None:
        from microcosm.build.uk_runtime.etb_services import (
            UK_ETB_SERVICES_NONNEGATIVE_OUTPUT_COLUMNS,
            UK_ETB_SERVICES_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.etb_vat import (
            UK_ETB_VAT_NONNEGATIVE_OUTPUT_COLUMNS,
            UK_ETB_VAT_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_brma import FRS_BRMA_OUTPUT_COLUMNS
        from microcosm.build.uk_runtime.frs_council_tax import (
            FRS_COUNCIL_TAX_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_disability import (
            FRS_DISABILITY_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_education import (
            FRS_EDUCATION_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_education_grants import (
            FRS_EDUCATION_GRANT_OUTPUT_COLUMNS,
            FRS_EDUCATION_GRANT_REWRITES,
        )
        from microcosm.build.uk_runtime.frs_employment import (
            FRS_EMPLOYMENT_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_household_draws import (
            FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_legacy_proxies import (
            FRS_LEGACY_PROXY_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_person_draws import (
            FRS_PERSON_DRAW_NONNEGATIVE_OUTPUT_COLUMNS,
            FRS_PERSON_DRAW_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_take_up import (
            FRS_TAKE_UP_NONNEGATIVE_OUTPUT_COLUMNS,
            FRS_TAKE_UP_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.lcfs_consumption import (
            UK_LCFS_CONSUMPTION_NONNEGATIVE_OUTPUT_COLUMNS,
            UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.regional_uprating import (
            UK_REGIONAL_PROPERTY_REWRITES,
        )
        from microcosm.build.uk_runtime.uc_deduction_attributes import (
            UC_DEDUCTION_NONNEGATIVE_OUTPUT_COLUMNS,
            UC_DEDUCTION_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.was_wealth import (
            UK_WAS_WEALTH_NONNEGATIVE_OUTPUT_COLUMNS,
            UK_WAS_WEALTH_OUTPUT_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        assert stages["frs_employment"].outputs == FRS_EMPLOYMENT_OUTPUT_COLUMNS
        assert stages["frs_council_tax"].outputs == FRS_COUNCIL_TAX_OUTPUT_COLUMNS
        assert stages["frs_disability"].outputs == FRS_DISABILITY_OUTPUT_COLUMNS
        assert stages["frs_education"].outputs == FRS_EDUCATION_OUTPUT_COLUMNS
        assert stages["frs_legacy_proxies"].outputs == FRS_LEGACY_PROXY_OUTPUT_COLUMNS
        assert (
            stages["frs_education_grant_split"].outputs
            == FRS_EDUCATION_GRANT_OUTPUT_COLUMNS
        )
        assert (
            stages["frs_education_grant_split"].rewrites == FRS_EDUCATION_GRANT_REWRITES
        )
        assert stages["frs_take_up"].outputs == FRS_TAKE_UP_OUTPUT_COLUMNS
        assert (
            stages["frs_take_up"].nonnegative_outputs
            == FRS_TAKE_UP_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert stages["frs_person_draws"].outputs == FRS_PERSON_DRAW_OUTPUT_COLUMNS
        assert (
            stages["frs_person_draws"].nonnegative_outputs
            == FRS_PERSON_DRAW_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert (
            stages["frs_household_draws"].outputs == FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS
        )
        assert stages["frs_brma"].outputs == FRS_BRMA_OUTPUT_COLUMNS
        assert stages["was_wealth"].outputs == UK_WAS_WEALTH_OUTPUT_COLUMNS
        assert (
            stages["was_wealth"].nonnegative_outputs
            == UK_WAS_WEALTH_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert stages["uc_deduction_attributes"].outputs == UC_DEDUCTION_OUTPUT_COLUMNS
        assert (
            stages["uc_deduction_attributes"].nonnegative_outputs
            == UC_DEDUCTION_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert stages["regional_property_uprating"].outputs == ()
        assert (
            stages["regional_property_uprating"].rewrites
            == UK_REGIONAL_PROPERTY_REWRITES
        )
        assert stages["lcfs_consumption"].outputs == UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS
        assert (
            stages["lcfs_consumption"].nonnegative_outputs
            == UK_LCFS_CONSUMPTION_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert stages["etb_vat"].outputs == UK_ETB_VAT_OUTPUT_COLUMNS
        assert (
            stages["etb_vat"].nonnegative_outputs
            == UK_ETB_VAT_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert stages["etb_services"].outputs == UK_ETB_SERVICES_OUTPUT_COLUMNS
        assert (
            stages["etb_services"].nonnegative_outputs
            == UK_ETB_SERVICES_NONNEGATIVE_OUTPUT_COLUMNS
        )

    def test_e7_outputs_and_rewrites_are_backed_by_runtime_constants(self) -> None:
        from microcosm.build.uk_runtime.spi_spine import (
            UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS,
            UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS,
            UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS,
            UK_SPI_INCOME_SPINE_REWRITE_COLUMNS,
            UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        assert (
            stages["frs_hmrc_spine_leaves"].outputs
            == UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS
        )
        assert (
            stages["spi_support_channel"].outputs
            == UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS
        )
        income = stages["hmrc_spi_income_spine"]
        assert income.outputs == UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS
        assert income.nonnegative_outputs == (
            UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert income.rewrites == UK_SPI_INCOME_SPINE_REWRITE_COLUMNS
        assert not (set(income.outputs) & set(income.rewrites))

    def test_e8_outputs_and_rewrites_are_backed_by_runtime_constants(self) -> None:
        from microcosm.build.uk_runtime.cgt_structure import (
            HOUSEHOLD_IS_CGT_BAND_DONOR,
            HOUSEHOLD_IS_CGT_CLONE,
        )
        from microcosm.build.uk_runtime.salary_sacrifice import SALSAC_OUTPUT

        stages = load_country_spec("uk").sources.stage_map()

        assert stages["cgt_incidence_clone"].outputs == (
            HOUSEHOLD_IS_CGT_CLONE,
            "capital_gains",
        )
        assert stages["cgt_incidence_clone"].rewrites == ("capital_gains",)
        assert stages["cgt_band_donors"].outputs == (
            HOUSEHOLD_IS_CGT_BAND_DONOR,
            "capital_gains",
        )
        assert stages["cgt_band_donors"].rewrites == ("capital_gains",)
        assert stages["hmrc_cgt_gains_spine"].outputs == ("capital_gains",)
        assert stages["hmrc_cgt_gains_spine"].rewrites == ("capital_gains",)
        assert stages["salary_sacrifice"].outputs == (
            SALSAC_OUTPUT,
            "employee_pension_contributions",
        )
        assert stages["student_loans"].outputs == ("student_loan_plan",)


class TestE3ManifestLockstep:
    def test_e3_raw_tab_pins_match_spine_artifacts(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        spine_pins = {
            artifact["table"]: (
                artifact["locator"],
                artifact["sha256"],
                artifact["size_bytes"],
            )
            for artifact in stages["frs_spine"].artifacts
        }

        for stage_name in E3_STAGE_NAMES:
            for artifact in stages[stage_name].artifacts:
                assert (
                    artifact["locator"],
                    artifact["sha256"],
                    artifact["size_bytes"],
                ) == spine_pins[artifact["table"]]

    def test_e3_operation_kinds_are_declared_in_order(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        assert [op.kind for op in stages["frs_employment"].operations] == [
            "read_tables",
            "map_coded_amounts",
        ]
        assert [op.kind for op in stages["frs_council_tax"].operations] == [
            "read_tables",
            "impute_cell_means",
        ]
        assert [op.kind for op in stages["frs_disability"].operations] == [
            "derive",
            "derive",
        ]
        assert [op.kind for op in stages["frs_education"].operations] == [
            "read_tables",
            "derive",
            "impute_cell_means",
        ]
        assert [op.kind for op in stages["frs_legacy_proxies"].operations] == [
            "read_tables",
            "materialize_rules_engine_predictors",
            "derive",
        ]
        assert [op.kind for op in stages["frs_education_grant_split"].operations] == [
            "materialize_rules_engine_predictors",
            "materialize_rules_engine_predictors",
            "derive",
        ]
        assert [op.kind for op in stages["frs_take_up"].operations] == [
            "aggregate_person_to_benunit",
            "assign_binary_with_anchored_residual",
            "assign_binary_from_rate",
            "assign_binary_with_anchored_residual",
            "assign_binary_with_anchored_residual",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_clipped_normal",
        ]
        assert [op.kind for op in stages["frs_person_draws"].operations] == [
            "assign_binary_from_rate",
            "assign_binary_from_banded_rates",
            "assign_uniform_draw",
            "assign_period_constant",
        ]
        assert [op.kind for op in stages["frs_household_draws"].operations] == [
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
        ]
        assert [op.kind for op in stages["frs_brma"].operations] == [
            "materialize_rules_engine_predictors",
            "sample_categorical_from_count_table",
        ]
        assert [op.kind for op in stages["was_wealth"].operations] == [
            "derive",
            "materialize_rules_engine_predictors",
            "fit_weighted_qrf_chain",
            "fold_into",
            "support_clip",
            "allocate_within_group_waterfall",
        ]
        assert [op.kind for op in stages["regional_property_uprating"].operations] == [
            "uprate_to_regional_reference",
        ]
        assert [op.kind for op in stages["lcfs_consumption"].operations] == [
            "derive",
            "iterative_proportional_fit",
            "bridge_donor_column_via_qrf",
            "assign_binary_from_rate",
            "materialize_rules_engine_predictors",
            "fit_weighted_qrf_chain",
            "support_clip",
            "iterative_proportional_fit",
            "fold_into",
            "zero_when_false",
        ]
        assert [op.kind for op in stages["etb_vat"].operations] == [
            "derive",
            "materialize_rules_engine_predictors",
            "fit_weighted_qrf",
            "support_clip",
        ]
        assert [op.kind for op in stages["etb_services"].operations] == [
            "derive",
            "materialize_rules_engine_predictors",
            "fit_weighted_qrf_chain",
            "support_clip",
            "compute_ratio",
            "allocate_per_capita_from_cell_table",
        ]
        assert [op.kind for op in stages["frs_hmrc_spine_leaves"].operations] == [
            "retain_adjudicated_frs_hmrc_leaves",
            "derive",
        ]
        assert [op.kind for op in stages["spi_support_channel"].operations] == [
            "stack_zero_weight_donors",
            "gate_zero_weight_strata",
            "allocate_zero_weight_prior_mass",
        ]
        assert [op.kind for op in stages["hmrc_spi_income_spine"].operations] == [
            "verify_pinned_hmrc_source_pair",
            "strict_read_private_table",
            "fit_weighted_qrf_stage1",
            "fit_weighted_qrf_stage2",
            "redraw_columns_from_fitted_qrf",
            "materialize_hmrc_income_bands_fail_closed",
            "classify_hmrc_income_facts_with_reviewed_fences",
            "gate_distributional_effective_mass",
        ]
        assert [op.kind for op in stages["uc_reporter_redraw"].operations] == [
            "derive",
            "materialize_rules_engine_predictors",
            "aggregate_person_to_benunit",
            "redraw_spi_reported_uc",
        ]
        assert [op.kind for op in stages["uc_capital_coherence"].operations] == [
            "aggregate_person_to_benunit",
            "redraw_spi_reporter_capital",
            "derive",
        ]
        assert [op.kind for op in stages["uc_deduction_attributes"].operations] == [
            "assign_uniform_draw",
            "assign_uniform_draw",
            "map_uniform_to_banded_rate",
            "map_uniform_to_categorical",
        ]
        assert [op.kind for op in stages["cgt_incidence_clone"].operations] == [
            "clone_records",
            "draw_capital_gains_prior_from_banded_quantiles",
        ]
        assert [op.kind for op in stages["cgt_band_donors"].operations] == [
            "stack_band_donor_households"
        ]
        assert [op.kind for op in stages["hmrc_cgt_gains_spine"].operations] == [
            "verify_pinned_cgt_ods",
            "taxable_income_proxy",
            "rank_preserving_allocation",
            "within_band_draws",
            "sub_aea_remainder",
            "record_mass_conservation_receipt",
            "classify_cgt_band_facts_with_reviewed_fence",
        ]
        assert [op.kind for op in stages["salary_sacrifice"].operations] == [
            "fit_weighted_qrf",
            "convert_donors_to_target_stock",
        ]
        assert [op.kind for op in stages["student_loans"].operations] == [
            "assign_student_loan_plan_cohorts",
            "top_up_to_stock",
            "top_up_to_stock",
        ]

    def test_engine_predictor_and_rewrite_constants_match_manifest(self) -> None:
        from microcosm.build.uk_runtime.etb_services import (
            UK_ETB_SERVICES_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.etb_vat import UK_ETB_VAT_PREDICTORS
        from microcosm.build.uk_runtime.frs_brma import UK_BRMA_PREDICTORS
        from microcosm.build.uk_runtime.frs_education_grants import (
            DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES,
            FRS_EDUCATION_GRANT_REWRITES,
            UK_EDUCATION_GRANT_CAPACITY_PREDICTORS,
        )
        from microcosm.build.uk_runtime.frs_legacy_proxies import (
            UK_LEGACY_PROXY_PREDICTORS,
        )
        from microcosm.build.uk_runtime.frs_take_up import (
            UK_TAKE_UP_ANCHOR_AGGREGATES,
        )
        from microcosm.build.uk_runtime.lcfs_consumption import (
            UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS,
            UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS,
            UK_LCFS_CONSUMPTION_PREDICTORS,
            UK_LCFS_HAS_FUEL_PREDICTORS,
        )
        from microcosm.build.uk_runtime.uc_reporter_redraw import (
            UC_REPORTER_AGGREGATES,
            UC_REPORTER_PREDICTORS,
            UC_REPORTER_SCREEN_VARIABLES,
        )
        from microcosm.build.uk_runtime.was_wealth import (
            UK_WAS_ENGINE_PREDICTORS,
            UK_WAS_WEALTH_PREDICTORS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        legacy_predictors = (
            stages["frs_legacy_proxies"].operations[1].parameters["predictors"]
        )
        grant_predictors = (
            stages["frs_education_grant_split"].operations[0].parameters["predictors"]
        )
        dsa_predictors = (
            stages["frs_education_grant_split"].operations[1].parameters["predictors"]
        )
        assert tuple(legacy_predictors) == UK_LEGACY_PROXY_PREDICTORS
        assert tuple(grant_predictors) == UK_EDUCATION_GRANT_CAPACITY_PREDICTORS
        assert tuple(dsa_predictors) == (
            DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES
        )
        assert (
            stages["frs_education_grant_split"].rewrites == FRS_EDUCATION_GRANT_REWRITES
        )
        assert (
            stages["frs_take_up"].operations[0].parameters["aggregates"]
            == UK_TAKE_UP_ANCHOR_AGGREGATES
        )
        reporter = stages["uc_reporter_redraw"]
        assert (
            tuple(reporter.operations[1].parameters["predictors"])
            == UC_REPORTER_SCREEN_VARIABLES
        )
        assert reporter.operations[2].parameters["aggregates"] == UC_REPORTER_AGGREGATES
        assert (
            tuple(reporter.operations[3].parameters["predictors"])
            == UC_REPORTER_PREDICTORS
        )
        assert (
            tuple(stages["frs_brma"].operations[0].parameters["predictors"])
            == UK_BRMA_PREDICTORS
        )
        assert (
            tuple(stages["was_wealth"].operations[1].parameters["predictors"])
            == UK_WAS_ENGINE_PREDICTORS
        )
        assert (
            tuple(stages["was_wealth"].operations[2].parameters["predictors"])
            == UK_WAS_WEALTH_PREDICTORS
        )
        lcfs = stages["lcfs_consumption"]
        lcfs_ops = {op.kind: op for op in lcfs.operations}
        assert (
            tuple(lcfs_ops["bridge_donor_column_via_qrf"].parameters["predictors"])
            == UK_LCFS_HAS_FUEL_PREDICTORS
        )
        assert (
            tuple(
                lcfs_ops["materialize_rules_engine_predictors"].parameters["predictors"]
            )
            == UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS
        )
        assert (
            tuple(lcfs_ops["fit_weighted_qrf_chain"].parameters["predictors"])
            == UK_LCFS_CONSUMPTION_PREDICTORS
        )
        assert (
            tuple(lcfs_ops["fit_weighted_qrf_chain"].parameters["targets"])
            == UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS[:-1]
        )
        assert (
            tuple(stages["etb_vat"].operations[1].parameters["predictors"])
            == UK_ETB_VAT_PREDICTORS
        )
        from microcosm.build.uk_runtime.etb_services import (
            UK_ETB_SERVICES_EDUCATION_COUNTS,
            UK_ETB_SERVICES_ENGINE_VARIABLES,
        )

        assert (
            tuple(stages["etb_services"].operations[1].parameters["predictors"])
            == UK_ETB_SERVICES_ENGINE_VARIABLES
        )
        assert set(
            stages["etb_services"].operations[1].parameters["derived_predictors"]
        ) == set(UK_ETB_SERVICES_EDUCATION_COUNTS)
        assert (
            tuple(stages["etb_services"].operations[2].parameters["targets"])
            == UK_ETB_SERVICES_OUTPUT_COLUMNS[:3]
        )
        rate_keys = [
            op.parameters["rate_key"]
            for stage_name in (
                "frs_take_up",
                "frs_person_draws",
                "frs_household_draws",
            )
            for op in stages[stage_name].operations
            if "rate_key" in op.parameters
        ]
        assert rate_keys == [
            "child_benefit",
            "child_benefit_opts_out_rate",
            "pension_credit",
            "universal_credit",
            "tax_free_childcare",
            "extended_childcare",
            "universal_childcare",
            "targeted_childcare",
            "marriage_allowance",
            "tv_ownership_rate",
            "tv_licence_evasion_rate",
            "first_time_buyer_rate",
            "property_purchase_rate",
        ]
        scp_bands = stages["frs_person_draws"].operations[1].parameters["bands"]
        assert [band["rate_key"] for band in scp_bands] == [
            "scp_under_6",
            "scp_6_plus",
        ]

    def test_every_e4_stochastic_operation_declares_integer_seed(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        for stage_name in E4_STAGE_NAMES:
            for operation in stages[stage_name].operations:
                if operation.kind in {
                    "assign_binary_with_anchored_residual",
                    "assign_binary_from_rate",
                    "assign_binary_from_banded_rates",
                    "assign_uniform_draw",
                    "assign_clipped_normal",
                    "sample_categorical_from_count_table",
                }:
                    assert isinstance(operation.parameters.get("seed"), int)

    def test_e5_debt_segment_predictors_lockstep(self) -> None:
        from microcosm.build.uk_runtime.was_wealth import (
            UK_WAS_DEBT_SEGMENT_PREDICTORS,
            UK_WAS_WEALTH_PREDICTORS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        qrf = stages["was_wealth"].operations[2]

        assert qrf.kind == "fit_weighted_qrf_chain"
        assert tuple(qrf.parameters["debt_segment_predictors"]) == (
            UK_WAS_DEBT_SEGMENT_PREDICTORS
        )
        # The extra predictor belongs to the debt segment only: the shared
        # base list, and so E5's first three segments, are unchanged.
        assert not set(UK_WAS_DEBT_SEGMENT_PREDICTORS) & set(
            qrf.parameters["predictors"]
        )
        assert tuple(qrf.parameters["predictors"]) == UK_WAS_WEALTH_PREDICTORS

    def test_e5_qrf_operation_declares_integer_seed(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        qrf = stages["was_wealth"].operations[2]

        assert qrf.kind == "fit_weighted_qrf_chain"
        assert qrf.parameters["seed"] == 0

    def test_e6_declared_seed_lockstep(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        lcfs_seeded = {
            op.kind: op.parameters["seed"]
            for op in stages["lcfs_consumption"].operations
            if "seed" in op.parameters
        }
        assert lcfs_seeded == {
            "bridge_donor_column_via_qrf": 0,
            "assign_binary_from_rate": 0,
            "fit_weighted_qrf_chain": 0,
        }
        assert stages["etb_vat"].operations[2].parameters["seed"] == 0
        assert stages["etb_services"].operations[2].parameters["seed"] == 0

    def test_e7_declared_seed_lockstep(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        assert stages["spi_support_channel"].operations[0].parameters["seed"] == 42
        assert stages["hmrc_spi_income_spine"].operations[2].parameters["seed"] == 42
        assert stages["hmrc_spi_income_spine"].operations[3].parameters["seed"] == 43
        assert stages["uc_reporter_redraw"].operations[3].parameters["seed"] == 44
        assert stages["uc_capital_coherence"].operations[1].parameters["seed"] == 0

    def test_e8_declared_seed_lockstep(self) -> None:
        stages = load_country_spec("uk").sources.stage_map()

        assert stages["cgt_incidence_clone"].operations[1].parameters["seed"] == 0
        assert stages["cgt_band_donors"].operations[0].parameters["seed"] == 1
        assert (
            stages["hmrc_cgt_gains_spine"].operations[3].parameters["seed_base"] == 552
        )
        assert stages["salary_sacrifice"].operations[0].parameters["seed"] == 42
        assert stages["salary_sacrifice"].operations[1].parameters["seed"] == 2024
        assert [
            operation.parameters["seed"]
            for operation in stages["student_loans"].operations[1:]
        ] == [42, 42]

    def test_e9_declared_seed_lockstep(self) -> None:
        from microcosm.build.uk_runtime.uc_deduction_attributes import (
            UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS,
        )

        stage = load_country_spec("uk").sources.stage_map()["uc_deduction_attributes"]
        declared = {
            operation.parameters["output"]: operation.parameters["seed"]
            for operation in stage.operations
            if operation.kind == "assign_uniform_draw"
        }

        assert declared == UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS

    def test_full_uk_source_stage_plan_compiles_with_e4_stages(self) -> None:
        spec = load_country_spec("uk")
        implementations = {name: _identity for name in UK_SOURCE_STAGE_NAMES}

        driver_stage_names = tuple(_uk_graph_stage_names(spec))
        plan = country_stage_plan(spec, implementations, stage_names=driver_stage_names)

        assert [stage.name for stage in plan.stages] == list(driver_stage_names)

    def test_internal_disability_carriers_stay_out_of_export_registers(self) -> None:
        from microcosm.build.uk_runtime.frs_disability import (
            UK_INTERNAL_DISABILITY_REPORTED_COLUMNS,
        )
        from microcosm.build.uk_runtime.release_input_coverage import (
            uk_release_input_coverage_required_columns,
        )
        from microcosm.build.uk_runtime.terminal_gates import (
            UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
        )

        gates = _load_json(UK_PACKAGE / "gates.json")
        export_gate = next(
            gate for gate in gates["gates"] if gate["id"] == "uk_export_surface"
        )
        allowed_extra = set(export_gate["parameters"]["allowed_extra_columns"])
        allowed_extra.update(UK_ALLOWED_EXTRA_EXPORT_COLUMNS)
        required = uk_release_input_coverage_required_columns()

        for column in UK_INTERNAL_DISABILITY_REPORTED_COLUMNS:
            assert f"person.{column}" not in allowed_extra
            assert column not in required

    def test_stage2_outputs_are_backed_by_runtime_written_columns(self) -> None:
        from microcosm.build.uk_runtime.spi_support import (
            SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
            SPI_HMRC_QRF_AUXILIARY_COLUMNS,
            SPI_INCOME_IMPUTATION_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        stage2 = stages["hmrc_spi_income"]
        written = (
            set(SPI_INCOME_IMPUTATION_COLUMNS)
            | set(SPI_HMRC_QRF_AUXILIARY_COLUMNS)
            | set(SPI_HMRC_DERIVED_AUXILIARY_COLUMNS)
        )
        # The narrow PAY+EPB+TAXTERM employment input is written on SPI rows
        # by the stage even though the QRF output surface excludes it.
        written.add("employment_income")
        unbacked = [name for name in stage2.outputs if name not in written]
        assert unbacked == [], (
            "Declared outputs with no named runtime written-column constant "
            f"backing them: {unbacked}. Either the manifest declares a "
            "concept instead of a persisted column, or the runtime constant "
            "moved without the manifest following."
        )
