"""The declarative country spec: loading, refusals, and country packages.

Belgium is the first full consumer of the country-spec schema
(microcosm#261): its package declares sources, geography spine, target
references, gates, and release contract as pure data. The golden-file test
pins each greenfield package's loaded spec — stage order, gate selection,
release contract, and the sha256 of every resource — so any byte change is a
reviewed golden diff, never an accident.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from microcosm.build import (
    CountryResourceRow,
    CountrySpec,
    ResolvedCountrySpec,
    country_stage_plan,
    load_country_spec,
)
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.trace import canonical_json_bytes
from microcosm.build.uk_runtime import terminal_gates, weighted_integrity

COUNTRY_PACKAGE_ROOT = Path(__file__).parents[1] / "src/microcosm/build"
GOLDEN_ROOT = Path(__file__).parent / "golden"
GOLDEN_COUNTRIES = ("am", "be")
FORBIDDEN_TARGET_VALUE_KEYS = {"value", "values", "observed", "observed_value"}


def _loaded_spec_summary(country: str) -> dict[str, object]:
    spec = load_country_spec(country)
    return {
        "country": spec.country,
        "fingerprint": spec.fingerprint,
        "resources": list(spec.resources),
        "resource_hashes": dict(spec.resource_hashes),
        "stage_names": [stage.stage for stage in spec.sources.stages],
        "geography_spine_stage": spec.geography_spine.geography_spine.stage,
        "target_reference_names": [
            reference.name for reference in spec.target_references
        ],
        "gate_ids": [gate.id for gate in spec.gates.gates],
        "release": {
            "builder": spec.release_contract.builder,
            "artifact_repo": spec.release_contract.artifact_repo,
            "staging_repo": spec.release_contract.staging_repo,
            "dataset_filename_template": (
                spec.release_contract.dataset_filename_template
            ),
            "required_release_files": list(
                spec.release_contract.required_release_files
            ),
        },
    }


def _nested_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _nested_mapping_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _nested_mapping_keys(child)}
    return set()


def _armenia_scalar_ledger_fact(
    reference,
    ordinal: int,
    *,
    dimensions: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one synthetic scalar Ledger fact for an AM selector probe."""
    selector = reference.ledger_selector
    source_name = str(selector["source_name"])
    source_measure_id = str(selector["source_measure_id"])
    geography_level = str(selector["geography_level"])
    cell_id = f"cell-{ordinal}"
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:am-scalar-{ordinal}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:am-scalar-{ordinal}",
        "lineage": {
            "source_record_id": f"ledger_am.scalar_fixture.{reference.name}.{cell_id}",
            "source_cell_keys": [f"ledger.source_cell.v1:am-{ordinal}"],
            "source_row_keys": [],
        },
        "value": ordinal + 1,
        "period": {"type": "year", "value": reference.period},
        "geography": {
            "level": geography_level,
            "id": "AM" if geography_level == "country" else "AM-01",
            "name": "Armenia selector fixture",
            "vintage": "2022_census",
        },
        "entity": {"name": reference.entity},
        "observed_measure": {
            "source_name": source_name,
            "source_table": "Synthetic Armenia scalar selector fixture",
            "source_measure_id": source_measure_id,
            "source_concept": f"ledger-am:{source_measure_id}",
            "unit": "count",
        },
        "concept_alignment": {
            "source_concept": f"ledger-am:{source_measure_id}",
            "canonical_concept": f"ledger-am:{source_measure_id}",
            "relation": "exact",
            "authority": "ledger-am",
            "legal_vintage": "2024",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": source_name,
            "source_table": "Synthetic Armenia scalar selector fixture",
            "source_file": "synthetic_am_selector_fixture.jsonl",
            "url": "https://statbank.armstat.am/",
            "vintage": "2024",
        },
        "dimensions": dimensions or {},
        "universe_constraints": {
            "domain": "all households"
            if reference.entity == "household"
            else "population"
        },
        "layout": {
            "record_set_id": f"{source_name}.2024.synthetic_scalar_fixture",
            "groupby_dimension": "fixture_cell",
            "groupby_value_id": cell_id,
            "measure_id": source_measure_id,
        },
    }


def _write_package(root: Path, files: dict[str, dict]) -> Path:
    package_dir = root / "xx"
    package_dir.mkdir()
    for name, payload in files.items():
        (package_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    return package_dir


def _minimal_package(**overrides) -> dict[str, dict]:
    files = {
        "country_package.json": {
            "schema_version": 1,
            "country": "xx",
            "policy": "spec-only test package",
            "resources": ["gates.json"],
        },
        "gates.json": {
            "version": 1,
            "country": "xx",
            "policy": "test gates",
            "phases": ["terminal"],
            "gates": [
                {
                    "id": "fit",
                    "gate": "per_family_fit",
                    "phase": "terminal",
                    "criticality": "release_blocking",
                }
            ],
        },
    }
    files.update(overrides)
    return files


def _schema2_target_resource() -> dict[str, object]:
    return {
        "country": "xx",
        "allowed_value_operations": ["identity"],
        "target_references": [
            {
                "name": "population_anchor",
                "ledger_selector": {
                    "source_name": "official_population",
                    "source_measure_id": "people",
                    "period_type": "calendar_year",
                    "geography_level": "country",
                },
                "entity": "person",
                "measure": "people",
                "period": 2023,
                "family": "demography",
                "assertion_policy": "allow_source_projection",
                "period_match_policy": "exact",
                "metadata": {
                    "basis_period": "population_2023",
                    "criticality": "release_blocking",
                    "criticality_tier": "demography_release",
                    "publisher": "Official statistics office",
                    "target_role": "calibration",
                },
            }
        ],
        "target_profile": {
            "schema_version": 2,
            "required_families": ["demography"],
            "criticality_tiers": {
                "demography_release": {
                    "criticality": "release_blocking",
                    "relative_tolerance": 0.02,
                    "description": "Population cells.",
                }
            },
            "basis_periods": {
                "population_2023": {
                    "period": 2023,
                    "basis": "reference_date",
                    "fact_period_type": "calendar_year",
                    "mismatch_policy": "requires_source_projection",
                    "description": "Population reference year.",
                }
            },
            "hierarchy_reconciliations": [],
        },
    }


def _package_with_schema2_targets() -> dict[str, dict]:
    files = _minimal_package()
    files["country_package.json"]["resources"].append("target_references.json")
    files["target_references.json"] = _schema2_target_resource()
    return files


class TestArmenianPackage:
    @pytest.fixture(scope="class")
    def spec(self):
        return load_country_spec("am")

    def test_loads_with_every_declared_resource(self, spec) -> None:
        assert spec.country == "am"
        assert set(spec.resources) == {
            "spec/bundle.yaml",
            "spec/catalogs.yaml",
            "spec/geography.yaml",
            "spec/sources.yaml",
            "spec/spine.yaml",
            "spec/vintages.yaml",
            "source_stages.json",
            "geography_spine.json",
            "target_references.json",
            "gates.json",
            "release_contract.json",
        }
        assert set(spec.resource_hashes) == set(spec.resources) | {
            "country_package.json"
        }

    def test_source_stages_keep_us_donors_distinct_from_armenia(self, spec) -> None:
        stages = spec.sources.stage_map()
        assert tuple(stages) == (
            "load_populace_us_support_pool",
            "assign_am_marz",
        )

        support = stages["load_populace_us_support_pool"]
        assert support.grain == "person"
        assert {artifact["kind"] for artifact in support.artifacts} == {
            "public_microdata"
        }
        assert {"donor_country_code", "support_stratum"} <= set(support.outputs)
        assert "marz_code" not in support.outputs
        assert "US donor support records" in support.survey

        marz = stages["assign_am_marz"]
        assert marz.outputs == ("marz_code",)
        assert {artifact["kind"] for artifact in marz.artifacts} == {
            "public_aggregated_counts"
        }
        assert [operation.kind for operation in marz.operations] == [
            "sample_categorical_from_count_table"
        ]

    def test_geography_spine_is_census_vintage_aware(self, spec) -> None:
        spine = spec.geography_spine.geography_spine
        assert spine.stage == "clone_assign_communities"
        assert spine.method == "clone_assign_uniform"
        assert spine.clones_per_record == 1
        assert spine.geography_level == "community"
        assert spine.code_system == "am_census_community"
        assert spine.vintage == "2022"
        assert spine.vintage_policy == "error"
        assert spine.collision_avoidance is True
        assert spine.constrain_to_column == "marz_code"

    def test_targets_are_engine_free_ledger_references_without_values(
        self, spec
    ) -> None:
        references = {reference.name: reference for reference in spec.target_references}
        expected_names = {
            "armstat_population_by_age_sex_marz",
            "armstat_ilcs_households_by_size_marz",
            "armstat_ilcs_households_by_consumption_band_marz",
            "armstat_lfs_employed_by_age_sex_marz",
            "armstat_lfs_employees_by_industry_sex_marz",
            "armstat_src_payroll_employees_by_industry_sex_marz",
            "armstat_pensioner_caseload",
            "armstat_family_social_benefit_families",
        }
        assert set(references) == expected_names
        table_placeholders = expected_names - {
            "armstat_pensioner_caseload",
            "armstat_family_social_benefit_families",
        }
        assert {
            references[name].metadata["activation_status"]
            for name in table_placeholders
        } == {"requires_harvested_cell_references"}
        assert {
            references[name].metadata["activation_status"]
            for name in expected_names - table_placeholders
        } == {"requires_harvested_fact_reference"}
        assert all(
            reference.metadata.get("ledger_am_key") for reference in references.values()
        )
        assert {
            reference.metadata["target_role"] for reference in references.values()
        } == {"calibration"}
        assert {
            reference.metadata["measure_kind"] for reference in references.values()
        } == {"prepared_column"}
        assert {reference.value_operation for reference in references.values()} == {
            "identity"
        }
        assert {reference.measure for reference in references.values()} <= {
            "households",
            "is_employed",
            "is_employee",
            "is_payroll_employee",
            "people",
            "receives_family_benefit",
            "receives_pension",
        }
        assert {
            "employment_income",
            "household_consumption",
            "household_income",
            "pension_income",
        }.isdisjoint(reference.measure for reference in references.values())
        assert not any(
            token in name for name in references for token in ("poverty", "tax")
        )

        target_payload = json.loads(
            (COUNTRY_PACKAGE_ROOT / "am/target_references.json").read_text(
                encoding="utf-8"
            )
        )
        assert FORBIDDEN_TARGET_VALUE_KEYS.isdisjoint(
            _nested_mapping_keys(target_payload["target_references"])
        )
        assert spec.support_spine is None
        assert not {
            "battery",
            "calibration",
            "imputation",
            "take_up",
        } & {row.kind for row in spec.resource_rows}

    def test_target_selector_vocabulary_resolves_scalar_facts(self, spec) -> None:
        references = tuple(spec.target_references)
        assert all(not reference.ledger_fact_key for reference in references)
        assert all(not reference.ledger_source_record_id for reference in references)
        assert all(
            set(reference.ledger_selector)
            == {"source_name", "source_measure_id", "geography_level"}
            for reference in references
        )

        # Isolate each scalar probe: this exercises the real closed resolver
        # vocabulary through the shape a harvested replacement would carry,
        # without activating the packaged authoring placeholder.
        for ordinal, reference in enumerate(references, start=1):
            active_reference = replace(
                reference,
                metadata={
                    key: value
                    for key, value in reference.metadata.items()
                    if key != "activation_status"
                },
            )
            registry = compile_ledger_target_references(
                [_armenia_scalar_ledger_fact(active_reference, ordinal)],
                [active_reference],
                country="am",
            )
            assert len(registry.specs) == 1
            assert registry.specs[0].name == reference.name
            assert registry.specs[0].value > 0

    @pytest.mark.parametrize("fact_count", [1, 2])
    def test_table_placeholder_refuses_compilation_before_cell_fanout(
        self, spec, fact_count
    ) -> None:
        reference = next(
            reference
            for reference in spec.target_references
            if reference.name == "armstat_population_by_age_sex_marz"
        )
        facts = [
            _armenia_scalar_ledger_fact(
                reference,
                1,
                dimensions={"age_band": "0_to_4", "sex": "female"},
            ),
            _armenia_scalar_ledger_fact(
                reference,
                2,
                dimensions={"age_band": "5_to_9", "sex": "female"},
            ),
        ]

        with pytest.raises(ValueError, match="non-executable placeholder"):
            compile_ledger_target_references(
                facts[:fact_count], [reference], country="am"
            )

    def test_gates_use_greenfield_and_weight_health_posture(self, spec) -> None:
        selected = {gate.gate for gate in spec.gates.gates}
        assert {
            "aggregate_admin",
            "calibration_reference_coverage",
            "macro_realism",
            "per_family_fit",
            "support",
            "target_profile_coverage",
            "weight_ess",
            "weight_ratio",
            "weights_audit",
        } <= selected
        assert {"export_surface", "parity", "target_surface"}.isdisjoint(selected)
        active_blockers = {
            gate.id
            for gate in spec.gates.gates
            if gate.criticality == "release_blocking" and gate.not_applicable is None
        }
        assert {
            "calibration_weights_audit",
            "donor_support_bounds",
            "target_profile_coverage",
        } <= active_blockers
        reference_coverage = [
            gate
            for gate in spec.gates.gates
            if gate.gate == "calibration_reference_coverage"
        ]
        assert len(reference_coverage) == 1
        assert reference_coverage[0].criticality == "release_blocking"
        assert reference_coverage[0].not_applicable is None
        assert reference_coverage[0].evidence_absent_blocks is True

    def test_release_contract_is_public_and_ordinal_free(self, spec) -> None:
        contract = spec.release_contract
        assert contract.builder == "populace-am"
        assert contract.artifact_repo == "policyengine/populace-am"
        assert contract.artifact_repo_private is False
        assert contract.licence_restricted is False
        assert contract.dataset_filename_template == "populace_am_{year}.h5"
        assert contract.private_artifacts == ()
        assert set(contract.required_release_files) <= set(contract.public_artifacts)
        assert "source_coverage.json" in contract.required_release_files
        assert "validation_bands.json" in contract.required_release_files

    def test_fingerprint_is_stable_across_loads(self, spec) -> None:
        assert load_country_spec("am").fingerprint == spec.fingerprint


class TestBelgianPackage:
    @pytest.fixture(scope="class")
    def spec(self):
        return load_country_spec("be")

    def test_loads_with_every_declared_resource(self, spec) -> None:
        assert spec.country == "be"
        assert set(spec.resources) == {
            "spec/bundle.yaml",
            "spec/catalogs.yaml",
            "spec/geography.yaml",
            "spec/sources.yaml",
            "spec/spine.yaml",
            "spec/vintages.yaml",
            "source_stages.json",
            "geography_spine.json",
            "target_references.json",
            "gates.json",
            "release_contract.json",
        }
        assert set(spec.resource_hashes) == set(spec.resources) | {
            "country_package.json"
        }

    def test_source_stage_declares_the_silc_contract(self, spec) -> None:
        stage = spec.sources.stage_map()["silc_load"]
        assert stage.grain == "person"
        kinds = [operation.kind for operation in stage.operations]
        assert "declare_income_reference_offset" in kinds
        assert "map_columns" in kinds
        offsets = [
            operation.parameters["years"]
            for operation in stage.operations
            if operation.kind == "declare_income_reference_offset"
        ]
        assert offsets == [-1]  # SILC year N carries year N-1 incomes
        assert "belgium_pit_article_23_worker_remuneration" in stage.outputs
        assert "belgium_pit_article_23_worker_remuneration" in stage.nonnegative_outputs

    def test_geography_spine_is_vintage_aware(self, spec) -> None:
        spine = spec.geography_spine.geography_spine
        assert spine.geography_level == "commune"
        assert spine.code_system == "be_nis"
        assert spine.vintage == "2025"
        assert spine.vintage_policy == "error"
        assert spine.collision_avoidance is True
        assert spine.constrain_to_column == "region_nuts1"

    def test_targets_arrive_by_reference_with_no_values(self, spec) -> None:
        names = {reference.name for reference in spec.target_references}
        assert {
            "statbel_population_by_age_sex_region",
            "statbel_fiscal_income_by_commune",
            "spf_finances_pit_total",
            "onss_employee_contribution_total",
            "onem_unemployment_caseload",
            "nbb_household_disposable_income",
        } <= names
        by_name = {reference.name: reference for reference in spec.target_references}
        commune = by_name["statbel_fiscal_income_by_commune"]
        assert commune.metadata["nis_vintage"] == "2025"
        assert commune.metadata["geography_vintage"] == "nis_2025"
        assert commune.ledger_selector["geography_vintage"] == "nis_2025"
        assert commune.metadata["criticality"] == "diagnostic"
        assert {
            by_name[name].metadata["activation_status"]
            for name in {
                "statbel_population_by_age_sex_region",
                "statbel_fiscal_income_by_commune",
            }
        } == {"requires_harvested_cell_references"}

        payload = json.loads(
            (COUNTRY_PACKAGE_ROOT / "be/target_references.json").read_text(
                encoding="utf-8"
            )
        )
        assert FORBIDDEN_TARGET_VALUE_KEYS.isdisjoint(
            _nested_mapping_keys(payload["target_references"])
        )

    def test_target_selectors_declare_the_intended_chronicle_vocabulary(
        self, spec
    ) -> None:
        references = {reference.name: reference for reference in spec.target_references}
        expected = {
            "statbel_population_by_age_sex_region": (
                "statbel_population_structure",
                "people",
                "calendar_year",
                2023,
                "nuts1",
                "nuts1_2025",
            ),
            "statbel_fiscal_income_by_commune": (
                "statbel_fiscal_income",
                "taxable_income",
                "tax_year",
                2022,
                "commune",
                "nis_2025",
            ),
            "spf_finances_pit_total": (
                "spf_finances_pit",
                "tax_before_withholding",
                "tax_year",
                2022,
                "country",
                None,
            ),
            "onss_employee_contribution_total": (
                "onss_contributions",
                "worker_article_17_uncapped_component_contribution",
                "calendar_year",
                2022,
                "country",
                None,
            ),
            "onem_unemployment_caseload": (
                "onem_rva_unemployment",
                "receives_unemployment_benefit",
                "calendar_year",
                2022,
                "country",
                None,
            ),
            "nbb_household_disposable_income": (
                "nbb_national_accounts",
                "household_disposable_income",
                "calendar_year",
                2022,
                "country",
                None,
            ),
        }

        for name, (
            source_name,
            source_measure_id,
            period_type,
            period,
            geography_level,
            geography_vintage,
        ) in expected.items():
            reference = references[name]
            assert reference.ledger_selector["source_name"] == source_name
            assert reference.ledger_selector["source_measure_id"] == source_measure_id
            assert reference.ledger_selector["period_type"] == period_type
            assert reference.period == period
            assert reference.period_match_policy == "exact"
            assert reference.assertion_policy == "allow_source_projection"
            assert reference.ledger_selector["geography_level"] == geography_level
            assert (
                reference.ledger_selector.get("geography_vintage") == geography_vintage
            )

    def test_target_profile_declares_tiers_and_income_basis(self, spec) -> None:
        profile = spec.target_profile
        assert profile["schema_version"] == 2
        assert tuple(profile["required_families"]) == (
            "demography",
            "fiscal_income",
            "income_tax",
            "social_security",
            "caseloads",
        )
        tiers = profile["criticality_tiers"]
        assert tiers["core_fiscal_release"]["relative_tolerance"] == 0.05
        assert tiers["caseload_release"]["relative_tolerance"] == 0.15
        assert tiers["validation_only"]["relative_tolerance"] is None

        income_basis = profile["basis_periods"]["assessment_income_year_2022"]
        assert income_basis["period"] == 2022
        assert income_basis["fact_period_type"] == "tax_year"
        assert income_basis["survey_year"] == 2023
        assert income_basis["income_reference_offset_years"] == -1
        assert income_basis["mismatch_policy"] == "requires_source_projection"

        references = {reference.name: reference for reference in spec.target_references}
        assert (
            references["nbb_household_disposable_income"].metadata["target_role"]
            == "validation"
        )
        assert {
            reference.family
            for reference in references.values()
            if reference.metadata["target_role"] == "calibration"
        } >= set(profile["required_families"])

    def test_target_profile_tiers_and_roles_are_declaration_only(self, spec) -> None:
        assert all(reference.tolerance is None for reference in spec.target_references)
        description = json.loads(
            (COUNTRY_PACKAGE_ROOT / "be/target_references.json").read_text(
                encoding="utf-8"
            )
        )["description"]
        assert "validated declaration metadata only" in description
        assert "does not yet wire them into runtime calibration" in description
        assert "current Chronicle Belgian catalog does not satisfy" in description
        assert "#264" in description
        assert "Declaration-only intended Belgian gate posture" in spec.gates.policy
        assert "not implemented here" in spec.gates.policy

    @pytest.mark.parametrize(
        ("reference_name", "aliases"),
        [
            (
                "statbel_population_by_age_sex_region",
                ("be_nuts1_2025", "nuts1_2025", "2025_nuts1"),
            ),
            (
                "statbel_fiscal_income_by_commune",
                ("be_nis_2025", "nis_2025", "2025_nis"),
            ),
        ],
    )
    def test_subnational_targets_accept_only_declared_typed_vintage_aliases(
        self, tmp_path, reference_name, aliases
    ) -> None:
        for index, alias in enumerate(aliases):
            package_dir = tmp_path / f"case-{index}" / "be"
            shutil.copytree(COUNTRY_PACKAGE_ROOT / "be", package_dir)
            target_path = package_dir / "target_references.json"
            payload = json.loads(target_path.read_text(encoding="utf-8"))
            reference = next(
                row
                for row in payload["target_references"]
                if row["name"] == reference_name
            )
            reference["ledger_selector"]["geography_vintage"] = alias
            reference["metadata"]["geography_vintage"] = alias
            target_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_country_spec(package_dir)
            loaded_reference = next(
                row for row in loaded.target_references if row.name == reference_name
            )
            assert loaded_reference.ledger_selector["geography_vintage"] == alias

    @pytest.mark.parametrize(
        ("reference_name", "invalid_vintage"),
        [
            ("statbel_population_by_age_sex_region", "NUTS_2024"),
            ("statbel_fiscal_income_by_commune", "nis_2024"),
        ],
    )
    def test_subnational_targets_refuse_vintages_outside_typed_registry(
        self, tmp_path, reference_name, invalid_vintage
    ) -> None:
        package_dir = tmp_path / "be"
        shutil.copytree(COUNTRY_PACKAGE_ROOT / "be", package_dir)
        target_path = package_dir / "target_references.json"
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        reference = next(
            row for row in payload["target_references"] if row["name"] == reference_name
        )
        reference["ledger_selector"]["geography_vintage"] = invalid_vintage
        reference["metadata"]["geography_vintage"] = invalid_vintage
        target_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="not an exact typed authority alias"):
            load_country_spec(package_dir)

    def test_subnational_target_requires_a_typed_geography_layer(
        self, tmp_path
    ) -> None:
        package_dir = tmp_path / "be"
        shutil.copytree(COUNTRY_PACKAGE_ROOT / "be", package_dir)
        target_path = package_dir / "target_references.json"
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        reference = payload["target_references"][0]
        reference["ledger_selector"]["geography_level"] = "province"
        target_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="does not declare it"):
            load_country_spec(package_dir)

    @pytest.mark.parametrize(
        "reference_name",
        [
            "statbel_population_by_age_sex_region",
            "statbel_fiscal_income_by_commune",
        ],
    )
    def test_multicell_be_placeholders_cannot_compile_before_fanout(
        self, spec, reference_name
    ) -> None:
        reference = next(
            row for row in spec.target_references if row.name == reference_name
        )

        with pytest.raises(ValueError, match="non-executable placeholder"):
            compile_ledger_target_references([], [reference], country="be")

    def test_gates_select_no_incumbent_comparison(self, spec) -> None:
        selected = {gate.gate for gate in spec.gates.gates}
        assert "parity" not in selected  # no incumbent; #264 remains separate
        assert "export_surface" not in selected
        assert "per_family_fit" in selected
        assert "formula_owned_export" in selected
        blocking = [
            gate.id
            for gate in spec.gates.gates
            if gate.criticality == "release_blocking"
        ]
        assert "target_profile_coverage" in blocking
        diagnostic = [
            gate.id for gate in spec.gates.gates if gate.criticality == "diagnostic"
        ]
        assert "commune_fiscal_income_fit" in diagnostic

    def test_release_contract_is_private_and_ordinal_free(self, spec) -> None:
        contract = spec.release_contract
        assert contract.artifact_repo == "policyengine/populace-be-private"
        assert contract.artifact_repo_private is True
        assert contract.licence_restricted is True
        assert contract.dataset_filename_template == "populace_be_{year}.h5"
        assert "source_coverage.json" in contract.required_release_files
        assert "reform_validation.json" in contract.required_release_files

    def test_gates_declare_their_phase_order(self, spec) -> None:
        assert spec.gates.phases == ("terminal",)
        assert {gate.phase for gate in spec.gates.gates} == {"terminal"}

    def test_fingerprint_is_stable_across_loads(self, spec) -> None:
        assert load_country_spec("be").fingerprint == spec.fingerprint


class TestGoldenCountrySpecs:
    @pytest.mark.parametrize("country", GOLDEN_COUNTRIES)
    def test_loaded_spec_matches_the_golden_file_byte_for_byte(
        self, country: str
    ) -> None:
        golden = GOLDEN_ROOT / f"{country}_country_spec.json"
        summary = _loaded_spec_summary(country)
        rendered = canonical_json_bytes(summary)
        assert golden.exists(), (
            "Golden file missing. Generate it after reviewing the spec:\n"
            f'  python -c "..." > {golden}'
        )
        assert rendered == golden.read_bytes(), (
            f"The {country!r} country spec changed. If intentional, regenerate "
            f"tests/golden/{country}_country_spec.json from the loaded spec and "
            "review the diff; resource hashes pin every spec byte."
        )


class TestCountryStagePlan:
    def test_compiles_with_noop_implementations_in_declared_order(self) -> None:
        spec = load_country_spec("be")
        names = [stage.stage for stage in spec.sources.stages] + [
            spec.geography_spine.geography_spine.stage
        ]
        plan = country_stage_plan(spec, {name: (lambda frame: frame) for name in names})
        assert [stage.name for stage in plan.stages] == [
            "silc_load",
            "clone_assign_communes",
        ]
        donors = dict(plan.donors())
        assert donors["silc_load"].source.startswith("https://")
        assert donors["clone_assign_communes"].survey.startswith("Statbel")

    def test_missing_stage_refuses_to_assemble(self) -> None:
        spec = load_country_spec("be")
        with pytest.raises(ValueError, match="missing \\['clone_assign_communes'\\]"):
            country_stage_plan(spec, {"silc_load": lambda frame: frame})

    def test_unknown_stage_is_refused(self) -> None:
        spec = load_country_spec("be")
        names = [stage.stage for stage in spec.sources.stages] + [
            spec.geography_spine.geography_spine.stage,
            "silc_load_fallback",
        ]
        with pytest.raises(ValueError, match="Unknown stage implementation"):
            country_stage_plan(spec, {name: (lambda frame: frame) for name in names})

    def test_default_stage_selection_still_requires_all_declared_stages(self) -> None:
        spec = load_country_spec("be")

        with pytest.raises(ValueError, match="missing \\['clone_assign_communes'\\]"):
            country_stage_plan(spec, {"silc_load": lambda frame: frame})

    def test_explicit_stage_subset_uses_manifest_order(self) -> None:
        spec = load_country_spec("be")
        plan = country_stage_plan(
            spec,
            {
                "silc_load": lambda frame: frame,
                "clone_assign_communes": lambda frame: frame,
            },
            stage_names=("clone_assign_communes", "silc_load"),
        )

        assert [stage.name for stage in plan.stages] == [
            "silc_load",
            "clone_assign_communes",
        ]

    def test_explicit_stage_subset_refuses_empty_or_unknown_names(self) -> None:
        spec = load_country_spec("be")
        implementations = {
            "silc_load": lambda frame: frame,
            "clone_assign_communes": lambda frame: frame,
        }

        with pytest.raises(ValueError, match="stage_names must not be empty"):
            country_stage_plan(spec, implementations, stage_names=())

        with pytest.raises(ValueError, match="Unknown stage selection"):
            country_stage_plan(
                spec,
                implementations,
                stage_names=("silc_load", "silc_load_fallback"),
            )


class TestUKCountryPackage:
    def test_spi_spine_adds_no_country_package_resources(self) -> None:
        # The name records the #717 question this was written to answer; what
        # it does now is pin the whole legacy-JSON resource list, so any
        # increment that ships a new country-package resource lands here.
        # spine_swap_signed_differences.json is #686's deliberate addition.
        spec = load_country_spec("uk")

        legacy_rows = tuple(
            row.path for row in spec.resource_rows if row.kind == "legacy_json"
        )
        assert legacy_rows == (
            "cgt_source_stages.json",
            "degenerate_reviewed_exclusions.json",
            "target_fit_reviewed_exclusions.json",
            "efrs_parity_known_gaps.json",
            "efrs_parity_reference.json",
            "frs_release.json",
            "national_chronicle_feed.json",
            "gates.json",
            "brma_rent_counts.json",
            "calibration_measure_exclusions.json",
            "hmrc_cgt_size_bands.json",
            "advani_summers_capital_gains_distribution.json",
            "salary_sacrifice_anchor.json",
            "slc_liable_stocks.json",
            "cgt_band_donor_support_bounds.json",
            "hmrc_income_release_gate_report.json",
            "hmrc_income_replay_report.json",
            "hmrc_income_source_stages.json",
            "need_energy_targets.json",
            "lcfs_consumption_anchors.json",
            "etb_policy_anchors.json",
            "etb_services_anchors.json",
            "dwp_uc_deduction_distributions.json",
            "nhs_consumption_by_age_gender.json",
            "ons_age_tail_band_populations.json",
            "lcfs_consumption_support_bounds.json",
            "etb_vat_support_bounds.json",
            "etb_services_support_bounds.json",
            "regional_land_values.json",
            "source_stages.json",
            "take_up_contract.json",
            "target_reference_signed_exclusions.json",
            "input_mass_reviewed_exclusions.json",
            "spine_swap_signed_differences.json",
            "spine_candidate_acceptance.json",
            "ledger_compile_parity_incumbent_2025_signed_differences.json",
            "ledger_compile_parity_local_incumbent_2025_signed_differences.json",
            "ledger_compile_parity_production_2023_signed_differences.json",
            "national_staging_build_record.json",
            "parity_fixture_production_2023.json",
            "qrf_tail_reviewed_exclusions.json",
            "release_input_coverage_manifest.json",
            "registry_parity_fixture_2025.json",
            "local_registry_parity_fixture_2025.json",
            "was_wealth_support_bounds.json",
            "uc_deduction_support_bounds.json",
            "local_binding_adjudications.json",
            "local_area_support_exclusions.json",
            "uk_local_target_census.json",
            "uk_data_target_parity.json",
            "uk_data_target_inventory.json",
            "local_validation_levels.json",
            "uk_population_targets.json",
            "uk_firms_targets.json",
            "local_area_crosswalk.json",
            "target_references.json",
            "target_reference_membership.json",
            "local_target_references.json",
            "local_target_reference_membership.json",
        )

    def test_uk_source_manifest_loads_thirty_stages(self) -> None:
        spec = load_country_spec("uk")

        assert spec.sources is not None
        # 28 spine stages (uc_reporter_redraw #832, then uc_deduction_attributes
        # #685 as the newest) plus the
        # two certified-pair stages the June path still uses.
        assert len(spec.sources.stages) == 30


class TestExistingPackagesGeneralize:
    """The loader is country-neutral: the US and UK packages load unchanged."""

    def test_us_package_loads(self) -> None:
        spec = load_country_spec("us")
        assert spec.country == "us"
        assert spec.sources is not None
        assert spec.support_spine is not None
        # US target references live in fiscal_target_references.json (an
        # untyped resource its runtime interprets); Belgium and the UK use the
        # typed target_references.json convention.
        assert spec.target_references == ()

    def test_uk_package_loads(self) -> None:
        spec = load_country_spec("uk")
        assert spec.country == "uk"
        assert spec.resources == (
            "spec/bundle.yaml",
            "spec/catalogs.yaml",
            "spec/geography.yaml",
            "spec/sources.yaml",
            "spec/spine.yaml",
            "spec/vintages.yaml",
            "cgt_source_stages.json",
            "degenerate_reviewed_exclusions.json",
            "target_fit_reviewed_exclusions.json",
            "efrs_parity_known_gaps.json",
            "efrs_parity_reference.json",
            "frs_release.json",
            "national_chronicle_feed.json",
            "gates.json",
            "brma_rent_counts.json",
            "calibration_measure_exclusions.json",
            "hmrc_cgt_size_bands.json",
            "advani_summers_capital_gains_distribution.json",
            "salary_sacrifice_anchor.json",
            "slc_liable_stocks.json",
            "cgt_band_donor_support_bounds.json",
            "hmrc_income_release_gate_report.json",
            "hmrc_income_replay_report.json",
            "hmrc_income_source_stages.json",
            "need_energy_targets.json",
            "lcfs_consumption_anchors.json",
            "etb_policy_anchors.json",
            "etb_services_anchors.json",
            "dwp_uc_deduction_distributions.json",
            "nhs_consumption_by_age_gender.json",
            "ons_age_tail_band_populations.json",
            "lcfs_consumption_support_bounds.json",
            "etb_vat_support_bounds.json",
            "etb_services_support_bounds.json",
            "regional_land_values.json",
            "source_stages.json",
            "take_up_contract.json",
            "target_reference_signed_exclusions.json",
            "input_mass_reviewed_exclusions.json",
            "spine_swap_signed_differences.json",
            "spine_candidate_acceptance.json",
            "ledger_compile_parity_incumbent_2025_signed_differences.json",
            "ledger_compile_parity_local_incumbent_2025_signed_differences.json",
            "ledger_compile_parity_production_2023_signed_differences.json",
            "national_staging_build_record.json",
            "parity_fixture_production_2023.json",
            "qrf_tail_reviewed_exclusions.json",
            "release_input_coverage_manifest.json",
            "registry_parity_fixture_2025.json",
            "local_registry_parity_fixture_2025.json",
            "was_wealth_support_bounds.json",
            "uc_deduction_support_bounds.json",
            "local_binding_adjudications.json",
            "local_area_support_exclusions.json",
            "uk_local_target_census.json",
            "uk_data_target_parity.json",
            "uk_data_target_inventory.json",
            "local_validation_levels.json",
            "uk_population_targets.json",
            "uk_firms_targets.json",
            "local_area_crosswalk.json",
            "target_references.json",
            "target_reference_membership.json",
            "local_target_references.json",
            "local_target_reference_membership.json",
        )

    def test_uk_target_references_accept_regenerated_contract_fields(self) -> None:
        spec = load_country_spec("uk")

        references = {reference.name: reference for reference in spec.target_references}
        assert len(references) == 415
        assert references["obr.esa"].value_operation == "sum"
        assert references["dwp.uc.households"].value_operation == (
            "monthly_window_sum_average"
        )
        assert references["dwp.uc.households"].period_match_policy == "source_window"
        assert len(references["dwp.uc.households"].value_operands) == 10
        assert (
            references["obr.income_tax"].assertion_policy == "allow_source_projection"
        )

        fanout = references["hmrc/employment_income_income_band_100_000_to_150_000"]
        assert fanout.metadata == {
            "contract_target_id": (
                "hmrc.spi.employment_income.amount_by_total_income_band"
            ),
            "measure_kind": "prepared_column",
        }
        assert fanout.uprating_from_period == "2023"
        assert fanout.uprating_to_period == 2025


class TestResolvedCountrySpecSeam:
    def test_country_spec_is_the_exact_resolved_alias(self) -> None:
        assert CountrySpec is ResolvedCountrySpec

    def test_generation_one_rows_retain_explicit_legacy_evidence(self) -> None:
        spec = load_country_spec("be")
        assert spec.resources == tuple(row.path for row in spec.resource_rows)
        typed = [row for row in spec.resource_rows if row.kind != "legacy_json"]
        legacy = [row for row in spec.resource_rows if row.kind == "legacy_json"]
        assert {row.kind for row in typed} == {
            "bundle",
            "catalogs",
            "geography",
            "sources",
            "spine",
            "vintages",
        }
        assert all(
            row.kind == "legacy_json" and row.schema_id == "legacy_json"
            for row in legacy
        )
        assert spec.resolved_spec is not None

    def test_resource_rows_are_frozen(self) -> None:
        row = CountryResourceRow(
            path="spec/bundle.yaml",
            kind="bundle",
            schema_id="bundle.schema.json",
        )
        with pytest.raises(FrozenInstanceError):
            row.path = "spec/changed.yaml"

    def test_typed_json_and_yaml_descriptors_load_together(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"] = [
            {
                "path": "gates.json",
                "kind": "legacy_json",
                "schema_id": "legacy_json",
            },
            {
                "path": "spec/bundle.yaml",
                "kind": "bundle",
                "schema_id": "bundle.schema.json",
            },
        ]
        del files["country_package.json"]["policy"]
        package_dir = _write_package(tmp_path, files)
        spec_dir = package_dir / "spec"
        spec_dir.mkdir()
        (spec_dir / "bundle.yaml").write_text(
            "country: xx\nidentity_generation: 1\nseed_protocol: legacy-v1\n",
            encoding="utf-8",
        )

        spec = load_country_spec(package_dir)

        assert spec.resources == ("gates.json", "spec/bundle.yaml")
        assert spec.gates is not None
        assert spec.resource_rows[1] == CountryResourceRow(
            path="spec/bundle.yaml",
            kind="bundle",
            schema_id="bundle.schema.json",
        )
        assert set(spec.resource_hashes) == {
            "country_package.json",
            "gates.json",
            "spec/bundle.yaml",
        }
        assert spec.resolved_spec is not None

    def test_generation_one_manifest_does_not_require_legacy_policy(
        self, tmp_path
    ) -> None:
        package_dir = tmp_path / "xx"
        package_dir.mkdir()
        (package_dir / "country_package.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "country": "xx",
                    "resources": [
                        {
                            "path": "bundle.yaml",
                            "kind": "bundle",
                            "schema_id": "bundle.schema.json",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (package_dir / "bundle.yaml").write_text(
            "country: xx\nidentity_generation: 1\nseed_protocol: legacy-v1\n",
            encoding="utf-8",
        )

        spec = load_country_spec(package_dir)

        assert spec.policy == ""
        assert spec.sources is None
        assert spec.gates is None
        assert spec.resolved_spec is not None
        assert spec.resolved_spec.country == "xx"

    def test_generated_locks_are_admitted_but_excluded_from_authority_hashes(
        self, tmp_path
    ) -> None:
        files = _minimal_package()
        files.update(
            {
                "bundle.lock.json": {},
                "engine_abi.lock.json": {},
                "plan.lock.json": {},
            }
        )
        spec = load_country_spec(_write_package(tmp_path, files))

        assert set(spec.resource_hashes) == {"country_package.json", "gates.json"}
        assert all("lock.json" not in resource for resource in spec.resources)

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            (
                {
                    "path": "../escape.yaml",
                    "kind": "bundle",
                    "schema_id": "bundle.schema.json",
                },
                "normalized local POSIX path",
            ),
            (
                {
                    "path": "bundle.yaml",
                    "kind": "executable",
                    "schema_id": "bundle.schema.json",
                },
                "unknown kind",
            ),
            (
                {
                    "path": "bundle.yaml",
                    "kind": "bundle",
                    "schema_id": "bundle.schema.json",
                    "entrypoint": "microcosm.build:run",
                },
                "closed-world",
            ),
            (
                {
                    "path": "engine_abi.lock.json",
                    "kind": "legacy_json",
                    "schema_id": "legacy_json",
                },
                "generated locks cannot be authored",
            ),
        ],
    )
    def test_invalid_typed_resource_rows_are_refused(
        self, tmp_path, row, message
    ) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"] = [row]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match=message):
            load_country_spec(package_dir)

    def test_duplicate_typed_paths_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        row = {
            "path": "gates.json",
            "kind": "legacy_json",
            "schema_id": "legacy_json",
        }
        files["country_package.json"]["resources"] = [row, dict(row)]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="duplicate resource path"):
            load_country_spec(package_dir)


class TestUKGatesManifest:
    """The UK battery declared as data (microcosm#611 increment 1).

    Every threshold in ``uk/gates.json`` is pinned against the module
    constant the legacy battery still runs on, so spec and code cannot
    drift apart during the migration window (the constants retire when
    the national build swaps onto the battery executor).
    """

    @pytest.fixture(scope="class")
    def manifest(self):
        return load_country_spec("uk").gates

    def test_declares_the_uk_phases_in_order(self, manifest) -> None:
        assert manifest is not None
        assert manifest.phases == (
            "preflight",
            "assembled",
            "transferred",
            "terminal",
        )

    def test_declares_the_full_june_battery(self, manifest) -> None:
        assert [gate.id for gate in manifest.gates] == [
            "uk_release_input_coverage_manifest_current",
            "uk_release_family_build_stages",
            "uk_ledger_compile_parity_production_2023",
            "uk_ledger_compile_parity_incumbent_2025",
            "uk_ledger_compile_parity_local_incumbent_2025",
            "uk_target_surface_local_default_2025",
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
            "uk_release_input_coverage",
            "uk_degenerate_release_surface",
            "uk_zero_weight_strata",
            "uk_weight_ess",
            "uk_weight_ratio",
            "uk_weights_audit",
            "uk_nonnegative_columns",
            "uk_uc_capital_coherence",
            "uk_support",
            "uk_aggregate_admin",
            "uk_export_surface",
            "uk_take_up_signal",
            "uk_brma_enum_domain",
            "uk_uc_deduction_combination_enum_domain",
            "uk_student_loan_plan_enum_domain",
            "uk_calibration_reference_coverage",
            "uk_target_surface",
            "uk_target_fit",
            "uk_input_mass_parity",
            "uk_qrf_tail_concentration",
            "uk_local_geography_ladder_post_calibration",
            "uk_local_area_support",
            "uk_local_target_fit",
            "uk_local_per_family_fit",
            "uk_local_weight_ratio",
            "uk_local_weight_ess",
        ]
        # PR #870 review: the four local fit/weight gates are release-blocking
        # like the rest of the battery; nothing in the manifest is diagnostic.
        diagnostic: set[str] = set()
        assert {
            gate.id for gate in manifest.gates if gate.criticality == "diagnostic"
        } == diagnostic
        assert all(
            gate.criticality == "release_blocking"
            for gate in manifest.gates
            if gate.id not in diagnostic
        )

    def test_ledger_compile_parity_gates_pin_their_fixture_periods(
        self, manifest
    ) -> None:
        params = {gate.id: gate.parameters for gate in manifest.gates}

        assert (
            params["uk_ledger_compile_parity_production_2023"]["target_period"] == 2023
        )
        assert (
            params["uk_ledger_compile_parity_incumbent_2025"]["target_period"] == 2025
        )
        assert (
            params["uk_ledger_compile_parity_local_incumbent_2025"]["target_period"]
            == 2025
        )
        assert (
            params["uk_ledger_compile_parity_local_incumbent_2025"]["registry_artifact"]
            == "uk_ledger_compiled_local_registries"
        )
        assert (
            params["uk_target_surface_local_default_2025"]["expected"]
            == "local_default_surface"
        )
        assert (
            params["uk_target_surface_local_default_2025"]["registry_artifact"]
            == "uk_ledger_compiled_local_registries"
        )

    def test_strict_absent_evidence_entries_are_declared(self, manifest) -> None:
        # "An absent audit is not a passing audit" — the retired schema-3
        # path blocked every posture on a missing fit-weight audit, and the
        # battery keeps that strictness via the entry flag (#654, #691
        # review). Stage-health gates also block on absent receipts because
        # the spine build cannot silently skip a checkpoint's own evidence.
        flagged = [g.id for g in manifest.gates if g.evidence_absent_blocks]
        assert flagged == [
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
            "uk_weights_audit",
        ]
        assert all(g.not_applicable is None for g in manifest.gates)

    def test_gate_names_are_country_neutral(self, manifest) -> None:
        by_id = {gate.id: gate.gate for gate in manifest.gates}
        # The two legacy names the bindings re-mint to the shared vocabulary.
        assert by_id["uk_release_input_coverage"] == "release_input_coverage"
        assert by_id["uk_qrf_tail_concentration"] == "tail_concentration"
        assert not any(name.startswith("uk_") for name in by_id.values())

    def test_thresholds_match_the_schema4_manifest(self, manifest) -> None:
        params = {gate.id: gate.parameters for gate in manifest.gates}
        assert params["uk_weight_ess"]["minimum_ess_fraction"] == 0.01
        assert (
            params["uk_weight_ratio"]["maximum_max_to_median_ratio"]
            == 1_151.2542195939373
        )
        assert params["uk_input_mass_parity"]["relative_tolerance"] == 4.521811483823806
        assert params["uk_input_mass_parity"]["minimum_reference_total"] == 0.0
        assert params["uk_qrf_tail_concentration"]["top_k"] == 100
        assert (
            params["uk_qrf_tail_concentration"]["max_top_share"] == 0.9994670564654868
        )
        assert params["uk_qrf_tail_concentration"]["min_nonzero_records"] == 104
        assert (
            params["uk_target_fit"]["max_abs_relative_error"]
            == terminal_gates.UK_MAX_TARGET_ABS_RELATIVE_ERROR
        )
        assert params["uk_support"]["support_bounds_resources"] == (
            "was_wealth_support_bounds.json",
            "lcfs_consumption_support_bounds.json",
            "etb_vat_support_bounds.json",
            "etb_services_support_bounds.json",
            "uc_deduction_support_bounds.json",
        )
        aggregate = params["uk_aggregate_admin"]
        assert aggregate["default_rtol"] == 0.15
        assert [anchor["name"] for anchor in aggregate["anchors"]] == [
            "need_electricity_mean_spending",
            "need_gas_mean_spending",
            "nhs_spending_total",
        ]

    def test_zero_weight_declarations_match_the_june_strata(self, manifest) -> None:
        params = {gate.id: gate.parameters for gate in manifest.gates}
        declared = params["uk_zero_weight_strata"]["declarations"]
        strata = terminal_gates.UK_DEFAULT_ZERO_WEIGHT_STRATA
        assert len(declared) == len(strata)
        for entry, stratum in zip(declared, strata, strict=True):
            assert entry["name"] == stratum.name
            assert dict(entry["selector"]) == stratum.selector
            assert entry["maximum_zero_weight_rows"] == (
                stratum.maximum_zero_weight_rows
            )
            assert entry["reason"] == stratum.reason

    def test_export_surface_registers_match_the_reviewed_constants(
        self, manifest
    ) -> None:
        params = {gate.id: gate.parameters for gate in manifest.gates}
        export = params["uk_export_surface"]
        assert (
            export["allowed_extra_columns"]
            == terminal_gates.UK_ALLOWED_EXTRA_EXPORT_COLUMNS
        )
        assert (
            dict(export["reviewed_exclusions"])
            == terminal_gates.UK_REVIEWED_EXPORT_EXCLUSIONS
        )

    def test_input_mass_reference_is_a_declared_pinned_input(self, manifest) -> None:
        # The microcosm#327 rule: a parity gate's reference and exclusion
        # register are declared per-country inputs, never implicit code.
        params = {gate.id: gate.parameters for gate in manifest.gates}
        input_mass = params["uk_input_mass_parity"]
        assert input_mass["reference"] in input_mass["reference_registry"]
        expected_registry = {
            name: descriptor.spec_payload()
            for name, descriptor in (
                weighted_integrity.UK_INPUT_MASS_REFERENCE_REGISTRY.items()
            )
        }
        assert input_mass["reference_registry"] == expected_registry
        assert (
            input_mass["reviewed_exclusions_resource"]
            == weighted_integrity.UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE
        )
        qrf = params["uk_qrf_tail_concentration"]
        assert (
            qrf["reviewed_exclusions_resource"]
            == weighted_integrity.UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE
        )
        degenerate = params["uk_degenerate_release_surface"]
        assert (
            degenerate["reviewed_exclusions_resource"]
            == weighted_integrity.UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE
        )


class TestRefusals:
    def test_undeclared_file_on_disk_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        package_dir = _write_package(tmp_path, files)
        (package_dir / "stray.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="stray.json"):
            load_country_spec(package_dir)

    def test_missing_declared_resource_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"] = ["gates.json", "absent.json"]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(FileNotFoundError, match="absent.json"):
            load_country_spec(package_dir)

    def test_country_mismatch_in_a_resource_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["country"] = "yy"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="declares country 'yy'"):
            load_country_spec(package_dir)

    def test_unknown_gate_function_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["gate"] = "vibes"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="unknown gate function 'vibes'"):
            load_country_spec(package_dir)

    def test_non_bool_evidence_absent_blocks_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["evidence_absent_blocks"] = "yes"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="evidence_absent_blocks must be"):
            load_country_spec(package_dir)

    def test_evidence_absent_blocks_on_an_excused_entry_is_refused(
        self, tmp_path
    ) -> None:
        files = _minimal_package()
        entry = files["gates.json"]["gates"][0]
        entry.pop("parameters", None)
        entry["not_applicable"] = "reviewed: no surface yet"
        entry["evidence_absent_blocks"] = True
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_country_spec(package_dir)

    def test_all_diagnostic_gates_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["criticality"] = "diagnostic"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="release_blocking"):
            load_country_spec(package_dir)

    def test_gate_without_a_phase_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        del files["gates.json"]["gates"][0]["phase"]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="phase must be a non-empty string"):
            load_country_spec(package_dir)

    def test_unknown_gate_phase_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["phase"] = "someday"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="unknown phase 'someday'"):
            load_country_spec(package_dir)

    def test_missing_phase_order_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        del files["gates.json"]["phases"]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="phases must be a non-empty list"):
            load_country_spec(package_dir)

    def test_gate_phase_outside_the_declared_order_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["phase"] = "preflight"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="not in the declared phase order"):
            load_country_spec(package_dir)

    def test_duplicate_phases_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["phases"] = ["terminal", "terminal"]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="duplicate phase"):
            load_country_spec(package_dir)

    def test_unknown_gate_entry_key_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["paramters"] = {"within": 0.1}
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(
            ValueError, match=r"gate entry 'fit' has unknown keys \['paramters'\]"
        ):
            load_country_spec(package_dir)

    def test_not_applicable_with_parameters_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["not_applicable"] = "no surface yet"
        files["gates.json"]["gates"][0]["parameters"] = {"within": 0.1}
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_country_spec(package_dir)

    def test_empty_not_applicable_reason_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["not_applicable"] = "  "
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(
            ValueError, match="not_applicable must be a non-empty string"
        ):
            load_country_spec(package_dir)

    def test_target_reference_carrying_a_value_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("target_references.json")
        files["target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "smuggled",
                    "ledger_selector": {"source_name": "somewhere"},
                    "entity": "person",
                    "measure": "people",
                    "value": 123.0,
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="values live in Ledger"):
            load_country_spec(package_dir)

    def test_target_reference_carrying_a_nested_value_is_refused(
        self, tmp_path
    ) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("target_references.json")
        files["target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "smuggled_nested",
                    "ledger_selector": {"source_name": "somewhere", "value": 123.0},
                    "entity": "person",
                    "measure": "people",
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="values live in Ledger"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_loads_as_value_free_policy(self, tmp_path) -> None:
        files = _package_with_schema2_targets()
        package_dir = _write_package(tmp_path, files)

        spec = load_country_spec(package_dir)

        assert spec.target_profile["schema_version"] == 2
        assert (
            spec.target_profile["criticality_tiers"]["demography_release"][
                "relative_tolerance"
            ]
            == 0.02
        )
        assert spec.target_references[0].period_match_policy == "exact"

    @pytest.mark.parametrize("schema_version", [True, 1.0])
    def test_target_profile_refuses_non_integer_schema_version(
        self, tmp_path, schema_version
    ) -> None:
        files = _package_with_schema2_targets()
        files["target_references.json"]["target_profile"]["schema_version"] = (
            schema_version
        )
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="schema_version must be an integer"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_refuses_invalid_tolerance(self, tmp_path) -> None:
        files = _package_with_schema2_targets()
        files["target_references.json"]["target_profile"]["criticality_tiers"][
            "demography_release"
        ]["relative_tolerance"] = 0.0
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="finite number in \\(0, 1\\]"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_refuses_unknown_reference_tier(
        self, tmp_path
    ) -> None:
        files = _package_with_schema2_targets()
        files["target_references.json"]["target_references"][0]["metadata"][
            "criticality_tier"
        ] = "undeclared"
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="unknown criticality_tier"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_refuses_income_offset_drift(self, tmp_path) -> None:
        files = _package_with_schema2_targets()
        basis = files["target_references.json"]["target_profile"]["basis_periods"][
            "population_2023"
        ]
        basis["survey_year"] = 2023
        basis["income_reference_offset_years"] = -1
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="does not equal survey_year"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_refuses_reference_period_drift(
        self, tmp_path
    ) -> None:
        files = _package_with_schema2_targets()
        files["target_references.json"]["target_references"][0]["period"] = 2022
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="does not match basis period"):
            load_country_spec(package_dir)

    @pytest.mark.parametrize(
        ("reference_period", "basis_period"),
        [
            ("academic_year_2023_24", "ay2023_24"),
            ("ay_2023_24", "academic-year-2023-2024"),
            ("academic_year_1999_00", "ay1999_2000"),
        ],
    )
    def test_schema2_target_profile_uses_shared_academic_period_semantics(
        self, tmp_path, reference_period, basis_period
    ) -> None:
        files = _package_with_schema2_targets()
        reference = files["target_references.json"]["target_references"][0]
        reference["period"] = reference_period
        reference["ledger_selector"]["period_type"] = "academic_year"
        basis = files["target_references.json"]["target_profile"]["basis_periods"][
            "population_2023"
        ]
        basis["period"] = basis_period
        basis["fact_period_type"] = "academic_year"
        package_dir = _write_package(tmp_path, files)

        spec = load_country_spec(package_dir)

        assert spec.target_references[0].period == reference_period

    @pytest.mark.parametrize(
        ("reference_period", "basis_period"),
        [
            ("academic_year_2023_24", "ay2023_25"),
            ("academic_year_2023_24", "ay2023_04"),
            ("academic_year_2023_24", "ay2023"),
            ("academic_year_2023_25", "academic_year_2023_25"),
        ],
    )
    def test_schema2_target_profile_preserves_academic_period_range_end(
        self, tmp_path, reference_period, basis_period
    ) -> None:
        files = _package_with_schema2_targets()
        reference = files["target_references.json"]["target_references"][0]
        reference["period"] = reference_period
        reference["ledger_selector"]["period_type"] = "academic_year"
        basis = files["target_references.json"]["target_profile"]["basis_periods"][
            "population_2023"
        ]
        basis["period"] = basis_period
        basis["fact_period_type"] = "academic_year"
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="does not match basis period"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_refuses_typed_period_kind_drift(
        self, tmp_path
    ) -> None:
        files = _package_with_schema2_targets()
        files["target_references.json"]["target_references"][0]["period"] = "ay2023_24"
        basis = files["target_references.json"]["target_profile"]["basis_periods"][
            "population_2023"
        ]
        basis["period"] = "academic_year_2023_24"
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="implies period type 'academic_year'"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_refuses_fact_period_type_drift(
        self, tmp_path
    ) -> None:
        files = _package_with_schema2_targets()
        files["target_references.json"]["target_references"][0]["ledger_selector"][
            "period_type"
        ] = "tax_year"
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="fact_period_type"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_requires_explicit_projection_policy(
        self, tmp_path
    ) -> None:
        files = _package_with_schema2_targets()
        files["target_references.json"]["target_references"][0]["assertion_policy"] = (
            "observed_only"
        )
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="allow_source_projection"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_requires_subnational_vintage_binding(
        self, tmp_path
    ) -> None:
        files = _package_with_schema2_targets()
        reference = files["target_references.json"]["target_references"][0]
        reference["ledger_selector"]["geography_level"] = "nuts1"
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="same non-empty geography_vintage"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_refuses_subnational_vintage_drift(
        self, tmp_path
    ) -> None:
        files = _package_with_schema2_targets()
        reference = files["target_references.json"]["target_references"][0]
        reference["ledger_selector"].update(
            {"geography_level": "nuts1", "geography_vintage": "NUTS_2024"}
        )
        reference["metadata"]["geography_vintage"] = "NUTS_2021"
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="same non-empty geography_vintage"):
            load_country_spec(package_dir)

    def test_schema2_target_profile_requires_every_calibration_family(
        self, tmp_path
    ) -> None:
        files = _package_with_schema2_targets()
        files["target_references.json"]["target_profile"]["required_families"].append(
            "income_tax"
        )
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="no calibration reference"):
            load_country_spec(package_dir)

    def test_sum_target_reference_roundtrips(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("target_references.json")
        files["target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "summed",
                    "ledger_selector": {"source_name": "somewhere"},
                    "value_operation": "sum",
                    "entity": "person",
                    "measure": "people",
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)

        spec = load_country_spec(package_dir)

        assert spec.target_references[0].value_operation == "sum"

    def test_local_target_reference_roundtrips_with_crosswalk_roster(
        self, tmp_path
    ) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].extend(
            ["local_area_crosswalk.json", "local_target_references.json"]
        )
        files["local_area_crosswalk.json"] = {
            "country": "xx",
            "levels": {
                "constituency": {
                    "expected_vintage": "test_vintage",
                    "area_ids": ["A1"],
                }
            },
        }
        files["local_target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "ons.age.0_10@A1",
                    "ledger_selector": {
                        "source_name": "ons",
                        "source_measure_id": "population",
                        "geography_level": "constituency",
                        "geography_id": "A1",
                    },
                    "value_operation": "sum",
                    "entity": "person",
                    "measure": "age/0_10",
                    "metadata": {"contract_target_id": "ons.age.0_10"},
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)

        spec = load_country_spec(package_dir)

        assert spec.target_references == ()
        assert len(spec.local_target_references) == 1
        assert spec.local_target_references[0].name == "ons.age.0_10@A1"

    def test_local_target_reference_refuses_unknown_roster_area(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].extend(
            ["local_area_crosswalk.json", "local_target_references.json"]
        )
        files["local_area_crosswalk.json"] = {
            "country": "xx",
            "levels": {
                "constituency": {
                    "expected_vintage": "test_vintage",
                    "area_ids": ["A1"],
                }
            },
        }
        files["local_target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "ons.age.0_10@A2",
                    "ledger_selector": {
                        "source_name": "ons",
                        "source_measure_id": "population",
                        "geography_level": "constituency",
                        "geography_id": "A2",
                    },
                    "entity": "person",
                    "measure": "age/0_10",
                    "metadata": {"contract_target_id": "ons.age.0_10"},
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="test_vintage"):
            load_country_spec(package_dir)

    def test_local_target_reference_refuses_unpinned_name(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].extend(
            ["local_area_crosswalk.json", "local_target_references.json"]
        )
        files["local_area_crosswalk.json"] = {
            "country": "xx",
            "levels": {
                "constituency": {
                    "expected_vintage": "test_vintage",
                    "area_ids": ["A1"],
                }
            },
        }
        files["local_target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "ons.age.0_10",
                    "ledger_selector": {
                        "source_name": "ons",
                        "source_measure_id": "population",
                        "geography_level": "constituency",
                        "geography_id": "A1",
                    },
                    "entity": "person",
                    "measure": "age/0_10",
                    "metadata": {"contract_target_id": "ons.age.0_10"},
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="target_id@geography_id"):
            load_country_spec(package_dir)

    def test_restricted_licence_requires_a_private_repo(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("release_contract.json")
        files["release_contract.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "builder": "populace-xx",
            "hf": {
                "artifact_repo": "policyengine/populace-xx",
                "private": False,
                "staging_repo": "policyengine/populace-xx-staging",
            },
            "dataset_filename_template": "microcosm_xx_{year}.h5",
            "required_release_files": ["release_manifest.json"],
            "boundary": {"private": ["microcosm_xx_{year}.h5"], "public": []},
            "licence": {"name": "restricted survey", "restricted": True},
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="requires a private artifact repo"):
            load_country_spec(package_dir)

    def test_ordinal_version_tokens_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("release_contract.json")
        files["release_contract.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "builder": "populace-xx-v2",
            "hf": {
                "artifact_repo": "policyengine/populace-xx-private",
                "private": True,
                "staging_repo": "policyengine/populace-xx-staging",
            },
            "dataset_filename_template": "microcosm_xx_{year}.h5",
            "required_release_files": ["release_manifest.json"],
            "boundary": {"private": ["microcosm_xx_{year}.h5"], "public": []},
            "licence": {"name": "restricted survey", "restricted": True},
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="ordinal version token"):
            load_country_spec(package_dir)

    def test_embedded_ordinal_version_tokens_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("release_contract.json")
        files["release_contract.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "builder": "populace_xx_v2_staging",
            "hf": {
                "artifact_repo": "policyengine/populace-xx-private",
                "private": True,
                "staging_repo": "policyengine/populace-xx-staging",
            },
            "dataset_filename_template": "microcosm_xx_{year}.h5",
            "required_release_files": ["release_manifest.json"],
            "boundary": {"private": ["microcosm_xx_{year}.h5"], "public": []},
            "licence": {"name": "restricted survey", "restricted": True},
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="ordinal version token"):
            load_country_spec(package_dir)

    def test_version_like_substrings_without_ordinal_tokens_load(
        self, tmp_path
    ) -> None:
        # "sha-v2x" is not an ordinal token: the digits run into a letter.
        files = _minimal_package()
        files["country_package.json"]["resources"].append("release_contract.json")
        files["release_contract.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "builder": "populace-xx-sha-v2x",
            "hf": {
                "artifact_repo": "policyengine/populace-xx-private",
                "private": True,
                "staging_repo": "policyengine/populace-xx-staging",
            },
            "dataset_filename_template": "microcosm_xx_{year}.h5",
            "required_release_files": ["release_manifest.json"],
            "boundary": {"private": ["microcosm_xx_{year}.h5"], "public": []},
            "licence": {"name": "restricted survey", "restricted": True},
        }
        package_dir = _write_package(tmp_path, files)
        spec = load_country_spec(package_dir)
        assert spec.release_contract is not None

    def test_geography_vintage_policy_must_be_error(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("geography_spine.json")
        files["geography_spine.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "geography_spine": {
                "stage": "clone_assign",
                "method": "clone_assign_uniform",
                "geography_level": "area",
                "code_system": "xx_codes",
                "code_column": "area_code",
                "vintage": "2025",
                "vintage_policy": "warn",
                "clones_per_record": 2,
                "collision_avoidance": True,
                "constrain_to_column": "",
                "assignment_source": {
                    "survey": "Test census",
                    "source": "https://example.test/census",
                },
            },
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="vintage_policy"):
            load_country_spec(package_dir)


@pytest.mark.parametrize(
    ("period_type", "invalid_period"),
    [
        ("academic_year", "2023_25"),
        ("academic_year", "2023_2025"),
        ("academic_year", "2023_04"),
        ("tax_year", "2023_25"),
        ("calendar_year", "2023_2025"),
        ("fiscal_year", "2023_00"),
        ("month", "2023_00"),
        ("month", "2023_13"),
        ("month", "2023_24"),
        ("month", "2023_2024"),
        ("month", "2023"),
    ],
)
def test_schema2_period_contract_refuses_equal_malformed_untyped_labels(
    tmp_path, period_type, invalid_period
) -> None:
    files = _package_with_schema2_targets()
    reference = files["target_references.json"]["target_references"][0]
    reference["period"] = invalid_period
    reference["ledger_selector"]["period_type"] = period_type
    basis = files["target_references.json"]["target_profile"]["basis_periods"][
        "population_2023"
    ]
    basis["period"] = invalid_period
    basis["fact_period_type"] = period_type
    package_dir = _write_package(tmp_path, files)

    with pytest.raises(ValueError, match="does not match basis period"):
        load_country_spec(package_dir)


@pytest.mark.parametrize(
    ("period_type", "reference_period", "basis_period"),
    [
        ("tax_year", 2023, "ty2023"),
        ("calendar_year", "2023", "calendar_year_2023"),
        ("fiscal_year", "2023_24", "fy2023_2024"),
        ("academic_year", "2003_04", "ay2003_04"),
        ("academic_year", "1999_00", "academic_year_1999_2000"),
        ("academic_year", 2023, "ay2023"),
        ("month", "2003_04", "month_2003_04"),
        ("month", "2023_1", "month_2023_01"),
        ("academic_year", "publisher_revision_a", "publisher_revision_a"),
        ("reporting_window", "publisher_release_a", "publisher_release_a"),
    ],
)
def test_schema2_period_contract_preserves_valid_aliases_and_opaque_labels(
    tmp_path, period_type, reference_period, basis_period
) -> None:
    files = _package_with_schema2_targets()
    reference = files["target_references.json"]["target_references"][0]
    reference["period"] = reference_period
    reference["ledger_selector"]["period_type"] = period_type
    basis = files["target_references.json"]["target_profile"]["basis_periods"][
        "population_2023"
    ]
    basis["period"] = basis_period
    basis["fact_period_type"] = period_type
    package_dir = _write_package(tmp_path, files)

    loaded = load_country_spec(package_dir)

    assert loaded.target_references[0].period == reference_period


@pytest.mark.parametrize(
    "identifier_field", ["ledger_fact_key", "ledger_source_record_id"]
)
@pytest.mark.parametrize("fact_vintage", ["nis_2025", "nis_2024", None])
def test_schema2_be_geography_vintage_contract_survives_identifier_resolution(
    tmp_path, identifier_field, fact_vintage
) -> None:
    package_dir = tmp_path / "be"
    shutil.copytree(COUNTRY_PACKAGE_ROOT / "be", package_dir)
    target_path = package_dir / "target_references.json"
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    raw_reference = next(
        row
        for row in payload["target_references"]
        if row["name"] == "statbel_fiscal_income_by_commune"
    )
    raw_reference["metadata"]["activation_status"] = "active"
    raw_reference["ledger_selector"]["geography_id"] = "21004"
    fact_key = "ledger.aggregate_fact.v2:synthetic-be-commune"
    record_id = "statbel_fiscal_income.synthetic-commune"
    raw_reference[identifier_field] = (
        fact_key if identifier_field == "ledger_fact_key" else record_id
    )
    target_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_country_spec(package_dir)
    reference = next(
        row
        for row in loaded.target_references
        if row.name == "statbel_fiscal_income_by_commune"
    )
    fact = {
        "aggregate_fact_key": fact_key,
        "lineage": {"source_record_id": record_id},
        "value": 10.0,  # Synthetic resolver probe, not a Belgian source value.
        "period": {"type": "tax_year", "value": 2022},
        "geography": {"level": "commune", "id": "21004"},
        "entity": {"name": "household"},
        "observed_measure": {
            "source_name": "statbel_fiscal_income",
            "source_measure_id": "taxable_income",
            "unit": "eur",
        },
        "aggregation": {"method": "sum"},
    }
    if fact_vintage is not None:
        fact["geography"]["vintage"] = fact_vintage

    if fact_vintage != "nis_2025":
        with pytest.raises(ValueError, match="vintage"):
            compile_ledger_target_references([fact], [reference], country="be")
    else:
        (spec,) = compile_ledger_target_references(
            [fact], [reference], country="be"
        ).specs
        assert spec.metadata["ledger_geography_vintage"] == "nis_2025"
        assert spec.value == 10.0
