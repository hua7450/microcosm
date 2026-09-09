"""Shared-core country compile proofs for the F0 CountrySpec seam."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.spec_engine import load_bundle
from microcosm.build.spec_engine.compiler_ir import compile_spec
from microcosm.build.spec_engine.resolver import (
    F0_CONTRACT_ONLY_KERNEL_IDS,
    F0_IMPLEMENTED_KERNEL_IDS,
    F0_KERNEL_REGISTRY,
    KernelRegistry,
)

EXPECTED_RESOURCES = {
    "bundle",
    "catalogs",
    "geography",
    "sources",
    "spine",
    "vintages",
}
AM_SPEC_SHA256 = "b081464d575f63e54e13ce7a2a7c9009ed0bea62289b4f9ab5dd3bc214544b16"


@pytest.mark.parametrize(
    ("country", "expected_spec_sha256", "expected_columns", "expected_entities"),
    [
        (
            "am",
            AM_SPEC_SHA256,
            {
                "household.household_id",
                "person.age",
                "person.marz_code",
                "person.person_id",
                "person.sex",
            },
            {"household", "person"},
        ),
        (
            "be",
            "70fc6b15de937f3d88c173d586bbe4856070644eb9e1c850bcdc93501a3534f9",
            {
                "household.household_id",
                "person.person_id",
                "person.region_nuts1",
            },
            {"household", "person"},
        ),
        (
            "uk",
            "6a8003d9246f8e4c7bd60b95f3f9ff599dfb5e4ffdc4323f6603f75a46f611b6",
            {
                "benunit.benunit_id",
                "household.household_id",
                "household.region",
                "person.person_id",
            },
            {"benunit", "household", "person"},
        ),
    ],
)
def test_country_bundle_loads_once_and_compiles_through_the_shared_core(
    country: str,
    expected_spec_sha256: str,
    expected_columns: set[str],
    expected_entities: set[str],
) -> None:
    if country == "us":
        pytest.importorskip(
            "policyengine_us",
            reason="US compile proof reads the live engine ABI",
            exc_type=ModuleNotFoundError,
        )
    direct = load_bundle(country)
    country_spec = load_country_spec(country)

    assert country_spec.resolved_spec is not None
    assert country_spec.resolved_spec.spec_sha256 == direct.spec_sha256
    assert direct.spec_sha256 == expected_spec_sha256

    compiled = compile_spec(direct)
    assert set(compiled.resources_wire()) == EXPECTED_RESOURCES
    assert not compiled.producer_graph.present
    assert compiled.nodes == ()
    assert {column.key for column in direct.columns} == expected_columns
    assert {column.entity.id for column in direct.columns} == expected_entities


def test_country_bundles_exercise_distinct_support_and_geography_kinds() -> None:
    am = compile_spec(load_bundle("am"))
    be = compile_spec(load_bundle("be"))
    uk = compile_spec(load_bundle("uk"))

    assert am.resource("spine")["support_roles"] == [
        {"id": "populace_us_donor_base", "kind": "none"}
    ]
    assert be.resource("spine")["support_roles"] == [
        {"id": "silc_base", "kind": "none"}
    ]
    assert uk.resource("spine")["support_roles"] == [
        {
            "id": "spi_income_support",
            "kind": "synthetic_prior_replacement",
            "clone_index": 1,
        }
    ]
    assert am.resource("geography")["assignment"]["kernels"] == {
        "assign": "kernel:clone_assign_communities",
        "validate": "kernel:am_community_geography_gate",
    }
    assert be.resource("geography")["assignment"]["kernels"] == {
        "assign": "kernel:clone_assign_communes",
        "validate": "kernel:be_commune_geography_gate",
    }
    assert uk.resource("geography")["assignment"]["kernels"] == {
        "assign": "kernel:assign_uk_geography_ladder",
        "validate": "kernel:uk_geography_ladder_gate",
    }


def test_country_kernel_contract_ids_are_closed_in_the_compiler_registry() -> None:
    country_contract_ids = {
        "am_community_geography_gate",
        "assign_am_marz",
        "clone_assign_communities",
        "load_populace_us_support_pool",
        "silc_load",
        "clone_assign_communes",
        "be_commune_geography_gate",
        "load_uk_national_frame",
        "assign_uk_geography_ladder",
        "uk_geography_ladder_gate",
    }

    assert country_contract_ids == F0_CONTRACT_ONLY_KERNEL_IDS
    assert country_contract_ids <= F0_KERNEL_REGISTRY.ids
    assert all(F0_KERNEL_REGISTRY.contains(value) for value in country_contract_ids)
    assert not any(
        F0_KERNEL_REGISTRY.has_implementation(value) for value in country_contract_ids
    )


def test_country_contract_ids_do_not_change_the_implementation_digest() -> None:
    implemented_only = KernelRegistry.from_ids(F0_IMPLEMENTED_KERNEL_IDS)

    assert F0_KERNEL_REGISTRY.implemented_ids == implemented_only.ids
    assert (
        F0_KERNEL_REGISTRY.implementation_sha256
        == implemented_only.implementation_sha256
    )


def test_am_generation_zero_views_come_from_the_country_spec_seam() -> None:
    spec = load_country_spec("am")

    assert spec.sources is not None
    assert tuple(spec.sources.stage_map()) == (
        "load_populace_us_support_pool",
        "assign_am_marz",
    )
    assert spec.geography_spine is not None
    assert spec.geography_spine.geography_spine.stage == "clone_assign_communities"
    assert spec.gates is not None
    assert spec.release_contract is not None
    assert not spec.release_contract.artifact_repo_private
    assert {row.path for row in spec.resource_rows if row.kind == "legacy_json"} == {
        "gates.json",
        "geography_spine.json",
        "release_contract.json",
        "source_stages.json",
        "target_references.json",
    }


def test_be_generation_zero_views_still_come_from_the_country_spec_seam() -> None:
    spec = load_country_spec("be")

    assert spec.sources is not None
    assert tuple(spec.sources.stage_map()) == ("silc_load",)
    assert spec.geography_spine is not None
    assert spec.geography_spine.geography_spine.stage == "clone_assign_communes"
    assert spec.gates is not None
    assert spec.release_contract is not None
    assert spec.release_contract.artifact_repo_private
    assert {row.path for row in spec.resource_rows if row.kind == "legacy_json"} == {
        "gates.json",
        "geography_spine.json",
        "release_contract.json",
        "source_stages.json",
        "target_references.json",
    }


@pytest.mark.parametrize(
    ("before", "after", "field"),
    [
        (
            "kernel:clone_assign_communes",
            "kernel:assign_uk_geography_ladder",
            "assignment/kernels/assign",
        ),
        ("anchor: commune", "anchor: municipality", "assignment/anchor"),
        ("- commune_nis", "- municipality_nis", "assignment/derive"),
        (
            "commune: vintage:be_nis_2025",
            "commune: vintage:be_nuts1_2025",
            "assignment/layer_vintages",
        ),
        (
            "- geography_vintage_exact",
            "- geography_vintage_warning",
            "assignment/assertions/geography_vintage_exact",
        ),
        (
            "- vintage_refusal",
            "- vintage_warning",
            "assignment/validation/vintage_refusal",
        ),
        (
            "universe: commune_within_nuts1",
            "universe: commune_nationally",
            "assignment/draw/universe",
        ),
        (
            "- source_nuts1_preserved",
            "- source_nuts1_warning",
            "assignment/assertions/source_nuts1_preserved",
        ),
    ],
)
def test_be_typed_geography_drift_from_legacy_evidence_refuses(
    tmp_path: Path,
    before: str,
    after: str,
    field: str,
) -> None:
    package = tmp_path / "be"
    source = Path(__file__).parents[1] / "src/microcosm/build/be"
    shutil.copytree(source, package)
    geography = package / "spec/geography.yaml"
    authored = geography.read_text(encoding="utf-8")
    assert authored.count(before) == 1
    geography.write_text(authored.replace(before, after), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_country_spec(package)
    assert "typed geography compatibility drift" in str(error.value)
    assert field in str(error.value)


def test_be_smoke_build_refuses_without_gated_inputs_and_stage_bindings() -> None:
    """Compile succeeds, but the repository has no honest runnable BE stage map."""

    spec = load_country_spec("be")
    compiled = compile_spec(spec.resolved_spec)
    assert compiled.spec_binding.country == "be"

    silc_stage = spec.sources.stage_map()["silc_load"]
    assert {artifact["kind"] for artifact in silc_stage.artifacts} == {
        "restricted_microdata"
    }
    build_package = Path(__file__).parents[1] / "src/microcosm/build"
    assert not (build_package / "be_runtime").exists()

    with pytest.raises(
        ValueError,
        match=(
            r"missing \['silc_load', 'clone_assign_communes'\].*"
            r"There are no stubs or fallbacks"
        ),
    ):
        country_stage_plan(spec, {})


def test_am_smoke_build_refuses_without_harvests_and_stage_bindings() -> None:
    """Compile succeeds, but engine-free v1 has no runnable AM stage map."""

    spec = load_country_spec("am")
    compiled = compile_spec(spec.resolved_spec)
    assert compiled.spec_binding.country == "am"

    stages = spec.sources.stage_map()
    assert {
        artifact["kind"]
        for artifact in stages["load_populace_us_support_pool"].artifacts
    } == {"public_microdata"}
    assert {artifact["kind"] for artifact in stages["assign_am_marz"].artifacts} == {
        "public_aggregated_counts"
    }
    build_package = Path(__file__).parents[1] / "src/microcosm/build"
    assert not (build_package / "am_runtime").exists()

    with pytest.raises(
        ValueError,
        match=(
            r"missing \['load_populace_us_support_pool', 'assign_am_marz', "
            r"'clone_assign_communities'\].*There are no stubs or fallbacks"
        ),
    ):
        country_stage_plan(spec, {})
