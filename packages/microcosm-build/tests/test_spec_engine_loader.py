from __future__ import annotations

import json
from dataclasses import replace
from importlib.resources import files as resource_files
from pathlib import Path

import pytest

import microcosm.build.spec_engine.loader as loader_module
from microcosm.build.spec_engine import (
    KernelRegistry,
    ResourceKind,
    SpecResolutionError,
    SpecValidationError,
    bundle_lock_bytes,
    bundle_lock_payload,
    load_bundle,
    load_schema_registry,
)
from microcosm.build.spec_engine.errors import SpecParseError
from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL, SeedProtocol

ZERO_SHA = "0" * 64
SELECTION_KERNEL_IDS = [
    "assert_exact_k_support",
    "exact_k_ladder_manifest_payload",
    "exact_k_pcg64_rng",
    "select_exact_k",
]


def _write_bundle(root: Path, files: dict[str, str], rows: list[dict]) -> Path:
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "country": root.name,
        "resources": rows,
    }
    (root / "country_package.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _row(path: str, kind: str) -> dict[str, str]:
    return {"path": path, "kind": kind, "schema_id": f"{kind}.schema.json"}


def _rich_minimal(root: Path, *, note: str = "first", store: str = "local:a") -> Path:
    files = {
        "bundle.yaml": (
            "country: xx\n"
            "dataset_run: {target_period: 2024}\n"
            "identity_generation: 1\n"
            "seed_protocol: legacy-v1\n"
            "status: documentation only\n"
        ),
        "vintages.yaml": (
            "records:\n"
            "  - id: ty2024\n"
            "    kind: target_period_ref\n"
            "    authority_ref:\n"
            "      kind: dataset_run\n"
            "      pointer: /dataset_run/target_period\n"
            "    compatible_with: [vintage:release2024]\n"
            "  - id: release2024\n"
            "    kind: release_series_ref\n"
            "    authority_ref:\n"
            "      kind: publication_release\n"
            "      pointer: /release/line/value\n"
            "    compatible_with: [vintage:ty2024]\n"
        ),
        "catalogs.yaml": (
            "columns:\n"
            "  - key: person.age\n"
            "    contract:\n"
            "      entity: person\n"
            "      dtype: int64\n"
            "      unit: count\n"
            "      period: vintage:ty2024\n"
            "      nullable: false\n"
            "      domain: demographics\n"
            "      public_stability: public_stable\n"
            "    docs:\n"
            f"      description: {note}\n"
            "      citations: [official]\n"
        ),
        "publication.yaml": (
            "attempts:\n"
            "  model: append_only_events_then_terminal_seal\n"
            "  terminal_states: [landed, failed, expired]\n"
            "promotion:\n"
            "  latest_flip: human_gate\n"
            "  idempotency: required_key\n"
            "  recovery: [seal_ok_append_fail]\n"
            "release:\n"
            "  line:\n"
            "    value: microcosm-xx-2024\n"
            "    normative: true\n"
            "    note: documentation note\n"
            "  pattern: '{line}-stacked-f001-s0'\n"
            "  rung_fractions:\n"
            "    - {fraction: 0.01, token: f001, percent_basis_points: 100}\n"
            "    - {fraction: 0.04, token: f004, percent_basis_points: 400}\n"
            "    - {fraction: 0.10, token: f010, percent_basis_points: 1000}\n"
            "    - {fraction: 0.25, token: f025, percent_basis_points: 2500}\n"
            "    - {fraction: 1.0, token: f100, percent_basis_points: 10000}\n"
            "audit_chain:\n"
            "  kind: strict_linear\n"
            f"  store: {store}\n"
            "release_graph:\n"
            "  relations: [derived_from]\n"
        ),
        "selection.yaml": resource_files("microcosm.build.us")
        .joinpath("spec/selection.yaml")
        .read_text(encoding="utf-8"),
    }
    rows = [
        _row("bundle.yaml", "bundle"),
        _row("vintages.yaml", "vintages"),
        _row("catalogs.yaml", "catalogs"),
        _row("publication.yaml", "publication"),
        _row("selection.yaml", "selection"),
    ]
    return _write_bundle(root, files, rows)


def test_loads_typed_domains_injects_defaults_and_emits_valid_lock(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    spec = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )

    assert spec.country == "xx"
    assert (
        spec.resource(ResourceKind.CATALOGS).domain.to_wire()["columns"][0]["contract"][
            "entity"
        ]
        == "person"
    )
    selection = spec.resource(ResourceKind.SELECTION).domain.to_wire()
    assert selection["exact_k"]["k"]["precedence"] == ("run_request_overrides_default")
    assert spec.columns[0].key == "person.age"
    assert spec.spec_binding.attestation == "mirror-attested"
    lock = bundle_lock_payload(spec)
    load_schema_registry().validate(lock, "locks.schema.json#/$defs/bundle_lock")
    assert bundle_lock_bytes(spec) == bundle_lock_bytes(spec)
    assert spec.seed_protocol is LEGACY_V1_PROTOCOL
    assert lock["seed_protocol"] == spec.seed_protocol.to_wire()


def test_resolved_selector_drives_lock_and_derived_v2_fails_closed(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    spec = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    first = spec.seed_protocol.sites[0]
    changed = SeedProtocol(
        id=spec.seed_protocol.id,
        implementation_id=spec.seed_protocol.implementation_id,
        kernels=spec.seed_protocol.kernels,
        sites=(
            replace(first, reset_boundary="test_only_changed_boundary"),
            *spec.seed_protocol.sites[1:],
        ),
    )
    changed_spec = replace(spec, seed_protocol=changed)
    changed_lock = bundle_lock_payload(changed_spec)
    assert changed_lock["seed_protocol"] == changed.to_wire()
    assert (
        changed_lock["seed_protocol"]["implementation_sha256"]
        != bundle_lock_payload(spec)["seed_protocol"]["implementation_sha256"]
    )

    bundle_path = root / "bundle.yaml"
    bundle_path.write_text(
        bundle_path.read_text(encoding="utf-8").replace("legacy-v1", "derived-v2"),
        encoding="utf-8",
    )
    with pytest.raises(
        SpecResolutionError, match="unsupported F0 protocol 'derived-v2'"
    ):
        load_bundle(root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS))


def test_resolved_seed_protocol_expansion_is_spec_normative(
    tmp_path, monkeypatch
) -> None:
    root = _rich_minimal(tmp_path / "xx")
    first = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    original_resolver = loader_module.resolve_cross_references

    def changed_resolver(*args, **kwargs):
        resolved = original_resolver(*args, **kwargs)
        protocol = resolved.seed_protocol
        changed = SeedProtocol(
            id=protocol.id,
            implementation_id=protocol.implementation_id,
            kernels=protocol.kernels,
            sites=(
                replace(
                    protocol.sites[0],
                    reset_boundary="test_only_normative_protocol_mutation",
                ),
                *protocol.sites[1:],
            ),
        )
        return replace(resolved, seed_protocol=changed)

    monkeypatch.setattr(loader_module, "resolve_cross_references", changed_resolver)
    second = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )

    assert second.package_fingerprint == first.package_fingerprint
    assert second.spec_sha256 != first.spec_sha256
    assert (
        second.seed_protocol.implementation_sha256
        != first.seed_protocol.implementation_sha256
    )


def test_semantic_hash_has_golden_vector_and_surface_separation(tmp_path) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    first = load_bundle(
        _rich_minimal(tmp_path / "xx", note="first", store="local:a"),
        kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS),
    )
    # Pin the domain separator, normalization rules, schema-set receipt, and
    # exact normative projection as one reviewable golden vector.
    assert first.spec_sha256 == (
        "8761376e420bf184dcd4623185a876e4453f7d0d567ed687009ac3533752a88c"
    )

    second_root = _rich_minimal(tmp_path / "xy", note="second", store="local:b")
    manifest_path = second_root / "country_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["country"] = "xx"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (second_root / "bundle.yaml").write_text(
        (second_root / "bundle.yaml")
        .read_text(encoding="utf-8")
        .replace("country: xx", "country: xx"),
        encoding="utf-8",
    )
    # Directory name is part of the CountrySpec seam, but load_bundle's typed
    # manifest is the country authority and supports fixture locations.
    second = load_bundle(
        second_root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    assert second.spec_sha256 == first.spec_sha256
    assert second.documentation_sha256 != first.documentation_sha256
    assert second.package_fingerprint != first.package_fingerprint
    assert second.surfaces.operational != first.surfaces.operational


def test_manifest_declared_set_reordering_does_not_change_spec_hash(tmp_path) -> None:
    first_root = _rich_minimal(tmp_path / "xx")
    first = load_bundle(
        first_root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    second_root = _rich_minimal(tmp_path / "xy")
    manifest_path = second_root / "country_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["country"] = "xx"
    manifest["resources"] = list(reversed(manifest["resources"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    second = load_bundle(
        second_root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    assert second.spec_sha256 == first.spec_sha256
    assert second.package_fingerprint != first.package_fingerprint


def _append_legacy_json(root: Path, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")
    manifest_path = root / "country_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"].append(
        {"path": name, "kind": "legacy_json", "schema_id": "legacy_json"}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_legacy_json_resources_load_through_the_strict_json_path(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    _append_legacy_json(root, "extras.json", '{"rows": [1, 2]}\n')
    spec = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    assert spec.resource(ResourceKind.LEGACY_JSON).domain.to_wire() == {"rows": [1, 2]}


def test_legacy_json_resource_in_yaml_syntax_refuses(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    _append_legacy_json(root, "extras.json", "rows: [1, 2]\n")
    with pytest.raises(SpecParseError, match=r"extras\.json.*invalid JSON"):
        load_bundle(root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS))


def _cross_ref_bundle(root: Path, *, source_ref: str = "survey") -> Path:
    files = {
        "bundle.yaml": (
            "country: xx\n"
            "dataset_run: {target_period: 2024}\n"
            "identity_generation: 1\n"
            "seed_protocol: legacy-v1\n"
        ),
        "sources.yaml": (
            "sources:\n"
            "  - id: survey\n"
            "    role: survey\n"
            f"    sha256: '{ZERO_SHA}'\n"
            "    loader: kernel:survey_loader\n"
            "    vintages: [vintage:survey2024]\n"
            "    vintage_authorities:\n"
            "      - id: survey2024\n"
            "        kind: survey_period\n"
            "        value: 2024\n"
        ),
        "vintages.yaml": (
            "records:\n"
            "  - id: survey2024\n"
            "    kind: survey_period_ref\n"
            "    authority_ref:\n"
            "      kind: source_record\n"
            "      source: source:survey\n"
            "      authority: survey2024\n"
            "    compatible_with: [vintage:target2024]\n"
            "  - id: target2024\n"
            "    kind: target_period_ref\n"
            "    authority_ref:\n"
            "      kind: dataset_run\n"
            "      pointer: /dataset_run/target_period\n"
            "    compatible_with: [vintage:survey2024]\n"
        ),
        "spine.yaml": (
            "channels:\n"
            "  - id: survey_channel\n"
            f"    source: {source_ref}\n"
            "    observed_geography: state\n"
            "assembly:\n"
            "  mass_anchor_channel: survey_channel\n"
            "  shared_dtype_policy: canonical_string_storage\n"
            "support_roles: []\n"
        ),
        "geography.yaml": (
            "phase: legacy\n"
            "assignment:\n"
            "  anchor: puma\n"
            "  order: legacy_post_transfer\n"
            "  kernels:\n"
            "    assign: kernel:geo_assign\n"
            "    validate: kernel:geo_validate\n"
            "  draw: {}\n"
            "  identified_county_source: source:survey\n"
            "  derive: [state_fips]\n"
            "  assertions: [observed_preserved]\n"
            "  ladder_source: puma_ladder\n"
            "  seed: stream:geography_legacy\n"
        ),
    }
    rows = [
        _row("bundle.yaml", "bundle"),
        _row("sources.yaml", "sources"),
        _row("vintages.yaml", "vintages"),
        _row("spine.yaml", "spine"),
        _row("geography.yaml", "geography"),
    ]
    return _write_bundle(root, files, rows)


def test_cross_reference_resolution_accepts_selected_registry(tmp_path) -> None:
    spec = load_bundle(
        _cross_ref_bundle(tmp_path / "xx"),
        kernel_registry=KernelRegistry.from_ids(
            ["survey_loader", "geo_assign", "geo_validate", "unused_library_kernel"]
        ),
    )
    assert {reference.namespace for reference in spec.references} == {
        "kernel",
        "source",
        "stream",
        "vintage",
    }


@pytest.mark.parametrize(
    ("source_ref", "message"),
    [("missing", "dangling source reference")],
)
def test_dangling_cross_reference_refuses(
    tmp_path, source_ref: str, message: str
) -> None:
    with pytest.raises(SpecResolutionError, match=message):
        load_bundle(
            _cross_ref_bundle(tmp_path / "xx", source_ref=source_ref),
            kernel_registry=KernelRegistry.from_ids(
                ["survey_loader", "geo_assign", "geo_validate"]
            ),
        )


def test_manifest_path_traversal_refuses_before_io(tmp_path) -> None:
    root = _write_bundle(
        tmp_path / "xx",
        {
            "bundle.yaml": "country: xx\nidentity_generation: 1\nseed_protocol: legacy-v1\n"
        },
        [
            {
                "path": "../bundle.yaml",
                "kind": "bundle",
                "schema_id": "bundle.schema.json",
            }
        ],
    )
    with pytest.raises(
        SpecValidationError, match="normalized and relative|invalid portable"
    ):
        load_bundle(root)
