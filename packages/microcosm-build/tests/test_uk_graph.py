"""UK spine graph declarations, kernels, and structural runtime contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import (
    UK_SPINE_EXCLUSIONS,
    UK_SPINE_STRUCTURAL_STAGES,
    uk_registry,
    uk_spine_graph,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Graph,
    KernelResult,
    Node,
    StructuralDelta,
    compile_graph,
    graph_from_json,
    graph_to_json,
)
from microcosm.graph.population import Population, PopulationError, patch


def _expand_population() -> Population:
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": pd.Series([1, 2], dtype="int64"),
                    "person_benunit_id": pd.Series([100, 200], dtype="int64"),
                    "person_household_id": pd.Series([10, 20], dtype="int64"),
                    "hidden_payload": pd.Series([1.25, 9.5], dtype="float64"),
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": pd.Series([100, 200], dtype="int64"),
                    "capital": pd.Series([4.0, 7.0], dtype="float64"),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": pd.Series([10, 20], dtype="int64"),
                    "region": pd.Series(["LONDON", "WALES"], dtype="string"),
                }
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {
            "household": Weights(
                np.array([1.0, 2.0], dtype=np.float64), WeightKind.DESIGN
            )
        },
        pd.Series(["base", "base"], dtype="string", name="stratum"),
        metadata={"time_period": "2024"},
    )
    return Population.from_frame(frame, "root")


def _expand_node() -> Node:
    return Node(
        id="clone",
        kernel="uk.stage.expand.test@1",
        structural=StructuralDelta.EXPAND,
        base="root",
        params={
            "expand_cells": (("household", "is_clone", "bool"),),
            "expand_weight_entity": "household",
            "expand_weight_kind": "importance",
        },
        mass="conserve",
    )


def _expand_result(*, bad_source: bool = False) -> KernelResult:
    return KernelResult(
        expand={
            "person": pd.Series(
                [99 if bad_source else 1],
                index=pd.Index([3], name="person_id"),
                dtype="int64",
            ),
            "benunit": pd.Series(
                [100],
                index=pd.Index([300], name="benunit_id"),
                dtype="int64",
            ),
            "household": pd.Series(
                [10],
                index=pd.Index([30], name="household_id"),
                dtype="int64",
            ),
        },
        columns={
            ("household", "is_clone"): pd.Series(
                [False, False, True],
                index=pd.Index([10, 20, 30], name="household_id"),
                dtype="bool",
            ),
        },
        weights=Weights(
            np.array([0.5, 2.0, 0.5], dtype=np.float64),
            WeightKind.IMPORTANCE,
        ),
        receipt={
            "frame_mass_log_append": [
                {
                    "entity": "household",
                    "old_total": 3.0,
                    "new_total": 3.0,
                    "declared_factor": None,
                    "reason": "test clone mass is conserved",
                }
            ]
        },
    )


def test_uk_expand_contract_carries_cells_links_weights_and_design_lineage() -> None:
    expanded = patch(_expand_population(), _expand_node(), _expand_result())

    person = expanded.frame.table("person")
    assert person["person_id"].tolist() == [1, 2, 3]
    assert person["person_household_id"].tolist() == [10, 20, 30]
    assert person["hidden_payload"].tolist() == [1.25, 9.5, 1.25]
    assert expanded.frame.table("household")["is_clone"].tolist() == [
        False,
        False,
        True,
    ]
    assert expanded.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(
        expanded.frame.weights_for("household").values,
        np.array([0.5, 2.0, 0.5]),
    )
    np.testing.assert_array_equal(
        expanded.design_weights["household"], np.array([1.0, 2.0, 1.0])
    )
    assert expanded.frame.mass_log[-1].reason == "test clone mass is conserved"
    assert expanded.mass_ledger[-1].operation == "expand"


def test_uk_expand_contract_rejects_unknown_source_ids() -> None:
    with pytest.raises(PopulationError, match="unknown 'person' source ids"):
        patch(_expand_population(), _expand_node(), _expand_result(bad_source=True))


def test_uk_spine_graph_contains_manifest_stages_and_named_exclusions() -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    expected = tuple(
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    )
    graph = uk_spine_graph(spec)
    ids = {node.id for node in graph.nodes}

    # 28 with the #832 uc_reporter_redraw and #685 uc_deduction_attributes
    # stages; the two named exclusions are the certified-pair alternatives,
    # not steps of this pipeline.
    assert len(expected) == 28
    assert UK_SPINE_EXCLUSIONS == {
        "frs_hmrc_retained_leaves",
        "hmrc_spi_income",
    }
    assert set(expected) <= ids
    assert not (UK_SPINE_EXCLUSIONS & ids)
    root_dtypes = {
        (owned.entity, owned.column): owned.dtype
        for owned in graph.node("create_uk_frs").outputs
    }
    assert root_dtypes[("person", "is_uc_claimant")] == "bool"
    assert {
        node.id for node in graph.nodes if node.structural is StructuralDelta.EXPAND
    } == UK_SPINE_STRUCTURAL_STAGES


def test_uk_spine_compile_order_is_derived_from_declared_inputs() -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    expected = tuple(
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    )
    compiled = compile_graph(uk_spine_graph(spec))
    stage_order = tuple(node_id for node_id in compiled.order if node_id in expected)

    assert set(stage_order) == set(expected)
    assert len(stage_order) == len(expected)
    assert all(
        set(compiled.predecessors[node_id]) <= set(compiled.order[:index])
        for index, node_id in enumerate(compiled.order)
    )
    assert all(
        compiled.graph.node(node_id).inputs
        for node_id in expected[1:]
        if node_id not in UK_SPINE_STRUCTURAL_STAGES
    )
    graph = compiled.graph
    reversed_declaration = Graph(
        graph.country, graph.sources, tuple(reversed(graph.nodes))
    )
    assert compile_graph(reversed_declaration).order == compiled.order
    assert "frs_employment" in compiled.predecessors["frs_legacy_proxies"]
    assert "was_wealth" in compiled.predecessors["regional_property_uprating.boundary"]
    assert "regional_property_uprating" in compiled.predecessors["lcfs_consumption"]
    assert "spi_support_channel" in compiled.predecessors["hmrc_spi_income_spine"]


def test_uk_production_graph_binds_split_donor_sources_and_runtime_config() -> None:
    graph = uk_spine_graph(
        source_mode="split",
        sample_fraction=0.1,
        sample_seed=999,
    )

    assert {source.name for source in graph.sources} == {
        "frs",
        "was",
        "lcfs_household",
        "lcfs_person",
        "etb",
        "spi",
        "hmrc_income",
        "hmrc_cgt",
    }
    assert graph.node("lcfs_consumption").sources == (
        "lcfs_household",
        "lcfs_person",
        "was",
    )
    assert graph.node("hmrc_spi_income_spine").sources == (
        "spi",
        "hmrc_income",
    )
    assert graph.node("hmrc_cgt_gains_spine").sources == ("hmrc_cgt",)
    create = graph.node("create_uk_frs")
    assert create.params["sample_fraction"] == 0.1
    assert create.params["sample_seed"] == 999
    assert len(str(create.params["stage_contract_sha256"])) == 64
    assert all(
        len(str(node.params["stage_contract_sha256"])) == 64
        for node in graph.nodes
        if "stage" in node.params
    )


def test_uk_registry_covers_every_kernel_ref_and_hashes_stage_modules() -> None:
    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)

    assert set(registry.refs()) == {node.kernel for node in graph.nodes}
    assert registry.implementation_hash(
        "uk.stage.frs_employment@1"
    ) != registry.implementation_hash("uk.stage.frs_council_tax@1")


def test_uc_relationship_helper_changes_affected_kernel_hashes(monkeypatch) -> None:
    from microcosm.build.uk_runtime import uc_relationships

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    affected_refs = (
        graph.node("create_uk_frs").kernel,
        graph.node("uc_reporter_redraw").kernel,
        graph.node("uc_capital_coherence").kernel,
    )
    control_ref = graph.node("frs_council_tax").kernel
    before = {
        ref: registry.implementation_hash(ref) for ref in (*affected_refs, control_ref)
    }
    helper_path = Path(uc_relationships.__file__).resolve()
    original_read_bytes = Path.read_bytes

    def changed_helper_bytes(path: Path) -> bytes:
        content = original_read_bytes(path)
        if path.resolve() == helper_path:
            return content + b"\n# A helper-only source change.\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_helper_bytes)

    for ref in affected_refs:
        assert registry.implementation_hash(ref) != before[ref]
    assert registry.implementation_hash(control_ref) == before[control_ref]


def test_uk_adapter_source_changes_invalidate_all_consuming_stages(monkeypatch):
    from microcosm.frame.adapters import policyengine_uk

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    stages = (
        "frs_legacy_proxies",
        "frs_education_grant_split",
        "frs_brma",
        "was_wealth",
        "lcfs_consumption",
        "etb_vat",
        "etb_services",
        "uc_reporter_redraw",
    )
    affected = [graph.node(stage).kernel for stage in stages]
    controls = [
        graph.node(stage).kernel
        for stage in ("frs_council_tax", "uc_capital_coherence")
    ]
    before = {ref: registry.implementation_hash(ref) for ref in affected + controls}
    adapter_path = Path(policyengine_uk.__file__).resolve()
    original = Path.read_bytes

    def changed_adapter_bytes(path):
        content = original(path)
        return (
            content + b"\n# adapter-only change\n"
            if path.resolve() == adapter_path
            else content
        )

    monkeypatch.setattr(Path, "read_bytes", changed_adapter_bytes)
    for ref in affected:
        assert registry.implementation_hash(ref) != before[ref]
    for ref in controls:
        assert registry.implementation_hash(ref) == before[ref]


def test_uk_graph_json_round_trip_is_canonical() -> None:
    graph = uk_spine_graph()
    serialized = graph_to_json(graph)

    assert graph_from_json(serialized) == graph
    assert graph_to_json(graph_from_json(serialized)) == serialized


def _mixed_size_population() -> Population:
    """Two households of different size: one person in 10, two in 20."""

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": pd.Series([1, 2, 3], dtype="int64"),
                    "person_benunit_id": pd.Series([100, 200, 200], dtype="int64"),
                    "person_household_id": pd.Series([10, 20, 20], dtype="int64"),
                }
            ),
            "benunit": pd.DataFrame(
                {"benunit_id": pd.Series([100, 200], dtype="int64")}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": pd.Series([10, 20], dtype="int64"),
                    "region": pd.Series(["LONDON", "WALES"], dtype="string"),
                }
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {
            "household": Weights(
                np.array([1.0, 2.0], dtype=np.float64), WeightKind.DESIGN
            )
        },
        pd.Series(["base", "base", "base"], dtype="string", name="stratum"),
        metadata={"time_period": "2024"},
    )
    return Population.from_frame(frame, "root")


def _mass_shifting_expand_result(*, declared: bool) -> KernelResult:
    """Clone the one-person household and move mass onto it from the larger one.

    Household mass is conserved (1 + 2 == 0.5 + 1.5 + 1.0) while person mass is
    not (1 + 2*2 = 5 against 0.5 + 1.5*2 + 1.0 = 4.5): the shape of the SPI
    support channel, whose prior-mass allocation moves half the household mass
    onto stacked households whose composition differs from the FRS households
    it is taken from.
    """

    receipt: dict[str, object] = {
        "frame_mass_log_append": [
            {
                "entity": "household",
                "old_total": 3.0,
                "new_total": 3.0,
                "declared_factor": None,
                "reason": "test stack conserves household mass, not person mass",
            }
        ]
    }
    if declared:
        receipt["mass"] = {
            "policy": "declared",
            "before": 5.0,
            "after": 4.5,
            "stratum_before": {"base": 5.0},
            "stratum_after": {"base": 4.5},
        }
    return KernelResult(
        expand={
            "person": pd.Series(
                [1], index=pd.Index([4], name="person_id"), dtype="int64"
            ),
            "benunit": pd.Series(
                [100], index=pd.Index([300], name="benunit_id"), dtype="int64"
            ),
            "household": pd.Series(
                [10], index=pd.Index([30], name="household_id"), dtype="int64"
            ),
        },
        columns={
            ("household", "is_clone"): pd.Series(
                [False, False, True],
                index=pd.Index([10, 20, 30], name="household_id"),
                dtype="bool",
            ),
        },
        weights=Weights(
            np.array([0.5, 1.5, 1.0], dtype=np.float64),
            WeightKind.IMPORTANCE,
        ),
        receipt=receipt,
    )


def test_conserve_rejects_a_mass_shift_across_household_sizes() -> None:
    # The executor's ledger is person mass: an expansion that conserves the
    # weight entity's mass but shifts it between households of different size
    # cannot pass ``conserve``.  This is the class that refused the SPI support
    # channel on the licensed FRS 2024-25 spine (68.25m -> 65.44m persons).
    with pytest.raises(PopulationError, match="changed stratum"):
        patch(
            _mixed_size_population(),
            _expand_node(),
            _mass_shifting_expand_result(declared=False),
        )


def test_declared_accepts_the_same_expansion_with_the_kernel_ledger() -> None:
    node = Node(
        id="stack",
        kernel="uk.stage.expand.test@1",
        structural=StructuralDelta.EXPAND,
        base="root",
        params=_expand_node().params,
        mass="declared",
    )
    expanded = patch(
        _mixed_size_population(),
        node,
        _mass_shifting_expand_result(declared=True),
    )

    assert expanded.frame.table("person")["person_household_id"].tolist() == [
        10,
        20,
        20,
        30,
    ]
    assert expanded.frame.weights_for("household").total == pytest.approx(3.0)
    record = expanded.mass_ledger[-1]
    assert record.policy == "declared"
    assert record.before_total == pytest.approx(5.0)
    assert record.after_total == pytest.approx(4.5)


def test_spi_support_channel_declares_its_mass_change_and_cgt_clones_conserve() -> None:
    graph = uk_spine_graph(load_country_spec("uk"))

    assert graph.node("spi_support_channel").mass == "declared"
    assert graph.node("cgt_incidence_clone").mass == "conserve"
    assert graph.node("cgt_band_donors").mass == "free"


@pytest.mark.requires_uk
def test_spi_support_fixture_changes_person_mass_and_conserves_household_mass() -> None:
    """Keep H2 sensitive to support-channel composition changes."""

    from pathlib import Path

    from microcosm.build.uk_runtime.frs_spine import uk_frs_spine_seed_frame
    from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs

    root = Path(__file__).resolve().parents[3]
    fixture = root / "packages/microcosm-graph/tests/fixtures/parity/uk_spine"
    _, implementations = fixture_stage_plan_inputs(fixture / "sources")
    before = implementations["frs_spine"](uk_frs_spine_seed_frame())
    after = implementations["spi_support_channel"](before)

    before_person_mass = before.stratum_mass()
    after_person_mass = after.stratum_mass()
    assert before_person_mass.index.equals(after_person_mass.index)
    assert not np.isclose(
        float(after_person_mass.sum()),
        float(before_person_mass.sum()),
        rtol=1e-9,
        atol=0.0,
    )

    before_household_mass = before.weights_for("household").total
    assert after.weights_for("household").total == before_household_mass
    assert after.mass_log[-1].old_total == before_household_mass
    assert after.mass_log[-1].new_total == before_household_mass


@pytest.mark.requires_uk
def test_driver_projects_a_stage_record_for_every_graph_stage_on_the_fixture(
    tmp_path,
) -> None:
    """The driver's record projection must cover every declared output.

    ``frs_spine`` declares the entity ids and memberships among its outputs,
    but the executor carries those outside owned cells, so the root node
    exposes no artifact for them.  The first full licensed run through the
    graph completed every stage and then died here, on ``person_id``; this
    test runs the projection on the hermetic H2 fixture so the class fails
    in CI's engine lane instead.
    """

    import importlib.util
    from pathlib import Path

    from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs
    from microcosm.graph import ContentStore, run_graph

    root = Path(__file__).resolve().parents[3]
    fixture = root / "packages/microcosm-graph/tests/fixtures/parity/uk_spine"
    if not fixture.exists():
        pytest.skip("UK spine parity fixture is not present")
    spec = importlib.util.spec_from_file_location(
        "build_uk_frs_spine", root / "tools" / "build_uk_frs_spine.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    country = load_country_spec("uk")
    stages = [
        stage
        for stage in country.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    ]
    _, implementations = fixture_stage_plan_inputs(fixture / "sources")
    graph = uk_spine_graph()
    compiled = compile_graph(graph)
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        compiled,
        sources={"frs": fixture / "sources"},
        store=store,
        kernels=uk_registry(dict(implementations)),
        resume="forbid",
        decisions=(),
    )
    final = manifest.population(compiled.versions[compiled.order[-1]])

    records = driver._graph_stage_records(
        manifest=manifest, store=store, stages=stages, frame=final
    )

    assert [record.stage for record in records] == [stage.stage for stage in stages]
    by_stage = {record.stage: record for record in records}
    for stage in stages:
        assert set(by_stage[stage.stage].nonzero_share) == set(stage.outputs), (
            stage.stage
        )
    root_shares = by_stage["frs_spine"].nonzero_share
    for column in (
        "person_id",
        "person_benunit_id",
        "person_household_id",
        "benunit_id",
        "household_id",
    ):
        assert root_shares[column] == 1.0
