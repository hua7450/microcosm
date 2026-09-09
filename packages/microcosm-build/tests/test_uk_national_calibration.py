"""The UK national Ledger-backed calibration stage."""

from __future__ import annotations

import json
from importlib import resources as importlib_resources

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime import (
    UK_NATIONAL_L0_LAMBDA,
    UK_NATIONAL_LEARNING_RATE,
    UK_NATIONAL_MASS_RULE,
    UK_NATIONAL_MAX_WEIGHT_RATIO,
    UK_NATIONAL_SEED,
    UK_NATIONAL_SOLVE_DOCTRINE,
    UK_NATIONAL_SOLVE_EPOCHS,
    UK_NATIONAL_TARGET_LOSS_CAP,
    UK_NATIONAL_TARGET_WEIGHT_RULE,
    UKNationalSolveDoctrine,
    national_calibration,
    uk_doctrine_with_overrides,
    uk_national_target_loss_weights,
)
from microcosm.build.uk_runtime.ledger_targets import (
    UKLedgerTargetCompilation,
    _uk_contract_targets,
)
from microcosm.build.uk_runtime.national_calibration import (
    UKNationalCalibrationStage,
    _post_solve_calibration_record,
    national_calibration_mass_reason,
)
from microcosm.build.uk_runtime.national_frame import (
    validate_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

ACTIVE_REFERENCE_COUNT = 415


def _uc_reference(**overrides) -> LedgerTargetReference:
    values = {
        "name": "dwp.uc.households",
        "ledger_selector": {
            "source_name": "dwp",
            "source_concept": "dwp.uc_benefit_units",
            "geography_level": "country",
        },
        "entity": "benunit",
        "measure": "dwp/uc/households",
        "family": "dwp_uc",
        "period": 2025,
        "metadata": {"contract_target_id": "dwp.uc.households"},
    }
    values.update(overrides)
    return LedgerTargetReference(**values)


def _fact(
    *,
    concept: str = "dwp.uc_benefit_units",
    source_name: str = "dwp",
    value: float = 30.0,
) -> dict:
    return {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:uc-fixture",
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "entity": {"name": "person"},
        "geography": {"level": "country", "id": "K02000001"},
        "observed_measure": {
            "source_name": source_name,
            "source_concept": concept,
            "source_measure_id": "total_units",
            "unit": "count",
        },
        "period": {"type": "month", "value": "2025-12"},
        "value": value,
    }


def _registry(*, value: float = 30.0) -> TargetRegistry:
    return TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=value,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )


def _frame() -> Frame:
    ids = np.arange(4, dtype="int64")
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": ids,
                    "person_benunit_id": ids,
                    "person_household_id": ids,
                }
            ),
            "benunit": pd.DataFrame(
                {"benunit_id": ids, "universal_credit": [1.0, 1.0, 0.0, 0.0]}
            ),
            "household": pd.DataFrame({"household_id": ids, "region": "LONDON"}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.full(4, 10.0), WeightKind.DESIGN)},
        metadata={"time_period": "2023"},
    )


def _frame_without_uc_column() -> Frame:
    frame = _frame()
    return Frame(
        {
            "person": frame.table("person"),
            "benunit": frame.table("benunit").drop(columns=["universal_credit"]),
            "household": frame.table("household"),
        },
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


class StubMeasureResolver:
    contract_targets = {
        "dwp.uc.households": {
            "bindings": {
                "policyengine": {
                    "from_entity": "benunit",
                    "value_variable": "universal_credit",
                }
            }
        }
    }

    def __init__(self):
        self.calls = []

    def knows(self, entity, variable):
        return (entity, variable) == ("benunit", "universal_credit")

    def compute(self, entity, variable):
        self.calls.append((entity, variable))
        return np.array([1.0, 1.0, 0.0, 0.0]), "stub_uc"

    def receipt(self):
        return {"provider": "stub_uc"}


class StubCrosstabResolver:
    """Supply separate prepared benefit-unit flags and affected-child counts.

    The national stage must inject both measurements temporarily, materialize
    the target count, then remove the scratch inputs from the returned frame.
    """

    contract_targets = None
    measures = {
        "uc_tcl_affected_benunit_proxy": np.array([True, False, True]),
        "uc_tcl_affected_child_count_proxy": np.array([2.0, 0.0, 3.0]),
    }

    def knows(self, entity, variable):
        return entity == "benunit" and variable in self.measures

    def compute(self, entity, variable):
        assert self.knows(entity, variable)
        return self.measures[variable].copy(), "stub_benunit_tcl_measure"

    def receipt(self):
        return {"provider": "stub_crosstab_flag"}


def _nested_frame() -> Frame:
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(6, dtype="int64"),
                    "person_benunit_id": [0, 0, 1, 2, 3, 3],
                    "person_household_id": [0, 0, 0, 1, 2, 2],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(4, dtype="int64"),
                    "universal_credit": [1.0, 0.0, 1.0, 1.0],
                }
            ),
            "household": pd.DataFrame(
                {"household_id": np.arange(3, dtype="int64"), "region": "LONDON"}
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0, 20.0, 30.0]), WeightKind.DESIGN)},
        metadata={"time_period": "2023"},
    )


def _reference_by_name(name: str) -> LedgerTargetReference:
    from microcosm.build.country_spec import load_country_spec

    return next(
        reference
        for reference in load_country_spec("uk").target_references
        if reference.name == name
    )


def _fact_for_reference(
    reference: LedgerTargetReference,
    value: float,
) -> dict:
    selector = dict(reference.ledger_selector)
    dimensions = dict(selector.get("dimension_values", {}))
    layout = {
        key: selector[key]
        for key in (
            "groupby_dimension",
            "groupby_value_id",
            "record_set_spec_id",
        )
        if key in selector
    }
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{reference.name}",
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "entity": {"name": selector.get("entity_name", reference.entity)},
        "dimensions": dimensions,
        "geography": {
            "level": selector.get("geography_level", "country"),
            "id": selector.get("geography_id", "K02000001"),
        },
        "layout": layout,
        "observed_measure": {
            "source_name": selector["source_name"],
            "source_concept": selector["source_concept"],
            "source_measure_id": selector.get("source_measure_id", "value"),
            "unit": "gbp",
        },
        "period": {"type": "month", "value": f"{reference.period}-12"},
        "value": value,
    }


def _facts_for_reference(reference: LedgerTargetReference, value: float) -> list[dict]:
    """Expand the paid headline into its declared synthetic month/cell grid."""
    if reference.name != "dwp.uc.households":
        return [_fact_for_reference(reference, value)]

    facts = []
    for month in reference.ledger_selector["period_value"]:
        for cell, operand in enumerate(reference.value_operands):
            fact = _fact_for_reference(reference, value / len(reference.value_operands))
            fact["aggregate_fact_key"] += f":{month}:{cell}"
            fact["dimensions"] = dict(operand["dimension_values"])
            fact["period"] = {"type": "month", "value": month}
            fact["source_release_key"] = "ledger.source_release.v2:uc-paid-fixture"
            fact["source"] = {"source_sha256": "a" * 64}
            fact["observed_measure"]["unit"] = "count"
            facts.append(fact)
    return facts


def _materialization_binding_frame(
    *,
    include_counterfactual_delta: bool = True,
) -> Frame:
    household = pd.DataFrame(
        {
            "household_id": np.arange(3, dtype="int64"),
            "region": "LONDON",
            "esa_income": [10.0, 20.0, 0.0],
            "esa_contrib": [1.0, 2.0, 0.0],
        }
    )
    salary_sacrifice_metric = "hmrc/salary_sacrifice_it_relief_basic_rate"
    person_columns = {
        "person_id": np.arange(6, dtype="int64"),
        "person_benunit_id": [0, 0, 1, 2, 2, 2],
        "person_household_id": [0, 0, 1, 2, 2, 2],
        "capital_gains": [0.0, 7_000.0, 12_000.0, 500.0, 0.0, 0.0],
        # Two flagged children in household 0, none in 1, three in 2.
        "uc_is_child_limit_affected": [True, True, False, True, True, True],
    }
    if include_counterfactual_delta:
        person_columns[salary_sacrifice_metric] = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    return Frame(
        {
            "person": pd.DataFrame(person_columns),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(3, dtype="int64"),
                    "universal_credit": [1.0, 0.0, 1.0],
                }
            ),
            "household": household,
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.full(3, 10.0), WeightKind.DESIGN)},
        metadata={"time_period": "2025"},
    )


def test_uc_calibration_compiles_and_moves_weighted_count_towards_fact() -> None:
    frame = _frame()
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=200, learning_rate=0.05),
    )

    result = stage(frame)

    before = 20.0
    after = float(result.weights_for("household").values[:2].sum())
    assert abs(after - 30.0) < abs(before - 30.0)
    assert stage.manifest["activated_reference_count"] == 1
    assert stage.manifest["resolved_reference_count"] == 1
    assert stage.manifest["matrix_target_count"] == 1
    assert stage.diagnostics[0]["target"] == 30.0
    assert result.weights_for("household").kind is WeightKind.CALIBRATED
    assert len(result.mass_log) == 1
    mass_change = stage.manifest["weights"]["calibration_mass_change"]
    assert mass_change["entity"] == "household"
    assert "National doctrine calibration" in mass_change["reason"]
    assert stage.manifest["weights"]["household_weight_kind_chain"] == [
        {"stage": "staging", "kind": "design"},
        {"stage": "national_calibration", "kind": "calibrated"},
    ]
    assert stage.manifest["weights"]["mass_log_records_before_calibration"] == 0
    assert stage.manifest["weights"]["mass_log_records"] == 1
    assert stage.manifest["solve"]["n_targets"] == 1
    assert stage.manifest["solve"]["n_households"] == 4


def test_exact_k_calibration_selects_and_refits_exact_household_count() -> None:
    frame = _frame()
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=3, learning_rate=0.02),
        exact_k=2,
        exact_k_pi_hi=1.0,
        exact_k_seed=17,
    )

    result = stage(frame)

    assert result.n("household") == 2
    assert result.n("benunit") == 2
    assert result.n("person") == 2
    assert result.weights_for("household").kind is WeightKind.CALIBRATED
    assert set(result.table("household")["household_id"]) <= set(
        frame.table("household")["household_id"]
    )
    exact_k = stage.manifest["exact_k_ladder"]
    assert exact_k["k"] == 2
    assert exact_k["seed"] == 17
    assert exact_k["pool_households"] == 4
    assert exact_k["realized_households"] == 2
    assert exact_k["selection_receipt"]["design"] == "sampford"
    assert exact_k["refit_baseline_diagnostics"]["selected_size"] == 2
    assert stage.manifest["solve"]["n_households"] == 2
    validate_uk_national_frame(result)


def test_exact_k_calibration_refuses_count_above_input_pool() -> None:
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=1),
        exact_k=5,
        exact_k_pi_hi=0.95,
        exact_k_seed=0,
    )

    with pytest.raises(ValueError, match="k=5 exceeds the pool size 4"):
        stage(_frame())


def test_stage_forwards_solver_progress_and_measure_lifecycle() -> None:
    progress: list[dict[str, object]] = []
    stages: list[tuple[str, str, dict[str, object]]] = []
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
        progress_callback=progress.append,
        stage_callback=lambda stage_id, status, details: stages.append(
            (stage_id, status, dict(details))
        ),
    )

    stage(_frame())

    assert progress
    assert all(event["kind"] == "calibration_epoch" for event in progress)
    assert [(stage_id, status) for stage_id, status, _details in stages] == [
        ("measure_resolution", "started"),
        ("measure_resolution", "completed"),
    ]
    assert stages[-1][2] == {"resolved_measure_count": 0}


def test_uc_calibration_stage_accepts_benunit_grain_reference_on_nested_frame() -> None:
    frame = _nested_frame()
    stage = UKNationalCalibrationStage(
        _registry(value=60.0),
        band_edge_registry=_registry(value=60.0),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )

    result = stage(frame)

    assert stage.manifest["activated_reference_count"] == 1
    assert stage.manifest["resolved_reference_count"] == 1
    assert stage.manifest["matrix_target_count"] == 1
    assert stage.diagnostics[0]["target"] == 60.0
    assert stage.diagnostics[0]["estimate"] == pytest.approx(60.0)
    validate_uk_national_frame(result)


def test_stage_measure_resolver_injects_columns_then_restores_pristine_output() -> None:
    frame = _frame_without_uc_column()
    resolver = StubMeasureResolver()
    original_columns = {
        entity: set(frame.table(entity).columns) for entity in frame.entities
    }
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
        measure_resolver=resolver,
    )

    result = stage(frame)

    assert resolver.calls == [("benunit", "universal_credit")]
    assert stage.manifest["measure_resolution"]["provider"] == {"provider": "stub_uc"}
    assert stage.manifest["measure_resolution"]["attached"] == {
        "benunit.universal_credit": "stub_uc"
    }
    for entity in frame.entities:
        assert set(result.table(entity).columns) == original_columns[entity]
    assert "universal_credit" not in result.table("benunit")
    validate_uk_national_frame(result)


def test_stage_manifest_omits_measure_resolution_without_resolver() -> None:
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )

    stage(_frame())

    assert "measure_resolution" not in stage.manifest


def test_stage_threads_band_edge_registry_to_materialization(monkeypatch) -> None:
    captured = []
    real_materialize = national_calibration.materialize_uk_ledger_targets

    def capture_materialize(*args, **kwargs):
        captured.append(kwargs)
        return real_materialize(*args, **kwargs)

    monkeypatch.setattr(
        national_calibration,
        "materialize_uk_ledger_targets",
        capture_materialize,
    )
    sentinel = TargetRegistry([], country="uk")
    stage = UKNationalCalibrationStage(
        _registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=1),
        band_edge_registry=sentinel,
    )

    stage(_frame())

    assert captured[-1]["band_edge_registry"] is sentinel

    # The parameter is required, never defaulted: a stage cannot tell a
    # pruned registry from a full one (#803 review finding 1).
    with pytest.raises(TypeError, match="band_edge_registry"):
        UKNationalCalibrationStage(
            _registry(),
            period=2025,
            doctrine=UKNationalSolveDoctrine(epochs=1),
        )


def test_activated_unresolvable_compiled_reference_aborts_loudly() -> None:
    stage = UKNationalCalibrationStage(
        UKLedgerTargetCompilation(
            registry=TargetRegistry([], country="uk"),
            unsupported=(
                {
                    "name": "dwp.uc.households",
                    "period": "2025",
                    "reason": "did not match a Ledger fact selector",
                },
            ),
        ),
        band_edge_registry=TargetRegistry([], country="uk"),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=1),
    )

    with pytest.raises(RuntimeError, match="did not match a Ledger fact selector"):
        stage(_frame())


def test_chronicle_184_uc_and_obr_references_compile_fail_closed() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.ledger_targets import compile_ledger_target_references

    spec = load_country_spec("uk")
    assert len(spec.target_references) == ACTIVE_REFERENCE_COUNT
    reference_names = {reference.name for reference in spec.target_references}
    assert "obr.universal_credit_in_cap" in reference_names
    assert "dwp.uc.households" in reference_names

    membership = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("target_reference_membership.json")
        .read_text()
    )
    assert membership["targets"]["dwp.uc.households"]["status"] == "active"
    uc_reference = next(
        reference
        for reference in spec.target_references
        if reference.name == "dwp.uc.households"
    )
    assert uc_reference.value_operation == "monthly_window_sum_average"
    assert uc_reference.period_match_policy == "source_window"
    assert uc_reference.ledger_selector["period_value"] == [
        f"2025-{month:02d}" for month in range(1, 13)
    ]
    assert len(uc_reference.value_operands) == 10
    assert {
        tuple(
            operand["dimension_values"][key]
            for key in ("family_type", "payment_indicator", "child_entitlement")
        )
        for operand in uc_reference.value_operands
    } == {
        (family, "Yes", entitled)
        for family in (
            "Single, no children",
            "Single, with children",
            "Couple, no children",
            "Couple, with children",
            "Unknown or missing family type",
        )
        for entitled in ("No", "Yes")
    }

    references = tuple(
        reference
        for reference in spec.target_references
        if reference.name == "obr.universal_credit_in_cap"
    )
    registry = compile_ledger_target_references(
        [
            _fact(
                concept="obr.universal_credit_in_cap",
                source_name="obr",
                value=40_000_000_000,
            ),
        ],
        references,
        country="uk",
    )

    assert {spec.name for spec in registry.specs} == {"obr.universal_credit_in_cap"}


def test_packaged_binding_classes_materialize_through_national_stage() -> None:
    selected_names = (
        "dwp.uc.households",
        "obr.esa",
        "hmrc.cgt.taxpayers_total",
        "dwp.uc.two_child_limit.children_affected",
        "hmrc.salary_sacrifice.it_relief_basic_rate",
    )
    references = tuple(_reference_by_name(name) for name in selected_names)
    facts = [
        fact
        for reference, value in zip(
            references,
            (20.0, 33.0, 2.0, 5.0, 3.0),
            strict=True,
        )
        for fact in _facts_for_reference(reference, value)
    ]
    from microcosm.build.ledger_targets import compile_ledger_target_references
    from microcosm.build.uk_runtime.ledger_targets import (
        UKFrameTargetAdapter,
        materialize_uk_ledger_targets,
    )

    registry = compile_ledger_target_references(facts, references, country="uk")
    headline = next(spec for spec in registry.specs if spec.name == "dwp.uc.households")
    assert headline.value == 20.0
    assert headline.metadata["ledger_member_fact_count"] == "120"
    resolver = StubCrosstabResolver()
    resolver.contract_targets = _uk_contract_targets()
    stage = UKNationalCalibrationStage(
        registry,
        band_edge_registry=registry,
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=1, learning_rate=0.01),
        measure_resolver=resolver,
    )

    input_frame = _materialization_binding_frame()
    original_columns = {
        entity: set(input_frame.table(entity).columns)
        for entity in input_frame.entities
    }

    result = stage(input_frame)

    assert stage.manifest["activated_reference_count"] == len(selected_names)
    assert stage.manifest["resolved_reference_count"] == len(selected_names)
    assert stage.manifest["matrix_target_count"] == len(selected_names)
    # The staged frame stays writer-clean: no prepared scratch column survives
    # onto the returned tables (adjudicated lifecycle; slash-named scratch
    # crashes the HDFStore staging writer). Original input columns — including
    # the fixture's precomputed counterfactual delta — are exactly preserved.
    for entity in input_frame.entities:
        assert set(result.table(entity).columns) == original_columns[entity]
    # The binding classes produce the right prepared values on the adapter…
    adapter = UKFrameTargetAdapter(_materialization_binding_frame())
    # The same table-scoped injection the resolution loop performs.
    for variable, values in resolver.measures.items():
        adapter.tables["benunit"][variable] = values.copy()
    materialize_uk_ledger_targets(adapter, registry, period=2025)
    materialized = {
        ("benunit", "dwp/uc/households"): [1.0, 0.0, 1.0],
        ("household", "obr/esa"): [11.0, 22.0, 0.0],
        ("person", "hmrc/cgt_taxpayers"): [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        # The prepared affected-child counts stay distinct from the claim
        # indicator [1.0, 0.0, 1.0] and legacy person-native entitlement flags.
        ("benunit", "dwp/uc/two_child_limit/children_affected"): [2.0, 0.0, 3.0],
        ("person", "hmrc/salary_sacrifice_it_relief_basic_rate"): [
            1.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    }
    for (entity, measure), expected in materialized.items():
        assert adapter.tables[entity][measure].tolist() == expected


def test_packaged_materialization_skip_aborts_national_stage() -> None:
    from microcosm.build.ledger_targets import compile_ledger_target_references

    reference = _reference_by_name("hmrc.salary_sacrifice.it_relief_basic_rate")
    registry = compile_ledger_target_references(
        [_fact_for_reference(reference, 3.0)],
        [reference],
        country="uk",
    )
    stage = UKNationalCalibrationStage(
        registry,
        band_edge_registry=registry,
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=1),
    )

    with pytest.raises(RuntimeError, match="could not materialize every"):
        stage(_materialization_binding_frame(include_counterfactual_delta=False))


def test_calibration_preserves_entity_ids_and_national_integrity() -> None:
    frame = _frame()
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )

    result = stage(frame)

    for entity in frame.entities:
        id_column = f"{entity}_id"
        assert result.table(entity)[id_column].equals(frame.table(entity)[id_column])
    validate_uk_national_frame(result)


def test_checkpoint_metadata_round_trips_calibration_evidence() -> None:
    frame = _frame()
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )

    staged = stage(frame)
    metadata = json.loads(json.dumps(stage.checkpoint_metadata()))

    resumed = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )
    resumed.resume_from_checkpoint(metadata, staged)

    assert resumed.manifest == stage.manifest
    assert resumed.diagnostics == stage.diagnostics
    assert resumed.output_content_identity == metadata["output_content_identity"]

    drifted = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )
    with pytest.raises(RuntimeError, match="drifted record"):
        drifted.resume_from_checkpoint(metadata, frame)

    empty = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )
    with pytest.raises(RuntimeError, match="calibration counts"):
        empty.resume_from_checkpoint({}, staged)

    missing_count = dict(metadata)
    missing_count["calibration"] = {
        key: value
        for key, value in metadata["calibration"].items()
        if key != "activated_reference_count"
    }
    with pytest.raises(RuntimeError, match="calibration counts"):
        empty.resume_from_checkpoint(missing_count, staged)

    unrun = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )
    with pytest.raises(RuntimeError, match="has not run"):
        unrun.checkpoint_metadata()


def test_prepared_slash_columns_are_not_returned_to_the_writer(tmp_path) -> None:
    pytest.importorskip("tables")
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5),
    )

    result = stage(_frame())

    assert "dwp/uc/households" not in result.table("benunit")
    write_uk_national_frame(result, tmp_path / "staging.h5")


def test_national_calibration_mass_reason_is_canonical() -> None:
    assert national_calibration_mass_reason(
        ["dwp_universal_credit", "hmrc", "dwp_universal_credit"]
    ) == (
        "National doctrine calibration to bound target family(ies) "
        "dwp_universal_credit, hmrc; total household mass moved with the targets."
    )
    with pytest.raises(ValueError, match="bound_families"):
        national_calibration_mass_reason([])


def test_post_solve_fence_requires_calibrated_kind_and_mass_record() -> None:
    before = _frame()
    uncalibrated = Frame(
        {entity: before.table(entity) for entity in before.entities},
        before.schema,
        {"household": Weights(np.full(4, 10.0), WeightKind.DESIGN)},
        before.strata,
        mass_log=before.mass_log,
        metadata=before.metadata,
    )

    with pytest.raises(RuntimeError, match="not 'calibrated'"):
        _post_solve_calibration_record(before, uncalibrated, before_count=0)

    calibrated_without_record = Frame(
        {entity: before.table(entity) for entity in before.entities},
        before.schema,
        {"household": Weights(np.full(4, 10.0), WeightKind.CALIBRATED)},
        before.strata,
        mass_log=before.mass_log,
        metadata=before.metadata,
    )
    with pytest.raises(RuntimeError, match="exactly one mass record"):
        _post_solve_calibration_record(
            before,
            calibrated_without_record,
            before_count=0,
        )


def test_national_doctrine_constants_are_the_declared_contract() -> None:
    assert UK_NATIONAL_SOLVE_EPOCHS == 256
    assert UK_NATIONAL_LEARNING_RATE == 0.02
    assert UK_NATIONAL_MAX_WEIGHT_RATIO == 10.0
    assert UK_NATIONAL_SEED == 0
    assert UK_NATIONAL_TARGET_LOSS_CAP == 10.0
    assert UK_NATIONAL_L0_LAMBDA == 0.0
    assert UK_NATIONAL_MASS_RULE == "free"
    # María's ruling (2026-08-24): family_equal is vocabulary, never the
    # default — she passes it as an explicit per-run override.
    assert UK_NATIONAL_TARGET_WEIGHT_RULE == "uniform"
    assert UK_NATIONAL_SOLVE_DOCTRINE == UKNationalSolveDoctrine()
    assert UK_NATIONAL_SOLVE_DOCTRINE.scale_rule == "default_target_loss_scales"
    assert UK_NATIONAL_SOLVE_DOCTRINE.target_weight_rule == "uniform"
    assert UKNationalSolveDoctrine(target_weight_rule="family_equal")


def test_uk_doctrine_with_overrides_receipts_effective_diffs_only() -> None:
    doctrine, receipt = uk_doctrine_with_overrides()
    assert doctrine == UK_NATIONAL_SOLVE_DOCTRINE
    assert receipt == {}

    doctrine, receipt = uk_doctrine_with_overrides(
        epochs=UK_NATIONAL_SOLVE_EPOCHS,
        target_weight_rule=UK_NATIONAL_TARGET_WEIGHT_RULE,
    )
    assert doctrine == UK_NATIONAL_SOLVE_DOCTRINE
    assert receipt == {}

    doctrine, receipt = uk_doctrine_with_overrides(
        epochs=128,
        learning_rate=0.01,
        target_weight_rule="family_equal",
        target_loss_cap=5.0,
    )
    assert doctrine.epochs == 128
    assert doctrine.learning_rate == 0.01
    assert doctrine.target_weight_rule == "family_equal"
    assert doctrine.target_loss_cap == 5.0
    assert receipt == {
        "epochs": {"default": 256, "effective": 128},
        "learning_rate": {"default": 0.02, "effective": 0.01},
        "target_loss_cap": {"default": 10.0, "effective": 5.0},
        "target_weight_rule": {"default": "uniform", "effective": "family_equal"},
    }


def test_uk_doctrine_with_overrides_refuses_invalid_or_frozen_fields() -> None:
    with pytest.raises(ValueError, match="target_weight_rule"):
        uk_doctrine_with_overrides(target_weight_rule="per_target")
    with pytest.raises(ValueError, match="epochs"):
        uk_doctrine_with_overrides(epochs=0)
    with pytest.raises(ValueError, match="reviewed constants"):
        uk_doctrine_with_overrides(seed=1)
    with pytest.raises(ValueError, match="unknown"):
        uk_doctrine_with_overrides(not_a_field=1)


def test_family_equal_gives_each_family_one_equal_share() -> None:
    weights = uk_national_target_loss_weights(
        ["hmrc"] * 3 + ["obr"], rule="family_equal"
    )
    assert weights is not None
    assert weights.sum() == pytest.approx(1.0)
    # Three hmrc rows share half the objective; the single obr row holds the
    # other half, so an over-supplied family cannot outvote by count.
    assert weights[:3].sum() == pytest.approx(0.5)
    assert weights[3] == pytest.approx(0.5)


def test_uniform_rule_defers_to_the_kernel_default() -> None:
    assert uk_national_target_loss_weights(["hmrc", "obr"], rule="uniform") is None


def test_family_equal_refuses_an_undeclared_family() -> None:
    with pytest.raises(ValueError, match="must declare a family"):
        uk_national_target_loss_weights(["hmrc", ""], rule="family_equal")


def test_national_doctrine_rejects_tampered_bounds() -> None:
    with pytest.raises(ValueError, match="epochs"):
        UKNationalSolveDoctrine(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        UKNationalSolveDoctrine(learning_rate=0.0)
    with pytest.raises(ValueError, match="max_weight_ratio"):
        UKNationalSolveDoctrine(max_weight_ratio=1.0)
    with pytest.raises(ValueError, match="seed"):
        UKNationalSolveDoctrine(seed=-1)
    with pytest.raises(ValueError, match="target_loss_cap"):
        UKNationalSolveDoctrine(target_loss_cap=float("nan"))
    with pytest.raises(ValueError, match="scale_rule"):
        UKNationalSolveDoctrine(scale_rule="bespoke")
    with pytest.raises(ValueError, match="target_weight_rule"):
        UKNationalSolveDoctrine(target_weight_rule="per_target")
    with pytest.raises(ValueError, match="mass_rule"):
        UKNationalSolveDoctrine(mass_rule="conserve")
    with pytest.raises(ValueError, match="l0_lambda"):
        UKNationalSolveDoctrine(l0_lambda=-0.1)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [("uniform", None), ("family_equal", [1.0])],
)
def test_doctrine_target_weight_rule_reaches_the_solver(
    monkeypatch, rule, expected
) -> None:
    """The declared rule must reach calibrate(), not just the manifest echo.

    First-armed-run finding (2026-08-23): the doctrine declared a
    target_weight_rule the stage never passed to the kernel, so the solve
    silently ran uniform whatever the doctrine said. The doctrine vector now
    travels explicitly; "uniform" maps to None — the kernel's own default —
    so the shipped identity is unchanged under the default rule.
    """

    from microcosm.calibrate import calibrate as real_calibrate

    captured: dict[str, object] = {}

    def capturing_calibrate(*args, **kwargs):
        captured["target_loss_weights"] = kwargs["target_loss_weights"]
        return real_calibrate(*args, **kwargs)

    monkeypatch.setattr(
        "microcosm.build.uk_runtime.national_calibration.calibrate",
        capturing_calibrate,
    )
    stage = UKNationalCalibrationStage(
        _registry(),
        band_edge_registry=_registry(),
        period=2025,
        doctrine=UKNationalSolveDoctrine(epochs=5, target_weight_rule=rule),
    )

    stage(_frame())

    weights = captured["target_loss_weights"]
    if expected is None:
        assert weights is None
    else:
        assert weights is not None
        assert weights.tolist() == expected


def test_measure_resolution_never_touches_the_source_frame() -> None:
    """Injection lands on adapter table copies, not on the caller's frame.

    A test that inspects the frame after ``restore`` would pass either way,
    so this one holds the property where it actually lives: the source frame
    gains no column while the resolution loop probes round after round.
    """

    from microcosm.build.target_materialization import resolve_target_measures
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    frame = _materialization_binding_frame()
    before = {entity: list(frame.table(entity).columns) for entity in frame.entities}

    class ProbeResolver:
        contract_targets = {
            "probe.target": {
                "bindings": {
                    "policyengine": {
                        "from_entity": "household",
                        "value_variable": "probe_measure",
                    }
                }
            }
        }

        def knows(self, entity, variable):
            return (entity, variable) == ("household", "probe_measure")

        def compute(self, entity, variable):
            return np.array([1.0, 2.0, 3.0]), "probe"

        def receipt(self):
            return {"provider": "probe"}

    registry = TargetRegistry(
        [
            TargetSpec(
                name="probe.target",
                entity="household",
                measure="probe/measure",
                value=6.0,
                source="test",
                metadata={"contract_target_id": "probe.target"},
            )
        ],
        country="uk",
    )

    resolution = resolve_target_measures(
        lambda: UKFrameTargetAdapter(frame),
        registry,
        ProbeResolver(),
        period=2025,
    )

    assert ("household", "probe_measure") in resolution.measure_inputs
    for entity in frame.entities:
        assert list(frame.table(entity).columns) == before[entity]


def test_published_child_reduction_translates_to_the_age_predicate():
    """A published fact keeps its publisher's semantics; binding it is our job.

    DWP names a dependent-child concept in `any_child_under`. The model carries
    no `is_child` column because it has no need of one — dependency is derived
    from age where it is wanted — so "any child under N" is exactly "any person
    aged under N" here. The translation is declared, not aliased at the call
    site, and refusing the fact instead would drop a real target.
    """

    from microcosm.build.uk_runtime.ledger_targets import (
        UK_TRANSLATED_HOUSEHOLD_REDUCTIONS,
        UKFrameTargetAdapter,
    )

    assert UK_TRANSLATED_HOUSEHOLD_REDUCTIONS["any_child_under"] == "any"

    frame = _materialization_binding_frame()
    adapter = UKFrameTargetAdapter(frame)
    # Households 0 and 2 each carry an infant; household 1 does not.
    adapter.tables["person"]["age"] = [0.0, 40.0, 35.0, 0.0, 38.0, 41.0]
    condition = {
        "variable": "age",
        "entity": "person",
        "reduce": "any_child_under",
        "operator": "<",
        "value": 1,
    }

    translated = adapter.household_condition(condition)

    assert list(translated) == [True, False, True]
    assert list(translated) == list(
        adapter.household_condition({**condition, "reduce": "any"})
    )


def test_unmapped_household_reduction_names_the_published_reducer():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_materialization_binding_frame())

    with pytest.raises(ValueError, match="any_grandchild_under"):
        adapter.household_condition(
            {
                "variable": "capital_gains",
                "entity": "person",
                "reduce": "any_grandchild_under",
                "operator": "<",
                "value": 1,
            }
        )


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_packaged_uc_headline_refuses_incomplete_month_cell_fixture(defect) -> None:
    from microcosm.build.ledger_targets import compile_ledger_target_references

    reference = _reference_by_name("dwp.uc.households")
    facts = _facts_for_reference(reference, 20.0)
    if defect == "missing":
        # Keep every month represented but omit one paid/entitlement cell.
        facts.pop(0)
    else:
        # Keep the expected total size and unique IDs; repeat a cell instead.
        facts[0]["dimensions"] = dict(facts[1]["dimensions"])

    with pytest.raises(ValueError, match="monthly window"):
        compile_ledger_target_references(facts, [reference], country="uk")
