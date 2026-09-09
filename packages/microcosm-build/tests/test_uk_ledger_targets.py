import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.ledger_targets import (
    UK_CROSS_GRAIN_BRIDGES,
    UK_CROSS_GRAIN_GRAIN_PRECEDENCE,
    UK_CROSS_GRAIN_RULE,
    _spec_geography,
    _uk_licensed_empty_legs_from_membership,
    align_uk_local_registry_parity_fixture,
    apply_uk_cross_grain_reconciliation,
    compile_uk_local_target_registry,
    compile_uk_target_registry,
    materialize_uk_ledger_targets,
    uk_ladder_household_uprating,
    uk_ledger_households_total,
    uk_local_target_surface,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

FIXTURE_FEED_ROWS = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)


def test_uk_local_target_surface_uses_registry_names_and_reconciles() -> None:
    registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="household",
                value=90.0,
                measure="uc_households",
                period=2025,
                source="DWP",
                family="uc_households",
                metadata={
                    "contract_target_id": "dwp.uc.households",
                    "ledger_geography_level": "country",
                    "ledger_geography_id": "K03000001",
                },
            ),
            TargetSpec(
                name="dwp.uc.households_by_area@E14000001",
                entity="household",
                value=30.0,
                measure="uc_households",
                period=2025,
                source="DWP",
                family="uc_households",
                metadata={
                    "contract_target_id": "dwp.uc.households_by_area",
                    "geography_level": "constituency",
                    "geography_id": "E14000001",
                },
            ),
            TargetSpec(
                name="dwp.uc.households_by_area@S14000001",
                entity="household",
                value=15.0,
                measure="uc_households",
                period=2025,
                source="DWP",
                family="uc_households",
                metadata={
                    "contract_target_id": "dwp.uc.households_by_area",
                    "geography_level": "constituency",
                    "geography_id": "S14000001",
                },
            ),
        ],
        country="uk",
    )
    ladder = SimpleNamespace(
        households=np.asarray([10.0, 20.0]),
        constituency_code=np.asarray(["E14000001", "S14000001"]),
        local_authority_code=np.asarray(["E06000001", "S12000005"]),
    )

    surface, receipt = uk_local_target_surface(
        registry,
        ladder,
        bound_national_target_ids=("dwp.uc.households",),
        period=2025,
    )

    uc = surface.loc[surface["metric"] == "uc_households"]
    assert uc["value"].tolist() == [60.0, 30.0]
    assert uc["target_name"].tolist() == [
        "dwp.uc.households_by_area@E14000001",
        "dwp.uc.households_by_area@S14000001",
    ]
    ladder_rows = surface.loc[surface["metric"] == "households"]
    assert set(ladder_rows["area_type"]) == {"constituency", "la"}
    assert all(ladder_rows["period"] == 2025)
    assert "national_uc_caseload_vs_uc_households_by_area" in {
        group["bridge_id"] for group in receipt["groups"]
    }
    uc_group = next(
        group
        for group in receipt["groups"]
        if group["bridge_id"] == "national_uc_caseload_vs_uc_households_by_area"
    )
    assert uc_group["winning_grain"] == "country"
    assert {leg["parent_geography_id"] for leg in uc_group["legs"]} == {"K03000001"}


def test_uk_local_target_surface_fires_k020_household_partition_bridge() -> None:
    composition_ids = UK_CROSS_GRAIN_BRIDGES[0].higher_target_ids
    registry = TargetRegistry(
        [
            TargetSpec(
                name=target_id,
                entity="household",
                value=10.0,
                measure=f"measure/{position}",
                period=2025,
                source="ONS",
                family="household_composition",
                metadata={
                    "contract_target_id": target_id,
                    "ledger_geography_level": "country",
                    "ledger_geography_id": "K02000001",
                },
            )
            for position, target_id in enumerate(composition_ids)
        ],
        country="uk",
    )
    ladder = SimpleNamespace(
        households=np.asarray([10.0, 20.0]),
        constituency_code=np.asarray(["E14000001", "S14000001"]),
        local_authority_code=np.asarray(["E06000001", "S12000005"]),
    )

    surface, receipt = uk_local_target_surface(
        registry,
        ladder,
        bound_national_target_ids=composition_ids,
        period=2025,
    )

    ladder_rows = surface.loc[surface["metric"] == "households"]
    assert ladder_rows.groupby("area_type")["value"].sum().to_dict() == {
        "constituency": pytest.approx(100.0),
        "la": pytest.approx(100.0),
    }
    groups = [
        group
        for group in receipt["groups"]
        if group["bridge_id"]
        == "national_household_composition_partition_vs_census_households"
    ]
    assert {group["winning_grain"] for group in groups} == {"country"}
    assert {
        leg["parent_geography_id"] for group in groups for leg in group["legs"]
    } == {"K02000001"}


def _national_geography_spec(metadata: dict[str, str]) -> TargetSpec:
    return TargetSpec(
        name="dwp.uc.households",
        entity="household",
        value=90.0,
        measure="uc_households",
        period=2025,
        source="DWP",
        family="uc_households",
        metadata={"contract_target_id": "dwp.uc.households", **metadata},
    )


def _minimal_target_surface_ladder() -> SimpleNamespace:
    return SimpleNamespace(
        households=np.asarray([10.0]),
        constituency_code=np.asarray(["E14000001"]),
        local_authority_code=np.asarray(["E06000001"]),
    )


def _national_control_spec(
    name: str,
    *,
    value: float,
    target_id: str = "dwp.uc.payment_distribution_single",
    geography_level: str = "country",
    geography_id: str = "K03000001",
) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity="household",
        value=value,
        measure=f"measure/{name}",
        period=2025,
        source="synthetic national control fixture",
        family="national_control",
        metadata={
            "contract_target_id": target_id,
            "ledger_geography_level": geography_level,
            "ledger_geography_id": geography_id,
        },
    )


def test_uk_local_target_surface_excludes_fanout_from_cross_grain_controls() -> None:
    specs = [
        _national_control_spec(f"payment-band-{index}", value=value)
        for index, value in enumerate((10.0, 20.0, 30.0))
    ]
    specs.extend(
        [
            TargetSpec(
                name=f"dwp.uc.households_by_area@{area}",
                entity="household",
                value=value,
                measure="uc_households",
                period=2025,
                source="synthetic local UC fixture",
                family="uc_households",
                metadata={
                    "contract_target_id": "dwp.uc.households_by_area",
                    "geography_level": "constituency",
                    "geography_id": area,
                },
            )
            for area, value in (("E14000001", 30.0), ("S14000001", 15.0))
        ]
    )

    surface, receipt = uk_local_target_surface(
        TargetRegistry(specs, country="uk"),
        SimpleNamespace(
            households=np.asarray([30.0, 15.0]),
            constituency_code=np.asarray(["E14000001", "S14000001"]),
            local_authority_code=np.asarray(["E06000001", "S12000005"]),
        ),
        bound_national_target_ids=("dwp.uc.payment_distribution_single",),
        period=2025,
    )

    assert surface.loc[surface["metric"] == "uc_households", "value"].tolist() == [
        30.0,
        15.0,
    ]
    assert receipt["bound_higher_targets"] == []
    assert all(
        group["bridge_id"] != "national_uc_caseload_vs_uc_households_by_area"
        for group in receipt["groups"]
    )
    assert receipt["fanout_targets_not_controls"] == [
        {
            "target_id": "dwp.uc.payment_distribution_single",
            "geography_id": "K03000001",
            "cells": 3,
            "cell_names": ["payment-band-0", "payment-band-1", "payment-band-2"],
            "activated_sum": 60.0,
            "reason": (
                "The activated cells are a band subset, so this distribution "
                "is not a cross-grain control."
            ),
        }
    ]
    assert "fanout_controls_summed" not in receipt


def test_uk_local_target_surface_keeps_single_national_control(monkeypatch) -> None:
    from microcosm.build.uk_runtime import ledger_targets as module

    captured: dict[str, pd.DataFrame] = {}

    def capture(frame, bound_higher_targets, *_args, **_kwargs):
        captured["frame"] = frame.copy(deep=True)
        captured["bound"] = tuple(bound_higher_targets)
        return frame.copy(deep=True), {"groups": []}

    monkeypatch.setattr(module, "apply_uk_cross_grain_reconciliation", capture)
    spec = _national_control_spec("only-cell", value=42.0)

    _, receipt = module.uk_local_target_surface(
        TargetRegistry([spec], country="uk"),
        _minimal_target_surface_ladder(),
        bound_national_target_ids=("dwp.uc.payment_distribution_single",),
        period=2025,
    )

    controls = captured["frame"].loc[
        captured["frame"]["target_id"] == "dwp.uc.payment_distribution_single"
    ]
    assert controls[["grain", "geography_id", "value"]].to_dict("records") == [
        {"grain": "country", "geography_id": "K03000001", "value": 42.0}
    ]
    assert captured["bound"] == ("dwp.uc.payment_distribution_single",)
    assert receipt["fanout_targets_not_controls"] == []
    assert "fanout_controls_summed" not in receipt


def test_uk_local_target_surface_refuses_named_nonfinite_national_cell() -> None:
    specs = [
        _national_control_spec("finite-cell", value=10.0),
        _national_control_spec("bad-cell", value=20.0),
    ]
    object.__setattr__(specs[1], "value", float("nan"))

    with pytest.raises(ValueError, match="bad-cell.*non-finite"):
        uk_local_target_surface(
            TargetRegistry(specs, country="uk"),
            _minimal_target_surface_ladder(),
            bound_national_target_ids=("dwp.uc.payment_distribution_single",),
            period=2025,
        )


def test_uk_local_target_surface_refuses_fanout_across_levels() -> None:
    target_id = "voa.council_tax_stock.band_a"
    specs = [
        _national_control_spec(
            "country-cell",
            value=10.0,
            target_id=target_id,
            geography_level="country",
            geography_id="K02000001",
        ),
        _national_control_spec(
            "region-cell",
            value=20.0,
            target_id=target_id,
            geography_level="region",
            geography_id="K02000001",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="country-cell.*region-cell.*mixed geography levels",
    ):
        uk_local_target_surface(
            TargetRegistry(specs, country="uk"),
            _minimal_target_surface_ladder(),
            bound_national_target_ids=(target_id,),
            period=2025,
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "dwp.uc.households.*names no geography"),
        (
            {
                "geography_level": "country",
                "geography_id": "K02000001",
                "ledger_geography_level": "country",
                "ledger_geography_id": "K03000001",
            },
            "dwp.uc.households.*disagree",
        ),
        (
            {
                "ledger_geography_level": "constituency",
                "ledger_geography_id": "E14000001",
            },
            "dwp.uc.households.*contract.*country",
        ),
    ],
)
def test_uk_local_target_surface_refuses_invalid_compiled_geography(
    metadata: dict[str, str],
    message: str,
) -> None:
    registry = TargetRegistry([_national_geography_spec(metadata)], country="uk")

    with pytest.raises(ValueError, match=message):
        uk_local_target_surface(
            registry,
            _minimal_target_surface_ladder(),
            bound_national_target_ids=(),
            period=2025,
        )


def test_compiled_geography_resolver_refuses_blank_spelling() -> None:
    spec = SimpleNamespace(
        name="dwp.uc.households",
        metadata={
            "contract_target_id": "dwp.uc.households",
            "ledger_geography_level": "",
            "ledger_geography_id": "K02000001",
        },
    )

    with pytest.raises(
        ValueError,
        match="dwp.uc.households.*blank ledger geography",
    ):
        _spec_geography(spec)


def test_local_parity_fixture_aligns_legacy_council_tax_band_names():
    fixture = {
        "rows": [
            {
                "name": "voa/council_tax/A@E06000001",
                "metric": "voa/council_tax/A",
                "geography_id": "E06000001",
            }
        ]
    }

    aligned = align_uk_local_registry_parity_fixture(fixture)

    assert aligned["rows"][0]["name"] == (
        "voa.council_tax_stock.by_area.band_a@E06000001"
    )


class StubUKAdapter:
    """Prepared benefit-unit measures distinguish flags and child counts.

    These independent prepared vectors test binding selection, not the
    demographic calculation: the flag (1/0/1), affected counts (3/0/2), and
    qualifying counts (4/0/2) must not substitute for one another.
    """

    def __init__(self):
        person_household = np.array([0, 0, 0, 0, 1, 2, 2])
        child_flags = np.array([True, True, True, False, False, True, True])
        self.tables = {
            "person": {
                "capital_gains": np.array([0.0, 0.0, 0.0, 5_000.0, 0.0, 0.0, 20_000.0]),
                "person_household_id": person_household,
                "uc_is_child_limit_affected": child_flags,
                "is_child": np.array([True, True, True, True, False, True, True]),
                "pip": np.array([0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0]),
            },
            "household": {
                "household_id": np.array([0, 1, 2]),
                # The household-grain flag is the any-collapse: "this
                # household contains at least one flagged child".
                "uc_is_child_limit_affected": np.array([1.0, 0.0, 1.0]),
            },
            "benunit": {
                "benunit_id": np.array([0, 1, 2]),
                "uc_tcl_affected_benunit_proxy": np.array([True, False, True]),
                "uc_tcl_affected_child_count_proxy": np.array([3.0, 0.0, 2.0]),
                "uc_tcl_qualifying_child_count": np.array([4.0, 0.0, 2.0]),
                "uc_tcl_claimant_receives_pip": np.array([True, True, False]),
            },
        }

    def column(self, entity, variable):
        return self.tables[entity][variable]

    def set_column(self, entity, variable, values):
        self.tables.setdefault(entity, {})[variable] = np.asarray(values)

    def entity_reduction(self, reduction):
        assert reduction["reduce"] == "sum"
        assert reduction["entity"] == "person"
        variable = reduction["variable"]
        values = self.tables["person"][variable].astype(float)
        households = self.tables["person"]["person_household_id"]
        return np.asarray(
            [
                values[households == household_id].sum()
                for household_id in self.tables["household"]["household_id"]
            ],
            dtype=float,
        )

    def household_condition(self, condition):
        if condition["variable"] == "region":
            assert condition["map_to"] == "benunit"
            return np.array([True, True, True])
        assert condition["variable"] == "pip"
        assert condition["entity"] == "person"
        assert condition["reduce"] == "sum"
        assert condition["operator"] == ">"
        return self.entity_reduction(condition) > float(condition["value"])

    def parameter(self, name, period):
        assert name == "gov.hmrc.cgt.annual_exempt_amount"
        assert period == 2025
        return 6_000.0

    def counterfactual_delta(self, binding, period):
        return np.zeros(3)


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]


def _local_crosswalk(area_ids: list[str]) -> dict:
    return {
        "country": "uk",
        "levels": {
            "constituency": {
                "expected_vintage": "pcon_2024",
                "area_ids": area_ids,
            }
        },
    }


def _local_fact(value: float, *, area_id: str = "A1", fact_key: str) -> dict:
    return {
        "aggregate_fact_key": fact_key,
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "concept_alignment": {
            "authority": "ons",
            "canonical_concept": "ons.population",
            "relation": "source_label",
            "source_concept": "ons.population",
        },
        "dimensions": {},
        "entity": {"name": "person", "role": "resident"},
        "geography": {
            "level": "constituency",
            "id": area_id,
            "vintage": "pcon_2024",
        },
        "layout": {
            "measure_id": "population",
            "record_set_id": "uk.local_geography.population.age_0_10",
            "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
        },
        "lineage": {"source_record_id": f"{fact_key}.source_record"},
        "observed_measure": {
            "source_concept": "ons.population",
            "source_measure_id": "population",
            "source_name": "ons",
            "source_table": "synthetic local population",
            "unit": "count",
        },
        "period": {"type": "calendar_year", "value": 2025},
        "source": {
            "source_name": "ons",
            "source_table": "synthetic local population",
            "source_file": "fixture.csv",
            "vintage": "fixture",
            "url": "https://example.test/local-population",
        },
        "universe_constraints": {"domain": "population"},
        "value": value,
    }


def test_compile_uk_target_registry_compiles_fixture_subset():
    result = compile_uk_target_registry(_fixture_rows(), target_period=2025)
    names = {spec.name for spec in result.registry.specs}

    assert {
        "obr.income_tax",
        "dwp.uc.two_child_limit.households_affected",
        "slc.repayments.england_plan_2",
        "hmrc.cgt.gains_total",
        "hmrc.cgt.taxpayers_total",
    } <= names
    assert result.unsupported


def test_compile_uk_local_target_registry_compiles_synthetic_area(monkeypatch):
    reference = LedgerTargetReference(
        name="ons.age.0_10@A1",
        ledger_selector={
            "source_name": "ons",
            "source_measure_id": "population",
            "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
            "geography_level": "constituency",
            "geography_id": "A1",
        },
        value_operation="sum",
        entity="person",
        measure="age/0_10",
        period=2025,
        family="ons_population",
        metadata={"contract_target_id": "ons.age.0_10"},
    )
    monkeypatch.setattr(
        "microcosm.build.uk_runtime.ledger_targets.load_country_spec",
        lambda country: SimpleNamespace(local_target_references=(reference,)),
    )

    result = compile_uk_local_target_registry(
        [
            _local_fact(10.0, fact_key="ledger.aggregate_fact.v2:a"),
            _local_fact(20.0, fact_key="ledger.aggregate_fact.v2:b"),
        ],
        target_period=2025,
        crosswalk=_local_crosswalk(["A1"]),
    )

    assert result.unsupported == ()
    assert len(result.registry.specs) == 1
    spec = result.registry.specs[0]
    assert spec.name == "ons.age.0_10@A1"
    assert spec.value == 30.0
    assert spec.metadata["ledger_geography_level"] == "constituency"
    assert spec.metadata["ledger_geography_id"] == "A1"


def test_compile_uk_local_target_registry_refuses_crosswalk_mismatch(monkeypatch):
    reference = LedgerTargetReference(
        name="ons.age.0_10@A9",
        ledger_selector={
            "source_name": "ons",
            "source_measure_id": "population",
            "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
            "geography_level": "constituency",
            "geography_id": "A9",
        },
        value_operation="sum",
        entity="person",
        measure="age/0_10",
        period=2025,
        family="ons_population",
        metadata={"contract_target_id": "ons.age.0_10"},
    )
    monkeypatch.setattr(
        "microcosm.build.uk_runtime.ledger_targets.load_country_spec",
        lambda country: SimpleNamespace(local_target_references=(reference,)),
    )

    with pytest.raises(ValueError, match="A9.*pcon_2024"):
        compile_uk_local_target_registry(
            [_local_fact(10.0, area_id="A9", fact_key="ledger.aggregate_fact.v2:a")],
            target_period=2025,
            crosswalk=_local_crosswalk(["A1"]),
        )


def test_compile_uk_local_target_registry_refuses_wrong_boundary_vintage(
    monkeypatch,
):
    """PR #795 review closing note: the crosswalk's declared vintage is
    operative, not decorative -- a matched fact on a different boundary frame
    fails the compile by name. Equivalent frames (ONS lists devolved areas on
    its lad_2023 lookup over unchanged boundaries) are accept-set members and
    do not refuse."""

    reference = LedgerTargetReference(
        name="ons.age.0_10@A1",
        ledger_selector={
            "source_name": "ons",
            "source_measure_id": "population",
            "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
            "geography_level": "constituency",
            "geography_id": "A1",
        },
        value_operation="sum",
        entity="person",
        measure="age/0_10",
        period=2025,
        family="ons_population",
        metadata={"contract_target_id": "ons.age.0_10"},
    )
    monkeypatch.setattr(
        "microcosm.build.uk_runtime.ledger_targets.load_country_spec",
        lambda country: SimpleNamespace(local_target_references=(reference,)),
    )
    fact = _local_fact(10.0, area_id="A1", fact_key="ledger.aggregate_fact.v2:a")
    fact["geography"]["vintage"] = "pcon_2010"

    with pytest.raises(ValueError, match="pcon_2010.*accepts.*pcon_2024"):
        compile_uk_local_target_registry(
            [fact],
            target_period=2025,
            crosswalk=_local_crosswalk(["A1"]),
        )


def test_materialize_uk_ledger_targets_with_stub_adapter():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="hmrc.cgt.taxpayers_total",
                entity="person",
                measure="hmrc/cgt_taxpayers",
                value=378_000.0,
                source="test",
                metadata={"contract_target_id": "hmrc.cgt.taxpayers_total"},
            ),
            TargetSpec(
                name="dwp.uc.two_child_limit.children_affected",
                entity="benunit",
                measure="dwp/uc/two_child_limit/children_affected",
                value=1.0,
                source="test",
                metadata={
                    "contract_target_id": "dwp.uc.two_child_limit.children_affected"
                },
            ),
            TargetSpec(
                name="dwp.uc.two_child_limit.children_claimant_pip",
                entity="benunit",
                measure="dwp/uc/two_child_limit/adult_pip_children",
                value=1.0,
                source="test",
                metadata={
                    "contract_target_id": (
                        "dwp.uc.two_child_limit.children_claimant_pip"
                    )
                },
            ),
        ],
        country="uk",
    )
    adapter = StubUKAdapter()

    result = materialize_uk_ledger_targets(adapter, registry, period=2025)

    assert result.skipped == ()
    assert adapter.tables["person"]["hmrc/cgt_taxpayers"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    # The binding reads the prepared affected-child count, preserving its
    # difference from the benefit-unit indicator [1.0, 0.0, 1.0].
    assert adapter.tables["benunit"][
        "dwp/uc/two_child_limit/children_affected"
    ].tolist() == [3.0, 0.0, 2.0]
    # Sheet 04B counts all qualifying children in affected claims whose own
    # claimant receives PIP. The first prepared unit contributes four, not
    # its three affected children or its one affected-unit indicator.
    assert adapter.tables["benunit"][
        "dwp/uc/two_child_limit/adult_pip_children"
    ].tolist() == [4.0, 0.0, 0.0]


def _composition_frame():
    """Two households: one lone parent with a child, one older couple."""

    import pandas as pd

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(5),
                    "person_benunit_id": [0, 0, 1, 2, 2],
                    "person_household_id": [0, 0, 0, 1, 1],
                    "is_child": [0.0, 1.0, 0.0, 0.0, 0.0],
                    "age": [40.0, 8.0, 30.0, 70.0, 72.0],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(3),
                    "family_type": ["LONE_PARENT", "SINGLE", "COUPLE_NO_CHILDREN"],
                }
            ),
            "household": pd.DataFrame({"household_id": np.arange(2)}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0, 20.0]), WeightKind.DESIGN)},
        metadata={"time_period": "2025"},
    )


def _uc_composition_frame():
    """Two multibenunit households that distinguish UC claim composition."""

    import pandas as pd

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(6),
                    "person_benunit_id": [0, 0, 1, 2, 2, 3],
                    "person_household_id": [0, 0, 0, 1, 1, 1],
                    "is_child": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(4),
                    "uc_calibration_family_type": [
                        "LONE_PARENT",
                        "SINGLE",
                        "LONE_PARENT",
                        "SINGLE",
                    ],
                    "uc_calibration_administrative_family_type": [
                        "LONE_PARENT",
                        "UNKNOWN",
                        "UNKNOWN",
                        "SINGLE",
                    ],
                    "universal_credit": [100.0, 0.0, 0.0, 100.0],
                    "uc_calibration_child_count": [1, 0, 1, 0],
                }
            ),
            "household": pd.DataFrame(
                {"household_id": np.arange(2), "region": ["LONDON", "SCOTLAND"]}
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0, 20.0]), WeightKind.DESIGN)},
        metadata={"time_period": "2025"},
    )


def test_uc_composition_materializes_at_benunit_grain():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households_children_1",
                entity="benunit",
                measure="dwp/uc/claimants_with_1_children",
                value=1.0,
                source="test",
                metadata={"contract_target_id": "dwp.uc.households_children_1"},
            ),
            TargetSpec(
                name="dwp.uc.households_single_no_children",
                entity="benunit",
                measure="dwp/uc/claimants_single_no_children",
                value=1.0,
                source="test",
                metadata={"contract_target_id": "dwp.uc.households_single_no_children"},
            ),
        ],
        country="uk",
    )
    adapter = UKFrameTargetAdapter(_uc_composition_frame())

    result = materialize_uk_ledger_targets(adapter, registry, period=2025)

    assert result.skipped == ()
    assert adapter.tables["benunit"]["dwp/uc/claimants_with_1_children"].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    # The non-UC single sharing household 0 with the claimant fails its own
    # filters. The childless UC single in household 1 counts even though the
    # other benunit in that dwelling contains a child.
    assert adapter.tables["benunit"][
        "dwp/uc/claimants_single_no_children"
    ].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_uc_national_counts_exclude_ni_and_unknown_regions_at_benunit_grain():
    """The GB source excludes Northern Ireland, not just on the fact selector.

    DWP methodology: Creating a Great Britain dataset. Multiple benefit units
    share each dwelling; order differs between benefit-unit and household ids.
    """
    import pandas as pd

    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    regions = ["NORTHERN_IRELAND", "LONDON", "WALES", "SCOTLAND", "UNKNOWN"]
    household_ids = np.array([4, 3, 2, 1, 0, 4, 3, 2, 1, 0])
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(10),
                    "person_benunit_id": np.arange(10),
                    "person_household_id": household_ids,
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(10),
                    "universal_credit": np.full(10, 29_000.0),
                    "uc_calibration_family_type": ["COUPLE_NO_CHILDREN"] * 10,
                    "uc_calibration_administrative_family_type": ["COUPLE_NO_CHILDREN"]
                    * 10,
                    "uc_calibration_child_count": np.ones(10),
                }
            ),
            "household": pd.DataFrame(
                {"household_id": np.arange(5), "region": regions}
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.ones(5), WeightKind.DESIGN)},
    )
    ids = [
        "dwp.uc.households",
        "dwp.uc.households_children_1",
        "dwp.uc.households_couple_no_children",
        "dwp.uc.payment_distribution_couple_no_children",
    ]
    registry = TargetRegistry(
        [
            TargetSpec(
                name=target_id,
                entity="benunit",
                measure=target_id,
                value=1.0,
                source="synthetic; DWP UC_Households GB",
                metadata={
                    "contract_target_id": target_id,
                    "ledger_filter_monthly_award_amount_bands": "£2400.01 to £2500.00",
                },
            )
            for target_id in ids
        ],
        country="uk",
    )
    adapter = UKFrameTargetAdapter(frame)
    result = materialize_uk_ledger_targets(adapter, registry, period=2025)
    assert not result.skipped
    expected = [0.0, 1.0, 1.0, 1.0, 0.0] * 2
    for target_id in ids:
        assert adapter.tables["benunit"][target_id].tolist() == expected


def test_household_geography_condition_projects_through_benefit_unit_links():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_uc_composition_frame())
    adapter.tables["household"]["region"] = ["NORTHERN_IRELAND", "SCOTLAND"]
    result = adapter.household_condition(
        {
            "entity": "household",
            "variable": "region",
            "reduce": "any",
            "operator": "in",
            "value": ["SCOTLAND"],
            "map_to": "benunit",
        }
    )
    assert result.tolist() == [False, False, True, True]


def test_household_geography_projection_refuses_benunits_spanning_dwellings():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_uc_composition_frame())
    adapter.tables["person"].loc[1, "person_household_id"] = 1
    with pytest.raises(ValueError, match="groups span multiple households"):
        adapter.household_condition(
            {
                "entity": "household",
                "variable": "region",
                "reduce": "any",
                "operator": "==",
                "value": "SCOTLAND",
                "map_to": "benunit",
            }
        )


def test_household_condition_reduces_person_level_conditions():
    # Person-level conditions collapse through person_household_id. Building
    # the group-membership column here would ask for "person_person_id",
    # which cannot exist — the failure that excluded every ONS
    # household-composition reference from calibration.
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_composition_frame())

    children = adapter.household_condition(
        {
            "variable": "is_child",
            "entity": "person",
            "reduce": "sum",
            "operator": ">=",
            "value": 1,
        }
    )

    assert list(children) == [True, False]


def test_household_condition_still_reduces_group_entities():
    from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter

    adapter = UKFrameTargetAdapter(_composition_frame())

    single = adapter.household_condition(
        {
            "variable": "family_type",
            "entity": "benunit",
            "reduce": "any",
            "operator": "==",
            "value": "SINGLE",
        }
    )

    assert list(single) == [True, False]


def test_uk_cross_grain_rule_constants_are_review_pinned():
    assert UK_CROSS_GRAIN_GRAIN_PRECEDENCE == (
        "country",
        "constituency",
        "la",
    )
    assert UK_CROSS_GRAIN_RULE.grain_precedence == (
        "country",
        "constituency",
        "la",
    )
    assert [bridge.bridge_id for bridge in UK_CROSS_GRAIN_BRIDGES] == [
        "national_household_composition_partition_vs_census_households",
        "national_uc_caseload_vs_uc_households_by_area",
        "national_age_0_9_vs_local_age_0_10",
        "national_age_10_19_vs_local_age_10_20",
        "national_age_20_29_vs_local_age_20_30",
        "national_age_30_39_vs_local_age_30_40",
        "national_age_40_49_vs_local_age_40_50",
        "national_age_50_59_vs_local_age_50_60",
        "national_age_60_69_vs_local_age_60_70",
        "national_age_70_79_vs_local_age_70_80",
    ]
    assert UK_CROSS_GRAIN_BRIDGES[0].higher_target_ids == (
        "ons.household_composition.lone_households_under_65",
        "ons.household_composition.lone_households_over_65",
        "ons.household_composition.unrelated_adult_households",
        "ons.household_composition.couple_no_children_households",
        "ons.household_composition.couple_under_3_children_households",
        "ons.household_composition.couple_3_plus_children_households",
        "ons.household_composition.couple_non_dependent_children_only_households",
        "ons.household_composition.lone_parent_dependent_children_households",
        "ons.household_composition.lone_parent_non_dependent_children_households",
        "ons.household_composition.multi_family_households",
    )
    assert UK_CROSS_GRAIN_BRIDGES[1].higher_target_ids == ("dwp.uc.households",)


def test_uk_age_cross_grain_bridges_are_complete_and_exclude_80_89():
    age_bridges = {
        bridge.bridge_id: (
            bridge.concept,
            bridge.higher_target_ids,
            bridge.lower_side,
        )
        for bridge in UK_CROSS_GRAIN_BRIDGES
        if bridge.bridge_id.startswith("national_age_")
    }

    assert age_bridges == {
        f"national_age_{lower}_{upper - 1}_vs_local_age_{lower}_{upper}": (
            "uk.person.count",
            (f"ons.population.age_{lower}_{upper - 1}_by_region",),
            f"contract:ons.age.{lower}_{upper}",
        )
        for lower, upper in zip(range(0, 80, 10), range(10, 90, 10), strict=True)
    }
    assert all(
        "80_89" not in target_id
        for bridge in UK_CROSS_GRAIN_BRIDGES
        for target_id in bridge.higher_target_ids
    )


def test_uk_age_bridge_rescales_constituency_and_la_to_uk_control():
    national_id = "ons.population.age_0_9_by_region"
    local_id = "ons.age.0_10"
    constituency_values = (25.0, 25.0, 25.0, 24.999)
    la_values = (25.0, 25.0, 25.0, 25.001)
    surface = pd.DataFrame(
        [
            {
                "grain": "country",
                "geography_id": "K02000001",
                "target_id": national_id,
                "value": 100.0,
            },
            *[
                {
                    "grain": "constituency",
                    "geography_id": area,
                    "target_id": local_id,
                    "value": value,
                }
                for area, value in zip(
                    ("E14000001", "W07000041", "S14000001", "N06000001"),
                    constituency_values,
                    strict=True,
                )
            ],
            *[
                {
                    "grain": "la",
                    "geography_id": area,
                    "target_id": local_id,
                    "value": value,
                }
                for area, value in zip(
                    ("E06000001", "W06000001", "S12000005", "N09000001"),
                    la_values,
                    strict=True,
                )
            ],
        ]
    )

    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        surface,
        (national_id,),
    )

    assert reconciled.loc[reconciled["grain"] == "constituency", "value"].sum() == (
        pytest.approx(100.0)
    )
    assert reconciled.loc[reconciled["grain"] == "la", "value"].sum() == pytest.approx(
        100.0
    )
    groups = [
        group
        for group in receipt["groups"]
        if group["bridge_id"] == "national_age_0_9_vs_local_age_0_10"
    ]
    assert {group["legs"][0]["reason"] for group in groups} == {
        "standing cross-grain rule: country controls constituency",
        "standing cross-grain rule: country controls la",
    }
    assert all(group["winning_grain"] == "country" for group in groups)
    for group in groups:
        assert len(group["legs"]) == 1
        leg = group["legs"][0]
        assert leg["parent_geography_id"] == "K02000001"
        assert leg["leg"] == "England+Wales+Scotland+Northern Ireland"
        assert leg["new_total"] == pytest.approx(100.0)
        assert leg["declared_factor"] == pytest.approx(1.0, abs=2e-5)


def test_uk_empty_leg_licences_require_complete_signed_deferral_coverage():
    membership = {
        "areas_by_geography_level": {
            "local_authority": ["E06000001", "S12000005", "S12000006"],
        },
        "signed_deferrals": [
            {
                "target_id": "voa.council_tax_stock.by_area.band_a",
                "geography_level": "local_authority",
                "area_ids": ["S12000005", "S12000006"],
            },
            {
                "target_id": "voa.council_tax_stock.by_area.band_b",
                "geography_level": "local_authority",
                "area_ids": ["S12000005"],
            },
        ],
    }

    assert _uk_licensed_empty_legs_from_membership(membership) == {
        "voa.council_tax_stock.by_area.band_a": frozenset({"Scotland"}),
    }


def test_committed_contract_detects_exact_uc_payment_partition():
    payment_ids = (
        "dwp.uc.payment_distribution_single",
        "dwp.uc.payment_distribution_lone_parent",
        "dwp.uc.payment_distribution_couple_no_children",
        "dwp.uc.payment_distribution_couple_with_children",
    )
    surface = pd.DataFrame(
        [
            *[
                {
                    "grain": "country",
                    "geography_id": "K02000001",
                    "target_id": target_id,
                    "value": 25.0,
                }
                for target_id in payment_ids
            ],
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 30.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 20.0,
            },
        ]
    )

    reconciled, receipt = apply_uk_cross_grain_reconciliation(surface, payment_ids)

    assert len(receipt["groups"]) == 1
    assert receipt["groups"][0]["bridge_id"] is None
    assert reconciled.loc[4:, "value"].tolist() == [60.0, 40.0]


def test_council_tax_stock_country_control_rescales_la_band_counts():
    surface = pd.DataFrame(
        [
            {
                "grain": "country",
                "geography_id": "S92000003",
                "target_id": "scotgov.council_tax_stock.band_a",
                "value": 100.0,
            },
            {
                "grain": "la",
                "geography_id": "S12000005",
                "target_id": "voa.council_tax_stock.by_area.band_a",
                "value": 30.0,
            },
            {
                "grain": "la",
                "geography_id": "S12000006",
                "target_id": "voa.council_tax_stock.by_area.band_a",
                "value": 20.0,
            },
        ]
    )

    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        surface, ("scotgov.council_tax_stock.band_a",)
    )

    assert receipt["groups"][0]["bridge_id"] is None
    assert receipt["groups"][0]["winning_grain"] == "country"
    assert reconciled.loc[1:, "value"].tolist() == [60.0, 40.0]


def test_real_uk_bridges_resolve_contract_and_external_lower_sides():
    bridges = {bridge.bridge_id: bridge for bridge in UK_CROSS_GRAIN_BRIDGES}
    household_bridge = bridges[
        "national_household_composition_partition_vs_census_households"
    ]
    uc_bridge = bridges["national_uc_caseload_vs_uc_households_by_area"]
    household_surface = pd.DataFrame(
        [
            *[
                {
                    "grain": "country",
                    "geography_id": "K02000001",
                    "target_id": target_id,
                    "value": 10.0,
                }
                for target_id in household_bridge.higher_target_ids
            ],
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "external:census_households/households",
                "value": 40.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "external:census_households/households",
                "value": 10.0,
            },
        ]
    )
    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        household_surface, household_bridge.higher_target_ids
    )
    assert receipt["groups"][0]["bridge_id"] == household_bridge.bridge_id
    assert reconciled.loc[10:, "value"].tolist() == [80.0, 20.0]

    uc_surface = pd.DataFrame(
        [
            {
                "grain": "country",
                "geography_id": "K02000001",
                "target_id": "dwp.uc.households",
                "value": 90.0,
            },
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 30.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 15.0,
            },
        ]
    )
    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        uc_surface, uc_bridge.higher_target_ids
    )
    assert receipt["groups"][0]["bridge_id"] == uc_bridge.bridge_id
    assert reconciled.loc[1:, "value"].tolist() == [60.0, 30.0]


def test_reviewed_household_composition_gap_leaves_census_ladder_unbound():
    household_bridge = UK_CROSS_GRAIN_BRIDGES[0]
    missing = {
        "ons.household_composition.unrelated_adult_households",
        "ons.household_composition.lone_parent_non_dependent_children_households",
        "ons.household_composition.multi_family_households",
    }
    selected = tuple(
        target_id
        for target_id in household_bridge.higher_target_ids
        if target_id not in missing
    )
    surface = pd.DataFrame(
        [
            *[
                {
                    "grain": "country",
                    "geography_id": "K02000001",
                    "target_id": target_id,
                    "value": 10.0,
                }
                for target_id in selected
            ],
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "external:census_households/households",
                "value": 40.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "external:census_households/households",
                "value": 10.0,
            },
        ]
    )
    reviewed = {
        target_id: {
            "tracking": "microcosm#791",
            "reason": "relationship-to-head is unavailable",
        }
        for target_id in missing
    }

    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        surface,
        selected,
        reviewed_unbound_higher_targets=reviewed,
    )

    assert reconciled.loc[len(selected) :, "value"].tolist() == [40.0, 10.0]
    assert receipt["groups"] == []
    assert receipt["unbound_bridges"] == [
        {
            "bridge_id": household_bridge.bridge_id,
            "missing": sorted(missing),
            "basis": "reviewed_exclusion",
            "records": {
                target_id: reviewed[target_id] for target_id in sorted(missing)
            },
        }
    ]


def test_uk_front_door_reconciles_per_country_legs_and_builds_uniform_surface():
    surface = pd.DataFrame(
        [
            {
                "grain": "country",
                "geography_id": "E92000001",
                "target_id": "dwp.uc.households",
                "value": 60.0,
            },
            {
                "grain": "country",
                "geography_id": "S92000003",
                "target_id": "dwp.uc.households",
                "value": 40.0,
            },
            {
                "grain": "constituency",
                "geography_id": "E14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 30.0,
            },
            {
                "grain": "constituency",
                "geography_id": "S14000001",
                "target_id": "dwp.uc.households_by_area",
                "value": 10.0,
            },
        ]
    )
    reconciled, _ = apply_uk_cross_grain_reconciliation(surface, ("dwp.uc.households",))
    local = reconciled.loc[reconciled["grain"] == "constituency"]
    targets = local.rename(columns={"geography_id": "code", "value": "uc_households"})[
        ["code", "uc_households"]
    ]
    metrics = pd.DataFrame(
        {"uc_households": [1.0, 1.0]},
        index=pd.Index([1, 2], name="household_id"),
    )
    assigned = pd.Series(
        ["E14000001", "S14000001"],
        index=metrics.index,
    )

    from microcosm.build.uk_runtime.local_rowwise import (
        _require_uniform_target_surface,
        build_uk_rowwise_local_matrix,
    )

    problem = build_uk_rowwise_local_matrix(
        metrics,
        assigned,
        targets,
        area_type="constituency",
    )
    _require_uniform_target_surface(problem)
    assert problem.targets.tolist() == [60.0, 40.0]


def test_uk_local_target_surface_refuses_bridge_control_fanout() -> None:
    # dwp.uc.households is a cross-grain bridge control: two cells at one
    # geography would ride through as duplicate reconciliation rows.
    specs = [
        _national_control_spec(
            f"uc-households-cell-{index}", value=value, target_id="dwp.uc.households"
        )
        for index, value in enumerate((40.0, 5.0))
    ]
    with pytest.raises(ValueError, match="bridge control"):
        uk_local_target_surface(
            TargetRegistry(specs, country="uk"),
            _minimal_target_surface_ladder(),
            bound_national_target_ids=("dwp.uc.households",),
            period=2025,
        )


def test_uk_local_target_surface_receipts_a_single_cell_dropped_by_the_fanout_rule() -> (
    None
):
    specs = [
        _national_control_spec(f"payment-band-{index}", value=value)
        for index, value in enumerate((10.0, 20.0))
    ]
    specs.append(
        _national_control_spec(
            "single-cell-elsewhere", value=7.0, geography_id="K02000001"
        )
    )
    _, receipt = uk_local_target_surface(
        TargetRegistry(specs, country="uk"),
        _minimal_target_surface_ladder(),
        bound_national_target_ids=("dwp.uc.payment_distribution_single",),
        period=2025,
    )
    entries = {
        (entry["geography_id"], entry["cells"]): entry["reason"]
        for entry in receipt["fanout_targets_not_controls"]
    }
    assert ("K03000001", 2) in entries
    assert entries[("K02000001", 1)].startswith("Single cell at this geography")


def _households_total_fact(period: int, value: float, **overrides):
    fact = {
        "concept_alignment": {"canonical_concept": "ons.households_total"},
        "geography": {"id": "K02000001", "level": "country"},
        "period": {"type": "calendar_year", "value": period},
        "dimensions": {},
        "value": value,
        "semantic_fact_key": f"ledger.semantic_fact.v2:{period}",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{period}",
        "lineage": {"source_record_id": f"ons.table5.all_households.cy{period}"},
    }
    fact.update(overrides)
    return fact


def test_uk_ledger_households_total_selects_exactly_one_fact() -> None:
    facts = (
        _households_total_fact(2021, 28_119_000.0),
        _households_total_fact(2025, 29_003_000.0),
        _households_total_fact(
            2025,
            1.0,
            concept_alignment={"canonical_concept": "ons.average_household_size"},
        ),
        _households_total_fact(2025, 5.0, geography={"id": "E92000001"}),
        _households_total_fact(2025, 7.0, dimensions={"household_type": "lone"}),
    )
    reference = uk_ledger_households_total(facts, period=2025)
    assert reference == {
        "concept": "ons.households_total",
        "geography_id": "K02000001",
        "period": 2025,
        "value": 29_003_000.0,
        "semantic_fact_key": "ledger.semantic_fact.v2:2025",
        "aggregate_fact_key": "ledger.aggregate_fact.v2:2025",
        "source_record_id": "ons.table5.all_households.cy2025",
    }
    with pytest.raises(ValueError, match="found 0"):
        uk_ledger_households_total(facts, period=2030)
    with pytest.raises(ValueError, match="found 2"):
        uk_ledger_households_total(
            (*facts, _households_total_fact(2025, 29_003_000.0)), period=2025
        )
    with pytest.raises(ValueError, match="positive finite"):
        uk_ledger_households_total((_households_total_fact(2025, 0.0),), period=2025)


def test_uk_ladder_household_uprating_scales_ladder_rows_and_receipts() -> None:
    small_ladder = SimpleNamespace(
        households=np.asarray([10.0, 20.0]),
        constituency_code=np.asarray(["E14000001", "S14000001"]),
        local_authority_code=np.asarray(["E06000001", "S12000005"]),
        metadata={"oa_vintage": "ew:2021_census;scotland:2022_census;ni:dz2021"},
    )
    small_reference = uk_ledger_households_total(
        (_households_total_fact(2025, 33.0),), period=2025
    )
    small = uk_ladder_household_uprating(small_ladder, small_reference, period=2025)
    assert small["applied"] is True
    assert small["factor"] == pytest.approx(1.1)
    assert small["ladder_households_total"] == 30.0
    assert small["ladder_oa_vintage"].startswith("ew:2021_census")
    assert small["reference"] == small_reference
    assert "A15" in small["adjudication"]
    with pytest.raises(ValueError, match="calibration period"):
        uk_ladder_household_uprating(small_ladder, small_reference, period=2024)

    def tenure_spec(name, value, **metadata):
        return TargetSpec(
            name=name,
            entity="household",
            value=value,
            measure="tenure/social_rent",
            period=2025,
            source="ONS",
            family="ons_housing",
            metadata={
                "contract_target_id": "ons.tenure.social_rent",
                "geography_level": "local_authority",
                **metadata,
            },
        )

    registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households@2025",
                entity="household",
                value=90.0,
                measure="uc_households",
                period=2025,
                source="DWP",
                family="dwp_universal_credit",
                metadata={
                    "contract_target_id": "dwp.uc.households",
                    "geography_level": "country",
                    "geography_id": "K03000001",
                },
            ),
            # A census-2021 tenure cell held to 2025 (A17: uprated), a
            # Scottish census-2022 hold (uprated), and a tenure cell compiled
            # from a 2025 fact with no hold (never touched).
            tenure_spec(
                "ons.tenure.social_rent@E06000001@2025",
                50.0,
                geography_id="E06000001",
                uprating_from_period=2021,
                uprating_to_period=2025,
            ),
            tenure_spec(
                "ons.tenure.social_rent@S12000005@2025",
                40.0,
                geography_id="S12000005",
                uprating_from_period=2022,
                uprating_to_period=2025,
            ),
            tenure_spec(
                "ons.tenure.social_rent@E06000002@2025",
                30.0,
                geography_id="E06000002",
            ),
        ],
        country="uk",
    )
    ladder = SimpleNamespace(
        households=np.asarray([10.0, 20.0, 5.0]),
        constituency_code=np.asarray(["E14000001", "S14000001", "E14000002"]),
        local_authority_code=np.asarray(["E06000001", "S12000005", "E06000002"]),
        metadata={"oa_vintage": "ew:2021_census;scotland:2022_census;ni:dz2021"},
    )
    reference = uk_ledger_households_total(
        (_households_total_fact(2025, 38.5),), period=2025
    )
    uprating = uk_ladder_household_uprating(ladder, reference, period=2025)
    assert uprating["factor"] == pytest.approx(1.1)
    as_published, receipt = uk_local_target_surface(
        registry, ladder, bound_national_target_ids=(), period=2025
    )
    assert receipt["ladder_household_uprating"]["applied"] is False
    assert receipt["ladder_household_uprating"]["tenure_cells"]["applied"] is False
    published_rows = as_published.loc[as_published["metric"] == "households"]
    assert published_rows["value"].tolist() == [10.0, 5.0, 20.0, 10.0, 5.0, 20.0]
    tenure_published = as_published.loc[as_published["metric"] == "tenure/social_rent"]
    assert tenure_published["value"].tolist() == [50.0, 40.0, 30.0]

    uprated, receipt = uk_local_target_surface(
        registry,
        ladder,
        bound_national_target_ids=(),
        period=2025,
        ladder_household_uprating=uprating,
    )
    rows = uprated.loc[uprated["metric"] == "households"]
    assert rows["value"].tolist() == pytest.approx([11.0, 5.5, 22.0, 11.0, 5.5, 22.0])
    tenure_rows = uprated.loc[uprated["metric"] == "tenure/social_rent"]
    assert tenure_rows["value"].tolist() == pytest.approx([55.0, 44.0, 30.0])
    tenure_receipt = receipt["ladder_household_uprating"]["tenure_cells"]
    assert tenure_receipt["applied"] is True
    assert tenure_receipt["cells"] == 2
    assert tenure_receipt["by_census_vintage"] == {"2021": 1, "2022": 1}
    assert "A17" in tenure_receipt["adjudication"]
    assert {
        k: v
        for k, v in receipt["ladder_household_uprating"].items()
        if k != "tenure_cells"
    } == uprating
    declined, receipt = uk_local_target_surface(
        registry,
        ladder,
        bound_national_target_ids=(),
        period=2025,
        ladder_household_uprating={"applied": False, "reason": "no facts"},
    )
    assert declined.loc[declined["metric"] == "households", "value"].tolist() == [
        10.0,
        5.0,
        20.0,
        10.0,
        5.0,
        20.0,
    ]
    declined_receipt = receipt["ladder_household_uprating"]
    assert declined_receipt["tenure_cells"]["applied"] is False
    assert {k: v for k, v in declined_receipt.items() if k != "tenure_cells"} == {
        "applied": False,
        "reason": "no facts",
    }
    with pytest.raises(ValueError, match="factor is invalid"):
        uk_local_target_surface(
            registry,
            ladder,
            bound_national_target_ids=(),
            period=2025,
            ladder_household_uprating={"applied": True, "factor": 0.0},
        )


def test_census_vintage_hold_follows_the_ladder_vintage_not_a_constant() -> None:
    from microcosm.build.uk_runtime.ledger_targets import (
        _census_vintage_years,
        _is_census_vintage_hold,
    )

    assert _census_vintage_years("ew:2021_census;scotland:2022_census;ni:dz2021") == {
        2021,
        2022,
    }
    later = _census_vintage_years("ew:2031_census;scotland:2032_census;ni:dz2031")
    held_2031 = {"uprating_from_period": "2031", "uprating_to_period": "2035"}
    assert _is_census_vintage_hold(held_2031, 2035, census_years=later) is True
    # A hold from a non-census vintage never takes the census household factor.
    held_2033 = {"uprating_from_period": "2033", "uprating_to_period": "2035"}
    assert _is_census_vintage_hold(held_2033, 2035, census_years=later) is False
    assert _census_vintage_years("") == frozenset()


@pytest.mark.parametrize(
    "vintage,from_period,reason,eligible",
    [
        (None, 2021, "missing_ladder_oa_vintage", False),
        ("midyear_2024", 2021, "non_census_ladder_vintage", False),
        (
            "ew:2021_census",
            2020,
            "hold_not_from_ladder_census_vintage_or_wrong_period",
            False,
        ),
        ("ew:2021_census", 2021, "census_vintage_hold_uprated", True),
    ],
)
def test_tenure_receipt_counts_attempted_and_skipped_holds(
    vintage, from_period, reason, eligible
):
    ladder = SimpleNamespace(
        households=np.asarray([10.0]),
        constituency_code=np.asarray(["E14000001"]),
        local_authority_code=np.asarray(["E06000001"]),
        metadata={} if vintage is None else {"oa_vintage": vintage},
    )
    reference = uk_ledger_households_total(
        (_households_total_fact(2025, 11.0),), period=2025
    )
    uprating = uk_ladder_household_uprating(ladder, reference, period=2025)
    registry = TargetRegistry(
        [
            TargetSpec(
                name="ons.tenure.social_rent@E06000001@2025",
                entity="household",
                value=5.0,
                measure="tenure/social_rent",
                period=2025,
                source="ONS",
                family="ons_housing",
                metadata={
                    "contract_target_id": "ons.tenure.social_rent",
                    "geography_level": "local_authority",
                    "geography_id": "E06000001",
                    "uprating_from_period": from_period,
                    "uprating_to_period": 2025,
                },
            )
        ],
        country="uk",
    )
    surface, receipt = uk_local_target_surface(
        registry,
        ladder,
        bound_national_target_ids=(),
        period=2025,
        ladder_household_uprating=uprating,
    )
    tenure = receipt["ladder_household_uprating"]["tenure_cells"]
    assert tenure["attempted_cells"] == tenure["total_cells"] == 1
    assert tenure["eligible_cells"] == tenure["cells"] == int(eligible)
    assert tenure["skipped_cells"] == int(not eligible)
    assert tenure["holds"][0]["reason"] == reason
    assert tenure["holds"][0]["ladder_oa_vintage"] == (vintage or "")
    assert surface.loc[
        surface["metric"] == "tenure/social_rent", "value"
    ].tolist() == pytest.approx([5.5 if eligible else 5.0])
