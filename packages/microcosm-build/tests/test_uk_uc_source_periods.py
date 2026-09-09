"""UK UC source windows must not drift when monthly fact availability changes."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.target_reference_authoring import (
    TargetReferenceAuthoringConfig,
    author_target_references,
    target_references_resource,
)
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry
from microcosm.build.uk_runtime.uc_source_periods import (
    EXPECTED_SOURCE_MONTHS,
    uc_source_month_metadata,
)
from tools.generate_uk_target_references import (
    _reference_metadata,
    _value_operation_by_target_id,
)

MONTHS = [f"2025-{month:02}" for month in range(4, 13)]
FISCAL_MONTHS = MONTHS + [f"2026-{month:02}" for month in range(1, 4)]


@pytest.mark.parametrize("family_count", [1, 5])
def test_generator_and_uk_guard_count_months_separately_from_joint_cells(
    monkeypatch, family_count
):
    months = [f"2025-{month:02}" for month in range(1, 13)]
    operands = [
        {
            "dimension_values": {
                "family_type": family,
                "payment_indicator": "In payment",
                "child_entitlement": entitlement,
            }
        }
        for family in [
            "single",
            "couple",
            "single_children",
            "couple_children",
            "unknown",
        ][:family_count]
        for entitlement in ["No", "Yes"]
    ]
    facts = []
    for index, month in enumerate(months):
        for cell, operand in enumerate(operands):
            fact = _fact(month, suffix=f"-{cell}", value=index + cell)
            fact["dimensions"] = operand["dimension_values"]
            fact["layout"]["groupby_dimension"] = "dwp.uc_family_type"
            fact["layout"]["record_set_id"] = f"dwp.uc_family_joint.{month}"
            facts.append(fact)
    target = {
        "target_id": "uc.paid",
        "family": "dwp_universal_credit",
        "value_operation": "monthly_window_sum_average",
        "period_match_policy": "source_window",
        "value_operands": operands,
        "ledger_selector": {
            "source_concept": "dwp.uc_benefit_units",
            "period_type": "month",
            "period_value": months,
            "groupby_dimension": "dwp.uc_family_type",
            "dimensions": ["family_type", "payment_indicator", "child_entitlement"],
            "dimension_values": {
                "family_type": [
                    item["dimension_values"]["family_type"] for item in operands
                ],
                "payment_indicator": "In payment",
                "child_entitlement": ["No", "Yes"],
            },
        },
        "measurement": {"source_months": months},
        "bindings": {
            "policyengine": {"metric_name": "uc_households", "from_entity": "benunit"}
        },
    }
    contract = {"country": "uk", "targets": [target]}
    config = TargetReferenceAuthoringConfig(
        target_period=2025,
        value_operation_by_target_id=_value_operation_by_target_id(contract),
        reference_metadata_by_target_id=_reference_metadata(contract),
    )
    (row,) = author_target_references(contract, facts, config).references
    assert row["value_operands"] == operands
    result = _compile(monkeypatch, facts, LedgerTargetReference(**row))
    assert not result.unsupported
    (spec,) = result.registry.specs
    assert spec.value == 5.5 * len(operands) + sum(range(len(operands)))
    assert spec.metadata["uk_uc_source_month_count"] == "12"
    assert spec.metadata["ledger_member_fact_count"] == str(12 * len(operands))
    assert json.loads(spec.metadata["uk_uc_resolved_source_months"]) == months
    assert (
        spec.metadata["uk_uc_source_period_basis"]
        == "mean_of_declared_monthly_cell_sums"
    )
    mismatch = replace(
        LedgerTargetReference(**row), metadata=uc_source_month_metadata(MONTHS)
    )
    assert (
        "must equal" in _compile(monkeypatch, facts, mismatch).unsupported[0]["reason"]
    )


def test_legacy_calendar_authoring_keeps_original_resource_shape():
    target = {
        "target_id": "uc.calendar",
        "family": "dwp_universal_credit",
        "ledger_selector": {"source_concept": "dwp.uc_benefit_units"},
        "bindings": {
            "policyengine": {"metric_name": "uc_households", "from_entity": "benunit"}
        },
    }
    contract = {"country": "uk", "targets": [target]}
    authored = author_target_references(
        contract,
        [_fact(month) for month in MONTHS],
        TargetReferenceAuthoringConfig(
            target_period=2025,
            value_operation_by_target_id=_value_operation_by_target_id(contract),
        ),
    )
    resource = target_references_resource(
        country="uk", description="Legacy", authored=authored
    )
    assert resource == {
        "country": "uk",
        "description": "Legacy",
        "allowed_value_operations": [
            "identity",
            "sum",
            "difference",
            "calendar_year_average",
            "latest_plateau",
            "count_x_mean",
        ],
        "target_references": [
            {
                "name": "uc.calendar",
                "ledger_selector": target["ledger_selector"],
                "entity": "benunit",
                "measure": "uc_households",
                "family": "dwp_universal_credit",
                "period": 2025,
                "metadata": {
                    "contract_target_id": "uc.calendar",
                    "measure_kind": "prepared_column",
                },
                "value_operation": "calendar_year_average",
                "uprating_from_period": "2025-12",
                "uprating_to_period": 2025,
            }
        ],
    }
    target["value_operation"] = "sum"
    assert _value_operation_by_target_id(contract) == {
        "uc.calendar": "calendar_year_average"
    }


def test_generator_authors_explicit_cross_year_uc_window_and_preserves_model_period(
    monkeypatch,
):
    contract = {
        "country": "uk",
        "targets": [
            {
                "target_id": "uc.fiscal",
                "family": "dwp_universal_credit",
                "value_operation": "monthly_window_average",
                "period_match_policy": "source_window",
                "ledger_selector": {
                    "source_concept": "dwp.uc_benefit_units",
                    "period_type": "month",
                    "period_value": FISCAL_MONTHS,
                },
                "measurement": {"source_months": FISCAL_MONTHS},
                "bindings": {
                    "policyengine": {
                        "metric_name": "uc_households",
                        "from_entity": "benunit",
                    }
                },
            }
        ],
    }
    facts = [_fact(month, value=index) for index, month in enumerate(FISCAL_MONTHS)]
    config = TargetReferenceAuthoringConfig(
        target_period=2025,
        value_operation_by_target_id=_value_operation_by_target_id(contract),
        reference_metadata_by_target_id=_reference_metadata(contract),
    )
    authored = author_target_references(contract, facts, config)
    (row,) = authored.references
    assert row["period_match_policy"] == "source_window"
    assert row["value_operation"] == "monthly_window_average"
    assert row["period"] == 2025
    assert "uprating_from_period" not in row
    assert (
        "monthly_window_average"
        in target_references_resource(
            country="uk", description="Synthetic source window", authored=authored
        )["allowed_value_operations"]
    )
    result = _compile(monkeypatch, facts, LedgerTargetReference(**row))
    assert not result.unsupported
    (target,) = result.registry.specs
    assert target.value == 5.5
    assert target.period == 2025
    assert target.metadata["uk_uc_source_month_count"] == "12"
    assert json.loads(target.metadata["uk_uc_resolved_source_months"]) == FISCAL_MONTHS
    entry = authored.membership_report["targets"]["uc.fiscal"]["candidates"][0]
    assert entry["matched_fact_count_at_or_before_period"] == 9
    assert entry["matched_fact_count_in_source_window"] == 12


def test_calendar_uc_selection_can_explicitly_choose_nine_from_twelve(monkeypatch):
    reference = replace(
        _reference(),
        ledger_selector={
            "source_concept": "dwp.uc_benefit_units",
            "period_type": "month",
            "period_value": MONTHS,
        },
    )
    facts = [_fact(f"2025-{month:02}", value=month) for month in range(1, 13)]
    result = _compile(monkeypatch, facts, reference)
    (target,) = result.registry.specs
    assert target.value == 8
    assert target.metadata["uk_uc_source_month_count"] == "9"


def _reference() -> LedgerTargetReference:
    return LedgerTargetReference(
        name="dwp.uc.households",
        ledger_selector={"source_concept": "dwp.uc_benefit_units"},
        value_operation="calendar_year_average",
        entity="benunit",
        measure="uc_households",
        period=2025,
        family="dwp_universal_credit",
        metadata=uc_source_month_metadata(MONTHS),
    )


def _fact(month: str, *, suffix: str = "", value: float = 100.0) -> dict:
    return {
        "aggregate_fact_key": f"uc-{month}{suffix}",
        "source_release_key": "ledger.source_release.v2:uc-window-fixture",
        "source": {"source_sha256": "a" * 64},
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "geography": {"level": "country", "id": "K03000001"},
        "layout": {
            "groupby_dimension": "dwp.uc_deductions_month",
            "measure_id": "total_units",
            "record_set_id": f"dwp.uc_deductions.{month}.total_units",
        },
        "observed_measure": {
            "source_name": "dwp",
            "source_concept": "dwp.uc_benefit_units",
            "source_measure_id": "total_units",
            "unit": "count",
        },
        "period": {"type": "month", "value": month},
        "value": value,
    }


def _compile(monkeypatch, facts: list[dict], reference=None, *, period=2025):
    monkeypatch.setattr(
        "microcosm.build.uk_runtime.ledger_targets.load_country_spec",
        lambda country: SimpleNamespace(target_references=[reference or _reference()]),
    )
    return compile_uk_target_registry(facts, target_period=period)


def test_declared_window_preserves_mean_and_receipts_each_actual_month(monkeypatch):
    result = _compile(
        monkeypatch,
        [_fact(month, value=100.0 + index) for index, month in enumerate(MONTHS)],
    )
    assert not result.unsupported
    (target,) = result.registry.specs
    assert (target.value, target.entity, target.measure) == (
        104.0,
        "benunit",
        "uc_households",
    )
    assert json.loads(target.metadata[EXPECTED_SOURCE_MONTHS]) == MONTHS
    assert json.loads(target.metadata["uk_uc_resolved_source_months"]) == MONTHS
    assert target.metadata["uk_uc_source_month_count"] == "9"
    assert target.metadata["uk_uc_source_period_basis"] == "mean_of_declared_months"


@pytest.mark.parametrize(
    "observed",
    [
        MONTHS[1:],
        ["2025-01", *MONTHS],
        [f"2025-{month:02}" for month in range(1, 10)],
        [*MONTHS, "2025-09"],
    ],
    ids=["missing", "extra", "same-count-wrong-months", "duplicate-month"],
)
def test_changed_month_coverage_is_unsupported_without_replacement(
    monkeypatch, observed
):
    result = _compile(
        monkeypatch,
        [_fact(month, suffix=f"-{index}") for index, month in enumerate(observed)],
    )
    assert not result.registry.specs
    assert len(result.unsupported) == 1
    assert "expected exactly one fact" in result.unsupported[0]["reason"]
    assert "resolved" in result.unsupported[0]["reason"]


def test_restamping_model_year_does_not_silently_restamp_source_window(monkeypatch):
    result = _compile(
        monkeypatch,
        [_fact(month.replace("2025", "2026")) for month in MONTHS],
        period=2026,
    )
    assert not result.registry.specs
    assert "UK UC source-month coverage" in result.unsupported[0]["reason"]


def test_missing_member_identity_is_reported_instead_of_aborting_compilation(
    monkeypatch,
):
    facts = [_fact(month) for month in MONTHS]
    facts[0].pop("aggregate_fact_key")
    facts[0]["lineage"] = None
    result = _compile(monkeypatch, facts)
    assert not result.registry.specs
    assert "must identify exactly one fact" in result.unsupported[0]["reason"]


def test_single_declared_month_uses_the_resolved_representative_fact(monkeypatch):
    reference = replace(_reference(), metadata=uc_source_month_metadata(["2025-04"]))
    result = _compile(monkeypatch, [_fact("2025-04")], reference)
    (target,) = result.registry.specs
    assert target.metadata["uk_uc_source_month_count"] == "1"
    assert target.value == 100.0


def test_undeclared_non_uc_reference_keeps_existing_available_month_behavior(
    monkeypatch,
):
    reference = replace(_reference(), family="other", metadata={})
    result = _compile(monkeypatch, [_fact("2025-06")], reference)
    (target,) = result.registry.specs
    assert target.value == 100.0
    assert EXPECTED_SOURCE_MONTHS not in target.metadata


@pytest.mark.parametrize(
    "months",
    [
        [],
        "2025-04",
        [4],
        ["2025-13"],
        ["2025-4"],
        MONTHS[::-1],
        [*MONTHS, "2025-12"],
        ["2024-12", "2025-01"],
    ],
)
def test_invalid_source_month_declarations_fail(months):
    with pytest.raises(ValueError, match="source_months"):
        uc_source_month_metadata(months)


def test_generator_preserves_observation_basis_and_explicit_months():
    contract = {
        "targets": [
            {
                "target_id": "uc",
                "family": "dwp_universal_credit",
                "measurement": {
                    "observation_basis": "monthly_stock",
                    "source_months": MONTHS,
                },
            }
        ]
    }
    assert _reference_metadata(contract) == {
        "uc": {"observation_basis": "monthly_stock", **uc_source_month_metadata(MONTHS)}
    }
    contract["targets"][0]["family"] = "other"
    with pytest.raises(ValueError, match="UK UC-only"):
        _reference_metadata(contract)


def test_shipped_uc_monthly_references_preserve_each_declared_window():
    references = load_country_spec("uk").target_references
    monthly = [
        reference
        for reference in references
        if reference.family == "dwp_universal_credit"
    ]
    assert len(monthly) == 111
    new_paid = {
        "dwp.uc.households",
        *{
            f"dwp.uc.households_{family}"
            for family in (
                "single_no_children",
                "single_with_children",
                "couple_no_children",
                "couple_with_children",
            )
        },
        *{
            f"dwp.uc.households_children_{children}"
            for children in ("1", "2", "3", "4", "5_or_more")
        },
    }
    assert len(new_paid) == 10
    for reference in monthly:
        if reference.name in new_paid:
            expected_months = [f"2025-{month:02}" for month in range(1, 13)]
            assert reference.ledger_selector["period_value"] == expected_months
            if reference.value_operation == "monthly_window_sum_average":
                assert reference.period_match_policy == "source_window"
                assert len(reference.value_operands) == (
                    10 if reference.name == "dwp.uc.households" else 2
                )
            else:
                assert reference.value_operation == "calendar_year_average"
                assert reference.name.startswith("dwp.uc.households_children_")
        else:
            expected_months = MONTHS
            assert reference.value_operation == "calendar_year_average"
        assert json.loads(reference.metadata[EXPECTED_SOURCE_MONTHS]) == expected_months
    assert new_paid <= {reference.name for reference in monthly}
    assert not any(
        EXPECTED_SOURCE_MONTHS in reference.metadata
        for reference in references
        if reference.family != "dwp_universal_credit"
    )
