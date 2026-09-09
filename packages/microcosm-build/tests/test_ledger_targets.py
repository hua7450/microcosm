import json
from dataclasses import asdict, replace

import pytest

from microcosm.build.gates import (
    TargetCoverageRequirement,
    target_profile_coverage_gate,
)
from microcosm.build.ledger_targets import (
    LedgerTargetMapping,
    LedgerTargetReference,
    apply_ledger_target_profile,
    compile_ledger_target_references,
    ledger_target_registry_parity_report,
    period_values_semantically_equal,
    select_ledger_targets,
    select_ledger_targets_from_jsonl,
    target_spec_from_ledger_reference,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

_FISCAL_MONTHS = [f"2025-{month:02d}" for month in range(4, 13)] + [
    f"2026-{month:02d}" for month in range(1, 4)
]


def _window_fact_row(month, *, value):
    row = _monthly_consumer_fact_row(month, value=value)
    row["source"]["vintage"] = "release_2026_06"
    row["source"]["source_file"] = "published-monthly-series.csv"
    row["source_release_key"] = "ledger.source_release.v2:window-fixture"
    row["source"]["source_sha256"] = "a" * 64
    return row


def _window_reference(**overrides):
    fields = {
        "name": "explicit fiscal observation window",
        "ledger_selector": {
            "source_name": "cms_medicaid",
            "period_type": "month",
            "period_value": _FISCAL_MONTHS,
        },
        "value_operation": "monthly_window_average",
        "period_match_policy": "source_window",
        "entity": "person",
        "measure": "enrollment",
        "period": 2025,
    }
    fields.update(overrides)
    return LedgerTargetReference(**fields)


def test_explicit_monthly_window_averages_cross_year_without_restamping_model():
    rows = [
        _window_fact_row(month, value=index)
        for index, month in enumerate(_FISCAL_MONTHS)
    ]
    # These observations are deliberately outside the selected window.
    rows += [
        _window_fact_row("2025-03", value=1000),
        _window_fact_row("2026-04", value=1000),
    ]
    (target,) = compile_ledger_target_references(
        rows[::-1], [_window_reference()], country="us"
    ).specs
    assert target.period == 2025
    assert target.value == 5.5  # Mean of the twelve source cells, including zero.
    assert target.metadata["ledger_member_fact_count"] == "12"
    assert target.metadata["ledger_fact_period"] == "2026-03"
    assert target.metadata["ledger_period_match_policy"] == "source_window"
    assert json.loads(target.metadata["ledger_source_months"]) == _FISCAL_MONTHS
    assert target.metadata["ledger_source_month_count"] == "12"
    assert (
        target.metadata["ledger_source_release_key"]
        == "ledger.source_release.v2:window-fixture"
    )
    assert target.metadata["ledger_source_sha256"] == "a" * 64


@pytest.mark.parametrize("defect", ["missing", "duplicate", "other-series"])
def test_monthly_window_requires_complete_unique_single_series(defect):
    rows = [_window_fact_row(month, value=1) for month in _FISCAL_MONTHS]
    if defect == "missing":
        rows.pop(0)
    elif defect == "duplicate":
        rows.append(_window_fact_row(_FISCAL_MONTHS[0], value=2))
        rows[-1]["aggregate_fact_key"] += "-duplicate"
    else:
        rows[0]["layout"]["groupby_value_id"] = "other-enrollment"
    with pytest.raises(ValueError, match="monthly window"):
        compile_ledger_target_references(rows, [_window_reference()], country="us")


@pytest.mark.parametrize(
    "fields",
    [
        {"value_operation": "sum"},
        {"period_match_policy": "latest_not_after"},
        {"period_match_policy": "exact"},
        {"period": None},
        {"ledger_selector": {"period_type": "year", "period_value": _FISCAL_MONTHS}},
        {"ledger_selector": {"period_type": "month", "period_value": []}},
        {"ledger_selector": {"period_type": "month", "period_value": "2025-04"}},
        {"ledger_selector": {"period_type": "month", "period_value": ["2025-4"]}},
        {"ledger_selector": {"period_type": "month", "period_value": ["2025-13"]}},
        {
            "ledger_selector": {
                "period_type": "month",
                "period_value": _FISCAL_MONTHS[::-1],
            }
        },
        {
            "ledger_selector": {
                "period_type": "month",
                "period_value": ["2025-04", "2025-04"],
            }
        },
    ],
)
def test_monthly_window_requires_paired_policy_and_explicit_ordered_months(fields):
    with pytest.raises(ValueError):
        _window_reference(**fields)


def test_monthly_window_direct_fact_route_cannot_bypass_coverage_or_selector():
    reference = _window_reference()
    with pytest.raises(ValueError, match="monthly window"):
        target_spec_from_ledger_reference(
            _window_fact_row("2026-03", value=99), reference
        )
    rows = [_window_fact_row(month, value=1) for month in _FISCAL_MONTHS]
    rows[-1]["period"]["value"] = "2026-04"
    with pytest.raises(ValueError, match="monthly window"):
        target_spec_from_ledger_reference(tuple(rows), reference)


def test_calendar_average_keeps_same_year_selection_and_future_fact_refusal():
    reference = _window_reference(
        value_operation="calendar_year_average", period_match_policy="latest_not_after"
    )
    rows = [
        _window_fact_row(month, value=index)
        for index, month in enumerate(_FISCAL_MONTHS)
    ]
    (target,) = compile_ledger_target_references(rows, [reference], country="us").specs
    assert target.value == 4  # Existing calendar semantics consume April–December.
    assert target.metadata["ledger_member_fact_count"] == "9"
    with pytest.raises(ValueError, match="at or before"):
        target_spec_from_ledger_reference(rows[-1], reference)


def _window_cell_rows():
    rows = []
    for index, month in enumerate(_FISCAL_MONTHS):
        for category, value in (("No", index), ("Yes", index + 20)):
            row = _window_fact_row(month, value=value)
            row["aggregate_fact_key"] += category
            row["legacy_fact_key"] += category
            row["semantic_fact_key"] += category
            row["dimensions"] = {"family": "unknown", "entitlement": category}
            row["source"]["vintage"] = "release_2026_06"
            row["source"]["source_file"] = "published-joint.csv"
            rows.append(row)
    return rows


def _sum_window_reference(**overrides):
    fields = {
        "value_operation": "monthly_window_sum_average",
        "value_operands": tuple(
            {"dimension_values": {"family": "unknown", "entitlement": category}}
            for category in ("No", "Yes")
        ),
    }
    fields.update(overrides)
    return _window_reference(**fields)


def test_monthly_window_sums_declared_cells_before_averaging_months():
    (target,) = compile_ledger_target_references(
        _window_cell_rows()[::-1], [_sum_window_reference()], country="us"
    ).specs
    assert target.value == 31  # Mean(index + index + 20), not mean of 24 cells.
    assert target.metadata["ledger_member_fact_count"] == "24"
    assert target.metadata["ledger_source_month_count"] == "12"
    assert target.metadata["ledger_source_cell_count_per_month"] == "2"
    assert len(json.loads(target.metadata["ledger_member_fact_keys"])) == 24


@pytest.mark.parametrize(
    "defect",
    [
        "missing-cell",
        "duplicate-cell",
        "extra-cell",
        "source-release",
        "measure",
        "geography",
        "entity",
        "source-package",
        "release-key",
        "raw-hash",
        "hidden-universe",
    ],
)
def test_monthly_window_cell_grid_refuses_incomplete_or_incompatible_members(defect):
    rows = _window_cell_rows()
    if defect == "missing-cell":
        rows.pop()
    elif defect in {"duplicate-cell", "extra-cell"}:
        row = _window_cell_rows()[0]
        row["aggregate_fact_key"] += "extra"
        row["legacy_fact_key"] += "extra"
        row["semantic_fact_key"] += "extra"
        if defect == "extra-cell":
            row["dimensions"]["entitlement"] = "all"
        rows.append(row)
    elif defect == "source-release":
        rows[0]["source"]["vintage"] = "different_release"
    elif defect == "source-package":
        rows[0]["layout"]["record_set_id"] = "different_package.2025-04"
    elif defect == "release-key":
        rows[0]["source_release_key"] = "other-release"
    elif defect == "raw-hash":
        rows[0]["source"]["source_sha256"] = "other-raw-source"
    elif defect == "hidden-universe":
        rows[0]["universe_constraints"]["constraints"] = [
            {"variable": "hidden_subset", "operator": "==", "value": "other"}
        ]
    elif defect == "measure":
        rows[0]["observed_measure"]["source_measure_id"] = "different_measure"
    elif defect == "geography":
        rows[0]["geography"]["id"] = "different_geography"
    else:
        rows[0]["entity"]["name"] = "different_entity"
    with pytest.raises(ValueError, match="monthly window"):
        compile_ledger_target_references(rows, [_sum_window_reference()], country="us")


@pytest.mark.parametrize(
    "operands",
    [
        (),
        ({"dimension_values": {"family": ["single", "couple"]}},),
        ({"dimension_values": {"family": "single"}},) * 2,
        (
            {"dimension_values": {"family": "single"}},
            {"dimension_values": {"entitlement": "Yes"}},
        ),
    ],
)
def test_monthly_window_sum_requires_disjoint_complete_scalar_operands(operands):
    with pytest.raises(ValueError, match="monthly window"):
        _sum_window_reference(value_operands=operands)


def test_monthly_window_does_not_override_projection_policy_or_declared_grid_size():
    rows = _window_cell_rows()
    rows[-1]["assertion"] = "source_projection"
    with pytest.raises(ValueError, match="monthly window"):
        compile_ledger_target_references(rows, [_sum_window_reference()], country="us")
    with pytest.raises(ValueError, match="assertion_policy"):
        target_spec_from_ledger_reference(tuple(rows), _sum_window_reference())
    (target,) = compile_ledger_target_references(
        rows,
        [_sum_window_reference(assertion_policy="allow_source_projection")],
        country="us",
    ).specs
    assert target.value == 31
    with pytest.raises(ValueError, match="expected_member_count"):
        compile_ledger_target_references(
            rows,
            [
                _sum_window_reference(
                    assertion_policy="allow_source_projection", expected_member_count=12
                )
            ],
            country="us",
        )


def test_monthly_window_preserves_distinct_fact_identity_for_each_member():
    rows = _window_cell_rows()
    rows[-1]["aggregate_fact_key"] = rows[0]["aggregate_fact_key"]
    with pytest.raises(ValueError, match="member fact identities"):
        target_spec_from_ledger_reference(tuple(rows), _sum_window_reference())


@pytest.mark.parametrize("field", ["source_release_key", "source_sha256"])
def test_single_series_monthly_window_refuses_mixed_publications(field):
    rows = [_window_fact_row(month, value=1) for month in _FISCAL_MONTHS]
    if field == "source_release_key":
        rows[0][field] = "other-release"
    else:
        rows[0]["source"][field] = "other-raw-hash"
    with pytest.raises(ValueError, match="source publication"):
        compile_ledger_target_references(rows, [_window_reference()], country="us")


def _ledger_fact(**overrides):
    fact = {
        "fact_key": "ledger.fact.v1:abc123",
        "source_record_id": "irs_soi.ty2023.table_1_1.all.adjusted_gross_income",
        "value": 15_286_017_359_000,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "country",
            "id": "0100000US",
            "name": "United States",
            "vintage": "2020_census",
        },
        "entity": {"name": "tax_unit", "role": "filing_unit"},
        "measure": {
            "concept": "us:statutes/26/62#adjusted_gross_income",
            "unit": "usd",
            "source_concept": "irs_soi.adjusted_gross_income",
            "concept_relation": "exact",
            "concept_authority": "ledger-us",
            "legal_vintage": "tax_year_2023",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_file": "23in11si.xls",
            "url": "https://www.irs.gov/pub/irs-soi/23in11si.xls",
            "vintage": "tax_year_2023",
        },
        "filters": {"income_range": "all", "filing_status": "all"},
        "domain": "all_individual_income_tax_returns",
        "layout": {
            "record_set_id": "irs_soi.ty2023.table_1_1",
            "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            "groupby_value_id": "all",
            "measure_id": "adjusted_gross_income",
        },
    }
    fact.update(overrides)
    return fact


def _consumer_fact_row(**overrides):
    row = {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:abc123",
        "legacy_fact_key": "ledger.fact.v1:abc123",
        "lineage": {
            "source_record_id": "irs_soi.ty2023.table_1_1.all.adjusted_gross_income",
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
            "source_row_keys": [],
        },
        "value": 15_286_017_359_000,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "country",
            "id": "0100000US",
            "name": "United States",
            "vintage": "2020_census",
        },
        "entity": {"name": "tax_unit", "role": "filing_unit"},
        "observed_measure": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_measure_id": "adjusted_gross_income",
            "source_concept": "irs_soi.adjusted_gross_income",
            "unit": "usd",
        },
        "concept_alignment": {
            "source_concept": "irs_soi.adjusted_gross_income",
            "canonical_concept": "us:statutes/26/62#adjusted_gross_income",
            "relation": "exact",
            "authority": "ledger-us",
            "legal_vintage": "tax_year_2023",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_file": "23in11si.xls",
            "url": "https://www.irs.gov/pub/irs-soi/23in11si.xls",
            "vintage": "tax_year_2023",
        },
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "universe_constraints": {"domain": "all_individual_income_tax_returns"},
        "layout": {
            "record_set_id": "irs_soi.ty2023.table_1_1",
            "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            "groupby_value_id": "all",
            "measure_id": "adjusted_gross_income",
        },
    }
    row.update(overrides)
    return row


def _consumer_fact_row_for_period(source_period: int, *, value: float):
    return _consumer_fact_row(
        aggregate_fact_key=f"ledger.aggregate_fact.v2:agi-{source_period}",
        legacy_fact_key=f"ledger.fact.v1:agi-{source_period}",
        semantic_fact_key=f"ledger.semantic_fact.v2:agi-{source_period}",
        lineage={
            "source_record_id": (
                f"irs_soi.ty{source_period}.table_1_1.all.adjusted_gross_income"
            ),
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
            "source_row_keys": [],
        },
        value=value,
        period={"type": "tax_year", "value": source_period},
        source={
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_file": f"{str(source_period)[-2:]}in11si.xls",
            "url": f"https://www.irs.gov/pub/irs-soi/{str(source_period)[-2:]}in11si.xls",
            "vintage": f"tax_year_{source_period}",
        },
        layout={
            "record_set_id": f"irs_soi.ty{source_period}.table_1_1",
            "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            "groupby_value_id": "all",
            "measure_id": "adjusted_gross_income",
        },
    )


def _exact_agi_reference(**overrides) -> LedgerTargetReference:
    values = {
        "name": "exact SOI AGI total",
        "ledger_selector": {
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "period_type": "tax_year",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        "entity": "tax_unit",
        "measure": "adjusted_gross_income",
        "period": 2023,
        "family": "irs_soi",
        "period_match_policy": "exact",
    }
    values.update(overrides)
    return LedgerTargetReference(**values)


def _monthly_consumer_fact_row(source_period: str, *, value: float):
    normalized_period = source_period.replace("-", "_")
    return _consumer_fact_row(
        aggregate_fact_key=f"ledger.aggregate_fact.v2:medicaid-{normalized_period}",
        legacy_fact_key=f"ledger.fact.v1:medicaid-{normalized_period}",
        semantic_fact_key=f"ledger.semantic_fact.v2:medicaid-{normalized_period}",
        lineage={
            "source_record_id": (
                f"cms_medicaid.month{normalized_period}."
                "us.total_medicaid_chip_enrollment"
            ),
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
            "source_row_keys": [],
        },
        value=value,
        period={"type": "month", "value": source_period},
        source={
            "source_name": "cms_medicaid",
            "source_table": "Medicaid and CHIP enrollment",
            "source_file": f"enrollment_{normalized_period}.csv",
            "url": "https://data.medicaid.gov/",
            "vintage": f"month_{normalized_period}",
        },
        entity={"name": "person"},
        layout={
            "record_set_id": f"cms_medicaid.month{normalized_period}",
            "groupby_dimension": "program",
            "groupby_value_id": "total_medicaid_chip_enrollment",
            "measure_id": "total_medicaid_chip_enrollment",
        },
        observed_measure={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "source_concept": "cms.total_medicaid_chip_enrollment",
            "unit": "people",
        },
    )


def _spi_area_fact(
    measure_id: str,
    *,
    value: float,
    aggregation: str,
    fact_key: str,
) -> dict[str, object]:
    return _consumer_fact_row(
        aggregate_fact_key=fact_key,
        legacy_fact_key=fact_key.replace("aggregate_fact.v2", "fact.v1"),
        semantic_fact_key=fact_key.replace("aggregate_fact.v2", "semantic_fact.v2"),
        lineage={
            "source_record_id": f"hmrc.spi.local.{measure_id}",
            "source_cell_keys": ["ledger.source_cell.v1:spi"],
            "source_row_keys": [],
        },
        value=value,
        period={"type": "tax_year", "value": 2025},
        geography={
            "level": "constituency",
            "id": "E14000001",
            "name": "Example constituency",
            "vintage": "pcon_2024",
        },
        entity={"name": "person", "role": "taxpayer"},
        observed_measure={
            "source_name": "hmrc",
            "source_measure_id": measure_id,
            "source_concept": f"hmrc.{measure_id}",
            "unit": "count" if measure_id.endswith("_count") else "gbp",
        },
        concept_alignment={
            "source_concept": f"hmrc.{measure_id}",
            "canonical_concept": f"hmrc.{measure_id}",
            "relation": "exact",
            "authority": "hmrc",
            "legal_vintage": "tax_year_2025",
        },
        aggregation={"method": aggregation},
        source={
            "source_name": "hmrc",
            "source_table": "Synthetic SPI local income by area",
            "source_file": "synthetic.ods",
            "url": "https://example.invalid/synthetic-spi",
            "vintage": "tax_year_2025",
        },
        dimensions={},
        layout={
            "record_set_id": "hmrc.spi.local.by_area",
            "record_set_spec_id": "uk.local_geography.spi_income.by_constituency.v1",
            "groupby_dimension": "",
            "groupby_value_id": "",
            "measure_id": measure_id,
        },
    )


def _hierarchy_target(
    name: str,
    *,
    value: float,
    geography_level: str,
    geography_id: str,
    state_fips: str | None = None,
) -> TargetSpec:
    metadata = {
        "ledger_geography_level": geography_level,
        "ledger_geography_id": geography_id,
        "target_role": "population_age",
        "source_measure_id": "population",
        "target_period": "2024",
        "materializer": "population_age",
        "measure_mode": "indicator_sum",
        "age_lower_bound": "0",
        "age_upper_bound": "5",
        "age_group": "age_0_to_4",
    }
    if state_fips is not None:
        metadata["state_fips"] = state_fips
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        source="source",
        family="census_population",
        metadata=metadata,
    )


def _hierarchy_profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "hierarchy_reconciliations": [
            {
                "id": "test_cd_to_state",
                "method": "scale_children_to_parent",
                "enabled_when": {"apply_hierarchy": True},
                "child_geography_level": "congressional_district",
                "parent_geography_level": "state",
                "parent_geography_id": {"template": "0400000US{state_fips}"},
                "child_completeness": {
                    "child_id_metadata_key": "ledger_geography_id",
                    "parent_key_metadata_key": "state_fips",
                    "on_incomplete": "fail",
                    "expected_child_count_by_parent_key": {"01": 2},
                },
                "match": {
                    "spec_fields": ["entity", "period", "family", "filter"],
                    "metadata_keys": [
                        "target_period",
                        "target_role",
                        "source_measure_id",
                        "materializer",
                        "measure_mode",
                        "age_lower_bound",
                        "age_upper_bound",
                        "age_group",
                    ],
                },
            }
        ],
    }


def test__given_supported_ledger_fact__then_microcosm_target_preserves_lineage() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        family_by_source_name={"irs_soi": "irs_soi"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact()], mapping)
    registry = selection.to_registry(country="us")

    # Then
    assert not selection.unsupported
    assert len(registry) == 1
    spec = registry.specs[0]
    assert spec.name == "ledger.fact.v1:abc123"
    assert spec.measure == "adjusted_gross_income"
    assert spec.filter == "is_tax_return"
    assert spec.family == "irs_soi"
    assert spec.metadata["ledger_source"] == "policyengine-ledger-data"
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )
    assert spec.metadata["ledger_filter_income_range"] == "all"
    assert spec.metadata["ledger_layout_record_set_id"] == "irs_soi.ty2023.table_1_1"


def test__given_consumer_contract_row__then_microcosm_target_preserves_lineage() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_consumer_fact_row()], mapping)

    # Then
    assert not selection.unsupported
    spec = selection.specs[0]
    assert spec.name == "ledger.aggregate_fact.v2:abc123"
    assert spec.measure == "adjusted_gross_income"
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )
    assert spec.metadata["ledger_fact_key"] == "ledger.aggregate_fact.v2:abc123"
    assert spec.metadata["ledger_source_concept"] == "irs_soi.adjusted_gross_income"


def test__given_consumer_contract_jsonl__then_microcosm_selects_targets(
    tmp_path,
) -> None:
    # Given
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text(json.dumps(_consumer_fact_row()) + "\n")
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets_from_jsonl(path, mapping)

    # Then
    assert not selection.unsupported
    assert selection.specs[0].name == "ledger.aggregate_fact.v2:abc123"


def test__given_ledger_target_reference__then_it_compiles_model_mapping() -> None:
    # Given
    reference = LedgerTargetReference(
        name="nation/irs/adjusted gross income/total",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        entity="tax_unit",
        measure="adjusted_gross_income",
        filter="is_tax_return",
        period=2024,
        source="IRS SOI Table 1.1",
        family="irs_soi",
        metadata={"target_role": "soi_fiscal_distribution"},
        uprating_index="cpi_u",
        uprating_from_period=2023,
        uprating_to_period=2024,
    )

    # When
    registry = compile_ledger_target_references(
        [_consumer_fact_row()],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.name == "nation/irs/adjusted gross income/total"
    assert spec.entity == "tax_unit"
    assert spec.measure == "adjusted_gross_income"
    assert spec.filter == "is_tax_return"
    assert spec.period == 2024
    assert spec.source.startswith("irs_soi | Publication 1304 Table 1.1")
    assert spec.metadata["target_role"] == "soi_fiscal_distribution"
    assert spec.metadata["uprating_index"] == "cpi_u"
    assert spec.metadata["uprating_from_period"] == "2023"
    assert spec.metadata["uprating_to_period"] == "2024"


def test__given_mean_fact_without_time_mean_contract__then_compilation_fails() -> None:
    # Given a mean-aggregated fact whose Microcosm mapping does NOT declare the
    # fact_aggregation=time_mean contract
    reference = LedgerTargetReference(
        name="usda_snap.fy2024.national_average_monthly_persons.national_total",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        entity="household",
        measure="snap_person_indicator",
        period=2024,
        source="USDA FNS",
        family="usda_snap",
    )

    # When / Then
    with pytest.raises(ValueError, match="unsupported aggregation 'mean'"):
        compile_ledger_target_references(
            [
                _consumer_fact_row(
                    value=41_700_000,
                    aggregation={"method": "mean"},
                    observed_measure={
                        "source_name": "usda_snap",
                        "source_table": "SNAP participation summary",
                        "source_measure_id": "average_monthly_persons",
                        "source_concept": "usda_snap.average_monthly_persons",
                        "unit": "count",
                    },
                )
            ],
            [reference],
            country="us",
        )


def test__given_count_ledger_target_reference__then_compilation_fails() -> None:
    # Given
    reference = LedgerTargetReference(
        name="census_pep.cy2024.national_resident_population_age.0_to_4.population",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        entity="person",
        measure="person_count",
        filter="age_0_to_4",
        period=2024,
        source="Census PEP",
        family="census_population",
    )

    # When / Then
    with pytest.raises(ValueError, match="unsupported aggregation 'count'"):
        compile_ledger_target_references(
            [
                _consumer_fact_row(
                    value=18_000_000,
                    aggregation={"method": "count"},
                    observed_measure={
                        "source_name": "census_pep",
                        "source_table": ("Annual Estimates of the Resident Population"),
                        "source_measure_id": "population",
                        "source_concept": "census_pep.resident_population",
                        "unit": "count",
                    },
                )
            ],
            [reference],
            country="us",
        )


def test__given_duplicate_semantic_facts__then_aggregate_reference_still_compiles() -> (
    None
):
    # Given
    first = _consumer_fact_row(semantic_fact_key="ledger.semantic_fact.v2:shared")
    second = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:def456",
        legacy_fact_key="ledger.fact.v1:def456",
        semantic_fact_key="ledger.semantic_fact.v2:shared",
        lineage={
            "source_record_id": "irs_soi.ty2023.table_1_1.all.total_tax",
            "source_cell_keys": ["ledger.source_cell.v1:cell2"],
            "source_row_keys": [],
        },
        value=2_000_000_000_000,
        observed_measure={
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_measure_id": "total_tax",
            "source_concept": "irs_soi.total_income_tax",
            "unit": "usd",
        },
        concept_alignment={
            "source_concept": "irs_soi.total_income_tax",
            "canonical_concept": "us:statutes/26/1#income_tax",
            "relation": "exact",
            "authority": "ledger-us",
            "legal_vintage": "tax_year_2023",
        },
        layout={
            "record_set_id": "irs_soi.ty2023.table_1_1",
            "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            "groupby_value_id": "all",
            "measure_id": "total_tax",
        },
    )
    reference = LedgerTargetReference(
        name="nation/irs/total tax/total",
        ledger_fact_key="ledger.aggregate_fact.v2:def456",
        entity="tax_unit",
        measure="income_tax",
        period=2024,
        family="irs_soi",
    )

    # When
    registry = compile_ledger_target_references(
        [first, second],
        [reference],
        country="us",
    )

    # Then
    assert registry.specs[0].value == 2_000_000_000_000
    assert (
        registry.specs[0].metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.total_tax"
    )


def test__given_ambiguous_ledger_reference_identifier__then_compilation_fails() -> None:
    # Given
    first = _consumer_fact_row(semantic_fact_key="ledger.semantic_fact.v2:shared")
    second = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:def456",
        legacy_fact_key="ledger.fact.v1:def456",
        semantic_fact_key="ledger.semantic_fact.v2:shared",
        lineage={
            "source_record_id": "irs_soi.ty2023.table_1_1.all.total_tax",
            "source_cell_keys": ["ledger.source_cell.v1:cell2"],
            "source_row_keys": [],
        },
    )
    reference = LedgerTargetReference(
        name="ambiguous semantic reference",
        ledger_fact_key="ledger.semantic_fact.v2:shared",
        entity="tax_unit",
        measure="adjusted_gross_income",
    )

    # When / Then
    with pytest.raises(ValueError, match="matched multiple Ledger facts"):
        compile_ledger_target_references(
            [first, second],
            [reference],
            country="us",
        )


def test__given_missing_ledger_reference_fact__then_compilation_fails() -> None:
    # Given
    reference = LedgerTargetReference(
        name="missing fact target",
        ledger_fact_key="ledger.aggregate_fact.v2:missing",
        entity="tax_unit",
        measure="adjusted_gross_income",
    )

    # When / Then
    with pytest.raises(ValueError, match="did not match a Ledger fact identifier"):
        compile_ledger_target_references(
            [_consumer_fact_row()], [reference], country="us"
        )


def test__given_selector_matches_multiple_years__then_latest_source_period_is_used() -> (
    None
):
    # Given
    reference = LedgerTargetReference(
        name="latest SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _consumer_fact_row_for_period(2022, value=14_000_000_000_000),
            _consumer_fact_row_for_period(2023, value=15_000_000_000_000),
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.value == 15_000_000_000_000
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )


def _fiscal_year_row(label: int, *, value: float, coverage: bool):
    row = _consumer_fact_row_for_period(label, value=value)
    row["period"] = {"type": "fiscal_year", "value": label}
    if coverage:
        # DfT labels the reporting year by its March end year: label 2025
        # covers April 2024 to March 2025.
        row["period_coverage"] = {
            "basis": "fiscal",
            "start_date": f"{label - 1}-04-01",
            "end_date": f"{label}-03-31",
            "notes": "DfT labels the reporting year by its March end year.",
        }
    return row


def test__given_fiscal_year_facts_with_coverage__then_the_start_year_is_compared() -> (
    None
):
    # Given: a closing-year publisher (label 2026 covers FY2025-26) and the
    # 2025 calibration period.
    reference = LedgerTargetReference(
        name="latest SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2025,
        family="irs_soi",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _fiscal_year_row(2025, value=1.0, coverage=True),
            _fiscal_year_row(2026, value=2.0, coverage=True),
            _fiscal_year_row(2027, value=3.0, coverage=True),
        ],
        [reference],
        country="us",
    )

    # Then: the label-2026 fact is FY2025-26, the latest not after 2025.
    spec = registry.specs[0]
    assert spec.value == 2.0
    assert spec.metadata["ledger_fact_period"] == "2025"
    assert spec.metadata["ledger_fact_period_label"] == "2026"


def test__given_fiscal_year_facts_without_coverage__then_the_label_is_compared() -> (
    None
):
    reference = LedgerTargetReference(
        name="latest SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2025,
        family="irs_soi",
    )

    registry = compile_ledger_target_references(
        [
            _fiscal_year_row(2025, value=1.0, coverage=False),
            _fiscal_year_row(2026, value=2.0, coverage=False),
        ],
        [reference],
        country="us",
    )

    spec = registry.specs[0]
    assert spec.value == 1.0
    assert spec.metadata["ledger_fact_period"] == "2025"
    assert "ledger_fact_period_label" not in spec.metadata


def test__given_exact_period_policy__then_only_the_target_period_is_used() -> None:
    registry = compile_ledger_target_references(
        [
            _consumer_fact_row_for_period(2022, value=14_000_000_000_000),
            _consumer_fact_row_for_period(2023, value=15_000_000_000_000),
        ],
        [_exact_agi_reference()],
        country="us",
    )

    (spec,) = registry.specs
    assert spec.value == 15_000_000_000_000
    assert spec.period == 2023
    assert spec.metadata["ledger_period_match_policy"] == "exact"


@pytest.mark.parametrize(
    ("reference_period", "fact_period"),
    [
        (2023, "tax_year_2023"),
        ("tax_year_2023", 2023),
    ],
)
def test__given_equivalent_exact_period_labels__then_target_period_is_used(
    reference_period, fact_period
) -> None:
    fact = _consumer_fact_row_for_period(2023, value=15_000_000_000_000)
    fact["period"]["value"] = fact_period

    registry = compile_ledger_target_references(
        [fact],
        [_exact_agi_reference(period=reference_period)],
        country="us",
    )

    (spec,) = registry.specs
    assert spec.value == 15_000_000_000_000
    assert spec.period == reference_period


@pytest.mark.parametrize(
    ("reference_period", "fact_period"),
    [
        ("academic_year_2023_24", "ay2023_24"),
        ("ay_2023_24", "academic-year-2023-2024"),
        ("academic_year_1999_00", "ay1999_2000"),
    ],
)
def test__given_equivalent_academic_period_labels__then_exact_match_uses_shared_parser(
    reference_period, fact_period
) -> None:
    fact = _consumer_fact_row_for_period(2023, value=15_000_000_000_000)
    fact["period"] = {"type": "academic_year", "value": fact_period}
    reference = LedgerTargetReference(
        name="academic-year total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "period_type": "academic_year",
            "geography_level": "country",
            "geography_id": "0100000US",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=reference_period,
        family="irs_soi",
        period_match_policy="exact",
    )

    registry = compile_ledger_target_references([fact], [reference], country="us")

    (spec,) = registry.specs
    assert spec.period == reference_period


@pytest.mark.parametrize(
    ("reference_period", "fact_period"),
    [
        ("academic_year_2023_24", "ay2023_25"),
        ("academic_year_2023_24", "ay2023_04"),
        ("academic_year_2023_24", "ay2023"),
        ("academic_year_2023_25", "academic_year_2023_25"),
    ],
)
@pytest.mark.parametrize(
    "resolution_route", ["selector", "ledger_fact_key", "ledger_source_record_id"]
)
def test_exact_academic_periods_do_not_discard_the_range_end(
    reference_period, fact_period, resolution_route
) -> None:
    fact = _consumer_fact_row_for_period(2023, value=15_000_000_000_000)
    fact["period"] = {"type": "academic_year", "value": fact_period}
    identifiers = (
        {}
        if resolution_route == "selector"
        else {
            resolution_route: (
                fact["aggregate_fact_key"]
                if resolution_route == "ledger_fact_key"
                else fact["lineage"]["source_record_id"]
            )
        }
    )
    reference = LedgerTargetReference(
        name="academic-year total",
        **identifiers,
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "period_type": "academic_year",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=reference_period,
        period_match_policy="exact",
    )

    with pytest.raises(ValueError, match="exact (target )?period"):
        compile_ledger_target_references([fact], [reference], country="us")


@pytest.mark.parametrize(
    "value_operation",
    ["calendar_year_average", "latest_plateau"],
)
def test__given_exact_period_with_subperiod_operation__then_reference_is_refused(
    value_operation,
) -> None:
    with pytest.raises(
        ValueError,
        match="does not support value_operation",
    ):
        LedgerTargetReference(
            name="unsupported exact aggregate",
            ledger_selector={"source_name": "official_series"},
            value_operation=value_operation,
            entity="person",
            measure="people",
            period=2025,
            period_match_policy="exact",
        )


def test__given_equivalent_exact_period_label__then_period_type_still_matches() -> None:
    fact = _consumer_fact_row_for_period(2023, value=15_000_000_000_000)
    fact["period"] = {"type": "calendar_year", "value": "tax_year_2023"}

    with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
        compile_ledger_target_references(
            [fact],
            [_exact_agi_reference(period=2023)],
            country="us",
        )


def test__given_exact_identifier__then_declared_period_type_still_matches() -> None:
    fact = _consumer_fact_row_for_period(2023, value=15_000_000_000_000)
    fact["period"]["type"] = "calendar_year"
    reference = _exact_agi_reference(
        ledger_fact_key=fact["aggregate_fact_key"],
        period="tax_year_2023",
    )

    with pytest.raises(ValueError, match="requires exact period"):
        compile_ledger_target_references([fact], [reference], country="us")


def test__given_exact_period_policy__then_stale_observation_is_refused() -> None:
    with pytest.raises(ValueError, match="exact target period 2023"):
        compile_ledger_target_references(
            [_consumer_fact_row_for_period(2022, value=14_000_000_000_000)],
            [_exact_agi_reference()],
            country="us",
        )


def test__given_exact_period_projection__then_explicit_projection_is_used() -> None:
    projection = _consumer_fact_row_for_period(2023, value=15_000_000_000_000)
    projection["assertion"] = "source_projection"
    registry = compile_ledger_target_references(
        [projection],
        [
            _exact_agi_reference(
                assertion_policy="allow_source_projection",
            )
        ],
        country="us",
    )

    (spec,) = registry.specs
    assert spec.value == 15_000_000_000_000
    assert spec.metadata["ledger_resolved_assertion"] == "source_projection"
    assert spec.metadata["ledger_assertion_policy"] == "allow_source_projection"
    assert spec.metadata["ledger_period_match_policy"] == "exact"


def test__given_geography_vintage_selector__then_only_that_vintage_matches() -> None:
    old_vintage = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:old-vintage",
        legacy_fact_key="ledger.fact.v1:old-vintage",
        value=14_000_000_000_000,
        geography={
            "level": "country",
            "id": "0100000US",
            "name": "United States",
            "vintage": "2010_census",
        },
    )
    current_vintage = _consumer_fact_row(value=15_000_000_000_000)
    reference = _exact_agi_reference(
        ledger_selector={
            **dict(_exact_agi_reference().ledger_selector),
            "geography_vintage": "2020_census",
        }
    )

    registry = compile_ledger_target_references(
        [old_vintage, current_vintage],
        [reference],
        country="us",
    )

    (spec,) = registry.specs
    assert spec.value == 15_000_000_000_000
    assert spec.metadata["ledger_selector_geography_vintage"] == "2020_census"


def test__given_exact_period_policy_without_period__then_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="requires an explicit target period"):
        _exact_agi_reference(period=None)


def test__given_period_bearing_groupby_value__then_latest_source_period_is_used() -> (
    None
):
    reference = LedgerTargetReference(
        name="latest CGT total",
        ledger_selector={
            "source_name": "hmrc",
            "source_measure_id": "total_gains",
            "source_concept": "hmrc.cgt_gains_total",
            "geography_level": "country",
            "geography_id": "K02000001",
        },
        entity="person",
        measure="capital_gains",
        period=2025,
        family="hmrc_cgt",
    )
    older = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:cgt-2022",
        legacy_fact_key="ledger.fact.v1:cgt-2022",
        value=60_000_000_000,
        source={"source_name": "hmrc"},
        observed_measure={
            "source_name": "hmrc",
            "source_measure_id": "total_gains",
            "source_concept": "hmrc.cgt_gains_total",
            "unit": "gbp",
        },
        period={"type": "tax_year", "value": 2022},
        geography={"level": "country", "id": "K02000001"},
        entity={"name": "person"},
        dimensions={},
        layout={
            "record_set_id": "hmrc.cgt_statistics_2025.table1.ty2022",
            "groupby_dimension": "hmrc.cgt_table1_line",
            "groupby_value_id": "ty2022",
            "measure_id": "total_gains",
        },
    )
    newer = _consumer_fact_row(
        **{
            **older,
            "aggregate_fact_key": "ledger.aggregate_fact.v2:cgt-2023",
            "legacy_fact_key": "ledger.fact.v1:cgt-2023",
            "value": 65_900_000_000,
            "period": {"type": "tax_year", "value": 2023},
            "layout": {
                "record_set_id": "hmrc.cgt_statistics_2025.table1.ty2023",
                "groupby_dimension": "hmrc.cgt_table1_line",
                "groupby_value_id": "ty2023",
                "measure_id": "total_gains",
            },
        }
    )

    registry = compile_ledger_target_references(
        [older, newer], [reference], country="uk"
    )

    assert registry.specs[0].value == 65_900_000_000


@pytest.mark.parametrize(
    ("layout_field", "older_value", "newer_value"),
    [
        ("record_set_id", "irs_soi.ty2022.table_1_1", "irs_soi.ty2023.table_1_2"),
        ("record_set_id", "irs_soi.ty2022.1", "irs_soi.ty2023.2"),
        ("groupby_value_id", "1", "2"),
    ],
)
def test_period_normalization_preserves_non_year_numeric_series_identifiers(
    layout_field, older_value, newer_value
) -> None:
    older = _consumer_fact_row_for_period(2022, value=14_000_000_000_000)
    newer = _consumer_fact_row_for_period(2023, value=15_000_000_000_000)
    older["layout"][layout_field] = older_value
    newer["layout"][layout_field] = newer_value
    reference = LedgerTargetReference(
        name="ambiguous source tables",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2023,
    )

    with pytest.raises(ValueError, match="multiple Ledger facts"):
        compile_ledger_target_references([older, newer], [reference], country="us")


def test__given_academic_year_record_sets__then_latest_source_period_is_used() -> None:
    """The SLC series key one record set per academic year (…ay2023, …ay2024).

    The ay-prefixed token must normalize away like ty/fy tokens do, so the
    per-year record sets collapse to one period-invariant series and the
    resolver can take the latest fact at or before the target period instead
    of refusing with multi_fact.
    """
    reference = LedgerTargetReference(
        name="latest maintenance-loan recipients",
        ledger_selector={
            "source_name": "slc",
            "source_concept": "slc.maintenance_loan_recipients",
            "geography_level": "country",
            "geography_id": "E92000001",
        },
        entity="person",
        measure="slc/maintenance_loan_recipients",
        period=2025,
        family="slc_student_support",
    )
    older = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:slc-2023",
        legacy_fact_key="ledger.fact.v1:slc-2023",
        value=1_100_000,
        source={"source_name": "slc"},
        observed_measure={
            "source_name": "slc",
            "source_measure_id": "maintenance_loan_recipients",
            "source_concept": "slc.maintenance_loan_recipients",
            "unit": "count",
        },
        period={"type": "academic_year", "value": 2023},
        geography={"level": "country", "id": "E92000001"},
        entity={"name": "person"},
        dimensions={},
        layout={
            "record_set_id": "slc.support.table_3a.recipients.ay2023",
            "groupby_dimension": "slc.support_line",
            "groupby_value_id": "maintenance_loan",
            "measure_id": "maintenance_loan_recipients",
        },
    )
    newer = _consumer_fact_row(
        **{
            **older,
            "aggregate_fact_key": "ledger.aggregate_fact.v2:slc-2024",
            "legacy_fact_key": "ledger.fact.v1:slc-2024",
            "value": 1_159_761,
            "period": {"type": "academic_year", "value": 2024},
            "layout": {
                "record_set_id": "slc.support.table_3a.recipients.ay2024",
                "groupby_dimension": "slc.support_line",
                "groupby_value_id": "maintenance_loan",
                "measure_id": "maintenance_loan_recipients",
            },
        }
    )

    registry = compile_ledger_target_references(
        [older, newer], [reference], country="uk"
    )

    assert registry.specs[0].value == 1_159_761


def test__given_year_prefixed_record_sets__then_latest_source_period_is_used() -> None:
    older = _consumer_fact_row_for_period(2024, value=95_031)
    newer = _consumer_fact_row_for_period(2025, value=85_629)
    older["layout"]["record_set_id"] = "dfe.release2026.year2024.early_learning"
    newer["layout"]["record_set_id"] = "dfe.release2026.year2025.early_learning"
    reference = LedgerTargetReference(
        name="latest DfE childcare count",
        ledger_selector={"source_name": "irs_soi"},
        entity="person",
        measure="person_count",
        period=2025,
    )

    registry = compile_ledger_target_references(
        [older, newer], [reference], country="uk"
    )

    assert registry.specs[0].value == 85_629


def test__given_one_difference_fact_per_operand__then_difference_compiles() -> None:
    minuend = _consumer_fact_row_for_period(2025, value=775_994)
    subtrahend = _consumer_fact_row_for_period(2025, value=379_029)
    minuend["dimensions"] = {"entitlement_type": "Universal"}
    subtrahend["dimensions"] = {"entitlement_type": "Working parents"}
    reference = LedgerTargetReference(
        name="universal-only childcare",
        ledger_selector={"source_name": "irs_soi"},
        value_operation="difference",
        value_operands=(
            {
                "role": "minuend",
                "dimension_values": {"entitlement_type": "Universal"},
            },
            {
                "role": "subtrahend",
                "dimension_values": {"entitlement_type": "Working parents"},
            },
        ),
        entity="person",
        measure="person_count",
        period=2025,
    )

    registry = compile_ledger_target_references(
        [minuend, subtrahend], [reference], country="uk"
    )

    assert registry.specs[0].value == 396_965
    assert registry.specs[0].metadata["ledger_value_formula"] == (
        "minuend - subtrahend"
    )
    assert json.loads(registry.specs[0].metadata["ledger_member_fact_keys"]) == [
        minuend["aggregate_fact_key"],
        subtrahend["aggregate_fact_key"],
    ]


@pytest.mark.parametrize(
    ("minuend_value", "subtrahend_value"),
    [(1.0, 2.0), (1e308, -1e308)],
)
def test__given_invalid_difference__then_compilation_refuses_it(
    minuend_value: float, subtrahend_value: float
) -> None:
    minuend = _consumer_fact_row_for_period(2025, value=minuend_value)
    subtrahend = _consumer_fact_row_for_period(2025, value=subtrahend_value)
    minuend["dimensions"] = {"role": "whole"}
    subtrahend["dimensions"] = {"role": "part"}
    reference = LedgerTargetReference(
        name="invalid difference",
        ledger_selector={"source_name": "irs_soi"},
        value_operation="difference",
        value_operands=(
            {"role": "minuend", "dimension_values": {"role": "whole"}},
            {"role": "subtrahend", "dimension_values": {"role": "part"}},
        ),
        entity="person",
        measure="person_count",
        period=2025,
    )

    with pytest.raises(ValueError, match="difference produced invalid value"):
        compile_ledger_target_references(
            [minuend, subtrahend], [reference], country="uk"
        )


def test__given_difference_operands_at_different_periods__then_compilation_fails() -> (
    None
):
    minuend = _consumer_fact_row_for_period(2025, value=10.0)
    subtrahend = _consumer_fact_row_for_period(2024, value=2.0)
    minuend["dimensions"] = {"role": "whole"}
    subtrahend["dimensions"] = {"role": "part"}
    reference = LedgerTargetReference(
        name="period-mismatched difference",
        ledger_selector={"source_name": "irs_soi"},
        value_operation="difference",
        value_operands=(
            {"role": "minuend", "dimension_values": {"role": "whole"}},
            {"role": "subtrahend", "dimension_values": {"role": "part"}},
        ),
        entity="person",
        measure="person_count",
        period=2025,
    )

    with pytest.raises(ValueError, match="same latest period"):
        compile_ledger_target_references(
            [minuend, subtrahend], [reference], country="uk"
        )


def test__given_sum_member_shortfall__then_compilation_fails_loudly() -> None:
    only_member = _consumer_fact_row_for_period(2025, value=10.0)
    reference = LedgerTargetReference(
        name="guarded sum",
        ledger_selector={"source_name": "irs_soi"},
        value_operation="sum",
        expected_member_count=2,
        entity="person",
        measure="person_count",
        period=2025,
    )

    with pytest.raises(ValueError, match="expected 2 members.*resolved 1"):
        compile_ledger_target_references([only_member], [reference], country="uk")


def test__given_identifier_bypasses_selector__then_entity_pin_still_applies() -> None:
    fact = _consumer_fact_row_for_period(2025, value=10.0)
    fact["entity"] = {"name": "government"}
    reference = LedgerTargetReference(
        name="entity-guarded reference",
        ledger_fact_key=fact["aggregate_fact_key"],
        ledger_selector={"entity_name": "person"},
        entity="person",
        measure="person_count",
        period=2025,
    )

    with pytest.raises(ValueError, match="requires entity_name 'person'"):
        compile_ledger_target_references([fact], [reference], country="uk")


def test__given_selector_matches_future_year__then_latest_eligible_period_is_used() -> (
    None
):
    # Given
    reference = LedgerTargetReference(
        name="latest eligible SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _consumer_fact_row_for_period(2023, value=15_000_000_000_000),
            _consumer_fact_row_for_period(2025, value=17_000_000_000_000),
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.value == 15_000_000_000_000
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )


def test__given_selector_matches_future_month__then_latest_eligible_period_is_used() -> (
    None
):
    # Given
    reference = LedgerTargetReference(
        name="latest eligible Medicaid enrollment",
        ledger_selector={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "person",
            "layout_groupby_value_id": "total_medicaid_chip_enrollment",
        },
        entity="person",
        measure="total_medicaid_chip_enrollment",
        period=2024,
        family="cms_medicaid",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _monthly_consumer_fact_row("2024-12", value=80_000_000),
            _monthly_consumer_fact_row("2025-12", value=82_000_000),
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.value == 80_000_000
    assert (
        spec.metadata["ledger_source_record_id"]
        == "cms_medicaid.month2024_12.us.total_medicaid_chip_enrollment"
    )


def test__given_selector_matches_multiple_eligible_months__then_latest_month_is_used() -> (
    None
):
    # Given
    reference = LedgerTargetReference(
        name="latest eligible Medicaid enrollment",
        ledger_selector={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "person",
            "layout_groupby_value_id": "total_medicaid_chip_enrollment",
        },
        entity="person",
        measure="total_medicaid_chip_enrollment",
        period=2024,
        family="cms_medicaid",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _monthly_consumer_fact_row("2024-11", value=79_000_000),
            _monthly_consumer_fact_row("2024-12", value=80_000_000),
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.value == 80_000_000
    assert (
        spec.metadata["ledger_source_record_id"]
        == "cms_medicaid.month2024_12.us.total_medicaid_chip_enrollment"
    )


class TestLedgerSelectorExtensions:
    def test_dimension_values_use_strict_typed_equality(self) -> None:
        reference = LedgerTargetReference(
            name="typed dimension selector",
            ledger_selector={"dimension_values": {"band": "5"}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
            compile_ledger_target_references(
                [_consumer_fact_row(dimensions={"band": 5})],
                [reference],
                country="us",
            )

    def test_dimension_values_accept_list_membership(self) -> None:
        reference = LedgerTargetReference(
            name="list dimension selector",
            ledger_selector={"dimension_values": {"band": ["A", "B"]}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row(dimensions={"band": "B"})],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "list dimension selector"

    def test_dimension_values_require_present_dimension(self) -> None:
        reference = LedgerTargetReference(
            name="missing dimension selector",
            ledger_selector={"dimension_values": {"band": "A"}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
            compile_ledger_target_references(
                [_consumer_fact_row(dimensions={"income_range": "all"})],
                [reference],
                country="us",
            )

    def test_empty_membership_list_raises_instead_of_matching_nothing(self) -> None:
        reference = LedgerTargetReference(
            name="empty membership selector",
            ledger_selector={"source_measure_id": []},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        with pytest.raises(ValueError, match="empty list"):
            compile_ledger_target_references(
                [_consumer_fact_row()],
                [reference],
                country="us",
            )

    def test_empty_dimension_values_pin_list_raises(self) -> None:
        reference = LedgerTargetReference(
            name="empty pin list selector",
            ledger_selector={"dimension_values": {"band": []}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        with pytest.raises(ValueError, match="empty pin"):
            compile_ledger_target_references(
                [_consumer_fact_row(dimensions={"band": "A"})],
                [reference],
                country="us",
            )

    def test_empty_dimensions_list_matches_dimensionless_fact(self) -> None:
        reference = LedgerTargetReference(
            name="dimensionless selector",
            ledger_selector={"dimensions": []},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row(dimensions={})],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "dimensionless selector"

    def test_dimensions_list_matches_exact_name_set_order_insensitively(self) -> None:
        fact = _consumer_fact_row(dimensions={"band": "A", "sex": "all"})
        reference = LedgerTargetReference(
            name="dimension-name selector",
            ledger_selector={"dimensions": ["sex", "band"]},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )
        missing_name = LedgerTargetReference(
            name="partial dimension-name selector",
            ledger_selector={"dimensions": ["band"]},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references([fact], [reference], country="us")
        with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
            compile_ledger_target_references([fact], [missing_name], country="us")

        assert registry.specs[0].name == "dimension-name selector"

    def test_chronicle_layout_aliases_match_fact_layout(self) -> None:
        reference = LedgerTargetReference(
            name="layout alias selector",
            ledger_selector={
                "record_set_id": "irs_soi.ty2023.table_1_1",
                "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            },
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row()],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "layout alias selector"

    def test_scalar_selector_accepts_list_expected_values(self) -> None:
        reference = LedgerTargetReference(
            name="list source measure selector",
            ledger_selector={
                "source_measure_id": ["total_tax", "adjusted_gross_income"],
            },
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row()],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "list source measure selector"

    def test_us_mapping_dimensions_keep_subset_semantics(self) -> None:
        reference = LedgerTargetReference(
            name="mapping dimensions selector",
            ledger_selector={"dimensions": {"income_range": "all"}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row()],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "mapping dimensions selector"

    @pytest.mark.parametrize(
        ("older_record_set_id", "newer_record_set_id", "older_period", "newer_period"),
        [
            (
                "obr.fy2024_25.tax_receipts",
                "obr.fy2025_26.tax_receipts",
                "2024-25",
                "2025-26",
            ),
            (
                "obr.ty2024_25.tax_receipts",
                "obr.ty2025_26.tax_receipts",
                "2024-25",
                "2025-26",
            ),
            (
                "obr.cy2024_25.tax_receipts",
                "obr.cy2025_26.tax_receipts",
                "2024-25",
                "2025-26",
            ),
            (
                "ons.2024_2025.population",
                "ons.2025_2026.population",
                "2024_2025",
                "2025_2026",
            ),
            ("isc.census_2023.pupils", "isc.census_2024.pupils", 2023, 2024),
            ("ons.dec2023.population", "ons.dec2024.population", "2023-12", "2024-12"),
            (
                "dfe.pse_march2025.pupils",
                "dfe.pse_march2026.pupils",
                "2025-03",
                "2026-03",
            ),
            (
                "ons.families_households_2024.table7",
                "ons.families_households_2025.table7",
                2024,
                2025,
            ),
            (
                "voa.apr2023_mar2024.stock",
                "voa.apr2024_mar2025.stock",
                "2023-04",
                "2024-04",
            ),
        ],
    )
    def test_period_tokens_collapse_across_uk_record_set_spellings(
        self,
        older_record_set_id: str,
        newer_record_set_id: str,
        older_period: int | str,
        newer_period: int | str,
    ) -> None:
        older = _consumer_fact_row(
            aggregate_fact_key="ledger.aggregate_fact.v2:older",
            legacy_fact_key="ledger.fact.v1:older",
            lineage={"source_record_id": f"{older_record_set_id}.all.population"},
            period={"type": "year", "value": older_period},
            value=1.0,
            source={"source_name": "ons"},
            observed_measure={
                "source_name": "ons",
                "source_measure_id": "population",
                "source_concept": "ons.population",
                "unit": "people",
            },
            geography={"level": "country", "id": "K02000001"},
            entity={"name": "person"},
            layout={
                "record_set_id": older_record_set_id,
                "groupby_dimension": "population",
                "groupby_value_id": "all",
                "measure_id": "population",
            },
        )
        newer = _consumer_fact_row(
            **{
                **older,
                "aggregate_fact_key": "ledger.aggregate_fact.v2:newer",
                "legacy_fact_key": "ledger.fact.v1:newer",
                "lineage": {
                    "source_record_id": f"{newer_record_set_id}.all.population"
                },
                "period": {"type": "year", "value": newer_period},
                "value": 2.0,
                "layout": {
                    "record_set_id": newer_record_set_id,
                    "groupby_dimension": "population",
                    "groupby_value_id": "all",
                    "measure_id": "population",
                },
            }
        )
        reference = LedgerTargetReference(
            name="latest UK population",
            ledger_selector={
                "source_name": "ons",
                "source_measure_id": "population",
                "geography_id": "K02000001",
                "entity_name": "person",
                "layout_groupby_value_id": "all",
            },
            entity="person",
            measure="population",
            period=2026,
            family="ons",
        )

        registry = compile_ledger_target_references(
            [older, newer], [reference], country="uk"
        )

        assert registry.specs[0].value == 2.0


def test__given_selector_matches_only_future_year__then_compilation_fails() -> None:
    # Given
    reference = LedgerTargetReference(
        name="latest eligible SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    # When / Then
    with pytest.raises(ValueError, match="at or before target period"):
        compile_ledger_target_references(
            [_consumer_fact_row_for_period(2025, value=17_000_000_000_000)],
            [reference],
            country="us",
        )


def test__given_selector_matches_only_unparseable_period__then_compilation_fails() -> (
    None
):
    reference = LedgerTargetReference(
        name="unparseable SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    with pytest.raises(ValueError, match="at or before target period"):
        compile_ledger_target_references(
            [
                _consumer_fact_row(
                    aggregate_fact_key="ledger.aggregate_fact.v2:unknown-period",
                    period={"type": "source", "value": "current"},
                )
            ],
            [reference],
            country="us",
        )


def test__given_aggregation_method_selector__then_compilation_fails() -> None:
    # Given
    reference = LedgerTargetReference(
        name="no aggregation selector",
        ledger_selector={"aggregation_method": "sum"},
        entity="tax_unit",
        measure="adjusted_gross_income",
    )

    # When / Then
    with pytest.raises(
        ValueError,
        match="Unsupported Ledger fact selector field 'aggregation_method'",
    ):
        compile_ledger_target_references(
            [_consumer_fact_row()],
            [reference],
            country="us",
        )


@pytest.mark.parametrize("measure", [None, ""])
def test__given_non_count_reference_without_measure__then_compilation_fails(
    measure,
) -> None:
    # Given
    reference = LedgerTargetReference(
        name="missing measure target",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        entity="tax_unit",
        measure=measure,
    )

    # When / Then
    with pytest.raises(ValueError, match="measure is required"):
        compile_ledger_target_references(
            [_consumer_fact_row()], [reference], country="us"
        )


def test__given_reference_identifiers_match_different_facts__then_compilation_fails() -> (
    None
):
    # Given
    first = _consumer_fact_row()
    second = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:def456",
        legacy_fact_key="ledger.fact.v1:def456",
        lineage={"source_record_id": "irs_soi.ty2023.table_1_1.all.total_tax"},
    )
    reference = LedgerTargetReference(
        name="inconsistent reference",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        ledger_source_record_id="irs_soi.ty2023.table_1_1.all.total_tax",
        entity="tax_unit",
        measure="adjusted_gross_income",
    )

    # When / Then
    with pytest.raises(ValueError, match="resolve to different Ledger facts"):
        compile_ledger_target_references([first, second], [reference], country="us")


def test__given_sum_reference_matches_same_vintage_facts__then_values_sum() -> None:
    first = _consumer_fact_row_for_period(2023, value=100.0)
    first["aggregate_fact_key"] = "ledger.aggregate_fact.v2:first"
    first["layout"]["groupby_value_id"] = "first"
    first["dimensions"] = {"band": "first"}
    second = _consumer_fact_row_for_period(2023, value=250.0)
    second["aggregate_fact_key"] = "ledger.aggregate_fact.v2:second"
    second["layout"]["groupby_value_id"] = "second"
    second["dimensions"] = {"band": "second"}
    reference = LedgerTargetReference(
        name="summed bands",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
        },
        value_operation="sum",
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    registry = compile_ledger_target_references(
        [first, second], [reference], country="us"
    )

    spec = registry.specs[0]
    assert spec.value == 350.0
    assert spec.metadata["ledger_member_fact_count"] == "2"
    assert json.loads(spec.metadata["ledger_member_fact_keys"]) == [
        "ledger.aggregate_fact.v2:first",
        "ledger.aggregate_fact.v2:second",
    ]
    assert spec.metadata["ledger_fact_key"] == "ledger.aggregate_fact.v2:second"
    assert spec.metadata["ledger_fact_period"] == "2023"
    assert spec.metadata["ledger_value_operation"] == "sum"


def test__given_sum_reference_matches_two_vintages__then_latest_partition_wins() -> (
    None
):
    old_first = _consumer_fact_row_for_period(2022, value=10.0)
    old_first["aggregate_fact_key"] = "ledger.aggregate_fact.v2:old-first"
    old_first["layout"]["groupby_value_id"] = "first"
    old_first["dimensions"] = {"band": "first"}
    old_second = _consumer_fact_row_for_period(2022, value=20.0)
    old_second["aggregate_fact_key"] = "ledger.aggregate_fact.v2:old-second"
    old_second["layout"]["groupby_value_id"] = "second"
    old_second["dimensions"] = {"band": "second"}
    new_first = _consumer_fact_row_for_period(2023, value=100.0)
    new_first["aggregate_fact_key"] = "ledger.aggregate_fact.v2:new-first"
    new_first["layout"]["groupby_value_id"] = "first"
    new_first["dimensions"] = {"band": "first"}
    new_second = _consumer_fact_row_for_period(2023, value=200.0)
    new_second["aggregate_fact_key"] = "ledger.aggregate_fact.v2:new-second"
    new_second["layout"]["groupby_value_id"] = "second"
    new_second["dimensions"] = {"band": "second"}
    reference = LedgerTargetReference(
        name="latest summed bands",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
        },
        value_operation="sum",
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    registry = compile_ledger_target_references(
        [old_first, old_second, new_first, new_second],
        [reference],
        country="us",
    )

    spec = registry.specs[0]
    assert spec.value == 300.0
    assert spec.metadata["ledger_member_fact_count"] == "2"
    assert "old-first" not in spec.metadata["ledger_member_fact_keys"]
    assert "new-first" in spec.metadata["ledger_member_fact_keys"]


def test__given_selector_uses_record_set_spec_id__then_layout_spec_id_matches() -> None:
    row = _consumer_fact_row_for_period(2023, value=100.0)
    row["layout"]["record_set_spec_id"] = "irs_soi.table_1_1.v1"
    other = _consumer_fact_row_for_period(2023, value=250.0)
    other["aggregate_fact_key"] = "ledger.aggregate_fact.v2:other-spec"
    other["layout"]["record_set_spec_id"] = "irs_soi.table_1_2.v1"
    reference = LedgerTargetReference(
        name="spec selected",
        ledger_selector={
            "source_name": "irs_soi",
            "record_set_spec_id": "irs_soi.table_1_1.v1",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    registry = compile_ledger_target_references([other, row], [reference], country="us")

    spec = registry.specs[0]
    assert spec.value == 100.0
    assert spec.metadata["ledger_layout_record_set_spec_id"] == "irs_soi.table_1_1.v1"
    assert spec.metadata["ledger_selector_record_set_spec_id"] == (
        "irs_soi.table_1_1.v1"
    )


def test__given_count_x_mean_reference__then_amount_multiplies_ordered_members() -> (
    None
):
    count = _spi_area_fact(
        "employment_income_count",
        value=25.0,
        aggregation="sum",
        fact_key="ledger.aggregate_fact.v2:employment-count",
    )
    mean = _spi_area_fact(
        "employment_income_mean",
        value=40_000.0,
        aggregation="mean",
        fact_key="ledger.aggregate_fact.v2:employment-mean",
    )
    reference = LedgerTargetReference(
        name="employment amount",
        ledger_selector={
            "source_name": "hmrc",
            "source_measure_id": [
                "employment_income_count",
                "employment_income_mean",
            ],
            "record_set_spec_id": "uk.local_geography.spi_income.by_constituency.v1",
            "geography_level": "constituency",
            "geography_id": "E14000001",
            "entity_name": "person",
        },
        value_operation="count_x_mean",
        entity="person",
        measure="hmrc/employment_income/amount",
        period=2025,
        family="hmrc",
    )

    registry = compile_ledger_target_references(
        [mean, count], [reference], country="uk"
    )

    spec = registry.specs[0]
    assert spec.value == 1_000_000.0
    assert spec.metadata["ledger_value_operation"] == "count_x_mean"
    assert spec.metadata["ledger_member_fact_count"] == "2"
    assert json.loads(spec.metadata["ledger_member_fact_keys"]) == [
        "ledger.aggregate_fact.v2:employment-count",
        "ledger.aggregate_fact.v2:employment-mean",
    ]
    assert (
        spec.metadata["ledger_fact_key"] == "ledger.aggregate_fact.v2:employment-mean"
    )


def test__given_count_x_mean_reference_without_pair__then_refuses() -> None:
    count = _spi_area_fact(
        "employment_income_count",
        value=25.0,
        aggregation="sum",
        fact_key="ledger.aggregate_fact.v2:employment-count",
    )
    reference = LedgerTargetReference(
        name="employment amount",
        ledger_selector={
            "source_name": "hmrc",
            "source_measure_id": [
                "employment_income_count",
                "employment_income_mean",
            ],
            "record_set_spec_id": "uk.local_geography.spi_income.by_constituency.v1",
            "geography_level": "constituency",
            "geography_id": "E14000001",
            "entity_name": "person",
        },
        value_operation="count_x_mean",
        entity="person",
        measure="hmrc/employment_income/amount",
        period=2025,
        family="hmrc",
    )

    with pytest.raises(ValueError, match="exactly one count fact and one mean fact"):
        compile_ledger_target_references([count], [reference], country="uk")


def test__given_calendar_year_average_reference__then_monthly_mean_compiles() -> None:
    rows = [
        _monthly_consumer_fact_row("2025-04", value=6_380_000.0),
        *(
            _monthly_consumer_fact_row(f"2025-{month:02d}", value=6_600_000.0)
            for month in range(5, 9)
        ),
        *(
            _monthly_consumer_fact_row(f"2025-{month:02d}", value=6_960_000.0)
            for month in range(9, 12)
        ),
        _monthly_consumer_fact_row("2025-12", value=7_170_000.0),
    ]
    reference = LedgerTargetReference(
        name="calendar average enrollment",
        ledger_selector={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "person",
        },
        value_operation="calendar_year_average",
        entity="person",
        measure="total_medicaid_chip_enrollment",
        period=2025,
        family="cms_medicaid",
    )

    registry = compile_ledger_target_references(rows, [reference], country="us")

    spec = registry.specs[0]
    assert spec.value == pytest.approx(
        (6_380_000.0 + 4 * 6_600_000.0 + 3 * 6_960_000.0 + 7_170_000.0) / 9
    )
    assert spec.metadata["ledger_member_fact_count"] == "9"
    assert json.loads(spec.metadata["ledger_member_fact_keys"])[-1] == (
        "ledger.aggregate_fact.v2:medicaid-2025_12"
    )
    assert spec.metadata["ledger_fact_key"] == (
        "ledger.aggregate_fact.v2:medicaid-2025_12"
    )
    assert spec.metadata["ledger_fact_period"] == "2025-12"
    assert spec.metadata["ledger_value_operation"] == "calendar_year_average"


def test__given_latest_plateau_reference__then_latest_equal_month_run_wins() -> None:
    rows = [
        _monthly_consumer_fact_row("2025-04", value=6_380_000.0),
        *(
            _monthly_consumer_fact_row(f"2025-{month:02d}", value=6_600_000.0)
            for month in range(5, 9)
        ),
        *(
            _monthly_consumer_fact_row(f"2025-{month:02d}", value=6_960_000.0)
            for month in range(9, 12)
        ),
        _monthly_consumer_fact_row("2025-12", value=7_170_000.0),
        _monthly_consumer_fact_row("2026-01", value=7_170_000.0),
        _monthly_consumer_fact_row("2026-02", value=7_170_000.0),
    ]
    reference = LedgerTargetReference(
        name="latest plateau enrollment",
        ledger_selector={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "person",
        },
        value_operation="latest_plateau",
        entity="person",
        measure="total_medicaid_chip_enrollment",
        period=2026,
        family="cms_medicaid",
    )

    registry = compile_ledger_target_references(rows, [reference], country="us")

    spec = registry.specs[0]
    assert spec.value == 7_170_000.0
    assert spec.metadata["ledger_member_fact_count"] == "3"
    assert json.loads(spec.metadata["ledger_member_fact_keys"]) == [
        "ledger.aggregate_fact.v2:medicaid-2025_12",
        "ledger.aggregate_fact.v2:medicaid-2026_01",
        "ledger.aggregate_fact.v2:medicaid-2026_02",
    ]
    assert spec.metadata["ledger_fact_key"] == (
        "ledger.aggregate_fact.v2:medicaid-2026_02"
    )
    assert spec.metadata["ledger_fact_period"] == "2026-02"
    assert spec.metadata["ledger_value_operation"] == "latest_plateau"


def test__given_month_operation_matches_non_month_fact__then_refuses_mixing() -> None:
    monthly = _monthly_consumer_fact_row("2025-04", value=6_380_000.0)
    annual = _monthly_consumer_fact_row("2025-05", value=6_600_000.0)
    annual["period"] = {"type": "tax_year", "value": 2025}
    reference = LedgerTargetReference(
        name="invalid calendar average enrollment",
        ledger_selector={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "person",
        },
        value_operation="calendar_year_average",
        entity="person",
        measure="total_medicaid_chip_enrollment",
        period=2025,
        family="cms_medicaid",
    )

    with pytest.raises(ValueError, match="requires monthly Ledger facts"):
        compile_ledger_target_references([monthly, annual], [reference], country="us")


def test__given_identity_reference_matches_multiple_facts__then_still_fails() -> None:
    first = _consumer_fact_row_for_period(2023, value=100.0)
    first["aggregate_fact_key"] = "ledger.aggregate_fact.v2:first"
    first["layout"]["groupby_value_id"] = "first"
    second = _consumer_fact_row_for_period(2023, value=250.0)
    second["aggregate_fact_key"] = "ledger.aggregate_fact.v2:second"
    second["layout"]["groupby_value_id"] = "second"
    reference = LedgerTargetReference(
        name="ambiguous bands",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    with pytest.raises(ValueError, match="matched multiple Ledger facts"):
        compile_ledger_target_references([first, second], [reference], country="us")


def test__given_lineage_only_metadata_diff__then_parity_report_names_full_digest() -> (
    None
):
    # Given
    expected = TargetRegistry(
        [
            TargetSpec(
                name="agi",
                entity="tax_unit",
                measure="adjusted_gross_income",
                value=100.0,
                source="source",
            )
        ],
        country="us",
    )
    actual = TargetRegistry(
        [
            TargetSpec(
                name="agi",
                entity="tax_unit",
                measure="adjusted_gross_income",
                value=100.0,
                source="source",
                metadata={"ledger_fact_key": "fact"},
            )
        ],
        country="us",
    )

    # When
    report = ledger_target_registry_parity_report(expected, actual)

    # Then
    assert not report.passed
    assert any(
        "full target registry digest differs" in line for line in report.failures
    )
    assert (
        report.details["expected_calibration_digest"]
        == report.details["actual_calibration_digest"]
    )
    assert asdict(expected.specs[0]) != asdict(actual.specs[0])


def test__given_domain_scoped_fact_without_filter_mapping__then_it_is_unsupported() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        }
    )

    # When
    selection = select_ledger_targets([_ledger_fact()], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_model_filter_mapping"


def test__given_domain_scoped_fact_with_domain_filter__then_target_uses_filter() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact()], mapping)

    # Then
    assert not selection.unsupported
    assert selection.specs[0].filter == "is_tax_return"


def test__given_scoped_fact_without_filter_mapping__then_it_is_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        }
    )
    fact = _ledger_fact(
        source_record_id="irs_soi.ty2023.table_1_1.under_1.adjusted_gross_income",
        filters={
            "income_range": "under_1",
            "filing_status": "all",
            "agi_upper_usd": 1,
        },
        constraints=[
            {
                "variable": "us:statutes/26/62#adjusted_gross_income",
                "operator": "<",
                "value": 1,
                "unit": "usd",
                "role": "filter",
            }
        ],
    )

    # When
    selection = select_ledger_targets([fact], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_model_filter_mapping"


def test__given_scoped_fact_with_only_domain_filter__then_it_is_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )
    fact = _ledger_fact(
        source_record_id="irs_soi.ty2023.table_1_1.under_1.adjusted_gross_income",
        filters={
            "income_range": "under_1",
            "filing_status": "all",
            "agi_upper_usd": 1,
        },
    )

    # When
    selection = select_ledger_targets([fact], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_model_filter_mapping"


def test__given_scoped_fact_with_filter_mapping__then_target_uses_filter() -> None:
    # Given
    source_record_id = "irs_soi.ty2023.table_1_1.under_1.adjusted_gross_income"
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_source_record_id={source_record_id: "agi_under_1"},
    )
    fact = _ledger_fact(
        source_record_id=source_record_id,
        domain="",
        filters={"income_range": "under_1", "filing_status": "all"},
    )

    # When
    selection = select_ledger_targets([fact], mapping)

    # Then
    assert not selection.unsupported
    assert selection.specs[0].filter == "agi_under_1"


def test__given_unmapped_ledger_fact__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping()

    # When
    selection = select_ledger_targets([_ledger_fact()], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_model_measure_mapping"
    assert (
        selection.unsupported[0].source_record_id
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )


def test__given_rate_fact__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )
    fact = _ledger_fact(aggregation={"method": "rate"})

    # When
    selection = select_ledger_targets([fact], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "unsupported_aggregation:rate"


def test__given_count_fact__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets(
        [_ledger_fact(aggregation={"method": "count"}, value=10)],
        mapping,
    )

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "unsupported_aggregation:count"


def test__given_hierarchy_profile__then_children_scale_to_parent() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01",
                value=30.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_02",
                value=70.0,
                geography_level="congressional_district",
                geography_id="5001900US0102",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When
    reconciled = apply_ledger_target_profile(
        registry,
        _hierarchy_profile(),
        context={"apply_hierarchy": True},
    )

    # Then
    first, second, parent = reconciled.specs
    assert first.value == pytest.approx(60.0)
    assert second.value == pytest.approx(140.0)
    assert parent.value == pytest.approx(200.0)
    assert first.metadata["hierarchy_reconciliation_rule"] == "test_cd_to_state"
    assert first.metadata["hierarchy_raw_value"] == "30"
    assert first.metadata["hierarchy_child_sum_raw"] == "100"
    assert first.metadata["hierarchy_parent_value"] == "200"
    assert first.metadata["hierarchy_coverage_ratio"] == "0.5"
    assert first.metadata["hierarchy_reconciliation_factor"] == "2"
    assert first.metadata["hierarchy_parent_geography_id"] == "0400000US01"
    assert first.metadata["hierarchy_parent_key"] == "01"
    assert first.metadata["hierarchy_expected_child_count"] == "2"
    assert first.metadata["hierarchy_observed_child_count"] == "2"
    assert first.metadata["hierarchy_child_ids"] == "5001900US0101,5001900US0102"


def test__given_disabled_hierarchy_profile__then_children_are_unchanged() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01",
                value=30.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When
    reconciled = apply_ledger_target_profile(
        registry,
        _hierarchy_profile(),
        context={"apply_hierarchy": False},
    )

    # Then
    assert reconciled.specs[0].value == pytest.approx(30.0)
    assert "hierarchy_reconciliation_rule" not in reconciled.specs[0].metadata


def test__given_schema2_profile_without_hierarchy__then_registry_is_unchanged() -> None:
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "be_population",
                value=11_500_000.0,
                geography_level="country",
                geography_id="BE",
            )
        ],
        country="be",
    )

    result = apply_ledger_target_profile(
        registry,
        {"schema_version": 2, "hierarchy_reconciliations": []},
    )

    assert result is registry


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test__given_non_integer_target_profile_schema__then_profile_is_refused(
    schema_version,
) -> None:
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "be_population",
                value=11_500_000.0,
                geography_level="country",
                geography_id="BE",
            )
        ],
        country="be",
    )

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        apply_ledger_target_profile(
            registry,
            {"schema_version": schema_version, "hierarchy_reconciliations": []},
        )


def test__given_nonzero_parent_and_zero_children__then_hierarchy_fails() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01",
                value=0.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_02",
                value=0.0,
                geography_level="congressional_district",
                geography_id="5001900US0102",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When / Then
    with pytest.raises(ValueError, match="zero-valued children to nonzero parent"):
        apply_ledger_target_profile(
            registry,
            _hierarchy_profile(),
            context={"apply_hierarchy": True},
        )


def test__given_incomplete_hierarchy_children__then_reconciliation_fails() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01",
                value=30.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When / Then
    with pytest.raises(ValueError, match="expected 2 child target"):
        apply_ledger_target_profile(
            registry,
            _hierarchy_profile(),
            context={"apply_hierarchy": True},
        )


def test__given_duplicate_hierarchy_child_ids__then_reconciliation_fails() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01a",
                value=30.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_01b",
                value=70.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When / Then
    with pytest.raises(ValueError, match="duplicate child ids"):
        apply_ledger_target_profile(
            registry,
            _hierarchy_profile(),
            context={"apply_hierarchy": True},
        )


def test__given_malformed_value__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact(value="not numeric")], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "invalid_value"


def test__given_overflow_value__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact(value="1e10000")], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "invalid_value"


def test__given_negative_value_without_signed_mapping__then_it_is_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact(value=-1)], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_signed_target_mapping"


def test__given_negative_value_with_signed_mapping__then_target_is_signed() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
        signed_by_concept=frozenset({"us:statutes/26/62#adjusted_gross_income"}),
    )

    # When
    selection = select_ledger_targets([_ledger_fact(value=-1)], mapping)

    # Then
    assert not selection.unsupported
    assert selection.specs[0].signed is True


def test__given_ledger_target_metadata__then_coverage_gate_uses_structured_fields() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )
    selection = select_ledger_targets([_ledger_fact()], mapping)
    requirement = TargetCoverageRequirement(
        requirement_id="soi_agi_total",
        label="SOI AGI total",
        accepted_measures=("adjusted_gross_income",),
        required_metadata=(
            ("ledger_source", "policyengine-ledger-data"),
            ("ledger_measure_concept", "us:statutes/26/62#adjusted_gross_income"),
            ("ledger_geography_level", "country"),
            ("ledger_filter_income_range", "all"),
        ),
    )

    # When
    result = target_profile_coverage_gate(selection.specs, [requirement])

    # Then
    assert result.passed


def test__given_current_soi_like_row__then_ledger_adapter_matches_current_target_shape() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        family_by_source_name={"irs_soi": "irs_soi"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    spec = select_ledger_targets([_ledger_fact()], mapping).specs[0]

    # Then
    assert spec.entity == "tax_unit"
    assert spec.measure == "adjusted_gross_income"
    assert spec.value == 15_286_017_359_000
    assert spec.period == 2023
    assert spec.family == "irs_soi"
    assert spec.signed is False


def test_ledger_metadata_records_assertion_and_fact_period():
    from microcosm.build.ledger_targets import (
        LedgerTargetMapping,
        target_spec_from_ledger_fact,
    )

    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_filer"},
    )
    # Legacy rows that omit the assertion field are not stamped in metadata
    # (absence means observation-by-default to readers, matching the loader).
    legacy = target_spec_from_ledger_fact(_consumer_fact_row(), mapping)
    assert "ledger_assertion" not in legacy.metadata
    assert legacy.metadata["ledger_fact_period"] == "2023"

    observation = target_spec_from_ledger_fact(
        _consumer_fact_row(assertion="observation"),
        mapping,
    )
    assert observation.metadata["ledger_assertion"] == "observation"

    projection = target_spec_from_ledger_fact(
        _consumer_fact_row(assertion="source_projection"),
        mapping,
    )
    assert projection.metadata["ledger_assertion"] == "source_projection"


def test_ledger_reference_selector_matches_assertion_key():
    reference = LedgerTargetReference(
        name="cbo_projected_agi",
        ledger_selector={
            "source_measure_id": "adjusted_gross_income",
            "assertion": "source_projection",
        },
        entity="household",
        measure="adjusted_gross_income",
        family="cbo",
        assertion_policy="allow_source_projection",
    )

    registry = compile_ledger_target_references(
        [_consumer_fact_row(assertion="source_projection")],
        [reference],
        country="us",
    )

    (spec,) = registry.specs
    assert spec.metadata["ledger_resolved_assertion"] == "source_projection"


def test_ledger_reference_selector_treats_absent_assertion_as_observation():
    reference = LedgerTargetReference(
        name="observed_agi",
        ledger_selector={
            "source_measure_id": "adjusted_gross_income",
            "assertion": "observation",
        },
        entity="household",
        measure="adjusted_gross_income",
        family="irs_soi",
        period=2024,
    )

    registry = compile_ledger_target_references(
        [_consumer_fact_row(value=15_000_000_000_000)],
        [reference],
        country="us",
    )

    (spec,) = registry.specs
    assert spec.value == 15_000_000_000_000
    assert "ledger_assertion" not in spec.metadata
    assert spec.metadata["ledger_resolved_assertion"] == "observation"


def test_ledger_reference_projection_fact_excluded_by_default():
    reference = LedgerTargetReference(
        name="cbo_projected_agi",
        ledger_selector={"source_measure_id": "adjusted_gross_income"},
        entity="household",
        measure="adjusted_gross_income",
        family="cbo",
        period=2024,
    )

    with pytest.raises(ValueError, match="at or before target period"):
        compile_ledger_target_references(
            [
                _consumer_fact_row(
                    aggregate_fact_key="ledger.aggregate_fact.v2:proj123",
                    legacy_fact_key="ledger.fact.v1:proj123",
                    assertion="source_projection",
                    value=16_000_000_000_000,
                )
            ],
            [reference],
            country="us",
        )


@pytest.mark.parametrize(
    "identifier_field",
    ["ledger_fact_key", "ledger_source_record_id"],
)
def test_ledger_reference_identifier_cannot_bypass_observed_only_policy(
    identifier_field,
) -> None:
    fact = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:projected-identifier",
        assertion="source_projection",
    )
    identifier = (
        fact["aggregate_fact_key"]
        if identifier_field == "ledger_fact_key"
        else fact["lineage"]["source_record_id"]
    )
    reference = LedgerTargetReference(
        name="identifier-bound observed target",
        **{identifier_field: identifier},
        entity="household",
        measure="adjusted_gross_income",
        family="irs_soi",
        period=2023,
        period_match_policy="exact",
    )

    with pytest.raises(
        ValueError,
        match="assertion_policy='observed_only'.*source_projection",
    ):
        compile_ledger_target_references([fact], [reference], country="us")


@pytest.mark.parametrize(
    "identifier_field", ["ledger_fact_key", "ledger_source_record_id"]
)
@pytest.mark.parametrize("fact_period", [2022, 2023, 2024])
def test_ledger_reference_identifier_enforces_latest_not_after_policy(
    identifier_field, fact_period
) -> None:
    fact = _consumer_fact_row_for_period(fact_period, value=15_000_000_000_000)
    identifier = (
        fact["aggregate_fact_key"]
        if identifier_field == "ledger_fact_key"
        else fact["lineage"]["source_record_id"]
    )
    reference = LedgerTargetReference(
        name="identifier-bound latest target",
        **{identifier_field: identifier},
        entity="household",
        measure="adjusted_gross_income",
        family="irs_soi",
        period=2023,
        period_match_policy="latest_not_after",
    )

    if fact_period > 2023:
        with pytest.raises(ValueError, match="at or before target period"):
            compile_ledger_target_references([fact], [reference], country="us")
    else:
        registry = compile_ledger_target_references([fact], [reference], country="us")
        assert registry.specs[0].value == 15_000_000_000_000


def test_ledger_reference_projection_fact_allowed_by_policy():
    reference = LedgerTargetReference(
        name="cbo_projected_agi",
        ledger_selector={"source_measure_id": "adjusted_gross_income"},
        entity="household",
        measure="adjusted_gross_income",
        family="cbo",
        period=2024,
        assertion_policy="allow_source_projection",
    )

    registry = compile_ledger_target_references(
        [
            _consumer_fact_row(
                aggregate_fact_key="ledger.aggregate_fact.v2:proj123",
                legacy_fact_key="ledger.fact.v1:proj123",
                assertion="source_projection",
                value=16_000_000_000_000,
            )
        ],
        [reference],
        country="us",
    )

    (spec,) = registry.specs
    assert spec.value == 16_000_000_000_000
    assert spec.metadata["ledger_assertion"] == "source_projection"
    assert spec.metadata["ledger_resolved_assertion"] == "source_projection"
    assert spec.metadata["ledger_assertion_policy"] == "allow_source_projection"


def test_ledger_reference_absent_assertion_fact_is_observation_by_default():
    reference = LedgerTargetReference(
        name="observed_agi",
        ledger_selector={"source_measure_id": "adjusted_gross_income"},
        entity="household",
        measure="adjusted_gross_income",
        family="irs_soi",
        period=2024,
    )

    registry = compile_ledger_target_references(
        [_consumer_fact_row(value=15_000_000_000_000)],
        [reference],
        country="us",
    )

    (spec,) = registry.specs
    assert spec.value == 15_000_000_000_000
    assert "ledger_assertion" not in spec.metadata
    assert spec.metadata["ledger_resolved_assertion"] == "observation"
    assert spec.metadata["ledger_assertion_policy"] == "observed_only"


def test_ledger_reference_latest_selection_uses_policy_eligible_facts():
    observed = _consumer_fact_row_for_period(2023, value=15_000_000_000_000)
    projection = _consumer_fact_row_for_period(2024, value=16_000_000_000_000)
    projection["assertion"] = "source_projection"
    reference = LedgerTargetReference(
        name="latest observed agi",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    registry = compile_ledger_target_references(
        [observed, projection], [reference], country="us"
    )

    assert registry.specs[0].value == 15_000_000_000_000


def _target_spec_via_contract_route(fact, reference, resolution_route):
    if resolution_route == "direct_helper":
        return target_spec_from_ledger_reference(fact, reference)
    if resolution_route == "ledger_fact_key":
        reference = replace(reference, ledger_fact_key=fact["aggregate_fact_key"])
    elif resolution_route == "ledger_source_record_id":
        reference = replace(
            reference,
            ledger_source_record_id=fact["lineage"]["source_record_id"],
        )
    else:
        assert resolution_route == "selector"
    (spec,) = compile_ledger_target_references([fact], [reference], country="us").specs
    return spec


@pytest.mark.parametrize(
    "resolution_route",
    ["selector", "ledger_fact_key", "ledger_source_record_id", "direct_helper"],
)
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
def test_exact_period_contract_refuses_equal_malformed_untyped_labels(
    resolution_route, period_type, invalid_period
) -> None:
    fact = _consumer_fact_row(period={"type": period_type, "value": invalid_period})
    reference = _exact_agi_reference(
        ledger_selector={"period_type": period_type},
        period=invalid_period,
    )

    with pytest.raises(ValueError, match="exact (target )?period"):
        _target_spec_via_contract_route(fact, reference, resolution_route)


@pytest.mark.parametrize(
    "resolution_route",
    ["selector", "ledger_fact_key", "ledger_source_record_id", "direct_helper"],
)
@pytest.mark.parametrize(
    ("period_type", "reference_period", "fact_period"),
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
def test_exact_period_contract_preserves_valid_aliases_and_opaque_labels(
    resolution_route, period_type, reference_period, fact_period
) -> None:
    fact = _consumer_fact_row(period={"type": period_type, "value": fact_period})
    reference = _exact_agi_reference(
        ledger_selector={"period_type": period_type},
        period=reference_period,
    )

    spec = _target_spec_via_contract_route(fact, reference, resolution_route)

    assert spec.value == fact["value"]
    assert spec.period == reference_period


@pytest.mark.parametrize(
    "resolution_route",
    ["selector", "ledger_fact_key", "ledger_source_record_id", "direct_helper"],
)
@pytest.mark.parametrize("period_match_policy", ["exact", "latest_not_after"])
@pytest.mark.parametrize("vintage_pin", ["2010_census", ["2010_census", "2000_census"]])
@pytest.mark.parametrize("fact_vintage", ["2010_census", "2020_census", None, ""])
def test_geography_vintage_contract_applies_to_every_resolution_route(
    resolution_route, period_match_policy, vintage_pin, fact_vintage
) -> None:
    fact = _consumer_fact_row()
    if fact_vintage is None:
        fact["geography"].pop("vintage")
    else:
        fact["geography"]["vintage"] = fact_vintage
    reference = _exact_agi_reference(
        ledger_selector={"geography_vintage": vintage_pin},
        period_match_policy=period_match_policy,
    )

    if fact_vintage != "2010_census":
        with pytest.raises(ValueError, match="(vintage|fact selector)"):
            _target_spec_via_contract_route(fact, reference, resolution_route)
    else:
        spec = _target_spec_via_contract_route(fact, reference, resolution_route)
        assert spec.metadata["ledger_geography_vintage"] == "2010_census"
        assert spec.value == fact["value"]


@pytest.mark.parametrize(
    "resolution_route",
    ["selector", "ledger_fact_key", "ledger_source_record_id", "direct_helper"],
)
def test_geography_vintage_contract_does_not_add_an_undeclared_pin(
    resolution_route,
) -> None:
    fact = _consumer_fact_row()
    fact["geography"].pop("vintage")

    spec = _target_spec_via_contract_route(
        fact, _exact_agi_reference(), resolution_route
    )

    assert spec.value == fact["value"]


@pytest.mark.parametrize(
    "invalid_period", ["2023_25", "2023_2025", "2023_00", "2023_13"]
)
def test_period_contract_comparator_rejects_malformed_numeric_literal_equality(
    invalid_period,
) -> None:
    assert not period_values_semantically_equal(invalid_period, invalid_period)


def test_exact_period_contract_sum_keeps_equivalent_untyped_annual_range_cells() -> (
    None
):
    first = _consumer_fact_row_for_period(2003, value=10.0)
    first["period"] = {"type": "academic_year", "value": "2003_04"}
    second = _consumer_fact_row_for_period(2003, value=20.0)
    second["aggregate_fact_key"] += "-second"
    second["legacy_fact_key"] += "-second"
    second["semantic_fact_key"] += "-second"
    second["lineage"]["source_record_id"] += ".second"
    second["period"] = {"type": "academic_year", "value": "ay2003_04"}
    reference = _exact_agi_reference(
        ledger_selector={"period_type": "academic_year"},
        period="academic_year_2003_2004",
        value_operation="sum",
    )

    (spec,) = compile_ledger_target_references(
        [first, second], [reference], country="us"
    ).specs

    assert spec.value == 30.0  # Both synthetic cells belong to the same academic year.


def test__given_mixed_epoch_fact_feed__then_both_eras_compile_to_targets() -> None:
    """Ledger-era and chronicle-era rows calibrate side by side.

    During Chronicle's rename cutover a feed carries history under
    ``ledger.*`` domains beside newly emitted rows under ``chronicle.*``
    (PolicyEngine/chronicle#143). Keys are opaque to this compiler, so both
    must select — and each target's name and metadata must carry its own
    row's key verbatim rather than being normalised onto one epoch.
    """
    # Given
    ledger_era = _consumer_fact_row()
    chronicle_era = _consumer_fact_row(
        aggregate_fact_key="chronicle.aggregate_fact.v3:def456",
        semantic_fact_key="chronicle.semantic_fact.v3:def456",
        lineage={
            "source_record_id": "irs_soi.ty2024.table_1_1.all.adjusted_gross_income",
            "source_cell_keys": ["chronicle.source_cell.v2:cell"],
            "source_row_keys": [],
        },
    )
    chronicle_era.pop("legacy_fact_key", None)
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([ledger_era, chronicle_era], mapping)

    # Then
    assert not selection.unsupported
    assert [spec.name for spec in selection.specs] == [
        "ledger.aggregate_fact.v2:abc123",
        "chronicle.aggregate_fact.v3:def456",
    ]
    chronicle_spec = selection.specs[1]
    assert (
        chronicle_spec.metadata["ledger_fact_key"]
        == "chronicle.aggregate_fact.v3:def456"
    )
    # The diagnostic field names stay ledger-era: they are frozen at v1
    # (microcosm#639) and name a slot, not an epoch.
    assert (
        chronicle_spec.metadata["ledger_aggregate_fact_key"]
        == "chronicle.aggregate_fact.v3:def456"
    )
    assert (
        chronicle_spec.metadata["ledger_semantic_fact_key"]
        == "chronicle.semantic_fact.v3:def456"
    )


def test__given_chronicle_era_reference_pin__then_it_resolves_against_the_feed() -> (
    None
):
    """A reference pinned to a chronicle-era key resolves without a code change."""
    # Given
    reference = LedgerTargetReference(
        name="nation/irs/adjusted gross income/total",
        ledger_fact_key="chronicle.aggregate_fact.v3:def456",
        entity="tax_unit",
        measure="adjusted_gross_income",
        filter="is_tax_return",
        period=2024,
        source="IRS SOI Table 1.1",
        family="irs_soi",
    )
    fact = _consumer_fact_row(
        aggregate_fact_key="chronicle.aggregate_fact.v3:def456",
        semantic_fact_key="chronicle.semantic_fact.v3:def456",
    )
    fact.pop("legacy_fact_key", None)

    # When
    registry = compile_ledger_target_references([fact], [reference], country="us")

    # Then
    assert [spec.name for spec in registry.specs] == [
        "nation/irs/adjusted gross income/total"
    ]
    assert (
        registry.specs[0].metadata["ledger_fact_key"]
        == "chronicle.aggregate_fact.v3:def456"
    )


# Equal missing identities must not count as a known common publication.
@pytest.mark.parametrize("summed", [False, True])
@pytest.mark.parametrize("field", ["source_release_key", "source_sha256"])
@pytest.mark.parametrize("bad_value", [None, "", " ", 1])
def test_monthly_window_requires_actual_publication_identity(summed, field, bad_value):
    rows = (
        _window_cell_rows()
        if summed
        else [_window_fact_row(month, value=1) for month in _FISCAL_MONTHS]
    )
    reference = _sum_window_reference() if summed else _window_reference()
    for row in rows:
        owner = row if field == "source_release_key" else row["source"]
        if bad_value is None:
            owner.pop(field)
        else:
            owner[field] = bad_value
    with pytest.raises(ValueError, match="requires nonempty source_release_key"):
        compile_ledger_target_references(rows, [reference], country="us")
    # The same authority must run if a caller bypasses selector resolution.
    with pytest.raises(ValueError, match="requires nonempty source_release_key"):
        target_spec_from_ledger_reference(tuple(rows), reference)
