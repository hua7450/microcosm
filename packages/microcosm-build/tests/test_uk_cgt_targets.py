"""Tests for the UK capital gains Ledger target references.

These tests pin individuals-only facts, provenance and required coverage.
The fixture also contains historical trust-inclusive totals; those must not
enter a person-level calibration by accident.
"""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.uk_runtime.fiscal_targets import (
    UK_CGT_REQUIRED_COLUMNS,
    UK_CGT_TARGET_COVERAGE_REQUIREMENTS,
    UK_CGT_TARGET_SPECS,
    UK_FISCAL_TARGET_REGISTRY,
)
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry
from microcosm.calibrate import TargetRegistry

FIXTURE_FEED_ROWS = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)
CGT_TARGET_NAMES = {
    "hmrc.cgt.gains_total",
    "hmrc.cgt.taxpayers_total",
    "hmrc.cgt.liability_total",
}


def _facts():
    return [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]


def _compiled_cgt_registry():
    spec = load_country_spec("uk")
    references = [
        reference
        for reference in spec.target_references
        if reference.name in CGT_TARGET_NAMES
    ]
    return compile_ledger_target_references(_facts(), references, country="uk")


@pytest.fixture
def compile_cgt(monkeypatch):
    # Exercise the production compiler on the three reviewed references only;
    # regeneration tests separately verify the complete national/UC roster.
    from microcosm.build.uk_runtime import ledger_targets

    references = tuple(
        r
        for r in load_country_spec("uk").target_references
        if r.name in CGT_TARGET_NAMES
    )
    monkeypatch.setattr(
        ledger_targets,
        "load_country_spec",
        lambda country: SimpleNamespace(target_references=references),
    )
    return compile_uk_target_registry


def test_inline_cgt_target_specs_are_retired():
    assert UK_CGT_TARGET_SPECS == ()
    assert len(UK_FISCAL_TARGET_REGISTRY) == 0


def test_compiled_references_declare_three_observed_cgt_totals():
    registry = _compiled_cgt_registry()

    assert {spec.name for spec in registry.specs} == CGT_TARGET_NAMES
    assert "obr.capital_gains_tax" not in {
        reference.name for reference in load_country_spec("uk").target_references
    }


def test_every_compiled_fact_carries_provenance():
    """A fact without a citation is not a fact."""
    for spec in _compiled_cgt_registry().specs:
        assert "gov.uk" in spec.source
        assert spec.family == "hmrc_cgt"


def test_compiled_facts_match_hmrc_2024_25_individuals_observations():
    by_name = {spec.name: spec for spec in _compiled_cgt_registry().specs}
    assert by_name["hmrc.cgt.gains_total"].value == 119_258_000_000
    assert by_name["hmrc.cgt.taxpayers_total"].value == 551_000
    liability = by_name["hmrc.cgt.liability_total"]
    # Verbatim Chronicle 6fb700e Table 1 provisional observation, not OBR cash.
    assert liability.value == 22_503_000_000
    assert liability.metadata["ledger_aggregate_fact_key"] == (
        "ledger.aggregate_fact.v2:222c397017de7bff0a6583a7"
    )
    assert all(spec.period == 2025 for spec in by_name.values())
    assert all(
        spec.metadata["measurement_period"] == "2024" for spec in by_name.values()
    )
    assert all(
        spec.metadata["source_period_policy"] == "exact_observation"
        for spec in by_name.values()
    )
    assert all(
        spec.metadata["ledger_fact_period"] == "2024" for spec in by_name.values()
    )


def test_individual_scope_is_pinned_to_table1_not_age_marginals():
    references = load_country_spec("uk").target_references
    cgt = [r for r in references if r.name.startswith("hmrc.cgt.")]
    assert len(cgt) == 3
    expected_keys = {
        "hmrc.cgt.taxpayers_total": "31d709fc393c2bf4d04efca5",
        "hmrc.cgt.gains_total": "12060d20a417d85d67cf24e8",
        "hmrc.cgt.liability_total": "222c397017de7bff0a6583a7",
    }
    for reference in cgt:
        assert reference.ledger_selector["aggregate_fact_key"] == (
            "ledger.aggregate_fact.v2:" + expected_keys[reference.name]
        )
        assert reference.ledger_selector["period_type"] == "tax_year"
        assert reference.ledger_selector["period_value"] == 2024
    assert {r.ledger_selector["source_concept"] for r in cgt} == {
        "hmrc.cgt_gains_individuals",
        "hmrc.cgt_taxpayers_individuals",
        "hmrc.cgt_tax_individuals",
    }
    assert all(
        r.ledger_selector["groupby_dimension"] == "hmrc.cgt_table1_line" for r in cgt
    )


def test_measures_are_declared_columns():
    """The registry refuses callables, so measures must be prepared columns."""
    measures = {spec.measure for spec in _compiled_cgt_registry().specs}
    assert measures == {
        "hmrc/capital_gains_total",
        "hmrc/cgt_taxpayers",
        "hmrc/cgt_liability",
    }
    assert set(UK_CGT_REQUIRED_COLUMNS) == {
        "uk_cgt_measure_gains_amount",
        "uk_cgt_measure_taxpayer_count",
    }


def test_facts_are_person_grain():
    """UK measures are person-level, matching the hmrc_calibration convention.

    The weights stay household-level; the frame carries them as household
    ``Weights`` while the constraint rows live on the person table.
    """
    assert all(spec.entity == "person" for spec in _compiled_cgt_registry().specs)


def test_registry_is_uk_and_content_addressed():
    assert UK_FISCAL_TARGET_REGISTRY.country == "uk"
    assert UK_FISCAL_TARGET_REGISTRY.version


def test_coverage_requires_all_three_observed_facts():
    """A build that drops liability, gains or counts must fail coverage."""
    (requirement,) = UK_CGT_TARGET_COVERAGE_REQUIREMENTS
    assert requirement.min_matches == 3
    assert set(requirement.accepted_names) == CGT_TARGET_NAMES


def test_original_cash_forecast_survives_only_as_diagnostic_metadata(
    tmp_path, compile_cgt
):
    compilation = compile_cgt(_facts(), target_period=2025)
    by_name = {spec.name: spec for spec in compilation.registry.specs}
    assert "obr.capital_gains_tax" not in by_name
    metadata = by_name["hmrc.cgt.liability_total"].metadata
    assert metadata["cgt_cash_diagnostic_status"] == "available"
    assert metadata["cgt_cash_diagnostic_role"] == "diagnostic_only_not_in_fit"
    assert metadata["cgt_cash_reconciliation_status"] == "unresolved"
    assert float(metadata["cgt_cash_diagnostic_value_gbp"]) == 21_801_546_197.09165
    assert metadata["cgt_cash_diagnostic_period"] == "2025"
    assert metadata["cgt_cash_diagnostic_ledger_period_type"] == "fiscal_year"
    assert metadata["cgt_cash_diagnostic_ledger_assertion"] == "source_projection"
    assert metadata["cgt_cash_diagnostic_ledger_aggregate_fact_key"] == (
        "ledger.aggregate_fact.v2:93699bb9caa7ec0d6833f420"
    )
    assert "obr.uk" in metadata["cgt_cash_diagnostic_source"]
    # TargetSpec's normal serialization is used in national/local receipts.
    path = compilation.registry.to_json(tmp_path / "registry.json")
    restored = TargetRegistry.from_json(path)
    assert metadata == next(
        spec.metadata
        for spec in restored.specs
        if spec.name == "hmrc.cgt.liability_total"
    )


@pytest.mark.parametrize(
    "cash_change",
    [
        "missing",
        "wrong_year",
        "observation",
        "different_forecast",
        "wrong_period_type",
        "duplicate",
    ],
)
def test_missing_cash_diagnostic_does_not_remove_observed_targets(
    cash_change, compile_cgt, tmp_path
):
    facts = _facts()
    cash = [
        fact
        for fact in facts
        if fact["observed_measure"]["source_concept"] == "obr.capital_gains_tax"
    ]
    assert cash
    if cash_change == "missing":
        facts = [fact for fact in facts if fact not in cash]
    elif cash_change == "duplicate":
        facts.append(deepcopy(cash[0]))
    else:
        for fact in cash:
            if cash_change == "wrong_year":
                fact["period"]["value"] = 2026
            elif cash_change == "observation":
                fact["assertion"] = "observation"
            elif cash_change == "wrong_period_type":
                fact["period"]["type"] = "tax_year"
            else:
                fact["aggregate_fact_key"] = "ledger.aggregate_fact.v2:replacement"
                fact["value"] += 1
    compilation = compile_cgt(facts, target_period=2025)
    assert not compilation.unsupported
    by_name = {spec.name: spec for spec in compilation.registry.specs}
    assert {name: spec.value for name, spec in by_name.items()} == {
        "hmrc.cgt.gains_total": 119_258_000_000,
        "hmrc.cgt.taxpayers_total": 551_000,
        "hmrc.cgt.liability_total": 22_503_000_000,
    }
    metadata = by_name["hmrc.cgt.liability_total"].metadata
    assert metadata["cgt_cash_diagnostic_status"] == "unavailable"
    assert metadata["cgt_cash_diagnostic_unavailable_reason"]
    assert metadata["cgt_cash_diagnostic_expected_fact_key"] == (
        "ledger.aggregate_fact.v2:93699bb9caa7ec0d6833f420"
    )
    assert metadata["cgt_cash_diagnostic_expected_period"] == "2025"
    assert metadata["cgt_cash_diagnostic_expected_period_type"] == "fiscal_year"
    assert metadata["cgt_cash_diagnostic_expected_assertion"] == "source_projection"
    assert "cgt_cash_diagnostic_value_gbp" not in metadata
    assert "cgt_cash_diagnostic_source" not in metadata
    restored = TargetRegistry.from_json(
        compilation.registry.to_json(tmp_path / "registry.json")
    )
    assert metadata == next(
        r.metadata for r in restored if r.name == "hmrc.cgt.liability_total"
    )


@pytest.mark.parametrize("name", sorted(CGT_TARGET_NAMES))
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "wrong_key",
        "wrong_scope",
        "wrong_entity",
        "wrong_period_type",
        "wrong_year",
    ],
)
def test_observed_reference_refuses_missing_or_mismatched_individual_fact(
    name, mutation, compile_cgt
):
    facts = _facts()
    reference = next(
        r for r in load_country_spec("uk").target_references if r.name == name
    )
    fact = next(
        f
        for f in facts
        if f["observed_measure"]["source_concept"]
        == reference.ledger_selector["source_concept"]
        and str(f["period"]["value"]) == "2024"
    )
    if mutation == "missing":
        facts.remove(fact)
    elif mutation == "wrong_key":
        fact["aggregate_fact_key"] = "ledger.aggregate_fact.v2:other_release"
    elif mutation == "wrong_period_type":
        fact["period"]["type"] = "fiscal_year"
    elif mutation == "wrong_year":
        fact["period"]["value"] = 2025
    elif mutation == "wrong_entity":
        fact["entity"]["name"] = "trust"
    else:
        # The shared selector accepts a source or canonical concept. Change
        # both representations so this fixture actually describes other scope.
        wrong_concept = "hmrc.cgt_gains_total"
        fact["observed_measure"]["source_concept"] = wrong_concept
        fact["concept_alignment"]["source_concept"] = wrong_concept
        fact["concept_alignment"]["canonical_concept"] = wrong_concept
    compilation = compile_cgt(facts, target_period=2025)
    assert name not in {row.name for row in compilation.registry.specs}
    assert name in {row["name"] for row in compilation.unsupported}


@pytest.mark.parametrize("change", ["missing", "receiver", "fact_key", "assertion"])
def test_malformed_cash_declaration_remains_a_compile_error(
    change, tmp_path, monkeypatch, compile_cgt
):
    from microcosm.build.uk_runtime import ledger_targets

    resource = ledger_targets.importlib_resources.files("microcosm.build.uk")
    contract = json.loads(resource.joinpath("uk_population_targets.json").read_text())
    declaration = contract["diagnostic_references"]["obr.capital_gains_tax"]
    if change == "missing":
        del contract["diagnostic_references"]
    elif change == "receiver":
        declaration["attach_to_target"] = "hmrc.cgt.gains_total"
    elif change == "fact_key":
        declaration["reference"]["ledger_fact_key"] = None
    else:
        declaration["required_assertion"] = "observation"
    (tmp_path / "uk_population_targets.json").write_text(json.dumps(contract))
    original_files = ledger_targets.importlib_resources.files
    monkeypatch.setattr(
        ledger_targets.importlib_resources,
        "files",
        lambda package: (
            tmp_path if package == "microcosm.build.uk" else original_files(package)
        ),
    )
    compilation = compile_cgt(_facts(), target_period=2025)
    assert "hmrc.cgt.liability_total" not in {r.name for r in compilation.registry}
    assert any(
        "cash diagnostic declaration" in r["reason"] for r in compilation.unsupported
    )


@pytest.mark.parametrize("name", sorted(CGT_TARGET_NAMES))
def test_new_key_revision_does_not_silently_replace_pinned_observation(
    name, compile_cgt
):
    facts = _facts()
    reference = next(
        r for r in load_country_spec("uk").target_references if r.name == name
    )
    original = next(
        f
        for f in facts
        if f["aggregate_fact_key"] == reference.ledger_selector["aggregate_fact_key"]
    )
    revision = deepcopy(original)
    revision["aggregate_fact_key"] = "ledger.aggregate_fact.v2:future_revision"
    revision["value"] += 1
    compilation = compile_cgt([*facts, revision], target_period=2025)
    assert not compilation.unsupported
    selected = next(r for r in compilation.registry if r.name == name)
    assert selected.value == original["value"]
    assert (
        selected.metadata["ledger_aggregate_fact_key"] == original["aggregate_fact_key"]
    )


@pytest.mark.parametrize("name", sorted(CGT_TARGET_NAMES))
def test_duplicate_pinned_observation_fails_loudly(name, compile_cgt):
    facts = _facts()
    reference = next(
        r for r in load_country_spec("uk").target_references if r.name == name
    )
    original = next(
        f
        for f in facts
        if f["aggregate_fact_key"] == reference.ledger_selector["aggregate_fact_key"]
    )
    compilation = compile_cgt([*facts, deepcopy(original)], target_period=2025)
    assert name not in {r.name for r in compilation.registry}
    assert name in {r["name"] for r in compilation.unsupported}
