"""Guarantees for the UK active-subset Ledger target references.

``uk/target_references.json`` is the typed activation surface derived from the
value-free UK national contract. Observed values remain in Ledger facts; this
resource only declares which country-level facts Microcosm activates for 2025.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from importlib import resources as importlib_resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.target_reference_authoring import (
    TargetReferenceAuthoringConfig,
    author_target_references,
)
from microcosm.build.uk_runtime.local_target_census import _LEDGER_FACT_FEED_PIN
from microcosm.build.uk_runtime.national_chronicle_feed import (
    load_uk_national_chronicle_feed,
)
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from tools.generate_uk_target_references import (
    POLICYENGINE_BINDING_KEYS,
    _annual_uc_award_band_token,
    _geography_pins,
    _reference_metadata,
    _sum_target_ids,
    _value_operation_by_target_id,
)

ACTIVE_REFERENCE_COUNT = 415
UK_DATA_REPO = "policyengine-" + "uk-data"

FIXTURE_REFERENCE_NAMES = {
    "obr.income_tax",
    "dwp.uc.two_child_limit.households_affected",
    "ons.population.uk_total",
    "hmrc/employment_income_income_band_12_570_to_15_000",
    "slc.repayments.england_plan_2",
    "obr.esa",
    "hmrc.cgt.gains_total",
    "hmrc.cgt.taxpayers_total",
    "dwp.uc.households",
    "dwp.uc.households_single_no_children",
}

FIXTURE_FEED_ROWS = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)

STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"


def _load_uk_resource(name: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("surface", ["national", "local"])
def test_committed_surfaces_regenerate_from_pinned_feed(
    tmp_path: Path, surface: str
) -> None:
    root = Path(__file__).resolve().parents[3]
    # Local references keep their independently reviewed historical input.
    # Never fall back to the configured national feed for the local surface.
    environment_key = (
        "CHRONICLE_UK_FACTS" if surface == "national" else "CHRONICLE_UK_LOCAL_FACTS"
    )
    configured = os.environ.get(environment_key)
    feed = Path(configured) if configured else root / STABLE_UK_FACT_FEED_NAME
    if not feed.exists():
        pytest.skip(f"pinned UK Chronicle {surface} consumer feed is not present")

    if feed.is_dir():
        artifact_path = feed
    else:
        default_manifest = root / ".codex-work/consumer_facts_uk_manifest.json"
        manifest = (
            default_manifest if not configured else feed.with_name("manifest.json")
        )
        if not manifest.is_file():
            pytest.skip(
                f"pinned UK Chronicle {surface} consumer manifest is not present"
            )
        artifact_path = tmp_path / "consumer-artifact"
        artifact_path.mkdir()
        (artifact_path / "consumer_facts.jsonl").symlink_to(feed.resolve())
        (artifact_path / "manifest.json").symlink_to(manifest.resolve())

    if surface == "national":
        pin = load_uk_national_chronicle_feed()
        facts_sha256, manifest_sha256 = pin.facts_sha256, pin.manifest_sha256
        expected_rows = pin.fact_row_count
    else:
        facts_sha256 = _LEDGER_FACT_FEED_PIN["facts_sha256"]
        manifest_sha256 = _LEDGER_FACT_FEED_PIN["manifest_sha256"]
        expected_rows = 128_717
    artifact = load_ledger_consumer_artifact(
        artifact_path,
        expected_facts_sha256=facts_sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    assert artifact.fact_row_count == expected_rows
    facts_path = (
        artifact.path / "consumer_facts.jsonl"
        if artifact.path.is_dir()
        else artifact.path
    )

    package = root / "packages/microcosm-build/src/microcosm/build/uk"
    generated = tmp_path / "generated"
    generated.mkdir()
    prefix = "" if surface == "national" else "local_"
    references_name = f"{prefix}target_references.json"
    membership_name = f"{prefix}target_reference_membership.json"
    generator = f"generate_uk_{prefix}target_references.py"
    # This stable provenance label is independent of the artifact location;
    # the facts and manifest above must still match the surface-specific pin.
    source_fact_feed = _load_uk_resource(membership_name)["source_fact_feed"]
    arguments = [
        sys.executable,
        str(root / "tools" / generator),
        "--contract",
        str(package / "uk_population_targets.json"),
        "--ledger-facts",
        str(facts_path),
        "--source-fact-feed",
        source_fact_feed,
        "--output",
        str(generated / references_name),
        "--membership-report",
        str(generated / membership_name),
    ]
    if surface == "local":
        arguments.extend(["--crosswalk", str(package / "local_area_crosswalk.json")])
    subprocess.run(arguments, cwd=root, check=True)
    for name in (references_name, membership_name):
        assert _sha256(generated / name) == _sha256(package / name), name


def _contract_targets_by_id() -> dict[str, dict]:
    contract = _load_uk_resource("uk_population_targets.json")
    return {target["target_id"]: target for target in contract["targets"]}


def _expected_reference_entity(target: dict) -> str:
    binding = target["bindings"]["policyengine"]
    return binding.get("from_entity") or binding.get("map_to") or "household"


def _fixture_feed_rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE_FEED_ROWS.read_text().splitlines()]


def test_uk_target_references_load_as_typed_non_empty_resource() -> None:
    spec = load_country_spec("uk")

    assert len(spec.target_references) == ACTIVE_REFERENCE_COUNT
    assert {reference.name for reference in spec.target_references}


def test_uk_target_references_follow_contract_derivation_rules() -> None:
    resource = _load_uk_resource("target_references.json")
    targets_by_id = _contract_targets_by_id()
    names = [reference["name"] for reference in resource["target_references"]]

    assert resource["country"] == "uk"
    assert resource["allowed_value_operations"] == [
        "identity",
        "sum",
        "difference",
        "calendar_year_average",
        "latest_plateau",
        "count_x_mean",
        "monthly_window_sum_average",
    ]
    assert len(names) == len(set(names))

    for reference in resource["target_references"]:
        contract_target_id = reference["metadata"]["contract_target_id"]
        target = targets_by_id[contract_target_id]
        binding = target["bindings"]["policyengine"]

        for key, value in target["ledger_selector"].items():
            if key == "dimension_values":
                assert value.items() <= reference["ledger_selector"][key].items()
                continue
            assert reference["ledger_selector"][key] == value
        assert reference["ledger_selector"]["geography_level"] == "country"
        assert reference["ledger_selector"]["geography_id"]
        assert reference["entity"] == _expected_reference_entity(target)
        expected_measure = (
            binding["metric_name"]
            if reference["name"] == contract_target_id
            else reference["name"]
        )
        assert reference["measure"] == expected_measure
        assert reference["family"] == target["family"]
        assert reference["period"] == 2025
        expected_metadata = {
            "contract_target_id": contract_target_id,
            "measure_kind": "prepared_column",
        }
        observation_basis = target["measurement"].get("observation_basis")
        if observation_basis is not None:
            expected_metadata["observation_basis"] = observation_basis
        source_months = target["measurement"].get("source_months")
        if source_months is not None:
            expected_metadata["uk_uc_expected_source_months"] = json.dumps(
                source_months, separators=(",", ":")
            )
        assert reference["metadata"] == expected_metadata
        # The measure is a prepared column, so the pointed-to contract binding
        # must carry what the microcosm#622 materializer needs to prepare it.
        assert (
            binding.get("value_variable")
            or binding.get("value_expression")
            or binding.get("kind")
        ), contract_target_id


def test_childcare_and_bus_references_compile_with_declared_provenance() -> None:
    root = Path(__file__).resolve().parents[3]
    feed = root / STABLE_UK_FACT_FEED_NAME
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")

    references = [
        reference
        for reference in load_country_spec("uk").target_references
        if reference.family in {"hmrc_tfc", "dfe_funded_childcare", "dft_local_bus"}
    ]
    facts = []
    for line in feed.open():
        fact = json.loads(line)
        source_name = fact.get("source", {}).get("source_name") or fact.get(
            "observed_measure", {}
        ).get("source_name")
        if source_name in {"hmrc", "dfe", "dft"}:
            facts.append(fact)

    registry = compile_ledger_target_references(facts, references, country="uk")
    specs = {spec.name: spec for spec in registry.specs}
    assert {name: spec.value for name, spec in specs.items()} == {
        "hmrc.tfc.government_top_up": 599_800_000.0,
        "hmrc.tfc.children_with_used_accounts": 1_151_515.0,
        "dfe.funded_childcare.working_parent_children_2_to_4": 621_482.0,
        "dfe.funded_childcare.early_learning_2_year_olds": 95_031.0,
        "dfe.funded_childcare.universal_only_children": 396_965.0,
        "dft.bus_fare_receipts.england": 3_417_388_656.43538,
        "dft.bus_net_support.england": 3_024_904_320.8399997,
    }
    assert {
        name: spec.metadata["ledger_entity_name"] for name, spec in specs.items()
    } == {
        "hmrc.tfc.government_top_up": "government",
        "hmrc.tfc.children_with_used_accounts": "person",
        "dfe.funded_childcare.working_parent_children_2_to_4": "person",
        "dfe.funded_childcare.early_learning_2_year_olds": "person",
        "dfe.funded_childcare.universal_only_children": "person",
        "dft.bus_fare_receipts.england": "institutional_sector",
        "dft.bus_net_support.england": "institutional_sector",
    }
    assert (
        specs["dfe.funded_childcare.universal_only_children"].metadata[
            "ledger_value_formula"
        ]
        == "minuend - subtrahend"
    )


def test_dfe_extended_sum_refuses_the_suppressed_2024_member() -> None:
    root = Path(__file__).resolve().parents[3]
    feed = root / STABLE_UK_FACT_FEED_NAME
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")

    reference = next(
        reference
        for reference in load_country_spec("uk").target_references
        if reference.name == "dfe.funded_childcare.working_parent_children_2_to_4"
    )
    facts = [json.loads(line) for line in feed.open() if '"source_name": "dfe"' in line]

    with pytest.raises(ValueError, match="expected 2 members.*resolved 1"):
        compile_ledger_target_references(
            facts,
            [replace(reference, period=2024)],
            country="uk",
        )


def test_uk_target_references_do_not_bind_known_mismatched_property_amounts() -> None:
    resource = _load_uk_resource("target_references.json")

    assert not [
        reference["name"]
        for reference in resource["target_references"]
        if reference["metadata"]["contract_target_id"]
        == "hmrc.spi.property_income.amount_by_total_income_band"
    ]


def test_uk_target_references_do_not_emit_nan_uc_payment_bands() -> None:
    resource = _load_uk_resource("target_references.json")

    assert not [
        reference["name"]
        for reference in resource["target_references"]
        if "uc_payment_dist" in reference["name"] and "nan_to_nan" in reference["name"]
    ]


def test_uc_payment_fanout_excludes_source_only_unbounded_rows() -> None:
    assert _annual_uc_award_band_token("No payment") == ""
    assert _annual_uc_award_band_token("£1,500.01 or over") == ""
    assert _annual_uc_award_band_token("£2,500.01 or over") == ""


def test_ons_age_total_targets_pin_exact_age_dimension_set() -> None:
    references = {
        reference["name"]: reference
        for reference in _load_uk_resource("target_references.json")[
            "target_references"
        ]
    }

    assert references["ons.population.scotland_babies_under_1"]["ledger_selector"][
        "dimensions"
    ] == ["age"]
    assert references["ons.population.scotland_children_under_16"]["ledger_selector"][
        "dimensions"
    ] == ["age"]


def test_uc_composition_targets_pin_paid_cells_and_explicit_month_windows() -> None:
    contract = _load_uk_resource("uk_population_targets.json")
    targets = {target["target_id"]: target for target in contract["targets"]}
    sum_target_ids = _sum_target_ids(contract)
    operations = _value_operation_by_target_id(contract)
    composition = {
        target_id: target
        for target_id, target in targets.items()
        if target["family"] == "dwp_universal_credit"
        and "dimension_values" in target["ledger_selector"]
    }
    assert composition
    for target_id, target in composition.items():
        selector = target["ledger_selector"]
        pins = selector["dimension_values"]
        assert set(pins) <= set(selector["dimensions"])
        if target_id.startswith("dwp.uc.households_children_"):
            assert selector["dimensions"] == [
                "number_of_children",
                "payment_indicator",
                "child_entitlement",
            ]
            assert pins["payment_indicator"] == "Yes"
            assert pins["child_entitlement"] == "all"
            assert selector["source_measure_id"] == "total_benefit_units"
            assert operations[target_id] == "calendar_year_average"
        elif target_id == "dwp.uc.households" or target_id.startswith(
            ("dwp.uc.households_single_", "dwp.uc.households_couple_")
        ):
            assert selector["dimensions"] == [
                "family_type",
                "payment_indicator",
                "child_entitlement",
            ]
            assert pins["payment_indicator"] == "Yes"
            assert pins["child_entitlement"] == ["No", "Yes"]
            assert selector["source_measure_id"] == "benefit_units"
            assert operations[target_id] == "monthly_window_sum_average"
        else:
            # Merely naming a dimension set still does not request a sum.
            assert target_id not in sum_target_ids
            continue
        # The list of months makes these candidates for aggregation, but the
        # declared operation averages months rather than summing a year's stock.
        assert target_id in sum_target_ids
        assert selector["period_value"] == [
            f"2025-{month:02d}" for month in range(1, 13)
        ]


def test_prefix_geography_pins_carry_scotgov_and_england_scoped_slc_families() -> None:
    """Prefix pins carry the families the nation-substring rule cannot.

    Chronicle stamps Scottish Government facts S92000003, but the substring
    rule sees no "scotland" in "scotgov" or "scottish_child_payment" and used
    to pin the family to the UK, so it could never match. The SLC
    borrower-plan forecasts and the student-support publication are England
    publications (facts stamped E92000001) whose contract side is
    England-scoped too — the borrower bindings filter country == ENGLAND
    explicitly and the support model variables are England-gated by
    construction — while the GB default pin could never match.
    slc.repayments.devolved_total and the dwp.pip claimant counts keep the GB
    pin and stay held: activating them needs contract redesign (per-nation
    repayment rows; an England-and-Wales PIP binding), not a pin change.
    """
    contract = _load_uk_resource("uk_population_targets.json")
    pins = _geography_pins(contract)
    scotgov_ids = {
        str(target["target_id"])
        for target in contract["targets"]
        if str(target["target_id"]).startswith("scotgov.")
    }
    assert scotgov_ids == {
        f"scotgov.council_tax_stock.band_{band}" for band in "abcdefgh"
    } | {
        "scotgov.council_tax_stock.total",
        "scotgov.scottish_child_payment_spending",
    }
    assert {pins[target_id]["geography_id"] for target_id in scotgov_ids} == {
        "S92000003"
    }

    england_slc_ids = {
        str(target["target_id"])
        for target in contract["targets"]
        if str(target["target_id"]).startswith(("slc.borrowers.", "slc.support."))
    }
    assert len(england_slc_ids) == 10
    assert {pins[target_id]["geography_id"] for target_id in england_slc_ids} == {
        "E92000001"
    }
    assert pins["slc.repayments.devolved_total"]["geography_id"] == "K03000001"
    assert (
        pins["dwp.pip.daily_living_standard_claimants"]["geography_id"] == "K03000001"
    )
    assert (
        pins["dwp.pip.daily_living_enhanced_claimants"]["geography_id"] == "K03000001"
    )

    def haystack(target: dict) -> str:
        selector = target.get("ledger_selector") or {}
        return " ".join(
            (
                str(target["target_id"]).lower(),
                str(selector.get("source_concept", "")).lower(),
                str(selector.get("source_measure_id", "")).lower(),
            )
        )

    substring_scotland = {
        str(target["target_id"])
        for target in contract["targets"]
        if "scotland" in haystack(target)
        and "northern" not in haystack(target)
        and "domestic_rates" not in haystack(target)
    }
    scotland_pinned = {
        target_id
        for target_id, pin in pins.items()
        if pin["geography_id"] == "S92000003"
    }
    assert scotland_pinned == scotgov_ids | substring_scotland

    membership = _load_uk_resource("target_reference_membership.json")
    for target_id in sorted(scotgov_ids):
        assert membership["geography_pins"][target_id]["geography_id"] == "S92000003"
        assert membership["targets"][target_id]["status"] == "active"
    for target_id in sorted(england_slc_ids):
        assert membership["geography_pins"][target_id]["geography_id"] == "E92000001"
        assert membership["targets"][target_id]["status"] == "active"
    for target_id in (
        "slc.repayments.devolved_total",
        "dwp.pip.daily_living_standard_claimants",
        "dwp.pip.daily_living_enhanced_claimants",
    ):
        assert (
            membership["targets"][target_id]["status"] == "no_fact_at_or_before_period"
        )


def test_uk_target_reference_membership_report_is_packaged() -> None:
    membership = _load_uk_resource("target_reference_membership.json")

    assert membership["target_period"] == 2025
    assert membership["active_reference_count"] == ACTIVE_REFERENCE_COUNT
    assert membership["status_counts"] == {
        "active": 415,
        "no_fact_at_or_before_period": 7,
        "signed_excluded": 8,
    }
    assert membership["genuine_sum_residue"]
    assert membership["uprating_holds"]
    assert membership["fanout_family_outcomes"] == [
        {
            "family": "hmrc_spi",
            "status": "active_with_signed_property_amount_exclusion",
            "active_reference_count": 143,
            "signed_rationale": (
                "SPI income-band targets fan out by strict total-income-band "
                "dimension pins, except the HMRC property-income amount "
                "surface. Those 13 rows are signed out because Ledger carries "
                "the official SPI Table 3.7 net property-income amounts, "
                "while the incumbent target applies the populace-side x1.9 "
                "property-income undercount adjustment traced to "
                f"{UK_DATA_REPO} PR #311 / issue #230 and HMRC Property Rental "
                "Income Statistics."
            ),
        },
        {
            "family": "dwp_universal_credit",
            "status": "active_with_unmapped_vintage_residue_skipped",
            "active_reference_count": 100,
            "skipped_unmapped_fact_count": 12,
            "signed_rationale": (
                "UC payment-distribution targets fan out over family_type and "
                "monthly_award_amount_bands into incumbent-compatible "
                "annual-payment rows. The four source-only 'No payment' facts "
                "and eight overlapping source-only 'or over' facts are left "
                "out; no active reference may use the legacy nan_to_nan band "
                "name."
            ),
        },
        {
            "family": "council_tax_stock",
            "status": "active_declared_rows",
            "active_reference_count": 18,
            "signed_rationale": (
                "VOA (England and Wales) and Scottish Government CTAXBASE "
                "(Scotland) council-tax stock bands are declared as nine "
                "explicit target rows each, including total, and each resolves "
                "with its country-level geography and band pin."
            ),
        },
    ]
    assert membership["signed_exclusion_rationales"] == [
        {
            "family": "hmrc_spi",
            "target_id": "hmrc.spi.property_income.amount_by_total_income_band",
            "status": "signed_excluded",
            "signed_rationale": (
                "Signed out pending a first-class value-scaling operation or "
                "a Chronicle package for HMRC Property Rental Income "
                "Statistics with declared reconciliation: the Ledger facts "
                "are official HMRC SPI Table 3.7 net property-income amounts, "
                "while the incumbent calibration target applies the "
                "populace-side x1.9 property-income undercount adjustment. "
                "The x1.9 trace is uk-data PR #311 / issue uk-data#230: SPI "
                "covers only taxpayers with liability, "
                "and HMRC Property Rental Income Statistics show GBP 46.68bn "
                "versus SPI about GBP 24.5bn for 2020-21. Binding the raw SPI "
                "facts would knowingly calibrate to 10/19 of the incumbent "
                "surface."
            ),
        },
        {
            "family": "ons_population",
            "target_id": "ons.population.scotland_households_3plus_children",
            "status": "signed_excluded",
            "signed_rationale": (
                "Signed out pending a Scotland household-composition fact: "
                "the current selector reaches person-level ONS mid-year "
                "population age rows, not households with three or more "
                "children. microcosm#736 tracks the missing declaration."
            ),
        },
    ]
    assert membership["multi_fact_rationales"] == []


def test_uk_fixture_b_signed_differences_carry_ruled_rationales() -> None:
    differences = {
        row["name"]: row
        for row in _load_uk_resource(
            "ledger_compile_parity_incumbent_2025_signed_differences.json"
        )["differences"]
    }

    assert "2023-24 outturn" in differences["hmrc.cgt.gains_total"]["reason"]
    assert "forecast/uprated value" in differences["hmrc.cgt.gains_total"]["reason"]
    assert "single-age-90 share" in differences["ons.population.female_85_89"]["reason"]
    assert "single-age-90 share" in differences["ons.population.male_85_89"]["reason"]
    assert (
        "ages 91+ unconstrained"
        in differences["ons.population.female_90_plus"]["reason"]
    )
    assert (
        "ages 91+ unconstrained" in differences["ons.population.male_90_plus"]["reason"]
    )


def test_uk_target_references_compile_from_real_staged_feed_rows() -> None:
    spec = load_country_spec("uk")
    references = [
        reference
        for reference in spec.target_references
        if reference.name in FIXTURE_REFERENCE_NAMES
    ]

    registry = compile_ledger_target_references(
        [
            json.loads(line)
            for line in FIXTURE_FEED_ROWS.read_text().splitlines()
            if line.strip()
        ],
        references,
        country="uk",
    )

    targets = {target.name: target for target in registry.specs}
    assert set(targets) == FIXTURE_REFERENCE_NAMES

    income_tax = targets["obr.income_tax"]
    assert income_tax.value == pytest.approx(331_437_583_074.4429)
    assert income_tax.period == 2025
    assert income_tax.metadata["ledger_assertion"] == "source_projection"
    assert income_tax.metadata["ledger_assertion_policy"] == ("allow_source_projection")

    tcl_households = targets["dwp.uc.two_child_limit.households_affected"]
    assert tcl_households.value == 469_780
    assert tcl_households.metadata["ledger_fact_period"] == "2025-04"

    population = targets["ons.population.uk_total"]
    assert population.value == 69_281_437
    assert population.metadata["ledger_value_operation"] == "sum"
    assert population.metadata["uprating_from_period"] == "2024"

    spi_band = targets["hmrc/employment_income_income_band_12_570_to_15_000"]
    assert spi_band.value == 16_900_000_000
    assert (
        spi_band.metadata["contract_target_id"]
        == "hmrc.spi.employment_income.amount_by_total_income_band"
    )

    slc_plan_2 = targets["slc.repayments.england_plan_2"]
    assert slc_plan_2.value == pytest.approx(2_778_253_361.64)
    assert slc_plan_2.metadata["ledger_member_fact_count"] == "2"

    assert targets["hmrc.cgt.gains_total"].value == 65_937_000_000
    assert targets["hmrc.cgt.taxpayers_total"].value == 378_000
    assert targets["hmrc.cgt.gains_total"].metadata["ledger_fact_period"] == "2023"

    caseload = targets["dwp.uc.households"]
    assert caseload.value == 6_197_311
    assert caseload.metadata["ledger_member_fact_count"] == "120"
    assert caseload.metadata["ledger_value_operation"] == "monthly_window_sum_average"
    assert caseload.metadata["ledger_source_month_count"] == "12"
    assert caseload.metadata["ledger_source_cell_count_per_month"] == "10"

    family_type = targets["dwp.uc.households_single_no_children"]
    assert family_type.value == pytest.approx(2_990_070.1666666665)
    assert family_type.metadata["ledger_member_fact_count"] == "24"
    assert (
        family_type.metadata["ledger_value_operation"] == "monthly_window_sum_average"
    )
    assert family_type.metadata["ledger_source_month_count"] == "12"
    assert family_type.metadata["ledger_source_cell_count_per_month"] == "2"


def test_uk_generator_averages_paid_monthly_sums_and_preserves_other_uc_operations() -> (
    None
):
    contract = _load_uk_resource("uk_population_targets.json")
    operations = _value_operation_by_target_id(contract)
    monthly_sum_ids = {
        "dwp.uc.households",
        "dwp.uc.households_single_no_children",
        "dwp.uc.households_single_with_children",
        "dwp.uc.households_couple_no_children",
        "dwp.uc.households_couple_with_children",
    }
    uc_target_ids = {
        target["target_id"]
        for target in contract["targets"]
        if target["family"] == "dwp_universal_credit"
    }
    assert monthly_sum_ids < uc_target_ids
    assert {
        target_id
        for target_id in uc_target_ids
        if operations[target_id] == "monthly_window_sum_average"
    } == monthly_sum_ids
    assert all(
        operations[target_id] == "calendar_year_average"
        for target_id in uc_target_ids - monthly_sum_ids
    )
    facts = _fixture_feed_rows()
    authored = author_target_references(
        contract,
        facts,
        TargetReferenceAuthoringConfig(
            target_period=2025,
            geography_pins=_geography_pins(contract),
            value_operation_by_target_id=operations,
            reference_metadata_by_target_id=_reference_metadata(contract),
            binding_vocabulary=POLICYENGINE_BINDING_KEYS,
            source_fact_feed="uk_target_reference_feed_rows.jsonl",
        ),
    )
    references = {reference["name"]: reference for reference in authored.references}
    uc_reference = references["dwp.uc.households"]
    assert uc_reference["value_operation"] == "monthly_window_sum_average"
    assert uc_reference["period_match_policy"] == "source_window"
    assert len(uc_reference["value_operands"]) == 10
    assert (
        authored.membership_report["targets"]["dwp.uc.households"]["status"] == "active"
    )
    registry = compile_ledger_target_references(
        facts, [LedgerTargetReference(**uc_reference)], country="uk"
    )
    # Public DWP detail cells sum to 74,367,732 across the twelve months;
    # the reference is a mean monthly stock, including unknown family type.
    assert registry.specs[0].value == 74_367_732 / 12


def test_paid_uc_public_fixture_preserves_complete_source_cells_including_zeros() -> (
    None
):
    # Exact Chronicle ec7169b5 producer rows for the public Stat-Xplore response.
    source_sha256 = "f4fb46fefd66a20b8d6d12c051a224743b6754fee53e1cee0be5ead4d34de550"
    rows = [
        row
        for row in _fixture_feed_rows()
        if row["source"]["source_sha256"] == source_sha256
    ]
    assert len(rows) == 120
    assert {row["layout"]["table_record_kind"] for row in rows} == {"detail"}
    assert {row["dimensions"]["payment_indicator"] for row in rows} == {"Yes"}
    assert {row["period"]["value"] for row in rows} == {
        f"2025-{month:02d}" for month in range(1, 13)
    }
    assert (
        len(
            {
                (
                    row["period"]["value"],
                    row["dimensions"]["family_type"],
                    row["dimensions"]["child_entitlement"],
                )
                for row in rows
            }
        )
        == 120
    )
    assert sum(row["value"] for row in rows) == 74_367_732
    missing_zero = next(
        row
        for row in rows
        if row["dimensions"]["family_type"] == "Single, no children"
        and row["dimensions"]["child_entitlement"] == "Yes"
    )
    assert missing_zero["value"] == 0
    reference = next(
        reference
        for reference in load_country_spec("uk").target_references
        if reference.name == "dwp.uc.households_single_no_children"
    )
    incomplete = [
        row
        for row in _fixture_feed_rows()
        if row["aggregate_fact_key"] != missing_zero["aggregate_fact_key"]
    ]
    with pytest.raises(
        ValueError, match="requires exactly 24 selected facts.*resolved 23"
    ):
        compile_ledger_target_references(incomplete, [reference], country="uk")


def test_uk_target_references_constrain_a_frame_with_prepared_columns() -> None:
    """The full active subset compiles into constraint rows, end to end.

    Measures name prepared columns (the US prepared-indicator-column doctrine):
    the microcosm#622 materializer owns building them from the contract
    bindings, so this test hand-prepares one column per measure on the
    reference's entity table and proves the compiled registry constrains a
    household-weighted frame with zero skipped targets and exact row
    aggregates.
    """

    spec = load_country_spec("uk")
    references = [
        reference
        for reference in spec.target_references
        if reference.name in FIXTURE_REFERENCE_NAMES
    ]
    feed_rows = [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]

    registry = compile_ledger_target_references(feed_rows, references, country="uk")
    assert len(registry.specs) == len(FIXTURE_REFERENCE_NAMES)

    n_households = 3
    weights = np.array([10.0, 20.0, 30.0])
    household_columns: dict[str, np.ndarray] = {}
    benunit_columns: dict[str, np.ndarray] = {}
    person_columns: dict[str, np.ndarray] = {}
    expected_aggregates: dict[str, float] = {}
    for index, compiled in enumerate(registry.specs):
        column = np.array([index + 1.0, 2.0 * (index + 1.0), 0.0])
        columns = {
            "person": person_columns,
            "benunit": benunit_columns,
            "household": household_columns,
        }[compiled.entity]
        columns[compiled.measure] = column
        expected_aggregates[f"{compiled.name}@{compiled.period}"] = float(
            (column * weights).sum()
        )

    household_ids = np.arange(n_households, dtype="int64")
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": household_ids,
                    "person_household_id": household_ids,
                    "person_benunit_id": household_ids,
                    **person_columns,
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": household_ids, **benunit_columns}),
            "household": pd.DataFrame(
                {"household_id": household_ids, **household_columns}
            ),
        },
        EntitySchema(group_entities=("household", "benunit")),
        {"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )

    problem = build_constraint_matrix(frame, registry.to_target_set())

    assert problem.skipped == ()
    assert len(problem.names) == len(FIXTURE_REFERENCE_NAMES)
    achieved = problem.matrix @ problem.initial_weights.values
    for name, estimate in zip(problem.names, achieved, strict=True):
        assert estimate == expected_aggregates[name], name
    fact_values_by_name = {
        f"{compiled.name}@{compiled.period}": compiled.value
        for compiled in registry.specs
    }
    for name, target_value in zip(problem.names, problem.target_vector, strict=True):
        assert target_value == fact_values_by_name[name], name


def _real_uk_consumer_fact_rows() -> list[dict]:
    """Public rows copied from .codex-work/consumer_facts_uk.jsonl."""

    return [
        {
            "aggregate_fact_key": "ledger.aggregate_fact.v2:a36696207826083b6dd7e878",
            "aggregation": {"method": "sum"},
            "assertion": "observation",
            "concept_alignment": {
                "authority": "isc",
                "canonical_concept": "isc.pupils_at_member_schools",
                "relation": "source_label",
                "source_concept": "isc.pupils_at_member_schools",
            },
            "dimension_set_key": "ledger.dimension_set.v2:798fe9e6b99defdb40a97be3",
            "dimensions": {"school_membership": "isc"},
            "entity": {"name": "person", "role": "private_school_pupil"},
            "geography": {
                "id": "K02000001",
                "level": "country",
                "vintage": "current",
            },
            "layout": {
                "groupby_dimension": "isc.census_line",
                "groupby_value_id": "pupils_at_isc_schools",
                "measure_id": "pupils",
                "record_set_id": "isc.census_2023.pupils",
            },
            "legacy_fact_key": "ledger.fact.v1:6862b35fc66e993fd1b3a496",
            "lineage": {
                "source_record_id": (
                    "isc.census_2023.pupils.pupils_at_isc_schools.pupils"
                )
            },
            "observed_measure": {
                "source_concept": "isc.pupils_at_member_schools",
                "source_measure_id": "pupils",
                "source_name": "isc",
                "source_table": "ISC annual Census 2023",
                "unit": "count",
            },
            "period": {"type": "month", "value": "2023-01"},
            "semantic_fact_key": "ledger.semantic_fact.v2:dcdb6fab7a97e0d16f5332a0",
            "source": {
                "source_file": "isc_census_2023_final.pdf",
                "source_name": "isc",
                "source_table": "ISC annual Census 2023",
                "url": "https://www.isc.co.uk/media/9316/isc_census_2023_final.pdf",
                "vintage": "census_2023",
            },
            "universe_constraint_set_key": (
                "ledger.universe_constraint_set.v2:6128dc3fab81d86b2f1e7d92"
            ),
            "universe_constraints": {"domain": "independent_school_education"},
            "value": 554_243,
            "value_type": "integer",
        },
        {
            "aggregate_fact_key": "ledger.aggregate_fact.v2:2d8306be080946696b35e1ee",
            "aggregation": {"method": "sum"},
            "assertion": "observation",
            "concept_alignment": {
                "authority": "ons",
                "canonical_concept": "ons.household_interest_resources",
                "relation": "source_label",
                "source_concept": "ons.ukea_haxv",
            },
            "dimension_set_key": "ledger.dimension_set.v2:44136fa355b3678a1146ad16",
            "dimensions": {},
            "entity": {"name": "household", "role": "resident_household"},
            "geography": {
                "id": "K02000001",
                "level": "country",
                "vintage": "current",
            },
            "layout": {
                "groupby_dimension": "ons.ukea_series",
                "groupby_value_id": "haxv",
                "measure_id": "amount",
                "record_set_id": "ons.ukea_haxv.savings_interest.cy2023",
            },
            "legacy_fact_key": "ledger.fact.v1:26472d1cd1730a29a966ad7c",
            "lineage": {
                "source_record_id": "ons.ukea_haxv.savings_interest.cy2023.haxv.amount"
            },
            "observed_measure": {
                "source_concept": "ons.ukea_haxv",
                "source_measure_id": "amount",
                "source_name": "ons",
                "source_table": (
                    "UKEA time series HAXV: Households (S.14) interest (D.41) "
                    "resources, current price GBP million NSA"
                ),
                "unit": "gbp",
            },
            "period": {"type": "calendar_year", "value": 2023},
            "semantic_fact_key": "ledger.semantic_fact.v2:9ac263134990690ff4d4aeca",
            "source": {
                "source_file": "haxv.csv",
                "source_name": "ons",
                "source_table": (
                    "UKEA time series HAXV: Households (S.14) interest (D.41) "
                    "resources, current price GBP million NSA"
                ),
                "url": (
                    "https://www.ons.gov.uk/generator?format=csv&uri=/economy/"
                    "grossdomesticproductgdp/timeseries/haxv/ukea"
                ),
                "vintage": "ukea_2026_06_30",
            },
            "universe_constraint_set_key": (
                "ledger.universe_constraint_set.v2:fa6e32cc24b49ed112373325"
            ),
            "universe_constraints": {"domain": "national_accounts"},
            "value": 86_040_000_000,
            "value_type": "integer",
        },
    ]
