#!/usr/bin/env python
"""Generate UK Ledger target references from the national contract.

This is an offline authoring tool. It consumes an already-exported Ledger
consumer fact JSONL feed; it does not fetch data or contact Chronicle.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.build.ledger_targets import MONTHLY_WINDOW_OPERATIONS
from microcosm.build.target_reference_authoring import (
    TargetReferenceAuthoringConfig,
    author_target_references,
    target_references_resource,
)
from microcosm.build.uk_runtime.uc_source_periods import uc_source_month_metadata

UK_GEOGRAPHY_IDS = {
    "uk": "K02000001",
    "great_britain": "K03000001",
    "england": "E92000001",
    "scotland": "S92000003",
    "wales": "W92000004",
    "northern_ireland": "N92000002",
}

POLICYENGINE_BINDING_KEYS = frozenset(
    {
        "affected_flag_variable",
        "band",
        "band_filter_dimension",
        "band_period_factor",
        "band_upper_bound",
        "band_upper_bound_inclusive",
        "count_of",
        "filters",
        "folded_into",
        "from_entity",
        "gate_comparison",
        "gate_parameter",
        "gated_variable",
        "groupby_variable",
        "household_conditions",
        "kind",
        "map_to",
        "metric_name",
        "measurement_period",
        "notes",
        "output_delta",
        "output_variable",
        "reduce",
        "require_matching_fact_period",
        "source_lines",
        "threshold_price_base_year",
        "value_expression",
        "value_reduction",
        "value_variable",
        "zeroed_input",
    }
)

_UK_DATA_REPO = "policyengine-" + "uk-data"

DESCRIPTION = (
    "UK active-subset Ledger target references for the FRS 2024-25 line. "
    "Rows are generated from the national rows in uk_population_targets.json: "
    "name is the contract "
    "target_id or an incumbent-compatible fan-out row name; ledger_selector "
    "is the contract selector plus geography pins; entity is from_entity, "
    "then map_to, then household; measure is the prepared-column metric name "
    "or fan-out row name; family is the contract family; period is the model "
    "and calibration year 2025, distinct from the FRS 2024-25 base period 2024. "
    "Observation windows are declared separately; metadata.measurement_period "
    "records observed-year exceptions. Observed values stay in Ledger facts "
    "and resolve through each reference's declared value operation. Deferred "
    "classes and geography-pin decisions are recorded in "
    "uk/target_reference_membership.json. metadata.measure_kind records that "
    "measures are prepared columns produced from the contract binding payload "
    "referenced by metadata.contract_target_id."
)
NATIONAL_GEOGRAPHY_LEVELS = frozenset({"country", "region"})


def main() -> None:
    args = _parser().parse_args()
    contract = _filter_contract_by_geography_levels(
        json.loads(args.contract.read_text()),
        allowed_levels=NATIONAL_GEOGRAPHY_LEVELS,
    )
    _validate_amount_entity_pins(contract)
    facts = [
        json.loads(line)
        for line in args.ledger_facts.read_text().splitlines()
        if line.strip()
    ]
    inverse_mapping = _registry_inverse(contract)
    config = TargetReferenceAuthoringConfig(
        target_period=args.period,
        geography_pins=_geography_pins(contract),
        fanout_name=lambda target, fact: _fanout_name(
            target,
            fact,
            inverse_mapping,
        ),
        sum_target_ids=_sum_target_ids(contract),
        value_operation_by_target_id=_value_operation_by_target_id(contract),
        selector_pins_by_target_id=_selector_pins(contract),
        signed_exclusions_by_target_id=_signed_exclusions(contract),
        reference_metadata_by_target_id=_reference_metadata(contract),
        binding_vocabulary=POLICYENGINE_BINDING_KEYS,
        source_fact_feed=args.source_fact_feed or str(args.ledger_facts),
    )
    authored = author_target_references(contract, facts, config)
    _add_uk_membership_accounting(authored.membership_report, authored.references)
    resource = target_references_resource(
        country="uk",
        description=DESCRIPTION,
        authored=authored,
    )
    args.output.write_text(json.dumps(resource, indent=2) + "\n")
    args.membership_report.write_text(
        json.dumps(authored.membership_report, indent=2) + "\n"
    )
    print(json.dumps(authored.membership_report["status_counts"], sort_keys=True))
    print(
        f"active_reference_count={authored.membership_report['active_reference_count']}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument(
        "--source-fact-feed",
        help="Stable display name recorded in the generated membership report.",
    )
    parser.add_argument("--period", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membership-report", type=Path, required=True)
    return parser


def _filter_contract_by_geography_levels(
    contract: Mapping[str, Any],
    *,
    allowed_levels: frozenset[str],
) -> dict[str, Any]:
    filtered = dict(contract)
    filtered["targets"] = [
        target
        for target in contract.get("targets", ())
        if set(target.get("geography_levels") or ()) <= allowed_levels
    ]
    return filtered


def _validate_amount_entity_pins(contract: Mapping[str, Any]) -> None:
    missing = [
        str(target["target_id"])
        for target in contract.get("targets", ())
        if str(target.get("measurement", {}).get("concept", "")).endswith(".amount")
        and not str(target.get("ledger_selector", {}).get("entity_name", ""))
    ]
    if missing:
        raise ValueError(
            "National amount targets must pin ledger_selector.entity_name: "
            f"{missing!r}."
        )


def _registry_inverse(contract: Mapping[str, Any]) -> dict[str, list[str]]:
    inverse: dict[str, list[str]] = {}
    for ancestor, target_id in contract["registry_parity"]["mapped"].items():
        inverse.setdefault(str(target_id), []).append(str(ancestor))
    return {key: sorted(value) for key, value in inverse.items()}


def _geography_pins(contract: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        str(target["target_id"]): {
            "geography_level": "country",
            "geography_id": _geography_id_for_target(target),
        }
        for target in contract.get("targets", ())
    }


SCOTGOV_COUNCIL_TAX_STOCK_PREFIX = "scotgov.council_tax_stock."

# Target-id prefixes whose geography pins cannot come from the nation-substring
# rule below. Each entry is justified by the layer that fixes the geography:
# the Chronicle fact stamp, the contract binding, or the model variable's own
# country gate.
TARGET_PREFIX_GEOGRAPHY_PINS: tuple[tuple[str, str], ...] = (
    # Scottish Government statistics are Scotland-scoped and Chronicle stamps
    # them S92000003 (CTAXBASE chargeable dwellings, Scottish Budget
    # social-security lines); the substring rule sees no "scotland" in
    # "scotgov" or "scottish_child_payment" and would fall through to the UK
    # pin, which never matches a Scotland-stamped fact.
    ("scotgov.", "scotland"),
    # The SLC borrower-plan forecasts Chronicle carries are England-scoped
    # (facts stamped E92000001) and the contract bindings already filter
    # country == ENGLAND explicitly, so the GB default could never match.
    ("slc.borrowers.", "england"),
    # The SLC student-support publication is England-scoped (facts stamped
    # E92000001) and the bound model variables are England-gated by
    # construction (maintenance_loan_in_england_system,
    # parents_learning_allowance_eligible and adult_dependants_grant_eligible
    # all require country == ENGLAND). slc.repayments.* stays with the
    # substring rule: its england_* ids name their nation, and devolved_total
    # needs a per-nation redesign before it can activate.
    ("slc.support.", "england"),
    ("dfe.", "england"),
)
# DfT BUS05i rows name their area in the selector; the geography follows the
# declared area, never a prefix, so a London or UK row can never be stamped
# England if it is activated. Sub-national DfT areas carry DfT's own ids.
DFT_BUS_AREA_GEOGRAPHY_IDS = {
    "england": UK_GEOGRAPHY_IDS["england"],
    "london": "E12000007",
    "england_outside_london": "dft:england_outside_london",
    "uk": UK_GEOGRAPHY_IDS["uk"],
}


def _geography_id_for_target(target: Mapping[str, Any]) -> str:
    target_id = str(target["target_id"]).lower()
    if target_id.startswith("dft."):
        area = str(
            (target.get("ledger_selector") or {}).get("layout_groupby_value_id", "")
        )
        if area not in DFT_BUS_AREA_GEOGRAPHY_IDS:
            raise ValueError(
                f"{target_id}: DfT rows must declare layout_groupby_value_id in "
                f"{sorted(DFT_BUS_AREA_GEOGRAPHY_IDS)}, got {area!r}."
            )
        return DFT_BUS_AREA_GEOGRAPHY_IDS[area]
    for prefix, geography_key in TARGET_PREFIX_GEOGRAPHY_PINS:
        if target_id.startswith(prefix):
            return UK_GEOGRAPHY_IDS[geography_key]
    selector = target.get("ledger_selector") or {}
    concept = str(selector.get("source_concept", "")).lower()
    measure = str(selector.get("source_measure_id", "")).lower()
    haystack = " ".join((target_id, concept, measure))
    if "northern" in haystack or "domestic_rates" in haystack:
        return UK_GEOGRAPHY_IDS["northern_ireland"]
    if "scotland" in haystack:
        return UK_GEOGRAPHY_IDS["scotland"]
    if "wales" in haystack:
        return UK_GEOGRAPHY_IDS["wales"]
    if "england" in haystack or target_id.startswith("voa."):
        return UK_GEOGRAPHY_IDS["england"]
    if target_id.startswith("dwp.") or target_id.startswith("slc."):
        return UK_GEOGRAPHY_IDS["great_britain"]
    return UK_GEOGRAPHY_IDS["uk"]


def _sum_target_ids(contract: Mapping[str, Any]) -> frozenset[str]:
    target_ids: set[str] = set()
    for target in contract.get("targets", ()):
        selector = target["ledger_selector"]
        binding = target["bindings"]["policyengine"]
        if "value_expression" in binding:
            target_ids.add(str(target["target_id"]))
        if any(
            key != "dimensions" and isinstance(value, list)
            for key, value in selector.items()
        ):
            target_ids.add(str(target["target_id"]))
        dimension_values = selector.get("dimension_values")
        if isinstance(dimension_values, Mapping) and any(
            isinstance(value, list) for value in dimension_values.values()
        ):
            target_ids.add(str(target["target_id"]))
    return frozenset(target_ids)


def _value_operation_by_target_id(contract: Mapping[str, Any]) -> dict[str, str]:
    operations = {target_id: "sum" for target_id in _sum_target_ids(contract)}
    for target in contract.get("targets", ()):
        target_id = str(target["target_id"])
        declared = target.get("value_operation")
        if declared is not None:
            operations[target_id] = str(declared)
        if (
            target.get("family") == "dwp_universal_credit"
            and declared not in MONTHLY_WINDOW_OPERATIONS
        ):
            operations[target_id] = "calendar_year_average"
    return operations


def _selector_pins(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    pins: dict[str, dict[str, Any]] = {}
    for target in contract.get("targets", ()):
        selector = target["ledger_selector"]
        dimension_values = selector.get("dimension_values")
        if selector.get(
            "source_concept"
        ) == "ons.mid_year_population_estimate" and isinstance(
            dimension_values, Mapping
        ):
            pins[str(target["target_id"])] = {
                "dimensions": sorted(str(key) for key in dimension_values)
            }
    return pins


def _signed_exclusions(contract: Mapping[str, Any]) -> dict[str, str]:
    target_ids = {str(target["target_id"]) for target in contract.get("targets", ())}
    resource = json.loads(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(
            "packages/microcosm-build/src/microcosm/build/uk/"
            "target_reference_signed_exclusions.json"
        )
        .read_text()
    )
    exclusions = {
        str(entry["target_id"]): str(entry["rationale"])
        for entry in resource["exclusions"]
    }
    return {
        target_id: rationale
        for target_id, rationale in exclusions.items()
        if target_id in target_ids
    }


def _reference_metadata(contract: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for target in contract.get("targets", ()):
        measurement = target.get("measurement", {})
        metadata = {}
        if measurement.get("observation_basis") is not None:
            metadata["observation_basis"] = str(measurement["observation_basis"])
        binding = target.get("bindings", {}).get("policyengine", {})
        if binding.get("measurement_period") is not None:
            metadata["measurement_period"] = str(binding["measurement_period"])
        if binding.get("require_matching_fact_period"):
            metadata["source_period_policy"] = "exact_observation"
        if "source_months" in measurement:
            if target.get("family") != "dwp_universal_credit":
                raise ValueError("source_months is currently a UK UC-only declaration.")
            metadata.update(
                uc_source_month_metadata(
                    measurement["source_months"],
                    value_operation=str(
                        target.get("value_operation", "calendar_year_average")
                    ),
                )
            )
        if metadata:
            result[str(target["target_id"])] = metadata
    return result


def _fanout_name(
    target: Mapping[str, Any],
    fact: Mapping[str, Any],
    inverse_mapping: Mapping[str, list[str]],
) -> str | None:
    target_id = str(target["target_id"])
    value_id = str(fact.get("layout", {}).get("groupby_value_id") or "")
    preferred_tokens, fallback_tokens = _dimension_tokens(fact)
    candidates = inverse_mapping.get(target_id, ())
    for candidate in candidates:
        if value_id and value_id not in _GEOGRAPHY_VALUE_IDS and value_id in candidate:
            return candidate
        if any(token in candidate for token in preferred_tokens):
            return candidate
    for candidate in candidates:
        if any(token in candidate for token in fallback_tokens):
            return candidate
    if len(candidates) == 1:
        return None
    if candidates:
        return None
    safe_value = value_id.replace("/", "_") or "detail"
    return f"{target_id}.{safe_value}"


def _dimension_tokens(fact: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    preferred: list[str] = []
    fallback: list[str] = []
    dimensions = fact.get("dimensions") or {}
    if not isinstance(dimensions, Mapping):
        return preferred, fallback
    for value in dimensions.values():
        if isinstance(value, int):
            token = _int_token(value)
            preferred.append(f"income_band_{token}_to")
            fallback.append(f"_{token}")
        elif isinstance(value, str) and value.isdigit():
            token = _int_token(int(value))
            preferred.append(f"income_band_{token}_to")
            fallback.append(f"_{token}")
        elif isinstance(value, str):
            annual_band = _annual_uc_award_band_token(value)
            if annual_band:
                preferred.append(annual_band)
                fallback.append(annual_band.removeprefix("annual_payment_"))
    return preferred, fallback


def _int_token(value: int) -> str:
    return f"{value:,}".replace(",", "_")


def _annual_uc_award_band_token(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "no payment" or "or over" in normalized:
        return ""
    if " to " not in normalized:
        return ""
    numbers = [
        float(number.replace(",", ""))
        for number in re.findall(r"[0-9][0-9,]*(?:\.[0-9]+)?", value)
    ]
    if len(numbers) != 2:
        return ""
    lower = int((numbers[0] // 100) * 100 * 12)
    upper = int(numbers[1] * 12)
    return f"annual_payment_{_int_token(lower)}_to_{_int_token(upper)}"


def _add_uk_membership_accounting(
    report: dict[str, Any],
    references: tuple[dict[str, Any], ...],
) -> None:
    fanout_counts = Counter(
        reference["family"]
        for reference in references
        if reference["name"] != reference["metadata"]["contract_target_id"]
    )
    council_tax_count = sum(
        1
        for reference in references
        if reference["metadata"]["contract_target_id"].startswith(
            ("voa.council_tax_stock.", SCOTGOV_COUNCIL_TAX_STOCK_PREFIX)
        )
    )
    report["fanout_family_outcomes"] = [
        {
            "family": "hmrc_spi",
            "status": "active_with_signed_property_amount_exclusion",
            "active_reference_count": fanout_counts.get("hmrc_spi", 0),
            "signed_rationale": (
                "SPI income-band targets fan out by strict total-income-band "
                "dimension pins, except the HMRC property-income amount "
                "surface. Those 13 rows are signed out because Ledger carries "
                "the official SPI Table 3.7 net property-income amounts, "
                "while the incumbent target applies the populace-side x1.9 "
                "property-income undercount adjustment traced to "
                f"{_UK_DATA_REPO} PR #311 / issue #230 and HMRC "
                "Property Rental Income Statistics."
            ),
        },
        {
            "family": "dwp_universal_credit",
            "status": "active_with_unmapped_vintage_residue_skipped",
            "active_reference_count": fanout_counts.get("dwp_universal_credit", 0),
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
            "active_reference_count": council_tax_count,
            "signed_rationale": (
                "VOA (England and Wales) and Scottish Government CTAXBASE "
                "(Scotland) council-tax stock bands are declared as nine "
                "explicit target rows each, including total, and each resolves "
                "with its country-level geography and band pin."
            ),
        },
    ]
    report["signed_exclusion_rationales"] = [
        {
            "family": "hmrc_spi",
            "target_id": "hmrc.spi.property_income.amount_by_total_income_band",
            "status": "signed_excluded",
            "signed_rationale": report["targets"][
                "hmrc.spi.property_income.amount_by_total_income_band"
            ]["candidates"][0]["signed_rationale"],
        },
        {
            "family": "ons_population",
            "target_id": "ons.population.scotland_households_3plus_children",
            "status": "signed_excluded",
            "signed_rationale": report["targets"][
                "ons.population.scotland_households_3plus_children"
            ]["candidates"][0]["signed_rationale"],
        },
    ]
    report["multi_fact_rationales"] = []


_GEOGRAPHY_VALUE_IDS = frozenset(
    {
        "england",
        "great_britain",
        "northern_ireland",
        "scotland",
        "uk",
        "wales",
    }
)


if __name__ == "__main__":
    main()
