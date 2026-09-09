"""Replay UC on captured source tables with fixed weights and explicit model pins.

This development diagnostic never recalibrates, changes source artifacts, or
certifies a release. Person/benefit-unit output is private; the JSON is aggregate.
An optional list of variable source files isolates model fixes without upgrading
unrelated rules in the installed country model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask
from microcosm.frame import WeightKind

GB_REGIONS = {
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
}
BU_VARIABLES = (
    "universal_credit",
    "universal_credit_pre_benefit_cap",
    "uc_maximum_amount",
    "uc_income_reduction",
    "uc_earned_income",
    "uc_unearned_income",
    "uc_net_earned_income",
    "uc_assessable_capital",
    "uc_capital_income",
    "uc_standard_allowance",
    "uc_standard_allowance_claimant_type",
    "uc_work_allowance",
    "is_uc_work_allowance_eligible",
    "uc_child_element",
    "uc_disability_elements",
    "uc_housing_costs_element",
    "uc_childcare_element",
    "uc_carer_element",
    "uc_deductions",
    "benefit_cap_reduction",
    "is_benefit_cap_exempt",
    "is_uc_eligible",
    "family_type",
    "relation_type",
    "benunit_rent",
    "would_claim_uc",
    "uc_calibration_family_type",
    "uc_calibration_child_count",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def variable_overlay(paths: list[Path]):
    """Apply explicit variable bytes before loading their dataset inputs."""
    from policyengine_uk.utils.scenario import Scenario

    classes = []
    identities = []
    for index, path in enumerate(paths):
        spec = importlib.util.spec_from_file_location(f"uc_diagnostic_{index}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes.append(getattr(module, path.stem))
        identities.append(
            {"variable": path.stem, "path": str(path), "sha256": sha256(path)}
        )

    def apply(simulation):
        for variable in classes:
            simulation.tax_benefit_system.update_variable(variable)

    return Scenario(
        simulation_modifier=apply, applied_before_data_load=True
    ), identities


def summarize_group(table: pd.DataFrame, mask: np.ndarray) -> dict:
    rows = table.loc[mask]
    claim = rows["universal_credit"].gt(0)
    result = {
        "benefit_unit_rows": len(rows),
        "unique_source_benefit_units": int(rows["benunit_source_id"].nunique()),
        "unique_source_households": int(rows["household_source_id"].nunique()),
        "positive_uc_rows": int(claim.sum()),
    }
    for weight in ("prior_weight", "fixed_calibrated_weight"):
        w = rows[weight]
        source_mass = rows.loc[claim].groupby("household_source_id")[weight].sum()
        result[weight] = {
            "benefit_units": float(w.sum()),
            "positive_uc": float(w[claim].sum()),
            "uc_amount": float((w * rows["universal_credit"]).sum()),
            "source_household_ess_positive_uc": (
                float(source_mass.sum() ** 2 / (source_mass**2).sum())
                if (source_mass**2).sum()
                else 0.0
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--solve-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-year", type=int, default=2024)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--variable-overlay", type=Path, nargs="*", default=[])
    parser.add_argument("--supply-frs-claimants", action="store_true")
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument(
        "--allow-changed-source-tables",
        action="store_true",
        help="Explicitly compare a diagnosed source-stage intervention.",
    )
    args = parser.parse_args()
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = [args.capture_dir / f"{e}.pkl" for e in ("person", "benunit", "household")]
    inputs.extend(
        [args.solve_dir / "solve.npz", args.solve_dir / "uc_composition_private.pkl"]
    )
    identities = {str(path): sha256(path) for path in inputs}
    person, benunit, household = [pd.read_pickle(path) for path in inputs[:3]]
    solution = np.load(args.solve_dir / "solve.npz", allow_pickle=False)
    composition = pd.read_pickle(args.solve_dir / "uc_composition_private.pkl")
    pd.testing.assert_series_equal(
        benunit["benunit_id"], composition["benunit_id"], check_names=False
    )
    membership = person.groupby("person_benunit_id")["person_household_id"].first()
    hh_ids = benunit["benunit_id"].map(membership)
    np.testing.assert_array_equal(hh_ids, composition["household_id"])
    prior = pd.Series(solution["initial_weights"], index=household["household_id"])
    weights = pd.Series(solution["weights"], index=household["household_id"])
    np.testing.assert_allclose(
        hh_ids.map(weights), composition["final_weight"], rtol=0, atol=0
    )
    roles = frs_uc_claimant_mask(person, benunit)
    if args.supply_frs_claimants:
        person = person.copy()
        person["is_uc_claimant"] = roles
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=args.source_year,
        household_weights=solution["initial_weights"],
        weight_kind=WeightKind.IMPORTANCE,
    )
    overlay = []
    factory = None
    if args.variable_overlay:
        from policyengine_uk import Microsimulation

        scenario, overlay = variable_overlay(args.variable_overlay)

        def factory(**kwargs):
            return Microsimulation(**kwargs, scenario=scenario)

    elif args.supply_frs_claimants:
        raise ValueError(
            "Supplying new claimant input requires an explicit model overlay."
        )
    resolver = UKMeasureResolver(
        simulation_source=None,
        scratch_dir=args.output_dir / "engine",
        year=args.year,
        frame=frame,
        microsimulation_factory=factory,
    )
    print("Simulation initialized; computing UC components.", flush=True)
    if args.supply_frs_claimants:
        np.testing.assert_array_equal(
            resolver.compute("person", "is_uc_claimant")[0], roles
        )
    result = benunit.copy()
    result["household_id"] = hh_ids.to_numpy()
    for name in ("region", "household_source_id", "household_support_channel", "rent"):
        result[name] = hh_ids.map(household.set_index("household_id")[name]).to_numpy()
    result["prior_weight"] = hh_ids.map(prior).to_numpy()
    result["fixed_calibrated_weight"] = hh_ids.map(weights).to_numpy()
    missing = []
    for variable in BU_VARIABLES:
        if resolver.knows("benunit", variable):
            values, _ = resolver.compute("benunit", variable)
            result[variable] = values
        else:
            missing.append(variable)
    np.testing.assert_array_equal(result["would_claim_uc"], benunit["would_claim_uc"])
    baseline_exact = None
    if not overlay:
        baseline_exact = np.array_equal(result["universal_credit"], composition["uc"])
        if not baseline_exact:
            raise AssertionError(
                "Baseline UC awards do not reproduce the retained full-model capture."
            )
        np.testing.assert_array_equal(
            result["uc_calibration_family_type"], composition["family"]
        )
        np.testing.assert_array_equal(
            result["uc_calibration_child_count"], composition["child_count"]
        )
    person_metrics = person[["person_id", "person_benunit_id", "age"]].copy()
    person_metrics["frs_claimant"] = roles
    person_metrics["uc_individual_child_element"] = resolver.compute(
        "person", "uc_individual_child_element"
    )[0]
    for variable in (
        "is_WA_adult",
        "is_adult",
        "is_child_or_qualifying_young_person_for_universal_credit",
    ):
        values, _ = resolver.compute("person", variable)
        person_metrics[variable] = values.astype(bool)
    grouping = person_metrics.groupby("person_benunit_id")
    for column in ("frs_claimant", "is_adult", "is_WA_adult"):
        result[f"count_{column}"] = (
            benunit["benunit_id"].map(grouping[column].sum()).to_numpy()
        )
    qualifying = "is_child_or_qualifying_young_person_for_universal_credit"
    person_metrics["dependent_qyp"] = ~roles & person_metrics[qualifying]
    person_metrics["older_dependent"] = ~roles & person_metrics["age"].ge(18)
    for column in ("dependent_qyp", "older_dependent"):
        result[f"count_{column}"] = (
            benunit["benunit_id"]
            .map(person_metrics.groupby("person_benunit_id")[column].sum())
            .to_numpy()
        )
    result["raw_reported_uc"] = (
        benunit["benunit_id"]
        .map(person.groupby("person_benunit_id")["universal_credit_reported"].sum())
        .to_numpy()
    )
    capital_limit = resolver.simulation.tax_benefit_system.parameters(
        args.year
    ).gov.dwp.universal_credit.means_test.capital.limit
    result["blocked_capital"] = result["uc_assessable_capital"].gt(capital_limit)
    result["blocked_age"] = result["count_is_WA_adult"].eq(0)
    result["pre_takeup_positive"] = (
        result["uc_maximum_amount"] - result["uc_income_reduction"]
    ).gt(0)
    result["blocked_takeup"] = ~result["would_claim_uc"].astype(bool)
    result["income_exhausts_award"] = (
        result["is_uc_eligible"].astype(bool)
        & result["uc_maximum_amount"].gt(0)
        & ~result["pre_takeup_positive"]
    )
    result["loss_reason"] = np.select(
        [
            result["universal_credit"].gt(0),
            result["blocked_age"],
            result["blocked_capital"],
            result["uc_maximum_amount"].le(0),
            result["income_exhausts_award"],
            result["blocked_takeup"],
        ],
        ["positive_uc", "age", "capital", "no_maximum_elements", "income", "takeup"],
        default="cap_deductions_or_other",
    )
    result.to_pickle(args.output_dir / "benefit_units_private.pkl")
    person_metrics.to_pickle(args.output_dir / "people_private.pkl")
    groups = {}
    for scope, mask in (
        ("UK", np.ones(len(result), dtype=bool)),
        ("GB", result["region"].isin(GB_REGIONS).to_numpy()),
        ("NI", result["region"].eq("NORTHERN_IRELAND").to_numpy()),
    ):
        groups[scope] = summarize_group(result, mask)
        for family in (
            "SINGLE",
            "LONE_PARENT",
            "COUPLE_NO_CHILDREN",
            "COUPLE_WITH_CHILDREN",
        ):
            groups[f"{scope}/{family}"] = summarize_group(
                result,
                mask & result["uc_calibration_family_type"].eq(family).to_numpy(),
            )
    for channel in result["benunit_support_channel"].unique():
        for reason in result["loss_reason"].unique():
            mask = (
                result["benunit_support_channel"].eq(channel)
                & result["loss_reason"].eq(reason)
                & result["uc_calibration_family_type"].eq("LONE_PARENT")
            )
            groups[f"lone_parent/{channel}/{reason}"] = summarize_group(
                result, mask.to_numpy()
            )
    for name, mask in {
        "adult_count_disagrees_with_claimants": result["count_is_adult"].ne(
            result["count_frs_claimant"]
        ),
        "older_dependants": result["count_older_dependent"].gt(0),
        "childless_couple_tail": result["uc_calibration_family_type"].eq(
            "COUPLE_NO_CHILDREN"
        )
        & result["universal_credit"].ge(26400.12),
    }.items():
        groups[name] = summarize_group(result, mask.to_numpy())
    transitions = (
        result.groupby(["family_type", "uc_calibration_family_type"], observed=True)
        .agg(rows=("benunit_id", "size"), mass=("fixed_calibrated_weight", "sum"))
        .reset_index()
        .to_dict("records")
    )
    comparison = None
    if args.baseline_output:
        baseline_receipt = json.loads(
            (args.baseline_output / "aggregate.json").read_text()
        )
        assert baseline_receipt["source_year"] == args.source_year
        assert baseline_receipt["model_year"] == args.year
        if not args.allow_changed_source_tables:
            assert baseline_receipt["input_sha256"] == identities
        old = pd.read_pickle(args.baseline_output / "benefit_units_private.pkl")
        np.testing.assert_array_equal(old["benunit_id"], result["benunit_id"])
        np.testing.assert_array_equal(old["household_id"], result["household_id"])
        for weight in ("prior_weight", "fixed_calibrated_weight"):
            np.testing.assert_array_equal(old[weight], result[weight])
        comparison = {}
        for variable in (
            "universal_credit",
            "uc_standard_allowance",
            "uc_work_allowance",
            "uc_child_element",
            "uc_housing_costs_element",
            "uc_income_reduction",
        ):
            delta = result[variable] - old[variable]
            comparison[variable] = {
                "increased_rows": int(delta.gt(0.01).sum()),
                "decreased_rows": int(delta.lt(-0.01).sum()),
                "fixed_weight_delta": float(
                    (delta * result["fixed_calibrated_weight"]).sum()
                ),
                "prior_weight_delta": float((delta * result["prior_weight"]).sum()),
            }
        comparison["uc_zero_to_positive_rows"] = int(
            (old["universal_credit"].le(0) & result["universal_credit"].gt(0)).sum()
        )
        comparison["uc_positive_to_zero_rows"] = int(
            (old["universal_credit"].gt(0) & result["universal_credit"].le(0)).sum()
        )
    receipt = {
        "purpose": "Development diagnostic; fixed inherited weights; no recalibration or certified release",
        "source_year": args.source_year,
        "model_year": args.year,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("policyengine-uk", "policyengine-core", "numpy", "pandas")
        },
        "input_sha256": identities,
        "model_overlay": overlay,
        "diagnostic_tool_sha256": sha256(Path(__file__)),
        "loss_reason_semantics": "Ordered first-failing-condition partition, not independent causal attribution",
        "changed_source_tables_allowed": args.allow_changed_source_tables,
        "explicit_frs_claimant_input": args.supply_frs_claimants,
        "baseline_awards_exact": baseline_exact,
        "missing_variables": missing,
        "groups": groups,
        "family_transitions": transitions,
        "comparison": comparison,
    }
    if identities != {str(path): sha256(path) for path in inputs}:
        raise AssertionError("Diagnostic changed an input artifact.")
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "baseline_awards_exact": baseline_exact,
                "UK": groups["UK"],
                "GB": groups["GB"],
                "comparison": comparison,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
