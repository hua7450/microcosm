"""The UK FRS spine as a cell-ownership graph.

The packaged source manifest is the single stage roster.  Declaration order
is intentionally irrelevant after construction: every stage conservatively
declares the complete live slice it can observe, and ``compile_graph`` derives
the same chain from those ownership edges.  The legacy driver's second,
hand-maintained ordering is therefore unnecessary.

Declared rewrites use :class:`microcosm.graph.Owned` cells on a population
version with a base.  A keep-all population boundary opens such a version
where needed.  Structural nodes still cannot own cells, so an EXPAND node is
followed by an ordinary claim node for the cells it materialized.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from microcosm.graph import (
    Graph,
    KernelRegistry,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
)

from ..country_spec import CountrySpec, load_country_spec
from .national_sampling import UK_SAMPLE_SEED_DEFAULT

__all__ = [
    "UK_SPINE_EXCLUSIONS",
    "UK_SPINE_STRUCTURAL_STAGES",
    "uk_registry",
    "uk_spine_graph",
]


UK_SPINE_EXCLUSIONS = frozenset(
    {
        # These are the certified-candidate/H5 alternatives to the raw-FRS
        # spine stages named below, not additional steps in this pipeline.
        "frs_hmrc_retained_leaves",
        "hmrc_spi_income",
    }
)

UK_SPINE_STRUCTURAL_STAGES = frozenset(
    {"spi_support_channel", "cgt_incidence_clone", "cgt_band_donors"}
)

# The executor's mass ledger is weighted *person* mass per stratum
# (``Frame.stratum_mass``: household weights broadcast through membership), so
# ``conserve`` is satisfiable only by an expansion that keeps household
# composition fixed.  CGT cloning does (a clone is its source household at
# half weight).  The SPI support channel does not: it stacks synthetic
# households whose person counts differ from the FRS households whose mass
# they take over, so household mass is conserved exactly (the stage's
# ``allocate_zero_weight_prior_mass`` declares ``conservation: exact_total``)
# while person mass moves with the composition change.  On the FRS 2024-25
# spine that is 68.25m -> 65.44m persons, which ``conserve`` rejects at the
# node.  The node therefore *declares* its mass change: the kernel states the
# person-mass ledger the executor verifies and asserts the household-mass
# invariant itself (``UKExpandStageKernel``).
_STRUCTURAL_MASS = {
    "spi_support_channel": "declared",
    "cgt_incidence_clone": "conserve",
    "cgt_band_donors": "free",
}

_STRUCTURAL_WEIGHT_KIND = {
    "spi_support_channel": "importance",
    "cgt_incidence_clone": "importance",
    "cgt_band_donors": "importance",
}

# ``hmrc_spi_income_spine`` has an intentionally conservative open input
# surface.  Opening a version before the following UC rewrite prevents that
# earlier reader from resolving its incumbent UC cells to the later owner and
# forming a declaration cycle.
_READER_ISOLATION_BOUNDARIES = frozenset(
    {
        "uc_capital_coherence",
        # ``frs_education_grant_split`` rewrites the root cell
        # ``education_grants`` that the open-surface ``frs_legacy_proxies``
        # reader already bound to.  In the root version that rewrite opened a
        # boundary by the same-version owner rule; now that ``age_tail`` opens
        # its own version right after the root, the root cells are inherited
        # rather than owned there, so the isolation must be declared.
        "frs_education_grant_split",
    }
)
# ``age_tail`` needs no isolation entry: it is declared immediately after the
# root stage, and the root version already owns ``age``, so the same-version
# owner rule opens its boundary.  Every later age reader then binds to the
# disaggregated cell — the #785 contract — instead of the top-coded root.

_SPLIT_STAGE_SOURCES: Mapping[str, tuple[str, ...]] = {
    "frs_spine": ("frs",),
    "frs_employment": ("frs",),
    "frs_council_tax": ("frs",),
    "frs_education": ("frs",),
    "frs_legacy_proxies": ("frs",),
    "was_wealth": ("was",),
    "lcfs_consumption": ("lcfs_household", "lcfs_person", "was"),
    "etb_vat": ("etb",),
    "etb_services": ("etb",),
    "frs_hmrc_spine_leaves": ("frs",),
    "hmrc_spi_income_spine": ("spi", "hmrc_income"),
    "hmrc_cgt_gains_spine": ("hmrc_cgt",),
}

_SPLIT_SOURCE_DESCRIPTIONS = {
    "frs": "Pinned local FRS table directory.",
    "was": "Pinned local WAS household donor table.",
    "lcfs_household": "Pinned local LCFS household donor table.",
    "lcfs_person": "Pinned local LCFS person donor table.",
    "etb": "Pinned local ETB household donor table.",
    "spi": "Pinned local SPI donor table.",
    "hmrc_income": "Pinned local HMRC income facts workbook.",
    "hmrc_cgt": "Pinned local HMRC capital-gains facts workbook.",
}

# ``None`` means the implementation genuinely has an open formula/model
# surface and therefore receives every live cell.  Finite sets are the direct
# Frame reads inventoried from the existing transform.  Structural ids,
# memberships, weights, and strata are executor-carried context rather than
# ordinary owned cells.
_STAGE_CONSUMES: Mapping[str, frozenset[tuple[str, str]] | None] = {
    "frs_employment": frozenset(),
    "frs_council_tax": frozenset(),
    "frs_disability": frozenset(
        ("person", column)
        for column in (
            "attendance_allowance_reported",
            "dla_sc_reported",
            "dla_m_reported",
            "pip_m_reported",
            "pip_dl_reported",
            "sda_reported",
            "incapacity_benefit_reported",
            "iidb_reported",
            "afcs_reported",
            "esa_contrib_reported",
            "esa_income_reported",
        )
    ),
    "frs_education": frozenset(
        ("person", column)
        for column in (
            "age",
            "universal_credit_reported",
            "jsa_contrib_reported",
            "jsa_income_reported",
            "esa_contrib_reported",
            "esa_income_reported",
        )
    ),
    "frs_legacy_proxies": None,
    "frs_education_grant_split": None,
    "frs_take_up": frozenset(
        ("person", column)
        for column in (
            "child_benefit_reported",
            "pension_credit_reported",
            "universal_credit_reported",
        )
    ),
    "frs_person_draws": frozenset({("person", "age")}),
    "frs_household_draws": frozenset(),
    "frs_brma": None,
    "was_wealth": None,
    "regional_property_uprating": frozenset(
        {
            ("household", "region"),
            ("household", "main_residence_value"),
            ("household", "property_wealth"),
        }
    ),
    "lcfs_consumption": None,
    "etb_vat": None,
    "etb_services": None,
    "frs_hmrc_spine_leaves": frozenset({("person", "employee_pension_contributions")}),
    "spi_support_channel": None,
    "hmrc_spi_income_spine": None,
    # Runs one temporary engine materialization over the whole frame for its
    # award screen, so its input surface is genuinely open.
    "uc_reporter_redraw": None,
    "uc_capital_coherence": frozenset(
        {
            ("person", "person_support_channel"),
            ("person", "universal_credit_reported"),
            ("benunit", "benunit_support_channel"),
            ("benunit", "dependent_children"),
            ("benunit", "frs_benunit_capital"),
            ("benunit", "is_married"),
            ("benunit", "would_claim_uc"),
        }
    ),
    "uc_deduction_attributes": frozenset({("household", "region")}),
    "cgt_incidence_clone": None,
    "cgt_band_donors": None,
    "hmrc_cgt_gains_spine": frozenset(
        ("person", column)
        for column in (
            "capital_gains",
            "employment_income",
            "self_employment_income",
            "savings_interest_income",
            "dividend_income",
            "miscellaneous_income",
            "private_pension_income",
            "property_income",
            "state_pension_reported",
            "tax_free_savings_income",
        )
    ),
    "salary_sacrifice": None,
    "student_loans": frozenset(
        {
            ("person", "age"),
            ("person", "student_loans"),
            ("person", "student_loan_repayments"),
            ("person", "current_education"),
            ("person", "highest_education"),
            ("household", "region"),
        }
    ),
    # Keyed on the structural person_id before any clone provenance exists;
    # the transform refuses a frame that already carries person_source_id.
    "age_tail": frozenset(
        {
            ("person", "age"),
            ("person", "gender"),
        }
    ),
}

_CONTEXT_CARRIERS = frozenset(
    {
        ("person", "age"),
        ("benunit", "frs_benunit_capital"),
        ("household", "region"),
    }
)


@dataclass(frozen=True)
class _Cell:
    entity: str
    column: str
    dtype: str

    @property
    def coordinate(self) -> tuple[str, str]:
        return self.entity, self.column

    def owned(self, *, rewrite: bool = False) -> Owned:
        return Owned(self.entity, self.column, self.dtype, rewrite=rewrite)


_ROOT_PERSON_STRING = {"gender", "marital_status"}
_ROOT_PERSON_BOOL = {
    "is_household_head",
    "is_benunit_head",
    "is_parent",
    "is_uc_claimant",
}
_ROOT_PERSON_INT = {"age"}
_ROOT_PERSON_FLOAT: set[str] = set()
_ROOT_BENUNIT_TYPES = {
    "frs_benunit_capital": "float64",
    "is_married": "bool",
    "dependent_children": "int64",
}
_ROOT_HOUSEHOLD_STRING = {
    "region",
    "tenure_type",
    "accommodation_type",
    "council_tax_band",
}
_ROOT_HOUSEHOLD_INT = {"num_bedrooms", "council_tax_single_adult_raw"}
_STRUCTURAL_COLUMNS = {
    "person_id",
    "person_benunit_id",
    "person_household_id",
    "benunit_id",
    "household_id",
}


def _root_cells(stage_outputs: Iterable[str]) -> tuple[_Cell, ...]:
    cells: list[_Cell] = []
    for column in stage_outputs:
        if column in _STRUCTURAL_COLUMNS:
            continue
        if column in _ROOT_BENUNIT_TYPES:
            cells.append(_Cell("benunit", column, _ROOT_BENUNIT_TYPES[column]))
        elif column in _ROOT_HOUSEHOLD_STRING:
            cells.append(_Cell("household", column, "string"))
        elif column in _ROOT_HOUSEHOLD_INT:
            cells.append(_Cell("household", column, "int64"))
        elif column in _ROOT_PERSON_STRING:
            cells.append(_Cell("person", column, "string"))
        elif column in _ROOT_PERSON_BOOL:
            cells.append(_Cell("person", column, "bool"))
        elif column in _ROOT_PERSON_INT:
            # frs_spine casts age to int64 after checking the raw codes are
            # integral, and age_tail keeps an integer input integer (its
            # bands are integral), so the legacy runner and this declaration
            # agree without relying on the CREATE-time cast (#845).  Rewrites
            # cannot change their base's dtype.
            cells.append(_Cell("person", column, "int64"))
        elif column in _ROOT_PERSON_FLOAT:
            cells.append(_Cell("person", column, "float64"))
        elif column.startswith("household_"):
            # No root data column currently takes this spelling; keep an
            # explicit failure if the manifest grows rather than guessing.
            raise ValueError(f"Unknown root household cell {column!r}.")
        elif column in {
            "council_tax_reported",
            "council_tax_rebate",
            "water_and_sewerage_charges",
            "domestic_rates",
            "rent",
            "subrent",
            "mortgage_interest_repayment",
            "mortgage_capital_repayment",
            "structural_insurance_payments",
            "housing_service_charges",
            "external_child_payments",
        }:
            cells.append(_Cell("household", column, "float64"))
        else:
            cells.append(_Cell("person", column, "float64"))
    return tuple(cells)


def _cells(
    entity: str,
    columns: Iterable[str],
    dtype: str = "float64",
) -> tuple[_Cell, ...]:
    return tuple(_Cell(entity, column, dtype) for column in columns)


_STAGE_CELLS: Mapping[str, tuple[_Cell, ...]] = {
    "frs_employment": (
        _Cell("person", "employment_status", "string"),
        _Cell("person", "employment_sector", "string"),
        _Cell("person", "sic_industry_division", "int64"),
    ),
    "frs_council_tax": (_Cell("household", "council_tax", "float64"),),
    "frs_disability": (
        *_cells(
            "person",
            (
                "aa_category",
                "dla_sc_category",
                "dla_m_category",
                "pip_m_category",
                "pip_dl_category",
            ),
            "string",
        ),
        *_cells(
            "person",
            (
                "is_disabled_for_benefits",
                "is_enhanced_disabled_for_benefits",
                "is_severely_disabled_for_benefits",
            ),
            "bool",
        ),
    ),
    "frs_education": (
        _Cell("person", "current_education", "string"),
        _Cell("person", "highest_education", "string"),
        _Cell("person", "is_in_non_advanced_education", "bool"),
        _Cell("person", "is_in_approved_training", "bool"),
        _Cell(
            "person",
            "age_started_or_accepted_current_education_or_training",
            "int64",
        ),
        _Cell(
            "person",
            "is_before_universal_credit_qualifying_young_person_terminal_date",
            "bool",
        ),
        _Cell("person", "adult_ema", "float64"),
        _Cell("person", "child_ema", "float64"),
        _Cell("person", "receives_benefits_in_own_right", "bool"),
    ),
    "frs_legacy_proxies": _cells(
        "person",
        (
            "legacy_jobseeker_proxy",
            "esa_health_condition_proxy",
            "esa_support_group_proxy",
        ),
        "bool",
    ),
    "frs_education_grant_split": _cells(
        "person",
        ("disabled_students_allowance_eligible_expenses", "education_grants"),
    ),
    "frs_take_up": (
        *_cells(
            "benunit",
            (
                "would_claim_child_benefit",
                "child_benefit_opts_out",
                "would_claim_pc",
                "would_claim_uc",
                "would_claim_tfc",
                "would_claim_extended_childcare",
                "would_claim_universal_childcare",
                "would_claim_targeted_childcare",
            ),
            "bool",
        ),
        _Cell("benunit", "maximum_extended_childcare_hours_usage", "float64"),
    ),
    "frs_person_draws": (
        _Cell("person", "would_claim_marriage_allowance", "bool"),
        _Cell("person", "would_claim_scp", "bool"),
        _Cell("person", "attends_private_school_random_draw", "float64"),
        _Cell("person", "tax_free_childcare_spend_routed_share", "float64"),
    ),
    "frs_household_draws": _cells(
        "household",
        (
            "household_owns_tv",
            "would_evade_tv_licence_fee",
            "main_residential_property_purchased_is_first_home",
            "property_purchased",
        ),
        "bool",
    ),
    "frs_brma": (_Cell("household", "brma", "string"),),
    "was_wealth": (
        *_cells(
            "household",
            (
                "owned_land",
                "property_wealth",
                "corporate_wealth",
                "private_pension_wealth",
                "gross_financial_wealth",
                "net_financial_wealth",
                "main_residence_value",
                "other_residential_property_value",
                "non_residential_property_value",
                "savings",
            ),
        ),
        _Cell("household", "num_vehicles", "int64"),
        *_cells("household", ("cash_isa", "stocks_and_shares_isa")),
        *_cells("household", ("mortgage_debt", "consumer_debt")),
        _Cell("person", "student_loan_balance", "float64"),
    ),
    "regional_property_uprating": _cells(
        "household", ("main_residence_value", "property_wealth")
    ),
    "lcfs_consumption": (
        *_cells(
            "household",
            (
                "food_and_non_alcoholic_beverages_consumption",
                "alcohol_and_tobacco_consumption",
                "clothing_and_footwear_consumption",
                "housing_water_and_electricity_consumption",
                "household_furnishings_consumption",
                "health_consumption",
                "transport_consumption",
                "communication_consumption",
                "recreation_consumption",
                "education_consumption",
                "restaurants_and_hotels_consumption",
                "miscellaneous_consumption",
                "petrol_spending",
                "diesel_spending",
                "bus_fare_spending",
                "domestic_energy_consumption",
                "electricity_consumption",
                "gas_consumption",
            ),
        ),
        _Cell("household", "has_fuel_consumption", "bool"),
    ),
    "etb_vat": (_Cell("household", "full_rate_vat_expenditure_rate", "float64"),),
    "etb_services": (
        *_cells(
            "household",
            (
                "dfe_education_spending",
                "rail_subsidy_spending",
                "bus_subsidy_spending",
                "rail_usage",
            ),
        ),
        *_cells(
            "person",
            (
                "a_and_e_visits",
                "admitted_patient_visits",
                "outpatient_visits",
                "nhs_a_and_e_spending",
                "nhs_admitted_patient_spending",
                "nhs_outpatient_spending",
            ),
        ),
    ),
    "frs_hmrc_spine_leaves": _cells(
        "person",
        (
            "hmrc_spi_pay",
            "hmrc_spi_unemployment_benefit_income",
            "hmrc_spi_incapacity_benefit_income",
            "ossben_identifiable_subset",
            "srp_regular_code5",
            "employer_pension_contributions",
        ),
    ),
    "spi_support_channel": (
        _Cell("person", "person_source_id", "int64"),
        _Cell("person", "person_support_channel", "string"),
        _Cell("person", "person_support_clone_index", "int64"),
        _Cell("benunit", "benunit_source_id", "int64"),
        _Cell("benunit", "benunit_support_channel", "string"),
        _Cell("benunit", "benunit_support_clone_index", "int64"),
        _Cell("household", "source_household_id", "int64"),
        _Cell("household", "source_year", "int64"),
        _Cell("household", "source_household_key", "string"),
        _Cell("household", "household_source_id", "int64"),
        _Cell("household", "household_support_channel", "string"),
        _Cell("household", "household_support_clone_index", "int64"),
        _Cell("household", "household_is_spi_synthetic", "bool"),
    ),
    "hmrc_spi_income_spine": (),  # populated below from typed groups
    "uc_reporter_redraw": (_Cell("person", "universal_credit_reported", "float64"),),
    "uc_capital_coherence": (
        _Cell("benunit", "uc_reported_capital", "float64"),
        _Cell("benunit", "frs_benunit_capital", "float64"),
        _Cell("benunit", "would_claim_uc", "bool"),
    ),
    "uc_deduction_attributes": (
        _Cell("benunit", "uc_deduction_random_draw", "float64"),
        _Cell("benunit", "uc_deduction_type_random_draw", "float64"),
        _Cell("benunit", "uc_latent_deduction_rate", "float64"),
        _Cell("benunit", "uc_deduction_combination", "string"),
    ),
    "cgt_incidence_clone": (
        _Cell("household", "household_is_capital_gains_clone", "bool"),
        _Cell("person", "capital_gains", "float64"),
    ),
    "cgt_band_donors": (
        _Cell("household", "household_is_cgt_band_donor", "bool"),
        _Cell("person", "capital_gains", "float64"),
    ),
    "hmrc_cgt_gains_spine": (_Cell("person", "capital_gains", "float64"),),
    "salary_sacrifice": _cells(
        "person",
        (
            "pension_contributions_via_salary_sacrifice",
            "employee_pension_contributions",
        ),
    ),
    "student_loans": (_Cell("person", "student_loan_plan", "string"),),
    "age_tail": (_Cell("person", "age", "int64"),),
}

_HMRC_SPI_FLOAT_COLUMNS = (
    "charitable_investment_gifts",
    "gift_aid",
    "other_investment_income",
    "hmrc_spi_employment_benefits",
    "hmrc_spi_employment_expenses",
    "hmrc_spi_other_social_security_income",
    "hmrc_spi_taxable_termination_pay",
    "hmrc_spi_miscellaneous_employment_income",
    "hmrc_spi_other_income",
    "hmrc_spi_state_pension_income",
    "hmrc_spi_employed_income",
    "hmrc_spi_total_earned_income",
    "hmrc_spi_total_investment_income",
    "hmrc_spi_assessable_income",
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
    "employee_pension_contributions",
    "employer_pension_contributions",
    "personal_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    "tax_free_savings_income",
    "universal_credit_reported",
    "pension_credit_reported",
    "child_benefit_reported",
    "housing_benefit_reported",
    "income_support_reported",
    "working_tax_credit_reported",
    "child_tax_credit_reported",
    "attendance_allowance_reported",
    "state_pension_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "pip_m_reported",
    "pip_dl_reported",
    "sda_reported",
    "carers_allowance_reported",
    "iidb_reported",
    "afcs_reported",
    "bsp_reported",
    "winter_fuel_allowance_reported",
    "council_tax_benefit_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
    "hmrc_spi_pay",
    "hmrc_spi_unemployment_benefit_income",
    "hmrc_spi_incapacity_benefit_income",
)
_HMRC_SPI_HIDDEN_STRING = (
    "aa_category",
    "dla_sc_category",
    "dla_m_category",
    "pip_m_category",
    "pip_dl_category",
)
_HMRC_SPI_HIDDEN_BOOL = (
    "is_disabled_for_benefits",
    "is_enhanced_disabled_for_benefits",
    "is_severely_disabled_for_benefits",
)
_STAGE_CELLS = {
    **_STAGE_CELLS,
    "hmrc_spi_income_spine": (
        *_cells("person", _HMRC_SPI_FLOAT_COLUMNS),
        *_cells("person", _HMRC_SPI_HIDDEN_STRING, "string"),
        *_cells("person", _HMRC_SPI_HIDDEN_BOOL, "bool"),
    ),
}


def _slices(
    live: Mapping[tuple[str, str], _Cell],
    *,
    exclude: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[Slice, ...]:
    by_entity: dict[str, list[str]] = {}
    for coordinate, cell in live.items():
        if coordinate not in exclude:
            by_entity.setdefault(cell.entity, []).append(cell.column)
    return tuple(
        Slice(entity, tuple(columns))
        for entity, columns in sorted(by_entity.items())
        if columns
    )


def _stage_slices(
    live: Mapping[tuple[str, str], _Cell],
    stage: str,
    *,
    exclude: frozenset[tuple[str, str]] = frozenset(),
    shape_anchor: tuple[str, str] | None = None,
) -> tuple[Slice, ...]:
    consumes = _STAGE_CONSUMES[stage]
    if consumes is None:
        return _slices(live, exclude=exclude)
    requested = consumes | _CONTEXT_CARRIERS
    # Frame column order is covered by uk_frame_content_identity.  Until the
    # frozen graph declaration has a shape-only dependency, one cell from the
    # immediately preceding stage anchors that schema-order dependency.  It is
    # intentionally additional to the transform's value reads above.
    if shape_anchor is not None:
        requested = requested | {shape_anchor}
    requested = requested - exclude
    missing = requested - live.keys()
    if missing:
        raise ValueError(
            f"UK graph stage {stage!r} consumes unavailable cells: {sorted(missing)}."
        )
    selected = {coordinate: live[coordinate] for coordinate in sorted(requested)}
    return _slices(selected)


def _deduplicate(cells: Iterable[_Cell]) -> tuple[_Cell, ...]:
    resolved: dict[tuple[str, str], _Cell] = {}
    for cell in cells:
        incumbent = resolved.get(cell.coordinate)
        if incumbent is not None and incumbent.dtype != cell.dtype:
            raise ValueError(
                f"Conflicting declarations for {cell.entity}.{cell.column}: "
                f"{incumbent.dtype!r} and {cell.dtype!r}."
            )
        resolved[cell.coordinate] = cell
    return tuple(resolved.values())


def _manifest_stages(spec: CountrySpec) -> tuple[object, ...]:
    if spec.sources is None:
        raise ValueError("The UK graph requires a source-stage manifest.")
    selected = tuple(
        stage for stage in spec.sources.stages if stage.stage not in UK_SPINE_EXCLUSIONS
    )
    if not selected or selected[0].stage != "frs_spine":
        raise ValueError("The UK FRS spine manifest must begin with 'frs_spine'.")
    unknown = [stage.stage for stage in selected[1:] if stage.stage not in _STAGE_CELLS]
    if unknown:
        raise ValueError(f"The UK graph has no typed cell inventory for {unknown}.")
    return selected


def _declared_stage_cells(manifest_stage: object) -> tuple[_Cell, ...]:
    stage_name = manifest_stage.stage
    declared = set((*manifest_stage.outputs, *manifest_stage.rewrites))
    if stage_name == "hmrc_spi_income_spine" and manifest_stage.rewrites:
        # The production transform refreshes disability categories/flags on
        # SPI rows in addition to the packaged declared rewrites.  Synthetic
        # reduced manifests which omit the rewrite surface do not perform it.
        declared.update((*_HMRC_SPI_HIDDEN_STRING, *_HMRC_SPI_HIDDEN_BOOL))
    cells = tuple(cell for cell in _STAGE_CELLS[stage_name] if cell.column in declared)
    unresolved = declared - {cell.column for cell in cells}
    if unresolved:
        raise ValueError(
            f"The UK graph has no typed cells for {stage_name}: {sorted(unresolved)}."
        )
    return _deduplicate(cells)


def _stage_contract_sha256(stage: object, spec: CountrySpec) -> str:
    """Bind a node to its manifest operations, seeds, artifacts, and resources."""

    resource_pins = {
        str(artifact["resource"]): str(spec.resource_hashes[artifact["resource"]])
        for artifact in stage.artifacts
        if "resource" in artifact
    }
    payload = {
        "stage": stage.stage,
        "survey": stage.survey,
        "source": stage.source,
        "grain": stage.grain,
        "artifacts": [dict(artifact) for artifact in stage.artifacts],
        "operations": [
            {"kind": operation.kind, **dict(operation.parameters)}
            for operation in stage.operations
        ],
        "outputs": list(stage.outputs),
        "rewrites": list(stage.rewrites),
        "nonnegative_outputs": list(stage.nonnegative_outputs),
        "resource_pins": resource_pins,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_names(stage_name: str, source_mode: str) -> tuple[str, ...]:
    if source_mode == "bundle":
        return ("frs",)
    return _SPLIT_STAGE_SOURCES.get(stage_name, ())


def _source_refs(source_mode: str) -> tuple[SourceRef, ...]:
    if source_mode == "bundle":
        return (
            SourceRef(
                "frs",
                "csv-tables",
                "Content-bound UK FRS and donor fixture/source bundle.",
            ),
        )
    return tuple(
        SourceRef(name, "csv-tables", description)
        for name, description in _SPLIT_SOURCE_DESCRIPTIONS.items()
    )


def uk_spine_graph(
    spec: CountrySpec | None = None,
    *,
    source_mode: str = "bundle",
    sample_fraction: float = 1.0,
    sample_seed: int = UK_SAMPLE_SEED_DEFAULT,
) -> Graph:
    """Return the source-bound graph for the packaged UK FRS spine."""

    from .frs_spine import UKFRSSpineStageTransform

    if source_mode not in {"bundle", "split"}:
        raise ValueError("UK graph source_mode must be 'bundle' or 'split'.")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("UK graph sample_fraction must be in (0, 1].")
    if sample_seed < 0:
        raise ValueError("UK graph sample_seed must be non-negative.")
    resolved = load_country_spec("uk") if spec is None else spec
    stages = _manifest_stages(resolved)
    # The root transform loads the complete national-frame seed schema even
    # when a reduced hermetic manifest names only the output under test.
    # CREATE must declare every loaded cell, never merely the StagePlan's
    # historically incomplete ownership surface.
    root_cells = _root_cells(UKFRSSpineStageTransform.output_columns())
    live: dict[tuple[str, str], _Cell] = {cell.coordinate: cell for cell in root_cells}

    nodes: list[Node] = [
        Node(
            id="create_uk_frs",
            kernel="uk.create@1",
            outputs=tuple(cell.owned() for cell in root_cells),
            structural=StructuralDelta.CREATE,
            sources=_source_names("frs_spine", source_mode),
            params={
                "time_period": "2024",
                "stage_contract_sha256": _stage_contract_sha256(stages[0], resolved),
                "sample_fraction": float(sample_fraction),
                "sample_seed": int(sample_seed),
            },
            description="Load the source-bound UK FRS root population.",
        )
    ]
    current_population = "create_uk_frs"

    # CREATE owns the loaded cells.  The named root stage claims them after a
    # population boundary, keeping the stage roster visible without executing
    # the FRS assembly twice.
    root_boundary = "frs_spine.boundary"
    nodes.append(
        Node(
            id=root_boundary,
            kernel="uk.identity@1",
            inputs=_slices(live),
            structural=StructuralDelta.FILTER,
            base=current_population,
            description="Ownership boundary for the source-assembling root stage.",
        )
    )
    nodes.append(
        Node(
            id="frs_spine",
            kernel="uk.claim@1",
            outputs=tuple(cell.owned(rewrite=True) for cell in root_cells),
            population=root_boundary,
            description="Claim the cells assembled by the UK FRS root transform.",
        )
    )
    current_population = root_boundary
    version_owned = set(live)
    shape_anchor = root_cells[-1].coordinate

    for manifest_stage in stages[1:]:
        stage_name = manifest_stage.stage
        cells = _declared_stage_cells(manifest_stage)
        coordinates = frozenset(cell.coordinate for cell in cells)
        incumbent = frozenset(coordinates & live.keys())

        if stage_name in UK_SPINE_STRUCTURAL_STAGES:
            nodes.append(
                Node(
                    id=stage_name,
                    kernel=f"uk.stage.expand.{stage_name}@1",
                    inputs=_stage_slices(
                        live,
                        stage_name,
                        shape_anchor=shape_anchor,
                    ),
                    params={
                        "stage": stage_name,
                        "time_period": "2024",
                        "expand_cells": tuple(
                            (cell.entity, cell.column, cell.dtype) for cell in cells
                        ),
                        "expand_weight_entity": "household",
                        "expand_weight_kind": _STRUCTURAL_WEIGHT_KIND[stage_name],
                        "stage_contract_sha256": _stage_contract_sha256(
                            manifest_stage, resolved
                        ),
                    },
                    structural=StructuralDelta.EXPAND,
                    base=current_population,
                    sources=_source_names(stage_name, source_mode),
                    mass=_STRUCTURAL_MASS[stage_name],
                    description=f"Run structural UK stage {stage_name}.",
                )
            )
            nodes.append(
                Node(
                    id=f"{stage_name}.owned",
                    kernel="uk.claim@1",
                    outputs=tuple(
                        cell.owned(rewrite=cell.coordinate in incumbent)
                        for cell in cells
                    ),
                    population=stage_name,
                    params={
                        # Amendment 8 projects declared rewrites directly.
                        # These are only the new cells installed physically by
                        # the preceding EXPAND node; structural declarations
                        # still cannot own them (the one remaining interface
                        # gap from lane F).
                        "materialized_expand_outputs": tuple(
                            f"{cell.entity}.{cell.column}"
                            for cell in cells
                            if cell.coordinate not in incumbent
                        )
                    },
                    description=f"Own the cells materialized by {stage_name}.",
                )
            )
            current_population = stage_name
            version_owned = set(coordinates)
        else:
            # A rewrite needs a version with a base, but it needs a fresh
            # boundary only when this same version already has an owner for
            # that coordinate.  Structural versions already provide a base;
            # do not retain identity FILTERs that existed only to transport
            # lane F's params-carried incumbents.
            if (
                coordinates & version_owned
                or stage_name in _READER_ISOLATION_BOUNDARIES
            ):
                boundary = f"{stage_name}.boundary"
                nodes.append(
                    Node(
                        id=boundary,
                        kernel="uk.identity@1",
                        inputs=_slices(live),
                        structural=StructuralDelta.FILTER,
                        base=current_population,
                        description=f"Ownership boundary before {stage_name} rewrites.",
                    )
                )
                current_population = boundary
                version_owned = set()
            nodes.append(
                Node(
                    id=stage_name,
                    kernel=f"uk.stage.{stage_name}@1",
                    inputs=_stage_slices(
                        live,
                        stage_name,
                        exclude=coordinates,
                        shape_anchor=shape_anchor,
                    ),
                    outputs=tuple(
                        cell.owned(rewrite=cell.coordinate in incumbent)
                        for cell in cells
                    ),
                    population=current_population,
                    params={
                        "stage": stage_name,
                        "time_period": "2024",
                        "stage_contract_sha256": _stage_contract_sha256(
                            manifest_stage, resolved
                        ),
                    },
                    sources=_source_names(stage_name, source_mode),
                    description=f"Run UK spine stage {stage_name}.",
                )
            )
            version_owned.update(coordinates)

        for cell in cells:
            live[cell.coordinate] = cell
        if cells:
            shape_anchor = cells[-1].coordinate

        if stage_name == "frs_brma":
            checkpoint = "frs_brma.checkpoint"
            nodes.append(
                Node(
                    id=checkpoint,
                    kernel="uk.identity@1",
                    inputs=_slices(live),
                    structural=StructuralDelta.FILTER,
                    base=current_population,
                    description="Freeze the assembled-spine gate population.",
                )
            )
            current_population = checkpoint
            version_owned = set()

    return Graph(
        country="uk",
        sources=_source_refs(source_mode),
        nodes=tuple(nodes),
    )


def uk_registry(
    implementations: Mapping[str, object] | None = None,
    *,
    graph: Graph | None = None,
) -> KernelRegistry:
    """Return kernels for ``graph``, optionally bound to real stage transforms."""

    from .graph_kernels import build_uk_registry

    return build_uk_registry(
        uk_spine_graph() if graph is None else graph,
        {} if implementations is None else implementations,
    )
