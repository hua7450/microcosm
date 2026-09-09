"""Raw FRS tab ingest for the UK spine Frame."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.trace import sha256_file
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask
from microcosm.frame import Frame, WeightKind

__all__ = [
    "FRS_SPINE_TABLES",
    "OUTPUT_COLUMNS",
    "REGION_MAP",
    "TIME_PERIOD",
    "UC_CAPITAL_UNAVAILABLE",
    "WEEKS_IN_YEAR",
    "UKFRSSpineStageTransform",
    "artifact_by_table",
    "build_uk_frs_spine_frame",
    "map_codes",
    "normalize_ids",
    "number",
    "positive",
    "raw_number",
    "read_pinned_tab",
    "reject_nan",
    "sum_to_entity",
    "uk_frs_spine_seed_frame",
]

FRS_SPINE_TABLES = (
    "accounts",
    "adult",
    "benefits",
    "benunit",
    "child",
    "chldcare",
    "extchild",
    "househol",
    "job",
    "maint",
    "mortgage",
    "oddjob",
    "penprov",
    "pension",
)

WEEKS_IN_YEAR = 365.25 / 7
TIME_PERIOD = "2024"

# Raw TOTCAPB4 blanks, nonnumeric values, and negative codes mean that the
# benefit-unit capital observation is unavailable. They map to this engine
# sentinel; observed zero remains a valid capital value.
UC_CAPITAL_UNAVAILABLE = -1.0

# FRS GVTREGNO uses skip-3 coding: code 3 (the retired Merseyside code) is
# absent from the domain, so the real codes are [1, 2, 4..13] with
# 12 = Scotland and 13 = Northern Ireland — matching the incumbent's
# ``[1, 2] + range(4, 15)`` categorical and its ``SCOTLAND_GVTREGNO = 12``
# water-charge branch below. Verified on the 2023-24 tabs: zero code-3
# households, 1,844 code-13 (Northern Ireland) households. A contiguous
# 1-12 map shifts every label from Yorkshire up by one region and drops
# Northern Ireland to UNKNOWN (#692 review). The SPI tape's GORCODE is a
# different coding with its own history — do not import lessons across.
REGION_MAP = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    4: "YORKSHIRE",
    5: "EAST_MIDLANDS",
    6: "WEST_MIDLANDS",
    7: "EAST_OF_ENGLAND",
    8: "LONDON",
    9: "SOUTH_EAST",
    10: "SOUTH_WEST",
    11: "WALES",
    12: "SCOTLAND",
    13: "NORTHERN_IRELAND",
}

TENURE_MAP = {
    1: "RENT_FROM_COUNCIL",
    2: "RENT_FROM_HA",
    3: "RENT_PRIVATELY",
    4: "RENT_PRIVATELY",
    5: "OWNED_OUTRIGHT",
    6: "OWNED_WITH_MORTGAGE",
}

ACCOMMODATION_MAP = {
    1: "HOUSE_DETACHED",
    2: "HOUSE_SEMI_DETACHED",
    3: "HOUSE_TERRACED",
    4: "FLAT",
    5: "CONVERTED_HOUSE",
    6: "MOBILE",
    7: "OTHER",
}

COUNCIL_TAX_BAND_MAP = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
}

MARITAL_MAP = {
    1: "MARRIED",
    2: "SINGLE",
    3: "SINGLE",
    4: "WIDOWED",
    5: "SEPARATED",
    6: "DIVORCED",
}

BENEFIT_CODES = {
    "child_benefit": 3,
    "income_support": 19,
    "housing_benefit": 94,
    "attendance_allowance": 12,
    "dla_sc": 1,
    "dla_m": 2,
    "iidb": 15,
    "carers_allowance": 13,
    "sda": 10,
    "afcs": 8,
    "ssmg": 22,
    "pension_credit": 4,
    "child_tax_credit": 91,
    "working_tax_credit": 90,
    "state_pension": 5,
    "winter_fuel_allowance": 62,
    "incapacity_benefit": 17,
    "universal_credit": 95,
    "pip_m": 97,
    "pip_dl": 96,
}

OUTPUT_COLUMNS = (
    "person_id",
    "person_benunit_id",
    "person_household_id",
    "age",
    "gender",
    "marital_status",
    "hours_worked",
    "is_household_head",
    "is_benunit_head",
    "is_parent",
    "is_uc_claimant",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "tax_free_savings_income",
    "savings_interest_income",
    "dividend_income",
    "property_income",
    "maintenance_income",
    "miscellaneous_income",
    "private_transfer_income",
    "lump_sum_income",
    "student_loan_repayments",
    "statutory_sick_pay",
    "statutory_maternity_pay",
    "student_loans",
    "access_fund",
    "education_grants",
    "healthy_start_vouchers",
    "free_school_breakfasts",
    "free_school_fruit_veg",
    "free_school_meals",
    "council_tax_benefit_reported",
    "maintenance_expenses",
    "childcare_expenses",
    "personal_pension_contributions",
    "employee_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    "salary_sacrifice_reported",
    "salary_sacrifice_asked",
    "child_benefit_reported",
    "income_support_reported",
    "housing_benefit_reported",
    "attendance_allowance_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "iidb_reported",
    "carers_allowance_reported",
    "sda_reported",
    "afcs_reported",
    "ssmg_reported",
    "pension_credit_reported",
    "child_tax_credit_reported",
    "working_tax_credit_reported",
    "state_pension_reported",
    "winter_fuel_allowance_reported",
    "incapacity_benefit_reported",
    "universal_credit_reported",
    "pip_m_reported",
    "pip_dl_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
    "bsp_reported",
    "benunit_id",
    "frs_benunit_capital",
    "is_married",
    "dependent_children",
    "household_id",
    "region",
    "tenure_type",
    "accommodation_type",
    "num_bedrooms",
    "council_tax_reported",
    "council_tax_band",
    "council_tax_rebate",
    "council_tax_single_adult_raw",
    "water_and_sewerage_charges",
    "domestic_rates",
    "rent",
    "subrent",
    "mortgage_interest_repayment",
    "mortgage_capital_repayment",
    "structural_insurance_payments",
    "housing_service_charges",
    "external_child_payments",
)


class UKFRSSpineStageTransform:
    """Whole-stage callable for the root FRS spine ingest."""

    def __init__(self, raw_dir: str | Path, *, stage: SourceStageSpec) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage
        self._sentinel_mapped_rows: int | None = None

    def __call__(self, frame: Frame) -> Frame:
        result = build_uk_frs_spine_frame(self.raw_dir, stage=self.stage)
        capital = result.table("benunit")["frs_benunit_capital"]
        self._sentinel_mapped_rows = int((capital == UC_CAPITAL_UNAVAILABLE).sum())
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        """Report how loudly the FRS capital availability rule fired."""

        if self._sentinel_mapped_rows is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {
            "evidence": {
                "stage": "frs_spine",
                "frs_benunit_capital": {
                    "unavailable_sentinel": UC_CAPITAL_UNAVAILABLE,
                    "mapped_rows": self._sentinel_mapped_rows,
                },
            }
        }


def uk_frs_spine_seed_frame() -> Frame:
    """A minimal valid Frame for the root stage; the stage ignores its rows."""

    person = pd.DataFrame(
        {
            "person_id": [1],
            "person_benunit_id": [1],
            "person_household_id": [1],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1]})
    household = pd.DataFrame({"household_id": [1], "household_weight": [1.0]})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=TIME_PERIOD,
        weight_kind=WeightKind.DESIGN,
    )


def build_uk_frs_spine_frame(raw_dir: str | Path, *, stage: SourceStageSpec) -> Frame:
    """Build the direct raw FRS spine Frame from pinned local tab files."""

    raw_root = Path(raw_dir)
    artifacts = _artifact_by_table(stage)
    tables = {
        table: _read_pinned_tab(
            raw_root / str(artifacts[table]["locator"]), artifacts[table]
        )
        for table in FRS_SPINE_TABLES
    }
    normalized = {name: _normalize_ids(table) for name, table in tables.items()}
    frame = _assemble_frame(normalized)
    validate_uk_national_frame(frame)
    return frame


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table: dict[str, Mapping[str, Any]] = {}
    for artifact in stage.artifacts:
        table = artifact.get("table")
        if table in by_table:
            raise ValueError(f"Duplicate FRS spine artifact for table {table!r}.")
        if isinstance(table, str):
            by_table[table] = artifact
    missing = sorted(set(FRS_SPINE_TABLES) - set(by_table))
    if missing:
        raise ValueError(f"FRS spine manifest is missing tab artifact(s): {missing}.")
    unknown = sorted(set(by_table) - set(FRS_SPINE_TABLES))
    if unknown:
        raise ValueError(f"FRS spine manifest declares unknown tab(s): {unknown}.")
    return by_table


def _read_pinned_tab(path: Path, artifact: Mapping[str, Any]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"FRS spine tab is missing: {path}.")
    expected_size = int(artifact["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"FRS spine tab {path.name} is {actual_size} bytes, "
            f"not the pinned {expected_size}."
        )
    expected_sha = str(artifact["sha256"])
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"FRS spine tab {path.name} hashes to {actual_sha}, "
            f"not the pinned {expected_sha}."
        )
    raw = pd.read_csv(path, sep="\t")
    raw.columns = raw.columns.str.lower()
    converted = raw.apply(pd.to_numeric, errors="coerce")
    if path.stem == "job" and "salsac" in raw.columns:
        converted["salsac_raw"] = raw["salsac"].astype(str)
    return converted


def _normalize_ids(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    if "sernum" in frame.columns:
        frame = frame.rename(columns={"sernum": "household_id"})
    if "benunit" in frame.columns:
        frame = frame.rename(columns={"benunit": "benunit_id"})
        frame["benunit_id"] = (
            _number(frame, "household_id") * 100 + _number(frame, "benunit_id")
        ).astype("int64")
    if "person" in frame.columns:
        frame = frame.rename(columns={"person": "person_id"})
        frame["person_id"] = (
            _number(frame, "household_id") * 1000 + _number(frame, "person_id")
        ).astype("int64")
    return frame


def _assemble_frame(frs: Mapping[str, pd.DataFrame]) -> Frame:
    person = (
        pd.concat([frs["adult"], frs["child"]], ignore_index=True, sort=False)
        .fillna(0)
        .sort_values(["household_id", "person_id"])
        .reset_index(drop=True)
    )
    adult_ids = set(frs["adult"]["person_id"].astype("int64"))
    benunit_raw = frs["benunit"].sort_values("benunit_id").reset_index(drop=True)
    household_raw = frs["househol"].set_index("household_id").sort_index()
    household_ids = np.sort(person["household_id"].unique())
    household = household_raw.loc[household_ids]

    pe_person = pd.DataFrame(
        {
            "person_id": person["person_id"].astype("int64"),
            "person_benunit_id": person["benunit_id"].astype("int64"),
            "person_household_id": person["household_id"].astype("int64"),
        }
    )
    pe_benunit = pd.DataFrame({"benunit_id": benunit_raw["benunit_id"].astype("int64")})
    pe_benunit["frs_benunit_capital"] = _frs_benunit_capital(benunit_raw)
    pe_household = pd.DataFrame(
        {"household_id": household_ids.astype("int64")},
        index=household.index,
    )
    pe_household["household_weight"] = _raw_number(household, "gross4").to_numpy()

    age = _number(person, "age80") + _number(person, "age")
    # The graph declares person.age int64 from the root (#845); the raw
    # AGE80/AGE columns are integer codes, and blanks coerce to 0 above, so
    # a non-integral value here is a vintage defect rather than data.
    if not np.array_equal(age.to_numpy(), np.floor(age.to_numpy())):
        raise ValueError("FRS age80/age columns must be integral to build person.age.")
    pe_person["age"] = age.astype("int64")
    pe_person["gender"] = np.where(_number(person, "sex") == 1, "MALE", "FEMALE")
    pe_person["marital_status"] = _map_codes(person, "marital", MARITAL_MAP, "SINGLE")
    pe_person["hours_worked"] = _positive(person, "tothours") * WEEKS_IN_YEAR
    pe_person["is_household_head"] = _number(person, "hrpid") == 1
    pe_person["is_benunit_head"] = _number(person, "uperson") == 1
    dependent_children = _dependent_children(benunit_raw, frs["child"])
    dependent_by_benunit = pd.Series(
        dependent_children.to_numpy(), index=benunit_raw["benunit_id"]
    )
    pe_person["is_parent"] = pe_person["person_id"].isin(adult_ids) & pe_person[
        "person_benunit_id"
    ].map(dependent_by_benunit).fillna(0).gt(0)

    pe_person["employment_income"] = _positive(person, "inearns") * WEEKS_IN_YEAR
    pe_person["self_employment_income"] = _positive(person, "seincam2") * WEEKS_IN_YEAR
    _add_private_pension(pe_person, person, frs["pension"])
    _add_accounts(pe_person, person, frs["accounts"])
    _add_person_income(pe_person, person, household, frs["oddjob"])
    _add_benefits(pe_person, person, frs["benefits"])
    _add_person_expenses(pe_person, person, frs)

    pe_benunit["is_married"] = _number(benunit_raw, "famtypb2").isin([5, 7])
    pe_benunit["dependent_children"] = dependent_children
    # Preserve the FRS claimant/partner roles as a country-model input. Age
    # does not promote a dependent child to a partner, and legal marriage
    # alone does not establish that a partner lives in this benefit unit.
    pe_person["is_uc_claimant"] = frs_uc_claimant_mask(pe_person, pe_benunit)

    _add_household_columns(pe_household, household, frs)

    _reject_nan(pe_person, "person")
    _reject_nan(pe_benunit, "benunit")
    _reject_nan(pe_household, "household")
    return uk_national_frame(
        person=pe_person,
        benunit=pe_benunit,
        household=pe_household,
        time_period=TIME_PERIOD,
        weight_kind=WeightKind.DESIGN,
    )


def _add_private_pension(
    pe_person: pd.DataFrame, person: pd.DataFrame, pension: pd.DataFrame
) -> None:
    pension_payment = _sum_to_entity(
        _number(pension, "penpay") * (_number(pension, "penpay") > 0),
        pension.get("person_id", pd.Series(dtype="float64")),
        person["person_id"],
    )
    pension_tax_paid = _sum_to_entity(
        _number(pension, "ptamt")
        * ((_number(pension, "ptinc") == 2) & (_number(pension, "ptamt") > 0)),
        pension.get("person_id", pd.Series(dtype="float64")),
        person["person_id"],
    )
    pension_deductions_removed = _sum_to_entity(
        _number(pension, "poamt")
        * (
            ((_number(pension, "poinc") == 2) | (_number(pension, "penoth") == 1))
            & (_number(pension, "poamt") > 0)
        ),
        pension.get("person_id", pd.Series(dtype="float64")),
        person["person_id"],
    )
    pe_person["private_pension_income"] = (
        pension_payment + pension_tax_paid + pension_deductions_removed
    ) * WEEKS_IN_YEAR


def _add_accounts(
    pe_person: pd.DataFrame, person: pd.DataFrame, accounts: pd.DataFrame
) -> None:
    account = _number(accounts, "account")
    accint = _number(accounts, "accint")
    acctax = _number(accounts, "acctax")
    invtax = _number(accounts, "invtax")
    person_id = accounts.get("person_id", pd.Series(dtype="float64"))
    inverted_basic_rate = 1.25
    tax_free = (
        _sum_to_entity(accint * (account == 21), person_id, person["person_id"])
        * WEEKS_IN_YEAR
    )
    taxable = (
        _sum_to_entity(
            (accint * np.where(acctax == 1, inverted_basic_rate, 1))
            * account.isin((1, 3, 5, 27, 28)),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    dividends = (
        _sum_to_entity(
            (accint * np.where(invtax == 1, inverted_basic_rate, 1))
            * (((account == 6) & (invtax == 1)) | account.isin((7, 8))),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["tax_free_savings_income"] = np.maximum(0, tax_free)
    pe_person["savings_interest_income"] = np.maximum(0, taxable + tax_free)
    pe_person["dividend_income"] = np.maximum(0, dividends)


def _add_person_income(
    pe_person: pd.DataFrame,
    person: pd.DataFrame,
    household: pd.DataFrame,
    oddjob: pd.DataFrame,
) -> None:
    is_head = _number(person, "hrpid") == 1
    household_property = _number(household, "tentyp2").isin((5, 6)).astype(
        float
    ) * _number(household, "subrent")
    property_by_household = pd.Series(household_property.values, index=household.index)
    pe_person["property_income"] = (
        np.maximum(
            0,
            is_head.to_numpy(dtype=float)
            * person["household_id"].map(property_by_household).fillna(0).to_numpy()
            + _number(person, "cvpay").to_numpy()
            + _number(person, "royyr1").to_numpy(),
        )
        * WEEKS_IN_YEAR
    )
    maintenance_to_self = np.maximum(
        np.where(
            _number(person, "mntus1") == 2,
            _number(person, "mntusam1"),
            _number(person, "mntamt1"),
        ),
        0,
    )
    pe_person["maintenance_income"] = (
        maintenance_to_self + _positive(person, "mntamt2").to_numpy()
    ) * WEEKS_IN_YEAR
    pe_person["miscellaneous_income"] = (
        _odd_job_income(person, oddjob)
        + _sum_positive_fields(
            person, ("allpay2", "royyr2", "royyr3", "royyr4", "chamtern", "chamttst")
        )
    ) * WEEKS_IN_YEAR
    pe_person["private_transfer_income"] = (
        _sum_positive_fields(
            person, ("apamt", "apdamt", "pareamt", "allpay2", "allpay3", "allpay4")
        )
        * WEEKS_IN_YEAR
    )
    pe_person["lump_sum_income"] = _number(person, "redamt")
    pe_person["student_loan_repayments"] = _number(person, "slrepamt") * WEEKS_IN_YEAR
    pe_person["statutory_sick_pay"] = _number(person, "sspadj") * WEEKS_IN_YEAR
    pe_person["statutory_maternity_pay"] = _number(person, "smpadj") * WEEKS_IN_YEAR
    pe_person["student_loans"] = _positive(person, "tuborr")
    pe_person["access_fund"] = _positive(person, "accssamt") * WEEKS_IN_YEAR
    pe_person["education_grants"] = np.maximum(
        _number(person, "grtdir1") + _number(person, "grtdir2"), 0
    )
    # In-kind benefits recorded per person on the FRS tapes. Each is a direct
    # weeklyised amount with no derivation: healthy-start vouchers appear on
    # both the adult and child tapes, the three school ones only on the child
    # tape, so absent columns read as zero for adults through `_number`.
    pe_person["healthy_start_vouchers"] = _positive(person, "heartval") * WEEKS_IN_YEAR
    pe_person["free_school_breakfasts"] = _positive(person, "fsbval") * WEEKS_IN_YEAR
    pe_person["free_school_fruit_veg"] = _positive(person, "fsfvval") * WEEKS_IN_YEAR
    pe_person["free_school_meals"] = _positive(person, "fsmval") * WEEKS_IN_YEAR


def _odd_job_income(person: pd.DataFrame, oddjob: pd.DataFrame) -> np.ndarray:
    return _sum_to_entity(
        _number(oddjob, "ojamt") * (_number(oddjob, "ojnow") == 1),
        oddjob.get("person_id", pd.Series(dtype="float64")),
        person["person_id"],
    )


def _add_benefits(
    pe_person: pd.DataFrame, person: pd.DataFrame, benefits: pd.DataFrame
) -> None:
    benefit = _number(benefits, "benefit")
    var2 = _number(benefits, "var2")
    amount = _number(benefits, "benamt")
    person_id = benefits.get("person_id", pd.Series(dtype="float64"))
    for name, code in BENEFIT_CODES.items():
        pe_person[f"{name}_reported"] = (
            _sum_to_entity(amount * (benefit == code), person_id, person["person_id"])
            * WEEKS_IN_YEAR
        )
    pe_person["jsa_contrib_reported"] = (
        _sum_to_entity(
            amount * var2.isin((1, 3)) * (benefit == 14),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["jsa_income_reported"] = (
        _sum_to_entity(
            amount * var2.isin((2, 4)) * (benefit == 14),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["esa_contrib_reported"] = (
        _sum_to_entity(
            amount * var2.isin((1, 3)) * (benefit == 16),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["esa_income_reported"] = (
        _sum_to_entity(
            amount * var2.isin((2, 4)) * (benefit == 16),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["bsp_reported"] = (
        _sum_to_entity(amount * benefit.isin((6, 9)), person_id, person["person_id"])
        * WEEKS_IN_YEAR
    )
    pe_person["winter_fuel_allowance_reported"] = (
        pe_person["winter_fuel_allowance_reported"] / WEEKS_IN_YEAR
    )


def _add_person_expenses(
    pe_person: pd.DataFrame, person: pd.DataFrame, frs: Mapping[str, pd.DataFrame]
) -> None:
    household = frs["househol"].set_index("household_id").sort_index()
    ctrebamt = pd.Series(_number(household, "ctrebamt").values, index=household.index)
    pe_person["council_tax_benefit_reported"] = np.maximum(
        (_number(person, "hrpid") == 1).to_numpy(dtype=float)
        * person["household_id"].map(ctrebamt).fillna(0).to_numpy()
        * WEEKS_IN_YEAR,
        0,
    )
    maintenance = frs["maint"]
    pe_person["maintenance_expenses"] = (
        _sum_to_entity(
            np.where(
                _number(maintenance, "mrus") == 2,
                _number(maintenance, "mruamt"),
                _number(maintenance, "mramt"),
            ),
            maintenance.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    childcare = frs["chldcare"]
    pe_person["childcare_expenses"] = (
        _sum_to_entity(
            _number(childcare, "chamt")
            * (_number(childcare, "cost") == 1)
            * (_number(childcare, "registrd") == 1),
            childcare.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    penprov = frs["penprov"]
    pension_amount = _number(penprov, "penamt")
    pension_clip = float(pension_amount.quantile(0.95)) if len(pension_amount) else 0.0
    pe_person["personal_pension_contributions"] = np.maximum(
        0,
        _sum_to_entity(
            pension_amount[_number(penprov, "stemppen").isin((5, 6))],
            penprov.loc[_number(penprov, "stemppen").isin((5, 6)), "person_id"],
            person["person_id"],
        ).clip(0, pension_clip)
        * WEEKS_IN_YEAR,
    )
    job = frs["job"]
    pe_person["employee_pension_contributions"] = np.maximum(
        0,
        _sum_to_entity(
            _number(job, "deduc1").fillna(0),
            job.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        )
        * WEEKS_IN_YEAR,
    )
    pe_person["pension_contributions_via_salary_sacrifice"] = np.maximum(
        0,
        _sum_to_entity(
            _number(job, "spnamt").fillna(0),
            job.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        )
        * WEEKS_IN_YEAR,
    )
    salsac = job.get("salsac_raw", pd.Series(dtype=object)).map(
        {"1": 1, "2": 0, " ": -1, "": -1}
    )
    salsac = salsac.fillna(-1).astype(int)
    pe_person["salary_sacrifice_reported"] = np.clip(
        _sum_to_entity(
            (salsac == 1).astype(int),
            job.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        ),
        0,
        1,
    )
    pe_person["salary_sacrifice_asked"] = np.clip(
        _sum_to_entity(
            (salsac >= 0).astype(int),
            job.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        ),
        0,
        1,
    )


def _add_household_columns(
    pe_household: pd.DataFrame,
    household: pd.DataFrame,
    frs: Mapping[str, pd.DataFrame],
) -> None:
    pe_household["region"] = _map_codes(household, "gvtregno", REGION_MAP, "UNKNOWN")
    pe_household["tenure_type"] = _map_codes(
        household, "ptentyp2", TENURE_MAP, "RENT_PRIVATELY"
    )
    pe_household["accommodation_type"] = _map_codes(
        household, "typeacc", ACCOMMODATION_MAP, "HOUSE_DETACHED"
    )
    pe_household["num_bedrooms"] = _number(household, "bedroom6")
    pe_household["council_tax_reported"] = _positive(household, "ctannual")
    pe_household["council_tax_band"] = _map_codes(
        household, "ctband", COUNCIL_TAX_BAND_MAP, "D"
    )
    pe_household["council_tax_rebate"] = (
        _positive(household, "ctrebamt") * WEEKS_IN_YEAR
    )
    pe_household["council_tax_single_adult_raw"] = _number(household, "adulth")
    scotland = _number(household, "gvtregno") == 12
    pe_household["water_and_sewerage_charges"] = (
        np.where(
            scotland,
            scottish_water_and_sewerage_weekly(household),
            _number(household, "watsewrt"),
        )
        * WEEKS_IN_YEAR
    )
    pe_household["domestic_rates"] = (
        np.select(
            [
                _number(household, "niratlia") >= 0,
                _number(household, "rt2rebam") >= 0,
                True,
            ],
            [_number(household, "niratlia"), _number(household, "rt2rebam"), 0],
        )
        * WEEKS_IN_YEAR
    ).astype(float)
    pe_household["rent"] = _number(household, "hhrent") * WEEKS_IN_YEAR
    pe_household["subrent"] = _number(household, "subrent") * WEEKS_IN_YEAR
    pe_household["mortgage_interest_repayment"] = (
        _number(household, "mortint") * WEEKS_IN_YEAR
    )
    mortgage = frs["mortgage"]
    mortgage_capital = np.where(
        _number(mortgage, "rmort") == 1,
        _number(mortgage, "rmamt"),
        _number(mortgage, "borramt"),
    )
    mortgage_end = _number(mortgage, "mortend").replace(0, np.nan)
    pe_household["mortgage_capital_repayment"] = _sum_to_entity(
        pd.Series(mortgage_capital).divide(mortgage_end).fillna(0),
        mortgage.get("household_id", pd.Series(dtype="float64")),
        household.index,
    )
    pe_household["structural_insurance_payments"] = (
        _number(household, "struins") * WEEKS_IN_YEAR
    )
    pe_household["housing_service_charges"] = (
        sum(_positive(household, f"chrgamt{i}") for i in range(1, 10)) * WEEKS_IN_YEAR
    )
    extchild = frs["extchild"]
    pe_household["external_child_payments"] = _sum_to_entity(
        _number(extchild, "nhhamt") * WEEKS_IN_YEAR,
        extchild.get("household_id", pd.Series(dtype="float64")),
        household.index,
    )


def _dependent_children(benunit: pd.DataFrame, child: pd.DataFrame) -> pd.Series:
    if "depchldb" in benunit.columns:
        return _number(benunit, "depchldb")
    return (
        child.groupby("benunit_id")
        .size()
        .reindex(benunit["benunit_id"])
        .fillna(0)
        .astype(float)
        .reset_index(drop=True)
    )


def _sum_to_entity(values: Any, source_ids: Any, target_ids: pd.Series) -> np.ndarray:
    source = pd.Series(np.asarray(source_ids))
    value = pd.Series(np.asarray(values, dtype="float64")).fillna(0)
    if source.empty or value.empty:
        return np.zeros(len(target_ids), dtype="float64")
    totals = value.groupby(source).sum()
    return totals.reindex(target_ids).fillna(0).to_numpy(dtype="float64")


def _sum_positive_fields(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    total = np.zeros(len(frame), dtype="float64")
    for column in columns:
        total += _positive(frame, column).to_numpy()
    return total


def _map_codes(
    frame: pd.DataFrame,
    column: str,
    mapping: Mapping[int, str],
    default: str,
) -> pd.Series:
    values = _number(frame, column).astype("int64", errors="ignore")
    return values.map(mapping).fillna(default).astype(object)


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.zeros(len(frame), dtype="float64"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _raw_number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), np.nan), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _frs_benunit_capital(benunit: pd.DataFrame) -> pd.Series:
    """Map TOTCAPB4 to capital, using the sentinel only for unavailable values.

    The I1 FRS 2024-25 audit found all 18,850 rows populated and nonnegative.
    Future raw blanks, nonnumeric values, or negative codes map to
    ``UC_CAPITAL_UNAVAILABLE``; observed zero is preserved.
    """

    raw = _raw_number(benunit, "totcapb4")
    return raw.where(raw.notna() & raw.ge(0), UC_CAPITAL_UNAVAILABLE).astype(float)


def _positive(frame: pd.DataFrame, column: str) -> pd.Series:
    return np.maximum(_number(frame, column), 0)


def scottish_water_and_sewerage_weekly(household: pd.DataFrame) -> pd.Series:
    """Weekly Scottish water + sewerage charge, net of the household's discount.

    Scotland is not asked ``WATSEWRT`` — its water and sewerage charges ride on
    the council tax bill — so the amount must be assembled from the council-tax
    water/sewerage cells.

    FRS 2024-25 retired the two cells the incumbent used.  ``CWATAMT``/
    ``CSEWAMT`` ("Wat./Sew. Charge: Final value after discount") are still
    present as headers but carry no data at all in this vintage, and the FRS
    replaced them with the derived ``CWATAMT1``/``CSEWAMT1`` ("Weeklyised gross
    annual dom. water/sew. charge on bill", Scotland only, "DV created in
    2024-25 as variable was removed from the dataset for 2024-25" — SN 9563
    ``9563_dv_summary_2425.xlsx``).

    The replacements are **gross**, where the retired cells were **after
    discount**, and the FRS publishes no discounted sewerage counterpart.
    ``CWATAMTD`` ("Deriv Council Tax water charge -Scot") does carry the
    discount, so the household's own discount factor is observable as
    ``CWATAMTD / CWATAMT1`` and applies to the sewerage side of the same bill.
    That keeps the incumbent's semantics — what the household actually pays —
    across the vintage change rather than silently switching to a gross basis.

    Every value is weeklyised (all five cells sit in the FRS "weekly variables"
    listing), so callers apply ``WEEKS_IN_YEAR`` themselves.

    On the 2024-25 tab the domain splits cleanly: 1,641 Scottish households
    carry a positive gross water charge and a well-defined factor in
    (1/3, 1]; 22 carry a recorded ``CWATAMTD`` with no gross bill cell, and
    their ``CSEWAMT1`` is zero, so the fallback factor cannot move them; 21
    carry no council-tax cells at all and fall to zero.
    """

    water = _positive(household, "cwatamtd")
    water_gross = _positive(household, "cwatamt1")
    sewerage_gross = _positive(household, "csewamt1")
    billed = water_gross > 0
    # The correctness of the discount fallback rests on the domain claim
    # above: a household with no gross water bill carries no sewerage gross
    # either, so the fallback factor of 1.0 can never re-introduce the gross
    # basis this helper exists to avoid. That claim is a property of the
    # vintage, not of the code — so it is asserted, and a vintage that breaks
    # it refuses at build time instead of silently paying gross sewerage.
    undiscountable = ~billed & (sewerage_gross > 0)
    if bool(undiscountable.any()):
        raise ValueError(
            "Scottish water assembly: "
            f"{int(undiscountable.sum())} household(s) carry a gross sewerage "
            "charge (CSEWAMT1 > 0) with no gross water bill (CWATAMT1 == 0), "
            "so no discount factor is observable for them. Adding sewerage at "
            "gross would silently change the charge's after-discount meaning; "
            "this vintage needs an adjudicated rule for these households."
        )
    discount = np.where(billed, water / water_gross.where(billed, 1.0), 1.0)
    return water + sewerage_gross * discount


def _reject_nan(frame: pd.DataFrame, entity: str) -> None:
    if frame.isna().any().any():
        bad = sorted(frame.columns[frame.isna().any()].tolist())
        raise ValueError(f"FRS spine {entity} produced NaN column(s): {bad}.")


artifact_by_table = _artifact_by_table
read_pinned_tab = _read_pinned_tab
normalize_ids = _normalize_ids
sum_to_entity = _sum_to_entity
map_codes = _map_codes
number = _number
raw_number = _raw_number
positive = _positive
reject_nan = _reject_nan
