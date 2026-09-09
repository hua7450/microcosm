from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_support import (
    support_channel_column,
    support_clone_index_column,
)
from microcosm.build.uk_runtime.uc_capital_coherence import cohere_uc_capital
from microcosm.build.uk_runtime.uc_reporter_redraw import (
    UC_REPORTER_REDRAW_OUTPUT,
    UC_REPORTER_REDRAW_SEED,
    UC_REPORTER_SCREEN_VARIABLES,
    UKUCReporterRedrawStageTransform,
    _assert_stage_parameters,
    _benefit_unit_reporter_amounts,
    _materialize_screen_inputs,
    _positive_pre_takeup_award_screen,
    redraw_spi_reported_uc,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine


def _stage():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()["uc_reporter_redraw"]


class _StubEngine:
    country = "uk"

    def __init__(self, *, fail_all_spi: bool = False) -> None:
        self.calls: list[tuple[object, tuple[str, ...], str]] = []
        self.fail_all_spi = fail_all_spi

    def materialize(self, frame, variables, period):
        self.calls.append((frame, tuple(variables), str(period)))
        benunit = frame.table("benunit")
        person = frame.table("person")
        # Benefit unit 202 fails the hard award screen. All others pass —
        # including a child-only unit, because uc_maximum_amount is
        # mechanical and pays it.
        maximum = np.full(len(benunit), 100.0)
        reduction = np.zeros(len(benunit))
        reduction[benunit["benunit_id"].eq(202).to_numpy()] = 100.0
        if self.fail_all_spi:
            reduction[
                benunit[support_channel_column("benunit")].eq("spi").to_numpy()
            ] = 100.0
        return {
            "uc_maximum_amount": maximum,
            "uc_income_reduction": reduction,
            "is_child_or_qualifying_young_person_for_universal_credit": (
                person["is_uc_child"].to_numpy(dtype=bool)
            ),
            "is_SP_age": person["age"].to_numpy(dtype=float) >= 66.0,
        }


class _StubFittedQRF:
    def __init__(self, record: dict[str, object]) -> None:
        self.record = record

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        self.record["predict_index"] = table.index.tolist()
        draws = np.zeros(len(table), dtype=float)
        if len(draws):
            draws[0] = 250.0
        return pd.DataFrame(
            {"universal_credit_reported_amount": draws},
            index=table.index,
        )


class _StubQRF:
    def __init__(self, record: dict[str, object], *, seed: int) -> None:
        record["seed"] = seed
        self.record = record

    def fit(self, table, predictors, targets, *, weights):
        self.record["fit_table"] = table.copy()
        self.record["fit_index"] = table.index.tolist()
        self.record["fit_predictors"] = list(predictors)
        self.record["fit_targets"] = list(targets)
        self.record["fit_weights"] = np.asarray(weights).copy()
        return _StubFittedQRF(self.record)


class _StubQRFFactory:
    def __init__(self) -> None:
        self.record: dict[str, object] = {}

    def __call__(self, *, seed: int):
        return _StubQRF(self.record, seed=seed)


def _frame(*, child_only: bool = False):
    benunit_rows = [
        # Four canonical, screened FRS training benefit units. The first is a
        # couple-with-child reporter donor needed by uc_capital_coherence.
        (101, 1, "frs", 0, False, True, 1, 100.0, 2.0),
        (102, 2, "frs", 0, False, False, 0, 0.0, 3.0),
        # Both are screened but excluded from training by the exact mask.
        (103, 3, "frs", 1, False, False, 0, 70.0, 6.0),
        (104, 4, "frs", 0, True, False, 0, 60.0, 7.0),
        (105, 5, "frs", 0, False, False, 0, 80.0, 4.0),
        (106, 6, "frs", 0, False, True, 1, 0.0, 5.0),
        # SPI: 201 passes and receives a draw; 202 fails and is zeroed; 203
        # passes but the stub model returns zero.
        (201, 7, "spi", 1, False, True, 1, 10.0, 1.0),
        (202, 8, "spi", 1, False, False, 0, 50.0, 1.0),
        (203, 9, "spi", 1, False, False, 0, 0.0, 1.0),
    ]
    if child_only:
        # A 17-year-old qualifying young person heading their own SPI unit,
        # carrying a stage-2 chain fill of 300. Its only member is a child.
        benunit_rows.append((204, 10, "spi", 1, False, False, 0, 300.0, 1.0))
    people: list[dict[str, object]] = []
    for (
        benunit_id,
        household_id,
        _channel,
        _clone,
        _,
        married,
        children,
        reported,
        _,
    ) in benunit_rows:
        if married:
            ages = (45, 45)
        elif benunit_id == 204:
            ages = (17,)
        else:
            ages = (70,) if benunit_id == 203 else (40,)
        for member, age in enumerate(ages):
            person_id = benunit_id * 10 + member + 1
            # Reverse the IDs for BU 201 so lowest-id, not input order, proves
            # the tie-break. The first table row has the larger ID.
            if benunit_id == 201:
                person_id = 2012 - member
            people.append(
                {
                    "person_id": person_id,
                    "person_benunit_id": benunit_id,
                    "person_household_id": household_id,
                    "age": age,
                    "is_benunit_head": member == 0,
                    "is_parent": children > 0,
                    "is_uc_child": benunit_id == 204,
                    "employment_income": 10_000.0 + benunit_id + member,
                    "self_employment_income": 100.0 * member,
                    "savings_interest_income": 10.0,
                    "dividend_income": 20.0,
                    "property_income": 30.0,
                    "other_investment_income": 40.0,
                    UC_REPORTER_REDRAW_OUTPUT: reported if member == 0 else 0.0,
                }
            )
        if children:
            people.append(
                {
                    "person_id": benunit_id * 10 + 9,
                    "person_benunit_id": benunit_id,
                    "person_household_id": household_id,
                    "age": 10,
                    "is_benunit_head": False,
                    "is_parent": False,
                    "is_uc_child": True,
                    "employment_income": 0.0,
                    "self_employment_income": 0.0,
                    "savings_interest_income": 0.0,
                    "dividend_income": 0.0,
                    "property_income": 0.0,
                    "other_investment_income": 0.0,
                    UC_REPORTER_REDRAW_OUTPUT: 0.0,
                }
            )
    person = pd.DataFrame(people)
    benunit = pd.DataFrame(
        {
            "benunit_id": [row[0] for row in benunit_rows],
            support_channel_column("benunit"): [row[2] for row in benunit_rows],
            support_clone_index_column("benunit"): [row[3] for row in benunit_rows],
            "frs_benunit_capital": [1_000.0 + row[0] for row in benunit_rows],
            "is_married": [row[5] for row in benunit_rows],
            "dependent_children": [row[6] for row in benunit_rows],
            "would_claim_uc": False,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [row[1] for row in benunit_rows],
            "region": ["LONDON" if row[1] % 2 else "SCOTLAND" for row in benunit_rows],
            "household_is_capital_gains_clone": [row[4] for row in benunit_rows],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([row[8] for row in benunit_rows]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _stub_run():
    engine = _StubEngine()
    factory = _StubQRFFactory()
    result = redraw_spi_reported_uc(_frame(), engine=engine, qrf_factory=factory)
    return result, engine, factory.record


def test_reporter_predictor_and_receipt_use_claimant_couple_not_marriage():
    frame = _frame()
    benunit = frame.table("benunit")
    benunit.loc[benunit["benunit_id"].isin([101, 201]), "is_married"] = False
    benunit.loc[benunit["benunit_id"] == 102, "is_married"] = True
    marital_before = benunit["is_married"].copy()
    factory = _StubQRFFactory()

    result = redraw_spi_reported_uc(frame, engine=_StubEngine(), qrf_factory=factory)

    assert "is_married" not in factory.record["fit_predictors"]
    assert factory.record["fit_table"]["is_uc_couple"].tolist() == [1, 0, 0, 1]
    assert (
        result.reporter_transitions["spi"]["couple_with_children"]["held_reporter"] == 1
    )
    pd.testing.assert_series_equal(
        result.frame.table("benunit")["is_married"], marital_before
    )


def test_reporter_refuses_missing_relationship_inputs_before_engine_call():
    frame = _frame()
    del frame.table("person")["is_parent"]
    engine = _StubEngine()
    with pytest.raises(ValueError, match="person columns missing.*is_parent"):
        redraw_spi_reported_uc(frame, engine=engine, qrf_factory=_StubQRFFactory())
    assert engine.calls == []


def test_manifest_declares_benunit_rewrite_seed_and_exact_operations() -> None:
    stage = _stage()

    assert stage.grain == "benunit"
    assert stage.outputs == ()
    assert stage.rewrites == (UC_REPORTER_REDRAW_OUTPUT,)
    assert [operation.kind for operation in stage.operations] == [
        "derive",
        "materialize_rules_engine_predictors",
        "aggregate_person_to_benunit",
        "redraw_spi_reported_uc",
    ]
    assert stage.operations[-1].parameters["seed"] == UC_REPORTER_REDRAW_SEED
    _assert_stage_parameters(stage)


def test_stub_engine_fast_lane_reads_frs_capital_once_and_consumes_outputs() -> None:
    result, engine, _ = _stub_run()

    assert len(engine.calls) == 1
    temporary, variables, period = engine.calls[0]
    assert variables == UC_REPORTER_SCREEN_VARIABLES
    assert period == "2024"
    np.testing.assert_array_equal(
        temporary.table("benunit")["uc_reported_capital"],
        _frame().table("benunit")["frs_benunit_capital"],
    )
    for table in (result.frame.table("person"), result.frame.table("benunit")):
        assert not set(UC_REPORTER_SCREEN_VARIABLES) & set(table.columns)
    assert "uc_reported_capital" not in result.frame.table("benunit")


def test_training_mask_channel_containment_screen_and_landing_rule() -> None:
    before = _frame()
    result, _, record = _stub_run()
    after = result.frame

    assert record["seed"] == UC_REPORTER_REDRAW_SEED
    assert record["fit_index"] == [0, 1, 4, 5]
    assert record["predict_index"] == [6, 8]
    np.testing.assert_array_equal(record["fit_weights"], [2.0, 3.0, 4.0, 5.0])

    base_ids = {101, 102, 103, 104, 105, 106}
    before_base = before.table("person").loc[
        before.table("person")["person_benunit_id"].isin(base_ids)
    ]
    after_base = after.table("person").loc[
        after.table("person")["person_benunit_id"].isin(base_ids)
    ]
    pd.testing.assert_frame_equal(before_base, after_base, check_exact=True)

    person = after.table("person")
    spi_201 = person[person["person_benunit_id"].eq(201)]
    positive = spi_201[spi_201[UC_REPORTER_REDRAW_OUTPUT] > 0.0]
    assert positive["person_id"].tolist() == [2011]
    assert positive[UC_REPORTER_REDRAW_OUTPUT].tolist() == [250.0]
    assert (
        person.loc[person["person_benunit_id"].eq(202), UC_REPORTER_REDRAW_OUTPUT]
        .eq(0.0)
        .all()
    )
    amounts = _benefit_unit_reporter_amounts(person, after.table("benunit"))
    assert amounts[after.table("benunit")["benunit_id"].eq(201).to_numpy()][0] == 250.0
    assert result.training_benunits == 4
    assert result.screened_spi_benunits == 2
    assert result.screen_failed_spi_benunits == 1


def test_twin_calls_are_byte_identical_and_emit_transition_receipt() -> None:
    first, _, _ = _stub_run()
    twin, _, _ = _stub_run()

    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            first.frame.table(entity),
            twin.frame.table(entity),
            check_exact=True,
        )
    assert first.evidence() == twin.evidence()
    assert first.evidence()["reporter_transitions"]["spi"]["couple_with_children"] == {
        "promoted": 0,
        "demoted": 0,
        "held_reporter": 1,
        "held_nonreporter": 0,
    }


def test_seeded_regime_gated_qrf_twin_calls_are_byte_identical() -> None:
    first = redraw_spi_reported_uc(_frame(), engine=_StubEngine())
    twin = redraw_spi_reported_uc(_frame(), engine=_StubEngine())

    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            first.frame.table(entity),
            twin.frame.table(entity),
            check_exact=True,
        )


def test_downstream_capital_coherence_picks_up_redrawn_anchor() -> None:
    redrawn, _, _ = _stub_run()
    coherent = cohere_uc_capital(redrawn.frame)
    person = coherent.frame.table("person")
    benunit = coherent.frame.table("benunit").set_index("benunit_id")

    assert (
        person.loc[person["person_benunit_id"].eq(201), UC_REPORTER_REDRAW_OUTPUT].sum()
        == 250.0
    )
    assert benunit.loc[201, "would_claim_uc"]
    assert benunit.loc[201, "uc_reported_capital"] >= 0.0
    gate = next(
        gate
        for gate in load_country_spec("uk").gates.gates
        if gate.id == "uk_uc_capital_coherence"
    )
    verdict = UK_GATE_REGISTRY["column_implication"].evaluate(
        EvidenceContext(frame=coherent.frame),
        gate.parameters,
    )
    assert verdict.passed, verdict.failures


def test_spec_drift_guard_rejects_changed_seed() -> None:
    stage = _stage()
    redraw = stage.operations[-1]
    parameters = dict(redraw.parameters)
    parameters["seed"] = UC_REPORTER_REDRAW_SEED + 1
    drifted = replace(
        stage,
        operations=(*stage.operations[:-1], replace(redraw, parameters=parameters)),
    )

    with pytest.raises(ValueError, match="parameters drifted"):
        _assert_stage_parameters(drifted)


@pytest.mark.requires_uk
def test_live_engine_screen_reads_frs_capital() -> None:
    person = pd.DataFrame(
        {
            "person_id": [1001, 1002],
            "person_benunit_id": [101, 102],
            "person_household_id": [1, 2],
            "age": [40, 40],
            "is_benunit_head": [True, True],
            UC_REPORTER_REDRAW_OUTPUT: [0.0, 0.0],
            "employment_income": [0.0, 0.0],
            "self_employment_income": [0.0, 0.0],
            "savings_interest_income": [0.0, 0.0],
            "dividend_income": [0.0, 0.0],
            "property_income": [0.0, 0.0],
            "other_investment_income": [0.0, 0.0],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [101, 102],
            "frs_benunit_capital": [0.0, 16_001.0],
            "is_married": [False, False],
            support_channel_column("benunit"): ["frs", "frs"],
            support_clone_index_column("benunit"): [0, 0],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["LONDON", "LONDON"],
            "council_tax": [0.0, 0.0],
            "tenure_type": ["OWNED_OUTRIGHT", "OWNED_OUTRIGHT"],
            "rent": [0.0, 0.0],
            "savings": [100_000.0, 100_000.0],
            "other_residential_property_value": [0.0, 0.0],
            "non_residential_property_value": [0.0, 0.0],
            "corporate_wealth": [0.0, 0.0],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(2),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )
    materialized = _materialize_screen_inputs(
        frame,
        engine=PolicyEngineUKEngine(),
        person=person,
        benunit=benunit,
        household=household,
    )
    screen = _positive_pre_takeup_award_screen(
        materialized["uc_maximum_amount"],
        materialized["uc_income_reduction"],
        expected=2,
    )

    np.testing.assert_array_equal(screen, [True, False])


def test_transform_records_checkpoint_metadata() -> None:
    engine = _StubEngine()
    factory = _StubQRFFactory()
    transform = UKUCReporterRedrawStageTransform(
        stage=_stage(),
        engine=engine,
        qrf_factory=factory,
    )
    transform(_frame())

    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == "uc_reporter_redraw"
    assert evidence["seed"] == UC_REPORTER_REDRAW_SEED


def test_child_only_benunit_gets_its_eldest_member_as_claimant() -> None:
    """A qualifying-young-person-only benefit unit must not crash the stage.

    A 16-19 QYP heading their own unit is its only member, so no ~uc_child
    candidate exists; the unit's eldest member is its de-facto head. The
    licensed spine carries hundreds of such units, so the old total guard
    failed every full build.
    """

    from microcosm.build.uk_runtime.uc_reporter_redraw import _claimant_rows

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_benunit_id": [10, 10, 20],
            "age": [40.0, 12.0, 17.0],
        }
    )
    uc_child = np.array([False, True, True])

    claimant = _claimant_rows(person, uc_child=uc_child, sp_age=np.zeros(3, bool))

    assert claimant.tolist() == [True, False, True]


def test_claimant_prefers_working_age_by_the_engine_sp_age_flag() -> None:
    """Working age comes from the materialized is_SP_age, not a hard-coded 66."""

    from microcosm.build.uk_runtime.uc_reporter_redraw import _claimant_rows

    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_benunit_id": [10, 10],
            "age": [64.0, 60.0],
        }
    )
    uc_child = np.array([False, False])
    # The eldest adult is flagged state-pension-age by the engine (early SPA
    # cohort); the younger adult is the working-age claimant.
    claimant = _claimant_rows(person, uc_child=uc_child, sp_age=np.array([True, False]))

    assert claimant.tolist() == [False, True]


def test_positive_draw_on_a_child_claimant_refuses_at_the_landing() -> None:
    """The fail-closed half of the child-only fallback lives at the landing."""

    from microcosm.build.uk_runtime.uc_reporter_redraw import (
        _claimant_rows,
        _land_spi_draws,
    )

    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_benunit_id": [10, 20],
            "age": [40.0, 17.0],
            UC_REPORTER_REDRAW_OUTPUT: [0.0, 0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20]})
    uc_child = np.array([False, True])
    claimant = _claimant_rows(person, uc_child=uc_child, sp_age=np.zeros(2, bool))
    spi = np.array([True, True])

    with pytest.raises(ValueError, match="child claimant"):
        _land_spi_draws(
            person,
            benunit,
            spi=spi,
            claimant_rows=claimant,
            draws=np.array([500.0, 500.0]),
            uc_child=uc_child,
        )

    zero_for_child = np.array([500.0, 0.0])
    _land_spi_draws(
        person,
        benunit,
        spi=spi,
        claimant_rows=claimant,
        draws=zero_for_child,
        uc_child=uc_child,
    )
    assert person[UC_REPORTER_REDRAW_OUTPUT].tolist() == [500.0, 0.0]


def test_child_only_benunits_are_screened_out_of_the_drawn_domain() -> None:
    """uc_maximum_amount is mechanical, so the screen needs the member check.

    The engine computes a positive pre-take-up award for a QYP-only unit even
    though it cannot claim; without this restriction such units enter the
    drawn domain and positive draws land on child claimants.
    """

    from microcosm.build.uk_runtime.uc_reporter_redraw import _has_adult_member

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 30],
            "age": [40.0, 12.0, 17.0, 70.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20, 30]})
    uc_child = np.array([False, True, True, False])

    assert _has_adult_member(person, benunit, uc_child=uc_child).tolist() == [
        True,
        False,
        True,
    ]
    with pytest.raises(ValueError, match="at least one person row"):
        _has_adult_member(
            person, pd.DataFrame({"benunit_id": [10, 99]}), uc_child=uc_child
        )


def test_child_only_spi_benunit_end_to_end_is_zeroed_and_filed_as_child_only() -> None:
    """The licensed-build failure's own test: a QYP-only SPI unit through the stage.

    The stub engine pays the unit (uc_maximum_amount is mechanical), so only
    the non-child-member half of the declared screen keeps it out of the
    drawn domain; its chain fill of 300 exits at 0, and the receipt files it
    under ``child_only`` rather than the lone-parent cell.
    """

    result = redraw_spi_reported_uc(
        _frame(child_only=True), engine=_StubEngine(), qrf_factory=_StubQRFFactory()
    )

    person = result.frame.table("person")
    assert person.loc[
        person["person_benunit_id"].eq(204), UC_REPORTER_REDRAW_OUTPUT
    ].tolist() == [0.0]
    assert result.screen_failed_spi_benunits == 2
    transitions = result.evidence()["reporter_transitions"]["spi"]
    assert transitions["child_only"] == {
        "promoted": 0,
        "demoted": 1,
        "held_reporter": 0,
        "held_nonreporter": 0,
    }
    assert "single_with_children" not in transitions


def test_empty_screened_spi_domain_refuses_instead_of_zeroing_the_channel() -> None:
    """An empty draw domain is a refusal, not a success-shaped total wipe."""

    with pytest.raises(ValueError, match="no screened SPI benefit units"):
        redraw_spi_reported_uc(
            _frame(),
            engine=_StubEngine(fail_all_spi=True),
            qrf_factory=_StubQRFFactory(),
        )


def test_landing_invariants_refuse_base_writes_and_double_landings() -> None:
    from microcosm.build.uk_runtime.uc_reporter_redraw import (
        _assert_landing_invariants,
    )

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_benunit_id": [10, 20, 20],
            UC_REPORTER_REDRAW_OUTPUT: [0.0, 250.0, 0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20]})
    spi = np.array([False, True])
    draws = np.array([0.0, 250.0])
    base_before = np.array([0.0])

    _assert_landing_invariants(
        person, benunit, spi=spi, draws=draws, base_before=base_before
    )

    touched_base = person.copy()
    touched_base.loc[0, UC_REPORTER_REDRAW_OUTPUT] = 5.0
    with pytest.raises(RuntimeError, match="base-channel"):
        _assert_landing_invariants(
            touched_base, benunit, spi=spi, draws=draws, base_before=base_before
        )

    double_landed = person.copy()
    double_landed.loc[2, UC_REPORTER_REDRAW_OUTPUT] = 125.0
    double_landed.loc[1, UC_REPORTER_REDRAW_OUTPUT] = 125.0
    with pytest.raises(RuntimeError, match="exactly one"):
        _assert_landing_invariants(
            double_landed, benunit, spi=spi, draws=draws, base_before=base_before
        )

    # The sum-equals-draw branch is a consistency check between the write and
    # the draws that fed it, not an oracle — so it needs its own failing path:
    # one positive row (the one-row check passes) carrying the wrong total.
    wrong_total = person.copy()
    wrong_total.loc[1, UC_REPORTER_REDRAW_OUTPUT] = 200.0
    with pytest.raises(RuntimeError, match="disagree with the draws"):
        _assert_landing_invariants(
            wrong_total, benunit, spi=spi, draws=draws, base_before=base_before
        )
