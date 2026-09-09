"""The UK consumer half of the gate battery (microcosm#611 increment 1).

Fixtures are synthetic throughout: no UKDS unit records. The schema-3
aggregator retired in #654; these tests pin the battery-side behavior that
survived the differential receipt.
"""

from __future__ import annotations

import base64
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from microcosm.build import load_country_spec
from microcosm.build.country_spec import GatesManifest
from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryBlockedError,
    GateBatteryRun,
    GateStatus,
    _evaluate_gate,
    evaluate_phase,
    gate_signing_key_env,
    validate_gate_parameters,
)
from microcosm.build.gates import FitWeightRecord, GateResult
from microcosm.build.uk_runtime.battery_bindings import (
    UK_GATE_REGISTRY,
    UKGateBinding,
    _ledger_compile_parity_registry,
)
from microcosm.build.uk_runtime.national_frame import (
    _uk_gate_surface,
    uk_household_weight_kind,
    uk_national_frame,
)
from microcosm.build.uk_runtime.release_input_coverage import (
    UKReleaseInputColumn,
    UKReleaseInputCoverageManifest,
    load_uk_release_input_coverage_manifest,
    uk_release_input_coverage_gate,
)
from microcosm.build.uk_runtime.terminal_gates import (
    UKInputMassReference,
)
from microcosm.build.uk_runtime.weighted_integrity import UKReviewedExclusion
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import engine_tables

KEY = base64.b64encode(b"\x07" * 32).decode("ascii")
#: The shared exclusion-expiry clock, fixed inside the committed register's
#: validity window (latest approval 2026-09-05, expires 2027-02-10) so the suite
#: never drifts across an expiry boundary.
CLOCK = date(2026, 9, 15)

VALIDATE_REFERENCE = (
    "microcosm.build.uk_runtime.weighted_integrity."
    "_validate_input_mass_reference_for_descriptor"
)


@pytest.fixture(autouse=True)
def signing_env(monkeypatch) -> None:
    monkeypatch.setenv(gate_signing_key_env("uk"), KEY)


@pytest.fixture(scope="module")
def uk_gates():
    return load_country_spec("uk").gates


def test_declared_export_surface_preserves_claimants_and_matches_runtime(uk_gates):
    from microcosm.build.uk_runtime.terminal_gates import (
        UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    )

    entry = next(entry for entry in uk_gates.gates if entry.id == "uk_export_surface")
    declared = set(entry.parameters["allowed_extra_columns"])
    assert "person.is_uc_claimant" in declared
    assert declared == set(UK_ALLOWED_EXTRA_EXPORT_COLUMNS)
    battery = _run_battery(
        _tables(),
        parity=_parity(candidate_columns={"person.age", "person.is_uc_claimant"}),
    )
    outcome = next(o for o in battery.outcomes if o.entry.id == "uk_export_surface")
    assert outcome.status is GateStatus.PASSED


def _tables(*, n: int = 4, weights=None):
    if weights is None:
        weights = np.ones(n, dtype=float)
    household_ids = np.arange(1, n + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 101 + n, dtype=np.int64),
            "person_household_id": household_ids,
            "person_benunit_id": np.arange(201, 201 + n, dtype=np.int64),
            "employment_income": np.arange(1, n + 1, dtype=float),
            "universal_credit_reported": np.asarray([10.0, 0.0] * n)[:n],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": np.arange(201, 201 + n, dtype=np.int64),
            "would_claim_uc": np.asarray([True, False] * n)[:n],
            "frs_benunit_capital": np.arange(n, dtype=float),
            "uc_reported_capital": np.arange(n, dtype=float),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": np.asarray(weights, dtype=float),
            "household_is_spi_synthetic": np.arange(n) % 2 == 1,
            "household_is_capital_gains_clone": np.arange(n) % 4 >= 2,
        }
    )
    return person, benunit, household


def _coverage() -> GateResult:
    return GateResult(
        name="uk_release_input_coverage",
        passed=True,
        details={"fixture": True},
    )


def _parity(**overrides) -> SimpleNamespace:
    fields = {
        "candidate_columns": {"person.age"},
        "reference_columns": {"person.age"},
        "candidate_targets": {"ons/population"},
        "reference_targets": {"ons/population"},
        "target_relative_errors": {"ons/population": 0.01},
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _reference() -> UKInputMassReference:
    return UKInputMassReference(
        totals={"employment_income": 10.0},
        filename="enhanced_frs_2023_24.h5",
        revision="655dd07e4bb9c777b00dac044949611f1feb824f",
        sha256="584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d",
        vintage="2023_24",
    )


def _fixture_coverage_registry():
    """The UK registry with the coverage gate fed by the same fixture the
    differential harness used before retiring schema 3. The real coverage
    gate has its own dedicated tests. The fixture mints the legacy name, so
    the re-minting path stays exercised."""

    return {
        **UK_GATE_REGISTRY,
        "release_input_coverage": UKGateBinding(
            name="release_input_coverage",
            evaluator=lambda context, parameters: _coverage(),
            parameter_keys=frozenset({"check"}),
            legacy_name="uk_release_input_coverage",
            needs_frame=False,
        ),
        "take_up_signal": UKGateBinding(
            name="take_up_signal",
            evaluator=lambda context, parameters: GateResult(
                name="take_up_signal", passed=True
            ),
            parameter_keys=frozenset({"maximum_share_deviation"}),
        ),
        "enum_domain": UKGateBinding(
            name="enum_domain",
            evaluator=lambda context, parameters: GateResult(
                name="enum_domain", passed=True
            ),
            parameter_keys=frozenset({"columns"}),
        ),
    }


def _run_battery(tables, *, parity=None, fit_records=None, armed=True, clock=CLOCK):
    person, benunit, household = tables
    frame = uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2023"
    )
    artifacts: dict[str, object] = {
        "coverage_engine": object(),
        "exclusions_evaluated_on": clock,
        # The staging pipeline's two scheduled stages declare no nonnegative
        # outputs, so the nonnegative gate passes with zero required columns.
        "build_stage_names": ("frs_hmrc_retained_leaves", "hmrc_spi_income"),
    }
    if fit_records is not None:
        artifacts["fit_weight_records"] = fit_records
    if parity is not None:
        artifacts["parity_evidence"] = parity
    if armed:
        artifacts["input_mass_reference"] = _reference()
        artifacts["aggregate_admin"] = {
            "need_electricity_mean_spending": 882.91463,
            "need_gas_mean_spending": 700.3661,
            "nhs_spending_total": 202_000_000_000,
        }
    # Small synthetic totals exercise battery behavior without disclosing
    # the licensed 131-column reference (same patch as the legacy tests);
    # the binding's declared-pin check compares spec to runtime constant and
    # needs no patching.
    with patch(VALIDATE_REFERENCE, return_value=None):
        return evaluate_phase(
            load_country_spec("uk").gates,
            "terminal",
            EvidenceContext(frame=frame, artifacts=artifacts),
            registry=_fixture_coverage_registry(),
        )


class TestUKSurfaceAdapter:
    def test_surface_materializes_the_frame_not_fallbacks(self) -> None:
        # The one surviving copy of the legacy duck-attr evidence surface
        # (the national build's adapter consolidated into it at the
        # orchestration swap). Every attr must resolve to the frame's real
        # values — a gate reading household_weight_kind or time_period must
        # never see a fallback.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        surface = _uk_gate_surface(frame)
        tables = engine_tables(frame)

        pd.testing.assert_frame_equal(surface.person, tables["person"])
        pd.testing.assert_frame_equal(surface.benunit, tables["benunit"])
        pd.testing.assert_frame_equal(surface.household, tables["household"])
        assert surface.time_period == "2023"
        assert surface.household_weight_kind is uk_household_weight_kind(frame)
        assert surface.mass_log == frame.mass_log

    def _nonnegative_frame(self, *, sic: list[float] | None):
        person, benunit, household = _tables()
        if sic is not None:
            person["sic_industry_division"] = sic
        return uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )

    def test_uc_column_implication_binding_aggregates_and_checks_carrier(self) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )

        passing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )
        assert passing.passed

        frame.table("benunit").loc[0, "would_claim_uc"] = False
        frame.table("benunit").loc[1, "uc_reported_capital"] = -1.0
        failing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )
        assert not failing.passed
        assert any("must imply" in failure for failure in failing.failures)
        assert any("same sentinel" in failure for failure in failing.failures)
        assert any("must equal" in failure for failure in failing.failures)

    def test_uc_column_implication_binding_is_not_vacuous_at_amount_thresholds(
        self,
    ) -> None:
        # Adversarial-review finding 2: the configured threshold filters the
        # person-level amounts; the aggregated 0/1 indicator must always be
        # compared at 0. With the two conflated, any threshold >= 1 could
        # never flag a violation. This pins the de-conflation: a reporter
        # above a nonzero amount threshold with would_claim_uc=False must
        # still fail the gate.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )
        parameters = {**dict(entry.parameters), "threshold": 100.0}
        frame.table("person")["universal_credit_reported"] = 500.0
        frame.table("benunit")["would_claim_uc"] = False

        failing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), parameters
        )
        assert not failing.passed
        assert any("must imply" in failure for failure in failing.failures)
        assert failing.details["amount_threshold"] == 100.0
        assert failing.details["threshold"] == 0.0

    def test_uc_column_implication_binding_refuses_the_undefined_interval(
        self,
    ) -> None:
        # Adversarial-review round-2 residual (a): the -1 contract defines
        # exactly two regions — the sentinel and nonnegative amounts. A
        # corrupted -0.5 previously cleared every capital check: above the
        # bare floor, not isclose to the sentinel on either side, finite,
        # and equal to a carrier carrying the same corruption. The domain
        # predicate must refuse it, on both columns.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )
        frame.table("benunit")["uc_reported_capital"] = -0.5
        frame.table("benunit")["frs_benunit_capital"] = -0.5

        failing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )
        assert not failing.passed
        assert any("outside the declared domain" in f for f in failing.failures)
        assert failing.details["capital_domain_violation_count"] > 0
        assert failing.details["carrier_domain_violation_count"] > 0
        # The corruption must NOT be reported as a sentinel or equality
        # mismatch — those checks legitimately pass on it, which is exactly
        # why the domain check exists.
        assert failing.details["same_source_mismatch_count"] == 0

    def test_uc_column_implication_binding_refuses_near_sentinel_values(
        self,
    ) -> None:
        # #833: the isclose band around -1 admitted the top sliver of the
        # undefined interval and read it as a declared absence. Sentinel
        # equality is exact; a corrupted -1.000005 on both columns must be a
        # domain violation, not a tolerated sentinel.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        entry = next(
            gate
            for gate in load_country_spec("uk").gates.gates
            if gate.id == "uk_uc_capital_coherence"
        )
        frame.table("benunit")["uc_reported_capital"] = -1.000005
        frame.table("benunit")["frs_benunit_capital"] = -1.000005

        failing = UK_GATE_REGISTRY["column_implication"].evaluate(
            EvidenceContext(frame=frame), entry.parameters
        )
        assert not failing.passed
        assert any("outside the declared domain" in f for f in failing.failures)
        assert failing.details["capital_domain_violation_count"] > 0
        assert failing.details["carrier_domain_violation_count"] > 0
        assert failing.details["sentinel_mismatch_count"] == 0
        assert failing.details["same_source_mismatch_count"] == 0

    def test_nonnegative_binding_requires_scheduled_stage_columns(self) -> None:
        # frs_employment declares sic_industry_division nonnegative; a build
        # that scheduled the stage but lost the column must fail — the
        # missing-column path is the reason the required set is never
        # pre-filtered to present columns.
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        context = EvidenceContext(
            frame=self._nonnegative_frame(sic=None),
            artifacts={"build_stage_names": ("frs_employment",)},
        )

        result = binding.evaluate(context, {})

        assert result.passed is False
        assert "sic_industry_division" in result.failures[0]

    def test_nonnegative_binding_fails_on_negative_values(self) -> None:
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        context = EvidenceContext(
            frame=self._nonnegative_frame(sic=[1.0, -2.0, 3.0, 4.0]),
            artifacts={"build_stage_names": ("frs_employment",)},
        )

        result = binding.evaluate(context, {})

        assert result.name == "nonnegative_columns"
        assert result.passed is False
        assert (
            "sic_industry_division: 1 finite value(s) below zero" in result.failures[0]
        )

    def test_nonnegative_binding_passes_clean_scheduled_columns(self) -> None:
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        context = EvidenceContext(
            frame=self._nonnegative_frame(sic=[1.0, 0.0, 3.0, 4.0]),
            artifacts={"build_stage_names": ("frs_employment",)},
        )

        result = binding.evaluate(context, {})

        assert result.passed is True

    def test_nonnegative_binding_does_not_demand_unscheduled_stages(self) -> None:
        # The national staging build schedules only the two HMRC stages,
        # which declare no nonnegative outputs — the gate passes honestly
        # with zero required columns rather than by silent pre-filtering.
        binding = UK_GATE_REGISTRY["nonnegative_columns"]
        context = EvidenceContext(
            frame=self._nonnegative_frame(sic=None),
            artifacts={
                "build_stage_names": (
                    "frs_hmrc_retained_leaves",
                    "hmrc_spi_income",
                )
            },
        )

        result = binding.evaluate(context, {})

        assert result.passed is True

    def test_support_binding_passes_in_range_was_outputs(self) -> None:
        person, benunit, household = _tables(n=2)
        household["owned_land"] = [0.0, 100.0]
        household["cash_isa"] = [0.0, 1000.0]
        person["student_loan_balance"] = [0.0, 100.0]
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        binding = UK_GATE_REGISTRY["support"]

        result = binding.evaluate(
            EvidenceContext(frame=frame, artifacts={}),
            {
                "support_bounds_resources": [
                    "was_wealth_support_bounds.json",
                    "lcfs_consumption_support_bounds.json",
                    "etb_vat_support_bounds.json",
                    "etb_services_support_bounds.json",
                ]
            },
        )

        assert result.passed is True
        assert result.details["columns_checked"] == 3

    def test_support_binding_fails_out_of_range_was_outputs(self) -> None:
        person, benunit, household = _tables(n=1)
        household["cash_isa"] = [99_999_999.0]
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        binding = UK_GATE_REGISTRY["support"]

        result = binding.evaluate(
            EvidenceContext(frame=frame, artifacts={}),
            {"support_bounds_resource": "was_wealth_support_bounds.json"},
        )

        assert result.passed is False
        assert "cash_isa" in result.failures[0]

    def test_support_binding_checks_e6_support_resources(self) -> None:
        person, benunit, household = _tables(n=1)
        household["full_rate_vat_expenditure_rate"] = [999_999.0]
        frame = uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2023",
        )
        binding = UK_GATE_REGISTRY["support"]

        result = binding.evaluate(
            EvidenceContext(frame=frame, artifacts={}),
            {"support_bounds_resources": ["etb_vat_support_bounds.json"]},
        )

        assert result.passed is False
        assert "full_rate_vat_expenditure_rate" in result.failures[0]

    def test_aggregate_admin_binding_checks_declared_anchors(self) -> None:
        binding = UK_GATE_REGISTRY["aggregate_admin"]

        result = binding.evaluate(
            EvidenceContext(
                frame=None,
                artifacts={
                    "aggregate_admin": {
                        "need_electricity_mean_spending": 882.91463,
                        "nhs_spending_total": 202_000_000_000,
                    }
                },
            ),
            {
                "default_rtol": 0.15,
                "anchors": [
                    {
                        "name": "need_electricity_mean_spending",
                        "entity": "household",
                        "measure": "electricity_consumption",
                        "value": 882.91463,
                        "period": "2023",
                        "source": "test",
                        "family": "need_energy",
                    },
                    {
                        "name": "nhs_spending_total",
                        "entity": "person",
                        "measure": "nhs_spending",
                        "value": 202_000_000_000,
                        "period": "2025_26",
                        "source": "test",
                        "family": "nhs",
                    },
                ],
            },
        )

        assert result.passed is True
        assert result.details["anchors_checked"] == 2


class TestUKCompatibility:
    """The BE plumbing test, run over the UK spec: an empty evidence context
    resolves every declared entry to a named gap and blocks the release."""

    def test_the_uk_spec_runs_as_declared_with_named_gaps(
        self, tmp_path, uk_gates
    ) -> None:
        run = GateBatteryRun(
            uk_gates,
            release_id="uk-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry=UK_GATE_REGISTRY,
        )
        run.run_phase("preflight", EvidenceContext())
        run.run_phase("assembled", EvidenceContext())
        run.run_phase("transferred", EvidenceContext())
        run.run_phase("terminal", EvidenceContext())
        report = run.report_payload()

        assert set(report["gates"]) == {entry.id for entry in uk_gates.gates}
        assert {gate["status"] for gate in report["gates"].values()} == {
            "evidence_absent"
        }
        assert report["shippable"] is False
        with pytest.raises(GateBatteryBlockedError):
            run.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)

    def test_missing_evidence_names_its_keys(self, uk_gates) -> None:
        phase = evaluate_phase(
            uk_gates, "terminal", EvidenceContext(), registry=UK_GATE_REGISTRY
        )
        reasons = {o.entry.id: o.reason for o in phase.outcomes}
        assert reasons["uk_weights_audit"] == ("missing evidence: fit_weight_records")
        assert reasons["uk_input_mass_parity"] == (
            "missing evidence: frame, exclusions_evaluated_on, input_mass_reference"
        )
        assert reasons["uk_degenerate_release_surface"] == (
            "missing evidence: frame, exclusions_evaluated_on"
        )


class TestBatteryRegressions:
    def test_fully_armed_battery_evaluates_gate_for_gate(self) -> None:
        battery = _run_battery(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
        )

        by_id = {o.entry.id: o for o in battery.outcomes}
        passed = [
            entry_id for entry_id, o in by_id.items() if o.status is GateStatus.PASSED
        ]
        # 11 as on main (uk_nonnegative_columns passes with zero required
        # columns — the scheduled stages declare none), the take-up signal
        # gate, the E5 support gate, the E6 aggregate-admin gate, and the E8
        # student-loan enum gate; their evaluators have direct tests. The BRMA
        # enum gate is no longer among them: it moved to the spine battery's
        # assembled boundary, where its column is first written.
        # 15 before this lane, plus uk_uc_capital_coherence from #829 and the
        # UC-deduction enum gate from #685, plus
        # the two frame-only local weight diagnostics that also pass in this
        # unscoped compatibility probe. The local ladder gate fails because
        # this national fixture deliberately carries no ladder columns; the
        # three evidence-backed local arms are named gaps below.
        assert len(passed) == 19
        qrf = by_id["uk_qrf_tail_concentration"]
        assert qrf.status is GateStatus.FAILED
        assert "declared QRF output is absent" in qrf.result.failures[0]

    def test_empty_fit_records_fail_closed(self) -> None:
        # Present-but-empty is not absent: a fit stage that ran and emitted
        # nothing is a failed audit, never a vacuous pass.
        battery = _run_battery(_tables(), fit_records=())

        audit = {o.entry.id: o for o in battery.outcomes}["uk_weights_audit"]
        assert audit.status is GateStatus.FAILED
        assert "an absent audit is not a passing audit" in (audit.result.failures[0])

    def test_seeded_defects_fail_the_expected_gates(self) -> None:
        blown = _tables(weights=[1.0, 1.0, 1.0, 1.0e9])
        seeded_parity = _parity(
            candidate_columns={"person.age", "person.unreviewed_extra"},
            target_relative_errors={"ons/population": -0.40},
        )
        battery = _run_battery(
            blown,
            parity=seeded_parity,
            fit_records=(FitWeightRecord("spi_qrf", "none"),),
        )

        failed = {o.entry.id for o in battery.outcomes if o.status is GateStatus.FAILED}
        assert {
            "uk_weight_ratio",
            "uk_weights_audit",
            "uk_export_surface",
            "uk_target_fit",
        } <= failed


class TestUnevidencedArms:
    """Missing evidence is explicit; it blocks release candidates, plus any
    entry whose manifest declares absence non-excusable in every posture
    (``uk_weights_audit`` — "an absent audit is not a passing audit", the
    legacy strictness ported during the #654 retirement)."""

    def test_battery_records_evidence_absent(self, uk_gates) -> None:
        battery = _run_battery(_tables(), armed=False)

        absent = {
            o.entry.id: o.reason
            for o in battery.outcomes
            if o.status is GateStatus.EVIDENCE_ABSENT
        }
        assert set(absent) == {
            "uk_weights_audit",
            "uk_export_surface",
            "uk_calibration_reference_coverage",
            "uk_target_surface",
            "uk_target_fit",
            "uk_input_mass_parity",
            "uk_aggregate_admin",
            "uk_local_area_support",
            "uk_local_target_fit",
            "uk_local_per_family_fit",
        }
        for reason in absent.values():
            assert reason.startswith("missing evidence: ")

        # The audit's absence blocks even the default posture — its status
        # stays honestly evidence_absent; only the enforcement is strict.
        default_blocked = {
            o.entry.id for o in battery.blocking_outcomes(release_candidate=False)
        }
        assert default_blocked == {
            "uk_local_geography_ladder_post_calibration",
            "uk_qrf_tail_concentration",
            "uk_weights_audit",
        }
        blocked = {
            o.entry.id for o in battery.blocking_outcomes(release_candidate=True)
        }
        assert blocked == {
            *set(absent),
            "uk_local_geography_ladder_post_calibration",
            "uk_qrf_tail_concentration",
        }

    def test_absent_fit_evidence_is_named(self) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        battery = evaluate_phase(
            load_country_spec("uk").gates,
            "terminal",
            EvidenceContext(frame=frame, artifacts={"coverage_engine": object()}),
            registry=_fixture_coverage_registry(),
        )
        audit = {o.entry.id: o for o in battery.outcomes}["uk_weights_audit"]
        assert audit.status is GateStatus.EVIDENCE_ABSENT
        assert audit.reason == "missing evidence: fit_weight_records"


class TestExclusionDiscipline:
    """One expiry clock, a committed register of record, a loud override."""

    EXCLUSION_GATES = (
        "uk_degenerate_release_surface",
        "uk_input_mass_parity",
        "uk_qrf_tail_concentration",
    )

    def test_every_exclusion_gate_shares_the_injected_clock(self) -> None:
        battery = _run_battery(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
        )
        by_id = {o.entry.id: o for o in battery.outcomes}
        stamps = {
            entry_id: by_id[entry_id].result.details["exclusions_evaluated_on"]
            for entry_id in self.EXCLUSION_GATES
        }
        assert set(stamps.values()) == {CLOCK.isoformat()}, stamps

    def test_an_expired_register_fails_closed(self) -> None:
        battery = _run_battery(
            _tables(),
            parity=_parity(),
            fit_records=(FitWeightRecord("spi_qrf", "importance"),),
            clock=date(2027, 3, 1),
        )
        failed = {o.entry.id for o in battery.outcomes if o.status is GateStatus.FAILED}
        assert {
            "uk_degenerate_release_surface",
            "uk_input_mass_parity",
            "uk_qrf_tail_concentration",
        } <= failed

    def test_review_override_is_loud_in_the_evidence_payload(self) -> None:
        binding = UK_GATE_REGISTRY["degenerate_release_surface"]
        committed = binding.evidence_payload(
            EvidenceContext(artifacts={"exclusions_evaluated_on": CLOCK}), {}
        )
        assert committed["exclusions_policy"] == "committed"
        assert "household.source_year" in committed["reviewed_exclusions"]

        overridden = binding.evidence_payload(
            EvidenceContext(
                artifacts={
                    "exclusions_evaluated_on": CLOCK,
                    "reviewed_degenerate_exclusions": {},
                }
            ),
            {},
        )
        assert overridden["exclusions_policy"] == "override"
        assert overridden["reviewed_exclusions"] == {}
        assert overridden != committed, "an override must move the evidence digest"

    def test_resupplying_the_committed_register_is_not_an_override(self) -> None:
        # The label follows content, not the artifact's presence: a caller
        # routing the committed register through the artifact (as a driver
        # preflight might) runs the committed policy and must say so — and
        # a review file byte-identical to the register is no deviation.
        from microcosm.build.uk_runtime.terminal_gates import (
            uk_default_degenerate_reviewed_exclusions,
        )

        binding = UK_GATE_REGISTRY["degenerate_release_surface"]
        committed = binding.evidence_payload(
            EvidenceContext(artifacts={"exclusions_evaluated_on": CLOCK}), {}
        )
        resupplied = binding.evidence_payload(
            EvidenceContext(
                artifacts={
                    "exclusions_evaluated_on": CLOCK,
                    "reviewed_degenerate_exclusions": dict(
                        uk_default_degenerate_reviewed_exclusions()
                    ),
                }
            ),
            {},
        )
        assert resupplied == committed
        assert resupplied["exclusions_policy"] == "committed"

    def test_a_datetime_clock_is_refused(self, uk_gates) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = {e.id: e for e in uk_gates.gates}["uk_degenerate_release_surface"]
        binding = UK_GATE_REGISTRY["degenerate_release_surface"]
        result = _evaluate_gate(
            "degenerate_release_surface",
            lambda: binding.evaluate(
                EvidenceContext(
                    frame=frame,
                    artifacts={"exclusions_evaluated_on": datetime(2026, 9, 1, 12, 0)},
                ),
                entry.parameters,
            ),
        )
        assert result.passed is False
        assert "shared clock" in result.details["evaluation_error"]["message"]


class _TerminalCoverageEngine:
    """Minimal engine surface the coverage gate consults."""

    def default_values(self, names):
        return {name: 0.0 for name in names}

    def variables(self):
        return ["employment_income"]

    def variable_entities(self, names):
        return {name: "person" for name in names}


class TestTerminalCoverageBinding:
    def test_terminal_mode_threads_frame_engine_and_manifest(self) -> None:
        # The differential test feeds fixture coverage to both sides (the
        # real gate needs the committed licensed manifest), so this is the
        # one place the binding's terminal branch runs the real gate: same
        # frame surface, engine, and manifest as a direct call, re-minted
        # onto the declared neutral name.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        engine = _TerminalCoverageEngine()
        manifest = UKReleaseInputCoverageManifest(
            reference={"source": "test"},
            candidate_evidence={"source": "test"},
            columns=(UKReleaseInputColumn("employment_income", "required"),),
            family_coverage={},
        )
        binding = UK_GATE_REGISTRY["release_input_coverage"]
        result = binding.evaluate(
            EvidenceContext(
                frame=frame,
                artifacts={
                    "coverage_engine": engine,
                    "coverage_manifest": manifest,
                },
            ),
            {},
        )
        direct = uk_release_input_coverage_gate(
            _uk_gate_surface(frame), engine, manifest=manifest
        )

        assert direct.name == "uk_release_input_coverage"
        assert result.name == "release_input_coverage"
        assert direct.passed is True
        assert result.passed == direct.passed
        assert result.failures == direct.failures
        assert dict(result.details) == dict(direct.details)


class TestPreflightBindings:
    def test_preflight_passes_with_the_committed_manifest(self, uk_gates) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        phase = evaluate_phase(
            uk_gates,
            "preflight",
            EvidenceContext(
                artifacts={
                    "coverage_engine": None,
                    "build_stage_names": tuple(sorted(manifest.required_build_stages)),
                }
            ),
            registry=UK_GATE_REGISTRY,
        )
        statuses = {o.entry.id: o.status for o in phase.outcomes}
        assert statuses == {
            "uk_release_input_coverage_manifest_current": GateStatus.PASSED,
            "uk_release_family_build_stages": GateStatus.PASSED,
            "uk_ledger_compile_parity_production_2023": (GateStatus.EVIDENCE_ABSENT),
            "uk_ledger_compile_parity_incumbent_2025": GateStatus.EVIDENCE_ABSENT,
            "uk_ledger_compile_parity_local_incumbent_2025": (
                GateStatus.EVIDENCE_ABSENT
            ),
            "uk_target_surface_local_default_2025": GateStatus.EVIDENCE_ABSENT,
        }

    def test_ledger_compile_parity_selects_the_declared_period_registry(self) -> None:
        registry_2023 = TargetRegistry(
            (
                TargetSpec(
                    name="target",
                    entity="household",
                    measure="measure",
                    value=1.0,
                    period=2023,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        registry_2025 = TargetRegistry(
            (
                TargetSpec(
                    name="target",
                    entity="household",
                    measure="measure",
                    value=2.0,
                    period=2025,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        context = EvidenceContext(
            artifacts={
                "uk_ledger_compiled_registries": {
                    2023: registry_2023,
                    2025: registry_2025,
                }
            }
        )

        assert _ledger_compile_parity_registry(context, 2023) is registry_2023
        assert _ledger_compile_parity_registry(context, 2025) is registry_2025

    def test_ledger_compile_parity_can_select_the_local_registry_artifact(
        self,
    ) -> None:
        national = TargetRegistry(
            (
                TargetSpec(
                    name="target",
                    entity="household",
                    measure="measure",
                    value=1.0,
                    period=2025,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        local = TargetRegistry(
            (
                TargetSpec(
                    name="local_target",
                    entity="household",
                    measure="measure",
                    value=2.0,
                    period=2025,
                    source="synthetic",
                ),
            ),
            country="uk",
        )
        context = EvidenceContext(
            artifacts={
                "uk_ledger_compiled_registries": {2025: national},
                "uk_ledger_compiled_local_registries": {2025: local},
            }
        )

        assert _ledger_compile_parity_registry(context, 2025) is national
        assert (
            _ledger_compile_parity_registry(
                context,
                2025,
                registry_artifact="uk_ledger_compiled_local_registries",
            )
            is local
        )

    def test_local_default_target_surface_uses_metric_names_and_signed_deferrals(
        self, uk_gates
    ) -> None:
        spec = load_country_spec("uk")
        registry = TargetRegistry(
            (
                TargetSpec(
                    name=reference.name,
                    entity=reference.entity,
                    measure=reference.measure,
                    value=1.0,
                    period=2025,
                    source="synthetic",
                )
                for reference in spec.local_target_references
            ),
            country="uk",
        )
        entry = {entry.id: entry for entry in uk_gates.gates}[
            "uk_target_surface_local_default_2025"
        ]
        result = UK_GATE_REGISTRY["target_surface"].evaluate(
            EvidenceContext(
                artifacts={"uk_ledger_compiled_local_registries": {2025: registry}}
            ),
            entry.parameters,
        )

        assert result.passed is True
        assert result.details["candidate_targets"] == 19_419
        assert result.details["reference_targets"] == 22_530
        # 2,100 signed area deferrals from the membership file plus the 1,011
        # ladder-derived households@area rows: census_households binds from the
        # OA-ladder artifact (microcosm#542), never from Chronicle facts, so the
        # in-code default surface excludes it by rule rather than by absence.
        exclusions = result.details["reviewed_exclusions"]
        assert len(exclusions) == 2_100 + 1_011
        households = [
            name for name in exclusions if str(name).startswith("households@")
        ]
        assert len(households) == 1_011
        assert result.details["missing_reference_targets"] == []

    def test_missing_required_stage_fails_with_the_assertion_text(
        self, uk_gates
    ) -> None:
        phase = evaluate_phase(
            uk_gates,
            "preflight",
            EvidenceContext(
                artifacts={"coverage_engine": None, "build_stage_names": ()}
            ),
            registry=UK_GATE_REGISTRY,
        )
        stages = {o.entry.id: o for o in phase.outcomes}[
            "uk_release_family_build_stages"
        ]
        assert stages.status is GateStatus.FAILED
        assert "omits required release family stage(s)" in stages.result.failures[0]


def test_area_support_binding_resolves_register_and_rejects_expired_entry(
    monkeypatch,
    uk_gates,
) -> None:
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    entry = {entry.id: entry for entry in uk_gates.gates}["uk_local_area_support"]
    binding = UK_GATE_REGISTRY["area_support"]
    # The committed register carries the two micro local authorities Maria
    # excluded on the measured K=4 shortfalls (microcosm#762 A4); the
    # synthetic roster carries them below the floor so the committed entries
    # apply instead of reading as unknown.
    support = pd.DataFrame(
        {
            "geography_level": [
                "constituency",
                "local_authority",
                "local_authority",
                "local_authority",
            ],
            "area_code": ["E14000001", "E06000001", "E06000053", "E09000001"],
            "assigned_households": [50, 50, 7, 14],
            "nonzero_households": [50, 50, 7, 14],
            "effective_sample_size": [50.0, 50.0, 6.8, 12.1],
            "nonzero_source_households": [50, 50, 7, 14],
        }
    )
    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, _levels: {
            "constituency": ("E14000001",),
            "local_authority": ("E06000001", "E06000053", "E09000001"),
        },
    )
    context = EvidenceContext(
        artifacts={
            "uk_area_support_summary": support,
            "exclusions_evaluated_on": CLOCK,
        }
    )

    committed = binding.evaluate(context, entry.parameters)
    assert committed.passed is True
    assert set(committed.details["reviewed_exclusions"]) == {
        "local_authority/E06000053",
        "local_authority/E09000001",
    }
    assert committed.details["excluded_area_count"] == 2
    exclusions = battery_bindings.load_uk_reviewed_exclusion_register(
        None,
        resource="local_area_support_exclusions.json",
    )
    a14_suffix = (
        " Per microcosm#762 A14 the authority's own local-authority cells are "
        "signed-deferred (`local_authority_support_floor_excluded`)."
    )
    assert all(record.reason.endswith(a14_suffix) for record in exclusions.values())

    monkeypatch.setattr(
        battery_bindings,
        "load_uk_reviewed_exclusion_register",
        lambda *_args, **_kwargs: {
            "constituency/E14000001": UKReviewedExclusion(
                reason="synthetic expired support review",
                approved_by="reviewer",
                adjudication="microcosm#762",
                approved_on="2026-01-01",
                expires_on="2026-02-01",
            )
        },
    )
    expired = binding.evaluate(context, entry.parameters)
    assert expired.passed is False
    assert expired.details["invalid_reviewed_exclusions"] == ["constituency/E14000001"]
    assert "outside its approval window" in expired.failures[0]


class TestParameterVocabulary:
    def test_every_declared_uk_parameter_is_inside_its_binding_vocabulary(
        self, uk_gates
    ) -> None:
        # The whole shipped spec arms against the shipped registry: a
        # parameter either routes into its gate or the battery refuses to
        # start. Guards the vocabulary against drifting behind gates.json.
        validate_gate_parameters(uk_gates, UK_GATE_REGISTRY)

    def test_a_stray_preflight_parameter_is_refused_at_arm_time(self) -> None:
        # The preflight coverage evaluator reads `check` selectively rather
        # than splatting, so before vocabulary validation an extra key here
        # was the one place a declared parameter could ship inside
        # policy_sha256 while governing nothing.
        manifest = GatesManifest.from_mapping(
            {
                "version": 1,
                "country": "uk",
                "policy": "test battery",
                "phases": ["preflight"],
                "gates": [
                    {
                        "id": "uk_manifest_current",
                        "gate": "release_input_coverage",
                        "phase": "preflight",
                        "criticality": "release_blocking",
                        "parameters": {
                            "check": "manifest_current",
                            "cheks": "manifest_current",  # typo'd on purpose
                        },
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match=r"'uk_manifest_current'.*cheks"):
            evaluate_phase(
                manifest,
                "preflight",
                EvidenceContext(artifacts={"coverage_engine": None}),
                registry=UK_GATE_REGISTRY,
            )


class TestBindingUnits:
    def test_only_the_declared_legacy_name_is_reminted(self) -> None:
        binding = UKGateBinding(
            name="tail_concentration",
            evaluator=lambda context, parameters: GateResult(
                name="qrf_tail_concentration", passed=True, details={}
            ),
            legacy_name="qrf_tail_concentration",
        )
        result = binding.evaluate(EvidenceContext(), {})
        assert result.name == "tail_concentration"

        impostor = UKGateBinding(
            name="tail_concentration",
            evaluator=lambda context, parameters: GateResult(
                name="weight_ess", passed=True, details={}
            ),
            legacy_name="qrf_tail_concentration",
        )
        checked = _evaluate_gate(
            "tail_concentration",
            lambda: impostor.evaluate(EvidenceContext(), {}),
        )
        assert checked.passed is False
        assert checked.details["returned_gate"] == "weight_ess"

    def test_frozen_spec_parameters_construct_reviewed_strata(self, uk_gates) -> None:
        # The declared parameters arrive frozen (mappings as proxies, lists
        # as tuples); the binding must build the reviewed declarations from
        # exactly that shape.
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = {e.id: e for e in uk_gates.gates}["uk_zero_weight_strata"]
        binding = UK_GATE_REGISTRY["zero_weight_strata"]
        result = binding.evaluate(EvidenceContext(frame=frame), entry.parameters)
        assert result.passed is True

    def test_unknown_declaration_key_fails_closed(self, uk_gates) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = {e.id: e for e in uk_gates.gates}["uk_zero_weight_strata"]
        seeded = {
            "declarations": [
                {**dict(declaration), "surprise": 1}
                for declaration in entry.parameters["declarations"]
            ]
        }
        binding = UK_GATE_REGISTRY["zero_weight_strata"]
        result = _evaluate_gate(
            "zero_weight_strata",
            lambda: binding.evaluate(EvidenceContext(frame=frame), seeded),
        )
        assert result.passed is False
        assert "unknown keys ['surprise']" in result.failures[0]

    def test_unknown_declared_parameter_fails_closed(self, uk_gates) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        binding = UK_GATE_REGISTRY["weight_ess"]
        result = _evaluate_gate(
            "weight_ess",
            lambda: binding.evaluate(
                EvidenceContext(frame=frame),
                {"minimum_ess_fraction": 0.01, "unreviewed_knob": 1},
            ),
        )
        assert result.passed is False
        assert result.details["evaluation_error"]["type"] == "TypeError"

    def test_drifted_declared_pin_fails_closed(self, uk_gates) -> None:
        person, benunit, household = _tables()
        frame = uk_national_frame(
            person=person, benunit=benunit, household=household, time_period="2023"
        )
        entry = {e.id: e for e in uk_gates.gates}["uk_input_mass_parity"]
        drifted = dict(entry.parameters)
        drifted["reference_registry"] = {
            name: (
                {
                    **payload,
                    "totals_sha256": "0" * 64,
                }
                if name == "efrs-post-calibration"
                else payload
            )
            for name, payload in drifted["reference_registry"].items()
        }
        binding = UK_GATE_REGISTRY["input_mass_parity"]
        result = _evaluate_gate(
            "input_mass_parity",
            lambda: binding.evaluate(
                EvidenceContext(
                    frame=frame,
                    artifacts={
                        "input_mass_reference": _reference(),
                        "exclusions_evaluated_on": CLOCK,
                    },
                ),
                drifted,
            ),
        )
        assert result.passed is False
        assert "reference_registry" in result.failures[0]


@pytest.mark.parametrize(
    "gate_id,weights,relative_error",
    [
        ("uk_local_target_fit", [1.0] * 200, 0.30),
        ("uk_local_per_family_fit", [1.0] * 200, 0.30),
        ("uk_local_weight_ratio", [1000.0] + [1.0] * 199, 0.0),
        ("uk_local_weight_ess", [1e9] + [1.0] * 199, 0.0),
    ],
)
def test_measured_local_quality_failure_blocks_release(
    gate_id, weights, relative_error, tmp_path
):
    """Use real program errors and weights, then exercise battery enforcement."""
    from microcosm.build.uk_runtime.calibration_run import uk_scoped_gate_manifest

    manifest = uk_scoped_gate_manifest(
        frozenset({gate_id}), phases=("terminal",), policy_suffix="local_candidate"
    )
    person, benunit, household = _tables(n=len(weights), weights=weights)
    frame = uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2025"
    )
    diagnostics = pd.DataFrame(
        [
            {
                "family": "obr",
                "area_code": "UK",
                "metric": f"program_{i}",
                "relative_error": relative_error,
            }
            for i in range(5)
        ]
    )
    report_path = tmp_path / "gates.json"
    battery = GateBatteryRun(
        manifest,
        release_id="synthetic-quality-refusal",
        report_path=report_path,
        release_candidate=True,
        registry=UK_GATE_REGISTRY,
    )
    battery.run_phase(
        "terminal",
        EvidenceContext(
            frame=frame, artifacts={"local_target_diagnostics": diagnostics}
        ),
    )
    with pytest.raises(GateBatteryBlockedError):
        battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
    report = battery.report_payload()
    assert report["shippable"] is False
    assert report["gates"][gate_id]["status"] == "failed"
    assert report["gates"][gate_id]["criticality"] == "release_blocking"
