"""Behavioral contracts for the calibration solver.

Each test pins a property the charter promises: calibration reduces target loss
and hits feasible targets, produces CALIBRATED weights, conserves declared mass
only when asked, respects the hard weight-ratio bound (and so cannot detonate a
landmine), stacks multi-period targets over one weight vector, and prunes toward
a record budget with L0.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from microcosm.calibrate import (
    CalibrationResult,
    L0RefitResult,
    Target,
    TargetSet,
    calibrate,
    calibrate_l0_refit,
    default_target_loss_scales,
    effective_sample_size,
    refit_l0_selection,
    relative_error_loss,
    select_exact_k,
)
from microcosm.calibrate import solve as solve_module
from microcosm.calibrate.gates import hard_concrete_open_probability_threshold
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _income_target(truth: float, factor: float) -> Target:
    return Target(
        name="income",
        entity="household",
        value=truth * factor,
        measure="income",
    )


def _population_target(truth: float, factor: float) -> Target:
    return Target(
        name="population",
        entity="household",
        value=truth * factor,
        measure="household_count",
    )


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def _l2_concentration_fixture() -> tuple[Frame, TargetSet, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 160
    income = rng.lognormal(10.5, 1.0, n)
    is_renter = (income < np.quantile(income, 0.35)).astype(float)
    initial_weights = np.full(n, 1000.0)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": range(n),
                    "income": income,
                    "is_renter": is_renter,
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                value=float((income * initial_weights).sum()) * 1.3,
                measure="income",
            ),
            Target(
                name="renters",
                entity="household",
                value=float((is_renter * initial_weights).sum()) * 0.7,
                measure="is_renter",
            ),
        )
    )
    return frame, targets, initial_weights


def _exact_k_design_fixture(
    initial_weights: np.ndarray,
    measure_values: np.ndarray,
) -> tuple[Frame, TargetSet, CalibrationResult]:
    """Build a tiny design-weighted frame and a valid L0 selection receipt."""
    weights = np.asarray(initial_weights, dtype=np.float64)
    measure = np.asarray(measure_values, dtype=np.float64)
    n = len(weights)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": range(n),
                    "adjudicated_measure": measure,
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="adjudicated",
                entity="household",
                value=float(weights @ measure),
                measure="adjudicated_measure",
                tolerance=0.01,
            ),
        )
    )
    selection = calibrate(
        frame,
        targets,
        epochs=1,
        seed=0,
        mass="conserve",
        l0_lambda=1e-8,
    )
    return frame, targets, selection


def test_method_prox_l1_selects_sparse_subset() -> None:
    """method='prox' with an L1 penalty drives a strict subset to exact zero."""
    frame, targets, w0 = _l2_concentration_fixture()
    n = w0.size
    dense = calibrate(frame, targets, method="prox", l1_lambda=0.0, epochs=300, seed=0)
    assert dense.n_nonzero == n  # no penalty -> nothing pruned
    assert dense.final_loss < dense.initial_loss  # the prox path actually fits

    sparse = calibrate(frame, targets, method="prox", l1_lambda=0.5, epochs=300, seed=0)
    weights = np.asarray(sparse.weights)
    assert 0 < sparse.n_nonzero < n  # a budget-controllable sparse subset
    assert np.all(weights >= 0.0)  # weights stay non-negative
    assert (weights == 0.0).any()  # and exactly zero, not merely small


def test_l1_lambda_is_budget_monotone() -> None:
    """A larger L1 penalty retains no more records -- the budget knob is monotone."""
    frame, targets, _ = _l2_concentration_fixture()
    counts = [
        calibrate(
            frame, targets, method="prox", l1_lambda=lam, epochs=300, seed=0
        ).n_nonzero
        for lam in (0.3, 0.5, 1.0)
    ]
    assert counts[0] >= counts[1] >= counts[2]


def test_apg_method_alias_normalizes_to_adam() -> None:
    """Existing configs can pass 'apg', but manifests record the real Adam path."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.deprecated_call(match="method='apg' is deprecated"):
        result = calibrate(frame, targets, method="apg", epochs=10, seed=0)

    assert result.options["method"] == "adam"


def test_l1_lambda_requires_prox_method() -> None:
    """l1_lambda needs the proximal solver; Adam cannot soft-threshold to zero."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.raises(ValueError, match="l1_lambda requires method='prox'"):
        calibrate(frame, targets, method="adam", l1_lambda=0.5, epochs=10, seed=0)


def test_method_prox_rejects_l2_lambda() -> None:
    """The prox path must not record an L2 objective it does not optimize."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.raises(ValueError, match="method='prox' does not implement l2_lambda"):
        calibrate(frame, targets, method="prox", l2_lambda=0.001, epochs=10, seed=0)


def test_method_prox_conserve_cap_infeasible_names_l1_remedy() -> None:
    """The prox projection error must not tell users to tune L0 knobs."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.raises(
        ValueError,
        match=(
            "L1 proximal pruning.*mass='conserve'.*max_weight_ratio=.*lower l1_lambda"
        ),
    ):
        calibrate(
            frame,
            targets,
            method="prox",
            l1_lambda=0.5,
            epochs=300,
            seed=0,
            mass="conserve",
            max_weight_ratio=1.05,
        )


def test_method_prox_zero_target_all_zero_raises_named_error() -> None:
    """The prox path names all-zero optima before the frame kernel rejects them."""
    initial_weights = np.full(8, 100.0)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(8), "person_household_id": range(8)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(8), "household_count": np.ones(8)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="zero_population",
                entity="household",
                value=0.0,
                measure="household_count",
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="method='prox' returned every calibrated weight as zero",
    ):
        calibrate(
            frame,
            targets,
            method="prox",
            l1_lambda=0.0,
            learning_rate=1.0,
            epochs=1,
            seed=0,
            target_loss_cap=1_000_000.0,
        )


def test_method_prox_l1_uses_mean_ratio_penalty_scale() -> None:
    """The prox shrink matches l1_lambda * mean(weight / initial_weight)."""
    initial_weights = np.array([100.0, 200.0, 300.0, 400.0])
    n = initial_weights.size
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(n), "household_count": np.ones(n)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=float(initial_weights.sum()),
                measure="household_count",
            ),
        )
    )
    learning_rate = 0.2
    l1_lambda = 0.8

    result = calibrate(
        frame,
        targets,
        method="prox",
        l1_lambda=l1_lambda,
        learning_rate=learning_rate,
        epochs=1,
        seed=0,
    )

    expected_ratio = 1.0 - (learning_rate * l1_lambda / n)
    np.testing.assert_allclose(result.weights / initial_weights, expected_ratio)
    assert result.options["l1_penalty"] == "mean_initial_weight_ratio_abs"


def test_method_prox_l1_uses_effective_step_with_nonzero_gradient() -> None:
    """The L1 threshold uses the RMS-normalized smooth step, not raw lr."""
    initial_weights = np.array([100.0, 200.0, 300.0, 400.0])
    n = initial_weights.size
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(n), "household_count": np.ones(n)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    target = float(initial_weights.sum() * 2.0)
    targets = TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=target,
                measure="household_count",
            ),
        )
    )
    learning_rate = 0.2
    l1_lambda = 0.8

    result = calibrate(
        frame,
        targets,
        method="prox",
        l1_lambda=l1_lambda,
        learning_rate=learning_rate,
        epochs=1,
        seed=0,
    )

    grad = -initial_weights / target
    step_size = learning_rate / np.sqrt(np.mean(grad**2))
    expected_ratio = 1.0 - (step_size * grad) - (step_size * l1_lambda / n)
    np.testing.assert_allclose(result.weights / initial_weights, expected_ratio)


def test_calibration_reduces_loss_and_hits_feasible_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    # Shift both targets by the same factor so a uniform rescale hits both.
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.4),
            _income_target(truths["income"], 1.4),
        )
    )
    result = calibrate(frame, targets, epochs=400, seed=0)
    assert result.gate_open_probabilities is None
    assert result.final_loss < result.initial_loss * 0.01
    for diag in result.diagnostics:
        assert abs(diag.relative_error) < 0.01  # within 1%


def test_calibrated_weights_are_calibrated_kind(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet((_population_target(truths["population"], 1.2),))
    result = calibrate(frame, targets, epochs=200, seed=0)
    assert result.frame.resolve_weights("household").kind is WeightKind.CALIBRATED


def test_conserve_mass_holds_total_free_mass_moves(feasible_frame) -> None:
    frame, truths = feasible_frame()
    initial_total = frame.resolve_weights("household").values.sum()
    # A population target above the current total: free mass should grow to it.
    targets = TargetSet((_population_target(truths["population"], 1.5),))

    free = calibrate(frame, targets, epochs=300, seed=0, mass="free")
    free_total = free.frame.resolve_weights("household").values.sum()
    assert free_total > initial_total * 1.3  # moved toward the larger target

    conserved = calibrate(frame, targets, epochs=300, seed=0, mass="conserve")
    conserved_total = conserved.frame.resolve_weights("household").values.sum()
    assert abs(conserved_total - initial_total) / initial_total < 1e-6


def test_target_loss_weights_prioritize_conflicting_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    low = truths["population"] * 0.5
    high = truths["population"] * 1.5
    targets = TargetSet(
        (
            Target(
                name="population_low",
                entity="household",
                value=low,
                measure="household_count",
            ),
            Target(
                name="population_high",
                entity="household",
                value=high,
                measure="household_count",
            ),
        )
    )

    uniform = calibrate(frame, targets, epochs=300, seed=0)
    prioritized = calibrate(
        frame,
        targets,
        epochs=300,
        seed=0,
        target_loss_weights=np.asarray([1.0, 1_000.0]),
    )

    uniform_total = uniform.weights.sum()
    prioritized_total = prioritized.weights.sum()
    assert abs(prioritized_total - high) < abs(uniform_total - high)
    assert prioritized.options["target_loss_weights"]["kind"] == "provided"


def test_target_loss_weights_follow_skipped_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="missing_measure",
                entity="household",
                value=1.0,
                measure="missing_measure",
            ),
            _population_target(truths["population"], 1.0),
        )
    )

    result = calibrate(
        frame,
        targets,
        epochs=50,
        seed=0,
        target_loss_weights=np.asarray([1_000.0, 1.0]),
    )

    assert [skip.target.name for skip in result.skipped] == ["missing_measure"]
    assert result.options["target_loss_weights"]["n"] == 1


def test_target_loss_weights_must_survive_skipped_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="missing_measure",
                entity="household",
                value=1.0,
                measure="missing_measure",
            ),
            _population_target(truths["population"], 1.0),
        )
    )

    with pytest.raises(ValueError, match="positive total weight"):
        calibrate(
            frame,
            targets,
            epochs=50,
            seed=0,
            target_loss_weights=np.asarray([1.0, 0.0]),
        )


def test_target_loss_scales_follow_skipped_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="missing_measure",
                entity="household",
                value=1.0,
                measure="missing_measure",
            ),
            _population_target(truths["population"], 1.0),
        )
    )

    result = calibrate(
        frame,
        targets,
        epochs=50,
        seed=0,
        target_loss_scales=np.asarray([10.0, 20.0]),
    )

    assert [skip.target.name for skip in result.skipped] == ["missing_measure"]
    assert result.options["target_loss_scales"]["kind"] == "provided"
    assert result.options["target_loss_scales"]["n"] == 1
    assert result.options["target_loss_scales"]["min"] == pytest.approx(20.0)
    assert result.options["target_loss_scales"]["max"] == pytest.approx(20.0)


def test_target_loss_scales_must_survive_skipped_targets(feasible_frame) -> None:
    frame, truths = feasible_frame()
    targets = TargetSet(
        (
            Target(
                name="missing_measure",
                entity="household",
                value=1.0,
                measure="missing_measure",
            ),
            _population_target(truths["population"], 1.0),
        )
    )

    with pytest.raises(ValueError, match="target_loss_scales"):
        calibrate(
            frame,
            targets,
            epochs=50,
            seed=0,
            target_loss_scales=np.asarray([10.0, 0.0]),
        )


def test_max_weight_ratio_is_respected(feasible_frame) -> None:
    frame, truths = feasible_frame()
    w0 = frame.resolve_weights("household").values
    targets = TargetSet((_income_target(truths["income"], 3.0),))
    result = calibrate(frame, targets, epochs=300, seed=0, max_weight_ratio=2.0)
    w = result.frame.resolve_weights("household").values
    assert (w <= 2.0 * w0 + 1e-9).all()


def test_weight_ratio_bound_prevents_a_landmine(landmine_frame) -> None:
    """A capital-gains target reachable only by inflating one rare donor.

    Unbounded calibration detonates the donor's weight to hit the target;
    the ratio bound caps it, so the donor's weighted capital-gains stays sane.
    """
    frame, donor_index, donor_value = landmine_frame()
    w0 = frame.resolve_weights("household").values
    # Target far above the current weighted capital gains (donor weight is ~1).
    far_target = TargetSet(
        (
            Target(
                name="capital_gains",
                entity="household",
                value=donor_value * 5000.0,
                measure="capital_gains",
            ),
        )
    )

    unbounded = calibrate(frame, far_target, epochs=300, seed=0)
    unbounded_w = unbounded.frame.resolve_weights("household").values
    bounded = calibrate(frame, far_target, epochs=300, seed=0, max_weight_ratio=10.0)
    bounded_w = bounded.frame.resolve_weights("household").values

    # Unbounded blows the donor weight far past the bound; bounded cannot.
    assert unbounded_w[donor_index] > 100.0 * w0[donor_index]
    assert bounded_w[donor_index] <= 10.0 * w0[donor_index] + 1e-9


def test_multi_period_targets_share_one_weight_vector(multiperiod_frame) -> None:
    frame, truth_2026, truth_2030 = multiperiod_frame()
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                value=truth_2026 * 1.3,
                measure="income_2026",
                period=2026,
            ),
            Target(
                name="income",
                entity="household",
                value=truth_2030 * 1.3,
                measure="income_2030",
                period=2030,
            ),
        )
    )
    result = calibrate(frame, targets, epochs=400, seed=0)
    # One weight vector, both periods' targets improved.
    assert result.problem.matrix.shape[0] == 2  # two (target, period) rows
    for diag in result.diagnostics:
        assert abs(diag.relative_error) < 0.05


def test_l0_prunes_toward_a_record_budget(feasible_frame) -> None:
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    budget = 120
    result = calibrate(
        frame,
        targets,
        epochs=500,
        seed=0,
        target_records=budget,
        l0_lambda=1e-3,
    )
    w = result.frame.resolve_weights("household").values
    nonzero = int((w > 1e-6).sum())
    # Pruned well below the 400 records, in the neighborhood of the budget.
    assert nonzero < 250
    assert result.l0_lambda > 0.0


def test_target_records_controls_the_budget_not_just_lambda(
    feasible_frame,
) -> None:
    """``target_records`` actually drives the achieved non-zero count (Finding 3).

    Previously ``target_records`` was dead: the *number* never entered the
    optimization (pruning was controlled entirely by ``l0_lambda``), so budget
    10 and budget 350 at the same lambda produced bitwise-identical weights. With
    budget control, two different budgets must produce materially different
    non-zero counts, each tracking its budget.
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    low = calibrate(frame, targets, epochs=250, seed=0, target_records=100)
    high = calibrate(frame, targets, epochs=250, seed=0, target_records=250)

    # The budgets produce materially different non-zero counts...
    assert high.n_nonzero - low.n_nonzero > 50, (low.n_nonzero, high.n_nonzero)
    # ...and the lower budget really is the smaller pool, ordered by budget.
    assert low.n_nonzero < high.n_nonzero
    # Each tracks its budget within a loose tolerance (the search is on a noisy,
    # discrete count, so the band is generous but far tighter than "anything").
    assert abs(low.n_nonzero - 100) <= 40, low.n_nonzero
    assert abs(high.n_nonzero - 250) <= 60, high.n_nonzero
    # A budget-controlled run reports the penalty it settled on.
    assert low.l0_lambda > 0.0 and high.l0_lambda > 0.0
    assert low.gate_open_probabilities is not None
    assert high.gate_open_probabilities is not None
    assert low.gate_open_probabilities.shape == low.weights.shape
    assert high.gate_open_probabilities.shape == high.weights.shape
    # The smaller pool took the stronger penalty (more pruning pressure).
    assert low.l0_lambda > high.l0_lambda


def test_budget_search_returns_probabilities_from_the_winning_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = [5, 2, 1]

    def fake_optimize(*args, return_gate_open_probabilities=False, **kwargs):
        evaluation = len(counts_seen)
        counts_seen.append(counts[evaluation])
        weights = np.concatenate(
            (np.ones(counts[evaluation]), np.zeros(5 - counts[evaluation]))
        )
        trajectory = np.asarray([float(evaluation)])
        probabilities = np.full(5, float(evaluation + 1))
        assert return_gate_open_probabilities is True
        return weights, trajectory, probabilities

    counts_seen: list[int] = []
    monkeypatch.setattr(solve_module, "_optimize", fake_optimize)
    result = solve_module._search_l0_lambda_for_budget(
        None,
        None,
        None,
        None,
        10.0,
        np.ones(5),
        target_records=3,
        epochs=1,
        learning_rate=0.1,
        conserve_mass=False,
        max_weight_ratio=None,
        l2_lambda=0.0,
        init_mean=0.9,
        temperature=0.25,
        seed=0,
        prune_atol=1e-6,
        initial_lambda=None,
        budget_iters=3,
        return_gate_open_probabilities=True,
    )

    weights, trajectory, realized_lambda, n_nonzero, probabilities = result
    np.testing.assert_array_equal(weights, [1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(trajectory, [1.0])
    assert realized_lambda == pytest.approx(0.1)
    assert n_nonzero == 2
    np.testing.assert_array_equal(probabilities, np.full(5, 2.0))


def test_budget_search_default_preserves_legacy_optimizer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def legacy_optimize(*args, **kwargs):
        assert "return_gate_open_probabilities" not in kwargs
        return np.ones(5), np.asarray([0.0])

    monkeypatch.setattr(solve_module, "_optimize", legacy_optimize)
    result = solve_module._search_l0_lambda_for_budget(
        None,
        None,
        None,
        None,
        10.0,
        np.ones(5),
        target_records=3,
        epochs=1,
        learning_rate=0.1,
        conserve_mass=False,
        max_weight_ratio=None,
        l2_lambda=0.0,
        init_mean=0.9,
        temperature=0.25,
        seed=0,
        prune_atol=1e-6,
        initial_lambda=None,
        budget_iters=1,
    )

    assert len(result) == 4


def test_l0_lambda_alone_is_a_fixed_penalty_pruning_control(feasible_frame) -> None:
    """Without ``target_records``, ``l0_lambda`` alone prunes monotonically.

    The fixed-penalty path (no budget search) is still a real control: a stronger
    ``l0_lambda`` keeps strictly fewer records, and the reported ``l0_lambda`` is
    exactly the value passed in (no search overrides it).
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    weak = calibrate(frame, targets, epochs=300, seed=0, l0_lambda=3e-4)
    strong = calibrate(frame, targets, epochs=300, seed=0, l0_lambda=3.5e-3)
    assert strong.n_nonzero < weak.n_nonzero
    # Reported penalty is the value supplied, unchanged.
    assert weak.l0_lambda == 3e-4
    assert strong.l0_lambda == 3.5e-3
    assert weak.gate_open_probabilities is not None
    assert weak.gate_open_probabilities.shape == weak.weights.shape
    assert weak.gate_open_probabilities.dtype == np.float64
    assert np.isfinite(weak.gate_open_probabilities).all()
    assert (
        (0.0 <= weak.gate_open_probabilities) & (weak.gate_open_probabilities <= 1.0)
    ).all()

    # The stronger fit straddles the probability threshold, so this fixture
    # distinguishes the exact formula from thresholds that merely happen to
    # retain every record. Latent weights stay comfortably above the pruning
    # tolerance, making the gate boundary the only support boundary here.
    probability_threshold = hard_concrete_open_probability_threshold(0.25)
    assert (
        float(strong.gate_open_probabilities.min())
        < probability_threshold
        < float(strong.gate_open_probabilities.max())
    )
    current_support = strong.weights > 1e-6 * float(np.mean(strong.initial_weights))
    np.testing.assert_array_equal(
        strong.gate_open_probabilities > probability_threshold,
        current_support,
    )


def test_l0_refit_returns_refit_on_selected_support(feasible_frame) -> None:
    frame, truths = feasible_frame(n=220)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )

    result = calibrate_l0_refit(
        frame,
        targets,
        epochs=180,
        refit_epochs=180,
        seed=0,
        target_records=80,
        mass="conserve",
    )

    assert isinstance(result, L0RefitResult)
    assert result.selection.l0_lambda > 0.0
    assert result.refit.l0_lambda == 0.0
    assert result.l0_lambda == result.selection.l0_lambda
    assert result.final_loss == result.refit.final_loss
    assert result.frame is result.refit.frame
    assert result.frame.n("household") == result.selection.n_nonzero
    assert len(result.selected_entity_ids) == result.selection.n_nonzero
    assert result.n_nonzero == result.selection.n_nonzero
    # The production weights are the ordinary no-L0 refit on the L0-selected
    # support, not the raw gated selection weights.
    assert result.final_loss <= result.selection.final_loss + 0.02


def test_refit_l0_selection_reuses_existing_selection(feasible_frame) -> None:
    frame, truths = feasible_frame(n=180)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    selection = calibrate(
        frame,
        targets,
        epochs=160,
        seed=0,
        target_records=70,
        mass="conserve",
    )

    result = refit_l0_selection(
        frame,
        targets,
        selection,
        epochs=160,
        seed=0,
        mass="conserve",
    )

    assert result.selection is selection
    assert result.refit.l0_lambda == 0.0
    assert result.frame.n("household") == selection.n_nonzero
    assert len(result.selected_entity_ids) == selection.n_nonzero

    with pytest.raises(ValueError, match="only be provided with support and k"):
        refit_l0_selection(
            frame,
            targets,
            selection,
            support_inclusion_probabilities=np.ones(selection.n_nonzero),
            epochs=1,
        )


def test_refit_l0_selection_accepts_gate_asserted_exact_k_support(
    feasible_frame,
) -> None:
    frame, truths = feasible_frame(n=120)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    selection = calibrate(
        frame,
        targets,
        epochs=250,
        seed=0,
        target_records=30,
        mass="conserve",
    )
    threshold = 1e-6 * float(np.mean(selection.initial_weights))
    closed = np.flatnonzero(selection.weights <= threshold)
    assert closed.size > 0
    other = np.flatnonzero(np.arange(120) != closed[0])[:11]
    support = np.sort(np.concatenate(([closed[0]], other)))

    with pytest.raises(ValueError, match="exact-k cardinality gate failed"):
        refit_l0_selection(
            frame,
            targets,
            selection,
            support=support[:-1],
            k=12,
            epochs=10,
        )

    with pytest.raises(ValueError, match="inclusion_probabilities are required"):
        refit_l0_selection(
            frame,
            targets,
            selection,
            support=support,
            k=12,
            epochs=10,
        )

    result = refit_l0_selection(
        frame,
        targets,
        selection,
        support=support,
        k=12,
        support_inclusion_probabilities=np.full(12, 0.1),
        epochs=80,
        seed=1,
        mass="conserve",
    )

    expected_mask = np.zeros(120, dtype=bool)
    expected_mask[support] = True
    np.testing.assert_array_equal(result.selected_mask, expected_mask)
    np.testing.assert_array_equal(result.selected_entity_ids, support)
    assert result.frame.n("household") == 12
    assert len(result.weights) == 12
    # The sampled closed gate proves the explicit path initialized from the
    # original frame's positive design weights, not selection.frame's zeros.
    assert closed[0] in support
    np.testing.assert_array_equal(result.initial_weights, np.full(12, 10_000.0))


def test_exact_k_refit_conserves_full_pool_mass(feasible_frame) -> None:
    """A 3-of-12 refit conserves 12,000, not the selected-only 3,000."""
    frame, truths = feasible_frame(n=12, weight=1000.0)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    selection = calibrate(
        frame,
        targets,
        epochs=1,
        seed=0,
        mass="conserve",
        l0_lambda=1e-8,
    )

    result = refit_l0_selection(
        frame,
        targets,
        selection,
        support=np.array([0, 4, 8], dtype=np.int64),
        k=3,
        support_inclusion_probabilities=np.full(3, 0.25),
        epochs=1,
        seed=0,
        mass="conserve",
    )

    np.testing.assert_array_equal(result.initial_weights, np.full(3, 4_000.0))
    assert result.initial_weights.sum() == truths["population"]
    assert result.weights.sum() == pytest.approx(truths["population"])
    assert result.diagnostics[0].final_estimate == pytest.approx(truths["population"])


@pytest.mark.parametrize(
    "inclusion_probabilities",
    [
        pytest.param(
            np.asarray([0.2 + 7j, 0.6 + 8j, 0.8 + 9j], dtype=np.complex128),
            id="finite-imaginary",
        ),
        pytest.param(
            np.asarray(
                [complex(0.2, np.nan), 0.6 + 8j, 0.8 + 9j],
                dtype=np.complex128,
            ),
            id="nan-imaginary",
        ),
    ],
)
def test_exact_k_refit_rejects_complex_inclusion_probabilities_by_name(
    inclusion_probabilities: np.ndarray,
) -> None:
    frame, targets, selection = _exact_k_design_fixture(
        np.asarray([2.0, 3.0, 5.0]),
        np.ones(3),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        with pytest.raises(
            ValueError,
            match=(
                r"^support_inclusion_probabilities must be a one-dimensional "
                r"numeric vector aligned with support\.$"
            ),
        ):
            refit_l0_selection(
                frame,
                targets,
                selection,
                support=np.arange(3, dtype=np.int64),
                k=3,
                support_inclusion_probabilities=inclusion_probabilities,
                epochs=1,
            )


def test_exact_k_refit_cap_uses_full_pool_expansion_weights(feasible_frame) -> None:
    """The 2-of-12 free-mass refit can reach 12,000 under a 5x cap."""
    frame, truths = feasible_frame(n=12, weight=1000.0)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    selection = calibrate(
        frame,
        targets,
        epochs=1,
        seed=0,
        mass="conserve",
        l0_lambda=1e-8,
    )

    result = refit_l0_selection(
        frame,
        targets,
        selection,
        support=np.array([1, 9], dtype=np.int64),
        k=2,
        support_inclusion_probabilities=np.full(2, 1.0 / 6.0),
        epochs=10,
        learning_rate=0.001,
        seed=0,
        mass="free",
        max_weight_ratio=5.0,
    )

    np.testing.assert_array_equal(result.initial_weights, np.full(2, 6_000.0))
    assert result.diagnostics[0].final_estimate == pytest.approx(
        truths["population"], rel=1e-5
    )
    assert (result.weights <= 5.0 * result.initial_weights).all()


def test_exact_k_refit_reaches_sols_inclusion_aware_cap_case() -> None:
    """Sol's seed-113 case reaches [2, 1] only with the selected w/q baseline."""
    weights = np.asarray([0.125, 1.0, 0.9375, 0.9375])
    inclusion_probabilities = np.asarray([0.125, 0.5, 0.6875, 0.6875])
    low_q_group = np.asarray([1.0, 0.0, 1.0, 1.0])
    frame, targets, selection = _exact_k_design_fixture(weights, low_q_group)
    support, _, selected_q = select_exact_k(
        inclusion_probabilities,
        k=2,
        pi_hi=1.0,
        seed=113,
    )

    np.testing.assert_array_equal(support, [0, 1])
    np.testing.assert_array_equal(selected_q, [0.125, 0.5])
    result = refit_l0_selection(
        frame,
        targets,
        selection,
        support=support,
        k=2,
        support_inclusion_probabilities=selected_q,
        epochs=100,
        learning_rate=0.02,
        seed=0,
        mass="conserve",
        max_weight_ratio=5.0,
    )

    np.testing.assert_array_equal(result.initial_weights, [1.0, 2.0])
    np.testing.assert_array_equal(5.0 * result.initial_weights, [5.0, 10.0])
    np.testing.assert_allclose(result.weights, [2.0, 1.0], rtol=0.0, atol=0.01)
    assert abs(result.diagnostics[0].relative_error) < 0.005
    assert result.diagnostics[0].within_tolerance is True

    ratio_baseline = weights[support] * weights.sum() / weights[support].sum()
    assert 5.0 * ratio_baseline[0] == pytest.approx(5.0 / 3.0)
    assert (5.0 * ratio_baseline[0] - 2.0) / 2.0 == pytest.approx(-1.0 / 6.0)


def test_exact_k_refit_unequal_q_allocates_baseline_and_cap_by_w_over_q() -> None:
    weights = np.asarray([2.0, 3.0, 5.0, 10.0, 20.0])
    inclusion_probabilities = np.asarray([0.2, 0.6, 0.8, 0.7, 0.7])
    frame, targets, selection = _exact_k_design_fixture(
        weights,
        np.ones(len(weights)),
    )
    support, _, selected_q = select_exact_k(
        inclusion_probabilities,
        k=3,
        pi_hi=1.0,
        seed=13,
    )

    np.testing.assert_array_equal(support, [0, 1, 2])
    # Reverse both arrays to prove the refit seam preserves their alignment
    # while sorting the realized frame support back into pool order.
    result = refit_l0_selection(
        frame,
        targets,
        selection,
        support=support[::-1],
        k=3,
        support_inclusion_probabilities=selected_q[::-1],
        epochs=1,
        seed=0,
        mass="conserve",
        max_weight_ratio=5.0,
    )

    expected = np.asarray([320.0 / 17.0, 160.0 / 17.0, 200.0 / 17.0])
    ratio_baseline = weights[support] * weights.sum() / weights[support].sum()
    np.testing.assert_allclose(result.initial_weights, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        5.0 * result.initial_weights,
        5.0 * expected,
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(result.initial_weights, ratio_baseline)


def test_exact_k_refit_uniform_q_matches_ratio_rescale() -> None:
    weights = np.asarray([2.0, 3.0, 5.0, 10.0, 20.0])
    inclusion_probabilities = np.full(len(weights), 0.6)
    frame, targets, selection = _exact_k_design_fixture(
        weights,
        np.ones(len(weights)),
    )
    support, _, selected_q = select_exact_k(
        inclusion_probabilities,
        k=3,
        pi_hi=1.0,
        seed=13,
    )
    result = refit_l0_selection(
        frame,
        targets,
        selection,
        support=support,
        k=3,
        support_inclusion_probabilities=selected_q,
        epochs=1,
        seed=0,
        mass="conserve",
    )

    ratio_baseline = weights[support] * weights.sum() / weights[support].sum()
    np.testing.assert_allclose(
        result.initial_weights,
        ratio_baseline,
        rtol=1e-15,
        atol=0.0,
    )


def test_exact_k_refit_calls_named_gate_after_subsetting(
    feasible_frame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The named cardinality gate protects both sides of frame subsetting."""
    frame, truths = feasible_frame(n=12, weight=1000.0)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    selection = calibrate(
        frame,
        targets,
        epochs=1,
        seed=0,
        mass="conserve",
        l0_lambda=1e-8,
    )
    real_gate = solve_module.assert_exact_k_support
    calls: list[tuple[np.ndarray, int, int | None]] = []

    def recording_gate(support, k, *, pool_size=None):
        normalized = real_gate(support, k, pool_size=pool_size)
        calls.append((normalized.copy(), k, pool_size))
        if len(calls) == 2:
            raise ValueError("post-subset exact-k gate sentinel")
        return normalized

    monkeypatch.setattr(solve_module, "assert_exact_k_support", recording_gate)

    with pytest.raises(ValueError, match="post-subset exact-k gate sentinel"):
        refit_l0_selection(
            frame,
            targets,
            selection,
            support=np.array([0, 4, 8], dtype=np.int64),
            k=3,
            support_inclusion_probabilities=np.full(3, 0.25),
            epochs=1,
        )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], [0, 4, 8])
    np.testing.assert_array_equal(calls[1][0], [0, 1, 2])
    assert calls[0][2] == 12
    assert calls[1][2] == 3


def test_l0_refit_requires_l0_pruning_control(feasible_frame) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.0),))

    with pytest.raises(ValueError, match="target_records|l0_lambda"):
        calibrate_l0_refit(frame, targets, epochs=10, seed=0)


def test_l2_lambda_zero_matches_default(feasible_frame) -> None:
    frame, truths = feasible_frame(n=80)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.2),
            _income_target(truths["income"], 1.2),
        )
    )

    default = calibrate(frame, targets, epochs=120, seed=0)
    explicit_zero = calibrate(frame, targets, epochs=120, seed=0, l2_lambda=0.0)

    np.testing.assert_array_equal(explicit_zero.weights, default.weights)


@pytest.mark.parametrize("l2_lambda", [-1.0, float("inf"), float("nan")])
def test_l2_lambda_must_be_finite_and_non_negative(
    feasible_frame,
    l2_lambda: float,
) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.0),))

    with pytest.raises(ValueError, match="l2_lambda"):
        calibrate(frame, targets, epochs=50, seed=0, l2_lambda=l2_lambda)


def test_l2_lambda_records_provenance(feasible_frame) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.1),))

    result = calibrate(frame, targets, epochs=50, seed=0, l2_lambda=0.001)

    assert result.options["l2_lambda"] == 0.001
    assert result.options["l2_penalty"] == "mean_initial_pre_gate_weight_ratio_squared"


def test_l2_lambda_reduces_weight_concentration() -> None:
    """A positive L2 penalty is a smooth concentration/ESS control.

    The shifted targets below are easiest to fit by moving mass toward the
    highest-income non-renter records. A small L2 penalty keeps the fit close
    while making that concentration materially less extreme.
    """

    frame, targets, initial_weights = _l2_concentration_fixture()

    baseline = calibrate(
        frame,
        targets,
        epochs=500,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
    )
    penalized = calibrate(
        frame,
        targets,
        epochs=500,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l2_lambda=0.001,
    )

    baseline_ratio = baseline.weights / initial_weights
    penalized_ratio = penalized.weights / initial_weights
    assert penalized_ratio.max() < baseline_ratio.max() * 0.5
    assert _effective_sample_size(penalized.weights) > (
        _effective_sample_size(baseline.weights) * 2.0
    )
    assert penalized.final_loss <= baseline.final_loss + 0.02


def test_l2_lambda_reduces_concentration_with_l0_gates_active() -> None:
    frame, targets, initial_weights = _l2_concentration_fixture()

    baseline = calibrate(
        frame,
        targets,
        epochs=400,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l0_lambda=0.003,
    )
    penalized = calibrate(
        frame,
        targets,
        epochs=400,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l0_lambda=0.003,
        l2_lambda=0.001,
    )

    assert baseline.n_nonzero < len(initial_weights)
    assert penalized.n_nonzero < len(initial_weights)
    baseline_ratio = baseline.weights / initial_weights
    penalized_ratio = penalized.weights / initial_weights
    assert penalized_ratio.max() < baseline_ratio.max() * 0.5
    assert _effective_sample_size(penalized.weights) > (
        _effective_sample_size(baseline.weights) * 1.5
    )
    assert penalized.final_loss <= baseline.final_loss + 0.1


def test_l2_lambda_is_fixed_during_target_record_budget_search(feasible_frame) -> None:
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )

    result = calibrate(
        frame,
        targets,
        epochs=250,
        seed=0,
        target_records=120,
        l2_lambda=0.001,
    )

    assert abs(result.n_nonzero - 120) <= 60, result.n_nonzero
    assert result.l0_lambda > 0.0
    assert result.options["l2_lambda"] == 0.001
    assert result.options["l2_penalty"] == "mean_initial_pre_gate_weight_ratio_squared"


def test_effective_sample_size_contract() -> None:
    """Kish ESS: n for uniform weights, 1 for a single carrier, 0 for all-zero."""
    assert effective_sample_size(np.full(10, 3.0)) == pytest.approx(10.0)
    assert effective_sample_size(np.array([5.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert effective_sample_size(np.zeros(4)) == 0.0
    with pytest.raises(ValueError, match="effective_sample_size"):
        effective_sample_size(np.array([1.0, -0.5]))
    with pytest.raises(ValueError, match="effective_sample_size"):
        effective_sample_size(np.array([1.0, float("nan")]))


def test_calibration_result_reports_weight_concentration(feasible_frame) -> None:
    """The result's concentration scalars match their definitions on the weights."""
    frame, truths = feasible_frame(n=150)
    targets = TargetSet(
        (
            _income_target(truths["income"], 1.3),
            _population_target(truths["population"], 1.0),
        )
    )
    result = calibrate(frame, targets, epochs=120, seed=0, mass="conserve")

    weights = np.asarray(result.weights, dtype=np.float64)
    assert result.effective_sample_size == pytest.approx(
        _effective_sample_size(weights)
    )
    assert result.realized_max_weight_ratio == pytest.approx(
        float((weights / result.initial_weights).max())
    )
    k = max(1, int(np.ceil(0.01 * weights.size)))
    expected_share = float(np.sort(weights)[-k:].sum() / weights.sum())
    assert result.top_1pct_weight_share == pytest.approx(expected_share)


def test_refit_l0_selection_threads_l2_lambda_to_refit() -> None:
    """The refit stage — the weights that ship — receives its own L2 penalty.

    The refit previously hardcoded ``l2_lambda=0.0``, so a concentration
    penalty could shape support selection but never the published weights.
    The regression contract is *threading*, pinned via the recorded solver
    options: exactly what the hardcoded zero would have violated.

    Deliberately NOT asserted here: an ESS direction between the penalized
    and free refits. On this near-converged fixture the mass projection
    cancels the penalty's near-proportional push, so that difference sits at
    the run-to-run nondeterminism noise floor and flips sign intermittently
    (issue #307) — and the direction itself is anchor- and regime-dependent
    (an initial-anchored refit penalty *lowers* ESS at production scale).
    The behavioral direction contract lives in
    :func:`test_l2_anchor_direction_on_heterogeneous_starting_weights`, in a
    regime where the effect clears the noise floor by orders of magnitude.
    """
    frame, targets, _ = _l2_concentration_fixture()
    selection = calibrate(
        frame,
        targets,
        epochs=150,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l0_lambda=0.003,
    )

    baseline = refit_l0_selection(
        frame,
        targets,
        selection,
        epochs=150,
        learning_rate=0.05,
        mass="conserve",
        seed=0,
    )
    penalized = refit_l0_selection(
        frame,
        targets,
        selection,
        epochs=150,
        learning_rate=0.05,
        mass="conserve",
        seed=0,
        l2_lambda=0.001,
    )

    assert baseline.refit.options["l2_lambda"] == 0.0
    assert penalized.refit.options["l2_lambda"] == 0.001
    assert (
        penalized.refit.options["l2_penalty"]
        == "mean_initial_pre_gate_weight_ratio_squared"
    )
    # Both refits ran on the same selected support and are valid calibrations.
    assert penalized.n_nonzero == baseline.n_nonzero
    assert penalized.final_loss <= baseline.final_loss + 0.1


def test_calibrate_l0_refit_refit_l2_lambda_inherits_and_overrides() -> None:
    """``refit_l2_lambda`` defaults to ``l2_lambda``; an explicit value overrides.

    Both stages' penalties are recorded in the result options, so all three
    sweep regimes — both-stage, selection-only, refit-only — are auditable.
    """
    frame, targets, _ = _l2_concentration_fixture()
    common = dict(
        epochs=200, seed=0, mass="conserve", learning_rate=0.05, l0_lambda=0.003
    )

    inherited = calibrate_l0_refit(frame, targets, **common, l2_lambda=0.001)
    assert inherited.options["selection_options"]["l2_lambda"] == 0.001
    assert inherited.options["l2_lambda"] == 0.001

    selection_only = calibrate_l0_refit(
        frame, targets, **common, l2_lambda=0.001, refit_l2_lambda=0.0
    )
    assert selection_only.options["selection_options"]["l2_lambda"] == 0.001
    assert selection_only.options["l2_lambda"] == 0.0

    refit_only = calibrate_l0_refit(
        frame, targets, **common, l2_lambda=0.0, refit_l2_lambda=0.002
    )
    assert refit_only.options["selection_options"]["l2_lambda"] == 0.0
    assert refit_only.options["l2_lambda"] == 0.002


@pytest.mark.parametrize("refit_l2_lambda", [-1.0, float("inf"), float("nan")])
def test_refit_l2_lambda_must_be_finite_and_non_negative(
    refit_l2_lambda: float,
) -> None:
    """An invalid refit override fails before the expensive selection stage."""
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.raises(ValueError, match="refit_l2_lambda"):
        calibrate_l0_refit(
            frame,
            targets,
            epochs=50,
            seed=0,
            l0_lambda=0.003,
            refit_l2_lambda=refit_l2_lambda,
        )


def test_l2_anchor_must_be_known() -> None:
    frame, targets, _ = _l2_concentration_fixture()
    with pytest.raises(ValueError, match="l2_anchor"):
        calibrate(frame, targets, epochs=50, seed=0, l2_anchor="bogus")
    with pytest.raises(ValueError, match="refit_l2_anchor"):
        calibrate_l0_refit(
            frame,
            targets,
            epochs=50,
            seed=0,
            l0_lambda=0.003,
            refit_l2_anchor="bogus",
        )


def test_l2_anchor_records_provenance_and_inherits(feasible_frame) -> None:
    """``l2_anchor`` flows to options; ``refit_l2_anchor`` overrides per stage."""
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.1),))
    result = calibrate(
        frame, targets, epochs=50, seed=0, l2_lambda=0.001, l2_anchor="uniform"
    )
    assert result.options["l2_anchor"] == "uniform"
    assert result.options["l2_penalty"] == "mean_uniform_pre_gate_weight_ratio_squared"

    frame2, targets2, _ = _l2_concentration_fixture()
    two_stage = calibrate_l0_refit(
        frame2,
        targets2,
        epochs=100,
        seed=0,
        l0_lambda=0.003,
        l2_lambda=0.001,
        l2_anchor="uniform",
        refit_l2_anchor="initial",
    )
    assert two_stage.options["selection_options"]["l2_anchor"] == "uniform"
    assert two_stage.options["l2_anchor"] == "initial"


def test_l2_anchor_direction_on_heterogeneous_starting_weights() -> None:
    """On heterogeneous starts the anchor decides which way the penalty pulls.

    Under mass conservation the penalty's constrained optimum is
    ``w ∝ anchor²``. Anchored at heterogeneous starting weights
    (``l2_anchor="initial"``) it therefore pulls toward the *square* of the
    start — more concentration, ESS below the start — while the uniform
    anchor is a direct 1/ESS control that spreads weights above the start.
    This pins the production pathology observed in the full-scale L2 sweep:
    the post-L0 refit starts from the selection stage's concentrated weights,
    so an initial-anchored refit penalty collapsed ESS instead of raising it.
    The contract needs active fit gradients — on an already-converged problem
    the mass projection cancels the penalty's near-proportional push.
    """
    rng = np.random.default_rng(0)
    n = 160
    income = rng.lognormal(10.5, 1.0, n)
    initial_weights = rng.lognormal(6.5, 1.2, n)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame({"household_id": range(n), "income": income}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="income",
                entity="household",
                value=float((income * initial_weights).sum()) * 1.3,
                measure="income",
            ),
        )
    )
    start_ess = effective_sample_size(initial_weights)

    common = dict(
        epochs=300, learning_rate=0.05, mass="conserve", seed=0, l2_lambda=0.05
    )
    initial_anchor = calibrate(frame, targets, **common)
    uniform_anchor = calibrate(frame, targets, **common, l2_anchor="uniform")

    assert initial_anchor.options["l2_anchor"] == "initial"
    assert uniform_anchor.options["l2_anchor"] == "uniform"
    # Initial anchor concentrates below the start; uniform spreads above it.
    assert initial_anchor.effective_sample_size < start_ess * 0.6
    assert uniform_anchor.effective_sample_size > start_ess * 1.4
    assert uniform_anchor.final_loss < 0.05  # spreading did not break the fit


def test_l2_anchor_weights_override_reproduces_presets_exactly(
    feasible_frame,
) -> None:
    """An explicit anchor equal to the starting weights IS the initial anchor.

    The explicit-vector seam (how a refit expresses the "design" anchor) must
    change nothing when it encodes a preset — same anchor, bit-identical run —
    while recording its own provenance label.
    """
    frame, truths = feasible_frame(n=60)
    targets = TargetSet((_income_target(truths["income"], 1.3),))
    w0 = frame.resolve_weights("household").values

    preset = calibrate(
        frame, targets, epochs=80, seed=0, l2_lambda=0.05, l2_anchor="initial"
    )
    explicit = calibrate(
        frame,
        targets,
        epochs=80,
        seed=0,
        l2_lambda=0.05,
        l2_anchor="design",
        l2_anchor_weights=w0.copy(),
    )

    np.testing.assert_array_equal(explicit.weights, preset.weights)
    assert explicit.options["l2_anchor"] == "design"
    assert explicit.options["l2_anchor_weights_supplied"] is True
    assert (
        explicit.options["l2_penalty"]
        == "mean_explicit_anchor_pre_gate_weight_ratio_squared"
    )
    assert preset.options["l2_anchor_weights_supplied"] is False


def test_l2_anchor_weights_are_validated(feasible_frame) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_income_target(truths["income"], 1.2),))
    with pytest.raises(ValueError, match="finite and strictly positive"):
        calibrate(
            frame,
            targets,
            epochs=20,
            seed=0,
            l2_lambda=0.01,
            l2_anchor="design",
            l2_anchor_weights=np.full(40, -1.0),
        )
    with pytest.raises(ValueError, match="shape"):
        calibrate(
            frame,
            targets,
            epochs=20,
            seed=0,
            l2_lambda=0.01,
            l2_anchor="design",
            l2_anchor_weights=np.ones(7),
        )
    with pytest.raises(ValueError, match="non-empty label"):
        calibrate(
            frame,
            targets,
            epochs=20,
            seed=0,
            l2_lambda=0.01,
            l2_anchor="",
            l2_anchor_weights=np.ones(40),
        )


def test_refit_design_anchor_uses_pre_selection_weights() -> None:
    """The refit 'design' anchor resolves to the survivors' pre-selection start.

    Provenance must show the refit ran with an explicit anchor labelled
    'design' while the selection stage recorded the equivalent 'initial'
    anchor, so a design-weighted production run can regularize the shipped
    weights toward the survey design rather than toward the selection
    stage's concentrated output.
    """
    frame, targets, _ = _l2_concentration_fixture()
    result = calibrate_l0_refit(
        frame,
        targets,
        epochs=150,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l0_lambda=0.003,
        l2_lambda=0.001,
        l2_anchor="design",
    )

    assert result.options["l2_anchor"] == "design"
    assert result.options["l2_anchor_weights_supplied"] is True
    # At a fresh selection the design anchor IS the initial anchor.
    assert result.options["selection_options"]["l2_anchor"] == "initial"
    assert result.options["selection_options"]["l2_anchor_weights_supplied"] is False


def test_l0_refit_result_delegates_summary_metrics_to_refit() -> None:
    """The two-stage result reports the refit's (shipped) summary metrics."""
    frame, targets, _ = _l2_concentration_fixture()
    result = calibrate_l0_refit(
        frame,
        targets,
        epochs=200,
        seed=0,
        mass="conserve",
        learning_rate=0.05,
        l0_lambda=0.003,
    )
    assert result.fraction_within_10pct == result.refit.fraction_within_10pct
    assert result.effective_sample_size == result.refit.effective_sample_size
    assert result.realized_max_weight_ratio == result.refit.realized_max_weight_ratio
    assert result.top_1pct_weight_share == result.refit.top_1pct_weight_share


def test_budget_iters_must_be_positive(feasible_frame) -> None:
    """A non-positive ``budget_iters`` is rejected with a named error."""
    frame, truths = feasible_frame(n=50)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    with pytest.raises(ValueError, match="budget_iters"):
        calibrate(frame, targets, epochs=50, seed=0, target_records=10, budget_iters=0)


def test_prune_with_conserve_and_cap_keeps_pruned_records_pruned(
    feasible_frame,
) -> None:
    """L0 pruning survives mass-conservation + a weight cap (Finding 4).

    With ``mass="conserve"`` and a ``max_weight_ratio`` cap, the post-pruning
    mass deficit must be filled only over the surviving (gate-open) records.
    Filling it over *all* records with headroom — including the gate-closed,
    ~0-weight pruned ones — resurrects every record (the bug: free-mass pruned
    hundreds, conserve+cap returned zero pruned).

    The conserve+cap run uses ``target_records`` budget control rather than a
    fixed penalty, because a fixed ``l0_lambda`` prunes platform-dependently:
    3e-3 left tens of survivors on arm64 but 3 of 400 on x86 CI, where the
    freed mass had nowhere to go under the 100x cap (feasibility needs
    ``survivors >= n / ratio`` = 4) — while a cap loose enough to make any
    count feasible lets the conserve rescale feed an all-gates-closed
    collapse. The budget search adapts the penalty until the survivor count
    tracks the budget on *any* platform, keeping the scenario away from both
    cliffs. The resurrection bug is still caught: under it every probe
    returns ~400 nonzero records, nowhere near the budget or the assertion
    bound.
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    free = calibrate(frame, targets, epochs=400, seed=0, l0_lambda=3e-2)
    free_nonzero = int((free.frame.resolve_weights("household").values > 1e-6).sum())
    assert free_nonzero < 100  # free mass prunes hard

    capped = calibrate(
        frame,
        targets,
        epochs=400,
        seed=0,
        target_records=80,
        mass="conserve",
        max_weight_ratio=100.0,
    )
    capped_w = capped.frame.resolve_weights("household").values
    capped_nonzero = int((capped_w > 1e-6).sum())
    # The pruned records stay pruned: the survivor count tracks the budget,
    # nowhere near the full 400 the resurrection bug returned.
    assert capped_nonzero < 150, capped_nonzero
    # Mass is still conserved on the survivors.
    initial_total = frame.resolve_weights("household").values.sum()
    assert abs(capped_w.sum() - initial_total) / initial_total < 1e-6


def test_all_gates_closed_raises_a_named_error(feasible_frame) -> None:
    """An L0 penalty that closes every gate raises a named error.

    A penalty far above the fit loss drives every gate shut (under Adam the
    per-parameter step is sign-dominated, so the penalty's gradient marches
    every gate logit down regardless of scale) and the calibrated vector is
    all zeros. The kernel would reject that with an opaque "Weights cannot be
    all zero"; calibrate must instead name the cause — the penalty — and the
    remedies, before the kernel ever sees the weights.
    """
    frame, truths = feasible_frame(n=50)
    targets = TargetSet((_income_target(truths["income"], 1.0),))
    with pytest.raises(ValueError, match="l0_lambda"):
        calibrate(frame, targets, epochs=300, seed=0, l0_lambda=100.0)


def test_cap_below_one_with_conserve_is_rejected_a_priori(feasible_frame) -> None:
    """``max_weight_ratio < 1`` with ``mass="conserve"`` is infeasible (Finding 7).

    Every capped weight is below its initial, so ``sum(cap) < total`` and the
    input mass can never be restored. This is infeasible before any optimization
    and must be rejected in argument validation, naming the three causes — not
    surfaced later as an opaque kernel mass-conservation failure.
    """
    frame, truths = feasible_frame(n=100)
    targets = TargetSet((_income_target(truths["income"], 1.0),))
    with pytest.raises(ValueError, match="max_weight_ratio"):
        calibrate(
            frame,
            targets,
            epochs=50,
            seed=0,
            mass="conserve",
            max_weight_ratio=0.5,
        )


def test_prune_conserve_cap_infeasible_when_survivors_lack_headroom(
    feasible_frame,
) -> None:
    """If the surviving records cannot absorb the deficit under the cap, raise.

    A tight cap (just above 1) leaves the few survivors almost no headroom, so
    they cannot soak up the mass freed by pruning. That is genuinely infeasible
    and must raise a clear error naming pruning + conserve + cap, rather than
    silently resurrecting the pruned records to balance the books.
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    with pytest.raises(ValueError, match="prun.*conserve.*cap|conserve.*cap.*prun"):
        calibrate(
            frame,
            targets,
            epochs=400,
            seed=0,
            l0_lambda=5e-3,
            mass="conserve",
            max_weight_ratio=1.05,
        )


def test_final_loss_describes_the_returned_weights(feasible_frame) -> None:
    """``final_loss`` is the loss of the weights actually returned (Finding 8).

    The loss trajectory records each epoch's value *before* that epoch's step and
    before the closing mass/cap projections, so its tail does not describe the
    returned vector. With ``mass="conserve"`` and a cap, the closing projection
    moves the weights, so the trajectory tail and the true loss on the returned
    weights diverge. ``final_loss`` must equal the loss recomputed on the
    returned weights.
    """
    frame, truths = feasible_frame(n=200)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.5),
            _income_target(truths["income"], 1.5),
        )
    )
    result = calibrate(
        frame,
        targets,
        epochs=300,
        seed=0,
        mass="conserve",
        max_weight_ratio=2.0,
        learning_rate=0.1,
    )
    # Recompute the capped weighted-MAPE loss on the returned weights directly.
    # final_loss is a float64 closing eval, so it matches to machine epsilon.
    b = result.problem.target_vector
    scales = default_target_loss_scales(b)
    est = result.problem.estimates(result.weights)
    true_loss = relative_error_loss(est, b, target_loss_scales=scales)
    assert abs(result.final_loss - true_loss) < 1e-9
    # final_loss is NOT merely the trajectory tail (which is pre-projection): on
    # this conserve+cap run they differ materially.
    assert abs(result.final_loss - float(result.loss_trajectory[-1])) > 1e-4
    # initial_loss still describes the input weights (the trajectory head),
    # to float32 precision since the trajectory is computed in float32 torch.
    est0 = result.problem.estimates(result.initial_weights)
    true_initial = relative_error_loss(est0, b, target_loss_scales=scales)
    assert abs(result.initial_loss - true_initial) < 1e-5


def test__given_warm_start_weights__then_optimizer_starts_from_warm_loss(
    feasible_frame,
) -> None:
    frame, truths = feasible_frame(n=80)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.2),
            _income_target(truths["income"], 0.9),
        )
    )

    cold = calibrate(frame, targets, epochs=1, seed=0, mass="conserve")
    prior = calibrate(frame, targets, epochs=80, seed=0, mass="conserve")
    warm = calibrate(
        frame,
        targets,
        epochs=1,
        seed=0,
        mass="conserve",
        warm_start_weights=prior.weights,
    )

    assert warm.initial_loss < cold.initial_loss
    assert warm.options["warm_start_weights"] == {
        "enabled": True,
        "kind": "explicit",
    }
    np.testing.assert_allclose(warm.initial_weights, prior.initial_weights)
    assert warm.diagnostics[0].initial_estimate == cold.diagnostics[0].initial_estimate


def test_free_adam_preserves_a_better_warm_start_after_overshooting() -> None:
    """A finite Adam budget must not discard an already better feasible fit.

    The warm start nearly hits a count target. A deliberately large step jumps
    past it, so returning the last update makes the accepted warm start worse.
    """
    frame = Frame(
        {
            "person": pd.DataFrame({"person_id": [0], "person_household_id": [0]}),
            "household": pd.DataFrame({"household_id": [0], "household_count": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)},
    )
    targets = TargetSet((_population_target(2.0, 1.005),))
    result = calibrate(
        frame,
        targets,
        epochs=2,
        learning_rate=0.2,
        mass="free",
        max_weight_ratio=3,
        warm_start_weights=np.array([2.0]),
    )
    assert result.final_loss <= result.initial_loss + 1e-7
    np.testing.assert_allclose(result.weights, [2.0], rtol=1e-7)
    assert len(result.loss_trajectory) == 2
    assert result.options["iterate_selection_receipt"]["selected_epoch"] == 0


def test_free_adam_considers_the_final_post_update_candidate() -> None:
    """The final allowed step still counts when it improves the feasible fit."""
    frame = Frame(
        {
            "person": pd.DataFrame({"person_id": [0], "person_household_id": [0]}),
            "household": pd.DataFrame({"household_id": [0], "household_count": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)},
    )
    targets = TargetSet((_population_target(2.0, 1.0),))
    result = calibrate(
        frame,
        targets,
        epochs=1,
        learning_rate=0.1,
        mass="free",
        max_weight_ratio=1.05,
    )
    assert result.final_loss < result.initial_loss
    np.testing.assert_allclose(result.weights, [1.05], rtol=1e-7)
    assert result.weights[0] <= 1.05
    assert result.options["iterate_selection_receipt"]["selected_epoch"] == 1


def test_free_adam_retains_an_intermediate_fit_when_the_last_step_overshoots() -> None:
    """Selection includes intermediate candidates, not only start and end."""
    frame = Frame(
        {
            "person": pd.DataFrame({"person_id": [0], "person_household_id": [0]}),
            "household": pd.DataFrame({"household_id": [0], "household_count": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)},
    )
    result = calibrate(
        frame,
        TargetSet((_population_target(1.5, 1.0),)),
        epochs=3,
        learning_rate=0.2,
        mass="free",
        max_weight_ratio=3,
    )
    assert result.final_loss <= result.loss_trajectory.min() + 1e-7
    assert result.final_loss < 0.02
    assert result.options["iterate_selection_receipt"]["selected_epoch"] == 2


def test__given_bad_warm_start_shape__then_solver_rejects_it(feasible_frame) -> None:
    frame, truths = feasible_frame(n=20)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    bad = np.ones(19)

    with pytest.raises(ValueError, match="warm_start_weights shape"):
        calibrate(frame, targets, epochs=1, seed=0, warm_start_weights=bad)


def test__given_l0_budget_and_warm_start__then_solver_rejects_it(
    feasible_frame,
) -> None:
    frame, truths = feasible_frame(n=40)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    weights = frame.resolve_weights("household").values.copy()

    with pytest.raises(ValueError, match="warm_start_weights.*L0"):
        calibrate(
            frame,
            targets,
            epochs=1,
            seed=0,
            target_records=20,
            warm_start_weights=weights,
        )


def test_default_target_loss_scales_ignore_initial_estimates() -> None:
    targets = np.asarray([0.0, 10.0, 1_000.0])
    initial_estimates = np.asarray([1_000_000.0, 20_000.0, 1.0])

    scales = default_target_loss_scales(targets, initial_estimates)

    np.testing.assert_allclose(scales, np.asarray([1.0, 10.0, 1_000.0]))


def test_default_target_loss_scales_ignore_nonfinite_initial_estimates() -> None:
    targets = np.asarray([0.0, 10.0, 1_000.0])
    initial_estimates = np.asarray([np.inf, np.nan, -np.inf])

    scales = default_target_loss_scales(targets, initial_estimates)

    np.testing.assert_allclose(scales, np.asarray([1.0, 10.0, 1_000.0]))


def test_small_valued_targets_converge_to_the_value_not_value_minus_one() -> None:
    """The loss is minimized at est == target, not est == target - 1.

    A +1 in the loss numerator biases the optimum to target - 1 — negligible at
    $2T magnitudes, fatal for small targets: a count of 5 converges to 4. The
    numerator must be the raw residual.
    """
    import numpy as np
    import pandas as pd

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    n = 50
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(n), "household_count": np.ones(n)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.full(n, 0.1), kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        (
            Target(
                name="count",
                entity="household",
                measure="household_count",
                value=5.0,
            ),
        )
    )
    result = calibrate(frame, targets, epochs=400, seed=0)
    estimate = result.frame.resolve_weights("household").values.sum()
    assert abs(estimate - 5.0) < 0.05  # not 4.0


def test_budget_search_survives_a_mid_search_infeasible_probe(feasible_frame) -> None:
    """A feasible budget under conserve + cap must not crash on a search probe.

    The L0 budget search bisects ``l0_lambda``; an intermediate (too-large)
    penalty can over-prune past what the surviving records can carry under the
    cap, tripping the conserve+cap infeasibility raise *inside* the search. That
    must be caught and steered (toward a smaller penalty), not surfaced as a
    failure that wrongly blames the user's budget — which is itself feasible.
    """
    frame, truths = feasible_frame(n=400)
    targets = TargetSet(
        (
            _population_target(truths["population"], 1.0),
            _income_target(truths["income"], 1.0),
        )
    )
    # 395 survivors at cap 1.05x can carry the input total (395*1.05 > 400), so
    # the budget is feasible — the search must reach it, not raise.
    result = calibrate(
        frame,
        targets,
        epochs=200,
        seed=0,
        target_records=395,
        mass="conserve",
        max_weight_ratio=1.05,
    )
    assert result.n_nonzero > 0
    # Mass is still conserved on the survivors.
    initial_total = frame.resolve_weights("household").values.sum()
    final_total = result.frame.resolve_weights("household").values.sum()
    assert abs(final_total - initial_total) / initial_total < 1e-6


def test_calibrate_reports_epoch_progress(feasible_frame) -> None:
    frame, truths = feasible_frame(n=30)
    targets = TargetSet((_population_target(truths["population"], 1.0),))
    events: list[dict[str, object]] = []

    calibrate(frame, targets, epochs=5, seed=0, progress_callback=events.append)

    assert [event["kind"] for event in events] == ["calibration_epoch"] * 5
    assert [event["epoch"] for event in events] == [1, 2, 3, 4, 5]
    assert all(event["epochs"] == 5 for event in events)
    assert all(isinstance(event["loss"], float) for event in events)


def test_mass_reason_rides_the_free_mass_record() -> None:
    """A caller-supplied mass_reason lands verbatim on the mass record."""

    frame, targets, _ = _l2_concentration_fixture()
    reason = "Calibration to the census_households/constituency family."

    result = calibrate(frame, targets, epochs=4, mass_reason=reason)
    assert result.frame.mass_log[-1].reason == reason

    default = calibrate(frame, targets, epochs=4)
    assert "capped weighted-MAPE" in default.frame.mass_log[-1].reason

    with pytest.raises(ValueError, match="mass_reason requires mass='free'"):
        calibrate(frame, targets, epochs=4, mass="conserve", mass_reason=reason)
    with pytest.raises(ValueError, match="non-empty string"):
        calibrate(frame, targets, epochs=4, mass_reason="   ")


def _pre_best_iterate_oracle(fixture: dict):
    """Verify the immutable optimizer and installed dependency closure first."""
    from microcosm.calibrate import gates

    provenance = fixture["same_runtime_oracle"]
    path = Path(__file__).parent / "fixtures/pre_best_iterate" / provenance["module"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["module_sha256"]
    for name, expected in provenance["helper_source_sha256"].items():
        source = inspect.getsource(getattr(solve_module, name)).rstrip("\n")
        assert hashlib.sha256(source.encode()).hexdigest() == expected, name
    # Pin the entire gate module, including the class's stretch constants.
    assert (
        hashlib.sha256(Path(gates.__file__).read_bytes()).hexdigest()
        == (provenance["gates_module_sha256"])
    )
    assert solve_module._PRUNE_REL_ATOL == provenance["prune_rel_atol"]
    spec = importlib.util.spec_from_file_location("pre_best_iterate_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    source = inspect.getsource(oracle._optimize).rstrip("\n")
    assert (
        hashlib.sha256(source.encode()).hexdigest()
        == provenance["function_source_sha256"]
    )
    return oracle


@pytest.mark.parametrize("case_id", ["conserved_mass", "l2_regularized", "l0_gated"])
def test_best_iterate_preserves_pre_change_excluded_paths(case_id: str) -> None:
    """Old and current solvers return exact bytes in every executing runtime.

    The immutable optimizer comes from the recorded pre-change source, not
    the implementation under test. Mac snapshot bytes provide a separate
    provenance check only on their authoring runtime. No tolerance or platform
    skip replaces the exact old/new comparison.
    """
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/pre_best_iterate/controls.json").read_text()
    )
    oracle = _pre_best_iterate_oracle(fixture)
    inputs = fixture["inputs"]
    n = len(inputs["initial_weights"])
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": range(n),
                    "income": inputs["income"],
                    "eligible": inputs["eligible"],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array(inputs["initial_weights"]), WeightKind.DESIGN)},
    )
    targets = TargetSet(tuple(Target(**item) for item in fixture["targets"]))
    case = next(item for item in fixture["cases"] if item["id"] == case_id)
    options = case["options"]
    # Construct the oracle's inputs from frozen data, independently of the
    # current solver's compiled problem. These two targets preserve row order.
    target_values = np.array([item["value"] for item in fixture["targets"]])
    torch.manual_seed(options["seed"])
    old_weights, old_trajectory, old_gates = oracle._optimize(
        torch.tensor([inputs["income"], inputs["eligible"]], dtype=torch.float32),
        torch.tensor(target_values, dtype=torch.float32),
        None,
        torch.tensor(np.maximum(np.abs(target_values), 1.0), dtype=torch.float32),
        options["target_loss_cap"],
        np.array(inputs["initial_weights"]),
        epochs=options["epochs"],
        learning_rate=options["learning_rate"],
        conserve_mass=options["mass"] == "conserve",
        max_weight_ratio=options["max_weight_ratio"],
        l0_lambda=options.get("l0_lambda", 0.0),
        l2_lambda=options.get("l2_lambda", 0.0),
        target_records=None,
        init_mean=options.get("init_mean", 0.999),
        temperature=0.25,
        return_gate_open_probabilities=True,
    )
    torch.manual_seed(options["seed"])
    result = calibrate(frame, targets, **options)
    expected_arrays = {"weights": old_weights, "loss_trajectory": old_trajectory}
    if old_gates is not None:
        expected_arrays["gate_open_probabilities"] = old_gates
    else:
        assert result.gate_open_probabilities is None
    for name, expected in expected_arrays.items():
        actual = np.asarray(getattr(result, name))
        assert actual.dtype == expected.dtype == np.dtype("float64")
        assert actual.tobytes() == expected.tobytes(), name

    runtime = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
    }
    if runtime == fixture["authoring_runtime"] and all(
        version(name) == expected for name, expected in fixture["dependencies"].items()
    ):
        for name, expected in case["expected_float64_le_hex"].items():
            assert expected_arrays[name].astype("<f8").tobytes() == bytes.fromhex(
                expected
            ), name
    assert result.options["iterate_selection"] == "closing_state"
    assert result.n_nonzero == case["n_nonzero"]
    assert not np.array_equal(result.weights, inputs["initial_weights"])
    if case_id == "conserved_mass":
        assert result.weights.sum() == pytest.approx(sum(inputs["initial_weights"]))
    elif case_id == "l2_regularized":
        assert result.options["l2_lambda"] > 0
    else:
        assert result.l0_lambda > 0
        assert np.all(
            (result.gate_open_probabilities > 0) & (result.gate_open_probabilities < 1)
        )
