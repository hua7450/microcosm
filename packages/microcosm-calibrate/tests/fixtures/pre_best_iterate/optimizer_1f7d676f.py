"""Immutable pre-best-iterate optimizer, extracted from commit 1f7d676f.

The function body is copied verbatim from solve.py at the recorded revision.
Tests verify its SHA and the unchanged installed helper closure before use.
This module is only an oracle; it never replaces an installed solver function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch

from microcosm.calibrate.gates import HardConcrete
from microcosm.calibrate.solve import (
    _PRUNE_REL_ATOL,
    _apply_constraint,
    _prepare_warm_start_weights,
    _project_to_total,
    _relative_error_loss,
)


def _optimize(
    matrix: torch.Tensor,
    targets: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
    target_loss_scales: torch.Tensor,
    target_loss_cap: float,
    initial_weights: np.ndarray,
    *,
    warm_start_weights: np.ndarray | None = None,
    epochs: int,
    learning_rate: float,
    conserve_mass: bool,
    max_weight_ratio: float | None,
    l0_lambda: float,
    l2_lambda: float,
    l2_anchor: str = "initial",
    l2_anchor_weights: np.ndarray | None = None,
    target_records: int | None,
    init_mean: float,
    temperature: float,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_context: Mapping[str, object] | None = None,
    return_gate_open_probabilities: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run the torch optimization and return weights plus its trajectory.

    Optimizes the log-weights with Adam against capped weighted MAPE.
    Positivity is by construction (``w = exp(log_w)`` times optional gates). The
    hard constraints — mass conservation and ``max_weight_ratio`` — are applied
    by projecting the realized weights after each step, so they hold on the
    returned vector exactly, not merely in expectation.

    The private opt-in ``return_gate_open_probabilities`` adds a third return
    value without changing the two-item tuple used by existing workspace
    callers. The third value is aligned per-record ``pi_i`` for an L0 run and
    ``None`` when no gates were active.
    """
    w0 = np.asarray(initial_weights, dtype=np.float64)
    start = _prepare_warm_start_weights(
        w0,
        warm_start_weights,
        conserve_mass=conserve_mass,
        max_weight_ratio=max_weight_ratio,
    )
    total0 = float(w0.sum())
    # Same prune threshold the result's n_nonzero uses, so "pruned" here means
    # exactly what the reported non-zero count means.
    prune_atol = _PRUNE_REL_ATOL * float(np.mean(w0))
    log_w = torch.tensor(np.log(start), dtype=torch.float32, requires_grad=True)

    gates: HardConcrete | None = None
    params: list[torch.Tensor] = [log_w]
    if l0_lambda > 0.0 or target_records is not None:
        gates = HardConcrete(len(w0), init_mean=init_mean, temperature=temperature)
        params = [log_w, *gates.parameters()]

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    upper = (
        torch.tensor(max_weight_ratio * w0, dtype=torch.float32)
        if max_weight_ratio is not None
        else None
    )
    if l2_lambda > 0.0:
        # The L2 penalty's reference vector. "initial" divides by each record's
        # own starting weight; "uniform" divides by the shared mean weight; an
        # explicit vector (e.g. the pre-selection design weights during a
        # refit) overrides both. Under mass conservation the penalty's
        # constrained optimum is w_i ∝ anchor_i², so anchoring on
        # heterogeneous starting weights (e.g. a refit whose start is a
        # concentrated selection vector) pulls toward MORE concentration; the
        # uniform anchor makes the penalty a direct 1/ESS control regardless
        # of the starting distribution.
        if l2_anchor_weights is not None:
            anchor = np.asarray(l2_anchor_weights, dtype=np.float64)
        elif l2_anchor == "initial":
            anchor = w0
        else:
            anchor = np.full_like(w0, w0.mean())
        w0_t = torch.tensor(anchor, dtype=torch.float32)
    else:
        w0_t = None

    trajectory = np.empty(epochs, dtype=np.float64)
    for epoch in range(epochs):
        optimizer.zero_grad()
        weights = torch.exp(log_w)
        if gates is not None:
            weights = weights * gates()
        estimate = _apply_constraint(matrix, weights)
        loss = _relative_error_loss(
            estimate,
            targets,
            target_loss_weights,
            target_loss_scales,
            target_loss_cap,
        )
        penalty = (
            l0_lambda * gates.get_penalty()
            if (gates is not None and l0_lambda > 0.0)
            else torch.zeros((), dtype=torch.float32)
        )
        # Penalize latent pre-gate weights. Under L0, a nearly closed gate
        # should not be able to hide a very large exp(log_w).
        l2_penalty = (
            ((torch.exp(log_w) / w0_t) ** 2).mean()
            if w0_t is not None
            else torch.zeros((), dtype=torch.float32)
        )
        total_loss = loss + penalty + l2_lambda * l2_penalty
        trajectory[epoch] = float(loss.item())
        if progress_callback is not None:
            progress_callback(
                {
                    **dict(progress_context or {}),
                    "kind": "calibration_epoch",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": trajectory[epoch],
                }
            )
        total_loss.backward()
        optimizer.step()

        # Hard projections, applied to the realized weights every step so the
        # guarantees hold on the returned vector, not just in expectation.
        with torch.no_grad():
            if upper is not None:
                # Clamp log-weights so exp(log_w) <= max_weight_ratio*w0. This is
                # the landmine guard: a rare high-value near-zero-weight record
                # can never be inflated past its bound.
                log_w.clamp_(max=torch.log(upper))
            if conserve_mass:
                realized = torch.exp(log_w)
                if gates is not None:
                    realized = realized * gates()
                current_total = float(realized.sum().item())
                if current_total > 0:
                    log_w.add_(float(np.log(total0 / current_total)))
                    if upper is not None:
                        # Re-clamp: the rescale may have pushed a record over its
                        # bound. The mass invariant is held within rtol by the
                        # final-vector rescale below; per-step we keep the bound
                        # hard so it can never be violated mid-run.
                        log_w.clamp_(max=torch.log(upper))

    gate_open_probabilities: np.ndarray | None = None
    with torch.no_grad():
        weights = torch.exp(log_w)
        if gates is not None:
            gates.eval()
            weights = weights * gates()
            if return_gate_open_probabilities:
                gate_open_probabilities = (
                    gates.get_active_prob()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
        final = weights.detach().numpy().astype(np.float64)

    # Make the hard ratio bound exact on the returned vector. The per-step
    # clamp is in float32, so exp() can overshoot the bound by a float epsilon
    # (~1e-7 relative); a closing float64 cap guarantees no returned weight
    # exceeds max_weight_ratio * w0, which downstream code may assert.
    if max_weight_ratio is not None:
        final = np.minimum(final, max_weight_ratio * np.asarray(w0, dtype=np.float64))

    # Exact mass conservation on the returned vector: a single closing rescale
    # to the input total. When a max_weight_ratio is also set, the rescale is
    # capped at the bound and the residual is redistributed below the bound, so
    # both invariants hold together. When L0 pruning is active, the deficit is
    # redistributed only over surviving (gate-open) records so the cap fill never
    # resurrects a pruned one.
    if conserve_mass:
        pruned = (
            final <= prune_atol if (gates is not None and l0_lambda > 0.0) else None
        )
        final = _project_to_total(final, total0, max_weight_ratio, w0, pruned=pruned)
    if return_gate_open_probabilities:
        return final, trajectory, gate_open_probabilities
    return final, trajectory
