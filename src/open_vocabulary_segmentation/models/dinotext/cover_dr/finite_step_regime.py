"""Reusable finite-step RWR propagation kernel, matched k11-from-k12 graph
construction, and FP64 reference/diagnostic utilities for the k11/k12
stability gate.

Pure math over :class:`~.graph.DirectedTopKGraph` and plain tensors: this
module never parses TOML, never loads a model, and never touches a
dataset. It is intended to be reused unmodified by both the bounded
stability-gate harness and, later, the production matched-experiment
evaluator, so the finite-step recurrence is implemented in exactly one
place.

COVER-DR uses a directed row-stochastic graph and never symmetrizes it;
this module inherits that contract from ``DirectedTopKGraph`` and adds no
symmetrization, tie perturbation, or independent top-k re-selection of its
own. k11 is always constructed as the literal first-11-column prefix of an
already-built k12 graph -- see :func:`build_matched_k11_from_k12`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .graph import DirectedTopKGraph


class FiniteStepRegimeError(ValueError):
    """Raised on any finite-step kernel, graph-matching, or diagnostic
    invariant violation. Always fail closed: never silently substitute a
    convergence-based, tolerance-based, or solver-based fallback."""


_DEFAULT_EPSILON = 1e-6


def _require_exact_int(value, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FiniteStepRegimeError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise FiniteStepRegimeError(f"{label} must be >= {minimum}")
    return value


def _require_finite_float(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FiniteStepRegimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FiniteStepRegimeError(f"{label} must be finite")
    return result


def _require_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise FiniteStepRegimeError(f"{label} must be a torch.Tensor")
    return value


def _require_finite_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise FiniteStepRegimeError(f"{label} must be finite")
    return value


# ---------------------------------------------------------------------------
# Matched k11-from-k12 graph construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedGraphDiagnostics:
    """Bounded, tensor-free diagnostics comparing a k11 graph against the
    k12 graph it was constructed from."""

    prefix_mismatch_count: int
    fallback_row_count_k12: int
    fallback_row_count_k11: int
    fallback_row_mismatch_count: int
    tie_row_count: int
    row_sum_max_error_k11: float
    row_sum_max_error_k12: float
    negative_weight_count_k11: int
    negative_weight_count_k12: int
    non_fallback_self_edge_count_k11: int
    non_fallback_self_edge_count_k12: int
    directed_asymmetry_fraction_k12: float


def build_matched_k11_from_k12(graph12: DirectedTopKGraph) -> DirectedTopKGraph:
    """Construct the matched k=11 graph as the literal first-11-column
    prefix of ``graph12``'s ordered top-12 selection, then independently
    re-normalize and re-derive the zero-affinity fallback policy from the
    retained 11 affinities -- never by calling the top-k selector again.

    Because the underlying affinity ordering is descending, a row's rank-12
    entry can only be positive if its rank-1..11 entries are also positive;
    the zero-affinity fallback condition is therefore provably identical
    for k=11 and k=12, and this function's fallback re-derivation always
    agrees with ``graph12.self_loop_fallback`` on the same rows.
    """
    if not isinstance(graph12, DirectedTopKGraph):
        raise FiniteStepRegimeError("graph12 must be a DirectedTopKGraph")
    if graph12.k != 12:
        raise FiniteStepRegimeError(
            f"matched k11 construction requires a k=12 parent graph, got k={graph12.k}"
        )

    neighbor_11 = graph12.neighbor_indices[:, :11].clone()
    affinities_11 = graph12.edge_affinities[:, :11].clone()
    row_sums = affinities_11.sum(dim=1)
    fallback_11 = row_sums == 0
    weights_11 = torch.zeros_like(affinities_11)
    positive = ~fallback_11
    if torch.any(positive):
        weights_11[positive] = affinities_11[positive] / row_sums[positive, None]
    if torch.any(fallback_11):
        node_ids = torch.arange(
            graph12.num_nodes, device=graph12.neighbor_indices.device, dtype=torch.int64
        )
        neighbor_11[fallback_11, 0] = node_ids[fallback_11]
        weights_11[fallback_11, 0] = 1.0

    return DirectedTopKGraph(
        neighbor_indices=neighbor_11,
        transition_weights=weights_11,
        edge_affinities=affinities_11,
        self_loop_fallback=fallback_11,
        num_nodes=graph12.num_nodes,
        k=11,
        affinity_power=graph12.affinity_power,
    )


def compute_matched_graph_diagnostics(
    graph12: DirectedTopKGraph, graph11: DirectedTopKGraph
) -> MatchedGraphDiagnostics:
    """Bounded diagnostics comparing a matched (graph12, graph11) pair.

    Raises :class:`FiniteStepRegimeError` if the prefix or fallback
    invariants are violated -- these are structural guarantees of
    :func:`build_matched_k11_from_k12`, so a nonzero count here indicates a
    genuine construction defect, never an expected/tolerable condition.
    """
    if not isinstance(graph12, DirectedTopKGraph) or not isinstance(graph11, DirectedTopKGraph):
        raise FiniteStepRegimeError("graph12 and graph11 must be DirectedTopKGraph instances")
    if graph12.k != 12 or graph11.k != 11:
        raise FiniteStepRegimeError("expected a matched (k=12, k=11) graph pair")
    if graph12.num_nodes != graph11.num_nodes:
        raise FiniteStepRegimeError("graph12 and graph11 must share the same node count")

    prefix_mismatch = (graph11.neighbor_indices != graph12.neighbor_indices[:, :11]).any(dim=1)
    prefix_mismatch_count = int(prefix_mismatch.sum().item())
    if prefix_mismatch_count != 0:
        raise FiniteStepRegimeError(
            f"k11 neighbor indices are not an exact prefix of k12 on {prefix_mismatch_count} rows"
        )

    fallback_mismatch_count = int(
        (graph12.self_loop_fallback != graph11.self_loop_fallback).sum().item()
    )
    if fallback_mismatch_count != 0:
        raise FiniteStepRegimeError(
            f"fallback-row set differs between k11 and k12 on {fallback_mismatch_count} rows "
            "(expected k-invariant)"
        )

    tie_row_count = int(
        (graph12.edge_affinities[:, 10] == graph12.edge_affinities[:, 11]).sum().item()
    )

    def _row_sum_max_error(graph: DirectedTopKGraph) -> float:
        return float((graph.transition_weights.sum(dim=1) - 1.0).abs().max().item())

    def _negative_weight_count(graph: DirectedTopKGraph) -> int:
        return int((graph.transition_weights < 0).sum().item())

    def _non_fallback_self_edge_count(graph: DirectedTopKGraph) -> int:
        row_ids = torch.arange(graph.num_nodes, device=graph.neighbor_indices.device)
        ordinary = ~graph.self_loop_fallback
        if not torch.any(ordinary):
            return 0
        return int((graph.neighbor_indices[ordinary] == row_ids[ordinary, None]).sum().item())

    dense = graph12.to_dense()
    directed_present = dense > 0
    asymmetry_fraction = float(
        (directed_present != directed_present.T).float().mean().item()
    )

    return MatchedGraphDiagnostics(
        prefix_mismatch_count=prefix_mismatch_count,
        fallback_row_count_k12=int(graph12.self_loop_fallback.sum().item()),
        fallback_row_count_k11=int(graph11.self_loop_fallback.sum().item()),
        fallback_row_mismatch_count=fallback_mismatch_count,
        tie_row_count=tie_row_count,
        row_sum_max_error_k11=_row_sum_max_error(graph11),
        row_sum_max_error_k12=_row_sum_max_error(graph12),
        negative_weight_count_k11=_negative_weight_count(graph11),
        negative_weight_count_k12=_negative_weight_count(graph12),
        non_fallback_self_edge_count_k11=_non_fallback_self_edge_count(graph11),
        non_fallback_self_edge_count_k12=_non_fallback_self_edge_count(graph12),
        directed_asymmetry_fraction_k12=asymmetry_fraction,
    )


# ---------------------------------------------------------------------------
# Finite-step propagation kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiniteStepTrace:
    """Immutable result of one continuous finite-step recurrence run:
    ``snapshots`` maps a registered step index to its owned P tensor."""

    steps_completed: int
    snapshots: Mapping[int, torch.Tensor]
    alpha: float
    dtype: torch.dtype


def finite_step_propagate(
    graph: DirectedTopKGraph,
    s0: torch.Tensor,
    *,
    alpha: float,
    steps: int,
    snapshot_steps: Sequence[int],
) -> FiniteStepTrace:
    """Run the finite-step recurrence ``P(t+1) = alpha * A @ P(t) + (1-alpha) * S0``
    for exactly ``steps`` completed updates in one continuous pass, capturing
    an owned clone of P at every step named in ``snapshot_steps`` (0 means
    the initial iterate P0 = S0 itself).

    Exactly one ``(1 - alpha)`` factor is applied per step; there is no
    convergence test, tolerance, early stop, CGLS, GMRES, dense solve, or
    fallback anywhere in this function. ``s0`` is never mutated: a defensive
    clone is taken immediately and used for every step's forcing term.
    """
    if not isinstance(graph, DirectedTopKGraph):
        raise FiniteStepRegimeError("graph must be a DirectedTopKGraph")
    s0 = _require_tensor(s0, "s0")
    if s0.ndim != 2:
        raise FiniteStepRegimeError(f"s0 must have shape [N, C], got {tuple(s0.shape)}")
    if s0.shape[0] != graph.num_nodes:
        raise FiniteStepRegimeError("s0 node count does not match graph.num_nodes")
    if not s0.is_floating_point():
        raise FiniteStepRegimeError("s0 must be floating point")
    _require_finite_tensor(s0, "s0")

    alpha_value = _require_finite_float(alpha, "alpha")
    if not 0 <= alpha_value < 1:
        raise FiniteStepRegimeError("alpha must satisfy 0 <= alpha < 1")
    steps = _require_exact_int(steps, "steps", minimum=1)

    snapshot_steps = tuple(snapshot_steps)
    if not snapshot_steps:
        raise FiniteStepRegimeError("snapshot_steps must be non-empty")
    for step in snapshot_steps:
        _require_exact_int(step, "snapshot_steps entry", minimum=0)
        if step > steps:
            raise FiniteStepRegimeError(
                f"snapshot step {step} exceeds the total step count {steps}"
            )
    snapshot_step_set = set(snapshot_steps)

    s0_owned = s0.detach().clone()
    p = s0_owned.clone()
    snapshots: dict[int, torch.Tensor] = {}
    if 0 in snapshot_step_set:
        snapshots[0] = p.clone()

    one_minus_alpha = 1.0 - alpha_value
    for step in range(1, steps + 1):
        p = alpha_value * graph.matmul(p) + one_minus_alpha * s0_owned
        if step in snapshot_step_set:
            snapshots[step] = p.clone()

    _require_finite_tensor(p, "finite-step propagation result")
    for step, value in snapshots.items():
        _require_finite_tensor(value, f"finite-step snapshot at t={step}")
    missing = snapshot_step_set - set(snapshots)
    if missing:
        raise FiniteStepRegimeError(f"snapshot steps were not captured: {sorted(missing)}")

    return FiniteStepTrace(
        steps_completed=steps, snapshots=snapshots, alpha=alpha_value, dtype=p.dtype
    )


# ---------------------------------------------------------------------------
# FP64 references
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseEquilibriumReference:
    """Dense FP64 direct reference -- never called "exact": it is the
    direct solution of a dense FP64 linear system, itself subject to
    floating-point rounding, and is reported alongside its own residual."""

    p_equilibrium: torch.Tensor
    k_matrix: torch.Tensor
    b_vector: torch.Tensor
    residual: torch.Tensor
    residual_frobenius_norm: float
    residual_relative_frobenius_norm: float
    backward_error: float


def dense_fp64_equilibrium_reference(
    graph: DirectedTopKGraph, s0: torch.Tensor, *, alpha: float
) -> DenseEquilibriumReference:
    """Compute the dense FP64 direct reference ``P = (I - alpha*A)^-1 (1-alpha)*S0``
    via ``torch.linalg.solve`` (never a dedicated matrix-inversion call),
    and independently verify it via the true residual ``B - K @ P``."""
    if not isinstance(graph, DirectedTopKGraph):
        raise FiniteStepRegimeError("graph must be a DirectedTopKGraph")
    s0 = _require_tensor(s0, "s0")
    if s0.ndim != 2 or s0.shape[0] != graph.num_nodes:
        raise FiniteStepRegimeError("s0 must have shape [N, C] matching graph.num_nodes")
    _require_finite_tensor(s0, "s0")
    alpha_value = _require_finite_float(alpha, "alpha")
    if not 0 <= alpha_value < 1:
        raise FiniteStepRegimeError("alpha must satisfy 0 <= alpha < 1")

    dense_a = graph.to_dense().to(torch.float64)
    identity = torch.eye(graph.num_nodes, dtype=torch.float64, device=dense_a.device)
    k_matrix = identity - alpha_value * dense_a
    b_vector = (1.0 - alpha_value) * s0.to(torch.float64)

    p_equilibrium = torch.linalg.solve(k_matrix, b_vector)
    residual = b_vector - k_matrix @ p_equilibrium

    residual_norm = float(torch.linalg.matrix_norm(residual).item())
    b_norm = float(torch.linalg.matrix_norm(b_vector).item())
    k_norm = float(torch.linalg.matrix_norm(k_matrix).item())
    p_norm = float(torch.linalg.matrix_norm(p_equilibrium).item())
    relative_residual = residual_norm / max(b_norm, _DEFAULT_EPSILON)
    backward_error = residual_norm / max(k_norm * p_norm + b_norm, _DEFAULT_EPSILON)

    _require_finite_tensor(p_equilibrium, "dense FP64 direct reference")

    return DenseEquilibriumReference(
        p_equilibrium=p_equilibrium,
        k_matrix=k_matrix,
        b_vector=b_vector,
        residual=residual,
        residual_frobenius_norm=residual_norm,
        residual_relative_frobenius_norm=relative_residual,
        backward_error=backward_error,
    )


@dataclass(frozen=True)
class ConditionDiagnostics:
    """Condition-number diagnostics for ``K = I - alpha*A``, computed from
    the true singular values of K itself (never from ``K^T @ K``)."""

    sigma_min: float
    sigma_max: float
    kappa_2: float
    departure_from_normality_illustrative: float


def compute_condition_diagnostics(
    graph: DirectedTopKGraph, *, alpha: float
) -> ConditionDiagnostics:
    """Compute sigma_min(K), sigma_max(K), and kappa_2(K) = sigma_max/sigma_min
    from ``torch.linalg.svdvals(K)`` directly -- this is the condition
    number of K itself, not of K^T @ K. Also reports a clearly-labeled,
    illustrative-only Henrici-style departure-from-normality measure
    (``||K^T K - K K^T||_F / ||K||_F``); this is not a certified
    non-normality index and is not used in any gate threshold."""
    if not isinstance(graph, DirectedTopKGraph):
        raise FiniteStepRegimeError("graph must be a DirectedTopKGraph")
    alpha_value = _require_finite_float(alpha, "alpha")
    if not 0 <= alpha_value < 1:
        raise FiniteStepRegimeError("alpha must satisfy 0 <= alpha < 1")

    dense_a = graph.to_dense().to(torch.float64)
    identity = torch.eye(graph.num_nodes, dtype=torch.float64, device=dense_a.device)
    k_matrix = identity - alpha_value * dense_a

    singular_values = torch.linalg.svdvals(k_matrix)
    sigma_min = float(singular_values.min().item())
    sigma_max = float(singular_values.max().item())
    if sigma_min <= 0:
        raise FiniteStepRegimeError("K must be nonsingular (sigma_min must be positive)")
    kappa_2 = sigma_max / sigma_min

    k_transpose = k_matrix.transpose(0, 1)
    departure = torch.linalg.matrix_norm(
        k_transpose @ k_matrix - k_matrix @ k_transpose
    ) / max(float(torch.linalg.matrix_norm(k_matrix).item()), _DEFAULT_EPSILON)

    return ConditionDiagnostics(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        kappa_2=kappa_2,
        departure_from_normality_illustrative=float(departure.item()),
    )


# ---------------------------------------------------------------------------
# Comparison diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotComparison:
    max_absolute_error: float
    mean_absolute_error: float
    relative_frobenius_error: float
    argmax_disagreement_count: int
    argmax_disagreement_rate: float


def compare_snapshots(
    a: torch.Tensor, b: torch.Tensor, *, epsilon: float = _DEFAULT_EPSILON
) -> SnapshotComparison:
    """Compare two [N, C] score tensors (e.g. FP32 vs FP64, or T vs T')."""
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if a.shape != b.shape:
        raise FiniteStepRegimeError(f"snapshot shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    a64 = a.detach().to(torch.float64)
    b64 = b.detach().to(torch.float64)
    diff = a64 - b64
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    a_norm = float(torch.linalg.matrix_norm(a64).item())
    relative_frobenius = float(torch.linalg.matrix_norm(diff).item()) / max(a_norm, epsilon)
    argmax_a = a.argmax(dim=-1)
    argmax_b = b.argmax(dim=-1)
    disagreement_count = int((argmax_a != argmax_b).sum().item())
    disagreement_rate = disagreement_count / a.shape[0]
    return SnapshotComparison(
        max_absolute_error=max_abs,
        mean_absolute_error=mean_abs,
        relative_frobenius_error=relative_frobenius,
        argmax_disagreement_count=disagreement_count,
        argmax_disagreement_rate=disagreement_rate,
    )


@dataclass(frozen=True)
class MatchedDelta:
    """``D_T = P_k11_T - P_k12_T`` and its diagnostics at one step T."""

    frobenius_norm: float
    argmax_disagreement_count: int
    argmax_disagreement_rate: int


def compute_matched_delta(p_k11: torch.Tensor, p_k12: torch.Tensor) -> MatchedDelta:
    p_k11 = _require_tensor(p_k11, "p_k11")
    p_k12 = _require_tensor(p_k12, "p_k12")
    if p_k11.shape != p_k12.shape:
        raise FiniteStepRegimeError("p_k11 and p_k12 must share the same shape")
    delta = p_k11.detach().to(torch.float64) - p_k12.detach().to(torch.float64)
    norm = float(torch.linalg.matrix_norm(delta).item())
    argmax_k11 = p_k11.argmax(dim=-1)
    argmax_k12 = p_k12.argmax(dim=-1)
    disagreement_count = int((argmax_k11 != argmax_k12).sum().item())
    disagreement_rate = disagreement_count / p_k11.shape[0]
    return MatchedDelta(
        frobenius_norm=norm,
        argmax_disagreement_count=disagreement_count,
        argmax_disagreement_rate=disagreement_rate,
    )


def delta_stability_error(
    delta_a: torch.Tensor, delta_b: torch.Tensor, *, epsilon: float = _DEFAULT_EPSILON
) -> float:
    """``||delta_a - delta_b||_F / max(||delta_b||_F, epsilon)`` -- e.g.
    with ``delta_a = D160`` and ``delta_b = D320`` for the D160-D320
    diagnostic, or ``delta_a = D320``, ``delta_b = D640`` for D320-D640."""
    delta_a = _require_tensor(delta_a, "delta_a").detach().to(torch.float64)
    delta_b = _require_tensor(delta_b, "delta_b").detach().to(torch.float64)
    if delta_a.shape != delta_b.shape:
        raise FiniteStepRegimeError("delta_a and delta_b must share the same shape")
    denom = max(float(torch.linalg.matrix_norm(delta_b).item()), epsilon)
    return float(torch.linalg.matrix_norm(delta_a - delta_b).item()) / denom


# ---------------------------------------------------------------------------
# Diagnostic-only final-label sensitivity (never stitched, never mIoU)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSensitivity:
    """Pixel argmax-map disagreement within one crop -- diagnostic only.
    Never stitched across windows/images and never reported as mIoU."""

    argmax_disagreement_count: int
    argmax_disagreement_rate: float
    pixel_count: int


def diagnostic_pixel_argmax_map(
    patch_scores: torch.Tensor, grid_hw: tuple[int, int], crop_hw: tuple[int, int]
) -> torch.Tensor:
    """Apply the canonical downstream sigmoid + bilinear (align_corners=True)
    interpolation to one window's [N, C] patch scores, in a diagnostic-only
    path that never touches production stitching. Returns the crop-sized
    pixel argmax map ``[crop_h, crop_w]``."""
    patch_scores = _require_tensor(patch_scores, "patch_scores")
    if patch_scores.ndim != 2:
        raise FiniteStepRegimeError("patch_scores must have shape [N, C]")
    grid_h, grid_w = grid_hw
    if grid_h * grid_w != patch_scores.shape[0]:
        raise FiniteStepRegimeError("grid_hw does not match patch_scores node count")
    simmap = patch_scores.reshape(grid_h, grid_w, -1).permute(2, 0, 1).unsqueeze(0)
    probabilities = torch.sigmoid(simmap)
    upsampled = F.interpolate(probabilities, size=crop_hw, mode="bilinear", align_corners=True)
    return upsampled[0].argmax(dim=0)


def compare_label_sensitivity(map_a: torch.Tensor, map_b: torch.Tensor) -> LabelSensitivity:
    map_a = _require_tensor(map_a, "map_a")
    map_b = _require_tensor(map_b, "map_b")
    if map_a.shape != map_b.shape:
        raise FiniteStepRegimeError("map_a and map_b must share the same shape")
    disagreement = map_a != map_b
    count = int(disagreement.sum().item())
    total = int(disagreement.numel())
    return LabelSensitivity(
        argmax_disagreement_count=count,
        argmax_disagreement_rate=count / total,
        pixel_count=total,
    )


__all__ = [
    "ConditionDiagnostics",
    "DenseEquilibriumReference",
    "FiniteStepRegimeError",
    "FiniteStepTrace",
    "LabelSensitivity",
    "MatchedDelta",
    "MatchedGraphDiagnostics",
    "SnapshotComparison",
    "build_matched_k11_from_k12",
    "compare_label_sensitivity",
    "compare_snapshots",
    "compute_condition_diagnostics",
    "compute_matched_delta",
    "compute_matched_graph_diagnostics",
    "delta_stability_error",
    "dense_fp64_equilibrium_reference",
    "diagnostic_pixel_argmax_map",
    "finite_step_propagate",
]
