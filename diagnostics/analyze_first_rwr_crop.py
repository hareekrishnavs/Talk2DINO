#!/usr/bin/env python3
"""Diagnostic-only analysis of the external first-crop RWR fixture.

This file is intentionally outside the production import path.  It never
writes into the source cache and never changes the production solver.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import torch


REPOSITORY = Path("/project/6114407/haree/Talk2DINO")
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src/open_vocabulary_segmentation"))

from models.dinotext.cover_dr.graph import DirectedTopKGraph
from models.dinotext.cover_dr import rwr as rwr_module


ALPHA = 0.98
RTOL = 1e-5
ATOL = 1e-7
MAX_ITER = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--historical-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def graph_from_fixture(payload: dict[str, torch.Tensor], device: torch.device) -> DirectedTopKGraph:
    indices = payload["neighbor_indices"].to(device=device, dtype=torch.int64)
    weights = payload["transition_weights"].to(device=device, dtype=torch.float32)
    affinities = payload["edge_affinities"].to(device=device, dtype=torch.float32)
    fallback = payload["self_loop_fallback"].to(device=device, dtype=torch.bool)
    return DirectedTopKGraph(
        neighbor_indices=indices,
        transition_weights=weights,
        edge_affinities=affinities,
        self_loop_fallback=fallback,
        num_nodes=indices.shape[0],
        k=indices.shape[1],
        affinity_power=3.0,
    )


def dense_adjacency(indices: torch.Tensor, weights: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    node_count = indices.shape[0]
    result = torch.zeros((node_count, node_count), dtype=dtype, device=indices.device)
    result.scatter_add_(1, indices, weights.to(dtype=dtype))
    return result


def scaled_residual(residual: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Independent implementation of the production L2-per-column threshold.
    residual_norm = torch.linalg.vector_norm(residual, dim=0)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=0)
    threshold = ATOL * math.sqrt(residual.shape[0]) + RTOL * rhs_norm
    return residual_norm, threshold, residual_norm / threshold


def solution_metrics(
    name: str,
    solution: torch.Tensor,
    dense_k: torch.Tensor,
    rhs: torch.Tensor,
    reference: torch.Tensor,
    *,
    iterations: int | None,
    work_count: int | None,
    recursive_residual: torch.Tensor | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    value = solution.to(dtype=torch.float64)
    residual = rhs - dense_k @ value
    residual_norm, threshold, scaled = scaled_residual(residual, rhs)
    normal = dense_k.transpose(0, 1) @ residual
    reference_error = torch.linalg.vector_norm(value - reference, dim=0)
    reference_norm = torch.linalg.vector_norm(reference, dim=0).clamp_min(torch.finfo(torch.float64).tiny)
    record: dict[str, object] = {
        "name": name,
        "iterations": iterations,
        "work_count": work_count,
        "finite": bool(torch.isfinite(value).all().item()),
        "converged_rhs": int((scaled <= 1).sum().item()),
        "unconverged_rhs": (scaled > 1).nonzero().flatten().tolist(),
        "maximum_scaled_true_residual": float(scaled.max().item()),
        "median_scaled_true_residual": float(scaled.median().item()),
        "maximum_true_residual_l2": float(residual_norm.max().item()),
        "maximum_true_residual_inf": float(residual.abs().max().item()),
        "maximum_normal_residual_l2": float(torch.linalg.vector_norm(normal, dim=0).max().item()),
        "maximum_relative_dense_solution_error": float((reference_error / reference_norm).max().item()),
        "mean_relative_dense_solution_error": float((reference_error / reference_norm).mean().item()),
        "argmax_node_differences_vs_dense": int((value.argmax(dim=0) != reference.argmax(dim=0)).sum().item()),
        "per_rhs_scaled_true_residual": [float(item) for item in scaled.tolist()],
        "per_rhs_threshold": [float(item) for item in threshold.tolist()],
    }
    if recursive_residual is not None:
        recursive = recursive_residual.to(torch.float64)
        recursive_norm = torch.linalg.vector_norm(recursive, dim=0)
        record["maximum_recursive_residual_l2"] = float(recursive_norm.max().item())
        record["maximum_recursive_to_true_residual_ratio"] = float(
            (recursive_norm / residual_norm.clamp_min(torch.finfo(torch.float64).tiny)).max().item()
        )
    if extra:
        record.update(extra)
    return record


@torch.no_grad()
def historical_cgls(
    graph: DirectedTopKGraph,
    rhs: torch.Tensor,
    *,
    tolerance: float,
    max_iter: int,
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, bool, torch.Tensor]:
    """Exact recurrence from fd5d615:src/learned_affinity/implicit_solve.py."""
    epsilon = torch.finfo(rhs.dtype).tiny
    def forward(value: torch.Tensor) -> torch.Tensor:
        return value - ALPHA * graph.matmul(value)

    def transpose(value: torch.Tensor) -> torch.Tensor:
        return value - ALPHA * graph.transpose_matmul(value)

    solution = rhs.clone() if initial is None else initial.clone()
    residual = rhs - forward(solution)
    normal = transpose(residual)
    direction = normal.clone()
    normal_norm_squared = (normal * normal).sum(dim=0)
    rhs_norm = rhs.norm(dim=0).clamp_min(epsilon)
    completed = 0
    converged = False
    for completed in range(1, max_iter + 1):
        forward_direction = forward(direction)
        denominator = (forward_direction * forward_direction).sum(dim=0).clamp_min(epsilon)
        step = normal_norm_squared / denominator
        solution = solution + step[None, :] * direction
        residual = residual - step[None, :] * forward_direction
        if torch.all(residual.norm(dim=0) / rhs_norm < tolerance):
            converged = True
            break
        next_normal = transpose(residual)
        next_norm_squared = (next_normal * next_normal).sum(dim=0)
        beta = next_norm_squared / normal_norm_squared.clamp_min(epsilon)
        direction = next_normal + beta[None, :] * direction
        normal_norm_squared = next_norm_squared
    return solution, completed, converged, residual


def current_diagnostic_variant(
    *,
    count_completed: bool,
    periodic_steps: int | None = None,
    recursive_state: bool = False,
):
    """Compile current solver with narrowly selected diagnostic constants."""
    source = textwrap.dedent(inspect.getsource(rwr_module.solve_rwr_cgls))
    if count_completed:
        old = "budget_used < problem.max_iter"
        count = source.count(old)
        if count not in (0, 3):
            raise RuntimeError(f"unexpected current solver source: found {count} budget predicates")
        if count == 3:
            source = source.replace(old, "completed_iterations < problem.max_iter")
    if recursive_state:
        marker = "        previous_active = active.clone()\n        residual, converged_now, scaled = _residual_state(\n"
        replacement = (
            "        previous_active = active.clone()\n"
            "        residual = residual + forward_direction * step[None, :]\n"
            "        true_residual, converged_now, scaled = _residual_state(\n"
        )
        marker_count = source.count(marker)
        if marker_count == 1:
            source = source.replace(marker, replacement)
        elif "true_residual, converged_now, scaled = _residual_state(" not in source:
            raise RuntimeError("unexpected current solver source: residual-update marker missing")
    namespace = dict(vars(rwr_module))
    if periodic_steps is not None:
        namespace["_CGLS_PERIODIC_RESTART_STEPS"] = periodic_steps
    exec(compile(source, "<diagnostic-completed-iteration-variant>", "exec"), namespace)
    return namespace["solve_rwr_cgls"]


def failure_solution(error: Exception) -> tuple[torch.Tensor, dict[str, object]]:
    solution = getattr(error, "current_scores", None)
    if solution is None:
        raise
    detail = getattr(error, "detail", None)
    work = getattr(error, "work_count", None)
    if isinstance(detail, str):
        match = re.search(r"work_budget_used=(\d+)", detail)
        if match:
            work = int(match.group(1))
    return solution, {
        "reported_failure": type(error).__name__,
        "failure_reason": getattr(error, "reason", None),
        "failure_detail": detail,
        "reported_failing_rhs": list(getattr(error, "failing_rhs", ())),
        "restart_count": getattr(error, "total_restart_count", None),
        "restart_reason_counts": list(getattr(error, "restart_reason_counts", ())),
        "residual_replacement_count": getattr(error, "total_residual_replacement_count", None),
        "parsed_work_count": work,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.fixture, map_location="cpu", weights_only=True)
    graph = graph_from_fixture(payload, device)
    scores = payload["unary_scores"].to(device=device, dtype=torch.float32)
    rhs64 = ((1 - ALPHA) * scores).to(torch.float64)
    scores64 = scores.to(torch.float64)
    native_rhs64 = (1 - ALPHA) * scores64

    adjacency64 = dense_adjacency(
        graph.neighbor_indices, graph.transition_weights, dtype=torch.float64
    )
    identity64 = torch.eye(graph.num_nodes, dtype=torch.float64, device=device)
    dense_k64 = identity64 - ALPHA * adjacency64
    dense_reference = torch.linalg.solve(dense_k64, rhs64)
    native_dense_reference = torch.linalg.solve(dense_k64, native_rhs64)

    singular_values = torch.linalg.svdvals(dense_k64)
    eigenvalues = torch.linalg.eigvals(ALPHA * adjacency64)
    kt_k = dense_k64.transpose(0, 1) @ dense_k64
    k_kt = dense_k64 @ dense_k64.transpose(0, 1)
    nonnormality = torch.linalg.matrix_norm(kt_k - k_kt) / (
        torch.linalg.matrix_norm(dense_k64) ** 2
    )

    historical = torch.load(args.historical_shard, map_location="cpu", weights_only=True)
    historical_indices = historical["knn_indices"][0].to(device=device, dtype=torch.int64)
    historical_weights = historical["knn_weights"][0].to(device=device, dtype=torch.float32)
    current_indices = graph.neighbor_indices
    current_weights = graph.transition_weights
    current_sets = [set(row) for row in current_indices.cpu().tolist()]
    historical_sets = [set(row) for row in historical_indices.cpu().tolist()]
    set_intersections = [len(left & right) for left, right in zip(current_sets, historical_sets)]
    differing_rows = [index for index, (left, right) in enumerate(zip(current_sets, historical_sets)) if left != right]
    aligned_differences: list[float] = []
    for row in range(graph.num_nodes):
        current_map = dict(zip(current_indices[row].tolist(), current_weights[row].tolist()))
        historical_map = dict(zip(historical_indices[row].tolist(), historical_weights[row].tolist()))
        for destination in sorted(current_sets[row] & historical_sets[row]):
            aligned_differences.append(abs(current_map[destination] - historical_map[destination]))

    report: dict[str, object] = {
        "fixture": str(args.fixture),
        "device": str(device),
        "system": {
            "nodes": graph.num_nodes,
            "rhs": scores.shape[1],
            "top_k": graph.k,
            "row_sum_minimum": float(adjacency64.sum(dim=1).min().item()),
            "row_sum_maximum": float(adjacency64.sum(dim=1).max().item()),
            "a_one_error_inf": float((adjacency64 @ torch.ones(graph.num_nodes, dtype=torch.float64, device=device) - 1).abs().max().item()),
            "finite_weights": bool(torch.isfinite(graph.transition_weights).all().item()),
            "fallback_rows": int(graph.self_loop_fallback.sum().item()),
            "self_edges": int((graph.neighbor_indices == torch.arange(graph.num_nodes, device=device)[:, None]).sum().item()),
            "directed_asymmetry_frobenius": float(torch.linalg.matrix_norm(adjacency64 - adjacency64.T).item()),
            "directed_asymmetry_maximum": float((adjacency64 - adjacency64.T).abs().max().item()),
            "singular_value_minimum": float(singular_values.min().item()),
            "singular_value_maximum": float(singular_values.max().item()),
            "condition_k_2": float((singular_values.max() / singular_values.min()).item()),
            "condition_ktk_2": float(((singular_values.max() / singular_values.min()) ** 2).item()),
            "nonnormality_henrici_commutator_ratio": float(nonnormality.item()),
            "spectral_radius_alpha_a": float(eigenvalues.abs().max().item()),
            "rhs_formula_max_error": float((rhs64 - 0.02 * scores.to(torch.float64)).abs().max().item()),
            "quantized_fp32_system_rhs_dtype": str(rhs64.dtype),
            "native_fp64_system_rhs_dtype": str(native_rhs64.dtype),
            "quantized_vs_native_rhs_maximum_difference": float(
                (rhs64 - native_rhs64).abs().max().item()
            ),
        },
        "historical_graph": {
            "exact_index_element_match_rate": float((current_indices == historical_indices).to(torch.float64).mean().item()),
            "rows_with_any_ordered_index_difference": int((current_indices != historical_indices).any(dim=1).sum().item()),
            "rows_with_semantic_neighbor_difference": len(differing_rows),
            "top_k_set_overlap_mean": sum(set_intersections) / (graph.num_nodes * graph.k),
            "first_ten_semantic_mismatch_rows": differing_rows[:10],
            "aligned_weight_difference_maximum": max(aligned_differences),
            "aligned_weight_difference_mean": sum(aligned_differences) / len(aligned_differences),
            "current_row_sum_range": [float(current_weights.sum(1).min().item()), float(current_weights.sum(1).max().item())],
            "historical_row_sum_range": [float(historical_weights.sum(1).min().item()), float(historical_weights.sum(1).max().item())],
            "current_fallback_rows": int(graph.self_loop_fallback.sum().item()),
            "historical_fallback_rows": int(((historical_weights[:, 0] == 1) & (historical_weights[:, 1:] == 0).all(1)).sum().item()),
        },
        "solvers": [],
    }

    # Dense FP64 reference checked by the same independent residual code.
    report["solvers"].append(solution_metrics(
        "dense_fp64_quantized_fp32_system",
        dense_reference,
        dense_k64,
        rhs64,
        dense_reference,
        iterations=None, work_count=None,
    ))

    # Current production FP32 solver.
    try:
        current_result = rwr_module.solve_rwr_cgls(
            graph, scores, alpha=ALPHA, rtol=RTOL, atol=ATOL, max_iter=MAX_ITER
        )
        current_solution = current_result.scores
        current_extra = {
            "reported_converged": True,
            "restart_count": current_result.total_restart_count,
            "restart_reason_counts": list(current_result.restart_reason_counts),
        }
        current_iterations = current_result.iterations
        current_work = current_result.work_count
    except rwr_module.RWRSolverError as error:
        current_solution, current_extra = failure_solution(error)
        current_iterations = int(getattr(error, "iteration"))
        current_work = current_extra["parsed_work_count"]
    report["solvers"].append(solution_metrics(
        "current_restarted_fp32", current_solution, dense_k64, rhs64, dense_reference,
        iterations=current_iterations, work_count=current_work, extra=current_extra,
    ))

    # Diagnostic-only current recurrence with max_iter counting completed updates.
    variant = current_diagnostic_variant(count_completed=True)
    try:
        variant_result = variant(
            graph, scores, alpha=ALPHA, rtol=RTOL, atol=ATOL, max_iter=MAX_ITER
        )
        variant_solution = variant_result.scores
        variant_extra = {
            "reported_converged": True,
            "restart_count": variant_result.total_restart_count,
            "restart_reason_counts": list(variant_result.restart_reason_counts),
        }
        variant_iterations = variant_result.iterations
        variant_work = variant_result.work_count
    except rwr_module.RWRSolverError as error:
        variant_solution, variant_extra = failure_solution(error)
        variant_iterations = int(getattr(error, "iteration"))
        variant_work = getattr(error, "work_count", None)
    report["solvers"].append(solution_metrics(
        "current_fp32_5000_completed_updates", variant_solution, dense_k64, rhs64, dense_reference,
        iterations=variant_iterations, work_count=variant_work, extra=variant_extra,
    ))

    # Diagnostic-only ablation of the fixed periodic direction reset.  The
    # solver already recomputes the true primal residual after every update.
    no_periodic = current_diagnostic_variant(
        count_completed=True,
        periodic_steps=MAX_ITER + 1,
    )
    try:
        no_periodic_result = no_periodic(
            graph, scores, alpha=ALPHA, rtol=RTOL, atol=ATOL, max_iter=MAX_ITER
        )
        no_periodic_solution = no_periodic_result.scores
        no_periodic_extra = {
            "reported_converged": True,
            "restart_count": no_periodic_result.total_restart_count,
            "restart_reason_counts": list(no_periodic_result.restart_reason_counts),
        }
        no_periodic_iterations = no_periodic_result.iterations
        no_periodic_work = no_periodic_result.work_count
    except rwr_module.RWRSolverError as error:
        no_periodic_solution, no_periodic_extra = failure_solution(error)
        no_periodic_iterations = int(getattr(error, "iteration"))
        no_periodic_work = getattr(error, "work_count", None)
    report["solvers"].append(solution_metrics(
        "current_fp32_without_periodic_restart", no_periodic_solution,
        dense_k64, rhs64, dense_reference,
        iterations=no_periodic_iterations, work_count=no_periodic_work,
        extra=no_periodic_extra,
    ))

    # Diagnostic-only correction of recurrence-state consistency: retain the
    # recursive residual between reliable replacements, while the separately
    # recomputed true residual remains the sole convergence authority.
    recursive_variant = current_diagnostic_variant(
        count_completed=True,
        recursive_state=True,
    )
    try:
        recursive_result = recursive_variant(
            graph, scores, alpha=ALPHA, rtol=RTOL, atol=ATOL, max_iter=MAX_ITER
        )
        recursive_solution = recursive_result.scores
        recursive_extra = {
            "reported_converged": True,
            "restart_count": recursive_result.total_restart_count,
            "restart_reason_counts": list(recursive_result.restart_reason_counts),
        }
        recursive_iterations = recursive_result.iterations
        recursive_work = recursive_result.work_count
    except rwr_module.RWRSolverError as error:
        recursive_solution, recursive_extra = failure_solution(error)
        recursive_iterations = int(getattr(error, "iteration"))
        recursive_work = getattr(error, "work_count", None)
    report["solvers"].append(solution_metrics(
        "current_fp32_recursive_state_with_true_checks", recursive_solution,
        dense_k64, rhs64, dense_reference,
        iterations=recursive_iterations, work_count=recursive_work,
        extra=recursive_extra,
    ))

    # Historical recurrence, unchanged, with both current-relative and original canonical tolerance.
    rhs32 = rhs64.to(torch.float32)
    for tolerance in (RTOL, 1e-10):
        solution, iterations, recursive_converged, recursive_residual = historical_cgls(
            graph, rhs32, tolerance=tolerance, max_iter=MAX_ITER
        )
        report["solvers"].append(solution_metrics(
            f"historical_fp32_recursive_tol_{tolerance:g}",
            solution, dense_k64, rhs64, dense_reference,
            iterations=iterations,
            work_count=iterations,
            recursive_residual=recursive_residual,
            extra={"historical_recursive_converged": recursive_converged},
        ))
    same_initial_solution, same_initial_iterations, same_initial_converged, same_initial_recursive = historical_cgls(
        graph,
        rhs32,
        tolerance=RTOL,
        max_iter=MAX_ITER,
        initial=scores,
    )
    report["solvers"].append(solution_metrics(
        "historical_fp32_recurrence_with_current_initial",
        same_initial_solution,
        dense_k64,
        rhs64,
        dense_reference,
        iterations=same_initial_iterations,
        work_count=same_initial_iterations,
        recursive_residual=same_initial_recursive,
        extra={"historical_recursive_converged": same_initial_converged},
    ))

    # Current solver in FP64, diagnostic only.
    try:
        fp64_result = rwr_module.solve_rwr_cgls(
            graph, scores64, alpha=ALPHA, rtol=RTOL, atol=ATOL, max_iter=MAX_ITER
        )
        fp64_solution = fp64_result.scores
        fp64_extra = {
            "reported_converged": True,
            "restart_count": fp64_result.total_restart_count,
            "restart_reason_counts": list(fp64_result.restart_reason_counts),
        }
        fp64_iterations = fp64_result.iterations
        fp64_work = fp64_result.work_count
    except rwr_module.RWRSolverError as error:
        fp64_solution, fp64_extra = failure_solution(error)
        fp64_iterations = int(getattr(error, "iteration"))
        fp64_work = fp64_extra["parsed_work_count"]
    report["solvers"].append(solution_metrics(
        "current_restarted_fp64_native_system",
        fp64_solution,
        dense_k64,
        native_rhs64,
        native_dense_reference,
        iterations=fp64_iterations, work_count=fp64_work, extra=fp64_extra,
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "condition_k_2": report["system"]["condition_k_2"],
        "semantic_graph_rows": report["historical_graph"]["rows_with_semantic_neighbor_difference"],
        "solvers": [
            {
                "name": item["name"],
                "iterations": item["iterations"],
                "unconverged_rhs": len(item["unconverged_rhs"]),
                "max_scaled": item["maximum_scaled_true_residual"],
            }
            for item in report["solvers"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
