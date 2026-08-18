#!/usr/bin/env python3
"""Bounded trace of the base-HEAD first-crop CGLS recurrence."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


REPOSITORY = Path("/project/6114407/haree/Talk2DINO")
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src/open_vocabulary_segmentation"))

from models.dinotext.cover_dr import rwr as rwr_module
from models.dinotext.cover_dr.graph import DirectedTopKGraph


FAILING = (3, 10, 21, 55, 57, 60, 62, 90, 91, 96, 97, 103, 105, 106, 113, 140, 168, 169)
TRACE: list[dict[str, object]] = []
LAST: dict[int, dict[str, object]] = {}


def values(tensor, indices=FAILING):
    return {str(index): float(tensor[index].item()) for index in indices}


def update_hook(
    iteration, work, residual, normal, direction, gamma, denominator, step,
    best_scaled, since_progress, operator, solution, rhs, active,
):
    if not (iteration <= 4 or iteration % 128 == 0 or iteration >= 4600):
        return
    true = operator._matmul_prepared(solution) - rhs
    _norm, _threshold, scaled = rwr_module._scaled_residual_quantities(
        true, rhs, rtol=1e-5, atol=1e-7
    )
    recursive_norm = torch.linalg.vector_norm(residual, dim=0)
    true_norm = torch.linalg.vector_norm(true, dim=0)
    normal_norm = torch.linalg.vector_norm(normal, dim=0)
    gap = torch.linalg.vector_norm(residual - true, dim=0)
    record = {
        "event": "update",
        "iteration": iteration,
        "work_count": work,
        "active_failing_rhs": [index for index in FAILING if bool(active[index])],
        "true_scaled": values(scaled),
        "recursive_residual_l2": values(recursive_norm),
        "true_residual_l2": values(true_norm),
        "residual_gap_l2": values(gap),
        "normal_residual_l2": values(normal_norm),
        "gamma": values(gamma),
        "denominator": values(denominator),
        "step": values(step),
        "best_scaled": values(best_scaled),
        "iterations_since_improvement": {
            str(index): int(since_progress[index].item()) for index in FAILING
        },
    }
    TRACE.append(record)
    LAST.clear()
    LAST.update({index: record for index in FAILING})


def beta_hook(iteration, coefficient):
    for index in FAILING:
        if index in LAST and LAST[index]["iteration"] == iteration:
            LAST[index].setdefault("beta", {})[str(index)] = float(coefficient[index].item())


def restart_hook(iteration, work, reason, mask, scaled, best_scaled, since_progress, restore_best):
    selected = [index for index in FAILING if bool(mask[index])]
    if not selected:
        return
    TRACE.append({
        "event": "restart",
        "iteration": iteration,
        "work_count": work,
        "reason": reason,
        "rhs": selected,
        "rollback": bool(restore_best),
        "true_scaled": {str(index): float(scaled[index].item()) for index in selected},
        "best_scaled": {str(index): float(best_scaled[index].item()) for index in selected},
        "iterations_since_improvement": {
            str(index): int(since_progress[index].item()) for index in selected
        },
    })


def instrumented_solver(mode):
    if mode == "base":
        source = subprocess.check_output(
            ["git", "show", "HEAD:src/open_vocabulary_segmentation/models/dinotext/cover_dr/rwr.py"],
            cwd=REPOSITORY,
            text=True,
        )
    else:
        source = (
            REPOSITORY
            / "src/open_vocabulary_segmentation/models/dinotext/cover_dr/rwr.py"
        ).read_text()
    start = source.index("@torch.no_grad()\ndef solve_rwr_cgls(")
    end = source.index("\n\ndef solve_rwr(\n", start)
    source = source[start:end]
    marker = "        improved = previous_active & (scaled < best_scaled)"
    injection = (
        "        _DIAGNOSTIC_UPDATE(completed_iterations, budget_used, residual, "
        "normal_residual, direction, gamma, denominator, step, best_scaled, "
        "steps_since_progress, operator, solution, right_hand_side, active)\n"
        + marker
    )
    if source.count(marker) != 1:
        raise RuntimeError("update trace marker not found exactly once")
    source = source.replace(marker, injection)
    marker = "        updated_direction = next_normal_residual + direction * coefficient[None, :]"
    injection = "        _DIAGNOSTIC_BETA(completed_iterations, coefficient)\n" + marker
    if source.count(marker) != 1:
        raise RuntimeError("beta trace marker not found exactly once")
    source = source.replace(marker, injection)
    marker = "        repeated = restart_mask & ("
    injection = (
        "        _DIAGNOSTIC_RESTART(completed_iterations, budget_used, reason, "
        "restart_mask, scaled, best_scaled, steps_since_progress, restore_best)\n"
        + marker
    )
    if source.count(marker) != 1:
        raise RuntimeError("restart trace marker not found exactly once")
    source = source.replace(marker, injection)
    namespace = dict(vars(rwr_module))
    namespace.update({
        "_DIAGNOSTIC_UPDATE": update_hook,
        "_DIAGNOSTIC_BETA": beta_hook,
        "_DIAGNOSTIC_RESTART": restart_hook,
    })
    exec(compile(source, "<instrumented-base-cgls>", "exec"), namespace)
    return namespace["solve_rwr_cgls"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "repaired"), default="base")
    args = parser.parse_args()
    payload = torch.load(args.fixture, map_location="cpu", weights_only=True)
    device = torch.device("cuda")
    indices = payload["neighbor_indices"].to(device, torch.int64)
    graph = DirectedTopKGraph(
        neighbor_indices=indices,
        transition_weights=payload["transition_weights"].to(device),
        edge_affinities=payload["edge_affinities"].to(device),
        self_loop_fallback=payload["self_loop_fallback"].to(device),
        num_nodes=indices.shape[0], k=indices.shape[1], affinity_power=3.0,
    )
    scores = payload["unary_scores"].to(device, torch.float32)
    failure = None
    success = None
    try:
        result = instrumented_solver(args.mode)(
            graph, scores, alpha=0.98, rtol=1e-5, atol=1e-7, max_iter=5000
        )
        success = {
            "iterations": result.iterations,
            "work_count": result.work_count,
            "maximum_scaled": result.maximum_scaled_residual,
            "restart_reason_counts": list(result.restart_reason_counts),
        }
    except rwr_module.RWRSolverError as error:
        failure = {
            "type": type(error).__name__,
            "iteration": error.iteration,
            "failing_rhs": list(error.failing_rhs),
            "maximum_scaled": error.max_scaled_primal_residual,
            "restart_reason_counts": list(error.restart_reason_counts),
            "detail": error.detail,
        }
    report = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "failing_rhs_focus": list(FAILING),
        "trace": TRACE,
        "failure": failure,
        "success": success,
        "mode": args.mode,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "trace_records": len(TRACE),
        "failure": failure,
        "success": success,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
