#!/usr/bin/env python3
"""Run the base-HEAD CGLS over current/historical graph-score combinations."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch


REPOSITORY = Path("/project/6114407/haree/Talk2DINO")
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src/open_vocabulary_segmentation"))

from models.dinotext.cover_dr import rwr as rwr_module
from models.dinotext.cover_dr.graph import DirectedTopKGraph


def original_solver():
    source = subprocess.check_output(
        [
            "git", "show",
            "HEAD:src/open_vocabulary_segmentation/models/dinotext/cover_dr/rwr.py",
        ],
        cwd=REPOSITORY,
        text=True,
    )
    start = source.index("@torch.no_grad()\ndef solve_rwr_cgls(")
    end = source.index("\n\ndef solve_rwr(\n", start)
    namespace = dict(vars(rwr_module))
    exec(compile(source[start:end], "<base-head-cgls>", "exec"), namespace)
    return namespace["solve_rwr_cgls"]


def graph(indices, weights, *, device):
    indices = indices.to(device=device, dtype=torch.int64)
    weights = weights.to(device=device, dtype=torch.float32)
    fallback = (weights[:, 0] == 1) & (weights[:, 1:] == 0).all(1)
    affinities = weights.clone()
    affinities[fallback] = 0
    normalized = weights.clone()
    normalized[~fallback] = (
        weights[~fallback] / weights[~fallback].sum(dim=1, keepdim=True)
    )
    return DirectedTopKGraph(
        neighbor_indices=indices,
        transition_weights=normalized,
        edge_affinities=affinities,
        self_loop_fallback=fallback,
        num_nodes=indices.shape[0],
        k=indices.shape[1],
        affinity_power=3.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--historical-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    historical = torch.load(args.historical_shard, map_location="cpu", weights_only=True)
    device = torch.device("cuda")
    graphs = {
        "current_graph": graph(
            fixture["neighbor_indices"], fixture["transition_weights"], device=device
        ),
        "historical_graph": graph(
            historical["knn_indices"][0], historical["knn_weights"][0], device=device
        ),
    }
    scores = {
        "current_scores": fixture["unary_scores"].to(device, torch.float32),
        "historical_scores": historical["raw_scores"][0]
        .reshape(171, -1)
        .transpose(0, 1)
        .contiguous()
        .to(device, torch.float32),
    }
    solve = original_solver()
    report = {}
    for graph_name, graph_value in graphs.items():
        for score_name, score_value in scores.items():
            name = f"{graph_name}__{score_name}"
            try:
                result = solve(
                    graph_value,
                    score_value,
                    alpha=0.98,
                    rtol=1e-5,
                    atol=1e-7,
                    max_iter=5000,
                )
                report[name] = {
                    "converged": True,
                    "iterations": result.iterations,
                    "max_scaled": result.maximum_scaled_residual,
                    "restarts": result.total_restart_count,
                }
            except rwr_module.RWRSolverError as error:
                report[name] = {
                    "converged": False,
                    "iterations": error.iteration,
                    "max_scaled": error.max_scaled_primal_residual,
                    "failing_rhs": list(error.failing_rhs),
                    "restarts": error.total_restart_count,
                    "detail": error.detail,
                }
            print(name, report[name], flush=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
