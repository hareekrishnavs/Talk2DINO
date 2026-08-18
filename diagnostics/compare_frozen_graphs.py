#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def dense(indices, weights, dtype):
    nodes = indices.shape[0]
    value = torch.zeros(nodes, nodes, device=indices.device, dtype=dtype)
    value.scatter_add_(1, indices, weights.to(dtype))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--historical-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    historical = torch.load(args.historical_shard, map_location="cpu", weights_only=True)
    device = torch.device("cuda")
    current_i = fixture["neighbor_indices"].to(device, torch.int64)
    current_w = fixture["transition_weights"].to(device, torch.float64)
    old_i = historical["knn_indices"][0].to(device, torch.int64)
    old_w = historical["knn_weights"][0].to(device, torch.float64)
    scores = fixture["unary_scores"].to(device, torch.float64)
    eye = torch.eye(current_i.shape[0], device=device, dtype=torch.float64)
    current_k = eye - 0.98 * dense(current_i, current_w, torch.float64)
    old_k = eye - 0.98 * dense(old_i, old_w, torch.float64)
    rhs = 0.02 * scores
    current_solution = torch.linalg.solve(current_k, rhs)
    old_solution = torch.linalg.solve(old_k, rhs)
    difference = (current_solution - old_solution).abs()
    report = {
        "maximum_solution_absolute_difference": float(difference.max().item()),
        "mean_solution_absolute_difference": float(difference.mean().item()),
        "relative_frobenius_difference": float(
            (torch.linalg.matrix_norm(current_solution - old_solution) /
             torch.linalg.matrix_norm(current_solution)).item()
        ),
        "class_patch_argmax_differences": int(
            (current_solution.argmax(0) != old_solution.argmax(0)).sum().item()
        ),
        "patch_class_argmax_differences": int(
            (current_solution.argmax(1) != old_solution.argmax(1)).sum().item()
        ),
        "total_classes": scores.shape[1],
        "total_patches": scores.shape[0],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
