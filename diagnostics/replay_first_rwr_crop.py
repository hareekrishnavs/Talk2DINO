#!/usr/bin/env python3
"""Replay the untracked frozen first-crop fixture through production CGLS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


REPOSITORY = Path("/project/6114407/haree/Talk2DINO")
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src/open_vocabulary_segmentation"))

from models.dinotext.cover_dr.graph import DirectedTopKGraph
from models.dinotext.cover_dr.rwr import (
    SparseRWROperator,
    _sparse_fp64_residual_certificate,
    solve_rwr_cgls,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = torch.load(args.fixture, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    indices = payload["neighbor_indices"].to(device=device, dtype=torch.int64)
    graph = DirectedTopKGraph(
        neighbor_indices=indices,
        transition_weights=payload["transition_weights"].to(device=device),
        edge_affinities=payload["edge_affinities"].to(device=device),
        self_loop_fallback=payload["self_loop_fallback"].to(device=device),
        num_nodes=indices.shape[0],
        k=indices.shape[1],
        affinity_power=3.0,
    )
    scores = payload["unary_scores"].to(device=device, dtype=torch.float32)
    result = solve_rwr_cgls(
        graph,
        scores,
        alpha=0.98,
        rtol=1e-5,
        atol=1e-7,
        max_iter=5000,
    )
    operator = SparseRWROperator(graph, 0.98)
    rhs = (1 - 0.98) * scores
    residual = rhs - operator.matmul(result.scores)
    threshold = 1e-7 * (graph.num_nodes ** 0.5) + 1e-5 * torch.linalg.vector_norm(rhs, dim=0)
    scaled = torch.linalg.vector_norm(residual, dim=0) / threshold
    adjacency64 = torch.zeros(
        graph.num_nodes,
        graph.num_nodes,
        device=device,
        dtype=torch.float64,
    )
    adjacency64.scatter_add_(
        1,
        graph.neighbor_indices,
        graph.transition_weights.to(torch.float64),
    )
    system64 = (
        torch.eye(graph.num_nodes, device=device, dtype=torch.float64)
        - 0.98 * adjacency64
    )
    rhs64 = rhs.to(torch.float64)
    dense_reference = torch.linalg.solve(system64, rhs64)
    solution64 = result.scores.to(torch.float64)
    reference_error = torch.linalg.vector_norm(
        solution64 - dense_reference, dim=0
    ) / torch.linalg.vector_norm(dense_reference, dim=0)
    residual64 = rhs64 - system64 @ solution64
    threshold64 = 1e-7 * (graph.num_nodes ** 0.5) + 1e-5 * torch.linalg.vector_norm(
        rhs64, dim=0
    )
    sparse_certificate = _sparse_fp64_residual_certificate(
        graph,
        0.98,
        result.scores,
        rhs,
        rtol=1e-5,
        atol=1e-7,
    )
    record = {
        "fixture": str(args.fixture),
        "fixture_sha256": sha256(args.fixture),
        "device": str(device),
        "iterations": result.iterations,
        "work_count": result.work_count,
        "fp64_certificate_checks": result.fp64_certificate_checks,
        "fp64_certified_rhs": result.fp64_certified_rhs,
        "fp64_certificate_rejections": result.fp64_certificate_rejections,
        "fp64_certificate_restart_count": result.fp64_certificate_restart_count,
        "fp64_certificate_work": result.fp64_certificate_work,
        "certificate_dtype": result.certificate_dtype,
        "total_restart_count": result.total_restart_count,
        "restart_reason_counts": list(result.restart_reason_counts),
        "maximum_scaled_true_residual": float(scaled.max().item()),
        "reported_working_maximum_scaled_residual": (
            result.working_maximum_scaled_residual
        ),
        "reported_certified_maximum_scaled_residual": (
            result.certified_maximum_scaled_residual
        ),
        "unconverged_rhs": (scaled > 1).nonzero().flatten().tolist(),
        "maximum_scaled_fp64_reference_residual": float(
            (torch.linalg.vector_norm(residual64, dim=0) / threshold64).max().item()
        ),
        "sparse_dense_fp64_residual_maximum_difference": float(
            (sparse_certificate.residual - residual64).abs().max().item()
        ),
        "sparse_dense_fp64_scaled_maximum_difference": float(
            (
                sparse_certificate.scaled_residual
                - torch.linalg.vector_norm(residual64, dim=0) / threshold64
            ).abs().max().item()
        ),
        "maximum_relative_dense_fp64_solution_error": float(reference_error.max().item()),
        "mean_relative_dense_fp64_solution_error": float(reference_error.mean().item()),
        "argmax_differences_vs_dense_fp64": int(
            (solution64.argmax(dim=0) != dense_reference.argmax(dim=0)).sum().item()
        ),
        "output_argmax_sha256": hashlib.sha256(
            result.scores.argmax(dim=0).cpu().numpy().tobytes()
        ).hexdigest(),
        "scores_sha256": hashlib.sha256(
            result.scores.cpu().numpy().tobytes()
        ).hexdigest(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
