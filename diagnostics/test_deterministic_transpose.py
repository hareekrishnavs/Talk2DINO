#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    payload = torch.load(args.fixture, map_location="cpu", weights_only=True)
    indices = payload["neighbor_indices"].cuda().long()
    weights = payload["transition_weights"].cuda().float()
    nodes, k = indices.shape
    sources = torch.arange(nodes, device="cuda")[:, None].expand(-1, k).reshape(-1)
    destinations = indices.reshape(-1)
    order = torch.argsort(destinations, stable=True)
    sorted_destinations = destinations[order]
    sorted_sources = sources[order]
    sorted_weights = weights.reshape(-1)[order]
    counts = torch.bincount(sorted_destinations, minlength=nodes)
    crow = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64), counts.cumsum(0)))
    transpose = torch.sparse_csr_tensor(
        crow, sorted_sources, sorted_weights, size=(nodes, nodes), device="cuda"
    )
    generator = torch.Generator(device="cuda").manual_seed(17)
    rhs = torch.randn(nodes, 171, device="cuda", generator=generator)

    torch.use_deterministic_algorithms(True)
    first = torch.sparse.mm(transpose, rhs)
    torch.cuda.synchronize()
    started = time.perf_counter()
    hashes = []
    for _ in range(args.repetitions):
        value = torch.sparse.mm(transpose, rhs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    for value in (first, torch.sparse.mm(transpose, rhs)):
        hashes.append(hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest())
    contributions = sorted_weights[:, None] * rhs[sorted_sources]
    segment_first = torch.segment_reduce(contributions, "sum", lengths=counts)
    torch.cuda.synchronize()
    segment_started = time.perf_counter()
    for _ in range(args.repetitions):
        contributions = sorted_weights[:, None] * rhs[sorted_sources]
        segment_value = torch.segment_reduce(contributions, "sum", lengths=counts)
    torch.cuda.synchronize()
    segment_elapsed = time.perf_counter() - segment_started
    segment_hashes = [
        hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
        for value in (segment_first, segment_value)
    ]
    print({
        "hashes": hashes,
        "equal": hashes[0] == hashes[1],
        "seconds": elapsed,
        "microseconds_per_call": elapsed * 1e6 / args.repetitions,
        "layout": str(transpose.layout),
        "segment_hashes": segment_hashes,
        "segment_equal": segment_hashes[0] == segment_hashes[1],
        "segment_matches_csr": torch.equal(first, segment_first),
        "segment_microseconds_per_call": segment_elapsed * 1e6 / args.repetitions,
    })


if __name__ == "__main__":
    main()
