#!/usr/bin/env python3
"""G7: expressiveness diagnostic -- the direct counterpart to the ~13%
edge-flip measurement that condemned the unary design (Y4b's graph
diagnostic in RUN_PartF2fix_review.md). One EdgeGate instance: measure
edge weights at the shipped zero-init (mlp[0] at its normal random init,
mlp[-1] zeroed), THEN randomise ONLY the final layer to magnitude 0.5
in-place (holding mlp[0] fixed, isolating the final layer's effect --
matching G3's finding that mlp[0] starts blocked and only the final layer
moves first), and measure the SAME real features' edge weights again.

Reports, at both K=12 (apples-to-apples with the unary design's own
measurement) and K=32 (the design's actual operating point):
  - mean absolute change in row-normalised edge weight
  - fraction of edges changing by more than 2x
  - mean total-variation distance between old and new row distributions
No GPU needed -- real captured features, direct computation."""
from __future__ import annotations

import sys

sys.path.insert(0, "/project/6114407/haree/Talk2DINO")

import torch

from src.learned_affinity.edge_gate import EdgeGate, build_frozen_candidate_set

UNARY_EDGE_FLIP_FRACTION = 0.13  # Y4b's measured kNN neighbour-overlap complement (1 - 0.8734), K=12


def run_at_k(f_real: torch.Tensor, K: int, seed: int = 13, *, scramble_first_layer: bool = False):
    cand_idx = build_frozen_candidate_set(f_real, K=K)

    gate = EdgeGate(K=K)  # shipped init: mlp[0] normal random, mlp[-1] zero
    with torch.no_grad():
        old_weights = gate(f_real, cand_idx)

    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed)
        if scramble_first_layer:
            # mlp[0]'s PyTorch DEFAULT init (Kaiming-uniform, bound
            # 1/sqrt(2305)~=0.02) gives pre-GELU hidden activations with
            # std~=0.02 -- too small for even a magnitude-0.5 final layer to
            # produce a strongly-varying e_ij (measured std ~0.07, literal
            # G7 spec below). This variant scales mlp[0] to the SAME 0.5
            # magnitude, matching the unary design's own Y4b scramble
            # convention (which scaled BOTH layers) so the comparison to
            # the unary design's ~13% figure isn't handicapped by an
            # under-powered first layer.
            gate.mlp[0].weight.copy_(torch.randn(gate.mlp[0].weight.shape, generator=gen) * 0.5)
            gate.mlp[0].bias.copy_(torch.randn(gate.mlp[0].bias.shape, generator=gen) * 0.5)
        gate.mlp[2].weight.copy_(torch.randn(gate.mlp[2].weight.shape, generator=gen) * 0.5)
        gate.mlp[2].bias.copy_(torch.randn(gate.mlp[2].bias.shape, generator=gen) * 0.5)
        new_weights = gate(f_real, cand_idx)

    abs_change = (new_weights - old_weights).abs()
    mean_abs_change = abs_change.mean().item()

    ratio = new_weights / old_weights.clamp_min(1e-12)
    changed_2x_frac = ((ratio > 2.0) | (ratio < 0.5)).float().mean().item()

    tv_per_row = 0.5 * (old_weights - new_weights).abs().sum(dim=-1)
    mean_tv = tv_per_row.mean().item()

    # kNN-overlap-style figure for direct comparison to the unary design's ~13%:
    # fraction of each row's candidate set that would be DROPPED if we re-ranked
    # by the new weights and kept only the top ceil(K/... ) -- but candidates are
    # FIXED here (that's the whole point of the redesign), so the comparable
    # quantity is edge-weight movement, not neighbour-set overlap. Report both
    # framings so the comparison is not apples-to-oranges by omission.
    print(f"K={K}:")
    print(f"  mean absolute change in row-normalised edge weight: {mean_abs_change:.4f}")
    print(f"  fraction of edges changing by more than 2x: {changed_2x_frac:.4f}")
    print(f"  mean total-variation distance between old/new row distributions: {mean_tv:.4f}")
    print(f"  (unary design's comparable figure: ~{UNARY_EDGE_FLIP_FRACTION:.2f} of kNN NEIGHBOURS "
          f"changed identity, at K=12, under a full-magnitude-0.5 scramble of BOTH MLP layers -- "
          f"not directly the same quantity, since the pairwise design's candidate set never changes "
          f"by construction; total-variation distance is the more comparable notion of 'how much did "
          f"the graph move')")
    print()
    return {"mean_abs_change": mean_abs_change, "changed_2x_frac": changed_2x_frac, "mean_tv": mean_tv}


def main():
    shard = torch.load(
        "/scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full/shards/windows-000000.pt",
        map_location="cpu", weights_only=False,
    )
    f_real = shard[0].float()  # [1024, 768], real DINOv2 patch features

    print("=" * 70)
    print("G7a: LITERAL spec test -- default-init mlp[0], magnitude-0.5 final layer only")
    print("=" * 70)
    results_literal = {}
    for K in (12, 32):
        results_literal[K] = run_at_k(f_real, K)

    print("=" * 70)
    print("G7b: deconfounded test -- BOTH layers at magnitude 0.5 (matching the unary")
    print("     design's own Y4b scramble convention, which scaled both layers)")
    print("=" * 70)
    results_both = {}
    for K in (12, 32):
        results_both[K] = run_at_k(f_real, K, scramble_first_layer=True)

    moves_literal = results_literal[32]["mean_tv"] > 0.3
    moves_both = results_both[32]["mean_tv"] > 0.3
    print(f"G7a (literal spec) VERDICT: pairwise gate {'DOES' if moves_literal else 'does NOT'} move the "
          f"graph substantially (mean TV at K=32: {results_literal[32]['mean_tv']:.4f}). This test is "
          f"CONFOUNDED, though: mlp[0]'s PyTorch default init has std~=0.012, giving pre-GELU hidden "
          f"activations with std~=0.023 -- too small for even a magnitude-0.5 final layer to produce a "
          f"strongly-varying signal (measured raw e_ij std ~=0.07, e_ij mean ~=-0.64 dominated by the "
          f"bias term rather than spanning a meaningful range). A small TV distance here reflects this "
          f"under-powered perturbation, not necessarily the architecture's true capacity.")
    print()
    print(f"G7b (deconfounded) VERDICT: pairwise gate {'DOES' if moves_both else 'does NOT'} move the "
          f"graph substantially when BOTH layers are scaled comparably to the unary design's own test "
          f"(mean TV at K=32: {results_both[32]['mean_tv']:.4f}).")
    if not moves_both:
        print("*** G7b also fails -- this would mean the redesign did not solve the problem it was built for. ***")


if __name__ == "__main__":
    main()
