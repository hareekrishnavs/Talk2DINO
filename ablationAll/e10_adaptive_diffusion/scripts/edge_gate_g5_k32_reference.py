#!/usr/bin/env python3
"""G5: K=32 reference point. Widening the candidate set from K=12 to K=32
changes the graph EVEN WITH a neutral (zero-init) gate -- more candidates
means a different row-normalisation denominator and a different top-K set
entering the ReLU(cos)^kappa formula. This script evaluates the UNTRAINED
(zero-init) EdgeGate at K=32 so that later training-vs-reference comparisons
at K=32 are judged against THIS number, not against the K=12 canonical
29.877244 (which would overstate what training contributes by conflating
it with the K=12->K=32 widening effect)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/project/6114407/haree/Talk2DINO")

import importlib.util

spec = importlib.util.spec_from_file_location(
    "edge_gate_assert_identity",
    "/project/6114407/haree/Talk2DINO/ablationAll/e10_adaptive_diffusion/scripts/edge_gate_assert_identity.py",
)
identity_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity_mod)

from src.learned_affinity.edge_gate import EdgeGate

CANONICAL_MIOU_K12 = identity_mod.CANONICAL_MIOU


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    gate = EdgeGate(K=32).to(args.device)  # fresh, untrained, zero-init final layer
    metrics = identity_mod.evaluate_edge_gate_converged(gate, device=args.device, K=32, max_images=args.max_images)

    print(f"K=32 UNTRAINED reference mIoU: {metrics['mIoU']}")
    print(f"K=32 UNTRAINED reference aAcc: {metrics['aAcc']}")
    print(f"K=32 UNTRAINED reference mAcc: {metrics['mAcc']}")
    print(f"evaluated_images: {metrics['evaluated_images']}")
    print()
    print(f"K=12 canonical (existing production graph): {CANONICAL_MIOU_K12}")
    print(f"delta (K=32 untrained vs K=12 canonical): {metrics['mIoU'] - CANONICAL_MIOU_K12:+.6f}")
    print()
    if args.max_images is not None:
        print(f"NOTE: ran on only {args.max_images} images (converged CG, ~4s/image; full 5000 costs "
              f"~334 min) -- both this number and the K=12 canonical comparison are only literally "
              f"comparable at full scale; this capped run establishes the SIGN and rough scale of the "
              f"K=12->K=32 widening effect, not a precise reference point.")
    print()
    print("Report BOTH numbers when judging any later K=32 training run -- compare training's mIoU "
          "against the K=32 untrained reference above, not against the K=12 canonical 29.877244.")


if __name__ == "__main__":
    main()
