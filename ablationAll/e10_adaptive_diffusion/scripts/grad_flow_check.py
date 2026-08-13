#!/usr/bin/env python3
"""Gate 1 pre-flight check before committing GPU time to a real schedule:
re-initialise LearnedMetric fresh (F1's fixed init -- zero final MLP layer,
r=0.1 plain parameter) and run N REAL training steps (real COCO images/
captions, real model, real CG solves -- not synthetic data), printing the
gradient norm of the MLP's FIRST layer, LAST layer, and `r` after each step.

Even with F1's fix, the first MLP layer's gradient is expected to be
EXACTLY zero at step 1 -- it is gated by the still-zero-initialised final
layer (dL/dW1 involves a factor of W2, which is 0 at init, regardless of
r). The fix is that the LAST layer's gradient is now proportional to
r=0.1 instead of r~2.3e-5 (~4300x larger), so the final layer moves
substantially after the very first optimiser step -- which should unblock
the first layer's gradient within the first few steps after that. F3's
verdict below is computed directly from the measured numbers, not
asserted independently of them."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/project/6114407/haree/Talk2DINO")

import torch

from src.learned_affinity.losses import LossWeights
from src.learned_affinity.metric import LearnedMetric
from src.learned_affinity.train import TrainConfig, build_step_sample, setup_training_data, training_step

FIRST_LAYER_THRESHOLD = 1e-6
LAST_LAYER_THRESHOLD = 1e-6


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-image-subset", type=int, default=300)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TrainConfig(seed=args.seed)
    model, vocab, train_dataset, _held_out = setup_training_data(
        config, output_dir=Path("/scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/grad_flow_check_out"),
        device=args.device, vocab_image_subset=args.vocab_image_subset,
    )

    metric = LearnedMetric().to(args.device)  # fresh, untrained init (F1 fix)
    optimizer = torch.optim.AdamW(metric.parameters(), lr=1e-4)

    rows = []
    print(f"{'step':>4}  {'r':>12}  {'r_grad':>12}  {'first_layer_grad':>18}  {'last_layer_grad':>17}  {'total_grad_norm':>16}")
    for step in range(1, args.steps + 1):
        samples = []
        for i in range(args.batch_size):
            item = train_dataset[(step * args.batch_size + i) % len(train_dataset)]
            samples.append(build_step_sample(model, vocab, item["crop_bgr"], item["present_nouns"],
                                              n_distractor=32, device=args.device))
        log = training_step(metric, optimizer, samples, alpha=0.98, loss_weights=LossWeights())

        first_layer_grad = metric.mlp[0].weight.grad.norm().item()
        last_layer_grad = metric.mlp[2].weight.grad.norm().item()
        r_grad = log["r_grad"] if log["r_grad"] is not None else 0.0
        rows.append({
            "step": step, "r": log["r"], "r_grad": r_grad,
            "first_layer_grad": first_layer_grad, "last_layer_grad": last_layer_grad,
            "total_grad_norm": log["grad_norm"],
        })

        print(f"{step:>4}  {log['r']:>12.6e}  {r_grad:>12.6e}  {first_layer_grad:>18.6e}  "
              f"{last_layer_grad:>17.6e}  {log['grad_norm']:>16.6e}")

    step1 = rows[0]
    last_layer_nonzero_at_step1 = step1["last_layer_grad"] > LAST_LAYER_THRESHOLD
    first_layer_nonzero_step = next(
        (row["step"] for row in rows if row["first_layer_grad"] > FIRST_LAYER_THRESHOLD), None,
    )
    passed = last_layer_nonzero_at_step1 and first_layer_nonzero_step is not None

    print()
    print(f"last_layer_grad at step 1: {step1['last_layer_grad']:.6e} "
          f"({'>' if last_layer_nonzero_at_step1 else '<='} {LAST_LAYER_THRESHOLD:.0e}: "
          f"{'OK' if last_layer_nonzero_at_step1 else 'FAIL'})")
    if first_layer_nonzero_step is not None:
        print(f"first_layer_grad first exceeded {FIRST_LAYER_THRESHOLD:.0e} at step "
              f"{first_layer_nonzero_step} (within {args.steps}): OK")
    else:
        print(f"first_layer_grad never exceeded {FIRST_LAYER_THRESHOLD:.0e} in {args.steps} steps: FAIL")

    if passed:
        print(f"\nPASS: last_layer_grad({step1['last_layer_grad']:.6e}) > {LAST_LAYER_THRESHOLD:.0e} at step 1 "
              f"AND first_layer_grad exceeded {FIRST_LAYER_THRESHOLD:.0e} at step {first_layer_nonzero_step} "
              f"(<= {args.steps}).")
        sys.exit(0)
    else:
        print(f"\nFAIL: last_layer_nonzero_at_step1={last_layer_nonzero_at_step1}, "
              f"first_layer_nonzero_step={first_layer_nonzero_step}. Do not proceed to a real schedule.")
        sys.exit(1)


if __name__ == "__main__":
    main()
