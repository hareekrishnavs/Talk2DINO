#!/usr/bin/env python3
"""Gate 2: 200 real training steps (batch size 4), logging every 10 steps,
NO full-val eval (that is Gate 3's job -- doing it every 10 steps here
would cost ~20x a full-val eval for no reason at this stage). Verifies two
things before committing to Gate 3's 2-3 hour run:

  - |r - 0.1| > 0.005: r has visibly moved off its F1 initialisation.
  - l1_masked_ce - ln(C) is negative AND lower at step 200 than at step 20:
    reconstruction is genuinely better than chance AND still improving,
    not just moved-then-flat.

Both conditions are computed directly from the logged values; the verdict
printed at the end is not a separate human claim."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/project/6114407/haree/Talk2DINO")

import torch

from src.learned_affinity.losses import LossWeights
from src.learned_affinity.metric import LearnedMetric
from src.learned_affinity.train import TrainConfig, build_step_sample, setup_training_data, training_step

R_MOVEMENT_THRESHOLD = 0.005
R_INIT = 0.1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-image-subset", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TrainConfig(seed=args.seed)
    model, vocab, train_dataset, _held_out = setup_training_data(
        config, output_dir=Path("/scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/gate2_out"),
        device=args.device, vocab_image_subset=args.vocab_image_subset,
    )

    metric = LearnedMetric().to(args.device)  # fresh, untrained init (F1 fix)
    optimizer = torch.optim.AdamW(metric.parameters(), lr=1e-4)

    rows = []
    print(f"{'step':>5}  {'r':>10}  {'l1_masked_ce':>13}  {'gap_vs_chance':>14}  {'C':>6}  "
          f"{'first_layer_grad':>18}  {'last_layer_grad':>17}  {'r_grad':>12}  {'grad_norm':>12}")
    for step in range(1, args.steps + 1):
        samples = []
        for i in range(args.batch_size):
            item = train_dataset[(step * args.batch_size + i) % len(train_dataset)]
            samples.append(build_step_sample(model, vocab, item["crop_bgr"], item["present_nouns"],
                                              n_distractor=32, device=args.device))
        log = training_step(metric, optimizer, samples, alpha=0.98, loss_weights=LossWeights())

        if step % args.log_every == 0:
            first_layer_grad = metric.mlp[0].weight.grad.norm().item()
            last_layer_grad = metric.mlp[2].weight.grad.norm().item()
            row = {
                "step": step, "r": log["r"], "l1_masked_ce": log["l1_masked_ce"],
                "gap_vs_chance": log["l1_masked_ce_gap_vs_chance"], "num_classes": log["num_classes"],
                "first_layer_grad": first_layer_grad, "last_layer_grad": last_layer_grad,
                "r_grad": log["r_grad"] if log["r_grad"] is not None else 0.0, "grad_norm": log["grad_norm"],
            }
            rows.append(row)
            print(f"{step:>5}  {row['r']:>10.6f}  {row['l1_masked_ce']:>13.6f}  {row['gap_vs_chance']:>+14.6f}  "
                  f"{row['num_classes']:>6.1f}  {row['first_layer_grad']:>18.6e}  {row['last_layer_grad']:>17.6e}  "
                  f"{row['r_grad']:>12.6e}  {row['grad_norm']:>12.6e}")

    final_r = rows[-1]["r"]
    r_moved = abs(final_r - R_INIT) > R_MOVEMENT_THRESHOLD

    row20 = next((r for r in rows if r["step"] == 20), None)
    row200 = next((r for r in rows if r["step"] == args.steps), None)
    gap_negative_at_end = row200["gap_vs_chance"] < 0.0 if row200 else False
    gap_improved = (row200["gap_vs_chance"] < row20["gap_vs_chance"]) if (row20 and row200) else False

    passed = r_moved and gap_negative_at_end and gap_improved

    print()
    print(f"final r = {final_r:.6f} (init 0.1): |r-0.1| = {abs(final_r - R_INIT):.6f} "
          f"({'>' if r_moved else '<='} {R_MOVEMENT_THRESHOLD}: {'OK' if r_moved else 'FAIL'})")
    if row20 and row200:
        print(f"gap_vs_chance at step 20: {row20['gap_vs_chance']:+.6f}, at step {args.steps}: {row200['gap_vs_chance']:+.6f}")
        print(f"  negative at step {args.steps}: {'OK' if gap_negative_at_end else 'FAIL'}")
        print(f"  lower at step {args.steps} than step 20: {'OK' if gap_improved else 'FAIL'}")
    else:
        print(f"could not find both step 20 and step {args.steps} in the log (steps={args.steps}, "
              f"log_every={args.log_every}) -- FAIL")

    print("\nSummary at steps 10, 50, 100, 150, 200 (or nearest logged step):")
    for target in (10, 50, 100, 150, 200):
        nearest = min(rows, key=lambda r: abs(r["step"] - target)) if rows else None
        if nearest:
            print(f"  step {nearest['step']}: r={nearest['r']:.6f} l1_masked_ce={nearest['l1_masked_ce']:.6f} "
                  f"gap_vs_chance={nearest['gap_vs_chance']:+.6f} first_layer_grad={nearest['first_layer_grad']:.6e} "
                  f"last_layer_grad={nearest['last_layer_grad']:.6e} grad_norm={nearest['grad_norm']:.6e}")

    if passed:
        print(f"\nPASS: r moved off init ({final_r:.6f}), gap_vs_chance negative "
              f"({row200['gap_vs_chance']:+.6f}) and improved since step 20 ({row20['gap_vs_chance']:+.6f}).")
        sys.exit(0)
    else:
        print("\nFAIL: one or more Gate 2 conditions not met. Do not proceed to Gate 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()
