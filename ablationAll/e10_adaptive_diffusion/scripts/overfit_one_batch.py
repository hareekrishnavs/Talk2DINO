#!/usr/bin/env python3
"""Overfit-one-batch sanity test, before testing any of Gate 2's three
candidate explanations: can the model overfit a SINGLE fixed batch at all?
If not, none of "lr too high", "L2/L3 fighting L1", or "too few steps"
matter -- something more basic is blocking the objective from being
improvable, and that is cheaper to rule out first.

One batch (batch_size 4, seed 0), extracted ONCE and reused every step (no
resampling -- this tests expressiveness, not generalisation). The mask is
also re-seeded identically every step, so the reconstruction TARGET is
identical every step too -- the only thing that can change the loss across
steps is the model's own parameters. L2/L3 weight = 0.0: L1 alone.

Step 0 (before any optimiser step, MLP still exactly zero, g == f) is the
reference. PASS requires l1_masked_ce at step 500 to be at least 20% below
the step-0 value; plateau or rise is FAIL."""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/project/6114407/haree/Talk2DINO")

import torch

from src.learned_affinity.losses import LossWeights, compute_total_loss
from src.learned_affinity.metric import LearnedMetric, build_differentiable_knn_graph
from src.learned_affinity.train import TrainConfig, build_step_sample, setup_training_data, training_step

MASK_SEED = 12345
PASS_FRACTION = 0.20


def fixed_mask_generator(device):
    # Freshly re-seeded every call, deliberately -- reusing one generator
    # object across calls would advance its internal state and give a
    # DIFFERENT mask each time, defeating the "identical target every step"
    # requirement.
    return torch.Generator(device=device).manual_seed(MASK_SEED)


def eval_l1_no_grad(metric, samples, *, alpha, device):
    """Step-0 reference: forward only, no backward, no optimiser step,
    same batch-averaging convention as training_step, same fixed mask."""
    total = 0.0
    with torch.no_grad():
        for features, raw_scores, is_present in samples:
            g = metric(features)
            indices, weights = build_differentiable_knn_graph(g, k=metric.k, kappa=metric.kappa)
            result = compute_total_loss(
                f=features, g=g, indices=indices, weights=weights, raw_scores=raw_scores,
                is_present=is_present, alpha=alpha, loss_weights=LossWeights(l2=0.0, l3=0.0),
                generator=fixed_mask_generator(device),
            )
            total += float(result["l1_masked_ce"]) / len(samples)
    return total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-image-subset", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    config = TrainConfig(seed=args.seed, lr=args.lr)
    model, vocab, train_dataset, _held_out = setup_training_data(
        config, output_dir=Path("/scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/overfit_one_batch_out"),
        device=args.device, vocab_image_subset=args.vocab_image_subset,
    )

    # ONE fixed batch, extracted once, reused every step -- no resampling.
    samples = []
    for i in range(args.batch_size):
        item = train_dataset[i]  # first batch_size items, deterministic given the fixed dataset ordering
        samples.append(build_step_sample(model, vocab, item["crop_bgr"], item["present_nouns"],
                                          n_distractor=32, device=args.device, seed=args.seed))

    metric = LearnedMetric().to(args.device)  # fresh, untrained -- MLP zero, g == f exactly
    loss_weights = LossWeights(l2=0.0, l3=0.0)  # L1 alone; l4 already defaults to 0.0

    # num_classes is per-sample (depends on how many present nouns that crop has);
    # report it from the fixed batch directly, no need to re-derive per step since
    # the batch never changes.
    chance_ces = [math.log(rs.shape[0]) for _, rs, _ in samples]
    mean_chance_ce = statistics.mean(chance_ces)

    step0_l1 = eval_l1_no_grad(metric, samples, alpha=0.98, device=args.device)
    print(f"step0_l1_masked_ce (reference, MLP still zero, g==f): {step0_l1:.6f}")
    print(f"mean ln(C) over the fixed batch: {mean_chance_ce:.6f}  "
          f"(per-sample C: {[rs.shape[0] for _, rs, _ in samples]})")
    print(f"step0 gap_vs_chance: {step0_l1 - mean_chance_ce:+.6f}")
    print()

    optimizer = torch.optim.AdamW(metric.parameters(), lr=args.lr)

    rows = [{"step": 0, "l1_masked_ce": step0_l1, "gap_vs_chance": step0_l1 - mean_chance_ce,
             "r": 0.1, "first_layer_grad": None, "last_layer_grad": None, "grad_norm": None}]
    print(f"{'step':>5}  {'l1_masked_ce':>13}  {'gap_vs_chance':>14}  {'r':>10}  "
          f"{'first_layer_grad':>18}  {'last_layer_grad':>17}  {'grad_norm':>12}")
    print(f"{0:>5}  {step0_l1:>13.6f}  {step0_l1 - mean_chance_ce:>+14.6f}  {'0.100000':>10}  "
          f"{'--':>18}  {'--':>17}  {'--':>12}")

    for step in range(1, args.steps + 1):
        log = training_step(metric, optimizer, samples, alpha=0.98, loss_weights=loss_weights,
                             generator=fixed_mask_generator(args.device))
        if step % args.log_every == 0:
            first_layer_grad = metric.mlp[0].weight.grad.norm().item()
            last_layer_grad = metric.mlp[2].weight.grad.norm().item()
            row = {
                "step": step, "l1_masked_ce": log["l1_masked_ce"],
                "gap_vs_chance": log["l1_masked_ce_gap_vs_chance"], "r": log["r"],
                "first_layer_grad": first_layer_grad, "last_layer_grad": last_layer_grad,
                "grad_norm": log["grad_norm"],
            }
            rows.append(row)
            print(f"{step:>5}  {row['l1_masked_ce']:>13.6f}  {row['gap_vs_chance']:>+14.6f}  {row['r']:>10.6f}  "
                  f"{row['first_layer_grad']:>18.6e}  {row['last_layer_grad']:>17.6e}  {row['grad_norm']:>12.6e}")

    step500_l1 = rows[-1]["l1_masked_ce"]
    threshold = step0_l1 * (1 - PASS_FRACTION)
    passed = step500_l1 <= threshold
    pct_change = (step500_l1 - step0_l1) / step0_l1 * 100

    print()
    print(f"step0_l1_masked_ce:   {step0_l1:.6f}")
    print(f"step{args.steps}_l1_masked_ce: {step500_l1:.6f}")
    print(f"change: {pct_change:+.2f}%  (PASS requires <= -{PASS_FRACTION*100:.0f}%, "
          f"i.e. step{args.steps} <= {threshold:.6f})")
    if passed:
        print(f"\nPASS: l1_masked_ce dropped {abs(pct_change):.2f}% (>= {PASS_FRACTION*100:.0f}% required) "
              f"-- the model CAN overfit this single batch; the objective is improvable.")
        sys.exit(0)
    else:
        print(f"\nFAIL: l1_masked_ce changed {pct_change:+.2f}% ({'plateaued' if abs(pct_change) < PASS_FRACTION*100 else 'rose'}), "
              f"short of the {PASS_FRACTION*100:.0f}% drop required -- the model canNOT even overfit a single fixed "
              f"batch. This points to something more basic than lr/L2/L3 blocking the objective.")
        sys.exit(1)


if __name__ == "__main__":
    main()
