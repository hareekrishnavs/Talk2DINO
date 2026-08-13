#!/usr/bin/env python3
"""G6: overfit-one-batch test for EdgeGate. SAME protocol and thresholds as
the failed unary-design run, so the comparison is exact: one fixed batch
(batch_size 4, seed 0) reused every step, fixed mask seed (identical
reconstruction target every step), L2=L3=0.0 (L1 alone), 500 steps at the
current lr. cand_idx is computed ONCE from frozen features (K=32) and never
recomputed -- matching G4's "candidate set never changes during training"
requirement literally, not just in spirit.

PASS: l1_masked_ce at step 500 is at least 20% below the step-0 value
(step 0 = zero-init gate, i.e. the K=32 untrained reference's per-sample
loss, not the K=12 identity value -- G4/G5 already established these
differ). FAIL: plateau or rise."""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/project/6114407/haree/Talk2DINO")

import torch

from src.learned_affinity.edge_gate import EdgeGate, build_frozen_candidate_set
from src.learned_affinity.losses import LossWeights, compute_total_loss
from src.learned_affinity.train import TrainConfig, build_step_sample, setup_training_data

MASK_SEED = 12345
PASS_FRACTION = 0.20


def fixed_mask_generator(device):
    return torch.Generator(device=device).manual_seed(MASK_SEED)


def forward_loss(edge_gate, cand_idx, features, raw_scores, is_present, *, alpha, device):
    weights, gate = edge_gate(features, cand_idx, return_gate=True)
    result = compute_total_loss(
        f=features, g=features, indices=cand_idx, weights=weights, raw_scores=raw_scores,
        is_present=is_present, alpha=alpha, loss_weights=LossWeights(l2=0.0, l3=0.0),
        generator=fixed_mask_generator(device),
    )
    return result, gate


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
    parser.add_argument("--K", type=int, default=32)
    args = parser.parse_args()

    config = TrainConfig(seed=args.seed, lr=args.lr)
    model, vocab, train_dataset, _held_out = setup_training_data(
        config, output_dir=Path("/scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/edge_gate_g6_out"),
        device=args.device, vocab_image_subset=args.vocab_image_subset,
    )

    # ONE fixed batch, extracted once, reused every step -- no resampling.
    samples = []
    for i in range(args.batch_size):
        item = train_dataset[i]
        samples.append(build_step_sample(model, vocab, item["crop_bgr"], item["present_nouns"],
                                          n_distractor=32, device=args.device, seed=args.seed))

    # cand_idx computed ONCE from frozen features, per sample, K=32 -- never recomputed.
    cand_idxs = [build_frozen_candidate_set(features, K=args.K) for features, _, _ in samples]

    edge_gate = EdgeGate(K=args.K).to(args.device)  # fresh, zero-init final layer
    chance_ces = [math.log(rs.shape[0]) for _, rs, _ in samples]
    mean_chance_ce = statistics.mean(chance_ces)

    # Step 0 reference: forward only, no backward, no optimiser step.
    with torch.no_grad():
        step0_total = 0.0
        for (features, raw_scores, is_present), cand_idx in zip(samples, cand_idxs):
            result, _ = forward_loss(edge_gate, cand_idx, features, raw_scores, is_present,
                                      alpha=0.98, device=args.device)
            step0_total += float(result["l1_masked_ce"]) / len(samples)
    print(f"step0_l1_masked_ce (reference, zero-init gate): {step0_total:.6f}")
    print(f"mean ln(C) over the fixed batch: {mean_chance_ce:.6f}  (per-sample C: {[rs.shape[0] for _, rs, _ in samples]})")
    print(f"step0 gap_vs_chance: {step0_total - mean_chance_ce:+.6f}")
    print()

    optimizer = torch.optim.AdamW(edge_gate.parameters(), lr=args.lr)
    rows = [{"step": 0, "l1_masked_ce": step0_total, "gap_vs_chance": step0_total - mean_chance_ce}]

    print(f"{'step':>5}  {'l1_masked_ce':>13}  {'gap_vs_chance':>14}  {'first_layer_grad':>18}  "
          f"{'last_layer_grad':>17}  {'gate_mean':>10}  {'gate_std':>10}  {'grad_norm':>12}")
    print(f"{0:>5}  {step0_total:>13.6f}  {step0_total - mean_chance_ce:>+14.6f}  {'--':>18}  {'--':>17}  {'--':>10}  {'--':>10}  {'--':>12}")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        total_loss_value = 0.0
        l1_masked_ce_value = 0.0
        all_gates = []
        for (features, raw_scores, is_present), cand_idx in zip(samples, cand_idxs):
            result, gate = forward_loss(edge_gate, cand_idx, features, raw_scores, is_present,
                                         alpha=0.98, device=args.device)
            (result["total"] / len(samples)).backward()
            total_loss_value += float(result["total"].detach()) / len(samples)
            l1_masked_ce_value += float(result["l1_masked_ce"]) / len(samples)
            all_gates.append(gate.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(edge_gate.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            first_layer_grad = edge_gate.mlp[0].weight.grad.norm().item()
            last_layer_grad = edge_gate.mlp[2].weight.grad.norm().item()
            gate_cat = torch.cat([g.flatten() for g in all_gates])
            gate_mean = gate_cat.mean().item()
            gate_std = gate_cat.std().item()
            row = {
                "step": step, "l1_masked_ce": l1_masked_ce_value,
                "gap_vs_chance": l1_masked_ce_value - mean_chance_ce,
                "first_layer_grad": first_layer_grad, "last_layer_grad": last_layer_grad,
                "gate_mean": gate_mean, "gate_std": gate_std, "grad_norm": float(grad_norm),
            }
            rows.append(row)
            print(f"{step:>5}  {row['l1_masked_ce']:>13.6f}  {row['gap_vs_chance']:>+14.6f}  "
                  f"{first_layer_grad:>18.6e}  {last_layer_grad:>17.6e}  {gate_mean:>10.6f}  "
                  f"{gate_std:>10.6f}  {row['grad_norm']:>12.6e}")

    step500_l1 = rows[-1]["l1_masked_ce"]
    threshold = step0_total * (1 - PASS_FRACTION)
    passed = step500_l1 <= threshold
    pct_change = (step500_l1 - step0_total) / step0_total * 100

    print()
    print(f"step0_l1_masked_ce:   {step0_total:.6f}")
    print(f"step{args.steps}_l1_masked_ce: {step500_l1:.6f}")
    print(f"change: {pct_change:+.2f}%  (PASS requires <= -{PASS_FRACTION*100:.0f}%, "
          f"i.e. step{args.steps} <= {threshold:.6f})")
    if passed:
        print(f"\nPASS: l1_masked_ce dropped {abs(pct_change):.2f}% (>= {PASS_FRACTION*100:.0f}% required) "
              f"-- EdgeGate CAN overfit this single batch.")
        sys.exit(0)
    else:
        print(f"\nFAIL: l1_masked_ce changed {pct_change:+.2f}%, short of the {PASS_FRACTION*100:.0f}% drop "
              f"required.")
        sys.exit(1)


if __name__ == "__main__":
    main()
