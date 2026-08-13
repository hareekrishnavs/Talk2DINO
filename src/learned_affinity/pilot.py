#!/usr/bin/env python3
"""V1-V4: proxy validation pilot study. Trains for a short schedule,
checkpoints at fixed intervals, and at EVERY checkpoint records both the
held-out caption-split loss components AND the full COCO-Stuff val mIoU
(using the CONVERGED CGLS solve, not a fixed-T approximation -- monkeypatch
technique below, identical to CANONICAL.md's re-derivation, so this is
directly comparable to the 29.877244 canonical number). Then computes
Pearson/Spearman correlation between each loss component and mIoU across
checkpoints (V2), and selects a FINAL checkpoint using only a
positively-correlated loss component -- never mIoU itself (V3/V4)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .coco_captions import sample_step_vocabulary
from .train import TrainConfig, build_step_sample, cosine_lr_lambda, save_checkpoint, training_step

CACHE = Path("/scratch/haree/talk2dino_e3_affinity_oracle/cache/full")
CAPTURE_DIR = Path("/scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full")
CANONICAL_MIOU = 29.877244374599126
CANONICAL_AACC = 48.52867057377273
CANONICAL_MACC = 54.13703466982515


def evaluate_full_val_converged(metric, *, device: str = "cuda", max_images: int | None = None) -> dict[str, Any]:
    """E1-style full-val evaluation using the learned metric's graph and the
    CONVERGED CGLS solve (not the fixed-T=320 propagate_scores
    approximation) -- same monkeypatch technique as CANONICAL.md's
    re-derivation, applied here to evaluate_with_learned_metric so the
    graph-substitution-from-learned-metric logic (F1's own function) is
    reused unmodified and only the propagation step is upgraded."""
    import src.e3_affinity_oracle as oracle
    from .evaluate import evaluate_with_learned_metric
    from .implicit_solve import solve_fixed_point

    def propagate_scores_via_cg(raw_scores, knn_indices, knn_weights, alpha, *, propagation_steps=10, alpha_dim="class"):
        if torch.is_tensor(alpha):
            raise NotImplementedError("scalar alpha only")
        scalar = float(alpha)
        if scalar == 0:
            return raw_scores
        dev = raw_scores.device
        indices64 = knn_indices.to(device=dev, dtype=torch.int64)
        weights32 = knn_weights.to(device=dev, dtype=torch.float32)
        s0_pc = raw_scores.float().T.contiguous()
        s_star, _ = solve_fixed_point(s0_pc, indices64, weights32, scalar)
        return s_star.T.contiguous()

    original = oracle.propagate_scores
    oracle.propagate_scores = propagate_scores_via_cg
    try:
        metrics = evaluate_with_learned_metric(
            CAPTURE_DIR, CACHE, metric, 0.98, device=device, propagation_steps=320, max_images=max_images,
        )
    finally:
        oracle.propagate_scores = original
    return metrics


def evaluate_held_out_caption_losses(
    metric, model, vocab, held_out_dataset, *, n_samples: int, alpha: float,
    loss_weights, device, n_distractor: int, seed: int = 0,
) -> dict[str, float]:
    """V1: loss components on the held-out caption split (D5), for
    correlation against mIoU. No gradient / no optimizer step."""
    from .losses import compute_total_loss
    from .metric import build_differentiable_knn_graph

    totals = {
        "l1_masked_ce": 0.0, "l1_unmasked_ce": 0.0, "l2_ranking": 0.0, "l3_anchor": 0.0, "mean_row_entropy": 0.0,
        "num_classes": 0.0, "chance_ce": 0.0, "l1_masked_ce_gap_vs_chance": 0.0,
    }
    n = min(n_samples, len(held_out_dataset))
    with torch.no_grad():
        for i in range(n):
            item = held_out_dataset[i]
            features, raw_scores, is_present = build_step_sample(
                model, vocab, item["crop_bgr"], item["present_nouns"],
                n_distractor=n_distractor, device=device, seed=seed + i,
            )
            g = metric(features)
            indices, weights = build_differentiable_knn_graph(g, k=metric.k, kappa=metric.kappa)
            result = compute_total_loss(
                f=features, g=g, indices=indices, weights=weights, raw_scores=raw_scores,
                is_present=is_present, alpha=alpha, loss_weights=loss_weights,
            )
            for key in totals:
                totals[key] += float(result[key]) / n
    return totals


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks
    return pearson(rank(xs), rank(ys))


def compute_correlations(pilot_log: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """V2: Pearson and Spearman correlation between each loss component and
    mIoU across checkpoints. L1 reported separately from L2 (the prior
    anti-correlation in this project was specifically a caption-contrastive
    (L2-like) effect)."""
    miou = [row["full_val_mIoU"] for row in pilot_log]
    components = ["l1_masked_ce", "l1_unmasked_ce", "l2_ranking", "l3_anchor", "mean_row_entropy"]
    return {
        component: {
            "pearson": pearson([row[component] for row in pilot_log], miou),
            "spearman": spearman([row[component] for row in pilot_log], miou),
        }
        for component in components
    }


def select_checkpoint(
    pilot_log: list[dict[str, Any]], correlations: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    """V3/V4 (F5): select using ONLY a loss component whose measured Pearson
    correlation with mIoU is STRICTLY POSITIVE -- never mIoU itself, and
    never a negative correlation (a prior run selected on l3_anchor at
    Pearson -0.2381, a negatively correlated signal, which should not have
    been selectable). If no component has positive correlation, do NOT
    select -- record that plainly (V4) and fall back to the LAST checkpoint
    trained, with the fallback stated explicitly in the returned finding."""
    candidates = [
        (name, corr["pearson"]) for name, corr in correlations.items()
        if corr["pearson"] is not None and corr["pearson"] > 0
    ]
    if not candidates:
        fallback_row = max(pilot_log, key=lambda row: row["step"])
        return {
            "selected_checkpoint_step": fallback_row["step"],
            "selection_signal": None,
            "selection_signal_correlation": None,
            "no_positively_correlated_signal": True,
            "all_correlations": correlations,
            "finding": (
                "V4: NO loss component showed a positive Pearson correlation with mIoU "
                f"across checkpoints (measured: {correlations}). This is itself the "
                f"reportable finding -- no checkpoint was selected via a validated proxy "
                f"signal. Falling back to the LAST checkpoint trained (step "
                f"{fallback_row['step']}), stated explicitly, not chosen via any signal."
            ),
        }
    # Most positive correlation = strongest validated signal.
    best_name, best_corr = max(candidates, key=lambda item: item[1])
    best_row = max(pilot_log, key=lambda row: row[best_name])
    return {
        "selected_checkpoint_step": best_row["step"],
        "selection_signal": best_name,
        "selection_signal_correlation": best_corr,
        "no_positively_correlated_signal": False,
        "finding": f"selected step {best_row['step']} using {best_name} (Pearson r={best_corr:.4f} vs mIoU)",
    }


def run_pilot(
    config: TrainConfig, *, output_dir: Path, device: str = "cuda",
    held_out_dataset, train_dataset, model, vocab, n_held_out_samples: int = 50,
    full_val_max_images: int | None = None,
) -> list[dict[str, Any]]:
    """V1 loop. NOT invoked automatically by this module -- call explicitly
    with real data/model, see ablationAll/e10_adaptive_diffusion/scripts/run_pilot.py."""
    from .metric import LearnedMetric

    metric = LearnedMetric().to(device)
    optimizer = torch.optim.AdamW(metric.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_lr_lambda(s, total_steps=config.total_steps, warmup_steps=config.warmup_steps),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_log: list[dict[str, Any]] = []
    log_path = output_dir / "pilot_log.jsonl"
    import random as _random
    rng = _random.Random(config.seed)

    for step in range(1, config.total_steps + 1):
        indices_this_step = [rng.randrange(len(train_dataset)) for _ in range(config.batch_size)]
        samples = []
        for i in indices_this_step:
            item = train_dataset[i]
            samples.append(build_step_sample(
                model, vocab, item["crop_bgr"], item["present_nouns"],
                n_distractor=config.n_distractor, device=device,
            ))
        step_log = training_step(metric, optimizer, samples, alpha=0.98, loss_weights=config.loss_weights)
        scheduler.step()

        if step % config.checkpoint_every == 0 or step == config.total_steps:
            ckpt_path = output_dir / f"checkpoint_{step:06d}.pt"
            save_checkpoint(ckpt_path, metric, optimizer, step, config)

            held_out_losses = evaluate_held_out_caption_losses(
                metric, model, vocab, held_out_dataset, n_samples=n_held_out_samples,
                alpha=0.98, loss_weights=config.loss_weights, device=device, n_distractor=config.n_distractor,
            )
            t0 = time.monotonic()
            full_val = evaluate_full_val_converged(metric, device=device, max_images=full_val_max_images)
            eval_seconds = time.monotonic() - t0

            row = {
                "step": step, "checkpoint_path": str(ckpt_path),
                "train_total_loss": step_log["total_loss"], "r": step_log["r"],
                **{f"held_out_{k}": v for k, v in held_out_losses.items()},
                **held_out_losses,  # also unprefixed, for compute_correlations' plain component names
                "full_val_mIoU": full_val["mIoU"], "full_val_aAcc": full_val["aAcc"],
                "full_val_mAcc": full_val["mAcc"], "full_val_eval_seconds": eval_seconds,
            }
            pilot_log.append(row)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[pilot] step={step} train_loss={step_log['total_loss']:.4f} r={step_log['r']:.6f} "
                  f"held_out_l1_masked_ce={held_out_losses['l1_masked_ce']:.4f} "
                  f"(gap_vs_chance={held_out_losses['l1_masked_ce_gap_vs_chance']:+.4f}, "
                  f"C~{held_out_losses['num_classes']:.1f}, ln(C)={held_out_losses['chance_ce']:.4f}) "
                  f"full_val_mIoU={full_val['mIoU']:.4f} (eval took {eval_seconds:.1f}s)")

    correlations = compute_correlations(pilot_log)
    selection = select_checkpoint(pilot_log, correlations)
    summary = {"pilot_log": pilot_log, "correlations": correlations, "selection": selection}
    (output_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== V2: correlations (loss component vs full-val mIoU) ===")
    for name, corr in correlations.items():
        print(f"  {name}: pearson={corr['pearson']} spearman={corr['spearman']}")
    print(f"\n=== V3/V4: checkpoint selection ===\n{selection['finding']}")
    return pilot_log
