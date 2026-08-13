#!/usr/bin/env python3
"""T1-T4: training loop for LearnedMetric. Optimises ONLY LearnedMetric's
parameters (T1) -- DINOv2, CLIP, the E3 projection, and alpha (fixed at
0.98 throughout) are all frozen. No masks, no pixel labels, no COCO class
list anywhere in this file.

`training_step` (the per-optimizer-step logic: forward, loss, backward,
clip, step, log) is deliberately decoupled from real image/text extraction
-- it takes already-extracted (features, raw_scores, is_present) samples,
so it is unit-testable on synthetic data without a live model or GPU. The
real glue that produces those samples from COCO images/captions via the
frozen model lives in `run_training`/`build_step_sample`, below it.

NOT RUN by this session (per the user's chosen scope: build infrastructure,
pause before the pilot run) -- `main()` is implemented but not invoked."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .coco_captions import (
    build_noun_vocabulary,
    disjoint_by_image_split,
    load_captions,
    sample_step_vocabulary,
)
from .crop_dataset import CocoCaptionCropDataset, load_random_crop_bgr
from .extract import extract_training_sample
from .implicit_solve import LAST_SOLVE_STATS
from .losses import LossWeights, compute_total_loss
from .metric import LearnedMetric, build_differentiable_knn_graph
from .text_vocab import VocabularyEmbeddings, build_merged_config, load_or_build_vocabulary_embeddings

ALPHA = 0.98  # T1: fixed throughout -- never tuned jointly with g


@dataclass
class TrainConfig:
    lr: float = 1e-4
    total_steps: int = 10_000
    warmup_steps: int = 200
    batch_size: int = 4  # T2: gradient-accumulated -- see module docstring in the F2 writeup
    grad_clip: float = 1.0
    checkpoint_every: int = 500
    n_distractor: int = 32
    loss_weights: LossWeights = field(default_factory=LossWeights)
    seed: int = 0


def training_step(
    metric: LearnedMetric,
    optimizer: torch.optim.Optimizer,
    samples: list[tuple[torch.Tensor, torch.Tensor, list[bool]]],
    *,
    alpha: float = ALPHA,
    loss_weights: LossWeights = LossWeights(),
    grad_clip: float = 1.0,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """T1/T2/T3: one optimizer step over `samples` (features [P,D],
    raw_scores [C,P], is_present [C]), gradient-accumulated (loss averaged
    over the batch before backward, matching a normal mini-batch mean --
    NOT batched inside a single CG solve, since each sample has its own
    graph; see the F2 writeup for the per-sample solve cost this implies).
    Returns a dict with every quantity T3 asks to be logged every step."""
    if not samples:
        raise ValueError("training_step requires at least one sample")
    optimizer.zero_grad()

    totals = {
        "l1_masked_ce": 0.0, "l1_unmasked_ce": 0.0, "l2_ranking": 0.0,
        "l3_anchor": 0.0, "l4_entropy_floor": 0.0, "mean_row_entropy": 0.0,
        "num_classes": 0.0, "chance_ce": 0.0, "l1_masked_ce_gap_vs_chance": 0.0,
    }
    forward_iters: list[int] = []
    backward_iters: list[int] = []
    total_loss_value = 0.0

    for features, raw_scores, is_present in samples:
        g = metric(features)
        indices, weights = build_differentiable_knn_graph(g, k=metric.k, kappa=metric.kappa)
        result = compute_total_loss(
            f=features, g=g, indices=indices, weights=weights, raw_scores=raw_scores,
            is_present=is_present, alpha=alpha, loss_weights=loss_weights, generator=generator,
        )
        (result["total"] / len(samples)).backward()
        total_loss_value += float(result["total"].detach()) / len(samples)
        for key in totals:
            totals[key] += float(result[key]) / len(samples)
        forward_iters.append(LAST_SOLVE_STATS.get("forward_iters"))
        backward_iters.append(LAST_SOLVE_STATS.get("backward_iters"))

    grad_norm = torch.nn.utils.clip_grad_norm_(metric.parameters(), grad_clip)
    optimizer.step()

    # Report the EFFECTIVE (clamped) r, matching what forward() actually
    # used -- `metric.r` is the raw, unclamped parameter and nothing stops
    # it drifting outside [0, r_max] between clamp() calls in forward.
    effective_r = float(torch.clamp(metric.r, 0.0, metric.r_max).detach())

    return {
        "total_loss": total_loss_value,
        **totals,
        "r": effective_r,
        "r_raw": float(metric.r.detach()),
        "r_grad": float(metric.r.grad.detach()) if metric.r.grad is not None else None,
        "grad_norm": float(grad_norm),
        "cg_forward_iters": forward_iters,
        "cg_backward_iters": backward_iters,
        "batch_size": len(samples),
    }


def build_step_sample(
    model, vocab: VocabularyEmbeddings, crop_bgr: torch.Tensor, present_nouns: list[str],
    *, n_distractor: int, device, seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
    """D2/D3/D4 glue: real image -> (features, raw_scores, is_present) for
    one training sample, using the frozen model and the cached vocabulary
    embeddings (no re-encoding of text at step time -- D4's per-step class
    list is a pure gather, per D3)."""
    class_list, is_present = sample_step_vocabulary(
        present_nouns, vocab.vocabulary, n_distractor=n_distractor, seed=seed,
    )
    text_embedding = vocab.gather(class_list, device=device)
    features, raw_scores = extract_training_sample(model, crop_bgr, text_embedding)
    return features, raw_scores, is_present


def save_checkpoint(path: Path, metric: LearnedMetric, optimizer, step: int, config: TrainConfig) -> None:
    """T4: checkpoint at fixed intervals, save r with every checkpoint.
    Saves both the raw (unclamped) parameter and the effective (clamped)
    value forward() actually uses -- see training_step's effective_r note."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "metric_state_dict": metric.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "r": float(torch.clamp(metric.r, 0.0, metric.r_max).detach()),
        "r_raw": float(metric.r.detach()),
        "config": asdict(config),
    }, path)


def load_checkpoint(path: Path, metric: LearnedMetric, optimizer=None) -> dict[str, Any]:
    payload = torch.load(path, weights_only=False)
    metric.load_state_dict(payload["metric_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def cosine_lr_lambda(step: int, *, total_steps: int, warmup_steps: int) -> float:
    import math
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def setup_training_data(
    config: TrainConfig, *, output_dir: Path, device: str = "cuda", vocab_image_subset: int | None = None,
):
    """Shared setup for run_training and the pilot runner: load captions,
    build the noun vocabulary (optionally from a SUBSET of images -- the
    full 118,287-image corpus takes ~21 minutes of spaCy tagging alone,
    which both a smoke test and a time-boxed pilot want to avoid paying),
    load the frozen model, encode+cache the vocabulary, build the
    disjoint-by-image train/held-out datasets. Returns
    (model, vocab, train_dataset, held_out_dataset)."""
    import os
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src" / "open_vocabulary_segmentation"))
    import main  # noqa: F401 -- registers the FloatImage pipeline transform, see capture_dino_features.py
    from models import build_model

    import spacy
    nlp = spacy.load("en_core_web_sm")

    images = load_captions()
    if vocab_image_subset is not None:
        import random as _random
        subset_ids = _random.Random(config.seed).sample(list(images.keys()), min(vocab_image_subset, len(images)))
        images = {i: images[i] for i in subset_ids}
    vocabulary, per_image_nouns = build_noun_vocabulary(images, nlp, min_count=5)
    train_ids, val_ids = disjoint_by_image_split(list(images.keys()), val_fraction=0.05, seed=config.seed)

    cfg = build_merged_config()
    model = build_model(cfg.model)
    if device == "cuda":
        model.cuda()
    model.eval()

    vocab_cache = output_dir / "noun_vocabulary_embeddings.pt"
    payload = load_or_build_vocabulary_embeddings(
        vocab_cache, model, vocabulary, cfg.evaluate.template, overwrite=True,
    )
    vocab = VocabularyEmbeddings(payload["vocabulary"], payload["embeddings"])

    from .coco_captions import COCO_IMAGES_TRAIN
    file_names = {image_id: entry["file_name"] for image_id, entry in images.items()}
    train_dataset = CocoCaptionCropDataset(COCO_IMAGES_TRAIN, train_ids, file_names, per_image_nouns, seed=config.seed)
    held_out_dataset = CocoCaptionCropDataset(COCO_IMAGES_TRAIN, val_ids, file_names, per_image_nouns, seed=config.seed)
    return model, vocab, train_dataset, held_out_dataset


def run_training(
    config: TrainConfig, *, output_dir: Path, device: str = "cuda", vocab_image_subset: int | None = None,
) -> None:
    """Full T1-T4 loop against real COCO Captions + the frozen model. NOT
    invoked by this session -- see the module docstring."""
    model, vocab, train_dataset, _held_out_dataset = setup_training_data(
        config, output_dir=output_dir, device=device, vocab_image_subset=vocab_image_subset,
    )

    metric = LearnedMetric().to(device)
    optimizer = torch.optim.AdamW(metric.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_lr_lambda(s, total_steps=config.total_steps, warmup_steps=config.warmup_steps),
    )

    log_path = output_dir / "train_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    import random as _random
    rng = _random.Random(config.seed)

    for step in range(config.total_steps):
        indices_this_step = [rng.randrange(len(train_dataset)) for _ in range(config.batch_size)]
        samples = []
        for i in indices_this_step:
            item = train_dataset[i]
            samples.append(build_step_sample(
                model, vocab, item["crop_bgr"], item["present_nouns"],
                n_distractor=config.n_distractor, device=device,
            ))
        log = training_step(metric, optimizer, samples, alpha=ALPHA, loss_weights=config.loss_weights)
        scheduler.step()
        log["step"] = step
        log["lr"] = scheduler.get_last_lr()[0]
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")

        if step % config.checkpoint_every == 0:
            save_checkpoint(output_dir / f"checkpoint_{step:06d}.pt", metric, optimizer, step, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    config = TrainConfig(lr=args.lr, total_steps=args.total_steps, batch_size=args.batch_size)
    run_training(config, output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
