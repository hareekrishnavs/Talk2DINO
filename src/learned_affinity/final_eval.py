#!/usr/bin/env python3
"""E1-E4: final evaluation of the selected checkpoint. Full val mIoU/aAcc/
mAcc (E1), per-class IoU deltas vs the canonical fixed-diffusion run with
special attention to the classes that were individually negative there
(E2), thing/stuff group means (E3), and the r=0 identity regression (E4)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .pilot import CANONICAL_AACC, CANONICAL_MACC, CANONICAL_MIOU, evaluate_full_val_converged

CACHE = Path("/scratch/haree/talk2dino_e3_affinity_oracle/cache/full")
CANONICAL_METRICS_PATH = Path(
    "/project/6114407/haree/Talk2DINO/ablationAll/e10_adaptive_diffusion/results/canonical_cg_metrics.json"
)
# The classes reported individually negative under fixed diffusion (given
# directly by the task, not re-derived here).
PREVIOUSLY_NEGATIVE_THING_INDICES = [46, 12, 22, 50, 3]
PREVIOUSLY_NEGATIVE_STUFF_INDICES = [149, 132, 127, 120, 146]
THING_RANGE = range(0, 80)
STUFF_RANGE = range(80, 171)
IDENTITY_TOLERANCE = 5e-3


def load_canonical_per_class_iou() -> tuple[list[str], list[float]]:
    payload = json.loads(CANONICAL_METRICS_PATH.read_text())
    per_class_iou = payload["alpha_0.98_converged_cgls"]["per_class_iou"]
    manifest = json.loads((CACHE / "manifest.json").read_text())
    class_names = manifest["class_names"]
    return class_names, per_class_iou


def per_class_deltas(trained_per_class_iou: list[float], canonical_per_class_iou: list[float]) -> list[float | None]:
    deltas = []
    for trained, canonical in zip(trained_per_class_iou, canonical_per_class_iou):
        if trained is None or canonical is None:
            deltas.append(None)
        else:
            deltas.append(trained - canonical)
    return deltas


def group_mean(values: list[float | None], indices: range) -> float:
    present = [values[i] for i in indices if values[i] is not None]
    return sum(present) / len(present) if present else float("nan")


def run_e1_e3(metric, *, device: str = "cuda", max_images: int | None = None) -> dict[str, Any]:
    """E1 (full val) + E2 (per-class deltas, previously-negative classes
    highlighted) + E3 (thing/stuff group means). `max_images` caps the
    evaluation to a prefix of the val set -- for smoke-testing or a
    time-boxed run; the REPORTED, definitive E1 number should use the full
    5000 images (max_images=None)."""
    class_names, canonical_per_class_iou = load_canonical_per_class_iou()

    full_val = evaluate_full_val_converged(metric, device=device, max_images=max_images)
    deltas = per_class_deltas(full_val["per_class_iou"], canonical_per_class_iou)

    previously_negative = {}
    for idx in PREVIOUSLY_NEGATIVE_THING_INDICES + PREVIOUSLY_NEGATIVE_STUFF_INDICES:
        previously_negative[class_names[idx]] = {
            "class_index": idx,
            "canonical_iou": canonical_per_class_iou[idx],
            "trained_iou": full_val["per_class_iou"][idx],
            "delta": deltas[idx],
            "was_previously_negative_delta": canonical_per_class_iou[idx] is not None,  # informational
        }

    thing_mean_delta = group_mean(deltas, THING_RANGE)
    stuff_mean_delta = group_mean(deltas, STUFF_RANGE)

    return {
        "E1_full_val": {"mIoU": full_val["mIoU"], "aAcc": full_val["aAcc"], "mAcc": full_val["mAcc"]},
        "E1_delta_vs_canonical": {
            "mIoU": full_val["mIoU"] - CANONICAL_MIOU,
            "aAcc": full_val["aAcc"] - CANONICAL_AACC,
            "mAcc": full_val["mAcc"] - CANONICAL_MACC,
        },
        "E2_previously_negative_classes": previously_negative,
        "E3_group_mean_delta": {"things": thing_mean_delta, "stuff": stuff_mean_delta},
        "full_per_class_deltas": {class_names[i]: deltas[i] for i in range(len(class_names))},
    }


def run_e4_identity_regression(metric, *, device: str = "cuda", max_images: int | None = None) -> dict[str, Any]:
    """E4: with r forced to 0, the TRAINED model must still reproduce the
    canonical 29.877244 within 5e-3 -- otherwise the evaluation path has
    drifted and E1 is not comparable. `max_images` is for smoke-testing
    only: 29.877244 is itself a full-5000-image number, so a capped check
    here is a mechanism check, not a literal validation of the tolerance
    -- the REPORTED E4 result should use max_images=None (full val)."""
    # r_override must be threaded through evaluate_with_learned_metric directly
    # (evaluate_full_val_converged doesn't expose it, since E1's normal use
    # is with the learned r, not forced to 0).
    import src.e3_affinity_oracle as oracle
    from .evaluate import evaluate_with_learned_metric
    from .implicit_solve import solve_fixed_point
    from .pilot import CACHE as _CACHE, CAPTURE_DIR as _CAPTURE_DIR

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
            _CAPTURE_DIR, _CACHE, metric, 0.98, device=device, propagation_steps=320,
            r_override=0.0, max_images=max_images,
        )
    finally:
        oracle.propagate_scores = original

    deviation = abs(metrics["mIoU"] - CANONICAL_MIOU)
    passed = deviation <= IDENTITY_TOLERANCE
    return {
        "mIoU_at_r0": metrics["mIoU"], "expected": CANONICAL_MIOU, "deviation": deviation,
        "tolerance": IDENTITY_TOLERANCE, "passed": passed, "evaluated_images": metrics["evaluated_images"],
    }


def run_final_evaluation(
    checkpoint_path: Path, *, device: str = "cuda", output_path: Path | None = None, max_images: int | None = None,
    e4_max_images: int | None = "unset",
) -> dict[str, Any]:
    """`e4_max_images` lets E4 use a DIFFERENT (typically smaller) cap than
    E1-E3: E4 compares against the full-5000-image CANONICAL_MIOU constant,
    so it is only a literal tolerance check at full scale -- at any smaller
    scale it is a mechanism/no-drift sanity check only (see
    run_e4_identity_regression's docstring). E1 is the primary reported
    number and usually deserves the larger budget. Defaults to `max_images`
    if not given, for backward compatibility."""
    from .metric import LearnedMetric
    from .train import load_checkpoint

    if e4_max_images == "unset":
        e4_max_images = max_images

    metric = LearnedMetric().to(device)
    payload = load_checkpoint(checkpoint_path, metric)
    print(f"loaded checkpoint from step {payload['step']}, r={payload['r']:.6f}")

    print("E4: identity regression (r forced to 0)...")
    e4 = run_e4_identity_regression(metric, device=device, max_images=e4_max_images)
    print(f"  mIoU_at_r0={e4['mIoU_at_r0']} expected={e4['expected']} "
          f"deviation={e4['deviation']:.6f} (tol={e4['tolerance']}) "
          f"evaluated_images={e4['evaluated_images']}: {'PASS' if e4['passed'] else 'FAIL'}")
    if not e4["passed"]:
        print("E4 FAILED -- evaluation path has drifted; E1 is not comparable. Reporting anyway, not stopping silently.")
    if e4_max_images is not None:
        print(f"  NOTE: E4 ran on only {e4_max_images} images -- this is a mechanism check, "
              f"not a literal validation against the full-5000-image canonical tolerance.")

    print("E1/E2/E3: full evaluation with the trained (learned r) metric...")
    e1_e3 = run_e1_e3(metric, device=device, max_images=max_images)
    print(f"  E1 full_val: mIoU={e1_e3['E1_full_val']['mIoU']:.4f} "
          f"(delta {e1_e3['E1_delta_vs_canonical']['mIoU']:+.4f} vs canonical)")
    print(f"  E3 group means: things delta={e1_e3['E3_group_mean_delta']['things']:+.4f} "
          f"stuff delta={e1_e3['E3_group_mean_delta']['stuff']:+.4f}")
    print("  E2 previously-negative classes:")

    def fmt(value, spec):
        return "N/A (no ground-truth pixels for this class in the evaluated set)" if value is None else format(value, spec)

    for name, row in e1_e3["E2_previously_negative_classes"].items():
        print(f"    {name} (idx {row['class_index']}): canonical={fmt(row['canonical_iou'], '.4f')} "
              f"trained={fmt(row['trained_iou'], '.4f')} delta={fmt(row['delta'], '+.4f')}")

    result = {"checkpoint": str(checkpoint_path), "checkpoint_step": payload["step"],
              "trained_r": payload["r"], "E4_identity_regression": e4, **e1_e3}
    if output_path:
        output_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {output_path}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None,
                         help="Cap E1/E4 to N images (smoke-testing only -- "
                              "the reported result should omit this for the full 5000).")
    _E4_DEFAULT = object()  # argparse applies `type` to STRING defaults too, so a
                             # string sentinel like "unset" breaks under type=int; a
                             # plain object() is left alone when the flag is omitted.
    parser.add_argument("--e4-max-images", type=int, default=_E4_DEFAULT,
                         help="Independent cap for E4 only (defaults to --max-images). "
                              "E4 compares against a full-5000-image constant, so it is only "
                              "a literal tolerance check when unset; smaller values are a "
                              "mechanism/no-drift sanity check only.")
    args = parser.parse_args()
    e4_max_images = args.max_images if args.e4_max_images is _E4_DEFAULT else args.e4_max_images
    run_final_evaluation(args.checkpoint, device=args.device, output_path=args.output,
                          max_images=args.max_images, e4_max_images=e4_max_images)
