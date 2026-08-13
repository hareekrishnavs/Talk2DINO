"""Evaluation glue for EdgeGate (Part G), mirroring evaluate.py's
evaluate_with_learned_metric structure exactly, but swapping the unary
metric(features)->build_differentiable_knn_graph step for EdgeGate's
(cand_idx computed once from frozen features) -> EdgeGate(features,
cand_idx) step. No training here -- inference/eval only, torch.no_grad
throughout, matching evaluate.py's own scope note."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.e3_affinity_oracle import (
    AffinityOracleError,
    confusion_from_prediction,
    load_annotation,
    load_cache_manifest,
    load_capture_features,
    load_capture_manifest,
    iter_cached_images,
    metrics_from_confusion,
    replay_cached_image,
)

from .edge_gate import EdgeGate, build_frozen_candidate_set


def evaluate_with_edge_gate(
    capture_dir: Path, cache_path: Path, edge_gate: EdgeGate, alpha: float, *,
    device: str = "cpu", propagation_steps: int | None = None,
    K: int | None = None, max_images: int | None = None,
) -> dict[str, Any]:
    """Full-val evaluation using EdgeGate's pairwise weights over a
    candidate set recomputed (from frozen features, no_grad) once per
    window. `K` defaults to `edge_gate.K` if not given -- pass K=12
    explicitly for the G4 identity check against the existing K=12
    production graph; leave it at the module's own K (32) for G5's
    reference point and any other real use."""
    K = edge_gate.K if K is None else K

    started = time.monotonic()
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)

    edge_gate = edge_gate.to(device_value)
    edge_gate.eval()

    capture_manifest = load_capture_manifest(capture_dir)
    cache_manifest = load_cache_manifest(cache_path, verify_shards=False)

    captured_images = {image["image_id"]: image for image in capture_manifest["images"]}
    if max_images is not None:
        captured_images = dict(list(captured_images.items())[:max_images])
    needed_indices: set[int] = set()
    for image in captured_images.values():
        needed_indices.update(w["global_window_index"] for w in image["windows"])
    feature_by_global_index = load_capture_features(
        capture_dir, capture_manifest, device=device_value, needed_indices=needed_indices,
    )

    confusion = torch.zeros(
        (cache_manifest["class_count"], cache_manifest["class_count"]), dtype=torch.int64
    )
    evaluated = 0
    images_skipped = 0
    with torch.no_grad():
        for cache_image, cache_windows in iter_cached_images(cache_path, cache_manifest):
            capture_image = captured_images.get(cache_image["image_id"])
            if capture_image is None:
                images_skipped += 1
                continue
            if len(capture_image["windows"]) != len(cache_windows):
                raise AffinityOracleError(
                    f"window count mismatch for image {cache_image['image_id']!r}"
                )
            rebuilt_windows = []
            for capture_window, cache_row in zip(capture_image["windows"], cache_windows):
                if capture_window["coordinates"] != cache_row["window_coordinates"].tolist():
                    raise AffinityOracleError(
                        f"sliding-window ordering diverged for image {cache_image['image_id']!r}"
                    )
                features32 = F.normalize(
                    feature_by_global_index[capture_window["global_window_index"]].to(
                        device=device_value, dtype=torch.float32,
                    ),
                    dim=-1,
                )
                cand_idx = build_frozen_candidate_set(features32, K=K)
                weights = edge_gate(features32, cand_idx)
                row = dict(cache_row)
                row["knn_indices"] = cand_idx.to(torch.int16).cpu()
                row["knn_weights"] = weights.detach().to(torch.float16).cpu()
                rebuilt_windows.append(row)
            prediction = replay_cached_image(
                cache_image, rebuilt_windows, alpha=alpha, manifest=cache_manifest,
                device=device_value, propagation_steps=propagation_steps,
            )
            target = load_annotation(cache_image["annotation_path"])
            confusion += confusion_from_prediction(
                prediction, target, num_classes=cache_manifest["class_count"],
                ignore_index=cache_manifest["protocol"]["ignore_index"],
            )
            evaluated += 1
    if evaluated == 0:
        raise AffinityOracleError("no capture image matched any cache image by image_id")
    metrics = metrics_from_confusion(confusion)
    metrics.update({
        "evaluated_images": evaluated,
        "images_skipped_no_capture": images_skipped,
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda" else 0
        ),
    })
    return metrics
