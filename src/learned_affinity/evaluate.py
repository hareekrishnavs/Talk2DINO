"""Full-val evaluation using the differentiable graph construction, for the
F1f/F1g identity and equivalence checks ONLY -- no training, no loss,
inference/eval only (torch.no_grad throughout)."""
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

from .metric import LearnedMetric, build_differentiable_knn_graph


def evaluate_with_learned_metric(
    capture_dir: Path, cache_path: Path, metric: LearnedMetric, alpha: float, *,
    device: str = "cpu", propagation_steps: int | None = None,
    r_override: float | None = None, max_images: int | None = None,
) -> dict[str, Any]:
    """Mirrors src.e3_affinity_oracle.evaluate_with_rebuilt_graph's loop
    exactly, substituting build_knn_graph(features) with
    metric(features, r_override=...) -> build_differentiable_knn_graph(...).
    Reuses replay_cached_image (unmodified production code) for the actual
    propagation/stitch/rescale -- only the per-window graph is substituted,
    exactly as Part E's evaluate_with_rebuilt_graph does for raw features."""

    started = time.monotonic()
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)

    metric = metric.to(device_value)
    metric.eval()

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
                g = metric(features32, r_override=r_override)
                indices, weights = build_differentiable_knn_graph(g, k=metric.k, kappa=metric.kappa)
                row = dict(cache_row)
                row["knn_indices"] = indices.to(torch.int16).cpu()
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
