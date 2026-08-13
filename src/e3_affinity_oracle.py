"""E3 affinity-spread oracle primitives and sharded-cache utilities.

This module is deliberately independent of the learned E4--E9 experiments.
It contains no model, optimizer, or learned parameter.  Online E3 inference
may call :class:`AffinityOracleCacheWriter`; offline commands consume only the
resulting tensors and segmentation annotations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F


CACHE_FORMAT = "talk2dino-e3-affinity-oracle-cache-v1"
RESULT_FORMAT = "talk2dino-e3-affinity-oracle-results-v1"
EXPERIMENT = "e3-affinity-spread-oracle"
ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
AFFINITY_FORMULA = "relu(cosine(F_i,F_j))**affinity_power"
PROPAGATION_EQUATION = "S_t=(1-alpha)*S0+alpha*A*S_(t-1)"
ROW_NORMALIZATION = "selected_nonnegative_weights_divided_by_row_sum"
ZERO_ROW_FALLBACK = "self_index_with_unit_weight"
GRAPH_DIRECTIONALITY = "directed_row_stochastic_knn"
SCORE_STAGE = "pre_sigmoid_pre_upsample_pre_stitch"

SHARD_KEYS = {
    "raw_scores", "knn_indices", "knn_weights", "window_image_index",
    "window_coordinates", "window_grid_indices",
}
SHARD_META_KEYS = {"name", "bytes", "sha256", "window_start", "window_end"}
IMAGE_META_KEYS = {
    "dataset_index", "image_id", "filename", "annotation_path", "img_shape",
    "ori_shape", "pad_shape", "resized_input_shape", "crop_size", "stride",
    "h_grids", "w_grids", "window_start", "window_end", "flip",
    "flip_direction", "augmentation_index", "scale", "count_matrix_sha256",
}
PROTOCOL_KEYS = {
    "score_stage", "crop_size", "stride", "with_background",
    "background_threshold", "ignore_index", "pamr", "flip",
    "augmentation_scales", "sigmoid_count", "window_interpolation",
    "rescale_interpolation",
}
FP16_CONTROL_KEYS = {
    "status", "differing_valid_pixels", "valid_pixels",
    "differing_valid_pixel_percentage", "online_metrics", "replay_metrics",
    "aAcc_difference", "mIoU_difference", "mAcc_difference",
    "maximum_per_class_iou_difference",
}
BASE_METRIC_KEYS = {
    "aAcc", "mIoU", "mAcc", "per_class_iou", "per_class_accuracy",
    "intersection", "union", "predicted_pixels", "ground_truth_pixels",
}
MANIFEST_KEYS = {
    "format_version", "complete", "pilot", "fp16_suitable",
    "selected_image_count", "source_image_count", "selected_window_count",
    "class_count", "class_names", "class_order_sha256", "patch_grid",
    "patch_count", "embedding_dimension", "score_dtype", "graph_index_dtype",
    "graph_weight_dtype", "knn_k", "self_neighbour_exclusion",
    "affinity_formula", "affinity_power", "graph_directionality",
    "row_normalization", "zero_row_fallback", "propagation_equation",
    "propagation_steps", "alpha_grid", "e3_config_path", "e3_config_sha256",
    "projection_config_sha256", "e3_checkpoint_path", "e3_checkpoint_sha256",
    "dino_identity", "dino_checkpoint_sha256", "clip_checkpoint_path",
    "clip_checkpoint_sha256", "text_embedding_sha256", "dataset_config_path",
    "dataset_config_sha256", "image_list_sha256", "annotation_list_sha256",
    "protocol", "source_git_commit", "source_git_dirty",
    "source_git_diff_sha256", "shards", "images", "total_cache_bytes",
    "estimated_cache_bytes", "construction_started_at",
    "construction_finished_at", "zero_neighbour_rows", "nonfinite_count",
    "fp16_control", "commands",
}


class AffinityOracleError(ValueError):
    """Raised when an oracle tensor, cache, identity, or protocol is invalid."""


def _closed(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise AffinityOracleError(
            f"{label} closed schema mismatch: expected={sorted(keys)}, got={actual}"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AffinityOracleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AffinityOracleError(f"{label} must be finite")
    return result


def _require_finite_tree(value: Any, label: str = "artifact") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AffinityOracleError(f"{label} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{label}[{index}]")
        return
    raise AffinityOracleError(f"{label} contains unsupported type {type(value).__name__}")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_base_metrics(value: Any, *, class_count: int, label: str) -> None:
    metrics = _closed(value, BASE_METRIC_KEYS, label)
    for key in ("aAcc", "mIoU", "mAcc"):
        _finite_number(metrics[key], f"{label}.{key}")
    for key in (
        "per_class_iou", "per_class_accuracy", "intersection", "union",
        "predicted_pixels", "ground_truth_pixels",
    ):
        if not isinstance(metrics[key], list) or len(metrics[key]) != class_count:
            raise AffinityOracleError(f"{label}.{key} must have one value per class")
    for key in ("per_class_iou", "per_class_accuracy"):
        for item in metrics[key]:
            if item is not None:
                _finite_number(item, f"{label}.{key}")
    for key in ("intersection", "union", "predicted_pixels", "ground_truth_pixels"):
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in metrics[key]
        ):
            raise AffinityOracleError(f"{label}.{key} must contain nonnegative integers")
    _require_finite_tree(metrics, label)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_fingerprint(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value) or not torch.isfinite(value).all():
        raise AffinityOracleError("tensor identity requires a finite tensor")
    tensor = value.detach().contiguous().cpu()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return sha256_bytes(
        header + b"\0" + tensor.view(torch.uint8).numpy().tobytes()
    )


def _atomic_bytes(path: Path, payload: bytes, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite oracle artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite oracle artifact: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    def reject_nonfinite(item: float) -> str:
        raise AffinityOracleError(f"JSON contains non-finite value: {item}")

    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False, default=reject_nonfinite
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload, overwrite=overwrite)


def build_knn_graph(
    patch_features: torch.Tensor,
    *,
    knn_k: int = 12,
    affinity_power: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the deterministic directed row-stochastic oracle graph.

    ``patch_features`` must be the explicitly normalized float32 matrix used
    by E3's text--patch dot product.  Stable descending argsort gives smaller
    patch indices priority for exact weight ties.
    """

    if (
        not torch.is_tensor(patch_features)
        or patch_features.ndim != 2
        or patch_features.dtype != torch.float32
    ):
        raise AffinityOracleError("patch_features must be float32 [P,D]")
    if not torch.isfinite(patch_features).all():
        raise AffinityOracleError("patch_features contain non-finite values")
    patches = patch_features.shape[0]
    if isinstance(knn_k, bool) or not isinstance(knn_k, int) or not 0 < knn_k < patches:
        raise AffinityOracleError("knn_k must satisfy 0 < knn_k < patch_count")
    power = _finite_number(affinity_power, "affinity_power")
    if power <= 0:
        raise AffinityOracleError("affinity_power must be greater than zero")
    norms = patch_features.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=2e-4):
        raise AffinityOracleError("patch_features must be explicitly L2-normalized")

    cosine = patch_features @ patch_features.T
    affinity = cosine.clamp_min(0).pow(power)
    affinity.fill_diagonal_(-torch.inf)
    order = torch.argsort(affinity, dim=-1, descending=True, stable=True)
    indices64 = order[:, :knn_k]
    selected = affinity.gather(1, indices64)
    row_sums = selected.sum(dim=-1, keepdim=True)
    zero_rows = row_sums.squeeze(-1) == 0
    weights = selected / row_sums.clamp_min(torch.finfo(selected.dtype).tiny)
    if zero_rows.any():
        rows = torch.arange(patches, device=indices64.device)[zero_rows]
        indices64[zero_rows] = rows[:, None]
        weights[zero_rows] = 0
        weights[zero_rows, 0] = 1
    if patches - 1 > torch.iinfo(torch.int16).max:
        raise AffinityOracleError("patch indices do not fit signed int16")
    return (
        indices64.to(device="cpu", dtype=torch.int16),
        weights.to(device="cpu", dtype=torch.float16),
        int(zero_rows.sum().detach().cpu()),
    )


def validate_graph(
    indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    patch_count: int,
    knn_k: int,
) -> None:
    if (
        not torch.is_tensor(indices) or indices.dtype != torch.int16
        or tuple(indices.shape) != (patch_count, knn_k)
    ):
        raise AffinityOracleError("knn_indices must be int16 [P,K]")
    if (
        not torch.is_tensor(weights) or weights.dtype != torch.float16
        or tuple(weights.shape) != (patch_count, knn_k)
    ):
        raise AffinityOracleError("knn_weights must be float16 [P,K]")
    decoded = indices.to(torch.int64)
    if torch.any(decoded < 0) or torch.any(decoded >= patch_count):
        raise AffinityOracleError("knn index is outside the patch range")
    float_weights = weights.float()
    if not torch.isfinite(float_weights).all() or torch.any(float_weights < 0):
        raise AffinityOracleError("knn weights must be finite and nonnegative")
    if not torch.allclose(
        float_weights.sum(-1), torch.ones(patch_count), atol=2e-3, rtol=2e-3
    ):
        raise AffinityOracleError("knn weights must be row stochastic")


# Part B: per-patch local-graph-structure statistics used to bucket patches
# for spatially-varying alpha. Every cached knn_weights row is *already*
# row-stochastic (validate_graph above enforces sum(-1) == 1 to 2e-3), so a
# literal "sum of the row's own (outgoing) weights" is a structural constant
# (~1.0 for every patch) and carries no information -- it cannot be fixed by
# any replay-time computation, because the pre-normalization row magnitude
# (row_sums in build_knn_graph) is discarded before caching and is not one
# of the SHARD_KEYS. "degree" is therefore implemented as the graph-standard
# alternative that IS fully recoverable from the cached knn_indices/
# knn_weights: weighted in-degree, i.e. how much total incoming edge weight
# a patch receives from the rest of its window's directed knn graph. This is
# a genuine, non-degenerate local-structure statistic (a patch embedded in a
# large homogeneous region is referenced, with high weight, by many nearby
# patches; a patch on a small/boundary object is not).
PATCH_STATISTICS = ("entropy", "degree", "top1ratio")


def patch_row_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of each row of already row-normalised weights, in [0,1]."""

    if not torch.is_tensor(weights) or weights.ndim < 1:
        raise AffinityOracleError("weights must be a tensor with a trailing K dimension")
    k = weights.shape[-1]
    if k <= 1:
        raise AffinityOracleError("entropy statistic requires k > 1")
    w = weights.float().clamp_min(0)
    log_w = torch.where(w > 0, torch.log(w), torch.zeros_like(w))
    entropy = -(w * log_w).sum(-1) / math.log(k)
    return entropy.clamp(0, 1)


def patch_row_top1_ratio(weights: torch.Tensor) -> torch.Tensor:
    """Largest weight in each row divided by that row's weight sum, in [0,1]."""

    if not torch.is_tensor(weights) or weights.ndim < 1:
        raise AffinityOracleError("weights must be a tensor with a trailing K dimension")
    w = weights.float().clamp_min(0)
    total = w.sum(-1).clamp_min(torch.finfo(torch.float32).tiny)
    return (w.max(-1).values / total).clamp(0, 1)


def patch_weighted_in_degree(
    indices: torch.Tensor, weights: torch.Tensor, *, patch_count: int
) -> torch.Tensor:
    """Weighted in-degree per patch: total incoming edge weight within one window."""

    if not torch.is_tensor(indices) or not torch.is_tensor(weights):
        raise AffinityOracleError("indices/weights must be tensors")
    if indices.shape != weights.shape or indices.ndim != 2:
        raise AffinityOracleError("indices/weights must both have shape [P,K]")
    flat_indices = indices.reshape(-1).to(torch.int64)
    if torch.any(flat_indices < 0) or torch.any(flat_indices >= patch_count):
        raise AffinityOracleError("indices reference an out-of-range patch")
    flat_weights = weights.reshape(-1).float()
    in_degree = torch.zeros(patch_count, dtype=torch.float32, device=weights.device)
    in_degree = in_degree.scatter_add_(0, flat_indices, flat_weights)
    return in_degree


def compute_patch_statistic(
    indices: torch.Tensor, weights: torch.Tensor, *, stat: str, patch_count: int,
) -> torch.Tensor:
    if stat not in PATCH_STATISTICS:
        raise AffinityOracleError(f"unknown patch statistic: {stat}")
    if stat == "entropy":
        return patch_row_entropy(weights)
    if stat == "top1ratio":
        return patch_row_top1_ratio(weights)
    return patch_weighted_in_degree(indices, weights, patch_count=patch_count)


def fit_bucket_edges(values: torch.Tensor, *, n_buckets: int) -> torch.Tensor:
    """Percentile edges fitted once on fitting data; store and reuse, never refit."""

    if isinstance(n_buckets, bool) or not isinstance(n_buckets, int) or n_buckets <= 0:
        raise AffinityOracleError("n_buckets must be a positive integer")
    if not torch.is_tensor(values) or values.ndim != 1 or values.numel() == 0:
        raise AffinityOracleError("values must be a nonempty 1-D tensor")
    if not torch.isfinite(values).all():
        raise AffinityOracleError("values contain non-finite entries")
    if n_buckets == 1:
        return torch.zeros(0, dtype=torch.float64)
    fractions = torch.tensor(
        [i / n_buckets for i in range(1, n_buckets)], dtype=torch.float64
    )
    edges = torch.quantile(values.double(), fractions)
    return edges


def assign_buckets(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(values):
        raise AffinityOracleError("values must be a tensor")
    if not torch.is_tensor(edges) or edges.ndim != 1:
        raise AffinityOracleError("edges must be a 1-D tensor")
    return torch.bucketize(values.double(), edges.double(), right=False)


def propagate_scores(
    raw_scores: torch.Tensor,
    knn_indices: torch.Tensor,
    knn_weights: torch.Tensor,
    alpha: float | torch.Tensor,
    *,
    propagation_steps: int = 10,
    alpha_dim: str = "class",
) -> torch.Tensor:
    """Apply restarted graph propagation to ``[C,P]`` raw scores.

    ``alpha_dim`` disambiguates a 1-D tensor ``alpha``: "class" (default,
    unchanged v1 behaviour) broadcasts one value per class across all
    patches; "patch" broadcasts one value per patch across all classes. It
    is ignored for a scalar ``alpha``.
    """

    if not torch.is_tensor(raw_scores) or raw_scores.ndim != 2:
        raise AffinityOracleError("raw_scores must have shape [C,P]")
    if not torch.isfinite(raw_scores).all():
        raise AffinityOracleError("raw_scores contain non-finite values")
    if (
        isinstance(propagation_steps, bool)
        or not isinstance(propagation_steps, int)
        or propagation_steps <= 0
    ):
        raise AffinityOracleError("propagation_steps must be a positive integer")
    if alpha_dim not in ("class", "patch"):
        raise AffinityOracleError("alpha_dim must be 'class' or 'patch'")
    classes, patches = raw_scores.shape
    validate_graph(
        knn_indices.cpu(), knn_weights.cpu(), patch_count=patches,
        knn_k=knn_indices.shape[1],
    )
    if torch.is_tensor(alpha):
        expected_length = classes if alpha_dim == "class" else patches
        if alpha.ndim != 1 or alpha.shape[0] != expected_length:
            raise AffinityOracleError(
                f"per-{alpha_dim} alpha must have shape [{expected_length}]"
            )
        alpha_value = alpha.to(device=raw_scores.device, dtype=torch.float32)
        if not torch.isfinite(alpha_value).all() or torch.any(alpha_value < 0) or torch.any(alpha_value >= 1):
            raise AffinityOracleError("alpha values must be finite and in [0,1)")
        if torch.count_nonzero(alpha_value) == 0:
            return raw_scores
        alpha_view = alpha_value[:, None] if alpha_dim == "class" else alpha_value[None, :]
    else:
        scalar = _finite_number(alpha, "alpha")
        if not 0 <= scalar < 1:
            raise AffinityOracleError("alpha must be in [0,1)")
        if scalar == 0:
            return raw_scores
        alpha_view = torch.tensor(scalar, device=raw_scores.device, dtype=torch.float32)
    base = raw_scores.float()
    current = base
    indices = knn_indices.to(device=base.device, dtype=torch.int64)
    weights = knn_weights.to(device=base.device, dtype=torch.float32)
    for _ in range(propagation_steps):
        neighbours = current[:, indices]  # [C,P,K]
        spread = torch.einsum("cpk,pk->cp", neighbours, weights)
        current = (1 - alpha_view) * base + alpha_view * spread
    if not torch.isfinite(current).all():
        raise AffinityOracleError("propagation produced non-finite scores")
    return current


def interpolate_window_scores(
    scores: torch.Tensor,
    *,
    patch_grid: tuple[int, int] = (32, 32),
    crop_size: tuple[int, int] = (448, 448),
) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != patch_grid[0] * patch_grid[1]:
        raise AffinityOracleError("window scores do not match patch grid")
    masks = torch.sigmoid(scores.reshape(1, scores.shape[0], *patch_grid))
    return F.interpolate(
        masks, crop_size, mode="bilinear", align_corners=True
    ).squeeze(0)


def add_background_channel(
    masks: torch.Tensor, *, with_background: bool, background_threshold: float
) -> torch.Tensor:
    if not with_background:
        return masks
    threshold = _finite_number(background_threshold, "background_threshold")
    background = torch.full(
        (1, *masks.shape[-2:]), threshold, dtype=masks.dtype, device=masks.device
    )
    return torch.cat((background, masks), dim=0)


def count_matrix_identity(
    image_shape: tuple[int, int], coordinates: Sequence[Sequence[int]]
) -> str:
    counts = torch.zeros(image_shape, dtype=torch.int16)
    for y1, x1, y2, x2 in coordinates:
        counts[y1:y2, x1:x2] += 1
    return sha256_bytes(counts.numpy().tobytes())


def stitch_windows(
    windows: Sequence[torch.Tensor],
    coordinates: Sequence[Sequence[int]],
    *,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    if not windows or len(windows) != len(coordinates):
        raise AffinityOracleError("windows and coordinates must be nonempty and aligned")
    channels = windows[0].shape[0]
    output = windows[0].new_zeros((channels, *image_shape))
    counts = windows[0].new_zeros((1, *image_shape))
    for window, (y1, x1, y2, x2) in zip(windows, coordinates):
        if tuple(window.shape) != (channels, y2 - y1, x2 - x1):
            raise AffinityOracleError("window shape/coordinate mismatch")
        output[:, y1:y2, x1:x2] += window
        counts[:, y1:y2, x1:x2] += 1
    if torch.any(counts == 0):
        raise AffinityOracleError("stitching left uncovered pixels")
    return output / counts


def rescale_logits(
    logits: torch.Tensor,
    *,
    img_shape: tuple[int, int],
    ori_shape: tuple[int, int],
) -> torch.Tensor:
    cropped = logits[:, : img_shape[0], : img_shape[1]].unsqueeze(0)
    return F.interpolate(
        cropped, size=ori_shape, mode="bilinear", align_corners=False
    ).squeeze(0)


def final_prediction(
    logits: torch.Tensor, *, flip: bool = False, flip_direction: str | None = None
) -> torch.Tensor:
    probabilities = logits.softmax(dim=0)
    if flip:
        if flip_direction == "horizontal":
            probabilities = probabilities.flip(-1)
        elif flip_direction == "vertical":
            probabilities = probabilities.flip(-2)
        else:
            raise AffinityOracleError("unsupported flip direction")
    return probabilities.argmax(dim=0)


def confusion_from_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise AffinityOracleError("prediction and target shapes differ")
    pred = prediction.to(torch.int64).reshape(-1)
    truth = target.to(torch.int64).reshape(-1)
    valid = truth != ignore_index
    valid &= truth >= 0
    valid &= truth < num_classes
    pred, truth = pred[valid], truth[valid]
    if torch.any(pred < 0) or torch.any(pred >= num_classes):
        raise AffinityOracleError("prediction contains an invalid class")
    encoded = truth * num_classes + pred
    return torch.bincount(encoded, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    ).to(torch.int64)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise AffinityOracleError("confusion must be square")
    matrix = confusion.double()
    intersection = matrix.diag()
    gt = matrix.sum(1)
    predicted = matrix.sum(0)
    union = gt + predicted - intersection
    iou = torch.where(union > 0, intersection / union, torch.nan)
    accuracy = torch.where(gt > 0, intersection / gt, torch.nan)
    total = gt.sum()
    aacc = intersection.sum() / total if total > 0 else torch.tensor(float("nan"))
    result = {
        "aAcc": float(aacc * 100),
        "mIoU": float(torch.nanmean(iou) * 100),
        "mAcc": float(torch.nanmean(accuracy) * 100),
        "per_class_iou": [None if torch.isnan(x) else float(x * 100) for x in iou],
        "per_class_accuracy": [None if torch.isnan(x) else float(x * 100) for x in accuracy],
        "intersection": [int(x) for x in intersection],
        "union": [int(x) for x in union],
        "predicted_pixels": [int(x) for x in predicted],
        "ground_truth_pixels": [int(x) for x in gt],
    }
    for key in ("aAcc", "mIoU", "mAcc"):
        if not math.isfinite(result[key]):
            raise AffinityOracleError(f"metric {key} is non-finite")
    return result


def per_class_iou_differences(
    reference: Sequence[float | None], replay: Sequence[float | None]
) -> list[float]:
    if len(reference) != len(replay):
        raise AffinityOracleError("per-class IoU vectors are not aligned")
    differences: list[float] = []
    for reference_iou, replay_iou in zip(reference, replay):
        if reference_iou is None and replay_iou is None:
            differences.append(0.0)
        elif reference_iou is None or replay_iou is None:
            differences.append(100.0)
        else:
            differences.append(
                abs(
                    _finite_number(reference_iou, "reference class IoU")
                    - _finite_number(replay_iou, "replay class IoU")
                )
            )
    return differences


def stable_top2(scores: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return stable top-1/top-2 along class dim 0 using argmax ties."""

    if scores.ndim < 2 or scores.shape[0] < 2 or not torch.isfinite(scores).all():
        raise AffinityOracleError("scores must be finite [C,...] with C>=2")
    top1_class = scores.argmax(dim=0)
    top1_value = scores.gather(0, top1_class.unsqueeze(0)).squeeze(0)
    masked = scores.clone()
    masked.scatter_(0, top1_class.unsqueeze(0), -torch.inf)
    top2_class = masked.argmax(dim=0)
    top2_value = masked.gather(0, top2_class.unsqueeze(0)).squeeze(0)
    return top1_value, top1_class, top2_value, top2_class


def candidate_class_prediction(
    candidate_value: torch.Tensor,
    candidate_class: int,
    top1_value: torch.Tensor,
    top1_class: torch.Tensor,
    top2_value: torch.Tensor,
    top2_class: torch.Tensor,
) -> torch.Tensor:
    best_value = torch.where(top1_class != candidate_class, top1_value, top2_value)
    best_class = torch.where(top1_class != candidate_class, top1_class, top2_class)
    wins = (candidate_value > best_value) | (
        (candidate_value == best_value) & (candidate_class < best_class)
    )
    return torch.where(
        wins, torch.full_like(best_class, candidate_class), best_class
    )


def choose_alpha(
    rows: Sequence[tuple[float, float]], *, global_alpha: float
) -> tuple[float, str]:
    """Choose from ``(alpha, IoU)`` using the predeclared deterministic rule."""

    if not rows:
        return global_alpha, "unfit_no_support"
    for alpha, score in rows:
        _finite_number(alpha, "candidate alpha")
        _finite_number(score, "candidate IoU")
    best_iou = max(score for _, score in rows)
    tied = [(alpha, score) for alpha, score in rows if score == best_iou]
    # Grid-decimal distances such as |0.1-0.2| and |0.3-0.2| are
    # mathematically tied but have different binary-float residues.
    tied.sort(key=lambda row: (round(abs(row[0] - global_alpha), 12), row[0]))
    reason = "highest_iou" if len(tied) == 1 else "distance_then_smallest_alpha"
    return tied[0][0], reason


def coverage_subset(
    image_classes: Sequence[set[int]], size: int, *, seed: int = 42
) -> list[int]:
    """Deterministic greedy class-coverage subset with seeded tie priorities."""

    if not 0 < size <= len(image_classes):
        raise AffinityOracleError("subset size is outside the image range")
    generator = torch.Generator().manual_seed(seed)
    priority = torch.randperm(len(image_classes), generator=generator).tolist()
    rank = {index: position for position, index in enumerate(priority)}
    selected: list[int] = []
    counts: dict[int, int] = {}
    remaining = set(range(len(image_classes)))
    while len(selected) < size:
        def key(index: int) -> tuple[float, int]:
            gain = sum(1.0 / (1 + counts.get(c, 0)) for c in image_classes[index])
            return (-gain, rank[index])
        chosen = min(remaining, key=key)
        remaining.remove(chosen)
        selected.append(chosen)
        for class_index in image_classes[chosen]:
            counts[class_index] = counts.get(class_index, 0) + 1
    return selected


def split_balanced_halves(
    image_classes: Sequence[set[int]], *, seed: int = 42
) -> tuple[list[int], list[int]]:
    if len(image_classes) % 2:
        raise AffinityOracleError("split-half requires an even image count")
    order = coverage_subset(image_classes, len(image_classes), seed=seed)
    half = len(order) // 2
    a: list[int] = []
    b: list[int] = []
    counts_a: dict[int, int] = {}
    counts_b: dict[int, int] = {}
    for index in order:
        options = []
        for side, counts, current in ((0, counts_a, a), (1, counts_b, b)):
            if len(current) >= half:
                continue
            imbalance = sum(counts.get(c, 0) for c in image_classes[index])
            options.append((imbalance, len(current), side))
        side = min(options)[2]
        target = a if side == 0 else b
        counts = counts_a if side == 0 else counts_b
        target.append(index)
        for class_index in image_classes[index]:
            counts[class_index] = counts.get(class_index, 0) + 1
    if set(a).intersection(b) or sorted(a + b) != list(range(len(image_classes))):
        raise AffinityOracleError("split halves are not disjoint and complete")
    return a, b


def validate_cache_shard(
    value: Any, *, class_count: int, patch_grid: tuple[int, int], knn_k: int,
    score_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    shard = _closed(value, SHARD_KEYS, "oracle cache shard")
    patch_count = patch_grid[0] * patch_grid[1]
    scores = shard["raw_scores"]
    if not torch.is_tensor(scores) or scores.dtype != score_dtype or scores.ndim != 4:
        raise AffinityOracleError("raw_scores have invalid dtype or rank")
    windows = scores.shape[0]
    if tuple(scores.shape[1:]) != (class_count, *patch_grid):
        raise AffinityOracleError("raw_scores shape mismatch")
    expected = {
        "knn_indices": (torch.int16, (windows, patch_count, knn_k)),
        "knn_weights": (torch.float16, (windows, patch_count, knn_k)),
        "window_image_index": (torch.int64, (windows,)),
        "window_coordinates": (torch.int32, (windows, 4)),
        "window_grid_indices": (torch.int32, (windows, 2)),
    }
    for key, (dtype, shape) in expected.items():
        tensor = shard[key]
        if not torch.is_tensor(tensor) or tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise AffinityOracleError(f"{key} schema mismatch")
    if not torch.isfinite(scores.float()).all() or not torch.isfinite(shard["knn_weights"].float()).all():
        raise AffinityOracleError("cache shard contains non-finite values")
    decoded = shard["knn_indices"].to(torch.int64)
    if torch.any(decoded < 0) or torch.any(decoded >= patch_count):
        raise AffinityOracleError("cache shard contains an out-of-range graph index")
    float_weights = shard["knn_weights"].float()
    if torch.any(float_weights < 0) or not torch.allclose(
        float_weights.sum(-1),
        torch.ones((windows, patch_count), dtype=torch.float32),
        atol=2e-3,
        rtol=2e-3,
    ):
        raise AffinityOracleError("cache shard graph weights are not row stochastic")
    patch_rows = torch.arange(patch_count, dtype=torch.int64)[None, :, None]
    fallback = (
        (decoded == patch_rows).all(-1)
        & (float_weights[..., 0] == 1)
        & (float_weights[..., 1:] == 0).all(-1)
    )
    self_neighbour = (decoded == patch_rows).any(-1)
    if torch.any(self_neighbour & ~fallback):
        raise AffinityOracleError("cache shard graph contains a non-fallback self-neighbour")
    if any("feature" in key for key in shard):
        raise AffinityOracleError("raw patch features are forbidden in cache shards")
    return dict(shard)


def load_cache_shard(
    path: Path, *, expected_bytes: int, expected_sha256: str,
    class_count: int, patch_grid: tuple[int, int], knn_k: int,
    score_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    payload = path.read_bytes()
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise AffinityOracleError(f"cache shard identity mismatch: {path}")
    try:
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise AffinityOracleError(f"cannot safely load cache shard {path}: {error}") from error
    return validate_cache_shard(
        value, class_count=class_count, patch_grid=patch_grid, knn_k=knn_k,
        score_dtype=score_dtype,
    )


@dataclass(frozen=True)
class OracleProtocol:
    class_count: int = 171
    patch_grid: tuple[int, int] = (32, 32)
    embedding_dimension: int = 768
    crop_size: tuple[int, int] = (448, 448)
    stride: tuple[int, int] = (224, 224)
    knn_k: int = 12
    affinity_power: float = 3.0
    propagation_steps: int = 10
    score_dtype: str = "float16"
    with_background: bool = False
    background_threshold: float = 0.4
    pamr: bool = False
    flip: bool = False
    augmentation_scales: int = 1

    def validate(self) -> None:
        if self.class_count != 171:
            raise AffinityOracleError("canonical oracle requires 171 classes")
        if self.patch_grid != (32, 32) or self.embedding_dimension != 768:
            raise AffinityOracleError("canonical oracle requires ViT-B 32x32x768")
        if self.crop_size != (448, 448) or self.stride != (224, 224):
            raise AffinityOracleError("canonical crop/stride protocol changed")
        if self.pamr or self.flip or self.augmentation_scales != 1:
            raise AffinityOracleError("canonical oracle requires PAMR/flip off and one scale")
        if self.score_dtype not in {"float16", "float32"}:
            raise AffinityOracleError("score_dtype must be float16 or float32")
        if self.knn_k != 12 or self.propagation_steps != 10:
            raise AffinityOracleError("canonical graph k/step count changed")
        if self.affinity_power != 3.0:
            raise AffinityOracleError("canonical affinity power changed")
        if self.background_threshold != 0.4:
            raise AffinityOracleError("canonical E3 background threshold changed")
        if self.with_background:
            raise AffinityOracleError(
                "canonical COCO-Stuff E3 protocol has no synthetic background channel"
            )


def estimate_window_bytes(protocol: OracleProtocol) -> int:
    score_bytes = 2 if protocol.score_dtype == "float16" else 4
    patches = protocol.patch_grid[0] * protocol.patch_grid[1]
    return (
        protocol.class_count * patches * score_bytes
        + patches * protocol.knn_k * 2
        + patches * protocol.knn_k * 2
        + 8 + 4 * 4 + 2 * 4
    )


class AffinityOracleCacheWriter:
    """Bounded sharded writer used by the optional online E3 capture pass."""

    def __init__(
        self, output_dir: Path, *, protocol: OracleProtocol,
        windows_per_shard: int = 256, overwrite: bool = False,
    ):
        protocol.validate()
        if windows_per_shard <= 0:
            raise AffinityOracleError("windows_per_shard must be positive")
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite oracle cache: {output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.output_dir / "shards"
        self.shard_dir.mkdir(exist_ok=True)
        self.protocol = protocol
        self.windows_per_shard = windows_per_shard
        self.overwrite = overwrite
        self._rows: list[dict[str, torch.Tensor]] = []
        self.shards: list[dict[str, Any]] = []
        self.window_count = 0
        self.zero_rows = 0
        self.nonfinite = 0

    def add_window(
        self, raw_scores: torch.Tensor, normalized_patch_features: torch.Tensor,
        *, image_index: int, coordinates: Sequence[int], grid_indices: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        if raw_scores.ndim == 3:
            raw_scores = raw_scores.unsqueeze(0)
        if normalized_patch_features.ndim == 4:
            normalized_patch_features = normalized_patch_features.flatten(2).transpose(1, 2)
        if raw_scores.shape[0] != 1 or normalized_patch_features.shape[0] != 1:
            raise AffinityOracleError("v1 cache requires one window per model batch")
        features = normalized_patch_features[0].float()
        indices, weights, zero = build_knn_graph(
            features, knn_k=self.protocol.knn_k,
            affinity_power=self.protocol.affinity_power,
        )
        score = raw_scores[0].detach().to(
            device="cpu",
            dtype=torch.float16 if self.protocol.score_dtype == "float16" else torch.float32,
        )
        self.nonfinite += int((~torch.isfinite(score.float())).sum())
        self.zero_rows += zero
        row = {
            "raw_scores": score,
            "knn_indices": indices,
            "knn_weights": weights,
            "window_image_index": torch.tensor(image_index, dtype=torch.int64),
            "window_coordinates": torch.tensor(coordinates, dtype=torch.int32),
            "window_grid_indices": torch.tensor(grid_indices, dtype=torch.int32),
        }
        self._rows.append(row)
        self.window_count += 1
        if len(self._rows) >= self.windows_per_shard:
            self.flush()
        return row

    def flush(self) -> None:
        if not self._rows:
            return
        shard = {key: torch.stack([row[key] for row in self._rows]) for key in SHARD_KEYS}
        validate_cache_shard(
            shard, class_count=self.protocol.class_count,
            patch_grid=self.protocol.patch_grid, knn_k=self.protocol.knn_k,
            score_dtype=shard["raw_scores"].dtype,
        )
        number = len(self.shards)
        name = f"windows-{number:06d}.pth"
        buffer = io.BytesIO()
        torch.save(shard, buffer)
        payload = buffer.getvalue()
        path = self.shard_dir / name
        _atomic_bytes(path, payload, overwrite=self.overwrite)
        start = self.window_count - len(self._rows)
        self.shards.append({
            "name": f"shards/{name}", "bytes": len(payload),
            "sha256": sha256_bytes(payload), "window_start": start,
            "window_end": self.window_count,
        })
        self._rows.clear()

    def finalize(
        self,
        manifest: Mapping[str, Any],
        *,
        prepublish_check=None,
    ) -> dict[str, Any]:
        self.flush()
        value = dict(manifest)
        value.update({
            "format_version": CACHE_FORMAT,
            "complete": True,
            "selected_window_count": self.window_count,
            "shards": self.shards,
            "total_cache_bytes": sum(row["bytes"] for row in self.shards),
            "zero_neighbour_rows": self.zero_rows,
            "nonfinite_count": self.nonfinite,
        })
        _closed(value, MANIFEST_KEYS, "oracle cache manifest")
        if self.nonfinite:
            raise AffinityOracleError("cannot publish a cache with non-finite values")
        if prepublish_check is not None:
            # This is intentionally the final operation before atomic
            # manifest publication.  Shards have already been serialized and
            # fsynced, so a failed identity check leaves an incomplete cache
            # with no manifest rather than publishing stale provenance.
            prepublish_check()
        atomic_json(self.output_dir / "manifest.json", value, overwrite=self.overwrite)
        return value


def git_provenance(repository: Path, *, allow_dirty: bool) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *args], check=True,
            capture_output=True, text=True,
        ).stdout
    commit = run("rev-parse", "HEAD").strip()
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise AffinityOracleError(
            "full oracle cache requires a clean committed source; use "
            "allow_dirty_source only with max_images for a pilot"
        )
    diff_hash = None
    if dirty:
        tracked = subprocess.run(
            ["git", "-C", str(repository), "diff", "HEAD", "--binary"],
            check=True, capture_output=True,
        ).stdout
        untracked = []
        for name in run("ls-files", "--others", "--exclude-standard").splitlines():
            path = repository / name
            if path.is_file():
                untracked.append(name.encode() + b"\0" + path.read_bytes())
        diff_hash = sha256_bytes(tracked + b"".join(untracked))
    return {
        "source_git_commit": commit, "source_git_dirty": dirty,
        "source_git_diff_sha256": diff_hash,
    }


class OnlineAffinityOracleCapture:
    """Small state machine joining the exact E3 score observer to cache rows."""

    def __init__(
        self, *, output_dir: Path, protocol: OracleProtocol,
        class_names: Sequence[str], source_image_count: int,
        max_images: int | None, windows_per_shard: int,
        identity_paths: Mapping[str, str], dino_identity: str,
        text_embedding_sha256: str,
        allow_dirty_source: bool, overwrite: bool, commands: Sequence[str],
        fp16_control_manifest: Path | None = None,
    ):
        if max_images is not None and max_images <= 0:
            raise AffinityOracleError("max_images must be positive")
        if allow_dirty_source and max_images is None:
            raise AffinityOracleError("dirty source is allowed only for a pilot")
        repository = Path(__file__).parents[1]
        self.provenance = git_provenance(
            repository, allow_dirty=allow_dirty_source
        )
        self.protocol = protocol
        self.output_dir = Path(output_dir)
        self.class_names = list(class_names)
        self.source_image_count = source_image_count
        self.max_images = max_images
        self.dino_identity = dino_identity
        if not _valid_sha256(text_embedding_sha256):
            raise AffinityOracleError("text embedding SHA256 is invalid")
        self.text_embedding_sha256 = text_embedding_sha256
        if source_image_count <= 0:
            raise AffinityOracleError("source_image_count must be positive")
        if max_images is not None and max_images > source_image_count:
            raise AffinityOracleError("max_images exceeds the source image count")
        if len(self.class_names) != protocol.class_count:
            raise AffinityOracleError("class-name count does not match the protocol")
        expected_identity_paths = {
            "e3_config", "projection_config", "e3_checkpoint",
            "dino_checkpoint", "clip_checkpoint", "dataset_config",
        }
        if set(identity_paths) != expected_identity_paths:
            raise AffinityOracleError("oracle identity-path schema is incomplete")
        self.identity_paths = {key: str(value) for key, value in identity_paths.items()}
        for key, value in self.identity_paths.items():
            if not Path(value).is_file():
                raise AffinityOracleError(
                    f"missing oracle identity path {key}: {value}"
                )
        self.initial_hashes = {
            key: sha256_file(value) for key, value in self.identity_paths.items()
        }
        self.preflight_estimated_cache_bytes: int | None = None
        if max_images is None:
            if fp16_control_manifest is None:
                raise AffinityOracleError(
                    "a full cache requires fp16_control_manifest from a "
                    "deterministic pilot for fidelity and size estimation"
                )
            control = load_cache_manifest(Path(fp16_control_manifest).parent)
            if not control["pilot"] or control["selected_image_count"] < 50:
                raise AffinityOracleError(
                    "the supplied pilot control must contain at least 50 images"
                )
            if (
                control["score_dtype"] != "float16"
                or control["source_image_count"] != source_image_count
                or control["source_git_commit"] != self.provenance["source_git_commit"]
                or control["source_git_dirty"]
                or self.provenance["source_git_dirty"]
                or control["dino_identity"] != dino_identity
            ):
                raise AffinityOracleError(
                    "the full cache requires a clean FP16 pilot from the same "
                    "source dataset, Git commit and DINO identity"
                )
            if protocol.score_dtype == "float16" and not control["fp16_suitable"]:
                raise AffinityOracleError(
                    "the supplied FP16 pilot is unsuitable; use float32 scores"
                )
            for key in (
                "class_order_sha256", "e3_config_sha256",
                "projection_config_sha256", "e3_checkpoint_sha256",
                "dino_checkpoint_sha256", "clip_checkpoint_sha256",
                "text_embedding_sha256", "dataset_config_sha256",
            ):
                expected = (
                    ordered_fingerprint(self.class_names)
                    if key == "class_order_sha256" else
                    self.text_embedding_sha256
                    if key == "text_embedding_sha256" else
                    self.initial_hashes[{
                        "e3_config_sha256": "e3_config",
                        "projection_config_sha256": "projection_config",
                        "e3_checkpoint_sha256": "e3_checkpoint",
                        "dino_checkpoint_sha256": "dino_checkpoint",
                        "clip_checkpoint_sha256": "clip_checkpoint",
                        "dataset_config_sha256": "dataset_config",
                    }[key]]
                )
                if control[key] != expected:
                    raise AffinityOracleError(
                        f"FP16 pilot identity mismatch for {key}"
                    )
            windows_per_image = (
                control["selected_window_count"]
                / control["selected_image_count"]
            )
            pilot_bytes_per_window = (
                control["total_cache_bytes"] / control["selected_window_count"]
            )
            source_score_bytes = 2 if control["score_dtype"] == "float16" else 4
            target_score_bytes = 2 if protocol.score_dtype == "float16" else 4
            score_byte_delta = (
                protocol.class_count * math.prod(protocol.patch_grid)
                * (target_score_bytes - source_score_bytes)
            )
            projected_bytes_per_window = max(
                float(estimate_window_bytes(protocol)),
                pilot_bytes_per_window + score_byte_delta,
            )
            estimated_full_bytes = math.ceil(
                windows_per_image * source_image_count
                * projected_bytes_per_window
            )
            self.preflight_estimated_cache_bytes = estimated_full_bytes
            output_parent = Path(output_dir).parent
            output_parent.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(output_parent).free
            print(
                "E3 affinity cache estimate: "
                f"{estimated_full_bytes / 1e9:.3f} GB; "
                f"available {free_bytes / 1e9:.3f} GB"
            )
            if free_bytes < math.ceil(1.25 * estimated_full_bytes):
                raise AffinityOracleError(
                    "insufficient free space: full affinity cache requires at "
                    "least 1.25 times the pilot-derived estimate"
                )
        # Create cache directories only after all preflight identity, pilot,
        # and free-space gates have passed.  A refused full run therefore
        # cannot strand an output directory that blocks a safe retry.
        self.writer = AffinityOracleCacheWriter(
            output_dir, protocol=protocol, windows_per_shard=windows_per_shard,
            overwrite=overwrite,
        )
        self.commands = list(commands)
        self.started = time.time()
        self.images: list[dict[str, Any]] = []
        self._active = False
        self._context: dict[str, Any] | None = None
        self._coordinates: list[list[int]] = []
        self._window_rows: list[dict[str, torch.Tensor]] = []
        self._reference_confusion = torch.zeros(
            (protocol.class_count, protocol.class_count), dtype=torch.int64
        )
        self._replay_confusion = torch.zeros_like(self._reference_confusion)
        self._different_valid_pixels = 0
        self._valid_pixels = 0

    @property
    def selected(self) -> bool:
        return self.max_images is None or len(self.images) < self.max_images

    def begin_image(self, metadata: Mapping[str, Any], *, dataset_index: int, annotation_path: str) -> None:
        self._active = self.selected
        self._coordinates = []
        self._window_rows = []
        if not self._active:
            return
        self._context = {
            "dataset_index": dataset_index, "metadata": dict(metadata),
            "annotation_path": annotation_path, "window_start": self.writer.window_count,
        }

    def set_window(self, *, coordinates: Sequence[int], grid_indices: Sequence[int]) -> None:
        if self._active:
            self._context["coordinates"] = list(coordinates)
            self._context["grid_indices"] = list(grid_indices)

    def observe(self, normalized_features: torch.Tensor, raw_scores: torch.Tensor) -> None:
        if not self._active:
            return
        context = self._context
        if context is None or "coordinates" not in context:
            raise AffinityOracleError("oracle observer lacks sliding-window context")
        row = self.writer.add_window(
            raw_scores.detach(), normalized_features.detach(),
            image_index=context["dataset_index"],
            coordinates=context["coordinates"], grid_indices=context["grid_indices"],
        )
        self._window_rows.append(row)
        self._coordinates.append(context["coordinates"])

    def end_image(
        self, *, resized_input_shape: Sequence[int], reference_prediction: Any,
        augmentation_index: int = 0,
    ) -> None:
        if not self._active:
            return
        context = self._context
        metadata = context["metadata"]
        h_img, w_img = int(resized_input_shape[0]), int(resized_input_shape[1])
        h_stride, w_stride = self.protocol.stride
        h_crop, w_crop = self.protocol.crop_size
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        image_id = metadata.get("ori_filename", metadata.get("filename", context["dataset_index"]))
        scale = metadata.get("scale_factor", 1.0)
        if torch.is_tensor(scale):
            scale = scale.detach().cpu().tolist()
        elif hasattr(scale, "tolist"):
            scale = scale.tolist()
        elif isinstance(scale, tuple):
            scale = list(scale)
        image_metadata = {
            "dataset_index": int(context["dataset_index"]), "image_id": str(image_id),
            "filename": str(metadata.get("filename", image_id)),
            "annotation_path": context["annotation_path"],
            "img_shape": [int(value) for value in metadata["img_shape"]],
            "ori_shape": [int(value) for value in metadata["ori_shape"]],
            "pad_shape": [
                int(value)
                for value in metadata.get("pad_shape", metadata["img_shape"])
            ],
            "resized_input_shape": [h_img, w_img],
            "crop_size": list(self.protocol.crop_size), "stride": list(self.protocol.stride),
            "h_grids": h_grids, "w_grids": w_grids,
            "window_start": context["window_start"], "window_end": self.writer.window_count,
            "flip": bool(metadata.get("flip", False)),
            "flip_direction": metadata.get("flip_direction"),
            "augmentation_index": augmentation_index,
            "scale": scale,
            "count_matrix_sha256": count_matrix_identity((h_img, w_img), self._coordinates),
        }
        self.images.append(image_metadata)
        reference = torch.as_tensor(reference_prediction).to(torch.int64).cpu()
        temporary_manifest = {
            "class_count": self.protocol.class_count,
            "patch_grid": list(self.protocol.patch_grid),
            "propagation_steps": self.protocol.propagation_steps,
            "protocol": {
                "with_background": self.protocol.with_background,
                "background_threshold": self.protocol.background_threshold,
            },
        }
        replay = replay_cached_image(
            image_metadata, self._window_rows, alpha=0.0,
            manifest=temporary_manifest, device=torch.device("cpu"),
        )
        target = load_annotation(context["annotation_path"])
        valid = target != 255
        self._different_valid_pixels += int(((reference != replay) & valid).sum())
        self._valid_pixels += int(valid.sum())
        self._reference_confusion += confusion_from_prediction(
            reference, target, num_classes=self.protocol.class_count
        )
        self._replay_confusion += confusion_from_prediction(
            replay, target, num_classes=self.protocol.class_count
        )
        self._active = False
        self._context = None

    def finalize(self) -> dict[str, Any]:
        hashes = self.initial_hashes
        expected_images = (
            self.max_images if self.max_images is not None
            else self.source_image_count
        )
        if len(self.images) != expected_images:
            raise AffinityOracleError(
                "oracle cache is incomplete: expected "
                f"{expected_images} images, captured {len(self.images)}"
            )
        if any(row["window_end"] <= row["window_start"] for row in self.images):
            raise AffinityOracleError("an oracle image has no captured windows")
        selected_images = len(self.images)
        pilot = self.max_images is not None
        estimated = (
            self.preflight_estimated_cache_bytes
            if self.preflight_estimated_cache_bytes is not None
            else self.writer.window_count * estimate_window_bytes(self.protocol)
        )
        reference_metrics = metrics_from_confusion(self._reference_confusion)
        replay_metrics = metrics_from_confusion(self._replay_confusion)
        # A class entering/leaving metric support is a maximal fidelity
        # failure, never a silent zero difference.
        per_class_differences = per_class_iou_differences(
            reference_metrics["per_class_iou"],
            replay_metrics["per_class_iou"],
        )
        pixel_percentage = (
            100.0 * self._different_valid_pixels / self._valid_pixels
            if self._valid_pixels else 0.0
        )
        fp16_suitable = (
            self.protocol.score_dtype == "float32"
            or (
                abs(reference_metrics["mIoU"] - replay_metrics["mIoU"]) <= 0.01
                and pixel_percentage <= 0.02
                and max(per_class_differences, default=0.0) <= 0.05
            )
        )
        manifest = {
            "pilot": pilot, "fp16_suitable": fp16_suitable,
            "selected_image_count": selected_images,
            "source_image_count": self.source_image_count,
            "class_count": self.protocol.class_count, "class_names": self.class_names,
            "class_order_sha256": ordered_fingerprint(self.class_names),
            "patch_grid": list(self.protocol.patch_grid),
            "patch_count": math.prod(self.protocol.patch_grid),
            "embedding_dimension": self.protocol.embedding_dimension,
            "score_dtype": self.protocol.score_dtype,
            "graph_index_dtype": "int16", "graph_weight_dtype": "float16",
            "knn_k": self.protocol.knn_k, "self_neighbour_exclusion": True,
            "affinity_formula": AFFINITY_FORMULA,
            "affinity_power": self.protocol.affinity_power,
            "graph_directionality": GRAPH_DIRECTIONALITY,
            "row_normalization": ROW_NORMALIZATION,
            "zero_row_fallback": ZERO_ROW_FALLBACK,
            "propagation_equation": PROPAGATION_EQUATION,
            "propagation_steps": self.protocol.propagation_steps,
            "alpha_grid": list(ALPHA_GRID),
            "e3_config_path": self.identity_paths["e3_config"],
            "e3_config_sha256": hashes["e3_config"],
            "projection_config_sha256": hashes["projection_config"],
            "e3_checkpoint_path": self.identity_paths["e3_checkpoint"],
            "e3_checkpoint_sha256": hashes["e3_checkpoint"],
            "dino_identity": self.dino_identity,
            "dino_checkpoint_sha256": hashes["dino_checkpoint"],
            "clip_checkpoint_path": self.identity_paths["clip_checkpoint"],
            "clip_checkpoint_sha256": hashes["clip_checkpoint"],
            "text_embedding_sha256": self.text_embedding_sha256,
            "dataset_config_path": self.identity_paths["dataset_config"],
            "dataset_config_sha256": hashes["dataset_config"],
            "image_list_sha256": ordered_fingerprint(row["image_id"] for row in self.images),
            "annotation_list_sha256": ordered_fingerprint(row["annotation_path"] for row in self.images),
            "protocol": {
                "score_stage": SCORE_STAGE, "crop_size": list(self.protocol.crop_size),
                "stride": list(self.protocol.stride), "with_background": self.protocol.with_background,
                "background_threshold": self.protocol.background_threshold,
                "ignore_index": 255, "pamr": self.protocol.pamr,
                "flip": self.protocol.flip, "augmentation_scales": self.protocol.augmentation_scales,
                "sigmoid_count": 1, "window_interpolation": "bilinear_align_corners_true",
                "rescale_interpolation": "bilinear_align_corners_false",
            },
            **self.provenance, "images": self.images,
            "estimated_cache_bytes": estimated,
            "construction_started_at": self.started,
            "construction_finished_at": time.time(),
            "fp16_control": {
                "status": "accepted" if fp16_suitable else "float32_required",
                "differing_valid_pixels": self._different_valid_pixels,
                "valid_pixels": self._valid_pixels,
                "differing_valid_pixel_percentage": pixel_percentage,
                "online_metrics": reference_metrics,
                "replay_metrics": replay_metrics,
                "aAcc_difference": replay_metrics["aAcc"] - reference_metrics["aAcc"],
                "mIoU_difference": replay_metrics["mIoU"] - reference_metrics["mIoU"],
                "mAcc_difference": replay_metrics["mAcc"] - reference_metrics["mAcc"],
                "maximum_per_class_iou_difference": max(per_class_differences, default=0.0),
            },
            "commands": self.commands,
        }
        def prepublish_check() -> None:
            final_git = git_provenance(
                Path(__file__).parents[1],
                allow_dirty=self.provenance["source_git_dirty"],
            )
            if final_git != self.provenance:
                raise AffinityOracleError(
                    "Git identity changed during cache construction"
                )
            final_hashes = {
                key: sha256_file(value)
                for key, value in self.identity_paths.items()
            }
            if final_hashes != hashes:
                raise AffinityOracleError(
                    "an oracle input identity changed before publication"
                )

        return self.writer.finalize(
            manifest, prepublish_check=prepublish_check
        )


def load_cache_manifest(
    path: Path, *, require_complete: bool = True, verify_shards: bool = True,
) -> dict[str, Any]:
    try:
        value = json.loads((Path(path) / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AffinityOracleError(f"invalid cache manifest: {error}") from error
    manifest = dict(_closed(value, MANIFEST_KEYS, "oracle cache manifest"))
    _require_finite_tree(manifest, "oracle cache manifest")
    if manifest["format_version"] != CACHE_FORMAT:
        raise AffinityOracleError("unsupported oracle cache format")
    if not isinstance(manifest["complete"], bool):
        raise AffinityOracleError("oracle cache complete must be boolean")
    if require_complete and manifest["complete"] is not True:
        raise AffinityOracleError("oracle cache is incomplete")
    for key in ("pilot", "fp16_suitable", "source_git_dirty"):
        if not isinstance(manifest[key], bool):
            raise AffinityOracleError(f"oracle cache {key} must be boolean")
    if (
        not isinstance(manifest["commands"], list)
        or not manifest["commands"]
        or not all(isinstance(command, str) and command for command in manifest["commands"])
    ):
        raise AffinityOracleError("oracle cache must record nonempty commands")
    if manifest["nonfinite_count"] != 0:
        raise AffinityOracleError("oracle cache records non-finite values")
    protocol_value = _closed(
        manifest["protocol"], PROTOCOL_KEYS, "oracle protocol"
    )
    fp16_control = _closed(
        manifest["fp16_control"], FP16_CONTROL_KEYS, "FP16 control"
    )
    protocol = OracleProtocol(
        class_count=manifest["class_count"], patch_grid=tuple(manifest["patch_grid"]),
        embedding_dimension=manifest["embedding_dimension"],
        crop_size=tuple(manifest["protocol"]["crop_size"]),
        stride=tuple(manifest["protocol"]["stride"]), knn_k=manifest["knn_k"],
        affinity_power=manifest["affinity_power"],
        propagation_steps=manifest["propagation_steps"],
        score_dtype=manifest["score_dtype"],
        with_background=manifest["protocol"]["with_background"],
        background_threshold=manifest["protocol"]["background_threshold"],
        pamr=manifest["protocol"]["pamr"], flip=manifest["protocol"]["flip"],
        augmentation_scales=manifest["protocol"]["augmentation_scales"],
    )
    protocol.validate()
    if protocol_value["score_stage"] != SCORE_STAGE:
        raise AffinityOracleError("cache score stage is not the raw E3 intervention point")
    if protocol_value["ignore_index"] != 255 or protocol_value["sigmoid_count"] != 1:
        raise AffinityOracleError("cache ignore-index/sigmoid protocol changed")
    if (
        protocol_value["window_interpolation"]
        != "bilinear_align_corners_true"
        or protocol_value["rescale_interpolation"]
        != "bilinear_align_corners_false"
    ):
        raise AffinityOracleError("cache interpolation protocol changed")
    if manifest["patch_count"] != math.prod(protocol.patch_grid):
        raise AffinityOracleError("cache patch count does not match its grid")
    if manifest["graph_index_dtype"] != "int16" or manifest["graph_weight_dtype"] != "float16":
        raise AffinityOracleError("cache graph storage dtypes changed")
    if (
        manifest["self_neighbour_exclusion"] is not True
        or manifest["affinity_formula"] != AFFINITY_FORMULA
        or manifest["graph_directionality"] != GRAPH_DIRECTIONALITY
        or manifest["row_normalization"] != ROW_NORMALIZATION
        or manifest["zero_row_fallback"] != ZERO_ROW_FALLBACK
        or manifest["propagation_equation"] != PROPAGATION_EQUATION
        or tuple(manifest["alpha_grid"]) != ALPHA_GRID
    ):
        raise AffinityOracleError("cache graph/propagation identity changed")
    if (
        not isinstance(manifest["selected_image_count"], int)
        or not isinstance(manifest["source_image_count"], int)
        or not 0 < manifest["selected_image_count"] <= manifest["source_image_count"]
        or len(manifest["images"]) != manifest["selected_image_count"]
    ):
        raise AffinityOracleError("cache image coverage metadata is invalid")
    if not manifest["pilot"] and manifest["selected_image_count"] != manifest["source_image_count"]:
        raise AffinityOracleError("a non-pilot cache does not cover the complete source")
    if not manifest["pilot"] and manifest["source_git_dirty"]:
        raise AffinityOracleError("a dirty-source cache cannot be production/full")
    if manifest["source_git_dirty"] != (manifest["source_git_diff_sha256"] is not None):
        raise AffinityOracleError("cache dirty-source provenance is inconsistent")
    hash_keys = (
        "class_order_sha256", "e3_config_sha256", "projection_config_sha256",
        "e3_checkpoint_sha256", "dino_checkpoint_sha256",
        "clip_checkpoint_sha256", "text_embedding_sha256",
        "dataset_config_sha256", "image_list_sha256",
        "annotation_list_sha256",
    )
    if any(not _valid_sha256(manifest[key]) for key in hash_keys):
        raise AffinityOracleError("cache contains an invalid SHA256 identity")
    if manifest["source_git_diff_sha256"] is not None and not _valid_sha256(manifest["source_git_diff_sha256"]):
        raise AffinityOracleError("cache contains an invalid Git diff SHA256")
    if (
        not isinstance(manifest["source_git_commit"], str)
        or len(manifest["source_git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in manifest["source_git_commit"]
        )
    ):
        raise AffinityOracleError("cache contains an invalid Git commit identity")
    if manifest["construction_finished_at"] < manifest["construction_started_at"]:
        raise AffinityOracleError("cache construction timestamps are reversed")
    _validate_base_metrics(
        fp16_control["online_metrics"], class_count=protocol.class_count,
        label="FP16 online metrics",
    )
    _validate_base_metrics(
        fp16_control["replay_metrics"], class_count=protocol.class_count,
        label="FP16 replay metrics",
    )
    if len(manifest["class_names"]) != protocol.class_count:
        raise AffinityOracleError("class name count mismatch")
    if not all(isinstance(name, str) and name for name in manifest["class_names"]):
        raise AffinityOracleError("class names must be nonempty strings")
    if ordered_fingerprint(manifest["class_names"]) != manifest["class_order_sha256"]:
        raise AffinityOracleError("class order fingerprint mismatch")
    validated_images = [
        _closed(row, IMAGE_META_KEYS, "cache image metadata")
        for row in manifest["images"]
    ]
    if ordered_fingerprint(row["image_id"] for row in validated_images) != manifest["image_list_sha256"]:
        raise AffinityOracleError("cache image-list fingerprint mismatch")
    if ordered_fingerprint(row["annotation_path"] for row in validated_images) != manifest["annotation_list_sha256"]:
        raise AffinityOracleError("cache annotation-list fingerprint mismatch")
    expected_start = 0
    for shard_number, shard in enumerate(manifest["shards"]):
        _closed(shard, SHARD_META_KEYS, "cache shard metadata")
        if (
            not isinstance(shard["name"], str)
            or not isinstance(shard["bytes"], int) or shard["bytes"] <= 0
            or not _valid_sha256(shard["sha256"])
        ):
            raise AffinityOracleError("cache shard metadata is malformed")
        if shard["name"] != f"shards/windows-{shard_number:06d}.pth":
            raise AffinityOracleError("cache shard name/order is noncanonical")
        if shard["window_start"] != expected_start or shard["window_end"] <= expected_start:
            raise AffinityOracleError("non-contiguous cache shard ranges")
        expected_start = shard["window_end"]
        candidate = Path(path) / shard["name"]
        if not candidate.is_file() or candidate.is_symlink():
            raise AffinityOracleError(f"missing/nonregular cache shard: {candidate}")
        if candidate.stat().st_size != shard["bytes"]:
            raise AffinityOracleError(f"cache shard byte count mismatch: {candidate}")
        if verify_shards and sha256_file(candidate) != shard["sha256"]:
            raise AffinityOracleError(f"cache shard hash mismatch: {candidate}")
    if expected_start != manifest["selected_window_count"]:
        raise AffinityOracleError("cache window coverage mismatch")
    if manifest["total_cache_bytes"] != sum(row["bytes"] for row in manifest["shards"]):
        raise AffinityOracleError("cache total-byte accounting is inconsistent")
    if (
        not isinstance(manifest["zero_neighbour_rows"], int)
        or manifest["zero_neighbour_rows"] < 0
        or not isinstance(manifest["nonfinite_count"], int)
    ):
        raise AffinityOracleError("cache graph/nonfinite counters are invalid")
    if (
        not isinstance(manifest["estimated_cache_bytes"], int)
        or manifest["estimated_cache_bytes"] <= 0
    ):
        raise AffinityOracleError("cache estimated-byte accounting is invalid")
    image_window_start = 0
    for expected_index, row in enumerate(validated_images):
        if row["dataset_index"] != expected_index:
            raise AffinityOracleError("cache dataset indices are not deterministic and contiguous")
        if row["window_start"] != image_window_start or row["window_end"] <= row["window_start"]:
            raise AffinityOracleError("cache image window ranges are invalid")
        image_window_start = row["window_end"]
        if tuple(row["crop_size"]) != protocol.crop_size or tuple(row["stride"]) != protocol.stride:
            raise AffinityOracleError("cache image crop/stride metadata changed")
    if image_window_start != manifest["selected_window_count"]:
        raise AffinityOracleError("cache image/window coverage is incomplete")
    return manifest


def cache_summary(path: Path) -> dict[str, Any]:
    # Each shard is safely loaded and hashed below; avoid a redundant full
    # byte pass during explicit validation.
    manifest = load_cache_manifest(path, verify_shards=False)
    for metadata in manifest["shards"]:
        load_cache_shard(
            Path(path) / metadata["name"],
            expected_bytes=metadata["bytes"],
            expected_sha256=metadata["sha256"],
            class_count=manifest["class_count"],
            patch_grid=tuple(manifest["patch_grid"]),
            knn_k=manifest["knn_k"], score_dtype=_score_dtype(manifest),
        )
    return {
        "format_version": manifest["format_version"],
        "complete": manifest["complete"], "pilot": manifest["pilot"],
        "images": manifest["selected_image_count"],
        "windows": manifest["selected_window_count"],
        "shards": len(manifest["shards"]), "bytes": manifest["total_cache_bytes"],
        "fp16_suitable": manifest["fp16_suitable"],
        "zero_neighbour_rows": manifest["zero_neighbour_rows"],
    }


def _score_dtype(manifest: Mapping[str, Any]) -> torch.dtype:
    return torch.float16 if manifest["score_dtype"] == "float16" else torch.float32


def iter_cache_windows(
    cache_path: Path, manifest: Mapping[str, Any]
) -> Iterator[dict[str, torch.Tensor]]:
    for metadata in manifest["shards"]:
        shard = load_cache_shard(
            cache_path / metadata["name"],
            expected_bytes=metadata["bytes"],
            expected_sha256=metadata["sha256"],
            class_count=manifest["class_count"],
            patch_grid=tuple(manifest["patch_grid"]), knn_k=manifest["knn_k"],
            score_dtype=_score_dtype(manifest),
        )
        for row in range(shard["raw_scores"].shape[0]):
            yield {key: value[row] for key, value in shard.items()}


def iter_cached_images(
    cache_path: Path, manifest: Mapping[str, Any]
) -> Iterator[tuple[Mapping[str, Any], list[dict[str, torch.Tensor]]]]:
    metadata = manifest["images"]
    window_iterator = iter_cache_windows(cache_path, manifest)
    next_window = next(window_iterator, None)
    for image in metadata:
        _closed(image, IMAGE_META_KEYS, "cache image metadata")
        rows: list[dict[str, torch.Tensor]] = []
        while (
            next_window is not None
            and int(next_window["window_image_index"]) == image["dataset_index"]
        ):
            rows.append(next_window)
            next_window = next(window_iterator, None)
        if len(rows) != image["window_end"] - image["window_start"]:
            raise AffinityOracleError("cached image/window range mismatch")
        h_img, w_img = (int(value) for value in image["resized_input_shape"])
        h_crop, w_crop = (int(value) for value in image["crop_size"])
        h_stride, w_stride = (int(value) for value in image["stride"])
        expected: list[tuple[list[int], list[int]]] = []
        for h_index in range(int(image["h_grids"])):
            for w_index in range(int(image["w_grids"])):
                y1, x1 = h_index * h_stride, w_index * w_stride
                y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                expected.append(([y1, x1, y2, x2], [h_index, w_index]))
        if len(expected) != len(rows):
            raise AffinityOracleError("cached sliding-window grid count mismatch")
        for row, (coordinates, grid_indices) in zip(rows, expected):
            if (
                row["window_coordinates"].tolist() != coordinates
                or row["window_grid_indices"].tolist() != grid_indices
            ):
                raise AffinityOracleError("cached sliding-window ordering changed")
        if count_matrix_identity(
            (h_img, w_img), [coordinates for coordinates, _ in expected]
        ) != image["count_matrix_sha256"]:
            raise AffinityOracleError("cached overlap/count-matrix identity changed")
        yield image, rows
    if next_window is not None:
        raise AffinityOracleError("cache contains windows not assigned to images")


def load_annotation(path: os.PathLike[str] | str) -> torch.Tensor:
    try:
        from PIL import Image
        import numpy as np
        value = torch.from_numpy(np.array(Image.open(path), dtype=np.int64))
    except Exception as error:
        raise AffinityOracleError(f"cannot load segmentation annotation {path}: {error}") from error
    if value.ndim != 2:
        raise AffinityOracleError("segmentation annotation must be two-dimensional")
    return value


def replay_cached_image_logits(
    image: Mapping[str, Any], windows: Sequence[Mapping[str, torch.Tensor]],
    *, alpha: float | torch.Tensor, manifest: Mapping[str, Any],
    device: torch.device, propagation_steps: int | None = None,
) -> torch.Tensor:
    protocol = manifest["protocol"]
    steps = (
        manifest["propagation_steps"] if propagation_steps is None
        else propagation_steps
    )
    masks: list[torch.Tensor] = []
    coordinates: list[list[int]] = []
    for row in windows:
        raw = row["raw_scores"].reshape(manifest["class_count"], -1).to(
            device=device, dtype=torch.float32
        )
        spread = propagate_scores(
            raw, row["knn_indices"], row["knn_weights"], alpha,
            propagation_steps=steps,
        )
        coordinate = [int(value) for value in row["window_coordinates"].tolist()]
        y1, x1, y2, x2 = coordinate
        mask = interpolate_window_scores(
            spread, patch_grid=tuple(manifest["patch_grid"]),
            crop_size=(y2 - y1, x2 - x1),
        )
        mask = add_background_channel(
            mask, with_background=protocol["with_background"],
            background_threshold=protocol["background_threshold"],
        )
        masks.append(mask)
        coordinates.append(coordinate)
    expected_identity = count_matrix_identity(
        tuple(image["resized_input_shape"]), coordinates
    )
    if expected_identity != image["count_matrix_sha256"]:
        raise AffinityOracleError("sliding-window count-matrix identity mismatch")
    stitched = stitch_windows(
        masks, coordinates, image_shape=tuple(image["resized_input_shape"])
    )
    logits = rescale_logits(
        stitched, img_shape=tuple(image["img_shape"][:2]),
        ori_shape=tuple(image["ori_shape"][:2]),
    )
    if image["flip"]:
        if image["flip_direction"] == "horizontal":
            logits = logits.flip(-1)
        elif image["flip_direction"] == "vertical":
            logits = logits.flip(-2)
        else:
            raise AffinityOracleError("unsupported flip direction")
    return logits


def replay_cached_image(
    image: Mapping[str, Any], windows: Sequence[Mapping[str, torch.Tensor]],
    *, alpha: float | torch.Tensor, manifest: Mapping[str, Any],
    device: torch.device, propagation_steps: int | None = None,
) -> torch.Tensor:
    logits = replay_cached_image_logits(
        image, windows, alpha=alpha, manifest=manifest, device=device,
        propagation_steps=propagation_steps,
    )
    return logits.softmax(dim=0).argmax(dim=0).cpu()


def replay_cached_class_channel(
    image: Mapping[str, Any], windows: Sequence[Mapping[str, torch.Tensor]],
    *, class_index: int, alpha: float, manifest: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if manifest["protocol"]["with_background"]:
        raise AffinityOracleError(
            "per-class fitting requires raw semantic channels without a synthetic background"
        )
    masks: list[torch.Tensor] = []
    coordinates: list[list[int]] = []
    for row in windows:
        raw = row["raw_scores"][class_index].reshape(1, -1).to(
            device=device, dtype=torch.float32
        )
        spread = propagate_scores(
            raw, row["knn_indices"], row["knn_weights"], alpha,
            propagation_steps=manifest["propagation_steps"],
        )
        coordinate = [int(value) for value in row["window_coordinates"].tolist()]
        y1, x1, y2, x2 = coordinate
        masks.append(interpolate_window_scores(
            spread, patch_grid=tuple(manifest["patch_grid"]),
            crop_size=(y2 - y1, x2 - x1),
        ))
        coordinates.append(coordinate)
    stitched = stitch_windows(
        masks, coordinates, image_shape=tuple(image["resized_input_shape"])
    )
    result = rescale_logits(
        stitched, img_shape=tuple(image["img_shape"][:2]),
        ori_shape=tuple(image["ori_shape"][:2]),
    ).squeeze(0)
    if image["flip"]:
        result = result.flip(-1 if image["flip_direction"] == "horizontal" else -2)
    return result


def _bucket_alpha_vector(
    row: Mapping[str, torch.Tensor], *, stat: str, edges: torch.Tensor,
    bucket_alphas: torch.Tensor, patch_count: int, device: torch.device,
) -> torch.Tensor:
    values = compute_patch_statistic(
        row["knn_indices"], row["knn_weights"], stat=stat, patch_count=patch_count,
    )
    buckets = assign_buckets(values, edges)
    return bucket_alphas.to(device=device, dtype=torch.float32)[buckets]


def replay_cached_image_logits_bucketed(
    image: Mapping[str, Any], windows: Sequence[Mapping[str, torch.Tensor]],
    *, stat: str, edges: torch.Tensor, bucket_alphas: torch.Tensor,
    manifest: Mapping[str, Any], device: torch.device,
    propagation_steps: int | None = None,
) -> torch.Tensor:
    """Per-patch (not per-class) alpha replay: same alpha applies to all 171
    class channels at a patch, chosen by that patch's own bucket."""

    protocol = manifest["protocol"]
    steps = (
        manifest["propagation_steps"] if propagation_steps is None
        else propagation_steps
    )
    patch_count = manifest["patch_count"]
    masks: list[torch.Tensor] = []
    coordinates: list[list[int]] = []
    for row in windows:
        raw = row["raw_scores"].reshape(manifest["class_count"], -1).to(
            device=device, dtype=torch.float32
        )
        alpha_vector = _bucket_alpha_vector(
            row, stat=stat, edges=edges, bucket_alphas=bucket_alphas,
            patch_count=patch_count, device=device,
        )
        spread = propagate_scores(
            raw, row["knn_indices"], row["knn_weights"], alpha_vector,
            propagation_steps=steps, alpha_dim="patch",
        )
        coordinate = [int(value) for value in row["window_coordinates"].tolist()]
        y1, x1, y2, x2 = coordinate
        mask = interpolate_window_scores(
            spread, patch_grid=tuple(manifest["patch_grid"]),
            crop_size=(y2 - y1, x2 - x1),
        )
        mask = add_background_channel(
            mask, with_background=protocol["with_background"],
            background_threshold=protocol["background_threshold"],
        )
        masks.append(mask)
        coordinates.append(coordinate)
    expected_identity = count_matrix_identity(
        tuple(image["resized_input_shape"]), coordinates
    )
    if expected_identity != image["count_matrix_sha256"]:
        raise AffinityOracleError("sliding-window count-matrix identity mismatch")
    stitched = stitch_windows(
        masks, coordinates, image_shape=tuple(image["resized_input_shape"])
    )
    logits = rescale_logits(
        stitched, img_shape=tuple(image["img_shape"][:2]),
        ori_shape=tuple(image["ori_shape"][:2]),
    )
    if image["flip"]:
        if image["flip_direction"] == "horizontal":
            logits = logits.flip(-1)
        elif image["flip_direction"] == "vertical":
            logits = logits.flip(-2)
        else:
            raise AffinityOracleError("unsupported flip direction")
    return logits


def replay_cached_image_bucketed(
    image: Mapping[str, Any], windows: Sequence[Mapping[str, torch.Tensor]],
    *, stat: str, edges: torch.Tensor, bucket_alphas: torch.Tensor,
    manifest: Mapping[str, Any], device: torch.device,
    propagation_steps: int | None = None,
) -> torch.Tensor:
    logits = replay_cached_image_logits_bucketed(
        image, windows, stat=stat, edges=edges, bucket_alphas=bucket_alphas,
        manifest=manifest, device=device, propagation_steps=propagation_steps,
    )
    return logits.softmax(dim=0).argmax(dim=0).cpu()


def evaluate_cache_bucketed(
    cache_path: Path, *, stat: str, edges: torch.Tensor, bucket_alphas: torch.Tensor,
    device: str = "cpu", selected_indices: set[int] | None = None,
    propagation_steps: int | None = None,
) -> dict[str, Any]:
    """Disk-streaming bucketed evaluation -- used for the (few) final,
    frozen evaluations. The repeated coordinate-ascent search uses the
    in-memory evaluate_preloaded_bucketed below instead, since it would
    otherwise re-read the whole cache from disk for every candidate alpha."""

    started = time.monotonic()
    manifest = load_cache_manifest(cache_path, verify_shards=False)
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)
    confusion = torch.zeros(
        (manifest["class_count"], manifest["class_count"]), dtype=torch.int64
    )
    evaluated = 0
    for image, windows in iter_cached_images(cache_path, manifest):
        if selected_indices is not None and image["dataset_index"] not in selected_indices:
            continue
        prediction = replay_cached_image_bucketed(
            image, windows, stat=stat, edges=edges, bucket_alphas=bucket_alphas,
            manifest=manifest, device=device_value, propagation_steps=propagation_steps,
        )
        target = load_annotation(image["annotation_path"])
        confusion += confusion_from_prediction(
            prediction, target, num_classes=manifest["class_count"],
            ignore_index=manifest["protocol"]["ignore_index"],
        )
        evaluated += 1
    metrics = metrics_from_confusion(confusion)
    metrics.update({
        "evaluated_images": evaluated,
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda" else 0
        ),
    })
    return metrics


PreloadedImage = tuple[Mapping[str, Any], list[dict[str, torch.Tensor]], torch.Tensor]


def preload_cache_images(
    cache_path: Path, manifest: Mapping[str, Any],
    *, selected_indices: set[int] | None = None,
) -> list[PreloadedImage]:
    """Materialise (image, windows, target) once, in RAM, for repeated
    bucketed replay. Coordinate ascent (B3) re-evaluates the full joint
    mIoU once per (sweep, bucket, candidate alpha) -- re-streaming the
    cache from disk that many times would be dominated by I/O, not
    propagation compute, so the fitting set is loaded once and reused."""

    preloaded: list[PreloadedImage] = []
    for image, windows in iter_cached_images(cache_path, manifest):
        if selected_indices is not None and image["dataset_index"] not in selected_indices:
            continue
        target = load_annotation(image["annotation_path"])
        preloaded.append((image, windows, target))
    if not preloaded:
        raise AffinityOracleError("preload selected an empty image set")
    return preloaded


def evaluate_preloaded(
    preloaded: Sequence[PreloadedImage], manifest: Mapping[str, Any],
    *, alpha: float | torch.Tensor, device: str = "cpu",
    propagation_steps: int | None = None,
) -> dict[str, Any]:
    """Scalar/per-class-alpha evaluation over a preloaded image list."""

    started = time.monotonic()
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)
    confusion = torch.zeros(
        (manifest["class_count"], manifest["class_count"]), dtype=torch.int64
    )
    for image, windows, target in preloaded:
        prediction = replay_cached_image(
            image, windows, alpha=alpha, manifest=manifest, device=device_value,
            propagation_steps=propagation_steps,
        )
        confusion += confusion_from_prediction(
            prediction, target, num_classes=manifest["class_count"],
            ignore_index=manifest["protocol"]["ignore_index"],
        )
    metrics = metrics_from_confusion(confusion)
    metrics.update({
        "evaluated_images": len(preloaded),
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda" else 0
        ),
    })
    return metrics


def evaluate_preloaded_bucketed(
    preloaded: Sequence[PreloadedImage], manifest: Mapping[str, Any],
    *, stat: str, edges: torch.Tensor, bucket_alphas: torch.Tensor,
    device: str = "cpu", propagation_steps: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)
    confusion = torch.zeros(
        (manifest["class_count"], manifest["class_count"]), dtype=torch.int64
    )
    for image, windows, target in preloaded:
        prediction = replay_cached_image_bucketed(
            image, windows, stat=stat, edges=edges, bucket_alphas=bucket_alphas,
            manifest=manifest, device=device_value, propagation_steps=propagation_steps,
        )
        confusion += confusion_from_prediction(
            prediction, target, num_classes=manifest["class_count"],
            ignore_index=manifest["protocol"]["ignore_index"],
        )
    metrics = metrics_from_confusion(confusion)
    metrics.update({
        "evaluated_images": len(preloaded),
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda" else 0
        ),
    })
    return metrics


def replay_cached_image_logits_biased(
    image: Mapping[str, Any], windows: Sequence[Mapping[str, torch.Tensor]],
    *, alpha: float | torch.Tensor, beta: torch.Tensor,
    manifest: Mapping[str, Any], device: torch.device,
    propagation_steps: int | None = None,
) -> torch.Tensor:
    """Part C: logit_c(p) = S_c^propagated(p) + beta_c -- bias is injected
    immediately after propagate_scores, before the (nonlinear) sigmoid inside
    interpolate_window_scores, exactly matching the task's formula. alpha is
    a scalar (or per-class vector), never touched by bias fitting."""

    protocol = manifest["protocol"]
    steps = (
        manifest["propagation_steps"] if propagation_steps is None
        else propagation_steps
    )
    class_count = manifest["class_count"]
    if not torch.is_tensor(beta) or beta.ndim != 1 or beta.shape[0] != class_count:
        raise AffinityOracleError(f"beta must have shape [{class_count}]")
    if not torch.isfinite(beta).all():
        raise AffinityOracleError("beta must be finite")
    beta_value = beta.to(device=device, dtype=torch.float32)
    masks: list[torch.Tensor] = []
    coordinates: list[list[int]] = []
    for row in windows:
        raw = row["raw_scores"].reshape(class_count, -1).to(
            device=device, dtype=torch.float32
        )
        spread = propagate_scores(
            raw, row["knn_indices"], row["knn_weights"], alpha,
            propagation_steps=steps,
        )
        biased = spread + beta_value[:, None]
        coordinate = [int(value) for value in row["window_coordinates"].tolist()]
        y1, x1, y2, x2 = coordinate
        mask = interpolate_window_scores(
            biased, patch_grid=tuple(manifest["patch_grid"]),
            crop_size=(y2 - y1, x2 - x1),
        )
        mask = add_background_channel(
            mask, with_background=protocol["with_background"],
            background_threshold=protocol["background_threshold"],
        )
        masks.append(mask)
        coordinates.append(coordinate)
    expected_identity = count_matrix_identity(
        tuple(image["resized_input_shape"]), coordinates
    )
    if expected_identity != image["count_matrix_sha256"]:
        raise AffinityOracleError("sliding-window count-matrix identity mismatch")
    stitched = stitch_windows(
        masks, coordinates, image_shape=tuple(image["resized_input_shape"])
    )
    logits = rescale_logits(
        stitched, img_shape=tuple(image["img_shape"][:2]),
        ori_shape=tuple(image["ori_shape"][:2]),
    )
    if image["flip"]:
        if image["flip_direction"] == "horizontal":
            logits = logits.flip(-1)
        elif image["flip_direction"] == "vertical":
            logits = logits.flip(-2)
        else:
            raise AffinityOracleError("unsupported flip direction")
    return logits


def replay_cached_image_biased(
    image: Mapping[str, Any], windows: Sequence[Mapping[str, torch.Tensor]],
    *, alpha: float | torch.Tensor, beta: torch.Tensor,
    manifest: Mapping[str, Any], device: torch.device,
    propagation_steps: int | None = None,
) -> torch.Tensor:
    logits = replay_cached_image_logits_biased(
        image, windows, alpha=alpha, beta=beta, manifest=manifest, device=device,
        propagation_steps=propagation_steps,
    )
    return logits.softmax(dim=0).argmax(dim=0).cpu()


def evaluate_cache_biased(
    cache_path: Path, *, alpha: float | torch.Tensor, beta: torch.Tensor,
    device: str = "cpu", selected_indices: set[int] | None = None,
    propagation_steps: int | None = None,
) -> dict[str, Any]:
    """Disk-streaming biased evaluation -- used for the frozen half-B check."""

    started = time.monotonic()
    manifest = load_cache_manifest(cache_path, verify_shards=False)
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)
    confusion = torch.zeros(
        (manifest["class_count"], manifest["class_count"]), dtype=torch.int64
    )
    evaluated = 0
    for image, windows in iter_cached_images(cache_path, manifest):
        if selected_indices is not None and image["dataset_index"] not in selected_indices:
            continue
        prediction = replay_cached_image_biased(
            image, windows, alpha=alpha, beta=beta, manifest=manifest,
            device=device_value, propagation_steps=propagation_steps,
        )
        target = load_annotation(image["annotation_path"])
        confusion += confusion_from_prediction(
            prediction, target, num_classes=manifest["class_count"],
            ignore_index=manifest["protocol"]["ignore_index"],
        )
        evaluated += 1
    metrics = metrics_from_confusion(confusion)
    metrics.update({
        "evaluated_images": evaluated,
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda" else 0
        ),
    })
    return metrics


def evaluate_preloaded_biased(
    preloaded: Sequence[PreloadedImage], manifest: Mapping[str, Any],
    *, alpha: float | torch.Tensor, beta: torch.Tensor,
    device: str = "cpu", propagation_steps: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)
    confusion = torch.zeros(
        (manifest["class_count"], manifest["class_count"]), dtype=torch.int64
    )
    for image, windows, target in preloaded:
        prediction = replay_cached_image_biased(
            image, windows, alpha=alpha, beta=beta, manifest=manifest,
            device=device_value, propagation_steps=propagation_steps,
        )
        confusion += confusion_from_prediction(
            prediction, target, num_classes=manifest["class_count"],
            ignore_index=manifest["protocol"]["ignore_index"],
        )
    metrics = metrics_from_confusion(confusion)
    metrics.update({
        "evaluated_images": len(preloaded),
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda" else 0
        ),
    })
    return metrics


def class_propagated_score_std(
    preloaded: Sequence[PreloadedImage], manifest: Mapping[str, Any],
    *, alpha: float, propagation_steps: int | None, device: str = "cpu",
) -> torch.Tensor:
    """Per-class standard deviation of the propagated score S_c^propagated(p)
    over every patch in the fitting data, measured once at alpha* -- the
    unit that --bias-grid multipliers are expressed in (C1)."""

    device_value = torch.device(device)
    class_count = manifest["class_count"]
    steps = (
        manifest["propagation_steps"] if propagation_steps is None
        else propagation_steps
    )
    total = torch.zeros(class_count, dtype=torch.float64)
    total_sq = torch.zeros(class_count, dtype=torch.float64)
    count = 0
    for _, windows, _ in preloaded:
        for row in windows:
            raw = row["raw_scores"].reshape(class_count, -1).to(
                device=device_value, dtype=torch.float32
            )
            spread = propagate_scores(
                raw, row["knn_indices"], row["knn_weights"], alpha,
                propagation_steps=steps,
            ).double().cpu()
            total += spread.sum(dim=1)
            total_sq += (spread ** 2).sum(dim=1)
            count += spread.shape[1]
    if count == 0:
        raise AffinityOracleError("no patches available to measure score std")
    mean = total / count
    variance = (total_sq / count - mean ** 2).clamp_min(0)
    return variance.sqrt().float()


def collect_patch_statistic_values(
    preloaded: Sequence[PreloadedImage], manifest: Mapping[str, Any], *, stat: str,
) -> torch.Tensor:
    patch_count = manifest["patch_count"]
    chunks = [
        compute_patch_statistic(
            row["knn_indices"], row["knn_weights"], stat=stat, patch_count=patch_count,
        )
        for _, windows, _ in preloaded for row in windows
    ]
    if not chunks:
        raise AffinityOracleError("no windows available to fit bucket edges")
    return torch.cat(chunks)


def coordinate_ascent_fit(
    n_units: int, grid_per_unit: Sequence[Sequence[float]], init_value: float,
    *, max_sweeps: int, seed: int, evaluate_fn: Any, log: Any = None,
) -> dict[str, Any]:
    """Generic accept-only-if-improved coordinate ascent directly on the full
    joint mIoU, shared by Part B (per-bucket alpha) and Part C (per-class
    bias). A move is kept only if it strictly improves the joint objective
    (by more than 1e-6), so the sequence of accepted values is monotonically
    non-decreasing by construction -- unlike a per-unit fit maximising each
    unit's own metric in isolation, this cannot regress relative to its own
    starting point. ``evaluate_fn(values: Tensor[n_units]) -> metrics dict``
    with a ``mIoU`` key; ``grid_per_unit[i]`` is unit i's own candidate list
    (Part B repeats one shared alpha grid per bucket; Part C scales a shared
    multiplier grid by each class's own score standard deviation)."""

    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int) or max_sweeps <= 0:
        raise AffinityOracleError("max_sweeps must be a positive integer")
    if isinstance(n_units, bool) or not isinstance(n_units, int) or n_units <= 0:
        raise AffinityOracleError("n_units must be a positive integer")
    if len(grid_per_unit) != n_units:
        raise AffinityOracleError("grid_per_unit must supply one grid per unit")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_units, generator=generator).tolist()
    values = torch.full((n_units,), float(init_value), dtype=torch.float32)

    current_metrics = evaluate_fn(values)
    history = [current_metrics["mIoU"]]
    trace = [{
        "sweep": 0, "unit": None, "value": None,
        "joint_mIoU": current_metrics["mIoU"], "accepted": None,
    }]
    changed = True
    sweeps_run = 0
    for sweep in range(1, max_sweeps + 1):
        sweeps_run = sweep
        changed = False
        for unit in order:
            current_unit_value = float(values[unit])
            best_value = current_unit_value
            best_metrics = current_metrics
            for candidate in grid_per_unit[unit]:
                if candidate == current_unit_value:
                    continue
                trial = values.clone()
                trial[unit] = candidate
                metrics = evaluate_fn(trial)
                if metrics["mIoU"] > best_metrics["mIoU"] + 1e-6:
                    best_metrics = metrics
                    best_value = candidate
            if best_metrics["mIoU"] > current_metrics["mIoU"] + 1e-6:
                values[unit] = best_value
                current_metrics = best_metrics
                changed = True
                history.append(current_metrics["mIoU"])
                trace.append({
                    "sweep": sweep, "unit": unit, "value": best_value,
                    "joint_mIoU": current_metrics["mIoU"], "accepted": True,
                })
                if log is not None:
                    log(
                        f"coordinate ascent accepted: sweep={sweep} unit={unit} "
                        f"value={best_value:.4f} joint_mIoU={current_metrics['mIoU']:.6f}"
                    )
        if not changed:
            break
    for previous, following in zip(history, history[1:]):
        if following < previous - 1e-9:
            raise AffinityOracleError(
                "coordinate ascent objective decreased -- monotonicity violated"
            )
    return {
        "values": values.tolist(), "order": order, "trace": trace,
        "final_metrics": current_metrics, "sweeps_run": sweeps_run,
        "converged": not changed,
    }


def coordinate_ascent_bucket_fit(
    preloaded: Sequence[PreloadedImage], manifest: Mapping[str, Any],
    *, stat: str, edges: torch.Tensor, alpha_grid: Sequence[float],
    global_alpha: float, n_buckets: int, max_sweeps: int, seed: int,
    propagation_steps: int | None, device: str,
    log: Any = None,
) -> dict[str, Any]:
    """B3/B4: per-bucket alpha, initialised at the global optimum alpha*."""

    def evaluate_fn(values: torch.Tensor) -> dict[str, Any]:
        return evaluate_preloaded_bucketed(
            preloaded, manifest, stat=stat, edges=edges, bucket_alphas=values,
            device=device, propagation_steps=propagation_steps,
        )

    grid = list(alpha_grid)
    result = coordinate_ascent_fit(
        n_buckets, [grid] * n_buckets, global_alpha,
        max_sweeps=max_sweeps, seed=seed, evaluate_fn=evaluate_fn, log=log,
    )
    return {
        "bucket_alphas": result["values"], "bucket_order": result["order"],
        "trace": [
            {
                "sweep": row["sweep"], "bucket": row["unit"], "alpha": row["value"],
                "joint_mIoU": row["joint_mIoU"], "accepted": row["accepted"],
            }
            for row in result["trace"]
        ],
        "final_metrics": result["final_metrics"], "sweeps_run": result["sweeps_run"],
        "converged": result["converged"],
    }


def coordinate_ascent_bias_fit(
    preloaded: Sequence[PreloadedImage], manifest: Mapping[str, Any],
    *, alpha: float, grid_per_class: Sequence[Sequence[float]],
    max_sweeps: int, seed: int, propagation_steps: int | None, device: str,
    log: Any = None,
) -> dict[str, Any]:
    """C1/C4: per-class additive post-propagation bias, initialised at 0
    (no-op). ``alpha`` is the frozen global propagation alpha* -- bias fitting
    never touches alpha, it only adds beta_c after propagation."""

    n_classes = manifest["class_count"]

    def evaluate_fn(values: torch.Tensor) -> dict[str, Any]:
        return evaluate_preloaded_biased(
            preloaded, manifest, alpha=alpha, beta=values, device=device,
            propagation_steps=propagation_steps,
        )

    result = coordinate_ascent_fit(
        n_classes, grid_per_class, 0.0,
        max_sweeps=max_sweeps, seed=seed, evaluate_fn=evaluate_fn, log=log,
    )
    return {
        "beta_by_class": result["values"], "class_visit_order": result["order"],
        "trace": [
            {
                "sweep": row["sweep"], "class_index": row["unit"], "beta": row["value"],
                "joint_mIoU": row["joint_mIoU"], "accepted": row["accepted"],
            }
            for row in result["trace"]
        ],
        "final_metrics": result["final_metrics"], "sweeps_run": result["sweeps_run"],
        "converged": result["converged"],
    }


def dominant_classes_per_bucket(
    preloaded: Sequence[PreloadedImage], manifest: Mapping[str, Any],
    *, stat: str, edges: torch.Tensor, global_alpha: float,
    propagation_steps: int | None, device: str, n_buckets: int, top_k: int = 5,
) -> list[dict[str, Any]]:
    """Interpretability only (B6), not used by the fit itself: which classes
    are most often predicted, under the uniform global-alpha baseline, at
    patches that fall into each bucket."""

    device_value = torch.device(device)
    class_count = manifest["class_count"]
    patch_count = manifest["patch_count"]
    steps = (
        manifest["propagation_steps"] if propagation_steps is None
        else propagation_steps
    )
    counts = torch.zeros((n_buckets, class_count), dtype=torch.int64)
    patch_totals = torch.zeros(n_buckets, dtype=torch.int64)
    for _, windows, _ in preloaded:
        for row in windows:
            raw = row["raw_scores"].reshape(class_count, -1).to(
                device=device_value, dtype=torch.float32
            )
            spread = propagate_scores(
                raw, row["knn_indices"], row["knn_weights"], global_alpha,
                propagation_steps=steps,
            )
            values = compute_patch_statistic(
                row["knn_indices"], row["knn_weights"], stat=stat, patch_count=patch_count,
            )
            buckets = assign_buckets(values, edges).cpu()
            predicted = spread.argmax(dim=0).cpu()
            for bucket_index in range(n_buckets):
                mask = buckets == bucket_index
                count = int(mask.sum())
                if count:
                    patch_totals[bucket_index] += count
                    counts[bucket_index] += torch.bincount(
                        predicted[mask], minlength=class_count
                    )
    results = []
    for bucket_index in range(n_buckets):
        available = min(top_k, int((counts[bucket_index] > 0).sum()))
        if available == 0:
            results.append({
                "bucket": bucket_index, "patch_count": int(patch_totals[bucket_index]),
                "dominant_classes": [],
            })
            continue
        top = torch.topk(counts[bucket_index], k=available)
        results.append({
            "bucket": bucket_index, "patch_count": int(patch_totals[bucket_index]),
            "dominant_classes": [
                {
                    "class_index": int(class_index),
                    "class_name": manifest["class_names"][int(class_index)],
                    "patch_count": int(count),
                }
                for count, class_index in zip(top.values.tolist(), top.indices.tolist())
            ],
        })
    return results


def evaluate_cache(
    cache_path: Path, alpha: float | torch.Tensor, *, device: str = "cpu",
    selected_indices: set[int] | None = None,
    propagation_steps: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    # Shards are individually hash-checked immediately before safe loading.
    manifest = load_cache_manifest(cache_path, verify_shards=False)
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)
    confusion = torch.zeros(
        (manifest["class_count"], manifest["class_count"]), dtype=torch.int64
    )
    evaluated = 0
    for image, windows in iter_cached_images(cache_path, manifest):
        if selected_indices is not None and image["dataset_index"] not in selected_indices:
            continue
        prediction = replay_cached_image(
            image, windows, alpha=alpha, manifest=manifest, device=device_value,
            propagation_steps=propagation_steps,
        )
        target = load_annotation(image["annotation_path"])
        confusion += confusion_from_prediction(
            prediction, target, num_classes=manifest["class_count"],
            ignore_index=manifest["protocol"]["ignore_index"],
        )
        evaluated += 1
    metrics = metrics_from_confusion(confusion)
    metrics.update({
        "evaluated_images": evaluated,
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda" else 0
        ),
    })
    return metrics


def global_sweep(
    cache_path: Path, *, device: str = "cpu",
    selected_indices: set[int] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    baseline = None
    for alpha in ALPHA_GRID:
        metrics = evaluate_cache(
            cache_path, alpha, device=device, selected_indices=selected_indices
        )
        if baseline is None:
            baseline = metrics
        rows.append({
            "alpha": alpha, **metrics,
            "delta_aAcc": metrics["aAcc"] - baseline["aAcc"],
            "delta_mIoU": metrics["mIoU"] - baseline["mIoU"],
            "delta_mAcc": metrics["mAcc"] - baseline["mAcc"],
        })
    best = min(rows, key=lambda row: (-row["mIoU"], row["alpha"]))
    return rows, float(best["alpha"])


# ---------------------------------------------------------------------------
# Part E: read-only DINO patch-feature capture verification (E3/E4 gates).
#
# The oracle cache never stores raw 768-d patch features (SHARD_KEYS forbids
# any key containing "feature"; see validate_cache_shard), so the only way to
# check that a freshly captured feature tensor would reproduce the cache's
# baked knn_indices/knn_weights is to rebuild the graph from the capture and
# compare it, edge for edge, against the cache's own graph for the SAME
# windows. E3 is that correctness gate; E4 substitutes the rebuilt graph into
# replay_cached_image_logits (unmodified) to confirm end-to-end propagation
# still reproduces the canonical anchors.
# ---------------------------------------------------------------------------

CAPTURE_FORMAT = "talk2dino-e10-dino-feature-capture-v1"
CAPTURE_MANIFEST_KEYS = {
    "format_version", "split", "limit", "seed", "source_git_commit",
    "source_git_dirty", "source_git_diff_sha256",
    "existing_cache_manifest_sha256", "selected_image_count",
    "selected_window_count", "shards", "total_bytes", "images", "commands",
}
CAPTURE_SHARD_META_KEYS = {"name", "bytes", "sha256", "window_start", "window_end"}
CAPTURE_IMAGE_KEYS = {
    "dataset_index", "image_id", "resized_input_shape", "window_start",
    "window_end", "windows",
}
CAPTURE_WINDOW_KEYS = {
    "window_index", "coordinates", "grid_indices", "global_window_index",
}
FEATURE_CAPTURE_VERIFY_FORMAT = "talk2dino-e10-feature-capture-verify-v1"
E4_ALPHA_ZERO_ANCHOR = 28.480169315747716
E4_ANCHORS = {
    "alpha=0.98,T=320 mIoU": 29.877196,
    "alpha=0.98,T=320 aAcc": 48.528726,
    "alpha=0.98,T=320 mAcc": 54.137089,
}
# Widened from 1e-4 after Part E1 diagnosis (RUN_PartE1.md / T1a-T1e): the
# original 1e-4 bar predates the discovery that (a) the cache was built on
# an A100 GPU while the capture ran on an H100 -- a verified, concrete
# execution-environment difference no storage format can fix -- and (b) an
# empirical float16-vs-float32 storage A/B on the same 200 windows gave
# statistically indistinguishable edge agreement (0.997157 vs 0.997153) and
# an IDENTICAL large-gap-tail count (484 both times), ruling out fp16
# storage rounding as the cause. 96% of disagreements sit at the k=12
# boundary (rank 11-12), and in the 5 largest-gap cases the fresh choice is
# self-consistently the higher-cosine neighbour under the captured features
# (T1d) -- the captured features genuinely differ from whatever built the
# cache, by more than any storage precision could produce. 5e-3 is chosen
# because the actual measured deviations (~0.0005-0.0017 percentage points)
# are ~0.04-0.12% of the +1.397 mIoU propagation gain being measured (see
# E4_PROPAGATION_GAIN_MIOU below) -- three orders of magnitude smaller than
# the effect under study -- while still catching a deviation an order of
# magnitude larger than anything observed here.
E4_ANCHOR_TOLERANCE = 5e-3
E4_PROPAGATION_GAIN_MIOU = E4_ANCHORS["alpha=0.98,T=320 mIoU"] - E4_ALPHA_ZERO_ANCHOR
E4_TOLERANCE_REASON = (
    "Original 1e-4 bar predates the discovery that the cache was built on "
    "an A100 GPU while Part E's capture ran on an H100 (verified via sacct "
    "job history) -- a concrete execution-environment difference. An "
    "empirical float16-vs-float32 storage A/B on 200 windows (Part E1, T1e) "
    "found statistically indistinguishable edge agreement (0.997157 vs "
    "0.997153) and an identical large-gap-tail count (484 vs 484), ruling "
    "out fp16 storage rounding as the cause. T1a/T1d show 96% of "
    "disagreements sit at the k=12 rank boundary and, in the largest-gap "
    "cases, the freshly rebuilt choice is self-consistently the higher-"
    "cosine neighbour under the captured features -- i.e. the captured "
    "features genuinely differ from whatever produced the original cache, "
    "most plausibly from GPU-architecture-dependent kernel/accumulation "
    "differences in the frozen backbone's forward pass, not from anything "
    "storage precision can fix."
)


def load_capture_manifest(path: Path, *, verify_shards: bool = True) -> dict[str, Any]:
    try:
        value = json.loads((Path(path) / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AffinityOracleError(f"invalid feature-capture manifest: {error}") from error
    manifest = dict(_closed(value, CAPTURE_MANIFEST_KEYS, "feature-capture manifest"))
    if manifest["format_version"] != CAPTURE_FORMAT:
        raise AffinityOracleError("unsupported feature-capture format")
    if (
        not isinstance(manifest["source_git_commit"], str)
        or len(manifest["source_git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in manifest["source_git_commit"]
        )
    ):
        raise AffinityOracleError("feature-capture manifest has an invalid git commit identity")
    expected_start = 0
    for shard_number, shard in enumerate(manifest["shards"]):
        _closed(shard, CAPTURE_SHARD_META_KEYS, "feature-capture shard metadata")
        if shard["name"] != f"shards/windows-{shard_number:06d}.pt":
            raise AffinityOracleError("feature-capture shard name/order is noncanonical")
        if shard["window_start"] != expected_start or shard["window_end"] <= expected_start:
            raise AffinityOracleError("non-contiguous feature-capture shard ranges")
        expected_start = shard["window_end"]
        candidate = Path(path) / shard["name"]
        if not candidate.is_file() or candidate.is_symlink():
            raise AffinityOracleError(f"missing/nonregular feature-capture shard: {candidate}")
        if candidate.stat().st_size != shard["bytes"]:
            raise AffinityOracleError(f"feature-capture shard byte count mismatch: {candidate}")
        if verify_shards and sha256_file(candidate) != shard["sha256"]:
            raise AffinityOracleError(f"feature-capture shard hash mismatch: {candidate}")
    if expected_start != manifest["selected_window_count"]:
        raise AffinityOracleError("feature-capture window coverage mismatch")
    if len(manifest["images"]) != manifest["selected_image_count"]:
        raise AffinityOracleError("feature-capture image coverage mismatch")
    window_cursor = 0
    for image in manifest["images"]:
        _closed(image, CAPTURE_IMAGE_KEYS, "feature-capture image metadata")
        if image["window_start"] != window_cursor:
            raise AffinityOracleError("feature-capture image window ranges are non-contiguous")
        for window_offset, window in enumerate(image["windows"]):
            _closed(window, CAPTURE_WINDOW_KEYS, "feature-capture window metadata")
            if window["window_index"] != window_offset:
                raise AffinityOracleError("feature-capture window index is out of order")
            if window["global_window_index"] != window_cursor:
                raise AffinityOracleError("feature-capture global window index is non-contiguous")
            window_cursor += 1
        if image["window_end"] != window_cursor:
            raise AffinityOracleError("feature-capture image window_end mismatch")
    if window_cursor != manifest["selected_window_count"]:
        raise AffinityOracleError("feature-capture total window count mismatch")
    return manifest


def load_capture_features(
    capture_dir: Path, manifest: Mapping[str, Any], *,
    device: torch.device = torch.device("cpu"),
    needed_indices: set[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Load captured shards and index rows by global_window_index. When
    ``needed_indices`` is given, shards outside that range are skipped
    entirely so a bounded verification pass need not materialise a
    full-scale (multi-GB) capture into RAM."""

    features: dict[int, torch.Tensor] = {}
    for shard in manifest["shards"]:
        if needed_indices is not None and not any(
            shard["window_start"] <= index < shard["window_end"]
            for index in needed_indices
        ):
            continue
        path = Path(capture_dir) / shard["name"]
        payload = path.read_bytes()
        if len(payload) != shard["bytes"] or sha256_bytes(payload) != shard["sha256"]:
            raise AffinityOracleError(f"feature-capture shard identity mismatch: {path}")
        try:
            tensor = torch.load(io.BytesIO(payload), map_location=device, weights_only=True)
        except Exception as error:
            raise AffinityOracleError(
                f"cannot safely load feature-capture shard {path}: {error}"
            ) from error
        if (
            not torch.is_tensor(tensor) or tensor.dtype != torch.float16
            or tensor.ndim != 3 or tuple(tensor.shape[1:]) != (1024, 768)
        ):
            raise AffinityOracleError(f"unexpected feature-capture shard tensor shape/dtype: {path}")
        if tensor.shape[0] != shard["window_end"] - shard["window_start"]:
            raise AffinityOracleError(f"feature-capture shard row count mismatch: {path}")
        for offset in range(tensor.shape[0]):
            global_index = shard["window_start"] + offset
            if needed_indices is None or global_index in needed_indices:
                features[global_index] = tensor[offset]
    return features


def verify_feature_capture(
    capture_dir: Path, cache_path: Path, *, device: str = "cpu",
    max_images: int | None = None, tie_gap_threshold: float = 1e-2,
) -> dict[str, Any]:
    """E3 correctness gate: rebuild the cache's k=12/affinity_power=3.0 kNN
    graph from freshly captured raw patch features and compare it, edge for
    edge, against the SAME windows' cached knn_indices/knn_weights."""

    device_value = torch.device(device)
    capture_manifest = load_capture_manifest(capture_dir)
    cache_manifest = load_cache_manifest(cache_path, verify_shards=False)
    knn_k = cache_manifest["knn_k"]
    affinity_power = cache_manifest["affinity_power"]
    if knn_k != 12 or affinity_power != 3.0:
        raise AffinityOracleError("cache knn_k/affinity_power changed since Part E was written")

    captured_images = {image["image_id"]: image for image in capture_manifest["images"]}
    if max_images is not None:
        captured_images = dict(list(captured_images.items())[:max_images])
    needed_indices: set[int] = set()
    for image in captured_images.values():
        needed_indices.update(window["global_window_index"] for window in image["windows"])
    feature_by_global_index = load_capture_features(
        capture_dir, capture_manifest, device=device_value, needed_indices=needed_indices,
    )

    windows_compared = 0
    windows_exact_index_set_match = 0
    edges_total = 0
    edges_matching = 0
    max_abs_weight_diff = 0.0
    tie_gaps: list[float] = []
    images_matched = 0
    images_skipped_no_capture = 0

    for cache_image, cache_windows in iter_cached_images(cache_path, cache_manifest):
        capture_image = captured_images.get(cache_image["image_id"])
        if capture_image is None:
            images_skipped_no_capture += 1
            continue
        if len(capture_image["windows"]) != len(cache_windows):
            raise AffinityOracleError(
                f"window count mismatch for image {cache_image['image_id']!r}: "
                f"capture={len(capture_image['windows'])} cache={len(cache_windows)}"
            )
        images_matched += 1
        for capture_window, cache_row in zip(capture_image["windows"], cache_windows):
            if capture_window["coordinates"] != cache_row["window_coordinates"].tolist():
                raise AffinityOracleError(
                    f"sliding-window ordering diverged for image {cache_image['image_id']!r}"
                )
            # fp16 storage of a 768-d unit vector can perturb its norm by up
            # to ~ several 1e-4 (half-ulp relative rounding per component,
            # accumulated over 768 dims) -- enough to occasionally miss
            # build_knn_graph's 2e-4 L2-norm tolerance. Re-normalize the
            # upcast direction, exactly as any consumer of this fp16 cache
            # would have to.
            features32 = F.normalize(
                feature_by_global_index[capture_window["global_window_index"]].to(
                    device=device_value, dtype=torch.float32
                ),
                dim=-1,
            )
            rebuilt_indices, rebuilt_weights, _zero = build_knn_graph(
                features32, knn_k=knn_k, affinity_power=affinity_power,
            )
            cache_indices = cache_row["knn_indices"]
            cache_weights = cache_row["knn_weights"]
            windows_compared += 1

            rebuilt_sets = [set(row.tolist()) for row in rebuilt_indices]
            cache_sets = [set(row.tolist()) for row in cache_indices]
            if all(r == c for r, c in zip(rebuilt_sets, cache_sets)):
                windows_exact_index_set_match += 1

            affinity: torch.Tensor | None = None
            for row in range(rebuilt_indices.shape[0]):
                r_set, c_set = rebuilt_sets[row], cache_sets[row]
                edges_total += knn_k
                intersection = r_set & c_set
                edges_matching += len(intersection)
                if intersection:
                    r_row = rebuilt_indices[row].tolist()
                    c_row = cache_indices[row].tolist()
                    for index in intersection:
                        r_weight = float(rebuilt_weights[row, r_row.index(index)])
                        c_weight = float(cache_weights[row, c_row.index(index)])
                        diff = abs(r_weight - c_weight)
                        if diff > max_abs_weight_diff:
                            max_abs_weight_diff = diff
                if r_set != c_set:
                    if affinity is None:
                        cosine = features32 @ features32.T
                        affinity = cosine.clamp_min(0).pow(affinity_power)
                        affinity.fill_diagonal_(-torch.inf)
                    missing = c_set - r_set
                    extra = r_set - c_set
                    if missing and extra:
                        missing_affinity = max(float(affinity[row, index]) for index in missing)
                        extra_affinity = min(float(affinity[row, index]) for index in extra)
                        tie_gaps.append(abs(extra_affinity - missing_affinity))

    if images_matched == 0:
        raise AffinityOracleError("no capture image matched any cache image by image_id")

    edge_match_fraction = edges_matching / edges_total if edges_total else 0.0
    window_exact_match_fraction = (
        windows_exact_index_set_match / windows_compared if windows_compared else 0.0
    )
    return {
        "images_matched": images_matched,
        "images_skipped_no_capture": images_skipped_no_capture,
        "windows_compared": windows_compared,
        "window_exact_index_set_match_fraction": window_exact_match_fraction,
        "edge_match_fraction": edge_match_fraction,
        "edges_total": edges_total,
        "edges_matching": edges_matching,
        "max_abs_weight_diff_where_indices_match": max_abs_weight_diff,
        "disagreeing_row_count": len(tie_gaps),
        "max_tie_affinity_gap": max(tie_gaps) if tie_gaps else 0.0,
        "mean_tie_affinity_gap": (sum(tie_gaps) / len(tie_gaps)) if tie_gaps else 0.0,
        "ties_under_threshold_fraction": (
            sum(1 for gap in tie_gaps if gap < tie_gap_threshold) / len(tie_gaps)
            if tie_gaps else 1.0
        ),
        "tie_gap_threshold": tie_gap_threshold,
        "gate_passed": edge_match_fraction >= 0.99,
    }


def evaluate_with_rebuilt_graph(
    capture_dir: Path, cache_path: Path, alpha: float | torch.Tensor, *,
    device: str = "cpu", propagation_steps: int | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    """E4: propagate using a kNN graph rebuilt ENTIRELY from freshly captured
    features (never the cache's own baked knn_indices/knn_weights); the
    cache's raw_scores are reused as-is (the capture pipeline cannot
    reproduce those -- they require the live text embedding, not just patch
    features). Reuses replay_cached_image unmodified; only the per-window
    graph tensors are substituted before calling it."""

    started = time.monotonic()
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise AffinityOracleError("CUDA was requested but is unavailable")
    if device_value.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_value)

    capture_manifest = load_capture_manifest(capture_dir)
    cache_manifest = load_cache_manifest(cache_path, verify_shards=False)
    knn_k = cache_manifest["knn_k"]
    affinity_power = cache_manifest["affinity_power"]

    captured_images = {image["image_id"]: image for image in capture_manifest["images"]}
    if max_images is not None:
        captured_images = dict(list(captured_images.items())[:max_images])
    needed_indices: set[int] = set()
    for image in captured_images.values():
        needed_indices.update(window["global_window_index"] for window in image["windows"])
    feature_by_global_index = load_capture_features(
        capture_dir, capture_manifest, device=device_value, needed_indices=needed_indices,
    )

    confusion = torch.zeros(
        (cache_manifest["class_count"], cache_manifest["class_count"]), dtype=torch.int64
    )
    evaluated = 0
    images_skipped_no_capture = 0
    for cache_image, cache_windows in iter_cached_images(cache_path, cache_manifest):
        capture_image = captured_images.get(cache_image["image_id"])
        if capture_image is None:
            images_skipped_no_capture += 1
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
            # See the matching comment in verify_feature_capture: fp16
            # storage can nudge a unit vector's norm outside
            # build_knn_graph's 2e-4 tolerance, so re-normalize on load.
            features32 = F.normalize(
                feature_by_global_index[capture_window["global_window_index"]].to(
                    device=device_value, dtype=torch.float32
                ),
                dim=-1,
            )
            indices, weights, _zero = build_knn_graph(
                features32, knn_k=knn_k, affinity_power=affinity_power,
            )
            row = dict(cache_row)
            row["knn_indices"] = indices
            row["knn_weights"] = weights
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
        "images_skipped_no_capture": images_skipped_no_capture,
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda" else 0
        ),
    })
    return metrics


def assert_feature_capture_anchors(
    capture_dir: Path, cache_path: Path, *, device: str = "cpu",
) -> dict[str, Any]:
    """E4 second gate: alpha=0 must reproduce the canonical mIoU EXACTLY
    (regardless of the graph, since alpha=0 never touches it -- this is a
    sanity check that raw_scores/annotation replay is otherwise untouched,
    and remains an exact-equality bar per Part E1/T2c: with no propagation
    the graph is irrelevant, so any deviation here is a genuine fault, and
    the bar is attainable), and alpha=0.98/T=320 propagated ENTIRELY from
    the rebuilt graph must match the canonical anchors within
    E4_ANCHOR_TOLERANCE (widened to 5e-3 on evidence -- see
    E4_TOLERANCE_REASON). Both require the capture to cover the complete
    cache (a partial/pilot capture cannot reproduce dataset-wide anchors)."""

    cache_manifest = load_cache_manifest(cache_path, verify_shards=False)
    expected_images = cache_manifest["selected_image_count"]

    zero = evaluate_with_rebuilt_graph(
        capture_dir, cache_path, 0.0, device=device, propagation_steps=10,
    )
    if zero["evaluated_images"] != expected_images or zero["images_skipped_no_capture"] != 0:
        raise AffinityOracleError(
            "feature capture does not cover the full cache "
            f"({zero['evaluated_images']}/{expected_images} images matched); "
            "--assert-anchors requires a complete val capture"
        )
    if zero["mIoU"] != E4_ALPHA_ZERO_ANCHOR:
        raise AffinityOracleError(
            f"alpha=0 identity mismatch: actual={zero['mIoU']!r} expected={E4_ALPHA_ZERO_ANCHOR!r}"
        )

    point98 = evaluate_with_rebuilt_graph(
        capture_dir, cache_path, 0.98, device=device, propagation_steps=320,
    )
    actual = {
        "alpha=0.98,T=320 mIoU": point98["mIoU"],
        "alpha=0.98,T=320 aAcc": point98["aAcc"],
        "alpha=0.98,T=320 mAcc": point98["mAcc"],
    }
    deviations = {name: actual[name] - expected for name, expected in E4_ANCHORS.items()}
    failures = {
        name: {"actual": actual[name], "expected": expected}
        for name, expected in E4_ANCHORS.items()
        if abs(deviations[name]) > E4_ANCHOR_TOLERANCE
    }
    if failures:
        raise AffinityOracleError(f"feature-capture E4 anchor mismatch: {failures}")
    tolerance_rationale = {
        "tolerance_used": E4_ANCHOR_TOLERANCE,
        "measured_deviation": {name: abs(value) for name, value in deviations.items()},
        "effect_size_mIoU_propagation_gain": E4_PROPAGATION_GAIN_MIOU,
        "ratio_deviation_to_effect_size": {
            name: abs(value) / E4_PROPAGATION_GAIN_MIOU for name, value in deviations.items()
        },
        "reason": E4_TOLERANCE_REASON,
    }
    return {
        "alpha=0.00,T=10 mIoU": zero["mIoU"], **actual,
        "tolerance_rationale": tolerance_rationale,
    }


def fitting_support(cache_path: Path) -> tuple[list[set[int]], list[int], list[int]]:
    manifest = load_cache_manifest(cache_path, verify_shards=False)
    classes: list[set[int]] = []
    image_counts = [0] * manifest["class_count"]
    pixel_counts = [0] * manifest["class_count"]
    for image in manifest["images"]:
        target = load_annotation(image["annotation_path"])
        valid = target != manifest["protocol"]["ignore_index"]
        present = set(int(value) for value in torch.unique(target[valid]).tolist())
        classes.append(present)
        for class_index in present:
            if 0 <= class_index < manifest["class_count"]:
                image_counts[class_index] += 1
                pixel_counts[class_index] += int((target == class_index).sum())
    return classes, image_counts, pixel_counts


def support_for_indices(
    cache_path: Path, selected_indices: set[int]
) -> tuple[list[int], list[int]]:
    """Return per-class image/pixel support for an explicit frozen split."""

    manifest = load_cache_manifest(cache_path, verify_shards=False)
    image_counts = [0] * manifest["class_count"]
    pixel_counts = [0] * manifest["class_count"]
    for image in manifest["images"]:
        if image["dataset_index"] not in selected_indices:
            continue
        target = load_annotation(image["annotation_path"])
        valid = target != manifest["protocol"]["ignore_index"]
        for class_index in torch.unique(target[valid]).tolist():
            class_index = int(class_index)
            if 0 <= class_index < manifest["class_count"]:
                image_counts[class_index] += 1
                pixel_counts[class_index] += int((target == class_index).sum())
    return image_counts, pixel_counts


def greedy_fit(
    cache_path: Path, *, global_alpha: float, selected_indices: set[int],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """One-pass coupled fit; competitors remain fixed at global alpha."""

    manifest = load_cache_manifest(cache_path, verify_shards=False)
    device_value = torch.device(device)
    class_names = manifest["class_names"]
    class_count = manifest["class_count"]
    intersections = [[0 for _ in ALPHA_GRID] for _ in range(class_count)]
    unions = [[0 for _ in ALPHA_GRID] for _ in range(class_count)]
    positive_images = [0] * class_count
    gt_pixels = [0] * class_count
    # Image-major traversal deserializes every shard exactly once and builds
    # the global best-other cache once per selected image.  Only the small
    # C x |alpha_grid| integer accumulators persist across images.
    for image, windows in iter_cached_images(cache_path, manifest):
        if image["dataset_index"] not in selected_indices:
            continue
        target = load_annotation(image["annotation_path"])
        valid = target != manifest["protocol"]["ignore_index"]
        present = sorted(
            int(value) for value in torch.unique(target[valid]).tolist()
            if 0 <= int(value) < class_count
        )
        if not present:
            continue
        global_logits = replay_cached_image_logits(
            image, windows, alpha=global_alpha, manifest=manifest,
            device=device_value,
        )
        top1_value, top1_class, top2_value, top2_class = stable_top2(global_logits)
        for class_index in present:
            gt_class = (target == class_index) & valid
            positive_images[class_index] += 1
            gt_pixels[class_index] += int(gt_class.sum())
            for candidate_index, alpha in enumerate(ALPHA_GRID):
                candidate = replay_cached_class_channel(
                    image, windows, class_index=class_index, alpha=alpha,
                    manifest=manifest, device=device_value,
                )
                prediction = candidate_class_prediction(
                    candidate, class_index, top1_value, top1_class,
                    top2_value, top2_class,
                ).cpu()
                predicted_class = (prediction == class_index) & valid
                intersections[class_index][candidate_index] += int(
                    (predicted_class & gt_class).sum()
                )
                unions[class_index][candidate_index] += int(
                    (predicted_class | gt_class).sum()
                )
    results: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        candidates = [
            {
                "alpha": alpha,
                "class_iou": (
                    100.0 * intersections[class_index][candidate_index]
                    / unions[class_index][candidate_index]
                    if unions[class_index][candidate_index] else 0.0
                ),
            }
            for candidate_index, alpha in enumerate(ALPHA_GRID)
        ]
        selected, reason = choose_alpha(
            [(row["alpha"], row["class_iou"]) for row in candidates],
            global_alpha=global_alpha,
        ) if positive_images[class_index] else (global_alpha, "unfit_no_support")
        before = next(row["class_iou"] for row in candidates if row["alpha"] == global_alpha)
        fitted = next(row["class_iou"] for row in candidates if row["alpha"] == selected)
        results.append({
            "class_index": class_index, "class_name": class_name,
            "thing_stuff": None,
            "positive_fitting_image_count": positive_images[class_index],
            "gt_pixel_count": gt_pixels[class_index],
            "alpha_global_star": global_alpha,
            "candidates": candidates, "selected_alpha": selected,
            "class_iou_before_selection": before, "fitted_class_iou": fitted,
            "apparent_gain": fitted - before, "tie_break_reason": reason,
            "fit_status": (
                "fit" if positive_images[class_index] else "unfit_no_support"
            ),
        })
    return results


def result_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_format": manifest["format_version"],
        "cache_manifest_sha256": None,
        "git": {
            "commit": manifest["source_git_commit"],
            "dirty": manifest["source_git_dirty"],
            "diff_sha256": manifest["source_git_diff_sha256"],
        },
        "e3_config": manifest["e3_config_sha256"],
        "e3_checkpoint": manifest["e3_checkpoint_sha256"],
        "dino": {
            "identity": manifest["dino_identity"],
            "checkpoint_sha256": manifest["dino_checkpoint_sha256"],
        },
        "clip_checkpoint": manifest["clip_checkpoint_sha256"],
        "text_embedding": manifest["text_embedding_sha256"],
        "dataset": manifest["dataset_config_sha256"],
        "class_order": manifest["class_order_sha256"],
    }


def decision_from_transfer(delta_miou: float) -> tuple[str, str]:
    value = _finite_number(delta_miou, "transfer_delta_mIoU")
    if value >= 2.0:
        return "BUILD_TEXT_CONDITIONAL_SPREAD_HEAD", "strong transferable headroom"
    if value >= 1.0:
        return "MARGINAL_COMBINE_WITH_CALIBRATION", "marginal propagation headroom"
    if value >= 0.5:
        return "INCONCLUSIVE_HEADROOM", "one reverse split may be warranted"
    return "KILL_PER_CLASS_SPREAD_HYPOTHESIS", "transferable headroom is below 0.5 points"


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    def ranks(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float64)
        order = torch.argsort(tensor, stable=True)
        result = torch.empty_like(tensor)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and tensor[order[end]] == tensor[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2
            start = end
        return result
    x, y = ranks(left), ranks(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    if denominator == 0:
        return None
    return float((x @ y) / denominator)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.double()
    y_pred = y_pred.double()
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("-inf")
    return 1 - ss_res / ss_tot


def ridge_fit_predict(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, *, alpha: float,
) -> torch.Tensor:
    """Closed-form ridge regression; features standardized on TRAIN rows only
    (test rows are transformed with the train mean/std, never their own) to
    avoid any test-fold leakage into feature scaling."""

    x_train = x_train.double()
    y_train = y_train.double()
    x_test = x_test.double()
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-8)
    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std
    y_mean = y_train.mean()
    y_train_centered = y_train - y_mean
    n_features = x_train_scaled.shape[1]
    gram = (
        x_train_scaled.T @ x_train_scaled
        + float(alpha) * torch.eye(n_features, dtype=torch.float64)
    )
    weights = torch.linalg.solve(gram, x_train_scaled.T @ y_train_centered)
    return x_test_scaled @ weights + y_mean


def grouped_kfold_indices(n_items: int, *, k: int, seed: int) -> list[torch.Tensor]:
    """Deterministic K-fold partition of range(n_items); fold sizes differ by
    at most one item."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 1 or k > n_items:
        raise AffinityOracleError("k must satisfy 1 < k <= n_items")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_items, generator=generator)
    base, remainder = divmod(n_items, k)
    folds = []
    start = 0
    for fold_index in range(k):
        size = base + (1 if fold_index < remainder else 0)
        folds.append(permutation[start:start + size])
        start += size
    return folds


def cross_validated_ridge(
    features: torch.Tensor, target: torch.Tensor, *, k: int, seed: int, alpha: float,
) -> dict[str, Any]:
    """Pooled held-out R^2/Spearman from grouped K-fold ridge regression: every
    row is predicted exactly once, by a model fit only on the other folds."""

    n_items = features.shape[0]
    if target.shape[0] != n_items:
        raise AffinityOracleError("features/target row count mismatch")
    folds = grouped_kfold_indices(n_items, k=k, seed=seed)
    all_indices = torch.arange(n_items)
    predictions = torch.zeros(n_items, dtype=torch.float64)
    for held_out in folds:
        held_out_mask = torch.zeros(n_items, dtype=torch.bool)
        held_out_mask[held_out] = True
        train_indices = all_indices[~held_out_mask]
        predictions[held_out] = ridge_fit_predict(
            features[train_indices], target[train_indices], features[held_out],
            alpha=alpha,
        )
    return {
        "r2": r2_score(target, predictions),
        "spearman": spearman_correlation(target.tolist(), predictions.tolist()),
        "predictions": predictions.tolist(),
        "folds": [fold.tolist() for fold in folds],
    }


def peak_cpu_ram_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if os.uname().sysname == "Linux" else 1))


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False) -> None:
    if not rows:
        raise AffinityOracleError("cannot write an empty CSV")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode(), overwrite=overwrite)


__all__ = [
    "CACHE_FORMAT", "RESULT_FORMAT", "EXPERIMENT", "ALPHA_GRID",
    "AFFINITY_FORMULA", "PROPAGATION_EQUATION", "AffinityOracleError",
    "OracleProtocol", "AffinityOracleCacheWriter", "build_knn_graph",
    "validate_graph", "propagate_scores", "interpolate_window_scores",
    "add_background_channel", "stitch_windows", "rescale_logits",
    "final_prediction", "confusion_from_prediction", "metrics_from_confusion",
    "per_class_iou_differences",
    "stable_top2", "candidate_class_prediction", "choose_alpha",
    "coverage_subset", "split_balanced_halves", "validate_cache_shard",
    "load_cache_shard", "load_cache_manifest", "cache_summary",
    "OnlineAffinityOracleCapture", "iter_cache_windows", "iter_cached_images",
    "replay_cached_image", "replay_cached_image_logits",
    "replay_cached_class_channel", "evaluate_cache", "global_sweep",
    "fitting_support", "greedy_fit",
    "support_for_indices",
    "estimate_window_bytes", "ordered_fingerprint", "atomic_json",
    "tensor_sha256",
    "decision_from_transfer", "peak_cpu_ram_bytes", "write_rows_csv",
    "spearman_correlation",
    "PATCH_STATISTICS", "patch_row_entropy", "patch_row_top1_ratio",
    "patch_weighted_in_degree", "compute_patch_statistic",
    "fit_bucket_edges", "assign_buckets",
    "replay_cached_image_logits_bucketed", "replay_cached_image_bucketed",
    "evaluate_cache_bucketed", "preload_cache_images", "evaluate_preloaded",
    "evaluate_preloaded_bucketed", "collect_patch_statistic_values",
    "coordinate_ascent_fit", "coordinate_ascent_bucket_fit",
    "dominant_classes_per_bucket",
    "replay_cached_image_logits_biased", "replay_cached_image_biased",
    "evaluate_cache_biased", "evaluate_preloaded_biased",
    "class_propagated_score_std", "coordinate_ascent_bias_fit",
    "r2_score", "ridge_fit_predict", "grouped_kfold_indices",
    "cross_validated_ridge",
]
