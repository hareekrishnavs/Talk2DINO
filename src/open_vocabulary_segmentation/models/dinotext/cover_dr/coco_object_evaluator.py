"""COCO-Object protocol-confirmation evaluator core: one shared E3
snapshot and one shared canonical top-12 graph selection per window,
producing THREE variants -- raw E3 (no propagation), matched k11, and
k12 -- propagated/converted with the exact same primitives
:mod:`.matched_power_evaluator` already verified
(``build_directed_topk_graph``, ``build_matched_k11_from_k12``,
``finite_step_propagate``, ``inference.model.generate_patch_snapshot``,
``inference.model.masks_from_patch_scores`` -- every one imported
unmodified, never reimplemented).

The only genuinely new production logic in this module is
:func:`apply_background_channel`, which reproduces -- not redesigns --
the canonical constant-threshold background mechanism already used by
``segmentation.evaluation.dinotext_seg.DINOTextSegInference.encode_decode``
(``torch.full`` + ``torch.cat`` at channel 0). That production code path
injects the background channel per-window, before stitching; this module
injects it once, after stitching, on the already-averaged foreground
scores. These are mathematically identical: the background channel is a
spatially-uniform constant, and averaging a uniform constant over any
positive per-pixel window-coverage count reproduces that same constant
exactly -- see ``docs/coco_object_protocol_confirmation.md`` for the
worked equivalence argument. This lets every variant (E3, k11, k12)
reuse ``matched_power_evaluator``-style accumulation completely
unmodified over the 80 foreground-only channels, with background handled
as one small, shared, independently-testable post-stitch step -- exactly
the "production-equivalent helper shared by E3, k11 and k12" this
protocol confirmation requires.

Never calls ``DINOTextSegInference.encode_decode``/``inference()``/
``simple_test()`` (which would additionally require ``with_bg=False``
whenever RWR is enabled at construction time -- an existing, orthogonal
constraint this module's direct-primitive orchestration never triggers).
Never applies PAMR, Hann/majority/center-selection stitching, or any
second graph/propagation for k11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .finite_step_regime import (
    FiniteStepRegimeError,
    MatchedGraphDiagnostics,
    build_matched_k11_from_k12,
    compute_matched_graph_diagnostics,
    finite_step_propagate,
)
from .graph import build_directed_topk_graph


class CocoObjectEvaluatorError(RuntimeError):
    """Raised on any COCO-Object protocol-confirmation evaluator invariant
    violation. Always fail closed -- never silently substitute a fallback
    stitching, solver, background rule, or graph-construction path."""


# ---------------------------------------------------------------------------
# Background channel: the one genuinely new (but formula-reused) step
# ---------------------------------------------------------------------------


def apply_background_channel(masks: torch.Tensor, *, bg_thresh: float) -> torch.Tensor:
    """Prepend a spatially-uniform constant background channel at index 0,
    reproducing ``DINOTextSegInference.encode_decode``'s own formula
    exactly (``torch.full([B,1,H,W], bg_thresh)`` concatenated in front of
    the foreground channels). Pure tensor arithmetic -- no model, no
    CUDA required to exercise or test this function.

    ``masks`` must be ``[B, C, H, W]`` with ``C`` the foreground-only
    class count (no background channel yet). Returns ``[B, C+1, H, W]``.
    """
    if masks.ndim != 4:
        raise CocoObjectEvaluatorError(f"apply_background_channel requires a [B, C, H, W] tensor, got ndim={masks.ndim}")
    if not isinstance(bg_thresh, float) or not (0.0 <= bg_thresh <= 1.0):
        raise CocoObjectEvaluatorError(f"bg_thresh must be a float in [0, 1], got {bg_thresh!r}")
    batch, _, height, width = masks.shape
    background = torch.full((batch, 1, height, width), bg_thresh, dtype=masks.dtype, device=masks.device)
    return torch.cat([background, masks], dim=1)


# ---------------------------------------------------------------------------
# Per-window operation telemetry (E3 + k11 + k12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowOperationTelemetryE3:
    """Exact per-completed-window operation counts for the E3/k11/k12
    protocol confirmation: one shared backbone/snapshot/DINO-feature/
    top-12-selection pass, two graph normalizations (k11, k12 -- E3 has
    none), two finite-step propagations of exactly 320 updates each
    (E3 propagation is contractually zero), and three independent
    sigmoid+interpolation conversions (one per variant, including E3's
    own unpropagated conversion)."""

    backbone_snapshot_calls: int
    dino_feature_extractions: int
    topk_selection_calls: int
    graph_normalizations: int
    finite_step_propagations: int
    e3_propagations: int
    k11_updates: int
    k12_updates: int
    sigmoid_calls: int
    interpolation_calls: int

    def __post_init__(self) -> None:
        fixed = {
            "backbone_snapshot_calls": 1,
            "dino_feature_extractions": 1,
            "topk_selection_calls": 1,
            "graph_normalizations": 2,
            "finite_step_propagations": 2,
            "e3_propagations": 0,
            "sigmoid_calls": 3,
            "interpolation_calls": 3,
        }
        for name, expected in fixed.items():
            observed = getattr(self, name)
            if observed != expected:
                raise CocoObjectEvaluatorError(
                    f"WindowOperationTelemetryE3.{name} must be exactly {expected}, observed {observed!r}"
                )
        if self.k11_updates != 320:
            raise CocoObjectEvaluatorError(f"WindowOperationTelemetryE3.k11_updates must be exactly 320, observed {self.k11_updates!r}")
        if self.k12_updates != 320:
            raise CocoObjectEvaluatorError(f"WindowOperationTelemetryE3.k12_updates must be exactly 320, observed {self.k12_updates!r}")


def aggregate_telemetry(entries: list[WindowOperationTelemetryE3]) -> dict[str, int]:
    if not entries:
        raise CocoObjectEvaluatorError("aggregate_telemetry requires at least one window entry")
    fields = (
        "backbone_snapshot_calls", "dino_feature_extractions", "topk_selection_calls",
        "graph_normalizations", "finite_step_propagations", "e3_propagations",
        "k11_updates", "k12_updates", "sigmoid_calls", "interpolation_calls",
    )
    return {field: sum(getattr(entry, field) for entry in entries) for field in fields}


# ---------------------------------------------------------------------------
# Per-window matched execution (E3 + k11 + k12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowVariantResultE3:
    """One window's fully-processed E3/k11/k12 output: sigmoid+
    interpolated foreground-only mask tensors (shape ``[1, C, crop_h,
    crop_w]``, C = foreground class count, no background channel yet),
    the matched-graph diagnostics comparing k11 against k12, and this
    window's exact operation telemetry."""

    e3_masks: torch.Tensor
    k11_masks: torch.Tensor
    k12_masks: torch.Tensor
    graph_diagnostics: MatchedGraphDiagnostics
    telemetry: WindowOperationTelemetryE3


def process_one_window_with_e3(
    inference: Any,
    image_tensor: torch.Tensor,
    window: Any,
    *,
    alpha: float,
    steps: int,
    affinity_power: float,
) -> WindowVariantResultE3:
    """Run the shared per-window flow for one sliding-window crop,
    identical to :func:`matched_power_evaluator.process_one_window`
    (same snapshot call, same graph construction, same k11 matched-prefix
    derivation, same propagation kernel), with one addition: the raw
    unary scores ``s0`` -- the same tensor the k11/k12 propagations start
    from -- are also converted directly via
    ``inference.model.masks_from_patch_scores`` (no
    ``finite_step_propagate`` call), giving the unpropagated E3 variant
    at zero additional backbone/snapshot/graph cost."""
    crop_rows, crop_cols = window.crop_slice
    crop = image_tensor[:, :, crop_rows, crop_cols]

    snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
    s0 = snapshot.unary_scores[0]
    dino_features = snapshot.dino_features[0]
    grid_hw = snapshot.grid_hw

    graph12 = build_directed_topk_graph(dino_features, k=12, affinity_power=affinity_power)
    graph11 = build_matched_k11_from_k12(graph12)
    graph_diagnostics = compute_matched_graph_diagnostics(graph12, graph11)

    trace12 = finite_step_propagate(graph12, s0, alpha=alpha, steps=steps, snapshot_steps=(steps,))
    trace11 = finite_step_propagate(graph11, s0, alpha=alpha, steps=steps, snapshot_steps=(steps,))

    crop_hw = window.extent.as_tuple()
    masks_e3 = inference.model.masks_from_patch_scores(s0.unsqueeze(0), grid_hw, crop_hw)
    masks11 = inference.model.masks_from_patch_scores(trace11.snapshots[steps].unsqueeze(0), grid_hw, crop_hw)
    masks12 = inference.model.masks_from_patch_scores(trace12.snapshots[steps].unsqueeze(0), grid_hw, crop_hw)

    telemetry = WindowOperationTelemetryE3(
        backbone_snapshot_calls=1,
        dino_feature_extractions=1,
        topk_selection_calls=1,
        graph_normalizations=2,
        finite_step_propagations=2,
        e3_propagations=0,
        k11_updates=trace11.steps_completed,
        k12_updates=trace12.steps_completed,
        sigmoid_calls=3,
        interpolation_calls=3,
    )
    return WindowVariantResultE3(
        e3_masks=masks_e3, k11_masks=masks11, k12_masks=masks12,
        graph_diagnostics=graph_diagnostics, telemetry=telemetry,
    )


# ---------------------------------------------------------------------------
# Image-level stitching: mirrors matched_power_evaluator.stitch_one_image
# exactly, generalized from two accumulators to three (E3, k11, k12).
# ---------------------------------------------------------------------------


def _pad_into(preds: torch.Tensor, masks: torch.Tensor, window: Any) -> None:
    accum_rows, accum_cols = window.accumulation_slice
    preds += F.pad(
        masks,
        (
            accum_cols.start, preds.shape[3] - accum_cols.stop,
            accum_rows.start, preds.shape[2] - accum_rows.stop,
        ),
    )


def _increment_count(count_mat: torch.Tensor, window: Any) -> None:
    accum_rows, accum_cols = window.accumulation_slice
    count_mat[:, :, accum_rows, accum_cols] += 1


@dataclass(frozen=True)
class ImageVariantStitchResultE3:
    """One image's fully-stitched (averaged, not yet background-injected/
    rescaled/argmaxed) per-variant foreground-only score tensors, each
    shape ``[1, C, H, W]``, plus per-window diagnostics/telemetry in
    row-major window order."""

    e3_stitched: torch.Tensor
    k11_stitched: torch.Tensor
    k12_stitched: torch.Tensor
    window_count: int
    window_telemetry: tuple[WindowOperationTelemetryE3, ...]
    graph_diagnostics: tuple[MatchedGraphDiagnostics, ...]


def stitch_one_image_with_e3(
    inference: Any,
    image_tensor: torch.Tensor,
    plan: Any,
    *,
    class_count: int,
    alpha: float,
    steps: int,
    affinity_power: float,
) -> ImageVariantStitchResultE3:
    """Process every window of ``plan`` in row-major flat-index order,
    accumulating each of the three variants into its own FP32 numerator
    against one shared FP32 coverage/count map (incremented exactly once
    per window, never once per variant), then divide only after all of
    this image's windows have been accumulated. ``class_count`` here is
    the FOREGROUND-ONLY class count (80 for COCO-Object) -- the
    background channel is added later, once, after stitching, via
    :func:`apply_background_channel`. No Hann/majority/center-selection
    weighting and no subtract/add delta re-stitching -- uniform averaging
    only, exactly mirroring canonical E3/RWR stitching."""
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise CocoObjectEvaluatorError("stitch_one_image_with_e3 requires a batched [1, C, H, W] image tensor")
    h_img, w_img = plan.image_size.as_tuple()
    if tuple(image_tensor.shape[-2:]) != (h_img, w_img):
        raise CocoObjectEvaluatorError("image_tensor spatial extent does not match plan.image_size")

    preds_e3 = image_tensor.new_zeros((1, class_count, h_img, w_img))
    preds11 = image_tensor.new_zeros((1, class_count, h_img, w_img))
    preds12 = image_tensor.new_zeros((1, class_count, h_img, w_img))
    count_mat = image_tensor.new_zeros((1, 1, h_img, w_img))

    telemetry: list[WindowOperationTelemetryE3] = []
    diagnostics: list[MatchedGraphDiagnostics] = []
    for window in plan.windows:
        result = process_one_window_with_e3(
            inference, image_tensor, window, alpha=alpha, steps=steps, affinity_power=affinity_power
        )
        _pad_into(preds_e3, result.e3_masks, window)
        _pad_into(preds11, result.k11_masks, window)
        _pad_into(preds12, result.k12_masks, window)
        _increment_count(count_mat, window)
        telemetry.append(result.telemetry)
        diagnostics.append(result.graph_diagnostics)

    if torch.any(count_mat == 0):
        raise CocoObjectEvaluatorError("matched stitching left uncovered pixels")

    return ImageVariantStitchResultE3(
        e3_stitched=preds_e3 / count_mat,
        k11_stitched=preds11 / count_mat,
        k12_stitched=preds12 / count_mat,
        window_count=len(plan.windows),
        window_telemetry=tuple(telemetry),
        graph_diagnostics=tuple(diagnostics),
    )


def finalize_prediction(
    stitched: torch.Tensor, img_meta: Any, *, align_corners: bool, bg_thresh: float,
) -> torch.Tensor:
    """Apply the canonical background channel to one already-stitched,
    foreground-only variant tensor, then perform the canonical output
    crop/unpadding + rescale + per-pixel argmax, mirroring
    ``DINOTextSegInference._rescale``/``simple_test`` exactly. Background
    competes in the argmax like every other channel, at index 0 --
    matching ``COCOObjectDataset.CLASSES[0] == 'background'``."""
    with_background = apply_background_channel(stitched, bg_thresh=bg_thresh)
    resize_h, resize_w = img_meta["img_shape"][:2]
    cropped = with_background[:, :, :resize_h, :resize_w]
    rescaled = F.interpolate(
        cropped, size=tuple(img_meta["ori_shape"][:2]), mode="bilinear", align_corners=align_corners
    )
    return rescaled.argmax(dim=1)


__all__ = [
    "CocoObjectEvaluatorError",
    "FiniteStepRegimeError",
    "ImageVariantStitchResultE3",
    "WindowOperationTelemetryE3",
    "WindowVariantResultE3",
    "aggregate_telemetry",
    "apply_background_channel",
    "finalize_prediction",
    "process_one_window_with_e3",
    "stitch_one_image_with_e3",
]
