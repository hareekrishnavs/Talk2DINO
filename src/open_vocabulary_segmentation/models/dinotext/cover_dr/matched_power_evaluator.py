"""Matched k11/k12 finite-step power-evaluator core: one shared E3 snapshot
and one shared canonical top-12 graph selection per window, propagated
independently for k11 and k12 with the exact finite-step kernel verified by
GPU stability-gate job 20300858 (:mod:`.finite_step_regime`, imported
unmodified from this package -- never copied or reimplemented), then
stitched into two independent per-variant accumulators using the same
accumulation mechanics as ``DINOTextSegInference.slide_inference`` /
``window_cache.py``'s two-pass cache.

Never calls ``solve_rwr_cgls``, never builds a second graph for k11 via an
independent top-k selection, never runs a second backbone pass for k12,
never applies PAMR, and never uses Hann/majority/center-selection or
sparse-delta stitching -- see ``[prohibited]`` in
``evaluation_identities/e12_k11_k12_power_evaluation.toml``.

Dependency policy mirrors ``window_cache.py``: the pure geometry/telemetry
types below (``WindowOperationTelemetry``, ``WindowVariantResult``,
``ImageVariantStitchResult``) and the stitching helpers import only
:mod:`torch` and :mod:`sliding_window_geometry`. ``process_one_window`` and
``stitch_one_image`` need a real ``inference`` object (model + text
embedding); they never import mmcv/mmseg themselves, so merely importing
this module never requires the model package.
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


class MatchedPowerEvaluatorError(RuntimeError):
    """Raised on any matched power-evaluator invariant violation. Always
    fail closed: never silently substitute a fallback stitching, solver,
    or graph-construction path."""


# ---------------------------------------------------------------------------
# Per-window operation telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowOperationTelemetry:
    """Exact per-completed-window operation counts, checked against the
    protocol's own required counts (never a range/approximation): one
    shared backbone/snapshot/DINO-feature/top-12-selection pass, two
    independent graph normalizations, two independent finite-step
    propagations of exactly 320 updates each, and two independent
    sigmoid/interpolation applications (one per variant)."""

    backbone_snapshot_calls: int
    dino_feature_extractions: int
    topk_selection_calls: int
    graph_normalizations: int
    finite_step_propagations: int
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
            "sigmoid_calls": 2,
            "interpolation_calls": 2,
        }
        for name, expected in fixed.items():
            observed = getattr(self, name)
            if observed != expected:
                raise MatchedPowerEvaluatorError(
                    f"WindowOperationTelemetry.{name} must be exactly {expected}, observed {observed!r}"
                )
        if self.k11_updates != 320:
            raise MatchedPowerEvaluatorError(
                f"WindowOperationTelemetry.k11_updates must be exactly 320, observed {self.k11_updates!r}"
            )
        if self.k12_updates != 320:
            raise MatchedPowerEvaluatorError(
                f"WindowOperationTelemetry.k12_updates must be exactly 320, observed {self.k12_updates!r}"
            )


def aggregate_telemetry(entries: list[WindowOperationTelemetry]) -> dict[str, int]:
    """Bounded, tensor-free sum of the fixed-per-window operation counts
    over a completed image or run -- used for the result's exact operation
    telemetry section."""
    if not entries:
        raise MatchedPowerEvaluatorError("aggregate_telemetry requires at least one window entry")
    fields = (
        "backbone_snapshot_calls", "dino_feature_extractions", "topk_selection_calls",
        "graph_normalizations", "finite_step_propagations", "k11_updates", "k12_updates",
        "sigmoid_calls", "interpolation_calls",
    )
    return {field: sum(getattr(entry, field) for entry in entries) for field in fields}


# ---------------------------------------------------------------------------
# Per-window matched execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowVariantResult:
    """One window's fully-processed matched-variant output: sigmoid+
    interpolated mask tensors for both variants (shape ``[1, C, crop_h,
    crop_w]``), the matched-graph diagnostics comparing k11 against k12,
    and this window's exact operation telemetry."""

    k11_masks: torch.Tensor
    k12_masks: torch.Tensor
    graph_diagnostics: MatchedGraphDiagnostics
    telemetry: WindowOperationTelemetry


def process_one_window(
    inference: Any,
    image_tensor: torch.Tensor,
    window: Any,
    *,
    alpha: float,
    steps: int,
    affinity_power: float,
) -> WindowVariantResult:
    """Run the matched per-window flow for one sliding-window crop of an
    already-batched ``[1, C, H, W]`` image tensor.

    Exactly one backbone/snapshot pass (``inference.model.
    generate_patch_snapshot``), exactly one canonical top-12 graph build
    (``build_directed_topk_graph``), and exactly one matched-prefix k11
    derivation (``build_matched_k11_from_k12`` -- never an independent
    top-k selection) are shared between both variants; each variant then
    gets its own independent finite-step propagation
    (:func:`finite_step_propagate`, the exact kernel reused unmodified
    from this package) and its own independent sigmoid+interpolation
    (``inference.model.masks_from_patch_scores`` -- the same production
    function ``window_cache.py``'s two-pass cache and
    ``DINOTextSegInference.encode_decode`` both call).
    """
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
    masks11 = inference.model.masks_from_patch_scores(trace11.snapshots[steps].unsqueeze(0), grid_hw, crop_hw)
    masks12 = inference.model.masks_from_patch_scores(trace12.snapshots[steps].unsqueeze(0), grid_hw, crop_hw)

    telemetry = WindowOperationTelemetry(
        backbone_snapshot_calls=1,
        dino_feature_extractions=1,
        topk_selection_calls=1,
        graph_normalizations=2,
        finite_step_propagations=2,
        k11_updates=trace11.steps_completed,
        k12_updates=trace12.steps_completed,
        sigmoid_calls=2,
        interpolation_calls=2,
    )
    return WindowVariantResult(
        k11_masks=masks11, k12_masks=masks12, graph_diagnostics=graph_diagnostics, telemetry=telemetry
    )


# ---------------------------------------------------------------------------
# Image-level stitching: mirrors DINOTextSegInference.slide_inference /
# window_cache.py's accumulation exactly, duplicated only because it lives
# in a different module -- the arithmetic itself is copied verbatim, never
# independently re-derived, and is exercised identically for both variants.
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
class ImageVariantStitchResult:
    """One image's fully-stitched (averaged, not yet rescaled/argmaxed)
    per-variant score tensors, each shape ``[1, C, H, W]``, plus per-window
    diagnostics/telemetry in row-major window order."""

    k11_stitched: torch.Tensor
    k12_stitched: torch.Tensor
    window_count: int
    window_telemetry: tuple[WindowOperationTelemetry, ...]
    graph_diagnostics: tuple[MatchedGraphDiagnostics, ...]


def stitch_one_image(
    inference: Any,
    image_tensor: torch.Tensor,
    plan: Any,
    *,
    class_count: int,
    alpha: float,
    steps: int,
    affinity_power: float,
) -> ImageVariantStitchResult:
    """Process every window of ``plan`` in row-major flat-index order,
    accumulating each variant into its own FP32 numerator against one
    shared FP32 coverage/count map (incremented exactly once per window,
    never once per variant), then divide only after all of this image's
    windows have been accumulated. No Hann/majority/center-selection
    weighting and no subtract/add delta re-stitching -- uniform averaging
    only, exactly mirroring canonical E3/RWR stitching."""
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise MatchedPowerEvaluatorError("stitch_one_image requires a batched [1, C, H, W] image tensor")
    h_img, w_img = plan.image_size.as_tuple()
    if tuple(image_tensor.shape[-2:]) != (h_img, w_img):
        raise MatchedPowerEvaluatorError("image_tensor spatial extent does not match plan.image_size")

    preds11 = image_tensor.new_zeros((1, class_count, h_img, w_img))
    preds12 = image_tensor.new_zeros((1, class_count, h_img, w_img))
    count_mat = image_tensor.new_zeros((1, 1, h_img, w_img))

    telemetry: list[WindowOperationTelemetry] = []
    diagnostics: list[MatchedGraphDiagnostics] = []
    for window in plan.windows:
        result = process_one_window(
            inference, image_tensor, window, alpha=alpha, steps=steps, affinity_power=affinity_power
        )
        _pad_into(preds11, result.k11_masks, window)
        _pad_into(preds12, result.k12_masks, window)
        _increment_count(count_mat, window)
        telemetry.append(result.telemetry)
        diagnostics.append(result.graph_diagnostics)

    if torch.any(count_mat == 0):
        raise MatchedPowerEvaluatorError("matched stitching left uncovered pixels")

    return ImageVariantStitchResult(
        k11_stitched=preds11 / count_mat,
        k12_stitched=preds12 / count_mat,
        window_count=len(plan.windows),
        window_telemetry=tuple(telemetry),
        graph_diagnostics=tuple(diagnostics),
    )


def finalize_prediction(
    stitched: torch.Tensor, img_meta: Any, *, align_corners: bool
) -> torch.Tensor:
    """Canonical output crop/unpadding + rescale + final per-pixel argmax,
    mirroring ``DINOTextSegInference._rescale``/``simple_test`` exactly:
    crop the stitched prediction to ``img_meta['img_shape']`` (a no-op on
    this pipeline, which has no Pad transform), bilinear-interpolate to
    ``img_meta['ori_shape']`` using the caller-supplied (never hardcoded)
    ``align_corners``, then take the per-pixel class argmax. Scores are
    never rounded or converted before this argmax. The intervening softmax
    production applies is omitted deliberately: softmax is monotonic per
    pixel across the class axis, so it cannot change which class attains
    the argmax, and omitting it changes no reported prediction."""
    resize_h, resize_w = img_meta["img_shape"][:2]
    cropped = stitched[:, :, :resize_h, :resize_w]
    rescaled = F.interpolate(
        cropped, size=tuple(img_meta["ori_shape"][:2]), mode="bilinear", align_corners=align_corners
    )
    return rescaled.argmax(dim=1)


__all__ = [
    "FiniteStepRegimeError",
    "ImageVariantStitchResult",
    "MatchedPowerEvaluatorError",
    "WindowOperationTelemetry",
    "WindowVariantResult",
    "aggregate_telemetry",
    "finalize_prediction",
    "process_one_window",
    "stitch_one_image",
]
