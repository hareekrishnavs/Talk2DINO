"""Reusable, independently-testable stitching-control suite.

Holds one shared, immutable k12 finite-step per-window output fixed (same
image, same E3 patch scores/features, same directed top-12 graph, same
ReLU(cosine)^3 affinity, same alpha=0.98/steps=320 propagation, same crop
geometry and row-major crop order -- never rebuilt or rerun per variant)
and varies ONLY the crop-to-image aggregation rule across four frozen
variants: ``uniform_probability`` (must reproduce
``matched_power_evaluator.stitch_one_image``'s k12 path exactly),
``hann_probability``, ``uniform_score``, and ``hann_score``. See
``evaluation_identities/e12_stitching_control_suite.toml`` for the sole
authoritative variant/formula contract -- nothing here is a second source
of truth; every constant this module accepts is validated against that
identity by :mod:`src.stitching_control_identity`, never hardcoded twice.

This module never selects T4 targets, never builds cross-view consensus,
never touches DCR/SUR, never prunes or edits the graph, never runs a
counterfactual/adjoint solve, and never introduces a learned parameter --
it is a pure post-hoc aggregation-policy control over an already-computed,
immutable per-window score tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


class StitchingControlError(RuntimeError):
    """Raised on any stitching-control invariant violation. Always fail
    closed: never silently substitute a fallback weighting, accumulator,
    or crop order."""


STAGE_PROBABILITY = "probability"
STAGE_SCORE = "score"
WEIGHTING_UNIFORM = "uniform"
WEIGHTING_HANN = "hann"
SIGMOID_BEFORE_INTERPOLATION = "before_interpolation"
SIGMOID_AFTER_STITCH = "after_stitch"


# ---------------------------------------------------------------------------
# Frozen variant specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StitchingVariantSpec:
    """One immutable, validated stitching-variant specification. Never
    constructed with an unsupported stage/weighting/sigmoid_stage
    combination -- see ``CANONICAL_VARIANTS`` for the four frozen,
    identity-locked instances actually used."""

    name: str
    stage: str
    weighting: str
    sigmoid_stage: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise StitchingControlError("StitchingVariantSpec.name must be a non-empty string")
        if self.stage not in (STAGE_PROBABILITY, STAGE_SCORE):
            raise StitchingControlError(f"StitchingVariantSpec.stage must be 'probability' or 'score', observed {self.stage!r}")
        if self.weighting not in (WEIGHTING_UNIFORM, WEIGHTING_HANN):
            raise StitchingControlError(f"StitchingVariantSpec.weighting must be 'uniform' or 'hann', observed {self.weighting!r}")
        if self.sigmoid_stage not in (SIGMOID_BEFORE_INTERPOLATION, SIGMOID_AFTER_STITCH):
            raise StitchingControlError(f"StitchingVariantSpec.sigmoid_stage must be a supported stage, observed {self.sigmoid_stage!r}")
        # A probability-stage variant sigmoids before interpolation (matching
        # the existing evaluator's masks_from_patch_scores contract); a
        # score-stage variant sigmoids exactly once, after stitching. No
        # other combination is a valid frozen variant.
        if self.stage == STAGE_PROBABILITY and self.sigmoid_stage != SIGMOID_BEFORE_INTERPOLATION:
            raise StitchingControlError("a probability-stage variant must sigmoid before interpolation")
        if self.stage == STAGE_SCORE and self.sigmoid_stage != SIGMOID_AFTER_STITCH:
            raise StitchingControlError("a score-stage variant must sigmoid after stitching")


UNIFORM_PROBABILITY = StitchingVariantSpec(
    name="uniform_probability", stage=STAGE_PROBABILITY, weighting=WEIGHTING_UNIFORM, sigmoid_stage=SIGMOID_BEFORE_INTERPOLATION,
)
HANN_PROBABILITY = StitchingVariantSpec(
    name="hann_probability", stage=STAGE_PROBABILITY, weighting=WEIGHTING_HANN, sigmoid_stage=SIGMOID_BEFORE_INTERPOLATION,
)
UNIFORM_SCORE = StitchingVariantSpec(
    name="uniform_score", stage=STAGE_SCORE, weighting=WEIGHTING_UNIFORM, sigmoid_stage=SIGMOID_AFTER_STITCH,
)
HANN_SCORE = StitchingVariantSpec(
    name="hann_score", stage=STAGE_SCORE, weighting=WEIGHTING_HANN, sigmoid_stage=SIGMOID_AFTER_STITCH,
)
# Deterministic canonical order -- matches evaluation_identities/
# e12_stitching_control_suite.toml [variants].order exactly; validated
# against that identity by the caller, never redeclared as a second
# authority.
CANONICAL_VARIANTS: tuple[StitchingVariantSpec, ...] = (UNIFORM_PROBABILITY, HANN_PROBABILITY, UNIFORM_SCORE, HANN_SCORE)
CANONICAL_VARIANTS_BY_NAME: dict[str, StitchingVariantSpec] = {spec.name: spec for spec in CANONICAL_VARIANTS}


# ---------------------------------------------------------------------------
# Hann weight construction
# ---------------------------------------------------------------------------

_hann_weight_cache: dict[tuple[int, int, torch.dtype, str], torch.Tensor] = {}


def _hann_1d(n: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Pixel-centred Hann window: h_N(x) = 0.5 - 0.5*cos(2*pi*(x+0.5)/N)
    for x=0,...,N-1. Strictly positive for every finite N>=1 (the +0.5
    pixel-centre offset keeps the argument of cos strictly inside
    (0, 2*pi), so cos never reaches exactly 1 and h never reaches exactly
    0) -- no epsilon or floor is ever added. At N=1 this evaluates to
    exactly 1.0 (cos(pi) = -1), a natural consequence of the formula
    itself, never a special-cased branch."""
    if type(n) is not int or n <= 0:
        raise StitchingControlError(f"_hann_1d requires a positive exact int length, observed {n!r}")
    x = torch.arange(n, dtype=dtype, device=device)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * (x + 0.5) / n)


def build_stitch_weight(
    height: int, width: int, *, kind: str, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a ``[1, 1, height, width]`` per-pixel stitching weight map:
    all-ones for ``kind="uniform"``, or the separable pixel-centred Hann
    window ``W(y,x) = h_H(y) * h_W(x)`` for ``kind="hann"``.

    Cached by ``(height, width, dtype, str(device))``; the returned tensor
    is always a fresh clone of the cached template, so callers can never
    mutate the cache through the returned value, and repeated calls are
    deterministic and cheap after the first."""
    if type(height) is not int or height <= 0 or type(width) is not int or width <= 0:
        raise StitchingControlError(f"build_stitch_weight requires positive exact int height/width, observed ({height!r}, {width!r})")
    if kind not in (WEIGHTING_UNIFORM, WEIGHTING_HANN):
        raise StitchingControlError(f"build_stitch_weight kind must be 'uniform' or 'hann', observed {kind!r}")

    device_obj = torch.device(device)
    key = (height, width, dtype, kind, str(device_obj))
    cached = _hann_weight_cache.get(key)
    if cached is None:
        if kind == WEIGHTING_UNIFORM:
            weight = torch.ones((1, 1, height, width), dtype=dtype, device=device_obj)
        else:
            h_rows = _hann_1d(height, dtype=dtype, device=device_obj)
            h_cols = _hann_1d(width, dtype=dtype, device=device_obj)
            weight = (h_rows[:, None] * h_cols[None, :]).reshape(1, 1, height, width)
            if not torch.all(weight > 0):
                raise StitchingControlError("constructed Hann weight is not strictly positive everywhere -- refusing to use it")
        weight.requires_grad_(False)
        cached = weight
        _hann_weight_cache[key] = cached
    # Always return a fresh clone: the cache's own storage is never handed
    # to a caller, so no caller can mutate a future lookup's result.
    return cached.clone()


# ---------------------------------------------------------------------------
# Per-variant accumulator
# ---------------------------------------------------------------------------


class StitchAccumulator:
    """One variant's independent FP32 numerator/denominator accumulator
    over an image's canonical window set. Never shares storage with any
    other accumulator; never mutates a source window tensor in place;
    accumulates every window's contribution in canonical row-major crop
    order so summation order is reproducible; rejects duplicate/missing/
    out-of-order windows and any non-finite contribution; finalizes
    exactly once."""

    def __init__(self, spec: StitchingVariantSpec, *, class_count: int, image_hw: tuple[int, int], expected_window_count: int, device: torch.device | str, dtype: torch.dtype = torch.float32):
        if type(class_count) is not int or class_count <= 0:
            raise StitchingControlError("StitchAccumulator requires a positive exact int class_count")
        h_img, w_img = image_hw
        if type(h_img) is not int or h_img <= 0 or type(w_img) is not int or w_img <= 0:
            raise StitchingControlError("StitchAccumulator requires a positive exact int image_hw")
        if type(expected_window_count) is not int or expected_window_count <= 0:
            raise StitchingControlError("StitchAccumulator requires a positive exact int expected_window_count")
        self.spec = spec
        self.class_count = class_count
        self.image_hw = (h_img, w_img)
        self.expected_window_count = expected_window_count
        self.device = torch.device(device)
        self.dtype = dtype
        self._numerator = torch.zeros((1, class_count, h_img, w_img), dtype=dtype, device=self.device)
        self._denominator = torch.zeros((1, 1, h_img, w_img), dtype=dtype, device=self.device)
        self._seen_indices: list[int] = []
        self._finalized = False
        self._result: torch.Tensor | None = None

    def add_window(self, window: Any, crop: torch.Tensor, *, weight: torch.Tensor) -> None:
        """Accumulate one window's already-staged crop tensor (probability
        or raw-score, depending on ``self.spec.stage`` -- the caller is
        responsible for supplying the correctly-staged crop; this method
        performs no sigmoid/interpolation itself)."""
        if self._finalized:
            raise StitchingControlError(f"{self.spec.name}: cannot add_window after finalize() has been called")
        index = int(window.index)
        if self._seen_indices and index != self._seen_indices[-1] + 1:
            if index in self._seen_indices:
                raise StitchingControlError(f"{self.spec.name}: duplicate crop insertion for window index {index}")
            raise StitchingControlError(
                f"{self.spec.name}: window index {index} is out of canonical row-major order "
                f"(expected {self._seen_indices[-1] + 1})"
            )
        if not self._seen_indices and index != 0:
            raise StitchingControlError(f"{self.spec.name}: first window must have index 0, observed {index}")
        if not torch.is_tensor(crop) or crop.ndim != 4 or crop.shape[0] != 1 or crop.shape[1] != self.class_count:
            raise StitchingControlError(f"{self.spec.name}: crop must have shape [1, {self.class_count}, h, w], observed {tuple(crop.shape) if torch.is_tensor(crop) else type(crop)!r}")
        if crop.device != self.device:
            raise StitchingControlError(f"{self.spec.name}: crop device {crop.device} disagrees with accumulator device {self.device}")
        if crop.dtype != self.dtype:
            raise StitchingControlError(f"{self.spec.name}: crop dtype {crop.dtype} disagrees with accumulator dtype {self.dtype}")
        if not torch.isfinite(crop).all():
            raise StitchingControlError(f"{self.spec.name}: crop contains a non-finite value at window index {index}")
        if not torch.is_tensor(weight) or weight.shape[-2:] != crop.shape[-2:]:
            raise StitchingControlError(f"{self.spec.name}: weight spatial shape disagrees with crop shape")
        if not torch.isfinite(weight).all() or not torch.all(weight > 0):
            raise StitchingControlError(f"{self.spec.name}: weight must be finite and strictly positive everywhere")

        accum_rows, accum_cols = window.accumulation_slice
        weighted_crop = crop * weight  # never in-place: builds a new tensor, source crop untouched
        self._numerator[:, :, accum_rows, accum_cols] += weighted_crop
        self._denominator[:, :, accum_rows, accum_cols] += weight
        self._seen_indices.append(index)

    def finalize(self) -> torch.Tensor:
        """Divide numerator by denominator exactly once, apply the
        variant's own sigmoid-after-stitch contract when applicable, and
        freeze the accumulator against further ``add_window`` calls."""
        if self._finalized:
            raise StitchingControlError(f"{self.spec.name}: finalize() called twice")
        if len(self._seen_indices) != self.expected_window_count:
            raise StitchingControlError(
                f"{self.spec.name}: finalize() called with {len(self._seen_indices)} windows accumulated, "
                f"expected exactly {self.expected_window_count}"
            )
        if not torch.isfinite(self._denominator).all() or not torch.all(self._denominator > 0):
            raise StitchingControlError(f"{self.spec.name}: denominator must be finite and strictly positive at every pixel before dividing")
        stitched = self._numerator / self._denominator
        if self.spec.sigmoid_stage == SIGMOID_AFTER_STITCH:
            stitched = torch.sigmoid(stitched)
        if not torch.isfinite(stitched).all():
            raise StitchingControlError(f"{self.spec.name}: finalized stitched output contains a non-finite value")
        self._finalized = True
        self._result = stitched
        return stitched

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def windows_accumulated(self) -> int:
        return len(self._seen_indices)


# ---------------------------------------------------------------------------
# Multi-variant orchestration for one image
# ---------------------------------------------------------------------------


class MultiVariantStitchingSuite:
    """Owns one independent :class:`StitchAccumulator` per configured
    variant for a single image, and routes each window's two staged crops
    (a shared sigmoid+interpolated probability crop for all
    probability-stage variants, a shared interpolated-raw-score crop for
    all score-stage variants -- each computed at most once per window by
    the caller) to the variants that consume that stage."""

    def __init__(
        self, *, variants: tuple[StitchingVariantSpec, ...], class_count: int, image_hw: tuple[int, int],
        expected_window_count: int, device: torch.device | str, dtype: torch.dtype = torch.float32,
    ):
        if not variants:
            raise StitchingControlError("MultiVariantStitchingSuite requires at least one variant")
        names = [v.name for v in variants]
        if len(set(names)) != len(names):
            raise StitchingControlError("MultiVariantStitchingSuite variant names must be unique")
        self.variants = tuple(variants)
        self.accumulators: dict[str, StitchAccumulator] = {
            v.name: StitchAccumulator(
                v, class_count=class_count, image_hw=image_hw, expected_window_count=expected_window_count,
                device=device, dtype=dtype,
            )
            for v in variants
        }
        self.image_hw = image_hw
        self.class_count = class_count
        self.expected_window_count = expected_window_count
        self._finalized = False

    def add_window(
        self, window: Any, *, probability_crop: torch.Tensor | None, score_crop: torch.Tensor | None,
        hann_weight_probability: torch.Tensor | None = None, hann_weight_score: torch.Tensor | None = None,
    ) -> None:
        if self._finalized:
            raise StitchingControlError("cannot add_window after the suite has been finalized")
        needs_probability = any(v.stage == STAGE_PROBABILITY for v in self.variants)
        needs_score = any(v.stage == STAGE_SCORE for v in self.variants)
        if needs_probability and probability_crop is None:
            raise StitchingControlError("probability_crop is required for this suite's configured variants")
        if needs_score and score_crop is None:
            raise StitchingControlError("score_crop is required for this suite's configured variants")

        for spec in self.variants:
            accumulator = self.accumulators[spec.name]
            crop = probability_crop if spec.stage == STAGE_PROBABILITY else score_crop
            h, w = crop.shape[-2:]
            if spec.weighting == WEIGHTING_UNIFORM:
                weight = build_stitch_weight(h, w, kind="uniform", dtype=crop.dtype, device=crop.device)
            else:
                supplied = hann_weight_probability if spec.stage == STAGE_PROBABILITY else hann_weight_score
                weight = supplied if supplied is not None else build_stitch_weight(h, w, kind="hann", dtype=crop.dtype, device=crop.device)
            accumulator.add_window(window, crop, weight=weight)

    def finalize(self) -> dict[str, torch.Tensor]:
        if self._finalized:
            raise StitchingControlError("finalize() called twice on this suite")
        results = {name: accumulator.finalize() for name, accumulator in self.accumulators.items()}
        self._finalized = True
        return results

    @property
    def finalized(self) -> bool:
        return self._finalized


# ---------------------------------------------------------------------------
# Shared per-window crop staging
# ---------------------------------------------------------------------------


def interpolate_raw_scores(patch_scores: torch.Tensor, grid_hw: tuple[int, int], output_hw: tuple[int, int]) -> torch.Tensor:
    """Reshape+interpolate raw (pre-sigmoid) finite-step patch scores to
    crop pixels, mirroring ``DINOText.masks_from_patch_scores``'s own
    reshape/permute/interpolate steps exactly (same
    ``align_corners=True`` convention, matching
    ``evaluation_identities/e12_stitching_control_suite.toml``
    ``[geometry].align_corners``) -- but WITHOUT that method's internal
    sigmoid, since the score-space variants must interpolate before any
    sigmoid is ever applied. This is the one piece of unavoidable
    duplication the score-space variants require (sigmoid must be
    excludable), never a re-derivation of the interpolation formula
    itself."""
    if not torch.is_tensor(patch_scores) or patch_scores.ndim != 3:
        raise StitchingControlError("interpolate_raw_scores requires patch_scores of shape [B,N,C]")
    if (
        not isinstance(grid_hw, tuple) or len(grid_hw) != 2
        or any(type(v) is not int or v <= 0 for v in grid_hw)
        or grid_hw[0] * grid_hw[1] != patch_scores.shape[1]
    ):
        raise StitchingControlError("interpolate_raw_scores: patch score count does not match grid_hw")
    batch, _, classes = patch_scores.shape
    simmap = patch_scores.reshape(batch, *grid_hw, classes).permute(0, 3, 1, 2)
    return F.interpolate(simmap, tuple(output_hw), mode="bilinear", align_corners=True)


# ---------------------------------------------------------------------------
# Shared-execution per-window / per-image orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StitchingControlWindowTelemetry:
    """Exact per-completed-window operation counts for the shared-execution
    contract: one shared backbone/snapshot pass, one shared k12 graph
    build, one shared finite-step propagation, and AT MOST two
    interpolation calls (one shared probability-space crop for
    uniform_probability/hann_probability, one shared score-space crop for
    uniform_score/hann_score) -- never four (never one per variant)."""

    backbone_snapshot_calls: int
    graph_builds: int
    propagation_calls: int
    probability_interpolation_calls: int
    score_interpolation_calls: int

    def __post_init__(self) -> None:
        fixed = {"backbone_snapshot_calls": 1, "graph_builds": 1, "propagation_calls": 1}
        for name, expected in fixed.items():
            observed = getattr(self, name)
            if observed != expected:
                raise StitchingControlError(f"StitchingControlWindowTelemetry.{name} must be exactly {expected}, observed {observed!r}")
        for name in ("probability_interpolation_calls", "score_interpolation_calls"):
            observed = getattr(self, name)
            if observed not in (0, 1):
                raise StitchingControlError(f"StitchingControlWindowTelemetry.{name} must be 0 or 1, observed {observed!r}")


def process_one_window_shared(
    inference: Any, image_tensor: torch.Tensor, window: Any, *, alpha: float, steps: int, affinity_power: float,
    need_probability: bool, need_score: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, StitchingControlWindowTelemetry]:
    """Run the shared (variant-independent) per-window flow exactly once:
    one backbone/snapshot pass, one canonical k12 top-12 graph build, one
    finite-step propagation -- reusing
    :func:`.graph.build_directed_topk_graph` and
    :func:`.finite_step_regime.finite_step_propagate` unmodified, never a
    second graph/propagation for any variant. Returns the shared
    probability-space crop (sigmoid-then-interpolate, via the unmodified
    production ``masks_from_patch_scores``) and/or the shared score-space
    crop (interpolate-then-would-be-sigmoid, via
    :func:`interpolate_raw_scores`), each computed at most once."""
    from .finite_step_regime import finite_step_propagate
    from .graph import build_directed_topk_graph

    crop_rows, crop_cols = window.crop_slice
    crop = image_tensor[:, :, crop_rows, crop_cols]

    snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
    s0 = snapshot.unary_scores[0]
    dino_features = snapshot.dino_features[0]
    grid_hw = snapshot.grid_hw

    graph12 = build_directed_topk_graph(dino_features, k=12, affinity_power=affinity_power)
    trace = finite_step_propagate(graph12, s0, alpha=alpha, steps=steps, snapshot_steps=(steps,))
    patch_scores = trace.snapshots[steps].unsqueeze(0)

    crop_hw = window.extent.as_tuple()
    probability_crop = inference.model.masks_from_patch_scores(patch_scores, grid_hw, crop_hw) if need_probability else None
    score_crop = interpolate_raw_scores(patch_scores, grid_hw, crop_hw) if need_score else None

    telemetry = StitchingControlWindowTelemetry(
        backbone_snapshot_calls=1, graph_builds=1, propagation_calls=1,
        probability_interpolation_calls=1 if need_probability else 0,
        score_interpolation_calls=1 if need_score else 0,
    )
    return probability_crop, score_crop, telemetry


def stitch_one_image_multi_variant(
    inference: Any, image_tensor: torch.Tensor, plan: Any, *,
    variants: tuple[StitchingVariantSpec, ...], class_count: int, alpha: float, steps: int, affinity_power: float,
) -> tuple[dict[str, torch.Tensor], int, tuple[StitchingControlWindowTelemetry, ...]]:
    """Process every window of ``plan`` in row-major flat-index order,
    running the shared per-window flow exactly once and feeding its
    output to every configured variant's independent accumulator, then
    finalizing all variants together. Mirrors
    ``matched_power_evaluator.stitch_one_image``'s structure exactly,
    generalized from two hardcoded variants to an arbitrary
    (deterministically-ordered) variant tuple."""
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise StitchingControlError("stitch_one_image_multi_variant requires a batched [1, C, H, W] image tensor")
    h_img, w_img = plan.image_size.as_tuple()
    if tuple(image_tensor.shape[-2:]) != (h_img, w_img):
        raise StitchingControlError("image_tensor spatial extent does not match plan.image_size")

    need_probability = any(v.stage == STAGE_PROBABILITY for v in variants)
    need_score = any(v.stage == STAGE_SCORE for v in variants)

    suite = MultiVariantStitchingSuite(
        variants=variants, class_count=class_count, image_hw=(h_img, w_img),
        expected_window_count=len(plan.windows), device=image_tensor.device, dtype=torch.float32,
    )

    telemetry: list[StitchingControlWindowTelemetry] = []
    for window in plan.windows:
        probability_crop, score_crop, window_telemetry = process_one_window_shared(
            inference, image_tensor, window, alpha=alpha, steps=steps, affinity_power=affinity_power,
            need_probability=need_probability, need_score=need_score,
        )
        suite.add_window(window, probability_crop=probability_crop, score_crop=score_crop)
        telemetry.append(window_telemetry)

    finalized = suite.finalize()
    return finalized, len(plan.windows), tuple(telemetry)


def finalize_prediction(stitched: torch.Tensor, img_meta: Any, *, align_corners: bool) -> torch.Tensor:
    """Canonical output crop/unpadding + rescale + final per-pixel argmax
    for one already-stitched variant tensor -- identical contract to
    ``matched_power_evaluator.finalize_prediction`` (never independently
    re-derived): crop to ``img_meta['img_shape']``, bilinear-interpolate
    to ``img_meta['ori_shape']`` using the caller-supplied (never
    hardcoded) ``align_corners``, then argmax. No second sigmoid/softmax
    here regardless of variant: probability-space variants are already
    probabilities, and score-space variants already had their one
    post-stitch sigmoid applied inside :meth:`StitchAccumulator.finalize`."""
    resize_h, resize_w = img_meta["img_shape"][:2]
    cropped = stitched[:, :, :resize_h, :resize_w]
    rescaled = F.interpolate(cropped, size=tuple(img_meta["ori_shape"][:2]), mode="bilinear", align_corners=align_corners)
    return rescaled.argmax(dim=1)


def finalize_stitched_outputs(suite: MultiVariantStitchingSuite) -> dict[str, torch.Tensor]:
    """Thin, explicit wrapper over :meth:`MultiVariantStitchingSuite.finalize`
    -- kept as a standalone function so a caller can finalize a suite
    without needing to know it is a method, matching the module's
    suggested conceptual API."""
    return suite.finalize()


__all__ = [
    "CANONICAL_VARIANTS",
    "CANONICAL_VARIANTS_BY_NAME",
    "HANN_PROBABILITY",
    "HANN_SCORE",
    "MultiVariantStitchingSuite",
    "SIGMOID_AFTER_STITCH",
    "SIGMOID_BEFORE_INTERPOLATION",
    "STAGE_PROBABILITY",
    "STAGE_SCORE",
    "StitchAccumulator",
    "StitchingControlError",
    "StitchingVariantSpec",
    "UNIFORM_PROBABILITY",
    "UNIFORM_SCORE",
    "WEIGHTING_HANN",
    "WEIGHTING_UNIFORM",
    "build_stitch_weight",
    "finalize_prediction",
    "finalize_stitched_outputs",
    "interpolate_raw_scores",
    "process_one_window_shared",
    "stitch_one_image_multi_variant",
    "StitchingControlWindowTelemetry",
]
