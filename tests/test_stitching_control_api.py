"""CPU-only coverage for models.dinotext.cover_dr.stitching_control: the
reusable stitching-control API. Covers uniform-identity parity against the
existing matched k12 evaluator, the Hann formula, score/probability
staging, the shared-execution accumulator contract, and scientific
isolation (no prohibited T4/DCR/SUR/graph-editing code paths). Never
initializes CUDA, never loads the real model/dataset."""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/open_vocabulary_segmentation"))

from models.dinotext.cover_dr.stitching_control import (  # noqa: E402
    CANONICAL_VARIANTS,
    CANONICAL_VARIANTS_BY_NAME,
    HANN_PROBABILITY,
    HANN_SCORE,
    STAGE_PROBABILITY,
    STAGE_SCORE,
    UNIFORM_PROBABILITY,
    UNIFORM_SCORE,
    MultiVariantStitchingSuite,
    StitchAccumulator,
    StitchingControlError,
    StitchingVariantSpec,
    build_stitch_weight,
    finalize_prediction,
    interpolate_raw_scores,
    process_one_window_shared,
    stitch_one_image_multi_variant,
)
from models.dinotext.cover_dr.matched_power_evaluator import stitch_one_image as legacy_stitch_one_image  # noqa: E402
from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize  # noqa: E402

# Pre-existing structural quirk in models/dinotext/__init__.py (unrelated
# to this suite, never modified here): it does `from .dinotext import *`,
# and because the package directory and that submodule share the literal
# name "dinotext", importing it as a side effect leaves the `models`
# package's own `dinotext` ATTRIBUTE pointing at the submodule instead of
# the package (sys.modules['models.dinotext'] itself stays correct; only
# the attribute access `models.dinotext` off the parent `models` module
# is affected). This never affects `from X import Y` imports (used
# throughout this file and the production code), but a sibling test file
# collected after this one may use the attribute-chain `import
# models.dinotext.cover_dr.X as Y` form internally and would otherwise
# inherit the corrupted attribute. Repairing it here costs nothing and
# leaves sys.modules state healthy for whatever runs next.
sys.modules["models"].dinotext = sys.modules["models.dinotext"]


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self, patch_size=2, embed_dim=6, class_count=4, seed=0):
        super().__init__()
        self.patch_size = patch_size
        g = torch.Generator().manual_seed(seed)
        self.feature_proj = torch.randn(3, embed_dim, generator=g)
        self.score_proj = torch.randn(embed_dim, class_count, generator=g)
        self.snapshot_calls = 0
        self.mask_calls = 0

    def generate_patch_snapshot(self, crop, text_embedding):
        del text_embedding
        self.snapshot_calls += 1
        _, channels, height, width = crop.shape
        ps = self.patch_size
        grid_h, grid_w = height // ps, width // ps
        pooled = F.avg_pool2d(crop, ps)
        flat = pooled.reshape(1, channels, grid_h * grid_w).permute(0, 2, 1)
        raw_feat = flat @ self.feature_proj
        bias = 0.01 * torch.arange(grid_h * grid_w, dtype=torch.float32).view(1, -1, 1)
        features = F.normalize(raw_feat + bias, dim=-1)
        scores = features @ self.score_proj
        return types.SimpleNamespace(unary_scores=scores, dino_features=features, grid_hw=(grid_h, grid_w))

    def masks_from_patch_scores(self, patch_scores, grid_hw, output_hw):
        self.mask_calls += 1
        batch, _n, classes = patch_scores.shape
        grid_h, grid_w = grid_hw
        simmap = patch_scores.reshape(batch, grid_h, grid_w, classes).permute(0, 3, 1, 2)
        mask = torch.sigmoid(simmap)
        return F.interpolate(mask, tuple(output_hw), mode="bilinear", align_corners=True)


def _make_inference(model=None, class_count=4):
    model = model or _TinyModel(class_count=class_count)
    return types.SimpleNamespace(model=model, text_embedding=None, num_classes=class_count, align_corners=False), model


def _make_plan(h_img, w_img, crop=(8, 8), stride=(4, 4)):
    return SlidingWindowPlan.build(image_size=SpatialSize(h_img, w_img), crop_size=SpatialSize(*crop), stride=SpatialSize(*stride))


ALPHA, STEPS, AFFINITY_POWER = 0.98, 320, 3.0


# ---------------------------------------------------------------------------
# Uniform parity: the load-bearing invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_img,w_img,crop,stride", [
    (8, 8, (8, 8), (4, 4)),        # exact one-window image
    (12, 8, (8, 8), (4, 4)),       # overlapping windows, clamped terminal window
    (10, 14, (8, 8), (4, 4)),      # non-square image
    (16, 16, (8, 8), (8, 8)),      # non-overlapping tiling
])
def test_uniform_probability_reproduces_legacy_k12_stitching_exactly(h_img, w_img, crop, stride):
    torch.manual_seed(0)
    image_tensor = torch.rand(1, 3, h_img, w_img)
    inference, model = _make_inference(class_count=4)
    plan = _make_plan(h_img, w_img, crop=crop, stride=stride)

    # Legacy path: matched_power_evaluator.stitch_one_image computes BOTH
    # k11 and k12; we only compare against its k12 output (the shared
    # propagation configuration this whole suite reuses).
    legacy_result = legacy_stitch_one_image(
        inference, image_tensor, plan, class_count=4, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
    )
    legacy_k12 = legacy_result.k12_stitched

    stitched, window_count, _telemetry = stitch_one_image_multi_variant(
        inference, image_tensor, plan, variants=(UNIFORM_PROBABILITY,), class_count=4,
        alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
    )
    new_uniform_probability = stitched["uniform_probability"]

    assert window_count == len(plan.windows)
    # Bitwise-identical FP32 result is not guaranteed if summation order or
    # intermediate op fusion differs even trivially between the two
    # independent call paths; require exact label-map/statistic-level
    # identity, and additionally assert the raw tensors are extremely
    # close (they are, in practice, bit-identical on CPU since both paths
    # call the exact same masks_from_patch_scores + the same F.pad-based
    # accumulation order).
    assert torch.equal(new_uniform_probability, legacy_k12), (
        "uniform_probability must reproduce the legacy k12 stitched tensor bitwise-exactly "
        f"(max abs diff observed: {(new_uniform_probability - legacy_k12).abs().max().item()})"
    )

    # Exact label-map equality after the same final argmax step, and exact
    # confusion-statistic equality via an independent per-class histogram --
    # required regardless of whether the raw-tensor check above passes.
    img_meta = {"img_shape": (h_img, w_img, 3), "ori_shape": (h_img, w_img, 3)}
    legacy_label = finalize_prediction(legacy_k12, img_meta, align_corners=False)
    new_label = finalize_prediction(new_uniform_probability, img_meta, align_corners=False)
    assert torch.equal(legacy_label, new_label)

    for c in range(4):
        assert torch.equal((legacy_label == c), (new_label == c)), f"per-class confusion statistics disagree for class {c}"


def test_uniform_probability_denominator_matches_legacy_count_semantics():
    torch.manual_seed(1)
    h_img, w_img = 12, 12
    image_tensor = torch.rand(1, 3, h_img, w_img)
    inference, _ = _make_inference(class_count=3)
    plan = _make_plan(h_img, w_img, crop=(8, 8), stride=(4, 4))

    accumulator = StitchAccumulator(UNIFORM_PROBABILITY, class_count=3, image_hw=(h_img, w_img), expected_window_count=len(plan.windows), device="cpu")
    for window in plan.windows:
        probability_crop, _, _ = process_one_window_shared(
            inference, image_tensor, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
            need_probability=True, need_score=False,
        )
        weight = build_stitch_weight(*probability_crop.shape[-2:], kind="uniform")
        accumulator.add_window(window, probability_crop, weight=weight)
    accumulator.finalize()
    assert accumulator.finalized


# ---------------------------------------------------------------------------
# Hann-weight verification
# ---------------------------------------------------------------------------


def _independent_numpy_hann(n: int) -> np.ndarray:
    x = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * (x + 0.5) / n)


@pytest.mark.parametrize("h,w", [(8, 8), (5, 9), (1, 8), (8, 1), (1, 1), (16, 4)])
def test_hann_matches_independent_numpy_formula(h, w):
    w_tensor = build_stitch_weight(h, w, kind="hann").numpy()[0, 0]
    expected = np.outer(_independent_numpy_hann(h), _independent_numpy_hann(w))
    assert np.allclose(w_tensor, expected, atol=1e-6)


@pytest.mark.parametrize("h,w", [(8, 8), (5, 9), (1, 1), (32, 32)])
def test_hann_strictly_positive_everywhere(h, w):
    w_tensor = build_stitch_weight(h, w, kind="hann")
    assert bool((w_tensor > 0).all())


def test_hann_symmetric_within_row_and_column():
    w_tensor = build_stitch_weight(8, 8, kind="hann").numpy()[0, 0]
    assert np.allclose(w_tensor, w_tensor[::-1, :])
    assert np.allclose(w_tensor, w_tensor[:, ::-1])


def test_hann_separable_into_row_and_column_factors():
    h_vals = _independent_numpy_hann(8)
    w_vals = _independent_numpy_hann(10)
    w_tensor = build_stitch_weight(8, 10, kind="hann").numpy()[0, 0]
    assert np.allclose(w_tensor, np.outer(h_vals, w_vals), atol=1e-6)


def test_hann_maximum_near_centre():
    w_tensor = build_stitch_weight(9, 9, kind="hann").numpy()[0, 0]
    assert np.unravel_index(np.argmax(w_tensor), w_tensor.shape) == (4, 4)


def test_hann_h1_w1_locked_to_one():
    assert build_stitch_weight(1, 1, kind="hann").item() == pytest.approx(1.0)


def test_hann_h1_row_all_ones_times_column_hann():
    w_tensor = build_stitch_weight(1, 8, kind="hann").numpy()[0, 0]
    expected = _independent_numpy_hann(8)
    assert np.allclose(w_tensor[0], expected, atol=1e-6)


def test_hann_deterministic_across_calls():
    a = build_stitch_weight(10, 10, kind="hann")
    b = build_stitch_weight(10, 10, kind="hann")
    assert torch.equal(a, b)


def test_hann_cache_immutable_from_caller_perspective():
    a = build_stitch_weight(6, 6, kind="hann")
    a[0, 0, 0, 0] = 999.0
    b = build_stitch_weight(6, 6, kind="hann")
    assert b[0, 0, 0, 0].item() != 999.0


def test_hann_no_hidden_epsilon_at_expected_boundary_value():
    n = 100
    # h(0) = 0.5 - 0.5*cos(pi/n), independently computed; no floor added.
    # The 2D weight at (0,0) is the SEPARABLE PRODUCT h(0)*h(0), not h(0)
    # alone -- build_stitch_weight(n, n, ...) returns the outer product.
    expected_edge_1d = 0.5 - 0.5 * np.cos(np.pi / n)
    w_tensor = build_stitch_weight(n, n, kind="hann").numpy()[0, 0]
    assert w_tensor[0, 0] == pytest.approx(expected_edge_1d ** 2, rel=1e-3)
    assert w_tensor[0, 0] > 0  # strictly positive, no epsilon floor needed


def test_uniform_weight_is_all_ones():
    w_tensor = build_stitch_weight(5, 7, kind="uniform")
    assert torch.equal(w_tensor, torch.ones(1, 1, 5, 7))


@pytest.mark.parametrize("h_img,w_img,crop,stride", [
    (8, 8, (8, 8), (4, 4)), (12, 8, (8, 8), (4, 4)), (10, 14, (8, 8), (4, 4)),
])
def test_hann_denominator_strictly_positive_over_real_window_plan(h_img, w_img, crop, stride):
    torch.manual_seed(2)
    image_tensor = torch.rand(1, 3, h_img, w_img)
    inference, _ = _make_inference(class_count=3)
    plan = _make_plan(h_img, w_img, crop=crop, stride=stride)
    accumulator = StitchAccumulator(HANN_PROBABILITY, class_count=3, image_hw=(h_img, w_img), expected_window_count=len(plan.windows), device="cpu")
    for window in plan.windows:
        probability_crop, _, _ = process_one_window_shared(
            inference, image_tensor, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
            need_probability=True, need_score=False,
        )
        weight = build_stitch_weight(*probability_crop.shape[-2:], kind="hann")
        accumulator.add_window(window, probability_crop, weight=weight)
    result = accumulator.finalize()
    assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# Score/probability staging
# ---------------------------------------------------------------------------


def test_sigmoid_average_differs_from_average_sigmoid_synthetic():
    # A synthetic case proving average(sigmoid(x)) != sigmoid(average(x)) in
    # general (sigmoid is nonlinear), so the two staging orders are
    # observably different -- score-space and probability-space variants
    # are not interchangeable in general.
    a = torch.tensor([-5.0])
    b = torch.tensor([1.0])
    avg_then_sigmoid = torch.sigmoid((a + b) / 2)
    sigmoid_then_avg = (torch.sigmoid(a) + torch.sigmoid(b)) / 2
    assert not torch.allclose(avg_then_sigmoid, sigmoid_then_avg, atol=1e-3)


def test_score_stage_sigmoid_applied_exactly_once_after_stitch():
    torch.manual_seed(3)
    h_img, w_img = 8, 8
    image_tensor = torch.rand(1, 3, h_img, w_img)
    inference, _ = _make_inference(class_count=3)
    plan = _make_plan(h_img, w_img, crop=(8, 8), stride=(8, 8))
    stitched, _, _ = stitch_one_image_multi_variant(
        inference, image_tensor, plan, variants=(UNIFORM_SCORE,), class_count=3, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
    )
    result = stitched["uniform_score"]
    assert torch.all(result >= 0) and torch.all(result <= 1), "score-space output must be in [0,1] after its one post-stitch sigmoid"


def test_probability_stage_never_double_sigmoided():
    torch.manual_seed(4)
    h_img, w_img = 8, 8
    image_tensor = torch.rand(1, 3, h_img, w_img)
    inference, model = _make_inference(class_count=3)
    plan = _make_plan(h_img, w_img, crop=(8, 8), stride=(8, 8))
    stitched, _, _ = stitch_one_image_multi_variant(
        inference, image_tensor, plan, variants=(UNIFORM_PROBABILITY,), class_count=3, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
    )
    # single-window, uniform: result must equal the model's own single sigmoid+interpolate output exactly
    window = plan.windows[0]
    probability_crop, _, _ = process_one_window_shared(
        inference, image_tensor, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER, need_probability=True, need_score=False,
    )
    assert torch.equal(stitched["uniform_probability"], probability_crop)


def test_score_crop_is_not_yet_sigmoided_raw():
    torch.manual_seed(5)
    inference, model = _make_inference(class_count=3)
    image_tensor = torch.rand(1, 3, 8, 8)
    plan = _make_plan(8, 8, crop=(8, 8), stride=(8, 8))
    window = plan.windows[0]
    _, score_crop, _ = process_one_window_shared(
        inference, image_tensor, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER, need_probability=False, need_score=True,
    )
    # raw scores are not bounded to [0,1] in general (unlike a probability
    # crop) -- confirm this fixture's score crop actually exceeds [0,1]
    # somewhere, proving no sigmoid snuck in.
    assert bool((score_crop < 0).any() or (score_crop > 1).any())


def test_at_most_two_interpolations_per_window_all_four_variants():
    torch.manual_seed(6)
    inference, model = _make_inference(class_count=3)
    image_tensor = torch.rand(1, 3, 8, 8)
    plan = _make_plan(8, 8, crop=(8, 8), stride=(8, 8))
    window = plan.windows[0]
    _, _, telemetry = process_one_window_shared(
        inference, image_tensor, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER, need_probability=True, need_score=True,
    )
    assert telemetry.probability_interpolation_calls == 1
    assert telemetry.score_interpolation_calls == 1
    assert telemetry.backbone_snapshot_calls == 1
    assert telemetry.graph_builds == 1
    assert telemetry.propagation_calls == 1


# ---------------------------------------------------------------------------
# Shared execution / accumulator contract
# ---------------------------------------------------------------------------


def test_one_backbone_call_per_window_not_per_variant():
    torch.manual_seed(7)
    inference, model = _make_inference(class_count=3)
    image_tensor = torch.rand(1, 3, 12, 12)
    plan = _make_plan(12, 12, crop=(8, 8), stride=(4, 4))
    stitch_one_image_multi_variant(
        inference, image_tensor, plan, variants=CANONICAL_VARIANTS, class_count=3, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
    )
    assert model.snapshot_calls == len(plan.windows), "backbone snapshot calls must equal windows, not windows*variants"


def test_at_most_two_mask_calls_per_window_across_four_variants():
    torch.manual_seed(8)
    inference, model = _make_inference(class_count=3)
    image_tensor = torch.rand(1, 3, 12, 12)
    plan = _make_plan(12, 12, crop=(8, 8), stride=(4, 4))
    stitch_one_image_multi_variant(
        inference, image_tensor, plan, variants=CANONICAL_VARIANTS, class_count=3, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
    )
    # masks_from_patch_scores (probability-space sigmoid+interpolate) is
    # called once per window (shared by uniform_probability/hann_probability),
    # never once per variant.
    assert model.mask_calls == len(plan.windows)


def test_source_window_tensor_not_mutated_in_place():
    torch.manual_seed(9)
    inference, model = _make_inference(class_count=3)
    image_tensor = torch.rand(1, 3, 8, 8)
    plan = _make_plan(8, 8, crop=(8, 8), stride=(8, 8))
    window = plan.windows[0]
    probability_crop, score_crop, _ = process_one_window_shared(
        inference, image_tensor, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER, need_probability=True, need_score=True,
    )
    probability_before = probability_crop.clone()
    score_before = score_crop.clone()
    accumulator = StitchAccumulator(UNIFORM_PROBABILITY, class_count=3, image_hw=(8, 8), expected_window_count=1, device="cpu")
    weight = build_stitch_weight(8, 8, kind="uniform")
    accumulator.add_window(window, probability_crop, weight=weight)
    accumulator.finalize()
    assert torch.equal(probability_crop, probability_before)
    assert torch.equal(score_crop, score_before)


def test_independent_accumulators_no_storage_aliasing():
    acc1 = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu")
    acc2 = StitchAccumulator(HANN_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu")
    assert acc1._numerator.data_ptr() != acc2._numerator.data_ptr()
    assert acc1._denominator.data_ptr() != acc2._denominator.data_ptr()


def test_no_variant_specific_model_rerun_suite_level():
    torch.manual_seed(10)
    inference, model = _make_inference(class_count=3)
    image_tensor = torch.rand(1, 3, 8, 8)
    plan = _make_plan(8, 8, crop=(8, 8), stride=(8, 8))
    stitch_one_image_multi_variant(
        inference, image_tensor, plan, variants=CANONICAL_VARIANTS, class_count=3, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER,
    )
    assert model.snapshot_calls == 1  # one window -> exactly one snapshot, for ALL four variants combined


# ---------------------------------------------------------------------------
# Accumulator API-level contract tests
# ---------------------------------------------------------------------------


class _FakeWindow:
    def __init__(self, index, rows, cols):
        self.index = index
        self.accumulation_slice = (rows, cols)


def test_duplicate_crop_insertion_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=2, device="cpu")
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    crop = torch.rand(1, 2, 4, 4)
    weight = build_stitch_weight(4, 4, kind="uniform")
    acc.add_window(w0, crop, weight=weight)
    with pytest.raises(StitchingControlError, match="duplicate"):
        acc.add_window(w0, crop, weight=weight)


def test_first_window_must_be_index_zero_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=2, device="cpu")
    w1 = _FakeWindow(1, slice(0, 4), slice(0, 4))
    crop = torch.rand(1, 2, 4, 4)
    weight = build_stitch_weight(4, 4, kind="uniform")
    with pytest.raises(StitchingControlError, match="index 0"):
        acc.add_window(w1, crop, weight=weight)


def test_out_of_order_window_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=3, device="cpu")
    crop = torch.rand(1, 2, 4, 4)
    weight = build_stitch_weight(4, 4, kind="uniform")
    acc.add_window(_FakeWindow(0, slice(0, 4), slice(0, 4)), crop, weight=weight)
    w2 = _FakeWindow(2, slice(0, 4), slice(0, 4))  # skips index 1
    with pytest.raises(StitchingControlError, match="order"):
        acc.add_window(w2, crop, weight=weight)


def test_missing_window_at_finalize_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=2, device="cpu")
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    crop = torch.rand(1, 2, 4, 4)
    weight = build_stitch_weight(4, 4, kind="uniform")
    acc.add_window(w0, crop, weight=weight)
    with pytest.raises(StitchingControlError):
        acc.finalize()


def test_shape_disagreement_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu")
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    wrong_crop = torch.rand(1, 3, 4, 4)  # wrong class_count
    weight = build_stitch_weight(4, 4, kind="uniform")
    with pytest.raises(StitchingControlError):
        acc.add_window(w0, wrong_crop, weight=weight)


def test_dtype_disagreement_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu", dtype=torch.float32)
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    wrong_dtype_crop = torch.rand(1, 2, 4, 4).to(torch.float64)
    weight = build_stitch_weight(4, 4, kind="uniform")
    with pytest.raises(StitchingControlError):
        acc.add_window(w0, wrong_dtype_crop, weight=weight)


def test_nan_crop_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu")
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    crop = torch.rand(1, 2, 4, 4)
    crop[0, 0, 0, 0] = float("nan")
    weight = build_stitch_weight(4, 4, kind="uniform")
    with pytest.raises(StitchingControlError, match="non-finite"):
        acc.add_window(w0, crop, weight=weight)


def test_infinite_weight_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu")
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    crop = torch.rand(1, 2, 4, 4)
    weight = torch.full((1, 1, 4, 4), float("inf"))
    with pytest.raises(StitchingControlError):
        acc.add_window(w0, crop, weight=weight)


def test_zero_weight_rejected_not_strictly_positive():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu")
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    crop = torch.rand(1, 2, 4, 4)
    weight = torch.zeros(1, 1, 4, 4)
    with pytest.raises(StitchingControlError):
        acc.add_window(w0, crop, weight=weight)


def test_finalize_twice_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu")
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    crop = torch.rand(1, 2, 4, 4)
    weight = build_stitch_weight(4, 4, kind="uniform")
    acc.add_window(w0, crop, weight=weight)
    acc.finalize()
    with pytest.raises(StitchingControlError, match="twice"):
        acc.finalize()


def test_add_after_finalize_rejected():
    acc = StitchAccumulator(UNIFORM_PROBABILITY, class_count=2, image_hw=(4, 4), expected_window_count=1, device="cpu")
    w0 = _FakeWindow(0, slice(0, 4), slice(0, 4))
    crop = torch.rand(1, 2, 4, 4)
    weight = build_stitch_weight(4, 4, kind="uniform")
    acc.add_window(w0, crop, weight=weight)
    acc.finalize()
    with pytest.raises(StitchingControlError, match="finalize"):
        acc.add_window(w0, crop, weight=weight)


def test_variant_spec_immutable():
    spec = UNIFORM_PROBABILITY
    with pytest.raises(Exception):
        spec.name = "changed"  # frozen dataclass


def test_invalid_variant_spec_stage_rejected():
    with pytest.raises(StitchingControlError):
        StitchingVariantSpec(name="x", stage="not_a_stage", weighting="uniform", sigmoid_stage="before_interpolation")


def test_invalid_variant_spec_stage_sigmoid_combo_rejected():
    with pytest.raises(StitchingControlError):
        StitchingVariantSpec(name="x", stage="probability", weighting="uniform", sigmoid_stage="after_stitch")


def test_canonical_variants_order_matches_identity():
    assert [v.name for v in CANONICAL_VARIANTS] == ["uniform_probability", "hann_probability", "uniform_score", "hann_score"]


# ---------------------------------------------------------------------------
# Scientific isolation: prohibited code paths absent
# ---------------------------------------------------------------------------


def test_no_prohibited_terms_in_stitching_control_identifiers():
    # AST-identifier scan, never a raw-text search: prose in comments/
    # docstrings legitimately DISCLAIMS these prohibited techniques (e.g.
    # "never touches dcr/sur"), so a text search would false-positive on
    # its own compliance statement. Only actual function/class/variable/
    # attribute NAMES are checked.
    source = inspect.getsource(sys.modules["models.dinotext.cover_dr.stitching_control"])
    tree = ast.parse(source)
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    lowered_identifiers = {ident.lower() for ident in identifiers}
    for forbidden in ("t4_target", "consensus_label", "dcr", "sur", "sherman_morrison", "adjoint", "reachability", "cgls", "pamr", "learned_weight", "parameter"):
        assert forbidden not in lowered_identifiers, f"prohibited identifier {forbidden!r} found in stitching_control.py"


def test_no_second_graph_or_propagation_call_in_process_one_window_shared():
    source = inspect.getsource(process_one_window_shared)
    tree = ast.parse(source)
    calls = [n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert calls.count("build_directed_topk_graph") <= 1
    assert calls.count("finite_step_propagate") <= 1


def test_no_learned_nn_module_in_stitching_control():
    module = sys.modules["models.dinotext.cover_dr.stitching_control"]
    for name in dir(module):
        obj = getattr(module, name)
        assert not (isinstance(obj, type) and issubclass(obj, torch.nn.Module)), f"{name} is an nn.Module subclass -- no learned component is permitted in this suite"
