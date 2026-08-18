"""Tests for the reusable sliding-window / patch-grid geometry module.

The production ``DINOTextSegInference.slide_inference`` loop is loaded via
file-path import with lightweight stand-ins for ``mmcv``/``utils`` (mirroring
``tests/test_rwr_integration.py``) so these tests never require mmcv, cv2,
or a CUDA device.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[1]
GEOMETRY_PATH = (
    ROOT
    / "src/open_vocabulary_segmentation/segmentation/evaluation/sliding_window_geometry.py"
)
SEGMENTATION_PATH = (
    ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py"
)
COVER_DR_PACKAGE_PATH = (
    ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr"
)


def _load_geometry_module():
    spec = importlib.util.spec_from_file_location(
        "sliding_window_geometry_under_test", GEOMETRY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


geometry = _load_geometry_module()
SpatialSize = geometry.SpatialSize
PixelCoordinate = geometry.PixelCoordinate
Rectangle = geometry.Rectangle
WindowGeometry = geometry.WindowGeometry
SlidingWindowPlan = geometry.SlidingWindowPlan
PatchGridSpec = geometry.PatchGridSpec
BilinearStencil = geometry.BilinearStencil
SlidingWindowGeometryError = geometry.SlidingWindowGeometryError
local_to_global = geometry.local_to_global
global_to_local = geometry.global_to_local
contains_point = geometry.contains_point
intersect_windows = geometry.intersect_windows
overlap_area = geometry.overlap_area
bilinear_stencil = geometry.bilinear_stencil
bilinear_stencil_from_doubled = geometry.bilinear_stencil_from_doubled


# ---------------------------------------------------------------------------
# Independent literal legacy reference (transcribed from
# DINOTextSegInference.slide_inference, not derived from the new module).
# ---------------------------------------------------------------------------


def legacy_windows(h_img, w_img, h_crop, w_crop, h_stride, w_stride):
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    windows = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            windows.append((y1, y2, x1, x2))
    return h_grids, w_grids, windows


def legacy_stitch(h_img, w_img, h_crop, w_crop, h_stride, w_stride, num_classes, crop_fn):
    """Literal transcription of the pre-refactor slide_inference accumulation."""
    preds = torch.zeros((1, num_classes, h_img, w_img), dtype=torch.float64)
    count_mat = torch.zeros((1, 1, h_img, w_img), dtype=torch.float64)
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_logits = crop_fn(y1, y2, x1, x2)
            preds += torch.nn.functional.pad(
                crop_logits, (x1, preds.shape[3] - x2, y1, preds.shape[2] - y2)
            )
            count_mat[:, :, y1:y2, x1:x2] += 1
    return preds, count_mat


def plan_stitch(h_img, w_img, h_crop, w_crop, h_stride, w_stride, num_classes, crop_fn):
    """The refactored plan-based accumulation, mirroring the new production code."""
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(h_img, w_img),
        crop_size=SpatialSize(h_crop, w_crop),
        stride=SpatialSize(h_stride, w_stride),
    )
    preds = torch.zeros((1, num_classes, h_img, w_img), dtype=torch.float64)
    count_mat = torch.zeros((1, 1, h_img, w_img), dtype=torch.float64)
    for window in plan.windows:
        crop_rows, crop_cols = window.crop_slice
        crop_logits = crop_fn(
            crop_rows.start, crop_rows.stop, crop_cols.start, crop_cols.stop
        )
        accum_rows, accum_cols = window.accumulation_slice
        preds += torch.nn.functional.pad(
            crop_logits,
            (
                accum_cols.start,
                preds.shape[3] - accum_cols.stop,
                accum_rows.start,
                preds.shape[2] - accum_rows.stop,
            ),
        )
        count_mat[:, :, accum_rows, accum_cols] += 1
    return preds, count_mat


# ---------------------------------------------------------------------------
# Legacy parity
# ---------------------------------------------------------------------------

FIXED_CASES = [
    # (H, W, Ch, Cw, Sh, Sw)
    (300, 300, 448, 448, 224, 224),      # image smaller than crop
    (448, 448, 448, 448, 224, 224),      # image equal to crop
    (449, 449, 448, 448, 224, 224),      # image one pixel larger
    (896, 896, 448, 448, 224, 224),      # exact stride multiples
    (1000, 1000, 448, 448, 224, 224),    # nonmultiple
    (2048, 448, 448, 448, 224, 224),     # extreme aspect ratio (tall/wide)
    (448, 5000, 448, 448, 224, 224),     # extreme aspect ratio
    (447, 447, 448, 448, 224, 224),      # image one pixel smaller than crop
    (672, 672, 448, 448, 224, 224),      # terminal one-pixel-shift-adjacent
    (673, 673, 448, 448, 224, 224),      # terminal window shifts by 1px
    (520, 520, 448, 448, 224, 224),      # COCO-Stuff-like dimensions
    (375, 500, 448, 448, 224, 224),      # COCO-like (H<crop, W>crop)
    (1, 1, 448, 448, 224, 224),          # degenerate 1x1 image
    (1, 1000, 1, 224, 1, 224),           # degenerate crop height of 1
]


@pytest.mark.parametrize("dims", FIXED_CASES)
def test_legacy_parity_fixed_cases(dims):
    h_img, w_img, h_crop, w_crop, h_stride, w_stride = dims
    grid_rows, grid_cols, legacy = legacy_windows(
        h_img, w_img, h_crop, w_crop, h_stride, w_stride
    )
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(h_img, w_img),
        crop_size=SpatialSize(h_crop, w_crop),
        stride=SpatialSize(h_stride, w_stride),
    )
    assert plan.grid_rows == grid_rows
    assert plan.grid_cols == grid_cols
    assert plan.window_count == len(legacy)
    for window, (y1, y2, x1, x2) in zip(plan.windows, legacy):
        assert (window.origin.row, window.end.row, window.origin.col, window.end.col) == (
            y1, y2, x1, x2,
        )
        crop_rows, crop_cols = window.crop_slice
        assert (crop_rows.start, crop_rows.stop, crop_cols.start, crop_cols.stop) == (
            y1, y2, x1, x2,
        )
        nominal_y = window.grid_row * h_stride
        nominal_x = window.grid_col * w_stride
        assert (window.nominal_origin.row, window.nominal_origin.col) == (nominal_y, nominal_x)
        assert window.clamped_vertical == (window.origin.row != nominal_y)
        assert window.clamped_horizontal == (window.origin.col != nominal_x)


def test_legacy_parity_randomized_500_combinations():
    rng = random.Random(20260818)
    count = 0
    for _ in range(500):
        h_img = rng.randint(1, 3000)
        w_img = rng.randint(1, 3000)
        h_crop = rng.randint(1, 600)
        w_crop = rng.randint(1, 600)
        h_stride = rng.randint(1, 600)
        w_stride = rng.randint(1, 600)
        grid_rows, grid_cols, legacy = legacy_windows(
            h_img, w_img, h_crop, w_crop, h_stride, w_stride
        )
        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(h_img, w_img),
            crop_size=SpatialSize(h_crop, w_crop),
            stride=SpatialSize(h_stride, w_stride),
        )
        assert plan.grid_rows == grid_rows
        assert plan.grid_cols == grid_cols
        assert plan.window_count == len(legacy)
        for window, (y1, y2, x1, x2) in zip(plan.windows, legacy):
            assert (
                window.origin.row, window.end.row, window.origin.col, window.end.col
            ) == (y1, y2, x1, x2)
        # row-major order: flat index strictly increasing, grid_row major
        for previous, current in zip(plan.windows, plan.windows[1:]):
            assert current.index == previous.index + 1
            assert (current.grid_row, current.grid_col) > (
                previous.grid_row, previous.grid_col
            )
        count += 1
    assert count == 500


def test_terminal_window_shifts_by_at_most_one_pixel_step_relationship():
    # A terminal window may start only one pixel after where an exact
    # multiple would have placed it (never more).
    for h_img in range(448, 900):
        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(h_img, 448),
            crop_size=SpatialSize(448, 448),
            stride=SpatialSize(224, 224),
        )
        last_row_window = plan.window_at(plan.grid_rows - 1, 0)
        if not last_row_window.clamped_vertical:
            continue
        nominal_next_start = last_row_window.nominal_origin.row
        actual_start = last_row_window.origin.row
        assert actual_start <= nominal_next_start
        assert actual_start >= nominal_next_start - 224  # back-shift bounded by stride


def test_no_terminal_window_deduplication_small_images():
    # Even when clamping makes two adjacent windows identical in geometry,
    # both must still be emitted (legacy loop never deduplicates).
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(10, 10),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    assert plan.window_count == 1  # only one grid cell exists here; see next case
    plan2 = SlidingWindowPlan.build(
        image_size=SpatialSize(230, 10),
        crop_size=SpatialSize(224, 224),
        stride=SpatialSize(224, 224),
    )
    # h_grids = max(230-224+224-1,0)//224+1 = max(229,0)//224+1 = 1+1 = 2
    assert plan2.grid_rows == 2
    assert plan2.window_count == 2
    first, second = plan2.window_at(0, 0), plan2.window_at(1, 0)
    assert first.origin.row == 0 and first.end.row == 224
    assert second.origin.row == 6 and second.end.row == 230  # back-shifted, overlaps first


def test_height_and_width_are_never_swapped():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(300, 900),
        crop_size=SpatialSize(200, 400),
        stride=SpatialSize(100, 300),
    )
    assert plan.image_size.height == 300 and plan.image_size.width == 900
    for window in plan.windows:
        assert 0 <= window.origin.row < window.end.row <= 300
        assert 0 <= window.origin.col < window.end.col <= 900
        assert window.extent.height <= 200
        assert window.extent.width <= 400


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_every_pixel_covered_at_least_once():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 837),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    coverage = plan.build_coverage_map()
    for row in range(1000):
        for col in range(837):
            assert coverage[row][col] >= 1


def test_dense_coverage_map_matches_independent_legacy_accumulation():
    h_img, w_img, h_crop, w_crop, h_stride, w_stride = 673, 900, 224, 224, 100, 150
    legacy_count = [[0] * w_img for _ in range(h_img)]
    _, _, legacy = legacy_windows(h_img, w_img, h_crop, w_crop, h_stride, w_stride)
    for (y1, y2, x1, x2) in legacy:
        for row in range(y1, y2):
            for col in range(x1, x2):
                legacy_count[row][col] += 1

    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(h_img, w_img),
        crop_size=SpatialSize(h_crop, w_crop),
        stride=SpatialSize(h_stride, w_stride),
    )
    dense = plan.build_coverage_map()
    assert list(map(list, dense)) == legacy_count
    # point-query coverage_count() must agree with the dense reference too.
    rng = random.Random(7)
    for _ in range(200):
        row = rng.randrange(h_img)
        col = rng.randrange(w_img)
        assert plan.coverage_count(row, col) == legacy_count[row][col]


def test_canonical_corner_edge_interior_counts():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    # top-left corner pixel is only ever covered by the single top-left window.
    assert plan.coverage_count(0, 0) == 1
    # bottom-right corner pixel is only covered by the single bottom-right window.
    assert plan.coverage_count(999, 999) == 1
    # a deep interior pixel well inside overlap bands should have coverage > 1.
    assert plan.coverage_count(224, 224) > 1


def test_windows_covering_point_are_row_major_ordered_and_consistent():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    indices = plan.windows_covering_point(224, 224)
    assert list(indices) == sorted(indices)
    assert len(indices) == plan.coverage_count(224, 224)
    for index in indices:
        assert contains_point(plan.window_by_index(index), PixelCoordinate(224, 224))


# ---------------------------------------------------------------------------
# Stitch parity (deterministic random crop tensors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims",
    [
        (1000, 1000, 448, 448, 224, 224),
        (673, 900, 224, 224, 100, 150),
        (449, 449, 448, 448, 224, 224),
        (300, 300, 448, 448, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stitch_parity_bitwise_identical(dims, dtype):
    h_img, w_img, h_crop, w_crop, h_stride, w_stride = dims
    num_classes = 3
    generator = torch.Generator().manual_seed(1234)
    # Deterministic per-crop tensor keyed only by its coordinates, so both
    # stitching implementations request bit-identical crop content whenever
    # they visit the same (y1, y2, x1, x2) window in the same order.
    cache = {}

    def crop_fn(y1, y2, x1, x2):
        key = (y1, y2, x1, x2)
        if key not in cache:
            cache[key] = torch.rand(
                (1, num_classes, y2 - y1, x2 - x1),
                generator=generator,
                dtype=torch.float32,
            ).to(dtype)
        return cache[key]

    legacy_preds, legacy_count = legacy_stitch(
        h_img, w_img, h_crop, w_crop, h_stride, w_stride, num_classes, crop_fn
    )
    plan_preds, plan_count = plan_stitch(
        h_img, w_img, h_crop, w_crop, h_stride, w_stride, num_classes, crop_fn
    )
    assert torch.equal(legacy_count, plan_count)
    assert torch.equal(legacy_preds, plan_preds)
    # Also compare the final divided result exactly, matching production.
    legacy_result = legacy_preds / legacy_count
    plan_result = plan_preds / plan_count
    assert torch.equal(legacy_result, plan_result)


# ---------------------------------------------------------------------------
# Patch geometry
# ---------------------------------------------------------------------------


CANONICAL_SPEC = PatchGridSpec(
    crop_size=SpatialSize(448, 448),
    patch_size=SpatialSize(14, 14),
    grid_size=SpatialSize(32, 32),
    align_corners=True,
)


def test_patch_footprint_bounds():
    footprint = CANONICAL_SPEC.patch_footprint_local(0, 0)
    assert (footprint.origin.row, footprint.origin.col) == (0, 0)
    assert (footprint.end.row, footprint.end.col) == (14, 14)
    footprint_last = CANONICAL_SPEC.patch_footprint_local(31, 31)
    assert (footprint_last.origin.row, footprint_last.origin.col) == (434, 434)
    assert (footprint_last.end.row, footprint_last.end.col) == (448, 448)


def test_patch_footprint_global_matches_window_origin_offset():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    window = plan.window_at(1, 1)
    local = CANONICAL_SPEC.patch_footprint_local(3, 5)
    glob = CANONICAL_SPEC.patch_footprint_global(window, 3, 5)
    assert glob.origin.row == local.origin.row + window.origin.row
    assert glob.origin.col == local.origin.col + window.origin.col
    assert glob.end.row == local.end.row + window.origin.row
    assert glob.end.col == local.end.col + window.origin.col


def test_doubled_coordinate_centers_are_exact_half_integers():
    # Patch [0:14) has center 6.5 -> doubled = 13
    assert CANONICAL_SPEC.patch_center_doubled_local(0, 0) == (13, 13)
    # Patch [14:28) has center 20.5 -> doubled = 41
    assert CANONICAL_SPEC.patch_center_doubled_local(1, 1) == (41, 41)


def test_flatten_unflatten_node_round_trip():
    for patch_row in range(0, 32, 3):
        for patch_col in range(0, 32, 5):
            flat = CANONICAL_SPEC.flatten_node(patch_row, patch_col)
            assert CANONICAL_SPEC.unflatten_node(flat) == (patch_row, patch_col)
    assert CANONICAL_SPEC.flatten_node(0, 0) == 0
    assert CANONICAL_SPEC.flatten_node(31, 31) == 32 * 32 - 1


def test_canonical_224_shift_equals_16_node_offset():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    window_a = plan.window_at(0, 0)
    window_b = plan.window_at(1, 0)
    assert not window_a.clamped_vertical and not window_b.clamped_vertical
    assert window_b.origin.row - window_a.origin.row == 224
    for patch_row in range(16, 32):
        mapped = CANONICAL_SPEC.native_patch_node(window_a, patch_row, 10, window_b)
        assert mapped == (patch_row - 16, 10)
    # And the reverse direction.
    for patch_row in range(0, 16):
        mapped_back = CANONICAL_SPEC.native_patch_node(window_b, patch_row, 10, window_a)
        assert mapped_back == (patch_row + 16, 10)


def test_aligned_terminal_window():
    # 896x896 with crop 448/stride 224 -> last window origin (448,448) which
    # IS a multiple of both stride and patch size (14), so it remains aligned
    # even though it is the terminal window (grid exactly divides the image).
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(896, 896),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    last = plan.window_at(plan.grid_rows - 1, plan.grid_cols - 1)
    assert last.origin.row == 448 and last.origin.col == 448
    assert last.is_lattice_aligned(SpatialSize(14, 14))


def test_unaligned_clamped_terminal_window():
    # 1000x1000: the terminal row/col windows back-shift to origin 552,
    # which is NOT a multiple of the 14px patch lattice (552 / 14 = 39.43).
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    last = plan.window_at(plan.grid_rows - 1, plan.grid_cols - 1)
    assert last.clamped_vertical and last.clamped_horizontal
    assert last.origin.row == 552 and last.origin.col == 552
    assert not last.is_lattice_aligned(SpatialSize(14, 14))


def test_clamped_windows_are_not_all_unconditionally_unaligned():
    # A clamped window CAN still be lattice-aligned if its back-shifted
    # origin happens to land on a patch multiple; construct such a case
    # directly: image height 434 (=31*14) with crop 448 clamps to origin
    # max(434-448,0)=0, i.e. aligned (origin 0 is trivially a multiple).
    # Use a case where clamping produces a nonzero but still-aligned origin:
    # H=448+140=588, crop=448, stride=280 -> grids: max(588-448+280-1,0)//280+1
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(588, 448),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(280, 224),
    )
    last_row = plan.window_at(plan.grid_rows - 1, 0)
    assert last_row.clamped_vertical
    assert last_row.origin.row == 588 - 448  # = 140, a multiple of 14
    assert last_row.is_lattice_aligned(SpatialSize(14, 14))


def test_native_mapping_symmetry():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    window_a = plan.window_at(0, 0)
    window_b = plan.window_at(1, 1)
    forward = CANONICAL_SPEC.native_patch_node(window_a, 20, 20, window_b)
    if forward is not None:
        backward = CANONICAL_SPEC.native_patch_node(window_b, forward[0], forward[1], window_a)
        assert backward == (20, 20)


def test_native_mapping_out_of_range_returns_none():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    window_a = plan.window_at(0, 0)
    window_b = plan.window_at(3, 3)  # far away; most nodes map out of range
    mapped = CANONICAL_SPEC.native_patch_node(window_a, 0, 0, window_b)
    assert mapped is None


def test_native_mapping_undefined_for_non_multiple_displacement():
    unaligned_spec = PatchGridSpec(
        crop_size=SpatialSize(450, 450),
        patch_size=SpatialSize(15, 15),
        grid_size=SpatialSize(30, 30),
        align_corners=True,
    )
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(450, 450),
        stride=SpatialSize(224, 224),  # not a multiple of patch size 15
    )
    window_a = plan.window_at(0, 0)
    window_b = plan.window_at(1, 0)
    assert (window_b.origin.row - window_a.origin.row) % 15 != 0
    mapped = unaligned_spec.native_patch_node(window_a, 10, 10, window_b)
    assert mapped is None


# ---------------------------------------------------------------------------
# Direct score-sampling geometry (bilinear stencil)
# ---------------------------------------------------------------------------


def test_stencil_weights_are_finite_nonnegative_and_sum_to_one():
    rng = random.Random(99)
    for _ in range(150):
        row = rng.randrange(0, 448)
        col = rng.randrange(0, 448)
        stencil = bilinear_stencil(
            PixelCoordinate(row, col),
            SpatialSize(448, 448),
            SpatialSize(32, 32),
            align_corners=True,
        )
        weights = stencil.weights()
        assert all(w == w for w in weights)  # not NaN
        assert all(w >= -1e-12 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-9
    # Half-integer patch-center coordinates, exercised via the doubled-integer
    # entry point (no float coordinate ever constructed by the caller).
    for _ in range(150):
        doubled_row = rng.randrange(0, 2 * 447 + 1)
        doubled_col = rng.randrange(0, 2 * 447 + 1)
        stencil = bilinear_stencil_from_doubled(
            doubled_row, doubled_col,
            SpatialSize(448, 448),
            SpatialSize(32, 32),
            align_corners=True,
        )
        weights = stencil.weights()
        assert all(w == w for w in weights)
        assert all(w >= -1e-12 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-9


def test_stencil_from_doubled_agrees_with_integer_entry_point_on_whole_pixels():
    rng = random.Random(11)
    for _ in range(100):
        row = rng.randrange(0, 448)
        col = rng.randrange(0, 448)
        integer_stencil = bilinear_stencil(
            PixelCoordinate(row, col), SpatialSize(448, 448), SpatialSize(32, 32),
            align_corners=True,
        )
        doubled_stencil = bilinear_stencil_from_doubled(
            2 * row, 2 * col, SpatialSize(448, 448), SpatialSize(32, 32),
            align_corners=True,
        )
        assert integer_stencil == doubled_stencil


def test_stencil_from_doubled_uses_patch_center_representation():
    # patch_center_doubled_local(0, 0) for the canonical spec is (13, 13),
    # i.e. the true pixel center 6.5 — verify the stencil resolves it to
    # the same grid-space stencil as directly requesting coordinate 6.5
    # would (computed independently here, not via the module).
    doubled_row, doubled_col = CANONICAL_SPEC.patch_center_doubled_local(0, 0)
    assert (doubled_row, doubled_col) == (13, 13)
    stencil = bilinear_stencil_from_doubled(
        doubled_row, doubled_col, SpatialSize(448, 448), SpatialSize(32, 32),
        align_corners=True,
    )
    assert abs(sum(stencil.weights()) - 1.0) < 1e-12
    expected_u = 6.5 * (32 - 1) / (448 - 1)
    expected_low = int(expected_u)
    assert stencil.row_low == expected_low
    assert stencil.row_high == expected_low + 1
    assert abs((stencil.weight_10 + stencil.weight_11) - (expected_u - expected_low)) < 1e-9


def test_stencil_boundary_points_stay_in_range():
    for row, col in [(0, 0), (447, 447), (0, 447), (447, 0)]:
        stencil = bilinear_stencil(
            PixelCoordinate(row, col),
            SpatialSize(448, 448),
            SpatialSize(32, 32),
            align_corners=True,
        )
        for grid_row, grid_col in stencil.neighbor_grid_indices():
            assert 0 <= grid_row < 32
            assert 0 <= grid_col < 32
        assert abs(sum(stencil.weights()) - 1.0) < 1e-12


def test_stencil_align_corners_true_endpoint_mapping():
    # align_corners=True maps output pixel 0 -> grid 0 and output pixel
    # (L-1) -> grid (G-1) exactly, with zero residual weight on the far side.
    top_left = bilinear_stencil(
        PixelCoordinate(0, 0), SpatialSize(448, 448), SpatialSize(32, 32), align_corners=True
    )
    assert top_left.row_low == 0 and top_left.col_low == 0
    assert top_left.weight_00 == 1.0

    bottom_right = bilinear_stencil(
        PixelCoordinate(447, 447), SpatialSize(448, 448), SpatialSize(32, 32), align_corners=True
    )
    assert bottom_right.row_high == 31 and bottom_right.col_high == 31
    assert bottom_right.weight_11 == 1.0


def test_stencil_is_geometry_only_no_tensor_sampling():
    import inspect

    source = "".join(
        inspect.getsource(fn)
        for fn in (
            geometry.bilinear_stencil,
            geometry.bilinear_stencil_from_doubled,
            geometry._axis_stencil,
            geometry._combine_axis_stencils,
        )
    )
    assert "torch" not in source
    assert "numpy" not in source


# ---------------------------------------------------------------------------
# Validation and immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SpatialSize(True, 4),
        lambda: SpatialSize(4, True),
        lambda: SpatialSize(4.0, 4),
        lambda: SpatialSize("4", 4),
        lambda: SpatialSize(0, 4),
        lambda: SpatialSize(-1, 4),
        lambda: PixelCoordinate(True, 0),
        lambda: PixelCoordinate(-1, 0),
        lambda: PixelCoordinate(0.5, 0),
    ],
)
def test_wrong_types_and_nonpositive_sizes_are_rejected(factory):
    with pytest.raises(SlidingWindowGeometryError):
        factory()


def test_malformed_plan_inputs_rejected():
    with pytest.raises(SlidingWindowGeometryError):
        SlidingWindowPlan.build(
            image_size=(1000, 1000),  # raw tuple, not SpatialSize
            crop_size=SpatialSize(448, 448),
            stride=SpatialSize(224, 224),
        )


def test_dataclasses_are_frozen():
    size = SpatialSize(10, 10)
    with pytest.raises(Exception):
        size.height = 5
    coord = PixelCoordinate(1, 1)
    with pytest.raises(Exception):
        coord.row = 5
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(500, 500),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    with pytest.raises(Exception):
        plan.grid_rows = 99
    with pytest.raises(Exception):
        plan.windows[0].origin = PixelCoordinate(0, 0)


def test_caller_inputs_are_not_mutated_by_plan_construction():
    image_size = SpatialSize(1000, 1000)
    crop_size = SpatialSize(448, 448)
    stride = SpatialSize(224, 224)
    plan = SlidingWindowPlan.build(image_size=image_size, crop_size=crop_size, stride=stride)
    assert image_size == SpatialSize(1000, 1000)
    assert crop_size == SpatialSize(448, 448)
    assert stride == SpatialSize(224, 224)
    assert plan.image_size is image_size or plan.image_size == image_size


def test_repeated_plan_construction_is_deterministic():
    args = dict(
        image_size=SpatialSize(1789, 613),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    plan_one = SlidingWindowPlan.build(**args)
    plan_two = SlidingWindowPlan.build(**args)
    assert plan_one == plan_two


def test_window_by_index_matches_window_at():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    for row in range(plan.grid_rows):
        for col in range(plan.grid_cols):
            expected = plan.window_at(row, col)
            assert plan.window_by_index(expected.index) == expected


def test_local_global_round_trip_and_intersection():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(1000, 1000),
        crop_size=SpatialSize(448, 448),
        stride=SpatialSize(224, 224),
    )
    window = plan.window_at(1, 1)
    local = PixelCoordinate(10, 20)
    glob = local_to_global(window, local)
    assert global_to_local(window, glob) == local
    outside = PixelCoordinate(0, 0)
    if not contains_point(window, outside):
        assert global_to_local(window, outside) is None

    other = plan.window_at(2, 2)
    rectangle = intersect_windows(window, other)
    area = overlap_area(window, other)
    if rectangle is None:
        assert area == 0
    else:
        assert area == rectangle.area > 0


# ---------------------------------------------------------------------------
# Production integration
# ---------------------------------------------------------------------------


def _load_cover_dr_package():
    spec = importlib.util.spec_from_file_location(
        "cover_dr_geometry_test",
        COVER_DR_PACKAGE_PATH / "__init__.py",
        submodule_search_locations=[str(COVER_DR_PACKAGE_PATH)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_segmentation_module():
    """Load the real ``dinotext_seg.py`` with its actual relative import to
    ``sliding_window_geometry.py`` intact, using lightweight stand-ins only
    for the unrelated ``mmcv``/``utils``/``models.dinotext.cover_dr``
    dependencies (mirrors ``tests/test_rwr_integration.py``'s
    ``_load_segmentation_module``, extended with real package context so
    the new relative import resolves to the real geometry module on disk).

    Returns ``(module, cover_dr)`` — callers that need to build an
    ``RWRInferenceConfig`` must use the returned ``cover_dr``, since
    ``isinstance`` checks require the exact class object ``module`` bound
    at load time.
    """
    cover_dr = _load_cover_dr_package()

    mmcv_stub = types.ModuleType("mmcv")

    class Config(dict):
        __getattr__ = dict.__getitem__

    mmcv_stub.Config = Config
    utils_stub = types.ModuleType("utils")
    utils_stub.get_logger = lambda: types.SimpleNamespace(info=lambda *_args: None)
    models_stub = types.ModuleType("models")
    models_stub.__path__ = []
    dinotext_stub = types.ModuleType("models.dinotext")
    dinotext_stub.__path__ = []

    # Give dinotext_seg.py a real package context so its own
    # "from .sliding_window_geometry import ..." resolves the actual file
    # on disk, rather than requiring a hand-maintained stand-in.
    segmentation_stub = types.ModuleType("segmentation")
    segmentation_stub.__path__ = [str(SEGMENTATION_PATH.parent.parent)]
    evaluation_stub = types.ModuleType("segmentation.evaluation")
    evaluation_stub.__path__ = [str(SEGMENTATION_PATH.parent)]

    module_names = (
        "mmcv", "utils", "models", "models.dinotext", "models.dinotext.cover_dr",
        "segmentation", "segmentation.evaluation",
        "segmentation.evaluation.sliding_window_geometry",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(
        {
            "mmcv": mmcv_stub,
            "utils": utils_stub,
            "models": models_stub,
            "models.dinotext": dinotext_stub,
            "models.dinotext.cover_dr": cover_dr,
            "segmentation": segmentation_stub,
            "segmentation.evaluation": evaluation_stub,
        }
    )
    sys.modules.pop("segmentation.evaluation.sliding_window_geometry", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "segmentation.evaluation.dinotext_seg", SEGMENTATION_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, cover_dr
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        sys.modules.pop("segmentation.evaluation.dinotext_seg", None)


def test_production_slide_inference_uses_the_geometry_plan():
    module, _cover_dr = _load_segmentation_module()
    assert hasattr(module, "SlidingWindowPlan")
    import inspect

    source = inspect.getsource(module.DINOTextSegInference.slide_inference)
    assert "SlidingWindowPlan" in source
    assert "h_grids" not in source and "w_grids" not in source


def test_production_slide_inference_call_count_order_and_crop_coordinates():
    module, _cover_dr = _load_segmentation_module()
    calls = []

    class Model(nn.Module):
        def generate_masks(self, image, text, apply_pamr=False):
            del text, apply_pamr
            calls.append(tuple(image.shape[-2:]))
            return torch.zeros((1, 3, *image.shape[-2:])), torch.empty(0)

    model = Model()
    inference = module.DINOTextSegInference(
        model, torch.randn(3, 4), ["a", "b", "c"], with_bg=False,
        test_cfg=dict(mode="slide", crop_size=(224, 224), stride=(100, 150)),
    )
    img = torch.zeros(1, 3, 673, 900)
    result = inference.slide_inference(img, [{"img_shape": (673, 900, 3), "ori_shape": (673, 900, 3)}], rescale=False)

    expected_plan = SlidingWindowPlan.build(
        image_size=SpatialSize(673, 900),
        crop_size=SpatialSize(224, 224),
        stride=SpatialSize(100, 150),
    )
    assert len(calls) == expected_plan.window_count
    for observed_shape, window in zip(calls, expected_plan.windows):
        assert observed_shape == window.extent.as_tuple()
    assert result.shape == (1, 3, 673, 900)


def test_production_uniform_stitch_result_unchanged_by_refactor():
    module, _cover_dr = _load_segmentation_module()

    def make_model():
        class Model(nn.Module):
            def generate_masks(self, image, text, apply_pamr=False):
                del text, apply_pamr
                height, width = image.shape[-2:]
                value = float(height * 1000 + width)
                return torch.full((1, 3, height, width), value), torch.empty(0)

        return Model()

    inference = module.DINOTextSegInference(
        make_model(), torch.randn(3, 4), ["a", "b", "c"], with_bg=False,
        test_cfg=dict(mode="slide", crop_size=(224, 224), stride=(100, 150)),
    )
    img = torch.zeros(1, 3, 673, 900)
    meta = [{"img_shape": (673, 900, 3), "ori_shape": (673, 900, 3)}]
    refactored_result = inference.slide_inference(img, meta, rescale=False)

    legacy_preds, legacy_count = legacy_stitch(
        673, 900, 224, 224, 100, 150, 3,
        lambda y1, y2, x1, x2: torch.full(
            (1, 3, y2 - y1, x2 - x1), float((y2 - y1) * 1000 + (x2 - x1))
        ),
    )
    legacy_result = legacy_preds / legacy_count
    assert torch.equal(refactored_result.double(), legacy_result)


def test_production_rwr_still_runs_once_per_crop_no_two_pass():
    module, cover_dr = _load_segmentation_module()
    from src.rwr_reproduction_identity import load_identity

    identity = load_identity(repo_root=ROOT)
    # config must be built from the SAME cover_dr module instance that
    # dinotext_seg.py bound internally, since apply_rwr_to_e3_snapshot does
    # an isinstance(config, RWRInferenceConfig) check against that class.
    config = cover_dr.RWRInferenceConfig(
        enabled=True,
        identity_path="evaluation_identities/e3_canonical_directed_rwr.toml",
        graph_mode=identity["rwr"]["graph_mode"],
        alpha=0.5,
        top_k=2,
        affinity_power=identity["rwr"]["affinity_power"],
        solver=identity["solver"]["method"],
        solver_rtol=identity["solver"]["rtol"],
        solver_atol=identity["solver"]["atol"],
        solver_max_iterations=identity["solver"]["max_iterations"],
        expected_class_count=3,
        config_path=identity["canonical_config_path"],
    )
    config.validate()

    snapshot_calls = []

    class Model(nn.Module):
        def generate_patch_snapshot(self, image, text):
            del text
            snapshot_calls.append(1)
            features = torch.nn.functional.normalize(torch.randn(1, 4, 5), dim=-1)
            scores = torch.randn(1, 4, 3)
            return types.SimpleNamespace(
                unary_scores=scores, dino_features=features, grid_hw=(2, 2)
            )

        def masks_from_patch_scores(self, scores, grid_hw, output_hw):
            return torch.zeros((1, 3, *output_hw))

        def generate_masks(self, *_args, **_kwargs):
            raise AssertionError("RWR-enabled path must not call generate_masks")

    inference = module.DINOTextSegInference(
        Model(), torch.randn(3, 4), ["a", "b", "c"], with_bg=False,
        test_cfg=dict(mode="slide", crop_size=(4, 4), stride=(4, 4)),
    )
    # Bypass the constructor's canonical (171-class) identity loading so this
    # test can use a small synthetic 3-class problem, matching the pattern
    # used by tests/test_rwr_integration.py's equivalent production-path test.
    inference.rwr_config = config
    inference.rwr_runtime = cover_dr.RWRRuntimeSummary()
    img = torch.zeros(1, 3, 8, 4)
    meta = [{"img_shape": (8, 4, 3), "ori_shape": (8, 4, 3)}]
    inference.slide_inference(img, meta, rescale=False)

    expected_plan = SlidingWindowPlan.build(
        image_size=SpatialSize(8, 4), crop_size=SpatialSize(4, 4), stride=SpatialSize(4, 4)
    )
    assert len(snapshot_calls) == expected_plan.window_count  # exactly once per crop
    assert inference.rwr_runtime.window_count == expected_plan.window_count
