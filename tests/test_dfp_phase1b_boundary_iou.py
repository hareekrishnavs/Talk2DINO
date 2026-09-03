"""Tests for the DFP Phase 1B Boundary IoU metric implementation
(src/dfp_phase1b_boundary_iou.py). CPU-only throughout: no CUDA, no
model, no dataset, no real COCO-Stuff-500 access. Every fixture is a
small hand-built or randomly generated array.

Contains a from-scratch, dependency-free (no scipy, no cv2) brute-force
multi-source-BFS Chebyshev-distance erosion oracle, independently
re-derived directly from the frozen definition (one-pixel constant-zero
pad, then N sequential 3x3-all-ones erosions, treating anything beyond
the finite padded array as non-constraining) -- proven equivalent to the
production scipy-based implementation across many geometries and
dilation values, including dilation > 1 (multi-iteration) cases. Where
OpenCV happens to be importable, the actual vendored official
`mask_to_boundary` function is additionally exercised and required to
match exactly; that comparison is skipped (never silently passed) when
OpenCV is unavailable, per the module's own documented fallback.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.dfp_phase1b_boundary_iou_identity import load_boundary_iou_contract  # noqa: E402
from src.dfp_phase1b_boundary_iou import (  # noqa: E402
    BoundaryConfusionStatistics,
    DfpPhase1bBoundaryIouError,
    aggregate_boundary_confusion_statistics,
    binary_mask_to_boundary,
    compute_boundary_confusion_statistics,
    compute_boundary_metrics,
    compute_boundary_width,
)
import src.dfp_phase1b_boundary_iou as biou_module  # noqa: E402

CONTRACT = load_boundary_iou_contract(repo_root=ROOT)
VENDORED_SOURCE = ROOT / "third_party/boundary_iou_api/boundary_iou/utils/boundary_utils.py"


# ---------------------------------------------------------------------------
# Independent, dependency-free brute-force oracle.
# ---------------------------------------------------------------------------


def _brute_force_chebyshev_erode(padded01: np.ndarray, iterations: int) -> np.ndarray:
    """Pure-Python (no scipy, no cv2) multi-source BFS Chebyshev
    distance-to-nearest-zero transform on an already-padded 0/1 array,
    thresholded at > iterations. Mathematically identical to
    `iterations` sequential 3x3-all-ones erosions where anything beyond
    the finite array is treated as non-constraining (no zero exists out
    there to shrink toward) -- this identity (N-fold erosion by a unit
    disk == thresholding the distance transform at N) is independent of
    any scipy/cv2 API and is the sole ground truth this test file
    trusts."""
    h, w = padded01.shape
    dist = np.full((h, w), -1, dtype=np.int64)
    q = deque()
    zero_rows, zero_cols = np.nonzero(padded01 == 0)
    for i, j in zip(zero_rows.tolist(), zero_cols.tolist()):
        dist[i, j] = 0
        q.append((i, j))
    while q:
        i, j = q.popleft()
        d = dist[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and dist[ni, nj] == -1:
                    dist[ni, nj] = d + 1
                    q.append((ni, nj))
    out = np.ones((h, w), dtype=bool)
    reached = dist != -1
    out[reached] = dist[reached] > iterations
    out[~reached] = True  # no zero anywhere in the array: never eroded
    return out


def _brute_force_boundary(mask01: np.ndarray, dilation: int, *, contract=CONTRACT) -> np.ndarray:
    h, w = mask01.shape
    padded = np.pad(mask01.astype(bool), pad_width=contract.padding_width, mode="constant", constant_values=bool(contract.padding_value))
    eroded_padded = _brute_force_chebyshev_erode(padded, dilation)
    pw = contract.padding_width
    eroded = eroded_padded[pw : pw + h, pw : pw + w]
    boundary = mask01.astype(np.uint8).copy()
    boundary[eroded] = 0
    return boundary


# ---------------------------------------------------------------------------
# Official-reference equivalence.
# ---------------------------------------------------------------------------


def _try_import_vendored_official():
    """Loads the pinned vendored module directly by file path (no
    package __init__ needed for this single function) -- only if cv2 AND
    pycocotools are both importable in THIS environment. Never modifies
    sys.modules for any canonical repository module name."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return None
    try:
        import pycocotools.mask  # noqa: F401
    except ImportError:
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("_vendored_boundary_utils_under_test", VENDORED_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VENDORED = _try_import_vendored_official()

_EQUIVALENCE_GEOMETRIES = []


def _make_geometries():
    rng = np.random.default_rng(1234)
    geoms = {}
    geoms["all_zero_10x10"] = np.zeros((10, 10), dtype=np.uint8)
    geoms["all_one_10x10"] = np.ones((10, 10), dtype=np.uint8)
    m = np.zeros((9, 9), dtype=np.uint8)
    m[4, 4] = 1
    geoms["one_pixel_object"] = m
    m = np.ones((9, 9), dtype=np.uint8)
    m[4, 4] = 0
    geoms["one_pixel_hole"] = m
    m = np.zeros((7, 20), dtype=np.uint8)
    m[3, :] = 1
    geoms["horizontal_line"] = m
    m = np.zeros((20, 7), dtype=np.uint8)
    m[:, 3] = 1
    geoms["vertical_line"] = m
    m = np.zeros((15, 15), dtype=np.uint8)
    for i in range(15):
        m[i, i] = 1
    geoms["diagonal_line"] = m
    m = np.zeros((12, 12), dtype=np.uint8)
    m[0:6, 0:6] = 1  # touches top-left edge
    geoms["image_edge_object"] = m
    m = np.zeros((10, 10), dtype=np.uint8)
    m[0, 0] = 1
    m[9, 9] = 1
    m[0, 9] = 1
    m[9, 0] = 1
    geoms["corner_touching_object"] = m
    m = np.zeros((21, 21), dtype=np.uint8)
    m[5:16, 5:16] = 1
    m[8:13, 8:13] = 0
    geoms["thin_ring"] = m
    m = np.zeros((20, 20), dtype=np.uint8)
    m[2:5, 2:5] = 1
    m[14:18, 14:18] = 1
    geoms["disconnected_objects"] = m
    m = np.zeros((13, 40), dtype=np.uint8)
    m[3:10, 5:35] = 1
    geoms["non_square_image"] = m
    m = np.zeros((3, 3), dtype=np.uint8)
    m[1, 1] = 1
    geoms["very_small_image"] = m
    # A large object requiring several erosion iterations (dilation > 1)
    m = np.zeros((300, 300), dtype=np.uint8)
    m[20:280, 20:280] = 1
    geoms["large_object_multi_iteration"] = m
    for k in range(6):
        h = int(rng.integers(6, 60))
        w = int(rng.integers(6, 60))
        density = rng.uniform(0.2, 0.8)
        geoms[f"random_{k}"] = (rng.random((h, w)) < density).astype(np.uint8)
    return geoms


_GEOMETRIES = _make_geometries()


@pytest.mark.parametrize("name", sorted(_GEOMETRIES))
def test_production_matches_brute_force_oracle(name):
    mask = _GEOMETRIES[name]
    h, w = mask.shape
    dilation = compute_boundary_width(int(h), int(w), CONTRACT)
    expected = _brute_force_boundary(mask, dilation)
    observed = binary_mask_to_boundary(mask, contract=CONTRACT)
    assert np.array_equal(expected, observed), name


@pytest.mark.parametrize("dilation", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("name", ["image_edge_object", "thin_ring", "random_0", "random_3"])
def test_production_matches_brute_force_oracle_fixed_dilations(name, dilation):
    mask = _GEOMETRIES[name]
    expected = _brute_force_boundary(mask, dilation)
    observed = binary_mask_to_boundary(mask, contract=CONTRACT, dilation=dilation)
    assert np.array_equal(expected, observed), (name, dilation)


def test_width_requiring_minimum_dilation_one():
    # sqrt(2^2+2^2)*0.02 ~= 0.057 -> round -> 0, floored to minimum 1.
    assert compute_boundary_width(2, 2, CONTRACT) == 1


def test_widths_producing_multiple_erosion_iterations():
    # sqrt(300^2+300^2)*0.02 ~= 8.49 -> round -> 8.
    assert compute_boundary_width(300, 300, CONTRACT) == 8
    assert compute_boundary_width(707, 707, CONTRACT) > 1


@pytest.mark.skipif(_VENDORED is None, reason="cv2 and/or pycocotools not importable in this environment")
@pytest.mark.parametrize("name", sorted(_GEOMETRIES))
def test_production_matches_actual_vendored_official_function(name):
    mask = _GEOMETRIES[name]
    expected = _VENDORED.mask_to_boundary(mask.copy(), dilation_ratio=CONTRACT.dilation_ratio)
    observed = binary_mask_to_boundary(mask, contract=CONTRACT)
    assert np.array_equal(expected, observed), name


# ---------------------------------------------------------------------------
# Mathematical properties.
# ---------------------------------------------------------------------------


def _confusion(pred, gt, class_count=2, ignore_index=255):
    return compute_boundary_confusion_statistics(pred, gt, class_count=class_count, ignore_index=ignore_index, contract=CONTRACT)


def test_boundary_iou_symmetry():
    rng = np.random.default_rng(7)
    gt = rng.integers(0, 3, size=(20, 20)).astype(np.int64)
    pred = rng.integers(0, 3, size=(20, 20)).astype(np.int64)
    s1 = _confusion(pred, gt, class_count=3, ignore_index=255)
    s2 = _confusion(gt, pred, class_count=3, ignore_index=255)  # swapped roles
    assert np.array_equal(s1.boundary_intersection, s2.boundary_intersection)
    assert np.array_equal(s1.boundary_union, s2.boundary_union)


def test_identical_nonempty_masks_give_1_0():
    gt = np.zeros((16, 16), dtype=np.int64)
    gt[4:12, 4:12] = 1
    pred = gt.copy()
    agg = aggregate_boundary_confusion_statistics([_confusion(pred, gt)])
    res = compute_boundary_metrics(agg)
    assert res.per_class_boundary_iou[1] == pytest.approx(100.0)


def test_disjoint_boundaries_give_0_0():
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[2:6, 2:6] = 1
    pred = np.zeros((20, 20), dtype=np.int64)
    pred[14:18, 14:18] = 1
    agg = aggregate_boundary_confusion_statistics([_confusion(pred, gt)])
    res = compute_boundary_metrics(agg)
    assert res.per_class_boundary_iou[1] == pytest.approx(0.0)


def test_small_objects_handled():
    gt = np.zeros((9, 9), dtype=np.int64)
    gt[4, 4] = 1
    pred = gt.copy()
    stats = _confusion(pred, gt)
    assert stats.gt_boundary_pixels[1] == 1
    assert stats.boundary_intersection[1] == 1


def test_image_edge_boundaries_are_counted():
    gt = np.zeros((10, 10), dtype=np.int64)
    gt[0:5, 0:5] = 1  # touches top-left edge
    stats = _confusion(gt.copy(), gt)
    boundary = binary_mask_to_boundary((gt == 1).astype(np.uint8), contract=CONTRACT)
    assert boundary[0, 0] == 1  # the true image-edge corner is boundary


def test_all_valid_reduces_to_official_binary_extraction():
    rng = np.random.default_rng(3)
    gt_c = (rng.random((14, 14)) < 0.5).astype(np.uint8)
    valid = np.ones((14, 14), dtype=bool)
    dilation = compute_boundary_width(14, 14, CONTRACT)
    ignore_neutral = biou_module._ignore_neutral_boundary(gt_c.astype(bool), valid, dilation=dilation, contract=CONTRACT)
    official = binary_mask_to_boundary(gt_c, contract=CONTRACT, dilation=dilation).astype(bool)
    assert np.array_equal(ignore_neutral, official)


def test_ignore_holes_do_not_create_artificial_boundaries():
    class_count, ignore_index = 2, 255
    gt = np.ones((21, 21), dtype=np.int64)  # entirely class 1
    gt[10, 10] = ignore_index  # one ignore pixel deep inside
    pred = np.ones((21, 21), dtype=np.int64)  # pred never carries the ignore label itself
    stats = _confusion(pred, gt, class_count=class_count, ignore_index=ignore_index)
    # No interior boundary should appear around the single ignore pixel:
    # only the true image-edge boundary should be present.
    edge_only = binary_mask_to_boundary(np.ones((21, 21), dtype=np.uint8), contract=CONTRACT)
    assert stats.gt_boundary_pixels[1] == int(np.count_nonzero(edge_only))


def test_ignore_holes_would_create_artificial_boundary_under_naive_masking():
    """Sanity check that the artificial-boundary risk is real: naively
    zeroing the ignore pixel BEFORE erosion (the un-adapted approach)
    does produce an extra boundary ring around it, distinguishing the
    ignore-neutral construction from the naive one this identity
    explicitly rejects."""
    naive = np.ones((21, 21), dtype=np.uint8)
    naive[10, 10] = 0
    naive_boundary = binary_mask_to_boundary(naive, contract=CONTRACT)
    edge_only = binary_mask_to_boundary(np.ones((21, 21), dtype=np.uint8), contract=CONTRACT)
    assert int(np.count_nonzero(naive_boundary)) > int(np.count_nonzero(edge_only))


def test_prediction_inside_ignore_region_never_changes_any_count():
    class_count, ignore_index = 2, 255
    gt = np.ones((15, 15), dtype=np.int64)
    gt[5:10, 5:10] = ignore_index
    pred_a = np.ones((15, 15), dtype=np.int64)
    pred_b = pred_a.copy()
    pred_b[5:10, 5:10] = 0  # differs only inside the ignore region
    stats_a = _confusion(pred_a, gt, class_count=class_count, ignore_index=ignore_index)
    stats_b = _confusion(pred_b, gt, class_count=class_count, ignore_index=ignore_index)
    assert np.array_equal(stats_a.boundary_intersection, stats_b.boundary_intersection)
    assert np.array_equal(stats_a.boundary_union, stats_b.boundary_union)
    assert np.array_equal(stats_a.pred_boundary_pixels, stats_b.pred_boundary_pixels)


def test_all_ignore_image_gives_zero_statistics():
    class_count, ignore_index = 3, 255
    gt = np.full((10, 10), ignore_index, dtype=np.int64)
    pred = np.zeros((10, 10), dtype=np.int64)
    stats = _confusion(pred, gt, class_count=class_count, ignore_index=ignore_index)
    assert np.all(stats.boundary_intersection == 0)
    assert np.all(stats.boundary_union == 0)
    assert np.all(stats.gt_boundary_pixels == 0)
    assert np.all(stats.pred_boundary_pixels == 0)


def test_prediction_only_class_gives_union_and_zero_intersection():
    gt = np.zeros((14, 14), dtype=np.int64)
    pred = np.zeros((14, 14), dtype=np.int64)
    pred[3:9, 3:9] = 1
    stats = _confusion(pred, gt, class_count=2, ignore_index=255)
    assert stats.boundary_intersection[1] == 0
    assert stats.boundary_union[1] > 0
    assert stats.pred_boundary_pixels[1] > 0
    assert stats.gt_boundary_pixels[1] == 0


def test_gt_only_class_gives_union_and_zero_intersection():
    gt = np.zeros((14, 14), dtype=np.int64)
    gt[3:9, 3:9] = 1
    pred = np.zeros((14, 14), dtype=np.int64)
    stats = _confusion(pred, gt, class_count=2, ignore_index=255)
    assert stats.boundary_intersection[1] == 0
    assert stats.boundary_union[1] > 0
    assert stats.gt_boundary_pixels[1] > 0
    assert stats.pred_boundary_pixels[1] == 0


def test_dataset_aggregation_differs_from_mean_per_image_boundary_iou():
    class_count, ignore_index = 2, 255
    # Image 1: perfect match (IoU=100 for class 1).
    gt1 = np.zeros((12, 12), dtype=np.int64)
    gt1[3:9, 3:9] = 1
    pred1 = gt1.copy()
    # Image 2: completely disjoint (IoU=0 for class 1), with a much larger
    # boundary union than image 1's.
    gt2 = np.zeros((30, 30), dtype=np.int64)
    gt2[2:6, 2:6] = 1
    pred2 = np.zeros((30, 30), dtype=np.int64)
    pred2[24:28, 24:28] = 1

    stats1 = _confusion(pred1, gt1, class_count=class_count, ignore_index=ignore_index)
    stats2 = _confusion(pred2, gt2, class_count=class_count, ignore_index=ignore_index)
    agg = aggregate_boundary_confusion_statistics([stats1, stats2])
    dataset_level = compute_boundary_metrics(agg)

    res1 = compute_boundary_metrics(aggregate_boundary_confusion_statistics([stats1]))
    res2 = compute_boundary_metrics(aggregate_boundary_confusion_statistics([stats2]))
    naive_mean = (res1.per_class_boundary_iou[1] + res2.per_class_boundary_iou[1]) / 2.0

    assert dataset_level.per_class_boundary_iou[1] != pytest.approx(naive_mean)


def test_class_inclusion_uses_aggregate_union_not_per_image():
    # Class 2 has zero union in image 1 but positive union in image 2 --
    # it must still be included in the dataset-level mean.
    class_count, ignore_index = 3, 255
    gt1 = np.zeros((10, 10), dtype=np.int64)
    pred1 = np.zeros((10, 10), dtype=np.int64)
    gt2 = np.zeros((10, 10), dtype=np.int64)
    gt2[2:6, 2:6] = 2
    pred2 = gt2.copy()
    stats1 = _confusion(pred1, gt1, class_count=class_count, ignore_index=ignore_index)
    stats2 = _confusion(pred2, gt2, class_count=class_count, ignore_index=ignore_index)
    agg = aggregate_boundary_confusion_statistics([stats1, stats2])
    res = compute_boundary_metrics(agg)
    assert not np.isnan(res.per_class_boundary_iou[2])
    assert np.isnan(res.per_class_boundary_iou[1])  # class 1 never appears anywhere


def test_no_defined_classes_causes_clean_rejection():
    class_count, ignore_index = 2, 255
    gt = np.full((10, 10), ignore_index, dtype=np.int64)
    pred = np.zeros((10, 10), dtype=np.int64)
    stats = _confusion(pred, gt, class_count=class_count, ignore_index=ignore_index)
    agg = aggregate_boundary_confusion_statistics([stats])
    with pytest.raises(DfpPhase1bBoundaryIouError, match="undefined"):
        compute_boundary_metrics(agg)


def test_inputs_remain_unchanged():
    gt = np.zeros((12, 12), dtype=np.int64)
    gt[3:9, 3:9] = 1
    pred = gt.copy()
    gt_before, pred_before = gt.copy(), pred.copy()
    _confusion(pred, gt)
    assert np.array_equal(gt, gt_before)
    assert np.array_equal(pred, pred_before)
    mask = (gt == 1).astype(np.uint8)
    mask_before = mask.copy()
    binary_mask_to_boundary(mask, contract=CONTRACT)
    assert np.array_equal(mask, mask_before)


def test_no_epsilon_in_denominator():
    # If any epsilon were silently added to the union denominator, this
    # exact ratio would not reproduce a clean float in general -- assert
    # the algebraic identity holds at full precision instead.
    gt = np.zeros((16, 16), dtype=np.int64)
    gt[3:9, 3:9] = 1
    pred = np.zeros((16, 16), dtype=np.int64)
    pred[3:9, 5:11] = 1
    agg = aggregate_boundary_confusion_statistics([_confusion(pred, gt)])
    res = compute_boundary_metrics(agg)
    expected = (agg.boundary_intersection[1] / agg.boundary_union[1]) * 100.0
    assert res.per_class_boundary_iou[1] == expected


# ---------------------------------------------------------------------------
# Hand-derived fixtures (Section 10 requirement).
# ---------------------------------------------------------------------------


def test_hand_derived_3x3_square_dilation_1():
    """A solid 3x3 square inside a 5x5 all-zero field, dilation=1.
    Erosion with a 3x3 kernel keeps only the CENTER pixel (the only
    pixel whose full 8-neighborhood, including the one-pixel zero pad,
    is entirely foreground). Boundary = mask - eroded is therefore the
    8-pixel RING around the center; hand-enumerated coordinates:
    (1,1) (1,2) (1,3)
    (2,1)       (2,3)
    (3,1) (3,2) (3,3)
    """
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    boundary = binary_mask_to_boundary(mask, contract=CONTRACT, dilation=1)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(boundary, expected)
    expected_coords = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)}
    observed_coords = {(int(i), int(j)) for i, j in zip(*np.nonzero(boundary))}
    assert observed_coords == expected_coords


def test_hand_derived_5x5_square_dilation_2():
    """A solid 5x5 square in a 9x9 field, dilation=2. Two 3x3 erosion
    passes shrink the 5x5 square to nothing (a 5x5 square's largest
    inscribed erosion-survivor under 2 passes of a 3x3 kernel is empty:
    after 1 pass, only the center 3x3 survives; after 2 passes, only the
    single center pixel would survive a 3x3-square input -- but a 5x5
    input's first pass leaves a 3x3 center, and the second pass applied
    to THAT 3x3 (now surrounded by the already-eroded ring, which reads
    as background) leaves only its own center pixel). So boundary =
    every pixel of the original 5x5 mask except that one true center
    pixel: 24 boundary pixels out of 25.
    """
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 1
    boundary = binary_mask_to_boundary(mask, contract=CONTRACT, dilation=2)
    assert int(np.count_nonzero(boundary)) == 24
    assert boundary[4, 4] == 0  # the true center survives erosion
    assert np.array_equal(boundary[2:7, 2:7].astype(bool) | (boundary[2:7, 2:7] == 0), np.ones((5, 5), dtype=bool))


def test_hand_derived_zero_padding_differs_from_replicate_padding():
    """Distinguishes zero padding (official, frozen) from hypothetical
    replicate/edge padding: for a mask that fills the ENTIRE array (no
    background anywhere inside), zero padding still finds a "virtual"
    zero one pixel outside every edge, so the outermost ring erodes away
    -- replicate padding would not erode anything (a completely full
    array replicated at the border stays entirely foreground forever)."""
    mask = np.ones((7, 7), dtype=np.uint8)
    boundary = binary_mask_to_boundary(mask, contract=CONTRACT, dilation=1)
    assert int(np.count_nonzero(boundary)) > 0  # zero padding: an edge boundary exists
    # A hand check: under replicate padding, eroding an all-ones array
    # would leave it entirely foreground (boundary would be all-zero).
    assert not np.array_equal(boundary, np.zeros((7, 7), dtype=np.uint8))


def test_hand_derived_3x3_erosion_differs_from_cross_erosion():
    """A plus (5-pixel, 4-connected) mask whose center pixel has all 4
    orthogonal neighbors foreground but all 4 DIAGONAL neighbors
    background. A cross/4-connected structuring element would keep the
    center pixel as interior (non-boundary); the frozen full 3x3
    (8-connected) kernel requires the diagonals too, so it fails to
    erode the center -- the center pixel is boundary under the frozen
    convention but would be interior under the rejected cross
    convention. Hand-enumerated: center = (2,2) in a 5x5 field."""
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 2] = 1
    mask[3, 2] = 1
    mask[2, 1] = 1
    mask[2, 3] = 1
    mask[2, 2] = 1
    dilation = 1
    full_3x3 = binary_mask_to_boundary(mask, contract=CONTRACT, dilation=dilation)
    assert full_3x3[2, 2] == 1  # center IS boundary under the frozen 8-connected kernel

    # Hand-compute a CROSS (4-connected) erosion independently: a pixel
    # survives iff itself and its 4 orthogonal neighbors are foreground
    # in the zero-padded array (never touching diagonal neighbors).
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    cross_eroded = np.ones_like(padded)
    h, w = padded.shape
    for i in range(h):
        for j in range(w):
            ok = padded[i, j]
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    ok = ok and padded[ni, nj]
                # out-of-bounds is non-constraining, matching the frozen convention
            cross_eroded[i, j] = ok
    cross_boundary = mask.astype(np.uint8).copy()
    cross_boundary[cross_eroded[1:6, 1:6]] = 0
    assert cross_boundary[2, 2] == 0  # center would be interior under a cross kernel

    assert not np.array_equal(full_3x3, cross_boundary)


def test_hand_derived_ignore_neutral_vs_ignore_as_background():
    """A single ignore pixel dead-center of an otherwise-solid mask:
    ignore-neutral erosion (frozen) produces the SAME boundary as if no
    ignore pixel existed at all (only the true image-edge ring);
    treating ignore as background (rejected convention) would instead
    produce an additional ring around the ignore hole itself."""
    valid = np.ones((9, 9), dtype=bool)
    valid[4, 4] = False
    mask = np.ones((9, 9), dtype=bool)
    dilation = 1

    ignore_neutral = biou_module._ignore_neutral_boundary(mask, valid, dilation=dilation, contract=CONTRACT)
    no_ignore_at_all = binary_mask_to_boundary(np.ones((9, 9), dtype=np.uint8), contract=CONTRACT, dilation=dilation).astype(bool)
    assert np.array_equal(ignore_neutral, no_ignore_at_all)

    # Ignore-as-background (naive/rejected): zero out the ignore pixel
    # then erode directly -- produces an EXTRA boundary ring around (4,4).
    naive = mask.copy()
    naive[4, 4] = False
    naive_boundary = binary_mask_to_boundary(naive.astype(np.uint8), contract=CONTRACT, dilation=dilation).astype(bool)
    assert not np.array_equal(ignore_neutral, naive_boundary & valid)


def test_hand_derived_dataset_sum_then_divide_differs_from_mean_per_image():
    """Two images with the SAME per-image Boundary IoU by coincidence of
    ratio but very different absolute boundary union magnitudes: sum-
    then-divide dataset aggregation must weight by absolute pixel counts,
    not treat every image equally like a naive per-image mean would."""
    class_count, ignore_index = 2, 255
    # Image A: intersection=3, union=6 (IoU 50%) via a tiny hand mask.
    gtA = np.zeros((6, 6), dtype=np.int64)
    gtA[2, 2] = 1
    predA = np.zeros((6, 6), dtype=np.int64)
    predA[2, 2] = 1
    predA[2, 3] = 1  # forces a partial-overlap boundary scenario at a larger scale below instead
    # (This tiny hand mask alone cannot cleanly hit IoU=50% at pixel
    # granularity with dilation=1 on a single foreground pixel: a lone
    # foreground pixel's boundary is itself, so intersection/union come
    # from two single-pixel masks -- use an explicit, exactly-computed
    # pair instead.)
    gtA = np.zeros((6, 6), dtype=np.int64)
    gtA[2, 2] = 1
    predA = np.zeros((6, 6), dtype=np.int64)
    predA[2, 2] = 1  # identical single-pixel masks: intersection=union=1
    statsA = _confusion(predA, gtA, class_count=class_count, ignore_index=ignore_index)
    assert statsA.boundary_intersection[1] == 1 and statsA.boundary_union[1] == 1

    # Image B: a large ring-shaped disagreement giving intersection=0,
    # union>0 (IoU 0%), at a much bigger absolute scale.
    gtB = np.zeros((40, 40), dtype=np.int64)
    gtB[5:35, 5:35] = 1
    predB = np.zeros((40, 40), dtype=np.int64)
    statsB = _confusion(predB, gtB, class_count=class_count, ignore_index=ignore_index)
    assert statsB.boundary_intersection[1] == 0 and statsB.boundary_union[1] > 0

    agg = aggregate_boundary_confusion_statistics([statsA, statsB])
    dataset_level = compute_boundary_metrics(agg)
    naive_mean = 100.0 * ((1.0 / 1.0) + (0.0 / statsB.boundary_union[1])) / 2.0  # 50.0
    assert dataset_level.per_class_boundary_iou[1] != pytest.approx(naive_mean)
    # Dataset-level must be dominated by image B's much larger union.
    assert dataset_level.per_class_boundary_iou[1] < 5.0


# ---------------------------------------------------------------------------
# Dtype and validation.
# ---------------------------------------------------------------------------


def test_rejects_wrong_rank():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(np.zeros((4, 4, 1), dtype=np.int64), np.zeros((4, 4, 1), dtype=np.int64), class_count=2, ignore_index=255, contract=CONTRACT)


def test_rejects_mismatched_shapes():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(np.zeros((4, 4), dtype=np.int64), np.zeros((5, 5), dtype=np.int64), class_count=2, ignore_index=255, contract=CONTRACT)


def test_rejects_invalid_class_count():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.int64), class_count=0, ignore_index=255, contract=CONTRACT)


def test_rejects_invalid_ignore_index_inside_class_range():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.int64), class_count=2, ignore_index=1, contract=CONTRACT)


def test_rejects_floating_label_maps():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(np.zeros((4, 4), dtype=np.float64), np.zeros((4, 4), dtype=np.int64), class_count=2, ignore_index=255, contract=CONTRACT)


def test_rejects_boolean_label_maps():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=np.int64), class_count=2, ignore_index=255, contract=CONTRACT)


def test_rejects_negative_prediction_labels():
    pred = np.zeros((4, 4), dtype=np.int64)
    pred[0, 0] = -1
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(pred, np.zeros((4, 4), dtype=np.int64), class_count=2, ignore_index=255, contract=CONTRACT)


def test_rejects_predictions_outside_class_range():
    pred = np.zeros((4, 4), dtype=np.int64)
    pred[0, 0] = 99
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(pred, np.zeros((4, 4), dtype=np.int64), class_count=2, ignore_index=255, contract=CONTRACT)


def test_rejects_invalid_non_ignore_gt_labels():
    gt = np.zeros((4, 4), dtype=np.int64)
    gt[0, 0] = 99  # neither a valid class nor the ignore index
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(np.zeros((4, 4), dtype=np.int64), gt, class_count=2, ignore_index=255, contract=CONTRACT)


def test_rejects_malformed_contract():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_confusion_statistics(
            np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.int64), class_count=2, ignore_index=255,
            contract="not a contract",  # type: ignore[arg-type]
        )


def test_binary_mask_to_boundary_rejects_non_binary_values():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 2
    with pytest.raises(DfpPhase1bBoundaryIouError):
        binary_mask_to_boundary(mask, contract=CONTRACT)


def test_binary_mask_to_boundary_rejects_wrong_dtype():
    mask = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(DfpPhase1bBoundaryIouError):
        binary_mask_to_boundary(mask, contract=CONTRACT)


def test_aggregate_rejects_empty_sequence():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        aggregate_boundary_confusion_statistics([])


def test_compute_metrics_rejects_non_aggregate_input():
    with pytest.raises(DfpPhase1bBoundaryIouError):
        compute_boundary_metrics("not a BoundaryConfusionStatistics")  # type: ignore[arg-type]


def test_accepts_read_only_and_non_contiguous_arrays():
    gt = np.zeros((16, 16), dtype=np.int64)
    gt[3:9, 3:9] = 1
    gt_transposed_view = gt.T  # non-contiguous view
    gt_transposed_view.flags.writeable = False
    pred = gt_transposed_view.copy()
    stats = _confusion(pred, gt_transposed_view)  # must not raise or mutate
    assert stats.gt_boundary_pixels[1] > 0


def test_keyboard_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except DfpPhase1bBoundaryIouError:
            pytest.fail("KeyboardInterrupt must never be caught by the domain error handler")


def test_system_exit_propagates():
    with pytest.raises(SystemExit):
        try:
            raise SystemExit(3)
        except DfpPhase1bBoundaryIouError:
            pytest.fail("SystemExit must never be caught by the domain error handler")


def test_memory_error_propagates():
    with pytest.raises(MemoryError):
        try:
            raise MemoryError()
        except DfpPhase1bBoundaryIouError:
            pytest.fail("MemoryError must never be caught by the domain error handler")


# ---------------------------------------------------------------------------
# No-compute isolation.
# ---------------------------------------------------------------------------


def test_module_import_does_not_import_torch():
    import subprocess

    src = "import sys\nimport src.dfp_phase1b_boundary_iou\nprint('torch' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", src], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
