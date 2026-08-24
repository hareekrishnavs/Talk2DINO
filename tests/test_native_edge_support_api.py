"""CPU-only coverage for models.dinotext.cover_dr.native_edge_support: the
native cross-view directed-edge support audit API. Covers native
patch-coordinate mapping (including exhaustive cross-validation of the
vectorized implementation against the scalar
``PatchGridSpec.native_patch_node`` reference), the directed support
definition, the deterministic descriptive ranking, row/window summary
aggregation, GT reachability diagnostics, shared per-window execution, and
scientific isolation (no prohibited T4/DCR/SUR/pruning/counterfactual code
paths). Never initializes CUDA, never loads the real model/dataset."""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path

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

from models.dinotext.cover_dr.native_edge_support import (  # noqa: E402
    SUPPORT_FRACTION_BUCKETS,
    UNDEFINED_REASON_NO_ALIGNED_OBSERVER,
    UNDEFINED_REASON_SINGLE_WINDOW_IMAGE,
    DirectedEdgeSupport,
    NativeEdgeSupportError,
    NativeNodeCoordinate,
    NativeNodeMapping,
    build_edge_records_for_row,
    build_row_correctness_record,
    build_row_summary,
    build_window_summary,
    compute_dense_adjacency,
    compute_window_support,
    crop_edge_band,
    map_native_node,
    native_patch_centre_pixel,
    native_window_pair_offset,
    node_class_from_scores,
    process_one_window_for_audit,
    rank_edges_for_row,
    rescale_pixel_align_corners,
)
from models.dinotext.cover_dr.graph import build_directed_topk_graph  # noqa: E402
from segmentation.evaluation.sliding_window_geometry import (  # noqa: E402
    PatchGridSpec,
    SlidingWindowPlan,
    SpatialSize,
)

# Pre-existing structural quirk in models/dinotext/__init__.py -- see
# tests/test_stitching_control_api.py for the full explanation. Repairing
# it here costs nothing and leaves sys.modules state healthy for whatever
# runs next.
sys.modules["models"].dinotext = sys.modules["models.dinotext"]


# ---------------------------------------------------------------------------
# Native mapping: canonical stride / aligned / clamped / partial overlap
# ---------------------------------------------------------------------------


def test_canonical_stride_gives_exact_offset():
    # Real production values: patch=14, stride=224 -> offset = 16 patches.
    offset = native_window_pair_offset(224, 0, 0, 0, patch_size=14)
    assert offset == (16, 0)


def test_aligned_overlapping_windows_match_scalar_reference():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(14, 14), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4),
    )
    spec = PatchGridSpec(crop_size=SpatialSize(8, 8), patch_size=SpatialSize(2, 2), grid_size=SpatialSize(4, 4), align_corners=True)
    w0, w1 = plan.windows[0], plan.windows[1]
    for patch_row in range(4):
        for patch_col in range(4):
            expected = spec.native_patch_node(w0, patch_row, patch_col, w1)
            mapping = map_native_node(
                NativeNodeCoordinate(window_index=0, patch_row=patch_row, patch_col=patch_col, grid_size=4),
                target_window_index=1, source_origin_row=w0.origin.row, source_origin_col=w0.origin.col,
                target_origin_row=w1.origin.row, target_origin_col=w1.origin.col, patch_size=2,
            )
            if expected is None:
                assert mapping.aligned is False and mapping.target is None
            else:
                assert mapping.aligned is True
                assert (mapping.target.patch_row, mapping.target.patch_col) == expected


def test_unaligned_clamped_origin_returns_none():
    # image=13, crop=8, stride=4, patch=2: the clamped final window's
    # origin (5) is not reachable from window 0's origin (0) by a
    # patch-size-2 multiple (5 is odd).
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(13, 13), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4),
    )
    windows = [w for w in plan.windows if w.grid_col == 0]
    clamped = [w for w in windows if w.clamped_vertical]
    assert clamped, "fixture must produce a clamped window"
    w0 = windows[0]
    wc = clamped[0]
    assert (w0.origin.row - wc.origin.row) % 2 != 0
    offset = native_window_pair_offset(w0.origin.row, w0.origin.col, wc.origin.row, wc.origin.col, patch_size=2)
    assert offset is None
    mapping = map_native_node(
        NativeNodeCoordinate(window_index=0, patch_row=0, patch_col=0, grid_size=4),
        target_window_index=1, source_origin_row=w0.origin.row, source_origin_col=w0.origin.col,
        target_origin_row=wc.origin.row, target_origin_col=wc.origin.col, patch_size=2,
    )
    assert mapping.aligned is False
    assert mapping.target is None


def test_aligned_clamped_origin_with_patch_size_one():
    # patch_size=1 makes every integer displacement divisible by the
    # patch size, so a clamped window's origin is still natively aligned
    # -- this is the "aligned clamped origin" case required by spec.
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(13, 8), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4),
    )
    windows = [w for w in plan.windows if w.grid_col == 0]
    clamped = [w for w in windows if w.clamped_vertical]
    assert clamped
    w0 = windows[0]
    wc = clamped[0]
    offset = native_window_pair_offset(w0.origin.row, w0.origin.col, wc.origin.row, wc.origin.col, patch_size=1)
    assert offset is not None


def test_partial_overlap_one_endpoint_covered_not_other():
    # A source node near a shared edge maps inside the target grid; a
    # destination node near the far edge maps outside it under the same
    # offset.
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(12, 8), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4),
    )
    w0, w1 = plan.windows[0], plan.windows[1]
    assert (w1.origin.row - w0.origin.row) == 4  # aligned overlap by construction
    covered = map_native_node(
        NativeNodeCoordinate(window_index=0, patch_row=3, patch_col=0, grid_size=4),  # near the shared edge
        target_window_index=1, source_origin_row=w0.origin.row, source_origin_col=w0.origin.col,
        target_origin_row=w1.origin.row, target_origin_col=w1.origin.col, patch_size=2,
    )
    not_covered = map_native_node(
        NativeNodeCoordinate(window_index=0, patch_row=0, patch_col=0, grid_size=4),  # far from window 1
        target_window_index=1, source_origin_row=w0.origin.row, source_origin_col=w0.origin.col,
        target_origin_row=w1.origin.row, target_origin_col=w1.origin.col, patch_size=2,
    )
    assert covered.aligned is True
    assert not_covered.aligned is False


def test_source_only_row_single_window_image():
    # A single-window image has no other window at all: every edge is
    # undefined with reason single_window_image, never approximated.
    edges = build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[1.0 / 12] * 12, observer_count_row=[0] * 12, support_count_row=[0] * 12, window_count=1,
    )
    assert all(not e.defined for e in edges)
    assert all(e.undefined_reason == UNDEFINED_REASON_SINGLE_WINDOW_IMAGE for e in edges)


def test_non_square_image_and_window_plan():
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(10, 18), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4),
    )
    assert plan.grid_rows != plan.grid_cols
    spec = PatchGridSpec(crop_size=SpatialSize(8, 8), patch_size=SpatialSize(2, 2), grid_size=SpatialSize(4, 4), align_corners=True)
    w0 = plan.windows[0]
    for w1 in plan.windows[1:]:
        for patch_row in range(4):
            for patch_col in range(4):
                expected = spec.native_patch_node(w0, patch_row, patch_col, w1)
                offset = native_window_pair_offset(w0.origin.row, w0.origin.col, w1.origin.row, w1.origin.col, patch_size=2)
                if offset is None:
                    assert expected is None
                else:
                    dr, dc = offset
                    tr, tc = patch_row + dr, patch_col + dc
                    if 0 <= tr < 4 and 0 <= tc < 4:
                        assert expected == (tr, tc)
                    else:
                        assert expected is None


def test_no_float_tolerance_near_miss_still_rejected():
    # An off-by-one displacement (15 instead of 14) must never be
    # rounded/quantized to the nearest patch -- it is firmly unaligned.
    assert native_window_pair_offset(15, 0, 0, 0, patch_size=14) is None
    assert native_window_pair_offset(14, 1, 0, 0, patch_size=14) is None


def test_no_nearest_neighbor_fallback_stays_none():
    offset = native_window_pair_offset(1, 1, 0, 0, patch_size=14)
    assert offset is None
    mapping = map_native_node(
        NativeNodeCoordinate(window_index=0, patch_row=0, patch_col=0, grid_size=32),
        target_window_index=1, source_origin_row=1, source_origin_col=1, target_origin_row=0, target_origin_col=0,
        patch_size=14,
    )
    assert mapping.target is None and mapping.aligned is False


# ---------------------------------------------------------------------------
# Vectorized compute_window_support cross-validated against the scalar
# reference (map_native_node + explicit adjacency lookup)
# ---------------------------------------------------------------------------


def _random_graph(seed: int, num_nodes: int, k: int = 12, embed_dim: int = 6):
    g = torch.Generator().manual_seed(seed)
    features = F.normalize(torch.randn(num_nodes, embed_dim, generator=g), dim=-1)
    return build_directed_topk_graph(features, k=k, affinity_power=3.0)


@pytest.mark.parametrize("image_hw,crop,stride,patch,grid", [
    ((14, 14), (8, 8), (4, 4), 2, 4),
    ((13, 13), (8, 8), (4, 4), 2, 4),
    ((10, 18), (8, 8), (4, 4), 2, 4),
    ((13, 8), (8, 8), (4, 4), 1, 8),
])
def test_compute_window_support_matches_scalar_reference(image_hw, crop, stride, patch, grid):
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(*image_hw), crop_size=SpatialSize(*crop), stride=SpatialSize(*stride),
    )
    windows = plan.windows
    graphs = tuple(_random_graph(seed=100 + i, num_nodes=grid * grid) for i in range(len(windows)))
    dense = tuple(compute_dense_adjacency(g) for g in graphs)
    origins = tuple((w.origin.row, w.origin.col) for w in windows)

    for source_index in range(len(windows)):
        observer_count, support_count, reverse_support_count, aligned_mask = compute_window_support(
            source_index, window_origins=origins, graphs=graphs, dense_adjacencies=dense, grid_size=grid, patch_size=patch,
        )
        graph_s = graphs[source_index]
        for node in range(graph_s.num_nodes):
            i_row, i_col = divmod(node, grid)
            for rank in range(graph_s.k):
                dest = int(graph_s.neighbor_indices[node, rank])
                j_row, j_col = divmod(dest, grid)
                ref_observer = 0
                ref_support = 0
                ref_reverse_support = 0
                for t, target_window in enumerate(windows):
                    if t == source_index:
                        continue
                    i_map = map_native_node(
                        NativeNodeCoordinate(window_index=source_index, patch_row=i_row, patch_col=i_col, grid_size=grid),
                        target_window_index=t, source_origin_row=origins[source_index][0], source_origin_col=origins[source_index][1],
                        target_origin_row=origins[t][0], target_origin_col=origins[t][1], patch_size=patch,
                    )
                    j_map = map_native_node(
                        NativeNodeCoordinate(window_index=source_index, patch_row=j_row, patch_col=j_col, grid_size=grid),
                        target_window_index=t, source_origin_row=origins[source_index][0], source_origin_col=origins[source_index][1],
                        target_origin_row=origins[t][0], target_origin_col=origins[t][1], patch_size=patch,
                    )
                    if i_map.target is None or j_map.target is None:
                        continue
                    ref_observer += 1
                    if dense[t][i_map.target.flat_index, j_map.target.flat_index]:
                        ref_support += 1
                    if dense[t][j_map.target.flat_index, i_map.target.flat_index]:
                        ref_reverse_support += 1
                assert int(observer_count[node, rank]) == ref_observer, (source_index, node, rank)
                assert int(support_count[node, rank]) == ref_support, (source_index, node, rank)
                assert int(reverse_support_count[node, rank]) == ref_reverse_support, (source_index, node, rank)


def test_reverse_edge_does_not_count_as_support():
    # A target graph containing only j_v -> i_v (the reverse direction)
    # must not count as support for i -> j.
    plan = SlidingWindowPlan.build(image_size=SpatialSize(12, 12), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    assert plan.window_count == 4
    origins = tuple((w.origin.row, w.origin.col) for w in plan.windows)
    grid = 4
    graphs = []
    for i in range(4):
        features = F.normalize(torch.eye(grid * grid, 6 + i)[:, :6] + 0.01 * (i + 1), dim=-1)
        graphs.append(build_directed_topk_graph(features, k=12, affinity_power=3.0))
    graphs = tuple(graphs)
    dense = tuple(compute_dense_adjacency(g) for g in graphs)
    # Force window 1 to contain ONLY the reverse edge for a chosen pair.
    reverse_only = torch.zeros_like(dense[1])
    reverse_only[3, 2] = True  # 2 -> 3 forward absent, 3 -> 2 present
    dense = (dense[0], reverse_only, dense[2], dense[3])

    observer_count, support_count, reverse_support_count, _ = compute_window_support(
        0, window_origins=origins, graphs=graphs, dense_adjacencies=dense, grid_size=grid, patch_size=2,
    )
    # Node 2's neighbor list may or may not include node 3; if it does,
    # support through window 1 for that specific edge must be zero.
    row2 = graphs[0].neighbor_indices[2].tolist()
    if 3 in row2:
        rank = row2.index(3)
        assert int(support_count[2, rank]) == 0
        if int(observer_count[2, rank]) > 0:
            # window 1 IS a valid observer for this pair and contains only
            # the reverse edge -- confirms reverse_support_count is
            # populated from the same lookup, independent of (never
            # substituting for) the forward support_count.
            assert int(reverse_support_count[2, rank]) >= 1


# ---------------------------------------------------------------------------
# DirectedEdgeSupport / row / window summaries
# ---------------------------------------------------------------------------


def test_directed_edge_support_rejects_support_exceeding_observer():
    with pytest.raises(NativeEdgeSupportError):
        DirectedEdgeSupport(
            source_window_index=0, source_node=0, neighbor_rank=0, destination_node=1, source_graph_weight=0.1,
            observer_count=1, support_count=2, support_fraction=2.0, defined=True, undefined_reason=None,
        )


def test_directed_edge_support_defined_flag_must_match_observer_count():
    with pytest.raises(NativeEdgeSupportError):
        DirectedEdgeSupport(
            source_window_index=0, source_node=0, neighbor_rank=0, destination_node=1, source_graph_weight=0.1,
            observer_count=0, support_count=0, support_fraction=None, defined=True, undefined_reason=None,
        )


def test_directed_edge_support_undefined_reason_required_when_undefined():
    with pytest.raises(NativeEdgeSupportError):
        DirectedEdgeSupport(
            source_window_index=0, source_node=0, neighbor_rank=0, destination_node=1, source_graph_weight=0.1,
            observer_count=0, support_count=0, support_fraction=None, defined=False, undefined_reason=None,
        )


def test_build_row_summary_requires_twelve_edges():
    edges = build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[1.0 / 12] * 12, observer_count_row=[2] * 12, support_count_row=[1] * 12, window_count=3,
    )
    with pytest.raises(NativeEdgeSupportError):
        build_row_summary(edges[:11])


def test_build_row_summary_rejects_mismatched_window_index():
    edges = list(build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[1.0 / 12] * 12, observer_count_row=[2] * 12, support_count_row=[1] * 12, window_count=3,
    ))
    bad = DirectedEdgeSupport(
        source_window_index=1, source_node=0, neighbor_rank=11, destination_node=99, source_graph_weight=0.1,
        observer_count=2, support_count=1, support_fraction=0.5, defined=True, undefined_reason=None,
    )
    edges[-1] = bad
    with pytest.raises(NativeEdgeSupportError):
        build_row_summary(edges)


def test_row_summary_any_all_defined_flags():
    edges = build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[1.0 / 12] * 12, observer_count_row=[2] * 12, support_count_row=[0] * 12, window_count=3,
    )
    summary = build_row_summary(edges)
    assert summary.all_defined is True
    assert summary.any_defined is True
    assert summary.any_defined_zero_support is True
    assert summary.num_support_defined == 12
    assert summary.num_undefined == 0


def test_row_summary_zero_observer_all_undefined():
    edges = build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[1.0 / 12] * 12, observer_count_row=[0] * 12, support_count_row=[0] * 12, window_count=3,
    )
    summary = build_row_summary(edges)
    assert summary.any_defined is False
    assert summary.all_defined is False
    assert summary.any_defined_zero_support is False


def test_build_window_summary_containment_invariant():
    with pytest.raises(NativeEdgeSupportError):
        build_window_summary(
            window_index=0, is_clamped=False, aligned_observer_window_count=1,
            row_summaries=[],
        )


# ---------------------------------------------------------------------------
# Deterministic descriptive ranking
# ---------------------------------------------------------------------------


def _edge(rank, dest, observer, support, weight, defined=True, reason=None):
    fraction = support / observer if defined else None
    return DirectedEdgeSupport(
        source_window_index=0, source_node=0, neighbor_rank=rank, destination_node=dest,
        source_graph_weight=weight, observer_count=observer, support_count=support,
        support_fraction=fraction, defined=defined, undefined_reason=reason,
    )


def test_ranking_defined_before_undefined():
    # Criterion 1 dominates every later criterion: a defined edge always
    # sorts ahead of an undefined one, even a fully-supported defined edge
    # (fraction 1.0) ahead of an undefined edge with no evidence at all --
    # "no observation" is not treated as worse than "confirmed support."
    defined = _edge(0, 1, 4, 4, 0.5)  # fraction 1.0 -- fully supported
    undefined = _edge(1, 2, 0, 0, 0.5, defined=False, reason=UNDEFINED_REASON_NO_ALIGNED_OBSERVER)
    ranked = rank_edges_for_row([defined, undefined])
    assert ranked[0] is defined
    assert ranked[1] is undefined


def test_ranking_lower_fraction_first():
    high = _edge(0, 1, 4, 4, 0.5)  # fraction 1.0
    low = _edge(1, 2, 4, 1, 0.5)  # fraction 0.25
    ranked = rank_edges_for_row([high, low])
    assert ranked[0] is low


def test_ranking_lower_support_count_first_when_fraction_tied():
    a = _edge(0, 1, 8, 2, 0.5)  # fraction 0.25
    b = _edge(1, 2, 4, 1, 0.5)  # fraction 0.25
    ranked = rank_edges_for_row([a, b])
    assert ranked[0] is b  # lower support_count (1 < 2)


def test_ranking_larger_observer_count_first_when_tied_further():
    a = _edge(0, 1, 4, 0, 0.5)  # fraction 0.0, support 0, observer 4
    b = _edge(1, 2, 8, 0, 0.5)  # fraction 0.0, support 0, observer 8
    ranked = rank_edges_for_row([a, b])
    assert ranked[0] is b  # larger observer count first (stronger zero evidence)


def test_ranking_lower_source_weight_first_when_tied_further():
    a = _edge(0, 1, 4, 0, 0.9)
    b = _edge(1, 2, 4, 0, 0.1)
    ranked = rank_edges_for_row([a, b])
    assert ranked[0] is b


def test_ranking_larger_neighbor_rank_first_when_tied_further():
    a = _edge(2, 1, 4, 0, 0.5)
    b = _edge(9, 2, 4, 0, 0.5)
    ranked = rank_edges_for_row([a, b])
    assert ranked[0] is b  # larger rank (9 > 2) first


def test_ranking_lower_destination_node_final_tiebreak():
    a = _edge(3, 10, 4, 0, 0.5)
    b = _edge(3, 5, 4, 0, 0.5)
    ranked = rank_edges_for_row([a, b])
    assert ranked[0] is b  # lower destination_node (5 < 10)


def test_ranking_never_applies_an_edit():
    edges = build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[float(i) / 12 for i in range(12)], observer_count_row=[3] * 12,
        support_count_row=[i % 4 for i in range(12)], window_count=3,
    )
    ranked = rank_edges_for_row(edges)
    assert set(ranked) == set(edges)  # a pure reordering, nothing added/removed/mutated
    assert len(ranked) == 12


# ---------------------------------------------------------------------------
# crop_edge_band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distance,expected", [
    (0, "0"), (1, "1"), (2, "2"), (3, "3-4"), (4, "3-4"), (5, "5-7"),
    (6, "5-7"), (7, "5-7"), (8, ">=8"), (100, ">=8"),
])
def test_crop_edge_band_boundaries(distance, expected):
    assert crop_edge_band(distance) == expected


# ---------------------------------------------------------------------------
# GT reachability diagnostics -- computed after support is frozen, never
# influences support/ranking
# ---------------------------------------------------------------------------


def test_node_class_from_scores_is_argmax():
    scores = torch.tensor([-5.0, 3.0, 0.1])
    assert node_class_from_scores(scores) == 1


def test_native_patch_centre_pixel_matches_identity_declared_formula():
    # centre = origin + patch_size*index + patch_size//2 (patch_size=14
    # gives the spec's own origin+14*index+7 exactly).
    row, col = native_patch_centre_pixel(0, 0, 0, 0, patch_size=14)
    assert (row, col) == (7, 7)
    row, col = native_patch_centre_pixel(100, 200, 2, 3, patch_size=14)
    assert (row, col) == (100 + 28 + 7, 200 + 42 + 7)


def test_rescale_pixel_align_corners_identity_when_shapes_equal():
    row, col = rescale_pixel_align_corners(5, 7, from_hw=(20, 20), to_hw=(20, 20))
    assert (row, col) == (5, 7)


def test_rescale_pixel_align_corners_endpoints_preserved():
    row, col = rescale_pixel_align_corners(0, 0, from_hw=(10, 10), to_hw=(20, 20))
    assert (row, col) == (0, 0)
    row, col = rescale_pixel_align_corners(9, 9, from_hw=(10, 10), to_hw=(20, 20))
    assert (row, col) == (19, 19)


def test_build_row_correctness_record_respects_ignore_index():
    gt = torch.full((4, 4), 255, dtype=torch.int64)
    stitched = torch.zeros((4, 4), dtype=torch.int64)
    scores = torch.tensor([1.0, -1.0])
    record = build_row_correctness_record(
        window_index=0, source_node=0, window_origin_row=0, window_origin_col=0,
        patch_row=0, patch_col=0, patch_size=2, source_patch_scores_row=scores,
        canonical_stitched_label_map=stitched, gt_label_map=gt, ignore_index=255,
        processed_hw=(4, 4), ori_hw=(4, 4),
    )
    assert record.gt_ignored is True
    assert record.source_row_misclassified is None
    assert record.canonical_stitched_misclassified is None


def test_build_row_correctness_record_detects_misclassification():
    gt = torch.zeros((4, 4), dtype=torch.int64)
    # centre = origin + patch_row*patch_size + patch_size//2 = 0+0*2+1 = (1,1)
    gt[1, 1] = 1  # GT class 1 at the sampled pixel
    stitched = torch.zeros((4, 4), dtype=torch.int64)  # predicts class 0 everywhere -> wrong
    scores = torch.tensor([5.0, -5.0])  # source argmax -> class 0 -> wrong
    record = build_row_correctness_record(
        window_index=0, source_node=0, window_origin_row=0, window_origin_col=0,
        patch_row=0, patch_col=0, patch_size=2, source_patch_scores_row=scores,
        canonical_stitched_label_map=stitched, gt_label_map=gt, ignore_index=255,
        processed_hw=(4, 4), ori_hw=(4, 4),
    )
    assert record.gt_ignored is False
    assert record.source_row_misclassified is True
    assert record.canonical_stitched_misclassified is True


# ---------------------------------------------------------------------------
# Shared per-window execution
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self, patch_size=2, embed_dim=6, class_count=4, seed=0):
        super().__init__()
        self.patch_size = patch_size
        g = torch.Generator().manual_seed(seed)
        self.feature_proj = torch.randn(3, embed_dim, generator=g)
        self.score_proj = torch.randn(embed_dim, class_count, generator=g)

    def generate_patch_snapshot(self, crop, text_embedding):
        del text_embedding
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
        batch, _n, classes = patch_scores.shape
        grid_h, grid_w = grid_hw
        simmap = patch_scores.reshape(batch, grid_h, grid_w, classes).permute(0, 3, 1, 2)
        mask = torch.sigmoid(simmap)
        return F.interpolate(mask, tuple(output_hw), mode="bilinear", align_corners=True)


def _fake_inference(model):
    return types.SimpleNamespace(model=model, text_embedding=None, num_classes=4, align_corners=False)


def test_process_one_window_for_audit_diagnostics_off_skips_propagation():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(8, 8), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    window = plan.windows[0]
    model = _TinyModel(patch_size=2, embed_dim=6, class_count=4)
    inference = _fake_inference(model)
    image_tensor = torch.rand(1, 3, 8, 8)

    context, telemetry = process_one_window_for_audit(
        inference, image_tensor, window, k=12, alpha=0.98, steps=5, affinity_power=3.0, need_diagnostics=False,
    )
    assert telemetry.propagation_calls == 0
    assert telemetry.probability_interpolation_calls == 0
    assert context.patch_scores is None
    assert context.probability_crop is None
    assert context.graph.num_nodes == 16


def test_process_one_window_for_audit_diagnostics_on_runs_propagation_once():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(8, 8), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    window = plan.windows[0]
    model = _TinyModel(patch_size=2, embed_dim=6, class_count=4)
    inference = _fake_inference(model)
    image_tensor = torch.rand(1, 3, 8, 8)

    context, telemetry = process_one_window_for_audit(
        inference, image_tensor, window, k=12, alpha=0.98, steps=5, affinity_power=3.0, need_diagnostics=True,
    )
    assert telemetry.propagation_calls == 1
    assert telemetry.probability_interpolation_calls == 1
    assert context.patch_scores is not None
    assert context.probability_crop is not None
    assert context.probability_crop.shape == (1, 4, 8, 8)


def test_process_one_window_for_audit_no_second_graph_or_propagation_per_call():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(8, 8), crop_size=SpatialSize(8, 8), stride=SpatialSize(4, 4))
    window = plan.windows[0]
    model = _TinyModel(patch_size=2, embed_dim=6, class_count=4)
    inference = _fake_inference(model)
    image_tensor = torch.rand(1, 3, 8, 8)
    calls = {"snapshot": 0}
    real_snapshot = model.generate_patch_snapshot

    def counting_snapshot(*a, **k):
        calls["snapshot"] += 1
        return real_snapshot(*a, **k)

    model.generate_patch_snapshot = counting_snapshot
    process_one_window_for_audit(
        inference, image_tensor, window, k=12, alpha=0.98, steps=5, affinity_power=3.0, need_diagnostics=True,
    )
    assert calls["snapshot"] == 1


# ---------------------------------------------------------------------------
# Scientific isolation: AST-identifier scan (never a raw text search, which
# would false-positive on this module's own prose disclaiming these
# techniques) for prohibited identifiers.
# ---------------------------------------------------------------------------

_FORBIDDEN_IDENTIFIER_SUBSTRINGS = (
    "t4", "dcr", "sur", "pruning", "sherman", "adjoint", "pamr", "cgls",
    "consensus", "vote", "threshold", "gate", "centrality",
)
_ALLOWED_FALSE_POSITIVES = {
    # legitimate identifiers that happen to contain a forbidden substring
    # as a harmless English-word fragment, never the prohibited technique.
    "source_row_misclassified", "canonical_stitched_misclassified",
    "misclassified_source", "misclassified_stitched", "misclassified",
}


def _collect_identifiers(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_no_prohibited_identifiers_in_native_edge_support_module():
    # Token-based, not substring-based: a raw substring search would
    # false-positive on legitimate identifiers that merely contain a
    # forbidden fragment as part of an ordinary English/code word (e.g.
    # "propagate" contains "gate", "measure" would contain "sur") --
    # exactly the false-positive class this scan must avoid, per the
    # lesson already documented in tests/test_stitching_control_api.py's
    # own prohibited-terms scan.
    import models.dinotext.cover_dr.native_edge_support as module

    identifiers = _collect_identifiers(module)
    for name in identifiers:
        if name in _ALLOWED_FALSE_POSITIVES:
            continue
        tokens = set(name.lower().split("_"))
        for forbidden in _FORBIDDEN_IDENTIFIER_SUBSTRINGS:
            assert forbidden not in tokens, f"identifier {name!r} contains forbidden token {forbidden!r}"


def test_no_nn_module_subclass_in_native_edge_support_module():
    import models.dinotext.cover_dr.native_edge_support as module

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                assert base_name != "Module", f"class {node.name} must not subclass nn.Module -- no learned parameters"


# ---------------------------------------------------------------------------
# Least-support tie detection
# ---------------------------------------------------------------------------


def test_least_support_unique_when_no_tie():
    edges = build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[float(i) / 12 for i in range(12)], observer_count_row=[12] * 12,
        support_count_row=list(range(12)), window_count=3,  # strictly increasing -> unique minimum
    )
    summary = build_row_summary(edges)
    assert summary.any_defined is True
    assert summary.least_support_tied is False


def test_least_support_tied_when_two_edges_share_evidence():
    # Two edges with identical (fraction, support_count, observer_count)
    # as the ranked-first edge -- tied on the primary evidence criteria,
    # even though the deterministic ranking still picks one via the
    # weight/rank/destination tiebreaks.
    support = [0] * 12  # all zero support, all observer_count=4 -> all tied on (0.0, 0, 4)
    edges = build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[0.5] * 12, observer_count_row=[4] * 12, support_count_row=support, window_count=3,
    )
    summary = build_row_summary(edges)
    assert summary.any_defined is True
    assert summary.least_support_tied is True


def test_least_support_tied_is_none_when_no_edge_defined():
    edges = build_edge_records_for_row(
        window_index=0, source_node=0, neighbor_indices_row=list(range(1, 13)),
        weights_row=[0.5] * 12, observer_count_row=[0] * 12, support_count_row=[0] * 12, window_count=1,
    )
    summary = build_row_summary(edges)
    assert summary.any_defined is False
    assert summary.least_support_tied is None


# ---------------------------------------------------------------------------
# Decision-output classification: scientific constants come from the real
# identity, never duplicated as bare literals in this test.
# ---------------------------------------------------------------------------


def _base_funnel(**overrides):
    from src.native_edge_support_checkpoint import FUNNEL_KEYS

    funnel = {key: 0 for key in FUNNEL_KEYS}
    funnel.update(overrides)
    return funnel


def test_decision_inconclusive_below_minimum_misclassified_rows():
    from src.native_edge_support_identity import load_identity
    from src.native_edge_support_report import classify_reachability

    identity = load_identity(repo_root=ROOT)
    minimum = identity["decision"]["minimum_misclassified_rows_for_conclusive"]
    funnel = _base_funnel(
        directed_edges=1000, edges_with_observer=900, misclassified_rows=minimum - 1,
        misclassified_rows_any_defined=0, graph_rows=100, correct_rows_any_defined=0,
    )
    outcome, rationale = classify_reachability(funnel, identity)
    assert outcome == "INCONCLUSIVE"
    assert "minimum" in rationale


def test_decision_alignment_limited_when_undefined_dominates():
    from src.native_edge_support_identity import load_identity
    from src.native_edge_support_report import classify_reachability

    identity = load_identity(repo_root=ROOT)
    minimum = identity["decision"]["minimum_misclassified_rows_for_conclusive"]
    # 90% of edges undefined -- well above the identity's alignment-limited threshold.
    funnel = _base_funnel(
        directed_edges=1000, edges_with_observer=100, misclassified_rows=minimum + 10,
        misclassified_rows_any_defined=5, graph_rows=200, correct_rows_any_defined=5,
    )
    outcome, _ = classify_reachability(funnel, identity)
    assert outcome == "ALIGNMENT_LIMITED"


def test_decision_reachable_when_support_concentrated_at_errors():
    from src.native_edge_support_identity import load_identity
    from src.native_edge_support_report import classify_reachability

    identity = load_identity(repo_root=ROOT)
    minimum = identity["decision"]["minimum_misclassified_rows_for_conclusive"]
    # Almost all wrong rows are support-defined; almost none of the
    # correct rows are -- support clearly concentrates at the errors.
    funnel = _base_funnel(
        directed_edges=1000, edges_with_observer=900, misclassified_rows=minimum + 30,
        misclassified_rows_any_defined=minimum + 28, graph_rows=1000, correct_rows_any_defined=2,
    )
    outcome, rationale = classify_reachability(funnel, identity)
    assert outcome == "REACHABLE"
    assert "risk ratio" in rationale


def test_decision_structurally_unreachable_when_support_concentrated_at_correct():
    from src.native_edge_support_identity import load_identity
    from src.native_edge_support_report import classify_reachability

    identity = load_identity(repo_root=ROOT)
    minimum = identity["decision"]["minimum_misclassified_rows_for_conclusive"]
    # Almost none of the wrong rows are support-defined; almost all of
    # the correct rows are -- support concentrates where things already work.
    funnel = _base_funnel(
        directed_edges=1000, edges_with_observer=900, misclassified_rows=minimum + 30,
        misclassified_rows_any_defined=2, graph_rows=1000, correct_rows_any_defined=500,
    )
    outcome, _ = classify_reachability(funnel, identity)
    assert outcome == "STRUCTURALLY_UNREACHABLE"


def test_decision_never_edits_or_mutates_funnel():
    from src.native_edge_support_identity import load_identity
    from src.native_edge_support_report import classify_reachability

    identity = load_identity(repo_root=ROOT)
    minimum = identity["decision"]["minimum_misclassified_rows_for_conclusive"]
    funnel = _base_funnel(
        directed_edges=1000, edges_with_observer=900, misclassified_rows=minimum + 5,
        misclassified_rows_any_defined=3, graph_rows=500, correct_rows_any_defined=100,
    )
    before = dict(funnel)
    classify_reachability(funnel, identity)
    assert funnel == before
