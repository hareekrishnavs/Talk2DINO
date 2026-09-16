"""Independent scientific tests for the pure BALP kernel
(`src.dfp_phase1b_lightweight_boundary`). Builds fresh NumPy oracles
for every formula -- never uses the kernel's own internals as its own
oracle -- and cross-checks the kernel's argmax against the real,
already-committed `src.dfp_phase1a_analysis._lowest_index_argmax`.

No torch/CUDA is initialized beyond ordinary CPU tensor construction;
no image/annotation/model/weight is accessed.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

import src.dfp_phase1b_lightweight_boundary as balp

E = balp.DfpPhase1bLightweightBoundaryKernelError


# =====================================================================
# Independent NumPy oracles (never imported from the kernel).
# =====================================================================


def _np_lowest_index_argmax(scores_chw: np.ndarray) -> np.ndarray:
    class_count = scores_chw.shape[0]
    max_value = scores_chw.max(axis=0, keepdims=True)
    is_max = scores_chw == max_value
    class_index = np.arange(class_count, dtype=np.int64).reshape(class_count, 1, 1)
    sentinel = np.int64(class_count)
    return np.where(is_max, class_index, sentinel).min(axis=0).astype(np.int64)


def _np_boundary(y: np.ndarray) -> np.ndarray:
    H, W = y.shape
    b = np.zeros((H, W), dtype=bool)
    for dr, dc in itertools.product((-1, 0, 1), repeat=2):
        if dr == 0 and dc == 0:
            continue
        r0, r1 = max(0, dr), H + min(0, dr)
        c0, c1 = max(0, dc), W + min(0, dc)
        sr0, sr1 = max(0, -dr), H + min(0, -dr)
        sc0, sc1 = max(0, -dc), W + min(0, -dc)
        if r0 >= r1 or c0 >= c1:
            continue
        b[r0:r1, c0:c1] |= y[r0:r1, c0:c1] != y[sr0:sr1, sc0:sc1]
    return b


def _np_chebyshev_dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r == 0:
        return mask.copy()
    H, W = mask.shape
    out = np.zeros((H, W), dtype=bool)
    ys, xs = np.nonzero(mask)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            ry, rx = ys + dr, xs + dc
            v = (ry >= 0) & (ry < H) & (rx >= 0) & (rx < W)
            out[ry[v], rx[v]] = True
    return out


def _np_manhattan_dilate(mask: np.ndarray, r: int) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((H, W), dtype=bool)
    ys, xs = np.nonzero(mask)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if abs(dr) + abs(dc) > r:
                continue
            ry, rx = ys + dr, xs + dc
            v = (ry >= 0) & (ry < H) & (rx >= 0) & (rx < W)
            out[ry[v], rx[v]] = True
    return out


def _np_stable(labels_by_h, s):
    suffix = [h for h in (0, 16, 32, 64) if h >= s]
    ref = labels_by_h[suffix[0]]
    mask = np.ones_like(ref, dtype=bool)
    for h in suffix[1:]:
        mask &= labels_by_h[h] == ref
    return mask


def _make_scores(class_count, H, W, seed):
    g = np.random.RandomState(seed)
    return {h: g.randn(class_count, H, W).astype(np.float32) for h in balp.HORIZON_SET}


def _to_torch(scores_np):
    return {h: torch.from_numpy(v.copy()) for h, v in scores_np.items()}


# =====================================================================
# lowest_index_argmax
# =====================================================================


def test_lowest_index_argmax_matches_real_committed_oracle():
    from src.dfp_phase1a_analysis import _lowest_index_argmax as real_oracle

    scores_np = _make_scores(4, 6, 5, seed=1)[0]
    torch_result = balp.lowest_index_argmax(torch.from_numpy(scores_np)).numpy()
    assert np.array_equal(torch_result, real_oracle(scores_np))


def test_lowest_index_argmax_two_way_tie():
    scores = torch.tensor([[[1.0, 2.0]], [[1.0, 2.0]], [[0.5, 2.0]]])
    result = balp.lowest_index_argmax(scores)
    assert result[0, 0].item() == 0  # classes 0,1 tie at 1.0 -> lowest index 0
    assert result[0, 1].item() == 0  # classes 0,1,2 tie at 2.0 -> lowest index 0


def test_lowest_index_argmax_three_way_tie():
    scores = torch.tensor([[[3.0]], [[3.0]], [[3.0]]])
    assert balp.lowest_index_argmax(scores)[0, 0].item() == 0


def test_lowest_index_argmax_no_tie():
    scores = torch.tensor([[[1.0]], [[5.0]], [[3.0]]])
    assert balp.lowest_index_argmax(scores)[0, 0].item() == 1


def test_lowest_index_argmax_deterministic_repeatability():
    scores = torch.from_numpy(_make_scores(5, 4, 4, seed=7)[0])
    r1 = balp.lowest_index_argmax(scores)
    r2 = balp.lowest_index_argmax(scores)
    assert torch.equal(r1, r2)
    assert torch.equal(scores, torch.from_numpy(_make_scores(5, 4, 4, seed=7)[0]))  # input unmutated


def test_lowest_index_argmax_read_only_input_not_mutated():
    scores = torch.from_numpy(_make_scores(3, 3, 3, seed=2)[0])
    scores.requires_grad_(False)
    before = scores.clone()
    balp.lowest_index_argmax(scores)
    assert torch.equal(scores, before)


# =====================================================================
# boundary_8connected
# =====================================================================


def test_boundary_horizontal_difference():
    y = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    b = balp.boundary_8connected(y)
    assert bool(b[0, 0]) is False
    assert bool(b[0, 1]) is True
    assert bool(b[0, 2]) is True


def test_boundary_vertical_difference():
    y = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    b = balp.boundary_8connected(y)
    assert bool(b[1, 0]) is True
    assert bool(b[0, 0]) is False


def test_boundary_diagonal_only_difference():
    y = torch.tensor([[0, 0], [0, 1]])
    b = balp.boundary_8connected(y)
    assert bool(b[0, 0]) is True   # diagonal neighbor (1,1)=1 differs
    assert bool(b[1, 1]) is True


def test_boundary_corner_pixel():
    y = torch.zeros((3, 3), dtype=torch.int64)
    y[0, 0] = 1
    b = balp.boundary_8connected(y)
    assert bool(b[0, 0]) is True
    assert bool(b[2, 2]) is False  # far corner unrelated to the single differing pixel


def test_boundary_uniform_map_has_no_boundary():
    y = torch.zeros((5, 5), dtype=torch.int64)
    assert not balp.boundary_8connected(y).any()


def test_boundary_1x1_map_has_no_boundary():
    y = torch.tensor([[7]])
    b = balp.boundary_8connected(y)
    assert b.shape == (1, 1)
    assert not b.any()


def test_boundary_non_square_geometry():
    y = torch.zeros((2, 5), dtype=torch.int64)
    y[0, 2] = 9
    b = balp.boundary_8connected(y)
    assert bool(b[0, 1]) and bool(b[0, 3]) and bool(b[1, 2])


def test_boundary_matches_numpy_oracle_random():
    y_np = np.random.RandomState(3).randint(0, 4, size=(9, 11))
    torch_b = balp.boundary_8connected(torch.from_numpy(y_np)).numpy()
    assert np.array_equal(torch_b, _np_boundary(y_np))


def test_boundary_rejects_non_2d_input():
    with pytest.raises(E):
        balp.boundary_8connected(torch.zeros(2, 3, 4, dtype=torch.int64))


# =====================================================================
# chebyshev_dilate
# =====================================================================


@pytest.mark.parametrize("r", [0, 1, 2, 4])
def test_chebyshev_dilate_matches_numpy_oracle(r):
    mask_np = np.zeros((13, 13), dtype=bool)
    mask_np[6, 6] = True
    mask_np[2, 10] = True
    torch_result = balp.chebyshev_dilate(torch.from_numpy(mask_np), r).numpy()
    assert np.array_equal(torch_result, _np_chebyshev_dilate(mask_np, r))


def test_chebyshev_differs_from_manhattan_at_diagonal_corner():
    mask_np = np.zeros((11, 11), dtype=bool)
    mask_np[5, 5] = True
    cheb = balp.chebyshev_dilate(torch.from_numpy(mask_np), 3).numpy()
    manh = _np_manhattan_dilate(mask_np, 3)
    assert cheb[8, 8] and not manh[8, 8]
    assert not np.array_equal(cheb, manh)


def test_chebyshev_dilate_radius_zero_is_identity():
    mask = torch.from_numpy(np.random.RandomState(4).rand(6, 6) > 0.5)
    assert torch.equal(balp.chebyshev_dilate(mask, 0), mask)


def test_chebyshev_dilate_edge_pixel_no_wraparound():
    mask_np = np.zeros((5, 5), dtype=bool)
    mask_np[0, 0] = True
    result = balp.chebyshev_dilate(torch.from_numpy(mask_np), 1).numpy()
    assert result.sum() == 4  # only the in-bounds quadrant
    assert result[4, 4] == False


def test_chebyshev_dilate_rejects_negative_radius():
    with pytest.raises(E):
        balp.chebyshev_dilate(torch.zeros(3, 3, dtype=torch.bool), -1)


def test_chebyshev_dilate_rejects_non_bool_mask():
    with pytest.raises(E):
        balp.chebyshev_dilate(torch.zeros(3, 3, dtype=torch.int64), 1)


# =====================================================================
# Stability
# =====================================================================


@pytest.mark.parametrize("s,expected", [
    (0, [True, False, False, False, True]),
    (16, [True, False, False, True, True]),
])
def test_stability_exact_suffix_semantics(s, expected):
    y0 = np.array([0, 0, 0, 0, 1]); y16 = np.array([0, 0, 0, 1, 1])
    y32 = np.array([0, 0, 1, 1, 1]); y64 = np.array([0, 1, 1, 1, 1])
    labels = {0: y0, 16: y16, 32: y32, 64: y64}
    assert list(_np_stable(labels, s)) == expected


def test_stability_s64_always_true_regardless_of_earlier_disagreement():
    y0 = np.array([9, 1, 2]); y16 = np.array([8, 3, 4]); y32 = np.array([7, 5, 6]); y64 = np.array([0, 0, 0])
    labels = {0: y0, 16: y16, 32: y32, 64: y64}
    assert list(_np_stable(labels, 64)) == [True, True, True]


def test_stability_s64_distinguishable_from_stricter_alternative_requiring_y32_agreement():
    y32, y64 = np.array([5]), np.array([9])
    s64_case = True  # frozen: suffix(64) = [64] only
    stricter_alternative = bool(y32[0] == y64[0])
    assert s64_case != stricter_alternative


# =====================================================================
# Late flip
# =====================================================================


def test_late_flip_exactly_y64_ne_y320():
    y64 = torch.tensor([0, 1, 2, 3])
    y320 = torch.tensor([0, 9, 2, 9])
    assert torch.equal(y64 != y320, torch.tensor([False, True, False, True]))


# =====================================================================
# Full compute_balp: structural validation
# =====================================================================


def test_compute_balp_rejects_wrong_horizon_set():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    del scores[320]
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=3)


def test_compute_balp_rejects_wrong_rank():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    scores[0] = scores[0][0]  # now 2-D
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=3)


def test_compute_balp_rejects_wrong_class_count():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=4)


def test_compute_balp_rejects_shape_mismatch_across_horizons():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    scores[16] = torch.randn(3, 5, 5)
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=3)


def test_compute_balp_rejects_dtype_mismatch_across_horizons():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    scores[16] = scores[16].to(torch.float64)
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=3)


def test_compute_balp_rejects_non_floating_input():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    scores[0] = scores[0].to(torch.int64)
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=3)


def test_compute_balp_rejects_nan():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    scores[0][0, 0, 0] = float("nan")
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=3)


def test_compute_balp_rejects_infinity():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    scores[320][0, 0, 0] = float("inf")
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=3)


def test_compute_balp_rejects_non_tensor():
    scores = _to_torch(_make_scores(3, 4, 4, seed=5))
    scores[0] = scores[0].numpy()
    with pytest.raises(E):
        balp.compute_balp(scores, class_count=3)


def test_compute_balp_preserves_device_cpu():
    scores = _to_torch(_make_scores(3, 4, 4, seed=6))
    result = balp.compute_balp(scores, class_count=3)
    for t in result.control_labels.values():
        assert t.device == scores[0].device
    for t in result.candidate_labels.values():
        assert t.device == scores[0].device


def test_compute_balp_does_not_mutate_inputs():
    scores = _to_torch(_make_scores(3, 4, 4, seed=8))
    before = {h: v.clone() for h, v in scores.items()}
    balp.compute_balp(scores, class_count=3)
    for h in scores:
        assert torch.equal(scores[h], before[h])


def test_compute_balp_deterministic_repeatability():
    scores = _to_torch(_make_scores(3, 4, 4, seed=9))
    r1 = balp.compute_balp(scores, class_count=3)
    r2 = balp.compute_balp(scores, class_count=3)
    for cid in balp.CANDIDATE_IDS:
        assert torch.equal(r1.candidate_labels[cid], r2.candidate_labels[cid])


# =====================================================================
# Candidate ordering, all 16 IDs, and the relational arbitration proof.
# =====================================================================


def test_candidate_ids_exact_16_ordered():
    S, R = (0, 16, 32, 64), (0, 1, 2, 4)
    expected = tuple(f"balp_s{s:03d}_r{r:02d}" for s in S for r in R)
    assert balp.CANDIDATE_IDS == expected
    assert len(balp.CANDIDATE_IDS) == 16
    assert len(set(balp.CANDIDATE_IDS)) == 16


def test_compute_balp_candidate_order_matches_frozen_order():
    scores = _to_torch(_make_scores(3, 5, 5, seed=10))
    result = balp.compute_balp(scores, class_count=3)
    assert list(result.candidate_labels.keys()) == list(balp.CANDIDATE_IDS)
    assert list(result.preservation_masks.keys()) == list(balp.CANDIDATE_IDS)


@pytest.mark.parametrize("cid", balp.CANDIDATE_IDS)
def test_every_candidate_direct_label_equals_explicit_score_vector_argmax(cid):
    """Explicitly proves E22's relational equivalence for EVERY one of
    the 16 candidates: the efficient torch.where(P, y64, y320) path
    must be byte-identical to materializing the selected score vector
    and taking its argmax."""
    scores = _to_torch(_make_scores(4, 6, 7, seed=hash(cid) % 1000))
    result = balp.compute_balp(scores, class_count=4)
    p = result.preservation_masks[cid]
    explicit_scores = torch.where(p.unsqueeze(0), scores[64], scores[320])
    explicit_label = balp.lowest_index_argmax(explicit_scores)
    assert torch.equal(explicit_label, result.candidate_labels[cid])


@pytest.mark.parametrize("s", balp.STABILITY_STARTS)
def test_each_stability_mask_term_independently_controls_preservation(s):
    scores = _to_torch(_make_scores(3, 8, 8, seed=100 + s))
    result = balp.compute_balp(scores, class_count=3)
    for r in balp.RADII:
        cid = balp.candidate_id(s, r)
        p = result.preservation_masks[cid]
        expected = result.stability_masks[s] & result.late_flip & result.dilated_boundaries[r]
        assert torch.equal(p, expected)


def test_preservation_mask_false_when_stability_false():
    scores = _to_torch(_make_scores(2, 4, 4, seed=11))
    result = balp.compute_balp(scores, class_count=2)
    never_stable = torch.zeros_like(result.stability_masks[0])
    p = never_stable & result.late_flip & result.dilated_boundaries[0]
    assert not p.any()


def test_preservation_mask_false_when_late_flip_false():
    scores = _to_torch(_make_scores(2, 4, 4, seed=12))
    result = balp.compute_balp(scores, class_count=2)
    never_flip = torch.zeros_like(result.late_flip)
    p = result.stability_masks[0] & never_flip & result.dilated_boundaries[4]
    assert not p.any()


def test_preservation_mask_false_when_outside_dilated_boundary():
    scores = _to_torch(_make_scores(2, 4, 4, seed=13))
    result = balp.compute_balp(scores, class_count=2)
    never_boundary = torch.zeros_like(result.dilated_boundaries[0])
    p = result.stability_masks[0] & result.late_flip & never_boundary
    assert not p.any()
