"""Tests for the opt-in leave-one-window-out consensus / T0-T4 audit.

Loaded via file-path import with lightweight stand-ins for mmcv/utils
(mirroring tests/test_image_window_cache.py) so these tests never require
mmcv, cv2, or a CUDA device except where explicitly guarded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation"
GEOMETRY_PATH = EVAL_DIR / "sliding_window_geometry.py"
CACHE_PATH = EVAL_DIR / "window_cache.py"
AUDIT_PATH = EVAL_DIR / "t4_audit.py"
SEGMENTATION_PATH = EVAL_DIR / "dinotext_seg.py"
COVER_DR_PACKAGE_PATH = ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr"


def _install_segmentation_package_stub() -> None:
    if "segmentation" not in sys.modules:
        segmentation = types.ModuleType("segmentation")
        segmentation.__path__ = [str(EVAL_DIR.parent)]
        sys.modules["segmentation"] = segmentation
    if "segmentation.evaluation" not in sys.modules:
        evaluation = types.ModuleType("segmentation.evaluation")
        evaluation.__path__ = [str(EVAL_DIR)]
        sys.modules["segmentation.evaluation"] = evaluation


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_segmentation_package_stub()
geometry = _load("segmentation.evaluation.sliding_window_geometry", GEOMETRY_PATH)
wc = _load("segmentation.evaluation.window_cache", CACHE_PATH)
t4 = _load("segmentation.evaluation.t4_audit", AUDIT_PATH)

SpatialSize = geometry.SpatialSize
SlidingWindowPlan = geometry.SlidingWindowPlan


def _load_cover_dr_package():
    if "models" not in sys.modules:
        models_stub = types.ModuleType("models")
        models_stub.__path__ = []
        sys.modules["models"] = models_stub
    if "models.dinotext" not in sys.modules:
        dinotext_stub = types.ModuleType("models.dinotext")
        dinotext_stub.__path__ = []
        sys.modules["models.dinotext"] = dinotext_stub
    spec = importlib.util.spec_from_file_location(
        "models.dinotext.cover_dr",
        COVER_DR_PACKAGE_PATH / "__init__.py",
        submodule_search_locations=[str(COVER_DR_PACKAGE_PATH)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cover_dr = _load_cover_dr_package()
from src.rwr_reproduction_identity import load_identity  # noqa: E402

IDENTITY = load_identity(repo_root=ROOT)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _plan(image=(6, 6), crop=(4, 4), stride=(2, 2)):
    return SlidingWindowPlan.build(
        image_size=SpatialSize(*image), crop_size=SpatialSize(*crop), stride=SpatialSize(*stride)
    )


def _make_state(window, s0_rows, p_rows, patch_grid, class_count, *, seed=0, k=2):
    n = len(s0_rows)
    s0 = torch.tensor(s0_rows, dtype=torch.float32)
    p = torch.tensor(p_rows, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    feats = F.normalize(torch.randn(n, 5, generator=generator), dim=-1)
    indices = torch.zeros(n, k, dtype=torch.int64)
    weights = torch.full((n, k), 1.0 / k)
    affinities = torch.ones(n, k)
    fallback = torch.zeros(n, dtype=torch.bool)
    graph = wc.GraphSnapshot(indices, weights, affinities, fallback, n, k, 3.0)
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    return wc.CachedWindowState(
        geometry=window, window_index=window.index, patch_grid_shape=patch_grid,
        class_count=class_count, s0=s0, dino_features=feats, graph=graph,
        propagated_scores=p, solver_summary=telemetry,
    )


def _context(plan, class_count, patch_grid, cache):
    return wc.FirstPassImageContext(
        image_size=plan.image_size, plan=plan, class_count=class_count,
        common_patch_grid_shape=patch_grid, expected_window_count=plan.window_count,
        cached_window_count=plan.window_count, min_coverage=1, max_coverage=plan.window_count,
        stitched_scores=torch.zeros(1, class_count, *plan.image_size.as_tuple()),
        cache_total_bytes=cache.total_bytes(), pass_summary=wc._summarize_telemetry([]),
    )


def _overlap_cache_and_context(
    *, source_s0, source_p, other_s0, other_p, class_count=3, grid=(2, 2),
    image=(6, 6), crop=(4, 4), stride=(2, 2),
):
    """4-window, heavily overlapping fixture: window 0 is the 'source' with
    a distinct signal; windows 1-3 share ``other_s0``/``other_p``."""
    plan = _plan(image=image, crop=crop, stride=stride)
    n = grid[0] * grid[1]
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        if window.index == 0:
            cache.append(_make_state(window, [source_s0] * n, [source_p] * n, grid, class_count, seed=window.index))
        else:
            cache.append(_make_state(window, [other_s0] * n, [other_p] * n, grid, class_count, seed=window.index))
    cache.seal()
    context = _context(plan, class_count, grid, cache)
    return cache, context


REVERSAL_FIXTURE = dict(
    source_s0=[5.0, 0.0, 0.0],   # u = class 0
    source_p=[-10.0, 10.0, 0.0],  # d = class 1, high confidence
    other_s0=[5.0, 0.0, 0.0],
    other_p=[1.0, 0.0, 0.0],      # y = class 0, mild confidence
)


def _deep_overlap_observation(signal):
    matches = [o for o in signal.observations if o.source_window_id == 0]
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------------------
# 1-10: core funnel semantics
# ---------------------------------------------------------------------------


def test_source_window_excluded_from_its_own_jury():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert 0 not in obs.other_covering_window_ids
    assert set(obs.other_window_labels) == {0}  # unanimous WITHOUT the source's dissenting class 1


def test_one_other_window_insufficient_for_t1():
    # 2x2 grid but drop overlap so only 1 other window ever covers a node:
    # use stride == crop (no overlap at all) -> zero other coverage anywhere.
    cache, context = _overlap_cache_and_context(
        **REVERSAL_FIXTURE, image=(8, 8), crop=(4, 4), stride=(4, 4),
    )
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert signal.funnel_counts.t1 == 0
    assert signal.funnel_counts.t0 > 0


def test_exactly_two_other_agreeing_windows_satisfy_t1_and_t2():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert len(obs.other_covering_window_ids) >= 2
    assert obs.t1 is True
    assert obs.t2 is True


def test_strict_unanimity_not_majority_vote():
    # window 0 (source) is unanimous-with-itself trivially; make window 0's
    # OWN "others" {1,2,3} a 2-vs-1 majority (window 1 dissents) so T1 holds
    # but T2 must not: no per-node record is built for T1-only nodes (see
    # test_one_disagreeing_juror_invalidates_t2's docstring for why), so
    # this is checked via the aggregate funnel counts instead.
    plan = _plan()
    n = 4
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        if window.index == 0:
            cache.append(_make_state(window, [[5.0, 0.0, 0.0]] * n, [[5.0, 0.0, 0.0]] * n, (2, 2), 3))
        elif window.index == 1:
            # majority of "others" (2 of 3) agree on class 0, but one dissents to class 1
            cache.append(_make_state(window, [[5.0, 0.0, 0.0]] * n, [[0.0, 5.0, 0.0]] * n, (2, 2), 3, seed=1))
        else:
            cache.append(_make_state(window, [[5.0, 0.0, 0.0]] * n, [[5.0, 0.0, 0.0]] * n, (2, 2), 3, seed=2))
    cache.seal()
    context = _context(plan, 3, (2, 2), cache)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    # majority (2/3 of window 0's "others" {1,2,3}) says class 0, but window
    # 1 dissents, so it is NOT unanimous -> T1 reached for window 0's node,
    # but no T2 record is built for it (window 1, whose OWN "others" {0,2,3}
    # happen to be genuinely unanimous, independently does reach T2 -- that
    # is incidental to this fixture, not what this test is checking).
    assert signal.funnel_counts.t1 == 4  # every window's own corner node
    assert all(o.source_window_id != 0 for o in signal.observations)


def test_one_disagreeing_juror_invalidates_t2():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    baseline = _deep_overlap_observation(signal)
    assert baseline.t2 is True

    # now make window 1 (an "other" window) disagree
    plan = _plan()
    n = 4
    cache2 = wc.ImageWindowCache(plan)
    for window in plan.windows:
        if window.index == 0:
            cache2.append(_make_state(window, [REVERSAL_FIXTURE["source_s0"]] * n, [REVERSAL_FIXTURE["source_p"]] * n, (2, 2), 3))
        elif window.index == 1:
            cache2.append(_make_state(window, [[0.0, 0.0, 5.0]] * n, [[0.0, 0.0, 5.0]] * n, (2, 2), 3, seed=1))
        else:
            cache2.append(_make_state(window, [REVERSAL_FIXTURE["other_s0"]] * n, [REVERSAL_FIXTURE["other_p"]] * n, (2, 2), 3, seed=window.index))
    cache2.seal()
    context2 = _context(plan, 3, (2, 2), cache2)
    signal2 = t4.build_t4_signal_for_image(cache2, context2, image_id="img")
    # window 0's node still has >=2 other covering windows (T1 is a pure
    # geometry fact, unaffected by scores), but window 1's dissent breaks
    # unanimity, so no T2 record is built for it: only nodes that reach T2
    # get a full ConsensusObservation.
    assert signal2.funnel_counts.t1 == 4  # geometry-only count, same as every other fixture here
    assert all(o.source_window_id != 0 for o in signal2.observations)


def test_t3_requires_source_dissent():
    plan = _plan()
    n = 4
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        cache.append(_make_state(window, [[5.0, 0.0, 0.0]] * n, [[5.0, 0.0, 0.0]] * n, (2, 2), 3, seed=window.index))
    cache.seal()
    context = _context(plan, 3, (2, 2), cache)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert obs.t2 is True
    assert obs.source_dissent_label == obs.consensus_label
    assert obs.t3 is False


def test_actionable_requires_stitched_dissent():
    # source disagrees but is heavily outnumbered/outweighed -> uniform
    # stitching recovers y -> T3 True, actionable False.
    cache, context = _overlap_cache_and_context(
        source_s0=[5.0, 0.0, 0.0], source_p=[0.0, 5.0, 0.0],
        other_s0=[5.0, 0.0, 0.0], other_p=[5.0, 0.0, 0.0],
    )
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert obs.t3 is True
    assert obs.stitched_label == obs.consensus_label
    assert obs.actionable is False


def test_t4_requires_source_unary_top1_equals_y():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert obs.actionable is True
    assert obs.source_unary_label == obs.consensus_label
    assert obs.t4 is True

    # now flip the source's unary away from y -> T4 must fail even though
    # everything else about the fixture is unchanged.
    cache2, context2 = _overlap_cache_and_context(
        source_s0=[0.0, 0.0, 5.0],  # u = class 2, != y
        source_p=REVERSAL_FIXTURE["source_p"],
        other_s0=REVERSAL_FIXTURE["other_s0"], other_p=REVERSAL_FIXTURE["other_p"],
    )
    signal2 = t4.build_t4_signal_for_image(cache2, context2, image_id="img")
    obs2 = _deep_overlap_observation(signal2)
    assert obs2.actionable is True
    assert obs2.source_unary_label != obs2.consensus_label
    assert obs2.t4 is False


def test_t4_prime_uses_strict_pairwise_s0_y_greater_than_s0_d():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert obs.t4_prime_source_score_y > obs.t4_prime_source_score_d
    assert obs.t4_prime is True

    # equal S0 scores for y and d must NOT satisfy strict > (T4' False)
    cache2, context2 = _overlap_cache_and_context(
        source_s0=[5.0, 5.0, 0.0],  # S0(y=0) == S0(d=1), argmax ties toward class 0 (u=0=y, so T4 still True)
        source_p=REVERSAL_FIXTURE["source_p"],
        other_s0=REVERSAL_FIXTURE["other_s0"], other_p=REVERSAL_FIXTURE["other_p"],
    )
    signal2 = t4.build_t4_signal_for_image(cache2, context2, image_id="img")
    obs2 = _deep_overlap_observation(signal2)
    assert obs2.t4_prime_source_score_y == obs2.t4_prime_source_score_d
    assert obs2.t4_prime is False


def test_t4_prime_records_unary_ranks():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert obs.t4_prime_source_rank_y == 0  # class 0 has the highest S0 score
    assert obs.t4_prime_source_rank_d == 1  # class 1 (d) is second


# ---------------------------------------------------------------------------
# 11: funnel subset invariants
# ---------------------------------------------------------------------------


def test_funnel_subset_invariants_enforced_by_construction():
    with pytest.raises(t4.T4AuditError):
        t4.T4FunnelCounts(t0=1, t1=2, t2=0, t3=0, actionable=0, t4=0, t4_prime=0)
    with pytest.raises(t4.T4AuditError):
        t4.T4FunnelCounts(t0=5, t1=5, t2=5, t3=5, actionable=5, t4=5, t4_prime=6)


def test_funnel_subset_invariants_hold_for_real_signals():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    counts = signal.funnel_counts
    assert counts.t4 <= counts.actionable <= counts.t3 <= counts.t2 <= counts.t1 <= counts.t0
    assert counts.t4_prime <= counts.actionable


def test_consensus_observation_rejects_invariant_violations():
    base_kwargs = dict(
        image_id="img", source_window_id=0, node_index=0, node_row=0, node_col=0,
        local_anchor=(0.0, 0.0), global_anchor=(0.0, 0.0), window_origin=(0, 0),
        window_extent=(4, 4), total_coverage=3, other_covering_window_ids=(1, 2),
        other_window_labels=(0, 0), other_window_unary_labels=(0, 0),
        source_dissent_label=1, source_unary_label=0, consensus_label=0,
        stitched_label=1, g_equals_d=True, g_is_third_label=False,
        unary_other_agree_count=2, unary_other_agree_fraction=1.0,
        unary_all_other_agree_y=True, unary_all_covering_agree_y=True,
        t0=True, t1=True, t2=True, t3=True, actionable=True, t4=True, t4_prime=True,
        t4_prime_source_score_y=5.0, t4_prime_source_score_d=0.0,
        t4_prime_source_rank_y=0, t4_prime_source_rank_d=1,
        other_window_tie=False, source_post_rwr_tie=False, source_unary_tie=False,
        stitched_tie=False, source_normalized_edge_distance=0.0,
        source_normalized_center_distance=0.0, agreeing_window_edge_distances=(),
        agreeing_window_center_distances=(),
    )
    t4.ConsensusObservation(**base_kwargs)  # valid baseline must construct fine

    broken = dict(base_kwargs)
    broken["t1"] = False  # t2=True but t1=False -> violates T2 implies T1
    with pytest.raises(t4.T4AuditError):
        t4.ConsensusObservation(**broken)

    broken2 = dict(base_kwargs)
    broken2["actionable"] = False  # t4=True but actionable=False
    with pytest.raises(t4.T4AuditError):
        t4.ConsensusObservation(**broken2)

    broken3 = dict(base_kwargs)
    broken3["t4_prime_source_score_y"] = 0.0  # equal scores, but t4_prime True
    broken3["t4_prime_source_score_d"] = 0.0
    with pytest.raises(t4.T4AuditError):
        t4.ConsensusObservation(**broken3)


# ---------------------------------------------------------------------------
# 12-14: adversarial fixtures
# ---------------------------------------------------------------------------


def test_source_exclusion_changes_the_adversarial_answer():
    """If the source were wrongly counted as a juror, [d, y, y, y] is not
    unanimous and T2 would be False; excluding it correctly yields [y, y, y]
    (unanimous) with the source's own reversal recorded separately as d."""
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert obs.t2 is True
    all_including_source = list(obs.other_window_labels) + [obs.source_dissent_label]
    assert len(set(all_including_source)) == 2  # would NOT be unanimous if source were included


def test_global_stitched_third_label_is_handled():
    cache, context = _overlap_cache_and_context(
        source_s0=[5.0, 0.0, 0.0], source_p=[-10.0, 0.0, 10.0],  # d = class 2
        other_s0=[5.0, 0.0, 0.0], other_p=[1.0, 0.0, 0.0],       # y = class 0
    )
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    assert obs.t3 is True
    if obs.actionable:
        assert obs.g_is_third_label == (obs.stitched_label not in (obs.consensus_label, obs.source_dissent_label))


def test_deterministic_class_ties_use_first_index():
    scores = torch.tensor([3.0, 3.0, 1.0])
    label, tie = t4._argmax_with_tie(scores)
    assert label == 0  # first/lowest index wins, matching torch.argmax
    assert tie is True
    scores2 = torch.tensor([1.0, 2.0, 3.0])
    label2, tie2 = t4._argmax_with_tie(scores2)
    assert label2 == 2
    assert tie2 is False


# ---------------------------------------------------------------------------
# 15-17: sampling geometry
# ---------------------------------------------------------------------------


def test_sigmoid_occurs_before_interpolation():
    """Sampling raw P then sigmoid-ing the SAMPLED value must differ from
    sigmoid-then-sample whenever the stencil weights are non-degenerate;
    prove the module actually does sigmoid-before-sample by construction."""
    torch.manual_seed(0)
    grid = torch.randn(9, 3)  # 3x3 grid, 3 classes
    stencil = geometry.bilinear_stencil(
        geometry.PixelCoordinate(1, 1), geometry.SpatialSize(4, 4), geometry.SpatialSize(3, 3),
        align_corners=True,
    )
    sigmoid_then_sample = t4._sample_bilinear(torch.sigmoid(grid), (3, 3), stencil)
    sample_then_sigmoid = torch.sigmoid(t4._sample_bilinear(grid, (3, 3), stencil))
    assert not torch.allclose(sigmoid_then_sample, sample_then_sigmoid), (
        "test fixture accidentally produced a degenerate (order-independent) stencil"
    )
    # the module's own consensus sampling operates on Q_v = sigmoid(P_v)
    # BEFORE _sample_bilinear is called: it samples from a per-window cache
    # of torch.sigmoid(propagated_scores) (whole grid, sigmoided once), and
    # never calls torch.sigmoid() on the OUTPUT of _sample_bilinear.
    import inspect
    source = inspect.getsource(t4.build_t4_signal_for_image)
    assert "torch.sigmoid(sampled_p)" not in source  # would be sample-then-sigmoid (wrong order)
    assert "torch.sigmoid(_sample_bilinear(" not in source  # would also be the wrong order
    assert "sigmoid_scores_for(other.window_index)" in source
    assert "_sample_bilinear(other_q, other_grid, stencil)" in source


def test_sparse_sampling_matches_dense_bilinear_interpolate():
    torch.manual_seed(1)
    grid_h, grid_w, classes = 5, 5, 4
    grid = torch.randn(grid_h * grid_w, classes)
    output_h, output_w = 17, 13
    dense = F.interpolate(
        grid.reshape(1, grid_h, grid_w, classes).permute(0, 3, 1, 2),
        (output_h, output_w), mode="bilinear", align_corners=True,
    )[0]  # [C, H, W]

    for py in (0, 4, 8, 16):
        for px in (0, 3, 9, 12):
            stencil = geometry.bilinear_stencil(
                geometry.PixelCoordinate(py, px), geometry.SpatialSize(output_h, output_w),
                geometry.SpatialSize(grid_h, grid_w), align_corners=True,
            )
            sparse = t4._sample_bilinear(grid, (grid_h, grid_w), stencil)
            assert torch.allclose(sparse, dense[:, py, px], atol=1e-5)


def test_source_anchor_maps_exactly_back_to_its_own_node():
    plan = _plan(image=(20, 20), crop=(8, 8), stride=(8, 8))
    window = plan.windows[0]
    grid = (4, 4)
    for r in range(4):
        for c in range(4):
            global_y, global_x = t4.source_score_grid_anchor(window, grid, r, c)
            local_y = global_y - window.origin.row
            local_x = global_x - window.origin.col
            stencil = t4._bilinear_stencil_from_fraction(
                local_y, local_x, window.extent.as_tuple(), grid
            )
            weights = stencil.weights()
            indices = stencil.neighbor_grid_indices()
            expected_flat = r * grid[1] + c
            observed_flat = {
                idx[0] * grid[1] + idx[1]: weight
                for idx, weight in zip(indices, weights) if weight > 0
            }
            assert set(observed_flat) == {expected_flat}
            assert abs(observed_flat[expected_flat] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 18-20: geometry edge cases
# ---------------------------------------------------------------------------


def test_clamped_terminal_windows_do_not_crash_and_stay_within_bounds():
    plan = _plan(image=(9, 9), crop=(4, 4), stride=(3, 3))  # forces clamped terminal windows
    assert any(w.clamped_vertical or w.clamped_horizontal for w in plan.windows)
    n = 4
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        cache.append(_make_state(window, [[5.0, 0.0, 0.0]] * n, [[5.0, 0.0, 0.0]] * n, (2, 2), 3, seed=window.index))
    cache.seal()
    context = _context(plan, 3, (2, 2), cache)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert signal.funnel_counts.t0 == plan.window_count * n


def test_non_square_and_smaller_than_crop_image():
    plan = _plan(image=(3, 5), crop=(8, 8), stride=(8, 8))
    assert plan.window_count == 1
    n = 4
    cache = wc.ImageWindowCache(plan)
    cache.append(_make_state(plan.windows[0], [[5.0, 0.0, 0.0]] * n, [[5.0, 0.0, 0.0]] * n, (2, 2), 3))
    cache.seal()
    context = _context(plan, 3, (2, 2), cache)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert signal.funnel_counts.t1 == 0  # single window: no "other" coverage possible


def test_coverage_at_exact_boundaries_and_corners():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    plan = context.plan
    corner_window = plan.windows[0]
    y, x = t4.source_score_grid_anchor(corner_window, (2, 2), 0, 0)
    assert t4.covers_continuous_point(corner_window, y, x)
    assert y == corner_window.origin.row and x == corner_window.origin.col


# ---------------------------------------------------------------------------
# 21-22: GT and feature isolation
# ---------------------------------------------------------------------------


def test_no_dino_feature_interpolation_anywhere_in_signal_building():
    import inspect
    source = inspect.getsource(t4.build_t4_signal_for_image)
    assert "dino_features" not in source
    assert ".s0" in source and "propagated_scores" in source


def test_signal_generation_has_no_gt_dependency():
    import inspect
    import re

    signature = inspect.signature(t4.build_t4_signal_for_image)
    for name in signature.parameters:
        assert "gt" not in name.lower() and "ground_truth" not in name.lower()
    # scan the CODE body only (docstrings legitimately discuss "GT-free" in
    # prose); a standalone "gt" identifier token would indicate an actual
    # ground-truth reference reaching this function.
    source_lines, _ = inspect.getsourcelines(t4.build_t4_signal_for_image)
    in_docstring = False
    code_only = []
    for line in source_lines:
        stripped = line.strip()
        if stripped.startswith('"""'):
            in_docstring = not in_docstring or stripped.count('"""') > 1
            continue
        if in_docstring:
            continue
        # also drop trailing "#" comment prose (not executable code)
        code_only.append(line.split("#", 1)[0])
    code_text = "".join(code_only)
    assert re.search(r"\bgt\b", code_text, flags=re.IGNORECASE) is None
    assert "ground_truth" not in code_text.lower()


# ---------------------------------------------------------------------------
# 23-25: GT evaluation, actionable-vs-T2 precision, shared-unary fields
# ---------------------------------------------------------------------------


def test_gt_nearest_neighbor_sampling_and_ignore_label():
    gt = torch.tensor([[1, 2], [3, 255]], dtype=torch.int64)
    size = SpatialSize(2, 2)
    assert t4.sample_gt_nearest_neighbor(gt, 0.0, 0.0, size) == 1
    assert t4.sample_gt_nearest_neighbor(gt, 0.49, 0.0, size) == 1  # rounds down to row 0
    assert t4.sample_gt_nearest_neighbor(gt, 0.5, 0.0, size) == 3   # round-half-up -> row 1
    assert t4.sample_gt_nearest_neighbor(gt, -5.0, -5.0, size) == 1  # clamped
    assert t4.sample_gt_nearest_neighbor(gt, 99.0, 99.0, size) == 255  # clamped, ignore label present


def test_actionable_precision_reported_separately_from_t2_precision():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)  # everything GT=class 0=y
    evaluation = t4.evaluate_t4_signal_against_gt(
        signal, gt, ignore_label=255, image_size=context.image_size
    )
    assert evaluation.t2 is not evaluation.actionable
    assert evaluation.t2.total >= evaluation.actionable.total


def test_shared_unary_support_fields_do_not_alter_membership():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    obs = _deep_overlap_observation(signal)
    import dataclasses
    tampered = dataclasses.replace(
        obs, unary_all_other_agree_y=not obs.unary_all_other_agree_y,
        unary_other_agree_count=0, unary_other_agree_fraction=0.0,
    )
    # membership flags must be identical regardless of the shared-unary fields
    assert tampered.t4 == obs.t4
    assert tampered.actionable == obs.actionable
    assert tampered.t2 == obs.t2


# ---------------------------------------------------------------------------
# 26-30: immutability, no reruns, output parity, determinism
# ---------------------------------------------------------------------------


def test_cache_tensors_unchanged_after_audit():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    before = [state.propagated_scores.clone() for state in cache.windows_in_order()]
    t4.build_t4_signal_for_image(cache, context, image_id="img")
    after = [state.propagated_scores for state in cache.windows_in_order()]
    for a, b in zip(before, after):
        assert torch.equal(a, b)
    assert cache.state is wc.WindowCacheState.SEALED  # audit never begins/ends replay


def test_no_model_graph_or_solver_rerun_during_audit():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    import models.dinotext.cover_dr.graph as graph_mod
    import models.dinotext.cover_dr.rwr as rwr_mod
    real_build = graph_mod.build_directed_topk_graph
    real_solve = rwr_mod.solve_rwr_cgls
    calls = {"graph": 0, "solve": 0}

    def counting_build(*a, **kw):
        calls["graph"] += 1
        return real_build(*a, **kw)

    def counting_solve(*a, **kw):
        calls["solve"] += 1
        return real_solve(*a, **kw)

    graph_mod.build_directed_topk_graph = counting_build
    rwr_mod.solve_rwr_cgls = counting_solve
    try:
        t4.build_t4_signal_for_image(cache, context, image_id="img")
    finally:
        graph_mod.build_directed_topk_graph = real_build
        rwr_mod.solve_rwr_cgls = real_solve
    assert calls == {"graph": 0, "solve": 0}


def test_deterministic_repeated_cpu_execution():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal_a = t4.build_t4_signal_for_image(cache, context, image_id="img")
    signal_b = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert signal_a.funnel_counts == signal_b.funnel_counts
    assert signal_a.observations == signal_b.observations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_parity_where_available():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    cpu_signal = t4.build_t4_signal_for_image(cache, context, image_id="img")

    plan = context.plan
    cuda_cache = wc.ImageWindowCache(plan)
    for state in cache.windows_in_order():
        cuda_cache.append(
            wc.CachedWindowState(
                geometry=state.geometry, window_index=state.window_index,
                patch_grid_shape=state.patch_grid_shape, class_count=state.class_count,
                s0=state.s0.cuda(), dino_features=state.dino_features.cuda(),
                graph=state.graph_copy(), propagated_scores=state.propagated_scores.cuda(),
                solver_summary=state.solver_summary,
            )
        )
    cuda_cache.seal()
    cuda_signal = t4.build_t4_signal_for_image(cuda_cache, context, image_id="img")
    assert cpu_signal.funnel_counts == cuda_signal.funnel_counts


def _dinotext_seg_module():
    mmcv_stub = types.ModuleType("mmcv")

    class Config(dict):
        __getattr__ = dict.__getitem__

    mmcv_stub.Config = Config
    utils_stub = types.ModuleType("utils")
    utils_stub.get_logger = lambda: types.SimpleNamespace(info=lambda *_a: None)
    previous = {n: sys.modules.get(n) for n in ("mmcv", "utils")}
    sys.modules.update({"mmcv": mmcv_stub, "utils": utils_stub})
    sys.modules.pop("segmentation.evaluation.dinotext_seg", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "segmentation.evaluation.dinotext_seg", SEGMENTATION_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("segmentation.evaluation.dinotext_seg", None)
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _SyntheticModel(nn.Module):
    def __init__(self, patch_size=2, embed_dim=6, class_count=3, seed=0):
        super().__init__()
        self.patch_size = patch_size
        generator = torch.Generator().manual_seed(seed)
        self.feature_proj = torch.randn(3, embed_dim, generator=generator)
        self.score_proj = torch.randn(embed_dim, class_count, generator=generator)

    def generate_patch_snapshot(self, crop, text_embedding):
        del text_embedding
        _, channels, height, width = crop.shape
        ps = self.patch_size
        grid_h, grid_w = height // ps, width // ps
        pooled = F.avg_pool2d(crop, ps).reshape(1, channels, grid_h * grid_w).permute(0, 2, 1)
        raw = pooled @ self.feature_proj
        raw = raw + 0.01 * torch.arange(grid_h * grid_w, dtype=torch.float32).view(1, -1, 1)
        features = F.normalize(raw, dim=-1)
        scores = features @ self.score_proj
        return types.SimpleNamespace(unary_scores=scores, dino_features=features, grid_hw=(grid_h, grid_w))

    def masks_from_patch_scores(self, patch_scores, grid_hw, output_hw):
        batch, _n, classes = patch_scores.shape
        grid_h, grid_w = grid_hw
        simmap = patch_scores.reshape(batch, grid_h, grid_w, classes).permute(0, 3, 1, 2)
        return F.interpolate(torch.sigmoid(simmap), tuple(output_hw), mode="bilinear", align_corners=True)


def _canonical_config(class_count=3, top_k=3, alpha=0.5):
    return cover_dr.RWRInferenceConfig(
        enabled=True, identity_path="evaluation_identities/e3_canonical_directed_rwr.toml",
        graph_mode=IDENTITY["rwr"]["graph_mode"], alpha=alpha, top_k=top_k,
        affinity_power=IDENTITY["rwr"]["affinity_power"], solver=IDENTITY["solver"]["method"],
        solver_rtol=1e-4, solver_atol=1e-6, solver_max_iterations=2000,
        expected_class_count=class_count, config_path=IDENTITY["canonical_config_path"],
    )


def test_pass_two_and_final_output_unchanged_by_the_audit():
    module = _dinotext_seg_module()
    model_a = _SyntheticModel(class_count=3, seed=2)
    model_b = _SyntheticModel(class_count=3, seed=2)  # identical construction
    config = _canonical_config()

    inference_a = module.DINOTextSegInference(
        model_a, torch.randn(3, 4), ["a", "b", "c"], with_bg=False,
        test_cfg=dict(mode="slide", crop_size=(8, 8), stride=(4, 4)),
    )
    inference_a.rwr_config = config
    inference_b = module.DINOTextSegInference(
        model_b, torch.randn(3, 4), ["a", "b", "c"], with_bg=False,
        test_cfg=dict(mode="slide", crop_size=(8, 8), stride=(4, 4)),
    )
    inference_b.rwr_config = config

    img = torch.rand(1, 3, 12, 12)
    cache_inference_a = types.SimpleNamespace(
        model=model_a, text_embedding=inference_a.text_embedding, rwr_config=config,
        test_cfg=types.SimpleNamespace(stride=(4, 4), crop_size=(8, 8)), num_classes=3, with_bg=False,
    )
    cache_inference_b = types.SimpleNamespace(
        model=model_b, text_embedding=inference_b.text_embedding, rwr_config=config,
        test_cfg=types.SimpleNamespace(stride=(4, 4), crop_size=(8, 8)), num_classes=3, with_bg=False,
    )

    # WITHOUT audit
    stitched_a, cache_a, context_a = wc.run_pass_one(cache_inference_a, img)
    pass2_a = wc.run_pass_two(cache_inference_a, cache_a, context_a)
    cache_a.close()

    # WITH audit inserted between pass 1 and pass 2
    stitched_b, cache_b, context_b = wc.run_pass_one(cache_inference_b, img)
    t4.build_t4_signal_for_image(cache_b, context_b, image_id="img")  # <-- audit
    pass2_b = wc.run_pass_two(cache_inference_b, cache_b, context_b)
    cache_b.close()

    assert torch.equal(stitched_a, stitched_b)
    assert torch.equal(pass2_a, pass2_b)


def test_audit_enabled_disabled_parity_and_zero_overhead_source():
    import inspect
    source = inspect.getsource(wc)
    assert "t4_audit" not in source  # window_cache.py has zero references to this module


# ---------------------------------------------------------------------------
# 31 covered above (skipped without CUDA); 32-33 below
# ---------------------------------------------------------------------------


def test_no_dense_full_crop_score_materialization():
    import inspect
    source = inspect.getsource(t4.build_t4_signal_for_image)
    assert "448" not in source
    assert "interpolate(" not in source  # only sparse 4-point gathers via _sample_bilinear


def test_aggregate_counts_match_explicit_synthetic_enumeration():
    """Explicit hand enumeration for REVERSAL_FIXTURE's 2x2-window, 2x2-node
    geometry (crop 4x4, stride 2x2, image 6x6):

    * T0 = 4 windows x 4 nodes = 16.
    * Only each window's (1,1) corner node has coverage 4 (the deep
      4-way overlap region); every other node has coverage 1 or 2, so
      T1 = 4 (one qualifying node per window).
    * Window 0's "others" = {1,2,3}, all sharing identical uniform
      other_p -> unanimous -> T2. Windows 1/2/3 each have window 0 (whose
      values differ) among THEIR "others" -> not unanimous -> no T2 for
      them. So T2 = 1 (window 0's node only).
    * That one T2 node: d=1 != y=0 -> T3=1. G(anchor) averages sigmoid(P)
      over the source + 3 "other" windows; the 3 mild-confidence "other"
      contributions (sigmoid([1,0,0])=[.731,.5,.5]) are outweighed by the
      source's saturated dissent (sigmoid([-10,10,0])~=[~0,~1,.5]) enough
      to tip the class-1 mean above class-0's -> g=1=d -> actionable=1.
    * u=0=y -> T4=1. S0(y=0)=5.0 > S0(d=1)=0.0 -> T4'=1.
    """
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert signal.funnel_counts == t4.T4FunnelCounts(
        t0=16, t1=4, t2=1, t3=1, actionable=1, t4=1, t4_prime=1
    )


# ---------------------------------------------------------------------------
# 34: empty-funnel / no-overlap
# ---------------------------------------------------------------------------


def test_empty_funnel_single_window_no_overlap():
    plan = _plan(image=(4, 4), crop=(4, 4), stride=(4, 4))
    assert plan.window_count == 1
    cache = wc.ImageWindowCache(plan)
    cache.append(_make_state(plan.windows[0], [[5.0, 0.0, 0.0]] * 4, [[5.0, 0.0, 0.0]] * 4, (2, 2), 3))
    cache.seal()
    context = _context(plan, 3, (2, 2), cache)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert signal.funnel_counts == t4.T4FunnelCounts(t0=4, t1=0, t2=0, t3=0, actionable=0, t4=0, t4_prime=0)
    assert signal.observations == ()


# ---------------------------------------------------------------------------
# 35: fail-closed validation
# ---------------------------------------------------------------------------


def test_invalid_cache_state_fails_closed():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)  # OPEN, never sealed
    context = wc.FirstPassImageContext(
        image_size=plan.image_size, plan=plan, class_count=3, common_patch_grid_shape=(2, 2),
        expected_window_count=plan.window_count, cached_window_count=0, min_coverage=1,
        max_coverage=plan.window_count, stitched_scores=torch.zeros(1, 3, *plan.image_size.as_tuple()),
        cache_total_bytes=0, pass_summary=wc._summarize_telemetry([]),
    )
    with pytest.raises(t4.T4AuditError):
        t4.build_t4_signal_for_image(cache, context, image_id="img")


def test_nonfinite_scores_fail_closed():
    with pytest.raises(t4.T4AuditError):
        t4._argmax_with_tie(torch.tensor([1.0, float("nan"), 2.0]))
    with pytest.raises(t4.T4AuditError):
        t4._argmax_with_tie(torch.tensor([1.0, float("inf"), 2.0]))


def test_wrong_gt_shape_fails_closed():
    gt = torch.zeros(3, 3, dtype=torch.int64)
    with pytest.raises(t4.T4AuditError):
        t4.sample_gt_nearest_neighbor(gt, 0.0, 0.0, SpatialSize(4, 4))


# ---------------------------------------------------------------------------
# 36: eager GT-shape validation, independent of observation count
#
# Regression coverage for a real gap: gt's shape was previously validated
# only inside sample_gt_nearest_neighbor, which is called once per
# observation -- so a signal with zero T2 observations (a normal,
# unremarkable case: no anchor ever reached strict unanimity) let a
# completely wrong-shaped/wrong-type gt tensor pass through
# evaluate_t4_signal_against_gt silently, returning a valid-looking
# all-zero T4ImageGTEvaluation instead of failing closed.
# ---------------------------------------------------------------------------


def _no_overlap_signal_and_context():
    """Single, non-overlapping window -> zero T1/T2 observations."""
    plan = _plan(image=(4, 4), crop=(4, 4), stride=(4, 4))
    assert plan.window_count == 1
    cache = wc.ImageWindowCache(plan)
    cache.append(_make_state(plan.windows[0], [[5.0, 0.0, 0.0]] * 4, [[5.0, 0.0, 0.0]] * 4, (2, 2), 3))
    cache.seal()
    context = _context(plan, 3, (2, 2), cache)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert signal.observations == ()
    return signal, context


def test_empty_observations_wrong_gt_shape_fails_closed():
    signal, context = _no_overlap_signal_and_context()
    wrong_shape_gt = torch.zeros(9, 9, dtype=torch.int64)
    with pytest.raises(t4.T4AuditError):
        t4.evaluate_t4_signal_against_gt(
            signal, wrong_shape_gt, ignore_label=255, image_size=context.image_size
        )


def test_empty_observations_non_tensor_gt_fails_closed():
    signal, context = _no_overlap_signal_and_context()
    for bad_gt in ("not a tensor", None, [[0, 0], [0, 0]]):
        with pytest.raises(t4.T4AuditError):
            t4.evaluate_t4_signal_against_gt(
                signal, bad_gt, ignore_label=255, image_size=context.image_size
            )


def test_empty_observations_rank3_gt_fails_closed():
    signal, context = _no_overlap_signal_and_context()
    rank3_gt = torch.zeros(4, 4, 1, dtype=torch.int64)
    with pytest.raises(t4.T4AuditError):
        t4.evaluate_t4_signal_against_gt(
            signal, rank3_gt, ignore_label=255, image_size=context.image_size
        )


def test_empty_observations_correctly_shaped_gt_returns_valid_zero_summary():
    signal, context = _no_overlap_signal_and_context()
    correct_gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    evaluation = t4.evaluate_t4_signal_against_gt(
        signal, correct_gt, ignore_label=255, image_size=context.image_size
    )
    for stage in (evaluation.t2, evaluation.t3, evaluation.actionable, evaluation.t4, evaluation.t4_prime):
        assert stage == t4.T4StageAccuracy()  # a genuine (not erroring) all-zero result


def test_non_empty_gt_evaluation_unchanged_by_eager_validation():
    """The eager check must not alter behavior for a real (non-empty)
    signal: exact same accuracy counters as the established
    REVERSAL_FIXTURE baseline (see test_aggregate_counts_match_explicit_
    synthetic_enumeration for the hand-derived funnel counts)."""
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    assert len(signal.observations) == 1
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)  # GT = y(class 0) everywhere
    evaluation = t4.evaluate_t4_signal_against_gt(signal, gt, ignore_label=255, image_size=context.image_size)
    assert evaluation.t2.total == 1 and evaluation.t2.y_correct == 1 and evaluation.t2.d_correct == 0
    assert evaluation.actionable.total == 1 and evaluation.actionable.y_correct == 1
    assert evaluation.t4.total == 1 and evaluation.t4.y_correct == 1


def test_cache_contents_and_final_predictions_unchanged_by_eager_gt_validation():
    """Cache tensors are byte-identical, and pass-2's final stitched
    output is bitwise identical, whether or not
    evaluate_t4_signal_against_gt ran in between pass 1 and pass 2 --
    exercised through BOTH its empty-observations (new eager-check) path
    and its ordinary non-empty-observations path."""
    model_a = _SyntheticModel(class_count=3, seed=4)
    model_b = _SyntheticModel(class_count=3, seed=4)
    config = _canonical_config()

    cache_inference_a = types.SimpleNamespace(
        model=model_a, text_embedding=torch.randn(3, 4), rwr_config=config,
        test_cfg=types.SimpleNamespace(stride=(4, 4), crop_size=(8, 8)), num_classes=3, with_bg=False,
    )
    cache_inference_b = types.SimpleNamespace(
        model=model_b, text_embedding=cache_inference_a.text_embedding, rwr_config=config,
        test_cfg=types.SimpleNamespace(stride=(4, 4), crop_size=(8, 8)), num_classes=3, with_bg=False,
    )
    img = torch.rand(1, 3, 12, 12)

    stitched_a, cache_a, context_a = wc.run_pass_one(cache_inference_a, img)
    pass2_a = wc.run_pass_two(cache_inference_a, cache_a, context_a)
    cache_a.close()

    stitched_b, cache_b, context_b = wc.run_pass_one(cache_inference_b, img)

    def digest_cache(cache):
        parts = []
        for window in cache.windows_in_order():
            for name in ("s0", "dino_features", "propagated_scores"):
                parts.append(hashlib.sha256(getattr(window, name).numpy().tobytes()).hexdigest())
        return parts

    before = digest_cache(cache_b)
    signal = t4.build_t4_signal_for_image(cache_b, context_b, image_id="img")
    gt = torch.zeros(*context_b.image_size.as_tuple(), dtype=torch.int64)
    t4.evaluate_t4_signal_against_gt(signal, gt, ignore_label=255, image_size=context_b.image_size)
    after = digest_cache(cache_b)
    assert before == after, "cache tensors changed after evaluate_t4_signal_against_gt ran"

    pass2_b = wc.run_pass_two(cache_inference_b, cache_b, context_b)
    cache_b.close()

    assert torch.equal(stitched_a, stitched_b)
    assert torch.equal(pass2_a, pass2_b)


def test_wrong_types_fail_closed():
    with pytest.raises(t4.T4AuditError):
        t4.T4ImageGTEvaluation(image_id=123, t2=t4.T4StageAccuracy(), t3=t4.T4StageAccuracy(),
                                actionable=t4.T4StageAccuracy(), t4=t4.T4StageAccuracy(),
                                t4_prime=t4.T4StageAccuracy())
    with pytest.raises(t4.T4AuditError):
        t4.T4StageAccuracy(total=-1)
    with pytest.raises(t4.T4AuditError):
        t4.T4StageAccuracy(y_correct=5, total=2)


def test_accumulator_rejects_wrong_types():
    accumulator = t4.T4AuditAccumulator()
    with pytest.raises(t4.T4AuditError):
        accumulator.absorb_signal("not a signal")
    with pytest.raises(t4.T4AuditError):
        accumulator.absorb_gt_evaluation("not an evaluation")


# ---------------------------------------------------------------------------
# Accumulator / summary streaming behavior
# ---------------------------------------------------------------------------


def test_accumulator_streams_across_images_without_retaining_records():
    accumulator = t4.T4AuditAccumulator()
    for _ in range(3):
        cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
        signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
        accumulator.absorb_signal(signal)
        gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
        evaluation = t4.evaluate_t4_signal_against_gt(
            signal, gt, ignore_label=255, image_size=context.image_size
        )
        accumulator.absorb_gt_evaluation(evaluation)
        cache.close()
        del signal  # simulate the record being released after aggregation
    summary = accumulator.summary()
    assert summary.images_processed == 3
    assert summary.funnel_counts.t4 == 3
    assert summary.gt_evaluated is True
    assert summary.delta_trust_actionable is not None


def test_delta_trust_is_a_raw_point_estimate_without_confidence_intervals():
    accumulator = t4.T4AuditAccumulator()
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    accumulator.absorb_signal(signal)
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)  # GT = y everywhere
    evaluation = t4.evaluate_t4_signal_against_gt(signal, gt, ignore_label=255, image_size=context.image_size)
    accumulator.absorb_gt_evaluation(evaluation)
    summary = accumulator.summary()
    # accuracy(y)=1.0 (GT is always y here), accuracy(d)=0.0 -> delta = 1.0
    assert summary.delta_trust_actionable == pytest.approx(1.0)
    assert not hasattr(summary, "delta_trust_confidence_interval")
    assert not hasattr(summary, "bootstrap_iterations")


def test_run_t4_audit_for_image_convenience_wrapper():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal, evaluation = t4.run_t4_audit_for_image(cache, context, image_id="img")
    assert evaluation is None
    gt = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
    signal2, evaluation2 = t4.run_t4_audit_for_image(
        cache, context, image_id="img", gt=gt, ignore_label=255
    )
    assert evaluation2 is not None
    assert signal2.funnel_counts == signal.funnel_counts
