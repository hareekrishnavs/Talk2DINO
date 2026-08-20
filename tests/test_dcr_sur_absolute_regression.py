"""Absolute-reference regression tests for the DCR/SUR double-sigmoid fix.

Every reference value in this file is computed from first principles
(hand-derived arithmetic, torch.sigmoid/F.interpolate called directly, or a
from-scratch pixel loop) -- never by calling the same helper the code under
test calls. This file exists specifically to catch a defect class that
self-consistency tests (comparing two paths through the same shared helper)
cannot catch, which is exactly how the original double-sigmoid defect
escaped the first version of tests/test_baseline_evaluation_suite.py.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation"
GEOMETRY_PATH = EVAL_DIR / "sliding_window_geometry.py"
CACHE_PATH = EVAL_DIR / "window_cache.py"
AUDIT_PATH = EVAL_DIR / "t4_audit.py"
STITCH_PATH = EVAL_DIR / "stitching_baselines.py"
CONSENSUS_PATH = EVAL_DIR / "consensus_replacement.py"
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


_install_segmentation_package_stub()
geometry = _load("segmentation.evaluation.sliding_window_geometry", GEOMETRY_PATH)
wc = _load("segmentation.evaluation.window_cache", CACHE_PATH)
t4 = _load("segmentation.evaluation.t4_audit", AUDIT_PATH)
stitching = _load("segmentation.evaluation.stitching_baselines", STITCH_PATH)
consensus = _load("segmentation.evaluation.consensus_replacement", CONSENSUS_PATH)
cover_dr = _load_cover_dr_package()

SpatialSize = geometry.SpatialSize
SlidingWindowPlan = geometry.SlidingWindowPlan


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


def _overlap_cache_and_context(*, source_s0, source_p, other_s0, other_p, class_count=3, grid=(2, 2),
                                image=(6, 6), crop=(4, 4), stride=(2, 2)):
    plan = SlidingWindowPlan.build(image_size=SpatialSize(*image), crop_size=SpatialSize(*crop), stride=SpatialSize(*stride))
    n = grid[0] * grid[1]
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        if window.index == 0:
            cache.append(_make_state(window, [source_s0] * n, [source_p] * n, grid, class_count, seed=window.index))
        else:
            cache.append(_make_state(window, [other_s0] * n, [other_p] * n, grid, class_count, seed=window.index))
    cache.seal()
    return cache, _context(plan, class_count, grid, cache)


REVERSAL_FIXTURE = dict(
    source_s0=[5.0, 0.0, 0.0], source_p=[-10.0, 10.0, 0.0],
    other_s0=[5.0, 0.0, 0.0], other_p=[1.0, 0.0, 0.0],
)


# ---------------------------------------------------------------------------
# 1. Raw [2,-1,0] produces sigmoid([2,-1,0]), not sigmoid(sigmoid(...)).
# ---------------------------------------------------------------------------


def test_raw_score_produces_single_sigmoid_not_double():
    raw = torch.tensor([2.0, -1.0, 0.0])
    single = torch.sigmoid(raw)
    double = torch.sigmoid(torch.sigmoid(raw))
    assert not torch.allclose(single, double)

    plan = SlidingWindowPlan.build(image_size=SpatialSize(2, 2), crop_size=SpatialSize(2, 2), stride=SpatialSize(2, 2))
    window = plan.windows[0]
    n, c = 2, 3
    raw_grid = torch.stack([raw, raw], dim=0)
    feats = F.normalize(torch.randn(n, 4, generator=torch.Generator().manual_seed(0)), dim=-1)
    graph = wc.GraphSnapshot(torch.zeros(n, 1, dtype=torch.int64), torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n, dtype=torch.bool), n, 1, 3.0)
    state = wc.CachedWindowState(
        geometry=window, window_index=0, patch_grid_shape=(1, 2), class_count=c,
        s0=torch.zeros(n, c), dino_features=feats, graph=graph, propagated_scores=raw_grid,
        solver_summary=wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0),
    )
    cache = wc.ImageWindowCache(plan)
    cache.append(state)
    cache.seal()
    result, _report = consensus.apply_replacements_and_stitch(
        cache, [], image_size=plan.image_size, class_count=c,
        mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_k12_q",
    )
    observed = result.image[:, 0, 0]
    assert torch.allclose(observed, single, atol=1e-5)
    assert not torch.allclose(observed, double, atol=1e-3)


# ---------------------------------------------------------------------------
# 2. DCR-Hard source grid node remains exact [1,0,0] at its own anchor
#    before overlap stitching (single-window, no averaging with other data).
# ---------------------------------------------------------------------------


def test_dcr_hard_exact_one_hot_survives_single_window_reconstruction():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(2, 2), crop_size=SpatialSize(2, 2), stride=SpatialSize(2, 2))
    window = plan.windows[0]
    n, c = 2, 3
    replacement_vector = torch.tensor([1.0, 0.0, 0.0])
    grid = torch.stack([replacement_vector, torch.tensor([0.3, 0.3, 0.4])], dim=0)
    feats = F.normalize(torch.randn(n, 4, generator=torch.Generator().manual_seed(1)), dim=-1)
    graph = wc.GraphSnapshot(torch.zeros(n, 1, dtype=torch.int64), torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n, dtype=torch.bool), n, 1, 3.0)
    output = consensus.probability_grid_to_mask(grid, (1, 2), (2, 2), class_count=c)
    # align_corners=True with grid_w=out_w=2 maps pixel column 0 exactly to
    # grid node 0 -- so the exact one-hot must survive untouched.
    assert torch.equal(output[:, 0, 0], replacement_vector)
    assert torch.equal(output[:, 1, 0], replacement_vector)


# ---------------------------------------------------------------------------
# 3. DCR-JuryMean equals an independently calculated jury probability mean.
# ---------------------------------------------------------------------------


def test_dcr_jury_mean_absolute_reference():
    cache, ctx = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, ctx, image_id="img")
    obs = [o for o in signal.observations if o.t4][0]

    # independent full-dense computation: sigmoid every juror window's WHOLE
    # grid by hand, then average the specific rows corresponding to the
    # anchor via a from-scratch bilinear stencil (not t4_audit's private
    # helper -- reimplemented here from the align_corners=True formula).
    from fractions import Fraction

    source_state = cache.get(obs.source_window_id)
    gh, gw = source_state.patch_grid_shape
    height, width = source_state.geometry.extent.as_tuple()

    def local_anchor(index, grid_extent, extent):
        if extent <= 1:
            return Fraction(0)
        if grid_extent <= 1:
            return Fraction(extent - 1, 2)
        return Fraction(index * (extent - 1), grid_extent - 1)

    global_y = source_state.geometry.origin.row + local_anchor(obs.node_row, gh, height)
    global_x = source_state.geometry.origin.col + local_anchor(obs.node_col, gw, width)

    samples = []
    for wid in obs.other_covering_window_ids:
        juror = cache.get(wid)
        juror_q = torch.sigmoid(juror.propagated_scores)  # full dense sigmoid, by hand
        jh, jw = juror.patch_grid_shape
        jheight, jwidth = juror.geometry.extent.as_tuple()
        ly = global_y - juror.geometry.origin.row
        lx = global_x - juror.geometry.origin.col

        def axis(coord, grid_extent, out_extent):
            if out_extent <= 1 or grid_extent <= 1:
                return 0, 0, 0.0
            pos = coord * (grid_extent - 1) / (out_extent - 1)
            low = int(pos)
            if low >= grid_extent - 1:
                return grid_extent - 1, grid_extent - 1, 0.0
            return low, low + 1, float(pos - low)

        r0, r1, rf = axis(ly, jh, jheight)
        c0, c1, cf = axis(lx, jw, jwidth)
        w00, w01, w10, w11 = (1 - rf) * (1 - cf), (1 - rf) * cf, rf * (1 - cf), rf * cf
        idx00, idx01, idx10, idx11 = r0 * jw + c0, r0 * jw + c1, r1 * jw + c0, r1 * jw + c1
        samples.append(w00 * juror_q[idx00] + w01 * juror_q[idx01] + w10 * juror_q[idx10] + w11 * juror_q[idx11])
    reference = torch.stack(samples, dim=0).mean(dim=0)

    replacements = consensus.build_frozen_replacements(cache, signal, kind=consensus.DCR_JURY_MEAN, image_id="img", class_count=3)
    assert torch.allclose(replacements[0].vector, reference, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. SUR equals independently calculated sigmoid(S0).
# ---------------------------------------------------------------------------


def test_sur_absolute_reference():
    cache, ctx = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, ctx, image_id="img")
    obs = [o for o in signal.observations if o.t4][0]
    source_state = cache.get(obs.source_window_id)
    raw_row = source_state.s0[obs.node_index].clone()
    reference = torch.tensor([1 / (1 + torch.exp(-v).item()) for v in raw_row])
    replacements = consensus.build_frozen_replacements(cache, signal, kind=consensus.SUR, image_id="img", class_count=3)
    assert torch.allclose(replacements[0].vector, reference, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Single-window output matches direct F.interpolate of the probability grid.
# ---------------------------------------------------------------------------


def test_single_window_matches_direct_interpolate():
    n, c = 4, 3
    grid = torch.rand(n, c)
    grid_hw, output_hw = (2, 2), (5, 7)
    observed = consensus.probability_grid_to_mask(grid, grid_hw, output_hw, class_count=c)
    reference = F.interpolate(
        grid.reshape(1, *grid_hw, c).permute(0, 3, 1, 2), output_hw, mode="bilinear", align_corners=True
    )[0]
    assert torch.equal(observed, reference)


# ---------------------------------------------------------------------------
# 6. Multi-window output matches an independent pixel-loop stitcher.
# ---------------------------------------------------------------------------


def test_multi_window_matches_independent_pixel_loop_stitcher():
    plan = SlidingWindowPlan.build(image_size=SpatialSize(6, 6), crop_size=SpatialSize(4, 4), stride=SpatialSize(2, 2))
    cache = wc.ImageWindowCache(plan)
    n = 4
    for window in plan.windows:
        generator = torch.Generator().manual_seed(window.index)
        raw = torch.randn(n, 3, generator=generator)
        feats = F.normalize(torch.randn(n, 4, generator=generator), dim=-1)
        graph = wc.GraphSnapshot(torch.zeros(n, 1, dtype=torch.int64), torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n, dtype=torch.bool), n, 1, 3.0)
        cache.append(
            wc.CachedWindowState(
                geometry=window, window_index=window.index, patch_grid_shape=(2, 2), class_count=3,
                s0=torch.zeros(n, 3), dino_features=feats, graph=graph, propagated_scores=raw,
                solver_summary=wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0),
            )
        )
    cache.seal()

    result, _report = consensus.apply_replacements_and_stitch(
        cache, [], image_size=plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_k12_q",
    )

    h, w = 6, 6
    total = torch.zeros(3, h, w, dtype=torch.float64)
    count = torch.zeros(h, w, dtype=torch.float64)
    for state in cache.windows_in_order():
        probs = torch.sigmoid(state.propagated_scores.double())
        gh, gw = state.patch_grid_shape
        eh, ew = state.geometry.extent.as_tuple()
        oy, ox = state.geometry.origin.row, state.geometry.origin.col
        for y in range(eh):
            gy = y * (gh - 1) / (eh - 1) if eh > 1 and gh > 1 else 0.0
            r0, r1 = int(gy), min(int(gy) + 1, gh - 1)
            rf = gy - int(gy)
            for x in range(ew):
                gx = x * (gw - 1) / (ew - 1) if ew > 1 and gw > 1 else 0.0
                c0, c1 = int(gx), min(int(gx) + 1, gw - 1)
                cf = gx - int(gx)
                v = (
                    (1 - rf) * (1 - cf) * probs[r0 * gw + c0]
                    + (1 - rf) * cf * probs[r0 * gw + c1]
                    + rf * (1 - cf) * probs[r1 * gw + c0]
                    + rf * cf * probs[r1 * gw + c1]
                )
                total[:, oy + y, ox + x] += v
                count[oy + y, ox + x] += 1
    reference = (total / count).float()
    assert torch.allclose(result.image, reference, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. Overlap fixture where double sigmoid changes argmax: fixed output now
#    matches the single-sigmoid reference, not the double-sigmoid one.
# ---------------------------------------------------------------------------


def test_argmax_flip_fixture_matches_single_sigmoid_not_double():
    a = torch.tensor([11.385, -0.474, 0.371])
    b = torch.tensor([-1.290, -2.978, 0.338])
    single_avg = (torch.sigmoid(a) + torch.sigmoid(b)) / 2
    double_avg = (torch.sigmoid(torch.sigmoid(a)) + torch.sigmoid(torch.sigmoid(b))) / 2
    assert single_avg.argmax().item() != double_avg.argmax().item(), "fixture must actually exercise an argmax flip"

    n, c = 2, 3
    # A real 2x2 image with a 2x2 crop/stride only ever produces ONE window,
    # so cross-window averaging is reproduced explicitly below: two
    # independent single-window reconstructions (one per raw vector) via
    # apply_replacements_and_stitch, averaged by hand in this test -- this
    # is exactly the arithmetic uniform multi-window stitching performs.
    plan2 = SlidingWindowPlan.build(image_size=SpatialSize(2, 2), crop_size=SpatialSize(2, 2), stride=SpatialSize(2, 2))
    cache2 = wc.ImageWindowCache(plan2)
    window2 = plan2.windows[0]
    raw_grid_a = torch.stack([a, a], dim=0)
    feats_a = F.normalize(torch.randn(n, 4, generator=torch.Generator().manual_seed(10)), dim=-1)
    graph_a = wc.GraphSnapshot(torch.zeros(n, 1, dtype=torch.int64), torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n, dtype=torch.bool), n, 1, 3.0)
    cache2.append(
        wc.CachedWindowState(
            geometry=window2, window_index=0, patch_grid_shape=(1, 2), class_count=c,
            s0=torch.zeros(n, c), dino_features=feats_a, graph=graph_a, propagated_scores=raw_grid_a,
            solver_summary=wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0),
        )
    )
    cache2.seal()
    result_a, _ = consensus.apply_replacements_and_stitch(
        cache2, [], image_size=plan2.image_size, class_count=c, score_source="rwr_k12_q",
    )
    # Two independent single-window reconstructions (window carrying `a`,
    # window carrying `b`) manually averaged -- reproduces exactly what a
    # 2-window overlap would produce under uniform stitching, using ONLY
    # apply_replacements_and_stitch (never a second, possibly-buggy helper)
    # plus by-hand averaging performed in this test.
    raw_grid_b = torch.stack([b, b], dim=0)
    feats_b = F.normalize(torch.randn(n, 4, generator=torch.Generator().manual_seed(11)), dim=-1)
    graph_b = wc.GraphSnapshot(torch.zeros(n, 1, dtype=torch.int64), torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n, dtype=torch.bool), n, 1, 3.0)
    cache_b = wc.ImageWindowCache(plan2)
    cache_b.append(
        wc.CachedWindowState(
            geometry=window2, window_index=0, patch_grid_shape=(1, 2), class_count=c,
            s0=torch.zeros(n, c), dino_features=feats_b, graph=graph_b, propagated_scores=raw_grid_b,
            solver_summary=wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0),
        )
    )
    cache_b.seal()
    result_b, _ = consensus.apply_replacements_and_stitch(
        cache_b, [], image_size=plan2.image_size, class_count=c, score_source="rwr_k12_q",
    )
    manual_multiwindow_average = (result_a.image[:, 0, 0] + result_b.image[:, 0, 0]) / 2
    assert torch.allclose(manual_multiwindow_average, single_avg, atol=1e-4)
    assert not torch.allclose(manual_multiwindow_average, double_avg, atol=1e-2)
    assert manual_multiwindow_average.argmax().item() == single_avg.argmax().item()


# ---------------------------------------------------------------------------
# 8. Disabled replacement path matches canonical production, reference side
#    built WITHOUT reusing apply_replacements_and_stitch or any shared helper.
# ---------------------------------------------------------------------------


def test_disabled_path_matches_independently_built_canonical_reference():
    cache, ctx = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    disabled, _report = consensus.apply_replacements_and_stitch(
        cache, [], image_size=cache.plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_k12_q",
    )
    # Independent reference: raw sigmoid + F.interpolate, called directly in
    # this test body, then a hand-written accumulate/divide stitch -- shares
    # no function with apply_replacements_and_stitch or with
    # tests/test_baseline_evaluation_suite.py's own helper.
    h, w = cache.plan.image_size.as_tuple()
    total = torch.zeros(3, h, w, dtype=torch.float64)
    count = torch.zeros(h, w, dtype=torch.float64)
    for state in cache.windows_in_order():
        probs = torch.sigmoid(state.propagated_scores.double())
        gh, gw = state.patch_grid_shape
        logits_grid = probs.reshape(1, gh, gw, 3).permute(0, 3, 1, 2)
        extent = state.geometry.extent.as_tuple()
        mask = F.interpolate(logits_grid, extent, mode="bilinear", align_corners=True)[0]
        rows, cols = state.geometry.accumulation_slice
        total[:, rows, cols] += mask
        count[rows, cols] += 1
    reference = (total / count).float()
    assert torch.allclose(disabled.image, reference, atol=1e-5)


# ---------------------------------------------------------------------------
# 9. Cache tensors remain unchanged.
# ---------------------------------------------------------------------------


def test_cache_tensors_unchanged_by_reconstruction():
    cache, ctx = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, ctx, image_id="img")
    replacements = consensus.build_frozen_replacements(cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3)
    snapshots = {s.window_index: s.propagated_scores.clone() for s in cache.windows_in_order()}
    consensus.apply_replacements_and_stitch(
        cache, replacements, image_size=cache.plan.image_size, class_count=3, score_source="rwr_dcr_hard",
    )
    for state in cache.windows_in_order():
        assert torch.equal(state.propagated_scores, snapshots[state.window_index])


# ---------------------------------------------------------------------------
# 10. Raw-score APIs reject probability-domain misuse where detectable.
# ---------------------------------------------------------------------------


def test_probability_grid_to_mask_rejects_raw_score_looking_input():
    raw_looking = torch.tensor([[5.0, -3.0, 2.0], [1.0, 0.0, -1.0]])  # far outside [0,1]
    with pytest.raises(consensus.ConsensusReplacementError):
        consensus.probability_grid_to_mask(raw_looking, (1, 2), (4, 4), class_count=3)


# ---------------------------------------------------------------------------
# 11. Probability APIs reject out-of-range and nonfinite inputs.
# ---------------------------------------------------------------------------


def test_probability_grid_to_mask_rejects_nonfinite():
    bad = torch.tensor([[0.5, 0.5, float("nan")], [0.1, 0.2, 0.7]])
    with pytest.raises(consensus.ConsensusReplacementError):
        consensus.probability_grid_to_mask(bad, (1, 2), (4, 4), class_count=3)


def test_probability_grid_to_mask_accepts_valid_probabilities():
    good = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    output = consensus.probability_grid_to_mask(good, (1, 2), (4, 4), class_count=3)
    assert output.shape == (3, 4, 4)
    assert bool((output >= -1e-6).all()) and bool((output <= 1.0 + 1e-6).all())


def test_probability_grid_to_mask_rejects_wrong_class_count():
    grid = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(consensus.ConsensusReplacementError):
        consensus.probability_grid_to_mask(grid, (1, 2), (4, 4), class_count=3)


# ---------------------------------------------------------------------------
# Section 3 instrumentation: total sigmoid count is exactly one per
# original score grid, and zero after probability-domain replacement.
# ---------------------------------------------------------------------------


def test_sigmoid_call_count_is_exactly_one_per_window_then_zero_after_replacement():
    cache, ctx = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, ctx, image_id="img")
    replacements = consensus.build_frozen_replacements(cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3)

    calls = {"count": 0}
    original_sigmoid = torch.sigmoid

    def counted_sigmoid(*args, **kwargs):
        calls["count"] += 1
        return original_sigmoid(*args, **kwargs)

    torch.sigmoid = counted_sigmoid
    try:
        consensus.apply_replacements_and_stitch(
            cache, replacements, image_size=cache.plan.image_size, class_count=3,
            mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_dcr_hard",
        )
    finally:
        torch.sigmoid = original_sigmoid

    window_count = cache.plan.window_count
    assert calls["count"] == window_count, (
        f"expected exactly one sigmoid call per window ({window_count}), got {calls['count']}"
    )


def test_probability_grid_to_mask_never_calls_sigmoid():
    grid = torch.tensor([[0.5, 0.5, 0.5], [0.2, 0.3, 0.5]])
    calls = {"count": 0}
    original_sigmoid = torch.sigmoid

    def counted_sigmoid(*args, **kwargs):
        calls["count"] += 1
        return original_sigmoid(*args, **kwargs)

    torch.sigmoid = counted_sigmoid
    try:
        consensus.probability_grid_to_mask(grid, (1, 2), (4, 4), class_count=3)
    finally:
        torch.sigmoid = original_sigmoid
    assert calls["count"] == 0


# ---------------------------------------------------------------------------
# Strict-safe revalidation after the reconstruction fix: the final stitched
# image (built via apply_replacements_and_stitch on ONLY the accepted
# candidates) must be consistent with strict-safe's own internal sparse
# argmax decision at every accepted target's anchor -- no hidden second
# sigmoid, no divergence between the sparse acceptance computation and the
# dense final reconstruction.
# ---------------------------------------------------------------------------


def test_strict_safe_accepted_state_matches_final_stitched_image():
    cache, ctx = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, ctx, image_id="img")
    replacements = consensus.build_frozen_replacements(cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3)
    report = consensus.run_strict_safe_sequential(cache, signal, replacements)
    assert report.accepted >= 1

    final_image, _apply_report = consensus.apply_replacements_and_stitch(
        cache, report.accepted_candidates, image_size=cache.plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_dcr_hard_strict_safe",
    )

    from fractions import Fraction

    for candidate in report.accepted_candidates:
        source_state = cache.get(candidate.source_window_id)
        gh, gw = source_state.patch_grid_shape
        height, width = source_state.geometry.extent.as_tuple()

        def local_anchor(index, grid_extent, extent):
            if extent <= 1:
                return Fraction(0)
            if grid_extent <= 1:
                return Fraction(extent - 1, 2)
            return Fraction(index * (extent - 1), grid_extent - 1)

        global_y = source_state.geometry.origin.row + local_anchor(candidate.node_row, gh, height)
        global_x = source_state.geometry.origin.col + local_anchor(candidate.node_col, gw, width)
        # this fixture's anchors land on exact integer pixel coordinates
        # (small hand-built lattice), so direct pixel indexing is valid.
        py, px = int(global_y), int(global_x)
        assert Fraction(py) == global_y and Fraction(px) == global_x
        pixel_argmax = int(final_image.image[:, py, px].argmax().item())
        assert pixel_argmax == candidate.consensus_label


def test_probability_grid_to_mask_source_never_references_sigmoid():
    import ast

    tree = ast.parse(inspect.getsource(consensus.probability_grid_to_mask))
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "sigmoid" not in calls
    assert "softmax" not in calls
