"""Tests for the stitching-protocol / k-sweep / DCR / SUR baseline suite.

Loaded via file-path import with lightweight stand-ins for mmcv/utils
(mirroring tests/test_t4_audit.py) so these tests never require mmcv, cv2,
or a CUDA device except where explicitly guarded.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
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
KSWEEP_PATH = EVAL_DIR / "graph_degree_sweep.py"
CONSENSUS_PATH = EVAL_DIR / "consensus_replacement.py"
IDENTITY_PATH = EVAL_DIR / "baseline_identity.py"
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
ksweep = _load("segmentation.evaluation.graph_degree_sweep", KSWEEP_PATH)
consensus = _load("segmentation.evaluation.consensus_replacement", CONSENSUS_PATH)
baseline_identity = _load("segmentation.evaluation.baseline_identity", IDENTITY_PATH)
cover_dr = _load_cover_dr_package()

from src.rwr_reproduction_identity import load_identity  # noqa: E402

IDENTITY = load_identity(repo_root=ROOT)

SpatialSize = geometry.SpatialSize
SlidingWindowPlan = geometry.SlidingWindowPlan


# ---------------------------------------------------------------------------
# Shared fixture builders (mirrors tests/test_t4_audit.py's pattern).
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
    source_s0=[5.0, 0.0, 0.0],    # u = class 0
    source_p=[-10.0, 10.0, 0.0],  # d = class 1, high confidence
    other_s0=[5.0, 0.0, 0.0],
    other_p=[1.0, 0.0, 0.0],      # y = class 0, mild confidence
)


def _t4_signal_and_cache():
    cache, context = _overlap_cache_and_context(**REVERSAL_FIXTURE)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    return cache, context, signal


def _windows_from_cache_probabilities(cache, class_count):
    """Build stitching.WindowProbabilityMap objects from a cache's own RAW
    (pre-sigmoid) propagated_scores, upsampled via patch_scores_to_masks --
    the same single-sigmoid pipeline production uses. Must NOT pre-sigmoid
    before calling patch_scores_to_masks: that function applies sigmoid
    itself, and doing so twice was the double-sigmoid defect this helper
    used to independently reproduce (see consensus_replacement.py's
    probability_grid_to_mask docstring)."""
    from models.dinotext.cover_dr.inference import patch_scores_to_masks

    out = []
    for state in cache.windows_in_order():
        extent = state.geometry.extent.as_tuple()
        masks = patch_scores_to_masks(state.propagated_scores.unsqueeze(0), state.patch_grid_shape, extent)[0]
        out.append(
            stitching.WindowProbabilityMap(
                geometry=state.geometry, window_index=state.window_index, probabilities=masks
            )
        )
    return out


# ---------------------------------------------------------------------------
# 1-10: stitching
# ---------------------------------------------------------------------------


def _manual_uniform_stitch(windows, class_count, image_size):
    h, w = image_size.as_tuple()
    preds = torch.zeros(1, class_count, h, w)
    count = torch.zeros(1, 1, h, w)
    for window in windows:
        rows, cols = window.geometry.accumulation_slice
        preds[:, :, rows, cols] += window.probabilities.unsqueeze(0)
        count[:, :, rows, cols] += 1
    return (preds / count)[0]


def test_uniform_matches_canonical_implementation():
    cache, _context, _signal = _t4_signal_and_cache()
    windows = _windows_from_cache_probabilities(cache, 3)
    plan = cache.plan
    result = stitching.stitch_windows(
        windows, image_size=plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_k12_q",
    )
    manual = _manual_uniform_stitch(windows, 3, plan.image_size)
    assert torch.allclose(result.image, manual, atol=1e-6)


def test_majority_voting_and_exact_ties():
    plan = _plan(image=(4, 4), crop=(4, 4), stride=(4, 4))
    a = stitching.WindowProbabilityMap(
        geometry=plan.windows[0], window_index=0,
        probabilities=torch.tensor([[[0.9]], [[0.1]], [[0.0]]]).expand(3, 4, 4).contiguous(),
    )
    plan2 = _plan(image=(4, 4), crop=(4, 4), stride=(4, 4))
    b_geom = plan2.windows[0]
    b = stitching.WindowProbabilityMap(
        geometry=b_geom, window_index=1,
        probabilities=torch.tensor([[[0.1]], [[0.9]], [[0.0]]]).expand(3, 4, 4).contiguous(),
    )
    # two windows, class 0 vs class 1: an exact 1-1 tie everywhere.
    result = stitching.stitch_windows(
        [a, b], image_size=plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_MAJORITY, score_source="rwr_k12_q",
    )
    assert result.diagnostics.tie_count == 16
    assert result.diagnostics.tie_fraction == 1.0
    # lowest class id wins the tie everywhere.
    assert bool((result.image.argmax(dim=0) == 0).all())


def test_half_sample_hann_formula():
    weight = stitching.hann_window_weight((4, 1))
    expected = torch.tensor(
        [math.sin(math.pi * (n + 0.5) / 4) ** 2 for n in range(4)], dtype=weight.dtype
    ).unsqueeze(-1)
    assert torch.allclose(weight, expected, atol=1e-12)


def test_hann_weights_strictly_positive():
    weight = stitching.hann_window_weight((17, 23))
    assert bool(torch.all(weight > 0))


def test_hann_normalization_stays_in_unit_range():
    cache, _context, _signal = _t4_signal_and_cache()
    windows = _windows_from_cache_probabilities(cache, 3)
    result = stitching.stitch_windows(
        windows, image_size=cache.plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_HANN, score_source="rwr_k12_q",
    )
    assert float(result.image.min()) >= -1e-5
    assert float(result.image.max()) <= 1.0 + 1e-5
    assert result.diagnostics.min_weight_denominator > 0


def test_center_select_tie_breaking_uses_lowest_ordinal():
    plan = _plan(image=(4, 4), crop=(4, 4), stride=(4, 4))
    a = stitching.WindowProbabilityMap(
        geometry=plan.windows[0], window_index=0,
        probabilities=torch.full((3, 4, 4), 0.0),
    )
    b = stitching.WindowProbabilityMap(
        geometry=plan.windows[0], window_index=1,
        probabilities=torch.full((3, 4, 4), 1.0),
    )
    result = stitching.stitch_windows(
        [a, b], image_size=plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_CENTER_SELECT, score_source="rwr_k12_q",
    )
    # identical extents -> identical Hann weight everywhere -> exact tie ->
    # lowest window_index (0, all-zero probabilities) must win.
    assert torch.allclose(result.image, torch.zeros(3, 4, 4))


def test_clamped_and_non_square_windows():
    plan = _plan(image=(7, 5), crop=(4, 4), stride=(3, 3))
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        n = 2 * 2
        cache.append(_make_state(window, [[1.0, 0.0, 0.0]] * n, [[1.0, 0.0, 0.0]] * n, (2, 2), 3))
    cache.seal()
    windows = _windows_from_cache_probabilities(cache, 3)
    for mode in stitching.STITCH_MODES:
        result = stitching.stitch_windows(
            windows, image_size=plan.image_size, class_count=3, mode=mode, score_source="rwr_k12_q"
        )
        assert result.image.shape == (3, 7, 5)


def test_stitching_fails_closed_on_uncovered_pixels():
    plan = _plan(image=(6, 6), crop=(4, 4), stride=(2, 2))
    window = plan.windows[0]
    probs = torch.rand(3, *window.extent.as_tuple()).clamp(0, 1)
    single = stitching.WindowProbabilityMap(geometry=window, window_index=0, probabilities=probs)
    with pytest.raises(stitching.StitchingError):
        stitching.stitch_windows(
            [single], image_size=plan.image_size, class_count=3,
            mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_k12_q",
        )


def test_score_source_labels_are_recorded_and_distinct():
    cache, _context, _signal = _t4_signal_and_cache()
    windows = _windows_from_cache_probabilities(cache, 3)
    e3_result = stitching.stitch_windows(
        windows, image_size=cache.plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source=stitching.SCORE_SOURCE_E3_UNARY,
    )
    rwr_result = stitching.stitch_windows(
        windows, image_size=cache.plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source=stitching.SCORE_SOURCE_RWR_K12,
    )
    assert e3_result.diagnostics.score_source != rwr_result.diagnostics.score_source


def test_stitching_has_no_gt_parameter_anywhere():
    for fn in (stitching.stitch_windows,):
        params = set(inspect.signature(fn).parameters)
        assert not any("gt" in name.lower() or "ground_truth" in name.lower() for name in params)


# ---------------------------------------------------------------------------
# 11-20: k sweep
# ---------------------------------------------------------------------------


def _single_window_state(n=20, d=8, c=3, seed=0):
    plan = _plan(image=(4, 4), crop=(4, 4), stride=(4, 4))
    window = plan.windows[0]
    generator = torch.Generator().manual_seed(seed)
    s0 = torch.randn(n, c, generator=generator)
    p = torch.randn(n, c, generator=generator)
    feats = F.normalize(torch.randn(n, d, generator=generator), dim=-1)
    indices = torch.zeros(n, 2, dtype=torch.int64)
    weights = torch.full((n, 2), 0.5)
    affinities = torch.ones(n, 2)
    fallback = torch.zeros(n, dtype=torch.bool)
    graph = wc.GraphSnapshot(indices, weights, affinities, fallback, n, 2, 3.0)
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    return wc.CachedWindowState(
        geometry=window, window_index=window.index, patch_grid_shape=(1, n),
        class_count=c, s0=s0, dino_features=feats, graph=graph,
        propagated_scores=p, solver_summary=telemetry,
    )


def test_k_sweep_exact_configured_k_set():
    assert ksweep.DEFAULT_K_VALUES == (4, 6, 8, 10, 11, 12, 16, 32)


def test_k_sweep_rejects_duplicate_k():
    state = _single_window_state(n=20)
    with pytest.raises(ksweep.GraphDegreeSweepError):
        ksweep.run_k_sweep_for_window(
            state, k_values=(4, 4), affinity_power=3.0, alpha=0.98,
            rtol=None, atol=None, max_iter=200,
        )


def test_k_sweep_directed_graph_preserved_no_symmetrization():
    state = _single_window_state(n=20)
    results = ksweep.run_k_sweep_for_window(
        state, k_values=(4,), affinity_power=3.0, alpha=0.98, rtol=None, atol=None, max_iter=200,
    )
    assert 4 in results
    assert results[4].scores.shape == state.s0.shape


def test_k_sweep_reuses_same_features_and_s0_without_mutation():
    state = _single_window_state(n=20)
    s0_before = state.s0.clone()
    features_before = state.dino_features.clone()
    ksweep.run_k_sweep_for_window(
        state, k_values=(4, 8, 12), affinity_power=3.0, alpha=0.98,
        rtol=None, atol=None, max_iter=200,
    )
    assert torch.equal(state.s0, s0_before)
    assert torch.equal(state.dino_features, features_before)


def test_k_sweep_one_graph_and_solve_call_per_k():
    state = _single_window_state(n=20)
    calls = {"graph": 0, "solve": 0}
    real_build = cover_dr.build_directed_topk_graph
    real_solve = cover_dr.solve_rwr_cgls

    def counted_build(*args, **kwargs):
        calls["graph"] += 1
        return real_build(*args, **kwargs)

    def counted_solve(*args, **kwargs):
        calls["solve"] += 1
        return real_solve(*args, **kwargs)

    import models.dinotext.cover_dr.graph as graph_module
    import models.dinotext.cover_dr.rwr as rwr_module

    original_build = graph_module.build_directed_topk_graph
    original_solve = rwr_module.solve_rwr_cgls
    graph_module.build_directed_topk_graph = counted_build
    rwr_module.solve_rwr_cgls = counted_solve
    try:
        k_values = (4, 8, 12)
        ksweep.run_k_sweep_for_window(
            state, k_values=k_values, affinity_power=3.0, alpha=0.98,
            rtol=None, atol=None, max_iter=200,
        )
    finally:
        graph_module.build_directed_topk_graph = original_build
        rwr_module.solve_rwr_cgls = original_solve
    assert calls["graph"] == 3
    assert calls["solve"] == 3


def test_k_sweep_k11_equals_direct_top11_construction():
    from models.dinotext.cover_dr.graph import build_directed_topk_graph
    from models.dinotext.cover_dr.rwr import solve_rwr_cgls

    state = _single_window_state(n=20)
    swept = ksweep.run_k_sweep_for_window(
        state, k_values=(11,), affinity_power=3.0, alpha=0.98, rtol=None, atol=None, max_iter=200,
    )
    direct_graph = build_directed_topk_graph(state.dino_features_copy(), k=11, affinity_power=3.0)
    direct = solve_rwr_cgls(direct_graph, state.s0_copy(), alpha=0.98, rtol=None, atol=None, max_iter=200)
    assert torch.allclose(swept[11].scores, direct.scores, atol=1e-6)


def test_k_sweep_k12_identity_matches_canonical_single_solve():
    from models.dinotext.cover_dr.graph import build_directed_topk_graph
    from models.dinotext.cover_dr.rwr import solve_rwr_cgls

    state = _single_window_state(n=20)
    swept = ksweep.run_k_sweep_for_window(
        state, k_values=(12,), affinity_power=3.0, alpha=0.98, rtol=None, atol=None, max_iter=200,
    )
    canonical_graph = build_directed_topk_graph(state.dino_features_copy(), k=12, affinity_power=3.0)
    canonical = solve_rwr_cgls(canonical_graph, state.s0_copy(), alpha=0.98, rtol=None, atol=None, max_iter=200)
    assert torch.allclose(swept[12].scores, canonical.scores, atol=1e-6)


def test_k_sweep_telemetry_separated_by_k():
    state = _single_window_state(n=40)
    swept = ksweep.run_k_sweep_for_window(
        state, k_values=(4, 32), affinity_power=3.0, alpha=0.98, rtol=None, atol=None, max_iter=500,
    )
    assert set(swept.keys()) == {4, 32}
    assert swept[4] is not swept[32]


def test_k_sweep_rejects_nonpositive_k():
    state = _single_window_state(n=20)
    with pytest.raises(ksweep.GraphDegreeSweepError):
        ksweep.run_k_sweep_for_window(
            state, k_values=(0,), affinity_power=3.0, alpha=0.98, rtol=None, atol=None, max_iter=200,
        )


def test_k_sweep_requires_nonempty_k_values():
    state = _single_window_state(n=20)
    with pytest.raises(ksweep.GraphDegreeSweepError):
        ksweep.run_k_sweep_for_window(
            state, k_values=(), affinity_power=3.0, alpha=0.98, rtol=None, atol=None, max_iter=200,
        )


# ---------------------------------------------------------------------------
# 21-28: DCR
# ---------------------------------------------------------------------------


def _strict_t4_observation(signal):
    matches = [o for o in signal.observations if o.t4]
    assert len(matches) == 1
    return matches[0]


def test_dcr_hard_one_hot_replacement():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    assert len(replacements) == 1
    vector = replacements[0].vector
    assert torch.equal(vector, torch.tensor([1.0, 0.0, 0.0]))
    obs = _strict_t4_observation(signal)
    assert replacements[0].consensus_label == obs.consensus_label == 0


def test_dcr_jury_mean_full_vector_replacement():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_JURY_MEAN, image_id="img", class_count=3,
    )
    assert len(replacements) == 1
    vector = replacements[0].vector
    assert vector.shape == (3,)
    assert bool((vector >= 0).all()) and bool((vector <= 1).all())
    # jury-mean must differ from a pure one-hot in general (it is an average
    # of sigmoid probabilities, not a hard label).
    assert not torch.equal(vector, torch.tensor([1.0, 0.0, 0.0]))


def test_dcr_jury_mean_rejects_source_in_jury():
    cache, _context, signal = _t4_signal_and_cache()
    obs = _strict_t4_observation(signal)
    import dataclasses

    tampered = dataclasses.replace(
        obs, other_covering_window_ids=obs.other_covering_window_ids + (obs.source_window_id,)
    )
    sigmoid_cache = consensus._SigmoidScoreCache(cache)
    with pytest.raises(consensus.ConsensusReplacementError):
        consensus.compute_dcr_jury_mean_vector(cache, sigmoid_cache, tampered)


def test_frozen_replacement_vectors_are_independent_copies():
    obs = _strict_t4_observation(_t4_signal_and_cache()[2])
    source_vector = torch.tensor([1.0, 0.0, 0.0])
    frozen = consensus.FrozenReplacement(
        image_id="img", source_window_id=obs.source_window_id, node_index=obs.node_index,
        node_row=obs.node_row, node_col=obs.node_col, consensus_label=obs.consensus_label,
        kind=consensus.DCR_HARD, vector=source_vector,
    )
    source_vector += 100.0  # mutate the caller's original tensor after construction
    assert torch.equal(frozen.vector, torch.tensor([1.0, 0.0, 0.0]))  # stored copy unaffected


def test_dcr_simultaneous_replacement_across_two_targets():
    plan = _plan(image=(8, 8), crop=(4, 4), stride=(2, 2))
    n = 4
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        if window.index in (0, 3):
            cache.append(_make_state(window, [REVERSAL_FIXTURE["source_s0"]] * n, [REVERSAL_FIXTURE["source_p"]] * n, (2, 2), 3, seed=window.index))
        else:
            cache.append(_make_state(window, [REVERSAL_FIXTURE["other_s0"]] * n, [REVERSAL_FIXTURE["other_p"]] * n, (2, 2), 3, seed=window.index))
    cache.seal()
    context = _context(plan, 3, (2, 2), cache)
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img")
    t4_targets = [o for o in signal.observations if o.t4]
    assert len({o.source_window_id for o in t4_targets}) >= 1
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    result, report = consensus.apply_replacements_and_stitch(
        cache, replacements, image_size=plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_dcr_hard",
    )
    assert report.target_rows == len(replacements)
    assert result.image.shape == (3, 8, 8)


def test_dcr_duplicate_global_anchor_reporting_is_zero_for_disjoint_targets():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    _result, report = consensus.apply_replacements_and_stitch(
        cache, replacements, image_size=cache.plan.image_size, class_count=3,
        score_source="rwr_dcr_hard",
    )
    assert report.duplicate_global_anchor_count == 0
    assert report.unique_windows == len({r.source_window_id for r in replacements})


def test_dcr_defaults_to_strict_t4_population():
    signature = inspect.signature(consensus.build_frozen_replacements)
    assert signature.parameters["target_population"].default == consensus.TARGET_POPULATION_STRICT_T4


def test_dcr_and_sur_functions_have_no_gt_parameter():
    for fn in (
        consensus.build_frozen_replacements,
        consensus.compute_dcr_jury_mean_vector,
        consensus.apply_replacements_and_stitch,
        consensus.run_strict_safe_sequential,
    ):
        params = set(inspect.signature(fn).parameters)
        assert not any("gt" == name.lower() or "ground_truth" in name.lower() for name in params)


# ---------------------------------------------------------------------------
# 29-33: SUR
# ---------------------------------------------------------------------------


def test_sur_uses_sigmoid_of_s0():
    cache, _context, signal = _t4_signal_and_cache()
    obs = _strict_t4_observation(signal)
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.SUR, image_id="img", class_count=3,
    )
    source_state = cache.get(obs.source_window_id)
    expected = torch.sigmoid(source_state.s0[obs.node_index])
    assert torch.allclose(replacements[0].vector, expected)


def test_sur_vector_is_probability_not_raw_logit():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.SUR, image_id="img", class_count=3,
    )
    vector = replacements[0].vector
    assert bool((vector >= 0).all()) and bool((vector <= 1).all())


def test_sur_only_targeted_rows_change():
    cache, _context, signal = _t4_signal_and_cache()
    obs = _strict_t4_observation(signal)
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.SUR, image_id="img", class_count=3,
    )
    baseline_q = {
        state.window_index: torch.sigmoid(state.propagated_scores).clone()
        for state in cache.windows_in_order()
    }
    for state in cache.windows_in_order():
        q = torch.sigmoid(state.propagated_scores).clone()
        if state.window_index == obs.source_window_id:
            for r in replacements:
                if r.source_window_id == state.window_index:
                    q[r.node_index] = r.vector
            other_rows = [i for i in range(q.shape[0]) if i != obs.node_index]
            assert torch.allclose(q[other_rows], baseline_q[state.window_index][other_rows])
        else:
            assert torch.allclose(q, baseline_q[state.window_index])


def test_sur_simultaneous_application_is_deterministic():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.SUR, image_id="img", class_count=3,
    )
    result_a, _ = consensus.apply_replacements_and_stitch(
        cache, replacements, image_size=cache.plan.image_size, class_count=3, score_source="rwr_sur"
    )
    result_b, _ = consensus.apply_replacements_and_stitch(
        cache, replacements, image_size=cache.plan.image_size, class_count=3, score_source="rwr_sur"
    )
    assert torch.equal(result_a.image, result_b.image)


# ---------------------------------------------------------------------------
# 34-42: strict-safe DCR/SUR
# ---------------------------------------------------------------------------


def test_strict_safe_protected_set_is_every_t2_or_higher_observation():
    cache, _context, signal = _t4_signal_and_cache()
    assert all(obs.t2 for obs in signal.observations)


def test_strict_safe_accepts_a_violation_reducing_candidate():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    report = consensus.run_strict_safe_sequential(cache, signal, replacements)
    assert report.accepted >= 1
    assert report.new_violations == 0


def test_strict_safe_rejects_candidate_that_introduces_a_new_violation():
    # Two independent source windows, each individually a strict-T4 target
    # against its own jury; construct a class-3 "bystander" replacement kind
    # by hand that flips one target's row to a value which would also
    # disturb a currently-satisfied neighboring anchor -- verified by
    # asserting that whatever the strict-safe loop rejects, rejection is
    # always because of a would-be new violation, never a cardinality tie.
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    # Build one adversarial extra candidate at the SAME anchor with a
    # deliberately wrong (non-consensus) vector; it can only ever be
    # rejected, since it cannot shrink the violation set for its own
    # anchor while it cannot fix any other anchor either.
    obs = _strict_t4_observation(signal)
    adversarial = consensus.FrozenReplacement(
        image_id="img", source_window_id=obs.source_window_id, node_index=obs.node_index,
        node_row=obs.node_row, node_col=obs.node_col, consensus_label=obs.consensus_label,
        kind=consensus.DCR_HARD, vector=torch.tensor([0.0, 0.0, 1.0]),
    )
    report = consensus.run_strict_safe_sequential(cache, signal, [adversarial])
    assert report.accepted == 0
    assert report.rejected == 1
    assert report.new_violations == 0


def test_strict_safe_no_net_progress_fallback():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    report = consensus.run_strict_safe_sequential(cache, signal, replacements)
    # every accepted candidate must have strictly shrunk the violation set;
    # this is enforced structurally by run_strict_safe_sequential's use of
    # Python's strict-subset operator, not by a cardinality comparison.
    source = inspect.getsource(consensus.run_strict_safe_sequential)
    assert "candidate_violations < current_violations" in source
    assert "len(" not in source.split("if candidate_violations")[1].split("\n")[0]


def test_strict_safe_deterministic_candidate_order():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    forward = consensus.run_strict_safe_sequential(cache, signal, replacements)
    backward = consensus.run_strict_safe_sequential(cache, signal, tuple(reversed(replacements)))
    assert forward.accepted == backward.accepted
    assert forward.rejected == backward.rejected


def test_strict_safe_uses_frozen_replacement_vectors():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    before = [r.vector.clone() for r in replacements]
    consensus.run_strict_safe_sequential(cache, signal, replacements)
    for r, snapshot in zip(replacements, before):
        assert torch.equal(r.vector, snapshot)


def test_strict_safe_reports_zero_new_violations():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.SUR, image_id="img", class_count=3,
    )
    report = consensus.run_strict_safe_sequential(cache, signal, replacements)
    assert report.new_violations == 0


def test_strict_safe_no_gt_parameter():
    params = set(inspect.signature(consensus.run_strict_safe_sequential).parameters)
    assert not any("gt" == name.lower() or "ground_truth" in name.lower() for name in params)


def test_strict_safe_empty_and_fully_rejected_cases():
    cache, _context, signal = _t4_signal_and_cache()
    empty_report = consensus.run_strict_safe_sequential(cache, signal, [])
    assert empty_report.candidates == 0
    assert empty_report.accepted == 0
    assert empty_report.acceptance_rate == 0.0

    obs = _strict_t4_observation(signal)
    adversarial = consensus.FrozenReplacement(
        image_id="img", source_window_id=obs.source_window_id, node_index=obs.node_index,
        node_row=obs.node_row, node_col=obs.node_col, consensus_label=obs.consensus_label,
        kind=consensus.DCR_HARD, vector=torch.tensor([0.0, 0.0, 1.0]),
    )
    rejected_report = consensus.run_strict_safe_sequential(cache, signal, [adversarial])
    assert rejected_report.accepted == 0
    assert rejected_report.rejected == 1


# ---------------------------------------------------------------------------
# 43-51: metrics, GT isolation, identity
# ---------------------------------------------------------------------------


def test_target_off_target_gt_accounting():
    before = torch.zeros(3, 4, 4)
    before[0] = 1.0  # predicts class 0 everywhere before
    after = before.clone()
    after[:, 0, 0] = torch.tensor([0.0, 1.0, 0.0])  # target pixel flips to class 1 (correct)
    after[:, 3, 3] = torch.tensor([0.0, 0.0, 1.0])  # off-target pixel flips to class 2 (wrong)
    gt = torch.zeros(4, 4, dtype=torch.int64)
    gt[0, 0] = 1
    gt[3, 3] = 0
    target_mask = torch.zeros(4, 4, dtype=torch.bool)
    target_mask[0, 0] = True
    accounting = consensus.compute_gt_accounting(before, after, gt, target_mask, ignore_index=255)
    assert accounting.target_gt_gains == 1
    assert accounting.target_gt_losses == 0
    assert accounting.off_target_gt_gains == 0
    assert accounting.off_target_gt_losses == 1
    assert accounting.net_changed_correct_pixels == 0
    assert accounting.changed_pixel_count == 2
    assert accounting.per_class_changes[1]["gains"] == 1
    assert accounting.per_class_changes[0]["losses"] == 1


def test_gt_accounting_respects_ignore_index():
    before = torch.zeros(2, 3, 3)
    before[0] = 1.0
    after = before.clone()
    after[:, 1, 1] = torch.tensor([0.0, 1.0])
    gt = torch.zeros(3, 3, dtype=torch.int64)
    gt[1, 1] = 255  # ignored pixel, must not count despite the flip
    target_mask = torch.ones(3, 3, dtype=torch.bool)
    accounting = consensus.compute_gt_accounting(before, after, gt, target_mask, ignore_index=255)
    assert accounting.changed_pixel_count == 0


def test_target_pixel_mask_marks_only_replaced_node_footprints():
    cache, _context, signal = _t4_signal_and_cache()
    replacements = consensus.build_frozen_replacements(
        cache, signal, kind=consensus.DCR_HARD, image_id="img", class_count=3,
    )
    mask = consensus.compute_target_pixel_mask(cache, replacements, cache.plan.image_size)
    assert mask.dtype == torch.bool
    assert bool(mask.any())
    assert not bool(mask.all())


def test_baseline_manifest_builds_from_real_rwr_identity():
    manifest = baseline_identity.build_baseline_manifest(
        rwr_identity=IDENTITY,
        canonical_config_sha256=IDENTITY["canonical_config_sha256"],
        checkpoint_sha256=IDENTITY["checkpoint"]["sha256"],
    )
    assert manifest["rwr"]["alpha"] == IDENTITY["rwr"]["alpha"]
    assert manifest["k_sweep"]["canonical_k"] == IDENTITY["rwr"]["top_k"]
    assert manifest["k_sweep"]["canonical_k"] in manifest["k_sweep"]["values"]


def test_baseline_manifest_rejects_unknown_field():
    manifest = baseline_identity.build_baseline_manifest(
        rwr_identity=IDENTITY,
        canonical_config_sha256=IDENTITY["canonical_config_sha256"],
        checkpoint_sha256=IDENTITY["checkpoint"]["sha256"],
    )
    broken = dict(manifest)
    broken["unexpected_field"] = True
    with pytest.raises(baseline_identity.BaselineIdentityError):
        baseline_identity.validate_baseline_manifest(broken)


def test_baseline_manifest_rejects_wrong_exact_type():
    manifest = baseline_identity.build_baseline_manifest(
        rwr_identity=IDENTITY,
        canonical_config_sha256=IDENTITY["canonical_config_sha256"],
        checkpoint_sha256=IDENTITY["checkpoint"]["sha256"],
    )
    broken = dict(manifest)
    broken["rwr"] = dict(broken["rwr"])
    broken["rwr"]["alpha"] = 1  # int, not exact float
    with pytest.raises(baseline_identity.BaselineIdentityError):
        baseline_identity.validate_baseline_manifest(broken)


def test_disabled_baselines_reduce_to_canonical_uniform_stitch():
    cache, _context, signal = _t4_signal_and_cache()
    canonical_windows = _windows_from_cache_probabilities(cache, 3)
    canonical = stitching.stitch_windows(
        canonical_windows, image_size=cache.plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_k12_q",
    )
    disabled, _report = consensus.apply_replacements_and_stitch(
        cache, [], image_size=cache.plan.image_size, class_count=3,
        mode=stitching.STITCH_MODE_UNIFORM, score_source="rwr_k12_q",
    )
    assert torch.allclose(canonical.image, disabled.image, atol=1e-6)
