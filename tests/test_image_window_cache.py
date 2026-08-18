"""Tests for the opt-in image-scoped two-pass sliding-window cache.

Loaded via file-path import with lightweight stand-ins for ``mmcv``/
``utils`` (mirroring ``tests/test_rwr_integration.py`` and
``tests/test_sliding_window_geometry.py``) so these tests never require
mmcv, cv2, or a CUDA device. Most tests use a plain ``SimpleNamespace``
"inference" stand-in exposing only the attributes the cache orchestrator
reads (``model``, ``text_embedding``, ``rwr_config``, ``test_cfg``,
``num_classes``, ``with_bg``); the "production integration" section loads
the real ``DINOTextSegInference`` to compare against.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
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
SEGMENTATION_PATH = EVAL_DIR / "dinotext_seg.py"
COVER_DR_PACKAGE_PATH = ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr"


def _install_segmentation_package_stub() -> None:
    """Give ``segmentation.evaluation`` a real package context pointing at
    the actual directory on disk, so file-based module loads below share
    classes with each other (both resolve the geometry module by the same
    dotted name)."""
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

SpatialSize = geometry.SpatialSize
SlidingWindowPlan = geometry.SlidingWindowPlan
WindowGeometry = geometry.WindowGeometry


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


# ---------------------------------------------------------------------------
# Synthetic deterministic model (no mmcv/backbone needed)
# ---------------------------------------------------------------------------


class SyntheticModel(nn.Module):
    """Deterministic stand-in for DINOText: average-pools each patch, then
    a fixed random projection to features/scores. No randomness at call
    time, so the same crop always produces bit-identical output."""

    def __init__(self, patch_size=2, embed_dim=6, class_count=3, seed=0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.class_count = class_count
        generator = torch.Generator().manual_seed(seed)
        self.feature_proj = torch.randn(3, embed_dim, generator=generator)
        self.score_proj = torch.randn(embed_dim, class_count, generator=generator)
        self.snapshot_call_count = 0
        self.downstream_call_count = 0

    def generate_patch_snapshot(self, crop, text_embedding):
        del text_embedding
        self.snapshot_call_count += 1
        _, channels, height, width = crop.shape
        ps = self.patch_size
        grid_h, grid_w = height // ps, width // ps
        pooled = F.avg_pool2d(crop, ps)  # [1, C, gh, gw]
        flat = pooled.reshape(1, channels, grid_h * grid_w).permute(0, 2, 1)  # [1, N, C]
        raw_feat = flat @ self.feature_proj  # [1, N, embed_dim]
        # break positional degeneracy so distinct crops/positions differ
        position_bias = 0.01 * torch.arange(grid_h * grid_w, dtype=torch.float32).view(1, -1, 1)
        raw_feat = raw_feat + position_bias
        features = F.normalize(raw_feat, dim=-1)
        scores = features @ self.score_proj  # [1, N, class_count]
        return types.SimpleNamespace(
            unary_scores=scores, dino_features=features, grid_hw=(grid_h, grid_w)
        )

    def masks_from_patch_scores(self, patch_scores, grid_hw, output_hw):
        self.downstream_call_count += 1
        batch, _num_patches, classes = patch_scores.shape
        grid_h, grid_w = grid_hw
        simmap = patch_scores.reshape(batch, grid_h, grid_w, classes).permute(0, 3, 1, 2)
        mask = torch.sigmoid(simmap)
        return F.interpolate(mask, tuple(output_hw), mode="bilinear", align_corners=True)

    def generate_masks(self, image, text_emb, apply_pamr=False):
        del apply_pamr
        snapshot = self.generate_patch_snapshot(image, text_emb)
        masks = self.masks_from_patch_scores(
            snapshot.unary_scores, snapshot.grid_hw, tuple(image.shape[-2:])
        )
        return masks, None


def _config(**overrides):
    identity = load_identity(repo_root=ROOT)
    values = {
        "enabled": True,
        "identity_path": "evaluation_identities/e3_canonical_directed_rwr.toml",
        "graph_mode": identity["rwr"]["graph_mode"],
        "alpha": 0.5,
        "top_k": 4,
        "affinity_power": identity["rwr"]["affinity_power"],
        "solver": identity["solver"]["method"],
        "solver_rtol": 1e-4,
        "solver_atol": 1e-6,
        "solver_max_iterations": 2000,
        "expected_class_count": 3,
        "config_path": identity["canonical_config_path"],
    }
    values.update(overrides)
    config = cover_dr.RWRInferenceConfig(**values)
    config.validate()
    return config


def _make_inference(**config_overrides):
    """A minimal 'inference' stand-in exposing only the attributes
    run_pass_one/run_pass_two read from a real DINOTextSegInference."""
    model = SyntheticModel(patch_size=2, embed_dim=6, class_count=3, seed=0)
    return types.SimpleNamespace(
        model=model,
        text_embedding=torch.randn(3, 4),
        rwr_config=_config(**config_overrides),
        test_cfg=types.SimpleNamespace(stride=(4, 4), crop_size=(8, 8)),
        num_classes=3,
        with_bg=False,
    )


def _make_state(window, N=8, D=4, C=3, k=3, seed=None):
    generator = torch.Generator().manual_seed(seed if seed is not None else window.index)
    s0 = torch.randn(N, C, generator=generator)
    features = torch.randn(N, D, generator=generator)
    propagated = torch.randn(N, C, generator=generator)
    indices = torch.randint(0, N, (N, k), generator=generator).to(torch.int64)
    weights = torch.rand(N, k, generator=generator)
    weights = weights / weights.sum(dim=1, keepdim=True)
    affinities = torch.rand(N, k, generator=generator)
    fallback = torch.zeros(N, dtype=torch.bool)
    graph = wc.GraphSnapshot(indices, weights, affinities, fallback, N, k, 3.0)
    telemetry = wc.WindowSolverTelemetry(
        iterations=5, work_count=5, restarts=0, residual_replacements=0,
        fallback_rows=0, maximum_scaled_residual=0.1,
    )
    return wc.CachedWindowState(
        geometry=window, window_index=window.index, patch_grid_shape=(N // 2, 2), class_count=C,
        s0=s0, dino_features=features, graph=graph, propagated_scores=propagated,
        solver_summary=telemetry,
    )


def _plan(image=(20, 20), crop=(10, 10), stride=(10, 10)):
    return SlidingWindowPlan.build(
        image_size=SpatialSize(*image), crop_size=SpatialSize(*crop), stride=SpatialSize(*stride)
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_lifecycle_open_sealed_replaying_closed():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    assert cache.state is wc.WindowCacheState.OPEN
    for window in plan.windows:
        cache.append(_make_state(window))
    assert cache.cached_window_count == plan.window_count == 4
    cache.seal()
    assert cache.state is wc.WindowCacheState.SEALED
    cache.begin_replay()
    assert cache.state is wc.WindowCacheState.REPLAYING
    assert cache.get(0).window_index == 0
    cache.end_replay()
    assert cache.state is wc.WindowCacheState.SEALED
    cache.close()
    assert cache.state is wc.WindowCacheState.CLOSED


def test_expected_window_count():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    assert cache.expected_window_count == plan.window_count == 4


def test_missing_window_fails_seal():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    cache.append(_make_state(plan.windows[0]))
    with pytest.raises(wc.WindowCacheError, match="expected 4 windows"):
        cache.seal()
    cache.close()


def test_duplicate_window_fails():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    cache.append(_make_state(plan.windows[0]))
    with pytest.raises(wc.WindowCacheError, match="duplicate"):
        cache.append(_make_state(plan.windows[0]))
    cache.close()


def test_out_of_order_window_fails():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    with pytest.raises(wc.WindowCacheError, match="out-of-order"):
        cache.append(_make_state(plan.windows[2]))
    cache.close()


def test_replay_before_seal_fails():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    with pytest.raises(wc.WindowCacheError, match="sealed"):
        cache.begin_replay()
    cache.close()


def test_append_after_seal_fails():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        cache.append(_make_state(window))
    cache.seal()
    with pytest.raises(wc.WindowCacheError, match="sealed"):
        cache.append(_make_state(plan.windows[0]))
    cache.close()


def test_access_after_close_fails():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    for window in plan.windows:
        cache.append(_make_state(window))
    cache.seal()
    cache.close()
    with pytest.raises(wc.WindowCacheError, match="closed"):
        cache.get(0)
    with pytest.raises(wc.WindowCacheError, match="closed"):
        list(cache.windows_in_order())
    with pytest.raises(wc.WindowCacheError, match="closed"):
        cache.total_bytes()


def test_cleanup_after_exception_via_context_manager():
    plan = _plan()
    with pytest.raises(RuntimeError, match="boom"):
        with wc.ImageWindowCache(plan) as cache:
            cache.append(_make_state(plan.windows[0]))
            raise RuntimeError("boom")
    assert cache.state is wc.WindowCacheState.CLOSED
    assert cache.cached_window_count == 0


def test_geometry_mismatch_against_plan_fails():
    plan = _plan()
    cache = wc.ImageWindowCache(plan)
    forged_geometry = dataclasses.replace(plan.windows[3], index=0)
    forged_state = _make_state(forged_geometry, seed=0)
    with pytest.raises(wc.WindowCacheError, match="geometry"):
        cache.append(forged_state)
    cache.close()


# ---------------------------------------------------------------------------
# Tensor ownership
# ---------------------------------------------------------------------------


def test_source_mutation_cannot_affect_cache():
    plan = _plan()
    window = plan.windows[0]
    s0 = torch.zeros(4, 2)
    features = torch.zeros(4, 3)
    propagated = torch.zeros(4, 2)
    indices = torch.zeros(4, 2, dtype=torch.int64)
    weights = torch.full((4, 2), 0.5)
    affinities = torch.ones(4, 2)
    fallback = torch.zeros(4, dtype=torch.bool)
    graph = wc.GraphSnapshot(indices, weights, affinities, fallback, 4, 2, 3.0)
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    state = wc.CachedWindowState(
        geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
        s0=s0, dino_features=features, graph=graph, propagated_scores=propagated,
        solver_summary=telemetry,
    )
    s0[0, 0] = 999.0
    features[0, 0] = 999.0
    indices[0, 0] = 3
    weights[0, 0] = 0.99
    assert state.s0[0, 0].item() == 0.0
    assert state.dino_features[0, 0].item() == 0.0
    assert state.graph.neighbor_indices[0, 0].item() == 0
    assert state.graph.transition_weights[0, 0].item() == 0.5


def test_replay_clone_mutation_cannot_affect_cache():
    plan = _plan()
    state = _make_state(plan.windows[0])
    clone = state.propagated_scores_copy()
    original = state.propagated_scores.clone()
    clone[0, 0] = 12345.0
    assert torch.equal(state.propagated_scores, original)
    features_clone = state.dino_features_copy()
    features_clone.fill_(999.0)
    assert not torch.equal(state.dino_features, features_clone)
    graph_clone = state.graph_copy()
    graph_clone_indices = graph_clone.neighbor_indices
    graph_clone_indices[0, 0] = 999
    assert not torch.equal(state.graph.neighbor_indices, graph_clone_indices)


def test_no_aliasing_between_cached_fields():
    plan = _plan()
    state = _make_state(plan.windows[0])
    assert state.s0.data_ptr() != state.dino_features.data_ptr()
    assert state.s0.data_ptr() != state.propagated_scores.data_ptr()
    assert state.dino_features.data_ptr() != state.propagated_scores.data_ptr()


def test_alias_construction_is_rejected():
    plan = _plan()
    window = plan.windows[0]
    shared = torch.zeros(4, 2)
    graph = wc.GraphSnapshot(
        torch.zeros(4, 2, dtype=torch.int64), torch.full((4, 2), 0.5),
        torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    with pytest.raises(wc.WindowCacheError, match="alias"):
        wc.CachedWindowState(
            geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
            s0=shared, dino_features=torch.zeros(4, 3), graph=graph,
            propagated_scores=shared,  # same tensor object as s0
            solver_summary=telemetry,
        )


def test_cpu_ownership_and_contiguity_and_dtype_preserved():
    plan = _plan()
    window = plan.windows[0]
    s0 = torch.randn(4, 2, dtype=torch.float64)
    features = torch.randn(4, 3, dtype=torch.float64)
    propagated = torch.randn(4, 2, dtype=torch.float64)
    graph = wc.GraphSnapshot(
        torch.zeros(4, 2, dtype=torch.int64), torch.full((4, 2), 0.5),
        torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    state = wc.CachedWindowState(
        geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
        s0=s0, dino_features=features, graph=graph, propagated_scores=propagated,
        solver_summary=telemetry,
    )
    for tensor in (state.s0, state.dino_features, state.propagated_scores,
                   state.graph.neighbor_indices, state.graph.transition_weights):
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
    assert state.s0.dtype == torch.float64  # dtype preserved exactly, no silent cast


def test_detached_from_autograd():
    plan = _plan()
    window = plan.windows[0]
    s0 = torch.randn(4, 2, requires_grad=True)
    y = (s0 * 2).sum()
    y.backward()
    features = torch.zeros(4, 3)
    propagated = torch.zeros(4, 2)
    graph = wc.GraphSnapshot(
        torch.zeros(4, 2, dtype=torch.int64), torch.full((4, 2), 0.5),
        torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    state = wc.CachedWindowState(
        geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
        s0=s0, dino_features=features, graph=graph, propagated_scores=propagated,
        solver_summary=telemetry,
    )
    assert not state.s0.requires_grad
    assert state.s0.grad_fn is None


def test_checksums_stable():
    plan = _plan()
    window = plan.windows[0]
    torch.manual_seed(3)
    s0 = torch.randn(4, 2)
    features = torch.randn(4, 3)
    propagated = torch.randn(4, 2)
    indices = torch.zeros(4, 2, dtype=torch.int64)
    weights = torch.full((4, 2), 0.5)
    graph = wc.GraphSnapshot(indices, weights, torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0)
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    checksums = wc.compute_window_checksums(s0, features, propagated, indices, weights)
    state = wc.CachedWindowState(
        geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
        s0=s0, dino_features=features, graph=graph, propagated_scores=propagated,
        solver_summary=telemetry, checksums=checksums,
    )
    assert state.verify_checksums() is True
    # recomputing independently from the (now-owned) cached tensors matches too
    recomputed = wc.compute_window_checksums(
        state.s0, state.dino_features, state.propagated_scores,
        state.graph.neighbor_indices, state.graph.transition_weights,
    )
    assert recomputed == dict(checksums)


def test_no_silent_fp16_downcast():
    plan = _plan()
    window = plan.windows[0]
    s0 = torch.randn(4, 2, dtype=torch.float32)
    features = torch.randn(4, 3, dtype=torch.float32)
    propagated = torch.randn(4, 2, dtype=torch.float32)
    graph = wc.GraphSnapshot(
        torch.zeros(4, 2, dtype=torch.int64), torch.full((4, 2), 0.5),
        torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    state = wc.CachedWindowState(
        geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
        s0=s0, dino_features=features, graph=graph, propagated_scores=propagated,
        solver_summary=telemetry,
    )
    assert state.s0.dtype == torch.float32
    assert state.dino_features.dtype == torch.float32


# ---------------------------------------------------------------------------
# Graph reconstruction
# ---------------------------------------------------------------------------


def test_graph_reconstruction_exact_indices_weights_and_matmul():
    torch.manual_seed(0)
    N, D, k = 40, 16, 5
    features = F.normalize(torch.randn(N, D), dim=-1)
    real_graph = cover_dr.build_directed_topk_graph(features, k=k, affinity_power=3.0)

    snapshot = wc.GraphSnapshot(
        real_graph.neighbor_indices, real_graph.transition_weights,
        real_graph.edge_affinities, real_graph.self_loop_fallback,
        real_graph.num_nodes, real_graph.k, real_graph.affinity_power,
    )
    reconstructed = wc.reconstruct_directed_topk_graph(snapshot)

    assert torch.equal(reconstructed.neighbor_indices, real_graph.neighbor_indices)
    assert torch.equal(reconstructed.transition_weights, real_graph.transition_weights)
    assert torch.equal(reconstructed.edge_affinities, real_graph.edge_affinities)
    assert torch.equal(reconstructed.self_loop_fallback, real_graph.self_loop_fallback)
    assert reconstructed.k == real_graph.k
    assert reconstructed.affinity_power == real_graph.affinity_power

    rhs = torch.randn(N, 3)
    assert torch.equal(reconstructed.matmul(rhs), real_graph.matmul(rhs))
    assert torch.equal(reconstructed.transpose_matmul(rhs), real_graph.transpose_matmul(rhs))
    row_sums = reconstructed.transition_weights.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-5)


def test_graph_reconstruction_fallback_metadata_preserved():
    # Force a self-loop fallback row by giving one node identical (zero
    # affinity after ReLU) similarity to everything else via an
    # all-negative row of the affinity matrix; simplest reliable way is to
    # construct a DirectedTopKGraph by hand with a fallback row.
    N, k = 5, 2
    indices = torch.tensor([[1, 2], [0, 2], [0, 1], [0, 1], [4, 4]], dtype=torch.int64)
    weights = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float32
    )
    affinities = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32
    )
    fallback = torch.tensor([False, False, False, False, True])
    real_graph = cover_dr.DirectedTopKGraph(indices, weights, affinities, fallback, N, k, 3.0)
    snapshot = wc.GraphSnapshot(
        real_graph.neighbor_indices, real_graph.transition_weights,
        real_graph.edge_affinities, real_graph.self_loop_fallback,
        real_graph.num_nodes, real_graph.k, real_graph.affinity_power,
    )
    reconstructed = wc.reconstruct_directed_topk_graph(snapshot)
    assert torch.equal(reconstructed.self_loop_fallback, fallback)
    assert reconstructed.self_loop_fallback[4].item() is True


def test_pass_two_never_calls_graph_builder():
    """Structural proof: run_pass_two's source never references the graph
    builder or solver, and reconstruct_directed_topk_graph never rebuilds
    from features."""
    import inspect

    pass_two_source = inspect.getsource(wc.run_pass_two)
    assert "build_directed_topk_graph" not in pass_two_source
    assert "solve_rwr_cgls" not in pass_two_source
    reconstruct_source = inspect.getsource(wc.reconstruct_directed_topk_graph)
    assert "build_directed_topk_graph" not in reconstruct_source


# ---------------------------------------------------------------------------
# Identity replay: call counts, parity, geometry integration
# ---------------------------------------------------------------------------


def test_pass_one_call_counts_exactly_one_per_window():
    inference = _make_inference()
    img = torch.rand(1, 3, 12, 12)
    call_log = {"graph": 0, "solve": 0}
    real_build = cover_dr.build_directed_topk_graph
    real_solve = cover_dr.solve_rwr_cgls

    import models.dinotext.cover_dr.graph as graph_mod
    import models.dinotext.cover_dr.rwr as rwr_mod

    def counting_build(*a, **kw):
        call_log["graph"] += 1
        return real_build(*a, **kw)

    def counting_solve(*a, **kw):
        call_log["solve"] += 1
        return real_solve(*a, **kw)

    graph_mod.build_directed_topk_graph = counting_build
    rwr_mod.solve_rwr_cgls = counting_solve
    try:
        first_pass_output, cache, context = wc.run_pass_one(inference, img)
    finally:
        graph_mod.build_directed_topk_graph = real_build
        rwr_mod.solve_rwr_cgls = real_solve

    expected_windows = context.expected_window_count
    assert inference.model.snapshot_call_count == expected_windows
    assert call_log["graph"] == expected_windows
    assert call_log["solve"] == expected_windows
    assert inference.model.downstream_call_count == expected_windows
    assert cache.cached_window_count == expected_windows
    cache.close()


def test_pass_two_zero_backbone_graph_solver_calls():
    inference = _make_inference()
    img = torch.rand(1, 3, 12, 12)
    first_pass_output, cache, context = wc.run_pass_one(inference, img)
    snapshot_calls_before = inference.model.snapshot_call_count
    downstream_before = inference.model.downstream_call_count

    call_log = {"graph": 0, "solve": 0}
    import models.dinotext.cover_dr.graph as graph_mod
    import models.dinotext.cover_dr.rwr as rwr_mod
    real_build = cover_dr.build_directed_topk_graph
    real_solve = cover_dr.solve_rwr_cgls

    def counting_build(*a, **kw):
        call_log["graph"] += 1
        return real_build(*a, **kw)

    def counting_solve(*a, **kw):
        call_log["solve"] += 1
        return real_solve(*a, **kw)

    graph_mod.build_directed_topk_graph = counting_build
    rwr_mod.solve_rwr_cgls = counting_solve
    try:
        second_pass_output = wc.run_pass_two(inference, cache, context)
    finally:
        graph_mod.build_directed_topk_graph = real_build
        rwr_mod.solve_rwr_cgls = real_solve
        cache.close()

    assert call_log["graph"] == 0
    assert call_log["solve"] == 0
    assert inference.model.snapshot_call_count == snapshot_calls_before  # no new backbone calls
    assert inference.model.downstream_call_count == downstream_before + context.expected_window_count


def test_identity_replay_equals_first_pass_output():
    inference = _make_inference()
    img = torch.rand(1, 3, 12, 12)
    result = wc.run_two_pass_slide_inference(inference, img)
    assert torch.equal(result.first_pass_output, result.second_pass_output)


def test_first_pass_stitch_equals_legacy_slide_inference():
    """Load the REAL DINOTextSegInference and compare its RWR-enabled
    slide_inference output against run_pass_one's stitched output for an
    identical synthetic model/crop plan."""
    module = _load_dinotext_seg_module()
    identity = load_identity(repo_root=ROOT)
    model = SyntheticModel(patch_size=2, embed_dim=6, class_count=3, seed=1)
    config = _config()

    inference = module.DINOTextSegInference(
        model, torch.randn(3, 4), ["a", "b", "c"], with_bg=False,
        test_cfg=dict(mode="slide", crop_size=(8, 8), stride=(4, 4)),
    )
    inference.rwr_config = config
    inference.rwr_runtime = cover_dr.RWRRuntimeSummary()

    img = torch.rand(1, 3, 12, 12)
    legacy_output = inference.slide_inference(
        img, [{"img_shape": (12, 12, 3), "ori_shape": (12, 12, 3)}], rescale=False
    )

    cache_inference = types.SimpleNamespace(
        model=model, text_embedding=inference.text_embedding, rwr_config=config,
        test_cfg=types.SimpleNamespace(stride=(4, 4), crop_size=(8, 8)),
        num_classes=3, with_bg=False,
    )
    stitched, cache, context = wc.run_pass_one(cache_inference, img)
    cache.close()

    assert torch.equal(stitched, legacy_output)


def test_geometry_integration_one_to_one_ordered_correspondence():
    inference = _make_inference()
    img = torch.rand(1, 3, 12, 12)
    _, cache, context = wc.run_pass_one(inference, img)
    plan = context.plan
    for window, cached in zip(plan.windows, cache.windows_in_order()):
        assert cached.window_index == window.index
        assert cached.geometry == window
    cache.close()


# ---------------------------------------------------------------------------
# Disabled parity
# ---------------------------------------------------------------------------


def test_run_pass_one_rejects_disabled_rwr():
    inference = _make_inference()
    inference.rwr_config = cover_dr.RWRInferenceConfig(enabled=False)
    img = torch.rand(1, 3, 12, 12)
    with pytest.raises(wc.WindowCacheError, match="enabled"):
        wc.run_pass_one(inference, img)


def test_disabled_canonical_execution_allocates_no_cache_module_state():
    """No module-level mutable container acts as a shared/global cache:
    every non-dunder, non-class, non-function, non-type attribute of the
    module must not be a dict/list/set instance (the only expected list is
    the static ``__all__`` export declaration, checked separately)."""
    import inspect

    for name, value in vars(wc).items():
        if name.startswith("__") or name == "__all__":
            continue
        if inspect.isclass(value) or inspect.isfunction(value) or isinstance(value, type):
            continue
        if inspect.ismodule(value):
            continue
        assert not isinstance(value, (dict, list, set)), (
            f"module-level mutable container found: {name} = {value!r}"
        )
    assert isinstance(wc.__all__, list)  # the only list is the static export declaration


# ---------------------------------------------------------------------------
# Image isolation
# ---------------------------------------------------------------------------


def test_multiple_images_do_not_share_cache_state():
    inference = _make_inference()
    img_a = torch.rand(1, 3, 12, 12)
    img_b = torch.rand(1, 3, 12, 12) + 5.0

    _, cache_a, context_a = wc.run_pass_one(inference, img_a)
    _, cache_b, context_b = wc.run_pass_one(inference, img_b)

    assert cache_a is not cache_b
    assert not torch.equal(context_a.stitched_scores, context_b.stitched_scores)
    a_first = cache_a.get(0)
    b_first = cache_b.get(0)
    assert not torch.equal(a_first.s0, b_first.s0)
    cache_a.close()
    cache_b.close()


def test_window_indices_restart_per_image():
    inference = _make_inference()
    for _ in range(3):
        img = torch.rand(1, 3, 12, 12)
        _, cache, context = wc.run_pass_one(inference, img)
        assert cache.get(0).window_index == 0
        cache.close()


def test_exception_in_one_image_does_not_poison_the_next():
    inference = _make_inference()
    bad_img = torch.full((1, 3, 12, 12), float("nan"))
    with pytest.raises(wc.WindowCacheError):
        wc.run_pass_one(inference, bad_img)

    good_img = torch.rand(1, 3, 12, 12)
    stitched, cache, context = wc.run_pass_one(inference, good_img)
    assert context.cached_window_count == context.expected_window_count
    cache.close()


def test_memory_does_not_grow_monotonically_across_images():
    inference = _make_inference()
    byte_totals = []
    for _ in range(5):
        img = torch.rand(1, 3, 12, 12)
        _, cache, context = wc.run_pass_one(inference, img)
        byte_totals.append(context.cache_total_bytes)
        cache.close()  # released before next image
    # every image reports the same (bounded) footprint, not a growing one
    assert len(set(byte_totals)) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_wrong_shape_rejected():
    plan = _plan()
    window = plan.windows[0]
    graph = wc.GraphSnapshot(
        torch.zeros(4, 2, dtype=torch.int64), torch.full((4, 2), 0.5),
        torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    with pytest.raises(wc.WindowCacheError):
        wc.CachedWindowState(
            geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
            s0=torch.zeros(3, 2), dino_features=torch.zeros(4, 3), graph=graph,  # N mismatch: 3 vs 4
            propagated_scores=torch.zeros(4, 2), solver_summary=telemetry,
        )


def test_wrong_type_rejected():
    plan = _plan()
    window = plan.windows[0]
    graph = wc.GraphSnapshot(
        torch.zeros(4, 2, dtype=torch.int64), torch.full((4, 2), 0.5),
        torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    with pytest.raises(wc.WindowCacheError):
        wc.CachedWindowState(
            geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
            s0=[[1, 2], [3, 4]], dino_features=torch.zeros(4, 3), graph=graph,  # not a tensor
            propagated_scores=torch.zeros(4, 2), solver_summary=telemetry,
        )


def test_class_count_mismatch_rejected():
    plan = _plan()
    window = plan.windows[0]
    graph = wc.GraphSnapshot(
        torch.zeros(4, 2, dtype=torch.int64), torch.full((4, 2), 0.5),
        torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    with pytest.raises(wc.WindowCacheError, match="class_count"):
        wc.CachedWindowState(
            geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=5,  # wrong
            s0=torch.zeros(4, 2), dino_features=torch.zeros(4, 3), graph=graph,
            propagated_scores=torch.zeros(4, 2), solver_summary=telemetry,
        )


def test_graph_dimension_mismatch_rejected():
    plan = _plan()
    window = plan.windows[0]
    graph = wc.GraphSnapshot(  # graph built for N=6, but s0/features are N=4
        torch.zeros(6, 2, dtype=torch.int64), torch.full((6, 2), 0.5),
        torch.ones(6, 2), torch.zeros(6, dtype=torch.bool), 6, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    with pytest.raises(wc.WindowCacheError, match="graph node count"):
        wc.CachedWindowState(
            geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
            s0=torch.zeros(4, 2), dino_features=torch.zeros(4, 3), graph=graph,
            propagated_scores=torch.zeros(4, 2), solver_summary=telemetry,
        )


def test_nonfinite_tensors_rejected():
    plan = _plan()
    window = plan.windows[0]
    graph = wc.GraphSnapshot(
        torch.zeros(4, 2, dtype=torch.int64), torch.full((4, 2), 0.5),
        torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
    )
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    bad = torch.zeros(4, 2)
    bad[0, 0] = float("nan")
    with pytest.raises(wc.WindowCacheError, match="finite"):
        wc.CachedWindowState(
            geometry=window, window_index=0, patch_grid_shape=(2, 2), class_count=2,
            s0=bad, dino_features=torch.zeros(4, 3), graph=graph,
            propagated_scores=torch.zeros(4, 2), solver_summary=telemetry,
        )


def test_nonfinite_graph_weights_rejected():
    bad_weights = torch.full((4, 2), 0.5)
    bad_weights[0, 0] = float("inf")
    with pytest.raises(wc.WindowCacheError, match="finite"):
        wc.GraphSnapshot(
            torch.zeros(4, 2, dtype=torch.int64), bad_weights,
            torch.ones(4, 2), torch.zeros(4, dtype=torch.bool), 4, 2, 3.0,
        )


def test_unsupported_batch_size_rejected():
    inference = _make_inference()
    img = torch.rand(2, 3, 12, 12)  # batch size 2, unsupported
    with pytest.raises(wc.WindowCacheError, match="batch size"):
        wc.run_pass_one(inference, img)


def test_with_bg_rejected():
    inference = _make_inference()
    inference.with_bg = True
    inference.num_classes = 4
    img = torch.rand(1, 3, 12, 12)
    with pytest.raises(wc.WindowCacheError, match="with_bg"):
        wc.run_pass_one(inference, img)


def test_alpha_zero_rejected():
    inference = _make_inference(alpha=0.0)
    img = torch.rand(1, 3, 12, 12)
    with pytest.raises(wc.WindowCacheError, match="alpha"):
        wc.run_pass_one(inference, img)


def test_second_pass_processor_wrong_shape_rejected():
    inference = _make_inference()
    img = torch.rand(1, 3, 12, 12)
    _, cache, context = wc.run_pass_one(inference, img)

    def bad_processor(window_index, geometry, propagated_scores):
        del window_index, geometry
        return propagated_scores[:, :1]  # wrong shape

    with pytest.raises(wc.WindowCacheError, match="shape"):
        wc.run_pass_two(inference, cache, context, processor=bad_processor)
    cache.close()


def test_second_pass_processor_nonfinite_output_rejected():
    inference = _make_inference()
    img = torch.rand(1, 3, 12, 12)
    _, cache, context = wc.run_pass_one(inference, img)

    def nan_processor(window_index, geometry, propagated_scores):
        del window_index, geometry
        return propagated_scores * float("nan")

    with pytest.raises(wc.WindowCacheError, match="finite"):
        wc.run_pass_two(inference, cache, context, processor=nan_processor)
    cache.close()


def _load_dinotext_seg_module():
    mmcv_stub = types.ModuleType("mmcv")

    class Config(dict):
        __getattr__ = dict.__getitem__

    mmcv_stub.Config = Config
    utils_stub = types.ModuleType("utils")
    utils_stub.get_logger = lambda: types.SimpleNamespace(info=lambda *_a: None)

    names = ("mmcv", "utils")
    previous = {n: sys.modules.get(n) for n in names}
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
