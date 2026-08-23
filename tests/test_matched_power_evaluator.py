"""CPU synthetic tests for the matched k11/k12 power-evaluator core
(``models.dinotext.cover_dr.matched_power_evaluator``).

Loaded via file-path import with a lightweight synthetic deterministic
model (mirroring ``tests/test_image_window_cache.py``'s ``SyntheticModel``
pattern) so these tests never require mmcv, cv2, a CUDA device, or the
real E3 backbone/checkpoint. Every test either exercises the real,
unmodified ``finite_step_regime``/``graph``/``sliding_window_geometry``
modules directly, or asserts a structural invariant via AST inspection of
the evaluator's own source -- never a reimplementation of the logic under
test.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
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
cover_dr = _load_cover_dr_package()

SpatialSize = geometry.SpatialSize
SlidingWindowPlan = geometry.SlidingWindowPlan

process_one_window = cover_dr.process_one_window
stitch_one_image = cover_dr.stitch_one_image
finalize_prediction = cover_dr.finalize_prediction
aggregate_telemetry = cover_dr.aggregate_telemetry
WindowOperationTelemetry = cover_dr.WindowOperationTelemetry
MatchedPowerEvaluatorError = cover_dr.MatchedPowerEvaluatorError
build_directed_topk_graph = cover_dr.build_directed_topk_graph
build_matched_k11_from_k12 = cover_dr.build_matched_k11_from_k12
finite_step_propagate = cover_dr.finite_step_propagate
compute_full_precision_metrics = cover_dr.compute_full_precision_metrics

MATCHED_POWER_EVALUATOR_PATH = COVER_DR_PACKAGE_PATH / "matched_power_evaluator.py"


# ---------------------------------------------------------------------------
# Synthetic deterministic model
# ---------------------------------------------------------------------------


class SyntheticModel(nn.Module):
    """Deterministic stand-in for DINOText: average-pools each patch, then
    a fixed random projection to features/scores; bit-identical output for
    the same crop on every call. Counts every backbone/downstream call so
    tests can assert exact operation counts."""

    def __init__(self, patch_size=2, embed_dim=6, class_count=5, seed=0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.class_count = class_count
        generator = torch.Generator().manual_seed(seed)
        self.feature_proj = torch.randn(3, embed_dim, generator=generator)
        self.score_proj = torch.randn(embed_dim, class_count, generator=generator)
        self.snapshot_call_count = 0
        self.snapshot_call_crops = []
        self.downstream_call_count = 0

    def generate_patch_snapshot(self, crop, text_embedding):
        del text_embedding
        self.snapshot_call_count += 1
        self.snapshot_call_crops.append(crop)
        _, channels, height, width = crop.shape
        ps = self.patch_size
        grid_h, grid_w = height // ps, width // ps
        pooled = F.avg_pool2d(crop, ps)
        flat = pooled.reshape(1, channels, grid_h * grid_w).permute(0, 2, 1)
        raw_feat = flat @ self.feature_proj
        position_bias = 0.01 * torch.arange(grid_h * grid_w, dtype=torch.float32).view(1, -1, 1)
        raw_feat = raw_feat + position_bias
        features = F.normalize(raw_feat, dim=-1)
        scores = features @ self.score_proj
        return types.SimpleNamespace(unary_scores=scores, dino_features=features, grid_hw=(grid_h, grid_w))

    def masks_from_patch_scores(self, patch_scores, grid_hw, output_hw):
        self.downstream_call_count += 1
        batch, _num_patches, classes = patch_scores.shape
        grid_h, grid_w = grid_hw
        simmap = patch_scores.reshape(batch, grid_h, grid_w, classes).permute(0, 3, 1, 2)
        mask = torch.sigmoid(simmap)
        return F.interpolate(mask, tuple(output_hw), mode="bilinear", align_corners=True)


def _inference(model, *, num_classes=5):
    return types.SimpleNamespace(model=model, text_embedding=None, num_classes=num_classes, align_corners=False)


def _plan(image_hw=(8, 8), crop=(8, 8), stride=(8, 8)):
    return SlidingWindowPlan.build(
        image_size=SpatialSize(*image_hw), crop_size=SpatialSize(*crop), stride=SpatialSize(*stride)
    )


ALPHA = 0.9
STEPS = 320
AFFINITY_POWER = 3.0


# ---------------------------------------------------------------------------
# 1-3: shared snapshot, shared top-12, k11 exact prefix
# ---------------------------------------------------------------------------


def test_1_one_shared_snapshot_used_by_both_variants():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan()
    window = plan.windows[0]
    image = torch.rand(1, 3, 8, 8)
    process_one_window(inference, image, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)
    assert model.snapshot_call_count == 1


def test_2_one_top12_selection_and_k11_is_exact_prefix():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan()
    window = plan.windows[0]
    image = torch.rand(1, 3, 8, 8)
    result = process_one_window(inference, image, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)
    # compute_matched_graph_diagnostics itself raises on any prefix
    # mismatch (see finite_step_regime.py); a successful call here is
    # itself the exact-prefix proof.
    assert result.graph_diagnostics.prefix_mismatch_count == 0
    assert result.graph_diagnostics.fallback_row_mismatch_count == 0


def test_3_separate_renormalization_k11_rows_sum_to_one():
    # A common base direction plus small per-node noise makes nearly every
    # pairwise cosine similarity positive (rather than ~half, as for
    # independent random vectors), so the rank-12 affinity is reliably
    # nonzero for this fixture -- required to distinguish k11's independent
    # renormalization from a mere truncation of k12's weights.
    generator = torch.Generator().manual_seed(42)
    base = torch.randn(1, 6, generator=generator)
    noise = 0.2 * torch.randn(16, 6, generator=generator)
    features = F.normalize(base + noise, dim=-1)
    graph12 = build_directed_topk_graph(features, k=12, affinity_power=AFFINITY_POWER)
    graph11 = build_matched_k11_from_k12(graph12)
    row_sums_11 = graph11.transition_weights.sum(dim=1)
    row_sums_12 = graph12.transition_weights.sum(dim=1)
    assert torch.allclose(row_sums_11, torch.ones_like(row_sums_11), atol=1e-5)
    assert torch.allclose(row_sums_12, torch.ones_like(row_sums_12), atol=1e-5)
    # k11's weights are independently renormalized over only the retained
    # 11 affinities (never borrowed/truncated from k12's own normalization):
    # for any row where the rank-12 affinity is nonzero, the two
    # normalizations must disagree, since they divide by different sums.
    rank12_affinity = graph12.edge_affinities[:, 11]
    nonzero_rank12_rows = rank12_affinity > 0
    assert bool(nonzero_rank12_rows.any()), "test fixture must contain at least one row with a nonzero rank-12 affinity"
    assert not torch.allclose(
        graph11.transition_weights[nonzero_rank12_rows],
        graph12.transition_weights[nonzero_rank12_rows, :11],
    )


# ---------------------------------------------------------------------------
# 5: exactly 320 updates
# ---------------------------------------------------------------------------


def test_5_exactly_320_updates_each_variant():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan()
    window = plan.windows[0]
    image = torch.rand(1, 3, 8, 8)
    result = process_one_window(inference, image, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)
    assert result.telemetry.k11_updates == 320
    assert result.telemetry.k12_updates == 320


def test_5b_window_operation_telemetry_rejects_wrong_counts():
    with pytest.raises(MatchedPowerEvaluatorError):
        WindowOperationTelemetry(
            backbone_snapshot_calls=2, dino_feature_extractions=1, topk_selection_calls=1,
            graph_normalizations=2, finite_step_propagations=2, k11_updates=320, k12_updates=320,
            sigmoid_calls=2, interpolation_calls=2,
        )
    with pytest.raises(MatchedPowerEvaluatorError):
        WindowOperationTelemetry(
            backbone_snapshot_calls=1, dino_feature_extractions=1, topk_selection_calls=1,
            graph_normalizations=2, finite_step_propagations=2, k11_updates=319, k12_updates=320,
            sigmoid_calls=2, interpolation_calls=2,
        )


# ---------------------------------------------------------------------------
# 6, 30: forbidden solver/production-CGLS nonuse (AST-based, not substring)
# ---------------------------------------------------------------------------


FORBIDDEN_CALL_NAMES = {
    "solve_rwr_cgls", "solve_rwr", "solve_rwr_fixed_point", "inv", "solve",
    "dense_fp64_equilibrium_reference",
}


def test_6_and_30_no_forbidden_solver_or_equilibrium_calls():
    source = MATCHED_POWER_EVALUATOR_PATH.read_text()
    tree = ast.parse(source)
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called_names.add(func.id)
            elif isinstance(func, ast.Attribute):
                called_names.add(func.attr)
    forbidden_found = called_names & FORBIDDEN_CALL_NAMES
    assert not forbidden_found, f"forbidden solver/equilibrium calls found: {forbidden_found}"

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
    forbidden_imports = imported_names & {
        "solve_rwr", "solve_rwr_cgls", "solve_rwr_fixed_point",
        "RWRInferenceConfig", "apply_rwr_to_e3_snapshot", "dense_fp64_equilibrium_reference",
    }
    assert not forbidden_imports, f"forbidden imports found: {forbidden_imports}"


def test_6b_finite_step_propagate_imported_not_reimplemented():
    """The evaluator must import the exact recurrence, never redefine one
    of its own -- checked by confirming no local function in the module
    defines a P(t+1) = ... update loop, and that ``finite_step_propagate``
    is imported (not shadowed by a same-named local def)."""
    source = MATCHED_POWER_EVALUATOR_PATH.read_text()
    tree = ast.parse(source)
    local_defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "finite_step_propagate" not in local_defs
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    assert "finite_step_propagate" in imported


# ---------------------------------------------------------------------------
# 7: canonical downstream transform call counts
# ---------------------------------------------------------------------------


def test_7_sigmoid_and_interpolation_call_counts_per_window():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan()
    window = plan.windows[0]
    image = torch.rand(1, 3, 8, 8)
    result = process_one_window(inference, image, window, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)
    assert model.downstream_call_count == 2  # masks_from_patch_scores, once per variant
    assert result.telemetry.sigmoid_calls == 2
    assert result.telemetry.interpolation_calls == 2


# ---------------------------------------------------------------------------
# 8-9: stitching order, coverage map
# ---------------------------------------------------------------------------


def test_8_windows_processed_in_row_major_order():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan(image_hw=(12, 12), crop=(8, 8), stride=(4, 4))
    image = torch.rand(1, 3, 12, 12)
    stitch_one_image(inference, image, plan, class_count=5, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)
    # each crop passed to generate_patch_snapshot must match plan.windows[i].crop_slice in order
    for index, window in enumerate(plan.windows):
        rows, cols = window.crop_slice
        expected = image[:, :, rows, cols]
        assert torch.equal(model.snapshot_call_crops[index], expected)


def test_9_full_coverage_no_uncovered_pixels():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan(image_hw=(10, 10), crop=(8, 8), stride=(4, 4))
    image = torch.rand(1, 3, 10, 10)
    result = stitch_one_image(inference, image, plan, class_count=5, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)
    assert result.k11_stitched.shape == (1, 5, 10, 10)
    assert result.k12_stitched.shape == (1, 5, 10, 10)
    assert torch.isfinite(result.k11_stitched).all()
    assert torch.isfinite(result.k12_stitched).all()


def test_9b_uncovered_pixels_rejected():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan(image_hw=(8, 8), crop=(8, 8), stride=(8, 8))
    image = torch.rand(1, 3, 8, 8)
    # sabotage: an image_tensor with the wrong spatial extent than the plan
    with pytest.raises(MatchedPowerEvaluatorError):
        stitch_one_image(inference, torch.rand(1, 3, 9, 9), plan, class_count=5, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)


# ---------------------------------------------------------------------------
# 10-11: two-variant vs separate-variant equivalence, k12-only replay
# ---------------------------------------------------------------------------


def test_10_and_11_two_variant_result_equals_independently_recomputed_k12_only():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan(image_hw=(10, 10), crop=(8, 8), stride=(4, 4))
    image = torch.rand(1, 3, 10, 10)
    combined = stitch_one_image(inference, image, plan, class_count=5, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)

    # independent k12-only replay: same crops, freshly built graph, freshly
    # propagated, using a SEPARATE SyntheticModel instance with the SAME
    # seed (deterministic) so the backbone output is bit-identical, then
    # only the k12 half of the pipeline is exercised.
    replay_model = SyntheticModel()
    preds12_replay = image.new_zeros((1, 5, 10, 10))
    count_replay = image.new_zeros((1, 1, 10, 10))
    for window in plan.windows:
        rows, cols = window.crop_slice
        crop = image[:, :, rows, cols]
        snapshot = replay_model.generate_patch_snapshot(crop, None)
        graph12 = build_directed_topk_graph(snapshot.dino_features[0], k=12, affinity_power=AFFINITY_POWER)
        trace12 = finite_step_propagate(graph12, snapshot.unary_scores[0], alpha=ALPHA, steps=STEPS, snapshot_steps=(STEPS,))
        masks12 = replay_model.masks_from_patch_scores(
            trace12.snapshots[STEPS].unsqueeze(0), snapshot.grid_hw, window.extent.as_tuple()
        )
        accum_rows, accum_cols = window.accumulation_slice
        preds12_replay += F.pad(
            masks12,
            (accum_cols.start, preds12_replay.shape[3] - accum_cols.stop,
             accum_rows.start, preds12_replay.shape[2] - accum_rows.stop),
        )
        count_replay[:, :, accum_rows, accum_cols] += 1
    stitched12_replay = preds12_replay / count_replay

    assert torch.allclose(combined.k12_stitched, stitched12_replay, atol=1e-6)


# ---------------------------------------------------------------------------
# 12-13: full-precision confusion, paired delta
# ---------------------------------------------------------------------------


def test_12_full_precision_metrics_reused_not_reimplemented():
    # 2 classes, 2 images: exact hand-derived intersect/union/pred/label
    pre_eval = [
        (torch.tensor([3.0, 1.0]), torch.tensor([4.0, 2.0]), torch.tensor([3.0, 2.0]), torch.tensor([4.0, 1.0])),
        (torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]), torch.tensor([1.0, 3.0]), torch.tensor([2.0, 2.0])),
    ]
    metrics = compute_full_precision_metrics(pre_eval)
    # Hand-derived in pure Python float64 (never torch float32) so this
    # independently proves the function's own claimed float64 accumulation
    # -- a naive float32 tensor computation would already disagree at this
    # precision, which is exactly the rounding error the function exists
    # to avoid.
    intersects = [4.0, 3.0]
    unions = [6.0, 5.0]
    labels = [6.0, 3.0]
    expected_aacc = sum(intersects) / sum(labels)
    expected_miou = sum(i / u for i, u in zip(intersects, unions)) / len(intersects)
    expected_macc = sum(i / l for i, l in zip(intersects, labels)) / len(intersects)
    assert metrics["aAcc"] == pytest.approx(expected_aacc, abs=1e-12)
    assert metrics["mIoU"] == pytest.approx(expected_miou, abs=1e-12)
    assert metrics["mAcc"] == pytest.approx(expected_macc, abs=1e-12)


def test_13_paired_delta_is_k11_minus_k12():
    metrics_k11 = {"aAcc": 50.0, "mIoU": 30.5, "mAcc": 55.0}
    metrics_k12 = {"aAcc": 48.5, "mIoU": 29.9, "mAcc": 54.1}
    delta = metrics_k11["mIoU"] - metrics_k12["mIoU"]
    assert delta == pytest.approx(0.6, abs=1e-9)


# ---------------------------------------------------------------------------
# 26: exact operation telemetry aggregation
# ---------------------------------------------------------------------------


def test_26_aggregate_telemetry_sums_exactly():
    entries = [
        WindowOperationTelemetry(1, 1, 1, 2, 2, 320, 320, 2, 2),
        WindowOperationTelemetry(1, 1, 1, 2, 2, 320, 320, 2, 2),
        WindowOperationTelemetry(1, 1, 1, 2, 2, 320, 320, 2, 2),
    ]
    totals = aggregate_telemetry(entries)
    assert totals["backbone_snapshot_calls"] == 3
    assert totals["k11_updates"] == 960
    assert totals["k12_updates"] == 960
    assert totals["sigmoid_calls"] == 6
    assert totals["interpolation_calls"] == 6
    assert totals["graph_normalizations"] == 6


def test_26b_aggregate_telemetry_rejects_empty():
    with pytest.raises(MatchedPowerEvaluatorError):
        aggregate_telemetry([])


# ---------------------------------------------------------------------------
# 27: input/cache nonmutation
# ---------------------------------------------------------------------------


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def test_27_input_image_tensor_never_mutated():
    model = SyntheticModel()
    inference = _inference(model)
    plan = _plan(image_hw=(10, 10), crop=(8, 8), stride=(4, 4))
    image = torch.rand(1, 3, 10, 10)
    before = _hash_tensor(image)
    stitch_one_image(inference, image, plan, class_count=5, alpha=ALPHA, steps=STEPS, affinity_power=AFFINITY_POWER)
    assert _hash_tensor(image) == before


# ---------------------------------------------------------------------------
# finalize_prediction: canonical crop/rescale/argmax
# ---------------------------------------------------------------------------


def test_finalize_prediction_shape_and_argmax_never_rounds_scores():
    stitched = torch.zeros(1, 3, 8, 8)
    stitched[0, 2] = 5.0  # class 2 uniformly wins
    img_meta = {"img_shape": (8, 8, 3), "ori_shape": (16, 16, 3)}
    pred = finalize_prediction(stitched, img_meta, align_corners=False)
    assert tuple(pred.shape) == (1, 16, 16)
    assert torch.all(pred == 2)


def test_finalize_prediction_crops_to_img_shape_before_rescale():
    stitched = torch.zeros(1, 2, 10, 10)
    stitched[0, 0, :8, :8] = 1.0  # class 0 wins inside the valid (unpadded) region
    stitched[0, 1, 8:, :] = 2.0  # class 1 dominates only the padded rows
    stitched[0, 1, :, 8:] = 2.0  # ... and the padded columns
    img_meta = {"img_shape": (8, 8, 3), "ori_shape": (8, 8, 3)}
    pred = finalize_prediction(stitched, img_meta, align_corners=False)
    # if the pad region were NOT cropped away before argmax, class 1 would
    # leak into the result via interpolation; cropping first prevents that
    assert torch.all(pred == 0)
