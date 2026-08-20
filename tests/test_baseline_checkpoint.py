"""Tests for the baseline-suite checkpoint/resume system, including a
deterministic multi-image uninterrupted-vs-resumed identity proof.

If this shared node's cv2/opencv module is currently unavailable (a
transient environment issue, not a code issue), a dependency-free
``intersect_and_union`` equivalent is installed under ``mmseg.core.
evaluation.metrics`` so ``StreamingSegmentationMetricAccumulator.absorb``
still exercises its real code path; when mmcv/cv2 import cleanly, the real
mmseg implementation is used instead.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation"
COVER_DR_PACKAGE_PATH = ROOT / "src/open_vocabulary_segmentation/models/dinotext/cover_dr"


def _install_segmentation_package_stub() -> None:
    if "segmentation" not in sys.modules:
        m = types.ModuleType("segmentation")
        m.__path__ = [str(EVAL_DIR.parent)]
        sys.modules["segmentation"] = m
    if "segmentation.evaluation" not in sys.modules:
        m = types.ModuleType("segmentation.evaluation")
        m.__path__ = [str(EVAL_DIR)]
        sys.modules["segmentation.evaluation"] = m


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_cover_dr_package():
    if "models" not in sys.modules:
        m = types.ModuleType("models")
        m.__path__ = []
        sys.modules["models"] = m
    if "models.dinotext" not in sys.modules:
        m = types.ModuleType("models.dinotext")
        m.__path__ = []
        sys.modules["models.dinotext"] = m
    spec = importlib.util.spec_from_file_location(
        "models.dinotext.cover_dr",
        COVER_DR_PACKAGE_PATH / "__init__.py",
        submodule_search_locations=[str(COVER_DR_PACKAGE_PATH)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _real_mmseg_intersect_and_union_importable() -> bool:
    for name in ("mmseg", "mmseg.core", "mmseg.core.evaluation", "mmseg.core.evaluation.metrics"):
        sys.modules.pop(name, None)
    try:
        import importlib

        importlib.import_module("mmseg.core.evaluation.metrics")
        return True
    except Exception:
        for name in ("mmseg", "mmseg.core", "mmseg.core.evaluation", "mmseg.core.evaluation.metrics"):
            sys.modules.pop(name, None)
        return False


def _install_intersect_and_union_stub() -> None:
    """Installs a dependency-free ``intersect_and_union`` equivalent under
    ``mmseg.core.evaluation.metrics`` ONLY for the duration of this test
    module's own tests (see the ``_mmseg_stub_scope`` fixture below), since
    this shared login node's cv2/opencv module load is currently
    intermittently broken. Never left installed afterward: a stub module
    lingering in ``sys.modules`` after this file's tests finish would break
    other test files' own ``@requires_mmseg``-style availability checks
    (which expect a genuine ImportError, not a module missing ``eval_metrics``
    and other real mmseg exports this stub does not provide)."""

    def intersect_and_union(pred, gt, num_classes, ignore_index, label_map=None, reduce_zero_label=False):
        pred_t = torch.as_tensor(pred)
        gt_t = torch.as_tensor(gt)
        mask = gt_t != ignore_index
        pred_m = pred_t[mask]
        gt_m = gt_t[mask]
        intersect = pred_m[pred_m == gt_m]
        area_intersect = torch.histc(intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
        area_pred = torch.histc(pred_m.float(), bins=num_classes, min=0, max=num_classes - 1)
        area_label = torch.histc(gt_m.float(), bins=num_classes, min=0, max=num_classes - 1)
        area_union = area_pred + area_label - area_intersect
        return area_intersect, area_union, area_pred, area_label

    mmseg_stub = types.ModuleType("mmseg")
    mmseg_core_stub = types.ModuleType("mmseg.core")
    mmseg_eval_stub = types.ModuleType("mmseg.core.evaluation")
    mmseg_metrics_stub = types.ModuleType("mmseg.core.evaluation.metrics")
    mmseg_metrics_stub.intersect_and_union = intersect_and_union
    sys.modules["mmseg"] = mmseg_stub
    sys.modules["mmseg.core"] = mmseg_core_stub
    sys.modules["mmseg.core.evaluation"] = mmseg_eval_stub
    sys.modules["mmseg.core.evaluation.metrics"] = mmseg_metrics_stub


@pytest.fixture(scope="module", autouse=True)
def _mmseg_stub_scope():
    """Installs the stub (only if real mmseg is genuinely unimportable
    right now) for this module's tests, then always removes whatever
    mmseg-related sys.modules entries existed at teardown so later test
    files see a clean slate and can attempt their own real import."""
    saved = {
        name: sys.modules.get(name)
        for name in ("mmseg", "mmseg.core", "mmseg.core.evaluation", "mmseg.core.evaluation.metrics")
    }
    if not _real_mmseg_intersect_and_union_importable():
        _install_intersect_and_union_stub()
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


_install_segmentation_package_stub()
geometry = _load("segmentation.evaluation.sliding_window_geometry", EVAL_DIR / "sliding_window_geometry.py")
wc = _load("segmentation.evaluation.window_cache", EVAL_DIR / "window_cache.py")
t4 = _load("segmentation.evaluation.t4_audit", EVAL_DIR / "t4_audit.py")
stitching = _load("segmentation.evaluation.stitching_baselines", EVAL_DIR / "stitching_baselines.py")
ksweep = _load("segmentation.evaluation.graph_degree_sweep", EVAL_DIR / "graph_degree_sweep.py")
consensus = _load("segmentation.evaluation.consensus_replacement", EVAL_DIR / "consensus_replacement.py")
trust_harness = _load("segmentation.evaluation.trust_centrality_harness", EVAL_DIR / "trust_centrality_harness.py")
bc = _load("segmentation.evaluation.baseline_checkpoint", EVAL_DIR / "baseline_checkpoint.py")
cover_dr = _load_cover_dr_package()

SpatialSize = geometry.SpatialSize
SlidingWindowPlan = geometry.SlidingWindowPlan

K_VALUES = (4, 6, 8, 10, 11, 12, 16, 32)
CLASS_COUNT = 3
IGNORE_INDEX = 255


def _make_context() -> bc.BaselineCheckpointCompatibilityContext:
    return bc.BaselineCheckpointCompatibilityContext(
        git_head="a" * 40, git_branch="e11-cover-dr-1",
        baseline_identity_name="stitching-kdcr-baseline-suite",
        baseline_manifest_sha256="b" * 64, e3_identity_sha256="c" * 64, rwr_identity_sha256="d" * 64,
        canonical_config_sha256="e" * 64, checkpoint_sha256="f" * 64,
        tracked_diff_sha256="0" * 64, untracked_source_sha256="1" * 64,
        crop=(448, 448), stride=(224, 224), k_values=K_VALUES,
        stitching_modes=tuple(stitching.STITCH_MODES), dcr_variants=("dcr_hard", "dcr_jury_mean"),
        sur_definition="sigmoid(S0_w(i))", strict_safe_rule="E(after) proper-subset E(before)",
        class_count=CLASS_COUNT, ignore_label=IGNORE_INDEX,
        metric_unit_contract="percent_0_to_100", dataset_length=2,
    )


# ---------------------------------------------------------------------------
# 9. Checkpoint serialization tests
# ---------------------------------------------------------------------------


def _empty_accumulator_state():
    metric = {name: trust_harness.StreamingSegmentationMetricAccumulator(num_classes=CLASS_COUNT, ignore_index=IGNORE_INDEX)
              for name in bc.declared_baseline_names(K_VALUES)}
    ksweep_acc = {str(k): bc.KSweepTelemetryAccumulator() for k in K_VALUES}
    stitch_acc = {mode: bc.StitchingDiagnosticsAccumulator() for mode in stitching.STITCH_MODES}
    replace_acc = {name: bc.ReplacementDiagnosticsAccumulator() for name in bc.REPLACEMENT_VARIANT_NAMES}
    return metric, ksweep_acc, stitch_acc, replace_acc


def test_checkpoint_round_trips_through_state_dict():
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    checkpoint = bc.BaselineCheckpoint(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
        next_index=0, processed_image_ids=(), processed_count=0,
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state={k: v.state_dict() for k, v in ksweep_acc.items()},
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
        call_accounting_state=bc.CallAccounting().state_dict(),
        context=context,
    )
    payload = bc._checkpoint_to_json(checkpoint)
    import json
    reparsed = json.loads(json.dumps(payload))
    assert reparsed == payload


def test_checkpoint_save_and_load_atomic(tmp_path):
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    checkpoint = bc.BaselineCheckpoint(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
        next_index=1, processed_image_ids=("img0",), processed_count=1,
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state={k: v.state_dict() for k, v in ksweep_acc.items()},
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
        call_accounting_state=bc.CallAccounting(graph_builds=1, solver_calls=1).state_dict(),
        context=context,
    )
    path = tmp_path / "checkpoint.json"
    bc.save_checkpoint_atomic(checkpoint, path)
    loaded = bc.load_checkpoint(path)
    assert loaded.processed_image_ids == ("img0",)
    assert loaded.context.crop == (448, 448)
    bc.validate_checkpoint_internal_consistency(loaded)


# ---------------------------------------------------------------------------
# 10. Checkpoint incompatibility tests
# ---------------------------------------------------------------------------


def test_incompatibility_rejects_old_schema_version():
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    with pytest.raises(bc.BaselineCheckpointError):
        bc.BaselineCheckpoint(
            schema_version="talk2dino-baseline-suite-checkpoint-v0", run_status=bc.RUN_STATUS_PARTIAL,
            next_index=0, processed_image_ids=(), processed_count=0,
            metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
            ksweep_telemetry_state={k: v.state_dict() for k, v in ksweep_acc.items()},
            stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
            replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
            call_accounting_state=bc.CallAccounting().state_dict(), context=context,
        )


def _valid_checkpoint(context=None, **overrides):
    context = context or _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    kwargs = dict(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
        next_index=0, processed_image_ids=(), processed_count=0,
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state={k: v.state_dict() for k, v in ksweep_acc.items()},
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
        call_accounting_state=bc.CallAccounting().state_dict(), context=context,
    )
    kwargs.update(overrides)
    return bc.BaselineCheckpoint(**kwargs)


def test_incompatibility_rejects_changed_identity_hash():
    saved = _valid_checkpoint()
    expected = _make_context()
    expected = bc.BaselineCheckpointCompatibilityContext(
        **{**expected.__dict__, "baseline_manifest_sha256": "9" * 64}
    )
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_compatibility(saved, expected)


def test_incompatibility_rejects_changed_k_set():
    saved = _valid_checkpoint()
    expected = _make_context()
    expected = bc.BaselineCheckpointCompatibilityContext(**{**expected.__dict__, "k_values": (4, 6, 8)})
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_compatibility(saved, expected)


def test_incompatibility_rejects_changed_stitching_mode():
    saved = _valid_checkpoint()
    expected = _make_context()
    expected = bc.BaselineCheckpointCompatibilityContext(
        **{**expected.__dict__, "stitching_modes": ("uniform_probability_average",)}
    )
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_compatibility(saved, expected)


def test_incompatibility_rejects_changed_dcr_sur_definition():
    saved = _valid_checkpoint()
    expected = _make_context()
    expected = bc.BaselineCheckpointCompatibilityContext(**{**expected.__dict__, "sur_definition": "different"})
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_compatibility(saved, expected)


def test_incompatibility_rejects_wrong_dataset_length():
    saved = _valid_checkpoint()
    expected = _make_context()
    expected = bc.BaselineCheckpointCompatibilityContext(**{**expected.__dict__, "dataset_length": 999})
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_compatibility(saved, expected)


def test_incompatibility_rejects_duplicate_image():
    with pytest.raises(bc.BaselineCheckpointError):
        _valid_checkpoint(next_index=2, processed_image_ids=("a", "a"), processed_count=2)


def test_incompatibility_rejects_missing_baseline_accumulator():
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    metric_state = {k: v.state_dict() for k, v in metric.items()}
    del metric_state[bc.BASELINE_DCR_HARD]
    with pytest.raises(bc.BaselineCheckpointError):
        bc.BaselineCheckpoint(
            schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
            next_index=0, processed_image_ids=(), processed_count=0,
            metric_accumulator_state=metric_state,
            ksweep_telemetry_state={k: v.state_dict() for k, v in ksweep_acc.items()},
            stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
            replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
            call_accounting_state=bc.CallAccounting().state_dict(), context=context,
        )


def test_incompatibility_rejects_wrong_exact_type():
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    ksweep_state = {k: v.state_dict() for k, v in ksweep_acc.items()}
    ksweep_state["4"]["graph_count"] = "not-an-int"
    checkpoint = bc.BaselineCheckpoint(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
        next_index=0, processed_image_ids=(), processed_count=0,
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state=ksweep_state,
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
        call_accounting_state=bc.CallAccounting().state_dict(), context=context,
    )
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_internal_consistency(checkpoint)


def test_incompatibility_rejects_negative_count():
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    ksweep_state = {k: v.state_dict() for k, v in ksweep_acc.items()}
    ksweep_state["4"]["graph_count"] = -1
    checkpoint = bc.BaselineCheckpoint(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
        next_index=0, processed_image_ids=(), processed_count=0,
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state=ksweep_state,
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
        call_accounting_state=bc.CallAccounting().state_dict(), context=context,
    )
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_internal_consistency(checkpoint)


def test_incompatibility_rejects_nonfinite_value():
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    ksweep_state = {k: v.state_dict() for k, v in ksweep_acc.items()}
    ksweep_state["4"]["maximum_scaled_residual"] = float("nan")
    checkpoint = bc.BaselineCheckpoint(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
        next_index=0, processed_image_ids=(), processed_count=0,
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state=ksweep_state,
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
        call_accounting_state=bc.CallAccounting().state_dict(), context=context,
    )
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_internal_consistency(checkpoint)


def test_incompatibility_rejects_inconsistent_call_counts():
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    ksweep_state = {k: v.state_dict() for k, v in ksweep_acc.items()}
    ksweep_state["4"]["graph_count"] = 5
    ksweep_state["4"]["solve_count"] = 5
    checkpoint = bc.BaselineCheckpoint(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
        next_index=0, processed_image_ids=(), processed_count=0,
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state=ksweep_state,
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
        call_accounting_state=bc.CallAccounting(graph_builds=2, solver_calls=2).state_dict(),  # too low
        context=context,
    )
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_internal_consistency(checkpoint)


def test_incompatibility_rejects_incomplete_image_transaction():
    with pytest.raises(bc.BaselineCheckpointError):
        _valid_checkpoint(next_index=3, processed_image_ids=("a", "b"), processed_count=2)


def test_incompatibility_rejects_new_violations_in_replacement_diagnostics():
    context = _make_context()
    metric, ksweep_acc, stitch_acc, replace_acc = _empty_accumulator_state()
    replace_state = {k: v.state_dict() for k, v in replace_acc.items()}
    replace_state[bc.BASELINE_STRICT_SAFE_DCR_HARD]["new_violations"] = 1
    checkpoint = bc.BaselineCheckpoint(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION, run_status=bc.RUN_STATUS_PARTIAL,
        next_index=0, processed_image_ids=(), processed_count=0,
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state={k: v.state_dict() for k, v in ksweep_acc.items()},
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state=replace_state,
        call_accounting_state=bc.CallAccounting().state_dict(), context=context,
    )
    with pytest.raises(bc.BaselineCheckpointError):
        bc.validate_checkpoint_internal_consistency(checkpoint)


# ---------------------------------------------------------------------------
# 11. Atomic-write tests
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_no_partial_file_visible(tmp_path):
    context = _make_context()
    checkpoint = _valid_checkpoint(context=context)
    path = tmp_path / "ckpt.json"
    bc.save_checkpoint_atomic(checkpoint, path)
    assert path.exists()
    assert not (tmp_path / "ckpt.json.tmp").exists()


def test_atomic_write_preserves_previous_valid_checkpoint_on_interrupted_write(tmp_path):
    context = _make_context()
    first = _valid_checkpoint(context=context, next_index=1, processed_image_ids=("img0",), processed_count=1)
    path = tmp_path / "ckpt.json"
    bc.save_checkpoint_atomic(first, path)
    original_bytes = path.read_bytes()

    # Simulate an interrupted second write: the tmp file is created and
    # partially written, but os.replace() never runs.
    tmp_path_file = path.with_name(path.name + ".tmp")
    tmp_path_file.write_text("{not valid json, interrupted mid-write")
    assert path.read_bytes() == original_bytes  # previous checkpoint untouched
    loaded = bc.load_checkpoint(path)
    assert loaded.processed_image_ids == ("img0",)
    tmp_path_file.unlink()


# ---------------------------------------------------------------------------
# Shared multi-image baseline-suite fixture + orchestration for the
# uninterrupted/resumed identity proof (Section 7 / Section 12).
# ---------------------------------------------------------------------------


def _make_window_state(window, raw_scores, s0, feats, patch_grid, k=2):
    n = raw_scores.shape[0]
    indices = torch.zeros(n, k, dtype=torch.int64)
    weights = torch.full((n, k), 1.0 / k)
    affinities = torch.ones(n, k)
    fallback = torch.zeros(n, dtype=torch.bool)
    graph = wc.GraphSnapshot(indices, weights, affinities, fallback, n, k, 3.0)
    telemetry = wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)
    return wc.CachedWindowState(
        geometry=window, window_index=window.index, patch_grid_shape=patch_grid, class_count=CLASS_COUNT,
        s0=s0, dino_features=feats, graph=graph, propagated_scores=raw_scores, solver_summary=telemetry,
    )


def _build_image_fixture(image_id: str, *, seed: int):
    """One multi-window (2x2 overlapping), n=40-node-per-window image, with
    a strict-T4 target guaranteed via the REVERSAL_FIXTURE pattern, plus a
    small GT map with both target and off-target changes and majority/
    center ties elsewhere in the class distribution."""
    plan = SlidingWindowPlan.build(image_size=SpatialSize(8, 8), crop_size=SpatialSize(4, 4), stride=SpatialSize(2, 2))
    n, patch_grid = 40, (5, 8)
    cache = wc.ImageWindowCache(plan)
    generator = torch.Generator().manual_seed(seed)
    for window in plan.windows:
        feats = F.normalize(torch.randn(n, 6, generator=generator), dim=-1)
        s0 = torch.randn(n, CLASS_COUNT, generator=generator)
        raw = torch.randn(n, CLASS_COUNT, generator=generator)
        if window.index == 0:
            # force node 0 into the REVERSAL_FIXTURE pattern relative to
            # this window's own neighbors so a genuine strict-T4 target
            # exists deterministically at (window 0, node 0).
            s0[0] = torch.tensor([5.0, 0.0, 0.0])
            raw[0] = torch.tensor([-10.0, 10.0, 0.0])
        else:
            s0[0] = torch.tensor([5.0, 0.0, 0.0])
            raw[0] = torch.tensor([1.0, 0.0, 0.0])
        cache.append(_make_window_state(window, raw, s0, feats, patch_grid))
    cache.seal()
    context = wc.FirstPassImageContext(
        image_size=plan.image_size, plan=plan, class_count=CLASS_COUNT, common_patch_grid_shape=patch_grid,
        expected_window_count=plan.window_count, cached_window_count=plan.window_count,
        min_coverage=1, max_coverage=plan.window_count, stitched_scores=torch.zeros(1, CLASS_COUNT, 8, 8),
        cache_total_bytes=cache.total_bytes(), pass_summary=wc._summarize_telemetry([]),
    )
    gt = torch.zeros(8, 8, dtype=torch.int64)
    gt[0, 0] = 1  # near the T4 target anchor -- exercises target GT gain/loss accounting
    gt[7, 7] = 2  # far corner -- exercises off-target accounting
    gt[3, 3] = IGNORE_INDEX  # exercises ignore-label handling
    return cache, context, gt


def _run_one_image(image_id, cache, ctx, gt, accumulators, call_accounting):
    metric, ksweep_acc, stitch_acc, replace_acc = accumulators
    calls = call_accounting

    def sigmoid_masks(score_field):
        windows = []
        for state in cache.windows_in_order():
            q = torch.sigmoid(getattr(state, score_field))
            extent = state.geometry.extent.as_tuple()
            mask = consensus.probability_grid_to_mask(q, state.patch_grid_shape, extent, class_count=CLASS_COUNT)
            windows.append(stitching.WindowProbabilityMap(geometry=state.geometry, window_index=state.window_index, probabilities=mask))
        return windows

    calls = calls + bc.CallAccounting(sigmoid_calls=cache.plan.window_count)

    e3_windows = sigmoid_masks("s0")
    for mode, name in [(stitching.STITCH_MODE_UNIFORM, bc.BASELINE_E3_UNIFORM), (stitching.STITCH_MODE_HANN, bc.BASELINE_E3_HANN)]:
        result = stitching.stitch_windows(e3_windows, image_size=cache.plan.image_size, class_count=CLASS_COUNT, mode=mode, score_source="e3_unary_q0")
        calls = calls + bc.CallAccounting(stitching_calls=1)
        metric[name].absorb(image_id, result.image.argmax(dim=0), gt)
        stitch_acc[mode].absorb(result.diagnostics, mode=mode)

    rwr_windows = sigmoid_masks("propagated_scores")
    rwr_results = {}
    for mode, name in [
        (stitching.STITCH_MODE_UNIFORM, bc.BASELINE_RWR_K12_UNIFORM),
        (stitching.STITCH_MODE_MAJORITY, bc.BASELINE_RWR_K12_MAJORITY),
        (stitching.STITCH_MODE_HANN, bc.BASELINE_RWR_K12_HANN),
        (stitching.STITCH_MODE_CENTER_SELECT, bc.BASELINE_RWR_K12_CENTER_SELECT),
    ]:
        result = stitching.stitch_windows(rwr_windows, image_size=cache.plan.image_size, class_count=CLASS_COUNT, mode=mode, score_source="rwr_k12_q")
        calls = calls + bc.CallAccounting(stitching_calls=1)
        metric[name].absorb(image_id, result.image.argmax(dim=0), gt)
        stitch_acc[mode].absorb(result.diagnostics, mode=mode)
        rwr_results[mode] = result

    stitched_by_k, solve_by_window_by_k = ksweep.stitch_k_sweep_for_image(
        cache, image_size=cache.plan.image_size, class_count=CLASS_COUNT, k_values=K_VALUES,
        affinity_power=3.0, alpha=0.98, rtol=None, atol=None, max_iter=2000,
    )
    for k, result in stitched_by_k.items():
        calls = calls + bc.CallAccounting(stitching_calls=1)
        # k=12 uniform is the SAME declared baseline as BASELINE_RWR_K12_UNIFORM
        # (already absorbed above in the canonical-controls loop); avoid a
        # duplicate-image absorption into that one shared accumulator.
        if k != 12:
            metric[bc.k_sweep_baseline_name(k)].absorb(image_id, result.image.argmax(dim=0), gt)
        stitch_acc[stitching.STITCH_MODE_UNIFORM].absorb(result.diagnostics, mode=stitching.STITCH_MODE_UNIFORM)
    for per_k in solve_by_window_by_k.values():
        for k, solve_result in per_k.items():
            ksweep_acc[str(k)].absorb(solve_result)
            calls = calls + bc.CallAccounting(graph_builds=1, solver_calls=1, sigmoid_calls=1, interpolation_calls=1)

    signal = t4.build_t4_signal_for_image(cache, ctx, image_id=image_id)
    before_image = rwr_results[stitching.STITCH_MODE_UNIFORM].image

    for kind, name in [
        (consensus.DCR_HARD, bc.BASELINE_DCR_HARD),
        (consensus.DCR_JURY_MEAN, bc.BASELINE_DCR_JURY_MEAN),
        (consensus.SUR, bc.BASELINE_SUR),
    ]:
        replacements = consensus.build_frozen_replacements(cache, signal, kind=kind, image_id=image_id, class_count=CLASS_COUNT)
        result, report = consensus.apply_replacements_and_stitch(
            cache, replacements, image_size=cache.plan.image_size, class_count=CLASS_COUNT,
            mode=stitching.STITCH_MODE_UNIFORM, score_source=f"rwr_{name}",
        )
        calls = calls + bc.CallAccounting(stitching_calls=1, sigmoid_calls=cache.plan.window_count, interpolation_calls=cache.plan.window_count)
        metric[name].absorb(image_id, result.image.argmax(dim=0), gt)
        replace_acc[name].absorb_application(report)
        target_mask = consensus.compute_target_pixel_mask(cache, replacements, cache.plan.image_size)
        gt_accounting = consensus.compute_gt_accounting(before_image, result.image, gt, target_mask, ignore_index=IGNORE_INDEX)
        replace_acc[name].absorb_gt_accounting(gt_accounting)

        strict_name = {bc.BASELINE_DCR_HARD: bc.BASELINE_STRICT_SAFE_DCR_HARD,
                        bc.BASELINE_DCR_JURY_MEAN: bc.BASELINE_STRICT_SAFE_DCR_JURY_MEAN,
                        bc.BASELINE_SUR: bc.BASELINE_STRICT_SAFE_SUR}[name]
        strict_report = consensus.run_strict_safe_sequential(cache, signal, replacements)
        strict_result, strict_apply_report = consensus.apply_replacements_and_stitch(
            cache, strict_report.accepted_candidates, image_size=cache.plan.image_size, class_count=CLASS_COUNT,
            mode=stitching.STITCH_MODE_UNIFORM, score_source=f"rwr_{strict_name}",
        )
        calls = calls + bc.CallAccounting(stitching_calls=1, sigmoid_calls=cache.plan.window_count, interpolation_calls=cache.plan.window_count)
        metric[strict_name].absorb(image_id, strict_result.image.argmax(dim=0), gt)
        replace_acc[strict_name].absorb_application(strict_apply_report)
        replace_acc[strict_name].absorb_strict_safe(strict_report)
        strict_target_mask = consensus.compute_target_pixel_mask(cache, strict_report.accepted_candidates, cache.plan.image_size)
        strict_gt_accounting = consensus.compute_gt_accounting(before_image, strict_result.image, gt, strict_target_mask, ignore_index=IGNORE_INDEX)
        replace_acc[strict_name].absorb_gt_accounting(strict_gt_accounting)

    return calls


def _run_images(image_specs, accumulators, call_accounting, start_index=0):
    calls = call_accounting
    for image_id, cache, ctx, gt in image_specs[start_index:]:
        calls = _run_one_image(image_id, cache, ctx, gt, accumulators, calls)
    return calls


def _checkpoint_scientific_view(checkpoint: "bc.BaselineCheckpoint") -> dict:
    """Everything except (nothing -- this checkpoint has no timestamp or
    runtime-only fields to exclude in the first place)."""
    return bc._checkpoint_to_json(checkpoint)


def _build_checkpoint(processed_ids, accumulators, call_accounting, context, *, complete: bool):
    metric, ksweep_acc, stitch_acc, replace_acc = accumulators
    return bc.BaselineCheckpoint(
        schema_version=bc.CHECKPOINT_SCHEMA_VERSION,
        run_status=bc.RUN_STATUS_COMPLETE if complete else bc.RUN_STATUS_PARTIAL,
        next_index=len(processed_ids), processed_image_ids=tuple(processed_ids), processed_count=len(processed_ids),
        metric_accumulator_state={k: v.state_dict() for k, v in metric.items()},
        ksweep_telemetry_state={k: v.state_dict() for k, v in ksweep_acc.items()},
        stitching_diagnostics_state={k: v.state_dict() for k, v in stitch_acc.items()},
        replacement_diagnostics_state={k: v.state_dict() for k, v in replace_acc.items()},
        call_accounting_state=call_accounting.state_dict(),
        context=context,
    )


@pytest.fixture(scope="module")
def two_image_fixture():
    cache0, ctx0, gt0 = _build_image_fixture("img0", seed=0)
    cache1, ctx1, gt1 = _build_image_fixture("img1", seed=1)
    return [("img0", cache0, ctx0, gt0), ("img1", cache1, ctx1, gt1)]


def test_uninterrupted_vs_resumed_checkpoint_identity(tmp_path, two_image_fixture):
    context = bc.BaselineCheckpointCompatibilityContext(
        git_head="a" * 40, git_branch="e11-cover-dr-1", baseline_identity_name="stitching-kdcr-baseline-suite",
        baseline_manifest_sha256="b" * 64, e3_identity_sha256="c" * 64, rwr_identity_sha256="d" * 64,
        canonical_config_sha256="e" * 64, checkpoint_sha256="f" * 64, tracked_diff_sha256="0" * 64,
        untracked_source_sha256="1" * 64, crop=(4, 4), stride=(2, 2), k_values=K_VALUES,
        stitching_modes=tuple(stitching.STITCH_MODES), dcr_variants=("dcr_hard", "dcr_jury_mean"),
        sur_definition="sigmoid(S0_w(i))", strict_safe_rule="E(after) proper-subset E(before)",
        class_count=CLASS_COUNT, ignore_label=IGNORE_INDEX, metric_unit_contract="percent_0_to_100",
        dataset_length=2,
    )

    # --- (1) uninterrupted ---
    uninterrupted_accs = _empty_accumulator_state()
    calls = _run_images(two_image_fixture, uninterrupted_accs, bc.CallAccounting())
    uninterrupted_checkpoint = _build_checkpoint(["img0", "img1"], uninterrupted_accs, calls, context, complete=True)

    # --- (2) interrupted after image 1, then resumed for image 2 ---
    resumed_accs = _empty_accumulator_state()
    calls_partial = _run_images(two_image_fixture[:1], resumed_accs, bc.CallAccounting())
    checkpoint_after_img0 = _build_checkpoint(["img0"], resumed_accs, calls_partial, context, complete=False)
    path = tmp_path / "resume.json"
    bc.save_checkpoint_atomic(checkpoint_after_img0, path)

    # simulate a fresh process: reload from checkpoint, reconstruct accumulators
    loaded = bc.load_checkpoint(path)
    bc.validate_checkpoint_compatibility(loaded, context)
    bc.validate_checkpoint_internal_consistency(loaded)
    restored_metric = {k: trust_harness.StreamingSegmentationMetricAccumulator.from_state_dict(v) for k, v in loaded.metric_accumulator_state.items()}
    restored_ksweep = {k: bc.KSweepTelemetryAccumulator.from_state_dict(v) for k, v in loaded.ksweep_telemetry_state.items()}
    restored_stitch = {k: bc.StitchingDiagnosticsAccumulator.from_state_dict(v) for k, v in loaded.stitching_diagnostics_state.items()}
    restored_replace = {k: bc.ReplacementDiagnosticsAccumulator.from_state_dict(v) for k, v in loaded.replacement_diagnostics_state.items()}
    restored_calls = bc.CallAccounting.from_state_dict(loaded.call_accounting_state)
    resumed_accs2 = (restored_metric, restored_ksweep, restored_stitch, restored_replace)

    calls_resumed = _run_images(two_image_fixture, resumed_accs2, restored_calls, start_index=1)
    resumed_checkpoint = _build_checkpoint(["img0", "img1"], resumed_accs2, calls_resumed, context, complete=True)

    assert _checkpoint_scientific_view(uninterrupted_checkpoint) == _checkpoint_scientific_view(resumed_checkpoint)

    # --- (3) interruption in the middle of the SECOND image's processing:
    # since no checkpoint is ever written mid-image, "interrupting" here
    # just means the process died before img1 completed -- resuming means
    # discarding whatever in-memory state existed for img1 and restarting
    # it from the last COMPLETE checkpoint (after img0), which is exactly
    # scenario (2) above. This is proven by construction: _run_one_image
    # is the smallest unit ever checkpointed after, so there is no
    # "partial-image" checkpoint state to separately test.
    mid_interrupt_accs = (
        {k: trust_harness.StreamingSegmentationMetricAccumulator.from_state_dict(v) for k, v in loaded.metric_accumulator_state.items()},
        {k: bc.KSweepTelemetryAccumulator.from_state_dict(v) for k, v in loaded.ksweep_telemetry_state.items()},
        {k: bc.StitchingDiagnosticsAccumulator.from_state_dict(v) for k, v in loaded.stitching_diagnostics_state.items()},
        {k: bc.ReplacementDiagnosticsAccumulator.from_state_dict(v) for k, v in loaded.replacement_diagnostics_state.items()},
    )
    mid_calls = bc.CallAccounting.from_state_dict(loaded.call_accounting_state)
    mid_calls = _run_images(two_image_fixture, mid_interrupt_accs, mid_calls, start_index=1)
    mid_checkpoint = _build_checkpoint(["img0", "img1"], mid_interrupt_accs, mid_calls, context, complete=True)
    assert _checkpoint_scientific_view(mid_checkpoint) == _checkpoint_scientific_view(uninterrupted_checkpoint)

    # --- (4) multiple resume cycles: resume once more from the completed
    # checkpoint (a no-op resume, since processed_count already equals
    # dataset_length) and confirm state is still byte-identical.
    final_path = tmp_path / "final.json"
    bc.save_checkpoint_atomic(resumed_checkpoint, final_path)
    reloaded_again = bc.load_checkpoint(final_path)
    assert _checkpoint_scientific_view(reloaded_again) == _checkpoint_scientific_view(uninterrupted_checkpoint)
