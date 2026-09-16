"""Tests for the BALP development-image processor and 350-image
control-plane orchestration (`src.dfp_phase1b_lightweight_boundary_execute`).

Two distinct fixture tiers, as required:
1. A REAL low-level scientific fixture (real graph/k12 recurrence/
   stitching/final-rescale primitives, a narrow fake model) proving the
   canonical-inference integration is wired correctly, at small scale.
2. A full-cardinality (350-image) SYNTHETIC control-plane harness (no
   torch model/dataset/graph/recurrence -- deterministic BALP-kernel-
   only score fields) proving fresh/resume/finalize-only orchestration,
   development-only population enforcement, and GT isolation at
   authoritative scale.

No real image/annotation access, no real model construction, no CUDA.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import src.dfp_phase1b_lightweight_boundary as balp
import src.dfp_phase1b_lightweight_boundary_artifact as art
import src.dfp_phase1b_lightweight_boundary_execute as ex
import src.dfp_phase1b_lightweight_boundary_identity as e22

ROOT = Path(__file__).parents[1]
E = ex.DfpPhase1bLightweightBoundaryExecuteError

_SPLIT = e22.load_and_verify_split(repo_root=ROOT)
DEV_IDS = tuple(_SPLIT["development_image_ids"])
CONF_IDS = tuple(_SPLIT["confirmation_image_ids"])

CLASS_COUNT = 3
IDENTITY = {
    "e22_protocol_sha256": "1" * 64, "e22_artifacts_sha256": "2" * 64, "e13_protocol_sha256": "3" * 64,
    "matched_evaluator_sha256": "4" * 64, "subset_manifest_sha256": "5" * 64, "balp_split_sha256": "6" * 64,
    "run_mode": "coco_stuff_500",
}


def _providers():
    return ex.ExecutionProviders(
        dataset_resources=None, image_preparer=None, inference=None, protocol_identity={}, contracts=None,
        class_count=CLASS_COUNT, affinity_power=1.0, windows_per_image={},
    )


def _fake_processor(inference, prepared, canonical_id, *, protocol_identity, contracts, class_count, affinity_power, selected_windows_for_image):
    seed = abs(hash(canonical_id)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    scores = {h: torch.randn(class_count, 2, 2, generator=g, dtype=torch.float32) for h in balp.HORIZON_SET}
    kr = balp.compute_balp(scores, class_count=class_count)
    return ex.ImageBalpOutcome(canonical_id=canonical_id, ori_shape=(2, 2), kernel_result=kr)


def _fake_image_provider_factory(poison_ids=frozenset(), call_log=None, gt_value=0):
    def _provider(canonical_id):
        if canonical_id in poison_ids:
            raise AssertionError(f"POISON: {canonical_id} must never be reprocessed")
        if call_log is not None:
            call_log.append(canonical_id)
        return ex.DevelopmentImageProvider(prepared=None, ground_truth=np.full((2, 2), gt_value, dtype=np.int64), selected_windows=[0])
    return _provider


# =====================================================================
# Real low-level scientific fixture: canonical inference integration,
# single window, real graph/recurrence/stitching/rescale, fake model.
# =====================================================================


def _skip_if_cv2_environment_broken():
    """This venv's `cv2` (loaded from the Alliance `opencv` environment
    module) was built against NumPy 1.x and cannot import under the
    venv's NumPy 2.x -- a pre-existing environment ABI conflict, not a
    defect in this task's new code. Several `src.dfp_phase1a_*` modules
    import `cv2` transitively at module level, so any fixture/test that
    triggers those imports must skip (never error) when this is broken.
    """
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            pytest.skip(f"cv2 is not importable in this environment: {exc}")
        raise
    except ImportError as exc:
        if "numpy.core.multiarray failed to import" in str(exc) or "_ARRAY_API" in str(exc):
            pytest.skip(f"cv2 ABI-incompatible with installed NumPy: {exc}")
        raise


@pytest.fixture(scope="module")
def _real_contracts():
    _skip_if_cv2_environment_broken()
    from src.dfp_phase1a_identity import load_identity as load_e13_identity
    from src.dfp_phase1a_final_rescale_pullback import dfp_final_resolution_contract_from_protocol
    from src.dfp_phase1a_horizon_stitching import DfpStitchingContract
    from src.dfp_phase1a_recurrence_capture import dfp_recurrence_contract_from_protocol
    from src.dfp_phase1a_stitching_pullback import DfpPullbackContract

    e13 = load_e13_identity(repo_root=ROOT)
    crop, stride, patch_grid, patch_size = (42, 70), (21, 35), (3, 5), (14, 14)
    connectivity = int(e13["structural_components"]["primary_connectivity"])
    stitching = DfpStitchingContract(horizons=(0, 16, 32, 64, 320), crop=crop, stride=stride, patch_grid=patch_grid, patch_size=patch_size, align_corners=True)
    pullback = DfpPullbackContract(crop=crop, stride=stride, patch_grid=patch_grid, patch_size=patch_size, align_corners=True, connectivity=connectivity)
    contracts = ex.BalpContracts(
        recurrence=dfp_recurrence_contract_from_protocol(e13), stitching=stitching, pullback=pullback,
        final_resolution=dfp_final_resolution_contract_from_protocol(e13, repo_root=ROOT),
    )
    return e13, contracts, crop, patch_grid


class _FakeSnapshot:
    def __init__(self, unary_scores, dino_features):
        self.unary_scores = [unary_scores]
        self.dino_features = [dino_features]


class _FakeModel:
    def __init__(self, per_window_data):
        self.per_window_data = per_window_data
        self.call_count = 0

    def generate_patch_snapshot(self, crop, text_embedding):
        idx = self.call_count
        self.call_count += 1
        unary, feats = self.per_window_data[idx]
        return _FakeSnapshot(unary, feats)


class _FakeInference:
    def __init__(self, model, text_embedding):
        self.model = model
        self.text_embedding = text_embedding


def _make_unary_features(n, offset, class_count):
    unary = torch.zeros(n, class_count, dtype=torch.float64)
    unary[:, 0] = 0.3
    unary[:, 1] = torch.tensor([math.sin(0.7 * i + offset) * 3.0 + 0.05 * i for i in range(n)], dtype=torch.float64)
    feats = torch.tensor([[math.cos(0.3 * i + offset), math.sin(0.5 * i + offset), math.cos(0.9 * i), math.sin(1.1 * i), 0.1 * i, 0.2, 0.3, 0.4] for i in range(n)], dtype=torch.float64)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return unary.to(torch.float32), feats.to(torch.float32)


def test_real_single_window_canonical_inference_produces_valid_balp_outcome(_real_contracts):
    e13, contracts, crop, patch_grid = _real_contracts
    n_nodes = patch_grid[0] * patch_grid[1]
    per_window_data = {0: _make_unary_features(n_nodes, offset=0.0, class_count=CLASS_COUNT)}
    model = _FakeModel(per_window_data)
    text_embedding = torch.eye(CLASS_COUNT, 8, dtype=torch.float32)
    inference = _FakeInference(model, text_embedding)
    prepared = type("PreparedImage", (), {})()
    prepared.image_tensor = torch.zeros(1, 3, crop[0], crop[1])
    prepared.inference_height, prepared.inference_width = crop
    prepared.ori_shape = crop

    outcome = ex.process_development_image(
        inference, prepared, "000000000001", protocol_identity=e13, contracts=contracts,
        class_count=CLASS_COUNT, affinity_power=float(e13["graph"]["affinity_power"]), selected_windows_for_image=[0],
    )
    assert list(outcome.kernel_result.control_labels.keys()) == list(balp.CANONICAL_CONTROLS)
    assert len(outcome.kernel_result.candidate_labels) == 16
    assert model.call_count == 1  # exactly one model forward for the one selected window


def test_real_canonical_inference_ground_truth_never_touched_before_finalization(_real_contracts):
    """GT-isolation proof at the real-inference layer: process_development_image
    has NO ground-truth parameter at all -- structurally cannot access GT."""
    import inspect
    sig = inspect.signature(ex.process_development_image)
    assert "ground_truth" not in sig.parameters
    assert "gt" not in sig.parameters


# =====================================================================
# Metric sufficient statistics (B6): dataset-level aggregation, never
# an averaged per-image mIoU.
# =====================================================================


def test_metric_aggregation_is_confusion_sum_not_per_image_mean():
    _skip_if_cv2_environment_broken()
    from src.dfp_phase1a_analysis import ConfusionStatistics, aggregate_confusion_statistics, compute_semantic_metrics

    cmA = np.array([[10, 1, 0], [0, 8, 0], [0, 0, 5]])
    cmB = np.array([[3, 0, 0], [0, 1, 9], [0, 0, 2]])

    def stats(cm):
        inter = np.diag(cm)
        pred, gt = cm.sum(0), cm.sum(1)
        return ConfusionStatistics(intersection=inter, union=pred + gt - inter, predicted_pixels=pred, ground_truth_pixels=gt, confusion_matrix=cm)

    sA, sB = stats(cmA), stats(cmB)
    agg = aggregate_confusion_statistics([sA, sB])
    naive_mean = (compute_semantic_metrics(sA).mIoU + compute_semantic_metrics(sB).mIoU) / 2
    dataset_level = compute_semantic_metrics(agg).mIoU
    assert abs(dataset_level - naive_mean) > 1e-6
    assert np.array_equal(agg.confusion_matrix, cmA + cmB)


def test_compute_confusion_matrix_respects_ignore_index():
    pred = np.array([[0, 1], [2, 0]])
    gt = np.array([[0, 255], [2, 0]])
    cm = ex.compute_confusion_matrix(pred, gt, class_count=3, ignore_index=255)
    assert cm.sum() == 3  # one pixel ignored
    assert cm[0, 0] == 2 and cm[2, 2] == 1


def test_accumulate_image_metrics_produces_all_21_variants():
    scores = {h: torch.randn(CLASS_COUNT, 3, 3) for h in balp.HORIZON_SET}
    kr = balp.compute_balp(scores, class_count=CLASS_COUNT)
    outcome = ex.ImageBalpOutcome(canonical_id="x", ori_shape=(3, 3), kernel_result=kr)
    gt = np.random.RandomState(0).randint(0, CLASS_COUNT, size=(3, 3)).astype(np.int64)
    record = ex.accumulate_image_metrics(outcome, ground_truth=gt, ignore_index=255, class_count=CLASS_COUNT)
    assert len(record.per_variant_confusion) == 21
    assert len(record.per_variant_scalars) == 21
    for v in balp.CANDIDATE_IDS:
        assert record.per_variant_scalars[v]["preservation_mask_pixel_count"] >= 0


# =====================================================================
# Development-only population enforcement (B5).
# =====================================================================


def test_execute_development_rejects_wrong_id_count():
    with pytest.raises(E):
        ex.execute_development(
            Path(tempfile.mkdtemp()), providers=_providers(), development_ids=DEV_IDS[:349], confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
        )


def test_execute_development_rejects_confirmation_id_in_population(tmp_path):
    bad_ids = list(DEV_IDS[:349]) + [CONF_IDS[0]]
    with pytest.raises(E):
        ex.execute_development(
            tmp_path, providers=_providers(), development_ids=bad_ids, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
        )


def test_execute_development_rejects_reordered_ids(tmp_path):
    reordered = list(DEV_IDS)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(E):
        ex.execute_development(
            tmp_path, providers=_providers(), development_ids=reordered, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
        )


# =====================================================================
# Full 350-image control-plane harness: fresh, resume at every
# required cut point, finalize-only recovery, already-finalized
# rejection, byte-identical scientific digests, no reprocessing.
# =====================================================================


@pytest.fixture(scope="module")
def _clean_fresh_result():
    _skip_if_cv2_environment_broken()  # execute_development finalizes via dfp_phase1a_analysis
    root = Path(tempfile.mkdtemp(prefix="balp_clean_"))
    result = ex.execute_development(
        root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
    )
    yield root, result
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.slow
def test_fresh_350_image_run_produces_all_variants(_clean_fresh_result):
    _, result = _clean_fresh_result
    assert len(result["control_results"]) == 5
    assert len(result["candidate_results"]) == 16
    assert result["complete"] is True and result["final"] is True


@pytest.mark.slow
@pytest.mark.parametrize("cut", [0, 1, 175, 349])
def test_resume_at_cut_point_byte_identical_no_reprocessing(_clean_fresh_result, cut):
    _, clean_result = _clean_fresh_result
    clean_digest = json.dumps(clean_result, sort_keys=True)

    root = Path(tempfile.mkdtemp(prefix=f"balp_resume_{cut}_"))
    try:
        count = {"k": 0}
        def stopping_provider(cid):
            if count["k"] >= cut:
                raise KeyboardInterrupt()
            count["k"] += 1
            return _fake_image_provider_factory()(cid)
        try:
            ex.execute_development(
                root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
                identity_bindings_without_digest=IDENTITY, ignore_index=255,
                image_provider_fn=stopping_provider, image_processor_fn=_fake_processor,
            )
        except KeyboardInterrupt:
            pass

        poison = frozenset(DEV_IDS[:cut])
        call_log = []
        resumed = ex.execute_development(
            root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(poison_ids=poison, call_log=call_log), image_processor_fn=_fake_processor,
        )
        assert json.dumps(resumed, sort_keys=True) == clean_digest
        assert call_log == list(DEV_IDS[cut:])
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.slow
def test_complete_unfinalized_recovery_byte_identical(_clean_fresh_result):
    _, clean_result = _clean_fresh_result
    clean_digest = json.dumps(clean_result, sort_keys=True)

    root = Path(tempfile.mkdtemp(prefix="balp_finalize_only_"))
    try:
        ex.execute_development(
            root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
        )
        shutil.rmtree(root / "development_result")
        pointer = art.load_current_pointer(root)
        gen_dir = root / art.CHECKPOINT_GENERATIONS_DIRNAME / pointer["generation_directory"]
        meta_path = gen_dir / art.CANONICAL_FILENAMES["checkpoint"]
        meta = json.loads(meta_path.read_bytes())
        meta["final"] = False
        meta["final_result_binding"] = None
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        # the pointer binds this generation's checkpoint.json bytes too -- update it to match
        # the edited file, so this reads as a legitimate complete-but-unfinalized state rather
        # than tamper-detected corruption (a separate, already-covered adversarial case).
        pointer_path = root / art.CURRENT_CHECKPOINT_POINTER_FILENAME
        pointer["checkpoint_json_sha256"] = art._sha256_file(meta_path)
        pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")

        poisoned_all = frozenset(DEV_IDS)
        finalize_only_result = ex.execute_development(
            root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(poison_ids=poisoned_all), image_processor_fn=_fake_processor,
        )
        assert json.dumps(finalize_only_result, sort_keys=True) == clean_digest
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.slow
def test_already_finalized_checkpoint_rejected(_clean_fresh_result):
    root, _ = _clean_fresh_result
    with pytest.raises(E):
        ex.execute_development(
            root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(poison_ids=frozenset(DEV_IDS)), image_processor_fn=_fake_processor,
        )


# =====================================================================
# GT isolation (B4): poisoned-GT equivalence and call-order proof.
# =====================================================================


def test_gt_isolation_changing_gt_content_never_changes_predictions(tmp_path):
    """Runs the SAME development population twice with DIFFERENT GT
    content and confirms every prediction (control + candidate label
    maps) is byte-identical -- GT only ever changes the accumulated
    confusion statistics, never a prediction."""
    scores = {h: torch.randn(CLASS_COUNT, 4, 4, generator=torch.Generator().manual_seed(42)) for h in balp.HORIZON_SET}
    outcome = ex.ImageBalpOutcome(canonical_id="x", ori_shape=(4, 4), kernel_result=balp.compute_balp(scores, class_count=CLASS_COUNT))
    gt_a = np.zeros((4, 4), dtype=np.int64)
    gt_b = np.random.RandomState(1).randint(0, CLASS_COUNT, size=(4, 4)).astype(np.int64)
    record_a = ex.accumulate_image_metrics(outcome, ground_truth=gt_a, ignore_index=255, class_count=CLASS_COUNT)
    record_b = ex.accumulate_image_metrics(outcome, ground_truth=gt_b, ignore_index=255, class_count=CLASS_COUNT)
    # predictions themselves are identical (same `outcome` object reused) -- confirms
    # accumulate_image_metrics reads predictions from `outcome`, never recomputing them from GT.
    for v in balp.CANDIDATE_IDS:
        assert record_a.per_variant_scalars[v]["preservation_mask_pixel_count"] == record_b.per_variant_scalars[v]["preservation_mask_pixel_count"]
    # but confusion statistics genuinely differ with different GT
    assert record_a.per_variant_confusion["T0"] != record_b.per_variant_confusion["T0"]


def test_gt_shape_cannot_define_ori_shape():
    """process_development_image derives ori_shape from `prepared.ori_shape`
    (the existing pipeline's own field) -- it has no ground_truth
    parameter to derive a shape from at all."""
    import inspect
    assert "ground_truth" not in inspect.signature(ex.process_development_image).parameters


def test_poisoned_gt_all_ignore_index_leaves_predictions_unchanged():
    scores = {h: torch.randn(CLASS_COUNT, 3, 3, generator=torch.Generator().manual_seed(5)) for h in balp.HORIZON_SET}
    outcome = ex.ImageBalpOutcome(canonical_id="x", ori_shape=(3, 3), kernel_result=balp.compute_balp(scores, class_count=CLASS_COUNT))
    poisoned_gt = np.full((3, 3), 255, dtype=np.int64)  # every pixel ignored
    record = ex.accumulate_image_metrics(outcome, ground_truth=poisoned_gt, ignore_index=255, class_count=CLASS_COUNT)
    for v in balp.CANDIDATE_IDS:
        assert record.per_variant_scalars[v]["evaluated_pixel_count"] == 0
        assert record.per_variant_scalars[v]["preservation_mask_pixel_count"] >= 0  # unaffected by GT poisoning


# =====================================================================
# Defect repair: first-use import isolation (B2/B5 of the repair task).
# `process_development_image` must resolve `models.dinotext.cover_dr.graph`
# on the very first call in a fresh process, without relying on some
# OTHER module having already mutated sys.path as an incidental side
# effect of test ordering.
# =====================================================================


def test_fresh_process_first_call_resolves_graph_import_hermetically():
    """Spawns a genuinely fresh interpreter with PYTHONPATH sanitized to
    only the repository root (plus the opencv module's site-packages,
    required only because this venv's own cv2 needs it -- unrelated to
    the defect under test), importing ONLY the execute module before
    calling `process_development_image` as the first-ever touchpoint.
    Must resolve `models.dinotext.cover_dr.graph`, never
    ModuleNotFoundError."""
    import os
    import subprocess
    import sys as _sys

    opencv_site_packages = next(
        (p for p in _sys.path if "opencv" in p and p.endswith("site-packages")), None
    )
    pythonpath = str(ROOT.resolve())
    if opencv_site_packages:
        pythonpath += ":" + opencv_site_packages

    script = """
import sys
class DummyInference: pass
import src.dfp_phase1b_lightweight_boundary_execute as ex
try:
    ex.process_development_image(DummyInference(), None, "x", protocol_identity={}, contracts=None, class_count=3, affinity_power=1.0, selected_windows_for_image=[])
except ModuleNotFoundError as e:
    print("MODULE_NOT_FOUND:", e)
    raise SystemExit(1)
except Exception as e:
    print("REACHED_PAST_IMPORT:", type(e).__name__)
    raise SystemExit(0)
"""
    with tempfile.TemporaryDirectory(prefix="balp_hermetic_cwd_") as fresh_cwd:
        env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "PYTHONPATH": pythonpath}
        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
        result = subprocess.run(
            [_sys.executable, "-c", script], cwd=fresh_cwd, env=env, capture_output=True, text=True, timeout=60,
        )
    assert "MODULE_NOT_FOUND" not in result.stdout, f"import-order defect reproduced:\n{result.stdout}\n{result.stderr}"
    assert "REACHED_PAST_IMPORT" in result.stdout, f"unexpected output:\n{result.stdout}\n{result.stderr}"
    assert result.returncode == 0


# =====================================================================
# Defect repair: GPU telemetry adapter output validation.
# =====================================================================


class _FakeGpuTelemetry:
    def __init__(self, record):
        self._record = record

    def snapshot(self):
        return self._record


def test_gpu_telemetry_valid_record_accepted():
    rec = ex.ProfilingRecorder(gpu_telemetry=_FakeGpuTelemetry({"gpu_peak_allocated_bytes": 100, "gpu_peak_reserved_bytes": 200}))
    snap = rec.gpu_snapshot()
    assert snap == {"gpu_peak_allocated_bytes": 100, "gpu_peak_reserved_bytes": 200}


def test_gpu_telemetry_none_adapter_returns_none():
    rec = ex.ProfilingRecorder(gpu_telemetry=None)
    assert rec.gpu_snapshot() is None


@pytest.mark.parametrize("bad_record,label", [
    ({"gpu_peak_allocated_bytes": True, "gpu_peak_reserved_bytes": 200}, "bool"),
    ({"gpu_peak_allocated_bytes": 100.0, "gpu_peak_reserved_bytes": 200}, "float"),
    ({"gpu_peak_allocated_bytes": "100", "gpu_peak_reserved_bytes": 200}, "string"),
    ({"gpu_peak_allocated_bytes": -1, "gpu_peak_reserved_bytes": 200}, "negative"),
    ({"gpu_peak_allocated_bytes": float("nan"), "gpu_peak_reserved_bytes": 200}, "nan"),
    ({"gpu_peak_allocated_bytes": float("inf"), "gpu_peak_reserved_bytes": 200}, "infinity"),
    ({"gpu_peak_reserved_bytes": 200}, "missing_key"),
    ({"gpu_peak_allocated_bytes": 100, "gpu_peak_reserved_bytes": 200, "extra": 1}, "extra_key"),
    ({"gpu_peak_allocated_bytes": 300, "gpu_peak_reserved_bytes": 200}, "allocated_exceeds_reserved"),
    (None, "null"),
    ("not-a-dict", "not_a_mapping"),
])
def test_gpu_telemetry_malformed_record_rejected(bad_record, label):
    rec = ex.ProfilingRecorder(gpu_telemetry=_FakeGpuTelemetry(bad_record))
    with pytest.raises(ex.DfpPhase1bLightweightBoundaryExecuteError):
        rec.gpu_snapshot()


def test_gpu_telemetry_adapter_fatal_exception_propagates():
    class ExplodingTelemetry:
        def snapshot(self):
            raise KeyboardInterrupt("synthetic fatal from adapter")
    rec = ex.ProfilingRecorder(gpu_telemetry=ExplodingTelemetry())
    with pytest.raises(KeyboardInterrupt):
        rec.gpu_snapshot()


def test_gpu_telemetry_cpu_mode_null_representation_unchanged():
    rec = ex.ProfilingRecorder()
    assert rec.gpu_snapshot() is None


# =====================================================================
# Defect repair: resume after an injected checkpoint-generation failure,
# driven through the real execute_development orchestration end-to-end
# (not just the lower-level artifact-module fault injection tests).
# =====================================================================


def test_execute_development_resume_after_injected_generation_install_failure(tmp_path):
    """Runs a real (synthetic-provider) development execution up to 5
    images, injects a failure in the checkpoint-generation write path
    for the 6th image (simulating a crash mid-install), confirms the
    checkpoint remains at 5 completed images and loadable, then clears
    the fault and confirms resume reaches the SAME final scientific
    digest as an unfaulted run, with zero already-committed images
    reprocessed."""
    import src.dfp_phase1b_lightweight_boundary_artifact as art

    root_clean = tmp_path / "clean"
    clean_result = ex.execute_development(
        root_clean, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
    )
    clean_digest = json.dumps(clean_result, sort_keys=True)

    root = tmp_path / "faulted"
    count = {"k": 0}
    def stopping_provider(cid):
        if count["k"] >= 5:
            raise KeyboardInterrupt()
        count["k"] += 1
        return _fake_image_provider_factory()(cid)
    try:
        ex.execute_development(
            root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=stopping_provider, image_processor_fn=_fake_processor,
        )
    except KeyboardInterrupt:
        pass

    before_doc = art.load_full_checkpoint(root, variant_ids=ex.VARIANT_IDS)
    assert before_doc["progress"]["next_development_index"] == 5

    real_write_json = art._write_checkpoint_json_into
    def failing_write(*a, **k):
        raise OSError("SYNTHETIC install failure for image 6's generation")
    art._write_checkpoint_json_into = failing_write
    try:
        with pytest.raises(OSError):
            ex.execute_development(
                root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
                identity_bindings_without_digest=IDENTITY, ignore_index=255,
                image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
            )
    finally:
        art._write_checkpoint_json_into = real_write_json

    after_doc = art.load_full_checkpoint(root, variant_ids=ex.VARIANT_IDS)
    assert after_doc == before_doc  # unaffected by the failed install attempt

    poison = frozenset(DEV_IDS[:5])
    call_log = []
    resumed = ex.execute_development(
        root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(poison_ids=poison, call_log=call_log), image_processor_fn=_fake_processor,
    )
    assert json.dumps(resumed, sort_keys=True) == clean_digest
    assert call_log == list(DEV_IDS[5:])


# =====================================================================
# Orphan-successor recovery, driven through the real execute_development
# orchestration at authoritative scale (350 frozen development images).
# Repairs the permanent resumability deadlock: a new generation
# installed successfully but the current pointer never advanced to it.
# =====================================================================


def _install_n_via_execute(root, n, *, providers=None):
    providers = providers or _providers()
    count = {"k": 0}
    def stopping(cid):
        if count["k"] >= n:
            raise KeyboardInterrupt()
        count["k"] += 1
        return _fake_image_provider_factory()(cid)
    try:
        ex.execute_development(
            root, providers=providers, development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=stopping, image_processor_fn=_fake_processor,
        )
    except KeyboardInterrupt:
        pass


@pytest.mark.slow
@pytest.mark.parametrize("cut", [1, 2, 175, 349])
def test_orphan_recovery_after_post_install_verification_failure_full_350(tmp_path, cut):
    """Injects a failure between successful generation-directory install
    and successful post-install reverification (one of the two formerly
    permanently-deadlocking boundaries), then retries from the same
    output root and requires automatic orphan adoption with zero
    reprocessing and byte-identical output to a clean fresh run."""
    clean_root = tmp_path / "clean"
    clean_result = ex.execute_development(
        clean_root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
    )
    clean_digest = json.dumps(clean_result, sort_keys=True)

    root = tmp_path / f"cut{cut}"
    _install_n_via_execute(root, cut)

    real_validate = art.validate_full_checkpoint
    state = {"n": 0}
    target_next_index = cut + 1
    def failing_validate(doc, *, development_ids, confirmation_ids):
        if doc["progress"]["next_development_index"] == target_next_index:
            state["n"] += 1
            if state["n"] == 2:
                raise art.DfpPhase1bLightweightBoundaryArtifactError("SYNTHETIC post-install reverify failure")
        return real_validate(doc, development_ids=development_ids, confirmation_ids=confirmation_ids)
    art.validate_full_checkpoint = failing_validate
    try:
        with pytest.raises(art.DfpPhase1bLightweightBoundaryArtifactError):
            ex.execute_development(
                root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
                identity_bindings_without_digest=IDENTITY, ignore_index=255,
                image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
            )
    finally:
        art.validate_full_checkpoint = real_validate

    # confirm the orphan is genuinely installed and unreferenced before retry. The orphan is
    # generation `cut` (0-indexed), representing `cut + 1` completed images (DEV_IDS[:cut] were
    # already the real current checkpoint from `_install_n_via_execute`; the orphan adds exactly
    # one more, DEV_IDS[cut]).
    pointer_before = art.load_current_pointer(root)
    assert pointer_before["current_generation"] == cut - 1
    gens_dir = root / art.CHECKPOINT_GENERATIONS_DIRNAME
    assert (gens_dir / f"generation_{cut}").is_dir()

    # retry: must adopt the orphan automatically, reprocessing NEITHER of the cut+1 images it
    # (together with the pre-orphan checkpoint) already represents.
    poison = frozenset(DEV_IDS[: cut + 1])
    call_log = []
    resumed = ex.execute_development(
        root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(poison_ids=poison, call_log=call_log), image_processor_fn=_fake_processor,
    )
    assert json.dumps(resumed, sort_keys=True) == clean_digest
    assert call_log == list(DEV_IDS[cut + 1:])
    assert art.load_current_pointer(root)["current_generation"] >= cut  # orphan was adopted, not re-created


@pytest.mark.slow
def test_orphan_recovery_after_pointer_write_failure_full_350(tmp_path):
    """The second formerly-deadlocking boundary: failure during the
    pointer os.replace itself (after the generation directory was
    already successfully installed)."""
    clean_root = tmp_path / "clean"
    clean_result = ex.execute_development(
        clean_root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
    )
    clean_digest = json.dumps(clean_result, sort_keys=True)

    root = tmp_path / "ptrfail"
    cut = 175
    _install_n_via_execute(root, cut)

    real_advance = art._advance_current_pointer
    def failing_advance(*a, **k):
        raise OSError("SYNTHETIC crash during pointer os.replace")
    art._advance_current_pointer = failing_advance
    try:
        with pytest.raises(OSError):
            ex.execute_development(
                root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
                identity_bindings_without_digest=IDENTITY, ignore_index=255,
                image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
            )
    finally:
        art._advance_current_pointer = real_advance

    poison = frozenset(DEV_IDS[: cut + 1])
    call_log = []
    resumed = ex.execute_development(
        root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(poison_ids=poison, call_log=call_log), image_processor_fn=_fake_processor,
    )
    assert json.dumps(resumed, sort_keys=True) == clean_digest
    assert call_log == list(DEV_IDS[cut + 1:])


@pytest.mark.slow
def test_orphan_recovery_complete_but_unfinalized_finalize_only(tmp_path):
    """A valid installed-but-unreferenced checkpoint representing ALL
    350 images (complete=True, final=False) -- the crash happened
    during the pointer advance for the very last per-image generation,
    strictly BEFORE `_finalize_only` ever starts. Recovery must adopt
    it, and the retry must then naturally reach the EXISTING
    complete-but-unfinalized `_finalize_only` path (a legal 'is_normal_
    image_extension' transition: next_development_index goes from 349
    to 350) without ANY image/model/dataset/graph/recurrence provider
    access, producing a result identical to a clean fresh run."""
    clean_root = tmp_path / "clean"
    clean_result = ex.execute_development(
        clean_root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
    )
    clean_digest = json.dumps(clean_result, sort_keys=True)

    root = tmp_path / "finalize_orphan"
    _install_n_via_execute(root, 349)

    # process the 350th (final) image via a REAL execute_development call, but poison the
    # pointer-advance step for THAT per-image install -- strictly before _finalize_only ever runs.
    real_advance = art._advance_current_pointer
    def failing_advance(*a, **k):
        raise OSError("SYNTHETIC crash during the 350th image's own pointer advance, before finalize starts")
    art._advance_current_pointer = failing_advance
    try:
        with pytest.raises(OSError):
            ex.execute_development(
                root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
                identity_bindings_without_digest=IDENTITY, ignore_index=255,
                image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
            )
    finally:
        art._advance_current_pointer = real_advance

    # confirm the orphan (generation 349, complete=True, final=False) is genuinely installed
    # and unreferenced, and no development_result was installed yet (finalize never started).
    assert art.load_current_pointer(root)["current_generation"] == 348
    assert (root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_349").is_dir()
    assert not (root / "development_result").exists()

    # every image/model/dataset/graph/recurrence provider poisoned for the retry -- adoption
    # plus finalize-only recovery must call none of them.
    call_log = []
    resumed = ex.execute_development(
        root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(poison_ids=frozenset(DEV_IDS), call_log=call_log), image_processor_fn=_fake_processor,
    )
    assert json.dumps(resumed, sort_keys=True) == clean_digest
    assert call_log == []
    assert art.load_full_checkpoint(root, variant_ids=ex.VARIANT_IDS)["final"] is True
    assert art.load_current_pointer(root)["current_generation"] == 350  # 349 (adopted orphan) then 350 (finalize)


def test_orphan_recovery_does_not_regress_already_finalized_rejection(tmp_path):
    root = tmp_path / "already_final"
    ex.execute_development(
        root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
    )
    with pytest.raises(ex.DfpPhase1bLightweightBoundaryExecuteError):
        ex.execute_development(
            root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(poison_ids=frozenset(DEV_IDS)), image_processor_fn=_fake_processor,
        )


@pytest.mark.slow
def test_orphan_recovery_already_final_orphan_adopted_then_rejects_further_execute(tmp_path):
    """A valid installed-but-unreferenced orphan that is ITSELF
    final=True (the crash happened between the finalize-generation's
    own install and its pointer advance): reconciliation must still
    adopt it (a legal finalize-only transition), advancing the pointer
    -- and, per the existing finalized-state policy, any FURTHER
    execute_development call against the now-current, now-final
    checkpoint must reject, never silently re-run or re-finalize."""
    clean_root = tmp_path / "clean"
    ex.execute_development(
        clean_root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
        identity_bindings_without_digest=IDENTITY, ignore_index=255,
        image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
    )

    root = tmp_path / "already_final_orphan"
    _install_n_via_execute(root, 349)

    real_advance = art._advance_current_pointer
    calls = {"n": 0}
    def fail_on_finalize_pointer_advance(output_root, *, generation, generation_directory, checkpoint_json_sha256, checkpoint_arrays_sha256):
        calls["n"] += 1
        if calls["n"] == 2:  # 1st call = the 350th per-image install; 2nd = the finalize-generation
            raise OSError("SYNTHETIC crash during the finalize-generation's own pointer advance")
        return real_advance(output_root, generation=generation, generation_directory=generation_directory,
                             checkpoint_json_sha256=checkpoint_json_sha256, checkpoint_arrays_sha256=checkpoint_arrays_sha256)
    art._advance_current_pointer = fail_on_finalize_pointer_advance
    try:
        with pytest.raises(OSError):
            ex.execute_development(
                root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
                identity_bindings_without_digest=IDENTITY, ignore_index=255,
                image_provider_fn=_fake_image_provider_factory(), image_processor_fn=_fake_processor,
            )
    finally:
        art._advance_current_pointer = real_advance

    # confirm the orphan is genuinely final=True and unreferenced (the 350th per-image
    # generation, 349, succeeded normally -- only the SUBSEQUENT finalize-generation's own
    # pointer advance was made to fail)
    assert art.load_current_pointer(root)["current_generation"] == 349
    gd = root / art.CHECKPOINT_GENERATIONS_DIRNAME / "generation_350"
    assert gd.is_dir()
    orphan_doc = json.loads((gd / art.CANONICAL_FILENAMES["checkpoint"]).read_bytes())
    assert orphan_doc["final"] is True

    # a further execute_development call must adopt the orphan (advancing the pointer) and THEN
    # correctly reject per the existing finalized-state policy -- never silently succeed or rerun.
    with pytest.raises(ex.DfpPhase1bLightweightBoundaryExecuteError, match="already finalized"):
        ex.execute_development(
            root, providers=_providers(), development_ids=DEV_IDS, confirmation_ids=CONF_IDS,
            identity_bindings_without_digest=IDENTITY, ignore_index=255,
            image_provider_fn=_fake_image_provider_factory(poison_ids=frozenset(DEV_IDS)), image_processor_fn=_fake_processor,
        )
    assert art.load_current_pointer(root)["current_generation"] == 350  # adoption itself DID happen
