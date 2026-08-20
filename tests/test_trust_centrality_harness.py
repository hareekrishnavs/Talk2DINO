"""Tests for the full/partial trust-centrality evaluation harness: run-mode
decision, parity-check scheduling, the streaming segmentation-metric
accumulator, canonical E3/RWR reference-metric attribution, and checkpoint
save/resume.

Loaded via file-path import (mirroring tests/test_t4_audit.py) so most of
these tests never require mmcv/cv2/CUDA. The streaming-accumulator tests
that call the real ``mmseg.core.evaluation.metrics.intersect_and_union``
are skipped (not failed) when mmseg/cv2 are unavailable in the active
environment -- run with ``module load opencv/4.14.0`` first to exercise
them for real.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation"
HARNESS_PATH = EVAL_DIR / "trust_centrality_harness.py"
GEOMETRY_PATH = EVAL_DIR / "sliding_window_geometry.py"
CACHE_PATH = EVAL_DIR / "window_cache.py"
T4_AUDIT_PATH = EVAL_DIR / "t4_audit.py"
DIAG_PATH = EVAL_DIR / "trust_centrality_diagnostics.py"


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
t4 = _load("segmentation.evaluation.t4_audit", T4_AUDIT_PATH)
tcd = _load("segmentation.evaluation.trust_centrality_diagnostics", DIAG_PATH)
h = _load("segmentation.evaluation.trust_centrality_harness", HARNESS_PATH)

try:
    import mmseg  # noqa: F401
    _MMSEG_AVAILABLE = True
except Exception:
    _MMSEG_AVAILABLE = False

requires_mmseg = pytest.mark.skipif(not _MMSEG_AVAILABLE, reason="requires mmseg/cv2 (module load opencv/4.14.0)")
SpatialSize = geometry.SpatialSize


# ---------------------------------------------------------------------------
# 1-2: run-mode / stop decisions
# ---------------------------------------------------------------------------


def test_1_full_dataset_limit_never_stops_early():
    mode = h.determine_run_mode(5000, 5000)
    assert mode == h.RUN_MODE_FULL
    assert h.should_stop_after_image(mode=mode, images_processed=5000, image_limit=5000, window_floor_satisfied=True) is False
    assert h.should_stop_after_image(mode=mode, images_processed=1, image_limit=5000, window_floor_satisfied=True) is False


def test_1b_absent_limit_is_also_full_dataset_and_never_stops():
    mode = h.determine_run_mode(None, 5000)
    assert mode == h.RUN_MODE_FULL
    assert h.should_stop_after_image(mode=mode, images_processed=4999, image_limit=None, window_floor_satisfied=True) is False


def test_2_smaller_limit_stops_cleanly_as_partial():
    mode = h.determine_run_mode(100, 5000)
    assert mode == h.RUN_MODE_PARTIAL
    assert h.should_stop_after_image(mode=mode, images_processed=99, image_limit=100, window_floor_satisfied=True) is False
    assert h.should_stop_after_image(mode=mode, images_processed=100, image_limit=100, window_floor_satisfied=True) is True
    # window floor still gates the stop even once the image limit is hit
    assert h.should_stop_after_image(mode=mode, images_processed=100, image_limit=100, window_floor_satisfied=False) is False


# ---------------------------------------------------------------------------
# 3-4: full reaches finalization; partial never reports full-dataset mIoU
# ---------------------------------------------------------------------------


@requires_mmseg
def test_3_full_mode_reaches_metric_finalization():
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    acc.absorb("img0", torch.tensor([[0, 1], [2, 0]]), torch.tensor([[0, 1], [2, 0]]))
    result = acc.finalize(scope=h.RUN_MODE_FULL, complete=True)
    assert result.complete is True
    assert result.mIoU == pytest.approx(100.0)
    assert result.unavailable_reason is None


def test_4_partial_mode_never_reports_full_dataset_observed_miou():
    result = h.unavailable_observed_metrics(scope=h.RUN_MODE_PARTIAL, evaluated_images=100, classes=171, reason="partial run: 100 of 5000 images processed")
    assert result.complete is False
    assert result.mIoU is None
    assert result.aAcc is None
    assert result.mAcc is None
    assert result.unavailable_reason


def test_4b_observed_metrics_rejects_partial_metrics_smuggled_in():
    with pytest.raises(h.TrustCentralityHarnessError):
        h.ObservedRunMetrics(
            scope=h.RUN_MODE_PARTIAL, complete=False, evaluated_images=10, unique_images=10, classes=3,
            metric_source=h.METRIC_SOURCE_STREAMING, aAcc=50.0, mIoU=None, mAcc=None,
            per_class_iou=None, per_class_acc=None, unavailable_reason="partial",
        )


def test_4c_complete_run_requires_all_metrics_populated():
    with pytest.raises(h.TrustCentralityHarnessError):
        h.ObservedRunMetrics(
            scope=h.RUN_MODE_FULL, complete=True, evaluated_images=5000, unique_images=5000, classes=171,
            metric_source=h.METRIC_SOURCE_STREAMING, aAcc=None, mIoU=None, mAcc=None,
            per_class_iou=None, per_class_acc=None, unavailable_reason=None,
        )


# ---------------------------------------------------------------------------
# 5-7: streaming statistics match mmseg exactly
# ---------------------------------------------------------------------------


@requires_mmseg
def test_5_streaming_statistics_match_mmseg_exactly():
    from mmseg.core.evaluation.metrics import eval_metrics
    import numpy as np

    preds = [
        torch.tensor([[0, 1, 2], [1, 1, 0]]),
        torch.tensor([[2, 2, 0], [1, 0, 2]]),
    ]
    gts = [
        torch.tensor([[0, 1, 1], [1, 2, 0]]),
        torch.tensor([[2, 1, 0], [1, 0, 255]]),
    ]
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    for i, (p, g) in enumerate(zip(preds, gts)):
        acc.absorb(f"img{i}", p, g)
    ours = acc.finalize(scope=h.RUN_MODE_FULL, complete=True)

    ref = eval_metrics(
        [p.numpy() for p in preds], [g.numpy() for g in gts], num_classes=3, ignore_index=255, metrics=["mIoU"],
    )
    ref_aAcc = float(np.nanmean(ref["aAcc"])) * 100
    ref_mIoU = float(np.nanmean(ref["IoU"])) * 100
    ref_mAcc = float(np.nanmean(ref["Acc"])) * 100

    assert ours.aAcc == pytest.approx(ref_aAcc, abs=1e-9)
    assert ours.mIoU == pytest.approx(ref_mIoU, abs=1e-9)
    assert ours.mAcc == pytest.approx(ref_mAcc, abs=1e-9)


@requires_mmseg
def test_6_ignore_labels_handled_identically_to_mmseg():
    from mmseg.core.evaluation.metrics import intersect_and_union
    import numpy as np

    pred = torch.tensor([[0, 1], [2, 1]])
    gt = torch.tensor([[0, 255], [2, 1]])
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    acc.absorb("img0", pred, gt)
    result = acc.finalize(scope=h.RUN_MODE_FULL, complete=True)

    intersect, union, pred_area, label_area = intersect_and_union(
        pred.numpy(), gt.numpy(), 3, 255, label_map={}, reduce_zero_label=False,
    )
    expected_aAcc = float((intersect.sum() / label_area.sum()).item()) * 100
    assert result.aAcc == pytest.approx(expected_aAcc, abs=1e-9)
    # the ignored pixel (label=255) must not appear in either total
    assert label_area.sum().item() == 3  # only 3 of the 4 pixels are non-ignored


@requires_mmseg
def test_7_per_class_and_aggregate_metrics_match_independent_fixture():
    # Hand-derived fixture: 2x2 image, 2 classes, perfect prediction except
    # one pixel of class 1 predicted as class 0.
    pred = torch.tensor([[0, 0], [1, 1]])
    gt = torch.tensor([[0, 1], [1, 1]])
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=2, ignore_index=255)
    acc.absorb("img0", pred, gt)
    result = acc.finalize(scope=h.RUN_MODE_FULL, complete=True)

    # Hand computation: class0 intersect=1 (only (0,0)), pred_area=2 ((0,0)&(0,1)
    # predicted 0), gt_area=1 (only (0,0) is really class0) -> union=2+1-1=2 -> IoU=50%
    # class1: intersect=2 (bottom row), pred_area=2, gt_area=3 ((0,1),(1,0),(1,1))
    # -> union=2+3-2=3 -> IoU=66.667%
    expected_iou_0 = 50.0
    expected_iou_1 = pytest.approx(2 / 3 * 100, abs=1e-6)
    assert result.per_class_iou[0] == pytest.approx(expected_iou_0)
    assert result.per_class_iou[1] == expected_iou_1
    # aAcc: 3 correct out of 4 pixels
    assert result.aAcc == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# 8-11: counting discipline
# ---------------------------------------------------------------------------


@requires_mmseg
def test_8_9_each_prediction_counted_exactly_once_no_double_counting():
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=2, ignore_index=255)
    for i in range(5):
        acc.absorb(f"img{i}", torch.zeros(2, 2, dtype=torch.int64), torch.zeros(2, 2, dtype=torch.int64))
    assert acc.image_count() == 5
    assert len(acc.unique_image_ids()) == 5
    # a "parity" prediction for the SAME image must never be separately
    # absorbed under a different id scheme -- simulated here by confirming
    # the accumulator has no API that accepts anything but one (image_id,
    # prediction, gt) triple per call, and that re-using an id is rejected.
    with pytest.raises(h.TrustCentralityHarnessError):
        acc.absorb("img0", torch.zeros(2, 2, dtype=torch.int64), torch.zeros(2, 2, dtype=torch.int64))


def test_10_duplicate_image_ids_fail():
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=2, ignore_index=255) if _MMSEG_AVAILABLE else None
    if acc is None:
        pytest.skip("requires mmseg/cv2")
    acc.absorb("dup", torch.zeros(2, 2, dtype=torch.int64), torch.zeros(2, 2, dtype=torch.int64))
    with pytest.raises(h.TrustCentralityHarnessError):
        acc.absorb("dup", torch.zeros(2, 2, dtype=torch.int64), torch.zeros(2, 2, dtype=torch.int64))


def test_11_missing_final_images_prevent_complete_status():
    assert h.is_dataset_fully_covered(["a", "b", "c"], 3) is True
    assert h.is_dataset_fully_covered(["a", "b"], 3) is False  # missing one
    assert h.is_dataset_fully_covered(["a", "a", "b"], 3) is False  # duplicate, not 3 unique


# ---------------------------------------------------------------------------
# 12-14: checkpoint save/resume
# ---------------------------------------------------------------------------


def _make_context(**overrides):
    defaults = dict(
        git_head="a" * 40, git_branch="e11-cover-dr-1", canonical_config_sha256="b" * 64,
        identity_name="e3-canonical-directed-rwr", e3_identity_sha256="c" * 64, rwr_identity_sha256="d" * 64,
        tracked_diff_sha256="e" * 64, untracked_source_sha256="f" * 64,
        num_classes=3, ignore_index=255, reduce_zero_label=False, parity_check_interval=25, dataset_length=5000,
        bootstrap_resamples=1000, bootstrap_seed=1,
        centrality_bin_edges=tuple(-1.0 + i * 0.2 for i in range(11)),
        edge_band_labels=("<1", "[1,2)", "[2,4)", ">=4"),
        diagnostic_schema_version="talk2dino-trust-centrality-report-v3",
    )
    defaults.update(overrides)
    return h.CheckpointCompatibilityContext(**defaults)


@requires_mmseg
def test_12_atomic_checkpoint_round_trip_preserves_metric_state(tmp_path):
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    acc.absorb("img0", torch.tensor([[0, 1], [2, 0]]), torch.tensor([[0, 1], [2, 0]]))
    acc.absorb("img1", torch.tensor([[1, 1], [2, 2]]), torch.tensor([[1, 0], [2, 2]]))

    checkpoint = h.HarnessCheckpoint(
        schema_version=h.CHECKPOINT_SCHEMA_VERSION, run_status=h.RUN_STATUS_PARTIAL, next_index=2,
        processed_image_ids=("img0", "img1"), processed_count=2, accumulator_state=acc.state_dict(),
        context=_make_context(), diagnostic_settings={"parity_check_every": 25}, parity_checked_image_ids=("img0",),
    )
    path = tmp_path / "checkpoint.json"
    h.save_checkpoint_atomic(checkpoint, path)
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()  # temp file cleaned up via os.replace

    restored = h.load_checkpoint(path)
    assert restored.processed_image_ids == ("img0", "img1")
    assert restored.context == checkpoint.context

    restored_acc = h.StreamingSegmentationMetricAccumulator.from_state_dict(restored.accumulator_state)
    assert restored_acc.image_count() == 2
    assert restored_acc.unique_image_ids() == frozenset({"img0", "img1"})
    assert restored_acc.finalize(scope=h.RUN_MODE_FULL, complete=True) == acc.finalize(scope=h.RUN_MODE_FULL, complete=True)


@requires_mmseg
def test_13_resume_yields_same_metrics_as_uninterrupted_processing():
    images = [
        (f"img{i}", torch.randint(0, 3, (4, 4)), torch.randint(0, 3, (4, 4)))
        for i in range(6)
    ]

    uninterrupted = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    for image_id, pred, gt in images:
        uninterrupted.absorb(image_id, pred, gt)
    uninterrupted_result = uninterrupted.finalize(scope=h.RUN_MODE_FULL, complete=True)

    first_half = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    for image_id, pred, gt in images[:3]:
        first_half.absorb(image_id, pred, gt)
    state = first_half.state_dict()

    resumed = h.StreamingSegmentationMetricAccumulator.from_state_dict(state)
    for image_id, pred, gt in images[3:]:
        resumed.absorb(image_id, pred, gt)
    resumed_result = resumed.finalize(scope=h.RUN_MODE_FULL, complete=True)

    assert resumed_result == uninterrupted_result


def test_14_incompatible_checkpoints_fail_closed():
    saved = _make_context(git_head="a" * 40)
    current = _make_context(git_head="c" * 40)
    with pytest.raises(h.TrustCentralityHarnessError):
        h.validate_checkpoint_compatibility(
            h.HarnessCheckpoint(
                schema_version=h.CHECKPOINT_SCHEMA_VERSION, run_status=h.RUN_STATUS_PARTIAL, next_index=0,
                processed_image_ids=(), processed_count=0, accumulator_state={}, context=saved,
                diagnostic_settings={}, parity_checked_image_ids=(),
            ),
            current,
        )


def test_checkpoint_rejects_inconsistent_next_index_as_a_skipped_image_signal():
    # next_index tracks how many images this sequential harness has
    # processed; it must always equal processed_count (== len(
    # processed_image_ids)). A mismatch means an image's index advanced
    # without a corresponding recorded id -- i.e. a skipped image, or a
    # corrupted checkpoint -- and must fail closed at construction time.
    with pytest.raises(h.TrustCentralityHarnessError, match="next_index"):
        h.HarnessCheckpoint(
            schema_version=h.CHECKPOINT_SCHEMA_VERSION, run_status=h.RUN_STATUS_PARTIAL,
            next_index=5, processed_image_ids=("a", "b"), processed_count=2,
            accumulator_state={}, context=_make_context(), diagnostic_settings={}, parity_checked_image_ids=(),
        )


def test_14b_wrong_schema_version_fails_closed():
    with pytest.raises(h.TrustCentralityHarnessError):
        h.HarnessCheckpoint(
            schema_version="some-other-version", run_status=h.RUN_STATUS_PARTIAL, next_index=0,
            processed_image_ids=(), processed_count=0, accumulator_state={}, context=_make_context(),
            diagnostic_settings={}, parity_checked_image_ids=(),
        )


@pytest.mark.parametrize("override", [
    dict(git_head="c" * 40), dict(git_branch="other-branch"), dict(canonical_config_sha256="c" * 64),
    dict(identity_name="different-identity"), dict(e3_identity_sha256="z" * 64), dict(rwr_identity_sha256="c" * 64),
    dict(tracked_diff_sha256="c" * 64), dict(untracked_source_sha256="c" * 64),
    dict(num_classes=999), dict(ignore_index=254), dict(reduce_zero_label=True),
    dict(parity_check_interval=1), dict(dataset_length=1),
    dict(bootstrap_resamples=1), dict(bootstrap_seed=999),
    dict(centrality_bin_edges=tuple(-1.0 + i * 0.25 for i in range(9))),
    dict(edge_band_labels=("only_one_band",)), dict(diagnostic_schema_version="some-other-schema"),
])
def test_checkpoint_context_mismatch_fails_closed_for_every_provenance_field(override):
    saved = _make_context()
    current = _make_context(**override)
    with pytest.raises(h.TrustCentralityHarnessError):
        h.validate_checkpoint_compatibility(
            h.HarnessCheckpoint(
                schema_version=h.CHECKPOINT_SCHEMA_VERSION, run_status=h.RUN_STATUS_PARTIAL, next_index=0,
                processed_image_ids=(), processed_count=0, accumulator_state={}, context=saved,
                diagnostic_settings={}, parity_checked_image_ids=(),
            ),
            current,
        )


def test_checkpoint_load_fails_closed_on_corrupted_json(tmp_path):
    path = tmp_path / "corrupted.json"
    path.write_text("{not valid json")
    with pytest.raises(h.TrustCentralityHarnessError):
        h.load_checkpoint(path)


def test_checkpoint_load_fails_closed_on_missing_required_field(tmp_path):
    import json
    incomplete = {"schema_version": h.CHECKPOINT_SCHEMA_VERSION, "run_status": "partial"}  # missing everything else
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(incomplete))
    with pytest.raises(h.TrustCentralityHarnessError):
        h.load_checkpoint(path)


@requires_mmseg
def test_checkpoint_at_final_image_resumes_to_an_already_complete_run():
    image_ids = [f"img{i}" for i in range(3)]
    metric_acc, t4_acc, trust_acc, primary_acc, windows_processed, caches = _run_pipeline(image_ids)
    state = _full_accumulator_state(
        metric_acc=metric_acc, t4_acc=t4_acc, trust_acc=trust_acc,
        primary_acc=primary_acc, parity_solver={}, windows_processed=windows_processed,
    )
    h.validate_accumulator_state_complete(state)

    resumed_metric = h.StreamingSegmentationMetricAccumulator.from_state_dict(state["metric_accumulator"])
    resumed_t4 = t4.T4AuditAccumulator.from_state_dict(state["t4_accumulator"])
    resumed_trust = tcd.TrustCentralityAccumulator.from_state_dict(state["trust_accumulator"])
    # No further images to absorb -- checkpointing at the final image and
    # "resuming" must reproduce exactly the uninterrupted summary with zero
    # additional work.
    assert resumed_t4.summary() == t4_acc.summary()
    assert resumed_trust.image_ids() == trust_acc.image_ids()
    assert _metrics_equal(
        resumed_metric.finalize(scope=h.RUN_MODE_FULL, complete=True),
        metric_acc.finalize(scope=h.RUN_MODE_FULL, complete=True),
    )
    for cache in caches:
        cache.close()


@requires_mmseg
def test_interruption_before_any_t4_record_then_resume_matches_uninterrupted():
    # Interrupt after 0 images (a checkpoint taken before any record ever
    # existed) -- resuming from it must be equivalent to a fresh start.
    image_ids = [f"img{i}" for i in range(2)]
    uninterrupted_metric, uninterrupted_t4, uninterrupted_trust, _, _, uninterrupted_caches = _run_pipeline(image_ids)

    empty_metric = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    empty_t4 = t4.T4AuditAccumulator()
    empty_trust = tcd.TrustCentralityAccumulator()
    empty_primary = h.PrimarySolverSummaryAccumulator()
    state = _full_accumulator_state(
        metric_acc=empty_metric, t4_acc=empty_t4, trust_acc=empty_trust,
        primary_acc=empty_primary, parity_solver={}, windows_processed=0,
    )
    h.validate_accumulator_state_complete(state)
    resumed_t4 = t4.T4AuditAccumulator.from_state_dict(state["t4_accumulator"])
    resumed_trust = tcd.TrustCentralityAccumulator.from_state_dict(state["trust_accumulator"])
    resumed_metric = h.StreamingSegmentationMetricAccumulator.from_state_dict(state["metric_accumulator"])

    resumed_metric2, resumed_t42, resumed_trust2 = resumed_metric, resumed_t4, resumed_trust
    for image_id in image_ids:
        cache, context = _reversal_cache()
        signal = t4.build_t4_signal_for_image(cache, context, image_id=image_id)
        gt2d = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
        evaluation = t4.evaluate_t4_signal_against_gt(signal, gt2d, ignore_label=255, image_size=context.image_size)
        resumed_t42.absorb_signal(signal)
        resumed_t42.absorb_gt_evaluation(evaluation)
        stats = tcd.build_image_diagnostic_statistics(
            signal, gt2d, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
        )
        resumed_trust2.absorb_image(stats)
        resumed_metric2.absorb(image_id, torch.zeros(3, 3, dtype=torch.int64), gt2d)
        cache.close()

    assert resumed_t42.summary() == uninterrupted_t4.summary()
    assert resumed_trust2.image_ids() == uninterrupted_trust.image_ids()
    for cache in uninterrupted_caches:
        cache.close()


@requires_mmseg
def test_multiple_resume_cycles_reproduce_the_uninterrupted_summary():
    # Interrupt-resume twice (after image 1, then again after image 3) --
    # the final summary must still match a single uninterrupted run.
    image_ids = [f"img{i}" for i in range(5)]
    uninterrupted_metric, uninterrupted_t4, uninterrupted_trust, _, _, uninterrupted_caches = _run_pipeline(image_ids)

    t4_acc = t4.T4AuditAccumulator()
    trust_acc = tcd.TrustCentralityAccumulator()
    metric_acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    caches = []
    for cut in (1, 3, 5):
        already = len(trust_acc.image_ids())
        for image_id in image_ids[already:cut]:
            cache, context = _reversal_cache()
            caches.append(cache)
            signal = t4.build_t4_signal_for_image(cache, context, image_id=image_id)
            gt2d = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
            evaluation = t4.evaluate_t4_signal_against_gt(signal, gt2d, ignore_label=255, image_size=context.image_size)
            t4_acc.absorb_signal(signal)
            t4_acc.absorb_gt_evaluation(evaluation)
            stats = tcd.build_image_diagnostic_statistics(
                signal, gt2d, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
            )
            trust_acc.absorb_image(stats)
            metric_acc.absorb(image_id, torch.zeros(3, 3, dtype=torch.int64), gt2d)
        # "checkpoint and resume": round-trip every accumulator's state_dict
        t4_acc = t4.T4AuditAccumulator.from_state_dict(t4_acc.state_dict())
        trust_acc = tcd.TrustCentralityAccumulator.from_state_dict(trust_acc.state_dict())
        metric_acc = h.StreamingSegmentationMetricAccumulator.from_state_dict(metric_acc.state_dict())

    assert t4_acc.summary() == uninterrupted_t4.summary()
    assert trust_acc.image_ids() == uninterrupted_trust.image_ids()
    assert _metrics_equal(
        metric_acc.finalize(scope=h.RUN_MODE_FULL, complete=True),
        uninterrupted_metric.finalize(scope=h.RUN_MODE_FULL, complete=True),
    )
    for cache in caches + uninterrupted_caches:
        cache.close()


# ---------------------------------------------------------------------------
# 15-16: parity scheduling and honest coverage reporting
# ---------------------------------------------------------------------------


def test_15_first_periodic_and_final_parity_images_are_checked():
    indices = h.parity_check_indices(5000, 25)
    assert 0 in indices
    assert 25 in indices
    assert 4999 in indices  # final image, even though 4999 is not a multiple of 25
    assert len(indices) == len(range(0, 5000, 25)) + 1  # +1 for the final-image addition


def test_15b_interval_validated_as_positive_exact_integer():
    for bad in (0, -1, 1.5, True, "25"):
        with pytest.raises(h.TrustCentralityHarnessError):
            h.parity_check_indices(5000, bad)


def test_16_subset_parity_is_not_labeled_all_dataset_parity():
    evidence = h.build_parity_evidence(
        checked_image_ids=["img0", "img25", "img4999"], total_images=5000, mismatches=[],
        final_pass2_checked=5000, final_pass2_total=5000, final_pass2_all_identical=True,
    )
    assert evidence.checked_image_count == 3
    assert evidence.total_images == 5000
    assert evidence.coverage_fraction == pytest.approx(3 / 5000)
    assert evidence.coverage_fraction < 1.0  # never silently rounds up to "all"


def test_16b_fabricated_full_coverage_fails_closed():
    with pytest.raises(h.TrustCentralityHarnessError):
        h.ParityEvidence(
            checked_image_count=3, checked_image_ids=("a", "b", "c"), total_images=5000,
            coverage_fraction=1.0,  # inconsistent with 3/5000
            all_checked_identical=True, first_mismatch_image_id=None,
            final_pass2_vs_pass1_checked_count=5000, final_pass2_vs_pass1_total_count=5000,
            final_pass2_vs_pass1_all_identical=True,
        )


# ---------------------------------------------------------------------------
# 17-18: E3/RWR identity attribution, loaded from the authoritative TOMLs
# ---------------------------------------------------------------------------


def test_17_18_e3_and_rwr_reference_identities_not_confused():
    rwr_ref = h.load_rwr_reference_metrics(repo_root=ROOT)
    e3_ref = h.load_e3_reference_metrics(repo_root=ROOT)

    assert rwr_ref.source == "rwr"
    assert e3_ref.source == "e3"
    # Independently reload the raw TOMLs (not via the harness's own loader)
    # so the test itself never hardcodes the expected numeric values.
    with (ROOT / "evaluation_identities/e3_canonical_directed_rwr.toml").open("rb") as fh:
        rwr_toml = tomllib.load(fh)
    with (ROOT / "evaluation_identities/e3_paired_soft_routing.toml").open("rb") as fh:
        e3_toml = tomllib.load(fh)

    assert rwr_ref.aAcc == pytest.approx(rwr_toml["expected_metrics"]["rwr_aAcc"])
    assert rwr_ref.mIoU == pytest.approx(rwr_toml["expected_metrics"]["rwr_mIoU"])
    assert rwr_ref.mAcc == pytest.approx(rwr_toml["expected_metrics"]["rwr_mAcc"])
    assert e3_ref.aAcc == pytest.approx(e3_toml["expected_metrics"]["aAcc"])
    assert e3_ref.mIoU == pytest.approx(e3_toml["expected_metrics"]["mIoU"])
    assert e3_ref.mAcc == pytest.approx(e3_toml["expected_metrics"]["mAcc"])

    # The two identities' metrics must be genuinely distinct numbers --
    # if a future refactor accidentally aliased the two loaders, this
    # would catch it immediately.
    assert rwr_ref.mIoU != e3_ref.mIoU
    assert rwr_ref.aAcc != e3_ref.aAcc
    assert rwr_ref.mAcc != e3_ref.mAcc

    comparison = h.compare_to_reference(rwr_ref.mIoU, rwr_ref)
    assert comparison.status == "pass"
    assert comparison.delta == pytest.approx(0.0, abs=1e-9)

    # Comparing the RWR-observed number against the E3 reference must NOT
    # silently pass -- this is exactly the E3/RWR confusion this task exists
    # to prevent.
    mismatched = h.compare_to_reference(rwr_ref.mIoU, e3_ref)
    assert mismatched.status == "fail"


def test_18b_reference_metrics_never_placed_under_observed_type():
    rwr_ref = h.load_rwr_reference_metrics(repo_root=ROOT)
    assert not isinstance(rwr_ref, h.ObservedRunMetrics)
    assert isinstance(rwr_ref, h.CanonicalReferenceMetrics)


# ---------------------------------------------------------------------------
# 19-20: production invariance
# ---------------------------------------------------------------------------


def test_19_no_model_graph_cgls_execution_introduced():
    import ast

    tree = ast.parse(HARNESS_PATH.read_text())
    forbidden_modules = {"models", "models.dinotext", "models.dinotext.cover_dr"}
    forbidden_ids = {"DirectedTopKGraph", "cgls_solve", "rwr_solve", "reconstruct_directed_topk_graph", "generate_patch_snapshot"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden_ids
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_ids


@requires_mmseg
def test_20_diagnostic_enabled_predictions_remain_unchanged_by_absorb():
    pred = torch.tensor([[0, 1], [2, 0]])
    gt = torch.tensor([[0, 1], [2, 0]])
    pred_before = pred.clone()
    gt_before = gt.clone()
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    acc.absorb("img0", pred, gt)
    assert torch.equal(pred, pred_before)
    assert torch.equal(gt, gt_before)


# ---------------------------------------------------------------------------
# Additional fail-closed / validation coverage
# ---------------------------------------------------------------------------


@requires_mmseg
def test_shape_mismatch_fails_closed():
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=2, ignore_index=255)
    with pytest.raises(h.TrustCentralityHarnessError):
        acc.absorb("img0", torch.zeros(2, 2, dtype=torch.int64), torch.zeros(3, 3, dtype=torch.int64))


@requires_mmseg
def test_nonfinite_prediction_fails_closed():
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=2, ignore_index=255)
    bad = torch.tensor([[0.0, float("nan")], [1.0, 0.0]])
    with pytest.raises(h.TrustCentralityHarnessError):
        acc.absorb("img0", bad, torch.zeros(2, 2, dtype=torch.int64))


@requires_mmseg
def test_synthetic_full_dataset_run_reaches_natural_metric_finalization_without_stop_early():
    """End-to-end synthetic simulation of a full-dataset run: a small
    synthetic 'dataset' of 6 images is processed through the exact same
    determine_run_mode / should_stop_after_image / accumulator sequence
    the real pilot script uses, and must NEVER trigger a stop-early
    condition, reaching a complete, populated ObservedRunMetrics at the
    end -- proving natural-completion behavior without needing a GPU."""
    dataset_length = 6
    image_limit = None  # absent limit -> full_dataset, matching the real pilot's default
    mode = h.determine_run_mode(image_limit, dataset_length)
    assert mode == h.RUN_MODE_FULL

    acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    processed_ids = []
    torch.manual_seed(0)
    stopped_early = False
    for index in range(dataset_length):
        image_id = str(index)
        pred = torch.randint(0, 3, (4, 4))
        gt = torch.randint(0, 3, (4, 4))
        acc.absorb(image_id, pred, gt)
        processed_ids.append(image_id)
        if h.should_stop_after_image(mode=mode, images_processed=index + 1, image_limit=image_limit, window_floor_satisfied=True):
            stopped_early = True
            break

    assert stopped_early is False, "full_dataset mode must never trigger the early-stop condition"
    assert h.is_dataset_fully_covered(processed_ids, dataset_length) is True
    result = acc.finalize(scope=mode, complete=True)
    assert result.complete is True
    assert result.evaluated_images == dataset_length
    assert result.mIoU is not None and result.aAcc is not None and result.mAcc is not None


def test_synthetic_partial_run_stops_early_and_never_claims_full_dataset_metrics():
    dataset_length = 6
    image_limit = 3
    mode = h.determine_run_mode(image_limit, dataset_length)
    assert mode == h.RUN_MODE_PARTIAL

    processed_ids = []
    stopped_early = False
    for index in range(dataset_length):
        processed_ids.append(str(index))
        if h.should_stop_after_image(mode=mode, images_processed=index + 1, image_limit=image_limit, window_floor_satisfied=True):
            stopped_early = True
            break

    assert stopped_early is True
    assert len(processed_ids) == image_limit
    assert h.is_dataset_fully_covered(processed_ids, dataset_length) is False
    result = h.unavailable_observed_metrics(
        scope=mode, evaluated_images=len(processed_ids), classes=3,
        reason=f"partial run: {len(processed_ids)} of {dataset_length} images processed",
    )
    assert result.mIoU is None  # never a misleading partial mIoU


def test_run_mode_rejects_wrong_types():
    with pytest.raises(h.TrustCentralityHarnessError):
        h.determine_run_mode(1.5, 5000)
    with pytest.raises(h.TrustCentralityHarnessError):
        h.determine_run_mode(True, 5000)
    with pytest.raises(h.TrustCentralityHarnessError):
        h.determine_run_mode(100, 0)


# ---------------------------------------------------------------------------
# Primary vs. parity solver-phase accounting (section 5 / section 12)
# ---------------------------------------------------------------------------


def _telemetry_summary(window_count, total_iterations=10, min_it=2, max_it=5, restarts=0, nonzero_restart=0, replacements=0, fallback=0, max_residual=0.5):
    return wc.WindowSolverTelemetrySummary(
        window_count=window_count, total_iterations=total_iterations, minimum_iterations=min_it,
        maximum_iterations=max_it, total_restarts=restarts, nonzero_restart_windows=nonzero_restart,
        total_residual_replacements=replacements, total_fallback_rows=fallback, maximum_scaled_residual=max_residual,
    )


def test_primary_solver_every_window_counted_once():
    acc = h.PrimarySolverSummaryAccumulator()
    acc.absorb(_telemetry_summary(window_count=3, total_iterations=30, min_it=8, max_it=12))
    acc.absorb(_telemetry_summary(window_count=5, total_iterations=60, min_it=5, max_it=20))
    result = acc.as_dict()
    assert result["window_count"] == 8
    assert result["total_iterations"] == 90
    assert result["minimum_iterations"] == 5
    assert result["maximum_iterations"] == 20
    assert result["phase"] == "primary_inference"
    assert result["images_contributing"] == 2


def test_primary_solver_handles_zero_window_images_without_polluting_min():
    acc = h.PrimarySolverSummaryAccumulator()
    acc.absorb(_telemetry_summary(window_count=0, total_iterations=0, min_it=0, max_it=0))
    acc.absorb(_telemetry_summary(window_count=2, total_iterations=10, min_it=3, max_it=7))
    result = acc.as_dict()
    assert result["window_count"] == 2
    assert result["minimum_iterations"] == 3  # not polluted by the zero-window image's min_it=0
    assert result["images_contributing"] == 2


def test_parity_solver_labeled_separately_never_as_primary():
    parity = h.build_parity_solver_summary({
        "window_count": 452, "converged_window_count": 452, "total_iterations": 251562,
        "minimum_iterations": 385, "maximum_iterations": 1006, "total_restarts": 248107,
        "nonzero_restart_windows": 452, "total_residual_replacements": 248107,
        "total_fallback_rows": 0, "maximum_scaled_residual": 0.9999985957485961,
    })
    assert parity["phase"] == "parity_check"
    assert parity["window_count"] == 452


def test_primary_solver_count_must_equal_windows_processed():
    acc = h.PrimarySolverSummaryAccumulator()
    acc.absorb(_telemetry_summary(window_count=11075))
    h.validate_solver_phase_separation(primary=acc.as_dict(), windows_processed=11075)  # passes
    with pytest.raises(h.TrustCentralityHarnessError):
        h.validate_solver_phase_separation(primary=acc.as_dict(), windows_processed=452)


def test_pass_two_never_recorded_in_primary_solver_summary():
    # By construction: PrimarySolverSummaryAccumulator only ever absorbs
    # FirstPassImageContext.pass_summary (produced once, during pass ONE);
    # nothing in this module's API accepts a pass-2 telemetry object at
    # all, so pass-2 cannot add solves structurally.
    import inspect
    sig = inspect.signature(h.PrimarySolverSummaryAccumulator.absorb)
    assert list(sig.parameters) == ["self", "pass_summary"]


def test_bounded_and_resumed_runs_preserve_solver_counts():
    acc_uninterrupted = h.PrimarySolverSummaryAccumulator()
    for wcnt in (3, 5, 2, 7):
        acc_uninterrupted.absorb(_telemetry_summary(window_count=wcnt, total_iterations=wcnt * 10))
    uninterrupted = acc_uninterrupted.as_dict()

    acc_resumed = h.PrimarySolverSummaryAccumulator()
    for wcnt in (3, 5):
        acc_resumed.absorb(_telemetry_summary(window_count=wcnt, total_iterations=wcnt * 10))
    # simulate checkpoint/restore by copying fields (mirrors the pilot
    # script's own checkpoint round-trip of these exact fields)
    restored = h.PrimarySolverSummaryAccumulator()
    for field_name in ("window_count", "total_iterations", "minimum_iterations", "maximum_iterations",
                       "total_restarts", "nonzero_restart_windows", "total_residual_replacements",
                       "total_fallback_rows", "maximum_scaled_residual"):
        setattr(restored, field_name, getattr(acc_resumed, field_name))
    restored._images = acc_resumed._images
    for wcnt in (2, 7):
        restored.absorb(_telemetry_summary(window_count=wcnt, total_iterations=wcnt * 10))

    assert restored.as_dict() == uninterrupted


def test_instrumentation_does_not_alter_solver_output():
    # The accumulator receives an already-immutable dataclass and never
    # mutates it or anything it references.
    summary = _telemetry_summary(window_count=4, total_iterations=40)
    before = (summary.window_count, summary.total_iterations, summary.maximum_scaled_residual)
    acc = h.PrimarySolverSummaryAccumulator()
    acc.absorb(summary)
    after = (summary.window_count, summary.total_iterations, summary.maximum_scaled_residual)
    assert before == after


# ---------------------------------------------------------------------------
# Gain-over-E3 and natural-vs-streaming reconciliation (sections 2 & 4)
# ---------------------------------------------------------------------------


def test_gain_over_e3_uses_full_precision_operands():
    e3_ref = h.load_e3_reference_metrics(repo_root=ROOT)
    gain = h.compute_gain_over_e3(29.87819736908178, e3_ref)
    assert gain.calculation_precision == "full"
    assert gain.rwr_observed_full_precision == 29.87819736908178
    assert gain.e3_reference_full_precision == e3_ref.mIoU
    assert gain.absolute_gain == pytest.approx(29.87819736908178 - e3_ref.mIoU, abs=1e-12)


def test_gain_over_e3_rejects_rwr_reference_by_mistake():
    rwr_ref = h.load_rwr_reference_metrics(repo_root=ROOT)
    with pytest.raises(h.TrustCentralityHarnessError):
        h.compute_gain_over_e3(29.878, rwr_ref)  # wrong source type -- must be E3


def test_gain_arithmetic_reconciles_exactly():
    e3_ref = h.load_e3_reference_metrics(repo_root=ROOT)
    with pytest.raises(h.TrustCentralityHarnessError):
        h.GainOverE3(
            metric="mIoU", rwr_observed_full_precision=29.878197, e3_reference_full_precision=e3_ref.mIoU,
            absolute_gain=999.0,  # deliberately wrong
            calculation_precision="full",
        )


@requires_mmseg
def test_natural_vs_streaming_reconciliation_passes_when_rounded_values_agree():
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    acc.absorb("img0", torch.tensor([[0, 1], [2, 0]]), torch.tensor([[0, 1], [2, 0]]))
    observed = acc.finalize(scope=h.RUN_MODE_FULL, complete=True)
    natural = {"aAcc": round(observed.aAcc, 2), "mIoU": round(observed.mIoU, 2), "mAcc": round(observed.mAcc, 2)}
    h.reconcile_natural_vs_streaming(natural, observed)  # must not raise


@requires_mmseg
def test_natural_vs_streaming_reconciliation_fails_on_disagreement():
    acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    acc.absorb("img0", torch.tensor([[0, 1], [2, 0]]), torch.tensor([[0, 1], [2, 0]]))
    observed = acc.finalize(scope=h.RUN_MODE_FULL, complete=True)
    natural = {"aAcc": 12.34, "mIoU": 56.78, "mAcc": 90.12}  # deliberately wrong
    with pytest.raises(h.TrustCentralityHarnessError):
        h.reconcile_natural_vs_streaming(natural, observed)


def test_reconciliation_requires_complete_streaming_metrics():
    incomplete = h.unavailable_observed_metrics(scope=h.RUN_MODE_PARTIAL, evaluated_images=1, classes=3, reason="partial")
    with pytest.raises(h.TrustCentralityHarnessError):
        h.reconcile_natural_vs_streaming({"aAcc": 1.0, "mIoU": 1.0, "mAcc": 1.0}, incomplete)


# ---------------------------------------------------------------------------
# dataset.evaluate() capture-wrapper semantics (section 10, tested against
# a plain mock -- exercises the EXACT wrapper pattern used in the pilot
# script's patched_multi_gpu_test without needing mmseg/CUDA)
# ---------------------------------------------------------------------------


def _install_capturing_wrapper(real_dataset, sink: dict):
    """Mirrors patched_multi_gpu_test's own capturing_evaluate exactly."""
    original_evaluate = real_dataset.evaluate

    def capturing_evaluate(results, metric="mIoU", logger=None, **kw):
        try:
            result = original_evaluate(results, metric=metric, logger=logger, **kw)
        finally:
            real_dataset.evaluate = original_evaluate
        sink["captured"] = result
        return result

    real_dataset.evaluate = capturing_evaluate
    return original_evaluate


class _FakeDataset:
    def __init__(self):
        self.calls = 0

    def evaluate(self, results, metric="mIoU", logger=None):
        self.calls += 1
        return {"aAcc": 48.53, "mIoU": 29.88, "mAcc": 54.14}


class _FailingDataset:
    def evaluate(self, results, metric="mIoU", logger=None):
        raise ValueError("boom")


def test_capture_installed_early_remains_present_until_the_real_call():
    dataset = _FakeDataset()
    original = dataset.evaluate
    sink: dict = {}
    _install_capturing_wrapper(dataset, sink)
    assert dataset.evaluate is not original  # still installed immediately after "multi_gpu_test" returns
    result = dataset.evaluate(["r"], metric="mIoU", logger=None)  # the later, real call
    assert sink["captured"] == result


def test_original_evaluator_called_exactly_once():
    dataset = _FakeDataset()
    sink: dict = {}
    _install_capturing_wrapper(dataset, sink)
    dataset.evaluate(["r"])
    assert dataset.calls == 1


def test_original_return_value_unchanged():
    dataset = _FakeDataset()
    sink: dict = {}
    _install_capturing_wrapper(dataset, sink)
    result = dataset.evaluate(["r"])
    assert result == {"aAcc": 48.53, "mIoU": 29.88, "mAcc": 54.14}
    assert result is sink["captured"]


def test_capture_populated_after_call():
    dataset = _FakeDataset()
    sink: dict = {}
    _install_capturing_wrapper(dataset, sink)
    assert "captured" not in sink
    dataset.evaluate(["r"])
    assert "captured" in sink


def test_original_restored_after_success():
    dataset = _FakeDataset()
    original = dataset.evaluate
    sink: dict = {}
    _install_capturing_wrapper(dataset, sink)
    dataset.evaluate(["r"])
    assert dataset.evaluate == original


def test_original_restored_after_exception():
    dataset = _FailingDataset()
    original = dataset.evaluate
    sink: dict = {}
    _install_capturing_wrapper(dataset, sink)
    with pytest.raises(ValueError):
        dataset.evaluate(["r"])
    assert dataset.evaluate == original
    assert "captured" not in sink  # nothing recorded on the failure path


def test_second_evaluation_uses_the_original_method_not_the_wrapper():
    dataset = _FakeDataset()
    sink: dict = {}
    _install_capturing_wrapper(dataset, sink)
    dataset.evaluate(["r"])  # first call: captured, self-restores
    sink.clear()
    dataset.evaluate(["r"])  # second call: hits the original directly
    assert dataset.calls == 2
    assert "captured" not in sink  # the wrapper is gone; nothing captures the second call


# ---------------------------------------------------------------------------
# Complete trust/centrality report schema (section 6 / section 13)
# ---------------------------------------------------------------------------


def _trivial_graph(n, k=1):
    return wc.GraphSnapshot(
        torch.zeros(n, k, dtype=torch.int64), torch.full((n, k), 1.0 / k), torch.ones(n, k),
        torch.zeros(n, dtype=torch.bool), n, k, 1.0,
    )


def _telemetry():
    return wc.WindowSolverTelemetry(1, 1, 0, 0, 0, 0.0)


def _reversal_cache():
    plan = geometry.SlidingWindowPlan.build(image_size=SpatialSize(3, 3), crop_size=SpatialSize(2, 2), stride=SpatialSize(1, 1))
    cache = wc.ImageWindowCache(plan)
    node_of = {0: 3, 1: 2, 2: 1, 3: 0}
    specs = {
        0: (torch.tensor([3.0, 1.0, 0.0]), torch.tensor([-5.0, 5.0, -5.0])),
        1: (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.5, -0.5, -0.5])),
        2: (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.5, -0.5, -0.5])),
        3: (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.5, -0.5, -0.5])),
    }

    def full(local, cc, vec):
        base = torch.zeros(4, cc)
        for i in range(4):
            base[i, 0] = 100.0
        base[local] = vec
        return base

    for w in plan.windows:
        s0v, pv = specs[w.index]
        cache.append(wc.CachedWindowState(
            geometry=w, window_index=w.index, patch_grid_shape=(2, 2), class_count=3,
            s0=full(node_of[w.index], 3, s0v), dino_features=torch.zeros(4, 3), graph=_trivial_graph(4),
            propagated_scores=full(node_of[w.index], 3, pv), solver_summary=_telemetry(),
        ))
    cache.seal()
    context = wc.FirstPassImageContext(
        image_size=SpatialSize(3, 3), plan=plan, class_count=3, common_patch_grid_shape=(2, 2),
        expected_window_count=4, cached_window_count=4, min_coverage=1, max_coverage=4,
        stitched_scores=torch.zeros(1, 3, 3, 3), cache_total_bytes=1,
        pass_summary=wc._summarize_telemetry([_telemetry()] * 4),
    )
    return cache, context


@requires_mmseg
def test_full_report_schema_populated_from_hand_enumerated_fixture():
    t4_acc = t4.T4AuditAccumulator()
    trust_acc = tcd.TrustCentralityAccumulator()
    stage_image_counts = {stage: 0 for stage in ("t1", "t2", "t3", "actionable", "t4", "t4_prime")}
    for i in range(3):
        cache, context = _reversal_cache()
        signal = t4.build_t4_signal_for_image(cache, context, image_id=f"img{i}")
        gt = torch.zeros(3, 3, dtype=torch.int64)
        evaluation = t4.evaluate_t4_signal_against_gt(signal, gt, ignore_label=255, image_size=context.image_size)
        t4_acc.absorb_signal(signal)
        t4_acc.absorb_gt_evaluation(evaluation)
        stats = tcd.build_image_diagnostic_statistics(
            signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
        )
        trust_acc.absorb_image(stats)
        fc = signal.funnel_counts
        for stage in stage_image_counts:
            if getattr(fc, stage) > 0:
                stage_image_counts[stage] += 1
        cache.close()

    t4_summary = t4_acc.summary()
    trust_report = trust_acc.summary(settings=tcd.BootstrapSettings(resamples=500, seed=7))
    section = h.build_full_trust_centrality_section(
        t4_summary=t4_summary, trust_report=trust_report, stage_image_counts=stage_image_counts, images_processed=3,
    )

    # 6.1: funnel + subset invariant, independently re-verified
    fc = t4_summary.funnel_counts
    assert fc.t4 <= fc.actionable <= fc.t3 <= fc.t2 <= fc.t1 <= fc.t0
    assert section["funnel"]["t4"]["count"] == fc.t4 == 3
    assert section["funnel"]["t4"]["contributing_images"] == 3

    # 6.2: per-stage trust, hand-reconciled
    t4_trust = section["trust_by_stage"]["t4"]
    assert t4_trust["consensus_correct"] == 3 and t4_trust["dissent_correct"] == 0
    assert t4_trust["delta_trust"] == pytest.approx((3 - 0) / 3)

    # 6.3: bootstrap validity present for every registered population
    # 1 (t4) + 9 (3 stages x 3 strata) + 30 (3 stages x 10 fixed-width bins)
    # + 12 (3 stages x 4 edge bands) + 4 (shared-unary)
    assert len(section["bootstrap"]) == 56
    t4_bootstrap = section["bootstrap"]["t4"]
    for required_key in ("bootstrap_unit", "resamples_requested", "valid_replicate_count",
                        "invalid_replicate_count", "invalid_replicate_fraction", "seed",
                        "confidence_level", "point_estimate", "ci_low", "ci_high",
                        "contributing_images", "zero_target_images"):
        assert required_key in t4_bootstrap

    # 6.4/6.5: centrality present for all 3 extended stages, not just T4
    assert set(section["centrality"].keys()) == {"actionable", "t4", "t4_prime"}
    for stage in ("actionable", "t4", "t4_prime"):
        assert set(section["centrality"][stage]["strata"].keys()) <= {"negative", "zero", "positive"}

    # 6.6: crop-edge bands present for all 3 stages
    assert set(section["crop_edge"].keys()) == {"actionable", "t4", "t4_prime"}

    # 6.7: shared-unary strata
    assert "all" in section["shared_unary"]

    # 6.8: per-class + concentration
    assert "0" in section["per_class_t4"]
    assert section["class_concentration"]["top_1_fraction"] == pytest.approx(1.0)

    # 6.9: stitched-label categories, g==y forced to zero by construction
    assert section["stitched_label_categories"]["g_equals_y_count"] == 0

    # section 7: interpretation label present and deterministic
    assert section["trust_interpretation_t4"] in ("SUPPORTED", "CONTRADICTED", "INCONCLUSIVE", "UNAVAILABLE")

    import json
    json.dumps(section, default=str)  # must be fully JSON-serializable


@requires_mmseg
def test_schema_missing_required_section_is_detectable():
    t4_acc = t4.T4AuditAccumulator()
    trust_acc = tcd.TrustCentralityAccumulator()
    cache, context = _reversal_cache()
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img0")
    gt = torch.zeros(3, 3, dtype=torch.int64)
    evaluation = t4.evaluate_t4_signal_against_gt(signal, gt, ignore_label=255, image_size=context.image_size)
    t4_acc.absorb_signal(signal)
    t4_acc.absorb_gt_evaluation(evaluation)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    trust_acc.absorb_image(stats)
    cache.close()
    t4_summary = t4_acc.summary()
    trust_report = trust_acc.summary(settings=tcd.BootstrapSettings(resamples=200, seed=1))
    section = h.build_full_trust_centrality_section(
        t4_summary=t4_summary, trust_report=trust_report, stage_image_counts={"t4": 1}, images_processed=1,
    )
    required_top_level = (
        "funnel", "trust_by_stage", "bootstrap", "centrality", "crop_edge", "shared_unary",
        "per_class_t4", "stitched_label_categories", "trust_interpretation_t4",
    )
    for key in required_top_level:
        assert key in section, f"required section {key!r} missing from schema"

    # validate_trust_centrality_section_complete must accept this
    # genuinely complete section...
    h.validate_trust_centrality_section_complete(section)
    # ...and must fail closed when any required top-level or nested key is
    # absent -- a report marked final: true must never accept a section
    # missing something this checks for.
    for key in h.REQUIRED_TRUST_CENTRALITY_SECTION_KEYS:
        broken = dict(section)
        del broken[key]
        with pytest.raises(h.TrustCentralityHarnessError):
            h.validate_trust_centrality_section_complete(broken)

    broken_funnel = {**section, "funnel": {k: v for k, v in section["funnel"].items() if k != "t4"}}
    with pytest.raises(h.TrustCentralityHarnessError):
        h.validate_trust_centrality_section_complete(broken_funnel)

    broken_centrality = {**section, "centrality": {k: v for k, v in section["centrality"].items() if k != "t4"}}
    with pytest.raises(h.TrustCentralityHarnessError):
        h.validate_trust_centrality_section_complete(broken_centrality)

    stage_without_bins = dict(section["centrality"]["t4"])
    del stage_without_bins["bins"]
    broken_bins = {**section, "centrality": {**section["centrality"], "t4": stage_without_bins}}
    with pytest.raises(h.TrustCentralityHarnessError):
        h.validate_trust_centrality_section_complete(broken_bins)


@requires_mmseg
def test_every_fixed_width_centrality_bin_boundary_is_present_and_ordered():
    t4_acc = t4.T4AuditAccumulator()
    trust_acc = tcd.TrustCentralityAccumulator()
    cache, context = _reversal_cache()
    signal = t4.build_t4_signal_for_image(cache, context, image_id="img0")
    gt = torch.zeros(3, 3, dtype=torch.int64)
    evaluation = t4.evaluate_t4_signal_against_gt(signal, gt, ignore_label=255, image_size=context.image_size)
    t4_acc.absorb_signal(signal)
    t4_acc.absorb_gt_evaluation(evaluation)
    stats = tcd.build_image_diagnostic_statistics(
        signal, gt, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
    )
    trust_acc.absorb_image(stats)
    cache.close()
    t4_summary = t4_acc.summary()
    trust_report = trust_acc.summary(settings=tcd.BootstrapSettings(resamples=200, seed=1))
    section = h.build_full_trust_centrality_section(
        t4_summary=t4_summary, trust_report=trust_report, stage_image_counts={"t4": 1}, images_processed=1,
    )
    for stage in ("actionable", "t4", "t4_prime"):
        bins = section["centrality"][stage]["bins"]
        edges = section["centrality"][stage]["bin_edges"]
        assert len(bins) == len(edges) - 1 == tcd.DELTA_C_BIN_COUNT
        for bin_idx, entry in enumerate(bins):
            assert entry["bin_index"] == bin_idx
            assert entry["bin_lower"] == edges[bin_idx]
            assert entry["bin_upper"] == edges[bin_idx + 1]
            assert entry["bin_lower"] < entry["bin_upper"]
            for required_key in ("observations", "consensus_correct", "dissent_correct", "delta_trust",
                                  "third_label_fraction", "point_estimate", "ci_low", "ci_high",
                                  "valid_replicate_count", "invalid_replicate_count", "zero_target_images"):
                assert required_key in entry
        # 30 total bin populations = 3 stages x 10 bins; each stage's own
        # bin observation total must equal that stage's Delta_c pool size.
        assert sum(entry["observations"] for entry in bins) == section["centrality"][stage]["delta_c"]["count"]


def test_build_full_report_requires_gt_evaluated_summary():
    class _Fake:
        gt_evaluated = False

    with pytest.raises(h.TrustCentralityHarnessError):
        h.build_full_trust_centrality_section(t4_summary=_Fake(), trust_report=None, stage_image_counts={}, images_processed=0)


# ---------------------------------------------------------------------------
# Checkpoint v2: exact per-image sufficient statistics, deterministic
# resume identity, and fail-closed rejection (Sections 2/3/4/5 of the
# checkpoint/resume repair)
# ---------------------------------------------------------------------------


def _metrics_equal(a, b) -> bool:
    """Structural equality for ObservedRunMetrics that treats NaN as equal
    to NaN in per_class_iou/per_class_acc (an unrepresented class's IoU is
    legitimately NaN; two runs with the same NaN pattern are identical,
    but Python's float NaN != NaN would otherwise make dataclass ``==``
    always report a false mismatch)."""
    import math
    if type(a) is not type(b):
        return False
    for field_name in a.__dataclass_fields__:
        av, bv = getattr(a, field_name), getattr(b, field_name)
        if isinstance(av, tuple) and isinstance(bv, tuple):
            if len(av) != len(bv):
                return False
            for x, y in zip(av, bv):
                same = (x == y) or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
                if not same:
                    return False
        elif av != bv:
            return False
    return True


def _full_accumulator_state(*, metric_acc, t4_acc, trust_acc, primary_acc, parity_solver: dict, windows_processed: int) -> dict:
    return {
        "metric_accumulator": metric_acc.state_dict(),
        "t4_accumulator": t4_acc.state_dict(),
        "trust_accumulator": trust_acc.state_dict(),
        "primary_solver": primary_acc.state_dict(),
        "parity_solver": parity_solver,
        "windows_processed": windows_processed,
    }


def _run_pipeline(image_ids):
    """Processes ``image_ids`` through fresh metric/t4/trust/primary-solver
    accumulators from scratch and returns them plus windows_processed."""
    metric_acc = h.StreamingSegmentationMetricAccumulator(num_classes=3, ignore_index=255)
    t4_acc = t4.T4AuditAccumulator()
    trust_acc = tcd.TrustCentralityAccumulator()
    primary_acc = h.PrimarySolverSummaryAccumulator()
    windows_processed = 0
    caches = []
    for image_id in image_ids:
        cache, context = _reversal_cache()
        caches.append(cache)
        signal = t4.build_t4_signal_for_image(cache, context, image_id=image_id)
        gt2d = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
        evaluation = t4.evaluate_t4_signal_against_gt(signal, gt2d, ignore_label=255, image_size=context.image_size)
        t4_acc.absorb_signal(signal)
        t4_acc.absorb_gt_evaluation(evaluation)
        stats = tcd.build_image_diagnostic_statistics(
            signal, gt2d, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
        )
        trust_acc.absorb_image(stats)
        primary_acc.absorb(context.pass_summary)
        windows_processed += context.expected_window_count
        pred = torch.zeros(3, 3, dtype=torch.int64)
        metric_acc.absorb(image_id, pred, gt2d)
    return metric_acc, t4_acc, trust_acc, primary_acc, windows_processed, caches


def _build_report(*, metric_acc, t4_acc, trust_acc, images_processed, elapsed_seconds, gpu_model):
    t4_summary = t4_acc.summary()
    trust_report = trust_acc.summary(settings=tcd.BootstrapSettings(resamples=200, seed=3))
    stage_image_counts = {stage: images_processed for stage in ("t1", "t2", "t3", "actionable", "t4", "t4_prime")}
    full_section = h.build_full_trust_centrality_section(
        t4_summary=t4_summary, trust_report=trust_report, stage_image_counts=stage_image_counts, images_processed=images_processed,
    )
    metrics = metric_acc.finalize(scope=h.RUN_MODE_FULL, complete=True)
    return {
        "schema_version": "test-report-v1",
        "final": True,
        "complete": True,
        "run_mode": h.RUN_MODE_FULL,
        "images_processed": images_processed,
        "reconciliation_error": None,
        "observed_metrics": {
            "scope": metrics.scope, "complete": metrics.complete, "aAcc": metrics.aAcc,
            "mIoU": metrics.mIoU, "mAcc": metrics.mAcc, "evaluated_images": metrics.evaluated_images,
        },
        "natural_evaluate_result": {"captured": True, "aAcc": metrics.aAcc, "mIoU": metrics.mIoU, "mAcc": metrics.mAcc},
        "trust_centrality": {"status": "available", **full_section},
        "provenance": {
            "start_time_unix": 1000.0, "end_time_unix": 1000.0 + elapsed_seconds,
            "elapsed_seconds": elapsed_seconds, "gpu_model": gpu_model,
            "python_version": "3.11.0", "torch_version": "2.0.0", "cuda_version": "12.1",
        },
        "peak_gpu_bytes": 123456789,
        "median_diagnostic_seconds_per_image": elapsed_seconds / max(images_processed, 1),
    }


@requires_mmseg
def test_uninterrupted_vs_interrupted_resumed_scientific_projection_byte_identical():
    image_ids = [f"img{i}" for i in range(5)]

    metric_acc, t4_acc, trust_acc, primary_acc, windows_processed, caches = _run_pipeline(image_ids)
    uninterrupted_report = _build_report(
        metric_acc=metric_acc, t4_acc=t4_acc, trust_acc=trust_acc,
        images_processed=len(image_ids), elapsed_seconds=42.0, gpu_model="NVIDIA A100",
    )
    for cache in caches:
        cache.close()

    # Simulate interruption after 2 of 5 images: checkpoint, "restart the
    # process" (fresh accumulators reconstructed purely from state_dict),
    # then continue with the remaining images.
    first_metric, first_t4, first_trust, first_primary, first_windows, first_caches = _run_pipeline(image_ids[:2])
    checkpoint_state = _full_accumulator_state(
        metric_acc=first_metric, t4_acc=first_t4, trust_acc=first_trust,
        primary_acc=first_primary, parity_solver={}, windows_processed=first_windows,
    )
    h.validate_accumulator_state_complete(checkpoint_state)
    import json
    json.dumps(checkpoint_state)  # must be JSON-serializable

    resumed_metric = h.StreamingSegmentationMetricAccumulator.from_state_dict(checkpoint_state["metric_accumulator"])
    resumed_t4 = t4.T4AuditAccumulator.from_state_dict(checkpoint_state["t4_accumulator"])
    resumed_trust = tcd.TrustCentralityAccumulator.from_state_dict(checkpoint_state["trust_accumulator"])
    resumed_primary = h.PrimarySolverSummaryAccumulator.from_state_dict(checkpoint_state["primary_solver"])
    resumed_windows = checkpoint_state["windows_processed"]

    for cache in first_caches:
        cache.close()

    for image_id in image_ids[2:]:
        cache, context = _reversal_cache()
        signal = t4.build_t4_signal_for_image(cache, context, image_id=image_id)
        gt2d = torch.zeros(*context.image_size.as_tuple(), dtype=torch.int64)
        evaluation = t4.evaluate_t4_signal_against_gt(signal, gt2d, ignore_label=255, image_size=context.image_size)
        resumed_t4.absorb_signal(signal)
        resumed_t4.absorb_gt_evaluation(evaluation)
        stats = tcd.build_image_diagnostic_statistics(
            signal, gt2d, ignore_label=255, image_size=context.image_size, common_patch_grid_shape=context.common_patch_grid_shape,
        )
        resumed_trust.absorb_image(stats)
        resumed_primary.absorb(context.pass_summary)
        resumed_windows += context.expected_window_count
        pred = torch.zeros(3, 3, dtype=torch.int64)
        resumed_metric.absorb(image_id, pred, gt2d)
        cache.close()

    assert resumed_windows == windows_processed
    resumed_report = _build_report(
        metric_acc=resumed_metric, t4_acc=resumed_t4, trust_acc=resumed_trust,
        images_processed=len(image_ids), elapsed_seconds=999.0, gpu_model="NVIDIA H100",  # deliberately different
    )

    # The raw reports differ (volatile fields deliberately mismatched) --
    # sanity-check the test setup is not vacuous.
    assert uninterrupted_report["provenance"]["elapsed_seconds"] != resumed_report["provenance"]["elapsed_seconds"]
    assert uninterrupted_report != resumed_report

    # But their canonical scientific projections must be byte-identical.
    h.assert_scientific_projections_identical(uninterrupted_report, resumed_report)
    assert h.sha256_of_scientific_projection(uninterrupted_report) == h.sha256_of_scientific_projection(resumed_report)


def test_canonical_scientific_projection_strips_only_volatile_fields():
    report = {
        "final": True, "complete": True, "images_processed": 5,
        "observed_metrics": {"mIoU": 29.87},
        "provenance": {
            "start_time_unix": 1.0, "end_time_unix": 2.0, "elapsed_seconds": 1.0,
            "gpu_model": "A100", "python_version": "3.11", "torch_version": "2.0", "cuda_version": "12.1",
            "git_head": "abc123",
        },
        "peak_gpu_bytes": 999,
        "median_diagnostic_seconds_per_image": 0.2,
    }
    projection = h.canonical_scientific_projection(report)
    assert "peak_gpu_bytes" not in projection
    assert "median_diagnostic_seconds_per_image" not in projection
    for volatile_key in ("start_time_unix", "end_time_unix", "elapsed_seconds", "gpu_model", "python_version", "torch_version", "cuda_version"):
        assert volatile_key not in projection["provenance"]
    assert projection["provenance"]["git_head"] == "abc123"  # non-volatile field preserved
    assert projection["observed_metrics"]["mIoU"] == 29.87
    assert projection["images_processed"] == 5
    # original report is untouched
    assert "peak_gpu_bytes" in report


def test_assert_scientific_projections_identical_raises_on_real_difference():
    a = {"final": True, "observed_metrics": {"mIoU": 29.87}, "provenance": {"elapsed_seconds": 1.0}}
    b = {"final": True, "observed_metrics": {"mIoU": 30.00}, "provenance": {"elapsed_seconds": 2.0}}
    with pytest.raises(h.TrustCentralityHarnessError):
        h.assert_scientific_projections_identical(a, b)


def test_checkpoint_rejects_old_v1_schema_version(tmp_path):
    import json
    legacy = {
        "schema_version": h.CHECKPOINT_SCHEMA_VERSION_V1, "run_status": "partial", "next_index": 1,
        "processed_image_ids": ["img0"], "processed_count": 1, "accumulator_state": {},
        "context": {
            "git_head": "a" * 40, "canonical_config_sha256": "b" * 64, "identity_name": "x",
            "num_classes": 3, "ignore_index": 255, "reduce_zero_label": False,
            "parity_check_interval": 25, "dataset_length": 5000,
        },
        "diagnostic_settings": {}, "parity_checked_image_ids": [],
    }
    path = tmp_path / "legacy_checkpoint.json"
    path.write_text(json.dumps(legacy))
    with pytest.raises(h.TrustCentralityHarnessError, match="v1"):
        h.load_checkpoint(path)


def test_validate_accumulator_state_complete_rejects_each_missing_key():
    full_state = {key: {} for key in h.REQUIRED_ACCUMULATOR_STATE_KEYS}
    h.validate_accumulator_state_complete(full_state)  # sanity: complete state passes
    for key in h.REQUIRED_ACCUMULATOR_STATE_KEYS:
        broken = dict(full_state)
        del broken[key]
        with pytest.raises(h.TrustCentralityHarnessError):
            h.validate_accumulator_state_complete(broken)


def test_validate_checkpoint_internal_consistency_detects_metric_count_mismatch():
    context = _make_context()
    accumulator_state = {
        "metric_accumulator": {"seen_image_ids": ["a", "b", "c"]},
        "t4_accumulator": {}, "trust_accumulator": {},
        "primary_solver": {"window_count": 4}, "parity_solver": {}, "windows_processed": 4,
    }
    checkpoint = h.HarnessCheckpoint(
        schema_version=h.CHECKPOINT_SCHEMA_VERSION, run_status=h.RUN_STATUS_PARTIAL, next_index=2,
        processed_image_ids=("a", "b"), processed_count=2,  # mismatches the 3 seen_image_ids above
        accumulator_state=accumulator_state, context=context, diagnostic_settings={}, parity_checked_image_ids=(),
    )
    with pytest.raises(h.TrustCentralityHarnessError):
        h.validate_checkpoint_internal_consistency(checkpoint)


def test_validate_checkpoint_internal_consistency_detects_solver_window_mismatch():
    context = _make_context()
    accumulator_state = {
        "metric_accumulator": {"seen_image_ids": ["a"]},
        "t4_accumulator": {}, "trust_accumulator": {},
        "primary_solver": {"window_count": 4}, "parity_solver": {}, "windows_processed": 999,  # mismatch
    }
    checkpoint = h.HarnessCheckpoint(
        schema_version=h.CHECKPOINT_SCHEMA_VERSION, run_status=h.RUN_STATUS_PARTIAL, next_index=1,
        processed_image_ids=("a",), processed_count=1,
        accumulator_state=accumulator_state, context=context, diagnostic_settings={}, parity_checked_image_ids=(),
    )
    with pytest.raises(h.TrustCentralityHarnessError):
        h.validate_checkpoint_internal_consistency(checkpoint)


def test_validate_checkpoint_internal_consistency_accepts_consistent_state():
    context = _make_context()
    accumulator_state = {
        "metric_accumulator": {"seen_image_ids": ["a", "b"]},
        "t4_accumulator": {}, "trust_accumulator": {},
        "primary_solver": {"window_count": 8}, "parity_solver": {}, "windows_processed": 8,
    }
    checkpoint = h.HarnessCheckpoint(
        schema_version=h.CHECKPOINT_SCHEMA_VERSION, run_status=h.RUN_STATUS_PARTIAL, next_index=2,
        processed_image_ids=("a", "b"), processed_count=2,
        accumulator_state=accumulator_state, context=context, diagnostic_settings={}, parity_checked_image_ids=(),
    )
    h.validate_checkpoint_internal_consistency(checkpoint)  # must not raise


def _minimal_complete_trust_centrality_section():
    bin_edges = list(tcd.DELTA_C_BIN_EDGES)
    return {
        "status": "available",
        **{
            key: {}
            for key in h.REQUIRED_TRUST_CENTRALITY_SECTION_KEYS
            if key not in ("funnel", "centrality", "crop_edge", "shared_unary")
        },
        "funnel": {k: {} for k in h.REQUIRED_FUNNEL_STAGE_KEYS},
        "centrality": {
            stage: {
                "population_count": 0, "delta_c": {}, "source_centrality": {}, "jury_centrality": {},
                "bin_edges": bin_edges, "bin_histogram": [0] * tcd.DELTA_C_BIN_COUNT,
                "strata": {stratum: {} for stratum in ("negative", "zero", "positive")},
                "bins": [{}] * tcd.DELTA_C_BIN_COUNT,
            }
            for stage in h.CENTRALITY_REPORT_STAGES
        },
        "crop_edge": {stage: {} for stage in h.CENTRALITY_REPORT_STAGES},
        "shared_unary": {group: {} for group in h.SHARED_UNARY_GROUPS_HARNESS},
    }


def _minimal_complete_full_report(**overrides):
    report = {
        "run_mode": h.RUN_MODE_FULL, "final": True, "complete": True, "finalization_attempted": True,
        "reconciliation_error": None, "failure_category": None,
        "observed_metrics": {"mIoU": 29.87},
        "natural_evaluate_result": {"captured": True, "mIoU": 29.87},
        "trust_centrality": _minimal_complete_trust_centrality_section(),
    }
    report.update(overrides)
    return report


def test_require_full_run_complete_accepts_a_genuinely_complete_full_run():
    h.require_full_run_complete(_minimal_complete_full_report())  # must not raise


def test_require_full_run_complete_ignores_partial_runs():
    h.require_full_run_complete({"run_mode": h.RUN_MODE_PARTIAL, "final": True, "complete": False})  # must not raise
    h.require_full_run_complete({"run_mode": h.RUN_MODE_PARTIAL, "final": False, "complete": False})  # must not raise


def test_require_full_run_complete_ignores_non_final_full_runs():
    h.require_full_run_complete({"run_mode": h.RUN_MODE_FULL, "final": False, "complete": False})  # must not raise


def test_require_full_run_complete_rejects_final_but_incomplete_full_run():
    report = _minimal_complete_full_report(complete=False)
    with pytest.raises(h.TrustCentralityFullRunIncompleteError):
        h.require_full_run_complete(report)


def test_require_full_run_complete_rejects_unavailable_trust_centrality():
    report = _minimal_complete_full_report(trust_centrality={"status": "unavailable", "reason": "x"})
    with pytest.raises(h.TrustCentralityFullRunIncompleteError):
        h.require_full_run_complete(report)


def test_require_full_run_complete_rejects_missing_observed_miou():
    report = _minimal_complete_full_report(observed_metrics={"mIoU": None})
    with pytest.raises(h.TrustCentralityFullRunIncompleteError):
        h.require_full_run_complete(report)


def test_require_full_run_complete_rejects_uncaptured_natural_evaluation():
    report = _minimal_complete_full_report(natural_evaluate_result={"captured": False})
    with pytest.raises(h.TrustCentralityFullRunIncompleteError):
        h.require_full_run_complete(report)


def test_require_full_run_complete_rejects_incomplete_trust_centrality_section():
    report = _minimal_complete_full_report()
    del report["trust_centrality"]["funnel"]
    with pytest.raises(h.TrustCentralityFullRunIncompleteError):
        h.require_full_run_complete(report)
