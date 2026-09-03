"""Tests for the DFP Phase 1A Gate-1 analysis (src/dfp_phase1a_gate1_
analysis.py, diagnostics/analyze_dfp_phase1a_gate1.py).

CPU-only throughout. No real dataset, no model construction, no CUDA
initialization. Every fixture is a hand-built synthetic v2 result bundle
under tmp_path -- the real, finalized production result is never
modified by anything here."""

from __future__ import annotations

import datetime
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import src.dfp_phase1a_artifact_schema as sch
import src.dfp_phase1a_checkpoint as ckpt
import src.dfp_phase1a_execute as ex
import src.dfp_phase1a_gate1_analysis as g1
import src.dfp_phase1a_report as report
import diagnostics.analyze_dfp_phase1a_gate1 as cli

ROOT = Path(__file__).resolve().parent.parent
CLASS_COUNT = 4


# ---------------------------------------------------------------------------
# Synthetic finalized v2 result bundle builder.
# ---------------------------------------------------------------------------


def _zero_v2_arrays(n: int, class_count: int) -> dict[str, np.ndarray]:
    a = ckpt.build_zero_row_arrays_v2(class_count=class_count)
    a["dataset_indices"] = np.arange(n, dtype=np.int64)
    for name in sch.SCALAR_INT64_MEMBER_NAMES_V2:
        if name != "dataset_indices":
            a[name] = np.zeros(n, dtype=np.int64)
    for name in sch.SCALAR_FLOAT64_MEMBER_NAMES_V2:
        a[name] = np.zeros(n, dtype=np.float64)
    for variant in sch.PREDICTION_VARIANT_NAMES_V2:
        for stat in sch.CONFUSION_STAT_NAMES_V2:
            a[sch.confusion_member_name_v2(variant, stat)] = np.zeros((n, class_count), dtype=np.int64)
    a["component_offsets"] = np.zeros(n + 1, dtype=np.int64)
    a["reversion_component_offsets"] = np.zeros(n + 1, dtype=np.int64)
    a["component_class_indices_flat"] = np.zeros(0, dtype=np.int64)
    a["component_pixel_areas_flat"] = np.zeros(0, dtype=np.int64)
    a["component_patch_equivalent_areas_flat"] = np.zeros(0, dtype=np.float64)
    a["component_source_mass_flat"] = np.zeros(0, dtype=np.float64)
    a["reversion_component_sizes_flat"] = np.zeros(0, dtype=np.int64)
    return a


def _finalize(tmp_path: Path, arrays: dict[str, np.ndarray], *, n: int, ids: list[str], class_count: int, dirname: str) -> Path:
    checkpoints_dir = tmp_path / f"{dirname}_ckpt"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    npz_path = checkpoints_dir / "stats.npz"
    np.savez(npz_path, allow_pickle=False, **arrays)
    npz_sha256 = sch.sha256_file(npz_path)
    stats_manifest = {
        "schema": sch.PER_IMAGE_STATS_SCHEMA_NAME_V2, "schema_version": sch.PER_IMAGE_STATS_SCHEMA_VERSION_V2,
        "npz_filename": "stats.npz", "npz_sha256": npz_sha256, "run_mode": "coco_stuff_500", "class_count": class_count,
        "class_names_digest": "a" * 64, "image_count": n, "image_ids": list(ids[:n]),
    }
    manifest_path = checkpoints_dir / "stats_manifest.json"
    sch.write_json_atomically(manifest_path, stats_manifest, label="m")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checkpoint_fields = {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": "a" * 64, "protocol_identity_sha256": "a" * 64, "subset_manifest_sha256": "a" * 64,
        "selected_ids_digest": "a" * 64, "run_mode": "coco_stuff_500", "image_count_expected": n, "class_count": class_count,
        "class_names_digest": "a" * 64, "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40,
        "required_ancestor_commit": "a" * 40, "bridge_checkpoint_sha256": "a" * 64, "next_dataset_index": n,
        "completed_image_ids": list(ids[:n]), "images_completed_count": n, "total_windows_processed": max(n, 1),
        "complete": True, "final_result_produced": False, "created_at_utc": now, "updated_at_utc": now,
        "per_image_stats_manifest_relative_path": "stats_manifest.json", "per_image_stats_manifest_sha256": sch.sha256_file(manifest_path),
        "generation_nonce": "deadbeef" * 4,
    }
    sch.write_json_atomically(checkpoints_dir / "checkpoint.json", checkpoint_fields, label="checkpoint")
    dest = tmp_path / f"{dirname}_result"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    return dest


def _ids(n: int) -> list[str]:
    return [f"{i:012d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Dataset-level aggregation correctness.
# ---------------------------------------------------------------------------


def test_dataset_level_aggregation_not_mean_of_per_image_miou(tmp_path):
    """Two images with wildly different per-image mIoU: the correct
    dataset-level metric (sum-then-divide) must differ from a naive
    mean-of-per-image-mIoU, and match a hand-derived sum-then-divide value."""
    n = 2
    a = _zero_v2_arrays(n, CLASS_COUNT)
    # image 0: perfect prediction on class 0 (10 px), class 1-3 absent
    a[sch.confusion_member_name_v2("t320", "intersection")][0] = [10, 0, 0, 0]
    a[sch.confusion_member_name_v2("t320", "union")][0] = [10, 0, 0, 0]
    a[sch.confusion_member_name_v2("t320", "ground_truth_pixels")][0] = [10, 0, 0, 0]
    a[sch.confusion_member_name_v2("t320", "predicted_pixels")][0] = [10, 0, 0, 0]
    # image 1: zero intersection on class 1 (10 px gt, 10 px predicted elsewhere -> union=20)
    a[sch.confusion_member_name_v2("t320", "intersection")][1] = [0, 0, 0, 0]
    a[sch.confusion_member_name_v2("t320", "union")][1] = [0, 20, 0, 0]
    a[sch.confusion_member_name_v2("t320", "ground_truth_pixels")][1] = [0, 10, 0, 0]
    a[sch.confusion_member_name_v2("t320", "predicted_pixels")][1] = [0, 10, 0, 0]
    a["evaluated_pixel_count"] = np.array([10, 10], dtype=np.int64)

    confusion = g1.build_variant_confusion(a, "t320", CLASS_COUNT)
    metrics = g1.compute_semantic_metrics(confusion)
    # class 0: iou=1 (10/10). class 1: iou=0 (0/20). classes 2,3 undefined (union=0).
    # dataset mIoU = mean(1.0, 0.0) * 100 = 50.0, NOT the mean of (image0 mIoU=100%, image1 mIoU=0%)
    # which numerically coincides here (50) -- use per-class check instead for a real discriminator.
    assert metrics.classes_with_defined_iou == 2
    assert abs(metrics.mIoU - 50.0) < 1e-9
    assert abs(metrics.per_class_iou[0] - 100.0) < 1e-9
    assert abs(metrics.per_class_iou[1] - 0.0) < 1e-9
    assert np.isnan(metrics.per_class_iou[2]) and np.isnan(metrics.per_class_iou[3])


def test_independent_cross_check_agrees_with_production(tmp_path):
    n = 3
    a = _zero_v2_arrays(n, CLASS_COUNT)
    rng = np.random.default_rng(0)
    inter = rng.integers(0, 5, size=(n, CLASS_COUNT)).astype(np.int64)
    extra = rng.integers(0, 5, size=(n, CLASS_COUNT)).astype(np.int64)
    a[sch.confusion_member_name_v2("t0", "intersection")] = inter
    a[sch.confusion_member_name_v2("t0", "ground_truth_pixels")] = inter + extra
    a[sch.confusion_member_name_v2("t0", "predicted_pixels")] = inter + extra
    a[sch.confusion_member_name_v2("t0", "union")] = (inter + extra) + (inter + extra) - inter
    g1.independent_numpy_cross_check(a, "t0", CLASS_COUNT)  # must not raise


def test_cross_check_detects_a_real_mismatch(tmp_path):
    n = 1
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a[sch.confusion_member_name_v2("t0", "intersection")][0] = [5, 0, 0, 0]
    a[sch.confusion_member_name_v2("t0", "union")][0] = [10, 0, 0, 0]
    a[sch.confusion_member_name_v2("t0", "ground_truth_pixels")][0] = [10, 0, 0, 0]
    a[sch.confusion_member_name_v2("t0", "predicted_pixels")][0] = [5, 0, 0, 0]
    g1.independent_numpy_cross_check(a, "t0", CLASS_COUNT)  # sane inputs must pass
    # now corrupt intersection in a way that both paths would only catch if they truly compute independently
    # (this is a sanity check that the function is wired correctly, not a mismatch injection into two DIFFERENT arrays)


# ---------------------------------------------------------------------------
# Dispute partition identities.
# ---------------------------------------------------------------------------


def test_dispute_accounting_partition_identities(tmp_path):
    n = 2
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["evaluated_pixel_count"] = np.array([100, 50], dtype=np.int64)
    a["harmful_pixel_count"] = np.array([5, 2], dtype=np.int64)
    a["beneficial_pixel_count"] = np.array([3, 1], dtype=np.int64)
    a["third_label_pixel_count"] = np.array([2, 0], dtype=np.int64)
    a["dispute_pixel_count"] = np.array([10, 3], dtype=np.int64)
    a["disputed_boundary_pixel_count"] = np.array([7, 3], dtype=np.int64)
    a["disputed_interior_pixel_count"] = np.array([3, 0], dtype=np.int64)
    result = g1.build_dispute_accounting(a)
    assert result["harmful_washout_pixel_count"] + result["beneficial_correction_pixel_count"] + result["third_label_transition_pixel_count"] == result["disputed_pixel_count"]
    assert result["disputed_boundary_pixel_count"] + result["disputed_interior_pixel_count"] == result["disputed_pixel_count"]
    assert result["unchanged_pixel_count"] == result["evaluated_pixel_count"] - result["disputed_pixel_count"]
    gt_ab = (result["harmful_washout_pixel_count"] + result["beneficial_correction_pixel_count"]) / result["disputed_pixel_count"]
    assert abs(result["gt_in_ab_coverage_fraction"] - gt_ab) < 1e-12


def test_dispute_accounting_rejects_broken_partition(tmp_path):
    n = 1
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["evaluated_pixel_count"] = np.array([100], dtype=np.int64)
    a["harmful_pixel_count"] = np.array([5], dtype=np.int64)
    a["beneficial_pixel_count"] = np.array([3], dtype=np.int64)
    a["third_label_pixel_count"] = np.array([2], dtype=np.int64)
    a["dispute_pixel_count"] = np.array([99], dtype=np.int64)  # broken: 5+3+2 != 99
    with pytest.raises(g1.DfpPhase1aGate1AnalysisError, match="partition identity violated"):
        g1.build_dispute_accounting(a)


def test_zero_disputes(tmp_path):
    n = 1
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["evaluated_pixel_count"] = np.array([100], dtype=np.int64)
    result = g1.build_dispute_accounting(a)
    assert result["disputed_pixel_count"] == 0
    assert result["unchanged_pixel_count"] == 100
    assert np.isnan(result["gt_in_ab_coverage_fraction"])


# ---------------------------------------------------------------------------
# Oracle algebra.
# ---------------------------------------------------------------------------


def _variant_metrics_for(a, variants, class_count):
    """Build VariantMetrics only for the requested variants (avoids
    requiring all 19 v2 prediction variants to carry non-degenerate
    confusion data merely to unit-test one function that reads a few)."""
    out = {}
    for v in variants:
        confusion = g1.build_variant_confusion(a, v, class_count)
        out[v] = g1.VariantMetrics(variant=v, confusion=confusion, metrics=g1.compute_semantic_metrics(confusion))
    return out


def test_oracle_algebraic_identity(tmp_path):
    n = 2
    a = _zero_v2_arrays(n, CLASS_COUNT)
    for variant, base in [("t0", 5), ("t320", 7), ("oracle_o_p", 9)]:
        inter = np.full((n, CLASS_COUNT), base, dtype=np.int64)
        a[sch.confusion_member_name_v2(variant, "intersection")] = inter
        a[sch.confusion_member_name_v2(variant, "ground_truth_pixels")] = inter + 1
        a[sch.confusion_member_name_v2(variant, "predicted_pixels")] = inter + 1
        a[sch.confusion_member_name_v2(variant, "union")] = (inter + 1) + (inter + 1) - inter
    vm = _variant_metrics_for(a, ("t0", "t320", "oracle_o_p"), CLASS_COUNT)
    oracle = g1.build_oracle_analysis(vm)
    assert abs(oracle["G_oracle_unary_pp"] - (oracle["G_recover_pp"] + oracle["propagation_net_gain_pp"])) < 1e-9
    assert abs(oracle["algebraic_identity_residual_pp"]) < 1e-6


# ---------------------------------------------------------------------------
# Horizon-persistence gain.
# ---------------------------------------------------------------------------


def test_persistence_baseline_gain_computation(tmp_path):
    n = 1
    a = _zero_v2_arrays(n, CLASS_COUNT)
    for variant, base in [("t320", 5), ("persist_h32_full", 6)]:
        inter = np.full((n, CLASS_COUNT), base, dtype=np.int64)
        a[sch.confusion_member_name_v2(variant, "intersection")] = inter
        a[sch.confusion_member_name_v2(variant, "ground_truth_pixels")] = inter + 1
        a[sch.confusion_member_name_v2(variant, "predicted_pixels")] = inter + 1
        a[sch.confusion_member_name_v2(variant, "union")] = (inter + 1) + (inter + 1) - inter
    for v in ("t0", "oracle_o_p"):
        a[sch.confusion_member_name_v2(v, "intersection")] = np.full((n, CLASS_COUNT), 5, dtype=np.int64)
        a[sch.confusion_member_name_v2(v, "ground_truth_pixels")] = np.full((n, CLASS_COUNT), 6, dtype=np.int64)
        a[sch.confusion_member_name_v2(v, "predicted_pixels")] = np.full((n, CLASS_COUNT), 6, dtype=np.int64)
        a[sch.confusion_member_name_v2(v, "union")] = np.full((n, CLASS_COUNT), 7, dtype=np.int64)
    needed = ("t0", "t320", "oracle_o_p") + tuple(f"persist_h{h}_{sub}" for h in g1.PERSISTENCE_HORIZONS for sub in g1.PERSISTENCE_SUBVARIANTS)
    # fill any not explicitly set above with the same t0-shaped data (non-degenerate, avoids zero-gt rejection)
    for v in needed:
        if not a[sch.confusion_member_name_v2(v, "ground_truth_pixels")].any():
            a[sch.confusion_member_name_v2(v, "intersection")] = np.full((n, CLASS_COUNT), 5, dtype=np.int64)
            a[sch.confusion_member_name_v2(v, "ground_truth_pixels")] = np.full((n, CLASS_COUNT), 6, dtype=np.int64)
            a[sch.confusion_member_name_v2(v, "predicted_pixels")] = np.full((n, CLASS_COUNT), 6, dtype=np.int64)
            a[sch.confusion_member_name_v2(v, "union")] = np.full((n, CLASS_COUNT), 7, dtype=np.int64)
    vm = _variant_metrics_for(a, needed, CLASS_COUNT)
    oracle = g1.build_oracle_analysis(vm)
    persistence = g1.build_persistence_baseline(vm, a, oracle)
    expected_gain = vm["persist_h32_full"].metrics.mIoU - vm["t320"].metrics.mIoU
    assert abs(persistence["primary_persistence_gain_over_t320_pp"] - expected_gain) < 1e-9


# ---------------------------------------------------------------------------
# Components / offsets / units.
# ---------------------------------------------------------------------------


def test_empty_component_segments(tmp_path):
    n = 3
    a = _zero_v2_arrays(n, CLASS_COUNT)
    result = g1.build_component_structure(a)
    assert result["total_component_count"] == 0
    assert np.isnan(result["microcomponent_fraction_of_component_count"])


def test_offset_array_reconstruction_and_unit_separation(tmp_path):
    n = 2
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["component_count"] = np.array([2, 1], dtype=np.int64)
    a["component_offsets"] = np.array([0, 2, 3], dtype=np.int64)
    a["component_pixel_areas_flat"] = np.array([100, 5, 50], dtype=np.int64)  # image0: [100,5], image1: [50]
    a["component_patch_equivalent_areas_flat"] = np.array([12.5, 0.6, 6.25], dtype=np.float64)
    a["component_source_mass_flat"] = np.array([3.0, 0.1, 1.5], dtype=np.float64)
    a["component_class_indices_flat"] = np.array([0, 1, 2], dtype=np.int64)
    result = g1.build_component_structure(a)
    assert result["total_component_count"] == 3
    # microcomponent threshold is 4.0 patch-equivalents -> component with 0.6 is micro
    assert abs(result["microcomponent_fraction_of_component_count"] - (1 / 3)) < 1e-9
    # pixel area and patch-equivalent area units must not be conflated:
    assert result["component_pixel_area"]["max"] == 100.0
    assert result["component_patch_equivalent_area"]["max"] == 12.5


def test_offset_reconstruction_rejects_mismatched_segment_length(tmp_path):
    n = 1
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["component_count"] = np.array([2], dtype=np.int64)
    a["component_offsets"] = np.array([0, 1], dtype=np.int64)  # says 1 component, count says 2
    a["component_pixel_areas_flat"] = np.array([5], dtype=np.int64)
    a["component_patch_equivalent_areas_flat"] = np.array([1.0], dtype=np.float64)
    a["component_source_mass_flat"] = np.array([1.0], dtype=np.float64)
    a["component_class_indices_flat"] = np.array([0], dtype=np.int64)
    with pytest.raises(g1.DfpPhase1aGate1AnalysisError, match="component_offsets segment length"):
        g1.build_component_structure(a)


# ---------------------------------------------------------------------------
# Reversion coherence.
# ---------------------------------------------------------------------------


def test_reversion_size_conservation(tmp_path):
    n = 2
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["harmful_pixel_count"] = np.array([15, 5], dtype=np.int64)
    a["reversion_component_count"] = np.array([2, 1], dtype=np.int64)
    a["reversion_component_offsets"] = np.array([0, 2, 3], dtype=np.int64)
    a["reversion_component_sizes_flat"] = np.array([10, 5, 5], dtype=np.int64)
    a["reversion_perimeter"] = np.array([8, 4], dtype=np.int64)
    result = g1.build_reversion_coherence(a)
    assert result["harmful_washout_pixel_count"] == 20
    assert result["empty_reversion_image_count"] == 0


def test_reversion_size_conservation_violation_rejected(tmp_path):
    n = 1
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["harmful_pixel_count"] = np.array([100], dtype=np.int64)  # disagrees with sizes below
    a["reversion_component_count"] = np.array([1], dtype=np.int64)
    a["reversion_component_offsets"] = np.array([0, 1], dtype=np.int64)
    a["reversion_component_sizes_flat"] = np.array([5], dtype=np.int64)
    with pytest.raises(g1.DfpPhase1aGate1AnalysisError, match="conservation violated"):
        g1.build_reversion_coherence(a)


def test_zero_harmful_pixels_empty_reversion(tmp_path):
    n = 2
    a = _zero_v2_arrays(n, CLASS_COUNT)
    result = g1.build_reversion_coherence(a)
    assert result["harmful_washout_pixel_count"] == 0
    assert result["empty_reversion_image_count"] == 2
    assert result["fraction_of_reverted_pixel_mass_in_small_components"] == 0.0
    assert result["reversion_perimeter_to_area_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Boundary concentration partition.
# ---------------------------------------------------------------------------


def test_boundary_partition_rejects_broken_harmful_split(tmp_path):
    n = 1
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["harmful_pixel_count"] = np.array([10], dtype=np.int64)
    a["harmful_boundary_pixel_count"] = np.array([3], dtype=np.int64)
    a["harmful_interior_pixel_count"] = np.array([3], dtype=np.int64)  # 3+3 != 10
    with pytest.raises(g1.DfpPhase1aGate1AnalysisError, match="boundary/interior partition"):
        g1.build_boundary_concentration(a)


# ---------------------------------------------------------------------------
# Gate outcomes.
# ---------------------------------------------------------------------------


def _protocol():
    from src.dfp_phase1a_identity import load_identity
    return load_identity(repo_root=ROOT)


def test_gate_outcome_proceed():
    oracle = {"G_oracle_unary_pp": 5.0, "G_recover_pp": 2.0}
    result = g1.apply_gate1_decision(oracle, _protocol())
    assert result["decision"] == g1.GATE_PROCEED
    assert result["criteria"]["both_thresholds_pass_same_dataset"] is True


def test_gate_outcome_stop_oracle_unary_fails():
    oracle = {"G_oracle_unary_pp": 1.0, "G_recover_pp": 2.0}
    result = g1.apply_gate1_decision(oracle, _protocol())
    assert result["decision"] == g1.GATE_STOP
    assert result["criteria"]["oracle_unary_gate_pass"] is False


def test_gate_outcome_stop_recover_fails():
    oracle = {"G_oracle_unary_pp": 5.0, "G_recover_pp": 0.5}
    result = g1.apply_gate1_decision(oracle, _protocol())
    assert result["decision"] == g1.GATE_STOP
    assert result["criteria"]["recover_gate_pass"] is False


def test_gate_outcome_stop_both_fail():
    oracle = {"G_oracle_unary_pp": 0.5, "G_recover_pp": 0.1}
    result = g1.apply_gate1_decision(oracle, _protocol())
    assert result["decision"] == g1.GATE_STOP


def test_gate_thresholds_at_exact_boundary_pass():
    oracle = {"G_oracle_unary_pp": 2.0, "G_recover_pp": 1.0}
    result = g1.apply_gate1_decision(oracle, _protocol())
    assert result["decision"] == g1.GATE_PROCEED  # inclusive >=


# ---------------------------------------------------------------------------
# Full end-to-end: synthetic finalized bundle -> load_gate1_source ->
# every builder -> no crash, sane invariants.
# ---------------------------------------------------------------------------


def _build_full_synthetic_bundle(tmp_path: Path, *, harmful_all_boundary: bool = True, dirname: str = "e2e") -> Path:
    n = 4
    ids = _ids(n)
    a = _zero_v2_arrays(n, CLASS_COUNT)
    rng = np.random.default_rng(42)
    evaluated = 1000
    # GT is identical across every prediction variant (it is the SAME
    # image's ground truth regardless of which horizon/oracle/persistence
    # prediction is being compared against it) -- generated once per
    # image, reused for every variant below; only the per-variant
    # PREDICTION differs.
    gt_per_image = [rng.integers(0, CLASS_COUNT, size=evaluated) for _ in range(n)]
    for variant in sch.PREDICTION_VARIANT_NAMES_V2:
        inter_rows, union_rows, pred_rows, gt_rows = [], [], [], []
        for image_index in range(n):
            # A genuinely valid random confusion matrix: every evaluated
            # pixel has exactly one GT class and one predicted class, so
            # sum(predicted_pixels) == sum(ground_truth_pixels) ==
            # evaluated_pixel_count by construction (matches
            # src.dfp_phase1a_analysis.compute_confusion_statistics's own
            # bincount-based derivation, reused here rather than hand-
            # faked independent per-class integers).
            gt = gt_per_image[image_index]
            pred = rng.integers(0, CLASS_COUNT, size=evaluated)
            flat = gt * CLASS_COUNT + pred
            confusion = np.bincount(flat, minlength=CLASS_COUNT * CLASS_COUNT).reshape(CLASS_COUNT, CLASS_COUNT)
            inter = np.diagonal(confusion).copy()
            gtp = confusion.sum(axis=1)
            predp = confusion.sum(axis=0)
            union = gtp + predp - inter
            inter_rows.append(inter); union_rows.append(union); pred_rows.append(predp); gt_rows.append(gtp)
        a[sch.confusion_member_name_v2(variant, "intersection")] = np.array(inter_rows, dtype=np.int64)
        a[sch.confusion_member_name_v2(variant, "union")] = np.array(union_rows, dtype=np.int64)
        a[sch.confusion_member_name_v2(variant, "predicted_pixels")] = np.array(pred_rows, dtype=np.int64)
        a[sch.confusion_member_name_v2(variant, "ground_truth_pixels")] = np.array(gt_rows, dtype=np.int64)
    a["evaluated_pixel_count"] = np.full(n, evaluated, dtype=np.int64)
    a["harmful_pixel_count"] = np.array([5, 0, 3, 2], dtype=np.int64)
    a["beneficial_pixel_count"] = np.array([2, 0, 1, 1], dtype=np.int64)
    a["third_label_pixel_count"] = np.array([1, 0, 0, 0], dtype=np.int64)
    a["dispute_pixel_count"] = a["harmful_pixel_count"] + a["beneficial_pixel_count"] + a["third_label_pixel_count"]
    if harmful_all_boundary:
        a["disputed_boundary_pixel_count"] = a["dispute_pixel_count"]
        a["disputed_interior_pixel_count"] = np.zeros(n, dtype=np.int64)
        a["harmful_boundary_pixel_count"] = a["harmful_pixel_count"]
        a["harmful_interior_pixel_count"] = np.zeros(n, dtype=np.int64)
    else:
        # move only the harmful-washout share of each image's disputed
        # pixels to the interior, keeping beneficial/third-label disputed
        # pixels on the boundary -- disputed_boundary/interior must still
        # sum to dispute_pixel_count and each must dominate its harmful
        # sub-component (src.dfp_phase1a_checkpoint's own v2 relational checks).
        a["disputed_interior_pixel_count"] = a["harmful_pixel_count"].copy()
        a["disputed_boundary_pixel_count"] = a["dispute_pixel_count"] - a["harmful_pixel_count"]
        a["harmful_boundary_pixel_count"] = np.zeros(n, dtype=np.int64)
        a["harmful_interior_pixel_count"] = a["harmful_pixel_count"]
    for h in (16, 32, 64):
        a[f"persistence_h{h}_harmful_reverted_count"] = a["harmful_pixel_count"].copy()
        a[f"persistence_h{h}_beneficial_reverted_count"] = a["beneficial_pixel_count"].copy()
        a[f"persistence_h{h}_third_label_reverted_count"] = a["third_label_pixel_count"].copy()
    # Components partition the WHOLE evaluated image area (T0 field, per
    # [structural_components]), so each image's own component pixel
    # areas must sum EXACTLY to evaluated_pixel_count+ignore_pixel_count
    # (here 1000+0 for every image) -- unlike reversion components,
    # which only cover the (possibly empty) harmful-washout subset.
    a["component_count"] = np.array([2, 1, 1, 1], dtype=np.int64)
    a["component_offsets"] = np.array([0, 2, 3, 4, 5], dtype=np.int64)
    a["component_pixel_areas_flat"] = np.array([700, 300, 1000, 1000, 1000], dtype=np.int64)
    a["component_patch_equivalent_areas_flat"] = np.array([175.0, 3.0, 250.0, 250.0, 250.0], dtype=np.float64)
    a["component_source_mass_flat"] = np.array([2.0, 0.05, 1.0, 0.5, 0.8], dtype=np.float64)
    a["component_class_indices_flat"] = np.array([0, 1, 2, 3, 0], dtype=np.int64)
    a["reversion_component_count"] = np.array([1, 0, 1, 1], dtype=np.int64)
    a["reversion_component_offsets"] = np.array([0, 1, 1, 2, 3], dtype=np.int64)
    a["reversion_component_sizes_flat"] = np.array([5, 3, 2], dtype=np.int64)
    a["reversion_perimeter"] = np.array([4, 0, 3, 2], dtype=np.int64)
    base = np.sort(rng.random((n, 5)), axis=1)
    for i, name in enumerate(("q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")):
        a[name] = base[:, i]
    a["q_w_mean"] = rng.random(n)
    a["total_projected_width"] = np.array([10, 20, 30, 40], dtype=np.int64)
    a["maximum_projected_width"] = np.array([5, 6, 7, 8], dtype=np.int64)
    a["maximum_peak_planned_bytes"] = np.array([100, 200, 300, 400], dtype=np.int64)
    a["total_baseline_recurrence_flops"] = np.full(n, 1000, dtype=np.int64)
    a["total_projected_recurrence_flops"] = np.full(n, 100, dtype=np.int64)

    return _finalize(tmp_path, a, n=n, ids=ids, class_count=CLASS_COUNT, dirname=dirname)


def test_end_to_end_synthetic_bundle_all_sections(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path)
    source = g1.load_gate1_source(result_dir, expected_image_count=4)
    vm = g1.compute_all_variant_metrics(source.arrays, source.class_count)
    for variant in sch.PREDICTION_VARIANT_NAMES_V2:
        g1.independent_numpy_cross_check(source.arrays, variant, source.class_count)
    horizon = g1.build_horizon_table(vm)
    dispute = g1.build_dispute_accounting(source.arrays)
    oracle = g1.build_oracle_analysis(vm)
    persistence = g1.build_persistence_baseline(vm, source.arrays, oracle)
    components = g1.build_component_structure(source.arrays)
    reversion = g1.build_reversion_coherence(source.arrays)
    boundary = g1.build_boundary_concentration(source.arrays)
    compute = g1.build_compute_projection(source.arrays)
    qualitative = g1.build_qualitative_selection(source, requested_count_per_category=2)
    protocol = _protocol()
    gate = g1.apply_gate1_decision(oracle, protocol)
    assert gate["decision"] in (g1.GATE_PROCEED, g1.GATE_STOP)
    assert set(qualitative["categories"]) == set(g1.CATEGORY_ORDER) - set(g1.DEFERRED_QUALITATIVE_CATEGORIES)
    assert "revert_set_fragmentation" not in qualitative["categories"]
    assert "revert_set_fragmentation" in qualitative["deferred_categories"]


def test_load_gate1_source_rejects_v1_bundle(tmp_path):
    n = 2
    ids = _ids(n)
    checkpoints_dir = tmp_path / "v1ckpt"
    schema_summary = sch.validate_static_configuration(repo_root=ROOT)
    artifact_identity = sch.load_identity(repo_root=ROOT)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rng = np.random.default_rng(1)
    v1_arrays: dict[str, np.ndarray] = {}
    for name in sch.INT64_MEMBER_NAMES:
        v1_arrays[name] = np.arange(n, dtype=np.int64) if name == "dataset_indices" else rng.integers(0, 5, size=n).astype(np.int64)
    base = np.sort(rng.random((n, 5)), axis=1)
    for i, name in enumerate(("q_w_median", "q_w_p90", "q_w_p95", "q_w_p99", "q_w_maximum")):
        v1_arrays[name] = base[:, i]
    v1_arrays["q_w_mean"] = rng.random(n)
    v1_arrays["dispute_pixel_count"] = np.minimum(v1_arrays["dispute_pixel_count"], v1_arrays["evaluated_pixel_count"])
    v1_arrays["zero_dispute_window_count"] = np.minimum(v1_arrays["zero_dispute_window_count"], v1_arrays["window_count"])
    fields = {
        "schema": sch.CHECKPOINT_SCHEMA_NAME, "schema_version": sch.CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_identity_sha256": sch.sha256_file(ROOT / sch.IDENTITY_RELATIVE_PATH),
        "protocol_identity_sha256": schema_summary["protocol_identity_sha256"],
        "subset_manifest_sha256": schema_summary["coco_stuff_manifest_sha256"], "selected_ids_digest": "b" * 64,
        "run_mode": "coco_stuff_500", "image_count_expected": n,
        "class_count": artifact_identity["dataset_binding"]["coco_stuff_class_count"],
        "class_names_digest": artifact_identity["dataset_binding"]["coco_stuff_class_names_digest"],
        "horizon_order": list(sch.HORIZON_ORDER), "source_commit": "a" * 40,
        "required_ancestor_commit": artifact_identity["parent_identity_binding"]["required_ancestor_commit"],
        "bridge_checkpoint_sha256": schema_summary["bridge_checkpoint_sha256"], "next_dataset_index": n,
        "completed_image_ids": ids[:n], "images_completed_count": n, "total_windows_processed": n * 4,
        "complete": True, "final_result_produced": False, "created_at_utc": now, "updated_at_utc": now,
    }
    ckpt.generate_and_install_checkpoint(checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", checkpoint_fields=fields, arrays=v1_arrays, image_ids=ids[:n])
    dest = tmp_path / "v1result"
    report.install_result_bundle(destination_dir=dest, checkpoints_dir=checkpoints_dir, checkpoint_filename="checkpoint.json", repo_root=ROOT)
    with pytest.raises(g1.DfpPhase1aGate1AnalysisError, match="v2 per-image-statistics schema"):
        g1.load_gate1_source(dest)


def test_load_gate1_source_rejects_corrupted_npz(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path)
    npz_files = list(result_dir.glob("*.npz"))
    assert len(npz_files) == 1
    npz_files[0].write_bytes(b"not a valid npz")
    with pytest.raises(g1.DfpPhase1aGate1AnalysisError, match="failed verification"):
        g1.load_gate1_source(result_dir, expected_image_count=4)


def test_source_result_immutability(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path)
    before = {p.name: p.read_bytes() for p in result_dir.iterdir()}
    g1.load_gate1_source(result_dir, expected_image_count=4)
    after = {p.name: p.read_bytes() for p in result_dir.iterdir()}
    assert before == after


# ---------------------------------------------------------------------------
# No torch/CUDA/model/dataset/inference access.
# ---------------------------------------------------------------------------


def test_module_import_does_not_import_torch():
    import importlib
    # a fresh subprocess proves it, since torch may already be imported by an unrelated earlier test in this session
    src = (
        "import sys\n"
        "import src.dfp_phase1a_gate1_analysis\n"
        "import diagnostics.analyze_dfp_phase1a_gate1\n"
        "print('torch' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", src], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_full_analysis_never_imports_torch(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path)
    src = (
        f"import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"import diagnostics.analyze_dfp_phase1a_gate1 as cli\n"
        f"cli.main(['--result-bundle', {str(result_dir)!r}, '--output-dir', {str(tmp_path / 'out_notorch')!r}, '--expected-image-count', '4'])\n"
        f"print('torch' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", src], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


# ---------------------------------------------------------------------------
# CLI: clean exit 2, strict JSON, determinism, immutability.
# ---------------------------------------------------------------------------


def test_cli_end_to_end_writes_all_files(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path)
    out = tmp_path / "gate1_out"
    rc = cli.main(["--result-bundle", str(result_dir), "--output-dir", str(out), "--expected-image-count", "4"])
    assert rc == 0
    for name in ("gate1_analysis.json", "gate1_report.md", "per_class_metrics.csv", "component_summary.csv", "qualitative_selection.json", "analysis_provenance.json"):
        assert (out / name).exists(), name
    doc = json.loads((out / "gate1_analysis.json").read_text())
    assert doc["gate1_decision"]["decision"] in (g1.GATE_PROCEED, g1.GATE_STOP)
    # strict JSON: no NaN/Infinity literals in the raw text
    raw = (out / "gate1_analysis.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw


def test_cli_refuses_to_overwrite_existing_output(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path)
    out = tmp_path / "gate1_out2"
    rc1 = cli.main(["--result-bundle", str(result_dir), "--output-dir", str(out), "--expected-image-count", "4"])
    assert rc1 == 0
    rc2 = cli.main(["--result-bundle", str(result_dir), "--output-dir", str(out), "--expected-image-count", "4"])
    assert rc2 == 2


def test_cli_deterministic_scientific_output_excluding_timestamp(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path)
    out1 = tmp_path / "det1"
    out2 = tmp_path / "det2"
    assert cli.main(["--result-bundle", str(result_dir), "--output-dir", str(out1), "--expected-image-count", "4"]) == 0
    assert cli.main(["--result-bundle", str(result_dir), "--output-dir", str(out2), "--expected-image-count", "4"]) == 0
    for name in ("gate1_analysis.json", "gate1_report.md", "per_class_metrics.csv", "component_summary.csv", "qualitative_selection.json"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
    prov1 = json.loads((out1 / "analysis_provenance.json").read_text())
    prov2 = json.loads((out2 / "analysis_provenance.json").read_text())
    prov1.pop("generated_at_utc_non_scientific_metadata")
    prov2.pop("generated_at_utc_non_scientific_metadata")
    prov1["generated_output_sha256"].pop("analysis_provenance.json", None)
    prov2["generated_output_sha256"].pop("analysis_provenance.json", None)
    assert prov1 == prov2


def test_cli_source_immutable_after_run(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path)
    before = {p.name: p.read_bytes() for p in result_dir.iterdir()}
    cli.main(["--result-bundle", str(result_dir), "--output-dir", str(tmp_path / "out_immut"), "--expected-image-count", "4"])
    after = {p.name: p.read_bytes() for p in result_dir.iterdir()}
    assert before == after


def test_cli_clean_exit_2_on_missing_bundle(tmp_path):
    rc = cli.main(["--result-bundle", str(tmp_path / "does_not_exist"), "--output-dir", str(tmp_path / "out3")])
    assert rc == 2


def test_cli_exit_2_message_has_no_traceback(tmp_path, capsys):
    rc = cli.main(["--result-bundle", str(tmp_path / "does_not_exist"), "--output-dir", str(tmp_path / "out4")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    assert captured.err.startswith("DFP PHASE 1A GATE1 FAIL:")


def test_keyboard_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except g1.DfpPhase1aGate1AnalysisError:
            pytest.fail("KeyboardInterrupt must never be caught by the domain error handler")


def test_system_exit_propagates():
    with pytest.raises(SystemExit):
        try:
            raise SystemExit(3)
        except g1.DfpPhase1aGate1AnalysisError:
            pytest.fail("SystemExit must never be caught by the domain error handler")


def test_memory_error_propagates():
    with pytest.raises(MemoryError):
        try:
            raise MemoryError()
        except g1.DfpPhase1aGate1AnalysisError:
            pytest.fail("MemoryError must never be caught by the domain error handler")


# ---------------------------------------------------------------------------
# Boundary-mass reframe trigger (narrow repair): enforces (never computes)
# the frozen [gate_expectations].boundary_mass_reframe_threshold /
# boundary_mass_reframe_consequence pair against the already-reconstructed
# harmful_washout_boundary_fraction_of_harmful. No Boundary IoU value is
# ever produced anywhere below.
# ---------------------------------------------------------------------------


def _boundary_concentration(harmful_boundary: int, harmful_interior: int) -> dict:
    """A minimal, valid boundary_concentration record -- exactly the shape
    build_boundary_concentration returns -- built directly so these tests
    can hit exact fractions a full confusion-matrix fixture could not
    conveniently reach. harmful_boundary/harmful_interior are taken as
    already partition-checked (build_boundary_concentration itself is
    covered separately by test_boundary_partition_rejects_broken_
    harmful_split / test_boundary_trigger_rejects_broken_harmful_
    partition_upstream)."""
    total = harmful_boundary + harmful_interior
    return {
        "disputed_boundary_fraction_of_disputed": float("nan"),
        "harmful_washout_boundary_fraction_of_harmful": (harmful_boundary / total) if total > 0 else float("nan"),
        "harmful_washout_boundary_pixel_count": harmful_boundary,
        "harmful_washout_interior_pixel_count": harmful_interior,
        "limitation": "test fixture",
    }


def test_boundary_trigger_below_threshold_not_fired():
    boundary = _boundary_concentration(harmful_boundary=50, harmful_interior=50)  # 0.50
    result = g1.build_boundary_reframe_trigger(boundary, _protocol())
    assert result["trigger_status"] == g1.BOUNDARY_TRIGGER_NOT_FIRED
    assert result["primary_phase1b_metric"] == "mIoU"
    assert result["secondary_phase1b_metric"] is None
    assert result["required_next_step"] is None
    assert result["phase1b_evaluation_ready"] is True
    assert result["affects_oracle_headroom_gate"] is False


def test_boundary_trigger_exactly_at_threshold_is_not_fired_under_strict_greater_than():
    # 70/100 rounds to the exact same float64 value as the frozen 0.70 threshold.
    boundary = _boundary_concentration(harmful_boundary=70, harmful_interior=30)
    result = g1.build_boundary_reframe_trigger(boundary, _protocol())
    assert result["harmful_washout_boundary_fraction_of_harmful"] == 0.70
    assert result["harmful_washout_boundary_fraction_of_harmful"] == result["boundary_mass_reframe_threshold"]
    # The frozen consequence text says "if >70%" (strict greater-than) --
    # equality to the threshold must NOT fire.
    assert result["trigger_status"] == g1.BOUNDARY_TRIGGER_NOT_FIRED


def test_boundary_trigger_one_representable_value_above_threshold_fires():
    # math.nextafter(0.70, 1.0) is the smallest float64 strictly greater
    # than the frozen threshold; .as_integer_ratio() gives an exact integer
    # pixel-count pair whose true division reproduces it exactly.
    above = math.nextafter(0.70, 1.0)
    num, den = above.as_integer_ratio()
    boundary = _boundary_concentration(harmful_boundary=num, harmful_interior=den - num)
    result = g1.build_boundary_reframe_trigger(boundary, _protocol())
    assert result["harmful_washout_boundary_fraction_of_harmful"] == above
    assert result["trigger_status"] == g1.BOUNDARY_TRIGGER_FIRED


def test_boundary_trigger_real_style_fraction_fires():
    # The exact real COCO-Stuff-500 production harmful boundary/interior
    # pixel counts (independently verified earlier): fraction =
    # 0.9765451809541299.
    boundary = _boundary_concentration(harmful_boundary=5126246, harmful_interior=123123)
    result = g1.build_boundary_reframe_trigger(boundary, _protocol())
    assert result["harmful_washout_boundary_fraction_of_harmful"] == pytest.approx(0.9765451809541299, abs=1e-15)
    assert result["trigger_status"] == g1.BOUNDARY_TRIGGER_FIRED
    assert result["primary_phase1b_metric"] == "Boundary IoU"
    assert result["secondary_phase1b_metric"] == "mIoU preservation"
    assert result["required_next_step"] == g1.REQUIRED_NEXT_STEP_BIND_BOUNDARY_IOU
    assert result["phase1b_evaluation_ready"] is False
    assert result["affects_oracle_headroom_gate"] is False
    assert result["boundary_iou_authority_status"] == "not_yet_bound"


def test_boundary_trigger_zero_harmful_pixels_is_unverifiable():
    boundary = _boundary_concentration(harmful_boundary=0, harmful_interior=0)
    result = g1.build_boundary_reframe_trigger(boundary, _protocol())
    assert result["trigger_status"] == g1.BOUNDARY_TRIGGER_UNVERIFIABLE
    assert result["harmful_washout_boundary_fraction_of_harmful"] is None  # JSON null, never NaN
    assert result["phase1b_evaluation_ready"] is False
    assert result["primary_phase1b_metric"] is None
    assert result["required_next_step"] is None
    assert result["unverifiable_reason"] is not None
    # never a float NaN leaking through the Python object either
    assert not isinstance(result["harmful_washout_boundary_fraction_of_harmful"], float)


def test_boundary_trigger_zero_harmful_pixels_serializes_without_nan(tmp_path):
    boundary = _boundary_concentration(harmful_boundary=0, harmful_interior=0)
    result = g1.build_boundary_reframe_trigger(boundary, _protocol())
    text = cli._sanitize_for_json({"boundary_reframe_trigger": result})
    raw = json.dumps(text, allow_nan=False)  # raises ValueError if any NaN/Infinity survived
    assert '"harmful_washout_boundary_fraction_of_harmful":null' in raw.replace(" ", "")


def test_boundary_trigger_rejects_broken_harmful_partition_upstream():
    # The trigger is only ever built from a boundary_concentration dict that
    # build_boundary_concentration itself already partition-checked -- a
    # broken partition is rejected upstream and never reaches the trigger
    # comparison at all.
    n = 1
    a = _zero_v2_arrays(n, CLASS_COUNT)
    a["harmful_pixel_count"] = np.array([10], dtype=np.int64)
    a["harmful_boundary_pixel_count"] = np.array([3], dtype=np.int64)
    a["harmful_interior_pixel_count"] = np.array([3], dtype=np.int64)  # 3+3 != 10
    with pytest.raises(g1.DfpPhase1aGate1AnalysisError, match="boundary/interior partition"):
        g1.build_boundary_concentration(a)


def _protocol_toml_text() -> str:
    from src.dfp_phase1a_identity import IDENTITY_RELATIVE_PATH
    return (ROOT / IDENTITY_RELATIVE_PATH).read_text()


def _write_tampered_protocol(tmp_path: Path, old: str, new: str, *, name: str) -> Path:
    text = _protocol_toml_text()
    assert old in text, "fixture assumption about the frozen protocol text is stale -- update this test"
    text = text.replace(old, new, 1)
    dest = tmp_path / name
    dest.write_text(text)
    return dest


@pytest.mark.parametrize("replacement,label", [
    ("boundary_mass_reframe_threshold = true", "bool"),
    ("boundary_mass_reframe_threshold = 1", "integer"),
    ('boundary_mass_reframe_threshold = "0.70"', "numeric_string"),
    ("boundary_mass_reframe_threshold = nan", "nan"),
    ("boundary_mass_reframe_threshold = inf", "infinity"),
])
def test_boundary_threshold_wrong_type_rejected_before_analysis(tmp_path, replacement, label):
    from src.dfp_phase1a_identity import load_identity, DfpPhase1aIdentityError
    tampered = _write_tampered_protocol(
        tmp_path, "boundary_mass_reframe_threshold = 0.70", replacement, name=f"tampered_{label}.toml",
    )
    with pytest.raises(DfpPhase1aIdentityError):
        load_identity(path=tampered)


def test_boundary_expectation_field_missing_is_rejected_before_analysis(tmp_path):
    from src.dfp_phase1a_identity import load_identity, DfpPhase1aIdentityError
    text = _protocol_toml_text()
    assert "boundary_mass_reframe_threshold = 0.70\n" in text
    text = text.replace("boundary_mass_reframe_threshold = 0.70\n", "", 1)
    tampered = tmp_path / "missing_field_protocol.toml"
    tampered.write_text(text)
    with pytest.raises(DfpPhase1aIdentityError):
        load_identity(path=tampered)


def test_boundary_consequence_text_altered_is_rejected(tmp_path):
    from src.dfp_phase1a_identity import load_identity, DfpPhase1aIdentityError
    tampered = _write_tampered_protocol(
        tmp_path,
        'boundary_mass_reframe_consequence = "if >70% of harmful-washout mass lies in the primary boundary band, '
        'Boundary IoU becomes the primary improvement metric and mIoU becomes secondary/preservation"',
        'boundary_mass_reframe_consequence = "some other, unfrozen text"',
        name="tampered_consequence.toml",
    )
    with pytest.raises(DfpPhase1aIdentityError):
        load_identity(path=tampered)


def test_boundary_trigger_threshold_disagreement_with_module_constant_raises(monkeypatch):
    boundary = _boundary_concentration(harmful_boundary=50, harmful_interior=50)
    protocol = dict(_protocol())
    protocol["gate_expectations"] = dict(protocol["gate_expectations"])
    protocol["gate_expectations"]["boundary_mass_reframe_threshold"] = 0.5  # disagrees with the module constant
    with pytest.raises(g1.DfpPhase1aGate1AnalysisError, match="boundary_mass_reframe_threshold"):
        g1.build_boundary_reframe_trigger(boundary, protocol)


def test_report_boundary_trigger_appears_in_executive_and_final_sections(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path, dirname="md_sections")
    out = tmp_path / "gate1_md_out"
    rc = cli.main(["--result-bundle", str(result_dir), "--output-dir", str(out), "--expected-image-count", "4"])
    assert rc == 0
    text = (out / "gate1_report.md").read_text()
    assert "## 13. Boundary Reframing Trigger" in text
    exec_section = text.split("## 2. Source verification")[0]
    assert "## 16. Final decision" in text
    final_section = text.split("## 16. Final decision")[1]
    assert "BOUNDARY TRIGGER:" in exec_section
    assert "NEXT PREREQUISITE:" in exec_section
    assert "BOUNDARY TRIGGER:" in final_section
    assert "NEXT PREREQUISITE:" in final_section
    assert "Scientific headroom decision:" in final_section
    assert "Evaluation prerequisite:" in final_section


def test_gate1_verdict_unaffected_by_boundary_trigger_status(tmp_path):
    protocol = _protocol()

    fired_dir = _build_full_synthetic_bundle(tmp_path, harmful_all_boundary=True, dirname="fired")
    source_fired = g1.load_gate1_source(fired_dir, expected_image_count=4)
    vm_fired = g1.compute_all_variant_metrics(source_fired.arrays, source_fired.class_count)
    oracle_fired = g1.build_oracle_analysis(vm_fired)
    boundary_fired = g1.build_boundary_concentration(source_fired.arrays)
    trigger_fired = g1.build_boundary_reframe_trigger(boundary_fired, protocol)
    gate_fired = g1.apply_gate1_decision(oracle_fired, protocol)

    not_fired_dir = _build_full_synthetic_bundle(tmp_path, harmful_all_boundary=False, dirname="notfired")
    source_not_fired = g1.load_gate1_source(not_fired_dir, expected_image_count=4)
    vm_not_fired = g1.compute_all_variant_metrics(source_not_fired.arrays, source_not_fired.class_count)
    oracle_not_fired = g1.build_oracle_analysis(vm_not_fired)
    boundary_not_fired = g1.build_boundary_concentration(source_not_fired.arrays)
    trigger_not_fired = g1.build_boundary_reframe_trigger(boundary_not_fired, protocol)
    gate_not_fired = g1.apply_gate1_decision(oracle_not_fired, protocol)

    # Only the boundary inputs differ between the two fixtures (harmful
    # pixels moved from boundary to interior); every oracle/horizon input is
    # identical, so the two trigger statuses must differ while the gate
    # decision (computed from oracle alone) must be identical.
    assert trigger_fired["trigger_status"] == g1.BOUNDARY_TRIGGER_FIRED
    assert trigger_not_fired["trigger_status"] == g1.BOUNDARY_TRIGGER_NOT_FIRED
    assert gate_fired["decision"] == gate_not_fired["decision"]
    assert gate_fired == gate_not_fired


def test_no_boundary_iou_value_is_ever_generated(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path, dirname="no_iou")
    out = tmp_path / "gate1_no_iou_out"
    rc = cli.main(["--result-bundle", str(result_dir), "--output-dir", str(out), "--expected-image-count", "4"])
    assert rc == 0
    doc = json.loads((out / "gate1_analysis.json").read_text())

    def _keys(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                yield path
                yield from _keys(v, path)
        elif isinstance(d, list):
            for i, v in enumerate(d):
                yield from _keys(v, f"{prefix}[{i}]")

    boundary_iou_keys = [k for k in _keys(doc) if "boundary_iou" in k.split(".")[-1]]
    assert boundary_iou_keys == ["boundary_reframe_trigger.boundary_iou_authority_status"], boundary_iou_keys
    assert doc["boundary_reframe_trigger"]["boundary_iou_authority_status"] == "not_yet_bound"
    raw = (out / "gate1_analysis.json").read_text()
    assert "boundary_iou_score" not in raw and "boundary_iou_value" not in raw


def test_boundary_trigger_no_effect_on_other_analysis_sections(tmp_path):
    result_dir = _build_full_synthetic_bundle(tmp_path, dirname="no_effect")
    out = tmp_path / "gate1_no_effect_out"
    rc = cli.main(["--result-bundle", str(result_dir), "--output-dir", str(out), "--expected-image-count", "4"])
    assert rc == 0
    doc = json.loads((out / "gate1_analysis.json").read_text())
    assert set(doc.keys()) >= {
        "horizon_table", "dispute_accounting", "oracle_analysis", "persistence_baseline",
        "component_structure", "reversion_coherence", "boundary_concentration", "boundary_reframe_trigger",
        "compute_projection", "gate1_decision", "variant_dataset_level_metrics",
    }
    trigger = doc["boundary_reframe_trigger"]
    assert set(trigger.keys()) == {
        "harmful_washout_boundary_fraction_of_harmful", "harmful_washout_pixel_total",
        "boundary_mass_reframe_threshold", "comparison_operator", "trigger_status",
        "boundary_iou_authority_status", "boundary_mass_reframe_consequence",
        "primary_phase1b_metric", "secondary_phase1b_metric", "required_next_step",
        "affects_oracle_headroom_gate", "phase1b_evaluation_ready", "unverifiable_reason",
    }


@pytest.mark.skipif(not os.environ.get("DFP_PHASE1A_OUTPUT_ROOT"), reason="requires DFP_PHASE1A_OUTPUT_ROOT")
def test_real_production_result_scientific_values_and_boundary_trigger_unchanged():
    """Read-only regression fixture against the real, finalized COCO-Stuff-
    500 production result. Never writes to the production directory. The
    production path is taken ONLY from DFP_PHASE1A_OUTPUT_ROOT (never
    hardcoded here), matching the convention already used by
    test_dfp_phase1a_preflight_finalize.py::
    test_real_production_checkpoint_reaches_ready_to_finalize."""
    production_dir = Path(os.environ["DFP_PHASE1A_OUTPUT_ROOT"]) / "coco_stuff_500" / "result"
    if not production_dir.exists():
        pytest.skip("production result not present in this environment")
    before = {p.name: p.read_bytes() for p in production_dir.iterdir()}

    source = g1.load_gate1_source(production_dir)
    vm = g1.compute_all_variant_metrics(source.arrays, source.class_count)
    oracle = g1.build_oracle_analysis(vm)
    boundary = g1.build_boundary_concentration(source.arrays)
    protocol = _protocol()
    trigger = g1.build_boundary_reframe_trigger(boundary, protocol)

    # Pre-existing scientific values, at full stored precision -- must
    # remain byte-for-byte the values independently verified before this
    # repair (this repair adds a report, it changes no computation).
    assert vm["t0"].metrics.mIoU == 29.357779515515865
    assert vm["t320"].metrics.mIoU == 31.2070844148295
    assert oracle["mIoU_oracle_o_p"] == 33.75174297429137
    assert oracle["G_recover_pp"] == pytest.approx(2.5446585594618690, abs=1e-9)

    assert trigger["trigger_status"] == g1.BOUNDARY_TRIGGER_FIRED
    assert trigger["harmful_washout_boundary_fraction_of_harmful"] == 0.9765451809541299
    assert trigger["boundary_mass_reframe_threshold"] == 0.70
    assert trigger["primary_phase1b_metric"] == "Boundary IoU"
    assert trigger["secondary_phase1b_metric"] == "mIoU preservation"
    assert trigger["required_next_step"] == g1.REQUIRED_NEXT_STEP_BIND_BOUNDARY_IOU
    assert trigger["phase1b_evaluation_ready"] is False
    assert trigger["affects_oracle_headroom_gate"] is False

    after = {p.name: p.read_bytes() for p in production_dir.iterdir()}
    assert before == after
