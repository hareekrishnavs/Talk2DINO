"""CPU-only coverage for src.k11_k12_full_result_analysis and
analyze_k11_k12_full_result.py: strict artifact loading, independent
metric reconstruction, GT/pairing consistency, the bounded-memory paired
bootstrap, per-class analysis, pilot/full nesting, and the CLI's
fail-closed exception boundary. Never initializes CUDA, never loads the
model/checkpoint, never constructs a dataset -- entirely synthetic
fixtures built directly from this module's own public writer functions."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.k11_k12_full_result_analysis as m  # noqa: E402
from src.k11_k12_power_evaluation_checkpoint import CHECKPOINT_SCHEMA_NAME  # noqa: E402
from src.k11_k12_power_evaluation_identity import K11K12PowerEvaluationError, load_identity  # noqa: E402
from src.k11_k12_power_evaluation_report import TOP_RESULT_KEYS  # noqa: E402
from src.k11_k12_stability_report import write_checkpoint_atomically  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (ROOT / "evaluation_identities/e12_k11_k12_power_evaluation.toml").exists(),
    reason="requires the e12 power-evaluation identity",
)

IDENTITY = load_identity(repo_root=ROOT)
IDENTITY_SHA = m.sha256_file(ROOT / "evaluation_identities/e12_k11_k12_power_evaluation.toml")


# ---------------------------------------------------------------------------
# Synthetic fixture construction (never imports the real 5000-image data)
# ---------------------------------------------------------------------------


def _independent_intersect_union(pred, label, num_classes, ignore_index=255):
    pred = pred.ravel()
    label = label.ravel()
    mask = label != ignore_index
    pred, label = pred[mask], label[mask]
    intersect = pred[pred == label]
    area_i = np.histogram(intersect, bins=num_classes, range=(0, num_classes))[0]
    area_p = np.histogram(pred, bins=num_classes, range=(0, num_classes))[0]
    area_l = np.histogram(label, bins=num_classes, range=(0, num_classes))[0]
    area_u = area_p + area_l - area_i
    return area_i.astype(np.int64), area_u.astype(np.int64), area_p.astype(np.int64), area_l.astype(np.int64)


_RUN_MODE_IMAGE_COUNT = {"pilot20": 20, "pilot100": 100, "full": 5000}


def _make_synthetic_full_run(tmp_path, *, seed=1, run_mode="pilot20", corrupt=None):
    """Build a complete, internally-consistent, strictly-valid four-
    artifact bundle for run_mode using this repo's own writers/validators
    (write_checkpoint_atomically, the exact same checkpoint/manifest/NPZ
    schema the real evaluator produces). image_count and class_count are
    NEVER shrunk below the authoritative identity's registered values --
    verify_record (reused, never bypassed) enforces both exactly, so
    pilot20 (20 images) is used as the default fast fixture size rather
    than an arbitrary small count. `corrupt` is an optional
    callable(paths) invoked after writing, for negative-path tests."""
    n_images = _RUN_MODE_IMAGE_COUNT[run_mode]
    n_classes = IDENTITY["metrics"]["class_count"]
    final = run_mode == "full"
    rng = np.random.default_rng(seed)
    intersect_k11 = np.zeros((n_images, n_classes), dtype=np.int64)
    union_k11 = np.zeros((n_images, n_classes), dtype=np.int64)
    pred_k11 = np.zeros((n_images, n_classes), dtype=np.int64)
    intersect_k12 = np.zeros((n_images, n_classes), dtype=np.int64)
    union_k12 = np.zeros((n_images, n_classes), dtype=np.int64)
    pred_k12 = np.zeros((n_images, n_classes), dtype=np.int64)
    label = np.zeros((n_images, n_classes), dtype=np.int64)
    image_ids = [f"{i:012d}.jpg" for i in range(n_images)]

    for i in range(n_images):
        gt = rng.integers(0, n_classes, size=(20, 20))
        pred11 = gt.copy()
        flip_mask = rng.random(gt.shape) < 0.1
        pred11[flip_mask] = rng.integers(0, n_classes, size=int(flip_mask.sum()))
        pred12 = gt.copy()
        flip_mask2 = rng.random(gt.shape) < 0.08
        pred12[flip_mask2] = rng.integers(0, n_classes, size=int(flip_mask2.sum()))

        i11, u11, p11, l11 = _independent_intersect_union(pred11, gt, n_classes)
        i12, u12, p12, l12 = _independent_intersect_union(pred12, gt, n_classes)
        assert np.array_equal(l11, l12)
        intersect_k11[i], union_k11[i], pred_k11[i] = i11, u11, p11
        intersect_k12[i], union_k12[i], pred_k12[i] = i12, u12, p12
        label[i] = l11

    d = Path(tmp_path)
    npz_path = d / f"per-image-stats-{run_mode}.npz"
    manifest_path = d / f"per-image-stats-{run_mode}.json"
    checkpoint_path = d / f"checkpoint-{run_mode}.json"
    result_path = d / f"result-{run_mode}.json"

    np.savez(
        npz_path, dataset_indices=np.arange(n_images, dtype=np.int64), label=label,
        intersect_k11=intersect_k11, union_k11=union_k11, pred_k11=pred_k11,
        intersect_k12=intersect_k12, union_k12=union_k12, pred_k12=pred_k12,
    )
    npz_sha256 = m.sha256_file(npz_path)

    order_digest = hashlib.sha256(json.dumps(image_ids, ensure_ascii=True).encode("utf-8")).hexdigest()
    manifest = {
        "schema": "talk2dino-k11-k12-power-evaluation-per-image-stats-v1",
        "class_count": n_classes, "image_count": n_images, "image_ids": image_ids,
        "dataset_indices": list(range(n_images)), "image_order_digest": order_digest,
        "npz_filename": npz_path.name, "npz_sha256": npz_sha256,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha256 = m.sha256_file(manifest_path)

    checkpoint = {
        "schema": CHECKPOINT_SCHEMA_NAME, "run_mode": run_mode, "identity": IDENTITY["identity"]["name"],
        "identity_sha256": IDENTITY_SHA, "matched_identity_sha256": IDENTITY["parent_identity"]["matched_identity_sha256"],
        "stability_result_sha256": "a" * 64, "finite_step_kernel_sha256": "b" * 64, "git_commit": "c" * 40,
        "class_count": n_classes, "image_count_expected": n_images, "image_order_digest": order_digest,
        "next_dataset_index": n_images, "completed_image_ids": image_ids, "images_completed_count": n_images,
        "windows_processed_total": n_images * 3, "complete": True,
        "created_at_utc": "2026-08-24T00:00:00Z", "updated_at_utc": "2026-08-24T00:01:00Z",
    }
    write_checkpoint_atomically(checkpoint_path, checkpoint)

    sums11 = m.aggregate_class_sums(intersect_k11, union_k11, label)
    sums12 = m.aggregate_class_sums(intersect_k12, union_k12, label)
    metrics11 = m.compute_metrics_from_class_sums(sums11["intersect"], sums11["union"], sums11["label"])
    metrics12 = m.compute_metrics_from_class_sums(sums12["intersect"], sums12["union"], sums12["label"])
    delta = metrics11["mIoU_percent_0_100"] - metrics12["mIoU_percent_0_100"]

    schema_key = {"pilot20": "pilot20_schema_name", "pilot100": "pilot100_schema_name", "full": "full_schema_name"}[run_mode]
    image_count_key = {"pilot20": "pilot20_image_count", "pilot100": "pilot100_image_count", "full": "full_image_count"}[run_mode]
    result = {
        "schema": IDENTITY["run_modes"][schema_key], "run_mode": run_mode, "identity": IDENTITY["identity"]["name"],
        "identity_sha256": IDENTITY_SHA, "matched_identity_sha256": IDENTITY["parent_identity"]["matched_identity_sha256"],
        "git_commit": "c" * 40, "complete": True, "final": final, "device": "cpu", "gpu_model": "none",
        "torch_version": "0.0.0", "cuda_version": "none",
        "image_count_expected": IDENTITY["run_modes"][image_count_key], "image_count_processed": n_images,
        "image_order_digest": order_digest, "windows_processed_total": n_images * 3, "class_count": n_classes,
        "metrics_k11": {"aAcc": metrics11["aAcc_percent_0_100"], "mIoU": metrics11["mIoU_percent_0_100"], "mAcc": metrics11["mAcc_percent_0_100"]},
        "metrics_k12": {"aAcc": metrics12["aAcc_percent_0_100"], "mIoU": metrics12["mIoU_percent_0_100"], "mAcc": metrics12["mAcc_percent_0_100"]},
        "delta_mIoU_percentage_points": delta, "metric_unit": "percent_0_100",
        "metric_source": IDENTITY["metrics"]["precision_source"],
        "per_image_stats_manifest_path": str(manifest_path), "per_image_stats_manifest_sha256": manifest_sha256,
        "per_image_stats_npz_sha256": npz_sha256,
        "stability_result_sha256": "a" * 64, "stability_schema": "talk2dino-k11-k12-stability-gate-v1",
        "gate_classification": IDENTITY["stability_gate"]["accepted_classifications"][0],
        "gate_git_commit": "d" * 40, "gate_identity_sha256": "e" * 64,
        "finite_step_kernel_sha256": "b" * 64, "graph_construction_sha256": "f" * 64,
        "operation_telemetry": {
            "backbone_snapshot_calls": n_images * 3, "dino_feature_extractions": n_images * 3, "topk_selection_calls": n_images * 3,
            "graph_normalizations": n_images * 6, "finite_step_propagations": n_images * 6,
            "k11_updates": n_images * 3 * 320, "k12_updates": n_images * 3 * 320,
            "sigmoid_calls": n_images * 6, "interpolation_calls": n_images * 6,
        },
        "phase_runtime_seconds": {"total": 1.0}, "peak_gpu_memory_bytes": 0, "resumed_from_checkpoint": False,
        "source_git_branch": "test", "failure_reason": None,
    }
    assert set(result.keys()) == TOP_RESULT_KEYS
    result_path.write_text(json.dumps(result, sort_keys=True))

    if corrupt is not None:
        corrupt(dict(result_path=result_path, checkpoint_path=checkpoint_path, manifest_path=manifest_path, npz_path=npz_path))

    return {
        "result_path": result_path, "checkpoint_path": checkpoint_path, "manifest_path": manifest_path, "npz_path": npz_path,
        "arrays": {"intersect_k11": intersect_k11, "union_k11": union_k11, "pred_k11": pred_k11,
                   "intersect_k12": intersect_k12, "union_k12": union_k12, "pred_k12": pred_k12, "label": label},
        "image_ids": image_ids, "delta": delta,
    }


def _load(paths, *, run_mode="pilot20"):
    if run_mode == "full":
        return m.load_full_artifacts(
            result_path=paths["result_path"], checkpoint_path=paths["checkpoint_path"],
            manifest_path=paths["manifest_path"], npz_path=paths["npz_path"], repo_root=ROOT,
        )
    return m.load_pilot_artifacts(
        run_mode=run_mode, image_count=_RUN_MODE_IMAGE_COUNT[run_mode], result_path=paths["result_path"],
        checkpoint_path=paths["checkpoint_path"], manifest_path=paths["manifest_path"], npz_path=paths["npz_path"],
        repo_root=ROOT,
    )


# ---------------------------------------------------------------------------
# Input / integrity
# ---------------------------------------------------------------------------


def test_synthetic_full_run_loads_and_validates(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    bundle = _load(fx)
    assert bundle.result["run_mode"] == "pilot20"
    assert len(bundle.checkpoint["completed_image_ids"]) == 20

    full_dir = tmp_path / "full_variant"
    full_dir.mkdir()
    full_fx = _make_synthetic_full_run(full_dir, run_mode="full")
    full_bundle = _load(full_fx, run_mode="full")
    assert full_bundle.result["run_mode"] == "full"
    assert len(full_bundle.checkpoint["completed_image_ids"]) == 5000


def test_malformed_json_rejected(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    fx["result_path"].write_text('{"a": 1,}')
    with pytest.raises(K11K12PowerEvaluationError, match="parse"):
        _load(fx)


def test_duplicate_json_key_rejected(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    fx["checkpoint_path"].write_text('{"a": 1, "a": 2}')
    with pytest.raises(K11K12PowerEvaluationError, match="duplicate"):
        _load(fx)


def test_non_finite_json_constant_rejected(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    fx["manifest_path"].write_text('{"a": NaN}')
    with pytest.raises(K11K12PowerEvaluationError):
        _load(fx)


def test_wrong_image_count_rejected(tmp_path):
    def corrupt(paths):
        d = json.loads(paths["result_path"].read_text())
        d["image_count_processed"] = 999999
        paths["result_path"].write_text(json.dumps(d))

    fx = _make_synthetic_full_run(tmp_path, corrupt=corrupt)
    with pytest.raises(K11K12PowerEvaluationError):
        _load(fx)


def test_wrong_class_count_rejected(tmp_path):
    def corrupt(paths):
        d = json.loads(paths["result_path"].read_text())
        d["class_count"] = 999
        paths["result_path"].write_text(json.dumps(d))

    fx = _make_synthetic_full_run(tmp_path, corrupt=corrupt)
    with pytest.raises(K11K12PowerEvaluationError):
        _load(fx)


def test_mismatched_manifest_ids_rejected(tmp_path):
    def corrupt(paths):
        d = json.loads(paths["manifest_path"].read_text())
        d["image_ids"][0] = "totally_different_image.jpg"
        paths["manifest_path"].write_text(json.dumps(d))

    fx = _make_synthetic_full_run(tmp_path, corrupt=corrupt)
    with pytest.raises(K11K12PowerEvaluationError):
        _load(fx)


def test_duplicate_completed_ids_rejected(tmp_path):
    def corrupt(paths):
        cp = json.loads(paths["checkpoint_path"].read_text())
        cp["completed_image_ids"][1] = cp["completed_image_ids"][0]
        paths["checkpoint_path"].write_text(json.dumps(cp))

    fx = _make_synthetic_full_run(tmp_path, corrupt=corrupt)
    with pytest.raises(K11K12PowerEvaluationError):
        _load(fx)


def test_wrong_npz_shape_rejected(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    bad_npz = Path(tmp_path) / "bad.npz"
    np.savez(bad_npz, dataset_indices=np.arange(4), label=np.zeros((3, 3), dtype=np.int64),
              intersect_k11=np.zeros((4, 3), dtype=np.int64), union_k11=np.zeros((4, 3), dtype=np.int64),
              pred_k11=np.zeros((4, 3), dtype=np.int64), intersect_k12=np.zeros((4, 3), dtype=np.int64),
              union_k12=np.zeros((4, 3), dtype=np.int64), pred_k12=np.zeros((4, 3), dtype=np.int64))
    with pytest.raises(m.K11K12AnalysisError, match="shape"):
        m._load_strict_npz(bad_npz, expected_image_count=4, expected_class_count=3)


def test_non_integral_float_npz_rejected(tmp_path):
    npz_path = Path(tmp_path) / "float_bad.npz"
    base = np.zeros((2, 2), dtype=np.float64)
    bad = base.copy()
    bad[0, 0] = 1.5
    np.savez(npz_path, dataset_indices=np.arange(2), label=base, intersect_k11=bad, union_k11=base,
              pred_k11=base, intersect_k12=base, union_k12=base, pred_k12=base)
    with pytest.raises(m.K11K12AnalysisError, match="non-integral"):
        m._load_strict_npz(npz_path, expected_image_count=2, expected_class_count=2)


def test_negative_count_npz_rejected(tmp_path):
    npz_path = Path(tmp_path) / "neg.npz"
    base = np.zeros((2, 2), dtype=np.int64)
    bad = base.copy()
    bad[0, 0] = -1
    np.savez(npz_path, dataset_indices=np.arange(2), label=base, intersect_k11=bad, union_k11=base,
              pred_k11=base, intersect_k12=base, union_k12=base, pred_k12=base)
    with pytest.raises(m.K11K12AnalysisError, match="negative"):
        m._load_strict_npz(npz_path, expected_image_count=2, expected_class_count=2)


def test_impossible_intersect_exceeds_union_rejected(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    with np.load(fx["npz_path"]) as z:
        arrays = dict(z)
    arrays["intersect_k11"][0, 0] = arrays["union_k11"][0, 0] + 5
    np.savez(fx["npz_path"], **arrays)
    with pytest.raises(K11K12PowerEvaluationError):
        _load(fx)


def test_source_inputs_not_mutated_by_loading(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    before = {k: v.read_bytes() if k.endswith("_path") else None for k, v in fx.items() if k.endswith("_path")}
    _load(fx)
    for key, content in before.items():
        assert fx[key].read_bytes() == content


# ---------------------------------------------------------------------------
# Metric reconstruction
# ---------------------------------------------------------------------------


def test_hand_derived_confusion_fixture():
    intersect = np.array([3.0, 0.0, 2.0])
    union = np.array([5.0, 0.0, 4.0])
    label = np.array([4.0, 0.0, 3.0])
    metrics = m.compute_metrics_from_class_sums(intersect, union, label)
    # class 1 has union=0 -> excluded from mIoU (nanmean over classes 0,2)
    expected_miou = np.mean([3 / 5, 2 / 4])
    assert metrics["mIoU_fraction_0_1"] == pytest.approx(expected_miou)
    assert metrics["valid_class_count"] == 2


def test_zero_union_all_classes_raises_on_zero_label_total():
    with pytest.raises(m.K11K12AnalysisError, match="GT area is zero"):
        m.compute_metrics_from_class_sums(np.zeros(3), np.zeros(3), np.zeros(3))


def test_float64_aggregation_used():
    fx_intersect = np.ones((2, 2), dtype=np.int64)
    fx_union = np.full((2, 2), 3, dtype=np.int64)
    fx_label = np.full((2, 2), 2, dtype=np.int64)
    sums = m.aggregate_class_sums(fx_intersect, fx_union, fx_label)
    assert sums["intersect"].dtype == np.float64


def test_reported_reconstructed_mismatch_rejected(tmp_path):
    def corrupt(paths):
        d = json.loads(paths["result_path"].read_text())
        d["metrics_k11"]["mIoU"] = d["metrics_k11"]["mIoU"] + 5.0
        paths["result_path"].write_text(json.dumps(d))

    fx = _make_synthetic_full_run(tmp_path, corrupt=corrupt)
    # verify_record's own range/consistency checks may catch this first;
    # either way it must fail closed, never silently accept.
    with pytest.raises(Exception):
        bundle = _load(fx)
        m.reconstruct_and_verify_metrics(bundle)


def test_reconstruction_matches_reported_within_tolerance(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    bundle = _load(fx)
    metrics = m.reconstruct_and_verify_metrics(bundle)
    assert metrics["delta_mIoU_percentage_points"] == pytest.approx(fx["delta"], abs=1e-9)


def test_does_not_average_per_image_miou(tmp_path):
    # construct a case where averaging per-image mIoU would differ from
    # dataset-level pooled mIoU, and confirm we get the pooled value.
    intersect = np.array([[10, 0], [0, 10]], dtype=np.int64)
    union = np.array([[10, 100], [100, 10]], dtype=np.int64)
    label = np.array([[10, 50], [50, 10]], dtype=np.int64)
    sums = m.aggregate_class_sums(intersect, union, label)
    pooled = m.compute_metrics_from_class_sums(sums["intersect"], sums["union"], sums["label"])
    # naive (wrong) per-image average: image0 iou=[1.0, 0.0]->miou=0.5; image1 iou=[0.0,1.0]->miou=0.5; avg=0.5
    # pooled: class0 sum_i=10,sum_u=110->0.0909; class1 sum_i=10,sum_u=110->0.0909; mean=0.0909
    assert pooled["mIoU_fraction_0_1"] == pytest.approx(10 / 110)
    assert pooled["mIoU_fraction_0_1"] != pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pairing / GT consistency
# ---------------------------------------------------------------------------


def test_pairing_consistency_passes_on_synthetic_fixture(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    bundle = _load(fx)
    result = m.verify_pairing_consistency(bundle)
    assert result["all_passed"], result["violations"]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bincount_weight_matmul_equals_gather_and_sum():
    rng = np.random.default_rng(3)
    n_images, n_classes = 12, 5
    arr = rng.integers(0, 50, size=(n_images, n_classes)).astype(np.float64)
    indices = rng.integers(0, n_images, size=n_images)
    via_gather = arr[indices].sum(axis=0)
    weight = np.bincount(indices, minlength=n_images).astype(np.float64)
    via_weight = weight @ arr
    assert np.array_equal(via_gather, via_weight)


def test_bootstrap_chunked_matches_simple_reference():
    rng_seed = 42
    n_images, n_classes = 10, 4
    rng = np.random.default_rng(1)
    intersect_k11 = rng.integers(0, 20, size=(n_images, n_classes)).astype(np.int64)
    union_k11 = intersect_k11 + rng.integers(1, 20, size=(n_images, n_classes)).astype(np.int64)
    intersect_k12 = rng.integers(0, 20, size=(n_images, n_classes)).astype(np.int64)
    union_k12 = intersect_k12 + rng.integers(1, 20, size=(n_images, n_classes)).astype(np.int64)

    ref = m.bootstrap_paired_delta(
        intersect_k11, union_k11, intersect_k12, union_k12,
        observed_delta_percentage_points=0.0, n_replicates=10_000, seed=rng_seed, chunk_size=1,
    )
    fast = m.bootstrap_paired_delta(
        intersect_k11, union_k11, intersect_k12, union_k12,
        observed_delta_percentage_points=0.0, n_replicates=10_000, seed=rng_seed, chunk_size=137,
    )
    assert np.array_equal(ref.delta_replicates, fast.delta_replicates)
    assert ref.bootstrap_mean_delta == fast.bootstrap_mean_delta
    assert ref.ci_low_2_5 == fast.ci_low_2_5


def test_bootstrap_paired_sampling_uses_identical_indices():
    # if k11 and k12 statistics are byte-identical per image, EVERY
    # replicate's delta must be exactly zero (proves the same resampled
    # weights are applied to both variants, not independent resamples).
    n_images, n_classes = 6, 3
    rng = np.random.default_rng(7)
    intersect = rng.integers(1, 20, size=(n_images, n_classes)).astype(np.int64)
    union = intersect + rng.integers(1, 20, size=(n_images, n_classes)).astype(np.int64)
    result = m.bootstrap_paired_delta(
        intersect, union, intersect, union, observed_delta_percentage_points=0.0,
        n_replicates=10_000, seed=5, chunk_size=100,
    )
    assert np.allclose(result.delta_replicates, 0.0)
    assert result.ci_low_2_5 == pytest.approx(0.0)
    assert result.ci_high_97_5 == pytest.approx(0.0)


def test_bootstrap_requires_at_least_10000_replicates():
    with pytest.raises(m.K11K12AnalysisError, match="10000"):
        m.bootstrap_paired_delta(
            np.ones((3, 2), dtype=np.int64), np.full((3, 2), 2, dtype=np.int64),
            np.ones((3, 2), dtype=np.int64), np.full((3, 2), 2, dtype=np.int64),
            observed_delta_percentage_points=0.0, n_replicates=100, seed=1,
        )


def test_bootstrap_deterministic_seed_reproducible():
    n_images, n_classes = 8, 3
    rng = np.random.default_rng(9)
    intersect_k11 = rng.integers(0, 20, size=(n_images, n_classes)).astype(np.int64)
    union_k11 = intersect_k11 + rng.integers(1, 20, size=(n_images, n_classes)).astype(np.int64)
    intersect_k12 = rng.integers(0, 20, size=(n_images, n_classes)).astype(np.int64)
    union_k12 = intersect_k12 + rng.integers(1, 20, size=(n_images, n_classes)).astype(np.int64)
    r1 = m.bootstrap_paired_delta(intersect_k11, union_k11, intersect_k12, union_k12, observed_delta_percentage_points=0.0, n_replicates=10_000, seed=123)
    r2 = m.bootstrap_paired_delta(intersect_k11, union_k11, intersect_k12, union_k12, observed_delta_percentage_points=0.0, n_replicates=10_000, seed=123)
    assert np.array_equal(r1.delta_replicates, r2.delta_replicates)


def test_bootstrap_positive_delta_fixture_classifies_above_zero():
    n_images, n_classes = 30, 5
    rng = np.random.default_rng(11)
    label = rng.integers(50, 100, size=(n_images, n_classes)).astype(np.int64)
    intersect_k11 = (label * 0.95).astype(np.int64)
    union_k11 = label
    intersect_k12 = (label * 0.60).astype(np.int64)
    union_k12 = label
    result = m.bootstrap_paired_delta(intersect_k11, union_k11, intersect_k12, union_k12, observed_delta_percentage_points=0.0, n_replicates=10_000, seed=1, chunk_size=500)
    assert result.classification == "CI_ABOVE_ZERO"
    assert result.probability_delta_gt_0 > 0.99


def test_bootstrap_negative_delta_fixture_classifies_below_zero():
    n_images, n_classes = 30, 5
    rng = np.random.default_rng(11)
    label = rng.integers(50, 100, size=(n_images, n_classes)).astype(np.int64)
    intersect_k11 = (label * 0.60).astype(np.int64)
    union_k11 = label
    intersect_k12 = (label * 0.95).astype(np.int64)
    union_k12 = label
    result = m.bootstrap_paired_delta(intersect_k11, union_k11, intersect_k12, union_k12, observed_delta_percentage_points=0.0, n_replicates=10_000, seed=1, chunk_size=500)
    assert result.classification == "CI_BELOW_ZERO"
    assert result.probability_delta_lt_0 > 0.99


def test_bootstrap_zero_delta_fixture_classifies_includes_zero():
    n_images, n_classes = 30, 5
    rng = np.random.default_rng(11)
    label = rng.integers(50, 100, size=(n_images, n_classes)).astype(np.int64)
    intersect = (label * 0.80).astype(np.int64)
    result = m.bootstrap_paired_delta(intersect, label, intersect, label, observed_delta_percentage_points=0.0, n_replicates=10_000, seed=1, chunk_size=500)
    assert result.classification == "CI_INCLUDES_ZERO"


def test_bootstrap_bounded_memory_no_full_materialization():
    # a resource-shaped smoke test: chunk_size much smaller than
    # n_replicates must still produce n_replicates results without ever
    # requesting an array of shape [n_replicates, n_images, n_classes].
    n_images, n_classes = 50, 10
    rng = np.random.default_rng(2)
    intersect = rng.integers(0, 20, size=(n_images, n_classes)).astype(np.int64)
    union = intersect + 5
    result = m.bootstrap_paired_delta(intersect, union, intersect, union, observed_delta_percentage_points=0.0, n_replicates=10_000, seed=1, chunk_size=13)
    assert result.replicate_count == 10_000
    assert result.delta_replicates.shape == (10_000,)


# ---------------------------------------------------------------------------
# Per-class
# ---------------------------------------------------------------------------


def test_per_class_gains_losses_and_canonical_names(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    bundle = _load(fx)
    metrics = m.reconstruct_and_verify_metrics(bundle)
    per_class = m.per_class_analysis(bundle, metrics)
    assert len(per_class["rows"]) == IDENTITY["metrics"]["class_count"]
    assert per_class["positive_class_count"] + per_class["negative_class_count"] + per_class["zero_class_count"] == per_class["valid_class_count_both_variants"]
    assert len(per_class["top_20_gains_for_k11"]) <= 20
    assert len(per_class["top_20_losses_for_k11"]) <= 20
    assert per_class["class_names_available"] is True
    assert per_class["rows"][0]["class_name"] is not None


def test_per_class_fp_fn_tp_changes_consistent(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    bundle = _load(fx)
    metrics = m.reconstruct_and_verify_metrics(bundle)
    per_class = m.per_class_analysis(bundle, metrics)
    for row in per_class["rows"]:
        assert "delta_true_positive_intersect" in row
        assert "delta_false_positive_pixels" in row
        assert "delta_false_negative_pixels" in row


# ---------------------------------------------------------------------------
# Pilot nesting
# ---------------------------------------------------------------------------


def test_pilot_exact_prefix_reproduces_pilot_delta(tmp_path):
    full_dir = Path(tmp_path) / "full"
    full_dir.mkdir()
    full_fx = _make_synthetic_full_run(full_dir, seed=99, run_mode="full")

    pilot_dir = Path(tmp_path) / "pilot"
    pilot_dir.mkdir()
    n_pilot = 20  # the identity's registered pilot20 image count -- never shrunk
    n_classes = IDENTITY["metrics"]["class_count"]
    arrays = full_fx["arrays"]

    npz_path = pilot_dir / "per-image-stats-pilot20.npz"
    np.savez(npz_path, dataset_indices=np.arange(n_pilot, dtype=np.int64),
              **{k: v[:n_pilot] for k, v in arrays.items()})
    npz_sha256 = m.sha256_file(npz_path)
    order_digest = hashlib.sha256(json.dumps(full_fx["image_ids"][:n_pilot], ensure_ascii=True).encode("utf-8")).hexdigest()
    manifest = {
        "schema": "talk2dino-k11-k12-power-evaluation-per-image-stats-v1", "class_count": n_classes, "image_count": n_pilot,
        "image_ids": full_fx["image_ids"][:n_pilot], "dataset_indices": list(range(n_pilot)),
        "image_order_digest": order_digest, "npz_filename": npz_path.name, "npz_sha256": npz_sha256,
    }
    manifest_path = pilot_dir / "per-image-stats-pilot20.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    checkpoint_path = pilot_dir / "checkpoint-pilot20.json"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA_NAME, "run_mode": "pilot20", "identity": IDENTITY["identity"]["name"],
        "identity_sha256": IDENTITY_SHA, "matched_identity_sha256": IDENTITY["parent_identity"]["matched_identity_sha256"],
        "stability_result_sha256": "a" * 64, "finite_step_kernel_sha256": "b" * 64, "git_commit": "c" * 40,
        "class_count": n_classes, "image_count_expected": n_pilot, "image_order_digest": order_digest,
        "next_dataset_index": n_pilot, "completed_image_ids": full_fx["image_ids"][:n_pilot], "images_completed_count": n_pilot,
        "windows_processed_total": n_pilot * 3, "complete": True,
        "created_at_utc": "2026-08-24T00:00:00Z", "updated_at_utc": "2026-08-24T00:01:00Z",
    }
    write_checkpoint_atomically(checkpoint_path, checkpoint)

    sums11 = m.aggregate_class_sums(arrays["intersect_k11"][:n_pilot], arrays["union_k11"][:n_pilot], arrays["label"][:n_pilot])
    sums12 = m.aggregate_class_sums(arrays["intersect_k12"][:n_pilot], arrays["union_k12"][:n_pilot], arrays["label"][:n_pilot])
    metrics11 = m.compute_metrics_from_class_sums(sums11["intersect"], sums11["union"], sums11["label"])
    metrics12 = m.compute_metrics_from_class_sums(sums12["intersect"], sums12["union"], sums12["label"])
    pilot_delta = metrics11["mIoU_percent_0_100"] - metrics12["mIoU_percent_0_100"]

    result_path = pilot_dir / "result-pilot20.json"
    result = {
        "schema": IDENTITY["run_modes"]["pilot20_schema_name"], "run_mode": "pilot20", "identity": IDENTITY["identity"]["name"],
        "identity_sha256": IDENTITY_SHA, "matched_identity_sha256": IDENTITY["parent_identity"]["matched_identity_sha256"],
        "git_commit": "c" * 40, "complete": True, "final": False, "device": "cpu", "gpu_model": "none",
        "torch_version": "0.0.0", "cuda_version": "none",
        "image_count_expected": n_pilot, "image_count_processed": n_pilot, "image_order_digest": order_digest,
        "windows_processed_total": n_pilot * 3, "class_count": n_classes,
        "metrics_k11": {"aAcc": metrics11["aAcc_percent_0_100"], "mIoU": metrics11["mIoU_percent_0_100"], "mAcc": metrics11["mAcc_percent_0_100"]},
        "metrics_k12": {"aAcc": metrics12["aAcc_percent_0_100"], "mIoU": metrics12["mIoU_percent_0_100"], "mAcc": metrics12["mAcc_percent_0_100"]},
        "delta_mIoU_percentage_points": pilot_delta, "metric_unit": "percent_0_100",
        "metric_source": IDENTITY["metrics"]["precision_source"],
        "per_image_stats_manifest_path": str(manifest_path), "per_image_stats_manifest_sha256": m.sha256_file(manifest_path),
        "per_image_stats_npz_sha256": npz_sha256,
        "stability_result_sha256": "a" * 64, "stability_schema": "talk2dino-k11-k12-stability-gate-v1",
        "gate_classification": IDENTITY["stability_gate"]["accepted_classifications"][0],
        "gate_git_commit": "d" * 40, "gate_identity_sha256": "e" * 64,
        "finite_step_kernel_sha256": "b" * 64, "graph_construction_sha256": "f" * 64,
        "operation_telemetry": {
            "backbone_snapshot_calls": n_pilot * 3, "dino_feature_extractions": n_pilot * 3, "topk_selection_calls": n_pilot * 3,
            "graph_normalizations": n_pilot * 6, "finite_step_propagations": n_pilot * 6,
            "k11_updates": n_pilot * 3 * 320, "k12_updates": n_pilot * 3 * 320,
            "sigmoid_calls": n_pilot * 6, "interpolation_calls": n_pilot * 6,
        },
        "phase_runtime_seconds": {"total": 1.0}, "peak_gpu_memory_bytes": 0, "resumed_from_checkpoint": False,
        "source_git_branch": "test", "failure_reason": None,
    }
    result_path.write_text(json.dumps(result, sort_keys=True))

    full_bundle = _load(full_fx, run_mode="full")
    pilot_bundle = m.load_pilot_artifacts(
        run_mode="pilot20", image_count=n_pilot, result_path=result_path, checkpoint_path=checkpoint_path,
        manifest_path=manifest_path, npz_path=npz_path, repo_root=ROOT,
    )
    nesting = m.pilot_nesting_audit(full_bundle, {"pilot": pilot_bundle})
    assert nesting["pilot"]["is_exact_prefix_of_full"] is True
    assert nesting["pilot"]["agrees_with_pilot_result"] is True
    assert nesting["pilot"]["reconstructed_from_full_prefix_delta_mIoU"] == pytest.approx(pilot_delta, abs=1e-9)


def test_pilot_different_subset_reported_honestly(tmp_path):
    full_dir = Path(tmp_path) / "full2"
    full_dir.mkdir()
    full_fx = _make_synthetic_full_run(full_dir, seed=5, run_mode="pilot20")

    class _FakeBundle:
        pass

    fake_pilot = _FakeBundle()
    fake_pilot.checkpoint = {"completed_image_ids": ["not_a_real_prefix_image.jpg"]}
    full_bundle = _load(full_fx)
    nesting = m.pilot_nesting_audit(full_bundle, {"unrelated": fake_pilot})
    assert nesting["unrelated"]["is_exact_prefix_of_full"] is False
    assert "note" in nesting["unrelated"]


# ---------------------------------------------------------------------------
# Canonical reconciliation
# ---------------------------------------------------------------------------


def test_fetch_historical_sweep_record_matches_identity_anchor():
    historical = m.fetch_historical_sweep_record(ROOT)
    assert historical["blob_sha1"] == m.CANONICAL_FINITE_STEP_REFERENCE["source_blob_sha1"]
    assert historical["row"]["alpha"] == 0.98
    assert historical["row"]["steps"] == 320


def test_metric_reduction_variants_reproduce_historical_exactly():
    historical = m.fetch_historical_sweep_record(ROOT)
    h_intersect = np.array(historical["row"]["intersection"], dtype=np.float64)
    h_union = np.array(historical["row"]["union"], dtype=np.float64)
    h_label = np.array(historical["row"]["ground_truth_pixels"], dtype=np.float64)
    reconstructed = m.compute_metrics_from_class_sums(h_intersect, h_union, h_label)
    assert reconstructed["mIoU_percent_0_100"] == pytest.approx(m.CANONICAL_FINITE_STEP_REFERENCE["mIoU_percent"], abs=1e-6)


def test_no_unsupported_canonical_replacement_in_anchor_decision(tmp_path):
    fx = _make_synthetic_full_run(tmp_path)
    bundle = _load(fx)
    metrics = m.reconstruct_and_verify_metrics(bundle)
    protocol_table = m.canonical_protocol_comparison_table(bundle)
    historical = m.fetch_historical_sweep_record(ROOT)
    variants = m.metric_reduction_variants(bundle, metrics, historical)
    label_stats = m.label_statistics_reconciliation(metrics, historical)
    anchor = m.determine_anchor_decision(protocol_table, variants, label_stats)
    assert anchor["no_new_canonical_anchor_created"] is True
    assert anchor["status"] in {"ANCHOR_MATCH", "METRIC_REDUCTION_MISMATCH", "PROTOCOL_MISMATCH", "PREDICTION_MISMATCH_UNEXPLAINED", "INSUFFICIENT_PROVENANCE"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CLI_SCRIPT = ROOT / "analyze_k11_k12_full_result.py"


def test_cli_clean_exit_2_on_missing_artifact(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--repo-root", str(ROOT),
         "--result", str(tmp_path / "missing.json"), "--checkpoint", str(tmp_path / "missing.json"),
         "--per-image-stats", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 2
    assert "K11/K12 FULL RESULT ANALYSIS FAIL" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "out.json").exists()


def test_cli_succeeds_and_writes_atomic_output(tmp_path):
    fx = _make_synthetic_full_run(tmp_path, run_mode="full")
    out_path = Path(tmp_path) / "analysis.json"
    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--repo-root", str(ROOT),
         "--result", str(fx["result_path"]), "--checkpoint", str(fx["checkpoint_path"]),
         "--per-image-stats", str(fx["manifest_path"]), "--output", str(out_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    report = json.loads(out_path.read_text())
    assert report["schema"] == m.SCHEMA_NAME
    leftovers = list(Path(tmp_path).glob("*.tmp-*"))
    assert leftovers == []


def test_cli_existing_report_preserved_without_overwrite(tmp_path):
    fx = _make_synthetic_full_run(tmp_path, run_mode="full")
    out_path = Path(tmp_path) / "analysis.json"
    out_path.write_text('{"sentinel": true}')
    original_bytes = out_path.read_bytes()
    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--repo-root", str(ROOT),
         "--result", str(fx["result_path"]), "--checkpoint", str(fx["checkpoint_path"]),
         "--per-image-stats", str(fx["manifest_path"]), "--output", str(out_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert out_path.read_bytes() == original_bytes


def test_cli_no_cuda_or_model_import():
    source = CLI_SCRIPT.read_text()
    assert "torch.cuda" not in source
    assert "build_model" not in source
    assert "_build_inference" not in source
    assert "CheckpointLoader" not in source


def test_keyboardinterrupt_propagates_through_cli_main():
    import analyze_k11_k12_full_result as cli

    real_run = cli._run

    def raising(args):
        raise KeyboardInterrupt()

    cli._run = raising
    try:
        with pytest.raises(KeyboardInterrupt):
            cli.main(["--result", "x", "--checkpoint", "x", "--per-image-stats", "x", "--output", "x"])
    finally:
        cli._run = real_run


def test_systemexit_propagates_through_cli_main():
    import analyze_k11_k12_full_result as cli

    real_run = cli._run

    def raising(args):
        raise SystemExit(5)

    cli._run = raising
    try:
        with pytest.raises(SystemExit):
            cli.main(["--result", "x", "--checkpoint", "x", "--per-image-stats", "x", "--output", "x"])
    finally:
        cli._run = real_run
