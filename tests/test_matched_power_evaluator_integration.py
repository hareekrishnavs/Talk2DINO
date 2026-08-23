"""Integration-level tests for the matched k11/k12 power evaluator: identity
loading, stability-gate result binding (against the real job 20300858
result), checkpoint/result schema verification, per-image-stats artifact
round-tripping, output-path safety, and CLI fail-closed behavior.

These never construct a model, dataset, or CUDA context -- they exercise
the identity/report/CLI modules directly with real or synthetic JSON
fixtures, and the real (already-committed) stability-gate result from GPU
job 20300858 where a genuine end-to-end binding proof is valuable.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.k11_k12_power_evaluation_identity import (  # noqa: E402
    K11K12PowerEvaluationError,
    load_identity,
    validate_stability_result_binding,
    validate_static_configuration,
)
from src.k11_k12_power_evaluation_report import verify_record  # noqa: E402
from src.k11_k12_power_evaluation_checkpoint import (  # noqa: E402
    validate_checkpoint_structure,
    resume_dataset_index,
)

REAL_STABILITY_RESULT = Path("/scratch/haree/e12_k11_k12_stability/result-20300858.json")


def _identity():
    return load_identity(repo_root=ROOT)


# ---------------------------------------------------------------------------
# Identity / preflight
# ---------------------------------------------------------------------------


def test_identity_loads_and_validates():
    identity = _identity()
    assert identity["identity"]["name"] == "e12-k11-k12-power-evaluation"


def test_static_configuration_preflight_passes():
    result = validate_static_configuration(repo_root=ROOT, check_git=True)
    assert result["pilot20_image_count"] == 20
    assert result["pilot100_image_count"] == 100
    assert result["full_image_count"] == 5000


# ---------------------------------------------------------------------------
# 17-20: stability-result binding accept/reject
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_17_real_stability_result_binds_successfully():
    identity = _identity()
    binding = validate_stability_result_binding(REAL_STABILITY_RESULT, identity=identity, repo_root=ROOT, check_git=True)
    assert binding["gate_classification"] == "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE"
    assert binding["gate_git_commit"] == "47584d4314c87bfc5389cf2beb937b79b1f327a9"
    assert len(binding["finite_step_kernel_sha256"]) == 64
    assert len(binding["graph_construction_sha256"]) == 64


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_18_invalid_classification_rejected(tmp_path):
    identity = _identity()
    record = json.loads(REAL_STABILITY_RESULT.read_text())
    record["gate_classification"] = "INVALID"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps(record))
    with pytest.raises(K11K12PowerEvaluationError):
        validate_stability_result_binding(corrupt, identity=identity, repo_root=ROOT, check_git=True)


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_18b_truncation_sensitive_classification_rejected(tmp_path):
    identity = _identity()
    record = json.loads(REAL_STABILITY_RESULT.read_text())
    record["gate_classification"] = "TRUNCATION_SENSITIVE"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps(record))
    with pytest.raises(K11K12PowerEvaluationError):
        validate_stability_result_binding(corrupt, identity=identity, repo_root=ROOT, check_git=True)


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_19_accepted_inconclusive_classification_is_permitted():
    identity = _identity()
    assert "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE" in identity["stability_gate"]["accepted_classifications"]
    binding = validate_stability_result_binding(REAL_STABILITY_RESULT, identity=identity, repo_root=ROOT, check_git=True)
    assert binding["gate_classification"] in identity["stability_gate"]["accepted_classifications"]


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_20_kernel_hash_mismatch_rejected(monkeypatch):
    """Simulate the finite-step kernel having changed since the gate ran:
    the gate's recorded git_commit is real and unchanged, so this mocks
    the historical (git-show-at-commit) hash lookup to return a value that
    disagrees with the real current file's hash -- directly exercising the
    mismatch branch without touching any real tracked file."""
    import src.k11_k12_power_evaluation_identity as identity_module

    identity = _identity()
    real_hash_blob_at_commit = identity_module._hash_blob_at_commit

    def _fake_hash_blob_at_commit(root, commit, relative_path):
        if relative_path == identity_module.KERNEL_MODULE_RELATIVE_PATH:
            return "0" * 64  # deliberately wrong, simulating a changed kernel
        return real_hash_blob_at_commit(root, commit, relative_path)

    monkeypatch.setattr(identity_module, "_hash_blob_at_commit", _fake_hash_blob_at_commit)
    with pytest.raises(K11K12PowerEvaluationError, match="kernel source has changed"):
        validate_stability_result_binding(REAL_STABILITY_RESULT, identity=identity, repo_root=ROOT, check_git=True)


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_20b_graph_construction_hash_mismatch_rejected(monkeypatch):
    import src.k11_k12_power_evaluation_identity as identity_module

    identity = _identity()
    real_hash_blob_at_commit = identity_module._hash_blob_at_commit

    def _fake_hash_blob_at_commit(root, commit, relative_path):
        if relative_path == identity_module.GRAPH_MODULE_RELATIVE_PATH:
            return "0" * 64
        return real_hash_blob_at_commit(root, commit, relative_path)

    monkeypatch.setattr(identity_module, "_hash_blob_at_commit", _fake_hash_blob_at_commit)
    with pytest.raises(K11K12PowerEvaluationError, match="graph construction source has changed"):
        validate_stability_result_binding(REAL_STABILITY_RESULT, identity=identity, repo_root=ROOT, check_git=True)


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_failure_reason_must_be_null(tmp_path):
    identity = _identity()
    record = json.loads(REAL_STABILITY_RESULT.read_text())
    record["failure_reason"] = "some failure"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps(record))
    with pytest.raises(K11K12PowerEvaluationError, match="failure_reason"):
        validate_stability_result_binding(corrupt, identity=identity, repo_root=ROOT, check_git=True)


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_incomplete_result_rejected(tmp_path):
    identity = _identity()
    record = json.loads(REAL_STABILITY_RESULT.read_text())
    record["complete"] = False
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps(record))
    with pytest.raises(K11K12PowerEvaluationError):
        validate_stability_result_binding(corrupt, identity=identity, repo_root=ROOT, check_git=True)


@pytest.mark.skipif(not REAL_STABILITY_RESULT.exists(), reason="real GPU job result not present on this machine")
def test_checkpoint_masquerading_as_result_rejected(tmp_path):
    """A checkpoint has a different schema/key-set than a result; passing
    one as --stability-result must fail schema validation, never be
    silently accepted as though it were a completed result."""
    identity = _identity()
    fake_checkpoint = {
        "schema": "talk2dino-k11-k12-stability-checkpoint-v1",
        "identity": "e12-k11-k12-stability-gate",
        "identity_sha256": "0" * 64,
        "manifest_digest": "0" * 64,
        "windows_expected": 100,
        "complete": False,
        "windows": [],
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "updated_at_utc": "2026-01-01T00:00:00+00:00",
    }
    corrupt = tmp_path / "checkpoint_as_result.json"
    corrupt.write_text(json.dumps(fake_checkpoint))
    with pytest.raises(K11K12PowerEvaluationError):
        validate_stability_result_binding(corrupt, identity=identity, repo_root=ROOT, check_git=True)


def test_malformed_json_rejected(tmp_path):
    identity = _identity()
    corrupt = tmp_path / "malformed.json"
    corrupt.write_text("{not valid json")
    with pytest.raises(K11K12PowerEvaluationError):
        validate_stability_result_binding(corrupt, identity=identity, repo_root=ROOT, check_git=True)


# ---------------------------------------------------------------------------
# 16, 21-24: schema separation, checkpoint/resume, rollback, corruption
# ---------------------------------------------------------------------------


def _base_result(run_mode="pilot20"):
    identity = _identity()
    schema_key = {"pilot20": "pilot20_schema_name", "pilot100": "pilot100_schema_name", "full": "full_schema_name"}[run_mode]
    return {
        "schema": identity["run_modes"][schema_key],
        "run_mode": run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": "a" * 64,
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "git_commit": "b" * 40,
        "complete": True,
        "final": run_mode == "full",
        "device": "cuda",
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "torch_version": "2.12.0",
        "cuda_version": "13.2",
        "image_count_expected": identity["run_modes"][{"pilot20": "pilot20_image_count", "pilot100": "pilot100_image_count", "full": "full_image_count"}[run_mode]],
        "image_count_processed": identity["run_modes"][{"pilot20": "pilot20_image_count", "pilot100": "pilot100_image_count", "full": "full_image_count"}[run_mode]],
        "image_order_digest": "c" * 64,
        "windows_processed_total": 10,
        "class_count": 171,
        "metrics_k11": {"aAcc": 50.0, "mIoU": 30.0, "mAcc": 55.0},
        "metrics_k12": {"aAcc": 49.0, "mIoU": 29.5, "mAcc": 54.0},
        "delta_mIoU_percentage_points": 0.5,
        "metric_unit": "percent_0_100",
        "metric_source": "full_precision_area_statistics_from_mmseg_pre_eval",
        "per_image_stats_manifest_path": "/scratch/haree/x/per_image_stats.json",
        "per_image_stats_manifest_sha256": "d" * 64,
        "per_image_stats_npz_sha256": "e" * 64,
        "stability_result_sha256": "f" * 64,
        "stability_schema": "talk2dino-k11-k12-stability-gate-v1",
        "gate_classification": "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE",
        "gate_git_commit": "0" * 40,
        "gate_identity_sha256": "1" * 64,
        "finite_step_kernel_sha256": "2" * 64,
        "graph_construction_sha256": "3" * 64,
        "operation_telemetry": {
            "backbone_snapshot_calls": 10, "dino_feature_extractions": 10, "topk_selection_calls": 10,
            "graph_normalizations": 20, "finite_step_propagations": 20, "k11_updates": 3200, "k12_updates": 3200,
            "sigmoid_calls": 20, "interpolation_calls": 20,
        },
        "phase_runtime_seconds": {"total": 12.5},
        "peak_gpu_memory_bytes": 1024,
        "resumed_from_checkpoint": False,
        "source_git_branch": "e12-connectivity-analysis",
        "failure_reason": None,
    }


def test_16_pilot_result_rejected_against_full_schema():
    identity = _identity()
    record = _base_result("pilot20")
    record["schema"] = identity["run_modes"]["full_schema_name"]  # tampered to claim full
    with pytest.raises(K11K12PowerEvaluationError):
        verify_record(record, identity, identity_sha256="a" * 64)


def test_16b_pilot_result_with_final_true_rejected():
    identity = _identity()
    record = _base_result("pilot20")
    record["final"] = True
    with pytest.raises(K11K12PowerEvaluationError):
        verify_record(record, identity, identity_sha256="a" * 64)


def test_valid_pilot20_result_passes():
    identity = _identity()
    record = _base_result("pilot20")
    message = verify_record(record, identity, identity_sha256="a" * 64)
    assert "PASS" in message


def test_delta_inconsistent_with_metrics_rejected():
    identity = _identity()
    record = _base_result("pilot20")
    record["delta_mIoU_percentage_points"] = 99.0
    with pytest.raises(K11K12PowerEvaluationError):
        verify_record(record, identity, identity_sha256="a" * 64)


def test_telemetry_inconsistent_with_windows_processed_rejected():
    identity = _identity()
    record = _base_result("pilot20")
    record["operation_telemetry"]["k11_updates"] = 1  # should be 320 * windows_processed_total
    with pytest.raises(K11K12PowerEvaluationError):
        verify_record(record, identity, identity_sha256="a" * 64)


def _base_checkpoint(run_mode="pilot20", *, next_index=0, completed=None, complete=False, class_count=171):
    identity = _identity()
    completed = completed if completed is not None else [f"img{i}.jpg" for i in range(next_index)]
    return {
        "schema": "talk2dino-k11-k12-power-evaluation-checkpoint-v1",
        "run_mode": run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": "a" * 64,
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "stability_result_sha256": "f" * 64,
        "finite_step_kernel_sha256": "2" * 64,
        "git_commit": "b" * 40,
        "class_count": class_count,
        "image_count_expected": identity["run_modes"]["pilot20_image_count"],
        "image_order_digest": "c" * 64,
        "next_dataset_index": next_index,
        "completed_image_ids": completed,
        "images_completed_count": len(completed),
        "windows_processed_total": 3 * next_index,
        "complete": complete,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "updated_at_utc": "2026-01-01T00:00:01+00:00",
    }


def test_21_valid_incomplete_checkpoint_passes():
    identity = _identity()
    record = _base_checkpoint(next_index=5)
    validate_checkpoint_structure(record, identity=identity, identity_sha256="a" * 64)  # must not raise
    assert resume_dataset_index(record) == 5


def test_21b_complete_checkpoint_cannot_resume():
    identity = _identity()
    record = _base_checkpoint(next_index=20, completed=[f"img{i}.jpg" for i in range(20)], complete=True)
    validate_checkpoint_structure(record, identity=identity, identity_sha256="a" * 64)  # structurally valid
    with pytest.raises(K11K12PowerEvaluationError):
        resume_dataset_index(record)


def test_22_partial_image_state_rejected():
    """next_dataset_index disagreeing with len(completed_image_ids) would
    mean a partially-processed image was checkpointed -- rejected. This is
    the exact independently-confirmed regression: next_dataset_index=6 with
    only 5 completed images must never be accepted."""
    identity = _identity()
    record = _base_checkpoint(next_index=5)
    record["next_dataset_index"] = 6  # one more than completed_image_ids actually has
    with pytest.raises(K11K12PowerEvaluationError, match="next_dataset_index"):
        validate_checkpoint_structure(record, identity=identity, identity_sha256="a" * 64)


def test_23_images_completed_count_mismatch_rejected():
    identity = _identity()
    record = _base_checkpoint(next_index=5)
    record["images_completed_count"] = 4
    with pytest.raises(K11K12PowerEvaluationError):
        validate_checkpoint_structure(record, identity=identity, identity_sha256="a" * 64)


def test_24_complete_checkpoint_with_wrong_image_count_rejected():
    identity = _identity()
    record = _base_checkpoint(next_index=5, complete=True)  # complete but not full 20
    with pytest.raises(K11K12PowerEvaluationError):
        validate_checkpoint_structure(record, identity=identity, identity_sha256="a" * 64)


# ---------------------------------------------------------------------------
# 25: output path safety (reuses diagnostics.run_k11_k12_stability's own
# helper directly -- never reimplemented)
# ---------------------------------------------------------------------------


def test_25_tracked_output_path_rejected():
    from diagnostics.run_k11_k12_stability import _reject_tracked_output_path
    from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

    with pytest.raises(K11K12StabilityGateError):
        _reject_tracked_output_path(ROOT, ROOT / "README.md")


def test_25b_untracked_scratch_path_accepted(tmp_path):
    from diagnostics.run_k11_k12_stability import _reject_tracked_output_path

    _reject_tracked_output_path(ROOT, tmp_path / "result.json")  # must not raise


# ---------------------------------------------------------------------------
# Per-image-stats artifact round-trip (NPZ, allow_pickle=False, no object
# arrays; JSON manifest carries image IDs)
# ---------------------------------------------------------------------------


def test_per_image_stats_npz_roundtrip_no_pickle(tmp_path):
    from diagnostics.run_matched_k11_k12_evaluation import _write_per_image_stats_atomically, _load_per_image_stats

    manifest_path = tmp_path / "stats.json"
    label = np.array([[5, 3], [2, 7]], dtype=np.int64)
    intersect_k11 = np.array([[3, 2], [1, 4]], dtype=np.int64)
    union_k11 = np.array([[6, 4], [3, 8]], dtype=np.int64)
    pred_k11 = np.array([[4, 3], [2, 5]], dtype=np.int64)
    intersect_k12 = np.array([[2, 2], [1, 3]], dtype=np.int64)
    union_k12 = np.array([[6, 4], [3, 8]], dtype=np.int64)
    pred_k12 = np.array([[3, 3], [2, 4]], dtype=np.int64)

    _write_per_image_stats_atomically(
        manifest_path, schema_name="talk2dino-k11-k12-power-evaluation-per-image-stats-v1",
        class_count=2, dataset_indices=[0, 1], image_ids=["a.jpg", "b.jpg"],
        label=label, intersect_k11=intersect_k11, union_k11=union_k11, pred_k11=pred_k11,
        intersect_k12=intersect_k12, union_k12=union_k12, pred_k12=pred_k12,
    )
    loaded = _load_per_image_stats(manifest_path)
    assert loaded["manifest"]["image_ids"] == ["a.jpg", "b.jpg"]
    assert np.array_equal(loaded["arrays"]["label"], label)
    assert np.array_equal(loaded["arrays"]["intersect_k11"], intersect_k11)
    assert np.array_equal(loaded["arrays"]["intersect_k12"], intersect_k12)

    npz_path = manifest_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as data:  # allow_pickle=False must not raise
        assert set(data.files) == {
            "dataset_indices", "label", "intersect_k11", "union_k11", "pred_k11",
            "intersect_k12", "union_k12", "pred_k12",
        }


def test_per_image_stats_sha_mismatch_detected(tmp_path):
    from diagnostics.run_matched_k11_k12_evaluation import _write_per_image_stats_atomically, _load_per_image_stats

    manifest_path = tmp_path / "stats.json"
    arr = np.array([[1, 2]], dtype=np.int64)
    _write_per_image_stats_atomically(
        manifest_path, schema_name="x", class_count=2, dataset_indices=[0], image_ids=["a.jpg"],
        label=arr, intersect_k11=arr, union_k11=arr, pred_k11=arr,
        intersect_k12=arr, union_k12=arr, pred_k12=arr,
    )
    npz_path = manifest_path.with_suffix(".npz")
    npz_path.write_bytes(npz_path.read_bytes() + b"corruption")
    with pytest.raises(K11K12PowerEvaluationError):
        _load_per_image_stats(manifest_path)


# ---------------------------------------------------------------------------
# CLI fail-closed behavior / --help
# ---------------------------------------------------------------------------


def test_cli_help_does_not_require_heavy_dependencies():
    result = subprocess.run(
        [sys.executable, str(ROOT / "diagnostics/run_matched_k11_k12_evaluation.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--stability-result" in result.stdout
    assert "--run-mode" in result.stdout


def test_cli_requires_stability_result_and_run_mode():
    result = subprocess.run(
        [sys.executable, str(ROOT / "diagnostics/run_matched_k11_k12_evaluation.py"),
         "--checkpoint", "/tmp/x.json", "--result", "/tmp/y.json", "--per-image-stats", "/tmp/z.json"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--stability-result" in result.stderr or "--run-mode" in result.stderr


def test_verifier_cli_help():
    result = subprocess.run(
        [sys.executable, str(ROOT / "verify_k11_k12_power_evaluation.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "preflight" in result.stdout
    assert "verify-result" in result.stdout


def test_verifier_cli_preflight():
    result = subprocess.run(
        [sys.executable, str(ROOT / "verify_k11_k12_power_evaluation.py"), "preflight"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert "PREFLIGHT PASS" in result.stdout
