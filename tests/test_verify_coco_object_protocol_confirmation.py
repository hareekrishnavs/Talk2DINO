"""Tests for verify_coco_object_protocol_confirmation.py. The preflight
tests that pass --materialization-manifest/--data-root exercise the real
production materialization artifact (a genuine subprocess call into
verify_coco_object_val_materialization.py verify-output); the
verify-checkpoint/verify-result tests use synthetic fixtures so they
don't depend on a completed GPU evaluation existing yet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import verify_coco_object_protocol_confirmation as verify_cli  # noqa: E402
from src.coco_object_protocol_confirmation_identity import load_identity  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e12_coco_object_protocol_confirmation.toml"
MANIFEST_PATH = Path("/scratch/haree/coco_object_protocol/manifests/manifest-20443250.json")
DATA_ROOT = Path("/scratch/haree/coco_object_protocol")
SOURCE_MASKS = Path("/scratch/haree/coco_stuff164k/annotations/val2017")
SOURCE_IMAGES = Path("/scratch/haree/coco_stuff164k/images/val2017")

pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the protocol-confirmation identity")


def test_preflight_basic_pass():
    exit_code = verify_cli.main(["preflight", "--repo-root", str(ROOT)])
    assert exit_code == 0


@pytest.mark.skipif(not (MANIFEST_PATH.exists() and DATA_ROOT.exists()), reason="requires real materialized data")
def test_preflight_with_materialization_binding_against_real_data():
    exit_code = verify_cli.main([
        "preflight", "--repo-root", str(ROOT),
        "--materialization-manifest", str(MANIFEST_PATH), "--data-root", str(DATA_ROOT),
        "--source-masks", str(SOURCE_MASKS), "--source-images", str(SOURCE_IMAGES),
    ])
    assert exit_code == 0


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="requires real materialized data")
def test_preflight_rejects_tampered_manifest(tmp_path):
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["image_count"] = 4999
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(manifest))
    exit_code = verify_cli.main([
        "preflight", "--repo-root", str(ROOT),
        "--materialization-manifest", str(tampered), "--data-root", str(DATA_ROOT),
        "--source-masks", str(SOURCE_MASKS), "--source-images", str(SOURCE_IMAGES),
    ])
    assert exit_code == 2


def test_verify_checkpoint_pass_on_valid_synthetic_checkpoint(tmp_path):
    identity = load_identity(repo_root=ROOT)
    import hashlib
    identity_sha256 = hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()
    checkpoint = {
        "schema": identity["checkpoint"]["schema_name"],
        "run_mode": "pilot20",
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "materialization_identity_sha256": identity["parent_identities"]["materialization_identity_sha256"],
        "materialization_manifest_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "class_count": identity["dataset"]["class_count"],
        "live_class_names_digest": identity["dataset"]["class_names_digest"],
        "image_count_expected": 20,
        "image_order_digest": "c" * 64,
        "next_dataset_index": 5,
        "completed_image_ids": [f"{i:012d}" for i in range(5)],
        "images_completed_count": 5,
        "windows_processed_total": 15,
        "complete": False,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "updated_at_utc": "2026-01-01T00:00:00+00:00",
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint))
    exit_code = verify_cli.main(["verify-checkpoint", "--repo-root", str(ROOT), "--checkpoint", str(checkpoint_path)])
    assert exit_code == 0


def test_verify_checkpoint_fails_on_malformed_json(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text('{"a": 1, "a": 2}')
    exit_code = verify_cli.main(["verify-checkpoint", "--repo-root", str(ROOT), "--checkpoint", str(checkpoint_path)])
    assert exit_code == 2


def test_verify_result_pass_on_valid_synthetic_result(tmp_path):
    identity = load_identity(repo_root=ROOT)
    import hashlib
    identity_sha256 = hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()
    metrics = {"aAcc": 50.0, "mIoU": 30.0, "mAcc": 55.0}
    metrics_k11 = {"aAcc": 50.0, "mIoU": 29.9, "mAcc": 55.0}
    metrics_k12 = {"aAcc": 50.0, "mIoU": 30.1, "mAcc": 55.0}
    windows = 60
    result = {
        "schema": identity["run_modes"]["pilot20_schema_name"],
        "run_mode": "pilot20",
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "materialization_identity_sha256": identity["parent_identities"]["materialization_identity_sha256"],
        "materialization_manifest_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "complete": True,
        "final": False,
        "device": "cuda",
        "gpu_model": "NVIDIA H100",
        "torch_version": "2.0.0",
        "cuda_version": "12.1",
        "image_count_expected": 20,
        "image_count_processed": 20,
        "image_order_digest": "c" * 64,
        "windows_processed_total": windows,
        "class_count": identity["dataset"]["class_count"],
        "background_class_index": identity["dataset"]["background_class_index"],
        "bg_thresh": identity["background_protocol"]["bg_thresh"],
        "live_class_names_digest": identity["dataset"]["class_names_digest"],
        "metrics_E3": metrics,
        "metrics_k11": metrics_k11,
        "metrics_k12": metrics_k12,
        "delta_mIoU_k11_minus_k12_percentage_points": metrics_k11["mIoU"] - metrics_k12["mIoU"],
        "delta_mIoU_k11_minus_E3_percentage_points": metrics_k11["mIoU"] - metrics["mIoU"],
        "delta_mIoU_k12_minus_E3_percentage_points": metrics_k12["mIoU"] - metrics["mIoU"],
        "metric_unit": "percent_0_100",
        "metric_source": "full_precision_area_statistics_from_mmseg_pre_eval",
        "per_image_stats_manifest_path": "/tmp/x.json",
        "per_image_stats_manifest_sha256": "d" * 64,
        "per_image_stats_npz_sha256": "e" * 64,
        "operation_telemetry": {
            "backbone_snapshot_calls": windows, "dino_feature_extractions": windows, "topk_selection_calls": windows,
            "graph_normalizations": windows * 2, "finite_step_propagations": windows * 2, "e3_propagations": 0,
            "k11_updates": windows * 320, "k12_updates": windows * 320, "sigmoid_calls": windows * 3,
            "interpolation_calls": windows * 3,
        },
        "phase_runtime_seconds": {"total": 12.5},
        "peak_gpu_memory_bytes": 1024,
        "resumed_from_checkpoint": False,
        "source_git_branch": "e12-connectivity-analysis",
        "failure_reason": None,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    exit_code = verify_cli.main(["verify-result", "--repo-root", str(ROOT), "--result", str(result_path)])
    assert exit_code == 0


def test_verify_result_fails_on_malformed_json(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text('{"a": 1, "a": 2}')
    exit_code = verify_cli.main(["verify-result", "--repo-root", str(ROOT), "--result", str(result_path)])
    assert exit_code == 2


def test_no_traceback_on_expected_failure(tmp_path, capsys):
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text('{"a": 1, "a": 2}')
    exit_code = verify_cli.main(["verify-checkpoint", "--repo-root", str(ROOT), "--checkpoint", str(checkpoint_path)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
