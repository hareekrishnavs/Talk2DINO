"""CPU-only tests for checkpoint/result schema validation. Every
scientific value (class_count, run-mode image counts, schema names) is
read from the real identity -- never duplicated as a literal."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.coco_object_protocol_confirmation_checkpoint import (  # noqa: E402
    resume_dataset_index,
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
)
from src.coco_object_protocol_confirmation_identity import (  # noqa: E402
    CocoObjectProtocolConfirmationIdentityError,
    load_identity,
)
from src.coco_object_protocol_confirmation_report import verify_record  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e12_coco_object_protocol_confirmation.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the protocol-confirmation identity")


@pytest.fixture(scope="module")
def identity():
    return load_identity(repo_root=ROOT)


@pytest.fixture
def identity_sha256(identity):
    import hashlib
    return hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()


def _valid_checkpoint(identity, identity_sha256, *, run_mode="pilot20", next_index=0, complete=False):
    image_count = identity["run_modes"][f"{run_mode}_images"]
    completed = [f"{i:012d}" for i in range(next_index)]
    return {
        "schema": identity["checkpoint"]["schema_name"],
        "run_mode": run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "materialization_identity_sha256": identity["parent_identities"]["materialization_identity_sha256"],
        "materialization_manifest_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "class_count": identity["dataset"]["class_count"],
        "live_class_names_digest": identity["dataset"]["class_names_digest"],
        "image_count_expected": image_count,
        "image_order_digest": "c" * 64,
        "next_dataset_index": next_index,
        "completed_image_ids": completed,
        "images_completed_count": len(completed),
        "windows_processed_total": max(next_index, 0),
        "complete": complete,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "updated_at_utc": "2026-01-01T00:00:00+00:00",
    }


def test_valid_checkpoint_structure_passes(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=5)
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_complete_checkpoint_at_full_prefix_passes(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=20, complete=True)
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_wrong_identity_name_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["identity"] = "some-other-identity"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_wrong_identity_sha256_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["identity_sha256"] = "0" * 64
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_coco_stuff_checkpoint_rejected_via_matched_identity_mismatch(identity, identity_sha256):
    """A COCO-Stuff checkpoint would carry a completely different
    identity/matched_identity_sha256 binding -- confirm that mismatch is
    rejected, the actual mechanism this repo uses to distinguish protocol
    checkpoints (never a bespoke 'is this COCO-Stuff' string check)."""
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["matched_identity_sha256"] = "9" * 64
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_wrong_class_count_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["class_count"] = 171  # COCO-Stuff's class count
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_wrong_image_count_for_run_mode_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["image_count_expected"] = 100  # doesn't match pilot20
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_run_mode_mismatch_against_explicit_arg_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, run_mode="pilot20")
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256, run_mode="pilot100")


def test_duplicate_image_id_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=3)
    checkpoint["completed_image_ids"] = [checkpoint["completed_image_ids"][0]] * 3
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_next_index_disagreeing_with_completed_count_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=5)
    checkpoint["next_dataset_index"] = 4
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_complete_but_wrong_next_index_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=5, complete=True)
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_unknown_key_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["extra_field"] = 1
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_wrong_exact_type_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256)
    checkpoint["complete"] = "false"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)


def test_canonical_order_prefix_match(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=3)
    expected_ids = [f"{i:012d}" for i in range(20)]
    validate_checkpoint_against_canonical_order(checkpoint, expected_ids, image_order_digest="c" * 64)


def test_canonical_order_digest_mismatch_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=3)
    expected_ids = [f"{i:012d}" for i in range(20)]
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_against_canonical_order(checkpoint, expected_ids, image_order_digest="d" * 64)


def test_reordered_prefix_rejected(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=3)
    checkpoint["completed_image_ids"] = list(reversed(checkpoint["completed_image_ids"]))
    expected_ids = [f"{i:012d}" for i in range(20)]
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        validate_checkpoint_against_canonical_order(checkpoint, expected_ids, image_order_digest="c" * 64)


def test_resume_dataset_index_rejects_already_complete(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=20, complete=True)
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        resume_dataset_index(checkpoint)


def test_resume_dataset_index_returns_next_index_when_incomplete(identity, identity_sha256):
    checkpoint = _valid_checkpoint(identity, identity_sha256, next_index=7, complete=False)
    assert resume_dataset_index(checkpoint) == 7


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


def _valid_result(identity, identity_sha256, *, run_mode="pilot20"):
    metrics = {"aAcc": 50.0, "mIoU": 30.0, "mAcc": 55.0}
    metrics_k11 = {"aAcc": 50.0, "mIoU": 29.9, "mAcc": 55.0}
    metrics_k12 = {"aAcc": 50.0, "mIoU": 30.1, "mAcc": 55.0}
    windows = 60
    return {
        "schema": identity["run_modes"][f"{run_mode}_schema_name"],
        "run_mode": run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "materialization_identity_sha256": identity["parent_identities"]["materialization_identity_sha256"],
        "materialization_manifest_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "complete": True,
        "final": run_mode == "full",
        "device": "cuda",
        "gpu_model": "NVIDIA H100",
        "torch_version": "2.0.0",
        "cuda_version": "12.1",
        "image_count_expected": identity["run_modes"][f"{run_mode}_images"],
        "image_count_processed": identity["run_modes"][f"{run_mode}_images"],
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


def test_valid_result_passes(identity, identity_sha256):
    verify_record(_valid_result(identity, identity_sha256), identity, identity_sha256=identity_sha256)


def test_pilot_result_rejected_as_full(identity, identity_sha256):
    result = _valid_result(identity, identity_sha256, run_mode="pilot20")
    result["schema"] = identity["run_modes"]["full_schema_name"]
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        verify_record(result, identity, identity_sha256=identity_sha256)


def test_final_flag_disagreeing_with_run_mode_rejected(identity, identity_sha256):
    result = _valid_result(identity, identity_sha256, run_mode="pilot20")
    result["final"] = True
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        verify_record(result, identity, identity_sha256=identity_sha256)


def test_delta_disagreeing_with_metrics_rejected(identity, identity_sha256):
    result = _valid_result(identity, identity_sha256)
    result["delta_mIoU_k11_minus_k12_percentage_points"] = 999.0
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        verify_record(result, identity, identity_sha256=identity_sha256)


def test_nonzero_e3_propagations_rejected(identity, identity_sha256):
    result = _valid_result(identity, identity_sha256)
    result["operation_telemetry"]["e3_propagations"] = 1
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        verify_record(result, identity, identity_sha256=identity_sha256)


def test_wrong_bg_thresh_rejected(identity, identity_sha256):
    result = _valid_result(identity, identity_sha256)
    result["bg_thresh"] = 0.5
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        verify_record(result, identity, identity_sha256=identity_sha256)


def test_non_null_failure_reason_on_complete_result_rejected(identity, identity_sha256):
    result = _valid_result(identity, identity_sha256)
    result["failure_reason"] = "something"
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        verify_record(result, identity, identity_sha256=identity_sha256)


def test_incomplete_result_rejected(identity, identity_sha256):
    result = _valid_result(identity, identity_sha256)
    result["complete"] = False
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        verify_record(result, identity, identity_sha256=identity_sha256)
