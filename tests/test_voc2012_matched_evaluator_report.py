"""Tests for the shared VOC2012 V20/V21 matched-evaluator result schema/
invariant validation. CPU-only synthetic fixtures; no CUDA."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.voc2012_matched_evaluator_identity import (  # noqa: E402
    Voc2012MatchedEvaluatorIdentityError,
    load_identity,
)
from src.voc2012_matched_evaluator_report import VARIANT_NAMES, verify_record  # noqa: E402

IDENTITY_PATH = ROOT / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
pytestmark = pytest.mark.skipif(not IDENTITY_PATH.exists(), reason="requires the voc2012 matched-evaluator identity")


@pytest.fixture(scope="module")
def identity():
    return load_identity(repo_root=ROOT)


IDENTITY_SHA = "a" * 64


def _valid_result(identity, *, run_mode="pilot20"):
    image_count = identity["run_modes"][f"{run_mode}_image_count"]
    metrics = {v: {"aAcc": 50.0, "mIoU": 30.0, "mAcc": 40.0} for v in VARIANT_NAMES}
    metrics["v20_k11"]["mIoU"] = 31.0
    metrics["v20_k12"]["mIoU"] = 30.5
    metrics["v20_e3"]["mIoU"] = 29.5
    metrics["v21_k11"]["mIoU"] = 29.0
    metrics["v21_k12"]["mIoU"] = 28.5
    metrics["v21_e3"]["mIoU"] = 27.5

    result = {
        "schema": identity["run_modes"][f"{run_mode}_schema_name"], "run_mode": run_mode,
        "identity": identity["identity"]["name"], "identity_sha256": IDENTITY_SHA,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "voc2012_source_identity_sha256": identity["parent_identities"]["voc2012_source_identity_sha256"],
        "source_manifest_sha256": "b" * 64,
        "bridge_checkpoint_sha256": identity["model_and_checkpoint"]["projection_checkpoint_sha256"],
        "git_commit": "c" * 40, "complete": True, "final": run_mode == "full",
        "device": "cuda", "gpu_model": "H100", "torch_version": "2.1.0", "cuda_version": "12.1",
        "image_count_expected": image_count, "image_count_processed": image_count,
        "image_order_digest": "d" * 64, "windows_processed_total": image_count * 5,
        "v20_class_count": identity["v20_protocol"]["class_count"],
        "v21_class_count": identity["v21_protocol"]["class_count"],
        "background_class_index": identity["v21_protocol"]["background_class_index"],
        "bg_thresh": identity["background_protocol"]["bg_thresh"],
        "live_v20_class_names_digest": "e" * 64, "live_v21_class_names_digest": "f" * 64,
        "metric_unit": "percent_0_100", "metric_source": "full_precision_area_statistics_from_mmseg_pre_eval",
        "per_image_stats_manifest_path": "x.json", "per_image_stats_manifest_sha256": "0" * 64,
        "per_image_stats_npz_sha256": "1" * 64,
        "operation_telemetry": {
            "backbone_snapshot_calls": image_count * 5, "dino_feature_extractions": image_count * 5,
            "topk_selection_calls": image_count * 5, "graph_normalizations": image_count * 10,
            "finite_step_propagations": image_count * 10, "e3_propagations": 0,
            "k11_updates": image_count * 5 * 320, "k12_updates": image_count * 5 * 320,
            "sigmoid_calls": image_count * 5 * 3, "interpolation_calls": image_count * 5 * 3,
        },
        "phase_runtime_seconds": {"total": 12.5}, "peak_gpu_memory_bytes": 1000,
        "resumed_from_checkpoint": False, "source_git_branch": "e12-connectivity-analysis", "failure_reason": None,
    }
    for v in VARIANT_NAMES:
        result[f"metrics_{v}"] = metrics[v]
    result["delta_mIoU_v20_k11_minus_k12_percentage_points"] = metrics["v20_k11"]["mIoU"] - metrics["v20_k12"]["mIoU"]
    result["delta_mIoU_v20_k11_minus_e3_percentage_points"] = metrics["v20_k11"]["mIoU"] - metrics["v20_e3"]["mIoU"]
    result["delta_mIoU_v20_k12_minus_e3_percentage_points"] = metrics["v20_k12"]["mIoU"] - metrics["v20_e3"]["mIoU"]
    result["delta_mIoU_v21_k11_minus_k12_percentage_points"] = metrics["v21_k11"]["mIoU"] - metrics["v21_k12"]["mIoU"]
    result["delta_mIoU_v21_k11_minus_e3_percentage_points"] = metrics["v21_k11"]["mIoU"] - metrics["v21_e3"]["mIoU"]
    result["delta_mIoU_v21_k12_minus_e3_percentage_points"] = metrics["v21_k12"]["mIoU"] - metrics["v21_e3"]["mIoU"]
    return result


def test_valid_pilot20_result_accepted(identity):
    result = _valid_result(identity, run_mode="pilot20")
    verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_valid_full_result_accepted(identity):
    result = _valid_result(identity, run_mode="full")
    verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_pilot_schema_not_accepted_as_full(identity):
    result = _valid_result(identity, run_mode="full")
    result["schema"] = identity["run_modes"]["pilot20_schema_name"]
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_full_schema_not_accepted_as_pilot(identity):
    result = _valid_result(identity, run_mode="pilot20")
    result["schema"] = identity["run_modes"]["full_schema_name"]
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_pilot_final_must_be_false(identity):
    result = _valid_result(identity, run_mode="pilot20")
    result["final"] = True
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_full_final_must_be_true(identity):
    result = _valid_result(identity, run_mode="full")
    result["final"] = False
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_incomplete_result_rejected(identity):
    result = _valid_result(identity)
    result["complete"] = False
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_inconsistent_delta_rejected(identity):
    result = _valid_result(identity)
    result["delta_mIoU_v20_k11_minus_k12_percentage_points"] = 999.0
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_all_six_deltas_checked_independently(identity):
    for delta_field in (
        "delta_mIoU_v20_k11_minus_k12_percentage_points", "delta_mIoU_v20_k11_minus_e3_percentage_points",
        "delta_mIoU_v20_k12_minus_e3_percentage_points", "delta_mIoU_v21_k11_minus_k12_percentage_points",
        "delta_mIoU_v21_k11_minus_e3_percentage_points", "delta_mIoU_v21_k12_minus_e3_percentage_points",
    ):
        result = _valid_result(identity)
        result[delta_field] = -12345.0
        with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
            verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_e3_propagations_nonzero_rejected(identity):
    result = _valid_result(identity)
    result["operation_telemetry"]["e3_propagations"] = 1
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_k11_updates_not_320_multiple_rejected(identity):
    result = _valid_result(identity)
    result["operation_telemetry"]["k11_updates"] = result["windows_processed_total"] * 319
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_wrong_bg_thresh_rejected(identity):
    result = _valid_result(identity)
    result["bg_thresh"] = 0.5
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_bg_thresh_int_rejected(identity):
    result = _valid_result(identity)
    result["bg_thresh"] = 1  # exact-int, not float
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_wrong_v20_class_count_rejected(identity):
    result = _valid_result(identity)
    result["v20_class_count"] = 19
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_wrong_v21_class_count_rejected(identity):
    result = _valid_result(identity)
    result["v21_class_count"] = 22
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_failure_reason_must_be_null_for_complete_result(identity):
    result = _valid_result(identity)
    result["failure_reason"] = "something went wrong"
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_unknown_field_rejected(identity):
    result = _valid_result(identity)
    result["extra_field"] = "x"
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_missing_field_rejected(identity):
    result = _valid_result(identity)
    del result["metrics_v21_e3"]
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_metric_block_wrong_type_rejected(identity):
    result = _valid_result(identity)
    result["metrics_v20_e3"]["mIoU"] = 30  # int, not float
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_metric_unit_must_be_percent_0_100(identity):
    result = _valid_result(identity)
    result["metric_unit"] = "fraction_0_1"
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_verify_record_never_mutates_input(identity):
    result = _valid_result(identity)
    before = copy.deepcopy(result)
    verify_record(result, identity, identity_sha256=IDENTITY_SHA)
    assert result == before


# ---------------------------------------------------------------------
# bridge_checkpoint_sha256 relational validation: the result's recorded
# value must equal identity["model_and_checkpoint"]["projection_checkpoint_sha256"]
# exactly, not merely be well-formed hex.
# ---------------------------------------------------------------------


def test_bridge_checkpoint_sha256_matching_value_accepted(identity):
    result = _valid_result(identity)
    assert result["bridge_checkpoint_sha256"] == identity["model_and_checkpoint"]["projection_checkpoint_sha256"]
    verify_record(result, identity, identity_sha256=IDENTITY_SHA)


def test_bridge_checkpoint_sha256_wrong_value_rejected_on_intermediate_pilot_result(identity):
    result = _valid_result(identity, run_mode="pilot20")
    result["bridge_checkpoint_sha256"] = "3" * 64
    before = copy.deepcopy(result)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)
    assert result == before


def test_bridge_checkpoint_sha256_wrong_value_rejected_on_final_result(identity):
    result = _valid_result(identity, run_mode="full")
    assert result["final"] is True
    result["bridge_checkpoint_sha256"] = "4" * 64
    before = copy.deepcopy(result)
    with pytest.raises(Voc2012MatchedEvaluatorIdentityError):
        verify_record(result, identity, identity_sha256=IDENTITY_SHA)
    assert result == before


def test_bridge_checkpoint_sha256_check_present_in_verify_record_source():
    import inspect

    source = inspect.getsource(verify_record)
    assert "projection_checkpoint_sha256" in source
