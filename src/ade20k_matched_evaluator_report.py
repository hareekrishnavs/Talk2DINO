"""Final-result schema/invariant validation for the ADE20K C150 matched
E3-vs-k11-vs-k12 evaluator. An incomplete run must never produce a
document this module accepts. A pilot result must never be accepted as
a full result, or vice versa."""

from __future__ import annotations

from typing import Any, Mapping

from src.ade20k_matched_evaluator_checkpoint import (
    _require_bool, _require_closed_mapping, _require_exact_string, _require_git_identity,
    _require_int, _require_sha256,
)
from src.ade20k_matched_evaluator_identity import Ade20kMatchedEvaluatorIdentityError

METRIC_NAMES = frozenset({"aAcc", "mIoU", "mAcc"})
TELEMETRY_KEYS = frozenset(
    {
        "backbone_snapshot_calls", "dino_feature_extractions", "topk_selection_calls",
        "graph_normalizations", "finite_step_propagations", "e3_propagations",
        "k11_updates", "k12_updates", "sigmoid_calls", "interpolation_calls",
    }
)
TOP_RESULT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "matched_identity_sha256",
        "ade20k_source_identity_sha256", "ade20k_source_manifest_sha256", "git_commit",
        "complete", "final", "device", "gpu_model", "torch_version", "cuda_version",
        "image_count_expected", "image_count_processed", "image_order_digest", "windows_processed_total",
        "class_count", "live_class_names_digest",
        "metrics_E3", "metrics_k11", "metrics_k12",
        "delta_mIoU_k11_minus_k12_percentage_points", "delta_mIoU_k11_minus_E3_percentage_points",
        "delta_mIoU_k12_minus_E3_percentage_points", "metric_unit", "metric_source",
        "per_image_stats_manifest_path", "per_image_stats_manifest_sha256", "per_image_stats_npz_sha256",
        "operation_telemetry", "phase_runtime_seconds", "peak_gpu_memory_bytes", "resumed_from_checkpoint",
        "source_git_branch", "failure_reason",
    }
)


def _require_metric_block(value: Any, label: str) -> Mapping[str, float]:
    block = _require_closed_mapping(value, METRIC_NAMES, label)
    for name in METRIC_NAMES:
        if type(block[name]) is not float:
            raise Ade20kMatchedEvaluatorIdentityError(f"{label}.{name} must be an exact float")
    return block


def verify_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> None:
    if not hasattr(record, "get"):
        raise Ade20kMatchedEvaluatorIdentityError("structured result must be a mapping")
    _require_closed_mapping(record, TOP_RESULT_KEYS, "result")

    run_mode = _require_exact_string(record["run_mode"], "result.run_mode")
    if run_mode not in ("pilot20", "pilot100", "full"):
        raise Ade20kMatchedEvaluatorIdentityError("result.run_mode must be one of pilot20, pilot100, full")
    expected_schema = identity["run_modes"][f"{run_mode}_schema_name"]
    if _require_exact_string(record["schema"], "result.schema") != expected_schema:
        raise Ade20kMatchedEvaluatorIdentityError(
            f"result.schema {record['schema']!r} does not match the {run_mode} schema {expected_schema!r} "
            "-- a pilot result must never be accepted as a full result, or vice versa"
        )
    if _require_exact_string(record["identity"], "result.identity") != identity["identity"]["name"]:
        raise Ade20kMatchedEvaluatorIdentityError("result.identity mismatch")
    if _require_sha256(record["identity_sha256"], "result.identity_sha256") != identity_sha256:
        raise Ade20kMatchedEvaluatorIdentityError("result.identity_sha256 does not match the loaded identity file")
    if record["matched_identity_sha256"] != identity["parent_identities"]["matched_identity_sha256"]:
        raise Ade20kMatchedEvaluatorIdentityError("result.matched_identity_sha256 disagrees with the identity's parent")
    if record["ade20k_source_identity_sha256"] != identity["parent_identities"]["ade20k_source_identity_sha256"]:
        raise Ade20kMatchedEvaluatorIdentityError("result.ade20k_source_identity_sha256 disagrees with the identity's parent")
    _require_sha256(record["ade20k_source_manifest_sha256"], "result.ade20k_source_manifest_sha256")
    _require_git_identity(record["git_commit"], "result.git_commit")

    if _require_bool(record["complete"], "result.complete") is not True:
        raise Ade20kMatchedEvaluatorIdentityError("result.complete must be true")
    expected_final = run_mode == "full"
    if _require_bool(record["final"], "result.final") is not expected_final:
        raise Ade20kMatchedEvaluatorIdentityError(f"result.final must be {expected_final} for run_mode={run_mode!r}")

    _require_exact_string(record["device"], "result.device")
    _require_exact_string(record["gpu_model"], "result.gpu_model")
    _require_exact_string(record["torch_version"], "result.torch_version")
    _require_exact_string(record["cuda_version"], "result.cuda_version")

    expected_image_count = identity["run_modes"][f"{run_mode}_images"]
    image_count_expected = _require_int(record["image_count_expected"], "result.image_count_expected", minimum=1)
    if image_count_expected != expected_image_count:
        raise Ade20kMatchedEvaluatorIdentityError("result.image_count_expected disagrees with the registered run-mode image count")
    image_count_processed = _require_int(record["image_count_processed"], "result.image_count_processed", minimum=1)
    if image_count_processed != image_count_expected:
        raise Ade20kMatchedEvaluatorIdentityError("result.image_count_processed disagrees with result.image_count_expected")

    _require_sha256(record["image_order_digest"], "result.image_order_digest")
    windows_processed_total = _require_int(record["windows_processed_total"], "result.windows_processed_total", minimum=1)

    class_count = _require_int(record["class_count"], "result.class_count", minimum=1)
    if class_count != identity["dataset"]["class_count"]:
        raise Ade20kMatchedEvaluatorIdentityError("result.class_count disagrees with identity.dataset.class_count")
    if _require_sha256(record["live_class_names_digest"], "result.live_class_names_digest") != identity["dataset"]["class_names_digest"]:
        raise Ade20kMatchedEvaluatorIdentityError(
            "result.live_class_names_digest disagrees with identity.dataset.class_names_digest"
        )

    metrics_e3 = _require_metric_block(record["metrics_E3"], "result.metrics_E3")
    metrics_k11 = _require_metric_block(record["metrics_k11"], "result.metrics_k11")
    metrics_k12 = _require_metric_block(record["metrics_k12"], "result.metrics_k12")

    for label, expected, observed in (
        ("delta_mIoU_k11_minus_k12_percentage_points", metrics_k11["mIoU"] - metrics_k12["mIoU"], record["delta_mIoU_k11_minus_k12_percentage_points"]),
        ("delta_mIoU_k11_minus_E3_percentage_points", metrics_k11["mIoU"] - metrics_e3["mIoU"], record["delta_mIoU_k11_minus_E3_percentage_points"]),
        ("delta_mIoU_k12_minus_E3_percentage_points", metrics_k12["mIoU"] - metrics_e3["mIoU"], record["delta_mIoU_k12_minus_E3_percentage_points"]),
    ):
        if type(observed) is not float:
            raise Ade20kMatchedEvaluatorIdentityError(f"result.{label} must be an exact float")
        if abs(observed - expected) > 1e-6:
            raise Ade20kMatchedEvaluatorIdentityError(f"result.{label} disagrees with its own metric blocks: expected {expected}, observed {observed}")

    if record["metric_unit"] != "percent_0_100":
        raise Ade20kMatchedEvaluatorIdentityError("result.metric_unit must be exactly 'percent_0_100'")
    _require_exact_string(record["metric_source"], "result.metric_source")

    _require_exact_string(record["per_image_stats_manifest_path"], "result.per_image_stats_manifest_path")
    _require_sha256(record["per_image_stats_manifest_sha256"], "result.per_image_stats_manifest_sha256")
    _require_sha256(record["per_image_stats_npz_sha256"], "result.per_image_stats_npz_sha256")

    telemetry = _require_closed_mapping(record["operation_telemetry"], TELEMETRY_KEYS, "result.operation_telemetry")
    for key in TELEMETRY_KEYS:
        _require_int(telemetry[key], f"result.operation_telemetry.{key}", minimum=0)
    if telemetry["e3_propagations"] != 0:
        raise Ade20kMatchedEvaluatorIdentityError("result.operation_telemetry.e3_propagations must be exactly 0")
    if telemetry["backbone_snapshot_calls"] != windows_processed_total:
        raise Ade20kMatchedEvaluatorIdentityError("result.operation_telemetry.backbone_snapshot_calls disagrees with windows_processed_total")
    if telemetry["sigmoid_calls"] != windows_processed_total * 3:
        raise Ade20kMatchedEvaluatorIdentityError("result.operation_telemetry.sigmoid_calls must equal windows_processed_total * 3 (E3, k11, k12)")
    if telemetry["k11_updates"] != windows_processed_total * 320:
        raise Ade20kMatchedEvaluatorIdentityError("result.operation_telemetry.k11_updates must equal windows_processed_total * 320")
    if telemetry["k12_updates"] != windows_processed_total * 320:
        raise Ade20kMatchedEvaluatorIdentityError("result.operation_telemetry.k12_updates must equal windows_processed_total * 320")

    runtime = _require_closed_mapping(record["phase_runtime_seconds"], frozenset({"total"}), "result.phase_runtime_seconds")
    if type(runtime["total"]) is not float or runtime["total"] < 0:
        raise Ade20kMatchedEvaluatorIdentityError("result.phase_runtime_seconds.total must be a non-negative float")

    _require_int(record["peak_gpu_memory_bytes"], "result.peak_gpu_memory_bytes", minimum=0)
    if type(record["resumed_from_checkpoint"]) is not bool:
        raise Ade20kMatchedEvaluatorIdentityError("result.resumed_from_checkpoint must be an exact boolean")
    _require_exact_string(record["source_git_branch"], "result.source_git_branch")
    if record["failure_reason"] is not None:
        raise Ade20kMatchedEvaluatorIdentityError("result.failure_reason must be null for a completed result")


__all__ = ["METRIC_NAMES", "TELEMETRY_KEYS", "TOP_RESULT_KEYS", "verify_record"]
