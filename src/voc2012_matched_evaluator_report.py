"""Final-result schema/invariant validation for the shared VOC2012 V20/V21
matched evaluator (E3 vs k11 vs k12, T=320 finite-step). An incomplete
run must never produce a document this module accepts. A pilot result
must never be accepted as a full result, or vice versa."""

from __future__ import annotations

from typing import Any, Mapping

from src.voc2012_matched_evaluator_checkpoint import (
    _require_bool, _require_closed_mapping, _require_exact_string, _require_git_identity,
    _require_int, _require_sha256,
)
from src.voc2012_matched_evaluator_identity import Voc2012MatchedEvaluatorIdentityError

METRIC_NAMES = frozenset({"aAcc", "mIoU", "mAcc"})
VARIANT_NAMES = ("v20_e3", "v20_k11", "v20_k12", "v21_e3", "v21_k11", "v21_k12")
METRIC_FIELD_NAMES = tuple(f"metrics_{variant}" for variant in VARIANT_NAMES)
DELTA_FIELD_NAMES = (
    "delta_mIoU_v20_k11_minus_k12_percentage_points", "delta_mIoU_v20_k11_minus_e3_percentage_points",
    "delta_mIoU_v20_k12_minus_e3_percentage_points", "delta_mIoU_v21_k11_minus_k12_percentage_points",
    "delta_mIoU_v21_k11_minus_e3_percentage_points", "delta_mIoU_v21_k12_minus_e3_percentage_points",
)
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
        "voc2012_source_identity_sha256", "source_manifest_sha256", "bridge_checkpoint_sha256", "git_commit",
        "complete", "final", "device", "gpu_model", "torch_version", "cuda_version",
        "image_count_expected", "image_count_processed", "image_order_digest", "windows_processed_total",
        "v20_class_count", "v21_class_count", "background_class_index", "bg_thresh",
        "live_v20_class_names_digest", "live_v21_class_names_digest",
        *METRIC_FIELD_NAMES, *DELTA_FIELD_NAMES,
        "metric_unit", "metric_source",
        "per_image_stats_manifest_path", "per_image_stats_manifest_sha256", "per_image_stats_npz_sha256",
        "operation_telemetry", "phase_runtime_seconds", "peak_gpu_memory_bytes", "resumed_from_checkpoint",
        "source_git_branch", "failure_reason",
    }
)


def _require_metric_block(value: Any, label: str) -> Mapping[str, float]:
    block = _require_closed_mapping(value, METRIC_NAMES, label)
    for name in METRIC_NAMES:
        if type(block[name]) is not float:
            raise Voc2012MatchedEvaluatorIdentityError(f"{label}.{name} must be an exact float")
    return block


def verify_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> None:
    if not hasattr(record, "get"):
        raise Voc2012MatchedEvaluatorIdentityError("structured result must be a mapping")
    _require_closed_mapping(record, TOP_RESULT_KEYS, "result")

    run_mode = _require_exact_string(record["run_mode"], "result.run_mode")
    if run_mode not in ("pilot20", "pilot100", "full"):
        raise Voc2012MatchedEvaluatorIdentityError("result.run_mode must be one of pilot20, pilot100, full")
    expected_schema = identity["run_modes"][f"{run_mode}_schema_name"]
    if _require_exact_string(record["schema"], "result.schema") != expected_schema:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"result.schema {record['schema']!r} does not match the {run_mode} schema {expected_schema!r} "
            "-- a pilot result must never be accepted as a full result, or vice versa"
        )
    if _require_exact_string(record["identity"], "result.identity") != identity["identity"]["name"]:
        raise Voc2012MatchedEvaluatorIdentityError("result.identity mismatch")
    if _require_sha256(record["identity_sha256"], "result.identity_sha256") != identity_sha256:
        raise Voc2012MatchedEvaluatorIdentityError("result.identity_sha256 does not match the loaded identity file")
    if record["matched_identity_sha256"] != identity["parent_identities"]["matched_identity_sha256"]:
        raise Voc2012MatchedEvaluatorIdentityError("result.matched_identity_sha256 disagrees with the identity's parent")
    if record["voc2012_source_identity_sha256"] != identity["parent_identities"]["voc2012_source_identity_sha256"]:
        raise Voc2012MatchedEvaluatorIdentityError("result.voc2012_source_identity_sha256 disagrees with the identity's parent")
    _require_sha256(record["source_manifest_sha256"], "result.source_manifest_sha256")
    if (
        _require_sha256(record["bridge_checkpoint_sha256"], "result.bridge_checkpoint_sha256")
        != identity["model_and_checkpoint"]["projection_checkpoint_sha256"]
    ):
        raise Voc2012MatchedEvaluatorIdentityError(
            "result.bridge_checkpoint_sha256 disagrees with identity.model_and_checkpoint.projection_checkpoint_sha256"
        )
    _require_git_identity(record["git_commit"], "result.git_commit")

    if _require_bool(record["complete"], "result.complete") is not True:
        raise Voc2012MatchedEvaluatorIdentityError("result.complete must be true")
    expected_final = run_mode == "full"
    if _require_bool(record["final"], "result.final") is not expected_final:
        raise Voc2012MatchedEvaluatorIdentityError(f"result.final must be {expected_final} for run_mode={run_mode!r}")

    _require_exact_string(record["device"], "result.device")
    _require_exact_string(record["gpu_model"], "result.gpu_model")
    _require_exact_string(record["torch_version"], "result.torch_version")
    _require_exact_string(record["cuda_version"], "result.cuda_version")

    expected_image_count = identity["run_modes"][f"{run_mode}_image_count"]
    image_count_expected = _require_int(record["image_count_expected"], "result.image_count_expected", minimum=1)
    if image_count_expected != expected_image_count:
        raise Voc2012MatchedEvaluatorIdentityError("result.image_count_expected disagrees with the registered run-mode image count")
    image_count_processed = _require_int(record["image_count_processed"], "result.image_count_processed", minimum=1)
    if image_count_processed != image_count_expected:
        raise Voc2012MatchedEvaluatorIdentityError("result.image_count_processed disagrees with result.image_count_expected")

    _require_sha256(record["image_order_digest"], "result.image_order_digest")
    windows_processed_total = _require_int(record["windows_processed_total"], "result.windows_processed_total", minimum=1)

    v20_class_count = _require_int(record["v20_class_count"], "result.v20_class_count", minimum=1)
    if v20_class_count != identity["v20_protocol"]["class_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("result.v20_class_count disagrees with identity.v20_protocol.class_count")
    v21_class_count = _require_int(record["v21_class_count"], "result.v21_class_count", minimum=1)
    if v21_class_count != identity["v21_protocol"]["class_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("result.v21_class_count disagrees with identity.v21_protocol.class_count")
    background_class_index = _require_int(record["background_class_index"], "result.background_class_index", minimum=0)
    if background_class_index != identity["v21_protocol"]["background_class_index"]:
        raise Voc2012MatchedEvaluatorIdentityError("result.background_class_index disagrees with identity.v21_protocol.background_class_index")
    if type(record["bg_thresh"]) is not float or record["bg_thresh"] != identity["background_protocol"]["bg_thresh"]:
        raise Voc2012MatchedEvaluatorIdentityError("result.bg_thresh disagrees with identity.background_protocol.bg_thresh")
    _require_sha256(record["live_v20_class_names_digest"], "result.live_v20_class_names_digest")
    _require_sha256(record["live_v21_class_names_digest"], "result.live_v21_class_names_digest")

    metrics: dict[str, Mapping[str, float]] = {}
    for field, variant in zip(METRIC_FIELD_NAMES, VARIANT_NAMES):
        metrics[variant] = _require_metric_block(record[field], f"result.{field}")

    expected_deltas = {
        "delta_mIoU_v20_k11_minus_k12_percentage_points": metrics["v20_k11"]["mIoU"] - metrics["v20_k12"]["mIoU"],
        "delta_mIoU_v20_k11_minus_e3_percentage_points": metrics["v20_k11"]["mIoU"] - metrics["v20_e3"]["mIoU"],
        "delta_mIoU_v20_k12_minus_e3_percentage_points": metrics["v20_k12"]["mIoU"] - metrics["v20_e3"]["mIoU"],
        "delta_mIoU_v21_k11_minus_k12_percentage_points": metrics["v21_k11"]["mIoU"] - metrics["v21_k12"]["mIoU"],
        "delta_mIoU_v21_k11_minus_e3_percentage_points": metrics["v21_k11"]["mIoU"] - metrics["v21_e3"]["mIoU"],
        "delta_mIoU_v21_k12_minus_e3_percentage_points": metrics["v21_k12"]["mIoU"] - metrics["v21_e3"]["mIoU"],
    }
    for label, expected in expected_deltas.items():
        observed = record[label]
        if type(observed) is not float:
            raise Voc2012MatchedEvaluatorIdentityError(f"result.{label} must be an exact float")
        if abs(observed - expected) > 1e-6:
            raise Voc2012MatchedEvaluatorIdentityError(f"result.{label} disagrees with its own metric blocks: expected {expected}, observed {observed}")

    if record["metric_unit"] != "percent_0_100":
        raise Voc2012MatchedEvaluatorIdentityError("result.metric_unit must be exactly 'percent_0_100'")
    _require_exact_string(record["metric_source"], "result.metric_source")

    _require_exact_string(record["per_image_stats_manifest_path"], "result.per_image_stats_manifest_path")
    _require_sha256(record["per_image_stats_manifest_sha256"], "result.per_image_stats_manifest_sha256")
    _require_sha256(record["per_image_stats_npz_sha256"], "result.per_image_stats_npz_sha256")

    telemetry = _require_closed_mapping(record["operation_telemetry"], TELEMETRY_KEYS, "result.operation_telemetry")
    for key in TELEMETRY_KEYS:
        _require_int(telemetry[key], f"result.operation_telemetry.{key}", minimum=0)
    if telemetry["e3_propagations"] != 0:
        raise Voc2012MatchedEvaluatorIdentityError("result.operation_telemetry.e3_propagations must be exactly 0")
    if telemetry["backbone_snapshot_calls"] != windows_processed_total:
        raise Voc2012MatchedEvaluatorIdentityError("result.operation_telemetry.backbone_snapshot_calls disagrees with windows_processed_total")
    if telemetry["sigmoid_calls"] != windows_processed_total * 3:
        raise Voc2012MatchedEvaluatorIdentityError("result.operation_telemetry.sigmoid_calls must equal windows_processed_total * 3 (E3, k11, k12)")
    if telemetry["k11_updates"] != windows_processed_total * 320:
        raise Voc2012MatchedEvaluatorIdentityError("result.operation_telemetry.k11_updates must equal windows_processed_total * 320")
    if telemetry["k12_updates"] != windows_processed_total * 320:
        raise Voc2012MatchedEvaluatorIdentityError("result.operation_telemetry.k12_updates must equal windows_processed_total * 320")

    runtime = _require_closed_mapping(record["phase_runtime_seconds"], frozenset({"total"}), "result.phase_runtime_seconds")
    if type(runtime["total"]) is not float or runtime["total"] < 0:
        raise Voc2012MatchedEvaluatorIdentityError("result.phase_runtime_seconds.total must be a non-negative float")

    _require_int(record["peak_gpu_memory_bytes"], "result.peak_gpu_memory_bytes", minimum=0)
    if type(record["resumed_from_checkpoint"]) is not bool:
        raise Voc2012MatchedEvaluatorIdentityError("result.resumed_from_checkpoint must be an exact boolean")
    _require_exact_string(record["source_git_branch"], "result.source_git_branch")
    if record["failure_reason"] is not None:
        raise Voc2012MatchedEvaluatorIdentityError("result.failure_reason must be null for a completed result")


__all__ = [
    "DELTA_FIELD_NAMES", "METRIC_FIELD_NAMES", "METRIC_NAMES", "TELEMETRY_KEYS", "TOP_RESULT_KEYS",
    "VARIANT_NAMES", "verify_record",
]
