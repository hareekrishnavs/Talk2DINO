"""Structured result schema validation for the matched k11/k12 finite-step
power evaluator.

This module never renders an efficacy PASS/FAIL verdict -- there is no
threshold section here, unlike the stability gate's ``classify_regime``.
It only verifies that a result is structurally complete, internally
consistent (recomputes windows/telemetry sums against the recorded
per-run-mode expectations rather than trusting an isolated scalar), and
correctly bound to its identity, run mode, and stability-gate provenance.
A pilot result is never accepted where a full result is required, and
vice versa.

Checkpoint validation lives in :mod:`src.k11_k12_power_evaluation_checkpoint`
-- the sole shared authority for checkpoint invariants, used identically by
the evaluator's resume path and by ``verify-checkpoint``. This module does
not duplicate any of it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from src.k11_k12_power_evaluation_checkpoint import parse_strict_json_document
from src.k11_k12_power_evaluation_identity import (
    K11K12PowerEvaluationError,
    RUN_MODE_IMAGE_COUNT_KEYS,
    RUN_MODE_SCHEMA_KEYS,
    load_identity,
    repository_root,
)


METRIC_KEYS = frozenset({"aAcc", "mIoU", "mAcc"})
OPERATION_TELEMETRY_KEYS = frozenset(
    {
        "backbone_snapshot_calls", "dino_feature_extractions", "topk_selection_calls",
        "graph_normalizations", "finite_step_propagations", "k11_updates", "k12_updates",
        "sigmoid_calls", "interpolation_calls",
    }
)
PHASE_RUNTIME_KEYS = frozenset({"total"})

TOP_RESULT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "matched_identity_sha256", "git_commit",
        "complete", "final", "device", "gpu_model", "torch_version", "cuda_version",
        "image_count_expected", "image_count_processed", "image_order_digest", "windows_processed_total",
        "class_count", "metrics_k11", "metrics_k12", "delta_mIoU_percentage_points", "metric_unit",
        "metric_source", "per_image_stats_manifest_path", "per_image_stats_manifest_sha256",
        "per_image_stats_npz_sha256", "stability_result_sha256", "stability_schema", "gate_classification",
        "gate_git_commit", "gate_identity_sha256", "finite_step_kernel_sha256", "graph_construction_sha256",
        "operation_telemetry", "phase_runtime_seconds", "peak_gpu_memory_bytes", "resumed_from_checkpoint",
        "source_git_branch", "failure_reason",
    }
)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise K11K12PowerEvaluationError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise K11K12PowerEvaluationError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise K11K12PowerEvaluationError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise K11K12PowerEvaluationError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise K11K12PowerEvaluationError(f"{label} must be at least {minimum}")
    return value


def _require_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise K11K12PowerEvaluationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise K11K12PowerEvaluationError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise K11K12PowerEvaluationError(f"{label} must be at least {minimum}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise K11K12PowerEvaluationError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise K11K12PowerEvaluationError(f"{label} must be a full Git identity")
    return token


def _validate_metrics(value: Any, label: str) -> Mapping[str, float]:
    metrics = _require_closed_mapping(value, METRIC_KEYS, label)
    for name in METRIC_KEYS:
        percent = _require_float(metrics[name], f"{label}.{name}", minimum=0.0)
        if percent > 100.0:
            raise K11K12PowerEvaluationError(f"{label}.{name} must be a percentage in [0, 100]")
    return metrics


def verify_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> str:
    if not hasattr(record, "get"):
        raise K11K12PowerEvaluationError("structured result must be a mapping")
    _require_closed_mapping(record, TOP_RESULT_KEYS, "result")

    run_mode = _require_exact_string(record["run_mode"], "result.run_mode")
    if run_mode not in RUN_MODE_IMAGE_COUNT_KEYS:
        raise K11K12PowerEvaluationError(f"result.run_mode must be one of {sorted(RUN_MODE_IMAGE_COUNT_KEYS)}")
    expected_schema = identity["run_modes"][RUN_MODE_SCHEMA_KEYS[run_mode]]
    if _require_exact_string(record["schema"], "result.schema") != expected_schema:
        raise K11K12PowerEvaluationError(
            f"result.schema {record['schema']!r} does not match the {run_mode} schema {expected_schema!r} "
            "-- a pilot result must never be accepted as a full result, or vice versa"
        )
    if _require_exact_string(record["identity"], "result.identity") != identity["identity"]["name"]:
        raise K11K12PowerEvaluationError("result.identity mismatch")
    if _require_sha256(record["identity_sha256"], "result.identity_sha256") != identity_sha256:
        raise K11K12PowerEvaluationError("result.identity_sha256 does not match the loaded identity file")
    if record["matched_identity_sha256"] != identity["parent_identity"]["matched_identity_sha256"]:
        raise K11K12PowerEvaluationError("result.matched_identity_sha256 disagrees with the identity's parent")
    _require_git_identity(record["git_commit"], "result.git_commit")

    if _require_bool(record["complete"], "result.complete") is not True:
        raise K11K12PowerEvaluationError("structured result is not complete")
    final = _require_bool(record["final"], "result.final")
    if run_mode == "full" and final is not True:
        raise K11K12PowerEvaluationError("a full-mode result must have final=true")
    if run_mode != "full" and final is not False:
        raise K11K12PowerEvaluationError("a pilot result must never claim final=true")

    _require_exact_string(record["device"], "result.device")
    _require_exact_string(record["gpu_model"], "result.gpu_model")
    _require_exact_string(record["torch_version"], "result.torch_version")
    _require_exact_string(record["cuda_version"], "result.cuda_version")

    expected_image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[run_mode]]
    image_count_expected = _require_int(record["image_count_expected"], "result.image_count_expected", minimum=1)
    if image_count_expected != expected_image_count:
        raise K11K12PowerEvaluationError(
            f"result.image_count_expected {image_count_expected} disagrees with the registered "
            f"{run_mode} image count {expected_image_count}"
        )
    image_count_processed = _require_int(record["image_count_processed"], "result.image_count_processed", minimum=0)
    if image_count_processed != image_count_expected:
        raise K11K12PowerEvaluationError("a complete result must have image_count_processed == image_count_expected")
    _require_sha256(record["image_order_digest"], "result.image_order_digest")
    _require_int(record["windows_processed_total"], "result.windows_processed_total", minimum=1)

    # class_count is validated here only against the identity's own
    # registered, relationally-derived value (never a bare module
    # constant) -- the evaluator itself is separately required to derive
    # this from the live inference.num_classes and to have already
    # cross-checked it against the identity before this field is ever
    # written; see K11K12PowerEvaluationClassCount in the evaluator CLI.
    if _require_int(record["class_count"], "result.class_count", minimum=1) != identity["metrics"]["class_count"]:
        raise K11K12PowerEvaluationError("result.class_count disagrees with the registered class count")

    metrics_k11 = _validate_metrics(record["metrics_k11"], "result.metrics_k11")
    metrics_k12 = _validate_metrics(record["metrics_k12"], "result.metrics_k12")
    delta = _require_float(record["delta_mIoU_percentage_points"], "result.delta_mIoU_percentage_points")
    recomputed_delta = float(metrics_k11["mIoU"]) - float(metrics_k12["mIoU"])
    if abs(delta - recomputed_delta) > 1e-6:
        raise K11K12PowerEvaluationError(
            "result.delta_mIoU_percentage_points is inconsistent with metrics_k11.mIoU - metrics_k12.mIoU"
        )
    if record["metric_unit"] != identity["metrics"]["unit"]:
        raise K11K12PowerEvaluationError("result.metric_unit disagrees with the registered metric unit")
    if record["metric_source"] != identity["metrics"]["precision_source"]:
        raise K11K12PowerEvaluationError("result.metric_source disagrees with the registered precision source")

    _require_exact_string(record["per_image_stats_manifest_path"], "result.per_image_stats_manifest_path")
    _require_sha256(record["per_image_stats_manifest_sha256"], "result.per_image_stats_manifest_sha256")
    _require_sha256(record["per_image_stats_npz_sha256"], "result.per_image_stats_npz_sha256")

    _require_sha256(record["stability_result_sha256"], "result.stability_result_sha256")
    _require_exact_string(record["stability_schema"], "result.stability_schema")
    classification = _require_exact_string(record["gate_classification"], "result.gate_classification")
    if classification not in identity["stability_gate"]["accepted_classifications"]:
        raise K11K12PowerEvaluationError(
            f"result.gate_classification {classification!r} is not one of the accepted classifications"
        )
    _require_git_identity(record["gate_git_commit"], "result.gate_git_commit")
    _require_sha256(record["gate_identity_sha256"], "result.gate_identity_sha256")
    # Cross-checking this hash against the *current* repository's live
    # kernel source is validate_stability_result_binding's job (it re-hashes
    # the file on disk at verification time); this function only checks the
    # recorded field is a well-formed SHA256, since a historical result's
    # kernel may legitimately differ from whatever HEAD is checked out now.
    _require_sha256(record["finite_step_kernel_sha256"], "result.finite_step_kernel_sha256")
    _require_sha256(record["graph_construction_sha256"], "result.graph_construction_sha256")

    telemetry = _require_closed_mapping(record["operation_telemetry"], OPERATION_TELEMETRY_KEYS, "result.operation_telemetry")
    windows_processed = record["windows_processed_total"]
    for name in ("backbone_snapshot_calls", "dino_feature_extractions", "topk_selection_calls"):
        if _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=0) != windows_processed:
            raise K11K12PowerEvaluationError(f"operation_telemetry.{name} must equal windows_processed_total ({windows_processed})")
    for name in ("graph_normalizations", "sigmoid_calls", "interpolation_calls"):
        if _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=0) != 2 * windows_processed:
            raise K11K12PowerEvaluationError(f"operation_telemetry.{name} must equal 2 * windows_processed_total ({2 * windows_processed})")
    for name in ("finite_step_propagations",):
        if _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=0) != 2 * windows_processed:
            raise K11K12PowerEvaluationError(f"operation_telemetry.{name} must equal 2 * windows_processed_total")
    for name in ("k11_updates", "k12_updates"):
        if _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=0) != 320 * windows_processed:
            raise K11K12PowerEvaluationError(f"operation_telemetry.{name} must equal 320 * windows_processed_total")

    phase_runtime = _require_closed_mapping(record["phase_runtime_seconds"], PHASE_RUNTIME_KEYS, "result.phase_runtime_seconds")
    _require_float(phase_runtime["total"], "phase_runtime_seconds.total", minimum=0.0)
    _require_int(record["peak_gpu_memory_bytes"], "result.peak_gpu_memory_bytes", minimum=0)
    _require_bool(record["resumed_from_checkpoint"], "result.resumed_from_checkpoint")
    _require_exact_string(record["source_git_branch"], "result.source_git_branch")
    if record["failure_reason"] is not None:
        _require_exact_string(record["failure_reason"], "result.failure_reason")

    return (
        f"K11/K12 POWER EVALUATION RESULT PASS run_mode={run_mode} images={image_count_processed} "
        f"mIoU_k11={metrics_k11['mIoU']:.6f} mIoU_k12={metrics_k12['mIoU']:.6f} delta_mIoU={delta:.6f}"
    )


def verify_result(path: Path, *, identity_path: Path | None = None, repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    resolved_identity_path = Path(identity_path) if identity_path is not None else root / "evaluation_identities/e12_k11_k12_power_evaluation.toml"
    identity_sha256 = _sha256_file(resolved_identity_path)
    record = parse_strict_json_document(Path(path), label="structured result")
    return verify_record(record, identity, identity_sha256=identity_sha256)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "OPERATION_TELEMETRY_KEYS",
    "TOP_RESULT_KEYS",
    "verify_record",
    "verify_result",
]
