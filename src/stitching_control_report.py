"""Structured result schema validation for the reusable stitching control
suite's evaluator.

Never renders an efficacy verdict -- this only verifies a result is
structurally complete, internally consistent (recomputes telemetry sums
against the recorded per-run-mode expectations rather than trusting an
isolated scalar), and correctly bound to its identity/run-mode/variant-set.
A pilot result is never accepted where a full5000 result is required, or
vice versa. Checkpoint validation lives exclusively in
:mod:`src.stitching_control_checkpoint` -- this module never duplicates
any of it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from src.stitching_control_checkpoint import parse_strict_json_document
from src.stitching_control_identity import (
    CANONICAL_VARIANT_NAMES,
    RUN_MODE_IMAGE_COUNT_KEYS,
    RUN_MODE_SCHEMA_KEYS,
    StitchingControlIdentityError,
    load_identity,
    repository_root,
)

METRIC_KEYS = frozenset({"aAcc", "mIoU", "mAcc"})
OPERATION_TELEMETRY_KEYS = frozenset(
    {
        "sample_pulls", "window_enumerations", "backbone_snapshot_calls", "graph_builds",
        "propagation_calls", "probability_interpolation_calls", "score_interpolation_calls",
        "accumulator_finalizations",
    }
)
PHASE_RUNTIME_KEYS = frozenset({"total", "shared", "per_variant"})
NON_REFERENCE_VARIANTS = tuple(v for v in CANONICAL_VARIANT_NAMES if v != "uniform_probability")

TOP_RESULT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "power_evaluation_identity_sha256",
        "matched_identity_sha256", "git_commit", "complete", "final", "device", "gpu_model",
        "torch_version", "cuda_version", "image_count_expected", "image_count_processed",
        "image_order_digest", "windows_processed_total", "class_count", "variant_names", "metrics",
        "delta_mIoU_percentage_points_vs_uniform_probability", "metric_unit", "metric_source",
        "per_image_stats_manifest_path", "per_image_stats_manifest_sha256", "per_image_stats_npz_sha256",
        "operation_telemetry", "phase_runtime_seconds", "peak_gpu_memory_bytes", "resumed_from_checkpoint",
        "source_git_branch", "failure_reason",
    }
)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise StitchingControlIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise StitchingControlIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise StitchingControlIdentityError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StitchingControlIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise StitchingControlIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StitchingControlIdentityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StitchingControlIdentityError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise StitchingControlIdentityError(f"{label} must be at least {minimum}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise StitchingControlIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise StitchingControlIdentityError(f"{label} must be a full Git identity")
    return token


def _validate_metrics(value: Any, label: str) -> Mapping[str, float]:
    metrics = _require_closed_mapping(value, METRIC_KEYS, label)
    for name in METRIC_KEYS:
        percent = _require_float(metrics[name], f"{label}.{name}", minimum=0.0)
        if percent > 100.0:
            raise StitchingControlIdentityError(f"{label}.{name} must be a percentage in [0, 100]")
    return metrics


def verify_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> str:
    if not hasattr(record, "get"):
        raise StitchingControlIdentityError("structured result must be a mapping")
    _require_closed_mapping(record, TOP_RESULT_KEYS, "result")

    run_mode = _require_exact_string(record["run_mode"], "result.run_mode")
    if run_mode not in RUN_MODE_IMAGE_COUNT_KEYS:
        raise StitchingControlIdentityError(f"result.run_mode must be one of {sorted(RUN_MODE_IMAGE_COUNT_KEYS)}")
    expected_schema = identity["run_modes"][RUN_MODE_SCHEMA_KEYS[run_mode]]
    if _require_exact_string(record["schema"], "result.schema") != expected_schema:
        raise StitchingControlIdentityError(
            f"result.schema {record['schema']!r} does not match the {run_mode} schema {expected_schema!r}"
        )
    if _require_exact_string(record["identity"], "result.identity") != identity["identity"]["name"]:
        raise StitchingControlIdentityError("result.identity mismatch")
    if _require_sha256(record["identity_sha256"], "result.identity_sha256") != identity_sha256:
        raise StitchingControlIdentityError("result.identity_sha256 does not match the loaded identity file")
    if record["power_evaluation_identity_sha256"] != identity["parent_identity"]["power_evaluation_identity_sha256"]:
        raise StitchingControlIdentityError("result.power_evaluation_identity_sha256 disagrees with the identity's parent")
    if record["matched_identity_sha256"] != identity["parent_identity"]["matched_identity_sha256"]:
        raise StitchingControlIdentityError("result.matched_identity_sha256 disagrees with the identity's parent")
    _require_git_identity(record["git_commit"], "result.git_commit")

    if _require_bool(record["complete"], "result.complete") is not True:
        raise StitchingControlIdentityError("structured result is not complete")
    final = _require_bool(record["final"], "result.final")
    if run_mode == "full5000" and final is not True:
        raise StitchingControlIdentityError("a full5000-mode result must have final=true")
    if run_mode != "full5000" and final is not False:
        raise StitchingControlIdentityError("a pilot result must never claim final=true")

    _require_exact_string(record["device"], "result.device")
    _require_exact_string(record["gpu_model"], "result.gpu_model")
    _require_exact_string(record["torch_version"], "result.torch_version")
    _require_exact_string(record["cuda_version"], "result.cuda_version")

    expected_image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[run_mode]]
    image_count_expected = _require_int(record["image_count_expected"], "result.image_count_expected", minimum=1)
    if image_count_expected != expected_image_count:
        raise StitchingControlIdentityError(f"result.image_count_expected disagrees with the registered {run_mode} image count")
    image_count_processed = _require_int(record["image_count_processed"], "result.image_count_processed", minimum=0)
    if image_count_processed != image_count_expected:
        raise StitchingControlIdentityError("a complete result must have image_count_processed == image_count_expected")
    _require_sha256(record["image_order_digest"], "result.image_order_digest")
    windows_processed = _require_int(record["windows_processed_total"], "result.windows_processed_total", minimum=1)

    if _require_int(record["class_count"], "result.class_count", minimum=1) != identity["metrics"]["class_count"]:
        raise StitchingControlIdentityError("result.class_count disagrees with the registered class count")

    variant_names = record["variant_names"]
    if type(variant_names) is not list or tuple(variant_names) != CANONICAL_VARIANT_NAMES:
        raise StitchingControlIdentityError(f"result.variant_names must be exactly {list(CANONICAL_VARIANT_NAMES)}")

    metrics_block = _require_closed_mapping(record["metrics"], frozenset(CANONICAL_VARIANT_NAMES), "result.metrics")
    per_variant_metrics = {name: _validate_metrics(metrics_block[name], f"result.metrics.{name}") for name in CANONICAL_VARIANT_NAMES}

    deltas = _require_closed_mapping(
        record["delta_mIoU_percentage_points_vs_uniform_probability"], frozenset(NON_REFERENCE_VARIANTS),
        "result.delta_mIoU_percentage_points_vs_uniform_probability",
    )
    reference_mIoU = per_variant_metrics["uniform_probability"]["mIoU"]
    for variant in NON_REFERENCE_VARIANTS:
        delta = _require_float(deltas[variant], f"delta.{variant}")
        recomputed = float(per_variant_metrics[variant]["mIoU"]) - float(reference_mIoU)
        if abs(delta - recomputed) > 1e-6:
            raise StitchingControlIdentityError(
                f"result.delta_mIoU_percentage_points_vs_uniform_probability.{variant} is inconsistent with "
                "metrics.{variant}.mIoU - metrics.uniform_probability.mIoU"
            )

    if record["metric_unit"] != identity["metrics"]["unit"]:
        raise StitchingControlIdentityError("result.metric_unit disagrees with the registered metric unit")
    if record["metric_source"] != identity["metrics"]["precision_source"]:
        raise StitchingControlIdentityError("result.metric_source disagrees with the registered precision source")

    _require_exact_string(record["per_image_stats_manifest_path"], "result.per_image_stats_manifest_path")
    _require_sha256(record["per_image_stats_manifest_sha256"], "result.per_image_stats_manifest_sha256")
    _require_sha256(record["per_image_stats_npz_sha256"], "result.per_image_stats_npz_sha256")

    telemetry = _require_closed_mapping(record["operation_telemetry"], OPERATION_TELEMETRY_KEYS, "result.operation_telemetry")
    # Required identities: model calls == windows, graph builds == windows,
    # propagation calls == windows -- NOT windows x variants. Interpolation
    # is at most two per window (one shared probability crop, one shared
    # score crop), so probability/score interpolation calls must each be
    # <= windows_processed_total, never > it.
    for name in ("sample_pulls",):
        _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=1)
    for name in ("window_enumerations", "backbone_snapshot_calls", "graph_builds", "propagation_calls"):
        if _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=0) != windows_processed:
            raise StitchingControlIdentityError(f"operation_telemetry.{name} must equal windows_processed_total ({windows_processed})")
    for name in ("probability_interpolation_calls", "score_interpolation_calls"):
        value = _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=0)
        if value > windows_processed:
            raise StitchingControlIdentityError(f"operation_telemetry.{name} must not exceed windows_processed_total ({windows_processed})")
    if _require_int(telemetry["accumulator_finalizations"], "operation_telemetry.accumulator_finalizations", minimum=0) != image_count_processed * len(CANONICAL_VARIANT_NAMES):
        raise StitchingControlIdentityError(
            "operation_telemetry.accumulator_finalizations must equal image_count_processed * variant_count"
        )

    phase_runtime = _require_closed_mapping(record["phase_runtime_seconds"], PHASE_RUNTIME_KEYS, "result.phase_runtime_seconds")
    _require_float(phase_runtime["total"], "phase_runtime_seconds.total", minimum=0.0)
    _require_float(phase_runtime["shared"], "phase_runtime_seconds.shared", minimum=0.0)
    per_variant_runtime = _require_closed_mapping(phase_runtime["per_variant"], frozenset(CANONICAL_VARIANT_NAMES), "phase_runtime_seconds.per_variant")
    for variant in CANONICAL_VARIANT_NAMES:
        _require_float(per_variant_runtime[variant], f"phase_runtime_seconds.per_variant.{variant}", minimum=0.0)

    _require_int(record["peak_gpu_memory_bytes"], "result.peak_gpu_memory_bytes", minimum=0)
    _require_bool(record["resumed_from_checkpoint"], "result.resumed_from_checkpoint")
    _require_exact_string(record["source_git_branch"], "result.source_git_branch")
    if record["failure_reason"] is not None:
        _require_exact_string(record["failure_reason"], "result.failure_reason")

    uniform_mIoU = per_variant_metrics["uniform_probability"]["mIoU"]
    return (
        f"STITCHING CONTROL SUITE RESULT PASS run_mode={run_mode} images={image_count_processed} "
        f"uniform_probability_mIoU={uniform_mIoU:.6f}"
    )


def verify_result(path: Path, *, identity_path: Path | None = None, repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    resolved_identity_path = Path(identity_path) if identity_path is not None else root / "evaluation_identities/e12_stitching_control_suite.toml"
    identity_sha256 = _sha256_file(resolved_identity_path)
    record = parse_strict_json_document(Path(path), label="structured result")
    return verify_record(record, identity, identity_sha256=identity_sha256)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "NON_REFERENCE_VARIANTS",
    "OPERATION_TELEMETRY_KEYS",
    "TOP_RESULT_KEYS",
    "verify_record",
    "verify_result",
]
