"""Structured result and checkpoint schema validation for the bounded
k11/k12 finite-step stability gate.

Reuses :mod:`src.k11_k12_stability_gate_identity` for identity loading and
TOML-defined tolerances/thresholds; never duplicates a canonical
diagnostic-only setting as a Python literal. Regime classification is
derived exclusively from TOML-defined thresholds via
:func:`classify_regime`.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from src.k11_k12_stability_gate_identity import (
    CHECKPOINT_SCHEMA_NAME,
    K11K12StabilityGateError,
    RESULT_SCHEMA_NAME,
    SUPPORTED_REGIME_CLASSIFICATIONS,
    load_identity,
    parse_structured_document,
    repository_root,
)


GRAPH_DIAGNOSTICS_KEYS = frozenset(
    {
        "prefix_mismatch_count", "fallback_row_count_k12", "fallback_row_count_k11",
        "fallback_row_mismatch_count", "tie_row_count", "row_sum_max_error_k11",
        "row_sum_max_error_k12", "negative_weight_count_k11", "negative_weight_count_k12",
        "non_fallback_self_edge_count_k11", "non_fallback_self_edge_count_k12",
        "directed_asymmetry_fraction_k12",
    }
)
SNAPSHOT_COMPARISON_KEYS = frozenset(
    {
        "max_absolute_error", "mean_absolute_error", "relative_frobenius_error",
        "argmax_disagreement_count", "argmax_disagreement_rate",
    }
)
RECURRENCE_DIAGNOSTICS_KEYS = frozenset(
    {
        "steps_completed_k11", "steps_completed_k12", "snapshot_steps",
        "early_termination", "solver_fallback_used", "cgls_call_count",
        "dense_solve_call_count_in_production_path",
        "t160_t320_k11", "t160_t320_k12", "t320_t640_k11", "t320_t640_k12",
        "t320_t640_relative_frobenius_change_mean",
    }
)
REFERENCE_DIAGNOSTICS_KEYS = frozenset(
    {
        "fp32_fp64_t160_k11", "fp32_fp64_t160_k12", "fp32_fp64_t320_k11", "fp32_fp64_t320_k12",
        "fp32_fp64_t640_k11", "fp32_fp64_t640_k12",
        "dense_equilibrium_t320_k11", "dense_equilibrium_t320_k12",
        "dense_equilibrium_t640_k11", "dense_equilibrium_t640_k12",
        "dense_residual_relative_k11", "dense_residual_relative_k12",
        "dense_backward_error_k11", "dense_backward_error_k12",
    }
)
CONDITION_DIAGNOSTICS_KEYS = frozenset(
    {
        "window_count", "sigma_min_min", "sigma_min_max", "kappa_2_min", "kappa_2_max",
        "kappa_2_mean", "departure_from_normality_illustrative_mean",
    }
)
MATCHED_DELTA_DIAGNOSTICS_KEYS = frozenset(
    {
        "d160_norm_mean", "d320_norm_mean", "d640_norm_mean", "d320_relative_norm_mean",
        "d160_d320_stability_error_mean", "d320_d640_stability_error_mean",
        "argmax_disagreement_rate_k11_k12_t320_mean",
        "label_sensitivity_argmax_disagreement_rate_mean",
    }
)
DETERMINISM_DIAGNOSTICS_KEYS = frozenset(
    {
        "replay_count", "manifest_matches", "graph_indices_match", "graph_weights_match",
        "p160_match", "p320_match", "p640_match", "argmax_digest_match",
        "drift_detected", "drift_description",
    }
)
RESUMABILITY_KEYS = frozenset({"resumed_from_checkpoint", "resumed_window_count", "checkpoint_path"})
PROVENANCE_KEYS = frozenset({"source_git_branch", "source_git_dirty", "elapsed_seconds_total"})
PHASE_RUNTIME_KEYS = frozenset(
    {"snapshot_extraction", "graph_construction", "finite_step_propagation", "reference_computation", "reporting"}
)
TOP_RESULT_KEYS = frozenset(
    {
        "schema", "identity", "identity_sha256", "matched_identity_sha256", "git_commit",
        "checkpoint_sha256", "complete", "final", "device", "gpu_model", "torch_version",
        "cuda_version", "manifest_digest", "windows_expected", "windows_processed",
        "graph_diagnostics", "recurrence_diagnostics", "reference_diagnostics",
        "condition_diagnostics", "matched_delta_diagnostics", "determinism_diagnostics",
        "phase_runtime_seconds", "peak_gpu_memory_bytes", "gate_classification",
        "numerical_validity_passed", "failure_reason", "resumability", "provenance",
    }
)
CHECKPOINT_WINDOW_ENTRY_KEYS = frozenset({"window_index", "image_id", "sha256"})
TOP_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "identity", "identity_sha256", "manifest_digest", "windows_expected",
        "complete", "windows", "created_at_utc", "updated_at_utc",
    }
)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise K11K12StabilityGateError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise K11K12StabilityGateError(f"{label} must be an exact non-empty string")
    return value


def _require_json_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise K11K12StabilityGateError(f"{label} must be an exact JSON integer")
    if minimum is not None and value < minimum:
        raise K11K12StabilityGateError(f"{label} must be at least {minimum}")
    return value


def _require_json_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise K11K12StabilityGateError(f"{label} must be an exact JSON boolean")
    return value


def _require_json_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, Decimal):
        raise K11K12StabilityGateError(f"{label} must be a JSON floating-point number")
    if not value.is_finite():
        raise K11K12StabilityGateError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise K11K12StabilityGateError(f"{label} is outside finite float range")
    if minimum is not None and result < minimum:
        raise K11K12StabilityGateError(f"{label} must be at least {minimum}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise K11K12StabilityGateError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise K11K12StabilityGateError(f"{label} must be a full Git identity")
    return token


def _validate_snapshot_comparison(value: Any, label: str) -> Mapping[str, Any]:
    comparison = _require_closed_mapping(value, SNAPSHOT_COMPARISON_KEYS, label)
    _require_json_float(comparison["max_absolute_error"], f"{label}.max_absolute_error", minimum=0.0)
    _require_json_float(comparison["mean_absolute_error"], f"{label}.mean_absolute_error", minimum=0.0)
    _require_json_float(comparison["relative_frobenius_error"], f"{label}.relative_frobenius_error", minimum=0.0)
    _require_json_int(comparison["argmax_disagreement_count"], f"{label}.argmax_disagreement_count", minimum=0)
    rate = _require_json_float(comparison["argmax_disagreement_rate"], f"{label}.argmax_disagreement_rate", minimum=0.0)
    if rate > 1.0:
        raise K11K12StabilityGateError(f"{label}.argmax_disagreement_rate must be in [0, 1]")
    return comparison


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def classify_regime(
    *,
    numerical_validity_passed: bool,
    d320_relative_norm: float,
    t320_t640_relative_change: float,
    thresholds: Mapping[str, Any],
) -> str:
    """Derive the regime classification exclusively from TOML-defined
    thresholds -- never claims k11 helps or hurts mIoU, and never reuses a
    label from an adjacent branch for a case that branch does not actually
    describe.

    Classification order (every step is load-bearing; do not reorder):

    1. Defensively validate the three registered thresholds are real,
       finite numbers (defends against a malformed identity reaching this
       function; the identity loader itself must already reject this).
    2. ``numerical_validity_passed`` must be exactly ``True``, else INVALID.
    3. ``d320_relative_norm``/``t320_t640_relative_change`` must both be
       real, finite numbers, else INVALID -- a NaN/Infinity diagnostic
       (e.g. from a genuine numerical breakdown upstream) must never be
       silently reported as a specific, false regime.
    4. Truncation sensitivity takes precedence over any effect-size claim.
    5. Equality at the near-noise maximum belongs to near-noise.
    6. Equality at the clear-signal minimum belongs to clear signal.
    7. Anything strictly between the two thresholds is an honest
       "inconclusive" result -- it is neither near noise nor clear signal,
       and must not be mislabeled as either.
    """
    for label, value in (
        ("gate_thresholds.effect_near_noise_relative_delta_norm_max", thresholds["effect_near_noise_relative_delta_norm_max"]),
        ("gate_thresholds.truncation_sensitive_relative_change_min", thresholds["truncation_sensitive_relative_change_min"]),
        ("gate_thresholds.clear_signal_relative_delta_norm_min", thresholds["clear_signal_relative_delta_norm_min"]),
    ):
        if not _finite_number(value):
            raise K11K12StabilityGateError(f"classify_regime: {label} must be a finite number, observed {value!r}")

    if numerical_validity_passed is not True:
        return "INVALID"
    if not _finite_number(d320_relative_norm) or not _finite_number(t320_t640_relative_change):
        return "INVALID"

    if t320_t640_relative_change >= thresholds["truncation_sensitive_relative_change_min"]:
        return "TRUNCATION_SENSITIVE"
    if d320_relative_norm <= thresholds["effect_near_noise_relative_delta_norm_max"]:
        return "NUMERICALLY_STABLE_BUT_EFFECT_NEAR_NOISE"
    if d320_relative_norm >= thresholds["clear_signal_relative_delta_norm_min"]:
        return "CLEAR_MATCHED_SIGNAL"
    return "NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE"


def verify_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> str:
    if not hasattr(record, "get"):
        raise K11K12StabilityGateError("structured result must be a mapping")
    _require_closed_mapping(record, TOP_RESULT_KEYS, "structured result")

    if _require_exact_string(record["schema"], "result.schema") != RESULT_SCHEMA_NAME:
        raise K11K12StabilityGateError("structured result schema mismatch")
    if _require_exact_string(record["identity"], "result.identity") != identity["identity"]["name"]:
        raise K11K12StabilityGateError("structured result identity mismatch")
    if _require_sha256(record["identity_sha256"], "result.identity_sha256") != identity_sha256:
        raise K11K12StabilityGateError("structured result identity_sha256 does not match the loaded identity file")
    _require_sha256(record["matched_identity_sha256"], "result.matched_identity_sha256")
    if record["matched_identity_sha256"] != identity["parent_identity"]["matched_identity_sha256"]:
        raise K11K12StabilityGateError("result matched_identity_sha256 disagrees with the gate identity's parent")
    _require_git_identity(record["git_commit"], "result.git_commit")
    if record["checkpoint_sha256"] is not None:
        _require_sha256(record["checkpoint_sha256"], "result.checkpoint_sha256")
    if _require_json_bool(record["complete"], "result.complete") is not True:
        raise K11K12StabilityGateError("structured result is not complete")
    if _require_json_bool(record["final"], "result.final") is not True:
        raise K11K12StabilityGateError("structured result is not final")

    _require_exact_string(record["device"], "result.device")
    _require_exact_string(record["gpu_model"], "result.gpu_model")
    _require_exact_string(record["torch_version"], "result.torch_version")
    _require_exact_string(record["cuda_version"], "result.cuda_version")
    _require_sha256(record["manifest_digest"], "result.manifest_digest")

    windows_expected = _require_json_int(record["windows_expected"], "result.windows_expected", minimum=1)
    if windows_expected != identity["sample_selection"]["canonical_window_count"]:
        raise K11K12StabilityGateError("result.windows_expected disagrees with sample_selection.canonical_window_count")
    windows_processed = _require_json_int(record["windows_processed"], "result.windows_processed", minimum=0)
    if windows_processed != windows_expected:
        raise K11K12StabilityGateError("a complete/final result must have windows_processed == windows_expected")

    graph = _require_closed_mapping(record["graph_diagnostics"], GRAPH_DIAGNOSTICS_KEYS, "result.graph_diagnostics")
    if _require_json_int(graph["prefix_mismatch_count"], "graph_diagnostics.prefix_mismatch_count", minimum=0) != 0:
        raise K11K12StabilityGateError("graph_diagnostics.prefix_mismatch_count must be 0")
    if _require_json_int(graph["fallback_row_mismatch_count"], "graph_diagnostics.fallback_row_mismatch_count", minimum=0) != 0:
        raise K11K12StabilityGateError("graph_diagnostics.fallback_row_mismatch_count must be 0")
    for name in (
        "fallback_row_count_k12", "fallback_row_count_k11", "tie_row_count",
        "negative_weight_count_k11", "negative_weight_count_k12",
        "non_fallback_self_edge_count_k11", "non_fallback_self_edge_count_k12",
    ):
        _require_json_int(graph[name], f"graph_diagnostics.{name}", minimum=0)
    if _require_json_int(graph["negative_weight_count_k11"], "graph_diagnostics.negative_weight_count_k11") != 0:
        raise K11K12StabilityGateError("graph_diagnostics.negative_weight_count_k11 must be 0")
    if _require_json_int(graph["negative_weight_count_k12"], "graph_diagnostics.negative_weight_count_k12") != 0:
        raise K11K12StabilityGateError("graph_diagnostics.negative_weight_count_k12 must be 0")
    if _require_json_int(graph["non_fallback_self_edge_count_k11"], "graph_diagnostics.non_fallback_self_edge_count_k11") != 0:
        raise K11K12StabilityGateError("graph_diagnostics.non_fallback_self_edge_count_k11 must be 0")
    if _require_json_int(graph["non_fallback_self_edge_count_k12"], "graph_diagnostics.non_fallback_self_edge_count_k12") != 0:
        raise K11K12StabilityGateError("graph_diagnostics.non_fallback_self_edge_count_k12 must be 0")
    _require_json_float(graph["row_sum_max_error_k11"], "graph_diagnostics.row_sum_max_error_k11", minimum=0.0)
    _require_json_float(graph["row_sum_max_error_k12"], "graph_diagnostics.row_sum_max_error_k12", minimum=0.0)
    directed_asymmetry = _require_json_float(
        graph["directed_asymmetry_fraction_k12"], "graph_diagnostics.directed_asymmetry_fraction_k12", minimum=0.0
    )
    if directed_asymmetry > 1.0:
        raise K11K12StabilityGateError("graph_diagnostics.directed_asymmetry_fraction_k12 must be in [0, 1]")

    recurrence = _require_closed_mapping(
        record["recurrence_diagnostics"], RECURRENCE_DIAGNOSTICS_KEYS, "result.recurrence_diagnostics"
    )
    for name in ("steps_completed_k11", "steps_completed_k12"):
        if _require_json_int(recurrence[name], f"recurrence_diagnostics.{name}", minimum=1) != 640:
            raise K11K12StabilityGateError(f"recurrence_diagnostics.{name} must be exactly 640")
    snapshot_steps = recurrence["snapshot_steps"]
    if type(snapshot_steps) is not list or [
        _require_json_int(s, "recurrence_diagnostics.snapshot_steps[i]", minimum=0) for s in snapshot_steps
    ] != [160, 320, 640]:
        raise K11K12StabilityGateError("recurrence_diagnostics.snapshot_steps must be exactly [160, 320, 640]")
    if _require_json_bool(recurrence["early_termination"], "recurrence_diagnostics.early_termination") is not False:
        raise K11K12StabilityGateError("recurrence_diagnostics.early_termination must be false")
    if _require_json_bool(recurrence["solver_fallback_used"], "recurrence_diagnostics.solver_fallback_used") is not False:
        raise K11K12StabilityGateError("recurrence_diagnostics.solver_fallback_used must be false")
    if _require_json_int(recurrence["cgls_call_count"], "recurrence_diagnostics.cgls_call_count", minimum=0) != 0:
        raise K11K12StabilityGateError("recurrence_diagnostics.cgls_call_count must be 0")
    if _require_json_int(
        recurrence["dense_solve_call_count_in_production_path"],
        "recurrence_diagnostics.dense_solve_call_count_in_production_path", minimum=0,
    ) != 0:
        raise K11K12StabilityGateError(
            "recurrence_diagnostics.dense_solve_call_count_in_production_path must be 0"
        )
    for name in ("t160_t320_k11", "t160_t320_k12", "t320_t640_k11", "t320_t640_k12"):
        _validate_snapshot_comparison(recurrence[name], f"recurrence_diagnostics.{name}")
    t320_t640_relative_change = _require_json_float(
        recurrence["t320_t640_relative_frobenius_change_mean"],
        "recurrence_diagnostics.t320_t640_relative_frobenius_change_mean", minimum=0.0,
    )
    if t320_t640_relative_change > identity["tolerances"]["t_snapshot_relative_frobenius_change_max"] * 100:
        # Sanity bound only (100x the pass/fail threshold): catches garbage
        # values without duplicating the actual gate-classification logic.
        raise K11K12StabilityGateError(
            "recurrence_diagnostics.t320_t640_relative_frobenius_change_mean is implausibly large"
        )

    reference = _require_closed_mapping(
        record["reference_diagnostics"], REFERENCE_DIAGNOSTICS_KEYS, "result.reference_diagnostics"
    )
    fp32_fp64_tolerance = identity["tolerances"]["fp32_fp64_relative_frobenius_error_max"]
    for name in (
        "fp32_fp64_t160_k11", "fp32_fp64_t160_k12", "fp32_fp64_t320_k11", "fp32_fp64_t320_k12",
        "fp32_fp64_t640_k11", "fp32_fp64_t640_k12",
    ):
        comparison = _validate_snapshot_comparison(reference[name], f"reference_diagnostics.{name}")
        if float(comparison["relative_frobenius_error"]) > fp32_fp64_tolerance:
            raise K11K12StabilityGateError(
                f"reference_diagnostics.{name}.relative_frobenius_error exceeds the registered tolerance"
            )
    for name in (
        "dense_equilibrium_t320_k11", "dense_equilibrium_t320_k12",
        "dense_equilibrium_t640_k11", "dense_equilibrium_t640_k12",
    ):
        _validate_snapshot_comparison(reference[name], f"reference_diagnostics.{name}")
    residual_tolerance = identity["tolerances"]["dense_equilibrium_residual_relative_max"]
    for name in ("dense_residual_relative_k11", "dense_residual_relative_k12"):
        residual = _require_json_float(reference[name], f"reference_diagnostics.{name}", minimum=0.0)
        if residual > residual_tolerance:
            raise K11K12StabilityGateError(f"reference_diagnostics.{name} exceeds the registered tolerance")
    for name in ("dense_backward_error_k11", "dense_backward_error_k12"):
        _require_json_float(reference[name], f"reference_diagnostics.{name}", minimum=0.0)

    condition = _require_closed_mapping(
        record["condition_diagnostics"], CONDITION_DIAGNOSTICS_KEYS, "result.condition_diagnostics"
    )
    _require_json_int(condition["window_count"], "condition_diagnostics.window_count", minimum=1)
    for name in ("sigma_min_min", "sigma_min_max", "kappa_2_min", "kappa_2_max", "kappa_2_mean"):
        value = _require_json_float(condition[name], f"condition_diagnostics.{name}")
        if value <= 0:
            raise K11K12StabilityGateError(f"condition_diagnostics.{name} must be positive")
    _require_json_float(
        condition["departure_from_normality_illustrative_mean"],
        "condition_diagnostics.departure_from_normality_illustrative_mean", minimum=0.0,
    )

    matched_delta = _require_closed_mapping(
        record["matched_delta_diagnostics"], MATCHED_DELTA_DIAGNOSTICS_KEYS, "result.matched_delta_diagnostics"
    )
    for name in ("d160_norm_mean", "d320_norm_mean", "d640_norm_mean"):
        _require_json_float(matched_delta[name], f"matched_delta_diagnostics.{name}", minimum=0.0)
    d320_relative_norm = _require_json_float(
        matched_delta["d320_relative_norm_mean"], "matched_delta_diagnostics.d320_relative_norm_mean", minimum=0.0
    )
    for name in ("d160_d320_stability_error_mean", "d320_d640_stability_error_mean"):
        _require_json_float(matched_delta[name], f"matched_delta_diagnostics.{name}", minimum=0.0)
    for name in ("argmax_disagreement_rate_k11_k12_t320_mean", "label_sensitivity_argmax_disagreement_rate_mean"):
        rate = _require_json_float(matched_delta[name], f"matched_delta_diagnostics.{name}", minimum=0.0)
        if rate > 1.0:
            raise K11K12StabilityGateError(f"matched_delta_diagnostics.{name} must be in [0, 1]")

    determinism = _require_closed_mapping(
        record["determinism_diagnostics"], DETERMINISM_DIAGNOSTICS_KEYS, "result.determinism_diagnostics"
    )
    replay_count = _require_json_int(determinism["replay_count"], "determinism_diagnostics.replay_count", minimum=2)
    if replay_count != identity["tolerances"]["determinism_replay_count"]:
        raise K11K12StabilityGateError("determinism_diagnostics.replay_count disagrees with the registered gate setting")
    for name in (
        "manifest_matches", "graph_indices_match", "graph_weights_match",
        "p160_match", "p320_match", "p640_match", "argmax_digest_match",
    ):
        if _require_json_bool(determinism[name], f"determinism_diagnostics.{name}") is not True:
            raise K11K12StabilityGateError(f"determinism_diagnostics.{name} must be true for a passing gate result")
    drift_detected = _require_json_bool(determinism["drift_detected"], "determinism_diagnostics.drift_detected")
    if drift_detected is not False:
        raise K11K12StabilityGateError("determinism_diagnostics.drift_detected must be false for a passing gate result")
    _require_exact_string(determinism["drift_description"], "determinism_diagnostics.drift_description", nonempty=False)

    phase_runtime = _require_closed_mapping(
        record["phase_runtime_seconds"], PHASE_RUNTIME_KEYS, "result.phase_runtime_seconds"
    )
    for name in PHASE_RUNTIME_KEYS:
        _require_json_float(phase_runtime[name], f"phase_runtime_seconds.{name}", minimum=0.0)
    _require_json_int(record["peak_gpu_memory_bytes"], "result.peak_gpu_memory_bytes", minimum=0)

    numerical_validity_passed = _require_json_bool(
        record["numerical_validity_passed"], "result.numerical_validity_passed"
    )
    gate_classification = _require_exact_string(record["gate_classification"], "result.gate_classification")
    if gate_classification not in SUPPORTED_REGIME_CLASSIFICATIONS:
        raise K11K12StabilityGateError(
            f"result.gate_classification must be one of {list(SUPPORTED_REGIME_CLASSIFICATIONS)}"
        )
    expected_classification = classify_regime(
        numerical_validity_passed=numerical_validity_passed,
        d320_relative_norm=d320_relative_norm,
        t320_t640_relative_change=t320_t640_relative_change,
        thresholds=identity["gate_thresholds"],
    )
    if gate_classification != expected_classification:
        raise K11K12StabilityGateError(
            "result.gate_classification is inconsistent with TOML-defined thresholds: "
            f"reported {gate_classification!r}, recomputed {expected_classification!r}"
        )
    if record["failure_reason"] is not None:
        _require_exact_string(record["failure_reason"], "result.failure_reason")

    resumability = _require_closed_mapping(record["resumability"], RESUMABILITY_KEYS, "result.resumability")
    _require_json_bool(resumability["resumed_from_checkpoint"], "resumability.resumed_from_checkpoint")
    _require_json_int(resumability["resumed_window_count"], "resumability.resumed_window_count", minimum=0)
    if resumability["checkpoint_path"] is not None:
        _require_exact_string(resumability["checkpoint_path"], "resumability.checkpoint_path")

    provenance = _require_closed_mapping(record["provenance"], PROVENANCE_KEYS, "result.provenance")
    _require_exact_string(provenance["source_git_branch"], "provenance.source_git_branch")
    _require_json_bool(provenance["source_git_dirty"], "provenance.source_git_dirty")
    _require_json_float(provenance["elapsed_seconds_total"], "provenance.elapsed_seconds_total", minimum=0.0)

    return (
        "K11/K12 STABILITY GATE RESULT PASS "
        f"windows={windows_processed} classification={gate_classification} "
        f"numerical_validity_passed={str(numerical_validity_passed).lower()}"
    )


def verify_checkpoint_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> str:
    if not hasattr(record, "get"):
        raise K11K12StabilityGateError("structured checkpoint must be a mapping")
    _require_closed_mapping(record, TOP_CHECKPOINT_KEYS, "checkpoint")
    if _require_exact_string(record["schema"], "checkpoint.schema") != CHECKPOINT_SCHEMA_NAME:
        raise K11K12StabilityGateError("structured checkpoint schema mismatch")
    if _require_exact_string(record["identity"], "checkpoint.identity") != identity["identity"]["name"]:
        raise K11K12StabilityGateError("structured checkpoint identity mismatch")
    if _require_sha256(record["identity_sha256"], "checkpoint.identity_sha256") != identity_sha256:
        raise K11K12StabilityGateError("checkpoint identity_sha256 does not match the loaded identity file")
    _require_sha256(record["manifest_digest"], "checkpoint.manifest_digest")
    windows_expected = _require_json_int(record["windows_expected"], "checkpoint.windows_expected", minimum=1)
    if windows_expected != identity["sample_selection"]["canonical_window_count"]:
        raise K11K12StabilityGateError("checkpoint.windows_expected disagrees with sample_selection.canonical_window_count")
    complete = _require_json_bool(record["complete"], "checkpoint.complete")

    windows = record["windows"]
    if type(windows) is not list:
        raise K11K12StabilityGateError("checkpoint.windows must be a list")
    seen_indices: set[int] = set()
    for position, entry in enumerate(windows):
        validated = _require_closed_mapping(entry, CHECKPOINT_WINDOW_ENTRY_KEYS, f"checkpoint.windows[{position}]")
        window_index = _require_json_int(
            validated["window_index"], f"checkpoint.windows[{position}].window_index", minimum=0
        )
        if window_index in seen_indices:
            raise K11K12StabilityGateError(f"checkpoint.windows contains a duplicate window_index {window_index}")
        if window_index != position:
            raise K11K12StabilityGateError(
                f"checkpoint.windows must be in contiguous row-major order; expected index {position}, got {window_index}"
            )
        seen_indices.add(window_index)
        _require_exact_string(validated["image_id"], f"checkpoint.windows[{position}].image_id")
        _require_sha256(validated["sha256"], f"checkpoint.windows[{position}].sha256")
    if len(windows) > windows_expected:
        raise K11K12StabilityGateError("checkpoint.windows exceeds windows_expected")
    if complete and len(windows) != windows_expected:
        raise K11K12StabilityGateError(
            "a checkpoint marked complete must have exactly windows_expected windows recorded"
        )

    _require_exact_string(record["created_at_utc"], "checkpoint.created_at_utc")
    _require_exact_string(record["updated_at_utc"], "checkpoint.updated_at_utc")

    status = "complete" if complete else f"resumable at window {len(windows)}"
    return f"K11/K12 STABILITY GATE CHECKPOINT PASS windows_recorded={len(windows)} status={status}"


def verify_result(path: Path, *, identity_path: Path | None = None, repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    resolved_identity_path = Path(identity_path) if identity_path is not None else root / "evaluation_identities/e12_k11_k12_stability_gate.toml"
    try:
        identity_sha256 = hashlib.sha256(resolved_identity_path.read_bytes()).hexdigest()
    except OSError as error:
        raise K11K12StabilityGateError(f"cannot hash identity file {resolved_identity_path}: {error}") from error
    record = parse_structured_document(path, label="structured result")
    return verify_record(record, identity, identity_sha256=identity_sha256)


def verify_checkpoint(path: Path, *, identity_path: Path | None = None, repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    resolved_identity_path = Path(identity_path) if identity_path is not None else root / "evaluation_identities/e12_k11_k12_stability_gate.toml"
    try:
        identity_sha256 = hashlib.sha256(resolved_identity_path.read_bytes()).hexdigest()
    except OSError as error:
        raise K11K12StabilityGateError(f"cannot hash identity file {resolved_identity_path}: {error}") from error
    record = parse_structured_document(path, label="structured checkpoint")
    return verify_checkpoint_record(record, identity, identity_sha256=identity_sha256)


def resume_window_start(checkpoint_record: Mapping[str, Any]) -> int:
    """Given an already-validated checkpoint record, return the window
    index a resumed run must start at. Raises if the checkpoint is already
    complete: a completed checkpoint must never be resumed as though it
    were incomplete (that would silently duplicate or skip windows)."""
    if checkpoint_record["complete"] is True:
        raise K11K12StabilityGateError(
            "checkpoint is already complete; refusing to resume it as though incomplete"
        )
    return len(checkpoint_record["windows"])


def write_checkpoint_atomically(path: Path, record: Mapping[str, Any]) -> None:
    """Atomic-replacement checkpoint write: write to a sibling temp file,
    then ``os.replace`` it into place, so a crash mid-write never leaves a
    corrupt/partial checkpoint at ``path``."""
    import os

    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


__all__ = [
    "CONDITION_DIAGNOSTICS_KEYS",
    "DETERMINISM_DIAGNOSTICS_KEYS",
    "GRAPH_DIAGNOSTICS_KEYS",
    "MATCHED_DELTA_DIAGNOSTICS_KEYS",
    "PHASE_RUNTIME_KEYS",
    "PROVENANCE_KEYS",
    "RECURRENCE_DIAGNOSTICS_KEYS",
    "REFERENCE_DIAGNOSTICS_KEYS",
    "RESUMABILITY_KEYS",
    "SNAPSHOT_COMPARISON_KEYS",
    "TOP_CHECKPOINT_KEYS",
    "TOP_RESULT_KEYS",
    "classify_regime",
    "resume_window_start",
    "verify_checkpoint",
    "verify_checkpoint_record",
    "verify_record",
    "verify_result",
    "write_checkpoint_atomically",
]
