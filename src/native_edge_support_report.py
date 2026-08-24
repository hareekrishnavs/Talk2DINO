"""Structured result schema validation for the native cross-view
directed-edge support audit.

Never renders a segmentation-accuracy or pruning-efficacy verdict -- this
only verifies a result is structurally complete, internally consistent
(recomputes funnel/telemetry identities rather than trusting an isolated
scalar), and correctly bound to its identity/run-mode. Checkpoint
validation lives exclusively in :mod:`src.native_edge_support_checkpoint`
-- this module never duplicates any of it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from src.native_edge_support_checkpoint import (
    AFFINITY_RANK_KEYS,
    CROP_EDGE_BAND_KEYS,
    CROSS_TAB_ROW_KEYS,
    CROSS_TAB_TABLE_NAMES,
    DECISION_OUTCOMES,
    DISPLACEMENT_BAND_KEYS,
    FUNNEL_KEYS,
    IMAGE_EDGE_BAND_KEYS,
    SUPPORT_COUNT_HISTOGRAM_KEYS,
    SUPPORT_FRACTION_HISTOGRAM_KEYS,
    UNDEFINED_REASON_KEYS,
    parse_strict_json_document,
    validate_funnel_invariants,
)
from src.native_edge_support_identity import (
    RUN_MODE_IMAGE_COUNT_KEYS,
    RUN_MODE_SCHEMA_KEYS,
    NativeEdgeSupportAuditIdentityError,
    load_identity,
    repository_root,
)

OPERATION_TELEMETRY_KEYS = frozenset(
    {
        "sample_pulls", "window_enumerations", "backbone_snapshot_calls", "graph_builds",
        "propagation_calls", "probability_interpolation_calls", "cross_view_comparisons_total",
    }
)
PHASE_RUNTIME_KEYS = frozenset({"total", "shared"})

TOP_RESULT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "stitching_control_identity_sha256",
        "power_evaluation_identity_sha256", "matched_identity_sha256", "e3_identity_sha256", "git_commit",
        "complete", "final", "device", "gpu_model", "torch_version", "cuda_version", "image_count_expected",
        "image_count_processed", "image_order_digest", "windows_processed_total", "rows_processed_total",
        "edges_processed_total", "class_count", "funnel", "undefined_reason_counts", "support_count_histogram",
        "support_fraction_histogram", "crop_edge_band_histogram", "image_edge_band_histogram",
        "displacement_band_histogram", "affinity_rank_histogram", "correctness_cross_tabs", "ignored_gt_count",
        "ranking_definition", "decision_output", "decision_rationale", "per_image_stats_manifest_path",
        "per_image_stats_manifest_sha256", "operation_telemetry", "phase_runtime_seconds", "peak_gpu_memory_bytes",
        "resumed_from_checkpoint", "source_git_branch", "failure_reason",
    }
)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise NativeEdgeSupportAuditIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be at least {minimum}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be a full Git identity")
    return token


def _require_counter_mapping(value: Any, keys: tuple[str, ...], label: str) -> dict[str, int]:
    mapping = _require_closed_mapping(value, frozenset(keys), label)
    return {key: _require_int(mapping[key], f"{label}.{key}", minimum=0) for key in keys}


def classify_reachability(
    funnel: Mapping[str, int], identity: Mapping[str, Any], *, ignored_gt_count: int = 0,
) -> tuple[str, str]:
    """Descriptive-only classification of whether native cross-view
    support is defined where current predictions are currently wrong
    (Section 17). Every threshold used here comes from
    ``identity["decision"]`` -- never a bare Python literal -- and this
    function never selects, prunes, or edits anything; it only labels the
    funnel that has already been computed. ``ignored_gt_count`` is a
    separate top-level result field (not part of ``funnel`` itself),
    threaded through explicitly rather than guessed with a default.
    Returns ``(outcome, rationale)``."""
    decision = identity["decision"]
    misclassified = funnel["misclassified_rows"]
    minimum_rows = decision["minimum_misclassified_rows_for_conclusive"]
    if misclassified < minimum_rows:
        return "INCONCLUSIVE", (
            f"only {misclassified} misclassified rows observed, below the identity-declared minimum "
            f"of {minimum_rows} required for a conclusive read"
        )

    undefined_edges = funnel["directed_edges"] - funnel["edges_with_observer"]
    undefined_fraction = undefined_edges / funnel["directed_edges"] if funnel["directed_edges"] > 0 else 1.0
    alignment_threshold = decision["alignment_limited_min_undefined_edge_fraction"]
    if undefined_fraction >= alignment_threshold:
        return "ALIGNMENT_LIMITED", (
            f"undefined-support edge fraction {undefined_fraction:.4f} meets or exceeds the identity-declared "
            f"alignment-limited threshold {alignment_threshold}"
        )

    fraction_wrong_defined = funnel["misclassified_rows_any_defined"] / misclassified
    correct_rows = funnel["graph_rows"] - ignored_gt_count - misclassified
    correct_rows_any_defined = funnel["correct_rows_any_defined"]
    fraction_correct_defined = correct_rows_any_defined / correct_rows if correct_rows > 0 else 0.0

    risk_ratio_threshold = decision["reachable_min_risk_ratio"]
    if fraction_correct_defined <= 0.0:
        risk_ratio = float("inf") if fraction_wrong_defined > 0 else 0.0
    else:
        risk_ratio = fraction_wrong_defined / fraction_correct_defined

    if risk_ratio > risk_ratio_threshold:
        return "REACHABLE", (
            f"support-defined coverage among wrong rows ({fraction_wrong_defined:.4f}) exceeds coverage among "
            f"correct rows ({fraction_correct_defined:.4f}); risk ratio {risk_ratio:.4f} > "
            f"identity-declared threshold {risk_ratio_threshold}"
        )
    return "STRUCTURALLY_UNREACHABLE", (
        f"support-defined coverage among wrong rows ({fraction_wrong_defined:.4f}) does not exceed coverage among "
        f"correct rows ({fraction_correct_defined:.4f}); risk ratio {risk_ratio:.4f} <= "
        f"identity-declared threshold {risk_ratio_threshold}"
    )


def verify_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> str:
    if not hasattr(record, "get"):
        raise NativeEdgeSupportAuditIdentityError("structured result must be a mapping")
    _require_closed_mapping(record, TOP_RESULT_KEYS, "result")

    run_mode = _require_exact_string(record["run_mode"], "result.run_mode")
    if run_mode not in RUN_MODE_IMAGE_COUNT_KEYS:
        raise NativeEdgeSupportAuditIdentityError(f"result.run_mode must be one of {sorted(RUN_MODE_IMAGE_COUNT_KEYS)}")
    expected_schema = identity["run_modes"][RUN_MODE_SCHEMA_KEYS[run_mode]]
    if _require_exact_string(record["schema"], "result.schema") != expected_schema:
        raise NativeEdgeSupportAuditIdentityError(
            f"result.schema {record['schema']!r} does not match the {run_mode} schema {expected_schema!r}"
        )
    if _require_exact_string(record["identity"], "result.identity") != identity["identity"]["name"]:
        raise NativeEdgeSupportAuditIdentityError("result.identity mismatch")
    if _require_sha256(record["identity_sha256"], "result.identity_sha256") != identity_sha256:
        raise NativeEdgeSupportAuditIdentityError("result.identity_sha256 does not match the loaded identity file")
    for prefix in ("stitching_control", "power_evaluation", "matched", "e3"):
        if record[f"{prefix}_identity_sha256"] != identity["parent_identity"][f"{prefix}_identity_sha256"]:
            raise NativeEdgeSupportAuditIdentityError(f"result.{prefix}_identity_sha256 disagrees with the identity's parent")
    _require_git_identity(record["git_commit"], "result.git_commit")

    if _require_bool(record["complete"], "result.complete") is not True:
        raise NativeEdgeSupportAuditIdentityError("structured result is not complete")
    if _require_bool(record["final"], "result.final") is not False:
        raise NativeEdgeSupportAuditIdentityError("a mechanics20 audit result must never claim final=true -- this stage is structural audit only")

    _require_exact_string(record["device"], "result.device")
    _require_exact_string(record["gpu_model"], "result.gpu_model")
    _require_exact_string(record["torch_version"], "result.torch_version")
    _require_exact_string(record["cuda_version"], "result.cuda_version")

    expected_image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[run_mode]]
    image_count_expected = _require_int(record["image_count_expected"], "result.image_count_expected", minimum=1)
    if image_count_expected != expected_image_count:
        raise NativeEdgeSupportAuditIdentityError(f"result.image_count_expected disagrees with the registered {run_mode} image count")
    image_count_processed = _require_int(record["image_count_processed"], "result.image_count_processed", minimum=0)
    if image_count_processed != image_count_expected:
        raise NativeEdgeSupportAuditIdentityError("a complete result must have image_count_processed == image_count_expected")
    _require_sha256(record["image_order_digest"], "result.image_order_digest")

    windows_processed = _require_int(record["windows_processed_total"], "result.windows_processed_total", minimum=1)
    rows_processed = _require_int(record["rows_processed_total"], "result.rows_processed_total", minimum=1)
    edges_processed = _require_int(record["edges_processed_total"], "result.edges_processed_total", minimum=1)
    if rows_processed != windows_processed * 1024:
        raise NativeEdgeSupportAuditIdentityError("result.rows_processed_total must equal windows_processed_total * 1024")
    if edges_processed != rows_processed * 12:
        raise NativeEdgeSupportAuditIdentityError("result.edges_processed_total must equal rows_processed_total * 12")

    if _require_int(record["class_count"], "result.class_count", minimum=1) != identity["dataset"]["classes"]:
        raise NativeEdgeSupportAuditIdentityError("result.class_count disagrees with the registered class count")

    funnel = _require_counter_mapping(record["funnel"], FUNNEL_KEYS, "result.funnel")
    if funnel["images"] != image_count_processed:
        raise NativeEdgeSupportAuditIdentityError("result.funnel.images disagrees with result.image_count_processed")
    if funnel["windows"] != windows_processed:
        raise NativeEdgeSupportAuditIdentityError("result.funnel.windows disagrees with result.windows_processed_total")
    if funnel["graph_rows"] != rows_processed:
        raise NativeEdgeSupportAuditIdentityError("result.funnel.graph_rows disagrees with result.rows_processed_total")
    if funnel["directed_edges"] != edges_processed:
        raise NativeEdgeSupportAuditIdentityError("result.funnel.directed_edges disagrees with result.edges_processed_total")
    validate_funnel_invariants(funnel)

    undefined_reason_counts = _require_counter_mapping(record["undefined_reason_counts"], UNDEFINED_REASON_KEYS, "result.undefined_reason_counts")
    if sum(undefined_reason_counts.values()) != funnel["directed_edges"] - funnel["edges_with_observer"]:
        raise NativeEdgeSupportAuditIdentityError(
            "result.undefined_reason_counts must sum to funnel.directed_edges - funnel.edges_with_observer"
        )

    support_count_histogram = _require_counter_mapping(record["support_count_histogram"], SUPPORT_COUNT_HISTOGRAM_KEYS, "result.support_count_histogram")
    if sum(support_count_histogram.values()) != funnel["edges_with_observer"]:
        raise NativeEdgeSupportAuditIdentityError("result.support_count_histogram must sum to funnel.edges_with_observer")

    support_fraction_histogram = _require_counter_mapping(record["support_fraction_histogram"], SUPPORT_FRACTION_HISTOGRAM_KEYS, "result.support_fraction_histogram")
    if sum(support_fraction_histogram.values()) != funnel["edges_with_observer"]:
        raise NativeEdgeSupportAuditIdentityError("result.support_fraction_histogram must sum to funnel.edges_with_observer")

    crop_edge_band_histogram = _require_counter_mapping(record["crop_edge_band_histogram"], CROP_EDGE_BAND_KEYS, "result.crop_edge_band_histogram")
    if sum(crop_edge_band_histogram.values()) != funnel["graph_rows"]:
        raise NativeEdgeSupportAuditIdentityError("result.crop_edge_band_histogram must sum to funnel.graph_rows (every row is banded)")
    image_edge_band_histogram = _require_counter_mapping(record["image_edge_band_histogram"], IMAGE_EDGE_BAND_KEYS, "result.image_edge_band_histogram")
    if sum(image_edge_band_histogram.values()) != funnel["graph_rows"]:
        raise NativeEdgeSupportAuditIdentityError("result.image_edge_band_histogram must sum to funnel.graph_rows")
    displacement_band_histogram = _require_counter_mapping(record["displacement_band_histogram"], DISPLACEMENT_BAND_KEYS, "result.displacement_band_histogram")
    if sum(displacement_band_histogram.values()) != funnel["directed_edges"]:
        raise NativeEdgeSupportAuditIdentityError("result.displacement_band_histogram must sum to funnel.directed_edges")
    affinity_rank_histogram = _require_counter_mapping(record["affinity_rank_histogram"], AFFINITY_RANK_KEYS, "result.affinity_rank_histogram")
    if sum(affinity_rank_histogram.values()) != funnel["directed_edges"]:
        raise NativeEdgeSupportAuditIdentityError("result.affinity_rank_histogram must sum to funnel.directed_edges")

    cross_tabs = _require_closed_mapping(record["correctness_cross_tabs"], frozenset(CROSS_TAB_TABLE_NAMES), "result.correctness_cross_tabs")
    for table_name in CROSS_TAB_TABLE_NAMES:
        table = _require_counter_mapping(cross_tabs[table_name], CROSS_TAB_ROW_KEYS, f"result.correctness_cross_tabs.{table_name}")
        total = sum(table.values())
        expected_total = funnel["graph_rows"] - record["ignored_gt_count"]
        if total != expected_total:
            raise NativeEdgeSupportAuditIdentityError(
                f"result.correctness_cross_tabs.{table_name} rows must sum to funnel.graph_rows - ignored_gt_count"
            )

    _require_int(record["ignored_gt_count"], "result.ignored_gt_count", minimum=0)
    _require_exact_string(record["ranking_definition"], "result.ranking_definition")
    if record["ranking_definition"] != " -> ".join(identity["ranking"]["criteria_in_order"]):
        raise NativeEdgeSupportAuditIdentityError("result.ranking_definition disagrees with the identity's own ranking.criteria_in_order")

    decision_output = _require_exact_string(record["decision_output"], "result.decision_output")
    if decision_output not in DECISION_OUTCOMES:
        raise NativeEdgeSupportAuditIdentityError(f"result.decision_output must be one of {DECISION_OUTCOMES}")
    _require_exact_string(record["decision_rationale"], "result.decision_rationale")
    recomputed_decision, _ = classify_reachability(funnel, identity, ignored_gt_count=record["ignored_gt_count"])
    if decision_output != recomputed_decision:
        raise NativeEdgeSupportAuditIdentityError(
            f"result.decision_output {decision_output!r} disagrees with the recomputed classification {recomputed_decision!r}"
        )

    _require_exact_string(record["per_image_stats_manifest_path"], "result.per_image_stats_manifest_path")
    _require_sha256(record["per_image_stats_manifest_sha256"], "result.per_image_stats_manifest_sha256")

    telemetry = _require_closed_mapping(record["operation_telemetry"], OPERATION_TELEMETRY_KEYS, "result.operation_telemetry")
    _require_int(telemetry["sample_pulls"], "operation_telemetry.sample_pulls", minimum=1)
    for name in ("window_enumerations", "backbone_snapshot_calls", "graph_builds"):
        if _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=0) != windows_processed:
            raise NativeEdgeSupportAuditIdentityError(f"operation_telemetry.{name} must equal windows_processed_total ({windows_processed})")
    for name in ("propagation_calls", "probability_interpolation_calls"):
        value = _require_int(telemetry[name], f"operation_telemetry.{name}", minimum=0)
        if value > windows_processed:
            raise NativeEdgeSupportAuditIdentityError(f"operation_telemetry.{name} must not exceed windows_processed_total ({windows_processed})")
    _require_int(telemetry["cross_view_comparisons_total"], "operation_telemetry.cross_view_comparisons_total", minimum=0)

    phase_runtime = _require_closed_mapping(record["phase_runtime_seconds"], PHASE_RUNTIME_KEYS, "result.phase_runtime_seconds")
    _require_float(phase_runtime["total"], "phase_runtime_seconds.total", minimum=0.0)
    _require_float(phase_runtime["shared"], "phase_runtime_seconds.shared", minimum=0.0)

    _require_int(record["peak_gpu_memory_bytes"], "result.peak_gpu_memory_bytes", minimum=0)
    _require_bool(record["resumed_from_checkpoint"], "result.resumed_from_checkpoint")
    _require_exact_string(record["source_git_branch"], "result.source_git_branch")
    if record["failure_reason"] is not None:
        _require_exact_string(record["failure_reason"], "result.failure_reason")

    misclassified = funnel["misclassified_rows"]
    coverage = (
        funnel["misclassified_rows_any_defined"] / misclassified if misclassified > 0 else float("nan")
    )
    return (
        f"NATIVE EDGE SUPPORT AUDIT RESULT PASS run_mode={run_mode} images={image_count_processed} "
        f"misclassified_rows={misclassified} support_defined_coverage_among_misclassified={coverage:.6f}"
    )


def verify_result(path: Path, *, identity_path: Path | None = None, repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else repository_root()
    identity = load_identity(identity_path, repo_root=root)
    resolved_identity_path = Path(identity_path) if identity_path is not None else root / "evaluation_identities/e12_native_edge_support_audit.toml"
    identity_sha256 = _sha256_file(resolved_identity_path)
    record = parse_strict_json_document(Path(path), label="structured result")
    return verify_record(record, identity, identity_sha256=identity_sha256)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "OPERATION_TELEMETRY_KEYS",
    "TOP_RESULT_KEYS",
    "classify_reachability",
    "verify_record",
    "verify_result",
]
