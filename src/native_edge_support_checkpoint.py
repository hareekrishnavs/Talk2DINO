"""Authoritative strict JSON loading and checkpoint/result-invariant
validation for the native cross-view directed-edge support audit.

Reuses :func:`src.k11_k12_power_evaluation_checkpoint.parse_strict_json_document`
directly (the exact same strict loader every E12 evaluator uses: rejects
duplicate keys, NaN/Infinity, malformed JSON, non-mapping roots) -- never a
second, independently reimplemented strict-JSON parser.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.k11_k12_power_evaluation_checkpoint import parse_strict_json_document
from src.native_edge_support_identity import (
    RUN_MODE_IMAGE_COUNT_KEYS,
    NativeEdgeSupportAuditIdentityError,
)

CHECKPOINT_SCHEMA_NAME = "talk2dino-native-edge-support-audit-checkpoint-v1"

FUNNEL_KEYS: tuple[str, ...] = (
    "images", "windows", "graph_rows", "directed_edges", "rows_with_other_crop",
    "rows_with_aligned_observer", "edges_with_observer", "rows_any_defined", "rows_all_defined",
    "rows_any_defined_zero_support", "misclassified_rows", "misclassified_rows_any_defined",
    "misclassified_rows_all_defined",
    # Window/pair-level and diagnostic-only additions (never used to
    # define, weight, or gate support itself -- descriptive counts only).
    "aligned_window_pairs", "unaligned_window_pairs", "clamped_windows", "non_clamped_windows",
    "unanimous_support_edges", "direction_reversal_only_cases",
    "rows_with_unique_least_support", "rows_tied_for_least_support",
    # non-ignored, non-misclassified ("correct") rows that are support-
    # defined -- paired with misclassified_rows_any_defined to compute the
    # descriptive risk ratio behind the decision-output classification.
    "correct_rows_any_defined",
)
SUPPORT_COUNT_HISTOGRAM_KEYS: tuple[str, ...] = tuple(str(n) for n in range(13))
SUPPORT_FRACTION_HISTOGRAM_KEYS: tuple[str, ...] = (
    "0.0", "(0.0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]",
    "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]", "(0.8,0.9]", "(0.9,1.0]",
)
CROP_EDGE_BAND_KEYS: tuple[str, ...] = ("0", "1", "2", "3-4", "5-7", ">=8")
IMAGE_EDGE_BAND_KEYS: tuple[str, ...] = CROP_EDGE_BAND_KEYS
DISPLACEMENT_BAND_KEYS: tuple[str, ...] = CROP_EDGE_BAND_KEYS
AFFINITY_RANK_KEYS: tuple[str, ...] = tuple(str(r) for r in range(12))
DECISION_OUTCOMES: tuple[str, ...] = ("REACHABLE", "STRUCTURALLY_UNREACHABLE", "ALIGNMENT_LIMITED", "INCONCLUSIVE")
UNDEFINED_REASON_KEYS: tuple[str, ...] = (
    "single_window_image", "no_exactly_aligned_observer_covering_both_endpoints",
)
CROSS_TAB_ROW_KEYS: tuple[str, ...] = (
    "defined_and_misclassified", "defined_and_correct", "undefined_and_misclassified", "undefined_and_correct",
)
CROSS_TAB_TABLE_NAMES: tuple[str, ...] = (
    "support_defined_vs_source_correct", "all_defined_vs_source_correct",
    "support_defined_vs_stitched_correct", "all_defined_vs_stitched_correct",
)

TOP_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "stitching_control_identity_sha256",
        "power_evaluation_identity_sha256", "matched_identity_sha256", "e3_identity_sha256", "git_commit",
        "class_count", "image_count_expected", "image_order_digest", "next_dataset_index", "completed_image_ids",
        "images_completed_count", "windows_processed_total", "rows_processed_total", "edges_processed_total",
        "funnel", "undefined_reason_counts", "support_count_histogram", "support_fraction_histogram",
        "crop_edge_band_histogram", "image_edge_band_histogram", "displacement_band_histogram",
        "affinity_rank_histogram", "correctness_cross_tabs", "ignored_gt_count", "complete",
        "created_at_utc", "updated_at_utc",
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


def _require_int(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be at least {minimum}, observed {value}")
    if maximum is not None and value > maximum:
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be at most {maximum}, observed {value}")
    return value


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


def _require_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise NativeEdgeSupportAuditIdentityError(f"{label} must be a list of exact strings")
    return value


def _require_counter_mapping(value: Any, keys: tuple[str, ...], label: str) -> dict[str, int]:
    mapping = _require_closed_mapping(value, frozenset(keys), label)
    return {key: _require_int(mapping[key], f"{label}.{key}", minimum=0) for key in keys}


def validate_funnel_invariants(funnel: Mapping[str, int]) -> None:
    """Exact subset/count invariants for the 13-step aggregate funnel
    (Section 10) -- never approximate, never skipped."""
    if funnel["graph_rows"] != funnel["windows"] * 1024:
        raise NativeEdgeSupportAuditIdentityError("funnel.graph_rows must equal funnel.windows * 1024")
    if funnel["directed_edges"] != funnel["graph_rows"] * 12:
        raise NativeEdgeSupportAuditIdentityError("funnel.directed_edges must equal funnel.graph_rows * 12")
    if funnel["rows_with_other_crop"] > funnel["graph_rows"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.rows_with_other_crop must not exceed funnel.graph_rows")
    if funnel["rows_with_aligned_observer"] > funnel["rows_with_other_crop"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.rows_with_aligned_observer must not exceed funnel.rows_with_other_crop")
    if funnel["edges_with_observer"] > funnel["directed_edges"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.edges_with_observer must not exceed funnel.directed_edges")
    if funnel["rows_any_defined"] > funnel["rows_with_aligned_observer"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.rows_any_defined must not exceed funnel.rows_with_aligned_observer")
    if funnel["rows_all_defined"] > funnel["rows_any_defined"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.rows_all_defined must not exceed funnel.rows_any_defined")
    if funnel["rows_any_defined_zero_support"] > funnel["rows_any_defined"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.rows_any_defined_zero_support must not exceed funnel.rows_any_defined")
    if funnel["misclassified_rows"] > funnel["graph_rows"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.misclassified_rows must not exceed funnel.graph_rows")
    if funnel["misclassified_rows_any_defined"] > funnel["misclassified_rows"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.misclassified_rows_any_defined must not exceed funnel.misclassified_rows")
    if funnel["misclassified_rows_any_defined"] > funnel["rows_any_defined"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.misclassified_rows_any_defined must not exceed funnel.rows_any_defined")
    if funnel["misclassified_rows_all_defined"] > funnel["misclassified_rows_any_defined"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.misclassified_rows_all_defined must not exceed funnel.misclassified_rows_any_defined")
    if funnel["clamped_windows"] + funnel["non_clamped_windows"] != funnel["windows"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.clamped_windows + funnel.non_clamped_windows must equal funnel.windows")
    if funnel["unanimous_support_edges"] > funnel["edges_with_observer"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.unanimous_support_edges must not exceed funnel.edges_with_observer")
    if funnel["direction_reversal_only_cases"] > funnel["edges_with_observer"]:
        raise NativeEdgeSupportAuditIdentityError("funnel.direction_reversal_only_cases must not exceed funnel.edges_with_observer")
    if funnel["rows_with_unique_least_support"] + funnel["rows_tied_for_least_support"] != funnel["rows_any_defined"]:
        raise NativeEdgeSupportAuditIdentityError(
            "funnel.rows_with_unique_least_support + funnel.rows_tied_for_least_support must equal funnel.rows_any_defined"
        )
    if funnel["correct_rows_any_defined"] > funnel["rows_any_defined"] - funnel["misclassified_rows_any_defined"]:
        raise NativeEdgeSupportAuditIdentityError(
            "funnel.correct_rows_any_defined must not exceed funnel.rows_any_defined - funnel.misclassified_rows_any_defined"
        )


def validate_checkpoint_structure(
    checkpoint: Mapping[str, Any], *, identity: Mapping[str, Any], identity_sha256: str,
    run_mode: str | None = None, class_count: int | None = None,
) -> None:
    """Validate everything checkable from the checkpoint document alone.
    The sole authority for this checkpoint's schema/type/binding/
    completeness/funnel-invariant checks -- never reimplemented at any
    call site."""
    _require_closed_mapping(checkpoint, TOP_CHECKPOINT_KEYS, "checkpoint")

    if _require_exact_string(checkpoint["schema"], "checkpoint.schema") != CHECKPOINT_SCHEMA_NAME:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.schema mismatch")
    if _require_exact_string(checkpoint["identity"], "checkpoint.identity") != identity["identity"]["name"]:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.identity mismatch")
    if _require_sha256(checkpoint["identity_sha256"], "checkpoint.identity_sha256") != identity_sha256:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.identity_sha256 does not match the loaded identity file")
    for prefix in ("stitching_control", "power_evaluation", "matched", "e3"):
        if checkpoint[f"{prefix}_identity_sha256"] != identity["parent_identity"][f"{prefix}_identity_sha256"]:
            raise NativeEdgeSupportAuditIdentityError(f"checkpoint.{prefix}_identity_sha256 disagrees with the identity's parent")

    _require_git_identity(checkpoint["git_commit"], "checkpoint.git_commit")

    recorded_run_mode = _require_exact_string(checkpoint["run_mode"], "checkpoint.run_mode")
    if recorded_run_mode not in RUN_MODE_IMAGE_COUNT_KEYS:
        raise NativeEdgeSupportAuditIdentityError(f"checkpoint.run_mode must be one of {sorted(RUN_MODE_IMAGE_COUNT_KEYS)}")
    if run_mode is not None and recorded_run_mode != run_mode:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.run_mode disagrees with --run-mode")
    registered_image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[recorded_run_mode]]
    expected_image_count = _require_int(checkpoint["image_count_expected"], "checkpoint.image_count_expected", minimum=1)
    if expected_image_count != registered_image_count:
        raise NativeEdgeSupportAuditIdentityError(
            f"checkpoint.image_count_expected ({expected_image_count}) disagrees with the registered "
            f"{recorded_run_mode} image count ({registered_image_count})"
        )
    recorded_class_count = _require_int(checkpoint["class_count"], "checkpoint.class_count", minimum=1)
    if class_count is not None and recorded_class_count != class_count:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.class_count disagrees with the live inference class count")

    _require_sha256(checkpoint["image_order_digest"], "checkpoint.image_order_digest")

    next_index = _require_int(
        checkpoint["next_dataset_index"], "checkpoint.next_dataset_index", minimum=0, maximum=expected_image_count,
    )
    completed = _require_string_list(checkpoint["completed_image_ids"], "checkpoint.completed_image_ids")
    if len(set(completed)) != len(completed):
        raise NativeEdgeSupportAuditIdentityError("checkpoint.completed_image_ids contains a duplicate image ID")
    images_completed_count = _require_int(checkpoint["images_completed_count"], "checkpoint.images_completed_count", minimum=0)
    if images_completed_count != len(completed):
        raise NativeEdgeSupportAuditIdentityError("checkpoint.images_completed_count disagrees with len(completed_image_ids)")
    if next_index != len(completed):
        raise NativeEdgeSupportAuditIdentityError(
            f"checkpoint.next_dataset_index ({next_index}) must equal the number of completed images "
            f"({len(completed)}) -- refusing to resume a checkpoint that could skip or duplicate an image"
        )

    complete = _require_bool(checkpoint["complete"], "checkpoint.complete")
    if complete and next_index != expected_image_count:
        raise NativeEdgeSupportAuditIdentityError(
            "checkpoint.complete is true but next_dataset_index does not equal the expected image count"
        )

    windows_processed_total = _require_int(checkpoint["windows_processed_total"], "checkpoint.windows_processed_total", minimum=0)
    rows_processed_total = _require_int(checkpoint["rows_processed_total"], "checkpoint.rows_processed_total", minimum=0)
    edges_processed_total = _require_int(checkpoint["edges_processed_total"], "checkpoint.edges_processed_total", minimum=0)
    if rows_processed_total != windows_processed_total * 1024:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.rows_processed_total must equal windows_processed_total * 1024")
    if edges_processed_total != rows_processed_total * 12:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.edges_processed_total must equal rows_processed_total * 12")

    funnel = _require_counter_mapping(checkpoint["funnel"], FUNNEL_KEYS, "checkpoint.funnel")
    if funnel["images"] != images_completed_count:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.funnel.images disagrees with images_completed_count")
    if funnel["windows"] != windows_processed_total:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.funnel.windows disagrees with windows_processed_total")
    validate_funnel_invariants(funnel)

    undefined_reason_counts = _require_counter_mapping(checkpoint["undefined_reason_counts"], UNDEFINED_REASON_KEYS, "checkpoint.undefined_reason_counts")
    if sum(undefined_reason_counts.values()) != funnel["directed_edges"] - funnel["edges_with_observer"]:
        raise NativeEdgeSupportAuditIdentityError(
            "checkpoint.undefined_reason_counts must sum to funnel.directed_edges - funnel.edges_with_observer"
        )

    support_count_histogram = _require_counter_mapping(checkpoint["support_count_histogram"], SUPPORT_COUNT_HISTOGRAM_KEYS, "checkpoint.support_count_histogram")
    if sum(support_count_histogram.values()) != funnel["edges_with_observer"]:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.support_count_histogram must sum to funnel.edges_with_observer (defined edges only)")

    support_fraction_histogram = _require_counter_mapping(checkpoint["support_fraction_histogram"], SUPPORT_FRACTION_HISTOGRAM_KEYS, "checkpoint.support_fraction_histogram")
    if sum(support_fraction_histogram.values()) != funnel["edges_with_observer"]:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.support_fraction_histogram must sum to funnel.edges_with_observer (defined edges only)")

    crop_edge_band_histogram = _require_counter_mapping(checkpoint["crop_edge_band_histogram"], CROP_EDGE_BAND_KEYS, "checkpoint.crop_edge_band_histogram")
    if sum(crop_edge_band_histogram.values()) != funnel["graph_rows"]:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.crop_edge_band_histogram must sum to funnel.graph_rows")
    image_edge_band_histogram = _require_counter_mapping(checkpoint["image_edge_band_histogram"], IMAGE_EDGE_BAND_KEYS, "checkpoint.image_edge_band_histogram")
    if sum(image_edge_band_histogram.values()) != funnel["graph_rows"]:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.image_edge_band_histogram must sum to funnel.graph_rows")
    displacement_band_histogram = _require_counter_mapping(checkpoint["displacement_band_histogram"], DISPLACEMENT_BAND_KEYS, "checkpoint.displacement_band_histogram")
    if sum(displacement_band_histogram.values()) != funnel["directed_edges"]:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.displacement_band_histogram must sum to funnel.directed_edges")
    affinity_rank_histogram = _require_counter_mapping(checkpoint["affinity_rank_histogram"], AFFINITY_RANK_KEYS, "checkpoint.affinity_rank_histogram")
    if sum(affinity_rank_histogram.values()) != funnel["directed_edges"]:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.affinity_rank_histogram must sum to funnel.directed_edges")

    cross_tabs = _require_closed_mapping(checkpoint["correctness_cross_tabs"], frozenset(CROSS_TAB_TABLE_NAMES), "checkpoint.correctness_cross_tabs")
    for table_name in CROSS_TAB_TABLE_NAMES:
        _require_counter_mapping(cross_tabs[table_name], CROSS_TAB_ROW_KEYS, f"checkpoint.correctness_cross_tabs.{table_name}")

    _require_int(checkpoint["ignored_gt_count"], "checkpoint.ignored_gt_count", minimum=0)

    _require_exact_string(checkpoint["created_at_utc"], "checkpoint.created_at_utc")
    _require_exact_string(checkpoint["updated_at_utc"], "checkpoint.updated_at_utc")


def validate_checkpoint_against_artifact(
    checkpoint: Mapping[str, Any], *, stats_manifest: Mapping[str, Any],
) -> None:
    """Cross-check the checkpoint's own claims against the per-image
    audit-stats artifact already on disk: summing the per-image funnel
    rows must reproduce the checkpoint's own running totals exactly.
    Requires :func:`validate_checkpoint_structure` to have already
    passed."""
    next_index = checkpoint["next_dataset_index"]
    completed = checkpoint["completed_image_ids"]

    manifest_class_count = _require_int(stats_manifest["class_count"], "per-image-stats manifest.class_count", minimum=1)
    if manifest_class_count != checkpoint["class_count"]:
        raise NativeEdgeSupportAuditIdentityError("per-image-stats manifest.class_count disagrees with checkpoint.class_count")

    recorded_ids = _require_string_list(stats_manifest["image_ids"], "per-image-stats manifest.image_ids")
    if recorded_ids != completed:
        raise NativeEdgeSupportAuditIdentityError("per-image-stats artifact image order disagrees with the checkpoint; refusing to resume")
    recorded_indices = stats_manifest["dataset_indices"]
    if list(recorded_indices) != list(range(next_index)):
        raise NativeEdgeSupportAuditIdentityError(
            f"per-image-stats artifact dataset indices are not the exact canonical prefix [0, ..., {next_index - 1}]"
        )

    per_image_rows = stats_manifest["per_image_funnel"]
    if len(per_image_rows) != next_index:
        raise NativeEdgeSupportAuditIdentityError("per-image-stats artifact per_image_funnel row count disagrees with next_dataset_index")

    summed = {key: 0 for key in FUNNEL_KEYS}
    for row in per_image_rows:
        row_funnel = _require_counter_mapping(row, FUNNEL_KEYS, "per-image-stats manifest.per_image_funnel row")
        for key in FUNNEL_KEYS:
            summed[key] += row_funnel[key]
    if summed != checkpoint["funnel"]:
        raise NativeEdgeSupportAuditIdentityError(
            "summed per-image funnel rows disagree with checkpoint.funnel; refusing to resume against an inconsistent artifact"
        )


def validate_checkpoint_against_canonical_order(
    checkpoint: Mapping[str, Any], expected_image_ids: list[str], *, image_order_digest: str,
) -> None:
    if checkpoint["image_order_digest"] != image_order_digest:
        raise NativeEdgeSupportAuditIdentityError("checkpoint.image_order_digest disagrees with the freshly-resolved dataset order")
    next_index = checkpoint["next_dataset_index"]
    expected_prefix = list(expected_image_ids[:next_index])
    if checkpoint["completed_image_ids"] != expected_prefix:
        raise NativeEdgeSupportAuditIdentityError(
            "checkpoint.completed_image_ids does not match the canonical dataset image-ID prefix "
            f"for the first {next_index} images; refusing to resume against a mismatched image order"
        )


def resume_dataset_index(checkpoint: Mapping[str, Any]) -> int:
    if checkpoint["complete"] is True:
        raise NativeEdgeSupportAuditIdentityError("checkpoint is already complete; refusing to resume it as though incomplete")
    return checkpoint["next_dataset_index"]


__all__ = [
    "CHECKPOINT_SCHEMA_NAME",
    "CROP_EDGE_BAND_KEYS",
    "CROSS_TAB_ROW_KEYS",
    "CROSS_TAB_TABLE_NAMES",
    "FUNNEL_KEYS",
    "SUPPORT_COUNT_HISTOGRAM_KEYS",
    "SUPPORT_FRACTION_HISTOGRAM_KEYS",
    "TOP_CHECKPOINT_KEYS",
    "UNDEFINED_REASON_KEYS",
    "parse_strict_json_document",
    "resume_dataset_index",
    "validate_checkpoint_against_artifact",
    "validate_checkpoint_against_canonical_order",
    "validate_checkpoint_structure",
    "validate_funnel_invariants",
]
