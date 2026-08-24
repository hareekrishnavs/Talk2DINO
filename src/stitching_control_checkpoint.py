"""Authoritative strict JSON loading and checkpoint/result-invariant
validation for the reusable stitching control suite.

Reuses :func:`src.k11_k12_power_evaluation_checkpoint.parse_strict_json_document`
directly (the exact same strict loader the matched k11/k12 evaluator uses:
rejects duplicate keys, NaN/Infinity, malformed JSON, non-mapping roots) --
never a second, independently reimplemented strict-JSON parser. Everything
in this module beyond that reused loader is specific to the four-variant
checkpoint/result/per-image-stats schema this suite's evaluator produces.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.k11_k12_power_evaluation_checkpoint import parse_strict_json_document
from src.stitching_control_identity import (
    CANONICAL_VARIANT_NAMES,
    RUN_MODE_IMAGE_COUNT_KEYS,
    StitchingControlIdentityError,
)

CHECKPOINT_SCHEMA_NAME = "talk2dino-stitching-control-checkpoint-v1"

TOP_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "power_evaluation_identity_sha256",
        "matched_identity_sha256", "variant_names", "git_commit", "class_count",
        "image_count_expected", "image_order_digest", "next_dataset_index", "completed_image_ids",
        "images_completed_count", "windows_processed_total", "complete", "created_at_utc", "updated_at_utc",
    }
)

_PER_VARIANT_ARRAY_SUFFIXES = ("intersect", "union", "pred")
_PER_IMAGE_ARRAY_KEYS = ("label",) + tuple(
    f"{suffix}_{variant}" for variant in CANONICAL_VARIANT_NAMES for suffix in _PER_VARIANT_ARRAY_SUFFIXES
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


def _require_int(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StitchingControlIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise StitchingControlIdentityError(f"{label} must be at least {minimum}, observed {value}")
    if maximum is not None and value > maximum:
        raise StitchingControlIdentityError(f"{label} must be at most {maximum}, observed {value}")
    return value


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


def _require_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise StitchingControlIdentityError(f"{label} must be a list of exact strings")
    return value


def validate_checkpoint_structure(
    checkpoint: Mapping[str, Any], *, identity: Mapping[str, Any], identity_sha256: str,
    run_mode: str | None = None, class_count: int | None = None,
) -> None:
    """Validate everything checkable from the checkpoint document alone.
    The sole authority for this checkpoint's schema/type/binding/
    completeness invariants -- never reimplemented at any call site."""
    _require_closed_mapping(checkpoint, TOP_CHECKPOINT_KEYS, "checkpoint")

    if _require_exact_string(checkpoint["schema"], "checkpoint.schema") != CHECKPOINT_SCHEMA_NAME:
        raise StitchingControlIdentityError("checkpoint.schema mismatch")
    if _require_exact_string(checkpoint["identity"], "checkpoint.identity") != identity["identity"]["name"]:
        raise StitchingControlIdentityError("checkpoint.identity mismatch")
    if _require_sha256(checkpoint["identity_sha256"], "checkpoint.identity_sha256") != identity_sha256:
        raise StitchingControlIdentityError("checkpoint.identity_sha256 does not match the loaded identity file")
    if checkpoint["power_evaluation_identity_sha256"] != identity["parent_identity"]["power_evaluation_identity_sha256"]:
        raise StitchingControlIdentityError("checkpoint.power_evaluation_identity_sha256 disagrees with the identity's parent")
    if checkpoint["matched_identity_sha256"] != identity["parent_identity"]["matched_identity_sha256"]:
        raise StitchingControlIdentityError("checkpoint.matched_identity_sha256 disagrees with the identity's parent")

    variant_names = _require_string_list(checkpoint["variant_names"], "checkpoint.variant_names")
    if tuple(variant_names) != CANONICAL_VARIANT_NAMES:
        raise StitchingControlIdentityError(
            f"checkpoint.variant_names must be exactly {list(CANONICAL_VARIANT_NAMES)} in that order; "
            "a checkpoint from a different variant set must never be resumed"
        )

    _require_git_identity(checkpoint["git_commit"], "checkpoint.git_commit")

    recorded_run_mode = _require_exact_string(checkpoint["run_mode"], "checkpoint.run_mode")
    if recorded_run_mode not in RUN_MODE_IMAGE_COUNT_KEYS:
        raise StitchingControlIdentityError(f"checkpoint.run_mode must be one of {sorted(RUN_MODE_IMAGE_COUNT_KEYS)}")
    if run_mode is not None and recorded_run_mode != run_mode:
        raise StitchingControlIdentityError("checkpoint.run_mode disagrees with --run-mode")
    registered_image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[recorded_run_mode]]
    expected_image_count = _require_int(checkpoint["image_count_expected"], "checkpoint.image_count_expected", minimum=1)
    if expected_image_count != registered_image_count:
        raise StitchingControlIdentityError(
            f"checkpoint.image_count_expected ({expected_image_count}) disagrees with the registered "
            f"{recorded_run_mode} image count ({registered_image_count})"
        )
    recorded_class_count = _require_int(checkpoint["class_count"], "checkpoint.class_count", minimum=1)
    if class_count is not None and recorded_class_count != class_count:
        raise StitchingControlIdentityError("checkpoint.class_count disagrees with the live inference class count")

    _require_sha256(checkpoint["image_order_digest"], "checkpoint.image_order_digest")

    next_index = _require_int(
        checkpoint["next_dataset_index"], "checkpoint.next_dataset_index", minimum=0, maximum=expected_image_count,
    )
    completed = _require_string_list(checkpoint["completed_image_ids"], "checkpoint.completed_image_ids")
    if len(set(completed)) != len(completed):
        raise StitchingControlIdentityError("checkpoint.completed_image_ids contains a duplicate image ID")
    images_completed_count = _require_int(checkpoint["images_completed_count"], "checkpoint.images_completed_count", minimum=0)
    if images_completed_count != len(completed):
        raise StitchingControlIdentityError("checkpoint.images_completed_count disagrees with len(completed_image_ids)")
    if next_index != len(completed):
        raise StitchingControlIdentityError(
            f"checkpoint.next_dataset_index ({next_index}) must equal the number of completed images "
            f"({len(completed)}) -- refusing to resume a checkpoint that could skip or duplicate an image"
        )

    complete = _require_bool(checkpoint["complete"], "checkpoint.complete")
    if complete and next_index != expected_image_count:
        raise StitchingControlIdentityError(
            "checkpoint.complete is true but next_dataset_index does not equal the expected image count"
        )

    _require_int(checkpoint["windows_processed_total"], "checkpoint.windows_processed_total", minimum=0)
    _require_exact_string(checkpoint["created_at_utc"], "checkpoint.created_at_utc")
    _require_exact_string(checkpoint["updated_at_utc"], "checkpoint.updated_at_utc")


def validate_checkpoint_against_artifact(
    checkpoint: Mapping[str, Any], *, stats_manifest: Mapping[str, Any], stats_arrays: Mapping[str, Any],
) -> None:
    """Cross-check the checkpoint's own claims against the per-image
    sufficient-statistics artifact (manifest + NPZ arrays) already on
    disk. Requires :func:`validate_checkpoint_structure` to have already
    passed."""
    next_index = checkpoint["next_dataset_index"]
    completed = checkpoint["completed_image_ids"]
    class_count = checkpoint["class_count"]

    manifest_class_count = _require_int(stats_manifest["class_count"], "per-image-stats manifest.class_count", minimum=1)
    if manifest_class_count != class_count:
        raise StitchingControlIdentityError("per-image-stats manifest.class_count disagrees with checkpoint.class_count")

    manifest_variants = _require_string_list(stats_manifest["variant_names"], "per-image-stats manifest.variant_names")
    if tuple(manifest_variants) != CANONICAL_VARIANT_NAMES:
        raise StitchingControlIdentityError("per-image-stats manifest.variant_names disagrees with the canonical variant set")

    recorded_ids = _require_string_list(stats_manifest["image_ids"], "per-image-stats manifest.image_ids")
    if recorded_ids != completed:
        raise StitchingControlIdentityError("per-image-stats artifact image order disagrees with the checkpoint; refusing to resume")
    recorded_indices = stats_manifest["dataset_indices"]
    if list(recorded_indices) != list(range(next_index)):
        raise StitchingControlIdentityError(
            "per-image-stats artifact dataset indices are not the exact canonical prefix "
            f"[0, ..., {next_index - 1}]"
        )

    import numpy as np

    array_row_counts: dict[str, int] = {}
    for key in _PER_IMAGE_ARRAY_KEYS:
        if key not in stats_arrays:
            raise StitchingControlIdentityError(f"per-image-stats artifact is missing array {key!r}")
        array = stats_arrays[key]
        if array.ndim != 2:
            raise StitchingControlIdentityError(f"per-image-stats artifact array {key!r} must be 2-dimensional")
        array_row_counts[key] = int(array.shape[0])
        if array.shape[0] != next_index:
            raise StitchingControlIdentityError(f"per-image-stats artifact array {key!r} has {array.shape[0]} rows, expected {next_index}")
        if array.shape[1] != class_count:
            raise StitchingControlIdentityError(f"per-image-stats artifact array {key!r} has {array.shape[1]} columns, expected class_count={class_count}")
        if array.size and (not np.isfinite(array).all() or (array < 0).any()):
            raise StitchingControlIdentityError(f"per-image-stats artifact array {key!r} contains a negative or non-finite value")

    if len(set(array_row_counts.values())) != 1:
        raise StitchingControlIdentityError(f"per-image-stats artifact arrays have mismatched row counts: {array_row_counts}")

    label = stats_arrays["label"]
    for variant in CANONICAL_VARIANT_NAMES:
        intersect = stats_arrays[f"intersect_{variant}"]
        union = stats_arrays[f"union_{variant}"]
        pred = stats_arrays[f"pred_{variant}"]
        if (intersect > union).any():
            raise StitchingControlIdentityError(f"per-image-stats artifact {variant}: intersect exceeds union")
        if (intersect > pred).any():
            raise StitchingControlIdentityError(f"per-image-stats artifact {variant}: intersect exceeds predicted-pixel count")
        if (intersect > label).any():
            raise StitchingControlIdentityError(f"per-image-stats artifact {variant}: intersect exceeds ground-truth pixel count")


def validate_checkpoint_against_canonical_order(
    checkpoint: Mapping[str, Any], expected_image_ids: list[str], *, image_order_digest: str,
) -> None:
    if checkpoint["image_order_digest"] != image_order_digest:
        raise StitchingControlIdentityError("checkpoint.image_order_digest disagrees with the freshly-resolved dataset order")
    next_index = checkpoint["next_dataset_index"]
    expected_prefix = list(expected_image_ids[:next_index])
    if checkpoint["completed_image_ids"] != expected_prefix:
        raise StitchingControlIdentityError(
            "checkpoint.completed_image_ids does not match the canonical dataset image-ID prefix "
            f"for the first {next_index} images; refusing to resume against a mismatched image order"
        )


def resume_dataset_index(checkpoint: Mapping[str, Any]) -> int:
    if checkpoint["complete"] is True:
        raise StitchingControlIdentityError("checkpoint is already complete; refusing to resume it as though incomplete")
    return checkpoint["next_dataset_index"]


__all__ = [
    "CHECKPOINT_SCHEMA_NAME",
    "TOP_CHECKPOINT_KEYS",
    "parse_strict_json_document",
    "resume_dataset_index",
    "validate_checkpoint_against_artifact",
    "validate_checkpoint_against_canonical_order",
    "validate_checkpoint_structure",
]
