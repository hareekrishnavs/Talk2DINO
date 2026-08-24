"""Checkpoint schema/invariant validation for the COCO-Object val2017
mask-materialization stage. Sole authority for the checkpoint's
schema/type/binding/completeness checks -- never reimplemented at any
call site."""

from __future__ import annotations

from typing import Any, Mapping

from src.coco_object_val_materialization import CLASS_COUNT
from src.coco_object_val_materialization_identity import CocoObjectValMaterializationIdentityError

TOP_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "identity", "identity_sha256", "git_commit", "split", "expected_image_count",
        "image_order_digest", "next_index", "completed_image_ids", "per_image_records",
        "aggregate_label_histogram", "total_pixels", "masks_with_foreground", "all_background_masks",
        "complete", "created_at_utc", "updated_at_utc",
    }
)
PER_IMAGE_RECORD_KEYS = frozenset(
    {"image_id", "width", "height", "raw_mask_sha256", "decoded_pixel_sha256", "encoded_png_sha256", "label_histogram"}
)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise CocoObjectValMaterializationIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise CocoObjectValMaterializationIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CocoObjectValMaterializationIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise CocoObjectValMaterializationIdentityError(f"{label} must be at most {maximum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise CocoObjectValMaterializationIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise CocoObjectValMaterializationIdentityError(f"{label} must be a full Git identity")
    return token


def _require_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise CocoObjectValMaterializationIdentityError(f"{label} must be a list of exact strings")
    return value


def _require_int_list(value: Any, label: str, *, length: int | None = None) -> list[int]:
    if type(value) is not list or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise CocoObjectValMaterializationIdentityError(f"{label} must be a list of exact integers")
    if length is not None and len(value) != length:
        raise CocoObjectValMaterializationIdentityError(f"{label} must have exactly {length} entries")
    return value


def validate_per_image_record(record: Any, label: str) -> Mapping[str, Any]:
    _require_closed_mapping(record, PER_IMAGE_RECORD_KEYS, label)
    _require_exact_string(record["image_id"], f"{label}.image_id")
    _require_int(record["width"], f"{label}.width", minimum=1)
    _require_int(record["height"], f"{label}.height", minimum=1)
    _require_sha256(record["raw_mask_sha256"], f"{label}.raw_mask_sha256")
    _require_sha256(record["decoded_pixel_sha256"], f"{label}.decoded_pixel_sha256")
    _require_sha256(record["encoded_png_sha256"], f"{label}.encoded_png_sha256")
    histogram = _require_int_list(record["label_histogram"], f"{label}.label_histogram", length=CLASS_COUNT)
    if any(count < 0 for count in histogram):
        raise CocoObjectValMaterializationIdentityError(f"{label}.label_histogram entries must be non-negative")
    if sum(histogram) != record["width"] * record["height"]:
        raise CocoObjectValMaterializationIdentityError(f"{label}.label_histogram must sum to width * height")
    return record


def validate_checkpoint_structure(
    checkpoint: Mapping[str, Any], *, identity: Mapping[str, Any], identity_sha256: str, checkpoint_schema_name: str,
    expected_image_count: int | None = None,
) -> None:
    _require_closed_mapping(checkpoint, TOP_CHECKPOINT_KEYS, "checkpoint")

    if _require_exact_string(checkpoint["schema"], "checkpoint.schema") != checkpoint_schema_name:
        raise CocoObjectValMaterializationIdentityError("checkpoint.schema mismatch")
    if _require_exact_string(checkpoint["identity"], "checkpoint.identity") != identity["identity"]["name"]:
        raise CocoObjectValMaterializationIdentityError("checkpoint.identity mismatch")
    if _require_sha256(checkpoint["identity_sha256"], "checkpoint.identity_sha256") != identity_sha256:
        raise CocoObjectValMaterializationIdentityError("checkpoint.identity_sha256 does not match the loaded identity file")
    _require_git_identity(checkpoint["git_commit"], "checkpoint.git_commit")

    if checkpoint["split"] != identity["protocol"]["split"]:
        raise CocoObjectValMaterializationIdentityError(
            f"checkpoint.split ({checkpoint['split']!r}) disagrees with the identity's protocol.split "
            f"({identity['protocol']['split']!r}) -- a COCO-Stuff or train-split checkpoint must be rejected"
        )
    recorded_image_count = _require_int(checkpoint["expected_image_count"], "checkpoint.expected_image_count", minimum=1)
    if expected_image_count is not None:
        if recorded_image_count != expected_image_count:
            raise CocoObjectValMaterializationIdentityError("checkpoint.expected_image_count disagrees with the effective --test-limit/canonical count for this run")
    elif recorded_image_count != identity["protocol"]["expected_image_count"]:
        raise CocoObjectValMaterializationIdentityError("checkpoint.expected_image_count disagrees with the identity")

    _require_sha256(checkpoint["image_order_digest"], "checkpoint.image_order_digest")

    next_index = _require_int(checkpoint["next_index"], "checkpoint.next_index", minimum=0, maximum=recorded_image_count)
    completed = _require_string_list(checkpoint["completed_image_ids"], "checkpoint.completed_image_ids")
    if len(set(completed)) != len(completed):
        raise CocoObjectValMaterializationIdentityError("checkpoint.completed_image_ids contains a duplicate image ID")
    if next_index != len(completed):
        raise CocoObjectValMaterializationIdentityError(
            f"checkpoint.next_index ({next_index}) must equal len(completed_image_ids) ({len(completed)}) -- "
            "refusing to resume a checkpoint that could skip, duplicate, or reorder an image"
        )

    per_image_records = checkpoint["per_image_records"]
    if type(per_image_records) is not list or len(per_image_records) != len(completed):
        raise CocoObjectValMaterializationIdentityError("checkpoint.per_image_records length disagrees with completed_image_ids")
    for index, (image_id, record) in enumerate(zip(completed, per_image_records)):
        validated = validate_per_image_record(record, f"checkpoint.per_image_records[{index}]")
        if validated["image_id"] != image_id:
            raise CocoObjectValMaterializationIdentityError(
                f"checkpoint.per_image_records[{index}].image_id ({validated['image_id']!r}) disagrees with "
                f"completed_image_ids[{index}] ({image_id!r})"
            )

    complete = _require_bool(checkpoint["complete"], "checkpoint.complete")
    if complete and next_index != recorded_image_count:
        raise CocoObjectValMaterializationIdentityError("checkpoint.complete is true but next_index does not equal expected_image_count")

    aggregate_histogram = _require_int_list(checkpoint["aggregate_label_histogram"], "checkpoint.aggregate_label_histogram", length=CLASS_COUNT)
    summed_histogram = [0] * CLASS_COUNT
    for record in per_image_records:
        for index in range(CLASS_COUNT):
            summed_histogram[index] += record["label_histogram"][index]
    if aggregate_histogram != summed_histogram:
        raise CocoObjectValMaterializationIdentityError("checkpoint.aggregate_label_histogram disagrees with the sum of per_image_records histograms")

    total_pixels = _require_int(checkpoint["total_pixels"], "checkpoint.total_pixels", minimum=0)
    if total_pixels != sum(aggregate_histogram):
        raise CocoObjectValMaterializationIdentityError("checkpoint.total_pixels disagrees with sum(aggregate_label_histogram)")

    masks_with_foreground = _require_int(checkpoint["masks_with_foreground"], "checkpoint.masks_with_foreground", minimum=0)
    all_background_masks = _require_int(checkpoint["all_background_masks"], "checkpoint.all_background_masks", minimum=0)
    if masks_with_foreground + all_background_masks != len(completed):
        raise CocoObjectValMaterializationIdentityError(
            "checkpoint.masks_with_foreground + checkpoint.all_background_masks must equal len(completed_image_ids)"
        )
    recomputed_with_fg = sum(1 for record in per_image_records if sum(record["label_histogram"][1:]) > 0)
    if recomputed_with_fg != masks_with_foreground:
        raise CocoObjectValMaterializationIdentityError("checkpoint.masks_with_foreground disagrees with a fresh recount of per_image_records")

    _require_exact_string(checkpoint["created_at_utc"], "checkpoint.created_at_utc")
    _require_exact_string(checkpoint["updated_at_utc"], "checkpoint.updated_at_utc")


def validate_checkpoint_against_canonical_order(
    checkpoint: Mapping[str, Any], expected_image_ids: list[str], *, image_order_digest: str,
) -> None:
    if checkpoint["image_order_digest"] != image_order_digest:
        raise CocoObjectValMaterializationIdentityError("checkpoint.image_order_digest disagrees with the freshly-resolved canonical image order")
    next_index = checkpoint["next_index"]
    expected_prefix = list(expected_image_ids[:next_index])
    if checkpoint["completed_image_ids"] != expected_prefix:
        raise CocoObjectValMaterializationIdentityError(
            "checkpoint.completed_image_ids does not match the canonical image-ID prefix for the first "
            f"{next_index} images; refusing to resume against a reordered/skipped image sequence"
        )


def resume_next_index(checkpoint: Mapping[str, Any]) -> int:
    if checkpoint["complete"] is True:
        raise CocoObjectValMaterializationIdentityError("checkpoint is already complete; refusing to resume it as though incomplete")
    return checkpoint["next_index"]


__all__ = [
    "PER_IMAGE_RECORD_KEYS",
    "TOP_CHECKPOINT_KEYS",
    "resume_next_index",
    "validate_checkpoint_against_canonical_order",
    "validate_checkpoint_structure",
    "validate_per_image_record",
]
