"""Checkpoint schema/invariant validation for the COCO-Object protocol
confirmation evaluator (E3 vs k11 vs k12). Sole authority for the
checkpoint's schema/type/binding/completeness checks -- never
reimplemented at any call site. A COCO-Stuff checkpoint (wrong dataset
binding) must be rejected."""

from __future__ import annotations

from typing import Any, Mapping

from src.coco_object_protocol_confirmation_identity import CocoObjectProtocolConfirmationIdentityError

TOP_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "matched_identity_sha256",
        "materialization_identity_sha256", "materialization_manifest_sha256", "git_commit",
        "class_count", "live_class_names_digest", "image_count_expected", "image_order_digest", "next_dataset_index",
        "completed_image_ids", "images_completed_count", "windows_processed_total",
        "complete", "created_at_utc", "updated_at_utc",
    }
)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be at most {maximum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be a full Git identity")
    return token


def _require_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise CocoObjectProtocolConfirmationIdentityError(f"{label} must be a list of exact strings")
    return value


def validate_checkpoint_structure(
    checkpoint: Mapping[str, Any], *, identity: Mapping[str, Any], identity_sha256: str,
    run_mode: str | None = None, class_count: int | None = None,
) -> None:
    """The sole authority for this checkpoint's schema/type/binding/
    completeness checks. Rejects a checkpoint bound to a different
    identity, protocol, run mode, or class count -- including, by
    construction, any COCO-Stuff checkpoint (which would carry a
    completely different identity/identity_sha256/class_count)."""
    _require_closed_mapping(checkpoint, TOP_CHECKPOINT_KEYS, "checkpoint")

    if _require_exact_string(checkpoint["schema"], "checkpoint.schema") != identity["checkpoint"]["schema_name"]:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.schema mismatch")
    if _require_exact_string(checkpoint["identity"], "checkpoint.identity") != identity["identity"]["name"]:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.identity mismatch")
    if _require_sha256(checkpoint["identity_sha256"], "checkpoint.identity_sha256") != identity_sha256:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.identity_sha256 does not match the loaded identity file")
    if checkpoint["matched_identity_sha256"] != identity["parent_identities"]["matched_identity_sha256"]:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.matched_identity_sha256 disagrees with the identity's parent")
    if checkpoint["materialization_identity_sha256"] != identity["parent_identities"]["materialization_identity_sha256"]:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.materialization_identity_sha256 disagrees with the identity's parent")
    _require_sha256(checkpoint["materialization_manifest_sha256"], "checkpoint.materialization_manifest_sha256")
    _require_git_identity(checkpoint["git_commit"], "checkpoint.git_commit")

    recorded_run_mode = _require_exact_string(checkpoint["run_mode"], "checkpoint.run_mode")
    if recorded_run_mode not in ("pilot20", "pilot100", "full"):
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.run_mode must be one of pilot20, pilot100, full")
    if run_mode is not None and recorded_run_mode != run_mode:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.run_mode disagrees with --run-mode")
    registered_image_count = identity["run_modes"][f"{recorded_run_mode}_images"]
    expected_image_count = _require_int(checkpoint["image_count_expected"], "checkpoint.image_count_expected", minimum=1)
    if expected_image_count != registered_image_count:
        raise CocoObjectProtocolConfirmationIdentityError(
            f"checkpoint.image_count_expected ({expected_image_count}) disagrees with the registered "
            f"{recorded_run_mode} image count ({registered_image_count})"
        )
    recorded_class_count = _require_int(checkpoint["class_count"], "checkpoint.class_count", minimum=1)
    if recorded_class_count != identity["dataset"]["class_count"]:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.class_count disagrees with identity.dataset.class_count")
    if class_count is not None and recorded_class_count != class_count:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.class_count disagrees with the live inference class count")
    if _require_sha256(checkpoint["live_class_names_digest"], "checkpoint.live_class_names_digest") != identity["dataset"]["class_names_digest"]:
        raise CocoObjectProtocolConfirmationIdentityError(
            "checkpoint.live_class_names_digest disagrees with identity.dataset.class_names_digest"
        )

    _require_sha256(checkpoint["image_order_digest"], "checkpoint.image_order_digest")

    next_index = _require_int(checkpoint["next_dataset_index"], "checkpoint.next_dataset_index", minimum=0, maximum=expected_image_count)
    completed = _require_string_list(checkpoint["completed_image_ids"], "checkpoint.completed_image_ids")
    if len(set(completed)) != len(completed):
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.completed_image_ids contains a duplicate image ID")
    images_completed_count = _require_int(checkpoint["images_completed_count"], "checkpoint.images_completed_count", minimum=0)
    if images_completed_count != len(completed):
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.images_completed_count disagrees with len(completed_image_ids)")
    if next_index != len(completed):
        raise CocoObjectProtocolConfirmationIdentityError(
            f"checkpoint.next_dataset_index ({next_index}) must equal the number of completed images "
            f"({len(completed)}) -- refusing to resume a checkpoint that could skip or duplicate an image"
        )

    complete = _require_bool(checkpoint["complete"], "checkpoint.complete")
    if complete and next_index != expected_image_count:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.complete is true but next_dataset_index does not equal the expected image count")

    windows_processed_total = _require_int(checkpoint["windows_processed_total"], "checkpoint.windows_processed_total", minimum=0)
    if images_completed_count > 0 and windows_processed_total < images_completed_count:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.windows_processed_total must be at least images_completed_count")

    _require_exact_string(checkpoint["created_at_utc"], "checkpoint.created_at_utc")
    _require_exact_string(checkpoint["updated_at_utc"], "checkpoint.updated_at_utc")


def validate_checkpoint_against_canonical_order(
    checkpoint: Mapping[str, Any], expected_image_ids: list[str], *, image_order_digest: str,
) -> None:
    if checkpoint["image_order_digest"] != image_order_digest:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint.image_order_digest disagrees with the freshly-resolved dataset order")
    next_index = checkpoint["next_dataset_index"]
    expected_prefix = list(expected_image_ids[:next_index])
    if checkpoint["completed_image_ids"] != expected_prefix:
        raise CocoObjectProtocolConfirmationIdentityError(
            "checkpoint.completed_image_ids does not match the canonical dataset image-ID prefix "
            f"for the first {next_index} images; refusing to resume against a mismatched image order"
        )


def resume_dataset_index(checkpoint: Mapping[str, Any]) -> int:
    if checkpoint["complete"] is True:
        raise CocoObjectProtocolConfirmationIdentityError("checkpoint is already complete; refusing to resume it as though incomplete")
    return checkpoint["next_dataset_index"]


__all__ = [
    "TOP_CHECKPOINT_KEYS",
    "resume_dataset_index",
    "validate_checkpoint_against_canonical_order",
    "validate_checkpoint_structure",
]
