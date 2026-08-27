"""Checkpoint schema/invariant validation for the shared VOC2012 V20/V21
matched evaluator (E3 vs k11 vs k12, T=320 finite-step). Sole authority
for the checkpoint's schema/type/binding/completeness checks -- never
reimplemented at any call site. A checkpoint bound to a different
identity/checkpoint/source-manifest must be rejected."""

from __future__ import annotations

from typing import Any, Mapping

from src.voc2012_matched_evaluator_identity import Voc2012MatchedEvaluatorIdentityError

TOP_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "run_mode", "identity", "identity_sha256", "matched_identity_sha256",
        "voc2012_source_identity_sha256", "source_manifest_sha256", "bridge_checkpoint_sha256", "git_commit",
        "v20_class_count", "v21_class_count", "live_v20_class_names_digest", "live_v21_class_names_digest",
        "image_count_expected", "image_order_digest", "next_dataset_index",
        "completed_image_ids", "images_completed_count", "windows_processed_total",
        "complete", "created_at_utc", "updated_at_utc",
    }
)


def _require_closed_mapping(value: Any, expected_keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} has an unexpected schema")
    return value


def _require_exact_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be at most {maximum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_git_identity(value: Any, label: str) -> str:
    token = _require_exact_string(value, label)
    if len(token) != 40 or any(c not in "0123456789abcdef" for c in token):
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be a full Git identity")
    return token


def _require_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise Voc2012MatchedEvaluatorIdentityError(f"{label} must be a list of exact strings")
    return value


def validate_checkpoint_structure(
    checkpoint: Mapping[str, Any], *, identity: Mapping[str, Any], identity_sha256: str,
    run_mode: str | None = None, v20_class_count: int | None = None, v21_class_count: int | None = None,
    source_manifest_sha256: str | None = None,
) -> None:
    """The sole authority for this checkpoint's schema/type/binding/
    completeness checks. Rejects a checkpoint bound to a different
    identity, run mode, class counts, or (when supplied) source manifest."""
    _require_closed_mapping(checkpoint, TOP_CHECKPOINT_KEYS, "checkpoint")

    if _require_exact_string(checkpoint["schema"], "checkpoint.schema") != identity["checkpoint"]["schema_name"]:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.schema mismatch")
    if _require_exact_string(checkpoint["identity"], "checkpoint.identity") != identity["identity"]["name"]:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.identity mismatch")
    if _require_sha256(checkpoint["identity_sha256"], "checkpoint.identity_sha256") != identity_sha256:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.identity_sha256 does not match the loaded identity file")
    if checkpoint["matched_identity_sha256"] != identity["parent_identities"]["matched_identity_sha256"]:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.matched_identity_sha256 disagrees with the identity's parent")
    if checkpoint["voc2012_source_identity_sha256"] != identity["parent_identities"]["voc2012_source_identity_sha256"]:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.voc2012_source_identity_sha256 disagrees with the identity's parent")
    recorded_source_manifest_sha256 = _require_sha256(checkpoint["source_manifest_sha256"], "checkpoint.source_manifest_sha256")
    if source_manifest_sha256 is not None and recorded_source_manifest_sha256 != source_manifest_sha256:
        raise Voc2012MatchedEvaluatorIdentityError(
            "checkpoint.source_manifest_sha256 disagrees with the currently-supplied --source-manifest"
        )
    if (
        _require_sha256(checkpoint["bridge_checkpoint_sha256"], "checkpoint.bridge_checkpoint_sha256")
        != identity["model_and_checkpoint"]["projection_checkpoint_sha256"]
    ):
        raise Voc2012MatchedEvaluatorIdentityError(
            "checkpoint.bridge_checkpoint_sha256 disagrees with identity.model_and_checkpoint.projection_checkpoint_sha256"
        )
    _require_git_identity(checkpoint["git_commit"], "checkpoint.git_commit")

    recorded_run_mode = _require_exact_string(checkpoint["run_mode"], "checkpoint.run_mode")
    if recorded_run_mode not in ("pilot20", "pilot100", "full"):
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.run_mode must be one of pilot20, pilot100, full")
    if run_mode is not None and recorded_run_mode != run_mode:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.run_mode disagrees with --run-mode")
    registered_image_count = identity["run_modes"][f"{recorded_run_mode}_image_count"]
    expected_image_count = _require_int(checkpoint["image_count_expected"], "checkpoint.image_count_expected", minimum=1)
    if expected_image_count != registered_image_count:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"checkpoint.image_count_expected ({expected_image_count}) disagrees with the registered "
            f"{recorded_run_mode} image count ({registered_image_count})"
        )

    recorded_v20_count = _require_int(checkpoint["v20_class_count"], "checkpoint.v20_class_count", minimum=1)
    if recorded_v20_count != identity["v20_protocol"]["class_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.v20_class_count disagrees with identity.v20_protocol.class_count")
    if v20_class_count is not None and recorded_v20_count != v20_class_count:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.v20_class_count disagrees with the live inference class count")
    recorded_v21_count = _require_int(checkpoint["v21_class_count"], "checkpoint.v21_class_count", minimum=1)
    if recorded_v21_count != identity["v21_protocol"]["class_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.v21_class_count disagrees with identity.v21_protocol.class_count")
    if v21_class_count is not None and recorded_v21_count != v21_class_count:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.v21_class_count disagrees with the live inference class count")

    _require_sha256(checkpoint["live_v20_class_names_digest"], "checkpoint.live_v20_class_names_digest")
    _require_sha256(checkpoint["live_v21_class_names_digest"], "checkpoint.live_v21_class_names_digest")
    _require_sha256(checkpoint["image_order_digest"], "checkpoint.image_order_digest")

    next_index = _require_int(checkpoint["next_dataset_index"], "checkpoint.next_dataset_index", minimum=0, maximum=expected_image_count)
    completed = _require_string_list(checkpoint["completed_image_ids"], "checkpoint.completed_image_ids")
    if len(set(completed)) != len(completed):
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.completed_image_ids contains a duplicate image ID")
    images_completed_count = _require_int(checkpoint["images_completed_count"], "checkpoint.images_completed_count", minimum=0)
    if images_completed_count != len(completed):
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.images_completed_count disagrees with len(completed_image_ids)")
    if next_index != len(completed):
        raise Voc2012MatchedEvaluatorIdentityError(
            f"checkpoint.next_dataset_index ({next_index}) must equal the number of completed images "
            f"({len(completed)}) -- refusing to resume a checkpoint that could skip or duplicate an image"
        )

    complete = _require_bool(checkpoint["complete"], "checkpoint.complete")
    if complete and next_index != expected_image_count:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.complete is true but next_dataset_index does not equal the expected image count")

    windows_processed_total = _require_int(checkpoint["windows_processed_total"], "checkpoint.windows_processed_total", minimum=0)
    if images_completed_count > 0 and windows_processed_total < images_completed_count:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.windows_processed_total must be at least images_completed_count")

    _require_exact_string(checkpoint["created_at_utc"], "checkpoint.created_at_utc")
    _require_exact_string(checkpoint["updated_at_utc"], "checkpoint.updated_at_utc")


def validate_checkpoint_against_canonical_order(
    checkpoint: Mapping[str, Any], expected_image_ids: list[str], *, image_order_digest: str,
) -> None:
    if checkpoint["image_order_digest"] != image_order_digest:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint.image_order_digest disagrees with the freshly-resolved dataset order")
    next_index = checkpoint["next_dataset_index"]
    expected_prefix = list(expected_image_ids[:next_index])
    if checkpoint["completed_image_ids"] != expected_prefix:
        raise Voc2012MatchedEvaluatorIdentityError(
            "checkpoint.completed_image_ids does not match the canonical dataset image-ID prefix "
            f"for the first {next_index} images; refusing to resume against a mismatched image order"
        )


def resume_dataset_index(checkpoint: Mapping[str, Any]) -> int:
    if checkpoint["complete"] is True:
        raise Voc2012MatchedEvaluatorIdentityError("checkpoint is already complete; refusing to resume it as though incomplete")
    return checkpoint["next_dataset_index"]


def validate_checkpoint_against_per_image_stats(
    checkpoint: Mapping[str, Any], stats_manifest: Mapping[str, Any], *, identity: Mapping[str, Any],
) -> None:
    """Bind a per-image-stats manifest to an already-structurally-validated
    checkpoint document (validate_checkpoint_structure has already bound
    that checkpoint to identity/source-manifest/bridge-checkpoint/run-mode/
    expected-count/class-counts) -- so this only needs to bind the STATS
    artifact to the CHECKPOINT, transitively inheriting every other
    binding without re-deriving it a second time. CPU-only; must run
    before any dataset/model/CUDA work."""
    expected_schema = identity["artifacts"]["per_image_stats_manifest_schema_name"]
    if stats_manifest["schema"] != expected_schema:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"per-image-stats manifest schema {stats_manifest['schema']!r} disagrees with the authoritative {expected_schema!r}"
        )
    if stats_manifest["v20_class_count"] != checkpoint["v20_class_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("per-image-stats manifest v20_class_count disagrees with the checkpoint")
    if stats_manifest["v21_class_count"] != checkpoint["v21_class_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("per-image-stats manifest v21_class_count disagrees with the checkpoint")
    if stats_manifest["live_v20_class_names_digest"] != checkpoint["live_v20_class_names_digest"]:
        raise Voc2012MatchedEvaluatorIdentityError("per-image-stats manifest live_v20_class_names_digest disagrees with the checkpoint")
    if stats_manifest["live_v21_class_names_digest"] != checkpoint["live_v21_class_names_digest"]:
        raise Voc2012MatchedEvaluatorIdentityError("per-image-stats manifest live_v21_class_names_digest disagrees with the checkpoint")

    next_index = checkpoint["next_dataset_index"]
    if stats_manifest["image_count"] != next_index:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"per-image-stats manifest image_count ({stats_manifest['image_count']}) disagrees with "
            f"checkpoint.next_dataset_index ({next_index})"
        )
    if list(stats_manifest["dataset_indices"]) != list(range(next_index)):
        raise Voc2012MatchedEvaluatorIdentityError(
            "per-image-stats manifest dataset_indices is not the exact canonical 0..next_dataset_index-1 prefix"
        )
    if list(stats_manifest["image_ids"]) != list(checkpoint["completed_image_ids"]):
        raise Voc2012MatchedEvaluatorIdentityError("per-image-stats manifest image_ids disagrees with checkpoint.completed_image_ids")


__all__ = [
    "TOP_CHECKPOINT_KEYS",
    "resume_dataset_index",
    "validate_checkpoint_against_canonical_order",
    "validate_checkpoint_against_per_image_stats",
    "validate_checkpoint_structure",
]
