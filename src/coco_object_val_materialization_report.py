"""Final-manifest schema/invariant validation for the COCO-Object val2017
mask-materialization stage. An incomplete run must never produce a
document this module accepts."""

from __future__ import annotations

from typing import Any, Mapping

from src.coco_object_val_materialization import CLASS_COUNT
from src.coco_object_val_materialization_checkpoint import (
    _require_bool, _require_closed_mapping, _require_exact_string, _require_git_identity,
    _require_int, _require_int_list, _require_sha256,
)
from src.coco_object_val_materialization_identity import CocoObjectValMaterializationIdentityError

TOP_MANIFEST_KEYS = frozenset(
    {
        "schema", "identity", "identity_sha256", "git_commit", "split", "complete", "final", "image_count",
        "image_order_digest", "completed_image_ids_digest", "aggregate_decoded_mask_digest",
        "aggregate_encoded_file_digest", "source_masks_digest", "converter_sha256", "dataset_class_sha256",
        "dataset_config_sha256", "materialization_identity_sha256", "class_count", "background_class_index",
        "output_mask_suffix", "dtype", "aggregate_label_histogram", "total_pixels", "masks_with_foreground",
        "all_background_masks", "crowd_overlap_policy", "no_source_mutation", "no_train_masks_generated",
        "software_versions", "created_at_utc",
    }
)
SOFTWARE_VERSION_KEYS = frozenset({"python", "numpy", "pillow"})


def verify_record(record: Mapping[str, Any], identity: Mapping[str, Any], *, identity_sha256: str) -> None:
    """The sole authority for the final manifest's schema/type/invariant
    contract. An incomplete run must never pass this check."""
    _require_closed_mapping(record, TOP_MANIFEST_KEYS, "manifest")

    manifest_schema_name = identity["manifest"]["schema_name"]
    if _require_exact_string(record["schema"], "manifest.schema") != manifest_schema_name:
        raise CocoObjectValMaterializationIdentityError("manifest.schema mismatch")
    if _require_exact_string(record["identity"], "manifest.identity") != identity["identity"]["name"]:
        raise CocoObjectValMaterializationIdentityError("manifest.identity mismatch")
    if _require_sha256(record["identity_sha256"], "manifest.identity_sha256") != identity_sha256:
        raise CocoObjectValMaterializationIdentityError("manifest.identity_sha256 does not match the loaded identity file")
    _require_git_identity(record["git_commit"], "manifest.git_commit")

    if record["split"] != identity["protocol"]["split"]:
        raise CocoObjectValMaterializationIdentityError("manifest.split disagrees with identity.protocol.split")
    if _require_bool(record["complete"], "manifest.complete") is not True:
        raise CocoObjectValMaterializationIdentityError("manifest.complete must be true")
    if _require_bool(record["final"], "manifest.final") is not True:
        raise CocoObjectValMaterializationIdentityError("manifest.final must be true")
    image_count = _require_int(record["image_count"], "manifest.image_count", minimum=1)
    if image_count != identity["protocol"]["expected_image_count"]:
        raise CocoObjectValMaterializationIdentityError("manifest.image_count disagrees with identity.protocol.expected_image_count")

    _require_sha256(record["image_order_digest"], "manifest.image_order_digest")
    _require_sha256(record["completed_image_ids_digest"], "manifest.completed_image_ids_digest")
    _require_sha256(record["aggregate_decoded_mask_digest"], "manifest.aggregate_decoded_mask_digest")
    _require_sha256(record["aggregate_encoded_file_digest"], "manifest.aggregate_encoded_file_digest")
    _require_sha256(record["source_masks_digest"], "manifest.source_masks_digest")

    if record["converter_sha256"] != identity["converter"]["canonical_converter_sha256"]:
        raise CocoObjectValMaterializationIdentityError("manifest.converter_sha256 disagrees with identity.converter.canonical_converter_sha256")
    if record["dataset_class_sha256"] != identity["converter"]["dataset_class_sha256"]:
        raise CocoObjectValMaterializationIdentityError("manifest.dataset_class_sha256 disagrees with identity.converter.dataset_class_sha256")
    if record["dataset_config_sha256"] != identity["converter"]["dataset_config_sha256"]:
        raise CocoObjectValMaterializationIdentityError("manifest.dataset_config_sha256 disagrees with identity.converter.dataset_config_sha256")
    if record["materialization_identity_sha256"] != identity_sha256:
        raise CocoObjectValMaterializationIdentityError("manifest.materialization_identity_sha256 disagrees with the loaded identity")

    if _require_int(record["class_count"], "manifest.class_count") != CLASS_COUNT:
        raise CocoObjectValMaterializationIdentityError(f"manifest.class_count must be exactly {CLASS_COUNT}")
    if _require_int(record["background_class_index"], "manifest.background_class_index") != identity["class_contract"]["background_class_index"]:
        raise CocoObjectValMaterializationIdentityError("manifest.background_class_index disagrees with identity.class_contract.background_class_index")
    if record["output_mask_suffix"] != identity["converter"]["output_mask_suffix"]:
        raise CocoObjectValMaterializationIdentityError("manifest.output_mask_suffix disagrees with identity.converter.output_mask_suffix")
    if record["dtype"] != identity["output"]["dtype"]:
        raise CocoObjectValMaterializationIdentityError("manifest.dtype disagrees with identity.output.dtype")

    histogram = _require_int_list(record["aggregate_label_histogram"], "manifest.aggregate_label_histogram", length=CLASS_COUNT)
    if any(count < 0 for count in histogram):
        raise CocoObjectValMaterializationIdentityError("manifest.aggregate_label_histogram entries must be non-negative")
    total_pixels = _require_int(record["total_pixels"], "manifest.total_pixels", minimum=0)
    if total_pixels != sum(histogram):
        raise CocoObjectValMaterializationIdentityError("manifest.total_pixels disagrees with sum(aggregate_label_histogram)")

    masks_with_foreground = _require_int(record["masks_with_foreground"], "manifest.masks_with_foreground", minimum=0)
    all_background_masks = _require_int(record["all_background_masks"], "manifest.all_background_masks", minimum=0)
    if masks_with_foreground + all_background_masks != image_count:
        raise CocoObjectValMaterializationIdentityError("manifest.masks_with_foreground + manifest.all_background_masks must equal manifest.image_count")

    _require_exact_string(record["crowd_overlap_policy"], "manifest.crowd_overlap_policy")
    if _require_bool(record["no_source_mutation"], "manifest.no_source_mutation") is not True:
        raise CocoObjectValMaterializationIdentityError("manifest.no_source_mutation must be true")
    if _require_bool(record["no_train_masks_generated"], "manifest.no_train_masks_generated") is not True:
        raise CocoObjectValMaterializationIdentityError("manifest.no_train_masks_generated must be true")

    versions = _require_closed_mapping(record["software_versions"], SOFTWARE_VERSION_KEYS, "manifest.software_versions")
    for key in SOFTWARE_VERSION_KEYS:
        _require_exact_string(versions[key], f"manifest.software_versions.{key}")

    _require_exact_string(record["created_at_utc"], "manifest.created_at_utc")


__all__ = ["TOP_MANIFEST_KEYS", "verify_record"]
