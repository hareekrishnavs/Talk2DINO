"""Dataset-root resolution and deterministic manifest generation/
verification for the ADE20K (ADEChallengeData2016) validation source.

CPU-only: never initializes CUDA, never loads a model, never imports
torch. Reads only images/validation/*.jpg, annotations/validation/*.png
(and, for the disjointness check, the training-split filenames only --
never their content) under the resolved dataset root. The manifest
contains logical relative paths/labels only -- never the physical
dataset root, username, hostname, or an absolute path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.ade20k_dataset_identity import Ade20kDatasetIdentityError

CLASS_COUNT = 150
# Raw pixel domain before reduce_zero_label: 0 (ignored/background), 1..150
# (evaluated classes), and 255 (already-ignored, if ever present).
LEGAL_RAW_LABELS = frozenset(range(0, 151)) | {255}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_digest(rows: list[Any]) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=True).encode("utf-8")).hexdigest()


def resolve_dataset_root(data_root: Path, identity: dict[str, Any]) -> Path:
    """Fail-closed ADE20K root resolution. Accepts either the
    ADEChallengeData2016 directory itself, or an unambiguous parent
    containing exactly one such subdirectory -- never assumes a fixed
    nesting depth, and reports ambiguity instead of guessing."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise Ade20kDatasetIdentityError(f"dataset root does not exist: {data_root}")
    if not data_root.is_dir():
        raise Ade20kDatasetIdentityError(f"dataset root is not a directory: {data_root}")

    required_subpaths = identity["expected_root_structure"]["required_subpaths"]
    ade_relative_root = identity["expected_root_structure"]["ade_relative_root"]

    def _satisfies_root_contract(candidate: Path) -> bool:
        return candidate.is_dir() and all((candidate / sub).is_dir() for sub in required_subpaths)

    candidates: list[Path] = []
    if _satisfies_root_contract(data_root):
        candidates.append(data_root)
    ade_child = data_root / ade_relative_root
    if ade_child != data_root and _satisfies_root_contract(ade_child):
        candidates.append(ade_child)

    # Reject ambiguity from *nested* ADEChallengeData2016 directories at
    # more than one depth under data_root (e.g. two independent copies).
    nested_matches = [p for p in data_root.rglob(ade_relative_root) if p.is_dir() and _satisfies_root_contract(p)]
    for match in nested_matches:
        if match not in candidates:
            candidates.append(match)

    if len(candidates) == 0:
        raise Ade20kDatasetIdentityError(
            f"cannot resolve an ADEChallengeData2016 root under {data_root}: neither it nor a "
            f"{ade_relative_root!r} descendant contains all of {list(required_subpaths)}"
        )
    if len(candidates) > 1:
        raise Ade20kDatasetIdentityError(
            f"ambiguous ADE20K root under {data_root}: multiple candidates satisfy the root contract: "
            f"{[str(c) for c in candidates]}"
        )
    resolved = candidates[0]
    # Reject a symlinked root that resolves outside the tree it appears
    # to live in.
    if resolved.is_symlink() and resolved.resolve().parent != resolved.parent.resolve():
        raise Ade20kDatasetIdentityError(f"resolved ADE20K root {resolved} is a symlink escaping its containing directory")
    return resolved


def _scan_ids(directory: Path, *, suffix: str) -> list[str]:
    if not directory.is_dir():
        raise Ade20kDatasetIdentityError(f"expected directory does not exist: {directory}")
    stems: list[str] = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.name.endswith(suffix) and entry.is_file(follow_symlinks=False):
                stems.append(entry.name[: -len(suffix)])
    return sorted(stems)


def canonical_validation_ids(ade_root: Path, identity: dict[str, Any]) -> list[str]:
    """The exact order authority: sorted validation image filenames
    (no split file is configured for this dataset -- matches mmseg
    CustomDataset.load_annotations' own directory-listing sort when no
    split is given). Fails closed on any count mismatch, duplicate,
    image/mask ID disagreement, or syntactically invalid ID."""
    image_suffix = identity["source"]["image_suffix"]
    annotation_suffix = identity["source"]["annotation_suffix"]
    img_dir = ade_root / identity["source"]["image_relative_root"]
    mask_dir = ade_root / identity["source"]["annotation_relative_root"]

    image_ids = _scan_ids(img_dir, suffix=image_suffix)
    mask_ids = _scan_ids(mask_dir, suffix=annotation_suffix)

    if len(set(image_ids)) != len(image_ids):
        raise Ade20kDatasetIdentityError(f"{img_dir} contains a duplicate image ID")
    if image_ids != mask_ids:
        only_images = sorted(set(image_ids) - set(mask_ids))
        only_masks = sorted(set(mask_ids) - set(image_ids))
        raise Ade20kDatasetIdentityError(
            f"validation image IDs and mask IDs disagree: {len(only_images)} image-only, "
            f"{len(only_masks)} mask-only (examples: {only_images[:3]}, {only_masks[:3]})"
        )

    expected_count = identity["protocol"]["expected_image_count"]
    if len(image_ids) != expected_count:
        raise Ade20kDatasetIdentityError(f"{img_dir} has {len(image_ids)} entries, expected exactly {expected_count}")

    for image_id in image_ids:
        if not image_id.startswith("ADE_val_") or not image_id[len("ADE_val_"):].isdigit():
            raise Ade20kDatasetIdentityError(f"validation image ID {image_id!r} does not match the ADE_val_######## syntax")

    return image_ids


def check_train_val_disjointness(ade_root: Path, identity: dict[str, Any], val_ids: list[str]) -> int:
    """If the training directories exist, require zero ID overlap with
    the validation split. Never required if training data is absent --
    evaluation never needs it. Returns the observed training count."""
    train_img_dir = ade_root / identity["source"]["training_image_relative_root"]
    if not train_img_dir.is_dir():
        return 0
    train_ids = _scan_ids(train_img_dir, suffix=identity["source"]["image_suffix"])
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise Ade20kDatasetIdentityError(f"train/validation ID overlap detected ({len(overlap)} ids, e.g. {sorted(overlap)[:5]})")
    return len(train_ids)


def image_order_digest(image_ids: list[str]) -> str:
    return _ordered_digest(image_ids)


def _decode_image(path: Path, *, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size  # (width, height)
    except (OSError, UnidentifiedImageError) as error:
        raise Ade20kDatasetIdentityError(f"cannot decode {label} {path}: {error}") from error


def _decode_mask(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as im:
            mode = im.mode
            arr = np.array(im)
    except (OSError, UnidentifiedImageError) as error:
        raise Ade20kDatasetIdentityError(f"cannot decode {label} {path}: {error}") from error
    if mode not in ("L", "P", "I"):
        raise Ade20kDatasetIdentityError(f"{label} {path} has mode {mode!r}, expected a single-channel mask")
    if arr.ndim != 2:
        raise Ade20kDatasetIdentityError(f"{label} {path} decoded to {arr.ndim} dimensions, expected a single-channel 2D array")
    return arr


def scan_validation_split(ade_root: Path, identity: dict[str, Any], image_ids: list[str]) -> dict[str, Any]:
    """Exhaustively decode and validate every validation image/mask pair.
    Never samples: every id in `image_ids` is checked."""
    img_dir = ade_root / identity["source"]["image_relative_root"]
    mask_dir = ade_root / identity["source"]["annotation_relative_root"]
    image_suffix = identity["source"]["image_suffix"]
    annotation_suffix = identity["source"]["annotation_suffix"]

    image_hashes: list[tuple[str, str]] = []
    mask_encoded_hashes: list[tuple[str, str]] = []
    mask_decoded_hashes: list[tuple[str, str]] = []
    dimension_mismatches: list[dict[str, Any]] = []
    observed_labels: set[int] = set()
    total_image_bytes = 0
    total_mask_bytes = 0

    for image_id in image_ids:
        image_path = img_dir / f"{image_id}{image_suffix}"
        mask_path = mask_dir / f"{image_id}{annotation_suffix}"
        if not image_path.is_file():
            raise Ade20kDatasetIdentityError(f"missing validation image for id {image_id!r}: {image_path}")
        if not mask_path.is_file():
            raise Ade20kDatasetIdentityError(f"missing validation mask for id {image_id!r}: {mask_path}")

        image_wh = _decode_image(image_path, label="validation image")
        mask_arr = _decode_mask(mask_path, label="validation mask")
        mask_wh = (mask_arr.shape[1], mask_arr.shape[0])
        if image_wh != mask_wh:
            dimension_mismatches.append({"image_id": image_id, "image_wh": list(image_wh), "mask_wh": list(mask_wh)})

        labels_here = set(np.unique(mask_arr).tolist())
        illegal = labels_here - LEGAL_RAW_LABELS
        if illegal:
            raise Ade20kDatasetIdentityError(f"mask {mask_path} contains illegal label value(s): {sorted(illegal)}")
        observed_labels |= labels_here

        image_hashes.append((image_id, sha256_file(image_path)))
        mask_encoded_hashes.append((image_id, sha256_file(mask_path)))
        mask_decoded_hashes.append((image_id, hashlib.sha256(mask_arr.tobytes()).hexdigest()))
        total_image_bytes += image_path.stat().st_size
        total_mask_bytes += mask_path.stat().st_size

    if dimension_mismatches:
        raise Ade20kDatasetIdentityError(f"image/mask dimension mismatch for {len(dimension_mismatches)} id(s): {dimension_mismatches[:5]}")

    return {
        "image_hashes": image_hashes,
        "mask_encoded_hashes": mask_encoded_hashes,
        "mask_decoded_hashes": mask_decoded_hashes,
        "observed_labels": sorted(observed_labels),
        "total_image_bytes": total_image_bytes,
        "total_mask_bytes": total_mask_bytes,
        "dimension_reconciliation": {"all_agree": True, "mismatches": []},
    }


def build_manifest(
    *, identity: dict[str, Any], identity_sha256: str, image_ids: list[str], scan: dict[str, Any],
    training_image_count: int, generated_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": identity["manifest"]["schema_name"],
        "identity_name": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "source_root_logical_label": identity["expected_root_structure"]["ade_relative_root"],
        "split": identity["protocol"]["split"],
        "image_count": len(image_ids),
        "class_count": identity["class_contract"]["class_count"],
        "image_order_digest": image_order_digest(image_ids),
        "image_content_digest": _ordered_digest(scan["image_hashes"]),
        "mask_encoded_content_digest": _ordered_digest(scan["mask_encoded_hashes"]),
        "mask_decoded_content_digest": _ordered_digest(scan["mask_decoded_hashes"]),
        "observed_label_set": scan["observed_labels"],
        "total_image_bytes": scan["total_image_bytes"],
        "total_mask_bytes": scan["total_mask_bytes"],
        "dimension_reconciliation": scan["dimension_reconciliation"],
        "training_image_count": training_image_count,
        "train_val_disjoint": True,
        "conversion_used": False,
        "generated_at_utc": generated_at_utc,
    }


MANIFEST_TOP_KEYS = frozenset(
    {
        "schema", "identity_name", "identity_sha256", "source_root_logical_label", "split", "image_count",
        "class_count", "image_order_digest", "image_content_digest", "mask_encoded_content_digest",
        "mask_decoded_content_digest", "observed_label_set", "total_image_bytes", "total_mask_bytes",
        "dimension_reconciliation", "training_image_count", "train_val_disjoint", "conversion_used",
        "generated_at_utc",
    }
)


def _require_manifest_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise Ade20kDatasetIdentityError(f"{label} must be an exact string")
    return value


def _require_manifest_sha256(value: Any, label: str) -> str:
    token = _require_manifest_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise Ade20kDatasetIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_manifest_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Ade20kDatasetIdentityError(f"{label} must be an exact boolean")
    return value


def _require_manifest_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise Ade20kDatasetIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise Ade20kDatasetIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_manifest_int_list(value: Any, label: str) -> list:
    if type(value) is not list:
        raise Ade20kDatasetIdentityError(f"{label} must be an exact list")
    for index, item in enumerate(value):
        if type(item) is not int:
            raise Ade20kDatasetIdentityError(f"{label}[{index}] must be an exact non-boolean integer")
    return value


def _require_manifest_timestamp(value: Any, label: str) -> str:
    token = _require_manifest_string(value, label)
    try:
        from datetime import datetime

        datetime.fromisoformat(token)
    except ValueError as error:
        raise Ade20kDatasetIdentityError(f"{label} must be an ISO-8601 timestamp string: {error}") from error
    return token


def verify_manifest_against_identity(manifest: dict[str, Any], identity: dict[str, Any], *, identity_sha256: str) -> None:
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_TOP_KEYS:
        raise Ade20kDatasetIdentityError("manifest has an unexpected schema")

    _require_manifest_string(manifest["schema"], "manifest.schema")
    _require_manifest_string(manifest["identity_name"], "manifest.identity_name")
    _require_manifest_sha256(manifest["identity_sha256"], "manifest.identity_sha256")
    source_label = _require_manifest_string(manifest["source_root_logical_label"], "manifest.source_root_logical_label")
    if "/" in source_label or "\\" in source_label or Path(source_label).is_absolute():
        raise Ade20kDatasetIdentityError("manifest.source_root_logical_label must not embed a filesystem path")
    _require_manifest_string(manifest["split"], "manifest.split")
    _require_manifest_int(manifest["image_count"], "manifest.image_count", minimum=0)
    _require_manifest_int(manifest["class_count"], "manifest.class_count", minimum=0)
    _require_manifest_sha256(manifest["image_order_digest"], "manifest.image_order_digest")
    _require_manifest_sha256(manifest["image_content_digest"], "manifest.image_content_digest")
    _require_manifest_sha256(manifest["mask_encoded_content_digest"], "manifest.mask_encoded_content_digest")
    _require_manifest_sha256(manifest["mask_decoded_content_digest"], "manifest.mask_decoded_content_digest")
    _require_manifest_int_list(manifest["observed_label_set"], "manifest.observed_label_set")
    _require_manifest_int(manifest["total_image_bytes"], "manifest.total_image_bytes", minimum=0)
    _require_manifest_int(manifest["total_mask_bytes"], "manifest.total_mask_bytes", minimum=0)
    dimension_reconciliation = manifest["dimension_reconciliation"]
    if not isinstance(dimension_reconciliation, dict) or set(dimension_reconciliation) != {"all_agree", "mismatches"}:
        raise Ade20kDatasetIdentityError("manifest.dimension_reconciliation must be an exact mapping with keys {'all_agree', 'mismatches'}")
    _require_manifest_bool(dimension_reconciliation["all_agree"], "manifest.dimension_reconciliation.all_agree")
    if type(dimension_reconciliation["mismatches"]) is not list:
        raise Ade20kDatasetIdentityError("manifest.dimension_reconciliation.mismatches must be an exact list")
    _require_manifest_int(manifest["training_image_count"], "manifest.training_image_count", minimum=0)
    _require_manifest_bool(manifest["train_val_disjoint"], "manifest.train_val_disjoint")
    _require_manifest_bool(manifest["conversion_used"], "manifest.conversion_used")
    _require_manifest_timestamp(manifest["generated_at_utc"], "manifest.generated_at_utc")

    if manifest["schema"] != identity["manifest"]["schema_name"]:
        raise Ade20kDatasetIdentityError("manifest.schema disagrees with identity.manifest.schema_name")
    if manifest["identity_name"] != identity["identity"]["name"]:
        raise Ade20kDatasetIdentityError("manifest.identity_name disagrees with identity.identity.name")
    if manifest["identity_sha256"] != identity_sha256:
        raise Ade20kDatasetIdentityError("manifest.identity_sha256 does not match the loaded identity file")
    if manifest["source_root_logical_label"] != identity["expected_root_structure"]["ade_relative_root"]:
        raise Ade20kDatasetIdentityError("manifest.source_root_logical_label disagrees with the identity's declared logical root label")
    if manifest["split"] != identity["protocol"]["split"]:
        raise Ade20kDatasetIdentityError("manifest.split disagrees with identity.protocol.split")
    if manifest["image_count"] != identity["protocol"]["expected_image_count"]:
        raise Ade20kDatasetIdentityError("manifest.image_count disagrees with identity.protocol.expected_image_count")
    if manifest["class_count"] != identity["class_contract"]["class_count"]:
        raise Ade20kDatasetIdentityError("manifest.class_count disagrees with identity.class_contract.class_count")
    if not (set(manifest["observed_label_set"]) <= LEGAL_RAW_LABELS):
        raise Ade20kDatasetIdentityError("manifest.observed_label_set contains an illegal label value")
    if dimension_reconciliation["all_agree"] is not True or dimension_reconciliation["mismatches"]:
        raise Ade20kDatasetIdentityError("manifest.dimension_reconciliation reports a dimension disagreement")
    if manifest["train_val_disjoint"] is not True:
        raise Ade20kDatasetIdentityError("manifest.train_val_disjoint must be true")
    if manifest["conversion_used"] is not False:
        raise Ade20kDatasetIdentityError("manifest.conversion_used must be false")


__all__ = [
    "CLASS_COUNT",
    "LEGAL_RAW_LABELS",
    "MANIFEST_TOP_KEYS",
    "build_manifest",
    "canonical_validation_ids",
    "check_train_val_disjointness",
    "image_order_digest",
    "resolve_dataset_root",
    "scan_validation_split",
    "sha256_file",
    "verify_manifest_against_identity",
]
