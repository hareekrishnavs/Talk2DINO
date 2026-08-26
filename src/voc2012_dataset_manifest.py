"""Dataset-root resolution and deterministic manifest generation/
verification for the shared PASCAL VOC 2012 validation source (V20/V21).

CPU-only: never initializes CUDA, never loads a model, never imports
torch. Reads only JPEGImages/*.jpg, SegmentationClass/*.png, and
ImageSets/Segmentation/val.txt under the resolved dataset root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.voc2012_dataset_identity import Voc2012DatasetIdentityError

CLASS_COUNT_V21 = 21
LEGAL_RAW_LABELS = frozenset(range(CLASS_COUNT_V21)) | {255}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_digest(rows: list[Any]) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=True).encode("utf-8")).hexdigest()


def resolve_dataset_root(data_root: Path, identity: dict[str, Any]) -> Path:
    """Fail-closed VOC2012 root resolution. Accepts either the VOC2012
    directory itself, or its immediate parent (a 'VOCdevkit'-shaped
    directory containing exactly one VOC2012 subdirectory) -- never
    assumes a fixed nesting depth, and reports ambiguity instead of
    guessing when more than one candidate satisfies the root contract."""
    data_root = Path(data_root)
    required_subpaths = identity["expected_root_structure"]["required_subpaths"]

    def _satisfies_root_contract(candidate: Path) -> bool:
        return candidate.is_dir() and all((candidate / sub).exists() for sub in required_subpaths)

    candidates: list[Path] = []
    if _satisfies_root_contract(data_root):
        candidates.append(data_root)
    voc_child = data_root / identity["expected_root_structure"]["voc2012_relative_root"]
    if voc_child != data_root and _satisfies_root_contract(voc_child):
        candidates.append(voc_child)

    if not data_root.is_dir():
        raise Voc2012DatasetIdentityError(f"--data-root does not exist or is not a directory: {data_root}")
    if len(candidates) == 0:
        raise Voc2012DatasetIdentityError(
            f"cannot resolve a VOC2012 root under {data_root}: neither it nor its "
            f"{identity['expected_root_structure']['voc2012_relative_root']!r} subdirectory contains all of "
            f"{list(required_subpaths)}"
        )
    if len(candidates) > 1:
        raise Voc2012DatasetIdentityError(
            f"ambiguous VOC2012 root under {data_root}: multiple candidates satisfy the root contract: "
            f"{[str(c) for c in candidates]}"
        )
    return candidates[0]


def canonical_validation_ids(voc_root: Path, identity: dict[str, Any]) -> list[str]:
    """The exact order authority: ImageSets/Segmentation/val.txt line
    order, each line stripped -- never re-sorted, never derived from a
    directory listing. Fails closed on any empty, duplicate, absolute,
    traversal, or case-colliding entry."""
    split_path = voc_root / identity["source"]["split_relative_path"]
    try:
        raw_lines = split_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Voc2012DatasetIdentityError(f"cannot read validation split file {split_path}: {error}") from error

    ids = [line.strip() for line in raw_lines]
    if any(image_id == "" for image_id in ids):
        raise Voc2012DatasetIdentityError(f"{split_path} contains a blank/whitespace-only line")
    for image_id in ids:
        if image_id.startswith("/") or Path(image_id).is_absolute():
            raise Voc2012DatasetIdentityError(f"{split_path} contains an absolute-looking entry: {image_id!r}")
        if ".." in Path(image_id).parts or "/" in image_id or "\\" in image_id:
            raise Voc2012DatasetIdentityError(f"{split_path} contains a path-traversal/separator entry: {image_id!r}")
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        duplicates = []
        for image_id in ids:
            if image_id in seen:
                duplicates.append(image_id)
            seen.add(image_id)
        raise Voc2012DatasetIdentityError(f"{split_path} contains duplicate id(s): {sorted(set(duplicates))[:5]}")
    lower_ids = [image_id.lower() for image_id in ids]
    if len(set(lower_ids)) != len(set(ids)):
        raise Voc2012DatasetIdentityError(f"{split_path} contains case-colliding id(s)")

    expected_count = identity["protocol"]["expected_image_count"]
    if len(ids) != expected_count:
        raise Voc2012DatasetIdentityError(f"{split_path} has {len(ids)} entries, expected exactly {expected_count}")
    return ids


def image_order_digest(image_ids: list[str]) -> str:
    return _ordered_digest(image_ids)


def _decode_image(path: Path, *, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size  # (width, height)
    except (OSError, UnidentifiedImageError) as error:
        raise Voc2012DatasetIdentityError(f"cannot decode {label} {path}: {error}") from error


def _decode_mask(path: Path, *, label: str) -> tuple[np.ndarray, str]:
    try:
        with Image.open(path) as im:
            mode = im.mode
            arr = np.array(im)
    except (OSError, UnidentifiedImageError) as error:
        raise Voc2012DatasetIdentityError(f"cannot decode {label} {path}: {error}") from error
    if mode != "P":
        raise Voc2012DatasetIdentityError(f"{label} {path} has mode {mode!r}, expected a single-channel palettized ('P') mask")
    if arr.ndim != 2:
        raise Voc2012DatasetIdentityError(f"{label} {path} decoded to {arr.ndim} dimensions, expected a single-channel 2D array")
    return arr, mode


def scan_validation_split(
    voc_root: Path, identity: dict[str, Any], image_ids: list[str],
) -> dict[str, Any]:
    """Exhaustively decode and validate every validation image/mask pair.
    Never samples: every id in `image_ids` is checked. Returns the raw
    evidence (per-id hashes, dimensions, label sets) needed to build or
    verify the deterministic manifest."""
    img_dir = voc_root / identity["source"]["image_relative_root"]
    mask_dir = voc_root / identity["source"]["annotation_relative_root"]
    image_suffix = identity["source"]["image_suffix"]
    annotation_suffix = identity["source"]["annotation_suffix"]

    image_hashes: list[tuple[str, str]] = []
    mask_encoded_hashes: list[tuple[str, str]] = []
    mask_decoded_hashes: list[tuple[str, str]] = []
    dimension_mismatches: list[dict[str, Any]] = []
    observed_labels: set[int] = set()
    label_histogram = [0] * CLASS_COUNT_V21
    ignore_pixel_count = 0

    for image_id in image_ids:
        image_path = img_dir / f"{image_id}{image_suffix}"
        mask_path = mask_dir / f"{image_id}{annotation_suffix}"
        if not image_path.is_file():
            raise Voc2012DatasetIdentityError(f"missing validation image for id {image_id!r}: {image_path}")
        if not mask_path.is_file():
            raise Voc2012DatasetIdentityError(f"missing validation mask for id {image_id!r}: {mask_path}")

        image_wh = _decode_image(image_path, label="validation image")
        mask_arr, _mode = _decode_mask(mask_path, label="validation mask")
        mask_wh = (mask_arr.shape[1], mask_arr.shape[0])
        if image_wh != mask_wh:
            dimension_mismatches.append({"image_id": image_id, "image_wh": list(image_wh), "mask_wh": list(mask_wh)})

        labels_here = set(np.unique(mask_arr).tolist())
        illegal = labels_here - LEGAL_RAW_LABELS
        if illegal:
            raise Voc2012DatasetIdentityError(f"mask {mask_path} contains illegal label value(s): {sorted(illegal)}")
        observed_labels |= labels_here
        counts = np.bincount(mask_arr.reshape(-1), minlength=256)
        for value in range(CLASS_COUNT_V21):
            label_histogram[value] += int(counts[value])
        ignore_pixel_count += int(counts[255]) if counts.shape[0] > 255 else 0

        image_hashes.append((image_id, sha256_file(image_path)))
        mask_encoded_hashes.append((image_id, sha256_file(mask_path)))
        mask_decoded_hashes.append((image_id, hashlib.sha256(mask_arr.tobytes()).hexdigest()))

    if dimension_mismatches:
        raise Voc2012DatasetIdentityError(f"image/mask dimension mismatch for {len(dimension_mismatches)} id(s): {dimension_mismatches[:5]}")

    return {
        "image_hashes": image_hashes,
        "mask_encoded_hashes": mask_encoded_hashes,
        "mask_decoded_hashes": mask_decoded_hashes,
        "observed_labels": sorted(observed_labels),
        "label_histogram": label_histogram,
        "ignore_pixel_count": ignore_pixel_count,
        "dimension_reconciliation": {"all_agree": True, "mismatches": []},
    }


def build_manifest(
    *, identity: dict[str, Any], identity_sha256: str, voc_root: Path, image_ids: list[str], scan: dict[str, Any],
    generated_at_utc: str,
) -> dict[str, Any]:
    split_path = voc_root / identity["source"]["split_relative_path"]
    return {
        "schema": identity["manifest"]["schema_name"],
        "identity_name": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "source_root_logical_label": identity["expected_root_structure"]["voc2012_relative_root"],
        "split": identity["protocol"]["split"],
        "image_count": len(image_ids),
        "v21_class_count": identity["class_contract"]["v21_class_count"],
        "v20_class_count": identity["class_contract"]["v20_class_count"],
        "split_file_sha256": sha256_file(split_path),
        "image_order_digest": image_order_digest(image_ids),
        "image_content_digest": _ordered_digest(scan["image_hashes"]),
        "mask_encoded_content_digest": _ordered_digest(scan["mask_encoded_hashes"]),
        "mask_decoded_content_digest": _ordered_digest(scan["mask_decoded_hashes"]),
        "observed_label_set": scan["observed_labels"],
        "label_histogram_v21": scan["label_histogram"],
        "ignore_pixel_count": scan["ignore_pixel_count"],
        "dimension_reconciliation": scan["dimension_reconciliation"],
        "v20_v21_shared_source_assertion": True,
        "generated_at_utc": generated_at_utc,
    }


MANIFEST_CONTENT_DIGEST_EXCLUDED_KEYS = frozenset({"generated_at_utc"})

MANIFEST_TOP_KEYS = frozenset(
    {
        "schema", "identity_name", "identity_sha256", "source_root_logical_label", "split", "image_count",
        "v21_class_count", "v20_class_count", "split_file_sha256", "image_order_digest", "image_content_digest",
        "mask_encoded_content_digest", "mask_decoded_content_digest", "observed_label_set", "label_histogram_v21",
        "ignore_pixel_count", "dimension_reconciliation", "v20_v21_shared_source_assertion", "generated_at_utc",
    }
)


def _require_manifest_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise Voc2012DatasetIdentityError(f"{label} must be an exact string")
    return value


def _require_manifest_sha256(value: Any, label: str) -> str:
    token = _require_manifest_string(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise Voc2012DatasetIdentityError(f"{label} must be a lowercase SHA256")
    return token


def _require_manifest_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Voc2012DatasetIdentityError(f"{label} must be an exact boolean")
    return value


def _require_manifest_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise Voc2012DatasetIdentityError(f"{label} must be an exact non-boolean integer")
    if minimum is not None and value < minimum:
        raise Voc2012DatasetIdentityError(f"{label} must be at least {minimum}")
    return value


def _require_manifest_int_list(value: Any, label: str) -> list:
    if type(value) is not list:
        raise Voc2012DatasetIdentityError(f"{label} must be an exact list")
    for index, item in enumerate(value):
        if type(item) is not int:
            raise Voc2012DatasetIdentityError(f"{label}[{index}] must be an exact non-boolean integer")
    return value


def _require_manifest_timestamp(value: Any, label: str) -> str:
    token = _require_manifest_string(value, label)
    try:
        from datetime import datetime

        datetime.fromisoformat(token)
    except ValueError as error:
        raise Voc2012DatasetIdentityError(f"{label} must be an ISO-8601 timestamp string: {error}") from error
    return token


def verify_manifest_against_identity(manifest: dict[str, Any], identity: dict[str, Any], *, identity_sha256: str) -> None:
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_TOP_KEYS:
        raise Voc2012DatasetIdentityError("manifest has an unexpected schema")

    # Exact-type validation for every field, before any value comparison
    # -- never a bare `!=`, which under Python's numeric-tower semantics
    # would silently accept e.g. 1449.0 for an int field or 1 for a bool
    # field.
    _require_manifest_string(manifest["schema"], "manifest.schema")
    _require_manifest_string(manifest["identity_name"], "manifest.identity_name")
    _require_manifest_sha256(manifest["identity_sha256"], "manifest.identity_sha256")
    _require_manifest_string(manifest["source_root_logical_label"], "manifest.source_root_logical_label")
    _require_manifest_string(manifest["split"], "manifest.split")
    _require_manifest_int(manifest["image_count"], "manifest.image_count", minimum=0)
    _require_manifest_int(manifest["v21_class_count"], "manifest.v21_class_count", minimum=0)
    _require_manifest_int(manifest["v20_class_count"], "manifest.v20_class_count", minimum=0)
    _require_manifest_sha256(manifest["split_file_sha256"], "manifest.split_file_sha256")
    _require_manifest_sha256(manifest["image_order_digest"], "manifest.image_order_digest")
    _require_manifest_sha256(manifest["image_content_digest"], "manifest.image_content_digest")
    _require_manifest_sha256(manifest["mask_encoded_content_digest"], "manifest.mask_encoded_content_digest")
    _require_manifest_sha256(manifest["mask_decoded_content_digest"], "manifest.mask_decoded_content_digest")
    _require_manifest_int_list(manifest["observed_label_set"], "manifest.observed_label_set")
    label_histogram = _require_manifest_int_list(manifest["label_histogram_v21"], "manifest.label_histogram_v21")
    if len(label_histogram) != CLASS_COUNT_V21 or any(v < 0 for v in label_histogram):
        raise Voc2012DatasetIdentityError(f"manifest.label_histogram_v21 must be a list of {CLASS_COUNT_V21} non-negative exact integers")
    _require_manifest_int(manifest["ignore_pixel_count"], "manifest.ignore_pixel_count", minimum=0)
    dimension_reconciliation = manifest["dimension_reconciliation"]
    if not isinstance(dimension_reconciliation, dict) or set(dimension_reconciliation) != {"all_agree", "mismatches"}:
        raise Voc2012DatasetIdentityError("manifest.dimension_reconciliation must be an exact mapping with keys {'all_agree', 'mismatches'}")
    _require_manifest_bool(dimension_reconciliation["all_agree"], "manifest.dimension_reconciliation.all_agree")
    if type(dimension_reconciliation["mismatches"]) is not list:
        raise Voc2012DatasetIdentityError("manifest.dimension_reconciliation.mismatches must be an exact list")
    _require_manifest_bool(manifest["v20_v21_shared_source_assertion"], "manifest.v20_v21_shared_source_assertion")
    _require_manifest_timestamp(manifest["generated_at_utc"], "manifest.generated_at_utc")

    # Relational/value checks, now that every field's exact type is proven.
    if manifest["schema"] != identity["manifest"]["schema_name"]:
        raise Voc2012DatasetIdentityError("manifest.schema disagrees with identity.manifest.schema_name")
    if manifest["identity_name"] != identity["identity"]["name"]:
        raise Voc2012DatasetIdentityError("manifest.identity_name disagrees with identity.identity.name")
    if manifest["identity_sha256"] != identity_sha256:
        raise Voc2012DatasetIdentityError("manifest.identity_sha256 does not match the loaded identity file")
    if manifest["source_root_logical_label"] != identity["expected_root_structure"]["voc2012_relative_root"]:
        raise Voc2012DatasetIdentityError("manifest.source_root_logical_label disagrees with the identity's declared logical root label")
    if "/" in str(manifest["source_root_logical_label"]).replace("VOC2012", "") and manifest["source_root_logical_label"] != "VOC2012":
        raise Voc2012DatasetIdentityError("manifest.source_root_logical_label must not embed a filesystem path")
    if manifest["split"] != identity["protocol"]["split"]:
        raise Voc2012DatasetIdentityError("manifest.split disagrees with identity.protocol.split")
    if manifest["image_count"] != identity["protocol"]["expected_image_count"]:
        raise Voc2012DatasetIdentityError("manifest.image_count disagrees with identity.protocol.expected_image_count")
    if manifest["v21_class_count"] != identity["class_contract"]["v21_class_count"]:
        raise Voc2012DatasetIdentityError("manifest.v21_class_count disagrees with identity.class_contract.v21_class_count")
    if manifest["v20_class_count"] != identity["class_contract"]["v20_class_count"]:
        raise Voc2012DatasetIdentityError("manifest.v20_class_count disagrees with identity.class_contract.v20_class_count")
    if manifest["v20_v21_shared_source_assertion"] is not True:
        raise Voc2012DatasetIdentityError("manifest.v20_v21_shared_source_assertion must be true")
    if not (set(manifest["observed_label_set"]) <= LEGAL_RAW_LABELS):
        raise Voc2012DatasetIdentityError("manifest.observed_label_set contains an illegal label value")
    if dimension_reconciliation["all_agree"] is not True or dimension_reconciliation["mismatches"]:
        raise Voc2012DatasetIdentityError("manifest.dimension_reconciliation reports a dimension disagreement")


__all__ = [
    "CLASS_COUNT_V21",
    "LEGAL_RAW_LABELS",
    "MANIFEST_TOP_KEYS",
    "build_manifest",
    "canonical_validation_ids",
    "image_order_digest",
    "resolve_dataset_root",
    "scan_validation_split",
    "sha256_file",
    "verify_manifest_against_identity",
]
