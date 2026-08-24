"""Core, CPU-only mask-materialization logic for the COCO-Object val2017
protocol-confirmation prerequisite.

Reuses the repository's own already-committed conversion authority
verbatim -- ``clsID_to_trID`` is imported directly from
``convert_dataset/convert_coco_object.py`` (hash-checked against the
identity before use), never re-derived or hand-copied. This module adds
only what that authority lacks for safe, resumable, verified batch
production: a validated 256-entry lookup table (fail-closed on any raw
pixel value the table does not cover -- the original script silently
passes such values through unchanged; on the real val2017 data every raw
value is covered, so this is a safety net, never an observed behavior
change), atomic same-directory temp-file writes, and per-image
provenance hashing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.coco_object_val_materialization_identity import CocoObjectValMaterializationIdentityError

CLASS_COUNT = 81
RAW_VALUE_DOMAIN = 256


class CocoObjectValMaterializationError(RuntimeError):
    """Raised for every expected, data-preparation-time failure: a missing
    or unmapped source pixel value, a source/output overlap, a corrupt
    temporary file, a checkpoint that disagrees with the canonical order,
    etc. Always fail closed -- never silently skip or approximate."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_canonical_mapping(root: Path, identity: dict[str, Any]) -> dict[int, int]:
    """Hash-verify and import ``clsID_to_trID`` directly from the real,
    already-committed converter file named by the identity. Never
    re-declares the table as a Python literal in this module."""
    converter = identity["converter"]
    path = root / converter["canonical_converter_relative_path"]
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise CocoObjectValMaterializationError(f"cannot read canonical converter {path}: {error}") from error
    observed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if observed_sha256 != converter["canonical_converter_sha256"]:
        raise CocoObjectValMaterializationError(
            f"canonical converter {path} SHA256 mismatch: identity declares "
            f"{converter['canonical_converter_sha256']}, observed {observed_sha256}"
        )

    spec = importlib.util.spec_from_file_location("_canonical_convert_coco_object", path)
    if spec is None or spec.loader is None:
        raise CocoObjectValMaterializationError(f"cannot construct an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mapping = getattr(module, "clsID_to_trID", None)
    if not isinstance(mapping, dict) or not mapping:
        raise CocoObjectValMaterializationError(f"{path} does not define a non-empty clsID_to_trID dict")

    mapping_sorted = {str(k): v for k, v in sorted(mapping.items())}
    mapping_digest = hashlib.sha256(json.dumps(mapping_sorted, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()
    if mapping_digest != identity["label_mapping"]["mapping_table_sha256"]:
        raise CocoObjectValMaterializationError(
            "imported clsID_to_trID digest disagrees with identity.label_mapping.mapping_table_sha256 -- "
            "refusing to use a mapping table that does not match the recorded provenance"
        )
    if len(mapping) != identity["label_mapping"]["mapping_table_entry_count"]:
        raise CocoObjectValMaterializationError("imported clsID_to_trID entry count disagrees with the identity")

    return {int(k): int(v) for k, v in mapping.items()}


def build_lookup_table(mapping: dict[int, int]) -> np.ndarray:
    """A 256-entry int16 LUT: ``lut[raw_value] = mapped_value`` for every
    raw value the canonical table covers, ``-1`` (an explicit sentinel,
    never silently passed through) everywhere else."""
    lut = np.full(RAW_VALUE_DOMAIN, -1, dtype=np.int16)
    for raw_value, mapped_value in mapping.items():
        if not (0 <= raw_value < RAW_VALUE_DOMAIN):
            raise CocoObjectValMaterializationError(f"mapping table contains an out-of-range raw value {raw_value}")
        if not (0 <= mapped_value < CLASS_COUNT):
            raise CocoObjectValMaterializationError(f"mapping table maps raw value {raw_value} to out-of-range class {mapped_value}")
        lut[raw_value] = mapped_value
    return lut


def apply_canonical_mapping(raw_mask: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Vectorized, fail-closed application of the canonical LUT. Raises if
    any pixel's raw value is not covered by the table -- the original
    converter would silently leave such a pixel unchanged instead."""
    if raw_mask.dtype != np.uint8 or raw_mask.ndim != 2:
        raise CocoObjectValMaterializationError(f"raw mask must be a 2-D uint8 array, got dtype={raw_mask.dtype} ndim={raw_mask.ndim}")
    mapped = lut[raw_mask]
    if (mapped < 0).any():
        bad_values = sorted(set(raw_mask[mapped < 0].tolist()))
        raise CocoObjectValMaterializationError(f"raw mask contains value(s) not covered by clsID_to_trID: {bad_values}")
    return mapped.astype(np.uint8)


def canonical_image_ids(source_images_dir: Path, *, image_suffix: str = ".jpg") -> list[str]:
    """The exact ordering ``mmseg.datasets.custom.CustomDataset`` derives
    for this dataset config (no ``split`` file configured): every file in
    ``img_dir`` matching ``image_suffix``, sorted by full filename. Since
    every COCO val2017 filename is a fixed-width zero-padded numeric
    string plus a common suffix, sorting by stem is equivalent."""
    if not source_images_dir.is_dir():
        raise CocoObjectValMaterializationError(f"source images directory does not exist: {source_images_dir}")
    ids = sorted(p.stem for p in source_images_dir.iterdir() if p.is_file() and p.name.endswith(image_suffix))
    return ids


def image_order_digest(image_ids: list[str]) -> str:
    payload = json.dumps(image_ids, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raw_mask_path(source_masks_dir: Path, image_id: str) -> Path:
    return source_masks_dir / f"{image_id}.png"


def output_mask_path(output_annotations_dir: Path, image_id: str, *, suffix: str) -> Path:
    return output_annotations_dir / f"{image_id}{suffix}"


def compute_label_histogram(mapped_mask: np.ndarray) -> list[int]:
    counts = np.bincount(mapped_mask.reshape(-1), minlength=CLASS_COUNT)
    if counts.shape[0] != CLASS_COUNT:
        raise CocoObjectValMaterializationError("mapped mask contains a class index outside the declared class contract")
    return counts.astype(np.int64).tolist()


def convert_one_image(
    image_id: str, *, source_masks_dir: Path, output_annotations_dir: Path, lut: np.ndarray, output_mask_suffix: str,
) -> dict[str, Any]:
    """Steps 1-9 of the atomic per-image conversion contract: read the
    raw mask, apply the canonical mapping, validate shape/dtype/label
    set, write to a same-directory temp file, reopen and validate the
    temporary PNG, atomically rename into place. Never marks an image
    complete before the final mask is validated and installed."""
    src_path = raw_mask_path(source_masks_dir, image_id)
    if not src_path.is_file():
        raise CocoObjectValMaterializationError(f"source raw mask does not exist: {src_path}")
    raw_bytes = src_path.read_bytes()
    raw_mask_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    raw_mask = np.array(Image.open(src_path))
    if raw_mask.ndim != 2:
        raise CocoObjectValMaterializationError(f"source raw mask {src_path} is not single-channel (shape={raw_mask.shape})")
    if raw_mask.dtype != np.uint8:
        raw_mask = raw_mask.astype(np.uint8)
    height, width = raw_mask.shape

    mapped_mask = apply_canonical_mapping(raw_mask, lut)
    if mapped_mask.shape != (height, width):
        raise CocoObjectValMaterializationError("mapped mask shape disagrees with the source raw mask shape")
    if mapped_mask.dtype != np.uint8:
        raise CocoObjectValMaterializationError("mapped mask must be uint8")
    observed_labels = set(np.unique(mapped_mask).tolist())
    if not observed_labels <= set(range(CLASS_COUNT)):
        raise CocoObjectValMaterializationError(f"mapped mask {image_id} contains out-of-contract label(s): {observed_labels - set(range(CLASS_COUNT))}")

    decoded_pixel_sha256 = hashlib.sha256(mapped_mask.tobytes()).hexdigest()

    final_path = output_mask_path(output_annotations_dir, image_id, suffix=output_mask_suffix)
    temp_path = final_path.with_name(final_path.name + f".tmp-{os.getpid()}")
    try:
        Image.fromarray(mapped_mask, mode="L").save(temp_path, "PNG")
        reread = np.array(Image.open(temp_path))
        if reread.dtype != np.uint8 or reread.shape != (height, width):
            raise CocoObjectValMaterializationError(f"temporary PNG for {image_id} failed reopen validation")
        if not np.array_equal(reread, mapped_mask):
            raise CocoObjectValMaterializationError(f"temporary PNG for {image_id} does not decode back to the mapped mask exactly")
        encoded_png_sha256 = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        os.replace(temp_path, final_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

    return {
        "image_id": image_id,
        "width": int(width),
        "height": int(height),
        "raw_mask_sha256": raw_mask_sha256,
        "decoded_pixel_sha256": decoded_pixel_sha256,
        "encoded_png_sha256": encoded_png_sha256,
        "label_histogram": compute_label_histogram(mapped_mask),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    histogram = [0] * CLASS_COUNT
    total_pixels = 0
    masks_with_foreground = 0
    all_background_masks = 0
    for record in records:
        row = record["label_histogram"]
        for index in range(CLASS_COUNT):
            histogram[index] += row[index]
        row_total = sum(row)
        total_pixels += row_total
        foreground_pixels = row_total - row[0]
        if foreground_pixels > 0:
            masks_with_foreground += 1
        else:
            all_background_masks += 1
    return {
        "aggregate_label_histogram": histogram,
        "total_pixels": total_pixels,
        "masks_with_foreground": masks_with_foreground,
        "all_background_masks": all_background_masks,
    }


def source_masks_digest(records: list[dict[str, Any]]) -> str:
    """Stands in for a single 'source JSON SHA-256' (this converter has no
    such file): a deterministic digest over every source raw mask's own
    content hash, keyed by canonical image ID. Ties a materialization run
    to the exact, immutable per-image source state it was produced from."""
    payload = json.dumps(
        {record["image_id"]: record["raw_mask_sha256"] for record in records}, ensure_ascii=True, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def aggregate_decoded_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [[record["image_id"], record["decoded_pixel_sha256"]] for record in records], ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def aggregate_encoded_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [[record["image_id"], record["encoded_png_sha256"]] for record in records], ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def validate_output_root_isolation(*, output_root: Path, source_masks_dir: Path, source_annotation_root_marker: str) -> None:
    """Fail closed if the resolved output root overlaps the source COCO-
    Stuff annotation tree, or resolves inside it."""
    resolved_output = output_root.resolve()
    resolved_source_masks = source_masks_dir.resolve()
    if resolved_output == resolved_source_masks or resolved_source_masks in resolved_output.parents or resolved_output in resolved_source_masks.parents:
        raise CocoObjectValMaterializationError(
            f"--output-root {resolved_output} overlaps the source masks directory {resolved_source_masks}"
        )
    if source_annotation_root_marker in str(resolved_output):
        raise CocoObjectValMaterializationError(
            f"--output-root {resolved_output} resolves inside the source COCO-Stuff annotation root "
            f"(matches marker {source_annotation_root_marker!r}); refusing to write there"
        )


__all__ = [
    "CLASS_COUNT",
    "CocoObjectValMaterializationError",
    "aggregate_decoded_digest",
    "aggregate_encoded_digest",
    "aggregate_records",
    "apply_canonical_mapping",
    "build_lookup_table",
    "canonical_image_ids",
    "compute_label_histogram",
    "convert_one_image",
    "image_order_digest",
    "load_canonical_mapping",
    "output_mask_path",
    "raw_mask_path",
    "sha256_file",
    "source_masks_digest",
    "validate_output_root_isolation",
    "write_json_atomically",
]
