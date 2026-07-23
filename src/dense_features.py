"""Streaming storage, validation, and loading for E5 dense features."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1
REQUIRED_RECORD_KEYS = {
    "image_id",
    "file_name",
    "disentangled_self_attn",
    "patch_tokens",
    "self_attn_maps",
    "captions",
    "ann_feats",
    "annotation_ids",
}
REQUIRED_MANIFEST_KEYS = {
    "source_images",
    "source_annotations",
    "selected_images",
    "selected_annotations",
    "complete",
    "failed_image_ids",
}


class DenseFeatureValidationError(ValueError):
    """Raised when a dense-feature shard or manifest is invalid."""


class IncompleteDenseFeatureExtraction(RuntimeError):
    """Raised after persisting an extraction that does not have exact coverage."""


def _id_set_digest(values: Iterable[Any]) -> str:
    encoded = sorted(
        json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values
    )
    digest = hashlib.sha256()
    for value in encoded:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_json_dump(value: Dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_record(file_object: Any) -> Dict[str, Any]:
    try:
        return torch.load(file_object, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise DenseFeatureValidationError(f"unable to decode record: {exc}") from exc


def _iter_dense_shard_records(
    path: os.PathLike[str] | str, *, allow_partial: bool
) -> Iterator[Dict[str, Any]]:
    shard_path = Path(path)
    if shard_path.name.endswith(".tmp") and not allow_partial:
        raise DenseFeatureValidationError(f"partial shard is not readable: {shard_path}")
    try:
        archive = tarfile.open(shard_path, mode="r")
    except (OSError, tarfile.TarError) as exc:
        raise DenseFeatureValidationError(f"unreadable archive {shard_path}: {exc}") from exc
    with archive:
        for member in archive:
            if not member.isfile():
                continue
            if not member.name.endswith(".pth"):
                raise DenseFeatureValidationError(
                    f"unexpected tar member {member.name!r} in {shard_path}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise DenseFeatureValidationError(
                    f"unable to read tar member {member.name!r} in {shard_path}"
                )
            yield _load_record(extracted)


def iter_dense_shard_records(path: os.PathLike[str] | str) -> Iterator[Dict[str, Any]]:
    """Yield one image record at a time without extracting the tar archive."""
    yield from _iter_dense_shard_records(path, allow_partial=False)


def _validate_record(record: Dict[str, Any], source: str) -> Dict[str, float | int]:
    if not isinstance(record, dict):
        raise DenseFeatureValidationError(f"{source}: record must be a dictionary")
    missing = REQUIRED_RECORD_KEYS.difference(record)
    if missing:
        raise DenseFeatureValidationError(f"{source}: missing keys {sorted(missing)}")

    captions = record["captions"]
    ann_feats = record["ann_feats"]
    annotation_ids = record["annotation_ids"]
    if not isinstance(captions, list) or not isinstance(ann_feats, list) or not isinstance(annotation_ids, list):
        raise DenseFeatureValidationError(
            f"{source}: captions, ann_feats, and annotation_ids must be lists"
        )
    if not captions or not (len(captions) == len(ann_feats) == len(annotation_ids)):
        raise DenseFeatureValidationError(
            f"{source}: caption/feature/annotation alignment is invalid"
        )
    if len(set(annotation_ids)) != len(annotation_ids):
        raise DenseFeatureValidationError(f"{source}: duplicate annotation IDs")
    if any(not isinstance(caption, str) for caption in captions):
        raise DenseFeatureValidationError(f"{source}: every caption must be a string")
    if any(not torch.is_tensor(feature) for feature in ann_feats):
        raise DenseFeatureValidationError(f"{source}: every ann_feats entry must be a tensor")
    for feature in ann_feats:
        if not torch.isfinite(feature).all():
            raise DenseFeatureValidationError(f"{source}: non-finite ann_feats")

    heads = record["disentangled_self_attn"]
    patches = record["patch_tokens"]
    maps = record["self_attn_maps"]
    if not all(torch.is_tensor(tensor) for tensor in (heads, patches, maps)):
        raise DenseFeatureValidationError(f"{source}: dense fields must be tensors")
    if heads.ndim != 2 or patches.ndim != 2 or maps.ndim != 2:
        raise DenseFeatureValidationError(f"{source}: dense fields must all be rank two")
    if heads.shape[0] != maps.shape[0]:
        raise DenseFeatureValidationError(f"{source}: attention-head counts do not match")
    if patches.shape[0] != maps.shape[1]:
        raise DenseFeatureValidationError(f"{source}: patch/map token counts do not match")
    if heads.shape[1] != patches.shape[1]:
        raise DenseFeatureValidationError(f"{source}: embedding dimensions do not match")
    if tuple(heads.shape) != (12, 768):
        raise DenseFeatureValidationError(
            f"{source}: expected disentangled_self_attn [12,768], got {list(heads.shape)}"
        )
    if tuple(patches.shape) != (1024, 768):
        raise DenseFeatureValidationError(
            f"{source}: expected patch_tokens [1024,768], got {list(patches.shape)}"
        )
    if tuple(maps.shape) != (12, 1024):
        raise DenseFeatureValidationError(
            f"{source}: expected self_attn_maps [12,1024], got {list(maps.shape)}"
        )
    if heads.dtype != torch.float32:
        raise DenseFeatureValidationError(f"{source}: disentangled_self_attn must be float32")
    if patches.dtype != torch.float16 or maps.dtype != torch.float16:
        raise DenseFeatureValidationError(
            f"{source}: patch_tokens and self_attn_maps must be float16"
        )
    for name, tensor in (("disentangled_self_attn", heads), ("patch_tokens", patches), ("self_attn_maps", maps)):
        if not torch.isfinite(tensor).all():
            raise DenseFeatureValidationError(f"{source}: non-finite {name}")
    if (maps < 0).any():
        raise DenseFeatureValidationError(f"{source}: negative attention probabilities")
    row_sums = maps.float().sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-3, rtol=2e-3):
        raise DenseFeatureValidationError(
            f"{source}: attention rows do not sum to one; "
            f"range=[{row_sums.min().item():.6g}, {row_sums.max().item():.6g}]"
        )

    reconstructed = (
        patches.float().unsqueeze(0) * maps.float().unsqueeze(-1)
    ).mean(dim=1)
    cosine = F.cosine_similarity(
        F.normalize(reconstructed, dim=-1),
        F.normalize(heads.float(), dim=-1),
        dim=-1,
    )
    if not torch.isfinite(cosine).all():
        raise DenseFeatureValidationError(f"{source}: non-finite reconstruction cosine")
    return {
        "annotations": len(annotation_ids),
        "cosine_sum": cosine.sum().item(),
        "cosine_count": cosine.numel(),
        "cosine_min": cosine.min().item(),
        "cosine_max": cosine.max().item(),
        "row_sum_sum": row_sums.sum().item(),
        "row_sum_count": row_sums.numel(),
        "row_sum_min": row_sums.min().item(),
        "row_sum_max": row_sums.max().item(),
        "reconstruction_failed": cosine.min().item() < 0.99,
    }


def _validate_archive(
    path: Path,
    expected_images: Optional[int] = None,
    *,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    image_ids = set()
    annotation_ids = set()
    cosine_sum = 0.0
    cosine_count = 0
    cosine_min = float("inf")
    cosine_max = float("-inf")
    row_sum_sum = 0.0
    row_sum_count = 0
    row_sum_min = float("inf")
    row_sum_max = float("-inf")
    annotations = 0
    images = 0
    failed_images: List[Any] = []

    for index, record in enumerate(
        _iter_dense_shard_records(path, allow_partial=allow_partial)
    ):
        image_id = record.get("image_id") if isinstance(record, dict) else None
        source = f"{path.name}:record-{index}:image-{image_id}"
        try:
            stats = _validate_record(record, source)
        except DenseFeatureValidationError:
            failed_images.append(image_id)
            raise
        if image_id in image_ids:
            raise DenseFeatureValidationError(f"{path}: duplicate image ID {image_id!r}")
        image_ids.add(image_id)
        overlap = annotation_ids.intersection(record["annotation_ids"])
        if overlap:
            raise DenseFeatureValidationError(
                f"{path}: annotation IDs appear in multiple images: {sorted(overlap)!r}"
            )
        annotation_ids.update(record["annotation_ids"])
        images += 1
        annotations += int(stats["annotations"])
        cosine_sum += float(stats["cosine_sum"])
        cosine_count += int(stats["cosine_count"])
        cosine_min = min(cosine_min, float(stats["cosine_min"]))
        cosine_max = max(cosine_max, float(stats["cosine_max"]))
        row_sum_sum += float(stats["row_sum_sum"])
        row_sum_count += int(stats["row_sum_count"])
        row_sum_min = min(row_sum_min, float(stats["row_sum_min"]))
        row_sum_max = max(row_sum_max, float(stats["row_sum_max"]))
        if bool(stats["reconstruction_failed"]):
            failed_images.append(image_id)

    if images == 0:
        raise DenseFeatureValidationError(f"{path}: shard contains no image records")
    if expected_images is not None and images != expected_images:
        raise DenseFeatureValidationError(
            f"{path}: expected {expected_images} images, found {images}"
        )
    return {
        "path": str(path),
        "images": images,
        "annotations": annotations,
        "image_ids": list(image_ids),
        "annotation_ids": list(annotation_ids),
        "cosine_mean": cosine_sum / cosine_count,
        "cosine_min": cosine_min,
        "cosine_max": cosine_max,
        "row_sum_mean": row_sum_sum / row_sum_count,
        "row_sum_min": row_sum_min,
        "row_sum_max": row_sum_max,
        "failed_images": failed_images,
    }


def validate_dense_shard(
    path: os.PathLike[str] | str, expected_images: Optional[int] = None
) -> Dict[str, Any]:
    """Validate a finalized shard and return numerical summary statistics."""
    shard_path = Path(path)
    if shard_path.name.endswith(".tmp"):
        raise DenseFeatureValidationError(f"partial shard is not valid: {shard_path}")
    if shard_path.suffix != ".tar":
        raise DenseFeatureValidationError(f"expected a finalized .tar shard: {shard_path}")
    return _validate_archive(shard_path, expected_images)


def _load_manifest(path: os.PathLike[str] | str) -> Dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_NAME
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseFeatureValidationError(f"invalid manifest {manifest_path}: {exc}") from exc
    if manifest.get("format_version") != FORMAT_VERSION:
        raise DenseFeatureValidationError(
            f"unsupported manifest format version: {manifest.get('format_version')!r}"
        )
    if not isinstance(manifest.get("shards"), list):
        raise DenseFeatureValidationError("manifest shards must be a list")
    missing = REQUIRED_MANIFEST_KEYS.difference(manifest)
    if missing:
        raise DenseFeatureValidationError(
            f"manifest is missing completeness fields: {sorted(missing)}"
        )
    if not isinstance(manifest["complete"], bool):
        raise DenseFeatureValidationError("manifest complete must be a boolean")
    if not isinstance(manifest["failed_image_ids"], list):
        raise DenseFeatureValidationError("manifest failed_image_ids must be a list")
    return manifest


def validate_dense_dataset(
    path: os.PathLike[str] | str, *, require_complete: bool = False
) -> Dict[str, Any]:
    """Validate every completed shard referenced by a manifest."""
    directory = Path(path)
    manifest = _load_manifest(directory)
    seen_images = set()
    seen_annotations = set()
    shard_results = []
    weighted_cosine = 0.0
    cosine_heads = 0
    cosine_min = float("inf")
    cosine_max = float("-inf")
    row_sum_weighted = 0.0
    row_sum_rows = 0
    row_sum_min = float("inf")
    row_sum_max = float("-inf")
    for shard in manifest["shards"]:
        shard_path = directory / shard["name"]
        result = validate_dense_shard(shard_path, expected_images=shard["images"])
        if shard.get("bytes") != shard_path.stat().st_size:
            raise DenseFeatureValidationError(f"size mismatch for {shard_path}")
        if shard.get("sha256") != _sha256(shard_path):
            raise DenseFeatureValidationError(f"checksum mismatch for {shard_path}")
        image_overlap = seen_images.intersection(result["image_ids"])
        annotation_overlap = seen_annotations.intersection(result["annotation_ids"])
        if image_overlap or annotation_overlap:
            raise DenseFeatureValidationError(
                f"duplicate IDs across shards: images={sorted(image_overlap)!r}, "
                f"annotations={sorted(annotation_overlap)!r}"
            )
        seen_images.update(result["image_ids"])
        seen_annotations.update(result["annotation_ids"])
        head_count = result["images"] * 12
        row_count = result["images"] * 12
        weighted_cosine += result["cosine_mean"] * head_count
        cosine_heads += head_count
        cosine_min = min(cosine_min, result["cosine_min"])
        cosine_max = max(cosine_max, result["cosine_max"])
        row_sum_weighted += result["row_sum_mean"] * row_count
        row_sum_rows += row_count
        row_sum_min = min(row_sum_min, result["row_sum_min"])
        row_sum_max = max(row_sum_max, result["row_sum_max"])
        shard_results.append(result)

    images = sum(result["images"] for result in shard_results)
    annotations = sum(result["annotations"] for result in shard_results)
    if images != manifest.get("images") or annotations != manifest.get("annotations"):
        raise DenseFeatureValidationError(
            "manifest totals do not agree with the completed shards"
        )
    image_coverage_matches = (
        images == manifest["selected_images"]
        and _id_set_digest(seen_images) == manifest.get("selected_image_ids_sha256")
    )
    annotation_coverage_matches = (
        annotations == manifest["selected_annotations"]
        and _id_set_digest(seen_annotations)
        == manifest.get("selected_annotation_ids_sha256")
    )
    coverage_complete = (
        image_coverage_matches
        and annotation_coverage_matches
        and not manifest["failed_image_ids"]
    )
    if manifest["complete"] and not coverage_complete:
        raise DenseFeatureValidationError(
            "manifest declares complete=true but exact selected ID coverage does not match"
        )
    is_pilot = manifest.get("is_pilot", False)
    if require_complete and not (
        manifest["complete"] and coverage_complete and not is_pilot
    ):
        raise DenseFeatureValidationError(
            "dense-feature extraction is incomplete or is a pilot: "
            f"images={images}/{manifest['selected_images']}, "
            f"annotations={annotations}/{manifest['selected_annotations']}, "
            f"failed_image_ids={manifest['failed_image_ids']!r}, "
            f"is_pilot={is_pilot}"
        )
    return {
        "images": images,
        "annotations": annotations,
        "shards": shard_results,
        "cosine_mean": weighted_cosine / cosine_heads if cosine_heads else None,
        "cosine_min": cosine_min if cosine_heads else None,
        "cosine_max": cosine_max if cosine_heads else None,
        "row_sum_mean": row_sum_weighted / row_sum_rows if row_sum_rows else None,
        "row_sum_min": row_sum_min if row_sum_rows else None,
        "row_sum_max": row_sum_max if row_sum_rows else None,
        "failed_images": [
            image_id
            for result in shard_results
            for image_id in result["failed_images"]
        ],
        "source_images": manifest["source_images"],
        "source_annotations": manifest["source_annotations"],
        "selected_images": manifest["selected_images"],
        "selected_annotations": manifest["selected_annotations"],
        "complete": manifest["complete"],
        "coverage_complete": coverage_complete,
        "failed_image_ids": manifest["failed_image_ids"],
        "max_images": manifest.get("max_images"),
        "is_pilot": is_pilot,
    }


class DenseFeatureShardWriter:
    """Continuously write one dense image record per tar member."""

    def __init__(
        self,
        output_dir: os.PathLike[str] | str,
        split: str,
        extraction_config: Dict[str, Any],
        source_commit: str,
        *,
        source_images: int,
        source_annotations: int,
        selected_image_ids: Iterable[Any],
        selected_annotation_ids: Iterable[Any],
        max_images: Optional[int],
        images_per_shard: int = 128,
        overwrite: bool = False,
    ) -> None:
        if images_per_shard <= 0:
            raise ValueError("images_per_shard must be greater than zero")
        if not split or any(character in split for character in "/\\"):
            raise ValueError("split must be a simple non-empty name")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.images_per_shard = images_per_shard
        self.extraction_config = extraction_config
        self.source_commit = source_commit
        self.source_images = source_images
        self.source_annotations = source_annotations
        selected_image_ids = list(selected_image_ids)
        selected_annotation_ids = list(selected_annotation_ids)
        self.expected_image_ids = set(selected_image_ids)
        self.expected_annotation_ids = set(selected_annotation_ids)
        if len(self.expected_image_ids) != len(selected_image_ids):
            raise ValueError("selected_image_ids contains duplicates")
        if len(self.expected_annotation_ids) != len(selected_annotation_ids):
            raise ValueError("selected_annotation_ids contains duplicates")
        if len(self.expected_image_ids) > source_images:
            raise ValueError("selected image count cannot exceed source_images")
        if len(self.expected_annotation_ids) > source_annotations:
            raise ValueError("selected annotation count cannot exceed source_annotations")
        self.max_images = max_images
        self.manifest_path = self.output_dir / MANIFEST_NAME
        self._archive: Optional[tarfile.TarFile] = None
        self._temporary_path: Optional[Path] = None
        self._current_records = 0
        self._current_annotations = 0

        existing_shards = sorted(self.output_dir.glob(f"{split}-*.tar"))
        if overwrite:
            for path in existing_shards:
                path.unlink()
            for path in self.output_dir.glob(f"{split}-*.tar.tmp"):
                path.unlink()
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            existing_shards = []

        if self.manifest_path.exists():
            self.manifest = _load_manifest(self.manifest_path)
            expected_identity = {
                "split": split,
                "images_per_shard": images_per_shard,
                "extraction_config": extraction_config,
                "source_commit": source_commit,
                "source_images": source_images,
                "source_annotations": source_annotations,
                "selected_images": len(self.expected_image_ids),
                "selected_annotations": len(self.expected_annotation_ids),
                "selected_image_ids_sha256": _id_set_digest(self.expected_image_ids),
                "selected_annotation_ids_sha256": _id_set_digest(
                    self.expected_annotation_ids
                ),
                "max_images": max_images,
            }
            actual_identity = {key: self.manifest.get(key) for key in expected_identity}
            if actual_identity != expected_identity:
                raise FileExistsError(
                    "completed output has a different extraction configuration; "
                    "pass overwrite=True only if replacement is intentional"
                )
            if self.manifest["complete"]:
                raise FileExistsError(
                    "completed extraction is immutable; pass overwrite=True only if "
                    "replacement is intentional"
                )
            referenced_shards = {shard["name"] for shard in self.manifest["shards"]}
            orphan_shards = [
                path for path in existing_shards if path.name not in referenced_shards
            ]
            for orphan_path in orphan_shards:
                expected_name = f"{split}-{len(self.manifest['shards']):06d}.tar"
                if orphan_path.name != expected_name:
                    raise FileExistsError(
                        f"unexpected completed shard not present in manifest: {orphan_path}"
                    )
                result = validate_dense_shard(orphan_path)
                self.manifest["shards"].append(
                    {
                        "name": orphan_path.name,
                        "images": result["images"],
                        "annotations": result["annotations"],
                        "bytes": orphan_path.stat().st_size,
                        "sha256": _sha256(orphan_path),
                    }
                )
                self.manifest["images"] += result["images"]
                self.manifest["annotations"] += result["annotations"]
                _atomic_json_dump(self.manifest, self.manifest_path)
        elif existing_shards:
            raise FileExistsError(
                "completed shards exist without a manifest; refusing to overwrite them"
            )
        else:
            self.manifest = {
                "format_version": FORMAT_VERSION,
                "split": split,
                "images_per_shard": images_per_shard,
                "extraction_config": extraction_config,
                "source_commit": source_commit,
                "source_images": source_images,
                "source_annotations": source_annotations,
                "selected_images": len(self.expected_image_ids),
                "selected_annotations": len(self.expected_annotation_ids),
                "selected_image_ids_sha256": _id_set_digest(self.expected_image_ids),
                "selected_annotation_ids_sha256": _id_set_digest(
                    self.expected_annotation_ids
                ),
                "max_images": max_images,
                "is_pilot": max_images is not None,
                "complete": False,
                "failed_image_ids": [],
                "images": 0,
                "annotations": 0,
                "shards": [],
            }
            _atomic_json_dump(self.manifest, self.manifest_path)

        self.completed_image_ids = set()
        self.completed_annotation_ids = set()
        for shard in self.manifest["shards"]:
            shard_path = self.output_dir / shard["name"]
            result = validate_dense_shard(shard_path, expected_images=shard["images"])
            if shard.get("bytes") != shard_path.stat().st_size:
                raise DenseFeatureValidationError(f"size mismatch for {shard_path}")
            if shard.get("sha256") != _sha256(shard_path):
                raise DenseFeatureValidationError(f"checksum mismatch for {shard_path}")
            overlap = self.completed_image_ids.intersection(result["image_ids"])
            if overlap:
                raise DenseFeatureValidationError(
                    f"duplicate completed image IDs: {sorted(overlap)!r}"
                )
            self.completed_image_ids.update(result["image_ids"])
            annotation_overlap = self.completed_annotation_ids.intersection(
                result["annotation_ids"]
            )
            if annotation_overlap:
                raise DenseFeatureValidationError(
                    f"duplicate completed annotation IDs: {sorted(annotation_overlap)!r}"
                )
            self.completed_annotation_ids.update(result["annotation_ids"])
        if len(self.completed_image_ids) != self.manifest["images"]:
            raise DenseFeatureValidationError("manifest image total is invalid")
        if len(self.completed_annotation_ids) != self.manifest["annotations"]:
            raise DenseFeatureValidationError("manifest annotation total is invalid")

        self._next_shard_index = len(self.manifest["shards"])
        next_final = self.output_dir / f"{split}-{self._next_shard_index:06d}.tar"
        if next_final.exists():
            raise FileExistsError(f"refusing to overwrite completed shard {next_final}")
        next_temporary = next_final.with_name(next_final.name + ".tmp")
        if next_temporary.exists():
            next_temporary.unlink()

    def __enter__(self) -> "DenseFeatureShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _open_shard(self) -> None:
        final_path = self.output_dir / f"{self.split}-{self._next_shard_index:06d}.tar"
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite completed shard {final_path}")
        self._temporary_path = final_path.with_name(final_path.name + ".tmp")
        if self._temporary_path.exists():
            self._temporary_path.unlink()
        self._archive = tarfile.open(self._temporary_path, mode="w")

    def add(self, record: Dict[str, Any]) -> bool:
        """Add an image. Return False when that verified image is already complete."""
        image_id = record.get("image_id")
        if image_id in self.completed_image_ids:
            return False
        if image_id is None:
            raise DenseFeatureValidationError("record has no image_id")
        if image_id not in self.expected_image_ids:
            raise DenseFeatureValidationError(
                f"image ID {image_id!r} is outside the selected extraction"
            )
        _validate_record(record, f"image-{image_id}")
        annotation_ids = record["annotation_ids"]
        unexpected_annotations = set(annotation_ids).difference(
            self.expected_annotation_ids
        )
        if unexpected_annotations:
            raise DenseFeatureValidationError(
                "annotation IDs are outside the selected extraction: "
                f"{sorted(unexpected_annotations, key=str)!r}"
            )
        duplicate_annotations = self.completed_annotation_ids.intersection(annotation_ids)
        if duplicate_annotations:
            raise DenseFeatureValidationError(
                f"annotation IDs already written: {sorted(duplicate_annotations)!r}"
            )
        if self._archive is None:
            self._open_shard()
        buffer = io.BytesIO()
        torch.save(record, buffer)
        payload = buffer.getvalue()
        member = tarfile.TarInfo(name=f"{self._current_records:08d}.pth")
        member.size = len(payload)
        self._archive.addfile(member, io.BytesIO(payload))
        self._current_records += 1
        self._current_annotations += len(annotation_ids)
        self.completed_image_ids.add(image_id)
        self.completed_annotation_ids.update(annotation_ids)
        if self._current_records == self.images_per_shard:
            self._finalize_shard()
        return True

    def _finalize_shard(self) -> None:
        if self._archive is None or self._temporary_path is None:
            return
        self._archive.close()
        self._archive = None
        # Internally validate the closed temporary archive before it becomes visible.
        result = _validate_archive(
            self._temporary_path, self._current_records, allow_partial=True
        )
        if result["annotations"] != self._current_annotations:
            raise DenseFeatureValidationError("annotation count changed during serialization")
        if result["failed_images"]:
            raise DenseFeatureValidationError(
                "stored heads do not match reconstructed heads for image IDs "
                f"{result['failed_images']!r}"
            )
        final_path = self._temporary_path.with_name(
            self._temporary_path.name.removesuffix(".tmp")
        )
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite completed shard {final_path}")
        os.replace(self._temporary_path, final_path)
        shard_entry = {
            "name": final_path.name,
            "images": self._current_records,
            "annotations": self._current_annotations,
            "bytes": final_path.stat().st_size,
            "sha256": _sha256(final_path),
        }
        self.manifest["shards"].append(shard_entry)
        self.manifest["images"] += self._current_records
        self.manifest["annotations"] += self._current_annotations
        _atomic_json_dump(self.manifest, self.manifest_path)
        self._next_shard_index += 1
        self._temporary_path = None
        self._current_records = 0
        self._current_annotations = 0

    def close(self) -> None:
        if self._archive is not None:
            self._finalize_shard()

    def record_failure(self, image_id: Any) -> None:
        """Persist a failed selected image without discarding earlier failures."""
        if image_id not in self.expected_image_ids:
            raise DenseFeatureValidationError(
                f"failed image ID {image_id!r} is outside the selected extraction"
            )
        failed = set(self.manifest["failed_image_ids"])
        failed.add(image_id)
        self.manifest["failed_image_ids"] = sorted(failed, key=str)
        self.manifest["complete"] = False
        _atomic_json_dump(self.manifest, self.manifest_path)

    def finish(self) -> None:
        """Finalize pending data and mark complete only after exact ID-set checks."""
        self.close()
        missing_images = self.expected_image_ids.difference(self.completed_image_ids)
        unexpected_images = self.completed_image_ids.difference(self.expected_image_ids)
        missing_annotations = self.expected_annotation_ids.difference(
            self.completed_annotation_ids
        )
        unexpected_annotations = self.completed_annotation_ids.difference(
            self.expected_annotation_ids
        )
        unresolved_failures = set(self.manifest["failed_image_ids"]).difference(
            self.completed_image_ids
        )
        self.manifest["failed_image_ids"] = sorted(unresolved_failures, key=str)
        self.manifest["complete"] = not any(
            (
                missing_images,
                unexpected_images,
                missing_annotations,
                unexpected_annotations,
                unresolved_failures,
            )
        )
        _atomic_json_dump(self.manifest, self.manifest_path)
        if not self.manifest["complete"]:
            raise IncompleteDenseFeatureExtraction(
                "dense-feature extraction is incomplete: "
                f"missing images={len(missing_images)}, "
                f"unexpected images={len(unexpected_images)}, "
                f"missing annotations={len(missing_annotations)}, "
                f"unexpected annotations={len(unexpected_annotations)}, "
                f"failed_image_ids={self.manifest['failed_image_ids']!r}"
            )

    def abort(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        # Keep the .tmp marker so an interrupted shard can never look complete.


class DenseFeatureStreamingDataset(IterableDataset):
    """Lazy, annotation-level view over image-level E5 shards."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        shuffle_shards: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 0,
        allow_incomplete: bool = False,
        allow_pilot: bool = False,
    ) -> None:
        super().__init__()
        if shuffle_buffer < 0:
            raise ValueError("shuffle_buffer cannot be negative")
        self.root = Path(root)
        self.manifest = _load_manifest(self.root)
        if not self.manifest["complete"] and not allow_incomplete:
            raise DenseFeatureValidationError(
                "dense-feature extraction is incomplete; pass allow_incomplete=True "
                "only for explicit inspection or tests"
            )
        if self.manifest.get("is_pilot", False) and not allow_pilot:
            raise DenseFeatureValidationError(
                "dense-feature dataset is a pilot; pass allow_pilot=True only for "
                "explicit pilot inspection or tests"
            )
        self.shards = [self.root / shard["name"] for shard in self.manifest["shards"]]
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return int(self.manifest["annotations"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)

    def _shards_for_worker(self) -> List[Path]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        shards = list(self.shards)
        if self.shuffle_shards:
            rng.shuffle(shards)
        return shards[worker_id::worker_count]

    def _expanded_samples(self, shards: Iterable[Path]) -> Iterator[Dict[str, Any]]:
        for shard in shards:
            for record in iter_dense_shard_records(shard):
                for caption, ann_feat, annotation_id in zip(
                    record["captions"], record["ann_feats"], record["annotation_ids"]
                ):
                    yield {
                        "annotation": ann_feat,
                        "image": record["disentangled_self_attn"],
                        "metadata": {
                            "image_id": record["image_id"],
                            "annotation_id": annotation_id,
                        },
                        "caption": caption,
                        "patch_tokens": record["patch_tokens"],
                        "self_attn_maps": record["self_attn_maps"],
                    }

    @staticmethod
    def _bounded_shuffle(
        samples: Iterable[Dict[str, Any]], buffer_size: int, rng: random.Random
    ) -> Iterator[Dict[str, Any]]:
        if buffer_size <= 1:
            yield from samples
            return
        buffer: List[Dict[str, Any]] = []
        for sample in samples:
            if len(buffer) < buffer_size:
                buffer.append(sample)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = sample
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        shards = self._shards_for_worker()
        samples = self._expanded_samples(shards)
        yield from self._bounded_shuffle(samples, self.shuffle_buffer, rng)
