"""Streaming spatial-bank storage and validation for E9.

E9 consumes only the documented image-level streaming records produced by the
E5 extraction pipeline.  It does not reuse any E5 objective or training code.
Legacy monolithic archives are intentionally rejected because they carry no
closed geometry/attention-format identity and cannot be streamed boundedly.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import stat
import tarfile
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from build_e7_training_bank import (
    _require_unchanged_git_provenance,
    source_git_provenance,
)
from src.e6_prototype_bank import sha256_file
from src.e6_prototype_bank import annotation_id_set_fingerprint


E9_SPATIAL_BANK_FORMAT = "talk2dino-e9-spatial-bank-v2"
E5_STREAMING_FORMAT = 1
MANIFEST_NAME = "manifest.json"
SOURCE_RECORD_KEYS = {
    "image_id",
    "file_name",
    "disentangled_self_attn",
    "patch_tokens",
    "self_attn_maps",
    "captions",
    "ann_feats",
    "annotation_ids",
}
SHARD_KEYS = {"patch_embeddings", "attention_priors", "image_ids"}
SHARD_METADATA_KEYS = {
    "name",
    "sha256",
    "bytes",
    "row_start",
    "row_end",
    "row_count",
}
IMAGE_INDEX_KEYS = {"image_id", "shard", "row"}
BUILDER_CONFIG_KEYS = {
    "shard_rows", "max_images", "source_selected_images",
    "source_selected_annotations", "source_max_images", "source_is_pilot",
    "source_complete",
}
DINO_IDENTITY_KEYS = {"model", "source_commit", "checkpoint_sha256"}
EXTRACTION_KEYS = {
    "annotation_path", "data_dir", "model", "resize_dim", "crop_dim",
    "patch_count", "embedding_dim", "attention_heads",
    "attention_map_format", "patch_tokens_dtype", "self_attn_maps_dtype",
    "disentangled_self_attn_dtype", "backbone_weights_sha256",
}
MANIFEST_KEYS = {
    "format_version",
    "split",
    "complete",
    "is_pilot",
    "production_eligible",
    "source_images",
    "source_annotations",
    "selected_images",
    "selected_annotations",
    "image_id_fingerprint",
    "dataset_identity",
    "source_feature_paths",
    "source_feature_sha256",
    "source_feature_format",
    "source_feature_schema",
    "dino_identity",
    "extraction",
    "geometry",
    "dtypes",
    "pooling_version",
    "attention_prior_version",
    "global_token_handling",
    "shard_count",
    "shards",
    "image_index",
    "source_git_commit",
    "source_git_dirty",
    "source_git_diff_sha256",
    "builder_config",
    "created_at",
}
DATASET_IDENTITY_KEYS = {
    "split",
    "source_image_count",
    "source_annotation_count",
    "selected_image_count",
    "selected_annotation_count",
    "image_id_fingerprint",
    "annotation_id_fingerprint",
    "annotation_to_image_fingerprint",
}
SUPPORTED_SOURCE_FEATURE_FORMATS = {"talk2dino-e5-streaming-dense-v1"}
CANONICAL_SOURCE_FEATURE_FORMAT = "talk2dino-e5-streaming-dense-v1"
CANONICAL_EXTRACTION = {
    "model": "dinov2_vitb14_reg",
    "resize_dim": 448,
    "crop_dim": 448,
    "patch_count": 1024,
    "embedding_dim": 768,
    "attention_heads": 12,
    "attention_map_format": "probabilities",
    "patch_tokens_dtype": "float16",
    "self_attn_maps_dtype": "float16",
    "disentangled_self_attn_dtype": "float32",
}
EXPECTED_GEOMETRY = {
    "patch_size": 14,
    "source_grid_height": 32,
    "source_grid_width": 32,
    "pooled_grid_height": 16,
    "pooled_grid_width": 16,
    "patch_embedding_dim": 768,
}
EXPECTED_DTYPES = {
    "source_patch_embeddings": "float16",
    "source_attention_maps": "float16",
    "serialized_patch_embeddings": "float16",
    "serialized_attention_priors": "float16",
    "serialized_image_ids": "int64",
    "compute": "float32",
}
POOLING_VERSION = "mean-2x2-float32-then-l2-v1"
ATTENTION_PRIOR_VERSION = "mean-head-probability-sum-2x2-renormalize-v1"
GLOBAL_TOKEN_HANDLING = (
    "source extractor selects CLS-query attention to patch tokens after "
    "excluding 1 CLS plus 4 register tokens"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_UTC_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00$"
)
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_UNSUPPORTED_RENAMEAT2_ERRNOS = {errno.EINVAL, errno.ENOSYS}
_PRIVATE_DIR_MODE = 0o700
_MAX_MANIFEST_BYTES = 256 * 1024 * 1024
_NOFOLLOW_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_NOFOLLOW_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


class E9SpatialBankValidationError(ValueError):
    """Raised when an E9 source or spatial bank violates its closed contract."""


class E9PublicationConflictError(FileExistsError):
    """Raised when another filesystem object wins E9 publication."""


class E9PublicationRecoveryError(RuntimeError):
    """Raised when publication cannot safely restore an earlier artifact."""


class E9AtomicOperationUnsupportedError(E9PublicationRecoveryError):
    """Raised when the filesystem/kernel lacks a required renameat2 primitive.

    Publication never falls back to an unsafe non-atomic path when this is
    raised: it fails before any existing destination is touched.
    """


def _closed(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value)
        raise E9SpatialBankValidationError(
            f"{label} closed schema mismatch: expected={sorted(keys)}, got={actual}"
        )
    return value


def _id_fingerprint(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in sorted(int(item) for item in values):
        digest.update(int(value).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _source_id_fingerprint(values: Iterable[int]) -> str:
    """Match the canonical JSON ID-set digest used by the E5 manifest."""

    encoded = sorted(
        json.dumps(int(value), sort_keys=True, separators=(",", ":"))
        for value in values
    )
    digest = hashlib.sha256()
    for value in encoded:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def annotation_to_image_fingerprint(
    annotation_ids: Iterable[int], image_ids: Iterable[int]
) -> str:
    """Fingerprint an exact annotation-to-image mapping independent of row order."""

    annotations = [int(value) for value in annotation_ids]
    images = [int(value) for value in image_ids]
    if len(annotations) != len(images):
        raise E9SpatialBankValidationError(
            "annotation/image mapping lengths do not match"
        )
    if len(set(annotations)) != len(annotations):
        raise E9SpatialBankValidationError(
            "annotation/image mapping contains duplicate annotation IDs"
        )
    digest = hashlib.sha256()
    for annotation_id, image_id in sorted(zip(annotations, images)):
        digest.update(annotation_id.to_bytes(8, "big", signed=True))
        digest.update(image_id.to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def dataset_identity_from_rows(
    *,
    split: str,
    image_ids: Iterable[int],
    annotation_ids: Iterable[int],
    source_image_count: int,
    source_annotation_count: int,
) -> dict[str, Any]:
    image_rows = [int(value) for value in image_ids]
    annotation_rows = [int(value) for value in annotation_ids]
    if split not in {"train", "val"}:
        raise E9SpatialBankValidationError("dataset identity split must be train or val")
    if len(image_rows) != len(annotation_rows):
        raise E9SpatialBankValidationError("dataset identity row lengths differ")
    unique_images = set(image_rows)
    return {
        "split": split,
        "source_image_count": int(source_image_count),
        "source_annotation_count": int(source_annotation_count),
        "selected_image_count": len(unique_images),
        "selected_annotation_count": len(annotation_rows),
        "image_id_fingerprint": _id_fingerprint(unique_images),
        "annotation_id_fingerprint": annotation_id_set_fingerprint(annotation_rows),
        "annotation_to_image_fingerprint": annotation_to_image_fingerprint(
            annotation_rows, image_rows
        ),
    }


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise E9SpatialBankValidationError(
            f"{label} must be a lowercase SHA256 digest"
        )
    return value


def _require_git_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise E9SpatialBankValidationError(
            f"{label} must be a 40-character lowercase Git commit"
        )
    return value


def _validate_created_at(value: Any) -> None:
    if not isinstance(value, str) or _UTC_RFC3339.fullmatch(value) is None:
        raise E9SpatialBankValidationError(
            "manifest.created_at must be a UTC RFC3339 timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise E9SpatialBankValidationError(
            "manifest.created_at is not a valid timestamp"
        ) from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise E9SpatialBankValidationError("manifest.created_at must use UTC")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise E9SpatialBankValidationError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise E9SpatialBankValidationError(f"{path} must contain a mapping")
    return value


def _source_manifest(source: Path, *, require_complete: bool = True) -> dict[str, Any]:
    if source.is_file():
        raise E9SpatialBankValidationError(
            "legacy monolithic DINO archives are not valid E9 sources; provide "
            "the documented streaming dense-feature directory"
        )
    manifest_path = source / MANIFEST_NAME
    manifest = _load_json(manifest_path)
    required = {
        "format_version",
        "split",
        "extraction_config",
        "source_commit",
        "source_images",
        "selected_images",
        "source_annotations",
        "selected_annotations",
        "selected_image_ids_sha256",
        "selected_annotation_ids_sha256",
        "max_images",
        "is_pilot",
        "complete",
        "failed_image_ids",
        "images",
        "annotations",
        "shards",
    }
    missing = required.difference(manifest)
    if missing:
        raise E9SpatialBankValidationError(
            f"source manifest is missing keys {sorted(missing)}"
        )
    if manifest["format_version"] != E5_STREAMING_FORMAT:
        raise E9SpatialBankValidationError("unsupported dense source format")
    for key in (
        "source_images", "source_annotations", "selected_images",
        "selected_annotations", "images", "annotations",
    ):
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise E9SpatialBankValidationError(
                f"dense source manifest.{key} must be a nonnegative integer"
            )
    for key in ("complete", "is_pilot"):
        if type(manifest[key]) is not bool:
            raise E9SpatialBankValidationError(
                f"dense source manifest.{key} must be boolean"
            )
    if manifest["max_images"] is not None and (
        isinstance(manifest["max_images"], bool)
        or not isinstance(manifest["max_images"], int)
        or manifest["max_images"] <= 0
    ):
        raise E9SpatialBankValidationError(
            "dense source max_images must be null or a positive integer"
        )
    if not isinstance(manifest["failed_image_ids"], list):
        raise E9SpatialBankValidationError(
            "dense source failed_image_ids must be a list"
        )
    for key in ("selected_image_ids_sha256", "selected_annotation_ids_sha256"):
        _require_sha256(manifest[key], f"dense source {key}")
    _require_git_commit(manifest["source_commit"], "dense source commit")
    derived_source_pilot = (
        manifest["max_images"] is not None
        or manifest["selected_images"] < manifest["source_images"]
        or manifest["selected_annotations"] < manifest["source_annotations"]
    )
    if manifest["is_pilot"] != derived_source_pilot:
        raise E9SpatialBankValidationError("dense source has a forged pilot state")
    derived_complete = not manifest["failed_image_ids"] and (
        manifest["images"] == manifest["selected_images"]
        and manifest["annotations"] == manifest["selected_annotations"]
    )
    if manifest["complete"] != derived_complete:
        raise E9SpatialBankValidationError("dense source has forged completeness")
    if require_complete and not derived_complete:
        raise E9SpatialBankValidationError("dense source is incomplete")
    extraction = manifest["extraction_config"]
    if not isinstance(extraction, Mapping):
        raise E9SpatialBankValidationError("source extraction_config must be a mapping")
    if "backbone_weights_sha256" not in extraction:
        raise E9SpatialBankValidationError(
            "source extraction_config is missing required "
            "backbone_weights_sha256; regenerate E5 dense features with exact "
            "DINO weight identity"
        )
    _require_sha256(
        extraction["backbone_weights_sha256"],
        "source extraction backbone_weights_sha256",
    )
    mismatches = {
        key: (extraction.get(key), expected_value)
        for key, expected_value in CANONICAL_EXTRACTION.items()
        if extraction.get(key) != expected_value
    }
    if mismatches:
        raise E9SpatialBankValidationError(
            f"source extraction identity mismatch: {mismatches}"
        )
    if 448 // 14 != 32 or extraction["patch_count"] != (448 // 14) ** 2:
        raise E9SpatialBankValidationError("source grid cannot be recovered exactly")
    if not isinstance(manifest["shards"], list) or not manifest["shards"]:
        raise E9SpatialBankValidationError("source manifest contains no shards")
    return manifest


def _source_artifacts(source: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    manifest_path = source / MANIFEST_NAME
    artifacts.append({"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)})
    for entry in manifest["shards"]:
        if not isinstance(entry, Mapping) or "name" not in entry:
            raise E9SpatialBankValidationError("invalid source shard entry")
        path = source / entry["name"]
        if not path.is_file():
            raise E9SpatialBankValidationError(f"missing source shard {path}")
        digest = sha256_file(path)
        if entry.get("sha256") != digest:
            raise E9SpatialBankValidationError(f"source shard SHA mismatch: {path}")
        artifacts.append({"path": str(path.resolve()), "sha256": digest})
    return artifacts


def _load_source_record(file_object: Any, source: str) -> dict[str, Any]:
    try:
        record = torch.load(file_object, map_location="cpu", weights_only=False)
    except Exception as error:
        raise E9SpatialBankValidationError(f"cannot decode {source}: {error}") from error
    if not isinstance(record, Mapping) or set(record) != SOURCE_RECORD_KEYS:
        raise E9SpatialBankValidationError(f"{source}: dense record schema mismatch")
    return dict(record)


def _iter_source_records(source: Path, manifest: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for shard_entry in manifest["shards"]:
        path = source / shard_entry["name"]
        try:
            archive = tarfile.open(path, mode="r")
        except (OSError, tarfile.TarError) as error:
            raise E9SpatialBankValidationError(f"cannot open {path}: {error}") from error
        with archive:
            for member in archive:
                if not member.isfile():
                    continue
                if not member.name.endswith(".pth"):
                    raise E9SpatialBankValidationError(
                        f"unexpected source member {member.name!r}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise E9SpatialBankValidationError(f"cannot read {member.name}")
                yield _load_source_record(extracted, f"{path}:{member.name}")


def _validate_patch_map_tensors(
    patch_tokens: Any, self_attn_maps: Any, *, image_id: int | str
) -> None:
    patches = patch_tokens
    maps = self_attn_maps
    if (
        not torch.is_tensor(patches)
        or patches.device.type != "cpu"
        or patches.dtype != torch.float16
        or tuple(patches.shape) != (1024, 768)
    ):
        raise E9SpatialBankValidationError(
            f"image {image_id}: patch_tokens must be CPU float16 [1024,768]"
        )
    if (
        not torch.is_tensor(maps)
        or maps.device.type != "cpu"
        or maps.dtype != torch.float16
        or tuple(maps.shape) != (12, 1024)
    ):
        raise E9SpatialBankValidationError(
            f"image {image_id}: self_attn_maps must be CPU float16 [12,1024]"
        )
    if not torch.isfinite(patches).all() or not torch.isfinite(maps).all():
        raise E9SpatialBankValidationError(f"image {image_id}: non-finite source tensor")
    if torch.any(maps < 0):
        raise E9SpatialBankValidationError(f"image {image_id}: negative attention probability")
    sums = maps.float().sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=2e-3, rtol=2e-3):
        raise E9SpatialBankValidationError(
            f"image {image_id}: attention rows are not probabilities"
        )


def validate_source_record(record: Mapping[str, Any]) -> None:
    image_id = record.get("image_id")
    if isinstance(image_id, bool) or not isinstance(image_id, int):
        raise E9SpatialBankValidationError("source image_id must be an integer")
    if not isinstance(record.get("file_name"), str) or not record["file_name"]:
        raise E9SpatialBankValidationError(
            f"image {image_id}: file_name must be a nonempty string"
        )
    heads = record.get("disentangled_self_attn")
    if (
        not torch.is_tensor(heads) or heads.device.type != "cpu"
        or heads.dtype != torch.float32 or tuple(heads.shape) != (12, 768)
        or not torch.isfinite(heads).all()
    ):
        raise E9SpatialBankValidationError(
            f"image {image_id}: disentangled_self_attn must be finite CPU "
            "float32 [12,768]"
        )
    captions = record.get("captions")
    annotation_features = record.get("ann_feats")
    annotation_ids = record.get("annotation_ids")
    if not all(isinstance(value, list) for value in (
        captions, annotation_features, annotation_ids
    )):
        raise E9SpatialBankValidationError(
            f"image {image_id}: captions, ann_feats, and annotation_ids must be lists"
        )
    if not captions or not (
        len(captions) == len(annotation_features) == len(annotation_ids)
    ):
        raise E9SpatialBankValidationError(
            f"image {image_id}: annotation rows are not aligned"
        )
    if any(not isinstance(caption, str) for caption in captions):
        raise E9SpatialBankValidationError(
            f"image {image_id}: every caption must be a string"
        )
    if any(
        isinstance(annotation_id, bool) or not isinstance(annotation_id, int)
        for annotation_id in annotation_ids
    ) or len(set(annotation_ids)) != len(annotation_ids):
        raise E9SpatialBankValidationError(
            f"image {image_id}: annotation IDs must be unique integers"
        )
    for feature in annotation_features:
        if (
            not torch.is_tensor(feature)
            or not feature.is_floating_point()
            or tuple(feature.shape) != (512,)
            or not torch.isfinite(feature).all()
        ):
            raise E9SpatialBankValidationError(
                f"image {image_id}: ann_feats must be finite floating [512] tensors"
            )
    _validate_patch_map_tensors(
        record.get("patch_tokens"), record.get("self_attn_maps"), image_id=image_id
    )


def pool_source_image(
    patch_tokens: torch.Tensor,
    self_attn_maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool a validated 32x32 source record into normalized 16x16 cells."""

    _validate_patch_map_tensors(
        patch_tokens, self_attn_maps, image_id="pool input"
    )
    patches = patch_tokens.float().reshape(32, 32, 768)
    raw = patches.reshape(16, 2, 16, 2, 768).mean(dim=(1, 3))
    pooled = F.normalize(raw.reshape(256, 768), dim=-1)
    if torch.any(raw.reshape(256, 768).norm(dim=-1) == 0):
        raise E9SpatialBankValidationError("pooled patch contains a zero vector")

    # Stored maps are probabilities over patch tokens for each CLS-attention
    # head.  Average heads, sum each 2x2 probability block, then renormalize.
    prior_32 = self_attn_maps.float().mean(dim=0).reshape(32, 32)
    prior = prior_32.reshape(16, 2, 16, 2).sum(dim=(1, 3)).reshape(256)
    prior = prior.clamp_min(0)
    total = prior.sum()
    if not torch.isfinite(total) or total <= 0:
        raise E9SpatialBankValidationError("attention prior has invalid mass")
    prior = prior / total
    if not torch.isfinite(pooled).all() or not torch.isfinite(prior).all():
        raise E9SpatialBankValidationError("pooling produced non-finite values")
    return pooled, prior


def _validate_shard(value: Any, expected_rows: int | None = None) -> dict[str, Any]:
    shard = _closed(value, SHARD_KEYS, "E9 spatial shard")
    patches = shard["patch_embeddings"]
    priors = shard["attention_priors"]
    image_ids = shard["image_ids"]
    if not torch.is_tensor(image_ids) or image_ids.dtype != torch.int64 or image_ids.device.type != "cpu" or image_ids.ndim != 1:
        raise E9SpatialBankValidationError("image_ids must be CPU int64 [N]")
    rows = image_ids.shape[0]
    if expected_rows is not None and rows != expected_rows:
        raise E9SpatialBankValidationError("spatial shard row count mismatch")
    if len(set(image_ids.tolist())) != rows:
        raise E9SpatialBankValidationError("spatial shard has duplicate image IDs")
    if not torch.is_tensor(patches) or patches.dtype != torch.float16 or patches.device.type != "cpu" or tuple(patches.shape) != (rows, 256, 768):
        raise E9SpatialBankValidationError("patch_embeddings must be CPU float16 [N,256,768]")
    if not torch.is_tensor(priors) or priors.dtype != torch.float16 or priors.device.type != "cpu" or tuple(priors.shape) != (rows, 256):
        raise E9SpatialBankValidationError("attention_priors must be CPU float16 [N,256]")
    if not torch.isfinite(patches).all() or not torch.isfinite(priors).all() or torch.any(priors < 0):
        raise E9SpatialBankValidationError("spatial shard contains invalid values")
    patch_norms = patches.float().norm(dim=-1)
    if torch.max(torch.abs(patch_norms - 1)) > 2e-3:
        raise E9SpatialBankValidationError("serialized patch embeddings are not normalized")
    sums = priors.float().sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=2e-3, rtol=2e-3):
        raise E9SpatialBankValidationError("serialized priors do not sum to one")
    return {"rows": rows, "image_ids": image_ids.tolist()}


def _read_spatial_shard(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise E9SpatialBankValidationError(f"cannot read shard {path}: {error}") from error
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise E9SpatialBankValidationError(f"spatial shard byte count changed: {path}")
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise E9SpatialBankValidationError(f"spatial shard SHA changed: {path}")
    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise E9SpatialBankValidationError(f"cannot load shard {path}: {error}") from error


def load_spatial_shard(
    path: os.PathLike[str] | str,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, torch.Tensor]:
    path = Path(path)
    shard = _read_spatial_shard(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    _validate_shard(shard)
    return {
        "patch_embeddings": F.normalize(shard["patch_embeddings"].float(), dim=-1),
        "attention_priors": shard["attention_priors"].float()
        / shard["attention_priors"].float().sum(dim=-1, keepdim=True),
        "image_ids": shard["image_ids"],
    }


def _read_fd_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 1 << 20))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def validate_e9_spatial_bank(
    path: os.PathLike[str] | str,
    *,
    require_production: bool = True,
    verify_source_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate a closed-contract E9 spatial bank.

    Every filesystem read below is descriptor based: the bank root and every
    manifest/shard are opened with ``O_NOFOLLOW`` relative to an already-open
    directory descriptor and read from that exact descriptor, so nothing can
    be swapped for a symlink between a preliminary check and the read.
    """

    root = Path(path)
    try:
        root_fd = os.open(root, _NOFOLLOW_DIR_FLAGS)
    except OSError as error:
        raise E9SpatialBankValidationError(
            f"cannot open E9 bank root {root}: {error}"
        ) from error
    try:
        return _validate_e9_spatial_bank_fd(
            root_fd,
            root,
            require_production=require_production,
            verify_source_artifacts=verify_source_artifacts,
        )
    finally:
        os.close(root_fd)


def _validate_e9_spatial_bank_fd(
    root_fd: int,
    root: Path,
    *,
    require_production: bool,
    verify_source_artifacts: bool,
) -> dict[str, Any]:
    manifest_display = root / MANIFEST_NAME
    try:
        manifest_fd = os.open(MANIFEST_NAME, _NOFOLLOW_FILE_FLAGS, dir_fd=root_fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise E9SpatialBankValidationError(
                "E9 manifest.json must be a non-symlink regular file"
            ) from error
        raise E9SpatialBankValidationError(
            f"invalid E9 manifest file {manifest_display}: {error}"
        ) from error
    try:
        manifest_stat = os.fstat(manifest_fd)
        if not stat.S_ISREG(manifest_stat.st_mode):
            raise E9SpatialBankValidationError(
                "E9 manifest.json must be a non-symlink regular file"
            )
        if manifest_stat.st_size > _MAX_MANIFEST_BYTES:
            raise E9SpatialBankValidationError(
                f"E9 manifest.json exceeds the maximum allowed size: "
                f"{manifest_stat.st_size} > {_MAX_MANIFEST_BYTES} bytes"
            )
        payload = _read_fd_exact(manifest_fd, manifest_stat.st_size)
        if len(payload) != manifest_stat.st_size:
            raise E9SpatialBankValidationError(
                f"E9 manifest.json was truncated while reading: {manifest_display}"
            )
        try:
            manifest_value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise E9SpatialBankValidationError(
                f"invalid JSON {manifest_display}: {error}"
            ) from error
        if not isinstance(manifest_value, dict):
            raise E9SpatialBankValidationError(
                f"{manifest_display} must contain a mapping"
            )
    finally:
        os.close(manifest_fd)
    manifest = _closed(manifest_value, MANIFEST_KEYS, "E9 manifest")
    if manifest["format_version"] != E9_SPATIAL_BANK_FORMAT:
        raise E9SpatialBankValidationError("unsupported E9 spatial-bank format")
    for key in ("complete", "is_pilot", "production_eligible", "source_git_dirty"):
        if type(manifest[key]) is not bool:
            raise E9SpatialBankValidationError(f"manifest.{key} must be boolean")
    if manifest["split"] not in {"train", "val"}:
        raise E9SpatialBankValidationError("manifest split must be train or val")
    for key in (
        "source_images", "source_annotations", "selected_images",
        "selected_annotations", "shard_count",
    ):
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise E9SpatialBankValidationError(f"manifest.{key} must be nonnegative")
    if manifest["selected_images"] > manifest["source_images"]:
        raise E9SpatialBankValidationError("selected images exceed source images")
    if manifest["selected_annotations"] > manifest["source_annotations"]:
        raise E9SpatialBankValidationError(
            "selected annotations exceed source annotations"
        )
    if manifest["source_feature_format"] not in SUPPORTED_SOURCE_FEATURE_FORMATS:
        raise E9SpatialBankValidationError("unsupported source feature format")
    if manifest["source_feature_schema"] != sorted(SOURCE_RECORD_KEYS):
        raise E9SpatialBankValidationError("source feature schema mismatch")
    paths = manifest["source_feature_paths"]
    digests = manifest["source_feature_sha256"]
    if (
        not isinstance(paths, list) or not isinstance(digests, list)
        or len(paths) != len(digests) or not paths
        or any(not isinstance(path, str) or not path for path in paths)
        or any(
            not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
            for digest in digests
        )
    ):
        raise E9SpatialBankValidationError("source artifact identity mismatch")
    dino_identity = _closed(
        manifest["dino_identity"], DINO_IDENTITY_KEYS, "DINO identity"
    )
    extraction = _closed(
        manifest["extraction"], EXTRACTION_KEYS, "extraction identity"
    )
    builder = _closed(
        manifest["builder_config"], BUILDER_CONFIG_KEYS, "builder configuration"
    )
    if dino_identity["model"] != "dinov2_vitb14_reg":
        raise E9SpatialBankValidationError("DINO model identity mismatch")
    _require_git_commit(dino_identity["source_commit"], "DINO source commit")
    _require_sha256(
        dino_identity["checkpoint_sha256"], "DINO checkpoint identity"
    )
    extraction_mismatches = {
        key: (extraction.get(key), expected)
        for key, expected in CANONICAL_EXTRACTION.items()
        if extraction.get(key) != expected
    }
    if extraction_mismatches:
        raise E9SpatialBankValidationError(
            f"canonical E9 extraction identity mismatch: {extraction_mismatches}"
        )
    for key in ("annotation_path", "data_dir"):
        if not isinstance(extraction[key], str) or not extraction[key]:
            raise E9SpatialBankValidationError(
                f"extraction.{key} must be a nonempty source description"
            )
    if extraction["model"] != dino_identity["model"]:
        raise E9SpatialBankValidationError("DINO/extraction model identity mismatch")
    if (
        extraction["backbone_weights_sha256"]
        != dino_identity["checkpoint_sha256"]
    ):
        raise E9SpatialBankValidationError(
            "DINO/extraction checkpoint identity mismatch"
        )
    _require_git_commit(manifest["source_git_commit"], "E9 source Git commit")
    if manifest["source_git_dirty"]:
        _require_sha256(
            manifest["source_git_diff_sha256"], "E9 dirty-source diff"
        )
    elif manifest["source_git_diff_sha256"] is not None:
        raise E9SpatialBankValidationError(
            "clean E9 provenance requires source_git_diff_sha256=null"
        )
    _validate_created_at(manifest["created_at"])
    if (
        isinstance(builder["shard_rows"], bool)
        or not isinstance(builder["shard_rows"], int)
        or builder["shard_rows"] <= 0
    ):
        raise E9SpatialBankValidationError("invalid builder shard_rows")
    if builder["max_images"] is not None and (
        isinstance(builder["max_images"], bool)
        or not isinstance(builder["max_images"], int)
        or builder["max_images"] <= 0
    ):
        raise E9SpatialBankValidationError("invalid builder max_images")
    if builder["source_max_images"] is not None and (
        isinstance(builder["source_max_images"], bool)
        or not isinstance(builder["source_max_images"], int)
        or builder["source_max_images"] <= 0
    ):
        raise E9SpatialBankValidationError("invalid builder source_max_images")
    if (
        isinstance(builder["source_selected_images"], bool)
        or not isinstance(builder["source_selected_images"], int)
        or not 0 <= builder["source_selected_images"] <= manifest["source_images"]
        or isinstance(builder["source_selected_annotations"], bool)
        or not isinstance(builder["source_selected_annotations"], int)
        or not 0 <= builder["source_selected_annotations"] <= manifest["source_annotations"]
        or type(builder["source_is_pilot"]) is not bool
        or type(builder["source_complete"]) is not bool
    ):
        raise E9SpatialBankValidationError("invalid builder source identity")
    expected_selected = min(
        builder["source_selected_images"],
        builder["max_images"] or builder["source_selected_images"],
    )
    if manifest["selected_images"] != expected_selected:
        raise E9SpatialBankValidationError("builder/selected image count mismatch")
    if manifest["geometry"] != EXPECTED_GEOMETRY or manifest["dtypes"] != EXPECTED_DTYPES:
        raise E9SpatialBankValidationError("spatial bank geometry/dtype mismatch")
    if manifest["pooling_version"] != POOLING_VERSION or manifest["attention_prior_version"] != ATTENTION_PRIOR_VERSION:
        raise E9SpatialBankValidationError("spatial bank equation identity mismatch")
    if manifest["global_token_handling"] != GLOBAL_TOKEN_HANDLING:
        raise E9SpatialBankValidationError("global-token handling mismatch")
    dataset_identity = _closed(
        manifest["dataset_identity"], DATASET_IDENTITY_KEYS, "dataset identity"
    )
    for key in (
        "source_image_count", "source_annotation_count",
        "selected_image_count", "selected_annotation_count",
    ):
        value = dataset_identity[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise E9SpatialBankValidationError(
                f"dataset_identity.{key} must be a nonnegative integer"
            )
    for key in (
        "image_id_fingerprint", "annotation_id_fingerprint",
        "annotation_to_image_fingerprint",
    ):
        _require_sha256(dataset_identity[key], f"dataset identity {key}")
    expected_dataset_counts = {
        "split": manifest["split"],
        "source_image_count": manifest["source_images"],
        "source_annotation_count": manifest["source_annotations"],
        "selected_image_count": manifest["selected_images"],
        "selected_annotation_count": manifest["selected_annotations"],
        "image_id_fingerprint": manifest["image_id_fingerprint"],
    }
    if any(
        dataset_identity[key] != value
        for key, value in expected_dataset_counts.items()
    ):
        raise E9SpatialBankValidationError(
            "dataset identity does not match manifest counts/fingerprint"
        )
    shards = manifest["shards"]
    index = manifest["image_index"]
    if not isinstance(shards, list) or len(shards) != manifest["shard_count"]:
        raise E9SpatialBankValidationError("manifest shard count mismatch")
    if not isinstance(index, list) or len(index) != manifest["selected_images"]:
        raise E9SpatialBankValidationError("manifest image index count mismatch")
    seen: list[int] = []
    seen_shard_names: set[str] = set()
    expected_start = 0
    for shard_number, entry in enumerate(shards):
        _closed(entry, SHARD_METADATA_KEYS, "E9 shard metadata")
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise E9SpatialBankValidationError("invalid spatial shard name")
        if entry["name"] in seen_shard_names:
            raise E9SpatialBankValidationError("duplicate spatial shard name")
        seen_shard_names.add(entry["name"])
        expected_name = f"{manifest['split']}-{shard_number:06d}.pth"
        if entry["name"] != expected_name:
            raise E9SpatialBankValidationError(
                "non-canonical spatial shard name: "
                f"expected={expected_name}, got={entry['name']!r}"
            )
        _require_sha256(entry["sha256"], "spatial shard")
        for key in ("bytes", "row_start", "row_end", "row_count"):
            item = entry[key]
            if (
                isinstance(item, bool) or not isinstance(item, int)
                or item < 0 or (key in {"bytes", "row_count"} and item == 0)
            ):
                raise E9SpatialBankValidationError(
                    f"invalid spatial shard {key}"
                )
        if entry["row_start"] != expected_start or entry["row_end"] != expected_start + entry["row_count"]:
            raise E9SpatialBankValidationError("non-contiguous shard row ranges")
        shard_display = root / entry["name"]
        try:
            shard_fd = os.open(entry["name"], _NOFOLLOW_FILE_FLAGS, dir_fd=root_fd)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise E9SpatialBankValidationError(
                    "spatial shard must be a non-symlink regular file: "
                    f"{shard_display}"
                ) from error
            raise E9SpatialBankValidationError(
                f"missing spatial shard: {shard_display}"
            ) from error
        try:
            shard_stat = os.fstat(shard_fd)
            if not stat.S_ISREG(shard_stat.st_mode):
                raise E9SpatialBankValidationError(
                    "spatial shard must be a non-symlink regular file: "
                    f"{shard_display}"
                )
            if shard_stat.st_size != entry["bytes"]:
                raise E9SpatialBankValidationError(
                    f"spatial shard byte count changed: {shard_display}"
                )
            payload = _read_fd_exact(shard_fd, shard_stat.st_size)
            if len(payload) != shard_stat.st_size:
                raise E9SpatialBankValidationError(
                    f"spatial shard was truncated while reading: {shard_display}"
                )
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise E9SpatialBankValidationError(
                    f"spatial shard SHA changed: {shard_display}"
                )
            try:
                raw = torch.load(
                    io.BytesIO(payload), map_location="cpu", weights_only=True
                )
            except Exception as error:
                raise E9SpatialBankValidationError(
                    f"cannot load shard {shard_display}: {error}"
                ) from error
        finally:
            os.close(shard_fd)
        result = _validate_shard(raw, entry["row_count"])
        for row, image_id in enumerate(result["image_ids"]):
            index_entry = index[expected_start + row]
            _closed(index_entry, IMAGE_INDEX_KEYS, "E9 image index")
            if index_entry != {"image_id": image_id, "shard": shard_number, "row": row}:
                raise E9SpatialBankValidationError("manifest image index mismatch")
        seen.extend(result["image_ids"])
        expected_start = entry["row_end"]
    if expected_start != manifest["selected_images"] or len(set(seen)) != len(seen):
        raise E9SpatialBankValidationError("spatial-bank image coverage mismatch")
    if _id_fingerprint(seen) != manifest["image_id_fingerprint"]:
        raise E9SpatialBankValidationError("image fingerprint mismatch")
    complete = (
        expected_start == manifest["selected_images"]
        and len(seen) == manifest["selected_images"]
        and len(set(seen)) == len(seen)
        and dataset_identity["selected_annotation_count"]
        == manifest["selected_annotations"]
    )
    source_is_pilot = (
        builder["source_max_images"] is not None
        or builder["source_selected_images"] < manifest["source_images"]
        or builder["source_selected_annotations"] < manifest["source_annotations"]
    )
    if builder["source_is_pilot"] != source_is_pilot:
        raise E9SpatialBankValidationError("forged source pilot identity")
    is_pilot = (
        source_is_pilot
        or builder["max_images"] is not None
        or manifest["selected_images"] < manifest["source_images"]
        or manifest["selected_annotations"] < manifest["source_annotations"]
    )
    production_eligible = (
        complete
        and not is_pilot
        and manifest["selected_images"] == manifest["source_images"]
        and manifest["selected_annotations"] == manifest["source_annotations"]
        and builder["source_selected_images"] == manifest["source_images"]
        and builder["source_selected_annotations"] == manifest["source_annotations"]
        and builder["source_complete"]
        and manifest["source_feature_format"] == CANONICAL_SOURCE_FEATURE_FORMAT
        and not manifest["source_git_dirty"]
    )
    if manifest["complete"] != complete:
        raise E9SpatialBankValidationError("forged spatial-bank completeness")
    if manifest["is_pilot"] != is_pilot:
        raise E9SpatialBankValidationError("forged spatial-bank pilot identity")
    if manifest["production_eligible"] != production_eligible:
        raise E9SpatialBankValidationError("forged production eligibility")
    if require_production and not production_eligible:
        raise E9SpatialBankValidationError("spatial bank is not production eligible")
    if verify_source_artifacts:
        for source_path, digest in zip(
            manifest["source_feature_paths"], manifest["source_feature_sha256"]
        ):
            candidate = Path(source_path)
            if not candidate.is_file() or sha256_file(candidate) != digest:
                raise E9SpatialBankValidationError("source feature SHA mismatch")
    return {
        "split": manifest["split"],
        "images": len(seen),
        "annotations": manifest["selected_annotations"],
        "shards": len(shards),
        "complete": complete,
        "is_pilot": is_pilot,
        "production_eligible": production_eligible,
        "dataset_identity": dict(dataset_identity),
        "estimated_bytes": manifest["selected_images"] * (256 * 768 * 2 + 256 * 2 + 8),
    }


DirectoryIdentity = tuple[int, int]


@dataclass
class _OwnedEntry:
    """An open, identity-checked handle to a filesystem object reached
    without following a symlink.

    The retained file descriptor pins the underlying inode for the handle's
    lifetime: even if the pathname used to reach it is later replaced, this
    handle keeps observing -- and, for a directory, keeps being able to read
    -- the exact object it was opened for.  The ``(dev, ino)`` pair is the
    ownership proof compared against whatever a later atomic capture turns
    up at a shared pathname; a separate lstat() of that pathname is never
    used as a substitute for this proof.
    """

    fd: int
    dev: int
    ino: int
    is_dir: bool

    @property
    def identity(self) -> DirectoryIdentity:
        return (self.dev, self.ino)

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


def _fstat_identity(fd: int) -> tuple[int, int, bool]:
    metadata = os.fstat(fd)
    return metadata.st_dev, metadata.st_ino, stat.S_ISDIR(metadata.st_mode)


def _stat_or_none(parent_fd: int, name: str) -> os.stat_result | None:
    """A single fstatat(..., AT_SYMLINK_NOFOLLOW) relative to an already-open
    directory descriptor.  Safe to call on a name nobody else can reach
    (a private reservation/quarantine name) or purely to classify an object
    that was *just* captured atomically; never used as a check that is
    separated from a later destructive pathname operation."""

    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_parent_nofollow(path: Path) -> int:
    try:
        return os.open(path, _NOFOLLOW_DIR_FLAGS)
    except OSError as error:
        raise E9PublicationRecoveryError(
            f"cannot open {path} as a non-symlink directory: {error}"
        ) from error


_renameat2_fn: Any = None


def _renameat2_symbol() -> Any:
    global _renameat2_fn
    if _renameat2_fn is None:
        library = ctypes.CDLL(None, use_errno=True)
        if not hasattr(library, "renameat2"):
            raise E9AtomicOperationUnsupportedError(
                "libc does not export renameat2(2); E9 publication requires "
                "RENAME_NOREPLACE and RENAME_EXCHANGE"
            )
        symbol = library.renameat2
        symbol.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        )
        symbol.restype = ctypes.c_int
        _renameat2_fn = symbol
    return _renameat2_fn


def _renameat2(
    old_dir_fd: int,
    old_name: str,
    new_dir_fd: int,
    new_name: str,
    flags: int,
    *,
    unsupported_label: str,
) -> None:
    symbol = _renameat2_symbol()
    result = symbol(
        old_dir_fd, os.fsencode(old_name), new_dir_fd, os.fsencode(new_name),
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in _UNSUPPORTED_RENAMEAT2_ERRNOS:
        raise E9AtomicOperationUnsupportedError(
            f"{unsupported_label} is not supported by this filesystem/"
            f"kernel (errno={error_number}: {os.strerror(error_number)}); "
            "E9 publication requires it and will not fall back to a "
            "non-atomic rename"
        )
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, os.strerror(error_number), new_name)
    raise OSError(error_number, os.strerror(error_number), old_name)


def _random_private_name(prefix: str) -> str:
    return f".e9-{prefix}-{os.getpid()}-{secrets.token_hex(8)}"


def _probe_atomic_rename_support(parent_fd: int, parent_display: str) -> None:
    """Fail closed, before any existing destination is ever touched, if this
    filesystem/kernel cannot provide RENAME_NOREPLACE or RENAME_EXCHANGE.
    Only private, freshly created objects are used for the probe."""

    first_name = _random_private_name("probe")
    second_name = _random_private_name("probe")
    try:
        os.mkdir(first_name, _PRIVATE_DIR_MODE, dir_fd=parent_fd)
    except OSError as error:
        raise E9PublicationRecoveryError(
            f"cannot create a probe object in {parent_display}: {error}"
        ) from error
    try:
        _renameat2(
            parent_fd, first_name, parent_fd, second_name,
            _RENAME_NOREPLACE, unsupported_label="RENAME_NOREPLACE",
        )
        os.mkdir(first_name, _PRIVATE_DIR_MODE, dir_fd=parent_fd)
        _renameat2(
            parent_fd, first_name, parent_fd, second_name,
            _RENAME_EXCHANGE, unsupported_label="RENAME_EXCHANGE",
        )
    finally:
        for name in (first_name, second_name):
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass


def _create_private_directory(
    parent_fd: int, *, prefix: str
) -> tuple[str, _OwnedEntry]:
    """Create a private mode-0700 directory under ``parent_fd`` and return an
    identity-checked handle, established before the object is ever exposed
    at a shared pathname."""

    for _ in range(8):
        name = _random_private_name(prefix)
        try:
            os.mkdir(name, _PRIVATE_DIR_MODE, dir_fd=parent_fd)
        except FileExistsError:
            continue
        fd = os.open(name, _NOFOLLOW_DIR_FLAGS, dir_fd=parent_fd)
        dev, ino, is_dir = _fstat_identity(fd)
        return name, _OwnedEntry(fd=fd, dev=dev, ino=ino, is_dir=is_dir)
    raise E9PublicationRecoveryError(
        f"cannot allocate a private reservation name for {prefix!r}"
    )


def _discard_unshared_reservation(
    parent_fd: int, name: str, entry: _OwnedEntry
) -> None:
    """Remove a private reservation that was never exposed at a shared
    pathname; its name was never known to any other process."""

    try:
        if entry.is_dir:
            _rmtree_via_fd(entry.fd, mount_dev=entry.dev)
    except OSError:
        pass
    entry.close()
    try:
        if entry.is_dir:
            os.rmdir(name, dir_fd=parent_fd)
        else:
            os.unlink(name, dir_fd=parent_fd)
    except OSError:
        pass


def _install_reservation(
    parent_fd: int,
    reservation_name: str,
    destination_name: str,
    *,
    conflict_label: str,
) -> None:
    try:
        _renameat2(
            parent_fd, reservation_name, parent_fd, destination_name,
            _RENAME_NOREPLACE, unsupported_label="RENAME_NOREPLACE",
        )
    except FileExistsError as error:
        raise E9PublicationConflictError(
            f"refusing to overwrite; concurrent {conflict_label} already "
            "exists"
        ) from error


def _rmtree_via_fd(dir_fd: int, *, mount_dev: int) -> None:
    """Recursively remove the contents of an already-open, identity-verified
    directory using descriptor-relative operations only.  Symlinks are never
    followed -- they are unlinked as themselves -- and traversal refuses to
    cross a mount point."""

    for entry in os.scandir(dir_fd):
        entry_stat = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(entry_stat.st_mode):
            os.unlink(entry.name, dir_fd=dir_fd)
        elif stat.S_ISDIR(entry_stat.st_mode):
            if entry_stat.st_dev != mount_dev:
                raise E9PublicationRecoveryError(
                    f"refusing to cross a mount point while removing "
                    f"{entry.name!r}"
                )
            child_fd = os.open(entry.name, _NOFOLLOW_DIR_FLAGS, dir_fd=dir_fd)
            try:
                _rmtree_via_fd(child_fd, mount_dev=mount_dev)
            finally:
                os.close(child_fd)
            os.rmdir(entry.name, dir_fd=dir_fd)
        else:
            os.unlink(entry.name, dir_fd=dir_fd)


def _capture_shared_entry(
    parent_fd: int, name: str, *, prefix: str
) -> tuple[str, _OwnedEntry]:
    """Atomically swap whatever currently sits at ``name`` for a fresh
    private placeholder.

    No lstat/open of ``name`` happens first: the RENAME_EXCHANGE call is the
    only observation, so there is no gap in which a concurrent replacement
    of ``name`` could go unnoticed between "look" and "act".  Returns the
    private name the captured object now lives under, plus a retained handle
    to the placeholder that took its place at ``name``.
    """

    placeholder_name, placeholder = _create_private_directory(
        parent_fd, prefix=f"{prefix}-quarantine"
    )
    try:
        _renameat2(
            parent_fd, placeholder_name, parent_fd, name,
            _RENAME_EXCHANGE, unsupported_label="RENAME_EXCHANGE",
        )
    except OSError:
        _discard_unshared_reservation(parent_fd, placeholder_name, placeholder)
        raise
    return placeholder_name, placeholder


def _restore_or_preserve(
    parent_fd: int,
    quarantine_name: str,
    name: str,
    *,
    label: str,
    parent_display: str,
) -> str:
    """Swap the object captured at ``quarantine_name`` back to ``name`` and
    retire the placeholder that returns to ``quarantine_name``.

    Used both when the captured object turned out to be foreign (restore,
    then the caller raises) and when it matched our own expectation (restore,
    then the caller continues) -- restoring is the same atomic operation
    either way.
    """

    try:
        _renameat2(
            parent_fd, quarantine_name, parent_fd, name,
            _RENAME_EXCHANGE, unsupported_label="RENAME_EXCHANGE",
        )
    except OSError as error:
        raise E9PublicationRecoveryError(
            f"cannot restore {label} to {parent_display}/{name}: {error}; "
            f"it remains preserved at {parent_display}/{quarantine_name}"
        ) from error
    try:
        os.rmdir(quarantine_name, dir_fd=parent_fd)
    except OSError:
        pass
    return f"{label} restored to {parent_display}/{name}"


def _retire_placeholder(
    parent_fd: int,
    name: str,
    placeholder_identity: DirectoryIdentity,
    *,
    parent_display: str,
) -> None:
    """Remove the empty placeholder directory left at ``name`` after a
    successful disposal, verifying with one more atomic capture that it is
    still exactly the placeholder that was put there."""

    retire_name = _random_private_name("retire")
    try:
        _renameat2(
            parent_fd, name, parent_fd, retire_name,
            _RENAME_NOREPLACE, unsupported_label="RENAME_NOREPLACE",
        )
    except FileNotFoundError:
        return
    captured_stat = _stat_or_none(parent_fd, retire_name)
    matches = (
        captured_stat is not None
        and stat.S_ISDIR(captured_stat.st_mode)
        and (captured_stat.st_dev, captured_stat.st_ino) == placeholder_identity
    )
    if matches:
        try:
            os.rmdir(retire_name, dir_fd=parent_fd)
        except OSError as error:
            raise E9PublicationRecoveryError(
                "cannot remove a retired E9 placeholder at "
                f"{parent_display}/{retire_name}: {error}"
            ) from error
        return
    try:
        _renameat2(
            parent_fd, retire_name, parent_fd, name,
            _RENAME_NOREPLACE, unsupported_label="RENAME_NOREPLACE",
        )
    except OSError as error:
        raise E9PublicationRecoveryError(
            "an unrecognized object was captured while retiring a private "
            f"E9 placeholder; preserved at {parent_display}/{retire_name}: "
            f"{error}"
        ) from error
    raise E9PublicationRecoveryError(
        f"refusing to remove an unrecognized object found at "
        f"{parent_display}/{name} while retiring a private placeholder; it "
        "was restored unmodified"
    )


def _dispose_owned(
    parent_fd: int,
    name: str,
    expected_identity: DirectoryIdentity | None,
    *,
    label: str,
    parent_display: str,
) -> None:
    """Atomically capture whatever sits at ``name``.

    If it is exactly the object expected (an object created and solely
    owned by this process), it is recursively removed through directory
    descriptors only, never by following a symlink or crossing a mount
    point.  Anything else is restored to ``name`` byte-for-byte /
    symlink-for-symlink and an :class:`E9PublicationRecoveryError` is
    raised naming it.
    """

    quarantine_name, placeholder = _capture_shared_entry(
        parent_fd, name, prefix="dispose"
    )
    placeholder_identity = placeholder.identity
    try:
        captured_stat = _stat_or_none(parent_fd, quarantine_name)
        owned = (
            captured_stat is not None
            and stat.S_ISDIR(captured_stat.st_mode)
            and expected_identity is not None
            and (captured_stat.st_dev, captured_stat.st_ino) == expected_identity
        )
        if not owned:
            _restore_or_preserve(
                parent_fd, quarantine_name, name,
                label=label, parent_display=parent_display,
            )
            raise E9PublicationRecoveryError(
                f"refusing to remove foreign {label} at {parent_display}/"
                f"{name}; it was restored unmodified"
            )
        captured_fd = os.open(quarantine_name, _NOFOLLOW_DIR_FLAGS, dir_fd=parent_fd)
        try:
            _rmtree_via_fd(captured_fd, mount_dev=captured_stat.st_dev)
        finally:
            os.close(captured_fd)
        os.rmdir(quarantine_name, dir_fd=parent_fd)
    finally:
        placeholder.close()
    _retire_placeholder(
        parent_fd, name, placeholder_identity, parent_display=parent_display
    )


def _dispose_private_directory(
    path: Path, identity: DirectoryIdentity, *, label: str
) -> None:
    parent_fd = _open_parent_nofollow(path.parent)
    try:
        _dispose_owned(
            parent_fd, path.name, identity,
            label=label, parent_display=str(path.parent),
        )
    finally:
        os.close(parent_fd)


def _preserve_orphaned_reservation(
    parent_fd: int, reservation: _OwnedEntry, *, prefix: str
) -> str:
    """Best-effort: relink every regular file out of a reservation directory
    that lost its shared name into a freshly named recovery directory, so a
    fully validated bank is not silently lost even if its intended pathname
    was taken by something else in the same instant."""

    recovery_name, recovery = _create_private_directory(parent_fd, prefix=prefix)
    try:
        for entry in os.scandir(reservation.fd):
            entry_stat = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(entry_stat.st_mode):
                continue
            os.link(
                entry.name, entry.name,
                src_dir_fd=reservation.fd, dst_dir_fd=recovery.fd,
            )
        os.fsync(recovery.fd)
    finally:
        recovery.close()
    return recovery_name


def _acquire_publication_lock(lock: Path) -> _OwnedEntry:
    parent_fd = _open_parent_nofollow(lock.parent)
    try:
        reservation_name, reservation = _create_private_directory(
            parent_fd, prefix="lock-reserve"
        )
        try:
            _install_reservation(
                parent_fd, reservation_name, lock.name,
                conflict_label=f"E9 publication lock at {lock}",
            )
        except E9PublicationConflictError as error:
            _discard_unshared_reservation(parent_fd, reservation_name, reservation)
            raise E9PublicationConflictError(
                "E9 publication lock already exists; inspect it manually at "
                f"{lock}"
            ) from error
        os.fsync(parent_fd)
        return reservation
    finally:
        os.close(parent_fd)


def _release_publication_lock(lock: Path, identity: _OwnedEntry) -> None:
    parent_fd = _open_parent_nofollow(lock.parent)
    try:
        try:
            _dispose_owned(
                parent_fd, lock.name, identity.identity,
                label="E9 publication lock", parent_display=str(lock.parent),
            )
        except E9PublicationRecoveryError as error:
            raise E9PublicationRecoveryError(
                "refusing to remove a replaced E9 publication lock; inspect "
                f"it manually at {lock}"
            ) from error
    finally:
        identity.close()
        os.close(parent_fd)


def _fsync_name(name: str, *, dir_fd: int) -> None:
    fd = os.open(name, _NOFOLLOW_FILE_FLAGS, dir_fd=dir_fd)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_manifest_last(
    parent_fd: int,
    output_name: str,
    staging_fd: int,
    *,
    output_display: Path,
    final_verification: Callable[[], None],
) -> _OwnedEntry:
    reservation_name, reservation = _create_private_directory(
        parent_fd, prefix="output-reserve"
    )
    installed = False
    try:
        _install_reservation(
            parent_fd, reservation_name, output_name,
            conflict_label=f"E9 destination {output_display}",
        )
        installed = True
        names = sorted(
            entry.name for entry in os.scandir(staging_fd)
            if entry.name != MANIFEST_NAME
        )
        for name in names:
            os.link(name, name, src_dir_fd=staging_fd, dst_dir_fd=reservation.fd)
            _fsync_name(name, dir_fd=reservation.fd)
        final_verification()
        os.link(
            MANIFEST_NAME, MANIFEST_NAME,
            src_dir_fd=staging_fd, dst_dir_fd=reservation.fd,
        )
        _fsync_name(MANIFEST_NAME, dir_fd=reservation.fd)
        os.fsync(reservation.fd)
        os.fsync(parent_fd)
        _validate_e9_spatial_bank_fd(
            reservation.fd, output_display,
            require_production=False, verify_source_artifacts=False,
        )
    except Exception:
        if installed:
            try:
                _dispose_owned(
                    parent_fd, output_name, reservation.identity,
                    label="partial E9 output reservation",
                    parent_display=str(output_display.parent),
                )
            finally:
                reservation.close()
        else:
            _discard_unshared_reservation(parent_fd, reservation_name, reservation)
        raise
    return reservation


def _checkpoint_and_commit_output(
    parent_fd: int,
    output_name: str,
    reservation: _OwnedEntry,
    *,
    parent_display: str,
) -> None:
    """Close the post-validation window.

    Atomically capture whatever is now at ``output_name`` and confirm, via
    the reservation fd/identity retained since before this bank was ever
    exposed at a shared pathname, that it is still exactly the bank that was
    just validated.  Only after that checkpoint succeeds is the caller
    allowed to retire the old pilot backup.
    """

    try:
        quarantine_name, checkpoint_placeholder = _capture_shared_entry(
            parent_fd, output_name, prefix="checkpoint"
        )
    except OSError as error:
        recovery_name = _preserve_orphaned_reservation(
            parent_fd, reservation, prefix="orphaned-output"
        )
        raise E9PublicationRecoveryError(
            "E9 output disappeared immediately after validation; "
            f"publication aborted ({error}). The validated new bank was "
            f"preserved at {parent_display}/{recovery_name}."
        ) from error
    try:
        captured_stat = _stat_or_none(parent_fd, quarantine_name)
        matches = (
            captured_stat is not None
            and stat.S_ISDIR(captured_stat.st_mode)
            and (captured_stat.st_dev, captured_stat.st_ino) == reservation.identity
        )
        restore_note = _restore_or_preserve(
            parent_fd, quarantine_name, output_name,
            label="E9 output", parent_display=parent_display,
        )
        if not matches:
            recovery_name = _preserve_orphaned_reservation(
                parent_fd, reservation, prefix="orphaned-output"
            )
            raise E9PublicationRecoveryError(
                "E9 output was replaced immediately after validation; "
                f"publication aborted. {restore_note}. The validated new "
                f"bank was preserved at {parent_display}/{recovery_name}."
            )
    finally:
        checkpoint_placeholder.close()


def _publish_directory(
    staging: Path,
    output: Path,
    *,
    overwrite: bool,
    final_verification: Callable[[], None],
    staging_identity: DirectoryIdentity | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = _open_parent_nofollow(output.parent)
    try:
        parent_display = str(output.parent)
        _probe_atomic_rename_support(parent_fd, parent_display)

        if staging_identity is None:
            probe_fd = os.open(staging, _NOFOLLOW_DIR_FLAGS)
            try:
                staging_identity = _fstat_identity(probe_fd)[:2]
            finally:
                os.close(probe_fd)

        output_name = output.name
        lock_path = output.with_name(f".{output_name}.publication-lock")
        backup_name = f".{output_name}.backup-{os.getpid()}"

        lock_identity = _acquire_publication_lock(lock_path)
        backup_moved = False
        backup_owner: _OwnedEntry | None = None
        try:
            if _stat_or_none(parent_fd, backup_name) is not None:
                raise FileExistsError(
                    "stale E9 publication backup exists: "
                    f"{output.with_name(backup_name)}"
                )
            output_present = _stat_or_none(parent_fd, output_name) is not None
            if not overwrite and output_present:
                raise FileExistsError(
                    f"refusing to overwrite E9 spatial bank: {output}"
                )
            if overwrite and output_present:
                try:
                    _renameat2(
                        parent_fd, output_name, parent_fd, backup_name,
                        _RENAME_NOREPLACE, unsupported_label="RENAME_NOREPLACE",
                    )
                except FileExistsError as error:
                    raise FileExistsError(
                        "stale E9 publication backup exists: "
                        f"{output.with_name(backup_name)}"
                    ) from error
                backup_moved = True
                backup_stat = os.stat(
                    backup_name, dir_fd=parent_fd, follow_symlinks=False
                )
                if stat.S_ISLNK(backup_stat.st_mode):
                    raise E9SpatialBankValidationError(
                        "refusing to overwrite a symbolic-link E9 destination"
                    )
                if not stat.S_ISDIR(backup_stat.st_mode):
                    raise E9SpatialBankValidationError(
                        "existing E9 overwrite destination is not a "
                        "spatial-bank directory"
                    )
                backup_fd = os.open(
                    backup_name, _NOFOLLOW_DIR_FLAGS, dir_fd=parent_fd
                )
                backup_owner = _OwnedEntry(
                    fd=backup_fd, dev=backup_stat.st_dev,
                    ino=backup_stat.st_ino, is_dir=True,
                )
                previous = validate_e9_spatial_bank(
                    output.with_name(backup_name), require_production=False
                )
                if not previous["is_pilot"] or previous["production_eligible"]:
                    raise E9SpatialBankValidationError(
                        "existing E9 spatial bank is not a replaceable pilot"
                    )

            staging_fd = os.open(staging, _NOFOLLOW_DIR_FLAGS)
            try:
                reservation = _publish_manifest_last(
                    parent_fd, output_name, staging_fd,
                    output_display=output,
                    final_verification=final_verification,
                )
            finally:
                os.close(staging_fd)
            try:
                _checkpoint_and_commit_output(
                    parent_fd, output_name, reservation,
                    parent_display=parent_display,
                )
            finally:
                reservation.close()
        except E9PublicationRecoveryError as publication_error:
            if backup_owner is not None:
                raise E9PublicationRecoveryError(
                    f"{publication_error} Old E9 pilot backup preserved "
                    f"at {output.with_name(backup_name)}."
                ) from publication_error
            raise
        except Exception as publication_error:
            if backup_moved:
                try:
                    _renameat2(
                        parent_fd, backup_name, parent_fd, output_name,
                        _RENAME_NOREPLACE,
                        unsupported_label="RENAME_NOREPLACE",
                    )
                except FileExistsError as restore_error:
                    raise E9PublicationRecoveryError(
                        "E9 publication conflict preserved both "
                        f"artifacts; output={output}, backup="
                        f"{output.with_name(backup_name)}"
                    ) from restore_error
                except OSError as restore_error:
                    raise E9PublicationRecoveryError(
                        "E9 pilot restoration failed; preserved "
                        f"recovery paths: output={output}, backup="
                        f"{output.with_name(backup_name)}"
                    ) from restore_error
            raise
        else:
            if backup_owner is not None:
                _dispose_owned(
                    parent_fd, backup_name, backup_owner.identity,
                    label="validated old E9 pilot backup",
                    parent_display=parent_display,
                )
            _dispose_owned(
                parent_fd, staging.name, staging_identity,
                label="E9 staging directory",
                parent_display=parent_display,
            )
        finally:
            if backup_owner is not None:
                backup_owner.close()
            _release_publication_lock(lock_path, lock_identity)
    finally:
        os.close(parent_fd)


def build_e9_spatial_bank(
    source: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    *,
    split: str,
    expected_dino_source_commit: str | None = None,
    expected_dino_checkpoint_sha256: str | None = None,
    shard_rows: int = 128,
    max_images: int | None = None,
    overwrite: bool = False,
    allow_dirty_source: bool = False,
    repository_root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if isinstance(shard_rows, bool) or not isinstance(shard_rows, int) or shard_rows <= 0:
        raise ValueError("shard_rows must be a positive integer")
    if max_images is not None and (isinstance(max_images, bool) or not isinstance(max_images, int) or max_images <= 0):
        raise ValueError("max_images must be a positive integer")
    if expected_dino_source_commit is not None:
        _require_git_commit(
            expected_dino_source_commit, "expected DINO source commit"
        )
    if expected_dino_checkpoint_sha256 is not None:
        _require_sha256(
            expected_dino_checkpoint_sha256, "expected DINO checkpoint"
        )
    source = Path(source).resolve()
    output_text = os.fspath(output)
    if not isinstance(output_text, str):
        raise ValueError("E9 output path must be text")
    if (
        not output_text
        or output_text.endswith(os.sep)
        or output_text.rsplit(os.sep, 1)[-1] in {"", ".", ".."}
    ):
        raise ValueError("E9 output path must have a nonempty final name")
    raw_output = Path(output_text).expanduser()
    if not raw_output.is_absolute():
        raw_output = Path.cwd() / raw_output
    raw_parent = raw_output.parent.absolute()
    resolved_parent = raw_output.parent.resolve()
    # Resolve every ancestor while retaining the final component verbatim so
    # a final-component symlink remains visible to os.path.lexists/lstat.
    output = resolved_parent / raw_output.name
    repository_root = Path(repository_root or Path(__file__).parents[1]).resolve()
    if (
        source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("E9 source and output directories must not overlap")
    try:
        output.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise ValueError("E9 banks must be published outside the source repository")
    try:
        output.parent.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "E9 output parent must resolve outside the source repository"
        )
    if raw_parent != resolved_parent and os.path.lexists(output):
        raise ValueError(
            "an existing E9 destination must be addressed through its "
            f"canonical parent path: {output}"
        )
    # Reject an existing final-component symlink immediately, before any
    # source record is read, before staging is created, before the
    # publication lock is taken, and before any destination movement.  This
    # is a cheap early rejection only: the race-safe publication layer below
    # still independently refuses to follow or destructively touch a
    # final-component symlink that appears or changes after this check.
    try:
        _early_final_component_stat = os.lstat(output)
    except OSError:
        _early_final_component_stat = None
    if _early_final_component_stat is not None and stat.S_ISLNK(
        _early_final_component_stat.st_mode
    ):
        raise E9SpatialBankValidationError(
            f"refusing to overwrite a symbolic-link E9 destination: {output}"
        )
    manifest_source = _source_manifest(source)
    if manifest_source["split"] != split:
        raise E9SpatialBankValidationError("source split mismatch")
    extraction_source = manifest_source["extraction_config"]
    if expected_dino_source_commit is None:
        # No independent expectation was supplied: trust the source
        # manifest's own already-validated extractor commit directly
        # (the latest value actually recorded for this source) rather than
        # cross-checking it against an external assertion.
        expected_dino_source_commit = manifest_source["source_commit"]
    if expected_dino_checkpoint_sha256 is None:
        # No independent expectation was supplied: trust the source
        # manifest's own already-validated checkpoint identity directly
        # rather than cross-checking it against an external assertion.
        expected_dino_checkpoint_sha256 = extraction_source[
            "backbone_weights_sha256"
        ]
    if manifest_source["source_commit"] != expected_dino_source_commit:
        raise E9SpatialBankValidationError(
            "dense source DINO extractor commit does not match the expected "
            "DINO source commit"
        )
    if (
        extraction_source["backbone_weights_sha256"]
        != expected_dino_checkpoint_sha256
    ):
        raise E9SpatialBankValidationError(
            "dense source DINO checkpoint SHA256 does not match the expected "
            "DINO checkpoint"
        )
    if extraction_source["model"] != "dinov2_vitb14_reg":
        raise E9SpatialBankValidationError(
            "dense source DINO model must be dinov2_vitb14_reg"
        )
    source_images = int(manifest_source["source_images"])
    source_annotations = int(manifest_source["source_annotations"])
    available_images = int(manifest_source["selected_images"])
    available_annotations = int(manifest_source["selected_annotations"])
    source_is_pilot = (
        available_images < source_images
        or available_annotations < source_annotations
        or manifest_source["max_images"] is not None
    )
    pilot = max_images is not None or source_is_pilot
    if overwrite and not pilot:
        raise ValueError(
            "full production spatial banks are immutable; publish to a new output path"
        )
    artifacts = _source_artifacts(source, manifest_source)
    initial_hashes = {entry["path"]: entry["sha256"] for entry in artifacts}
    try:
        provenance = source_git_provenance(
            repository_root,
            allow_dirty_source=allow_dirty_source,
        )
    except ValueError as error:
        raise E9SpatialBankValidationError(
            "E9 spatial-bank construction requires a clean Git worktree; "
            "commit E9 first, or use --allow_dirty_source only for a pilot"
        ) from error
    if provenance["source_git_dirty"] and not pilot:
        raise E9SpatialBankValidationError("dirty source is permitted only for pilot banks")

    selected_images = min(available_images, max_images or available_images)
    estimated_bytes = selected_images * (256 * 768 * 2 + 256 * 2 + 8)
    estimated_full_bytes = source_images * (256 * 768 * 2 + 256 * 2 + 8)
    print(
        f"E9 spatial bank estimate: building {selected_images} images "
        f"({estimated_bytes / 1e9:.3f} GB tensor payload); source-full "
        f"estimate {source_images} images ({estimated_full_bytes / 1e9:.3f} GB)"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.e9-", dir=output.parent))
    _staging_identity_fd = os.open(staging, _NOFOLLOW_DIR_FLAGS)
    try:
        staging_identity = _fstat_identity(_staging_identity_fd)[:2]
    finally:
        os.close(_staging_identity_fd)
    shard_entries = []
    image_index = []
    try:
        pooled: list[torch.Tensor] = []
        priors: list[torch.Tensor] = []
        ids: list[int] = []
        seen: set[int] = set()
        seen_annotations: set[int] = set()
        selected_ids: list[int] = []
        selected_annotation_ids: list[int] = []
        selected_annotation_image_ids: list[int] = []
        source_count = 0

        def finalize_shard() -> None:
            nonlocal pooled, priors, ids
            if not ids:
                return
            shard_number = len(shard_entries)
            start = len(image_index)
            shard = {
                "patch_embeddings": torch.stack(pooled),
                "attention_priors": torch.stack(priors),
                "image_ids": torch.tensor(ids, dtype=torch.int64),
            }
            _validate_shard(shard, len(ids))
            name = f"{split}-{shard_number:06d}.pth"
            shard_path = staging / name
            torch.save(shard, shard_path)
            _fsync_file(shard_path)
            shard_entries.append(
                {
                    "name": name,
                    "sha256": sha256_file(shard_path),
                    "bytes": shard_path.stat().st_size,
                    "row_start": start,
                    "row_end": start + len(ids),
                    "row_count": len(ids),
                }
            )
            image_index.extend(
                {"image_id": image_id, "shard": shard_number, "row": row}
                for row, image_id in enumerate(ids)
            )
            pooled, priors, ids = [], [], []

        for record in _iter_source_records(source, manifest_source):
            if max_images is not None and len(selected_ids) >= selected_images:
                break
            source_count += 1
            validate_source_record(record)
            image_id = int(record["image_id"])
            if image_id in seen:
                raise E9SpatialBankValidationError(
                    f"duplicate source image ID {image_id}"
                )
            seen.add(image_id)
            annotation_ids = [int(value) for value in record["annotation_ids"]]
            duplicate_annotations = seen_annotations.intersection(annotation_ids)
            if duplicate_annotations:
                raise E9SpatialBankValidationError(
                    "annotation IDs appear in multiple source images: "
                    f"{sorted(duplicate_annotations)[:20]}"
                )
            seen_annotations.update(annotation_ids)
            selected_annotation_ids.extend(annotation_ids)
            selected_annotation_image_ids.extend([image_id] * len(annotation_ids))
            patch, prior = pool_source_image(
                record["patch_tokens"], record["self_attn_maps"]
            )
            pooled.append(patch.to(torch.float16).cpu())
            priors.append(prior.to(torch.float16).cpu())
            ids.append(image_id)
            selected_ids.append(image_id)
            if len(ids) == shard_rows:
                finalize_shard()
        finalize_shard()
        if max_images is None and source_count != available_images:
            raise E9SpatialBankValidationError(
                "source image count does not match manifest"
            )
        if len(selected_ids) != selected_images:
            raise E9SpatialBankValidationError("selected source coverage mismatch")
        source_subset_complete = (
            source_count == available_images
            and len(selected_annotation_ids) == available_annotations
            and _source_id_fingerprint(selected_ids)
            == manifest_source["selected_image_ids_sha256"]
            and _source_id_fingerprint(selected_annotation_ids)
            == manifest_source["selected_annotation_ids_sha256"]
            and not manifest_source["failed_image_ids"]
        )
        if max_images is None:
            if not source_subset_complete:
                raise E9SpatialBankValidationError(
                    "dense source exact image/annotation coverage is incomplete"
                )
            if manifest_source["complete"] != source_subset_complete:
                raise E9SpatialBankValidationError(
                    "dense source has a forged completeness state"
                )
        extraction = dict(manifest_source["extraction_config"])
        source_paths = [entry["path"] for entry in artifacts]
        source_hashes = [entry["sha256"] for entry in artifacts]
        selected_annotations = len(selected_annotation_ids)
        dataset_identity = dataset_identity_from_rows(
            split=split,
            image_ids=selected_annotation_image_ids,
            annotation_ids=selected_annotation_ids,
            source_image_count=source_images,
            source_annotation_count=source_annotations,
        )
        complete = (
            len(selected_ids) == selected_images
            and dataset_identity["selected_image_count"] == selected_images
            and dataset_identity["selected_annotation_count"] == selected_annotations
        )
        production_eligible = (
            complete
            and source_subset_complete
            and not pilot
            and selected_images == source_images
            and selected_annotations == source_annotations
            and not provenance["source_git_dirty"]
        )
        manifest = {
            "format_version": E9_SPATIAL_BANK_FORMAT,
            "split": split,
            "complete": complete,
            "is_pilot": pilot,
            "production_eligible": production_eligible,
            "source_images": source_images,
            "source_annotations": source_annotations,
            "selected_images": selected_images,
            "selected_annotations": selected_annotations,
            "image_id_fingerprint": _id_fingerprint(selected_ids),
            "dataset_identity": dataset_identity,
            "source_feature_paths": source_paths,
            "source_feature_sha256": source_hashes,
            "source_feature_format": CANONICAL_SOURCE_FEATURE_FORMAT,
            "source_feature_schema": sorted(SOURCE_RECORD_KEYS),
            "dino_identity": {
                "model": extraction["model"],
                "source_commit": manifest_source["source_commit"],
                "checkpoint_sha256": extraction["backbone_weights_sha256"],
            },
            "extraction": extraction,
            "geometry": dict(EXPECTED_GEOMETRY),
            "dtypes": dict(EXPECTED_DTYPES),
            "pooling_version": POOLING_VERSION,
            "attention_prior_version": ATTENTION_PRIOR_VERSION,
            "global_token_handling": GLOBAL_TOKEN_HANDLING,
            "shard_count": len(shard_entries),
            "shards": shard_entries,
            "image_index": image_index,
            **provenance,
            "builder_config": {
                "shard_rows": shard_rows,
                "max_images": max_images,
                "source_selected_images": available_images,
                "source_selected_annotations": available_annotations,
                "source_max_images": manifest_source["max_images"],
                "source_is_pilot": source_is_pilot,
                "source_complete": bool(manifest_source["complete"]),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = staging / MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)

        # Validate the complete staging tree before entering the final
        # provenance/publication boundary.
        validate_e9_spatial_bank(staging, require_production=False)

        def final_verification() -> None:
            for source_path, digest in initial_hashes.items():
                if sha256_file(source_path) != digest:
                    raise E9SpatialBankValidationError(
                        "source artifact changed during spatial-bank "
                        f"publication: {source_path}"
                    )
            try:
                _require_unchanged_git_provenance(
                    repository_root,
                    provenance,
                    allow_dirty_source=allow_dirty_source,
                )
            except ValueError as error:
                raise E9SpatialBankValidationError(
                    "E9 Git source provenance changed during spatial-bank "
                    "publication"
                ) from error

        _publish_directory(
            staging,
            output,
            overwrite=overwrite,
            final_verification=final_verification,
            staging_identity=staging_identity,
        )
        staging = None
        return {**validate_e9_spatial_bank(output, require_production=False), "estimated_bytes": estimated_bytes}
    finally:
        if staging is not None:
            _dispose_private_directory(
                staging, staging_identity, label="E9 staging directory",
            )


class E9SpatialBank:
    """Lazy image-ID view with a bounded LRU shard cache."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        require_production: bool = True,
        cache_shards: int = 1,
    ):
        if isinstance(cache_shards, bool) or not isinstance(cache_shards, int) or cache_shards <= 0:
            raise ValueError("cache_shards must be a positive integer")
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        initial_manifest_sha256 = sha256_file(manifest_path)
        self.validation = validate_e9_spatial_bank(
            self.root, require_production=require_production
        )
        if sha256_file(manifest_path) != initial_manifest_sha256:
            raise E9SpatialBankValidationError(
                "spatial manifest changed during validation"
            )
        self.manifest = _load_json(manifest_path)
        if sha256_file(manifest_path) != initial_manifest_sha256:
            raise E9SpatialBankValidationError(
                "spatial manifest changed while it was loaded"
            )
        self.manifest_sha256 = initial_manifest_sha256
        self._locations = {
            int(entry["image_id"]): (int(entry["shard"]), int(entry["row"]))
            for entry in self.manifest["image_index"]
        }
        if len(self._locations) != len(self.manifest["image_index"]):
            raise E9SpatialBankValidationError("duplicate image index rows")
        self.cache_shards = cache_shards
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self.shard_deserializations = 0

    def __len__(self) -> int:
        return len(self._locations)

    @property
    def image_ids(self) -> set[int]:
        return set(self._locations)

    def _shard(self, index: int) -> dict[str, torch.Tensor]:
        if index in self._cache:
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        metadata = self.manifest["shards"][index]
        path = self.root / metadata["name"]
        while len(self._cache) >= self.cache_shards:
            self._cache.popitem(last=False)
        value = load_spatial_shard(
            path,
            expected_sha256=metadata["sha256"],
            expected_bytes=metadata["bytes"],
        )
        self.shard_deserializations += 1
        self._cache[index] = value
        return value

    def get(self, image_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            shard_index, row = self._locations[int(image_id)]
        except KeyError as error:
            raise E9SpatialBankValidationError(f"missing spatial image ID {image_id}") from error
        shard = self._shard(shard_index)
        if int(shard["image_ids"][row]) != int(image_id):
            raise E9SpatialBankValidationError("spatial row/index mismatch")
        return shard["patch_embeddings"][row], shard["attention_priors"][row]

    def shard_index(self, image_id: int) -> int:
        try:
            return self._locations[int(image_id)][0]
        except KeyError as error:
            raise E9SpatialBankValidationError(
                f"missing spatial image ID {image_id}"
            ) from error


class SpatialUniqueImageBatchSampler(Iterable[list[int]]):
    """Deterministic unique-image sampling with bounded shard-local I/O.

    Shards and image rows are independently shuffled each epoch.  Rows from a
    shard remain contiguous in the stream, so the lazy spatial bank need not
    repeatedly reopen a large shard merely to preserve random batch order.
    """

    def __init__(
        self,
        image_ids: torch.Tensor,
        spatial_bank: E9SpatialBank,
        batch_size: int,
        seed: int = 42,
    ):
        if (
            not torch.is_tensor(image_ids)
            or image_ids.ndim != 1
            or image_ids.dtype != torch.int64
        ):
            raise ValueError("image_ids must be a one-dimensional int64 tensor")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        groups: dict[int, list[int]] = {}
        for row, image_id in enumerate(image_ids.cpu().tolist()):
            groups.setdefault(int(image_id), []).append(row)
        shard_images: dict[int, list[int]] = {}
        for image_id in sorted(groups):
            shard_images.setdefault(spatial_bank.shard_index(image_id), []).append(
                image_id
            )
        self._groups = {key: tuple(value) for key, value in groups.items()}
        self._shard_images = {
            key: tuple(value) for key, value in shard_images.items()
        }
        self._shards = tuple(sorted(self._shard_images))
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self._groups) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(
            len(self._shards), generator=generator
        ).tolist()
        batch: list[int] = []
        for shard_position in shard_order:
            shard = self._shards[shard_position]
            image_group = self._shard_images[shard]
            image_order = torch.randperm(
                len(image_group), generator=generator
            ).tolist()
            for image_position in image_order:
                image_id = image_group[image_position]
                rows = self._groups[image_id]
                caption_position = (self.seed + self.epoch) % len(rows)
                batch.append(rows[caption_position])
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


@dataclass(frozen=True)
class E9JoinedRow:
    query_row: int
    image_id: int
    annotation_id: int


def validate_query_spatial_join(
    query_bank: Mapping[str, Any],
    spatial_bank: E9SpatialBank,
    *,
    expected_split: str,
) -> list[E9JoinedRow]:
    metadata = query_bank["metadata"]
    if metadata["split_name"] != expected_split or spatial_bank.manifest["split"] != expected_split:
        raise E9SpatialBankValidationError("query/spatial split mismatch")
    if metadata["e3_config_sha256"] is None or metadata["e3_checkpoint_sha256"] is None:
        raise E9SpatialBankValidationError("query bank lacks E3 identity")
    image_ids = query_bank["image_ids"].tolist()
    annotation_ids = query_bank["annotation_ids"].tolist()
    if len(annotation_ids) != len(set(annotation_ids)):
        raise E9SpatialBankValidationError("query bank has duplicate annotations")
    rows = []
    for row, (image_id, annotation_id) in enumerate(zip(image_ids, annotation_ids)):
        if image_id not in spatial_bank.image_ids:
            raise E9SpatialBankValidationError(f"missing spatial image ID {image_id}")
        rows.append(E9JoinedRow(row, image_id, annotation_id))
    query_identity = dataset_identity_from_rows(
        split=expected_split,
        image_ids=image_ids,
        annotation_ids=annotation_ids,
        source_image_count=metadata["source_image_count"],
        source_annotation_count=metadata["source_annotation_count"],
    )
    spatial_identity = spatial_bank.manifest["dataset_identity"]
    if query_identity != spatial_identity:
        differing = sorted(
            key for key in DATASET_IDENTITY_KEYS
            if query_identity.get(key) != spatial_identity.get(key)
        )
        raise E9SpatialBankValidationError(
            "query/spatial cryptographic dataset identity mismatch: "
            f"{differing}; rebuild the E7 query bank or E9 spatial bank from "
            "the same annotation-to-image mapping"
        )
    if set(image_ids) != spatial_bank.image_ids:
        raise E9SpatialBankValidationError(
            "query/spatial image coverage differs"
        )
    return rows


def reject_train_validation_overlap(train: E9SpatialBank, validation: E9SpatialBank) -> None:
    overlap = train.image_ids.intersection(validation.image_ids)
    if overlap:
        raise E9SpatialBankValidationError(
            f"train/validation image overlap: {sorted(overlap)[:20]}"
        )


__all__ = [
    "E9_SPATIAL_BANK_FORMAT",
    "E9SpatialBankValidationError",
    "EXPECTED_GEOMETRY",
    "EXPECTED_DTYPES",
    "POOLING_VERSION",
    "ATTENTION_PRIOR_VERSION",
    "DATASET_IDENTITY_KEYS",
    "annotation_to_image_fingerprint",
    "dataset_identity_from_rows",
    "validate_source_record",
    "pool_source_image",
    "load_spatial_shard",
    "validate_e9_spatial_bank",
    "build_e9_spatial_bank",
    "E9SpatialBank",
    "SpatialUniqueImageBatchSampler",
    "E9JoinedRow",
    "validate_query_spatial_join",
    "reject_train_validation_overlap",
]
