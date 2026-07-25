"""Schema, routing, and validation helpers for E6 RGTP prototype banks.

The bank is deliberately a plain, CPU-resident torch artifact.  It contains
only the two normalized embedding matrices and their integer identifiers; the
E3 projection and the source image-head tensors are never serialized into it.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


FORMAT_VERSION = "talk2dino-e6-rgtp-v3"
EXPECTED_E3_CONFIG_NAME = (
    "vitb_mlp_infonce_paired_soft_routing_tau010.yaml"
)
EXPECTED_E3_CHECKPOINT_NAME = (
    "vitb_mlp_infonce_paired_soft_routing_tau010.pth"
)
EXPECTED_ROUTING_TEMPERATURE = 0.10
CAPTION_EMBED_DIM = 512
ROUTED_DINO_EMBED_DIM = 768
EXPECTED_ATTENTION_HEADS = 12
DEFAULT_NORM_TOLERANCE = 5e-3
VALIDATION_ROW_CHUNK_SIZE = 32768

_REQUIRED_TOP_LEVEL_KEYS = {
    "caption_embeddings",
    "routed_dino_embeddings",
    "image_ids",
    "annotation_ids",
    "metadata",
}
_REQUIRED_METADATA_KEYS = {
    "format_version",
    "complete",
    "is_pilot",
    "source_feature_path",
    "source_feature_sha256",
    "source_image_count",
    "source_annotation_count",
    "selected_annotation_count",
    "e3_config_name",
    "e3_config_sha256",
    "e3_checkpoint_name",
    "checkpoint_sha256",
    "routing_temperature",
    "source_git_commit",
    "source_git_dirty",
    "source_git_diff_sha256",
    "dimensions",
    "dtypes",
    "annotation_id_fingerprint",
}
_OPTIONAL_METADATA_KEYS: set[str] = set()
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class PrototypeBankValidationError(ValueError):
    """Raised when an E6 prototype bank violates its on-disk contract."""


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of *path* without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _integer_ids(ids: Any, field_name: str) -> list[int]:
    if torch.is_tensor(ids):
        if ids.ndim != 1:
            raise PrototypeBankValidationError(
                f"{field_name} must be one-dimensional, got {tuple(ids.shape)}"
            )
        if ids.dtype != torch.int64:
            raise PrototypeBankValidationError(
                f"{field_name} must have dtype torch.int64, got {ids.dtype}"
            )
        values = ids.detach().cpu().tolist()
    else:
        try:
            values = list(ids)
        except TypeError as error:
            raise PrototypeBankValidationError(
                f"{field_name} must be an iterable of integer IDs"
            ) from error

    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise PrototypeBankValidationError(
                f"{field_name} contains a non-integer ID: {value!r}"
            )
        normalized_value = int(value)
        if not -(2**63) <= normalized_value < 2**63:
            raise PrototypeBankValidationError(
                f"{field_name} contains an ID outside the int64 range: "
                f"{normalized_value}"
            )
        normalized.append(normalized_value)
    return normalized


def annotation_id_set_fingerprint(annotation_ids: Any) -> str:
    """Fingerprint an exact annotation-ID set independently of input order."""

    ids = _integer_ids(annotation_ids, "annotation_ids")
    if len(ids) != len(set(ids)):
        raise PrototypeBankValidationError(
            "annotation_ids contains duplicates; an ID-set fingerprint would "
            "hide duplicate annotations"
        )
    digest = hashlib.sha256()
    for annotation_id in sorted(ids):
        digest.update(str(annotation_id).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def route_annotation_batch(
    projection: torch.nn.Module,
    annotation_features: torch.Tensor,
    disentangled_self_attn: torch.Tensor,
    routing_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized CLIP and E3-routed DINO bank vectors in float32.

    This is the Phase-1 E6 equation, kept independent from the retrieval and
    segmentation code so that it can be tested against a direct loop
    implementation.
    """

    try:
        routing_temperature = float(routing_temperature)
    except (TypeError, ValueError) as error:
        raise ValueError("routing_temperature must be a finite positive value") from error
    if not math.isfinite(routing_temperature) or routing_temperature <= 0:
        raise ValueError(
            "routing_temperature must be finite and strictly positive, got "
            f"{routing_temperature}"
        )
    if not torch.is_tensor(annotation_features) or annotation_features.ndim != 2:
        shape = getattr(annotation_features, "shape", None)
        raise ValueError(
            "annotation_features must have shape [B, 512], got "
            f"{tuple(shape) if shape is not None else type(annotation_features)}"
        )
    if annotation_features.shape[1] != CAPTION_EMBED_DIM:
        raise ValueError(
            "annotation_features must have shape [B, 512], got "
            f"{tuple(annotation_features.shape)}"
        )
    if (
        not torch.is_tensor(disentangled_self_attn)
        or disentangled_self_attn.ndim != 3
    ):
        shape = getattr(disentangled_self_attn, "shape", None)
        raise ValueError(
            "disentangled_self_attn must have shape [B, 12, 768], got "
            f"{tuple(shape) if shape is not None else type(disentangled_self_attn)}"
        )
    if tuple(disentangled_self_attn.shape[1:]) != (
        EXPECTED_ATTENTION_HEADS,
        ROUTED_DINO_EMBED_DIM,
    ):
        raise ValueError(
            "disentangled_self_attn must have shape [B, 12, 768], got "
            f"{tuple(disentangled_self_attn.shape)}"
        )
    if disentangled_self_attn.shape[0] != annotation_features.shape[0]:
        raise ValueError(
            "annotation and image-head batch sizes differ: "
            f"{annotation_features.shape[0]} != {disentangled_self_attn.shape[0]}"
        )
    if not annotation_features.is_floating_point():
        raise ValueError("annotation_features must be a floating-point tensor")
    if not disentangled_self_attn.is_floating_point():
        raise ValueError("disentangled_self_attn must be a floating-point tensor")

    annotation_features = annotation_features.float()
    disentangled_self_attn = disentangled_self_attn.float()
    if not torch.isfinite(annotation_features).all():
        raise ValueError("annotation_features contains non-finite values")
    if not torch.isfinite(disentangled_self_attn).all():
        raise ValueError("disentangled_self_attn contains non-finite values")

    caption_embeddings = F.normalize(annotation_features, p=2, dim=-1)
    projected_text = projection.project_clip_txt(annotation_features)
    if (
        not torch.is_tensor(projected_text)
        or tuple(projected_text.shape)
        != (annotation_features.shape[0], ROUTED_DINO_EMBED_DIM)
    ):
        shape = getattr(projected_text, "shape", None)
        raise ValueError(
            "E3 projection must produce shape [B, 768], got "
            f"{tuple(shape) if shape is not None else type(projected_text)}"
        )
    projected_text = F.normalize(projected_text.float(), p=2, dim=-1)
    normalized_heads = F.normalize(disentangled_self_attn, p=2, dim=-1)
    affinities = torch.einsum("bd,bhd->bh", projected_text, normalized_heads)
    routing_weights = torch.softmax(
        affinities / routing_temperature,
        dim=-1,
    )
    routed_dino_embeddings = torch.einsum(
        "bh,bhd->bd",
        routing_weights,
        normalized_heads,
    )
    routed_dino_embeddings = F.normalize(
        routed_dino_embeddings,
        p=2,
        dim=-1,
    )

    if not torch.isfinite(caption_embeddings).all():
        raise ValueError("normalized caption embeddings contain non-finite values")
    if not torch.isfinite(routed_dino_embeddings).all():
        raise ValueError("routed DINO embeddings contain non-finite values")
    if torch.any(caption_embeddings.norm(dim=-1) == 0):
        raise ValueError("annotation_features contains a zero-norm vector")
    if torch.any(routed_dino_embeddings.norm(dim=-1) == 0):
        raise ValueError("routing produced a zero-norm DINO vector")
    return caption_embeddings, routed_dino_embeddings


def _require_metadata_integer(
    metadata: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 0,
) -> int:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrototypeBankValidationError(
            f"metadata.{key} must be an integer, got {value!r}"
        )
    if value < minimum:
        raise PrototypeBankValidationError(
            f"metadata.{key} must be at least {minimum}, got {value}"
        )
    return value


def _validate_embedding_tensor(
    value: Any,
    *,
    field_name: str,
    rows: int,
    columns: int,
    norm_tolerance: float,
) -> tuple[float | None, float | None]:
    if not torch.is_tensor(value):
        raise PrototypeBankValidationError(f"{field_name} must be a tensor")
    if value.device.type != "cpu":
        raise PrototypeBankValidationError(
            f"{field_name} must be CPU-resident in the serialized bank"
        )
    if value.dtype != torch.float16:
        raise PrototypeBankValidationError(
            f"{field_name} must have dtype torch.float16, got {value.dtype}"
        )
    if tuple(value.shape) != (rows, columns):
        raise PrototypeBankValidationError(
            f"{field_name} must have shape [{rows}, {columns}], got "
            f"{tuple(value.shape)}"
        )
    norm_min: float | None = None
    norm_max: float | None = None
    for start in range(0, rows, VALIDATION_ROW_CHUNK_SIZE):
        chunk = value[start : start + VALIDATION_ROW_CHUNK_SIZE]
        if not torch.isfinite(chunk).all():
            raise PrototypeBankValidationError(
                f"{field_name} contains non-finite values"
            )
        norms = chunk.float().norm(dim=-1)
        chunk_min = float(norms.min())
        chunk_max = float(norms.max())
        norm_min = chunk_min if norm_min is None else min(norm_min, chunk_min)
        norm_max = chunk_max if norm_max is None else max(norm_max, chunk_max)
        if torch.max(torch.abs(norms - 1.0)).item() > norm_tolerance:
            raise PrototypeBankValidationError(
                f"{field_name} contains vectors outside the L2 norm tolerance "
                f"{norm_tolerance}"
            )
    return norm_min, norm_max


def validate_prototype_bank(
    bank: Mapping[str, Any],
    *,
    allow_pilot: bool = False,
    allow_dirty_source: bool = False,
    require_complete: bool = True,
    expected_config_path: os.PathLike[str] | str | None = None,
    expected_checkpoint_path: os.PathLike[str] | str | None = None,
    expected_source_features_path: os.PathLike[str] | str | None = None,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> dict[str, Any]:
    """Validate an in-memory bank and return a compact validation summary."""

    if not isinstance(bank, Mapping):
        raise PrototypeBankValidationError("prototype bank must be a mapping")
    try:
        norm_tolerance = float(norm_tolerance)
    except (TypeError, ValueError) as error:
        raise PrototypeBankValidationError(
            "norm_tolerance must be a finite non-negative value"
        ) from error
    if not math.isfinite(norm_tolerance) or norm_tolerance < 0:
        raise PrototypeBankValidationError(
            "norm_tolerance must be a finite non-negative value"
        )
    missing_keys = sorted(_REQUIRED_TOP_LEVEL_KEYS.difference(bank))
    if missing_keys:
        raise PrototypeBankValidationError(
            f"prototype bank is missing top-level keys: {missing_keys}"
        )
    unexpected_keys = sorted(set(bank).difference(_REQUIRED_TOP_LEVEL_KEYS))
    if unexpected_keys:
        raise PrototypeBankValidationError(
            f"prototype bank has unexpected top-level keys: {unexpected_keys}"
        )

    metadata = bank["metadata"]
    if not isinstance(metadata, Mapping):
        raise PrototypeBankValidationError("metadata must be a mapping")
    if "format_version" not in metadata:
        raise PrototypeBankValidationError(
            "metadata is missing keys: ['format_version']"
        )
    if metadata["format_version"] != FORMAT_VERSION:
        legacy_hint = (
            "; v1/v2 banks must be rebuilt"
            if metadata["format_version"]
            in {"talk2dino-e6-rgtp-v1", "talk2dino-e6-rgtp-v2"}
            else ""
        )
        raise PrototypeBankValidationError(
            "unsupported metadata.format_version: "
            f"{metadata['format_version']!r}; expected {FORMAT_VERSION!r}"
            f"{legacy_hint}"
        )
    missing_metadata = sorted(_REQUIRED_METADATA_KEYS.difference(metadata))
    if missing_metadata:
        raise PrototypeBankValidationError(
            f"metadata is missing keys: {missing_metadata}"
        )
    allowed_metadata = _REQUIRED_METADATA_KEYS | _OPTIONAL_METADATA_KEYS
    unexpected_metadata = sorted(set(metadata).difference(allowed_metadata))
    if unexpected_metadata:
        raise PrototypeBankValidationError(
            f"metadata has unknown keys: {unexpected_metadata}"
        )
    if type(metadata["complete"]) is not bool:
        raise PrototypeBankValidationError("metadata.complete must be a boolean")
    if type(metadata["is_pilot"]) is not bool:
        raise PrototypeBankValidationError("metadata.is_pilot must be a boolean")
    if type(metadata["source_git_dirty"]) is not bool:
        raise PrototypeBankValidationError(
            "metadata.source_git_dirty must be a boolean"
        )
    source_git_dirty = metadata["source_git_dirty"]
    source_git_diff_sha256 = metadata["source_git_diff_sha256"]
    if source_git_dirty:
        if (
            not isinstance(source_git_diff_sha256, str)
            or _HEX_SHA256.fullmatch(source_git_diff_sha256) is None
        ):
            raise PrototypeBankValidationError(
                "a dirty-source bank must record a lowercase "
                "metadata.source_git_diff_sha256"
            )
        if not allow_dirty_source:
            raise PrototypeBankValidationError(
                "prototype bank was built from a dirty Git worktree and is "
                "not permitted for evaluation"
            )
        if not metadata["is_pilot"] and metadata["complete"]:
            raise PrototypeBankValidationError(
                "a complete production bank cannot have source_git_dirty=true"
            )
    elif source_git_diff_sha256 is not None:
        raise PrototypeBankValidationError(
            "a clean-source bank must set metadata.source_git_diff_sha256 to null"
        )
    if require_complete and not metadata["complete"]:
        raise PrototypeBankValidationError(
            "prototype bank is incomplete; pass require_complete=False only for "
            "explicit inspection"
        )
    if metadata["is_pilot"] and not allow_pilot:
        raise PrototypeBankValidationError(
            "prototype bank is a pilot; pass allow_pilot=True only for explicit "
            "pilot inspection"
        )

    source_image_count = _require_metadata_integer(
        metadata, "source_image_count", minimum=0
    )
    source_annotation_count = _require_metadata_integer(
        metadata, "source_annotation_count", minimum=0
    )
    selected_annotation_count = _require_metadata_integer(
        metadata, "selected_annotation_count", minimum=0
    )
    if selected_annotation_count > source_annotation_count:
        raise PrototypeBankValidationError(
            "metadata.selected_annotation_count exceeds source_annotation_count"
        )
    if not metadata["is_pilot"] and (
        selected_annotation_count != source_annotation_count
    ):
        raise PrototypeBankValidationError(
            "a non-pilot bank must select every source annotation"
        )

    for key in ("annotation_ids", "image_ids"):
        value = bank[key]
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.int64
            or value.ndim != 1
        ):
            raise PrototypeBankValidationError(
                f"{key} must be a one-dimensional CPU torch.int64 tensor"
            )
    annotation_ids = _integer_ids(bank["annotation_ids"], "annotation_ids")
    image_ids = _integer_ids(bank["image_ids"], "image_ids")
    rows = len(annotation_ids)
    if len(image_ids) != rows:
        raise PrototypeBankValidationError(
            "image_ids and annotation_ids have different lengths"
        )
    if rows != selected_annotation_count:
        raise PrototypeBankValidationError(
            "stored row count does not equal metadata.selected_annotation_count: "
            f"{rows} != {selected_annotation_count}"
        )
    if len(annotation_ids) != len(set(annotation_ids)):
        raise PrototypeBankValidationError("annotation_ids contains duplicates")
    if len(set(image_ids)) > source_image_count:
        raise PrototypeBankValidationError(
            "the number of referenced image IDs exceeds source_image_count"
        )

    caption_norm_min, caption_norm_max = _validate_embedding_tensor(
        bank["caption_embeddings"],
        field_name="caption_embeddings",
        rows=rows,
        columns=CAPTION_EMBED_DIM,
        norm_tolerance=norm_tolerance,
    )
    routed_norm_min, routed_norm_max = _validate_embedding_tensor(
        bank["routed_dino_embeddings"],
        field_name="routed_dino_embeddings",
        rows=rows,
        columns=ROUTED_DINO_EMBED_DIM,
        norm_tolerance=norm_tolerance,
    )

    expected_dimensions = {
        "caption_embeddings": CAPTION_EMBED_DIM,
        "routed_dino_embeddings": ROUTED_DINO_EMBED_DIM,
    }
    if metadata["dimensions"] != expected_dimensions:
        raise PrototypeBankValidationError(
            "metadata.dimensions does not match the bank tensor contract"
        )
    expected_dtypes = {
        "caption_embeddings": "float16",
        "routed_dino_embeddings": "float16",
        "image_ids": "int64",
        "annotation_ids": "int64",
    }
    if metadata["dtypes"] != expected_dtypes:
        raise PrototypeBankValidationError(
            "metadata.dtypes does not match the bank tensor contract"
        )

    for key in (
        "source_feature_path",
        "e3_config_name",
        "e3_checkpoint_name",
    ):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise PrototypeBankValidationError(
                f"metadata.{key} must be a non-empty string"
            )
    if metadata["e3_config_name"] != EXPECTED_E3_CONFIG_NAME:
        raise PrototypeBankValidationError(
            "metadata.e3_config_name does not identify the required E3 "
            f"configuration: {metadata['e3_config_name']!r} != "
            f"{EXPECTED_E3_CONFIG_NAME!r}"
        )
    config_sha256 = metadata["e3_config_sha256"]
    if (
        not isinstance(config_sha256, str)
        or _HEX_SHA256.fullmatch(config_sha256) is None
    ):
        raise PrototypeBankValidationError(
            "metadata.e3_config_sha256 must be a lowercase SHA256 digest"
        )
    source_feature_sha256 = metadata["source_feature_sha256"]
    if (
        not isinstance(source_feature_sha256, str)
        or _HEX_SHA256.fullmatch(source_feature_sha256) is None
    ):
        raise PrototypeBankValidationError(
            "metadata.source_feature_sha256 must be a lowercase SHA256 digest"
        )
    if metadata["e3_checkpoint_name"] != EXPECTED_E3_CHECKPOINT_NAME:
        raise PrototypeBankValidationError(
            "metadata.e3_checkpoint_name does not identify the required E3 "
            f"checkpoint: {metadata['e3_checkpoint_name']!r} != "
            f"{EXPECTED_E3_CHECKPOINT_NAME!r}"
        )
    checkpoint_sha256 = metadata["checkpoint_sha256"]
    if (
        not isinstance(checkpoint_sha256, str)
        or _HEX_SHA256.fullmatch(checkpoint_sha256) is None
    ):
        raise PrototypeBankValidationError(
            "metadata.checkpoint_sha256 must be a lowercase SHA256 digest"
        )
    source_git_commit = metadata["source_git_commit"]
    if (
        not isinstance(source_git_commit, str)
        or _GIT_COMMIT.fullmatch(source_git_commit) is None
    ):
        raise PrototypeBankValidationError(
            "metadata.source_git_commit must be a full lowercase Git commit"
        )
    if isinstance(metadata["routing_temperature"], bool):
        raise PrototypeBankValidationError(
            "metadata.routing_temperature must be numeric"
        )
    try:
        routing_temperature = float(metadata["routing_temperature"])
    except (TypeError, ValueError) as error:
        raise PrototypeBankValidationError(
            "metadata.routing_temperature must be numeric"
        ) from error
    if not math.isfinite(routing_temperature) or not math.isclose(
        routing_temperature,
        EXPECTED_ROUTING_TEMPERATURE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PrototypeBankValidationError(
            "metadata.routing_temperature must be the E3 value 0.10"
        )

    actual_fingerprint = annotation_id_set_fingerprint(bank["annotation_ids"])
    if metadata["annotation_id_fingerprint"] != actual_fingerprint:
        raise PrototypeBankValidationError(
            "metadata.annotation_id_fingerprint does not match annotation_ids"
        )

    if expected_config_path is not None:
        expected_config_path = Path(expected_config_path)
        if not expected_config_path.is_file():
            raise PrototypeBankValidationError(
                f"expected E3 configuration does not exist: {expected_config_path}"
            )
        expected_config_digest = sha256_file(expected_config_path)
        if config_sha256 != expected_config_digest:
            raise PrototypeBankValidationError(
                "prototype bank E3 configuration SHA256 does not match the "
                f"expected configuration {expected_config_path}"
            )
        if metadata["e3_config_name"] != expected_config_path.name:
            raise PrototypeBankValidationError(
                "prototype bank E3 configuration name does not match the "
                f"expected configuration: {metadata['e3_config_name']!r} != "
                f"{expected_config_path.name!r}"
            )

    if expected_checkpoint_path is not None:
        expected_checkpoint_path = Path(expected_checkpoint_path)
        if not expected_checkpoint_path.is_file():
            raise PrototypeBankValidationError(
                f"expected E3 checkpoint does not exist: {expected_checkpoint_path}"
            )
        expected_digest = sha256_file(expected_checkpoint_path)
        if checkpoint_sha256 != expected_digest:
            raise PrototypeBankValidationError(
                "prototype bank checkpoint SHA256 does not match the expected "
                f"checkpoint {expected_checkpoint_path}"
            )
        if metadata["e3_checkpoint_name"] != expected_checkpoint_path.name:
            raise PrototypeBankValidationError(
                "prototype bank checkpoint name does not match the expected "
                f"checkpoint: {metadata['e3_checkpoint_name']!r} != "
                f"{expected_checkpoint_path.name!r}"
            )

    if expected_source_features_path is not None:
        expected_source_features_path = Path(expected_source_features_path)
        if not expected_source_features_path.is_file():
            raise PrototypeBankValidationError(
                "expected source feature archive does not exist: "
                f"{expected_source_features_path}"
            )
        expected_source_digest = sha256_file(expected_source_features_path)
        if source_feature_sha256 != expected_source_digest:
            raise PrototypeBankValidationError(
                "prototype bank source feature SHA256 does not match the "
                f"expected archive {expected_source_features_path}"
            )

    return {
        "format_version": metadata["format_version"],
        "complete": metadata["complete"],
        "is_pilot": metadata["is_pilot"],
        "entries": rows,
        "unique_images": len(set(image_ids)),
        "source_images": source_image_count,
        "source_annotations": source_annotation_count,
        "source_git_commit": source_git_commit,
        "source_git_dirty": source_git_dirty,
        "source_git_diff_sha256": source_git_diff_sha256,
        "source_feature_sha256": source_feature_sha256,
        "e3_config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "annotation_id_fingerprint": actual_fingerprint,
        "caption_norm_min": caption_norm_min,
        "caption_norm_max": caption_norm_max,
        "routed_dino_norm_min": routed_norm_min,
        "routed_dino_norm_max": routed_norm_max,
    }


def load_prototype_bank(
    path: os.PathLike[str] | str,
    *,
    allow_pilot: bool = False,
    allow_dirty_source: bool = False,
    require_complete: bool = True,
    expected_config_path: os.PathLike[str] | str | None = None,
    expected_checkpoint_path: os.PathLike[str] | str | None = None,
    expected_source_features_path: os.PathLike[str] | str | None = None,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> dict[str, Any]:
    """Load a CPU bank, enforcing full/non-pilot safety by default."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"E6 prototype bank does not exist: {path}")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    validate_prototype_bank(
        bank,
        allow_pilot=allow_pilot,
        allow_dirty_source=allow_dirty_source,
        require_complete=require_complete,
        expected_config_path=expected_config_path,
        expected_checkpoint_path=expected_checkpoint_path,
        expected_source_features_path=expected_source_features_path,
        norm_tolerance=norm_tolerance,
    )
    return bank
