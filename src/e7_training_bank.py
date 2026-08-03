"""Strict compact query-target bank contract for E7 adapter training."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from src.e6_prototype_bank import (
    CAPTION_EMBED_DIM,
    DEFAULT_NORM_TOLERANCE,
    EXPECTED_ATTENTION_HEADS,
    EXPECTED_E3_CHECKPOINT_NAME,
    EXPECTED_E3_CONFIG_NAME,
    EXPECTED_ROUTING_TEMPERATURE,
    ROUTED_DINO_EMBED_DIM,
    annotation_id_set_fingerprint,
    sha256_file,
)


FORMAT_VERSION = "talk2dino-e7-training-bank-v1"
MAPPED_QUERY_EMBED_DIM = ROUTED_DINO_EMBED_DIM
VALID_SPLITS = {"train", "val"}
VALIDATION_ROW_CHUNK_SIZE = 32768

_TOP_LEVEL_KEYS = {
    "caption_embeddings",
    "mapped_query_embeddings",
    "routed_target_embeddings",
    "image_ids",
    "annotation_ids",
    "metadata",
}
_METADATA_KEYS = {
    "format_version",
    "complete",
    "is_pilot",
    "split_name",
    "source_feature_path",
    "source_feature_sha256",
    "source_image_count",
    "source_annotation_count",
    "selected_annotation_count",
    "annotation_id_fingerprint",
    "e3_config_name",
    "e3_config_sha256",
    "e3_checkpoint_name",
    "e3_checkpoint_sha256",
    "routing_temperature",
    "source_git_commit",
    "source_git_dirty",
    "source_git_diff_sha256",
    "dimensions",
    "dtypes",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class E7TrainingBankValidationError(ValueError):
    """Raised when an E7 training bank violates its closed contract."""


def route_e7_annotation_batch(
    projection: torch.nn.Module,
    annotation_features: torch.Tensor,
    disentangled_self_attn: torch.Tensor,
    routing_temperature: float = EXPECTED_ROUTING_TEMPERATURE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized raw queries, mapped queries, and E3 routed targets."""

    try:
        routing_temperature = float(routing_temperature)
    except (TypeError, ValueError) as error:
        raise ValueError("routing_temperature must be finite and positive") from error
    if not math.isfinite(routing_temperature) or routing_temperature <= 0:
        raise ValueError("routing_temperature must be finite and positive")
    if (
        not torch.is_tensor(annotation_features)
        or tuple(annotation_features.shape[1:]) != (CAPTION_EMBED_DIM,)
        or annotation_features.ndim != 2
    ):
        shape = getattr(annotation_features, "shape", None)
        raise ValueError(
            "annotation_features must have shape [B, 512], got "
            f"{tuple(shape) if shape is not None else type(annotation_features)}"
        )
    if (
        not torch.is_tensor(disentangled_self_attn)
        or disentangled_self_attn.ndim != 3
        or tuple(disentangled_self_attn.shape[1:])
        != (EXPECTED_ATTENTION_HEADS, ROUTED_DINO_EMBED_DIM)
    ):
        shape = getattr(disentangled_self_attn, "shape", None)
        raise ValueError(
            "disentangled_self_attn must have shape [B, 12, 768], got "
            f"{tuple(shape) if shape is not None else type(disentangled_self_attn)}"
        )
    if annotation_features.shape[0] != disentangled_self_attn.shape[0]:
        raise ValueError("annotation and image-head batch sizes differ")
    if not annotation_features.is_floating_point():
        raise ValueError("annotation_features must be floating point")
    if not disentangled_self_attn.is_floating_point():
        raise ValueError("disentangled_self_attn must be floating point")

    annotation_features = annotation_features.float()
    heads = disentangled_self_attn.float()
    if not torch.isfinite(annotation_features).all():
        raise ValueError("annotation_features contains non-finite values")
    if not torch.isfinite(heads).all():
        raise ValueError("disentangled_self_attn contains non-finite values")
    if torch.any(annotation_features.norm(dim=-1) == 0):
        raise ValueError("annotation_features contains a zero-norm vector")

    caption_embeddings = F.normalize(annotation_features, dim=-1)
    # Deliberately project the original serialized ann_feats, not its normalized
    # form.  E3 contains biased nonlinear layers, so these are not equivalent.
    mapped_query_embeddings = projection.project_clip_txt(annotation_features)
    if (
        not torch.is_tensor(mapped_query_embeddings)
        or tuple(mapped_query_embeddings.shape)
        != (annotation_features.shape[0], MAPPED_QUERY_EMBED_DIM)
    ):
        shape = getattr(mapped_query_embeddings, "shape", None)
        raise ValueError(
            "E3 projection must return [B, 768], got "
            f"{tuple(shape) if shape is not None else type(mapped_query_embeddings)}"
        )
    mapped_query_embeddings = F.normalize(
        mapped_query_embeddings.float(), dim=-1
    )
    normalized_heads = F.normalize(heads, dim=-1)
    routing_weights = torch.softmax(
        torch.einsum(
            "bd,bhd->bh", mapped_query_embeddings, normalized_heads
        )
        / routing_temperature,
        dim=-1,
    )
    routed_target_embeddings = F.normalize(
        torch.einsum("bh,bhd->bd", routing_weights, normalized_heads),
        dim=-1,
    )
    for name, value in (
        ("caption embeddings", caption_embeddings),
        ("mapped query embeddings", mapped_query_embeddings),
        ("routed target embeddings", routed_target_embeddings),
    ):
        if not torch.isfinite(value).all() or torch.any(value.norm(dim=-1) == 0):
            raise ValueError(f"{name} are non-finite or zero norm")
    return (
        caption_embeddings,
        mapped_query_embeddings,
        routed_target_embeddings,
    )


def _validate_ids(value: Any, name: str, rows: int) -> list[int]:
    if (
        not torch.is_tensor(value)
        or value.device.type != "cpu"
        or value.dtype != torch.int64
        or tuple(value.shape) != (rows,)
    ):
        raise E7TrainingBankValidationError(
            f"{name} must be a CPU int64 tensor with shape [{rows}]"
        )
    return value.tolist()


def _validate_embeddings(
    value: Any,
    name: str,
    rows: int,
    dimension: int,
    norm_tolerance: float,
) -> tuple[float | None, float | None]:
    if (
        not torch.is_tensor(value)
        or value.device.type != "cpu"
        or value.dtype != torch.float16
        or tuple(value.shape) != (rows, dimension)
    ):
        raise E7TrainingBankValidationError(
            f"{name} must be a CPU float16 tensor with shape "
            f"[{rows}, {dimension}]"
        )
    minimum = maximum = None
    for start in range(0, rows, VALIDATION_ROW_CHUNK_SIZE):
        chunk = value[start : start + VALIDATION_ROW_CHUNK_SIZE]
        if not torch.isfinite(chunk).all():
            raise E7TrainingBankValidationError(f"{name} contains non-finite values")
        norms = chunk.float().norm(dim=-1)
        chunk_min = float(norms.min())
        chunk_max = float(norms.max())
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
        if torch.max(torch.abs(norms - 1.0)).item() > norm_tolerance:
            raise E7TrainingBankValidationError(
                f"{name} violates L2 norm tolerance {norm_tolerance}"
            )
    return minimum, maximum


def _require_sha(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata[key]
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise E7TrainingBankValidationError(
            f"metadata.{key} must be a lowercase SHA256 digest"
        )
    return value


def validate_e7_training_bank(
    bank: Mapping[str, Any],
    *,
    allow_pilot: bool = False,
    allow_dirty_source: bool = False,
    require_complete: bool = True,
    expected_split: str | None = None,
    expected_config_path: os.PathLike[str] | str | None = None,
    expected_checkpoint_path: os.PathLike[str] | str | None = None,
    expected_source_features_path: os.PathLike[str] | str | None = None,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> dict[str, Any]:
    """Validate a bank without accepting legacy or ambiguous metadata."""

    if not isinstance(bank, Mapping):
        raise E7TrainingBankValidationError("E7 training bank must be a mapping")
    if set(bank) != _TOP_LEVEL_KEYS:
        raise E7TrainingBankValidationError(
            f"top-level schema mismatch: expected {sorted(_TOP_LEVEL_KEYS)}, "
            f"got {sorted(bank)}"
        )
    metadata = bank["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != _METADATA_KEYS:
        actual = sorted(metadata) if isinstance(metadata, Mapping) else type(metadata)
        raise E7TrainingBankValidationError(
            f"closed metadata schema mismatch: {actual}"
        )
    if metadata["format_version"] != FORMAT_VERSION:
        raise E7TrainingBankValidationError(
            f"unsupported format_version {metadata['format_version']!r}"
        )
    for key in ("complete", "is_pilot", "source_git_dirty"):
        if type(metadata[key]) is not bool:
            raise E7TrainingBankValidationError(f"metadata.{key} must be boolean")
    if require_complete and not metadata["complete"]:
        raise E7TrainingBankValidationError("E7 training bank is incomplete")
    if metadata["is_pilot"] and not allow_pilot:
        raise E7TrainingBankValidationError("E7 training bank is a pilot")
    if metadata["source_git_dirty"] and not allow_dirty_source:
        raise E7TrainingBankValidationError(
            "E7 training bank was built from a dirty source tree"
        )
    if metadata["source_git_dirty"]:
        _require_sha(metadata, "source_git_diff_sha256")
        if metadata["complete"] and not metadata["is_pilot"]:
            raise E7TrainingBankValidationError(
                "a complete production bank cannot have dirty provenance"
            )
    elif metadata["source_git_diff_sha256"] is not None:
        raise E7TrainingBankValidationError(
            "clean provenance requires source_git_diff_sha256=null"
        )

    split = metadata["split_name"]
    if split not in VALID_SPLITS:
        raise E7TrainingBankValidationError("metadata.split_name must be train or val")
    if expected_split is not None and split != expected_split:
        raise E7TrainingBankValidationError(
            f"bank split mismatch: {split!r} != {expected_split!r}"
        )
    for key in (
        "source_image_count",
        "source_annotation_count",
        "selected_annotation_count",
    ):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise E7TrainingBankValidationError(
                f"metadata.{key} must be a non-negative integer"
            )
    rows = metadata["selected_annotation_count"]
    if rows > metadata["source_annotation_count"]:
        raise E7TrainingBankValidationError("selected annotations exceed source")
    if not metadata["is_pilot"] and rows != metadata["source_annotation_count"]:
        raise E7TrainingBankValidationError(
            "a full bank must contain all source annotations"
        )
    annotation_ids = _validate_ids(bank["annotation_ids"], "annotation_ids", rows)
    image_ids = _validate_ids(bank["image_ids"], "image_ids", rows)
    if len(annotation_ids) != len(set(annotation_ids)):
        raise E7TrainingBankValidationError("annotation_ids contains duplicates")
    if len(set(image_ids)) > metadata["source_image_count"]:
        raise E7TrainingBankValidationError("image ID count exceeds source images")
    fingerprint = annotation_id_set_fingerprint(bank["annotation_ids"])
    if metadata["annotation_id_fingerprint"] != fingerprint:
        raise E7TrainingBankValidationError("annotation ID fingerprint mismatch")

    norm_ranges = {}
    for name, dimension in (
        ("caption_embeddings", CAPTION_EMBED_DIM),
        ("mapped_query_embeddings", MAPPED_QUERY_EMBED_DIM),
        ("routed_target_embeddings", ROUTED_DINO_EMBED_DIM),
    ):
        norm_ranges[name] = _validate_embeddings(
            bank[name], name, rows, dimension, norm_tolerance
        )
    expected_dimensions = {
        "caption_embeddings": CAPTION_EMBED_DIM,
        "mapped_query_embeddings": MAPPED_QUERY_EMBED_DIM,
        "routed_target_embeddings": ROUTED_DINO_EMBED_DIM,
    }
    expected_dtypes = {
        "caption_embeddings": "float16",
        "mapped_query_embeddings": "float16",
        "routed_target_embeddings": "float16",
        "image_ids": "int64",
        "annotation_ids": "int64",
    }
    if metadata["dimensions"] != expected_dimensions:
        raise E7TrainingBankValidationError("metadata dimensions mismatch")
    if metadata["dtypes"] != expected_dtypes:
        raise E7TrainingBankValidationError("metadata dtypes mismatch")
    if metadata["e3_config_name"] != EXPECTED_E3_CONFIG_NAME:
        raise E7TrainingBankValidationError("unexpected E3 configuration name")
    if metadata["e3_checkpoint_name"] != EXPECTED_E3_CHECKPOINT_NAME:
        raise E7TrainingBankValidationError("unexpected E3 checkpoint name")
    if (
        not isinstance(metadata["source_feature_path"], str)
        or not metadata["source_feature_path"]
    ):
        raise E7TrainingBankValidationError(
            "metadata.source_feature_path must be a non-empty string"
        )
    config_sha = _require_sha(metadata, "e3_config_sha256")
    checkpoint_sha = _require_sha(metadata, "e3_checkpoint_sha256")
    source_sha = _require_sha(metadata, "source_feature_sha256")
    commit = metadata["source_git_commit"]
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise E7TrainingBankValidationError("invalid source Git commit")
    try:
        temperature = float(metadata["routing_temperature"])
    except (TypeError, ValueError) as error:
        raise E7TrainingBankValidationError("invalid routing temperature") from error
    if not math.isfinite(temperature) or not math.isclose(
        temperature, EXPECTED_ROUTING_TEMPERATURE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise E7TrainingBankValidationError("routing temperature must equal 0.10")

    for path_value, digest, label in (
        (expected_config_path, config_sha, "E3 configuration"),
        (expected_checkpoint_path, checkpoint_sha, "E3 checkpoint"),
        (expected_source_features_path, source_sha, "source feature archive"),
    ):
        if path_value is not None:
            path = Path(path_value)
            if not path.is_file() or sha256_file(path) != digest:
                raise E7TrainingBankValidationError(f"{label} SHA256 mismatch")
            if label == "E3 configuration" and path.name != metadata["e3_config_name"]:
                raise E7TrainingBankValidationError(
                    "E3 configuration name mismatch"
                )
            if label == "E3 checkpoint" and path.name != metadata["e3_checkpoint_name"]:
                raise E7TrainingBankValidationError("E3 checkpoint name mismatch")

    return {
        "format_version": FORMAT_VERSION,
        "complete": metadata["complete"],
        "is_pilot": metadata["is_pilot"],
        "split_name": split,
        "entries": rows,
        "unique_images": len(set(image_ids)),
        "annotation_id_fingerprint": fingerprint,
        "source_feature_sha256": source_sha,
        "e3_config_sha256": config_sha,
        "e3_checkpoint_sha256": checkpoint_sha,
        "source_git_dirty": metadata["source_git_dirty"],
        "norm_ranges": norm_ranges,
    }


def load_e7_training_bank(
    path: os.PathLike[str] | str,
    **validation_kwargs: Any,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"E7 training bank does not exist: {path}")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    validate_e7_training_bank(bank, **validation_kwargs)
    return bank


@dataclass(frozen=True)
class E7BankIdentity:
    format_version: str
    split_name: str
    source_feature_sha256: str
    annotation_id_fingerprint: str
    selected_annotation_count: int
    e3_config_sha256: str
    e3_checkpoint_sha256: str
    routing_temperature: float
    source_git_commit: str

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "E7BankIdentity":
        return cls(
            format_version=metadata["format_version"],
            split_name=metadata["split_name"],
            source_feature_sha256=metadata["source_feature_sha256"],
            annotation_id_fingerprint=metadata["annotation_id_fingerprint"],
            selected_annotation_count=metadata["selected_annotation_count"],
            e3_config_sha256=metadata["e3_config_sha256"],
            e3_checkpoint_sha256=metadata["e3_checkpoint_sha256"],
            routing_temperature=float(metadata["routing_temperature"]),
            source_git_commit=metadata["source_git_commit"],
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
