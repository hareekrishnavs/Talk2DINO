"""Caption-conditioned sparse region alignment for Talk2DINO E9."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.e6_prototype_bank import sha256_file
from src.e9_spatial_bank import (
    ATTENTION_PRIOR_VERSION,
    CANONICAL_EXTRACTION,
    DINO_IDENTITY_KEYS,
    E9_SPATIAL_BANK_FORMAT,
    EXPECTED_GEOMETRY,
    EXTRACTION_KEYS,
    GLOBAL_TOKEN_HANDLING,
    POOLING_VERSION,
)

E9_ADAPTER_CHECKPOINT_FORMAT = "talk2dino-e9-sparse-region-adapter-v2"
GATE_PRIOR_FEATURE_VERSION = "max-valid-normalized-attention-v1"
E9_CHECKPOINT_KEYS = {
    "format_version",
    "adapter_state_dict",
    "architecture_config",
    "training_config",
    "epoch",
    "best_validation_metric",
    "query_bank_identities",
    "spatial_bank_identities",
    "e3_identity",
    "source_feature_identities",
    "run_identity",
    "source_git_provenance",
    "diagnostic_summary",
}
RUN_IDENTITY_KEYS = {
    "pilot_training",
    "bounded_training",
    "bounded_validation",
    "train_query_complete",
    "train_query_is_pilot",
    "validation_query_complete",
    "validation_query_is_pilot",
    "train_spatial_complete",
    "train_spatial_is_pilot",
    "validation_spatial_complete",
    "validation_spatial_is_pilot",
    "source_git_dirty",
    "train_validation_overlap",
    "production_eligible",
}
GIT_KEYS = {"source_git_commit", "source_git_dirty", "source_git_diff_sha256"}
TRAINING_CONFIG_KEYS = {
    "seed", "num_epochs", "batch_size", "learning_rate", "weight_decay",
    "optimizer", "scheduler", "warmup_ratio", "amp_dtype", "save_best_model",
    "early_stopping_patience", "embedding_dim", "bottleneck_dim", "dropout",
    "residual_max", "gamma_max", "gate_hidden_dim", "pooled_grid_height",
    "pooled_grid_width", "mil_top_k", "mil_temperature",
    "infonce_temperature", "attention_selection_weight", "anchor_weight",
    "attention_support_weight", "gate_weight", "pair_chunk_size",
    "gate_prior_feature_version",
}
E7_IDENTITY_KEYS = {
    "format_version", "split_name", "bank_sha256", "source_feature_sha256",
    "annotation_id_fingerprint", "selected_annotation_count",
    "e3_config_sha256", "e3_checkpoint_sha256", "routing_temperature",
    "source_git_commit", "source_git_dirty", "source_git_diff_sha256",
    "complete", "is_pilot", "source_image_count", "source_annotation_count",
    "selected_image_count", "image_id_fingerprint",
    "annotation_to_image_fingerprint",
}
SPATIAL_IDENTITY_KEYS = {
    "format_version", "split", "manifest_sha256", "source_feature_sha256",
    "dataset_identity", "complete", "is_pilot", "production_eligible",
    "source_git_commit", "source_git_dirty", "source_git_diff_sha256",
    "dino_identity", "extraction", "geometry", "pooling_version",
    "attention_prior_version", "global_token_handling",
}
DATASET_IDENTITY_KEYS = {
    "split", "source_image_count", "source_annotation_count",
    "selected_image_count", "selected_annotation_count",
    "image_id_fingerprint", "annotation_id_fingerprint",
    "annotation_to_image_fingerprint",
}
SOURCE_FEATURE_IDENTITY_KEYS = {
    "query_bank_sha256", "query_source_feature_sha256",
    "spatial_manifest_sha256", "spatial_source_artifact_sha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")

CANONICAL_E9_TRAINING_CONFIG = {
    "seed": 42,
    "num_epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "optimizer": "AdamW",
    "scheduler": "cosine",
    "warmup_ratio": 0.05,
    "amp_dtype": "bfloat16",
    "save_best_model": True,
    "early_stopping_patience": 5,
    "embedding_dim": 768,
    "bottleneck_dim": 256,
    "dropout": 0.10,
    "residual_max": 0.25,
    "gamma_max": 0.30,
    "gate_hidden_dim": 128,
    "pooled_grid_height": 16,
    "pooled_grid_width": 16,
    "mil_top_k": 16,
    "mil_temperature": 0.10,
    "infonce_temperature": 0.07,
    "attention_selection_weight": 0.05,
    "anchor_weight": 0.10,
    "attention_support_weight": 0.02,
    "gate_weight": 0.001,
    "pair_chunk_size": 32,
    "gate_prior_feature_version": GATE_PRIOR_FEATURE_VERSION,
}

E9_EPOCH_DIAGNOSTIC_KEYS = frozenset({
    "loss",
    "nce_loss",
    "anchor_text_loss",
    "anchor_patch_loss",
    "anchor_loss",
    "attention_support_loss",
    "gate_loss",
    "gamma_min",
    "gamma_mean",
    "gamma_max",
    "text_residual_gate_min",
    "text_residual_gate_mean",
    "text_residual_gate_max",
    "patch_residual_gate_min",
    "patch_residual_gate_mean",
    "patch_residual_gate_max",
    "positive_image_score",
    "negative_image_score",
    "positive_minus_negative_margin",
    "positive_base_patch_score",
    "positive_adapted_patch_score",
    "positive_final_patch_score",
    "selected_region_attention_mass",
    "selected_region_responsibility_entropy",
    "unique_selected_patch_count",
    "top_k_index_diversity",
    "query_residual_angle",
    "patch_residual_angle",
    "peak_activation_elements",
    "retained_autograd_activation_elements",
    "duplicate_image_batch_violations",
    "nonfinite_counts",
    "train_validation_image_overlap_violations",
})


class E9ValidationError(ValueError):
    """Raised when E9 inputs or artifacts violate their closed contract."""


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise E9ValidationError(f"{label} must be a lowercase SHA256 digest")
    return value


def validate_e9_dino_identity(
    value: Mapping[str, Any], label: str = "E9 DINO identity"
) -> dict[str, str]:
    """Validate the closed DINOv2 extraction/runtime identity."""

    if not isinstance(value, Mapping) or set(value) != DINO_IDENTITY_KEYS:
        raise E9ValidationError(f"{label} closed schema mismatch")
    result = dict(value)
    if result["model"] != "dinov2_vitb14_reg":
        raise E9ValidationError(f"{label} model must be dinov2_vitb14_reg")
    if (
        not isinstance(result["source_commit"], str)
        or _GIT_COMMIT.fullmatch(result["source_commit"]) is None
    ):
        raise E9ValidationError(
            f"{label} source_commit must be 40 lowercase hexadecimal characters"
        )
    _require_sha256(result["checkpoint_sha256"], f"{label} checkpoint_sha256")
    return result


def validate_e9_training_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed scientific/training configuration without I/O."""

    if not isinstance(value, Mapping) or set(value) != TRAINING_CONFIG_KEYS:
        raise E9ValidationError("E9 training configuration has an invalid closed schema")
    result = dict(value)
    for name in (
        "seed", "num_epochs", "batch_size", "early_stopping_patience",
        "embedding_dim", "bottleneck_dim", "gate_hidden_dim",
        "pooled_grid_height", "pooled_grid_width", "mil_top_k",
        "pair_chunk_size",
    ):
        item = result[name]
        minimum = 0 if name == "seed" else 1
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            qualifier = "a non-negative" if name == "seed" else "a positive"
            raise E9ValidationError(f"{name} must be {qualifier} integer")
    positive = ("learning_rate", "mil_temperature", "infonce_temperature")
    nonnegative = (
        "weight_decay", "warmup_ratio", "dropout", "residual_max",
        "gamma_max", "attention_selection_weight", "anchor_weight",
        "attention_support_weight", "gate_weight",
    )
    for name in positive + nonnegative:
        item = result[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or (name in positive and float(item) <= 0)
            or (name in nonnegative and float(item) < 0)
        ):
            qualifier = "positive" if name in positive else "nonnegative"
            raise E9ValidationError(f"{name} must be finite and {qualifier}")
    if not 0 <= float(result["warmup_ratio"]) <= 1:
        raise E9ValidationError("warmup_ratio must be in [0,1]")
    if result["mil_top_k"] > result["pooled_grid_height"] * result["pooled_grid_width"]:
        raise E9ValidationError("mil_top_k exceeds the configured spatial grid")
    if type(result["save_best_model"]) is not bool:
        raise E9ValidationError("save_best_model must be boolean")
    if (
        result["optimizer"] != "AdamW"
        or result["scheduler"] != "cosine"
        or result["amp_dtype"] != "bfloat16"
    ):
        raise E9ValidationError("unsupported E9 optimizer/scheduler/AMP configuration")
    if result["gate_prior_feature_version"] != GATE_PRIOR_FEATURE_VERSION:
        raise E9ValidationError(
            "gate_prior_feature_version must equal "
            f"{GATE_PRIOR_FEATURE_VERSION}"
        )
    SparseRegionAlignmentConfig.from_mapping({
        key: result[key]
        for key in (
            "embedding_dim", "bottleneck_dim", "dropout", "residual_max",
            "gamma_max", "gate_hidden_dim",
        )
    })
    return result


def _closed_dataclass(cls, value: Mapping[str, Any], label: str):
    if not isinstance(value, Mapping):
        raise E9ValidationError(f"{label} must be a mapping")
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        raise E9ValidationError(
            f"{label} closed schema mismatch: unknown={sorted(set(value)-expected)}, "
            f"missing={sorted(expected-set(value))}"
        )
    return cls(**dict(value))


def _finite_bound(name: str, value: Any, *, lower: float, upper: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise E9ValidationError(f"{name} must be finite")
    if float(value) < lower or (upper is not None and float(value) > upper):
        raise E9ValidationError(f"{name} must be in [{lower}, {upper}]")


def max_valid_normalized_attention(
    attention_priors: torch.Tensor,
    valid_patch_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return detached, grid-invariant attention support in ``[0, 1]``.

    Invalid patches are removed before each image's maximum is calculated.  A
    row with no positive valid support is represented by exact zeros.
    """

    if (
        not torch.is_tensor(attention_priors)
        or attention_priors.ndim != 2
        or not attention_priors.is_floating_point()
    ):
        raise E9ValidationError("attention_priors must be floating point [J,P]")
    if valid_patch_mask is None:
        valid_patch_mask = torch.ones_like(attention_priors, dtype=torch.bool)
    if (
        not torch.is_tensor(valid_patch_mask)
        or valid_patch_mask.shape != attention_priors.shape
        or valid_patch_mask.dtype != torch.bool
        or valid_patch_mask.device != attention_priors.device
    ):
        raise E9ValidationError(
            "valid_patch_mask must be boolean [J,P] on the attention device"
        )
    priors = attention_priors.detach().to(dtype=torch.float32)
    if not torch.isfinite(priors).all() or torch.any(priors < 0):
        raise E9ValidationError("attention priors must be finite and nonnegative")
    valid_priors = torch.where(valid_patch_mask, priors, torch.zeros_like(priors))
    maxima = valid_priors.amax(dim=-1, keepdim=True)
    feature = torch.where(
        maxima > 0,
        valid_priors / maxima.clamp_min(torch.finfo(valid_priors.dtype).tiny),
        torch.zeros_like(valid_priors),
    )
    feature = torch.where(valid_patch_mask, feature, torch.zeros_like(feature))
    if (
        not torch.isfinite(feature).all()
        or torch.any(feature < 0)
        or torch.any(feature > 1)
    ):
        raise E9ValidationError(
            "max-valid-normalized attention feature must be finite in [0,1]"
        )
    return feature


def pool_native_grid_to_training_geometry(
    patch_grid: torch.Tensor,
    attention_prior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a native runtime DINO patch grid to the exact geometry E9 was
    trained on.

    The E9 spatial bank pools every source image from its native
    ``source_grid_height x source_grid_width`` DINO patch grid down to
    ``pooled_grid_height x pooled_grid_width`` (see
    ``src.e9_spatial_bank.pool_source_image``): 2x2 mean pooling followed by
    L2 renormalization for patch embeddings, and 2x2 sum pooling followed by
    renormalization for attention priors.  The adapter never sees anything
    but that pooled geometry during training.  Segmentation inference must
    apply the identical reduction before scoring, or every patch embedding
    and attention weight the adapter sees at test time differs
    systematically -- in scale, in noise, and in the effective area a single
    scored "patch" covers -- from what it learned.  This function is the
    single place that reduction happens.
    """

    if patch_grid.ndim != 4:
        raise E9ValidationError("patch_grid must have shape [B,D,H,W]")
    batch, dimension, height, width = patch_grid.shape
    source_h = EXPECTED_GEOMETRY["source_grid_height"]
    source_w = EXPECTED_GEOMETRY["source_grid_width"]
    if height != source_h or width != source_w:
        raise E9ValidationError(
            f"E9 inference requires the exact {source_h}x{source_w} native "
            "DINO patch grid the training pipeline used; got "
            f"{height}x{width}. Check resize_dim/patch_size for the "
            "evaluation configuration."
        )
    if (
        attention_prior.ndim != 2
        or attention_prior.shape[0] != batch
        or attention_prior.shape[1] != height * width
    ):
        raise E9ValidationError("attention_prior must have shape [B,H*W]")
    if not torch.isfinite(patch_grid).all() or not torch.isfinite(attention_prior).all():
        raise E9ValidationError("E9 pooling inputs must be finite")

    pooled_h = EXPECTED_GEOMETRY["pooled_grid_height"]
    pooled_w = EXPECTED_GEOMETRY["pooled_grid_width"]
    block_h = source_h // pooled_h
    block_w = source_w // pooled_w
    if block_h * pooled_h != source_h or block_w * pooled_w != source_w:
        raise E9ValidationError(
            "source grid is not an exact multiple of the pooled grid"
        )

    patches = patch_grid.permute(0, 2, 3, 1).float()
    pooled_patches = patches.reshape(
        batch, pooled_h, block_h, pooled_w, block_w, dimension
    ).mean(dim=(2, 4))
    pooled_patches = F.normalize(
        pooled_patches.reshape(batch, pooled_h * pooled_w, dimension), dim=-1
    )
    pooled_grid = pooled_patches.reshape(batch, pooled_h, pooled_w, dimension).permute(
        0, 3, 1, 2
    )

    prior = attention_prior.float().reshape(batch, source_h, source_w)
    pooled_prior = (
        prior.reshape(batch, pooled_h, block_h, pooled_w, block_w)
        .sum(dim=(2, 4))
        .reshape(batch, pooled_h * pooled_w)
        .clamp_min(0)
    )
    total = pooled_prior.sum(dim=-1, keepdim=True)
    pooled_prior = pooled_prior / total.clamp_min(torch.finfo(pooled_prior.dtype).tiny)

    if not torch.isfinite(pooled_grid).all() or not torch.isfinite(pooled_prior).all():
        raise E9ValidationError("E9 pooling produced non-finite values")
    return pooled_grid, pooled_prior


@dataclass(frozen=True)
class SparseRegionAlignmentConfig:
    embedding_dim: int = 768
    bottleneck_dim: int = 256
    dropout: float = 0.10
    residual_max: float = 0.25
    gamma_max: float = 0.30
    gate_hidden_dim: int = 128

    def __post_init__(self) -> None:
        for name in ("embedding_dim", "bottleneck_dim", "gate_hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise E9ValidationError(f"{name} must be a positive integer")
        _finite_bound("dropout", self.dropout, lower=0, upper=1 - 1e-12)
        _finite_bound("residual_max", self.residual_max, lower=0, upper=1)
        _finite_bound("gamma_max", self.gamma_max, lower=0, upper=1)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        return _closed_dataclass(cls, value, "E9 architecture configuration")


class _ResidualAdapter(nn.Module):
    def __init__(self, config: SparseRegionAlignmentConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.bottleneck_dim, config.embedding_dim),
        )
        nn.init.normal_(self.network[-1].weight, std=1e-4)
        nn.init.zeros_(self.network[-1].bias)
        self.gate = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.0)

    def forward(self, value: torch.Tensor, maximum: float) -> tuple[torch.Tensor, torch.Tensor]:
        gate = maximum * torch.sigmoid(self.gate(value)).squeeze(-1)
        adapted = F.normalize(value + gate.unsqueeze(-1) * self.network(value), dim=-1)
        return adapted, gate


@dataclass(frozen=True)
class SparseRegionRepresentations:
    base_query: torch.Tensor
    base_patches: torch.Tensor
    query: torch.Tensor
    patches: torch.Tensor
    text_residual_gate: torch.Tensor
    patch_residual_gate: torch.Tensor


@dataclass(frozen=True)
class SparseRegionScoreOutput:
    base_score: torch.Tensor
    adapted_score: torch.Tensor
    gamma: torch.Tensor
    final_score: torch.Tensor
    query: torch.Tensor
    patches: torch.Tensor
    text_residual_gate: torch.Tensor
    patch_residual_gate: torch.Tensor


class SparseRegionAlignmentAdapter(nn.Module):
    """Shared, class-parameter-free residual compatibility adapter."""

    def __init__(self, config: SparseRegionAlignmentConfig | Mapping[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = SparseRegionAlignmentConfig()
        elif not isinstance(config, SparseRegionAlignmentConfig):
            config = SparseRegionAlignmentConfig.from_mapping(config)
        self.config = config
        self.text_adapter = _ResidualAdapter(config)
        self.patch_adapter = _ResidualAdapter(config)
        self.compatibility_gate = nn.Sequential(
            nn.Linear(4, config.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(config.gate_hidden_dim, 1),
        )
        nn.init.normal_(self.compatibility_gate[-1].weight, std=1e-4)
        nn.init.constant_(self.compatibility_gate[-1].bias, -4.0)

    def representations(
        self,
        mapped_queries: torch.Tensor,
        patch_embeddings: torch.Tensor,
        *,
        inputs_normalized: bool = False,
    ) -> SparseRegionRepresentations:
        if mapped_queries.ndim != 2 or mapped_queries.shape[-1] != self.config.embedding_dim:
            raise E9ValidationError("mapped_queries must have shape [B,D]")
        if patch_embeddings.ndim != 3 or patch_embeddings.shape[-1] != self.config.embedding_dim:
            raise E9ValidationError("patch_embeddings must have shape [J,P,D]")
        if not mapped_queries.is_floating_point() or not patch_embeddings.is_floating_point():
            raise E9ValidationError("E9 embeddings must be floating point")
        if mapped_queries.device != patch_embeddings.device:
            raise E9ValidationError("E9 embeddings must share a device")
        query_source = mapped_queries.detach().float()
        patch_source = patch_embeddings.detach().float()
        if inputs_normalized:
            query_norms = query_source.norm(dim=-1)
            patch_norms = patch_source.norm(dim=-1)
            if (
                torch.max(torch.abs(query_norms - 1)) > 2e-3
                or torch.max(torch.abs(patch_norms - 1)) > 2e-3
            ):
                raise E9ValidationError(
                    "inputs_normalized=True requires unit-normalized embeddings"
                )
        else:
            query_source = F.normalize(query_source, dim=-1)
            patch_source = F.normalize(patch_source, dim=-1)
        if not torch.isfinite(query_source).all() or not torch.isfinite(patch_source).all():
            raise E9ValidationError("E9 embeddings contain non-finite values")
        query, text_gate = self.text_adapter(query_source, self.config.residual_max)
        patches, patch_gate = self.patch_adapter(patch_source, self.config.residual_max)
        return SparseRegionRepresentations(
            query_source, patch_source, query, patches, text_gate, patch_gate
        )

    def score_from_representations(
        self,
        base_queries: torch.Tensor,
        base_patches: torch.Tensor,
        representations: SparseRegionRepresentations,
        attention_priors: torch.Tensor,
        valid_patch_mask: torch.Tensor | None = None,
        precomputed_base_score: torch.Tensor | None = None,
        precomputed_prior_feature: torch.Tensor | None = None,
    ) -> SparseRegionScoreOutput:
        query_count = base_queries.shape[0]
        image_count, patch_count, dimension = base_patches.shape
        if (
            tuple(representations.base_query.shape) != tuple(base_queries.shape)
            or tuple(representations.base_patches.shape) != tuple(base_patches.shape)
        ):
            raise E9ValidationError("base and adapted representation shapes differ")
        if attention_priors.shape != (image_count, patch_count):
            raise E9ValidationError("attention_priors must have shape [J,P]")
        if attention_priors.device != base_patches.device:
            raise E9ValidationError("attention priors must share the patch device")
        if valid_patch_mask is None:
            valid_patch_mask = torch.ones_like(attention_priors, dtype=torch.bool)
        if valid_patch_mask.shape != attention_priors.shape or valid_patch_mask.dtype != torch.bool:
            raise E9ValidationError("valid_patch_mask must be boolean [J,P]")
        if valid_patch_mask.device != base_patches.device:
            raise E9ValidationError("valid_patch_mask must share the patch device")
        if not torch.isfinite(attention_priors).all() or torch.any(attention_priors < 0):
            raise E9ValidationError("attention priors must be finite and nonnegative")
        prior_sums = attention_priors.float().sum(dim=-1)
        if not torch.allclose(
            prior_sums, torch.ones_like(prior_sums), atol=2e-3, rtol=2e-3
        ):
            raise E9ValidationError("attention priors must sum to one per image")
        if precomputed_prior_feature is None:
            prior_feature = max_valid_normalized_attention(
                attention_priors, valid_patch_mask
            )
        else:
            if (
                precomputed_prior_feature.shape != attention_priors.shape
                or precomputed_prior_feature.device != attention_priors.device
            ):
                raise E9ValidationError(
                    "precomputed_prior_feature must have shape [J,P] on the "
                    "attention device"
                )
            prior_feature = precomputed_prior_feature.detach().to(
                dtype=torch.float32
            )
            if (
                not torch.isfinite(prior_feature).all()
                or torch.any(prior_feature < 0)
                or torch.any(prior_feature > 1)
                or torch.any(prior_feature[~valid_patch_mask] != 0)
            ):
                raise E9ValidationError(
                    "precomputed_prior_feature must be finite in [0,1] and "
                    "zero on invalid patches"
                )
        if precomputed_base_score is None:
            base = torch.einsum(
                "id,jpd->ijp",
                representations.base_query,
                representations.base_patches,
            )
        else:
            if precomputed_base_score.shape != (query_count, image_count, patch_count):
                raise E9ValidationError("precomputed_base_score must have shape [I,J,P]")
            if not torch.isfinite(precomputed_base_score).all():
                raise E9ValidationError("precomputed_base_score contains non-finite values")
            base = precomputed_base_score
        adapted = torch.einsum(
            "id,jpd->ijp", representations.query, representations.patches
        )
        prior = prior_feature[None].expand(query_count, -1, -1)
        gate_input = torch.stack((base, adapted, adapted - base, prior), dim=-1)
        gamma = self.config.gamma_max * torch.sigmoid(
            self.compatibility_gate(gate_input).squeeze(-1)
        )
        valid = valid_patch_mask[None].expand_as(base)
        gamma = torch.where(valid, gamma, torch.zeros_like(gamma))
        fused = (1 - gamma) * base + gamma * adapted
        final = torch.where(gamma == 0, base, fused)
        final = torch.where(valid, final, torch.zeros_like(final))
        result = SparseRegionScoreOutput(
            base, adapted, gamma, final,
            representations.query, representations.patches,
            representations.text_residual_gate,
            representations.patch_residual_gate,
        )
        if not all(torch.isfinite(value).all() for value in (
            result.base_score, result.adapted_score, result.gamma,
            result.final_score, result.query, result.patches,
        )):
            raise E9ValidationError("E9 adapter produced non-finite values")
        return result

    def forward(
        self,
        mapped_queries: torch.Tensor,
        patch_embeddings: torch.Tensor,
        attention_priors: torch.Tensor,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> SparseRegionScoreOutput:
        reps = self.representations(mapped_queries, patch_embeddings)
        return self.score_from_representations(
            mapped_queries, patch_embeddings, reps, attention_priors, valid_patch_mask
        )


@dataclass(frozen=True)
class E9MILOutput:
    image_scores: torch.Tensor
    selected_indices: torch.Tensor
    selected_scores: torch.Tensor
    selected_attention: torch.Tensor
    gamma: torch.Tensor
    base_score: torch.Tensor
    adapted_score: torch.Tensor
    final_score: torch.Tensor
    query: torch.Tensor
    patches: torch.Tensor
    base_query: torch.Tensor
    base_patches: torch.Tensor
    text_residual_gate: torch.Tensor
    patch_residual_gate: torch.Tensor
    peak_activation_elements: int
    retained_autograd_activation_elements: int
    gate_mean: torch.Tensor
    gamma_min: torch.Tensor
    gamma_max: torch.Tensor
    positive_base_patch_score: torch.Tensor
    positive_adapted_patch_score: torch.Tensor
    positive_final_patch_score: torch.Tensor


def compute_chunked_mil_scores(
    adapter: SparseRegionAlignmentAdapter,
    mapped_queries: torch.Tensor,
    patch_embeddings: torch.Tensor,
    attention_priors: torch.Tensor,
    *,
    valid_patch_mask: torch.Tensor | None = None,
    mil_top_k: int = 16,
    mil_temperature: float = 0.10,
    attention_selection_weight: float = 0.05,
    pair_chunk_size: int = 32,
    retain_pairwise_scores: bool = False,
    inputs_normalized: bool = False,
) -> E9MILOutput:
    for name, value in (
        ("mil_temperature", mil_temperature),
        ("attention_selection_weight", attention_selection_weight),
    ):
        _finite_bound(name, value, lower=0)
    if mil_temperature <= 0:
        raise E9ValidationError("mil_temperature must be positive")
    if isinstance(mil_top_k, bool) or not isinstance(mil_top_k, int) or mil_top_k <= 0:
        raise E9ValidationError("mil_top_k must be positive")
    if isinstance(pair_chunk_size, bool) or not isinstance(pair_chunk_size, int) or pair_chunk_size <= 0:
        raise E9ValidationError("pair_chunk_size must be positive")
    image_count, patch_count = patch_embeddings.shape[:2]
    if mil_top_k > patch_count:
        raise E9ValidationError("mil_top_k exceeds patch count")
    if valid_patch_mask is None:
        valid_patch_mask = torch.ones(
            (image_count, patch_count), dtype=torch.bool, device=patch_embeddings.device
        )
    if torch.any(valid_patch_mask.sum(dim=-1) < mil_top_k):
        raise E9ValidationError("each image must contain at least mil_top_k valid patches")
    prior_feature = max_valid_normalized_attention(
        attention_priors, valid_patch_mask
    )
    reps = adapter.representations(
        mapped_queries, patch_embeddings, inputs_normalized=inputs_normalized
    )
    image_chunks = []
    index_chunks = []
    selected_chunks = []
    attention_chunks = []
    gamma_chunks = []
    base_chunks = []
    adapted_chunks = []
    final_chunks = []
    peak = 0
    # Shared adapter activations remain live across every pair chunk.  This is
    # intentionally conservative: it counts normalized/adapted/residual paths,
    # bottleneck intermediates, and scalar residual gates.
    retained = (
        reps.base_query.numel() * 4
        + reps.base_patches.numel() * 4
        + reps.base_query.shape[0] * adapter.config.bottleneck_dim * 3
        + reps.base_patches.shape[0] * reps.base_patches.shape[1]
        * adapter.config.bottleneck_dim * 3
        + reps.text_residual_gate.numel()
        + reps.patch_residual_gate.numel()
    )
    gamma_sum = mapped_queries.new_zeros((), dtype=torch.float32)
    gamma_count = 0
    gamma_minimum = None
    gamma_maximum = None
    positive_base = []
    positive_adapted = []
    positive_final = []
    for start in range(0, mapped_queries.shape[0], pair_chunk_size):
        stop = min(start + pair_chunk_size, mapped_queries.shape[0])
        local_reps = SparseRegionRepresentations(
            reps.base_query[start:stop], reps.base_patches,
            reps.query[start:stop], reps.patches,
            reps.text_residual_gate[start:stop], reps.patch_residual_gate,
        )
        scored = adapter.score_from_representations(
            mapped_queries[start:stop], patch_embeddings, local_reps,
            attention_priors, valid_patch_mask,
            precomputed_prior_feature=prior_feature,
        )
        selection = (
            scored.final_score + attention_selection_weight * prior_feature[None]
        )
        selection = selection.masked_fill(~valid_patch_mask[None], -torch.inf)
        indices = torch.topk(selection.detach(), mil_top_k, dim=-1, sorted=True).indices
        selected = torch.gather(scored.final_score, -1, indices)
        selected_attention = torch.gather(
            prior_feature[None].expand(stop - start, -1, -1), -1, indices
        )
        image_score = mil_temperature * torch.logsumexp(
            selected / mil_temperature, dim=-1
        ) - mil_temperature * math.log(mil_top_k)
        image_chunks.append(image_score)
        index_chunks.append(indices)
        selected_chunks.append(selected)
        attention_chunks.append(selected_attention)
        gamma_sum = gamma_sum + scored.gamma.sum()
        gamma_count += scored.gamma.numel()
        local_minimum = scored.gamma.min()
        local_maximum = scored.gamma.max()
        gamma_minimum = local_minimum if gamma_minimum is None else torch.minimum(gamma_minimum, local_minimum)
        gamma_maximum = local_maximum if gamma_maximum is None else torch.maximum(gamma_maximum, local_maximum)
        local_rows = torch.arange(stop - start, device=mapped_queries.device)
        image_rows = torch.arange(start, stop, device=mapped_queries.device)
        positive_base.append(scored.base_score[local_rows, image_rows].mean(dim=-1))
        positive_adapted.append(scored.adapted_score[local_rows, image_rows].mean(dim=-1))
        positive_final.append(scored.final_score[local_rows, image_rows].mean(dim=-1))
        if retain_pairwise_scores:
            gamma_chunks.append(scored.gamma)
            base_chunks.append(scored.base_score)
            adapted_chunks.append(scored.adapted_score)
            final_chunks.append(scored.final_score)
        pair_elements = selection.numel()
        # Conservative accounting includes the four gate inputs, two scores,
        # gamma/fusion/selection temporaries, and the gate MLP hidden output.
        local_activation_elements = pair_elements * (
            adapter.config.gate_hidden_dim + 9
        ) + selected.numel() * 2
        peak = max(peak, local_activation_elements)
        retained += local_activation_elements
    return E9MILOutput(
        torch.cat(image_chunks), torch.cat(index_chunks), torch.cat(selected_chunks),
        torch.cat(attention_chunks),
        torch.cat(gamma_chunks) if gamma_chunks else mapped_queries.new_empty(0),
        torch.cat(base_chunks) if base_chunks else mapped_queries.new_empty(0),
        torch.cat(adapted_chunks) if adapted_chunks else mapped_queries.new_empty(0),
        torch.cat(final_chunks) if final_chunks else mapped_queries.new_empty(0),
        reps.query, reps.patches, reps.base_query, reps.base_patches,
        reps.text_residual_gate, reps.patch_residual_gate,
        peak, retained, gamma_sum / gamma_count, gamma_minimum, gamma_maximum,
        torch.cat(positive_base),
        torch.cat(positive_adapted), torch.cat(positive_final),
    )


def symmetric_infonce(scores: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    _finite_bound("infonce_temperature", temperature, lower=0)
    if temperature <= 0 or scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise E9ValidationError("InfoNCE scores must be square and temperature positive")
    labels = torch.arange(scores.shape[0], device=scores.device)
    return 0.5 * (
        F.cross_entropy(scores / temperature, labels)
        + F.cross_entropy(scores.T / temperature, labels)
    )


def compute_e9_loss(
    output: E9MILOutput,
    mapped_queries: torch.Tensor,
    patch_embeddings: torch.Tensor,
    *,
    mil_temperature: float = 0.10,
    infonce_temperature: float = 0.07,
    anchor_weight: float = 0.10,
    attention_support_weight: float = 0.02,
    gate_weight: float = 0.001,
) -> dict[str, torch.Tensor]:
    _finite_bound("mil_temperature", mil_temperature, lower=0)
    if mil_temperature <= 0:
        raise E9ValidationError("mil_temperature must be positive")
    for name, value in (
        ("anchor_weight", anchor_weight),
        ("attention_support_weight", attention_support_weight),
        ("gate_weight", gate_weight),
    ):
        _finite_bound(name, value, lower=0)
    nce = symmetric_infonce(output.image_scores, infonce_temperature)
    text_anchor = (
        1 - F.cosine_similarity(output.query, output.base_query, dim=-1)
    ).mean()
    patch_anchor = (
        1 - F.cosine_similarity(output.patches, output.base_patches, dim=-1)
    ).mean()
    anchor = 0.5 * (text_anchor + patch_anchor)
    batch = output.image_scores.shape[0]
    diagonal = torch.arange(batch, device=output.image_scores.device)
    positive_scores = output.selected_scores[diagonal, diagonal]
    positive_attention = output.selected_attention[diagonal, diagonal]
    omega = torch.softmax(positive_scores / mil_temperature, dim=-1)
    attention_support = (1 - (omega * positive_attention).sum(dim=-1)).mean()
    gate = output.gate_mean
    total = nce + anchor_weight * anchor + attention_support_weight * attention_support + gate_weight * gate
    losses = {
        "loss": total,
        "nce_loss": nce,
        "anchor_text_loss": text_anchor,
        "anchor_patch_loss": patch_anchor,
        "anchor_loss": anchor,
        "attention_support_loss": attention_support,
        "gate_loss": gate,
    }
    if not all(torch.isfinite(value) for value in losses.values()):
        raise E9ValidationError("E9 loss is non-finite")
    return losses


def _closed_mapping(
    value: Any, keys: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise E9ValidationError(f"{label} closed schema mismatch")
    return value


def _validate_git_provenance(value: Mapping[str, Any], label: str) -> None:
    _closed_mapping(value, GIT_KEYS, label)
    if (
        not isinstance(value["source_git_commit"], str)
        or _GIT_COMMIT.fullmatch(value["source_git_commit"]) is None
    ):
        raise E9ValidationError(f"{label} has an invalid Git commit")
    if type(value["source_git_dirty"]) is not bool:
        raise E9ValidationError(f"{label} dirty flag must be boolean")
    if value["source_git_dirty"]:
        _require_sha256(value["source_git_diff_sha256"], f"{label} diff")
    elif value["source_git_diff_sha256"] is not None:
        raise E9ValidationError(f"clean {label} requires no diff hash")


def _validate_dataset_identity(value: Any, expected_split: str, label: str) -> None:
    identity = _closed_mapping(value, DATASET_IDENTITY_KEYS, label)
    if identity["split"] != expected_split:
        raise E9ValidationError(f"{label} split mismatch")
    for key in (
        "source_image_count", "source_annotation_count",
        "selected_image_count", "selected_annotation_count",
    ):
        item = identity[key]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise E9ValidationError(f"{label}.{key} must be a positive integer")
    if (
        identity["selected_image_count"] > identity["source_image_count"]
        or identity["selected_annotation_count"]
        > identity["source_annotation_count"]
    ):
        raise E9ValidationError(f"{label} selected counts exceed source counts")
    for key in (
        "image_id_fingerprint", "annotation_id_fingerprint",
        "annotation_to_image_fingerprint",
    ):
        _require_sha256(identity[key], f"{label}.{key}")


def _validate_spatial_scientific_identity(
    spatial: Mapping[str, Any], label: str
) -> tuple[dict[str, str], dict[str, Any]]:
    dino = validate_e9_dino_identity(spatial["dino_identity"], f"{label} DINO")
    extraction = dict(
        _closed_mapping(
            spatial["extraction"], EXTRACTION_KEYS, f"{label} extraction"
        )
    )
    mismatches = {
        key: (extraction.get(key), expected)
        for key, expected in CANONICAL_EXTRACTION.items()
        if extraction.get(key) != expected
    }
    if mismatches:
        raise E9ValidationError(
            f"{label} canonical extraction identity mismatch: {mismatches}"
        )
    for key in ("annotation_path", "data_dir"):
        if not isinstance(extraction[key], str) or not extraction[key]:
            raise E9ValidationError(f"{label} extraction.{key} must be nonempty")
    _require_sha256(
        extraction["backbone_weights_sha256"],
        f"{label} extraction backbone_weights_sha256",
    )
    if extraction["model"] != dino["model"]:
        raise E9ValidationError(f"{label} DINO/extraction model mismatch")
    if extraction["backbone_weights_sha256"] != dino["checkpoint_sha256"]:
        raise E9ValidationError(f"{label} DINO/extraction checkpoint mismatch")
    if spatial["geometry"] != EXPECTED_GEOMETRY:
        raise E9ValidationError(f"{label} geometry mismatch")
    if spatial["pooling_version"] != POOLING_VERSION:
        raise E9ValidationError(f"{label} pooling version mismatch")
    if spatial["attention_prior_version"] != ATTENTION_PRIOR_VERSION:
        raise E9ValidationError(f"{label} attention-prior version mismatch")
    if spatial["global_token_handling"] != GLOBAL_TOKEN_HANDLING:
        raise E9ValidationError(f"{label} global-token handling mismatch")
    return dino, extraction


def _validate_diagnostic_summary(value: Any) -> dict[str, dict[str, float]]:
    summary = _closed_mapping(
        value, {"train", "validation"}, "diagnostic summary"
    )
    result: dict[str, dict[str, float]] = {}
    for split in ("train", "validation"):
        diagnostics = _closed_mapping(
            summary[split],
            set(E9_EPOCH_DIAGNOSTIC_KEYS),
            f"diagnostic summary {split}",
        )
        converted: dict[str, float] = {}
        for key, item in diagnostics.items():
            if type(item) not in (int, float):
                raise E9ValidationError(
                    f"diagnostic summary {split}.{key} must be a Python scalar"
                )
            scalar = float(item)
            if not math.isfinite(scalar):
                raise E9ValidationError(
                    f"diagnostic summary {split}.{key} must be finite"
                )
            converted[key] = scalar
        result[split] = converted
    return result


def _expected_adapter_state_shapes(
    architecture: SparseRegionAlignmentConfig,
) -> dict[str, tuple[int, ...]]:
    dimension = architecture.embedding_dim
    bottleneck = architecture.bottleneck_dim
    result: dict[str, tuple[int, ...]] = {}
    for prefix in ("text_adapter", "patch_adapter"):
        result.update({
            f"{prefix}.network.0.weight": (dimension,),
            f"{prefix}.network.0.bias": (dimension,),
            f"{prefix}.network.1.weight": (bottleneck, dimension),
            f"{prefix}.network.1.bias": (bottleneck,),
            f"{prefix}.network.4.weight": (dimension, bottleneck),
            f"{prefix}.network.4.bias": (dimension,),
            f"{prefix}.gate.0.weight": (dimension,),
            f"{prefix}.gate.0.bias": (dimension,),
            f"{prefix}.gate.1.weight": (1, dimension),
            f"{prefix}.gate.1.bias": (1,),
        })
    hidden = architecture.gate_hidden_dim
    result.update({
        "compatibility_gate.0.weight": (hidden, 4),
        "compatibility_gate.0.bias": (hidden,),
        "compatibility_gate.2.weight": (1, hidden),
        "compatibility_gate.2.bias": (1,),
    })
    return result


def validate_e9_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    require_production: bool = True,
    expected_e3_identity: Mapping[str, Any] | None = None,
    expected_checkpoint_path: str | Path | None = None,
    expected_source_git_commit: str | None = None,
    expected_dino_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate generic schema first, then the canonical production identity."""

    if not isinstance(checkpoint, Mapping) or set(checkpoint) != E9_CHECKPOINT_KEYS:
        raise E9ValidationError("E9 checkpoint closed schema mismatch")
    if checkpoint["format_version"] != E9_ADAPTER_CHECKPOINT_FORMAT:
        raise E9ValidationError("unsupported E9 checkpoint format")
    architecture = SparseRegionAlignmentConfig.from_mapping(
        checkpoint["architecture_config"]
    )
    training_config = validate_e9_training_config(checkpoint["training_config"])
    architecture_keys = {
        "embedding_dim", "bottleneck_dim", "dropout", "residual_max",
        "gamma_max", "gate_hidden_dim",
    }
    for key in architecture_keys:
        if training_config[key] != getattr(architecture, key):
            raise E9ValidationError(
                f"training/architecture configuration mismatch for {key}"
            )

    state = checkpoint["adapter_state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise E9ValidationError("invalid adapter state_dict")
    if any(
        forbidden in key.lower()
        for key in state
        for forbidden in ("clip", "dino", "projection", "bank")
    ):
        raise E9ValidationError("checkpoint contains a forbidden frozen component")
    expected_state_shapes = _expected_adapter_state_shapes(architecture)
    if set(state) != set(expected_state_shapes) or any(
        tuple(state[key].shape) != expected_state_shapes[key]
        for key in state
    ):
        raise E9ValidationError("adapter state_dict key/shape contract mismatch")
    for key, value in state.items():
        if (
            not value.is_floating_point() or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise E9ValidationError(
                f"adapter_state_dict.{key} must be a finite CPU floating tensor"
            )

    run = _closed_mapping(
        checkpoint["run_identity"], RUN_IDENTITY_KEYS, "run identity"
    )
    if any(type(run[key]) is not bool for key in RUN_IDENTITY_KEYS):
        raise E9ValidationError("every run-identity value must be boolean")
    provenance = _closed_mapping(
        checkpoint["source_git_provenance"], GIT_KEYS, "Git provenance"
    )
    _validate_git_provenance(provenance, "checkpoint Git provenance")
    if run["source_git_dirty"] != provenance["source_git_dirty"]:
        raise E9ValidationError("run/Git dirty provenance mismatch")

    query_identities = _closed_mapping(
        checkpoint["query_bank_identities"], {"train", "validation"},
        "query-bank identities",
    )
    spatial_identities = _closed_mapping(
        checkpoint["spatial_bank_identities"], {"train", "validation"},
        "spatial-bank identities",
    )
    source_identities = _closed_mapping(
        checkpoint["source_feature_identities"], {"train", "validation"},
        "source-feature identities",
    )
    validated_dino: dict[str, dict[str, str]] = {}
    validated_extraction: dict[str, dict[str, Any]] = {}
    for split_key, split_name in (("train", "train"), ("validation", "val")):
        query = _closed_mapping(
            query_identities[split_key], E7_IDENTITY_KEYS,
            f"{split_key} query identity",
        )
        if query["format_version"] != "talk2dino-e7-training-bank-v1":
            raise E9ValidationError("unsupported query-bank identity")
        if query["split_name"] != split_name:
            raise E9ValidationError(f"{split_key} query-bank split mismatch")
        for key in (
            "bank_sha256", "source_feature_sha256", "image_id_fingerprint",
            "annotation_id_fingerprint", "annotation_to_image_fingerprint",
            "e3_config_sha256", "e3_checkpoint_sha256",
        ):
            _require_sha256(query[key], f"{split_key} query identity {key}")
        for key in (
            "source_image_count", "source_annotation_count",
            "selected_image_count", "selected_annotation_count",
        ):
            item = query[key]
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise E9ValidationError(
                    f"{split_key} query identity {key} must be positive"
                )
        if type(query["complete"]) is not bool or type(query["is_pilot"]) is not bool:
            raise E9ValidationError("query completion/pilot flags must be boolean")
        _validate_git_provenance(
            {key: query[key] for key in GIT_KEYS},
            f"{split_key} query Git provenance",
        )
        if not math.isclose(
            float(query["routing_temperature"]), 0.10,
            rel_tol=0, abs_tol=1e-12,
        ):
            raise E9ValidationError("query routing temperature must equal 0.10")

        spatial = _closed_mapping(
            spatial_identities[split_key], SPATIAL_IDENTITY_KEYS,
            f"{split_key} spatial identity",
        )
        if spatial["format_version"] != E9_SPATIAL_BANK_FORMAT:
            raise E9ValidationError("unsupported spatial-bank identity")
        if spatial["split"] != split_name:
            raise E9ValidationError(f"{split_key} spatial-bank split mismatch")
        _require_sha256(
            spatial["manifest_sha256"], f"{split_key} spatial manifest"
        )
        if not isinstance(spatial["source_feature_sha256"], list) or not spatial["source_feature_sha256"]:
            raise E9ValidationError("spatial source identity must be a nonempty list")
        for digest in spatial["source_feature_sha256"]:
            _require_sha256(digest, f"{split_key} spatial source identity")
        for key in ("complete", "is_pilot", "production_eligible", "source_git_dirty"):
            if type(spatial[key]) is not bool:
                raise E9ValidationError(f"{split_key} spatial {key} must be boolean")
        _validate_git_provenance(
            {key: spatial[key] for key in GIT_KEYS},
            f"{split_key} spatial Git provenance",
        )
        _validate_dataset_identity(
            spatial["dataset_identity"], split_name,
            f"{split_key} spatial dataset identity",
        )
        dino, extraction = _validate_spatial_scientific_identity(
            spatial, f"{split_key} spatial identity"
        )
        validated_dino[split_key] = dino
        validated_extraction[split_key] = extraction
        query_dataset = {
            "split": split_name,
            "source_image_count": query["source_image_count"],
            "source_annotation_count": query["source_annotation_count"],
            "selected_image_count": query["selected_image_count"],
            "selected_annotation_count": query["selected_annotation_count"],
            "image_id_fingerprint": query["image_id_fingerprint"],
            "annotation_id_fingerprint": query["annotation_id_fingerprint"],
            "annotation_to_image_fingerprint": query[
                "annotation_to_image_fingerprint"
            ],
        }
        if query_dataset != dict(spatial["dataset_identity"]):
            raise E9ValidationError(
                f"{split_key} query/spatial dataset identity mismatch"
            )

        source = _closed_mapping(
            source_identities[split_key], SOURCE_FEATURE_IDENTITY_KEYS,
            f"{split_key} source-feature identity",
        )
        _require_sha256(
            source["query_bank_sha256"],
            f"{split_key} query bank identity",
        )
        _require_sha256(
            source["query_source_feature_sha256"],
            f"{split_key} query source-feature identity",
        )
        _require_sha256(
            source["spatial_manifest_sha256"],
            f"{split_key} spatial manifest identity",
        )
        if not isinstance(source["spatial_source_artifact_sha256"], list) or not source["spatial_source_artifact_sha256"]:
            raise E9ValidationError("spatial artifact identity must be a nonempty list")
        for digest in source["spatial_source_artifact_sha256"]:
            _require_sha256(digest, f"{split_key} spatial artifact identity")
        if source["query_bank_sha256"] != query["bank_sha256"]:
            raise E9ValidationError(
                f"{split_key} query-bank repeated identity mismatch"
            )
        if source["query_source_feature_sha256"] != query["source_feature_sha256"]:
            raise E9ValidationError(
                f"{split_key} query/source-feature identity mismatch"
            )
        if source["spatial_manifest_sha256"] != spatial["manifest_sha256"]:
            raise E9ValidationError(
                f"{split_key} spatial manifest identity mismatch"
            )
        if source["spatial_source_artifact_sha256"] != spatial["source_feature_sha256"]:
            raise E9ValidationError(
                f"{split_key} spatial/source-feature identity mismatch"
            )

        derived_query_pilot = (
            query["selected_image_count"] < query["source_image_count"]
            or query["selected_annotation_count"] < query["source_annotation_count"]
        )
        if query["is_pilot"] != derived_query_pilot:
            raise E9ValidationError(f"forged {split_key} query pilot state")
        derived_spatial_eligible = (
            spatial["complete"] and not spatial["is_pilot"]
            and not spatial["source_git_dirty"]
            and spatial["dataset_identity"]["selected_image_count"]
            == spatial["dataset_identity"]["source_image_count"]
            and spatial["dataset_identity"]["selected_annotation_count"]
            == spatial["dataset_identity"]["source_annotation_count"]
        )
        if spatial["production_eligible"] != derived_spatial_eligible:
            raise E9ValidationError(f"forged {split_key} spatial eligibility")

    if validated_dino["train"] != validated_dino["validation"]:
        raise E9ValidationError("train/validation DINO identities differ")
    invariant_extraction_keys = EXTRACTION_KEYS.difference(
        {"annotation_path", "data_dir"}
    )
    if any(
        validated_extraction["train"][key]
        != validated_extraction["validation"][key]
        for key in invariant_extraction_keys
    ):
        raise E9ValidationError(
            "train/validation invariant extraction identities differ"
        )
    for key in (
        "geometry", "pooling_version", "attention_prior_version",
        "global_token_handling",
    ):
        if spatial_identities["train"][key] != spatial_identities["validation"][key]:
            raise E9ValidationError(
                f"train/validation spatial {key} identities differ"
            )

    e3_identity = _closed_mapping(
        checkpoint["e3_identity"], {"config_sha256", "checkpoint_sha256"},
        "E3 identity",
    )
    for key, value in e3_identity.items():
        _require_sha256(value, f"E3 identity {key}")
    if not (
        query_identities["train"]["e3_config_sha256"]
        == query_identities["validation"]["e3_config_sha256"]
        == e3_identity["config_sha256"]
        and query_identities["train"]["e3_checkpoint_sha256"]
        == query_identities["validation"]["e3_checkpoint_sha256"]
        == e3_identity["checkpoint_sha256"]
    ):
        raise E9ValidationError("incompatible train/validation E3 identities")

    _validate_diagnostic_summary(checkpoint["diagnostic_summary"])
    if isinstance(checkpoint["epoch"], bool) or not isinstance(checkpoint["epoch"], int) or checkpoint["epoch"] <= 0:
        raise E9ValidationError("checkpoint epoch must be positive")
    metric = checkpoint["best_validation_metric"]
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
        raise E9ValidationError("best validation metric must be finite")

    relationships = {
        "train_query_complete": query_identities["train"]["complete"],
        "train_query_is_pilot": query_identities["train"]["is_pilot"],
        "validation_query_complete": query_identities["validation"]["complete"],
        "validation_query_is_pilot": query_identities["validation"]["is_pilot"],
        "train_spatial_complete": spatial_identities["train"]["complete"],
        "train_spatial_is_pilot": spatial_identities["train"]["is_pilot"],
        "validation_spatial_complete": spatial_identities["validation"]["complete"],
        "validation_spatial_is_pilot": spatial_identities["validation"]["is_pilot"],
        "source_git_dirty": provenance["source_git_dirty"],
    }
    if any(run[key] != value for key, value in relationships.items()):
        raise E9ValidationError("run identity does not match loaded artifact identities")

    canonical_architecture = {
        key: CANONICAL_E9_TRAINING_CONFIG[key] for key in architecture_keys
    }
    canonical_experiment = (
        dict(training_config) == CANONICAL_E9_TRAINING_CONFIG
        and dict(architecture.__dict__) == canonical_architecture
    )
    independently_eligible = (
        canonical_experiment
        and not run["pilot_training"] and not run["bounded_training"]
        and not run["bounded_validation"] and run["train_query_complete"]
        and not run["train_query_is_pilot"] and run["validation_query_complete"]
        and not run["validation_query_is_pilot"] and run["train_spatial_complete"]
        and not run["train_spatial_is_pilot"] and run["validation_spatial_complete"]
        and not run["validation_spatial_is_pilot"] and not run["source_git_dirty"]
        and not run["train_validation_overlap"] and not provenance["source_git_dirty"]
        and spatial_identities["train"]["production_eligible"]
        and spatial_identities["validation"]["production_eligible"]
        and not query_identities["train"]["source_git_dirty"]
        and not query_identities["validation"]["source_git_dirty"]
    )
    if run["production_eligible"] != independently_eligible:
        raise E9ValidationError("forged E9 production eligibility")
    if require_production and not independently_eligible:
        raise E9ValidationError("E9 checkpoint is not production eligible")
    if expected_e3_identity is not None and e3_identity != dict(expected_e3_identity):
        raise E9ValidationError("E3 identity mismatch")
    if expected_source_git_commit is not None:
        if _GIT_COMMIT.fullmatch(expected_source_git_commit) is None:
            raise E9ValidationError("expected E9 source Git commit is invalid")
        if provenance["source_git_commit"] != expected_source_git_commit:
            raise E9ValidationError("E9 source Git identity mismatch")
    elif require_production:
        raise E9ValidationError("production E9 requires an expected source Git commit")
    if expected_dino_identity is not None:
        expected_dino = validate_e9_dino_identity(
            expected_dino_identity, "expected runtime DINO identity"
        )
        if validated_dino["train"] != expected_dino:
            raise E9ValidationError("E9 checkpoint/runtime DINO identity mismatch")
    elif require_production:
        raise E9ValidationError("production E9 requires an expected DINO identity")
    if expected_checkpoint_path is not None and not Path(expected_checkpoint_path).is_file():
        raise E9ValidationError("E9 checkpoint path does not exist")
    return {
        "production_eligible": independently_eligible,
        "canonical_experiment": canonical_experiment,
        "architecture": architecture,
    }


def load_e9_adapter(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    require_production: bool = True,
    expected_e3_identity: Mapping[str, Any] | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_source_git_commit: str | None = None,
    expected_dino_identity: Mapping[str, Any] | None = None,
) -> SparseRegionAlignmentAdapter:
    path = Path(path)
    if require_production and expected_checkpoint_sha256 is None:
        raise E9ValidationError(
            "production E9 requires an external adapter checkpoint SHA256"
        )
    if expected_checkpoint_sha256 is not None:
        expected_checkpoint_sha256 = _require_sha256(
            expected_checkpoint_sha256, "expected E9 checkpoint"
        )
        if not path.is_file() or sha256_file(path) != expected_checkpoint_sha256:
            raise E9ValidationError("E9 adapter checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    result = validate_e9_checkpoint(
        checkpoint,
        require_production=require_production,
        expected_e3_identity=expected_e3_identity,
        expected_checkpoint_path=path,
        expected_source_git_commit=expected_source_git_commit,
        expected_dino_identity=expected_dino_identity,
    )
    adapter = SparseRegionAlignmentAdapter(result["architecture"])
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.to(device).eval()
    return adapter


__all__ = [
    "E9_ADAPTER_CHECKPOINT_FORMAT",
    "CANONICAL_E9_TRAINING_CONFIG",
    "E9_EPOCH_DIAGNOSTIC_KEYS",
    "GATE_PRIOR_FEATURE_VERSION",
    "E9ValidationError",
    "SparseRegionAlignmentConfig",
    "SparseRegionAlignmentAdapter",
    "SparseRegionRepresentations",
    "SparseRegionScoreOutput",
    "E9MILOutput",
    "compute_chunked_mil_scores",
    "max_valid_normalized_attention",
    "pool_native_grid_to_training_geometry",
    "symmetric_infonce",
    "compute_e9_loss",
    "validate_e9_training_config",
    "validate_e9_checkpoint",
    "validate_e9_dino_identity",
    "load_e9_adapter",
]
