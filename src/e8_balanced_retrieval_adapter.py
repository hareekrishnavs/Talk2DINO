"""Balanced retrieval modes and target-conditioned fusion for Talk2DINO E8.

This module is deliberately isolated from the finalized E7 implementation.  It
reuses only frozen-bank validation/retrieval utilities; none of the E7 adapter
or scoring mathematics is changed in place.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.e7_prototype_adapter import (
    _exact_chunked_cosine_topk,
    normalize_frozen_embeddings,
)
from src.e7_training_bank import (
    E7BankIdentity,
    MAPPED_QUERY_EMBED_DIM,
    load_e7_training_bank,
)
from src.retrieval_grounded_prototypes import (
    select_adaptive_unique_image_candidates,
)


E8_ADAPTER_CHECKPOINT_FORMAT = "talk2dino-e8-balanced-adapter-v2"


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _finite_nonnegative(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _closed_dataclass_from_mapping(cls, value: Mapping[str, Any], label: str):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    value = dict(value)
    expected = {field.name for field in fields(cls)}
    actual = set(value)
    unknown = sorted(actual.difference(expected))
    missing = sorted(expected.difference(actual))
    if unknown or missing:
        raise ValueError(
            f"{label} has an invalid closed schema; "
            f"unknown={unknown}, missing={missing}"
        )
    return cls(**value)


@dataclass(frozen=True)
class BalancedRetrievalPrototypeAdapterConfig:
    """Closed architecture configuration for the E8 adapter."""

    embedding_dim: int = 768
    num_prototypes: int = 3
    num_attention_heads: int = 8
    num_cross_attention_layers: int = 2
    ffn_dim: int = 1536
    dropout: float = 0.10
    alpha_max: float = 0.35
    beta_max: float = 0.30
    retrieval_count: int = 64
    assignment_temperature: float = 0.07
    retrieval_weight_temperature: float = 0.07
    responsibility_temperature: float = 0.10
    separation_margin: float = 0.50
    alpha_advantage_scale: float = 0.10
    beta_advantage_scale: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "embedding_dim",
            "num_prototypes",
            "num_attention_heads",
            "num_cross_attention_layers",
            "ffn_dim",
            "retrieval_count",
        ):
            _positive_integer(name, getattr(self, name))
        if self.embedding_dim % self.num_attention_heads:
            raise ValueError(
                "embedding_dim must be divisible by num_attention_heads"
            )
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0 <= float(self.dropout) < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        for name in ("alpha_max", "beta_max"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in (
            "assignment_temperature",
            "retrieval_weight_temperature",
            "responsibility_temperature",
            "alpha_advantage_scale",
            "beta_advantage_scale",
        ):
            _finite_positive(name, getattr(self, name))
        if (
            isinstance(self.separation_margin, bool)
            or not isinstance(self.separation_margin, (int, float))
            or not math.isfinite(float(self.separation_margin))
            or not -1 <= float(self.separation_margin) <= 1
        ):
            raise ValueError("separation_margin must be finite and in [-1, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        return _closed_dataclass_from_mapping(
            cls, value, "E8 adapter architecture"
        )


@dataclass(frozen=True)
class E8LossConfig:
    """Closed E8 objective configuration with the predeclared coefficients."""

    logit_temperature: float = 0.07
    anchor_weight: float = 0.10
    coverage_weight: float = 0.10
    balance_weight: float = 0.05
    sharpness_weight: float = 0.02
    separation_weight: float = 0.05
    alpha_calibration_weight: float = 0.05
    beta_calibration_weight: float = 0.10
    corrupt_abstention_weight: float = 0.10
    beta_usage_weight: float = 0.001

    def __post_init__(self) -> None:
        _finite_positive("logit_temperature", self.logit_temperature)
        for field in fields(self):
            if field.name != "logit_temperature":
                _finite_nonnegative(field.name, getattr(self, field.name))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        return _closed_dataclass_from_mapping(cls, value, "E8 loss")


E8_TRAINING_CONFIG_KEYS = frozenset(
    {
        "seed",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "scheduler",
        "warmup_ratio",
        "amp_dtype",
        "save_best_model",
        "early_stopping_patience",
        "adapter",
        "retrieval",
        "loss",
    }
)
E8_RETRIEVAL_CONFIG_KEYS = frozenset(
    {
        "queue_size",
        "candidate_pool",
        "retrieval_count",
        "retrieval_min_similarity",
    }
)
E8_LOSS_CONFIG_KEYS = frozenset(
    {
        "logit_temperature",
        "prototype_temperature",
        "anchor_weight",
        "coverage_weight",
        "balance_weight",
        "sharpness_weight",
        "separation_weight",
        "alpha_calibration_weight",
        "beta_calibration_weight",
        "corrupt_abstention_weight",
        "beta_usage_weight",
    }
)


def _require_closed_mapping(
    value: Any,
    *,
    expected_keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{label} has an invalid closed schema; "
            f"unknown={sorted(actual.difference(expected_keys))}, "
            f"missing={sorted(set(expected_keys).difference(actual))}"
        )
    return dict(value)


def validate_e8_training_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Deeply validate and copy the one canonical E8 training configuration."""

    config = _require_closed_mapping(
        value,
        expected_keys=E8_TRAINING_CONFIG_KEYS,
        label="E8 training configuration",
    )
    for name in ("seed", "early_stopping_patience"):
        item = config[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("num_epochs", "batch_size"):
        _positive_integer(name, config[name])
    for name in ("learning_rate", "weight_decay", "warmup_ratio"):
        item = config[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{name} must be finite and numeric")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError(
            "learning_rate must be positive and weight_decay non-negative"
        )
    if not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config["optimizer"] != "AdamW" or config["scheduler"] != "cosine":
        raise ValueError("E8 requires AdamW with cosine scheduling")
    if config["amp_dtype"] != "bfloat16":
        raise ValueError("E8 amp_dtype must be bfloat16")
    if type(config["save_best_model"]) is not bool:
        raise ValueError("save_best_model must be boolean")

    adapter_mapping = _require_closed_mapping(
        config["adapter"],
        expected_keys={field.name for field in fields(
            BalancedRetrievalPrototypeAdapterConfig
        )},
        label="E8 training configuration adapter",
    )
    adapter_config = BalancedRetrievalPrototypeAdapterConfig.from_mapping(
        adapter_mapping
    )
    retrieval = _require_closed_mapping(
        config["retrieval"],
        expected_keys=E8_RETRIEVAL_CONFIG_KEYS,
        label="E8 training configuration retrieval",
    )
    for name in ("queue_size", "candidate_pool", "retrieval_count"):
        _positive_integer(f"retrieval.{name}", retrieval[name])
    if retrieval["candidate_pool"] < retrieval["retrieval_count"]:
        raise ValueError("retrieval.candidate_pool must be at least retrieval_count")
    minimum = retrieval["retrieval_min_similarity"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or not -1 <= float(minimum) <= 1
    ):
        raise ValueError(
            "retrieval.retrieval_min_similarity must be finite and in [-1, 1]"
        )
    if retrieval["retrieval_count"] != adapter_config.retrieval_count:
        raise ValueError("adapter and retrieval retrieval_count values must match")

    loss = _require_closed_mapping(
        config["loss"],
        expected_keys=E8_LOSS_CONFIG_KEYS,
        label="E8 training configuration loss",
    )
    _finite_positive("loss.prototype_temperature", loss["prototype_temperature"])
    E8LossConfig.from_mapping(
        {
            name: loss[name]
            for name in E8_LOSS_CONFIG_KEYS
            if name != "prototype_temperature"
        }
    )
    config["adapter"] = adapter_mapping
    config["retrieval"] = retrieval
    config["loss"] = loss
    return config


class _BalancedCrossAttentionBlock(nn.Module):
    def __init__(self, config: BalancedRetrievalPrototypeAdapterConfig):
        super().__init__()
        dimension = config.embedding_dim
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, dimension),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        queries = queries + self.attention_dropout(attended)
        return queries + self.ffn(self.ffn_norm(queries))


def _masked_softmax(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Softmax with exact zero padding and a finite all-masked result."""

    if logits.shape != valid_mask.shape:
        raise ValueError("masked softmax logits and mask shapes must match")
    if valid_mask.dtype != torch.bool:
        raise ValueError("masked softmax mask must be boolean")
    if logits.shape[dim] == 0:
        return torch.zeros_like(logits)
    safe_mask = valid_mask.clone()
    no_valid = ~safe_mask.any(dim=dim, keepdim=True)
    first = [slice(None)] * logits.ndim
    first[dim] = slice(0, 1)
    safe_mask[tuple(first)] |= no_valid
    safe_logits = torch.where(safe_mask, logits, torch.zeros_like(logits))
    probabilities = torch.softmax(
        safe_logits.masked_fill(~safe_mask, -torch.inf), dim=dim
    )
    probabilities = torch.where(valid_mask, probabilities, torch.zeros_like(probabilities))
    return probabilities


def _validate_adapter_inputs(
    config: BalancedRetrievalPrototypeAdapterConfig,
    mapped_query: torch.Tensor,
    retrieved_routed_vectors: torch.Tensor,
    retrieval_scores: torch.Tensor,
    retrieval_valid_mask: torch.Tensor,
) -> None:
    dimension = config.embedding_dim
    if (
        not torch.is_tensor(mapped_query)
        or mapped_query.ndim != 2
        or mapped_query.shape[1] != dimension
        or not mapped_query.is_floating_point()
    ):
        raise ValueError(
            f"mapped_query must be a floating-point [B, {dimension}] tensor"
        )
    if (
        not torch.is_tensor(retrieved_routed_vectors)
        or retrieved_routed_vectors.ndim != 3
        or retrieved_routed_vectors.shape[0] != mapped_query.shape[0]
        or retrieved_routed_vectors.shape[2] != dimension
        or not retrieved_routed_vectors.is_floating_point()
    ):
        raise ValueError(
            "retrieved_routed_vectors must be a floating-point "
            "[B, M, embedding_dim] tensor"
        )
    retrieval_shape = retrieved_routed_vectors.shape[:2]
    if (
        not torch.is_tensor(retrieval_scores)
        or tuple(retrieval_scores.shape) != retrieval_shape
        or not retrieval_scores.is_floating_point()
    ):
        raise ValueError("retrieval_scores must be floating point with shape [B, M]")
    if (
        not torch.is_tensor(retrieval_valid_mask)
        or tuple(retrieval_valid_mask.shape) != retrieval_shape
        or retrieval_valid_mask.dtype != torch.bool
    ):
        raise ValueError("retrieval_valid_mask must be boolean with shape [B, M]")
    devices = {
        mapped_query.device,
        retrieved_routed_vectors.device,
        retrieval_scores.device,
        retrieval_valid_mask.device,
    }
    if len(devices) != 1:
        raise ValueError("all E8 adapter inputs must be on the same device")
    if not torch.isfinite(mapped_query).all():
        raise ValueError("mapped_query contains non-finite values")
    valid_vectors = retrieved_routed_vectors[retrieval_valid_mask]
    valid_scores = retrieval_scores[retrieval_valid_mask]
    if not torch.isfinite(valid_vectors).all():
        raise ValueError("valid retrieved vectors contain non-finite values")
    if not torch.isfinite(valid_scores).all():
        raise ValueError("valid retrieval scores contain non-finite values")
    if torch.any(mapped_query.norm(dim=-1) == 0):
        raise ValueError("mapped_query contains a zero-norm vector")
    if valid_vectors.numel() and torch.any(valid_vectors.norm(dim=-1) == 0):
        raise ValueError("valid retrieved vectors contain a zero-norm vector")


@dataclass(frozen=True)
class BalancedAdapterOutput:
    mode_vectors: torch.Tensor
    prototypes: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    candidate_assignments: torch.Tensor
    candidate_weights: torch.Tensor
    slot_mass: torch.Tensor
    retrieval_count: torch.Tensor


class BalancedRetrievalPrototypeAdapter(nn.Module):
    """Permutation-invariant, balanced retrieval-mode adapter for E8."""

    def __init__(
        self,
        config: BalancedRetrievalPrototypeAdapterConfig
        | Mapping[str, Any]
        | None = None,
    ):
        super().__init__()
        if config is None:
            config = BalancedRetrievalPrototypeAdapterConfig()
        elif not isinstance(config, BalancedRetrievalPrototypeAdapterConfig):
            config = BalancedRetrievalPrototypeAdapterConfig.from_mapping(config)
        self.config = config
        dimension = config.embedding_dim
        self.mode_tokens = nn.Parameter(
            torch.empty(config.num_prototypes, dimension)
        )
        nn.init.normal_(self.mode_tokens, std=0.02)
        self.score_encoder = nn.Sequential(
            nn.Linear(1, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.blocks = nn.ModuleList(
            [
                _BalancedCrossAttentionBlock(config)
                for _ in range(config.num_cross_attention_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.alpha_head = nn.Linear(dimension, 1)
        self.beta_head = nn.Linear(dimension, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, -3.0)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.constant_(self.beta_head.bias, -3.0)

    def forward(
        self,
        mapped_query: torch.Tensor,
        retrieved_routed_vectors: torch.Tensor,
        retrieval_scores: torch.Tensor,
        retrieval_valid_mask: torch.Tensor,
    ) -> BalancedAdapterOutput:
        _validate_adapter_inputs(
            self.config,
            mapped_query,
            retrieved_routed_vectors,
            retrieval_scores,
            retrieval_valid_mask,
        )
        batch_size, memory_count, _ = retrieved_routed_vectors.shape
        prototype_count = self.config.num_prototypes
        mapped = F.normalize(mapped_query.detach().to(torch.float32), dim=-1)
        valid_mask = retrieval_valid_mask.detach()
        retrieval_count = valid_mask.sum(dim=-1)
        has_retrieval = retrieval_count > 0

        if memory_count == 0:
            fallback = mapped[:, None, :].expand(-1, prototype_count, -1)
            return BalancedAdapterOutput(
                mode_vectors=fallback,
                prototypes=fallback,
                alpha=mapped.new_zeros((batch_size, prototype_count)),
                beta=mapped.new_zeros(batch_size),
                candidate_assignments=mapped.new_zeros(
                    (batch_size, 0, prototype_count)
                ),
                candidate_weights=mapped.new_zeros((batch_size, 0)),
                slot_mass=mapped.new_zeros((batch_size, prototype_count)),
                retrieval_count=retrieval_count,
            )

        # Invalid padded entries are zeroed before any operation, so their
        # stored values cannot affect attention, assignments, or gradients.
        detached_vectors = retrieved_routed_vectors.detach().to(torch.float32)
        detached_scores = retrieval_scores.detach().to(torch.float32)
        candidate_vectors = torch.where(
            valid_mask[..., None],
            detached_vectors,
            torch.zeros_like(detached_vectors),
        )
        candidate_norms = candidate_vectors.norm(dim=-1, keepdim=True)
        candidate_vectors = torch.where(
            valid_mask[..., None],
            candidate_vectors / candidate_norms.clamp_min(
                torch.finfo(candidate_vectors.dtype).tiny
            ),
            torch.zeros_like(candidate_vectors),
        )
        safe_scores = torch.where(
            valid_mask, detached_scores, torch.zeros_like(detached_scores)
        )
        memory = candidate_vectors + self.score_encoder(safe_scores.unsqueeze(-1))
        memory = torch.where(valid_mask[..., None], memory, torch.zeros_like(memory))

        # MultiheadAttention rejects all-masked rows.  A zero dummy is exposed
        # only internally; every public output for those rows is replaced by
        # the exact E3 fallback below.
        attention_mask = valid_mask.clone()
        missing = ~has_retrieval
        attention_mask[missing, 0] = True
        memory = memory.clone()
        memory[missing, 0] = 0

        tokens = mapped[:, None, :] + self.mode_tokens[None, :, :].to(mapped)
        for block in self.blocks:
            tokens = block(tokens, memory, attention_mask)
        mode_vectors = F.normalize(self.output_norm(tokens), dim=-1)

        cosine = torch.einsum(
            "bmd,bkd->bmk", candidate_vectors, mode_vectors
        ).clamp(-1.0, 1.0)
        assignment_mask = valid_mask[..., None].expand(-1, -1, prototype_count)
        assignments = _masked_softmax(
            cosine / self.config.assignment_temperature,
            assignment_mask,
            dim=-1,
        )
        candidate_weights = _masked_softmax(
            safe_scores / self.config.retrieval_weight_temperature,
            valid_mask,
            dim=-1,
        )
        slot_mass = torch.einsum("bm,bmk->bk", candidate_weights, assignments)

        alpha = self.config.alpha_max * torch.sigmoid(
            self.alpha_head(tokens).squeeze(-1)
        )
        mixed = (
            (1 - alpha[..., None]) * mapped[:, None, :]
            + alpha[..., None] * mode_vectors
        )
        prototypes = F.normalize(mixed, dim=-1)
        beta = self.config.beta_max * torch.sigmoid(
            self.beta_head(tokens.mean(dim=1)).squeeze(-1)
        )

        fallback = mapped[:, None, :].expand_as(prototypes)
        mode_vectors = torch.where(missing[:, None, None], fallback, mode_vectors)
        prototypes = torch.where(missing[:, None, None], fallback, prototypes)
        alpha = torch.where(missing[:, None], torch.zeros_like(alpha), alpha)
        beta = torch.where(missing, torch.zeros_like(beta), beta)
        assignments = torch.where(
            missing[:, None, None], torch.zeros_like(assignments), assignments
        )
        candidate_weights = torch.where(
            missing[:, None], torch.zeros_like(candidate_weights), candidate_weights
        )
        slot_mass = torch.where(
            missing[:, None], torch.zeros_like(slot_mass), slot_mass
        )

        outputs = (
            mode_vectors,
            prototypes,
            alpha,
            beta,
            assignments,
            candidate_weights,
            slot_mass,
        )
        if not all(torch.isfinite(value).all() for value in outputs):
            raise RuntimeError("E8 adapter produced non-finite outputs")
        return BalancedAdapterOutput(
            mode_vectors=mode_vectors,
            prototypes=prototypes,
            alpha=alpha,
            beta=beta,
            candidate_assignments=assignments,
            candidate_weights=candidate_weights,
            slot_mass=slot_mass,
            retrieval_count=retrieval_count,
        )


@dataclass(frozen=True)
class E8ScoreOutput:
    base_score: torch.Tensor
    prototype_score: torch.Tensor
    grounded_score: torch.Tensor
    responsibility: torch.Tensor
    reliability: torch.Tensor
    effective_beta: torch.Tensor
    final_score: torch.Tensor
    valid_query_mask: torch.Tensor

    @property
    def responsibilities(self) -> torch.Tensor:
        """Plural compatibility alias for the responsibility tensor."""

        return self.responsibility


def compute_responsibility_reliability(
    prototype_scores: torch.Tensor,
    temperature: float = 0.10,
    prototype_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-conditioned prototype responsibilities and reliability.

    ``prototype_scores`` is ``[Q, J, K]``.  The optional validity mask is
    ``[Q, K]`` and deliberately cannot vary with the target.
    """

    _finite_positive("responsibility_temperature", temperature)
    if (
        not torch.is_tensor(prototype_scores)
        or prototype_scores.ndim != 3
        or not prototype_scores.is_floating_point()
    ):
        raise ValueError("prototype_scores must be floating point [Q, J, K]")
    if not torch.isfinite(prototype_scores).all():
        raise ValueError("prototype_scores contain non-finite values")
    query_count, _, prototype_count = prototype_scores.shape
    if prototype_count == 0:
        raise ValueError("prototype_scores must contain at least one prototype")
    if prototype_valid_mask is None:
        valid = torch.ones(
            (query_count, prototype_count),
            dtype=torch.bool,
            device=prototype_scores.device,
        )
    else:
        if (
            not torch.is_tensor(prototype_valid_mask)
            or prototype_valid_mask.shape != (query_count, prototype_count)
            or prototype_valid_mask.dtype != torch.bool
        ):
            raise ValueError("prototype_valid_mask must be boolean [Q, K]")
        if prototype_valid_mask.device != prototype_scores.device:
            raise ValueError("prototype validity mask must share the score device")
        valid = prototype_valid_mask
    expanded_valid = valid[:, None, :].expand_as(prototype_scores)
    responsibility = _masked_softmax(
        prototype_scores / temperature,
        expanded_valid,
        dim=-1,
    )
    valid_count = valid.sum(dim=-1)
    valid_rows = valid_count > 0
    if prototype_count == 1:
        reliability = valid_rows[:, None].expand(-1, prototype_scores.shape[1]).to(
            prototype_scores.dtype
        )
    else:
        entropy_terms = torch.where(
            responsibility > 0,
            responsibility * responsibility.clamp_min(
                torch.finfo(responsibility.dtype).tiny
            ).log(),
            torch.zeros_like(responsibility),
        )
        entropy = -entropy_terms.sum(dim=-1)
        # E8 defines confidence against the fixed K-slot capacity, not against
        # a target-dependent or padding-dependent count.
        reliability = 1 - entropy / math.log(prototype_count)
        reliability = torch.where(
            valid_rows[:, None], reliability, torch.zeros_like(reliability)
        ).clamp(0.0, 1.0)
        # Entropy arithmetic for a uniform distribution can leave a tiny
        # round-off residual. Equal valid scores define exactly zero
        # reliability by the E8 control, so make that invariant explicit.
        masked_max = prototype_scores.masked_fill(
            ~expanded_valid, -torch.inf
        ).amax(dim=-1)
        masked_min = prototype_scores.masked_fill(
            ~expanded_valid, torch.inf
        ).amin(dim=-1)
        identical = (
            (valid_count > 1)[:, None]
            & (masked_max == masked_min)
        )
        reliability = torch.where(
            identical, torch.zeros_like(reliability), reliability
        )
    return responsibility, reliability


def _masked_normalized_logsumexp(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    expanded_valid = valid_mask[:, None, :].expand_as(scores)
    safe_mask = expanded_valid.clone()
    missing = ~valid_mask.any(dim=-1)
    safe_mask[missing, :, 0] = True
    safe_scores = torch.where(safe_mask, scores, torch.zeros_like(scores))
    count = valid_mask.sum(dim=-1).clamp_min(1).to(scores.dtype)
    grounded = temperature * torch.logsumexp(
        safe_scores.masked_fill(~safe_mask, -torch.inf) / temperature,
        dim=-1,
    ) - temperature * count.log()[:, None]
    return torch.where(missing[:, None], torch.zeros_like(grounded), grounded)


def compute_e8_scores(
    mapped_queries: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    beta: torch.Tensor,
    prototype_temperature: float = 0.10,
    responsibility_temperature: float = 0.10,
    *,
    has_retrieval: torch.Tensor | None = None,
    prototype_valid_mask: torch.Tensor | None = None,
    precomputed_base_score: torch.Tensor | None = None,
    targets_are_normalized: bool = False,
) -> E8ScoreOutput:
    """Compute E8 target-conditioned fusion for arbitrary Q and J counts.

    Inference may provide the exact E3 ``precomputed_base_score`` together
    with targets already normalized by the E3 image path.  This avoids a
    second normalization and, more importantly, preserves that exact base
    tensor wherever target-conditioned fusion is inactive.
    """

    _finite_positive("prototype_temperature", prototype_temperature)
    _finite_positive("responsibility_temperature", responsibility_temperature)
    if (
        not torch.is_tensor(mapped_queries)
        or mapped_queries.ndim != 2
        or not mapped_queries.is_floating_point()
    ):
        raise ValueError("mapped_queries must be floating point [Q, D]")
    if (
        not torch.is_tensor(targets)
        or targets.ndim != 2
        or not targets.is_floating_point()
        or targets.shape[1] != mapped_queries.shape[1]
    ):
        raise ValueError("targets must be floating point [J, D]")
    if (
        not torch.is_tensor(prototypes)
        or prototypes.ndim != 3
        or not prototypes.is_floating_point()
        or prototypes.shape[0] != mapped_queries.shape[0]
        or prototypes.shape[2] != mapped_queries.shape[1]
        or prototypes.shape[1] == 0
    ):
        raise ValueError("prototypes must be floating point [Q, K, D]")
    if (
        not torch.is_tensor(beta)
        or beta.shape != mapped_queries.shape[:1]
        or not beta.is_floating_point()
    ):
        raise ValueError("beta must be floating point [Q]")
    if not isinstance(targets_are_normalized, bool):
        raise TypeError("targets_are_normalized must be a bool")
    devices = {
        mapped_queries.device,
        targets.device,
        prototypes.device,
        beta.device,
    }
    if len(devices) != 1:
        raise ValueError("all E8 scoring tensors must share a device")
    if not all(
        torch.isfinite(value).all()
        for value in (mapped_queries, targets, prototypes, beta)
    ):
        raise ValueError("E8 scoring inputs must be finite")
    if torch.any((beta < 0) | (beta > 1)):
        raise ValueError("beta must be in [0, 1]")
    if precomputed_base_score is not None:
        if (
            not torch.is_tensor(precomputed_base_score)
            or precomputed_base_score.shape
            != (mapped_queries.shape[0], targets.shape[0])
            or not precomputed_base_score.is_floating_point()
        ):
            raise ValueError("precomputed_base_score must be floating point [Q, J]")
        if precomputed_base_score.device != mapped_queries.device:
            raise ValueError("precomputed_base_score must share the score device")
        if not torch.isfinite(precomputed_base_score).all():
            raise ValueError("precomputed_base_score contains non-finite values")
    for name, value in (
        ("mapped_queries", mapped_queries),
        ("targets", targets),
        ("prototypes", prototypes),
    ):
        if torch.any(value.norm(dim=-1) == 0):
            raise ValueError(f"{name} contains a zero-norm vector")

    query_count, prototype_count = prototypes.shape[:2]
    if has_retrieval is not None:
        if (
            not torch.is_tensor(has_retrieval)
            or has_retrieval.shape != (query_count,)
            or has_retrieval.dtype != torch.bool
        ):
            raise ValueError("has_retrieval must be boolean [Q]")
        if has_retrieval.device != mapped_queries.device:
            raise ValueError("has_retrieval must share the score device")
    if prototype_valid_mask is not None:
        if (
            not torch.is_tensor(prototype_valid_mask)
            or prototype_valid_mask.shape != (query_count, prototype_count)
            or prototype_valid_mask.dtype != torch.bool
        ):
            raise ValueError("prototype_valid_mask must be boolean [Q, K]")
        if prototype_valid_mask.device != mapped_queries.device:
            raise ValueError("prototype_valid_mask must share the score device")
    if prototype_valid_mask is None:
        valid_rows = (
            has_retrieval
            if has_retrieval is not None
            else torch.ones(query_count, dtype=torch.bool, device=mapped_queries.device)
        )
        prototype_valid_mask = valid_rows[:, None].expand(-1, prototype_count)
    else:
        mask_rows = prototype_valid_mask.any(dim=-1)
        if has_retrieval is not None and not torch.equal(mask_rows, has_retrieval):
            raise ValueError("has_retrieval and prototype_valid_mask disagree")
        valid_rows = mask_rows

    target = targets.detach().to(torch.float32)
    if not targets_are_normalized:
        target = F.normalize(target, dim=-1)
    prototype = F.normalize(prototypes.to(torch.float32), dim=-1)
    beta_value = beta.to(torch.float32)
    if precomputed_base_score is None:
        mapped = F.normalize(mapped_queries.detach().to(torch.float32), dim=-1)
        base = torch.einsum("qd,jd->qj", mapped, target)
    else:
        base = precomputed_base_score
    prototype_score = torch.einsum("qkd,jd->qjk", prototype, target)
    grounded = _masked_normalized_logsumexp(
        prototype_score,
        prototype_valid_mask,
        float(prototype_temperature),
    )
    responsibility, reliability = compute_responsibility_reliability(
        prototype_score,
        float(responsibility_temperature),
        prototype_valid_mask,
    )
    effective_beta = beta_value[:, None] * reliability
    effective_beta = torch.where(
        valid_rows[:, None], effective_beta, torch.zeros_like(effective_beta)
    )
    fused = (1 - effective_beta) * base + effective_beta * grounded
    # The explicit selection preserves bitwise E3 scores whenever fusion is
    # inactive, including beta_max=0 and no-retrieval rows.
    final = torch.where(effective_beta == 0, base, fused)
    result = E8ScoreOutput(
        base_score=base,
        prototype_score=prototype_score,
        grounded_score=grounded,
        responsibility=responsibility,
        reliability=reliability,
        effective_beta=effective_beta,
        final_score=final,
        valid_query_mask=valid_rows,
    )
    if not all(
        torch.isfinite(value).all()
        for value in (
            result.base_score,
            result.prototype_score,
            result.grounded_score,
            result.responsibility,
            result.reliability,
            result.effective_beta,
            result.final_score,
        )
    ):
        raise RuntimeError("E8 scoring produced non-finite outputs")
    return result


def _zero_with_adapter_gradient(output: BalancedAdapterOutput) -> torch.Tensor:
    """A scalar zero connected to adapter outputs for safe empty reductions."""

    return (
        output.mode_vectors.sum()
        + output.prototypes.sum()
        + output.alpha.sum()
        + output.beta.sum()
    ) * 0


def compute_balanced_mode_losses(
    output: BalancedAdapterOutput,
    retrieved_routed_vectors: torch.Tensor,
    retrieval_valid_mask: torch.Tensor,
    *,
    separation_margin: float = 0.50,
) -> dict[str, torch.Tensor]:
    """Coverage, balance, sharpness, and mode-separation losses."""

    if (
        isinstance(separation_margin, bool)
        or not isinstance(separation_margin, (int, float))
        or not math.isfinite(float(separation_margin))
        or not -1 <= float(separation_margin) <= 1
    ):
        raise ValueError("separation_margin must be finite and in [-1, 1]")
    batch, memory, prototype_count = output.candidate_assignments.shape
    dimension = output.mode_vectors.shape[-1]
    if retrieved_routed_vectors.shape != (batch, memory, dimension):
        raise ValueError("retrieved_routed_vectors shape is incompatible")
    if (
        retrieval_valid_mask.shape != (batch, memory)
        or retrieval_valid_mask.dtype != torch.bool
    ):
        raise ValueError("retrieval_valid_mask must be boolean [B, M]")
    if retrieval_valid_mask.device != output.mode_vectors.device:
        raise ValueError("retrieval_valid_mask must share the output device")
    valid_vectors = retrieved_routed_vectors[retrieval_valid_mask]
    if valid_vectors.numel() and not torch.isfinite(valid_vectors).all():
        raise ValueError("valid retrieved vectors contain non-finite values")
    vectors = torch.where(
        retrieval_valid_mask[..., None],
        retrieved_routed_vectors.detach().to(torch.float32),
        torch.zeros_like(retrieved_routed_vectors, dtype=torch.float32),
    )
    vector_norms = vectors.norm(dim=-1, keepdim=True)
    vectors = torch.where(
        retrieval_valid_mask[..., None],
        vectors / vector_norms.clamp_min(torch.finfo(vectors.dtype).tiny),
        torch.zeros_like(vectors),
    )
    modes = F.normalize(output.mode_vectors, dim=-1)
    cosine = torch.einsum("bmd,bkd->bmk", vectors, modes).clamp(-1.0, 1.0)
    assignment = output.candidate_assignments
    weight = output.candidate_weights
    valid_rows = retrieval_valid_mask.any(dim=-1)
    zero = _zero_with_adapter_gradient(output)

    coverage_rows = (
        weight
        * (1 - (assignment * cosine).sum(dim=-1))
    ).sum(dim=-1)
    coverage = coverage_rows[valid_rows].mean() if torch.any(valid_rows) else zero

    enough_candidates = retrieval_valid_mask.sum(dim=-1) >= prototype_count
    target_mass = 1.0 / prototype_count
    balance_rows = prototype_count * (
        (output.slot_mass - target_mass).square().mean(dim=-1)
    )
    balance = (
        balance_rows[enough_candidates].mean()
        if torch.any(enough_candidates)
        else zero
    )

    if prototype_count == 1:
        sharpness = zero
    else:
        entropy = -torch.where(
            assignment > 0,
            assignment
            * assignment.clamp_min(torch.finfo(assignment.dtype).tiny).log(),
            torch.zeros_like(assignment),
        ).sum(dim=-1) / math.log(prototype_count)
        sharpness_rows = (weight * entropy).sum(dim=-1)
        sharpness = (
            sharpness_rows[valid_rows].mean() if torch.any(valid_rows) else zero
        )

    if prototype_count < 2 or not torch.any(valid_rows):
        separation = zero
    else:
        pairwise = torch.einsum("bkd,bld->bkl", modes, modes)
        pair_mask = torch.triu(
            torch.ones(
                (prototype_count, prototype_count),
                dtype=torch.bool,
                device=modes.device,
            ),
            diagonal=1,
        )
        separation = F.relu(
            pairwise[valid_rows][:, pair_mask] - separation_margin
        ).mean()

    result = {
        "coverage_loss": coverage,
        "balance_loss": balance,
        "assignment_sharpness_loss": sharpness,
        "mode_separation_loss": separation,
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError("E8 balanced-mode loss is non-finite")
    return result


def compute_alpha_calibration_loss(
    output: BalancedAdapterOutput,
    mapped_queries: torch.Tensor,
    retrieved_routed_vectors: torch.Tensor,
    retrieval_valid_mask: torch.Tensor,
    *,
    alpha_max: float,
    alpha_advantage_scale: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return alpha calibration loss and its detached benefit target."""

    if (
        isinstance(alpha_max, bool)
        or not isinstance(alpha_max, (int, float))
        or not math.isfinite(float(alpha_max))
        or not 0 <= float(alpha_max) <= 1
    ):
        raise ValueError("alpha_max must be finite and in [0, 1]")
    _finite_positive("alpha_advantage_scale", alpha_advantage_scale)
    batch, memory, prototype_count = output.candidate_assignments.shape
    dimension = output.mode_vectors.shape[-1]
    if mapped_queries.shape != (batch, dimension):
        raise ValueError("mapped_queries shape is incompatible")
    if retrieved_routed_vectors.shape != (batch, memory, dimension):
        raise ValueError("retrieved_routed_vectors shape is incompatible")
    if (
        retrieval_valid_mask.shape != (batch, memory)
        or retrieval_valid_mask.dtype != torch.bool
    ):
        raise ValueError("retrieval_valid_mask must be boolean [B, M]")
    if alpha_max == 0:
        return _zero_with_adapter_gradient(output), torch.zeros_like(output.alpha)

    mapped = F.normalize(mapped_queries.detach().to(torch.float32), dim=-1)
    vectors = torch.where(
        retrieval_valid_mask[..., None],
        retrieved_routed_vectors.detach().to(torch.float32),
        torch.zeros_like(retrieved_routed_vectors, dtype=torch.float32),
    )
    if not torch.isfinite(vectors).all():
        raise ValueError("valid retrieved vectors contain non-finite values")
    vectors = torch.where(
        retrieval_valid_mask[..., None],
        F.normalize(vectors, dim=-1),
        torch.zeros_like(vectors),
    )
    mode = F.normalize(output.mode_vectors, dim=-1)
    mode_cosine = torch.einsum("bkd,bmd->bmk", mode, vectors)
    base_cosine = torch.einsum("bd,bmd->bm", mapped, vectors)

    # Detaching the complete normalized support prevents calibration from
    # changing the assignment merely to manufacture a more favorable target.
    support = (
        output.candidate_weights[..., None]
        * output.candidate_assignments
    ).detach()
    support = torch.where(
        retrieval_valid_mask[..., None], support, torch.zeros_like(support)
    )
    support_mass = support.sum(dim=1)
    denominator = support_mass.clamp_min(torch.finfo(support.dtype).tiny)
    visual_quality = (support * mode_cosine).sum(dim=1) / denominator
    base_quality = (
        support * base_cosine[..., None]
    ).sum(dim=1) / denominator
    supported = support_mass > 0
    target = ((visual_quality - base_quality) / alpha_advantage_scale).clamp(0, 1)
    target = torch.where(supported, target, torch.zeros_like(target)).detach()
    loss = F.smooth_l1_loss(output.alpha / alpha_max, target)
    if not torch.isfinite(loss):
        raise RuntimeError("E8 alpha calibration loss is non-finite")
    return loss, target


def compute_beta_calibration_loss(
    scores: E8ScoreOutput,
    beta: torch.Tensor,
    *,
    beta_max: float,
    logit_temperature: float = 0.07,
    beta_advantage_scale: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return benefit-aware beta calibration and its detached row-CE target."""

    if (
        isinstance(beta_max, bool)
        or not isinstance(beta_max, (int, float))
        or not math.isfinite(float(beta_max))
        or not 0 <= float(beta_max) <= 1
    ):
        raise ValueError("beta_max must be finite and in [0, 1]")
    _finite_positive("logit_temperature", logit_temperature)
    _finite_positive("beta_advantage_scale", beta_advantage_scale)
    batch = scores.base_score.shape[0]
    if scores.base_score.shape != (batch, batch):
        raise ValueError("beta calibration requires a square base score")
    if scores.grounded_score.shape != (batch, batch):
        raise ValueError("beta calibration requires a square grounded score")
    if beta.shape != (batch,):
        raise ValueError("beta must have shape [B]")
    target = torch.zeros_like(beta)
    if beta_max == 0:
        return beta.sum() * 0, target
    labels = torch.arange(batch, device=scores.base_score.device)
    base_ce = F.cross_entropy(
        scores.base_score.detach() / logit_temperature,
        labels,
        reduction="none",
    )
    grounded_ce = F.cross_entropy(
        scores.grounded_score.detach() / logit_temperature,
        labels,
        reduction="none",
    )
    target = ((base_ce - grounded_ce) / beta_advantage_scale).clamp(0, 1)
    target = torch.where(
        scores.valid_query_mask, target, torch.zeros_like(target)
    ).detach()
    # Invalid/no-retrieval rows already have both beta and target equal to zero.
    # Keeping them in the reduction makes the calibration strength independent
    # of how many rows happened to retrieve a candidate in a given batch.
    loss = F.smooth_l1_loss(
        beta / beta_max,
        target,
        reduction="mean",
    )
    if not torch.isfinite(loss):
        raise RuntimeError("E8 beta calibration loss is non-finite")
    return loss, target


def compute_corrupt_abstention_loss(
    corrupted_beta: torch.Tensor | None,
    corrupted_valid_rows: torch.Tensor | None,
    *,
    beta_max: float,
) -> torch.Tensor:
    """Penalize the mean squared gate on rotated, mismatched evidence."""

    if (
        isinstance(beta_max, bool)
        or not isinstance(beta_max, (int, float))
        or not math.isfinite(float(beta_max))
        or not 0 <= float(beta_max) <= 1
    ):
        raise ValueError("beta_max must be finite and in [0, 1]")
    if corrupted_beta is None:
        return torch.zeros((), dtype=torch.float32)
    if corrupted_beta.ndim != 1 or not corrupted_beta.is_floating_point():
        raise ValueError("corrupted_beta must be floating point [B]")
    if not torch.isfinite(corrupted_beta).all():
        raise ValueError("corrupted_beta contains non-finite values")
    if corrupted_valid_rows is None:
        valid = torch.ones_like(corrupted_beta, dtype=torch.bool)
    else:
        if (
            corrupted_valid_rows.shape != corrupted_beta.shape
            or corrupted_valid_rows.dtype != torch.bool
        ):
            raise ValueError("corrupted_valid_rows must be boolean [B]")
        valid = corrupted_valid_rows
    if beta_max == 0 or corrupted_beta.numel() == 0:
        return corrupted_beta.sum() * 0
    # Invalid rows are produced with beta=0 and deliberately remain in the
    # complete-batch reduction.  ``valid`` is retained only as a contract check
    # for callers that provide the retrieval-validity vector.
    return (corrupted_beta / beta_max).square().mean()


def compute_e8_loss(
    scores: E8ScoreOutput,
    adapter_output: BalancedAdapterOutput,
    mapped_queries: torch.Tensor,
    retrieved_routed_vectors: torch.Tensor,
    retrieval_valid_mask: torch.Tensor,
    corrupted_beta: torch.Tensor | None = None,
    corrupted_valid_rows: torch.Tensor | None = None,
    *,
    logit_temperature: float = 0.07,
    anchor_weight: float = 0.10,
    coverage_weight: float = 0.10,
    balance_weight: float = 0.05,
    sharpness_weight: float = 0.02,
    separation_weight: float = 0.05,
    alpha_calibration_weight: float = 0.05,
    beta_calibration_weight: float = 0.10,
    corrupt_abstention_weight: float = 0.10,
    beta_usage_weight: float = 0.001,
    alpha_max: float = 0.35,
    beta_max: float = 0.30,
    separation_margin: float | None = None,
    alpha_advantage_scale: float | None = None,
    beta_advantage_scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the complete, predeclared E8 training objective."""

    loss_config = E8LossConfig(
        logit_temperature=logit_temperature,
        anchor_weight=anchor_weight,
        coverage_weight=coverage_weight,
        balance_weight=balance_weight,
        sharpness_weight=sharpness_weight,
        separation_weight=separation_weight,
        alpha_calibration_weight=alpha_calibration_weight,
        beta_calibration_weight=beta_calibration_weight,
        corrupt_abstention_weight=corrupt_abstention_weight,
        beta_usage_weight=beta_usage_weight,
    )
    if (
        isinstance(alpha_max, bool)
        or not isinstance(alpha_max, (int, float))
        or not math.isfinite(float(alpha_max))
        or not 0 <= float(alpha_max) <= 1
    ):
        raise ValueError("alpha_max must be finite and in [0, 1]")
    if (
        isinstance(beta_max, bool)
        or not isinstance(beta_max, (int, float))
        or not math.isfinite(float(beta_max))
        or not 0 <= float(beta_max) <= 1
    ):
        raise ValueError("beta_max must be finite and in [0, 1]")
    if separation_margin is None:
        separation_margin = 0.50
    if alpha_advantage_scale is None:
        alpha_advantage_scale = 0.10
    if beta_advantage_scale is None:
        beta_advantage_scale = 0.10
    if (
        isinstance(separation_margin, bool)
        or not isinstance(separation_margin, (int, float))
        or not math.isfinite(float(separation_margin))
        or not -1 <= float(separation_margin) <= 1
    ):
        raise ValueError("separation_margin must be finite and in [-1, 1]")
    _finite_positive("alpha_advantage_scale", alpha_advantage_scale)
    _finite_positive("beta_advantage_scale", beta_advantage_scale)

    batch = scores.final_score.shape[0]
    if scores.final_score.shape != (batch, batch):
        raise ValueError("symmetric InfoNCE requires a square final score")
    if mapped_queries.shape != (
        batch,
        adapter_output.prototypes.shape[-1],
    ):
        raise ValueError("mapped_queries are incompatible with adapter output")
    labels = torch.arange(batch, device=scores.final_score.device)
    nce = 0.5 * (
        F.cross_entropy(scores.final_score / logit_temperature, labels)
        + F.cross_entropy(scores.final_score.T / logit_temperature, labels)
    )
    mapped = F.normalize(mapped_queries.detach().to(torch.float32), dim=-1)
    anchor = (
        1
        - F.cosine_similarity(
            adapter_output.prototypes,
            mapped[:, None, :],
            dim=-1,
        )
    ).mean()
    specialization = compute_balanced_mode_losses(
        adapter_output,
        retrieved_routed_vectors,
        retrieval_valid_mask,
        separation_margin=separation_margin,
    )
    alpha_calibration, _ = compute_alpha_calibration_loss(
        adapter_output,
        mapped_queries,
        retrieved_routed_vectors,
        retrieval_valid_mask,
        alpha_max=alpha_max,
        alpha_advantage_scale=alpha_advantage_scale,
    )
    beta_calibration, _ = compute_beta_calibration_loss(
        scores,
        adapter_output.beta,
        beta_max=beta_max,
        logit_temperature=logit_temperature,
        beta_advantage_scale=beta_advantage_scale,
    )
    corrupt_abstention = compute_corrupt_abstention_loss(
        corrupted_beta,
        corrupted_valid_rows,
        beta_max=beta_max,
    ).to(device=adapter_output.beta.device, dtype=adapter_output.beta.dtype)
    beta_usage = adapter_output.beta.mean()
    total = (
        nce
        + loss_config.anchor_weight * anchor
        + loss_config.coverage_weight * specialization["coverage_loss"]
        + loss_config.balance_weight * specialization["balance_loss"]
        + loss_config.sharpness_weight
        * specialization["assignment_sharpness_loss"]
        + loss_config.separation_weight
        * specialization["mode_separation_loss"]
        + loss_config.alpha_calibration_weight * alpha_calibration
        + loss_config.beta_calibration_weight * beta_calibration
        + loss_config.corrupt_abstention_weight * corrupt_abstention
        + loss_config.beta_usage_weight * beta_usage
    )
    result = {
        "loss": total,
        "nce_loss": nce,
        "anchor_loss": anchor,
        **specialization,
        "alpha_calibration_loss": alpha_calibration,
        "beta_calibration_loss": beta_calibration,
        "corrupt_abstention_loss": corrupt_abstention,
        "beta_usage_loss": beta_usage,
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError("E8 loss is non-finite")
    return result


@dataclass(frozen=True)
class CorruptedRetrieval:
    routed_vectors: torch.Tensor
    scores: torch.Tensor
    valid_mask: torch.Tensor
    image_ids: torch.Tensor
    annotation_ids: torch.Tensor | None
    exclusion_violations: torch.Tensor
    invalidated_same_image_count: torch.Tensor


def corrupt_retrieval_by_rotation(
    retrieved_routed_vectors: torch.Tensor,
    retrieval_scores: torch.Tensor,
    retrieval_valid_mask: torch.Tensor,
    retrieval_image_ids: torch.Tensor,
    query_image_ids: torch.Tensor,
    retrieval_annotation_ids: torch.Tensor | None = None,
) -> CorruptedRetrieval:
    """Rotate evidence by one row and reapply leave-one-image-out exclusion."""

    if retrieved_routed_vectors.ndim != 3:
        raise ValueError("retrieved_routed_vectors must have shape [B, M, D]")
    batch, memory = retrieved_routed_vectors.shape[:2]
    expected = (batch, memory)
    if retrieval_scores.shape != expected or not retrieval_scores.is_floating_point():
        raise ValueError("retrieval_scores must be floating point [B, M]")
    if retrieval_valid_mask.shape != expected or retrieval_valid_mask.dtype != torch.bool:
        raise ValueError("retrieval_valid_mask must be boolean [B, M]")
    if retrieval_image_ids.shape != expected or retrieval_image_ids.dtype != torch.int64:
        raise ValueError("retrieval_image_ids must be int64 [B, M]")
    if query_image_ids.shape != (batch,) or query_image_ids.dtype != torch.int64:
        raise ValueError("query_image_ids must be int64 [B]")
    if retrieval_annotation_ids is not None and (
        retrieval_annotation_ids.shape != expected
        or retrieval_annotation_ids.dtype != torch.int64
    ):
        raise ValueError("retrieval_annotation_ids must be int64 [B, M]")
    devices = {
        retrieved_routed_vectors.device,
        retrieval_scores.device,
        retrieval_valid_mask.device,
        retrieval_image_ids.device,
        query_image_ids.device,
    }
    if retrieval_annotation_ids is not None:
        devices.add(retrieval_annotation_ids.device)
    if len(devices) != 1:
        raise ValueError("corruption inputs must share a device")
    if batch < 2:
        valid = torch.zeros_like(retrieval_valid_mask)
        return CorruptedRetrieval(
            routed_vectors=torch.zeros_like(retrieved_routed_vectors),
            scores=torch.zeros_like(retrieval_scores),
            valid_mask=valid,
            image_ids=torch.full_like(retrieval_image_ids, -1),
            annotation_ids=(
                None
                if retrieval_annotation_ids is None
                else torch.full_like(retrieval_annotation_ids, -1)
            ),
            exclusion_violations=torch.zeros(
                (), dtype=torch.int64, device=query_image_ids.device
            ),
            invalidated_same_image_count=torch.zeros(
                (), dtype=torch.int64, device=query_image_ids.device
            ),
        )

    vectors = torch.roll(retrieved_routed_vectors.detach(), shifts=1, dims=0)
    scores = torch.roll(retrieval_scores.detach(), shifts=1, dims=0)
    valid = torch.roll(retrieval_valid_mask.detach(), shifts=1, dims=0)
    image_ids = torch.roll(retrieval_image_ids.detach(), shifts=1, dims=0)
    annotation_ids = (
        None
        if retrieval_annotation_ids is None
        else torch.roll(retrieval_annotation_ids.detach(), shifts=1, dims=0)
    )
    same_image = valid & (image_ids == query_image_ids[:, None])
    invalidated_count = same_image.sum()
    valid = valid & ~same_image
    vectors = torch.where(valid[..., None], vectors, torch.zeros_like(vectors))
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    image_ids = torch.where(valid, image_ids, torch.full_like(image_ids, -1))
    if annotation_ids is not None:
        annotation_ids = torch.where(
            valid, annotation_ids, torch.full_like(annotation_ids, -1)
        )
    violations = (valid & (image_ids == query_image_ids[:, None])).sum()
    return CorruptedRetrieval(
        routed_vectors=vectors,
        scores=scores,
        valid_mask=valid,
        image_ids=image_ids,
        annotation_ids=annotation_ids,
        exclusion_violations=violations,
        invalidated_same_image_count=invalidated_count,
    )


# Descriptive alias used by tests and external diagnostic code.
rotate_corrupted_retrieval = corrupt_retrieval_by_rotation


_E8_CHECKPOINT_KEYS = {
    "format_version",
    "adapter_state_dict",
    "architecture_config",
    "training_config",
    "e3_identity",
    "train_bank_identity",
    "validation_bank_identity",
    "run_identity",
    "source_git_provenance",
    "epoch",
    "best_validation_metric",
    "final_diagnostic_summary",
}
_BANK_IDENTITY_KEYS = {field.name for field in fields(E7BankIdentity)}
_E3_IDENTITY_KEYS = {"config_sha256", "checkpoint_sha256"}
_RUN_IDENTITY_KEYS = {
    "train_bank_is_pilot",
    "validation_bank_is_pilot",
    "train_bank_complete",
    "validation_bank_complete",
    "max_train_batches",
    "max_validation_batches",
    "bounded_training",
    "pilot_training",
    "production_eligible",
}
_SOURCE_GIT_PROVENANCE_KEYS = {
    "source_git_commit",
    "source_git_dirty",
    "source_git_diff_sha256",
}
_LOSS_METRIC_KEYS = {
    "loss",
    "nce_loss",
    "anchor_loss",
    "coverage_loss",
    "balance_loss",
    "assignment_sharpness_loss",
    "mode_separation_loss",
    "alpha_calibration_loss",
    "beta_calibration_loss",
    "corrupt_abstention_loss",
    "beta_usage_loss",
}
E8_DIAGNOSTIC_METRIC_KEYS = frozenset(
    _LOSS_METRIC_KEYS
    | {
        f"{name}_{stat}"
        for name in ("alpha", "beta", "corrupted_beta")
        for stat in ("min", "mean", "max", "std")
    }
    | {
        f"{name}_{stat}"
        for name in (
            "mode_pairwise_cosine",
            "prototype_pairwise_cosine",
            "slot_mass",
            "retrieval_count",
        )
        for stat in ("min", "mean", "max")
    }
    | {
        f"{name}_mean"
        for name in (
            "effective_slot_count",
            "positive_reliability",
            "negative_reliability",
            "effective_beta",
            "positive_base_score",
            "positive_grounded_score",
            "positive_final_score",
        )
    }
    | {
        "retrieval_exclusion_violations",
        "corrupted_retrieval_exclusion_violations",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _validate_sha256_mapping(
    value: Any,
    *,
    expected_keys: set[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{label} has an invalid closed schema")
    if not all(
        isinstance(item, str) and _SHA256.fullmatch(item) is not None
        for item in value.values()
    ):
        raise ValueError(f"{label} contains an invalid SHA256")


def _validate_bank_identity(identity: Any, name: str, split: str) -> None:
    if not isinstance(identity, Mapping) or set(identity) != _BANK_IDENTITY_KEYS:
        raise ValueError(f"{name} has an invalid closed schema")
    if identity["format_version"] != "talk2dino-e7-training-bank-v1":
        raise ValueError(f"{name} has an incompatible E7 bank format")
    if identity["split_name"] != split:
        raise ValueError(f"{name} must identify the {split} split")
    for sha_key in (
        "source_feature_sha256",
        "annotation_id_fingerprint",
        "e3_config_sha256",
        "e3_checkpoint_sha256",
    ):
        value = identity[sha_key]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name}.{sha_key} is not a SHA256")
    count = identity["selected_annotation_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{name} has an invalid annotation count")
    temperature = identity["routing_temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not math.isclose(float(temperature), 0.10, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError(f"{name} has an incompatible routing temperature")
    commit = identity["source_git_commit"]
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError(f"{name} has an invalid source Git commit")


def build_e8_run_identity(
    train_bank_metadata: Mapping[str, Any],
    validation_bank_metadata: Mapping[str, Any],
    *,
    max_train_batches: int | None,
    max_validation_batches: int | None,
    source_git_dirty: bool,
) -> dict[str, Any]:
    """Construct the immutable pilot/production identity from source facts."""

    for label, metadata in (
        ("train bank", train_bank_metadata),
        ("validation bank", validation_bank_metadata),
    ):
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{label} metadata must be a mapping")
        for key in ("is_pilot", "complete"):
            if type(metadata.get(key)) is not bool:
                raise ValueError(f"{label} metadata.{key} must be boolean")
    for name, limit in (
        ("max_train_batches", max_train_batches),
        ("max_validation_batches", max_validation_batches),
    ):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValueError(f"{name} must be a positive integer or null")
    if type(source_git_dirty) is not bool:
        raise ValueError("source_git_dirty must be boolean")

    train_is_pilot = train_bank_metadata["is_pilot"]
    validation_is_pilot = validation_bank_metadata["is_pilot"]
    if train_is_pilot != validation_is_pilot:
        raise ValueError(
            "train and validation banks must have identical is_pilot status"
        )
    train_complete = train_bank_metadata["complete"]
    validation_complete = validation_bank_metadata["complete"]
    bounded = max_train_batches is not None or max_validation_batches is not None
    pilot = train_is_pilot or validation_is_pilot or bounded or source_git_dirty
    production = (
        not pilot
        and train_complete
        and validation_complete
        and not train_is_pilot
        and not validation_is_pilot
        and not source_git_dirty
    )
    return {
        "train_bank_is_pilot": train_is_pilot,
        "validation_bank_is_pilot": validation_is_pilot,
        "train_bank_complete": train_complete,
        "validation_bank_complete": validation_complete,
        "max_train_batches": max_train_batches,
        "max_validation_batches": max_validation_batches,
        "bounded_training": bounded,
        "pilot_training": pilot,
        "production_eligible": production,
    }


def _validate_source_git_provenance(
    provenance: Any,
    *,
    allow_dirty_source: bool,
) -> dict[str, Any]:
    provenance = _require_closed_mapping(
        provenance,
        expected_keys=_SOURCE_GIT_PROVENANCE_KEYS,
        label="E8 checkpoint source Git provenance",
    )
    commit = provenance["source_git_commit"]
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("E8 checkpoint source Git commit is invalid")
    if type(provenance["source_git_dirty"]) is not bool:
        raise ValueError("E8 checkpoint source_git_dirty must be boolean")
    if provenance["source_git_dirty"]:
        diff_sha = provenance["source_git_diff_sha256"]
        if not isinstance(diff_sha, str) or _SHA256.fullmatch(diff_sha) is None:
            raise ValueError("dirty E8 checkpoint lacks a valid diff SHA256")
        if not allow_dirty_source:
            raise ValueError("dirty-source E8 checkpoints are not evaluable")
    elif provenance["source_git_diff_sha256"] is not None:
        raise ValueError("clean E8 checkpoint must have a null diff SHA256")
    return provenance


def _validate_e8_run_identity(
    value: Any,
    *,
    source_git_dirty: bool,
    allow_pilot_checkpoint: bool,
    expected_train_bank_is_pilot: bool | None,
    expected_train_bank_complete: bool | None,
    expected_validation_bank_is_pilot: bool | None,
    expected_validation_bank_complete: bool | None,
) -> dict[str, Any]:
    identity = _require_closed_mapping(
        value,
        expected_keys=_RUN_IDENTITY_KEYS,
        label="E8 checkpoint run_identity",
    )
    for name in (
        "train_bank_is_pilot",
        "validation_bank_is_pilot",
        "train_bank_complete",
        "validation_bank_complete",
        "bounded_training",
        "pilot_training",
        "production_eligible",
    ):
        if type(identity[name]) is not bool:
            raise ValueError(f"run_identity.{name} must be boolean")
    for name in ("max_train_batches", "max_validation_batches"):
        limit = identity[name]
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValueError(
                f"run_identity.{name} must be a positive integer or null"
            )
    if identity["train_bank_is_pilot"] != identity["validation_bank_is_pilot"]:
        raise ValueError(
            "run_identity train and validation bank pilot states differ"
        )
    expected_bounded = (
        identity["max_train_batches"] is not None
        or identity["max_validation_batches"] is not None
    )
    expected_pilot = (
        identity["train_bank_is_pilot"]
        or identity["validation_bank_is_pilot"]
        or expected_bounded
        or source_git_dirty
    )
    expected_production = (
        not expected_pilot
        and identity["train_bank_complete"]
        and identity["validation_bank_complete"]
        and not identity["train_bank_is_pilot"]
        and not identity["validation_bank_is_pilot"]
        and not source_git_dirty
    )
    for name, expected in (
        ("bounded_training", expected_bounded),
        ("pilot_training", expected_pilot),
        ("production_eligible", expected_production),
    ):
        if identity[name] is not expected:
            raise ValueError(f"run_identity.{name} is inconsistent with source facts")

    expected_states = (
        ("train_bank_is_pilot", expected_train_bank_is_pilot),
        ("train_bank_complete", expected_train_bank_complete),
        ("validation_bank_is_pilot", expected_validation_bank_is_pilot),
        ("validation_bank_complete", expected_validation_bank_complete),
    )
    for name, expected in expected_states:
        if expected is not None:
            if type(expected) is not bool:
                raise ValueError(f"expected {name} must be boolean")
            if identity[name] is not expected:
                raise ValueError(f"run_identity.{name} does not match the bank")
    if not identity["production_eligible"] and not allow_pilot_checkpoint:
        raise ValueError(
            "non-production E8 checkpoint requires allow_pilot_checkpoint=True"
        )
    return identity


def _validate_final_diagnostic_summary(value: Any) -> None:
    summary = _require_closed_mapping(
        value,
        expected_keys={"train", "validation"},
        label="E8 final_diagnostic_summary",
    )
    for split in ("train", "validation"):
        metrics = _require_closed_mapping(
            summary[split],
            expected_keys=E8_DIAGNOSTIC_METRIC_KEYS,
            label=f"E8 final_diagnostic_summary.{split}",
        )
        for name, metric in metrics.items():
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
            ):
                raise ValueError(
                    f"final_diagnostic_summary.{split}.{name} must be finite"
                )


def validate_e8_adapter_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_e3_identity: Mapping[str, Any] | None = None,
    expected_train_bank_identity: Mapping[str, Any] | None = None,
    expected_validation_bank_identity: Mapping[str, Any] | None = None,
    expected_architecture_config: Mapping[str, Any] | None = None,
    expected_retrieval_count: int | None = None,
    expected_responsibility_temperature: float | None = None,
    expected_prototype_temperature: float | None = None,
    expected_train_bank_is_pilot: bool | None = None,
    expected_train_bank_complete: bool | None = None,
    expected_validation_bank_is_pilot: bool | None = None,
    expected_validation_bank_complete: bool | None = None,
    allow_pilot_checkpoint: bool = False,
    allow_dirty_source: bool = False,
) -> BalancedRetrievalPrototypeAdapterConfig:
    """Validate the complete closed E8 compact-checkpoint contract."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("E8 adapter checkpoint must be a mapping")
    if checkpoint.get("format_version") != E8_ADAPTER_CHECKPOINT_FORMAT:
        legacy = checkpoint.get("format_version")
        if legacy == "talk2dino-e8-balanced-adapter-v1":
            raise ValueError(
                "legacy E8 v1 checkpoints are unsupported; a v2 artifact is required"
            )
        raise ValueError("unsupported E8 adapter checkpoint format")
    if set(checkpoint) != _E8_CHECKPOINT_KEYS:
        raise ValueError("E8 adapter checkpoint has an invalid closed schema")
    config = BalancedRetrievalPrototypeAdapterConfig.from_mapping(
        checkpoint["architecture_config"]
    )
    training_config = validate_e8_training_config(checkpoint["training_config"])
    if dict(checkpoint["architecture_config"]) != training_config["adapter"]:
        raise ValueError(
            "architecture_config must exactly equal training_config.adapter"
        )
    if expected_architecture_config is not None:
        expected_config = BalancedRetrievalPrototypeAdapterConfig.from_mapping(
            expected_architecture_config
        )
        if config != expected_config:
            raise ValueError("E8 adapter architecture is incompatible")
    if expected_retrieval_count is not None:
        _positive_integer("expected_retrieval_count", expected_retrieval_count)
        if config.retrieval_count != expected_retrieval_count:
            raise ValueError("E8 adapter and inference retrieval counts differ")
    for name, expected, actual in (
        (
            "responsibility temperature",
            expected_responsibility_temperature,
            config.responsibility_temperature,
        ),
        (
            "prototype temperature",
            expected_prototype_temperature,
            training_config["loss"]["prototype_temperature"],
        ),
    ):
        if expected is not None:
            _finite_positive(f"expected {name}", expected)
            if float(actual) != float(expected):
                raise ValueError(f"E8 checkpoint and inference {name}s differ")
    state = checkpoint["adapter_state_dict"]
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str)
        and torch.is_tensor(value)
        and torch.isfinite(value).all()
        for key, value in state.items()
    ):
        raise ValueError("E8 adapter_state_dict is invalid")
    _validate_sha256_mapping(
        checkpoint["e3_identity"],
        expected_keys=_E3_IDENTITY_KEYS,
        label="E8 checkpoint E3 identity",
    )
    train_identity = checkpoint["train_bank_identity"]
    validation_identity = checkpoint["validation_bank_identity"]
    _validate_bank_identity(train_identity, "train_bank_identity", "train")
    _validate_bank_identity(validation_identity, "validation_bank_identity", "val")
    for field_name in (
        "source_git_commit",
        "e3_config_sha256",
        "e3_checkpoint_sha256",
        "format_version",
        "routing_temperature",
    ):
        if train_identity[field_name] != validation_identity[field_name]:
            raise ValueError(
                "train and validation bank identities are incompatible for "
                f"{field_name}"
            )
    e3_identity = checkpoint["e3_identity"]
    for name, identity in (
        ("train_bank_identity", train_identity),
        ("validation_bank_identity", validation_identity),
    ):
        if identity["e3_config_sha256"] != e3_identity["config_sha256"]:
            raise ValueError(f"{name} does not match the E3 config identity")
        if identity["e3_checkpoint_sha256"] != e3_identity["checkpoint_sha256"]:
            raise ValueError(f"{name} does not match the E3 checkpoint identity")
    if expected_e3_identity is not None:
        _validate_sha256_mapping(
            expected_e3_identity,
            expected_keys=_E3_IDENTITY_KEYS,
            label="expected E3 identity",
        )
        if dict(e3_identity) != dict(expected_e3_identity):
            raise ValueError("E8 adapter E3 identity is incompatible")
    if expected_train_bank_identity is not None:
        _validate_bank_identity(
            expected_train_bank_identity,
            "expected_train_bank_identity",
            "train",
        )
        if dict(train_identity) != dict(expected_train_bank_identity):
            raise ValueError("E8 adapter train-bank identity is incompatible")
    if expected_validation_bank_identity is not None:
        _validate_bank_identity(
            expected_validation_bank_identity,
            "expected_validation_bank_identity",
            "val",
        )
        if dict(validation_identity) != dict(expected_validation_bank_identity):
            raise ValueError("E8 adapter validation-bank identity is incompatible")

    provenance = _validate_source_git_provenance(
        checkpoint["source_git_provenance"],
        allow_dirty_source=allow_dirty_source,
    )
    _validate_e8_run_identity(
        checkpoint["run_identity"],
        source_git_dirty=provenance["source_git_dirty"],
        allow_pilot_checkpoint=allow_pilot_checkpoint,
        expected_train_bank_is_pilot=expected_train_bank_is_pilot,
        expected_train_bank_complete=expected_train_bank_complete,
        expected_validation_bank_is_pilot=expected_validation_bank_is_pilot,
        expected_validation_bank_complete=expected_validation_bank_complete,
    )
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("E8 checkpoint epoch is invalid")
    metric = checkpoint["best_validation_metric"]
    if (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(float(metric))
    ):
        raise ValueError("best_validation_metric must be finite")
    _validate_final_diagnostic_summary(checkpoint["final_diagnostic_summary"])
    return config


def load_e8_adapter_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_e3_identity: Mapping[str, Any] | None = None,
    expected_train_bank_identity: Mapping[str, Any] | None = None,
    expected_validation_bank_identity: Mapping[str, Any] | None = None,
    expected_architecture_config: Mapping[str, Any] | None = None,
    expected_embedding_dim: int | None = None,
    expected_retrieval_count: int | None = None,
    expected_responsibility_temperature: float | None = None,
    expected_prototype_temperature: float | None = None,
    expected_train_bank_is_pilot: bool | None = None,
    expected_train_bank_complete: bool | None = None,
    expected_validation_bank_is_pilot: bool | None = None,
    expected_validation_bank_complete: bool | None = None,
    allow_pilot_checkpoint: bool = False,
    allow_dirty_source: bool = False,
) -> tuple[BalancedRetrievalPrototypeAdapter, dict[str, Any]]:
    """Validate identities before constructing and loading an E8 adapter."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"E8 adapter checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = validate_e8_adapter_checkpoint(
        checkpoint,
        expected_e3_identity=expected_e3_identity,
        expected_train_bank_identity=expected_train_bank_identity,
        expected_validation_bank_identity=expected_validation_bank_identity,
        expected_architecture_config=expected_architecture_config,
        expected_retrieval_count=expected_retrieval_count,
        expected_responsibility_temperature=expected_responsibility_temperature,
        expected_prototype_temperature=expected_prototype_temperature,
        expected_train_bank_is_pilot=expected_train_bank_is_pilot,
        expected_train_bank_complete=expected_train_bank_complete,
        expected_validation_bank_is_pilot=expected_validation_bank_is_pilot,
        expected_validation_bank_complete=expected_validation_bank_complete,
        allow_pilot_checkpoint=allow_pilot_checkpoint,
        allow_dirty_source=allow_dirty_source,
    )
    if expected_embedding_dim is not None:
        _positive_integer("expected_embedding_dim", expected_embedding_dim)
        if config.embedding_dim != expected_embedding_dim:
            raise ValueError(
                "E8 adapter embedding dimension is incompatible with production "
                f"segmentation: checkpoint={config.embedding_dim}, "
                f"required={expected_embedding_dim}"
            )
    adapter = BalancedRetrievalPrototypeAdapter(config)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.to(device).eval()
    return adapter, dict(checkpoint)


@dataclass(frozen=True)
class BalancedRetrievalSettings:
    prototype_candidate_pool: int = 256
    prototype_candidate_pool_max: int = 2048
    prototype_retrieval_count: int = 64
    retrieval_chunk_size: int = 32768
    retrieval_min_similarity: float = 0.18
    prototype_temperature: float = 0.10
    responsibility_temperature: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "prototype_candidate_pool",
            "prototype_candidate_pool_max",
            "prototype_retrieval_count",
            "retrieval_chunk_size",
        ):
            _positive_integer(name, getattr(self, name))
        if self.prototype_retrieval_count > self.prototype_candidate_pool:
            raise ValueError("retrieval count exceeds candidate pool")
        if self.prototype_candidate_pool > self.prototype_candidate_pool_max:
            raise ValueError("candidate pool exceeds maximum")
        minimum = self.retrieval_min_similarity
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not math.isfinite(float(minimum))
            or not -1 <= float(minimum) <= 1
        ):
            raise ValueError("retrieval_min_similarity must be finite and in [-1, 1]")
        _finite_positive("prototype_temperature", self.prototype_temperature)
        _finite_positive(
            "responsibility_temperature", self.responsibility_temperature
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        return _closed_dataclass_from_mapping(cls, value, "E8 retrieval settings")

    @property
    def identity(self) -> tuple[Any, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class BalancedPrototypeBatch:
    prototypes: torch.Tensor
    mode_vectors: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    candidate_assignments: torch.Tensor
    candidate_weights: torch.Tensor
    slot_mass: torch.Tensor
    valid_mask: torch.Tensor
    retrieval_indices: torch.Tensor
    retrieval_scores: torch.Tensor
    retrieval_count: torch.Tensor


class BalancedRetrievalPrototypes:
    """One-time class retrieval followed by the frozen balanced E8 adapter."""

    def __init__(
        self,
        bank: Mapping[str, Any],
        adapter: BalancedRetrievalPrototypeAdapter,
        settings: BalancedRetrievalSettings,
        *,
        bank_identity: Mapping[str, Any],
        checkpoint_sha256: str,
    ):
        self.bank = bank
        self.adapter = adapter
        self.settings = settings
        self.bank_identity = dict(bank_identity)
        self.checkpoint_sha256 = checkpoint_sha256
        self._cache: dict[str, BalancedPrototypeBatch] = {}

    def _cache_key(self, raw: torch.Tensor, mapped: torch.Tensor) -> str:
        digest = hashlib.sha256()
        for tensor in (raw, mapped):
            value = tensor.detach().float().cpu().contiguous()
            digest.update(value.numpy().tobytes())
            digest.update(repr(tuple(value.shape)).encode())
        digest.update(repr(self.settings.identity).encode())
        digest.update(repr(sorted(self.bank_identity.items())).encode())
        digest.update(self.checkpoint_sha256.encode())
        return digest.hexdigest()

    @torch.no_grad()
    def generate(
        self,
        raw_text_embeddings: torch.Tensor,
        mapped_embeddings: torch.Tensor,
    ) -> BalancedPrototypeBatch:
        if raw_text_embeddings.ndim != 2 or raw_text_embeddings.shape[1] != 512:
            raise ValueError("raw class embeddings must have shape [C, 512]")
        if (
            mapped_embeddings.ndim != 2
            or mapped_embeddings.shape[1] != 768
            or len(mapped_embeddings) != len(raw_text_embeddings)
        ):
            raise ValueError("mapped class embeddings must have shape [C, 768]")
        if not torch.isfinite(raw_text_embeddings).all() or not torch.isfinite(
            mapped_embeddings
        ).all():
            raise ValueError("class embeddings must be finite")
        key = self._cache_key(raw_text_embeddings, mapped_embeddings)
        if key in self._cache:
            return self._cache[key]
        raw = F.normalize(raw_text_embeddings.float(), dim=-1)
        mapped = F.normalize(mapped_embeddings.float(), dim=-1)
        top_scores, top_indices = _exact_chunked_cosine_topk(
            raw,
            self.bank["caption_embeddings"],
            self.settings.prototype_candidate_pool_max,
            self.settings.retrieval_chunk_size,
        )
        class_count = len(raw)
        retrieval_count = self.settings.prototype_retrieval_count
        vectors = mapped.new_zeros((class_count, retrieval_count, 768))
        scores = mapped.new_zeros((class_count, retrieval_count))
        valid = torch.zeros(
            (class_count, retrieval_count),
            dtype=torch.bool,
            device=mapped.device,
        )
        indices_out = torch.full(
            (class_count, retrieval_count),
            -1,
            dtype=torch.int64,
            device=mapped.device,
        )
        for class_index in range(class_count):
            selected_scores, selected_indices, _, _, _ = (
                select_adaptive_unique_image_candidates(
                    top_scores[class_index].cpu(),
                    top_indices[class_index].cpu(),
                    self.bank["image_ids"],
                    initial_pool=self.settings.prototype_candidate_pool,
                    maximum_pool=self.settings.prototype_candidate_pool_max,
                    retrieval_count=retrieval_count,
                    minimum_similarity=self.settings.retrieval_min_similarity,
                )
            )
            count = min(len(selected_scores), retrieval_count)
            if count:
                chosen = selected_indices[:count]
                vectors[class_index, :count] = normalize_frozen_embeddings(
                    self.bank["routed_target_embeddings"][chosen].to(mapped),
                    "E8 segmentation retrieval candidates",
                )
                scores[class_index, :count] = selected_scores[:count].to(mapped)
                valid[class_index, :count] = True
                indices_out[class_index, :count] = chosen[:count].to(mapped.device)
        output = self.adapter(mapped, vectors, scores, valid)
        prototype_valid = valid.any(dim=-1)[:, None].expand(
            -1, self.adapter.config.num_prototypes
        )
        result = BalancedPrototypeBatch(
            prototypes=output.prototypes,
            mode_vectors=output.mode_vectors,
            alpha=output.alpha,
            beta=output.beta,
            candidate_assignments=output.candidate_assignments,
            candidate_weights=output.candidate_weights,
            slot_mass=output.slot_mass,
            valid_mask=prototype_valid,
            retrieval_indices=indices_out,
            retrieval_scores=scores,
            retrieval_count=output.retrieval_count,
        )
        self._cache[key] = result
        return result


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_balanced_retrieval_prototypes(
    bank_path: str | Path,
    adapter_path: str | Path,
    settings: BalancedRetrievalSettings,
    *,
    e3_config_sha256: str,
    e3_checkpoint_sha256: str,
    device: torch.device | str,
) -> BalancedRetrievalPrototypes:
    bank = load_e7_training_bank(bank_path, expected_split="train")
    identity = E7BankIdentity.from_metadata(bank["metadata"]).as_dict()
    dimensions = bank["metadata"]["dimensions"]
    mapped_dimension = dimensions["mapped_query_embeddings"]
    routed_dimension = dimensions["routed_target_embeddings"]
    if not (
        mapped_dimension
        == routed_dimension
        == MAPPED_QUERY_EMBED_DIM
        == 768
    ):
        raise ValueError(
            "E8 embedding dimensions are incompatible with production "
            f"segmentation: mapped={mapped_dimension}, "
            f"routed={routed_dimension}, required=768"
        )
    expected_e3 = {
        "config_sha256": e3_config_sha256,
        "checkpoint_sha256": e3_checkpoint_sha256,
    }
    production_architecture = BalancedRetrievalPrototypeAdapterConfig()
    if (
        bank["metadata"]["e3_config_sha256"] != e3_config_sha256
        or bank["metadata"]["e3_checkpoint_sha256"] != e3_checkpoint_sha256
    ):
        raise ValueError("E8 retrieval bank is incompatible with E3 evaluation")
    adapter, _ = load_e8_adapter_checkpoint(
        adapter_path,
        device=device,
        expected_e3_identity=expected_e3,
        expected_train_bank_identity=identity,
        expected_architecture_config={
            field.name: getattr(production_architecture, field.name)
            for field in fields(BalancedRetrievalPrototypeAdapterConfig)
        },
        expected_embedding_dim=768,
        expected_retrieval_count=settings.prototype_retrieval_count,
        expected_responsibility_temperature=settings.responsibility_temperature,
        expected_prototype_temperature=settings.prototype_temperature,
        expected_train_bank_is_pilot=bank["metadata"]["is_pilot"],
        expected_train_bank_complete=bank["metadata"]["complete"],
    )
    return BalancedRetrievalPrototypes(
        bank,
        adapter,
        settings,
        bank_identity=identity,
        checkpoint_sha256=_sha256_file(adapter_path),
    )


__all__ = [
    "E8_ADAPTER_CHECKPOINT_FORMAT",
    "E8_DIAGNOSTIC_METRIC_KEYS",
    "E8_TRAINING_CONFIG_KEYS",
    "E8_RETRIEVAL_CONFIG_KEYS",
    "E8_LOSS_CONFIG_KEYS",
    "BalancedRetrievalPrototypeAdapterConfig",
    "E8LossConfig",
    "BalancedAdapterOutput",
    "BalancedRetrievalPrototypeAdapter",
    "E8ScoreOutput",
    "compute_responsibility_reliability",
    "compute_e8_scores",
    "compute_balanced_mode_losses",
    "compute_alpha_calibration_loss",
    "compute_beta_calibration_loss",
    "compute_corrupt_abstention_loss",
    "compute_e8_loss",
    "CorruptedRetrieval",
    "corrupt_retrieval_by_rotation",
    "rotate_corrupted_retrieval",
    "validate_e8_training_config",
    "build_e8_run_identity",
    "validate_e8_adapter_checkpoint",
    "load_e8_adapter_checkpoint",
    "BalancedRetrievalSettings",
    "BalancedPrototypeBatch",
    "BalancedRetrievalPrototypes",
    "load_balanced_retrieval_prototypes",
]
