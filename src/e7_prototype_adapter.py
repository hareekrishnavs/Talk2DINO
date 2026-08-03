"""Learned retrieval-conditioned visual prototypes for Talk2DINO E7."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.e7_training_bank import (
    E7BankIdentity,
    MAPPED_QUERY_EMBED_DIM,
    load_e7_training_bank,
)
from src.retrieval_grounded_prototypes import (
    normalized_logsumexp,
    select_adaptive_unique_image_candidates,
)


ADAPTER_CHECKPOINT_FORMAT = "talk2dino-e7-adapter-v1"


def normalize_frozen_embeddings(
    value: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Detach serialized/frozen embeddings and restore unit float32 geometry."""

    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    normalized_input = value.detach().to(dtype=torch.float32)
    if not torch.isfinite(normalized_input).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = normalized_input.norm(dim=-1, keepdim=True)
    if torch.any(norms == 0):
        raise ValueError(f"{name} contains a zero-norm vector")
    normalized = normalized_input / norms
    if not torch.isfinite(normalized).all():
        raise ValueError(f"{name} normalization produced non-finite values")
    return normalized


def normalize_frozen_retrieval_candidates(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Normalize valid frozen candidates while preserving zero padding."""

    if (
        not torch.is_tensor(value)
        or value.ndim != 3
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} must be a floating-point [B, M, D] tensor")
    if valid_mask.shape != value.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError(f"{name} valid mask must be boolean with shape [B, M]")
    detached = value.detach().to(dtype=torch.float32)
    valid_mask = valid_mask.detach().to(device=detached.device)
    if not torch.isfinite(detached).all():
        raise ValueError(f"{name} contains non-finite values")
    normalized = torch.zeros_like(detached)
    if torch.any(valid_mask):
        normalized[valid_mask] = normalize_frozen_embeddings(
            detached[valid_mask],
            name,
        )
    return normalized


def _stable_topk_by_score_and_index(
    scores: torch.Tensor,
    indices: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count == 0:
        return scores[:, :0], indices[:, :0]
    index_order = torch.argsort(indices, dim=-1, stable=True)
    scores = scores.gather(1, index_order)
    indices = indices.gather(1, index_order)
    score_order = torch.argsort(
        scores,
        dim=-1,
        descending=True,
        stable=True,
    )[:, :count]
    return scores.gather(1, score_order), indices.gather(1, score_order)


def _exact_chunked_cosine_topk(
    query_embeddings: torch.Tensor,
    bank_embeddings: torch.Tensor,
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact E7 top-k with float32 normalization at each serialized-bank chunk."""

    if (
        not torch.is_tensor(query_embeddings)
        or query_embeddings.ndim != 2
        or not query_embeddings.is_floating_point()
    ):
        raise ValueError("query_embeddings must be a floating-point matrix")
    if (
        not torch.is_tensor(bank_embeddings)
        or bank_embeddings.ndim != 2
        or not bank_embeddings.is_floating_point()
    ):
        raise ValueError("bank_embeddings must be a floating-point matrix")
    if query_embeddings.shape[1] != bank_embeddings.shape[1]:
        raise ValueError("query and bank embedding dimensions must match")
    for name, value in (("top_k", top_k), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    query = normalize_frozen_embeddings(query_embeddings, "retrieval queries")
    bank_size = bank_embeddings.shape[0]
    retained = min(top_k, bank_size)
    best_scores = query.new_empty((query.shape[0], 0))
    best_indices = torch.empty(
        (query.shape[0], 0),
        dtype=torch.int64,
        device=query.device,
    )
    for start in range(0, bank_size, chunk_size):
        stop = min(start + chunk_size, bank_size)
        chunk = normalize_frozen_embeddings(
            bank_embeddings[start:stop].to(device=query.device),
            "retrieval-bank caption embeddings",
        )
        chunk_scores = query @ chunk.T
        chunk_indices = torch.arange(
            start,
            stop,
            device=query.device,
            dtype=torch.int64,
        ).expand(query.shape[0], -1)
        best_scores, best_indices = _stable_topk_by_score_and_index(
            torch.cat((best_scores, chunk_scores), dim=1),
            torch.cat((best_indices, chunk_indices), dim=1),
            retained,
        )
    return best_scores, best_indices


@dataclass(frozen=True)
class RetrievalPrototypeAdapterConfig:
    embedding_dim: int = 768
    num_prototypes: int = 3
    num_attention_heads: int = 8
    num_cross_attention_layers: int = 2
    ffn_dim: int = 1536
    dropout: float = 0.10
    alpha_max: float = 0.35
    beta_max: float = 0.30
    retrieval_count: int = 64

    def __post_init__(self) -> None:
        for name in (
            "embedding_dim",
            "num_prototypes",
            "num_attention_heads",
            "num_cross_attention_layers",
            "ffn_dim",
            "retrieval_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.num_attention_heads:
            raise ValueError("embedding_dim must be divisible by num_attention_heads")
        for name in ("dropout", "alpha_max", "beta_max"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.alpha_max <= 1:
            raise ValueError("alpha_max must be in [0, 1]")
        if not 0 <= self.beta_max <= 1:
            raise ValueError("beta_max must be in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        value = dict(value)
        expected = {field.name for field in fields(cls)}
        unknown = sorted(set(value).difference(expected))
        if unknown:
            raise ValueError(f"unknown adapter architecture keys: {unknown}")
        return cls(**value)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, config: RetrievalPrototypeAdapterConfig):
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
        attended, _ = self.attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        queries = queries + self.attention_dropout(attended)
        return queries + self.ffn(self.ffn_norm(queries))


@dataclass(frozen=True)
class AdapterOutput:
    prototypes: torch.Tensor
    mode_vectors: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    retrieval_count: torch.Tensor


class RetrievalPrototypeAdapter(nn.Module):
    """Permutation-invariant query-conditioned prototype generator."""

    def __init__(
        self,
        config: RetrievalPrototypeAdapterConfig | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if config is None:
            config = RetrievalPrototypeAdapterConfig()
        elif not isinstance(config, RetrievalPrototypeAdapterConfig):
            config = RetrievalPrototypeAdapterConfig.from_mapping(config)
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
            [_CrossAttentionBlock(config) for _ in range(config.num_cross_attention_layers)]
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
    ) -> AdapterOutput:
        dimension = self.config.embedding_dim
        if (
            not torch.is_tensor(mapped_query)
            or mapped_query.ndim != 2
            or mapped_query.shape[1] != dimension
        ):
            raise ValueError(f"mapped_query must have shape [B, {dimension}]")
        if (
            not torch.is_tensor(retrieved_routed_vectors)
            or retrieved_routed_vectors.ndim != 3
            or retrieved_routed_vectors.shape[0] != mapped_query.shape[0]
            or retrieved_routed_vectors.shape[2] != dimension
        ):
            raise ValueError(
                "retrieved_routed_vectors must have shape [B, M, embedding_dim]"
            )
        expected_retrieval_shape = retrieved_routed_vectors.shape[:2]
        if tuple(retrieval_scores.shape) != expected_retrieval_shape:
            raise ValueError("retrieval_scores must have shape [B, M]")
        if tuple(retrieval_valid_mask.shape) != expected_retrieval_shape:
            raise ValueError("retrieval_valid_mask must have shape [B, M]")
        if not mapped_query.is_floating_point():
            raise ValueError("mapped_query must be floating point")
        if not retrieved_routed_vectors.is_floating_point():
            raise ValueError("retrieved vectors must be floating point")
        if not retrieval_scores.is_floating_point():
            raise ValueError("retrieval scores must be floating point")
        if retrieval_valid_mask.dtype != torch.bool:
            raise ValueError("retrieval_valid_mask must be boolean")
        if not torch.isfinite(mapped_query).all():
            raise ValueError("mapped_query contains non-finite values")
        if not torch.isfinite(retrieved_routed_vectors).all():
            raise ValueError("retrieved vectors contain non-finite values")
        if not torch.isfinite(retrieval_scores).all():
            raise ValueError("retrieval scores contain non-finite values")

        batch_size, memory_count, _ = retrieved_routed_vectors.shape
        mapped = F.normalize(mapped_query, dim=-1)
        retrieval_count = retrieval_valid_mask.sum(dim=-1)
        has_retrieval = retrieval_count > 0
        if memory_count == 0:
            prototypes = mapped[:, None, :].expand(
                -1, self.config.num_prototypes, -1
            )
            zeros_alpha = mapped.new_zeros(
                (batch_size, self.config.num_prototypes)
            )
            return AdapterOutput(
                prototypes=prototypes,
                mode_vectors=prototypes,
                alpha=zeros_alpha,
                beta=mapped.new_zeros(batch_size),
                retrieval_count=retrieval_count,
            )

        memory = F.normalize(retrieved_routed_vectors, dim=-1)
        score_features = self.score_encoder(retrieval_scores.unsqueeze(-1))
        memory = memory + score_features
        # MultiheadAttention cannot accept an all-masked row. A zero dummy is
        # exposed only for those rows, whose outputs are replaced exactly below.
        safe_mask = retrieval_valid_mask.clone()
        missing = ~has_retrieval
        if torch.any(missing):
            safe_mask[missing, 0] = True
            memory = memory.clone()
            memory[missing, 0] = 0

        tokens = mapped[:, None, :] + self.mode_tokens[None, :, :].to(mapped)
        for block in self.blocks:
            tokens = block(tokens, memory, safe_mask)
        mode_vectors = F.normalize(self.output_norm(tokens), dim=-1)
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
        if torch.any(missing):
            fallback = mapped[:, None, :].expand_as(prototypes)
            prototypes = torch.where(missing[:, None, None], fallback, prototypes)
            mode_vectors = torch.where(
                missing[:, None, None], fallback, mode_vectors
            )
            alpha = torch.where(missing[:, None], torch.zeros_like(alpha), alpha)
            beta = torch.where(missing, torch.zeros_like(beta), beta)
        if not all(
            torch.isfinite(value).all()
            for value in (prototypes, mode_vectors, alpha, beta)
        ):
            raise RuntimeError("E7 adapter produced non-finite outputs")
        return AdapterOutput(
            prototypes=prototypes,
            mode_vectors=mode_vectors,
            alpha=alpha,
            beta=beta,
            retrieval_count=retrieval_count,
        )


class UniqueImageBatchSampler(Iterable[list[int]]):
    """One deterministically rotating caption per image and epoch."""

    def __init__(self, image_ids: torch.Tensor, batch_size: int, seed: int = 42):
        if not torch.is_tensor(image_ids) or image_ids.ndim != 1:
            raise ValueError("image_ids must be a one-dimensional tensor")
        if image_ids.dtype != torch.int64:
            raise ValueError("image_ids must have dtype int64")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        groups: dict[int, list[int]] = {}
        for row, image_id in enumerate(image_ids.cpu().tolist()):
            groups.setdefault(image_id, []).append(row)
        self._groups = {key: tuple(value) for key, value in groups.items()}
        self._image_ids = tuple(sorted(self._groups))
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self._image_ids) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self._image_ids), generator=generator).tolist()
        selected = []
        for position in order:
            image_id = self._image_ids[position]
            rows = self._groups[image_id]
            caption_position = (self.seed + self.epoch) % len(rows)
            selected.append(rows[caption_position])
        for start in range(0, len(selected), self.batch_size):
            yield selected[start : start + self.batch_size]


@dataclass(frozen=True)
class QueueRetrieval:
    routed_vectors: torch.Tensor
    scores: torch.Tensor
    valid_mask: torch.Tensor
    image_ids: torch.Tensor
    annotation_ids: torch.Tensor
    exclusion_violations: torch.Tensor


class UniqueImageRetrievalQueue:
    """Bounded frozen retrieval memory with at most one row per image."""

    def __init__(
        self,
        *,
        queue_size: int = 16384,
        candidate_pool: int = 256,
        retrieval_count: int = 64,
        retrieval_min_similarity: float = 0.18,
        device: torch.device | str = "cpu",
    ):
        for name, value in (
            ("queue_size", queue_size),
            ("candidate_pool", candidate_pool),
            ("retrieval_count", retrieval_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if candidate_pool < retrieval_count:
            raise ValueError("candidate_pool must be at least retrieval_count")
        if not math.isfinite(retrieval_min_similarity) or not -1 <= retrieval_min_similarity <= 1:
            raise ValueError("retrieval_min_similarity must be finite and in [-1, 1]")
        self.queue_size = queue_size
        self.candidate_pool = candidate_pool
        self.retrieval_count = retrieval_count
        self.retrieval_min_similarity = retrieval_min_similarity
        self.device = torch.device(device)
        self.caption_embeddings = torch.empty((0, 512), device=self.device)
        self.routed_embeddings = torch.empty((0, 768), device=self.device)
        self.image_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.annotation_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self._image_to_slot: dict[int, int] = {}
        self._cursor = 0

    @property
    def active_count(self) -> int:
        return len(self.image_ids)

    def initialize_from_bank(self, bank: Mapping[str, Any], seed: int = 42) -> None:
        sampler = UniqueImageBatchSampler(
            bank["image_ids"],
            batch_size=max(1, len(torch.unique(bank["image_ids"]))),
            seed=seed,
        )
        indices = next(iter(sampler), [])[: self.queue_size]
        index = torch.tensor(indices, dtype=torch.int64)
        self.caption_embeddings = normalize_frozen_embeddings(
            bank["caption_embeddings"][index].to(self.device),
            "queue caption embeddings",
        )
        self.routed_embeddings = normalize_frozen_embeddings(
            bank["routed_target_embeddings"][index].to(self.device),
            "queue routed embeddings",
        )
        self.image_ids = bank["image_ids"][index].to(self.device).detach()
        self.annotation_ids = bank["annotation_ids"][index].to(self.device).detach()
        image_list = self.image_ids.cpu().tolist()
        if len(image_list) != len(set(image_list)):
            raise RuntimeError("queue initialization produced duplicate images")
        self._image_to_slot = {
            image_id: slot for slot, image_id in enumerate(image_list)
        }
        self._cursor = 0

    def retrieve(
        self,
        caption_queries: torch.Tensor,
        query_image_ids: torch.Tensor,
    ) -> QueueRetrieval:
        if caption_queries.ndim != 2 or caption_queries.shape[1] != 512:
            raise ValueError("caption_queries must have shape [B, 512]")
        if query_image_ids.shape != caption_queries.shape[:1] or query_image_ids.dtype != torch.int64:
            raise ValueError("query_image_ids must be int64 with shape [B]")
        if not torch.isfinite(caption_queries).all():
            raise ValueError("caption queries contain non-finite values")
        batch = len(caption_queries)
        output_shape = (batch, self.retrieval_count)
        if self.active_count == 0:
            return QueueRetrieval(
                routed_vectors=caption_queries.new_zeros(
                    (batch, self.retrieval_count, 768)
                ),
                scores=caption_queries.new_zeros(output_shape),
                valid_mask=torch.zeros(output_shape, dtype=torch.bool, device=caption_queries.device),
                image_ids=torch.full(output_shape, -1, dtype=torch.int64, device=caption_queries.device),
                annotation_ids=torch.full(output_shape, -1, dtype=torch.int64, device=caption_queries.device),
                exclusion_violations=torch.zeros((), dtype=torch.int64, device=caption_queries.device),
            )
        query = normalize_frozen_embeddings(caption_queries, "caption queries")
        memory = self.caption_embeddings.to(query)
        scores = query @ memory.T
        memory_image_ids = self.image_ids.to(query_image_ids.device)
        scores = scores.masked_fill(
            query_image_ids[:, None] == memory_image_ids[None, :], -torch.inf
        )
        candidate_count = min(self.candidate_pool, self.active_count)
        ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)[
            :, :candidate_count
        ]
        ranked_scores = scores.gather(1, ranked)
        ranked = ranked[:, : self.retrieval_count]
        ranked_scores = ranked_scores[:, : self.retrieval_count]
        available = ranked.shape[1]
        valid = torch.isfinite(ranked_scores) & (
            ranked_scores >= self.retrieval_min_similarity
        )
        gathered_images = memory_image_ids[ranked]
        gathered_annotations = self.annotation_ids.to(query_image_ids.device)[ranked]
        routed_memory = self.routed_embeddings.to(query.device)
        gathered_routed = routed_memory[ranked]
        if available < self.retrieval_count:
            padding = self.retrieval_count - available
            gathered_routed = F.pad(gathered_routed, (0, 0, 0, padding))
            ranked_scores = F.pad(ranked_scores, (0, padding))
            valid = F.pad(valid, (0, padding), value=False)
            gathered_images = F.pad(gathered_images, (0, padding), value=-1)
            gathered_annotations = F.pad(
                gathered_annotations, (0, padding), value=-1
            )
        gathered_routed = torch.where(
            valid[..., None], gathered_routed, torch.zeros_like(gathered_routed)
        )
        ranked_scores = torch.where(valid, ranked_scores, torch.zeros_like(ranked_scores))
        gathered_images = torch.where(valid, gathered_images, torch.full_like(gathered_images, -1))
        gathered_annotations = torch.where(valid, gathered_annotations, torch.full_like(gathered_annotations, -1))
        violations = (
            valid & (gathered_images == query_image_ids[:, None])
        ).sum()
        if int(violations.detach().cpu()) != 0:
            raise RuntimeError("same-image retrieval exclusion failed")
        return QueueRetrieval(
            routed_vectors=gathered_routed,
            scores=ranked_scores.to(caption_queries.dtype),
            valid_mask=valid,
            image_ids=gathered_images,
            annotation_ids=gathered_annotations,
            exclusion_violations=violations,
        )

    @torch.no_grad()
    def update(
        self,
        caption_embeddings: torch.Tensor,
        routed_embeddings: torch.Tensor,
        image_ids: torch.Tensor,
        annotation_ids: torch.Tensor,
    ) -> None:
        batch = len(image_ids)
        if caption_embeddings.shape != (batch, 512) or routed_embeddings.shape != (batch, 768):
            raise ValueError("queue update embedding shapes are invalid")
        if image_ids.shape != (batch,) or annotation_ids.shape != (batch,):
            raise ValueError("queue update IDs must be vectors")
        image_list = image_ids.detach().cpu().tolist()
        if len(image_list) != len(set(image_list)):
            raise ValueError("queue update batch contains duplicate image IDs")
        captions = normalize_frozen_embeddings(
            caption_embeddings.to(self.device),
            "queue-update caption embeddings",
        )
        routed = normalize_frozen_embeddings(
            routed_embeddings.to(self.device),
            "queue-update routed embeddings",
        )
        images = image_ids.detach().to(self.device)
        annotations = annotation_ids.detach().to(self.device)
        for row, image_id in enumerate(image_list):
            if image_id in self._image_to_slot:
                slot = self._image_to_slot[image_id]
            elif self.active_count < self.queue_size:
                slot = self.active_count
                self.caption_embeddings = torch.cat(
                    (self.caption_embeddings, captions[row : row + 1])
                )
                self.routed_embeddings = torch.cat(
                    (self.routed_embeddings, routed[row : row + 1])
                )
                self.image_ids = torch.cat((self.image_ids, images[row : row + 1]))
                self.annotation_ids = torch.cat(
                    (self.annotation_ids, annotations[row : row + 1])
                )
                self._image_to_slot[image_id] = slot
                continue
            else:
                slot = self._cursor
                old_image = int(self.image_ids[slot].detach().cpu())
                self._image_to_slot.pop(old_image)
                self._cursor = (self._cursor + 1) % self.queue_size
            self.caption_embeddings[slot].copy_(captions[row])
            self.routed_embeddings[slot].copy_(routed[row])
            self.image_ids[slot].copy_(images[row])
            self.annotation_ids[slot].copy_(annotations[row])
            self._image_to_slot[image_id] = slot
        if len(self._image_to_slot) != self.active_count:
            raise RuntimeError("retrieval queue contains duplicate images")


@dataclass(frozen=True)
class E7ScoreOutput:
    base_score: torch.Tensor
    prototype_score: torch.Tensor
    grounded_score: torch.Tensor
    final_score: torch.Tensor


def compute_e7_scores(
    mapped_queries: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    beta: torch.Tensor,
    prototype_temperature: float = 0.10,
) -> E7ScoreOutput:
    if not math.isfinite(prototype_temperature) or prototype_temperature <= 0:
        raise ValueError("prototype_temperature must be finite and positive")
    if mapped_queries.ndim != 2 or targets.ndim != 2 or mapped_queries.shape != targets.shape:
        raise ValueError("mapped_queries and targets must have matching [B, D]")
    if prototypes.ndim != 3 or prototypes.shape[0] != len(mapped_queries) or prototypes.shape[2] != mapped_queries.shape[1]:
        raise ValueError("prototypes must have shape [B, K, D]")
    if beta.shape != mapped_queries.shape[:1]:
        raise ValueError("beta must have shape [B]")
    if not all(torch.isfinite(value).all() for value in (mapped_queries, targets, prototypes, beta)):
        raise ValueError("E7 scoring inputs must be finite")
    if torch.any((beta < 0) | (beta > 1)):
        raise ValueError("beta must be in [0, 1]")
    if any(
        torch.any(value.norm(dim=-1) == 0)
        for value in (mapped_queries, targets, prototypes)
    ):
        raise ValueError("E7 scoring embeddings must have non-zero norms")
    mapped_queries = F.normalize(mapped_queries.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    base = torch.einsum("id,jd->ij", mapped_queries, targets)
    prototype_score = torch.einsum("ikd,jd->ijk", prototypes, targets)
    if torch.count_nonzero(beta).item() == 0:
        grounded = normalized_logsumexp(
            prototype_score, temperature=prototype_temperature, dim=-1
        )
        return E7ScoreOutput(base, prototype_score, grounded, base)
    grounded = normalized_logsumexp(
        prototype_score, temperature=prototype_temperature, dim=-1
    )
    final = (1 - beta[:, None]) * base + beta[:, None] * grounded
    return E7ScoreOutput(base, prototype_score, grounded, final)


def compute_e7_loss(
    scores: E7ScoreOutput,
    prototypes: torch.Tensor,
    mapped_queries: torch.Tensor,
    beta: torch.Tensor,
    *,
    logit_temperature: float = 0.07,
    anchor_weight: float = 0.10,
    diversity_weight: float = 0.05,
    gate_weight: float = 0.001,
    diversity_margin: float = 0.80,
) -> dict[str, torch.Tensor]:
    for name, value, positive in (
        ("logit_temperature", logit_temperature, True),
        ("anchor_weight", anchor_weight, False),
        ("diversity_weight", diversity_weight, False),
        ("gate_weight", gate_weight, False),
        ("diversity_margin", diversity_margin, False),
    ):
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"{name} has an invalid value")
    batch = scores.final_score.shape[0]
    if scores.final_score.shape != (batch, batch):
        raise ValueError("final score must be square")
    labels = torch.arange(batch, device=scores.final_score.device)
    nce = 0.5 * (
        F.cross_entropy(scores.final_score / logit_temperature, labels)
        + F.cross_entropy(scores.final_score.T / logit_temperature, labels)
    )
    anchor = (1 - F.cosine_similarity(
        prototypes, mapped_queries[:, None, :], dim=-1
    )).mean()
    prototype_count = prototypes.shape[1]
    if prototype_count < 2:
        diversity = prototypes.new_zeros(())
    else:
        pairwise = torch.einsum("ikd,ild->ikl", prototypes, prototypes)
        pair_mask = torch.triu(
            torch.ones(
                (prototype_count, prototype_count),
                dtype=torch.bool,
                device=prototypes.device,
            ),
            diagonal=1,
        )
        diversity = F.relu(pairwise[:, pair_mask] - diversity_margin).mean()
    gate = beta.mean()
    total = nce + anchor_weight * anchor + diversity_weight * diversity + gate_weight * gate
    result = {
        "loss": total,
        "nce_loss": nce,
        "anchor_loss": anchor,
        "diversity_loss": diversity,
        "gate_loss": gate,
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError("E7 loss is non-finite")
    return result


_CHECKPOINT_KEYS = {
    "format_version",
    "adapter_state_dict",
    "architecture_config",
    "training_config",
    "e3_identity",
    "train_bank_identity",
    "validation_bank_identity",
    "source_git_provenance",
    "epoch",
    "best_validation_metric",
}
_BANK_IDENTITY_KEYS = {field.name for field in fields(E7BankIdentity)}
_E3_IDENTITY_KEYS = {"config_sha256", "checkpoint_sha256"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def validate_e7_adapter_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_e3_identity: Mapping[str, Any] | None = None,
    expected_train_bank_identity: Mapping[str, Any] | None = None,
    allow_dirty_source: bool = False,
) -> RetrievalPrototypeAdapterConfig:
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != _CHECKPOINT_KEYS:
        raise ValueError("E7 adapter checkpoint has an invalid closed schema")
    if checkpoint["format_version"] != ADAPTER_CHECKPOINT_FORMAT:
        raise ValueError("unsupported E7 adapter checkpoint format")
    config = RetrievalPrototypeAdapterConfig.from_mapping(
        checkpoint["architecture_config"]
    )
    state = checkpoint["adapter_state_dict"]
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise ValueError("adapter_state_dict is invalid")
    if not isinstance(checkpoint["training_config"], Mapping):
        raise ValueError("training_config must be a mapping")
    e3_identity = checkpoint["e3_identity"]
    if not isinstance(e3_identity, Mapping) or set(e3_identity) != _E3_IDENTITY_KEYS:
        raise ValueError("checkpoint E3 identity has an invalid closed schema")
    for value in e3_identity.values():
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("checkpoint E3 identity contains an invalid SHA256")
    for name in ("train_bank_identity", "validation_bank_identity"):
        identity = checkpoint[name]
        if not isinstance(identity, Mapping) or set(identity) != _BANK_IDENTITY_KEYS:
            raise ValueError(f"{name} has an invalid closed schema")
        for sha_key in (
            "source_feature_sha256",
            "annotation_id_fingerprint",
            "e3_config_sha256",
            "e3_checkpoint_sha256",
        ):
            if not isinstance(identity[sha_key], str) or _SHA256.fullmatch(identity[sha_key]) is None:
                raise ValueError(f"{name}.{sha_key} is not a SHA256")
        if identity["format_version"] != "talk2dino-e7-training-bank-v1":
            raise ValueError(f"{name} has an incompatible bank format")
        if (
            isinstance(identity["selected_annotation_count"], bool)
            or not isinstance(identity["selected_annotation_count"], int)
            or identity["selected_annotation_count"] < 0
        ):
            raise ValueError(f"{name} has an invalid annotation count")
        if not math.isclose(
            float(identity["routing_temperature"]),
            0.10,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{name} has an incompatible routing temperature")
        if (
            not isinstance(identity["source_git_commit"], str)
            or _GIT_COMMIT.fullmatch(identity["source_git_commit"]) is None
        ):
            raise ValueError(f"{name} has an invalid source Git commit")
    if checkpoint["train_bank_identity"]["split_name"] != "train":
        raise ValueError("train_bank_identity must identify the train split")
    if checkpoint["validation_bank_identity"]["split_name"] != "val":
        raise ValueError("validation_bank_identity must identify the val split")
    train_identity = checkpoint["train_bank_identity"]
    validation_identity = checkpoint["validation_bank_identity"]
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
    for bank_name, identity in (
        ("train_bank_identity", train_identity),
        ("validation_bank_identity", validation_identity),
    ):
        if identity["e3_config_sha256"] != e3_identity["config_sha256"]:
            raise ValueError(
                f"{bank_name}.e3_config_sha256 does not match "
                "e3_identity.config_sha256"
            )
        if identity["e3_checkpoint_sha256"] != e3_identity["checkpoint_sha256"]:
            raise ValueError(
                f"{bank_name}.e3_checkpoint_sha256 does not match "
                "e3_identity.checkpoint_sha256"
            )
    if expected_e3_identity is not None and dict(e3_identity) != dict(
        expected_e3_identity
    ):
        raise ValueError("adapter E3 identity is incompatible")
    if expected_train_bank_identity is not None and dict(train_identity) != dict(
        expected_train_bank_identity
    ):
        raise ValueError("adapter retrieval-bank identity is incompatible")
    provenance = checkpoint["source_git_provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "source_git_commit", "source_git_dirty", "source_git_diff_sha256"
    }:
        raise ValueError("checkpoint source Git provenance is invalid")
    if (
        not isinstance(provenance["source_git_commit"], str)
        or _GIT_COMMIT.fullmatch(provenance["source_git_commit"]) is None
    ):
        raise ValueError("checkpoint source Git commit is invalid")
    if type(provenance["source_git_dirty"]) is not bool:
        raise ValueError("checkpoint source_git_dirty must be boolean")
    if provenance["source_git_dirty"]:
        diff_sha = provenance["source_git_diff_sha256"]
        if not isinstance(diff_sha, str) or _SHA256.fullmatch(diff_sha) is None:
            raise ValueError("dirty adapter checkpoint lacks a valid diff SHA256")
        if not allow_dirty_source:
            raise ValueError("dirty-source adapter checkpoints are not evaluable")
    elif provenance["source_git_diff_sha256"] is not None:
        raise ValueError("clean adapter checkpoint must have a null diff SHA256")
    if isinstance(checkpoint["epoch"], bool) or not isinstance(checkpoint["epoch"], int) or checkpoint["epoch"] < 0:
        raise ValueError("checkpoint epoch is invalid")
    metric = checkpoint["best_validation_metric"]
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
        raise ValueError("best_validation_metric must be finite")
    return config


def load_e7_adapter_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_e3_identity: Mapping[str, Any] | None = None,
    expected_train_bank_identity: Mapping[str, Any] | None = None,
    expected_embedding_dim: int | None = None,
    allow_dirty_source: bool = False,
) -> tuple[RetrievalPrototypeAdapter, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"E7 adapter checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = validate_e7_adapter_checkpoint(
        checkpoint,
        expected_e3_identity=expected_e3_identity,
        expected_train_bank_identity=expected_train_bank_identity,
        allow_dirty_source=allow_dirty_source,
    )
    if (
        expected_embedding_dim is not None
        and config.embedding_dim != expected_embedding_dim
    ):
        raise ValueError(
            "E7 adapter embedding dimension is incompatible with production "
            f"segmentation: checkpoint={config.embedding_dim}, "
            f"required={expected_embedding_dim}"
        )
    adapter = RetrievalPrototypeAdapter(config)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.to(device).eval()
    return adapter, dict(checkpoint)


@dataclass(frozen=True)
class LearnedRetrievalSettings:
    prototype_candidate_pool: int = 256
    prototype_candidate_pool_max: int = 2048
    prototype_retrieval_count: int = 64
    retrieval_chunk_size: int = 32768
    retrieval_min_similarity: float = 0.18
    prototype_temperature: float = 0.10

    def __post_init__(self):
        for name in (
            "prototype_candidate_pool",
            "prototype_candidate_pool_max",
            "prototype_retrieval_count",
            "retrieval_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.prototype_retrieval_count > self.prototype_candidate_pool:
            raise ValueError("retrieval count exceeds candidate pool")
        if self.prototype_candidate_pool > self.prototype_candidate_pool_max:
            raise ValueError("candidate pool exceeds maximum")
        if not math.isfinite(self.retrieval_min_similarity) or not -1 <= self.retrieval_min_similarity <= 1:
            raise ValueError("invalid minimum similarity")
        if not math.isfinite(self.prototype_temperature) or self.prototype_temperature <= 0:
            raise ValueError("invalid prototype temperature")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]):
        mapping = dict(mapping)
        expected = {field.name for field in fields(cls)}
        unknown = sorted(set(mapping).difference(expected))
        if unknown:
            raise ValueError(f"unknown E7 retrieval settings: {unknown}")
        return cls(**mapping)

    @property
    def identity(self):
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class LearnedPrototypeBatch:
    prototypes: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    valid_mask: torch.Tensor
    retrieval_indices: torch.Tensor
    retrieval_scores: torch.Tensor
    retrieval_count: torch.Tensor


class LearnedRetrievalPrototypes:
    """One-time class retrieval followed directly by the learned adapter."""

    def __init__(
        self,
        bank: Mapping[str, Any],
        adapter: RetrievalPrototypeAdapter,
        settings: LearnedRetrievalSettings,
        *,
        bank_identity: Mapping[str, Any],
        checkpoint_sha256: str,
    ):
        self.bank = bank
        self.adapter = adapter
        self.settings = settings
        self.bank_identity = dict(bank_identity)
        self.checkpoint_sha256 = checkpoint_sha256
        self._cache: dict[str, LearnedPrototypeBatch] = {}

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
    def generate(self, raw_text_embeddings: torch.Tensor, mapped_embeddings: torch.Tensor) -> LearnedPrototypeBatch:
        if raw_text_embeddings.ndim != 2 or raw_text_embeddings.shape[1] != 512:
            raise ValueError("raw class embeddings must have shape [C, 512]")
        if mapped_embeddings.ndim != 2 or mapped_embeddings.shape[1] != 768 or len(mapped_embeddings) != len(raw_text_embeddings):
            raise ValueError("mapped class embeddings must have shape [C, 768]")
        if not torch.isfinite(raw_text_embeddings).all() or not torch.isfinite(mapped_embeddings).all():
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
        valid = torch.zeros((class_count, retrieval_count), dtype=torch.bool, device=mapped.device)
        indices_out = torch.full((class_count, retrieval_count), -1, dtype=torch.int64, device=mapped.device)
        for class_index in range(class_count):
            selected_scores, selected_indices, _, _, _ = select_adaptive_unique_image_candidates(
                top_scores[class_index].cpu(),
                top_indices[class_index].cpu(),
                self.bank["image_ids"],
                initial_pool=self.settings.prototype_candidate_pool,
                maximum_pool=self.settings.prototype_candidate_pool_max,
                retrieval_count=retrieval_count,
                minimum_similarity=self.settings.retrieval_min_similarity,
            )
            count = min(len(selected_scores), retrieval_count)
            if count:
                selected_indices = selected_indices[:count]
                vectors[class_index, :count] = normalize_frozen_embeddings(
                    self.bank["routed_target_embeddings"][selected_indices].to(
                        mapped
                    ),
                    "segmentation retrieval candidates",
                )
                scores[class_index, :count] = selected_scores[:count].to(mapped)
                valid[class_index, :count] = True
                indices_out[class_index, :count] = selected_indices[:count].to(mapped.device)
        output = self.adapter(mapped, vectors, scores, valid)
        result = LearnedPrototypeBatch(
            prototypes=output.prototypes,
            alpha=output.alpha,
            beta=output.beta,
            valid_mask=valid.any(dim=-1)[:, None].expand(-1, self.adapter.config.num_prototypes),
            retrieval_indices=indices_out,
            retrieval_scores=scores,
            retrieval_count=output.retrieval_count,
        )
        self._cache[key] = result
        return result


def load_learned_retrieval_prototypes(
    bank_path: str | Path,
    adapter_path: str | Path,
    settings: LearnedRetrievalSettings,
    *,
    e3_config_sha256: str,
    e3_checkpoint_sha256: str,
    device: torch.device | str,
) -> LearnedRetrievalPrototypes:
    bank = load_e7_training_bank(
        bank_path,
        expected_split="train",
    )
    identity = E7BankIdentity.from_metadata(bank["metadata"]).as_dict()
    dimensions = bank["metadata"]["dimensions"]
    mapped_dimension = dimensions["mapped_query_embeddings"]
    routed_dimension = dimensions["routed_target_embeddings"]
    segmentation_dimension = MAPPED_QUERY_EMBED_DIM
    if not (
        mapped_dimension
        == routed_dimension
        == segmentation_dimension
        == 768
    ):
        raise ValueError(
            "E7 embedding dimensions are incompatible with production "
            "segmentation: "
            f"mapped={mapped_dimension}, routed={routed_dimension}, "
            f"segmentation={segmentation_dimension}"
        )
    expected_e3 = {
        "config_sha256": e3_config_sha256,
        "checkpoint_sha256": e3_checkpoint_sha256,
    }
    if bank["metadata"]["e3_config_sha256"] != e3_config_sha256 or bank["metadata"]["e3_checkpoint_sha256"] != e3_checkpoint_sha256:
        raise ValueError("E7 retrieval bank is incompatible with E3 evaluation")
    adapter, _ = load_e7_adapter_checkpoint(
        adapter_path,
        device=device,
        expected_e3_identity=expected_e3,
        expected_train_bank_identity=identity,
        expected_embedding_dim=segmentation_dimension,
    )
    if adapter.config.retrieval_count != settings.prototype_retrieval_count:
        raise ValueError("adapter and inference retrieval counts differ")
    from src.e6_prototype_bank import sha256_file
    return LearnedRetrievalPrototypes(
        bank,
        adapter,
        settings,
        bank_identity=identity,
        checkpoint_sha256=sha256_file(adapter_path),
    )
