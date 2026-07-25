"""Inference-only retrieval-grounded text prototypes for Talk2DINO E6."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from src.e6_prototype_bank import load_prototype_bank


@dataclass(frozen=True)
class RGTPSettings:
    prototype_candidate_pool: int = 256
    prototype_candidate_pool_max: int = 2048
    prototype_retrieval_count: int = 64
    retrieval_chunk_size: int = 32768
    retrieval_min_similarity: float = 0.18
    prototype_mmr_lambda: float = 0.70
    retrieval_temperature: float = 0.07
    prototype_count: int = 3
    prototype_confidence_center: float = 0.25
    prototype_confidence_scale: float = 0.05
    prototype_anchor_max: float = 0.30
    prototype_fusion_weight: float = 0.25
    prototype_temperature: float = 0.10

    def __post_init__(self) -> None:
        integer_fields = (
            "prototype_candidate_pool",
            "prototype_candidate_pool_max",
            "prototype_retrieval_count",
            "retrieval_chunk_size",
            "prototype_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.prototype_retrieval_count > self.prototype_candidate_pool:
            raise ValueError(
                "prototype_retrieval_count cannot exceed "
                "prototype_candidate_pool"
            )
        if self.prototype_candidate_pool > self.prototype_candidate_pool_max:
            raise ValueError(
                "prototype_candidate_pool cannot exceed "
                "prototype_candidate_pool_max"
            )
        finite_fields = (
            "retrieval_min_similarity",
            "prototype_mmr_lambda",
            "retrieval_temperature",
            "prototype_confidence_center",
            "prototype_confidence_scale",
            "prototype_anchor_max",
            "prototype_fusion_weight",
            "prototype_temperature",
        )
        for name in finite_fields:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not -1 <= self.retrieval_min_similarity <= 1:
            raise ValueError("retrieval_min_similarity must be in [-1, 1]")
        if not 0 <= self.prototype_mmr_lambda <= 1:
            raise ValueError("prototype_mmr_lambda must be in [0, 1]")
        if self.retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be strictly positive")
        if self.prototype_confidence_scale <= 0:
            raise ValueError(
                "prototype_confidence_scale must be strictly positive"
            )
        if not 0 <= self.prototype_anchor_max <= 1:
            raise ValueError("prototype_anchor_max must be in [0, 1]")
        if not 0 <= self.prototype_fusion_weight <= 1:
            raise ValueError("prototype_fusion_weight must be in [0, 1]")
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be strictly positive")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None):
        mapping = {} if mapping is None else dict(mapping)
        expected = {field.name for field in fields(cls)}
        unexpected = sorted(set(mapping).difference(expected))
        if unexpected:
            raise ValueError(f"unknown RGTP settings: {unexpected}")
        return cls(**mapping)

    @property
    def identity(self) -> tuple[Any, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class PrototypeBank:
    caption_embeddings: torch.Tensor
    routed_dino_embeddings: torch.Tensor
    image_ids: torch.Tensor
    annotation_ids: torch.Tensor
    metadata: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        path,
        *,
        allow_pilot: bool = False,
        allow_incomplete: bool = False,
        allow_dirty_source: bool = False,
        expected_config_sha256: str | None = None,
        expected_checkpoint_sha256: str | None = None,
    ):
        payload = load_prototype_bank(
            path,
            allow_pilot=allow_pilot,
            allow_dirty_source=allow_dirty_source,
            require_complete=not allow_incomplete,
        )
        actual_config_sha = payload["metadata"]["e3_config_sha256"]
        if (
            expected_config_sha256 is not None
            and actual_config_sha != expected_config_sha256
        ):
            raise ValueError(
                "prototype bank was built from a different E3 configuration"
            )
        actual_sha = payload["metadata"]["checkpoint_sha256"]
        if (
            expected_checkpoint_sha256 is not None
            and actual_sha != expected_checkpoint_sha256
        ):
            raise ValueError(
                "prototype bank was built from a different E3 checkpoint"
            )
        return cls(
            caption_embeddings=payload["caption_embeddings"],
            routed_dino_embeddings=payload["routed_dino_embeddings"],
            image_ids=payload["image_ids"],
            annotation_ids=payload["annotation_ids"],
            metadata=payload["metadata"],
        )

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            self.metadata["format_version"],
            self.metadata["source_feature_sha256"],
            self.metadata["checkpoint_sha256"],
            self.metadata["annotation_id_fingerprint"],
            self.metadata["selected_annotation_count"],
        )


@dataclass(frozen=True)
class GroundedPrototypeBatch:
    prototypes: torch.Tensor
    valid_mask: torch.Tensor
    confidence: torch.Tensor
    retrieval_indices: torch.Tensor
    retrieval_scores: torch.Tensor
    candidate_pool_used: torch.Tensor
    deduplicated_candidate_count: torch.Tensor
    threshold_valid_count: torch.Tensor
    selected_retrieval_count: torch.Tensor


def _validate_embedding_matrix(value, name):
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D]")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def _stable_topk(scores, indices, count):
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


def exact_chunked_topk(
    query_embeddings,
    bank_embeddings,
    top_k,
    chunk_size,
):
    """Exact deterministic top-k without constructing the full [C,N] matrix."""
    _validate_embedding_matrix(query_embeddings, "query_embeddings")
    _validate_embedding_matrix(bank_embeddings, "bank_embeddings")
    if query_embeddings.shape[1] != bank_embeddings.shape[1]:
        raise ValueError("query and bank embedding dimensions must match")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")

    query = query_embeddings.float()
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
        chunk = bank_embeddings[start:stop].to(
            device=query.device,
            dtype=query.dtype,
        )
        chunk_scores = query @ chunk.transpose(0, 1)
        chunk_indices = torch.arange(
            start,
            stop,
            device=query.device,
            dtype=torch.int64,
        ).expand(query.shape[0], -1)
        best_scores, best_indices = _stable_topk(
            torch.cat((best_scores, chunk_scores), dim=1),
            torch.cat((best_indices, chunk_indices), dim=1),
            retained,
        )
    return best_scores, best_indices


def deduplicate_by_image(scores, indices, image_ids):
    """Keep the highest-ranked caption for each image."""
    if scores.ndim != 1 or indices.ndim != 1 or scores.shape != indices.shape:
        raise ValueError("scores and indices must be aligned vectors")
    if image_ids.ndim != 1:
        raise ValueError("image_ids must be a vector")
    kept_scores = []
    kept_indices = []
    seen = set()
    for score, index in zip(scores, indices):
        bank_index = int(index.item())
        image_id = int(image_ids[bank_index].item())
        if image_id in seen:
            continue
        seen.add(image_id)
        kept_scores.append(score)
        kept_indices.append(index)
    if not kept_scores:
        return scores[:0], indices[:0]
    return torch.stack(kept_scores), torch.stack(kept_indices)


def select_adaptive_unique_image_candidates(
    ranked_scores,
    ranked_indices,
    image_ids,
    *,
    initial_pool,
    maximum_pool,
    retrieval_count,
    minimum_similarity,
):
    """Select a deterministic prefix with enough threshold-valid unique images."""

    if (
        ranked_scores.ndim != 1
        or ranked_indices.ndim != 1
        or ranked_scores.shape != ranked_indices.shape
    ):
        raise ValueError("ranked scores and indices must be aligned vectors")
    if ranked_indices.dtype != torch.int64:
        raise ValueError("ranked_indices must have dtype torch.int64")
    if image_ids.ndim != 1:
        raise ValueError("image_ids must be a vector")
    if not ranked_scores.is_floating_point() or not torch.isfinite(
        ranked_scores
    ).all():
        raise ValueError("ranked_scores must be finite and floating point")
    for name, value in (
        ("initial_pool", initial_pool),
        ("maximum_pool", maximum_pool),
        ("retrieval_count", retrieval_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if initial_pool > maximum_pool:
        raise ValueError("initial_pool cannot exceed maximum_pool")
    if not math.isfinite(minimum_similarity):
        raise ValueError("minimum_similarity must be finite")

    available = min(len(ranked_scores), maximum_pool)
    if available == 0:
        return (
            ranked_scores[:0],
            ranked_indices[:0],
            0,
            0,
            0,
        )

    prefix = min(initial_pool, available)
    while True:
        scores, indices = deduplicate_by_image(
            ranked_scores[:prefix],
            ranked_indices[:prefix],
            image_ids,
        )
        deduplicated_count = len(scores)
        threshold_mask = scores >= minimum_similarity
        valid_scores = scores[threshold_mask]
        valid_indices = indices[threshold_mask]
        valid_count = len(valid_scores)
        tail_below_threshold = bool(
            ranked_scores[prefix - 1] < minimum_similarity
        )
        if (
            valid_count >= retrieval_count
            or tail_below_threshold
            or prefix >= available
        ):
            return (
                valid_scores,
                valid_indices,
                prefix,
                deduplicated_count,
                valid_count,
            )
        prefix = min(prefix * 2, available)


def _deterministic_argmax(values, tie_break_ids):
    maximum = values.max()
    tied = torch.nonzero(values == maximum, as_tuple=False).flatten()
    if tied.numel() == 1:
        return int(tied[0])
    tied_ids = tie_break_ids[tied]
    return int(tied[torch.argmin(tied_ids)])


def deterministic_mmr(
    text_similarities,
    routed_embeddings,
    selection_count,
    mmr_lambda,
    candidate_indices=None,
):
    """Return selected local candidate positions in deterministic MMR order."""
    if text_similarities.ndim != 1:
        raise ValueError("text_similarities must be a vector")
    _validate_embedding_matrix(routed_embeddings, "routed_embeddings")
    if routed_embeddings.shape[0] != text_similarities.shape[0]:
        raise ValueError("MMR inputs must contain the same number of candidates")
    if not torch.isfinite(text_similarities).all():
        raise ValueError("text_similarities must be finite")
    if selection_count <= 0:
        raise ValueError("selection_count must be positive")
    if not math.isfinite(mmr_lambda) or not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be finite and in [0, 1]")
    count = min(selection_count, len(text_similarities))
    if candidate_indices is None:
        candidate_indices = torch.arange(
            len(text_similarities),
            device=text_similarities.device,
        )
    else:
        candidate_indices = candidate_indices.to(text_similarities.device)
    embeddings = F.normalize(routed_embeddings.float(), dim=-1)
    selected = []
    available = torch.ones(
        len(text_similarities),
        dtype=torch.bool,
        device=text_similarities.device,
    )
    for _ in range(count):
        if selected:
            redundancy = (
                embeddings
                @ embeddings[torch.tensor(selected, device=embeddings.device)].T
            ).max(dim=-1).values
            values = (
                mmr_lambda * text_similarities
                - (1 - mmr_lambda) * redundancy
            )
        else:
            values = text_similarities
        values = values.masked_fill(~available, -torch.inf)
        selected_index = _deterministic_argmax(values, candidate_indices)
        selected.append(selected_index)
        available[selected_index] = False
    return torch.tensor(
        selected,
        dtype=torch.int64,
        device=text_similarities.device,
    )


def deterministic_spherical_kmeans(
    embeddings,
    relevance,
    num_clusters,
    max_iterations=10,
):
    """Cluster normalized vectors with relevance-first farthest initialization."""
    _validate_embedding_matrix(embeddings, "embeddings")
    if relevance.ndim != 1 or len(relevance) != len(embeddings):
        raise ValueError("relevance must align with embeddings")
    if not torch.isfinite(relevance).all():
        raise ValueError("relevance must be finite")
    if num_clusters <= 0 or max_iterations <= 0:
        raise ValueError("cluster and iteration counts must be positive")
    count = len(embeddings)
    if count == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=embeddings.device),
            embeddings.new_empty((0, embeddings.shape[1])),
        )
    cluster_count = min(num_clusters, count)
    normalized = F.normalize(embeddings.float(), dim=-1)
    if torch.any(normalized.norm(dim=-1) == 0):
        raise ValueError("spherical k-means does not accept zero vectors")
    tie_ids = torch.arange(count, device=embeddings.device)
    first = _deterministic_argmax(relevance, tie_ids)
    center_indices = [first]
    while len(center_indices) < cluster_count:
        similarities = (
            normalized
            @ normalized[
                torch.tensor(center_indices, device=normalized.device)
            ].T
        )
        farthest_value = similarities.max(dim=-1).values
        farthest_value[center_indices] = torch.inf
        minimum = farthest_value.min()
        tied = torch.nonzero(
            farthest_value == minimum,
            as_tuple=False,
        ).flatten()
        center_indices.append(int(tied.min()))
    centers = normalized[
        torch.tensor(center_indices, device=normalized.device)
    ].clone()
    assignments = torch.full(
        (count,),
        -1,
        dtype=torch.int64,
        device=normalized.device,
    )
    for _ in range(max_iterations):
        next_assignments = (normalized @ centers.T).argmax(dim=-1)
        if torch.equal(next_assignments, assignments):
            break
        assignments = next_assignments
        next_centers = []
        for cluster in range(cluster_count):
            members = normalized[assignments == cluster]
            if len(members) == 0:
                next_centers.append(centers[cluster])
            else:
                center = members.mean(dim=0)
                center_norm = center.norm()
                if center_norm <= torch.finfo(center.dtype).eps:
                    next_centers.append(centers[cluster])
                else:
                    next_centers.append(center / center_norm)
        centers = torch.stack(next_centers)
    if not torch.isfinite(centers).all():
        raise RuntimeError("spherical k-means produced non-finite centers")
    return assignments, centers


def retrieval_confidence(similarities, center=0.25, scale=0.05):
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("confidence center/scale must be finite and scale positive")
    if similarities.numel() == 0:
        return similarities.new_zeros(())
    if not torch.isfinite(similarities).all():
        raise ValueError("retrieval similarities must be finite")
    best = torch.topk(
        similarities.flatten(),
        k=min(8, similarities.numel()),
    ).values
    return torch.sigmoid((best.mean() - center) / scale)


def anchor_prototypes(
    base_embeddings,
    mode_embeddings,
    confidence,
    anchor_max=0.30,
):
    if not math.isfinite(anchor_max) or not 0 <= anchor_max <= 1:
        raise ValueError("anchor_max must be finite and in [0, 1]")
    if not torch.isfinite(confidence).all() or torch.any(
        (confidence < 0) | (confidence > 1)
    ):
        raise ValueError("confidence must be finite and in [0, 1]")
    if (
        not torch.isfinite(base_embeddings).all()
        or not torch.isfinite(mode_embeddings).all()
    ):
        raise ValueError("base and mode embeddings must be finite")
    base = F.normalize(base_embeddings.float(), dim=-1)
    modes = F.normalize(mode_embeddings.float(), dim=-1)
    if base.ndim == 1 and modes.ndim == 2:
        if base.norm() == 0 or torch.any(modes.norm(dim=-1) == 0):
            raise ValueError("base and mode embeddings must have non-zero norms")
        alpha = confidence.to(modes) * anchor_max
        mixed = (1 - alpha) * base + alpha * modes
        mixed_norm = mixed.norm(dim=-1, keepdim=True)
        return torch.where(
            mixed_norm > 0,
            mixed / mixed_norm.clamp_min(torch.finfo(mixed.dtype).tiny),
            base.expand_as(mixed),
        )
    if base.ndim == 2 and modes.ndim == 3:
        if torch.any(base.norm(dim=-1) == 0) or torch.any(
            modes.norm(dim=-1) == 0
        ):
            raise ValueError("base and mode embeddings must have non-zero norms")
        alpha = confidence.to(modes)[:, None, None] * anchor_max
        mixed = (1 - alpha) * base[:, None, :] + alpha * modes
        mixed_norm = mixed.norm(dim=-1, keepdim=True)
        return torch.where(
            mixed_norm > 0,
            mixed / mixed_norm.clamp_min(torch.finfo(mixed.dtype).tiny),
            base[:, None, :].expand_as(mixed),
        )
    raise ValueError("anchoring expects [D]/[K,D] or [C,D]/[C,K,D]")


def normalized_logsumexp(
    scores,
    temperature=0.10,
    valid_mask=None,
    dim=-1,
):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        valid_mask = torch.broadcast_to(
            valid_mask.to(device=scores.device, dtype=torch.bool),
            scores.shape,
        )
    if not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError("scores must be a finite floating-point tensor")
    counts = valid_mask.sum(dim=dim)
    masked = scores.masked_fill(~valid_mask, -torch.inf)
    maximum = masked.amax(dim=dim, keepdim=True)
    maximum = torch.where(
        counts.unsqueeze(dim) > 0,
        maximum,
        torch.zeros_like(maximum),
    )
    result = maximum.squeeze(dim) + temperature * (
        torch.logsumexp((masked - maximum) / temperature, dim=dim)
        - counts.clamp_min(1).to(scores.dtype).log()
    )
    return torch.where(counts > 0, result, torch.zeros_like(result))


def fuse_prototype_scores(
    base_scores,
    prototype_scores,
    valid_mask,
    confidence,
    *,
    prototype_fusion_weight=0.25,
    prototype_temperature=0.10,
):
    """Fuse `[...,C]` base scores with `[...,C,K]` prototype scores."""
    if prototype_fusion_weight == 0:
        return base_scores
    if (
        not math.isfinite(prototype_fusion_weight)
        or not 0 <= prototype_fusion_weight <= 1
    ):
        raise ValueError("prototype_fusion_weight must be finite and in [0, 1]")
    if prototype_scores.shape[:-1] != base_scores.shape:
        raise ValueError("prototype_scores must have shape [..., C, K]")
    if (
        not torch.isfinite(base_scores).all()
        or not torch.isfinite(prototype_scores).all()
        or not torch.isfinite(confidence).all()
    ):
        raise ValueError("prototype fusion inputs must be finite")
    class_count, prototype_count = prototype_scores.shape[-2:]
    if valid_mask.shape != (class_count, prototype_count):
        raise ValueError("valid_mask must have shape [C, K]")
    if confidence.shape != (class_count,):
        raise ValueError("confidence must have shape [C]")
    if torch.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be in [0, 1]")
    expanded_mask = valid_mask.to(prototype_scores.device)
    for _ in range(prototype_scores.ndim - 2):
        expanded_mask = expanded_mask.unsqueeze(0)
    grounded = normalized_logsumexp(
        prototype_scores,
        temperature=prototype_temperature,
        valid_mask=expanded_mask,
        dim=-1,
    )
    class_valid = valid_mask.any(dim=-1).to(base_scores.device)
    beta = (
        confidence.to(device=base_scores.device, dtype=base_scores.dtype)
        * prototype_fusion_weight
        * class_valid.to(base_scores.dtype)
    )
    beta_shape = [1] * (base_scores.ndim - 1) + [class_count]
    beta = beta.reshape(beta_shape)
    return (1 - beta) * base_scores + beta * grounded


class RetrievalGroundedPrototypes:
    """CPU-bank retrieval and cached class-prototype construction."""

    def __init__(self, bank: PrototypeBank, settings: RGTPSettings | None = None):
        self.bank = bank
        self.settings = settings or RGTPSettings()
        self._cache: dict[str, GroundedPrototypeBatch] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self):
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(self, raw, mapped):
        digest = hashlib.sha256()
        for tensor in (raw, mapped):
            digest.update(str(tensor.device).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            digest.update(value.numpy().tobytes())
            digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(repr(self.settings.identity).encode("utf-8"))
        digest.update(repr(self.bank.identity).encode("utf-8"))
        return digest.hexdigest()

    @torch.no_grad()
    def generate(self, raw_text_embeddings, mapped_embeddings):
        _validate_embedding_matrix(raw_text_embeddings, "raw_text_embeddings")
        _validate_embedding_matrix(mapped_embeddings, "mapped_embeddings")
        if raw_text_embeddings.shape[0] != mapped_embeddings.shape[0]:
            raise ValueError("raw and mapped class counts must match")
        if raw_text_embeddings.shape[1] != 512:
            raise ValueError("raw_text_embeddings must have dimension 512")
        if mapped_embeddings.shape[1] != 768:
            raise ValueError("mapped_embeddings must have dimension 768")
        key = self._cache_key(raw_text_embeddings, mapped_embeddings)
        raw = F.normalize(raw_text_embeddings.float(), dim=-1)
        mapped = F.normalize(mapped_embeddings.float(), dim=-1)
        if torch.any(raw.norm(dim=-1) == 0) or torch.any(
            mapped.norm(dim=-1) == 0
        ):
            raise ValueError("class embeddings must have non-zero norms")
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        top_scores, top_indices = exact_chunked_topk(
            raw,
            self.bank.caption_embeddings,
            self.settings.prototype_candidate_pool_max,
            self.settings.retrieval_chunk_size,
        )
        top_scores = top_scores.cpu()
        top_indices = top_indices.cpu()
        mapped_cpu = mapped.cpu()
        class_count = len(raw)
        mode_count = self.settings.prototype_count
        prototypes = mapped_cpu.new_zeros((class_count, mode_count, 768))
        valid_mask = torch.zeros(
            (class_count, mode_count),
            dtype=torch.bool,
        )
        confidence = mapped_cpu.new_zeros(class_count)
        retrieval_indices = torch.full(
            (class_count, self.settings.prototype_retrieval_count),
            -1,
            dtype=torch.int64,
        )
        retrieval_scores = mapped_cpu.new_zeros(
            (class_count, self.settings.prototype_retrieval_count),
        )
        candidate_pool_used = torch.zeros(class_count, dtype=torch.int64)
        deduplicated_candidate_count = torch.zeros(
            class_count,
            dtype=torch.int64,
        )
        threshold_valid_count = torch.zeros(
            class_count,
            dtype=torch.int64,
        )
        selected_retrieval_count = torch.zeros(
            class_count,
            dtype=torch.int64,
        )

        for class_index in range(class_count):
            (
                scores,
                indices,
                pool_used,
                deduplicated_count,
                threshold_count,
            ) = select_adaptive_unique_image_candidates(
                top_scores[class_index],
                top_indices[class_index],
                self.bank.image_ids,
                initial_pool=self.settings.prototype_candidate_pool,
                maximum_pool=self.settings.prototype_candidate_pool_max,
                retrieval_count=self.settings.prototype_retrieval_count,
                minimum_similarity=self.settings.retrieval_min_similarity,
            )
            candidate_pool_used[class_index] = pool_used
            deduplicated_candidate_count[class_index] = deduplicated_count
            threshold_valid_count[class_index] = threshold_count
            if len(scores) == 0:
                continue
            routed = self.bank.routed_dino_embeddings[indices.cpu()].to(
                dtype=torch.float32,
            )
            selected_local = deterministic_mmr(
                scores,
                routed,
                self.settings.prototype_retrieval_count,
                self.settings.prototype_mmr_lambda,
                candidate_indices=indices,
            )
            selected_scores = scores[selected_local]
            selected_indices = indices[selected_local]
            selected_routed = routed[selected_local]
            confidence[class_index] = retrieval_confidence(
                selected_scores,
                self.settings.prototype_confidence_center,
                self.settings.prototype_confidence_scale,
            )
            selected_count = len(selected_scores)
            selected_retrieval_count[class_index] = selected_count
            retrieval_indices[class_index, :selected_count] = selected_indices
            retrieval_scores[class_index, :selected_count] = selected_scores

            assignments, _ = deterministic_spherical_kmeans(
                selected_routed,
                selected_scores,
                mode_count,
                max_iterations=10,
            )
            mode_embeddings = []
            for mode_index in range(min(mode_count, selected_count)):
                members = assignments == mode_index
                if not torch.any(members):
                    continue
                member_scores = selected_scores[members]
                relevance_logits = (
                    member_scores - member_scores.max()
                ) / self.settings.retrieval_temperature
                relevance_weights = torch.softmax(relevance_logits, dim=0)
                member_embeddings = selected_routed[members]
                weighted_sum = torch.sum(
                    relevance_weights[:, None] * member_embeddings,
                    dim=0,
                )
                weighted_norm = weighted_sum.norm()
                if weighted_norm <= torch.finfo(weighted_sum.dtype).eps:
                    local_best = _deterministic_argmax(
                        member_scores,
                        torch.nonzero(members, as_tuple=False).flatten(),
                    )
                    mode_embedding = member_embeddings[local_best]
                else:
                    mode_embedding = weighted_sum / weighted_norm
                mode_embeddings.append(mode_embedding)
            if mode_embeddings:
                anchored = anchor_prototypes(
                    mapped_cpu[class_index],
                    torch.stack(mode_embeddings),
                    confidence[class_index],
                    self.settings.prototype_anchor_max,
                )
                valid_count = len(anchored)
                prototypes[class_index, :valid_count] = anchored
                valid_mask[class_index, :valid_count] = True

        result = GroundedPrototypeBatch(
            prototypes=prototypes.to(mapped.device),
            valid_mask=valid_mask.to(mapped.device),
            confidence=confidence.to(mapped.device),
            retrieval_indices=retrieval_indices.to(mapped.device),
            retrieval_scores=retrieval_scores.to(mapped.device),
            candidate_pool_used=candidate_pool_used.to(mapped.device),
            deduplicated_candidate_count=(
                deduplicated_candidate_count.to(mapped.device)
            ),
            threshold_valid_count=threshold_valid_count.to(mapped.device),
            selected_retrieval_count=selected_retrieval_count.to(
                mapped.device
            ),
        )
        if not torch.isfinite(prototypes).all() or not torch.isfinite(
            confidence
        ).all():
            raise RuntimeError("RGTP generation produced non-finite outputs")
        self._cache[key] = result
        return result

    def fuse(
        self,
        patch_features,
        mapped_embeddings,
        generated,
        base_scores=None,
    ):
        if patch_features.shape[-1] != mapped_embeddings.shape[-1]:
            raise ValueError("patch and mapped embedding dimensions must match")
        patches = F.normalize(patch_features.float(), dim=-1)
        mapped = F.normalize(mapped_embeddings.float(), dim=-1)
        if base_scores is None:
            base_scores = torch.einsum("...d,cd->...c", patches, mapped)
        prototype_scores = torch.einsum(
            "...d,ckd->...ck",
            patches,
            generated.prototypes.to(patches),
        )
        return fuse_prototype_scores(
            base_scores,
            prototype_scores,
            generated.valid_mask,
            generated.confidence,
            prototype_fusion_weight=self.settings.prototype_fusion_weight,
            prototype_temperature=self.settings.prototype_temperature,
        )
