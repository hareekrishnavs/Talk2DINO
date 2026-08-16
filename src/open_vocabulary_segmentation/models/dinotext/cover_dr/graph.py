"""Directed sparse affinity graphs over frozen DINO patch features."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch


_FLOAT32_ROW_SUM_TOLERANCE = 2e-6


def _feature_norm_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return 5e-3
    if dtype == torch.bfloat16:
        return 1e-2
    return 1e-5


@dataclass(frozen=True)
class DirectedTopKGraph:
    """Owned sparse storage for one directed row-stochastic window graph.

    ``neighbor_indices[i, s] == j`` represents the directed edge ``i -> j``.
    PyTorch tensors are mutable, so immutability means frozen field bindings,
    detached owned storage, and a read-only consumer contract.

    COVER-DR uses a directed row-stochastic graph.
    Do not symmetrize: row edits and later exact counterfactuals depend on
    direction.
    """

    neighbor_indices: torch.Tensor
    transition_weights: torch.Tensor
    edge_affinities: torch.Tensor
    self_loop_fallback: torch.Tensor
    num_nodes: int
    k: int
    affinity_power: float

    def __post_init__(self) -> None:
        for name in (
            "neighbor_indices",
            "transition_weights",
            "edge_affinities",
            "self_loop_fallback",
        ):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                object.__setattr__(
                    self,
                    name,
                    value.detach().clone().contiguous(),
                )
        self.validate()

    def validate(self) -> None:
        """Validate sparse storage without constructing a dense adjacency."""
        if isinstance(self.num_nodes, bool) or not isinstance(self.num_nodes, int):
            raise TypeError("num_nodes must be an integer")
        if isinstance(self.k, bool) or not isinstance(self.k, int):
            raise TypeError("k must be an integer")
        if self.num_nodes < 2:
            raise ValueError("a directed top-k graph requires at least two nodes")
        if not 0 < self.k < self.num_nodes:
            raise ValueError(
                f"k must satisfy 0 < k < N, got k={self.k}, N={self.num_nodes}"
            )
        if (
            isinstance(self.affinity_power, bool)
            or not isinstance(self.affinity_power, (int, float))
            or not math.isfinite(float(self.affinity_power))
            or float(self.affinity_power) <= 0
        ):
            raise ValueError(
                "affinity_power must be finite and strictly positive, got "
                f"{self.affinity_power!r}"
            )

        expected_edges = (self.num_nodes, self.k)
        if self.neighbor_indices.shape != expected_edges:
            raise ValueError(
                "neighbor_indices must have shape "
                f"{expected_edges}, got {tuple(self.neighbor_indices.shape)}"
            )
        for name, value in (
            ("transition_weights", self.transition_weights),
            ("edge_affinities", self.edge_affinities),
        ):
            if value.shape != expected_edges:
                raise ValueError(
                    f"{name} must have shape {expected_edges}, got {tuple(value.shape)}"
                )
        if self.self_loop_fallback.shape != (self.num_nodes,):
            raise ValueError(
                "self_loop_fallback must have shape "
                f"{(self.num_nodes,)}, got {tuple(self.self_loop_fallback.shape)}"
            )
        if self.neighbor_indices.dtype != torch.int64:
            raise TypeError("neighbor_indices must have dtype torch.int64")
        if self.transition_weights.dtype != torch.float32:
            raise TypeError("transition_weights must have dtype torch.float32")
        if self.edge_affinities.dtype != torch.float32:
            raise TypeError("edge_affinities must have dtype torch.float32")
        if self.self_loop_fallback.dtype != torch.bool:
            raise TypeError("self_loop_fallback must have dtype torch.bool")

        devices = {
            self.neighbor_indices.device,
            self.transition_weights.device,
            self.edge_affinities.device,
            self.self_loop_fallback.device,
        }
        if len(devices) != 1:
            raise ValueError("all graph tensors must be on the same device")
        if any(
            value.requires_grad or value.grad_fn is not None
            for value in self._tensor_fields()
        ):
            raise ValueError("graph tensors must be detached from autograd")
        if not torch.isfinite(self.transition_weights).all():
            raise ValueError("transition_weights must be finite")
        if not torch.isfinite(self.edge_affinities).all():
            raise ValueError("edge_affinities must be finite")
        if torch.any(self.transition_weights < 0):
            raise ValueError("transition_weights must be non-negative")
        if torch.any(self.edge_affinities < 0):
            raise ValueError("edge_affinities must be non-negative")
        if torch.any(self.neighbor_indices < 0) or torch.any(
            self.neighbor_indices >= self.num_nodes
        ):
            raise ValueError("neighbor destination is outside [0, N)")

        row_ids = torch.arange(
            self.num_nodes, device=self.neighbor_indices.device
        )
        ordinary = ~self.self_loop_fallback
        if torch.any(self.neighbor_indices[ordinary] == row_ids[ordinary, None]):
            raise ValueError("ordinary graph rows must exclude self destinations")
        if torch.any(ordinary):
            sorted_destinations = self.neighbor_indices[ordinary].sort(dim=1).values
            if torch.any(sorted_destinations[:, 1:] == sorted_destinations[:, :-1]):
                raise ValueError("ordinary graph rows must not repeat destinations")

        affinity_sums = self.edge_affinities.sum(dim=1)
        expected_fallback = affinity_sums == 0
        if not torch.equal(self.self_loop_fallback, expected_fallback):
            raise ValueError(
                "self_loop_fallback must identify exactly the zero-affinity rows"
            )
        fallback = self.self_loop_fallback
        if torch.any(fallback):
            if not torch.equal(
                self.neighbor_indices[fallback, 0], row_ids[fallback]
            ):
                raise ValueError("fallback slot zero must be a self-loop")
            if not torch.equal(
                self.transition_weights[fallback, 0],
                torch.ones_like(self.transition_weights[fallback, 0]),
            ):
                raise ValueError("fallback self-loop must have transition weight one")
            if torch.count_nonzero(self.transition_weights[fallback, 1:]):
                raise ValueError("remaining fallback transition weights must be zero")
            if torch.count_nonzero(self.edge_affinities[fallback]):
                raise ValueError("fallback rows must retain zero edge affinities")

        if torch.any(ordinary):
            expected_weights = (
                self.edge_affinities[ordinary]
                / affinity_sums[ordinary, None]
            )
            if not torch.allclose(
                self.transition_weights[ordinary],
                expected_weights,
                rtol=_FLOAT32_ROW_SUM_TOLERANCE,
                atol=_FLOAT32_ROW_SUM_TOLERANCE,
            ):
                raise ValueError(
                    "positive graph rows must equal affinity divided by row sum"
                )
        if torch.any(
            (self.edge_affinities == 0) & (self.transition_weights != 0) & ordinary[:, None]
        ):
            raise ValueError("zero-affinity ordinary slots must have zero weight")
        row_sums = self.transition_weights.sum(dim=1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=_FLOAT32_ROW_SUM_TOLERANCE,
            atol=_FLOAT32_ROW_SUM_TOLERANCE,
        ):
            raise ValueError("every transition row must sum to one")

    def _tensor_fields(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), torch.Tensor)
        )

    def matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        """Compute ``A @ rhs`` directly from sparse outgoing neighbours."""
        if not isinstance(rhs, torch.Tensor):
            raise TypeError("graph matmul RHS must be a torch.Tensor")
        if rhs.ndim not in (1, 2):
            raise ValueError(
                "graph matmul RHS must have shape [N] or [N, R], got "
                f"{tuple(rhs.shape)}"
            )
        if rhs.shape[0] != self.num_nodes:
            raise ValueError(
                "graph matmul node mismatch: "
                f"graph={self.num_nodes}, rhs={rhs.shape[0]}"
            )
        if rhs.device != self.transition_weights.device:
            raise ValueError("graph and matmul RHS must be on the same device")
        if not rhs.is_floating_point():
            raise TypeError("graph matmul RHS must be floating point")

        gathered = rhs[self.neighbor_indices]
        weights = self.transition_weights.to(dtype=rhs.dtype)
        if rhs.ndim == 1:
            return (weights * gathered).sum(dim=1)
        return (weights.unsqueeze(-1) * gathered).sum(dim=1)

    def to_dense(self) -> torch.Tensor:
        """Materialize ``[N,N]`` adjacency for diagnostics and tests only."""
        dense = torch.zeros(
            (self.num_nodes, self.num_nodes),
            dtype=torch.float32,
            device=self.transition_weights.device,
        )
        dense.scatter_add_(1, self.neighbor_indices, self.transition_weights)
        return dense


@torch.no_grad()
def build_directed_topk_graph(
    dino_features: torch.Tensor,
    *,
    k: int = 12,
    affinity_power: float = 3.0,
) -> DirectedTopKGraph:
    """Build one directed top-k graph from normalized patch features."""
    if not isinstance(dino_features, torch.Tensor):
        raise TypeError("dino_features must be a torch.Tensor")
    if dino_features.ndim != 2:
        raise ValueError(
            "dino_features must have shape [N, D], got "
            f"{tuple(dino_features.shape)}"
        )
    if not dino_features.is_floating_point():
        raise TypeError("dino_features must be floating point")
    num_nodes, embed_dim = dino_features.shape
    if num_nodes < 2 or embed_dim <= 0:
        raise ValueError(
            "dino_features must contain at least two nodes and one feature"
        )
    if isinstance(k, bool) or not isinstance(k, int) or not 0 < k < num_nodes:
        raise ValueError(f"k must satisfy 0 < k < N, got k={k}, N={num_nodes}")
    if (
        isinstance(affinity_power, bool)
        or not isinstance(affinity_power, (int, float))
        or not math.isfinite(float(affinity_power))
        or float(affinity_power) <= 0
    ):
        raise ValueError(
            "affinity_power must be finite and strictly positive, got "
            f"{affinity_power!r}"
        )
    if not torch.isfinite(dino_features).all():
        raise ValueError("dino_features must be finite")

    features_float = dino_features.detach().to(torch.float32)
    norms = features_float.norm(dim=-1)
    if torch.any(norms == 0):
        raise ValueError("dino_features must have non-zero L2 norm")
    tolerance = _feature_norm_tolerance(dino_features.dtype)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=tolerance,
        atol=tolerance,
    ):
        maximum_error = (norms - 1).abs().max().item()
        raise ValueError(
            "dino_features must already be L2-normalized; maximum norm error "
            f"is {maximum_error:.8g}, tolerance is {tolerance:.8g}"
        )

    affinities = (features_float @ features_float.transpose(0, 1)).clamp_min_(0)
    affinities.pow_(float(affinity_power))
    affinities.fill_diagonal_(-torch.inf)

    # Stable sorting preserves ascending destination order for exact ties.
    neighbor_indices = torch.argsort(
        affinities,
        dim=1,
        descending=True,
        stable=True,
    )[:, :k]
    edge_affinities = affinities.gather(1, neighbor_indices)
    row_sums = edge_affinities.sum(dim=1)
    self_loop_fallback = row_sums == 0
    transition_weights = torch.zeros_like(edge_affinities)
    positive_rows = ~self_loop_fallback
    transition_weights[positive_rows] = (
        edge_affinities[positive_rows] / row_sums[positive_rows, None]
    )
    if torch.any(self_loop_fallback):
        fallback_rows = torch.arange(
            num_nodes, device=dino_features.device
        )[self_loop_fallback]
        neighbor_indices[self_loop_fallback, 0] = fallback_rows
        transition_weights[self_loop_fallback, 0] = 1

    return DirectedTopKGraph(
        neighbor_indices=neighbor_indices,
        transition_weights=transition_weights,
        edge_affinities=edge_affinities,
        self_loop_fallback=self_loop_fallback,
        num_nodes=num_nodes,
        k=k,
        affinity_power=float(affinity_power),
    )
