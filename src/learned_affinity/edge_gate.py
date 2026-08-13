"""Part G: pairwise learned edge gate over a FIXED candidate set, replacing
the unary LearnedMetric (src/learned_affinity/metric.py, kept unmodified in
the repository as the ablation baseline this design replaces -- see
RUN_PartG.md for why the unary design failed: a unary transform g(f) applied
identically to every patch structurally cannot move one edge without moving
all of them, and a scrambled g changed only ~13% of kNN edges; L1 could not
be overfit even on a single fixed batch).

    candidates: top-K neighbours of i by cos(f_i, f_j) on the FROZEN
                features, computed ONCE outside this module and passed in as
                `cand_idx` -- never recomputed, reordered, or gradient-
                tracked here. There is consequently no non-differentiable
                top-k selection inside this module's forward at all, and no
                detach is needed anywhere in it (see the module-level note
                below).

    e_ij  = MLP([ f_i, f_j, f_i * f_j, cos(f_i, f_j) ])
    w_ij  = ReLU(cos(f_i, f_j))^kappa * 2*sigmoid(e_ij)
    A     = row_normalize(symmetrize(w))

Symmetrisation is a NAMED NO-OP here, not an oversight: the existing
production graph builder (src/e3_affinity_oracle.py:build_knn_graph,
GRAPH_DIRECTIONALITY = "directed_row_stochastic_knn") has no symmetrisation
step at all -- confirmed by grepping the whole repo for "symmetri", which
returns only comments documenting its absence. "Match whatever reduction
the existing builder uses" therefore means matching NONE; symmetrizing here
would make G4's bitwise-identity requirement against that builder
impossible to satisfy. If the existing builder is ever changed to
symmetrise, this function's docstring is the place to update.

Detach audit: there is no `.detach()` call anywhere in this module.
`cand_idx` is supplied by the caller, already a fixed LongTensor computed
from frozen (non-grad-requiring) features -- it never originates from a
differentiable operation inside forward, so there is nothing to sever. This
is the structural difference from the unary design's
`build_differentiable_knn_graph`, which had to detach its own internally-
computed top-k indices every call."""
from __future__ import annotations

import torch
import torch.nn as nn


class EdgeGate(nn.Module):
    """Pairwise edge gate. See module docstring for the formula.

    Identity at init: the MLP's final layer is zero-initialised, so
    e_ij == 0 and 2*sigmoid(0) == 1.0 for every edge regardless of what the
    hidden layer computes -- w_ij reduces exactly to the existing
    ReLU(cos)^kappa formula, with no residual scale multiplying it (no `r`,
    `gamma`, or sigmoid-gated scalar anywhere in this class -- the ONLY
    zero at init is the final MLP layer itself, so the MLP receives
    full-magnitude gradient from step one, unlike the unary design's
    doubly-near-zero deadlock)."""

    def __init__(self, dim: int = 768, hidden: int = 256, kappa: float = 3.0, K: int = 32):
        super().__init__()
        if dim <= 0 or hidden <= 0:
            raise ValueError("dim and hidden must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if not 0 < K:
            raise ValueError("K must be positive")
        self.dim = dim
        self.kappa = kappa
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, f: torch.Tensor, cand_idx: torch.Tensor, *, return_gate: bool = False):
        """f: [P,dim], L2-normalised, frozen (no grad required on f itself).
        cand_idx: [P,K] int64, precomputed from frozen features by the
        caller -- used only to gather, never modified. Returns row-
        normalised edge weights [P,K], differentiable w.r.t. this module's
        parameters (and w.r.t. f, though f carries no grad in practice).
        `return_gate=True` additionally returns the raw `2*sigmoid(e_ij)`
        [P,K] tensor (pre row-normalisation) for diagnostics (G6 logs its
        mean/std across edges)."""
        if f.ndim != 2 or f.shape[-1] != self.dim:
            raise ValueError(f"f must be [P,{self.dim}]")
        if cand_idx.ndim != 2 or cand_idx.shape[0] != f.shape[0]:
            raise ValueError("cand_idx must be [P,K] matching f's patch count")

        patches, K = cand_idx.shape
        f_i = f.unsqueeze(1).expand(patches, K, self.dim)   # [P,K,D], view (no copy) until concatenated
        f_j = f[cand_idx]                                    # [P,K,D], gather -- cand_idx is a plain index tensor, not itself differentiable
        elementwise = f_i * f_j                               # [P,K,D]
        cos_ij = elementwise.sum(dim=-1, keepdim=True)        # [P,K,1] -- f is L2-normalised, so dot product IS cosine; reused (not recomputed) below

        edge_input = torch.cat([f_i, f_j, elementwise, cos_ij], dim=-1)  # [P,K,3*dim+1]
        e = self.mlp(edge_input).squeeze(-1)                   # [P,K]
        gate = 2.0 * torch.sigmoid(e)                          # [P,K], == 1.0 identically at init

        base = cos_ij.squeeze(-1).clamp_min(0).pow(self.kappa)  # [P,K]
        w = base * gate                                        # [P,K]

        # symmetrize(w): deliberate no-op -- see module docstring.
        row_sums = w.sum(dim=-1, keepdim=True)
        eps = torch.finfo(w.dtype).tiny
        weights = w / row_sums.clamp_min(eps)

        # Zero-row fallback, matching build_knn_graph's own convention
        # exactly: a patch with no positive-affinity candidate at all
        # (measure-zero for real continuous features) points at itself
        # with unit weight. cand_idx is unchanged (never remapped) -- only
        # the WEIGHT row is overridden, matching the unary design's
        # zero_rows handling in build_differentiable_knn_graph.
        zero_rows = (row_sums.squeeze(-1) == 0).detach()
        if zero_rows.any():
            weights = weights.clone()
            weights[zero_rows] = 0
            weights[zero_rows, 0] = 1

        if return_gate:
            return weights, gate
        return weights


def build_frozen_candidate_set(f: torch.Tensor, *, K: int = 32) -> torch.Tensor:
    """Computes cand_idx [P,K] int64 ONCE from frozen features f, matching
    build_knn_graph's own top-k selection exactly (cosine -> ReLU -> stable
    descending argsort -> top-K), so that at K=12 the candidate set EdgeGate
    receives is identical to the existing production graph's. This is the
    ONLY place a non-differentiable top-k selection happens anywhere in the
    G design -- it runs on frozen features, outside any gradient path, and
    its output is a plain index tensor with no grad_fn (torch.argsort never
    produces one), so no explicit .detach() is needed here either."""
    if f.ndim != 2:
        raise ValueError("f must be [P,dim]")
    patches = f.shape[0]
    if not 0 < K < patches:
        raise ValueError("K must satisfy 0 < K < patch_count")
    with torch.no_grad():
        cosine = f @ f.T
        affinity = cosine.clamp_min(0)
        affinity = affinity.clone()
        affinity.fill_diagonal_(float("-inf"))
        order = torch.argsort(affinity, dim=-1, descending=True, stable=True)
        cand_idx = order[:, :K].contiguous()
    return cand_idx
