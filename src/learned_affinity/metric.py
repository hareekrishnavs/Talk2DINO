"""Learned affinity metric g(f) = normalize(f + r*MLP(f)) and the
differentiable kNN graph construction from g(f) (F1). See
src.learned_affinity.implicit_solve for the propagation solve -- this
module only builds the module and the graph, it does not propagate scores
and contains no loss, training loop, or data pipeline (out of F1 scope)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# F1a discovery: build_knn_graph (src/e3_affinity_oracle.py:232-283) is
# directed and row-stochastic with NO symmetrisation (GRAPH_DIRECTIONALITY
# = "directed_row_stochastic_knn"; grepping the whole repo for "symmetri"
# returns nothing). Matching it exactly means adding no A+A^T/max/mean step
# here either.


class LearnedMetric(nn.Module):
    """g(f) = normalize(f + r * MLP(f)), r a plain scalar clamped to [0, r_max].

    Initialisation (F1, post-mortem fix -- see RUN_PartF2fix.md): the MLP's
    final Linear layer is zero-initialised (weight AND bias), so MLP(f) == 0
    exactly for any f -- an untrained module reproduces g(f) = f bitwise,
    regardless of r's value. `r` itself is a plain nn.Parameter initialised
    to 0.1 -- NOT gated through a sigmoid near zero.

    This matters because a gated residual must not start BOTH branches at
    zero. The previous parametrisation (r = r_max * sigmoid(gate), gate
    init -10.0) put r at ~2.3e-5 IN ADDITION to the zero final layer -- two
    independent near-zero factors multiplying together, a deadlock:
    dL/dr is proportional to MLP(f) (=0, from the zero final layer) and
    dL/d(final layer) is proportional to r (~2.3e-5, vanishing), so nothing
    in the MLP received a usable gradient at step 0, and what little did
    reach the gate was additionally damped by sigmoid'(-10) ~= 4.5e-5.
    Measured directly: all of the first ten real training steps showed the
    first MLP layer's and the gate's gradients at exactly 0.0, and a
    2400-step pilot left r at 2.3e-5 (unchanged to two significant figures)
    with full-val mIoU completely flat across every checkpoint.

    With r a plain parameter at 0.1, only ONE branch (the MLP's output)
    starts at zero; the scale does not. dL/d(final layer) is now
    proportional to 0.1 rather than 2.3e-5 (~4300x larger), so the MLP
    moves from the first step, and dL/dr becomes non-zero as soon as
    MLP(f) does. Exact identity at init is unaffected: MLP(f) = 0 still
    makes g(f) = normalize(f + 0.1*0) = f exactly, regardless of r's value
    -- the identity guarantee for a TRAINED module still comes only from
    forcing r to exactly 0 via `r_override` (F1f/--assert-identity),
    unchanged by this fix.
    """

    def __init__(self, dim: int = 768, hidden: int = 256, r_max: float = 0.5,
                 kappa: float = 3.0, k: int = 12):
        super().__init__()
        if dim <= 0 or hidden <= 0:
            raise ValueError("dim and hidden must be positive")
        if not 0 < k:
            raise ValueError("k must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        self.dim = dim
        self.r_max = r_max
        self.kappa = kappa
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.r = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, f: torch.Tensor, *, r_override: float | None = None, bypass_mlp_at_zero: bool = True,
    ) -> torch.Tensor:
        """f: [...,dim], already L2-normalised. r_override, when given,
        REPLACES the learned r with a literal python float -- used by the
        identity gate (F1f) to force r to exactly 0 regardless of the
        trained r parameter, without needing to touch it.

        F1f requires g(f)==f and the REBUILT GRAPH to match the graph built
        directly from f EXACTLY (not just to float32 eps) -- computing
        `f + 0*residual` and then re-normalising an already-unit vector is
        NOT bitwise identical to f (re-summing squares in float32 perturbs
        the last ~3 bits, ~1e-8 absolute here), which would make an argsort
        tie-break or a borderline fp16 weight rounding differ from
        build_knn_graph(f)'s own output on real data. r_override=0.0 (or a
        trained r that happens to equal exactly 0.0) therefore short-
        circuits and returns f itself, unmodified -- exact by construction
        rather than by floating-point luck.

        `bypass_mlp_at_zero=False` disables that short circuit even when the
        EFFECTIVE r is 0.0, forcing the full `f + r*mlp(f)` computation
        through the MLP -- needed to gradient-check d(loss)/d(MLP params) AT
        r=0 (the short circuit returns `f` directly, which is disconnected
        from the MLP in the autograd graph, so a gradient check through it
        would trivially see zero gradient to every MLP parameter regardless
        of whether the analytic formula is actually correct there)."""
        if r_override == 0.0 and bypass_mlp_at_zero:
            return f
        residual = self.mlp(f)
        # torch.clamp preserves gradient inside [0, r_max] (subgradient 0
        # outside it) -- deliberately not detached/hard-thresholded, so r
        # itself remains trainable through this clamp (F1).
        r_value = torch.clamp(self.r, 0.0, self.r_max) if r_override is None else r_override
        g = f + r_value * residual
        return F.normalize(g, dim=-1)


def build_differentiable_knn_graph(
    g: torch.Tensor, *, k: int = 12, kappa: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable counterpart of build_knn_graph (F1c). g: [P,D],
    already L2-normalised (as LearnedMetric.forward guarantees). Returns
    (indices [P,K] int64 detached, weights [P,K] differentiable w.r.t. g).

    F1c's required pattern, applied literally: `affinity` keeps gradient
    tracking; a DETACHED copy is used only to pick which k indices to keep
    (topk/argsort has no useful gradient anyway -- it's a selection, not a
    smooth function); the returned `weights` are then GATHERED from the
    still-differentiable `affinity` tensor at those (detached) indices, so
    gradient flows through the edge WEIGHTS but never through which edges
    were selected. See the two `.detach()` calls below.
    """
    if g.ndim != 2:
        raise ValueError("g must be [P,D]")
    patches = g.shape[0]
    if not 0 < k < patches:
        raise ValueError("k must satisfy 0 < k < patch_count")

    cosine = g @ g.T                                   # differentiable [P,P]
    affinity = cosine.clamp_min(0).pow(kappa)           # differentiable [P,P]

    selection_affinity = affinity.detach()              # <-- DETACH (selection only)
    selection_affinity = selection_affinity.clone()
    selection_affinity.fill_diagonal_(float("-inf"))
    order = torch.argsort(selection_affinity, dim=-1, descending=True, stable=True)
    indices = order[:, :k].detach()                     # <-- DETACH (no grad through indices)

    selected = affinity.gather(1, indices)               # <-- GATHER from the LIVE tensor: differentiable
    row_sums = selected.sum(dim=-1, keepdim=True)
    eps = torch.finfo(selected.dtype).tiny
    weights = selected / row_sums.clamp_min(eps)

    # Zero-row fallback (a patch with no positive-affinity neighbour at
    # all -- measure-zero for real continuous features, but handled for
    # robustness, matching build_knn_graph's own fallback exactly: point at
    # self with unit weight). This assignment is necessarily a hard
    # constant, not a function of g, so it locally has no gradient -- an
    # unavoidable consequence of the fallback being a discrete decision,
    # exactly as build_knn_graph's own (non-differentiable) fallback is.
    zero_rows = (row_sums.squeeze(-1) == 0).detach()
    if zero_rows.any():
        self_index = torch.arange(patches, device=g.device)[zero_rows]
        indices = indices.clone()
        indices[zero_rows] = self_index[:, None]
        weights = weights.clone()
        weights[zero_rows] = 0
        weights[zero_rows, 0] = 1

    return indices, weights
