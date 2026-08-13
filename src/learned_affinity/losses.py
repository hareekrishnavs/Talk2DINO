"""L1-L4: the training objective. L1 (masked reconstruction) is primary;
L2/L3/L4 are regularisers at small (or zero, for L4) weight. Operates on
S_0 [C,P] (P=1024 patches) and a differentiable kNN graph (indices,weights)
already built from g(f) -- this module never touches f or g directly, only
the graph and the raw scores, keeping it decoupled from LearnedMetric/
build_differentiable_knn_graph.

L1 and L2 each require their OWN implicit solve (L1 from a MASKED S_0, L2
from the FULL S_0 -- these are different inputs to the same linear system,
not reusable from one another) but SHARE the same graph, so
build_differentiable_knn_graph is called once per step by the caller, not
here."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .implicit_solve import apply_knn, implicit_propagate

DEFAULT_L2_WEIGHT = 0.1
DEFAULT_L3_WEIGHT = 0.05
DEFAULT_L4_WEIGHT = 0.0  # logged unconditionally regardless of weight


@dataclass
class LossWeights:
    l2: float = DEFAULT_L2_WEIGHT
    l3: float = DEFAULT_L3_WEIGHT
    l4: float = DEFAULT_L4_WEIGHT
    mask_fraction: float = 0.3
    tau: float = 0.1
    l2_margin: float = 0.2
    l4_entropy_floor: float = 0.0


def sample_patch_mask(num_patches: int, fraction: float, *, device, generator=None) -> torch.Tensor:
    """Boolean [P] mask, True at masked patches. `generator` (a
    torch.Generator) makes this reproducible in tests; omit for real
    per-step randomness."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0,1)")
    n_masked = max(1, int(round(num_patches * fraction)))
    perm = torch.randperm(num_patches, device=device, generator=generator)
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[perm[:n_masked]] = True
    return mask


def masked_reconstruction_loss(
    raw_scores: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, alpha: float, *,
    mask_fraction: float, tau: float, generator=None,
) -> dict[str, torch.Tensor]:
    """L1. raw_scores S_0: [C,P]. Zeros S_0 at a sampled patch mask, solves
    the fixed point from that masked S_0, and scores a soft-cross-entropy
    between softmax(S*_masked/tau) and softmax(S_0/tau) -- the UNDIFFUSED
    local evidence -- at the masked patches (the training signal) and,
    separately for logging only, at the unmasked patches (expected to be
    near-trivial, since those positions keep their true S_0 in the
    propagation's own (1-alpha)*S0 term)."""
    channels, patches = raw_scores.shape
    mask = sample_patch_mask(patches, mask_fraction, device=raw_scores.device, generator=generator)

    s0_masked = raw_scores.clone()
    s0_masked[:, mask] = 0.0

    s0_pc = s0_masked.T.contiguous()  # [P,C]
    s_star = implicit_propagate(s0_pc, indices, weights, alpha).T  # [C,P]

    target = F.softmax(raw_scores / tau, dim=0)  # [C,P], soft target from the UNDIFFUSED S_0
    pred_log_probs = F.log_softmax(s_star / tau, dim=0)
    per_patch_ce = -(target * pred_log_probs).sum(dim=0)  # [P]

    masked_loss = per_patch_ce[mask].mean()
    unmasked_loss = per_patch_ce[~mask].mean() if (~mask).any() else torch.zeros((), device=raw_scores.device)

    return {
        "loss": masked_loss,
        "masked_ce": masked_loss.detach(),
        "unmasked_ce": unmasked_loss.detach(),
        "mask": mask,
    }


def caption_noun_ranking_loss(
    raw_scores: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, alpha: float,
    is_present: list[bool], *, margin: float,
) -> torch.Tensor:
    """L2. raw_scores S_0: [C,P], FULL (unmasked). Propagates to the fixed
    point, max-pools over patches per class, and requires every present
    noun's score to outrank every distractor's score by `margin` (hinge /
    margin-ranking loss). ANTI-COLLAPSE REGULARISER ONLY -- see the module
    docstring in losses.py and the caller for the caution about
    caption-contrastive objectives previously anti-correlating with mIoU
    (Pearson -0.727) in this project; keep the caller's weight on this low."""
    if not any(is_present) or all(is_present):
        raise ValueError("caption_noun_ranking_loss needs both present and distractor classes")
    s0_pc = raw_scores.T.contiguous()  # [P,C]
    s_star = implicit_propagate(s0_pc, indices, weights, alpha).T  # [C,P]
    scores = s_star.max(dim=1).values  # [C]

    present_mask = torch.tensor(is_present, dtype=torch.bool, device=scores.device)
    present_scores = scores[present_mask]  # [n_present]
    distractor_scores = scores[~present_mask]  # [n_distractor]
    # all-pairs hinge: relu(margin - (present - distractor))
    diff = present_scores[:, None] - distractor_scores[None, :]
    return F.relu(margin - diff).mean()


def anchor_loss(g: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """L3. 1 - cos(g(f), f), averaged over patches. Both already
    L2-normalised, so cosine similarity is just the dot product."""
    return (1.0 - (g * f).sum(dim=-1)).mean()


def mean_row_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Mean entropy of A's rows (weights: [P,K], already row-stochastic
    from build_differentiable_knn_graph). Logged every step regardless of
    L4's weight (T3/L4's own instruction)."""
    eps = torch.finfo(weights.dtype).tiny
    row_entropy = -(weights * (weights.clamp_min(eps)).log()).sum(dim=-1)  # [P]
    return row_entropy.mean()


def row_entropy_floor_loss(weights: torch.Tensor, floor: float) -> torch.Tensor:
    """L4. One-sided: penalises ONLY mean row entropy falling below `floor`."""
    return F.relu(floor - mean_row_entropy(weights))


def compute_total_loss(
    *, f: torch.Tensor, g: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
    raw_scores: torch.Tensor, is_present: list[bool], alpha: float,
    loss_weights: LossWeights = LossWeights(), generator=None,
) -> dict[str, torch.Tensor]:
    """f, g: [P,D] (anchor). indices, weights: [P,K] (the differentiable
    graph built from g). raw_scores: [C,P] (S_0, FULL/unmasked -- L1 does
    its own masking internally). is_present: length-C list, True for
    caption-present nouns, False for distractors (D4's class-list order).

    Returns a dict with "total" (the backprop target) plus every component,
    each detached except where noted, for per-step logging (T3)."""
    num_classes = raw_scores.shape[0]

    l1 = masked_reconstruction_loss(
        raw_scores, indices, weights, alpha,
        mask_fraction=loss_weights.mask_fraction, tau=loss_weights.tau, generator=generator,
    )
    l2 = caption_noun_ranking_loss(
        raw_scores, indices, weights, alpha, is_present, margin=loss_weights.l2_margin,
    )
    l3 = anchor_loss(g, f)
    row_entropy = mean_row_entropy(weights)
    l4 = row_entropy_floor_loss(weights, loss_weights.l4_entropy_floor)

    total = l1["loss"] + loss_weights.l2 * l2 + loss_weights.l3 * l3 + loss_weights.l4 * l4

    # F4: random guessing over C text queries gives ln(C) -- a reconstruction
    # loss at or above this line is no better than chance. Logged alongside
    # the raw value (not backpropagated -- purely a reference line).
    chance_ce = math.log(num_classes)

    return {
        "total": total,
        "num_classes": num_classes,
        "chance_ce": chance_ce,
        "l1_masked_ce": l1["masked_ce"],
        "l1_masked_ce_gap_vs_chance": l1["masked_ce"] - chance_ce,
        "l1_unmasked_ce": l1["unmasked_ce"],
        "l2_ranking": l2.detach(),
        "l3_anchor": l3.detach(),
        "l4_entropy_floor": l4.detach(),
        "mean_row_entropy": row_entropy.detach(),
        "mask": l1["mask"],
    }
