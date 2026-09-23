"""Isolated, GT-free T0/T320 target generator, reproducing exactly the
source semantics frozen by the committed T320 target-source authority.

Authority: research_specs/semantic_evidence_t320_source_v1.toml,
  schema semantic_evidence_t320_source_authority, version 1.0.0.
Historical source commit: 00392527937253f93b9f3fe36b76347b4a6f1eb0
  (branch e12-connectivity-analysis).
Manifest: research_specs/semantic_evidence_t320_source_manifest_v1.json.

This module contains NO reference to ground truth, annotations, correctness
labels, or intervention outputs anywhere. It never imports the historical
branch, mmcv, a dataset provider, or a model builder. It never checks out
another branch, loads a checkpoint tensor, or runs real model inference --
it operates purely on already-validated `WindowSnapshot` objects (see
`semantic_evidence_e3_adapter.py`).

T320 execution on real data, the 96-image feasibility study, interventions,
and training all remain unauthorized by this module's existence -- see
`docs/semantic_evidence_t320_implementation.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from semantic_evidence_e3_adapter import WindowSnapshot


class T320Error(Exception):
    """Raised for any structural, shape, dtype, or scope violation in the
    T320 target generator."""


_FATAL = (KeyboardInterrupt, SystemExit, MemoryError)


def _reraise_if_fatal(exc: BaseException) -> None:
    if isinstance(exc, _FATAL):
        raise exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise T320Error(message)


def _require_finite(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise T320Error(f"{label} contains non-finite values")


# ---------------------------------------------------------------------------
# Frozen scientific constants -- source_manifest ordinals 1 (graph.py) and
# 2 (finite_step_regime.py); [graph_contract]/[recurrence_contract] in the
# authority TOML.
# ---------------------------------------------------------------------------

K = 12
AFFINITY_POWER = 3.0
ALPHA = 0.98
STEPS = 320
GOVERNED_DTYPE = torch.float32


def _feature_norm_tolerance(dtype: torch.dtype) -> float:
    """Matches graph.py's own dtype-dependent L2-norm tolerance exactly
    (larger tolerance for lower-precision dtypes)."""
    if dtype == torch.float16:
        return 5e-3
    if dtype == torch.bfloat16:
        return 1e-2
    return 1e-5


# ---------------------------------------------------------------------------
# Graph construction
# authority: [graph_contract], required_symbols DirectedTopKGraph /
# build_directed_topk_graph, source_manifest ordinal 1 (cover_dr/graph.py).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DirectedTopKGraph:
    """Sparse, row-stochastic, directed top-k graph. Not symmetrized (per
    [graph_contract].directed_symmetric_behavior). `neighbor_indices[i, s]
    == j` means the directed edge i -> j with weight
    `transition_weights[i, s]`."""

    neighbor_indices: torch.Tensor  # [N, k] int64
    transition_weights: torch.Tensor  # [N, k] float, rows sum to 1
    num_nodes: int
    k: int

    def matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        """Compute A @ rhs via the sparse row-stochastic representation,
        matching [graph_contract].matmul_orientation exactly: row i of the
        output is the weighted sum of rhs at row i's k selected neighbors."""
        _require(isinstance(rhs, torch.Tensor), "matmul rhs must be a torch.Tensor")
        _require(rhs.shape[0] == self.num_nodes, "matmul rhs node count mismatch")
        _require(rhs.device == self.transition_weights.device, "matmul rhs device mismatch")
        gathered = rhs[self.neighbor_indices]  # [N, k, C]
        weights = self.transition_weights.to(dtype=rhs.dtype)
        return (weights.unsqueeze(-1) * gathered).sum(dim=1)


def _build_directed_topk_graph_with_params(
    dino_features: torch.Tensor, *, k: int = K, affinity_power: float = AFFINITY_POWER
) -> DirectedTopKGraph:
    """Private primitive: builds the directed top-k affinity graph with
    caller-supplied k/affinity_power. Accepts these ONLY for isolated
    testing (e.g. small hand-computed graphs at k < 12) -- the governed
    public `build_t320_graph` below always calls this with the frozen
    K=12, AFFINITY_POWER=3.0 and never forwards a caller-supplied value.

    Authority: [graph_contract] in research_specs/semantic_evidence_t320_source_v1.toml.
    Historical source commit 00392527937253f93b9f3fe36b76347b4a6f1eb0,
    source_manifest ordinal 1, required_symbol build_directed_topk_graph
    (src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py).

    Formula: affinities = clamp_min(features @ features.T, 0) ** affinity_power,
    self excluded via -inf diagonal, top-k selected by a STABLE descending
    argsort (lowest original column index wins exact ties -- never `topk`),
    row-normalized (each nonzero row sums to 1); a row whose k selected
    affinities sum to exactly zero falls back to a self-loop of weight 1.
    Directed; never symmetrized.
    """
    _require(isinstance(dino_features, torch.Tensor), "dino_features must be a torch.Tensor")
    _require(dino_features.ndim == 2, f"dino_features must be [N, D], got {tuple(dino_features.shape)}")
    _require(dino_features.is_floating_point(), "dino_features must be floating point")
    num_nodes, _embed_dim = dino_features.shape
    _require(num_nodes >= 2, "dino_features must contain at least two nodes")
    _require(isinstance(k, int) and not isinstance(k, bool) and 0 < k < num_nodes,
              f"k must satisfy 0 < k < N, got k={k}, N={num_nodes}")
    _require(isinstance(affinity_power, (int, float)) and not isinstance(affinity_power, bool) and affinity_power > 0,
              f"affinity_power must be finite and positive, got {affinity_power!r}")
    _require_finite(dino_features, "dino_features")

    features_float = dino_features.detach().to(GOVERNED_DTYPE)
    norms = features_float.norm(dim=-1)
    _require(bool((norms > 0).all()), "dino_features must have non-zero L2 norm")
    tolerance = _feature_norm_tolerance(dino_features.dtype)
    _require(bool(torch.allclose(norms, torch.ones_like(norms), rtol=tolerance, atol=tolerance)),
              f"dino_features must already be L2-normalized (max norm error {(norms - 1).abs().max().item():.6g})")

    affinities = (features_float @ features_float.transpose(0, 1)).clamp_min(0.0)
    affinities = affinities.pow(float(affinity_power))
    affinities.fill_diagonal_(float("-inf"))

    neighbor_indices = torch.argsort(affinities, dim=1, descending=True, stable=True)[:, :k]
    edge_affinities = affinities.gather(1, neighbor_indices)
    row_sums = edge_affinities.sum(dim=1)
    self_loop_fallback = row_sums == 0

    transition_weights = torch.zeros_like(edge_affinities)
    positive_rows = ~self_loop_fallback
    transition_weights[positive_rows] = edge_affinities[positive_rows] / row_sums[positive_rows, None]
    if bool(self_loop_fallback.any()):
        fallback_rows = torch.arange(num_nodes, device=dino_features.device)[self_loop_fallback]
        neighbor_indices[self_loop_fallback, 0] = fallback_rows
        transition_weights[self_loop_fallback, 0] = 1.0

    _require_finite(transition_weights, "transition_weights")
    return DirectedTopKGraph(
        neighbor_indices=neighbor_indices.detach().clone().contiguous(),
        transition_weights=transition_weights.detach().clone().contiguous(),
        num_nodes=num_nodes,
        k=k,
    )


def build_t320_graph(dino_features: torch.Tensor) -> DirectedTopKGraph:
    """Governed public entry point: builds the frozen directed top-k
    affinity graph with K=12, AFFINITY_POWER=3.0 pinned internally. Accepts
    only the scientific input tensor -- no k or affinity_power parameter
    exists on this function; a caller cannot override either value.

    See `_build_directed_topk_graph_with_params` for the full formula
    documentation and authority citation.
    """
    return _build_directed_topk_graph_with_params(dino_features, k=K, affinity_power=AFFINITY_POWER)


# ---------------------------------------------------------------------------
# Finite-step recurrence
# authority: [recurrence_contract], required_symbol finite_step_propagate,
# source_manifest ordinal 2 (cover_dr/finite_step_regime.py).
# ---------------------------------------------------------------------------

def _finite_step_propagate(graph: DirectedTopKGraph, s0: torch.Tensor, *, alpha: float, steps: int) -> torch.Tensor:
    """Private primitive: P(0)=S0; P(t+1) = alpha*(A@P(t)) + (1-alpha)*S0,
    for exactly `steps` completed updates. Accepts alpha/steps as parameters
    ONLY for oracle/unit testing at small step counts -- the public governed
    path (`propagate_t320`) always calls this with alpha=0.98, steps=320 and
    never exposes these as caller-settable.

    Authority: [recurrence_contract]. No convergence test, no tolerance, no
    early stop, no adaptive step count (matches finite_step_propagate's own
    docstring: 'there is no convergence test, tolerance, early stop, ...').
    `s0` is never mutated: a defensive clone is taken once.
    """
    _require(isinstance(graph, DirectedTopKGraph), "graph must be a DirectedTopKGraph")
    _require(isinstance(s0, torch.Tensor) and s0.ndim == 2, f"s0 must be [N, C], got {tuple(getattr(s0, 'shape', ()))}")
    _require(s0.shape[0] == graph.num_nodes, "s0 node count does not match graph.num_nodes")
    _require(s0.is_floating_point(), "s0 must be floating point")
    _require_finite(s0, "s0")
    _require(isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and 0 <= alpha < 1,
              f"alpha must satisfy 0 <= alpha < 1, got {alpha!r}")
    _require(isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1, f"steps must be a positive int, got {steps!r}")

    s0_owned = s0.detach().clone()
    p = s0_owned.clone()
    one_minus_alpha = 1.0 - float(alpha)
    for _step in range(1, steps + 1):
        p = float(alpha) * graph.matmul(p) + one_minus_alpha * s0_owned

    _require_finite(p, "finite-step propagation result")
    return p


def propagate_t320(graph: DirectedTopKGraph, s0: torch.Tensor) -> torch.Tensor:
    """Governed public entry point: exactly 320 updates at alpha=0.98,
    float32. Does not accept alternative alpha/steps -- see module docstring
    on the non-overridable governed path."""
    s0_governed = s0.to(GOVERNED_DTYPE) if s0.dtype != GOVERNED_DTYPE else s0
    return _finite_step_propagate(graph, s0_governed, alpha=ALPHA, steps=STEPS)


# ---------------------------------------------------------------------------
# Internal per-window upsample (Operation A) vs. final rescale (Operation B)
# authority: [internal_upsample_contract] / [final_rescale_contract].
#
# Score-domain order (T0 and T320 both, confirmed from historical source):
#   raw pre-sigmoid S0 (T0) / raw post-320-step recurrence output (T320)
#     -> exactly one torch.sigmoid, per window (this function)
#     -> bilinear internal upsample, align_corners=True (this function)
#     -> overlap averaging across windows, in sigmoid-domain (stitch_windows)
#     -> bilinear final rescale, align_corners=False (finalize_prediction)
#     -> argmax (finalize_prediction)
# Sigmoid never occurs anywhere else: not before recurrence, not a second
# time before stitching, not after stitching, not after final rescale.
# ---------------------------------------------------------------------------

INTERNAL_UPSAMPLE_ALIGN_CORNERS = True
FINAL_RESCALE_ALIGN_CORNERS = False


def internal_upsample(patch_scores: torch.Tensor, grid_hw: tuple, output_hw: tuple) -> torch.Tensor:
    """Operation A: per-window patch-grid RAW scores -> sigmoid-domain,
    crop-pixel-upsampled mask.

    Authority: [internal_upsample_contract], source_function
    DINOText.masks_from_patch_scores, source_manifest ordinal 5. Historical
    source commit 00392527937253f93b9f3fe36b76347b4a6f1eb0
    (src/open_vocabulary_segmentation/models/dinotext/dinotext.py:437-455):

        mask = self.masker.mask_from_similarity(simmap, deterministic=True, hard=False)
        return F.interpolate(mask, output_hw, mode="bilinear", align_corners=True)

    `mask_from_similarity` (masker.py) resolves to `DINOTextSim2Mask.forward`'s
    unconditional soft-mask branch, `soft_mask = torch.sigmoid(x)` -- applied
    to every call regardless of `deterministic`, since `hard=False` always
    selects `soft_mask`. This function therefore applies exactly one
    `torch.sigmoid` to `patch_scores` (never mutating it in place -- a new
    tensor is produced by `.permute().reshape()` before the sigmoid), THEN
    bilinear-interpolates with `align_corners=True` (hardcoded, never
    caller-settable). `patch_scores` is [N, C] in canonical row-major node
    order; reshaped to [1, C, grid_h, grid_w] before the sigmoid+interpolation.

    Callers pass RAW (pre-sigmoid) scores here: S0 itself for T0, or the
    recurrence output after all 320 updates for T320 -- confirmed from
    source, `process_one_window` calls `masks_from_patch_scores` on
    `trace.snapshots[steps]` (the post-recurrence snapshot), never on a
    pre-sigmoid'd intermediate. The returned sigmoid-domain mask is exactly
    what the historical `stitch_one_image` accumulates
    (`_pad_into(preds, result.k11_masks/k12_masks, window)`, where those
    masks are `masks_from_patch_scores`'s own return value) -- so window
    overlap averaging (`stitch_windows`) must receive THIS function's
    output, never the raw `patch_scores` directly. Window-wise sigmoid
    must occur before overlap averaging: `mean(sigmoid(x)) != sigmoid(mean(x))`
    in general, so deferring sigmoid to after stitching would not reproduce
    the source-authorized behavior.
    """
    _require(isinstance(patch_scores, torch.Tensor) and patch_scores.ndim == 2, "patch_scores must be [N, C]")
    grid_h, grid_w = grid_hw
    _require(patch_scores.shape[0] == grid_h * grid_w, "patch_scores node count must equal grid_h*grid_w")
    _require_finite(patch_scores, "patch_scores")
    num_classes = patch_scores.shape[1]
    grid = patch_scores.permute(1, 0).reshape(1, num_classes, grid_h, grid_w)
    mask = torch.sigmoid(grid)
    upsampled = F.interpolate(mask, size=tuple(output_hw), mode="bilinear", align_corners=INTERNAL_UPSAMPLE_ALIGN_CORNERS)
    _require_finite(upsampled, "internal_upsample output")
    return upsampled


# ---------------------------------------------------------------------------
# Window stitching
# authority: [stitching_contract], required_symbol stitch_one_image,
# source_manifest ordinal 3.
# ---------------------------------------------------------------------------

def stitch_windows(window_scores: Sequence[torch.Tensor], window_boxes: Sequence[tuple], canvas_h: int, canvas_w: int) -> torch.Tensor:
    """Stitch per-window, already-sigmoid-transformed, upsampled mask
    tensors onto one canvas (i.e. `window_scores` entries are expected to
    be `internal_upsample`'s output, not raw pre-sigmoid scores -- matching
    the historical `stitch_one_image`, which accumulates
    `masks_from_patch_scores`'s sigmoid-domain return value directly).

    Authority: [stitching_contract]: uniform averaging only (no weighting
    beyond overlap count), fails closed on any uncovered pixel. `window_boxes`
    entries are (top, left, bottom, right) in canvas coordinates, in the same
    canonical order as `window_scores`. This function is domain-agnostic
    arithmetic (sum then divide by count); it is the caller's responsibility
    to have already applied the per-window sigmoid via `internal_upsample`
    before stitching.
    """
    _require(len(window_scores) == len(window_boxes), "window_scores and window_boxes length mismatch")
    _require(len(window_scores) >= 1, "at least one window is required")
    num_classes = window_scores[0].shape[1]
    dtype = window_scores[0].dtype
    device = window_scores[0].device
    accumulator = torch.zeros((1, num_classes, canvas_h, canvas_w), dtype=GOVERNED_DTYPE, device=device)
    count = torch.zeros((1, 1, canvas_h, canvas_w), dtype=GOVERNED_DTYPE, device=device)

    for scores, (top, left, bottom, right) in zip(window_scores, window_boxes):
        _require(scores.shape[1] == num_classes, "all windows must share the same class count")
        _require(scores.dtype == dtype and scores.device == device, "all windows must share dtype/device")
        _require_finite(scores, "window scores")
        h, w = bottom - top, right - left
        _require(scores.shape[2:] == (h, w), f"window score shape {tuple(scores.shape[2:])} does not match box ({h}, {w})")
        accumulator[:, :, top:bottom, left:right] += scores.to(GOVERNED_DTYPE)
        count[:, :, top:bottom, left:right] += 1.0

    if bool((count == 0).any()):
        raise T320Error("stitching left uncovered pixels")
    stitched = accumulator / count
    _require_finite(stitched, "stitched canvas")
    return stitched


# ---------------------------------------------------------------------------
# Final rescale + prediction extraction
# authority: [final_rescale_contract] / [prediction_contract], required
# symbol finalize_prediction, source_manifest ordinal 3.
# ---------------------------------------------------------------------------

def finalize_prediction(stitched: torch.Tensor, ori_hw: tuple) -> torch.Tensor:
    """Operation B (final rescale) + prediction extraction, exactly as
    matched_power_evaluator.finalize_prediction: crop to `stitched`'s own
    spatial extent (a no-op here since callers pass an exact-size canvas),
    bilinear rescale to `ori_hw` with align_corners=False (hardcoded,
    matching DINOTextSegInference.align_corners=False -- the sole such
    assignment in the historical source tree), then argmax(dim=1) (plain
    torch.argmax; tie-break is PyTorch's own, not independently guaranteed).
    """
    _require(isinstance(stitched, torch.Tensor) and stitched.ndim == 4, "stitched must be [1, C, H, W]")
    _require_finite(stitched, "stitched")
    rescaled = F.interpolate(stitched, size=tuple(ori_hw), mode="bilinear", align_corners=FINAL_RESCALE_ALIGN_CORNERS)
    _require_finite(rescaled, "rescaled prediction scores")
    return rescaled.argmax(dim=1)


# ---------------------------------------------------------------------------
# T0 generation -- authority: [t0_contract]. Bound to the committed E3
# feasibility contract; not redefined here. Original view only: no graph,
# no recurrence, same geometry/upsample/stitching/rescale pipeline as T320,
# applied to S0 directly (t=0, before any propagation).
# ---------------------------------------------------------------------------

def generate_t0_t320_targets(
    snapshots: Sequence[WindowSnapshot],
    window_boxes: Sequence[tuple],
    *,
    canvas_h: int,
    canvas_w: int,
    ori_h: int,
    ori_w: int,
) -> dict:
    """Top-level, GT-free T0/T320 target generator.

    Authority: research_specs/semantic_evidence_t320_source_v1.toml (schema
    semantic_evidence_t320_source_authority v1.0.0), historical source
    commit 00392527937253f93b9f3fe36b76347b4a6f1eb0.

    Accepts only validated `WindowSnapshot` objects and target geometry.
    Never accepts GT, annotations, alternative k/alpha/steps/affinity_power/
    align_corners, a debug subset, class remapping, or candidate-specific
    recurrence -- every scientific constant is pinned internally (see K,
    AFFINITY_POWER, ALPHA, STEPS, INTERNAL_UPSAMPLE_ALIGN_CORNERS,
    FINAL_RESCALE_ALIGN_CORNERS module constants).

    For N windows: exactly N graph constructions, N recurrence trajectories
    (N x STEPS updates total), zero candidate-specific recurrence, zero
    component/masked-forward calls. The caller is responsible for having
    produced each WindowSnapshot via exactly one E3 forward per window
    (see `semantic_evidence_e3_adapter.extract_e3_window_snapshot`); this
    function performs no model inference itself.

    Returns {"t0": LongTensor[1,ori_h,ori_w], "t320": LongTensor[1,ori_h,ori_w]}.
    """
    _require(len(snapshots) >= 1, "at least one window snapshot is required")
    _require(len(snapshots) == len(window_boxes), "snapshots and window_boxes length mismatch")

    seen_ids = set()
    for snap in snapshots:
        _require(isinstance(snap, WindowSnapshot), "every element of snapshots must be a WindowSnapshot")
        _require(snap.window_id not in seen_ids, f"duplicate window_id {snap.window_id!r}")
        seen_ids.add(snap.window_id)
        _require(snap.canvas_h == canvas_h and snap.canvas_w == canvas_w, "snapshot canvas size mismatch")

    t0_window_scores = []
    t320_window_scores = []
    for snap in snapshots:
        grid_hw = (snap.grid_h, snap.grid_w)
        output_hw = (snap.infer_h, snap.infer_w)

        # T0: original view, no graph, no recurrence -- S0 itself
        # sigmoid-transformed and upsampled (internal_upsample applies the
        # single sigmoid + bilinear interpolation, matching
        # masks_from_patch_scores exactly).
        t0_upsampled = internal_upsample(snap.s0, grid_hw, output_hw)
        t0_window_scores.append(t0_upsampled)

        # T320: one graph construction + one 320-step recurrence in raw
        # score space per window; sigmoid is applied only once, after all
        # 320 updates, by internal_upsample -- never before or during
        # recurrence.
        graph = build_t320_graph(snap.dino_features)
        propagated = propagate_t320(graph, snap.s0)
        t320_upsampled = internal_upsample(propagated, grid_hw, output_hw)
        t320_window_scores.append(t320_upsampled)

    t0_stitched = stitch_windows(t0_window_scores, window_boxes, canvas_h, canvas_w)
    t320_stitched = stitch_windows(t320_window_scores, window_boxes, canvas_h, canvas_w)

    t0_prediction = finalize_prediction(t0_stitched, (ori_h, ori_w))
    t320_prediction = finalize_prediction(t320_stitched, (ori_h, ori_w))

    return {"t0": t0_prediction, "t320": t320_prediction}
