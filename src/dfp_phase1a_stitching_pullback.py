"""DFP Phase 1A Step 7: the exact pullback (adjoint) of the canonical
Step 5 stitching operator, used to compute the soft source membership of
every window-patch copy in a Step 6 global unary component.

Authoritative forward operator A_w
-----------------------------------
Traced fresh from ``src.dfp_phase1a_horizon_stitching.
stitch_dfp_phase1a_horizons`` (never re-derived): for one window ``w``
and one class-agnostic scalar patch field ``u`` of shape ``[N]``
(``N = patch_grid[0] * patch_grid[1]``), the production code computes

    contribution[x] = interpolate_raw_scores(u, grid_hw, crop_hw)[local(x)]
                       * weight[local(x)]

for every pixel ``x`` in ``window.accumulation_slice`` (zero elsewhere),
accumulates it into a shared numerator, accumulates ``weight`` into a
shared denominator once per window (never once per class/horizon), and
finally divides: ``S(x) = numerator(x) / denominator(x)``. Composing the
per-window contribution with that division (the denominator does not
depend on ``u``'s *values*, only on window geometry) gives exactly

    A_w(u)(x) = interpolate_raw_scores(u, grid_hw, crop_hw)[local(x)]
                * weight[local(x)] / denominator(x)      for x in window w
              = 0                                         otherwise

``weight`` is DFP's frozen ``averaging="uniform"`` map (all ones); it is
carried through symbolically rather than dropped, so this remains
correct if a future step ever binds a different frozen weighting.

Why A_w is linear
-------------------
Every stage is linear in ``u``: ``interpolate_raw_scores`` is a
reshape + permute + ``F.interpolate(mode="bilinear", align_corners=True)``
-- a fixed convex combination of at most 4 input values per output pixel,
with weights that depend only on geometry, never on ``u``'s values;
elementwise multiplication by ``weight`` and division by ``denominator``
are diagonal (and therefore self-adjoint) linear maps, since neither
depends on ``u``; and scattering into ``window.accumulation_slice``
(zero elsewhere) is a fixed linear embedding. No sigmoid, softmax,
argmax, threshold, or calibration appears anywhere in this chain --
Step 5's own frozen ``raw_score_stage="pre_sigmoid_pre_upsample_pre_
stitch"`` guarantees this, and this module never adds one.

The pullback
------------
``A_w^T`` is computed via the exact vector-Jacobian product of
``interpolate_raw_scores`` itself (never re-derived by hand): for a
pixel-space vector ``v`` restricted to ``window.accumulation_slice``,
scaled by ``weight/denominator`` (the diagonal factor is self-adjoint,
so it is applied identically in both directions),

    A_w^T(v)[j] = d/du [ <interpolate_raw_scores(u, grid_hw, crop_hw),
                           weight/denominator * v> ]  at any u

is obtained with a single ``torch.autograd.grad`` call against a fresh,
disconnected zero probe tensor ``u`` -- never a model parameter, never
retaining a graph afterward, computing no parameter gradients. Because
``interpolate_raw_scores`` is genuinely linear, this is the *exact*
adjoint (autograd computes analytic gradients for linear operations, not
an approximation) of the literal function Step 5 calls, not of a
hand-derived approximation of it. Section 11's tests independently
verify this by constructing a small dense reference matrix from
repeated *forward* calls on patch-basis vectors -- never by reusing this
pullback for both sides of a comparison.

Soft membership
----------------
For a component ``Ω_r`` (from Step 6's ``component_ids``), window ``w``,
and patch node ``j``:

    M[r,w,j] = (A_w^T 1_{Ω_r})[j] / (A_w^T 1)[j]

``q[w,j] = (A_w^T 1)[j]`` is geometry-only (frozen identity:
``structural_components.a_transpose_ones_cached_invariant ==
"geometry_only"``) -- independent of horizon scores, class channels, and
component content -- so it is computed exactly once per image geometry
and reused for every component chunk.

No dense pixel-by-patch matrix
--------------------------------
This module never materializes an array of shape
``[image_pixels, window_patch_nodes]`` or any equivalent. Component
indicator fields are built one small chunk at a time (never all R
components' full-resolution masks simultaneously); the adjoint of
bilinear interpolation is obtained via autograd VJP over a tiny
``[1, N, chunk_size]`` probe, never a materialized interpolation matrix.

Excluded from this step
--------------------------
No masked recurrence, no component-propagated score field, no signed
evidence, no supercohort propagation, no k-star/phi/eta-net, no oracle
or Gate 1/2 execution, no final selector, no real-data driver, no
artifact/checkpoint schema, no SLURM/GPU code. **No real Phase 1A
dataset diagnostic has been run by this step.**
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from src.dfp_phase1a_global_unary_components import (
    DfpComponentContract,
    DfpPhase1aGlobalUnaryComponentsError,
    dfp_component_contract_from_protocol,
)
from src.dfp_phase1a_horizon_stitching import (
    DfpPhase1aHorizonStitchingError,
    DfpStitchingContract,
    SlidingWindowPlan,
    SpatialSize,
    build_stitch_weight,
    dfp_stitching_contract_from_protocol,
    interpolate_raw_scores,
)
from src.dfp_phase1a_identity import (
    DfpPhase1aIdentityError,
    validate_structural_components_block,
)


class DfpPhase1aStitchingPullbackError(ValueError):
    """Raised on any DFP Phase 1A stitching-pullback invariant
    violation. Always fail closed: never silently sort, reshape, coerce,
    clamp, or renormalize a malformed input or a failed conservation
    check."""


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1aStitchingPullbackError(f"{label} must be an exact non-boolean integer")
    return value


def _require_positive_exact_int(value: Any, label: str) -> int:
    value = _require_exact_int(value, label)
    if value <= 0:
        raise DfpPhase1aStitchingPullbackError(f"{label} must be positive")
    return value


# ---------------------------------------------------------------------------
# Identity-bound pullback contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DfpPullbackContract:
    """Independently validated pullback/geometry parameters extracted
    from the live protocol identity -- never hardcoded module
    constants. Delegates entirely to Step 5's and Step 6's own contract
    builders plus Step 6's ``validate_structural_components_block``
    (which already pins ``membership_dense_matrix_forbidden``,
    ``adjoint_identity_required``, ``membership_nonnegative_required``,
    ``membership_sums_to_one_per_window_patch_source_required``,
    ``flush_edge_windows_included``, and
    ``a_transpose_ones_cached_invariant == "geometry_only"``); this
    module introduces no new identity fields."""

    crop: tuple[int, int]
    stride: tuple[int, int]
    patch_grid: tuple[int, int]
    patch_size: tuple[int, int]
    align_corners: bool
    connectivity: int


def dfp_pullback_contract_from_protocol(
    protocol_identity: Mapping[str, Any], *, repo_root: Path,
) -> DfpPullbackContract:
    """Strictly extract and validate the pullback-relevant fields of an
    already-loaded protocol identity. Delegates to
    ``src.dfp_phase1a_identity.validate_structural_components_block``
    (soft-membership formula/operator/invariants), ``src.dfp_phase1a_
    horizon_stitching.dfp_stitching_contract_from_protocol`` (crop,
    stride, patch geometry, alignment), and ``src.dfp_phase1a_global_
    unary_components.dfp_component_contract_from_protocol``
    (connectivity, cross-bound for consistency with Step 6's own
    partition). Any failure from any delegate -- including
    ``DfpPhase1aIdentityError``, ``DfpPhase1aHorizonStitchingError``, or
    ``DfpPhase1aGlobalUnaryComponentsError`` -- is always re-raised as
    this module's own ``DfpPhase1aStitchingPullbackError`` (chained via
    ``from exc``), never left to escape unwrapped."""
    try:
        structural = protocol_identity["structural_components"]
        validate_structural_components_block(structural)
    except DfpPhase1aIdentityError as exc:
        raise DfpPhase1aStitchingPullbackError(str(exc)) from exc
    except KeyError as exc:
        raise DfpPhase1aStitchingPullbackError(f"protocol_identity is missing a required section: {exc}") from exc

    try:
        stitching_contract = dfp_stitching_contract_from_protocol(protocol_identity, repo_root=repo_root)
    except DfpPhase1aHorizonStitchingError as exc:
        raise DfpPhase1aStitchingPullbackError(str(exc)) from exc

    try:
        component_contract = dfp_component_contract_from_protocol(protocol_identity, repo_root=repo_root)
    except DfpPhase1aGlobalUnaryComponentsError as exc:
        raise DfpPhase1aStitchingPullbackError(str(exc)) from exc

    if component_contract.patch_pitch != stitching_contract.patch_size:
        raise DfpPhase1aStitchingPullbackError(
            "Step 6's patch pitch disagrees with Step 5's patch_size -- geometry is not self-consistent"
        )

    return DfpPullbackContract(
        crop=stitching_contract.crop, stride=stitching_contract.stride, patch_grid=stitching_contract.patch_grid,
        patch_size=stitching_contract.patch_size, align_corners=stitching_contract.align_corners,
        connectivity=component_contract.connectivity,
    )


# ---------------------------------------------------------------------------
# Geometry plan: window plan + geometry-only denominator + geometry-only q
# ---------------------------------------------------------------------------


def _geometry_fingerprint(image_size: tuple[int, int], contract: DfpPullbackContract, plan: SlidingWindowPlan) -> str:
    parts = [
        f"image_size={image_size}", f"crop={contract.crop}", f"stride={contract.stride}",
        f"patch_grid={contract.patch_grid}", f"align_corners={contract.align_corners}",
    ]
    for window in plan.windows:
        parts.append(
            f"w{window.index}:({window.grid_row},{window.grid_col}),"
            f"origin=({window.origin.row},{window.origin.col}),end=({window.end.row},{window.end.col})"
        )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest


@dataclass(frozen=True)
class StitchingPullbackPlan:
    """Geometry-only precomputed state for one image size: the window
    plan, the shared stitching denominator, and each window's
    geometry-only ``q[w] = A_w^T 1``. Contains no image scores, no
    ground truth, and no semantic graph data. Computed once per
    (image_size, contract) pair; reused for every component chunk."""

    image_size: tuple[int, int]
    plan: SlidingWindowPlan
    patch_grid: tuple[int, int]
    dtype: torch.dtype
    denominator: torch.Tensor  # [1, 1, H, W], read-only, geometry-only
    per_window_q: tuple[torch.Tensor, ...]  # one [N] tensor per window, read-only, geometry-only
    geometry_fingerprint: str

    def __post_init__(self) -> None:
        self.denominator.requires_grad_(False)
        for q in self.per_window_q:
            q.requires_grad_(False)


def _apply_window_bilinear_adjoint(v_scaled: torch.Tensor, *, grid_hw: tuple[int, int], crop_hw: tuple[int, int]) -> torch.Tensor:
    """Exact vector-Jacobian product of ``interpolate_raw_scores`` --
    the sole nonlinear-to-implement (but mathematically linear) stage of
    A_w. ``v_scaled`` has shape ``[K, crop_h, crop_w]`` (K = chunk size,
    or 1 for a single vector). Returns ``[K, N]``, always detached, no
    graph retained, no parameter gradients (the probe tensor is a fresh
    disconnected leaf, never a model parameter)."""
    chunk_size = v_scaled.shape[0]
    node_count = grid_hw[0] * grid_hw[1]
    probe = torch.zeros((1, node_count, chunk_size), dtype=v_scaled.dtype, device=v_scaled.device, requires_grad=True)
    forward = interpolate_raw_scores(probe, grid_hw, crop_hw)  # [1, K, crop_h, crop_w]
    loss = (forward.squeeze(0) * v_scaled).sum()
    (grad,) = torch.autograd.grad(loss, probe, retain_graph=False, create_graph=False)
    return grad.detach().squeeze(0).transpose(0, 1).contiguous()  # [K, N]


def build_stitching_pullback_plan(
    image_size: tuple[int, int], *, contract: DfpPullbackContract, dtype: torch.dtype = torch.float64,
) -> StitchingPullbackPlan:
    """Build the geometry-only plan for one image size: one
    ``SlidingWindowPlan`` (never rebuilt again), one shared denominator
    (accumulated exactly once per window, reusing Step 5's own
    ``build_stitch_weight`` primitive unmodified), and one geometry-only
    ``q[w] = A_w^T 1`` per window. Contains no scores; independent of
    horizon, class, and component content -- images sharing the exact
    same geometry produce an identical plan (verified by the returned
    ``geometry_fingerprint``)."""
    if not isinstance(contract, DfpPullbackContract):
        raise DfpPhase1aStitchingPullbackError("contract must be a DfpPullbackContract")
    if type(image_size) not in (tuple, list) or len(image_size) != 2:
        raise DfpPhase1aStitchingPullbackError("image_size must be an exact (height, width) tuple")
    h_img = _require_positive_exact_int(image_size[0], "image_size[0]")
    w_img = _require_positive_exact_int(image_size[1], "image_size[1]")

    try:
        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(h_img, w_img), crop_size=SpatialSize(*contract.crop), stride=SpatialSize(*contract.stride),
        )
    except Exception as exc:  # geometry construction errors from the reused canonical builder
        raise DfpPhase1aStitchingPullbackError(f"failed to build the canonical window plan: {exc}") from exc

    grid_hw = contract.patch_grid

    # Pass 1: the shared denominator, exactly Step 5's own accumulation
    # (build_stitch_weight reused unmodified), accumulated once per
    # window -- never once per component or horizon.
    denominator = torch.zeros((1, 1, h_img, w_img), dtype=dtype)
    for window in plan.windows:
        crop_hw = window.extent.as_tuple()
        weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=dtype)
        rows, cols = window.accumulation_slice
        denominator[:, :, rows, cols] += weight

    if not torch.isfinite(denominator).all() or not torch.all(denominator > 0):
        raise DfpPhase1aStitchingPullbackError("stitching pullback geometry left one or more pixels with non-positive denominator coverage")
    denominator = denominator.detach().clone()
    denominator.requires_grad_(False)

    # Pass 2: q[w] = A_w^T 1, geometry-only, computed once per window
    # using the now-complete denominator.
    per_window_q: list[torch.Tensor] = []
    for window in plan.windows:
        crop_hw = window.extent.as_tuple()
        weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=dtype)
        rows, cols = window.accumulation_slice
        ones_local = torch.ones((1, crop_hw[0], crop_hw[1]), dtype=dtype)
        scaled = ones_local * (weight.squeeze(0) / denominator[:, :, rows, cols].squeeze(0))
        q_w = _apply_window_bilinear_adjoint(scaled, grid_hw=grid_hw, crop_hw=crop_hw).squeeze(0)  # [N]
        q_w = q_w.detach().clone()
        q_w.requires_grad_(False)
        if not torch.isfinite(q_w).all() or not torch.all(q_w > 0):
            raise DfpPhase1aStitchingPullbackError(f"window {window.index}: q is not finite and strictly positive everywhere")
        per_window_q.append(q_w)

    fingerprint = _geometry_fingerprint((h_img, w_img), contract, plan)

    result = StitchingPullbackPlan(
        image_size=(h_img, w_img), plan=plan, patch_grid=grid_hw, dtype=dtype,
        denominator=denominator, per_window_q=tuple(per_window_q), geometry_fingerprint=fingerprint,
    )
    return result


# ---------------------------------------------------------------------------
# Forward and pullback for one window (reused/testable independently)
# ---------------------------------------------------------------------------


def _require_window_plan(window_plan: Any) -> StitchingPullbackPlan:
    if not isinstance(window_plan, StitchingPullbackPlan):
        raise DfpPhase1aStitchingPullbackError("window_plan must be a StitchingPullbackPlan")
    return window_plan


def _require_window_index(window_plan: StitchingPullbackPlan, window_index: Any) -> int:
    window_index = _require_exact_int(window_index, "window_index")
    if window_index < 0 or window_index >= len(window_plan.plan.windows):
        raise DfpPhase1aStitchingPullbackError(
            f"window_index {window_index} is out of range for {len(window_plan.plan.windows)} canonical windows"
        )
    return window_index


def _require_valid_field(tensor: Any, label: str, *, ndim: int) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise DfpPhase1aStitchingPullbackError(f"{label} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise DfpPhase1aStitchingPullbackError(f"{label} must be exactly {ndim}-D, observed {tensor.ndim}-D")
    if not tensor.is_floating_point():
        raise DfpPhase1aStitchingPullbackError(f"{label} must be floating point (observed dtype {tensor.dtype})")
    if not torch.isfinite(tensor).all():
        raise DfpPhase1aStitchingPullbackError(f"{label} contains a non-finite value (NaN or Infinity)")
    return tensor


def apply_window_stitching_forward(
    u: torch.Tensor, *, window_plan: StitchingPullbackPlan, window_index: int,
) -> torch.Tensor:
    """The per-window contribution to the final stitched field: ``A_w(u)``
    restricted to ``window.accumulation_slice`` (a ``[K, crop_h,
    crop_w]`` tensor for a ``[K, N]`` batch of class-agnostic patch
    fields ``u``), already divided by the shared denominator. This is
    the exact production computation Step 5's ``stitch_dfp_phase1a_
    horizons`` performs for one window, reused unmodified via
    ``interpolate_raw_scores`` and ``build_stitch_weight``, generalized
    only to accept an arbitrary ``K``-sized batch instead of a fixed
    class count."""
    window_plan = _require_window_plan(window_plan)
    window_index = _require_window_index(window_plan, window_index)
    _require_valid_field(u, "u", ndim=2)
    if u.shape[1] != window_plan.patch_grid[0] * window_plan.patch_grid[1]:
        raise DfpPhase1aStitchingPullbackError(
            f"u has {u.shape[1]} patch nodes, expected patch_grid product {window_plan.patch_grid[0] * window_plan.patch_grid[1]}"
        )
    window = window_plan.plan.windows[window_index]
    crop_hw = window.extent.as_tuple()
    k = u.shape[0]
    patch_scores = u.unsqueeze(0).transpose(1, 2)  # [1, N, K]
    crop = interpolate_raw_scores(patch_scores, window_plan.patch_grid, crop_hw).squeeze(0)  # [K, crop_h, crop_w]
    weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=u.dtype).squeeze(0)  # [1, crop_h, crop_w]
    rows, cols = window.accumulation_slice
    denom_local = window_plan.denominator[:, :, rows, cols].squeeze(0)  # [1, crop_h, crop_w]
    return crop * weight / denom_local


def apply_window_stitching_pullback(
    v: torch.Tensor, *, window_plan: StitchingPullbackPlan, window_index: int,
) -> torch.Tensor:
    """The exact adjoint ``A_w^T(v)`` for one window: ``v`` is a
    ``[K, H, W]`` full-image pixel-space batch (K component indicators,
    or a single all-ones vector for ``q``); the result is ``[K, N]``,
    the pullback restricted to this window's patch nodes."""
    window_plan = _require_window_plan(window_plan)
    window_index = _require_window_index(window_plan, window_index)
    _require_valid_field(v, "v", ndim=3)
    if tuple(v.shape[1:]) != window_plan.image_size:
        raise DfpPhase1aStitchingPullbackError(f"v spatial shape {tuple(v.shape[1:])} disagrees with the plan's image_size {window_plan.image_size}")
    window = window_plan.plan.windows[window_index]
    crop_hw = window.extent.as_tuple()
    rows, cols = window.accumulation_slice
    v_local = v[:, rows, cols]  # [K, crop_h, crop_w]
    weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=v.dtype).squeeze(0)
    denom_local = window_plan.denominator[:, :, rows, cols].squeeze(0)
    scaled = v_local * (weight / denom_local)
    return _apply_window_bilinear_adjoint(scaled, grid_hw=window_plan.patch_grid, crop_hw=crop_hw)


# ---------------------------------------------------------------------------
# Soft membership over a component chunk, for one window
# ---------------------------------------------------------------------------


def compute_window_soft_memberships(
    component_ids: np.ndarray, component_id_start: int, component_chunk_size: int, *,
    window_plan: StitchingPullbackPlan, window_index: int,
) -> torch.Tensor:
    """Soft membership ``M[r,w,j]`` for one window and one contiguous
    chunk of component ids ``[component_id_start, component_id_start +
    component_chunk_size)``. Returns ``[component_chunk_size, N]``. Never
    materializes more than one chunk's worth of full-image indicator
    fields at a time; never clips, renormalizes, or thresholds."""
    window_plan = _require_window_plan(window_plan)
    window_index = _require_window_index(window_plan, window_index)
    component_chunk_size = _require_positive_exact_int(component_chunk_size, "component_chunk_size")
    component_id_start = _require_exact_int(component_id_start, "component_id_start")
    if component_id_start < 0:
        raise DfpPhase1aStitchingPullbackError("component_id_start must be non-negative")
    if not isinstance(component_ids, np.ndarray) or component_ids.dtype != np.int64:
        raise DfpPhase1aStitchingPullbackError("component_ids must be a numpy.ndarray with dtype exactly int64")
    if component_ids.ndim != 2:
        raise DfpPhase1aStitchingPullbackError(f"component_ids must be exactly 2-D [H, W], observed {component_ids.ndim}-D")
    if tuple(component_ids.shape) != window_plan.image_size:
        raise DfpPhase1aStitchingPullbackError(
            f"component_ids shape {tuple(component_ids.shape)} disagrees with the plan's image_size {window_plan.image_size}"
        )

    ids_torch = torch.from_numpy(component_ids)
    target_ids = torch.arange(component_id_start, component_id_start + component_chunk_size, dtype=torch.int64)
    indicator = (ids_torch.unsqueeze(0) == target_ids.view(-1, 1, 1)).to(window_plan.dtype)  # [K, H, W]

    numerator = apply_window_stitching_pullback(indicator, window_plan=window_plan, window_index=window_index)  # [K, N]
    q_w = window_plan.per_window_q[window_index]  # [N]
    membership = numerator / q_w.unsqueeze(0)
    return membership.detach()


def _iter_image_soft_memberships_body(
    component_ids: np.ndarray, component_count: int, *, window_plan: StitchingPullbackPlan, component_chunk_size: int,
) -> Iterator[tuple[int, dict[int, torch.Tensor]]]:
    start = 0
    while start < component_count:
        size = min(component_chunk_size, component_count - start)
        chunk: dict[int, torch.Tensor] = {}
        for window_index in range(len(window_plan.plan.windows)):
            chunk[window_index] = compute_window_soft_memberships(
                component_ids, start, size, window_plan=window_plan, window_index=window_index,
            )
        yield start, chunk
        start += size


def iter_image_soft_memberships(
    component_ids: np.ndarray, component_count: int, *, window_plan: StitchingPullbackPlan, component_chunk_size: int,
) -> Iterator[tuple[int, dict[int, torch.Tensor]]]:
    """Yield ``(component_id_start, {window_index: membership[K, N]})``
    for every canonical-order component chunk covering ``0..component_
    count-1``, across every window in canonical Step 5 order. No
    component is dropped, merged, or reordered; the last chunk may be
    smaller than ``component_chunk_size``.

    Validation runs eagerly, before the returned generator's first item
    is ever produced -- this function is a plain function that validates
    then returns a generator, not a generator function itself, so a
    malformed ``component_count``/``component_chunk_size``/``window_
    plan`` is rejected immediately on the call, not deferred to the
    first ``next()``."""
    window_plan = _require_window_plan(window_plan)
    component_count = _require_positive_exact_int(component_count, "component_count")
    component_chunk_size = _require_positive_exact_int(component_chunk_size, "component_chunk_size")
    if not isinstance(component_ids, np.ndarray) or component_ids.dtype != np.int64:
        raise DfpPhase1aStitchingPullbackError("component_ids must be a numpy.ndarray with dtype exactly int64")
    if tuple(component_ids.shape) != window_plan.image_size:
        raise DfpPhase1aStitchingPullbackError(
            f"component_ids shape {tuple(component_ids.shape)} disagrees with the plan's image_size {window_plan.image_size}"
        )
    return _iter_image_soft_memberships_body(
        component_ids, component_count, window_plan=window_plan, component_chunk_size=component_chunk_size,
    )


# ---------------------------------------------------------------------------
# Validation and diagnostics (never gate/threshold/renormalize)
# ---------------------------------------------------------------------------


def validate_soft_memberships(
    membership: torch.Tensor, *, node_count: int, atol: float = 0.0,
) -> dict[str, float]:
    """Independently re-check ``membership`` (shape ``[K, N]``, for one
    window and ANY chunk of components -- possibly a strict subset of
    all R): non-negativity and finiteness only. Conservation
    (``sum_r M[r,w,j] == 1``) requires summing over every component, so
    it is checked separately by :func:`validate_conservation` on an
    accumulated column-sum vector -- a single chunk is never expected to
    sum to 1 by itself. Never mutates ``membership``, never clips or
    renormalizes on failure."""
    if membership.ndim != 2 or membership.shape[1] != node_count:
        raise DfpPhase1aStitchingPullbackError(f"membership must have shape [K, {node_count}], observed {tuple(membership.shape)}")
    if not torch.isfinite(membership).all():
        raise DfpPhase1aStitchingPullbackError("membership contains a non-finite value")
    min_value = float(membership.min())
    if min_value < -atol:
        raise DfpPhase1aStitchingPullbackError(f"membership contains a negative value below tolerance: {min_value}")
    return {"min_value": min_value, "chunk_size": membership.shape[0]}


def accumulate_membership_column_sums(
    running_total: torch.Tensor | None, membership: torch.Tensor,
) -> torch.Tensor:
    """Fold one component chunk's contribution into a running
    per-(window,patch) column-sum accumulator (``sum_r M[r,w,j]`` over
    the components seen so far), so a caller can verify conservation
    across the full component set without ever materializing all R
    components' memberships simultaneously. ``running_total=None``
    starts a fresh accumulator sized from ``membership``."""
    chunk_sum = membership.sum(dim=0)
    if running_total is None:
        return chunk_sum.detach().clone()
    if running_total.shape != chunk_sum.shape:
        raise DfpPhase1aStitchingPullbackError(
            f"running_total shape {tuple(running_total.shape)} disagrees with this chunk's shape {tuple(chunk_sum.shape)}"
        )
    return (running_total + chunk_sum).detach()


def validate_conservation(column_sums: torch.Tensor, *, node_count: int) -> dict[str, float]:
    """Independently re-check per-(window,patch)-source conservation
    (``sum_r M[r,w,j] == 1``) given a column-sum vector already
    accumulated over every component (see
    :func:`accumulate_membership_column_sums`). Never renormalizes or
    clips -- returns full-precision residual diagnostics; the protocol
    binds no runtime conservation tolerance, so none is invented here."""
    if column_sums.ndim != 1 or column_sums.shape[0] != node_count:
        raise DfpPhase1aStitchingPullbackError(f"column_sums must have shape [{node_count}], observed {tuple(column_sums.shape)}")
    if not torch.isfinite(column_sums).all():
        raise DfpPhase1aStitchingPullbackError("column_sums contains a non-finite value")
    residual = (column_sums - 1.0).abs()
    return {"max_conservation_residual": float(residual.max()), "mean_conservation_residual": float(residual.mean())}


def component_source_mass(chunk_memberships: Mapping[int, torch.Tensor]) -> torch.Tensor:
    """Diagnostic-only ``m_r = sum_{w,j} M[r,w,j]`` for one component
    chunk, given ``{window_index: membership[K, N]}``. Never used for a
    gate or threshold in this step -- ``source_copy_mass_is_diagnostic_
    only`` is a frozen identity invariant."""
    totals = None
    for tensor in chunk_memberships.values():
        window_total = tensor.sum(dim=1)
        totals = window_total if totals is None else totals + window_total
    if totals is None:
        raise DfpPhase1aStitchingPullbackError("chunk_memberships must be non-empty")
    return totals.detach()


__all__ = [
    "DfpPhase1aStitchingPullbackError",
    "DfpPullbackContract",
    "StitchingPullbackPlan",
    "dfp_pullback_contract_from_protocol",
    "build_stitching_pullback_plan",
    "apply_window_stitching_forward",
    "apply_window_stitching_pullback",
    "compute_window_soft_memberships",
    "iter_image_soft_memberships",
    "validate_soft_memberships",
    "accumulate_membership_column_sums",
    "validate_conservation",
    "component_source_mass",
]
