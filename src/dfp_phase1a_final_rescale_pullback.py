"""DFP Phase 1A -- composed final-resolution stitching pullback.

**Duplication gate.** This module never reimplements Step 7's window-level
pullback (`src.dfp_phase1a_stitching_pullback.apply_window_stitching_
pullback`, the adjoint of `A_stitch` -- one window's patch-grid-to-crop
bilinear interpolation plus uniform stitch-weight averaging) or Step 5's
stitching primitives (`interpolate_raw_scores`, `build_stitch_weight`).
Every one of those is imported and called exactly as committed, never
copied, never modified.

**What this module adds.** The `[final_resolution_analysis]` authority
block (frozen additively in `evaluation_identities/
e13_dfp_phase1a_protocol.toml`, validated by `src.dfp_phase1a_identity.
validate_final_resolution_analysis_block`) requires every downstream
analysis (dispute, oracle, persistence, confusion, global unary
components, reversion coherence) to consume the stitched score field
AFTER the exact existing matched-evaluator rescale-to-`ori_shape`
operation (`matched_power_evaluator.finalize_prediction`'s own crop-then-
bilinear-resize, minus its trailing argmax) -- never the raw pre-rescale
Step-5 output, and never an approximate (nearest-neighbor) resampling of
component IDs or membership masks. Calling that composed field `R`
(`stitched -> final ori-resolution scores`) and Step 7's own window
operator `A_stitch` (`patch sources -> stitched inference pixels`), the
soft-membership operator every downstream consumer actually needs is the
COMPOSITION:

    A_final = R composed with A_stitch
    A_final^T = A_stitch^T composed with R^T

This module implements exactly three new primitives, each provably
exact (never approximated) and each proven against small explicit dense
matrices in this module's own test suite:

  1. `apply_final_rescale_forward` -- R itself: the identical crop+
     bilinear-resize formula `finalize_prediction` already uses,
     reproduced here (never called through that function directly,
     since that function's own trailing argmax is not wanted here, and
     the frozen module is never modified to expose a no-argmax variant)
     and bound, byte-for-byte, to that function's own file-content
     SHA-256 via the frozen `[final_resolution_analysis].rescale_
     source_sha256` field.
  2. `apply_final_rescale_adjoint` -- the EXACT adjoint R^T of that same
     forward operator, decomposed as Crop^T (a trivial zero-pad scatter)
     composed with the bilinear resize's own adjoint (computed via a
     disconnected, gradient-isolated `torch.autograd.grad` probe --
     the SAME technique `dfp_phase1a_stitching_pullback._apply_window_
     bilinear_adjoint` already uses and already proves safe, applied
     here to a genuinely different linear map with its own dedicated
     adjoint-identity tests, never reusing that function's own narrower
     window-scoped contract).
  3. `apply_final_window_stitching_pullback` / `compute_window_final_
     soft_memberships` / `iter_image_final_soft_memberships` -- the
     composed `A_final^T`, built by applying `apply_final_rescale_
     adjoint` and THEN Step 7's own unmodified `apply_window_stitching_
     pullback` -- never a dense pixel-by-patch matrix, never a
     nearest-neighbor resample of a component ID or membership mask
     anywhere in this file.

No gradient ever flows into the model or backbone through any function
in this module: every `torch.autograd.grad` call here operates on a
freshly constructed, disconnected leaf tensor (`requires_grad_(True)`
set only on that fresh probe), with `retain_graph=False` and
`create_graph=False`, and every returned tensor is `.detach()`-ed before
being handed back to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from src.dfp_phase1a_stitching_pullback import (
    DfpPhase1aStitchingPullbackError,
    StitchingPullbackPlan,
    apply_window_stitching_pullback,
)


class DfpPhase1aFinalRescalePullbackError(DfpPhase1aStitchingPullbackError):
    """Raised on any composed final-resolution pullback invariant
    violation -- a subclass of Step 7's own domain error, since this
    module is additive orchestration over Step 7's pullback machinery
    and never introduces a second, unrelated failure taxonomy. Always
    fail closed: never approximate, resample, or silently substitute a
    fallback resize path."""


def _require_positive_exact_int(value: Any, label: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be an exact non-boolean integer")
    if value <= 0:
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be positive")
    return value


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be an exact non-boolean integer")
    return value


def _require_shape_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) not in (tuple, list) or len(value) != 2:
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be an exact (height, width) pair")
    h = _require_positive_exact_int(value[0], f"{label}[0]")
    w = _require_positive_exact_int(value[1], f"{label}[1]")
    return (h, w)


# ---------------------------------------------------------------------------
# Contract: the exact bound rescale operator, sourced only from the live
# [final_resolution_analysis] protocol authority.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DfpFinalResolutionContract:
    """Independently validated final-rescale parameters extracted from
    the live protocol identity -- never hardcoded module constants."""

    align_corners: bool
    antialias: bool
    interpolation_mode: str
    source_relative_path: str
    source_sha256: str
    source_function: str


def dfp_final_resolution_contract_from_protocol(
    protocol_identity: Mapping[str, Any], *, repo_root: Path,
) -> DfpFinalResolutionContract:
    """Strictly extract and validate the `[final_resolution_analysis]`
    fields of an already-loaded protocol identity, delegating entirely
    to `src.dfp_phase1a_identity.validate_final_resolution_analysis_block`
    (structural/relational validation, cross-checked against the live
    `[structural_components]` block) and `validate_final_resolution_
    rescale_source_binding` (the bound rescale source file's REAL bytes
    on disk, re-hashed and compared against the frozen `rescale_source_
    sha256` -- never trusted from the TOML alone). Any failure is
    re-raised as this module's own `DfpPhase1aFinalRescalePullbackError`."""
    from src.dfp_phase1a_identity import (
        DfpPhase1aIdentityError,
        validate_final_resolution_analysis_block,
        validate_final_resolution_rescale_source_binding,
    )

    try:
        block = protocol_identity["final_resolution_analysis"]
        structural = protocol_identity["structural_components"]
        validate_final_resolution_analysis_block(block, structural)
        validate_final_resolution_rescale_source_binding(Path(repo_root), protocol_identity)
    except DfpPhase1aIdentityError as exc:
        raise DfpPhase1aFinalRescalePullbackError(str(exc)) from exc
    except KeyError as exc:
        raise DfpPhase1aFinalRescalePullbackError(f"protocol_identity is missing a required section: {exc}") from exc

    if block["rescale_interpolation_mode"] != "bilinear":
        raise DfpPhase1aFinalRescalePullbackError(
            f"this module implements only mode='bilinear'; identity declares {block['rescale_interpolation_mode']!r}"
        )

    return DfpFinalResolutionContract(
        align_corners=block["rescale_align_corners"], antialias=block["rescale_antialias"],
        interpolation_mode=block["rescale_interpolation_mode"], source_relative_path=block["rescale_source_relative_path"],
        source_sha256=block["rescale_source_sha256"], source_function=block["rescale_source_function"],
    )


# ---------------------------------------------------------------------------
# R: the forward final-rescale operator (crop then bilinear resize).
# Byte-for-byte the same formula matched_power_evaluator.finalize_
# prediction uses, minus that function's own trailing argmax.
# ---------------------------------------------------------------------------


def _require_valid_batched_field(tensor: Any, label: str) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be a torch.Tensor")
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be a batched [1, K, H, W] tensor, observed shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be floating point (observed dtype {tensor.dtype})")
    if not torch.isfinite(tensor).all():
        raise DfpPhase1aFinalRescalePullbackError(f"{label} contains a non-finite value (NaN or Infinity)")
    return tensor


def apply_final_rescale_forward(
    stitched: torch.Tensor, *, img_shape: tuple[int, int], ori_shape: tuple[int, int], contract: DfpFinalResolutionContract,
) -> torch.Tensor:
    """R(stitched): crop `stitched` (shape `[1, K, H, W]`, `H >= img_shape[0]`,
    `W >= img_shape[1]`) to `img_shape` (removing any sliding-window
    padding -- a no-op slice when `stitched`'s own spatial extent
    already equals `img_shape`, exactly as in this pipeline, which has
    no Pad transform), then bilinearly interpolates to `ori_shape` using
    `contract.align_corners`. Identical to `matched_power_evaluator.
    finalize_prediction`'s own `cropped = stitched[:, :, :resize_h,
    :resize_w]; F.interpolate(cropped, size=ori_shape, mode="bilinear",
    align_corners=align_corners)`, minus that function's trailing
    argmax -- this module needs linear scores, not a finalized label
    map. Never mutates `stitched`."""
    stitched = _require_valid_batched_field(stitched, "stitched")
    img_h, img_w = _require_shape_pair(img_shape, "img_shape")
    ori_h, ori_w = _require_shape_pair(ori_shape, "ori_shape")
    if img_h > stitched.shape[2] or img_w > stitched.shape[3]:
        raise DfpPhase1aFinalRescalePullbackError(
            f"img_shape {img_shape} exceeds stitched's own spatial extent {tuple(stitched.shape[2:])}"
        )
    if not isinstance(contract, DfpFinalResolutionContract):
        raise DfpPhase1aFinalRescalePullbackError("contract must be a DfpFinalResolutionContract")

    cropped = stitched[:, :, :img_h, :img_w]
    rescaled = F.interpolate(cropped, size=(ori_h, ori_w), mode="bilinear", align_corners=contract.align_corners)
    if not torch.isfinite(rescaled).all():
        raise DfpPhase1aFinalRescalePullbackError("apply_final_rescale_forward produced a non-finite value")
    return rescaled


# ---------------------------------------------------------------------------
# R^T: the exact adjoint, decomposed as Crop^T (zero-pad scatter) composed
# with the bilinear resize's own adjoint (autograd-probe technique).
# ---------------------------------------------------------------------------


def _bilinear_resize_adjoint(v: torch.Tensor, *, in_size: tuple[int, int], out_size: tuple[int, int], align_corners: bool) -> torch.Tensor:
    """The exact vector-Jacobian product of
    ``F.interpolate(x, size=out_size, mode="bilinear", align_corners=...)``
    at any input of shape ``in_size`` -- linear, so independent of the
    (unused) input value itself. ``v`` has shape ``[K, out_h, out_w]``;
    returns ``[K, in_h, in_w]``. The probe tensor is a fresh, disconnected
    leaf (never a model parameter, never part of any external graph);
    the returned gradient is always detached -- no graph is retained and
    no parameter gradient is ever touched.

    Inference-context safe: this local autograd probe must produce the
    identical result whether the caller is in ordinary grad-enabled mode,
    ``torch.no_grad()``, ``torch.set_grad_enabled(False)``, or
    ``torch.inference_mode()`` -- exactly the ambient state every real
    inference pipeline runs under. ``v`` is detached before use (it may
    itself be an inference-mode tensor, which autograd otherwise refuses
    to touch even as a constant), and the probe/forward/backward triple
    is wrapped in a narrowly scoped ``torch.inference_mode(False)`` +
    ``torch.enable_grad()`` pair -- covering only this construction, never
    the caller's own tensors or any surrounding model computation. Both
    context managers restore the caller's exact prior state on every exit
    path (success, a raised domain error, or any BaseException, including
    KeyboardInterrupt/SystemExit/MemoryError) -- that restoration is a
    property of Python's own context-manager protocol, not additional
    logic here to get wrong. No model gradient is ever enabled: nothing
    inside this block references a model parameter, and the probe is a
    fresh leaf every call.

    ``v`` may itself have been created under an ACTIVE ``torch.inference_
    mode()`` (an ordinary occurrence: the whole surrounding pipeline runs
    under one ambient mode). Such a tensor carries an inference-tensor
    marker that survives a bare ``.detach()`` and can never be saved for
    backward, even once inference mode is locally disabled here -- so the
    detach-and-clone happens INSIDE the ``inference_mode(False)`` block,
    which is what actually produces an ordinary, non-inference tensor
    safe to use in this local autograd computation."""
    k = v.shape[0]
    in_h, in_w = in_size
    with torch.inference_mode(False):
        with torch.enable_grad():
            v_normal = v.detach().clone()  # strips any inference-tensor marker; never touches v itself
            probe = torch.zeros((1, k, in_h, in_w), dtype=v.dtype, device=v.device, requires_grad=True)
            forward = F.interpolate(probe, size=tuple(out_size), mode="bilinear", align_corners=align_corners)  # [1, K, out_h, out_w]
            loss = (forward.squeeze(0) * v_normal).sum()
            (grad,) = torch.autograd.grad(loss, probe, retain_graph=False, create_graph=False)
    return grad.detach().squeeze(0)  # [K, in_h, in_w]


def _require_valid_cotangent(v: Any, label: str, *, expected_shape: tuple[int, int]) -> torch.Tensor:
    if not torch.is_tensor(v):
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be a torch.Tensor")
    if v.ndim != 3:
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be exactly 3-D [K, H, W], observed {v.ndim}-D")
    if tuple(v.shape[1:]) != expected_shape:
        raise DfpPhase1aFinalRescalePullbackError(f"{label} spatial shape {tuple(v.shape[1:])} disagrees with expected {expected_shape}")
    if not v.is_floating_point():
        raise DfpPhase1aFinalRescalePullbackError(f"{label} must be floating point")
    if not torch.isfinite(v).all():
        raise DfpPhase1aFinalRescalePullbackError(f"{label} contains a non-finite value (NaN or Infinity)")
    return v


def apply_final_rescale_adjoint(
    v: torch.Tensor, *, stitched_shape: tuple[int, int], img_shape: tuple[int, int], ori_shape: tuple[int, int],
    contract: DfpFinalResolutionContract,
) -> torch.Tensor:
    """R^T(v) = Crop^T(Resize^T(v)): ``v`` is a ``[K, ori_h, ori_w]``
    full-image-resolution cotangent; the bilinear resize's exact adjoint
    (``_bilinear_resize_adjoint``) brings it back to ``img_shape``, then
    Crop's exact adjoint (zero-padding scatter into the top-left
    ``img_shape`` region of a ``stitched_shape``-sized zero tensor --
    trivial and exact, since Crop is itself a coordinate-selection
    linear map) brings it to ``stitched_shape`` -- the same spatial
    extent `apply_final_rescale_forward`'s own input, and Step 5's raw
    stitched output, both use."""
    img_h, img_w = _require_shape_pair(img_shape, "img_shape")
    ori_h, ori_w = _require_shape_pair(ori_shape, "ori_shape")
    stitched_h, stitched_w = _require_shape_pair(stitched_shape, "stitched_shape")
    if img_h > stitched_h or img_w > stitched_w:
        raise DfpPhase1aFinalRescalePullbackError(f"img_shape {img_shape} exceeds stitched_shape {stitched_shape}")
    if not isinstance(contract, DfpFinalResolutionContract):
        raise DfpPhase1aFinalRescalePullbackError("contract must be a DfpFinalResolutionContract")
    v = _require_valid_cotangent(v, "v", expected_shape=(ori_h, ori_w))

    resize_adjoint = _bilinear_resize_adjoint(v, in_size=(img_h, img_w), out_size=(ori_h, ori_w), align_corners=contract.align_corners)
    k = resize_adjoint.shape[0]
    out = resize_adjoint.new_zeros((k, stitched_h, stitched_w))
    out[:, :img_h, :img_w] = resize_adjoint
    if not torch.isfinite(out).all():
        raise DfpPhase1aFinalRescalePullbackError("apply_final_rescale_adjoint produced a non-finite value")
    return out.detach()


# ---------------------------------------------------------------------------
# Composed A_final^T = A_stitch^T composed with R^T, for one window.
# Reuses Step 7's apply_window_stitching_pullback completely unmodified.
# ---------------------------------------------------------------------------


def apply_final_window_stitching_pullback(
    v: torch.Tensor, *, window_plan: StitchingPullbackPlan, window_index: int, img_shape: tuple[int, int],
    ori_shape: tuple[int, int], contract: DfpFinalResolutionContract,
) -> torch.Tensor:
    """A_final^T(v) for one window: ``v`` is a ``[K, ori_h, ori_w]``
    full ori-resolution cotangent; ``apply_final_rescale_adjoint`` brings
    it to Step 7's own ``window_plan.image_size`` (== ``img_shape``,
    checked below), and Step 7's own, completely unmodified
    ``apply_window_stitching_pullback`` finishes the composition,
    restricted to this window's patch nodes -- ``[K, N]``."""
    if not isinstance(window_plan, StitchingPullbackPlan):
        raise DfpPhase1aFinalRescalePullbackError("window_plan must be a StitchingPullbackPlan")
    if tuple(window_plan.image_size) != tuple(_require_shape_pair(img_shape, "img_shape")):
        raise DfpPhase1aFinalRescalePullbackError(
            f"window_plan.image_size {window_plan.image_size} disagrees with img_shape {img_shape}"
        )
    pulled_to_stitch = apply_final_rescale_adjoint(
        v, stitched_shape=window_plan.image_size, img_shape=img_shape, ori_shape=ori_shape, contract=contract,
    )
    try:
        return apply_window_stitching_pullback(pulled_to_stitch, window_plan=window_plan, window_index=window_index)
    except DfpPhase1aStitchingPullbackError as exc:
        raise DfpPhase1aFinalRescalePullbackError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Geometry-only q_final[w] = A_final^T @ 1_ori, one per window, computed
# once per image and reused across every component chunk -- mirrors Step
# 7's own per_window_q exactly, composed with R^T.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalStitchingPullbackPlan:
    """Geometry-only precomputed state for one image's composed A_final
    operator: the underlying Step-7 `StitchingPullbackPlan`, the exact
    bound rescale contract, and each window's geometry-only ``q_final[w]
    = A_final^T @ 1`` (evaluated over the ori-resolution all-ones
    vector -- NOT Step 7's own `per_window_q`, which is geometry-only
    for `A_stitch` alone). Contains no component/membership content;
    computed once per (window_plan, img_shape, ori_shape, contract)
    tuple, reused for every component chunk."""

    window_plan: StitchingPullbackPlan
    img_shape: tuple[int, int]
    ori_shape: tuple[int, int]
    contract: DfpFinalResolutionContract
    per_window_q_final: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        for q in self.per_window_q_final:
            q.requires_grad_(False)


def build_final_stitching_pullback_plan(
    window_plan: StitchingPullbackPlan, *, img_shape: tuple[int, int], ori_shape: tuple[int, int], contract: DfpFinalResolutionContract,
) -> FinalStitchingPullbackPlan:
    if not isinstance(window_plan, StitchingPullbackPlan):
        raise DfpPhase1aFinalRescalePullbackError("window_plan must be a StitchingPullbackPlan")
    img_h, img_w = _require_shape_pair(img_shape, "img_shape")
    ori_h, ori_w = _require_shape_pair(ori_shape, "ori_shape")
    if tuple(window_plan.image_size) != (img_h, img_w):
        raise DfpPhase1aFinalRescalePullbackError(f"window_plan.image_size {window_plan.image_size} disagrees with img_shape {img_shape}")

    ones_ori = torch.ones((1, ori_h, ori_w), dtype=window_plan.dtype)
    per_window_q_final: list[torch.Tensor] = []
    for window_index in range(len(window_plan.plan.windows)):
        q_w = apply_final_window_stitching_pullback(
            ones_ori, window_plan=window_plan, window_index=window_index, img_shape=(img_h, img_w), ori_shape=(ori_h, ori_w), contract=contract,
        ).squeeze(0)
        if not torch.isfinite(q_w).all() or not torch.all(q_w > 0):
            raise DfpPhase1aFinalRescalePullbackError(f"window {window_index}: q_final is not finite and strictly positive everywhere")
        per_window_q_final.append(q_w.detach().clone())

    return FinalStitchingPullbackPlan(
        window_plan=window_plan, img_shape=(img_h, img_w), ori_shape=(ori_h, ori_w), contract=contract,
        per_window_q_final=tuple(per_window_q_final),
    )


# ---------------------------------------------------------------------------
# Soft membership over a component chunk (components defined at ori
# resolution), for one window, via the composed A_final operator.
# ---------------------------------------------------------------------------


def compute_window_final_soft_memberships(
    component_ids: np.ndarray, component_id_start: int, component_chunk_size: int, *,
    final_plan: FinalStitchingPullbackPlan, window_index: int,
) -> torch.Tensor:
    """Soft membership ``M[r,w,j]`` for one window and one contiguous
    chunk of component ids, with components defined DIRECTLY on the
    final ori-resolution stitched T0 field (``component_ids.shape ==
    final_plan.ori_shape`` -- never resized, never resampled). Returns
    ``[component_chunk_size, N]``. Never materializes more than one
    chunk's worth of full-ori-resolution indicator fields at a time."""
    if not isinstance(final_plan, FinalStitchingPullbackPlan):
        raise DfpPhase1aFinalRescalePullbackError("final_plan must be a FinalStitchingPullbackPlan")
    window_index = _require_exact_int(window_index, "window_index")
    if window_index < 0 or window_index >= len(final_plan.window_plan.plan.windows):
        raise DfpPhase1aFinalRescalePullbackError(f"window_index {window_index} is out of range")
    component_chunk_size = _require_positive_exact_int(component_chunk_size, "component_chunk_size")
    component_id_start = _require_exact_int(component_id_start, "component_id_start")
    if component_id_start < 0:
        raise DfpPhase1aFinalRescalePullbackError("component_id_start must be non-negative")
    if not isinstance(component_ids, np.ndarray) or component_ids.dtype != np.int64:
        raise DfpPhase1aFinalRescalePullbackError("component_ids must be a numpy.ndarray with dtype exactly int64")
    if component_ids.ndim != 2:
        raise DfpPhase1aFinalRescalePullbackError(f"component_ids must be exactly 2-D [H, W], observed {component_ids.ndim}-D")
    if tuple(component_ids.shape) != final_plan.ori_shape:
        raise DfpPhase1aFinalRescalePullbackError(
            f"component_ids shape {tuple(component_ids.shape)} disagrees with final_plan.ori_shape {final_plan.ori_shape}"
        )

    ids_torch = torch.from_numpy(np.ascontiguousarray(component_ids))
    target_ids = torch.arange(component_id_start, component_id_start + component_chunk_size, dtype=torch.int64)
    indicator = (ids_torch.unsqueeze(0) == target_ids.view(-1, 1, 1)).to(final_plan.window_plan.dtype)  # [K, ori_h, ori_w]

    numerator = apply_final_window_stitching_pullback(
        indicator, window_plan=final_plan.window_plan, window_index=window_index,
        img_shape=final_plan.img_shape, ori_shape=final_plan.ori_shape, contract=final_plan.contract,
    )  # [K, N]
    q_w = final_plan.per_window_q_final[window_index]  # [N]
    membership = numerator / q_w.unsqueeze(0)
    return membership.detach()


def _iter_image_final_soft_memberships_body(
    component_ids: np.ndarray, component_count: int, *, final_plan: FinalStitchingPullbackPlan, component_chunk_size: int,
) -> Iterator[tuple[int, dict[int, torch.Tensor]]]:
    start = 0
    while start < component_count:
        size = min(component_chunk_size, component_count - start)
        chunk: dict[int, torch.Tensor] = {}
        for window_index in range(len(final_plan.window_plan.plan.windows)):
            chunk[window_index] = compute_window_final_soft_memberships(
                component_ids, start, size, final_plan=final_plan, window_index=window_index,
            )
        yield start, chunk
        start += size


def iter_image_final_soft_memberships(
    component_ids: np.ndarray, component_count: int, *, final_plan: FinalStitchingPullbackPlan, component_chunk_size: int,
) -> Iterator[tuple[int, dict[int, torch.Tensor]]]:
    """Yield ``(component_id_start, {window_index: membership[K, N]})``
    for every canonical-order component chunk covering ``0..component_
    count-1``, across every window in canonical Step 5 order, via the
    composed A_final operator. Validation runs eagerly, before the
    returned generator's first item, exactly like Step 7's own
    `iter_image_soft_memberships`."""
    if not isinstance(final_plan, FinalStitchingPullbackPlan):
        raise DfpPhase1aFinalRescalePullbackError("final_plan must be a FinalStitchingPullbackPlan")
    component_count = _require_positive_exact_int(component_count, "component_count")
    component_chunk_size = _require_positive_exact_int(component_chunk_size, "component_chunk_size")
    if not isinstance(component_ids, np.ndarray) or component_ids.dtype != np.int64:
        raise DfpPhase1aFinalRescalePullbackError("component_ids must be a numpy.ndarray with dtype exactly int64")
    if tuple(component_ids.shape) != final_plan.ori_shape:
        raise DfpPhase1aFinalRescalePullbackError(
            f"component_ids shape {tuple(component_ids.shape)} disagrees with final_plan.ori_shape {final_plan.ori_shape}"
        )
    return _iter_image_final_soft_memberships_body(
        component_ids, component_count, final_plan=final_plan, component_chunk_size=component_chunk_size,
    )


def component_source_mass_float64(chunk_memberships: Mapping[int, torch.Tensor]) -> torch.Tensor:
    """``m_r = sum_{w,j} M[r,w,j]`` for one component chunk, accumulated
    in float64 regardless of the pullback plan's own working dtype --
    identical accumulation convention to Step 8's own `component_source_
    mass_float64`, reproduced here (never imported from Step 8, since
    that function's own docstring scopes it to Step 7's `A_stitch`-only
    chunk_memberships shape -- the numeric formula is identical, but this
    is a distinct, ori-resolution-composed input)."""
    if not isinstance(chunk_memberships, dict) or not chunk_memberships:
        raise DfpPhase1aFinalRescalePullbackError("chunk_memberships must be a non-empty dict")
    totals: torch.Tensor | None = None
    for tensor in chunk_memberships.values():
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise DfpPhase1aFinalRescalePullbackError("each chunk_memberships value must be a 2-D torch.Tensor [K, N]")
        window_total = tensor.to(torch.float64).sum(dim=1)
        totals = window_total if totals is None else totals + window_total
    return totals.detach()


__all__ = [
    "DfpPhase1aFinalRescalePullbackError",
    "DfpFinalResolutionContract",
    "dfp_final_resolution_contract_from_protocol",
    "apply_final_rescale_forward",
    "apply_final_rescale_adjoint",
    "apply_final_window_stitching_pullback",
    "FinalStitchingPullbackPlan",
    "build_final_stitching_pullback_plan",
    "compute_window_final_soft_memberships",
    "iter_image_final_soft_memberships",
    "component_source_mass_float64",
]
