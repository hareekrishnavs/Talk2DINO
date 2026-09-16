"""DFP Phase 1B: the pure Boundary-Aware Late-Flip Preservation (BALP)
scientific kernel, implementing exactly the formulas frozen in
`evaluation_identities/e22_dfp_phase1b_lightweight_boundary_preservation.toml`
(`[canonical_horizons]`, `[boundary_definition]`, `[stability_and_late_flip]`,
`[candidate_family]`). Introduces no new scientific formula, no learned
parameter, no confidence threshold, and no ground-truth input.

This module is pure tensor arithmetic: it never imports the E22
identity loader, never reads a file, and never imports any masked-
forward/component/pullback module. Importing this module never
initializes CUDA -- only whoever constructs a CUDA tensor and passes
it in does that.

Efficiency: E22 proves the relational equivalence
``argmax(where(P, S64, S320)) == where(P, y64, y320)`` -- so candidate
labels are produced directly via ``torch.where`` on the two ALREADY-
COMPUTED control label maps, never by materializing a third
``[class_count, H, W]`` score tensor per candidate.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping

import torch

STABILITY_STARTS: tuple[int, ...] = (0, 16, 32, 64)
RADII: tuple[int, ...] = (0, 1, 2, 4)
HORIZON_SET: tuple[int, ...] = (0, 16, 32, 64, 320)
CANONICAL_CONTROLS: tuple[str, ...] = ("T0", "T16", "T32", "T64", "T320")
_HORIZON_TO_CONTROL = {0: "T0", 16: "T16", 32: "T32", 64: "T64", 320: "T320"}


class DfpPhase1bLightweightBoundaryKernelError(ValueError):
    """Raised on any structural/scientific-input invariant violation of
    the BALP kernel. Always fail closed. Never raised for, and never
    catches, KeyboardInterrupt / SystemExit / MemoryError -- those
    always propagate unmodified."""


def candidate_id(stability_start: int, radius: int) -> str:
    return f"balp_s{stability_start:03d}_r{radius:02d}"


def reconstruct_candidate_ids() -> tuple[str, ...]:
    """The exact 16 ordered candidate IDs, regenerated from the frozen
    (stability_starts, radii) grid -- row-major, stability_starts outer
    ascending, radii inner ascending. Identical construction to
    `src.dfp_phase1b_lightweight_boundary_identity.reconstruct_candidate_ids`,
    reproduced here rather than imported so this module never depends
    on the E22 identity/loader machinery (a pure kernel has no need to
    parse TOML at import time)."""
    return tuple(candidate_id(s, r) for s in STABILITY_STARTS for r in RADII)


CANDIDATE_IDS: tuple[str, ...] = reconstruct_candidate_ids()


def _require_score_field(tensor: object, label: str, *, class_count: int) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} must be a torch.Tensor, observed {type(tensor).__name__}")
    if tensor.ndim != 3:
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} must be exactly 3-D [class_count, H, W], observed {tensor.ndim}-D")
    if tensor.shape[0] != class_count:
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} class dimension {tensor.shape[0]} disagrees with class_count {class_count}")
    if tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} spatial dimensions must be positive, observed {tuple(tensor.shape[1:])}")
    if not tensor.is_floating_point():
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} must be a floating-point tensor, observed dtype {tensor.dtype}")
    if not torch.isfinite(tensor).all():
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} contains a non-finite value (NaN or Infinity)")
    return tensor


def _require_identical_signature(reference: torch.Tensor, tensor: torch.Tensor, label: str, ref_label: str) -> None:
    if tensor.shape != reference.shape:
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} shape {tuple(tensor.shape)} disagrees with {ref_label} shape {tuple(reference.shape)}")
    if tensor.dtype != reference.dtype:
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} dtype {tensor.dtype} disagrees with {ref_label} dtype {reference.dtype}")
    if tensor.device != reference.device:
        raise DfpPhase1bLightweightBoundaryKernelError(f"{label} device {tensor.device} disagrees with {ref_label} device {reference.device}")


def lowest_index_argmax(scores_chw: torch.Tensor) -> torch.Tensor:
    """label(x) = lowest class index c maximizing scores_chw[c, x].

    Implemented explicitly via a tie mask + argmin-over-class-index,
    the identical construction to the already-committed, E22-bound
    `src.dfp_phase1a_analysis._lowest_index_argmax` -- never via
    `torch.argmax`'s undocumented, not-guaranteed-lowest-index tie
    behavior. Device-preserving; no `.item()`; no CPU round-trip."""
    class_count = scores_chw.shape[0]
    max_value = scores_chw.amax(dim=0, keepdim=True)
    is_max = scores_chw == max_value
    class_index = torch.arange(class_count, dtype=torch.int64, device=scores_chw.device).view(class_count, 1, 1)
    sentinel = torch.full((), class_count, dtype=torch.int64, device=scores_chw.device)
    masked = torch.where(is_max, class_index, sentinel)
    return masked.amin(dim=0)


def boundary_8connected(y: torch.Tensor) -> torch.Tensor:
    """B(y) = {x : some IN-IMAGE 8-connected neighbor x' has y(x') != y(x)}.

    The image exterior is never treated as a class: a shifted-in pixel
    with no in-image source contributes nothing (masked out), so an
    edge/corner pixel is a boundary only if an existing neighbor
    differs. Device-preserving; integer label map in, bool mask out."""
    if y.ndim != 2:
        raise DfpPhase1bLightweightBoundaryKernelError(f"label map must be exactly 2-D [H,W], observed {y.ndim}-D")
    H, W = y.shape
    boundary = torch.zeros((H, W), dtype=torch.bool, device=y.device)
    for dr, dc in itertools.product((-1, 0, 1), repeat=2):
        if dr == 0 and dc == 0:
            continue
        r0, r1 = max(0, dr), H + min(0, dr)
        c0, c1 = max(0, dc), W + min(0, dc)
        sr0, sr1 = max(0, -dr), H + min(0, -dr)
        sc0, sc1 = max(0, -dc), W + min(0, -dc)
        if r0 >= r1 or c0 >= c1:
            continue
        differs = y[r0:r1, c0:c1] != y[sr0:sr1, sc0:sc1]
        boundary[r0:r1, c0:c1] |= differs
    return boundary


def chebyshev_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """D_r(mask) = {x : min_{z in mask} ||x-z||_inf <= r}.

    Implemented as a single `max_pool2d` over a (2r+1)x(2r+1) window
    with stride 1 and padding r -- the standard, exact equivalence
    between Chebyshev (square-structuring-element) binary dilation and
    max-pooling; independently proven byte-identical against a direct
    O(r^2)-shift reference in this module's own test suite. radius=0
    is the identity (kernel_size=1)."""
    if radius < 0:
        raise DfpPhase1bLightweightBoundaryKernelError(f"radius must be >= 0, observed {radius}")
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise DfpPhase1bLightweightBoundaryKernelError(f"mask must be a 2-D bool tensor, observed dtype={mask.dtype} ndim={mask.ndim}")
    if radius == 0:
        return mask.clone()
    float_mask = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_size = 2 * radius + 1
    pooled = torch.nn.functional.max_pool2d(float_mask, kernel_size=kernel_size, stride=1, padding=radius)
    return (pooled.squeeze(0).squeeze(0) > 0.5)


@dataclass(frozen=True)
class BalpKernelResult:
    """Every intermediate and final quantity the development-image
    processor (Section B3/B6) needs, computed exactly once. All tensors
    share the input score fields' device."""

    control_labels: Mapping[str, torch.Tensor]          # {"T0"..."T320": [H,W] int64}
    boundary_union: torch.Tensor                          # [H,W] bool
    dilated_boundaries: Mapping[int, torch.Tensor]        # radius -> [H,W] bool
    stability_masks: Mapping[int, torch.Tensor]           # stability_start -> [H,W] bool
    late_flip: torch.Tensor                                # [H,W] bool
    preservation_masks: Mapping[str, torch.Tensor]        # candidate_id -> [H,W] bool
    candidate_labels: Mapping[str, torch.Tensor]          # candidate_id -> [H,W] int64


def compute_balp(scores: Mapping[int, torch.Tensor], *, class_count: int) -> BalpKernelResult:
    """The complete BALP kernel over one image's canonical horizon
    score fields. `scores` must contain exactly the 5 frozen horizons
    {0,16,32,64,320}, each `[class_count, H, W]`, same shape/dtype/
    device, finite. Computes:

    1. y_h = lowest_index_argmax(S_h) for every horizon.
    2. B(y64), B(y320), their union.
    3. Chebyshev dilations of the union at radii [0,1,2,4].
    4. Stability masks for starts [0,16,32,64].
    5. late_flip = [y64 != y320].
    6. All 16 preservation masks P_{s,r} = stable_s AND late_flip AND D_r.
    7. All 16 candidate label maps via y_BALP = where(P, y64, y320) --
       never via a materialized third score tensor.

    Fails closed with DfpPhase1bLightweightBoundaryKernelError on any
    structural violation. Never reads GT; contains no component,
    pullback, masked-forward, or optimization dependency; no learned
    weight or confidence threshold."""
    observed_horizons = frozenset(scores.keys())
    if observed_horizons != frozenset(HORIZON_SET):
        raise DfpPhase1bLightweightBoundaryKernelError(
            f"scores must contain exactly horizons {HORIZON_SET}, observed {sorted(observed_horizons)}"
        )
    class_count = int(class_count)
    if class_count <= 0:
        raise DfpPhase1bLightweightBoundaryKernelError(f"class_count must be positive, observed {class_count}")

    s0 = _require_score_field(scores[0], "S_0", class_count=class_count)
    validated: dict[int, torch.Tensor] = {0: s0}
    for h in HORIZON_SET[1:]:
        t = _require_score_field(scores[h], f"S_{h}", class_count=class_count)
        _require_identical_signature(s0, t, f"S_{h}", "S_0")
        validated[h] = t

    control_labels: dict[str, torch.Tensor] = {}
    for h in HORIZON_SET:
        control_labels[_HORIZON_TO_CONTROL[h]] = lowest_index_argmax(validated[h])

    y64, y320 = control_labels["T64"], control_labels["T320"]
    b64, b320 = boundary_8connected(y64), boundary_8connected(y320)
    boundary_union = b64 | b320

    dilated_boundaries = {r: chebyshev_dilate(boundary_union, r) for r in RADII}

    horizons_le_64 = (0, 16, 32, 64)
    stability_masks: dict[int, torch.Tensor] = {}
    for s in STABILITY_STARTS:
        suffix = [h for h in horizons_le_64 if h >= s]
        ref = control_labels[_HORIZON_TO_CONTROL[suffix[0]]]
        mask = torch.ones_like(ref, dtype=torch.bool)
        for h in suffix[1:]:
            mask &= (control_labels[_HORIZON_TO_CONTROL[h]] == ref)
        stability_masks[s] = mask

    late_flip = y64 != y320

    preservation_masks: dict[str, torch.Tensor] = {}
    candidate_labels: dict[str, torch.Tensor] = {}
    for s in STABILITY_STARTS:
        for r in RADII:
            cid = candidate_id(s, r)
            p = stability_masks[s] & late_flip & dilated_boundaries[r]
            preservation_masks[cid] = p
            candidate_labels[cid] = torch.where(p, y64, y320)

    return BalpKernelResult(
        control_labels=control_labels, boundary_union=boundary_union, dilated_boundaries=dilated_boundaries,
        stability_masks=stability_masks, late_flip=late_flip, preservation_masks=preservation_masks,
        candidate_labels=candidate_labels,
    )
