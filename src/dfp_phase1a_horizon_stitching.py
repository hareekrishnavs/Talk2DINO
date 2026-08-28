"""DFP Phase 1A Step 5: stitch every captured horizon into a canonical
full-image score field, reusing the existing, unmodified authoritative
geometry and per-window conversion primitives.

Traced authoritative path (see ``docs/dfp_phase1a_horizon_stitching.md``
for the full trace and citations):

- Window enumeration/geometry: ``sliding_window_geometry.SlidingWindowPlan``
  (pure, no torch dependency) -- reproduces the legacy clamped
  ``slide_inference`` grid exactly (back-shifted clamp to image bounds,
  row-major flat-index window order). Reused unmodified.
- Per-window raw-score interpolation: ``stitching_control.
  interpolate_raw_scores`` -- reshapes a ``[B, N, C]`` patch-score tensor
  to ``[B, C, grid_h, grid_w]`` and bilinear-interpolates
  (``align_corners=True``) to the window's pixel extent, WITHOUT any
  sigmoid. Reused unmodified.
- Per-pixel stitching weight: ``stitching_control.build_stitch_weight``
  (``kind="uniform"``) -- an all-ones ``[1, 1, h, w]`` weight map, exactly
  DFP's frozen ``averaging="uniform"``/``hann=false`` policy. Reused
  unmodified.
- Accumulation: none of the existing accumulators fit DFP's exact
  contract. ``matched_power_evaluator.stitch_one_image`` and
  ``stitching_control.StitchAccumulator`` are both single-numerator,
  single-denominator-per-variant, and ``StitchingVariantSpec`` forbids a
  score-stage variant that skips the post-stitch sigmoid (DFP's frozen
  ``raw_score_stage="pre_sigmoid_pre_upsample_pre_stitch"`` requires
  exactly that: no sigmoid anywhere, ever). DFP also needs five
  numerators (one per horizon) sharing ONE denominator, computed once per
  window rather than once per horizon -- a shape no existing accumulator
  supports. Per this step's own instructions ("If the existing stitcher
  is too tightly coupled ..., extract only the smallest pure reusable
  accumulation primitive necessary"), this module owns that one new
  primitive: a plain ``numerator[t][region] += crop_t * weight;
  denominator[region] += weight`` loop, mathematically identical to
  ``StitchAccumulator.add_window``'s formula, generalized to several
  numerators over one shared, horizon-independent denominator. Nothing
  about interpolation, geometry, or weighting is reimplemented here.

This module never runs the recurrence, never builds a graph, never loads
a model or dataset, never argmaxes, and never applies a sigmoid -- its
outputs are pre-argmax, pre-sigmoid, linear stitched score fields, ready
for ``src.dfp_phase1a_analysis``'s CPU analysis core once a caller
converts them to the numpy ``[class, height, width]`` layout that module
expects (out of scope here).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from segmentation.evaluation.sliding_window_geometry import (  # noqa: E402
    SlidingWindowGeometryError,
    SlidingWindowPlan,
    SpatialSize,
    WindowGeometry,
)
from models.dinotext.cover_dr.stitching_control import (  # noqa: E402
    StitchingControlError,
    build_stitch_weight,
    interpolate_raw_scores,
)

from src.dfp_phase1a_identity import (  # noqa: E402
    DfpPhase1aIdentityError,
    validate_propagation_block,
    validate_score_interpolation_stitching_block,
)
from src.matched_k11_k12_identity import load_identity as _load_matched_k11_k12_identity  # noqa: E402


class DfpPhase1aHorizonStitchingError(ValueError):
    """Raised on any DFP Phase 1A horizon-stitching invariant violation.
    Always fail closed: never silently sort, reshape, coerce, or repair
    a malformed input."""


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1aHorizonStitchingError(f"{label} must be an exact non-boolean integer")
    return value


def _require_exact_int_pair(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise DfpPhase1aHorizonStitchingError(f"{label} must be an exact 2-tuple")
    a, b = value
    if type(a) is not int or type(b) is not int or a <= 0 or b <= 0:
        raise DfpPhase1aHorizonStitchingError(f"{label} entries must be positive exact integers")
    return (a, b)


# ---------------------------------------------------------------------------
# Identity-bound stitching contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DfpStitchingContract:
    """Independently validated stitching parameters extracted from the
    live protocol identity (plus its matched-k11/k12 parent for
    patch/grid geometry) -- never hardcoded module constants."""

    horizons: tuple[int, ...]
    crop: tuple[int, int]
    stride: tuple[int, int]
    patch_grid: tuple[int, int]
    patch_size: tuple[int, int]
    align_corners: bool


def dfp_stitching_contract_from_protocol(
    protocol_identity: Mapping[str, Any], *, repo_root: Path,
) -> DfpStitchingContract:
    """Strictly extract and validate the stitching-relevant fields of an
    already-loaded protocol identity. Delegates to the single existing
    canonical authorities -- ``src.dfp_phase1a_identity.
    validate_propagation_block`` (horizons), ``validate_score_
    interpolation_stitching_block`` (crop, stride, align_corners,
    averaging/hann/majority_vote/center_selection/sparse_delta_
    restitching, window_enumeration, raw_score_stage,
    identical_across_horizons), and ``src.matched_k11_k12_identity.
    load_identity`` (patch/grid geometry, relationally cross-checked,
    never re-hardcoded as a second copy of 32x32/14x14) -- rather than
    performing merely structural checks on a caller-provided mapping.
    Never assumes ``protocol_identity`` already went through
    ``src.dfp_phase1a_identity.load_identity``."""
    try:
        propagation = protocol_identity["propagation"]
        stitching = protocol_identity["score_interpolation_stitching"]
        validate_propagation_block(propagation)
        validate_score_interpolation_stitching_block(stitching)
    except DfpPhase1aIdentityError as exc:
        raise DfpPhase1aHorizonStitchingError(str(exc)) from exc
    except KeyError as exc:
        raise DfpPhase1aHorizonStitchingError(f"protocol_identity is missing a required section: {exc}") from exc

    horizons = tuple(propagation["horizons"])
    crop = tuple(stitching["crop"])
    stride = tuple(stitching["stride"])
    align_corners = stitching["align_corners"]

    try:
        parents = protocol_identity["parent_identities"]
        matched = _load_matched_k11_k12_identity(
            Path(repo_root) / parents["matched_k11_k12_identity_path"], repo_root=Path(repo_root),
        )
    except KeyError as exc:
        raise DfpPhase1aHorizonStitchingError(f"protocol_identity is missing a required parent-identity key: {exc}") from exc

    matched_geometry = matched["geometry"]
    patch_grid = tuple(matched_geometry["patch_grid"])
    patch_size = tuple(matched_geometry["patch_size"])
    if tuple(matched_geometry["crop"]) != crop:
        raise DfpPhase1aHorizonStitchingError(
            "score_interpolation_stitching.crop disagrees with the live matched_k11_k12 parent identity's geometry.crop"
        )
    if tuple(matched_geometry["stride"]) != stride:
        raise DfpPhase1aHorizonStitchingError(
            "score_interpolation_stitching.stride disagrees with the live matched_k11_k12 parent identity's geometry.stride"
        )
    if matched_geometry["align_corners"] != align_corners:
        raise DfpPhase1aHorizonStitchingError(
            "score_interpolation_stitching.align_corners disagrees with the live matched_k11_k12 parent identity's geometry.align_corners"
        )
    if crop[0] != patch_size[0] * patch_grid[0] or crop[1] != patch_size[1] * patch_grid[1]:
        raise DfpPhase1aHorizonStitchingError("crop must equal patch_size * patch_grid on both axes")

    return DfpStitchingContract(
        horizons=horizons, crop=crop, stride=stride, patch_grid=patch_grid, patch_size=patch_size,
        align_corners=align_corners,
    )


# ---------------------------------------------------------------------------
# Per-window capture record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowHorizonCaptures:
    """One window's captured raw (pre-sigmoid, pre-upsample) patch-score
    tensors, keyed by horizon. ``window_index`` must equal this record's
    position in the sequence passed to
    :func:`stitch_dfp_phase1a_horizons` -- carried explicitly so a
    reordered, duplicated, or skipped window is rejected by comparing
    claimed identity against position, not merely assumed from list
    order. Each tensor has shape ``[N, C]`` (patch-node count, class
    count) -- the same layout ``finite_step_regime.finite_step_
    propagate`` / ``FiniteStepTrace.snapshots`` already use."""

    window_index: int
    captures: Mapping[int, torch.Tensor]


# ---------------------------------------------------------------------------
# Multi-horizon stitching
# ---------------------------------------------------------------------------


def stitch_dfp_phase1a_horizons(
    plan: SlidingWindowPlan,
    window_captures: Sequence[WindowHorizonCaptures],
    *,
    contract: DfpStitchingContract,
    class_count: int,
) -> dict[int, torch.Tensor]:
    """Stitch every window's captured trajectory into ``len(contract.
    horizons)`` full-image score fields, sharing one window plan and one
    denominator across all horizons (``identical_across_horizons=true``).

    For an image with N windows and H horizons: the plan is built by the
    caller (once, never rebuilt here); the denominator receives exactly N
    contributions (once per window, not once per horizon);
    ``interpolate_raw_scores`` is called exactly H*N times; each horizon's
    numerator is accumulated exactly H*N times in total (N per horizon);
    zero recurrence/graph/model calls occur anywhere in this function.

    Returns ``{horizon: tensor}`` with each tensor shape
    ``[1, class_count, image_height, image_width]`` -- pre-sigmoid,
    pre-argmax, linear score fields, never normalized/calibrated/
    thresholded here."""
    if not isinstance(plan, SlidingWindowPlan):
        raise DfpPhase1aHorizonStitchingError("plan must be a SlidingWindowPlan")
    if not isinstance(contract, DfpStitchingContract):
        raise DfpPhase1aHorizonStitchingError("contract must be a DfpStitchingContract")
    class_count = _require_exact_int(class_count, "class_count")
    if class_count <= 0:
        raise DfpPhase1aHorizonStitchingError("class_count must be positive")

    horizons = contract.horizons
    grid_hw = contract.patch_grid
    expected_node_count = grid_hw[0] * grid_hw[1]

    if type(window_captures) not in (list, tuple):
        raise DfpPhase1aHorizonStitchingError(
            f"window_captures must be a list or tuple (observed {type(window_captures).__name__})"
        )
    if len(window_captures) != len(plan.windows):
        raise DfpPhase1aHorizonStitchingError(
            f"window_captures has {len(window_captures)} entries, plan has {len(plan.windows)} windows"
        )

    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for position, record in enumerate(window_captures):
        if not isinstance(record, WindowHorizonCaptures):
            raise DfpPhase1aHorizonStitchingError(f"window_captures[{position}] must be a WindowHorizonCaptures")
        if type(record.window_index) is not int:
            raise DfpPhase1aHorizonStitchingError(f"window_captures[{position}].window_index must be an exact non-boolean integer")
        if record.window_index != position:
            raise DfpPhase1aHorizonStitchingError(
                f"window_captures[{position}].window_index is {record.window_index}, expected {position} "
                "(reordered, duplicated, or missing window)"
            )
        if not isinstance(record.captures, Mapping):
            raise DfpPhase1aHorizonStitchingError(f"window_captures[{position}].captures must be a mapping")
        observed_horizons = set(record.captures)
        for key in observed_horizons:
            if type(key) is not int:
                raise DfpPhase1aHorizonStitchingError(f"window_captures[{position}].captures key {key!r} must be an exact non-boolean integer")
        if observed_horizons != set(horizons):
            missing = sorted(set(horizons) - observed_horizons)
            unknown = sorted(observed_horizons - set(horizons))
            raise DfpPhase1aHorizonStitchingError(
                f"window_captures[{position}] horizon set mismatch: missing={missing}, unknown={unknown}"
            )
        for horizon in horizons:
            tensor = record.captures[horizon]
            label = f"window_captures[{position}].captures[{horizon}]"
            if not torch.is_tensor(tensor):
                raise DfpPhase1aHorizonStitchingError(f"{label} must be a torch.Tensor")
            if tensor.ndim != 2:
                raise DfpPhase1aHorizonStitchingError(f"{label} must be exactly 2-D [N, C], observed {tensor.ndim}-D")
            if tensor.shape[0] != expected_node_count:
                raise DfpPhase1aHorizonStitchingError(
                    f"{label} has {tensor.shape[0]} patch nodes, expected patch_grid product {expected_node_count}"
                )
            if tensor.shape[1] != class_count:
                raise DfpPhase1aHorizonStitchingError(f"{label} has {tensor.shape[1]} classes, expected class_count {class_count}")
            if not tensor.is_floating_point():
                raise DfpPhase1aHorizonStitchingError(f"{label} must be floating point")
            if not torch.isfinite(tensor).all():
                raise DfpPhase1aHorizonStitchingError(f"{label} contains a non-finite value (NaN or Infinity)")
            if dtype is None:
                dtype = tensor.dtype
                device = tensor.device
            elif tensor.dtype != dtype:
                raise DfpPhase1aHorizonStitchingError(f"{label} dtype {tensor.dtype} disagrees with the shared dtype {dtype}")
            elif tensor.device != device:
                raise DfpPhase1aHorizonStitchingError(f"{label} device {tensor.device} disagrees with the shared device {device}")

    assert dtype is not None and device is not None  # at least one window is always present (plan requires >= 1)

    h_img, w_img = plan.image_size.as_tuple()
    numerators = {t: torch.zeros((1, class_count, h_img, w_img), dtype=dtype, device=device) for t in horizons}
    denominator = torch.zeros((1, 1, h_img, w_img), dtype=dtype, device=device)

    for window, record in zip(plan.windows, window_captures):
        crop_hw = window.extent.as_tuple()
        try:
            weight = build_stitch_weight(crop_hw[0], crop_hw[1], kind="uniform", dtype=dtype, device=device)
        except StitchingControlError as exc:
            raise DfpPhase1aHorizonStitchingError(str(exc)) from exc
        accum_rows, accum_cols = window.accumulation_slice
        denominator[:, :, accum_rows, accum_cols] += weight
        for horizon in horizons:
            patch_scores = record.captures[horizon].detach().unsqueeze(0)  # [1, N, C], never retains autograd
            try:
                crop = interpolate_raw_scores(patch_scores, grid_hw, crop_hw)  # [1, C, crop_h, crop_w]
            except StitchingControlError as exc:
                raise DfpPhase1aHorizonStitchingError(str(exc)) from exc
            numerators[horizon][:, :, accum_rows, accum_cols] += crop * weight

    if not torch.isfinite(denominator).all() or not torch.all(denominator > 0):
        raise DfpPhase1aHorizonStitchingError("stitching left one or more pixels with non-positive denominator coverage")

    result: dict[int, torch.Tensor] = {}
    for horizon in horizons:
        stitched = numerators[horizon] / denominator
        if not torch.isfinite(stitched).all():
            raise DfpPhase1aHorizonStitchingError(f"stitched horizon {horizon} contains a non-finite value")
        result[horizon] = stitched
    return result


__all__ = [
    "DfpPhase1aHorizonStitchingError",
    "DfpStitchingContract",
    "WindowHorizonCaptures",
    "dfp_stitching_contract_from_protocol",
    "stitch_dfp_phase1a_horizons",
    "SlidingWindowGeometryError",
    "SlidingWindowPlan",
    "SpatialSize",
    "WindowGeometry",
]
