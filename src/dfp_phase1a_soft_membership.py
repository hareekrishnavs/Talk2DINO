"""DFP Phase 1A Step 8: soft source memberships.

**Duplication gate.** Step 7 (``src.dfp_phase1a_stitching_pullback``)
already implements, tests, documents, and independently verifies the
soft-membership formula itself:

    numerator[r,w,j] = (A_w^T 1_{Omega_r})[j]
    q[w,j]           = (A_w^T 1)[j]
    M[r,w,j]         = numerator[r,w,j] / q[w,j]

via ``compute_window_soft_memberships`` and ``iter_image_soft_
memberships`` -- including conservation (``sum_r numerator == q``,
``sum_r M == 1``), non-negativity, no clipping/renormalization,
chunking (sizes 1/2/3/non-divisor/R/>R all numerically equivalent),
component-chunk-bounded memory (no dense pixel-by-patch matrix, no all-R
masks simultaneously), geometry-only ``q`` reused once per window across
every component chunk, strict validation, immutability, and freedom
from ground truth or any other forbidden compute. This module does not
duplicate, rename, move, or rewrite any of that -- it imports and calls
Step 7's functions directly for every numerator/q/M computation.

**What Step 7 does not (yet) provide**, closed here as narrow additive
wrappers, never touching ``src/dfp_phase1a_stitching_pullback.py``:

1. Section 4 asks a streamed/chunked API to emit self-describing
   records carrying at least the window index, component start/stop
   index, the membership block, and a geometry fingerprint or plan
   identity. Step 7's ``iter_image_soft_memberships`` yields a bare
   ``(start, {window_index: tensor})`` pair -- correct, but not
   self-describing. ``iter_soft_membership_records`` wraps it,
   unchanged in every numeric respect, into ``MembershipChunkRecord``
   instances.
2. Section 7 requires stale or mismatched geometry fingerprints to be
   rejected *before* the first membership chunk is generated. Step 7
   computes ``StitchingPullbackPlan.geometry_fingerprint`` as a stored
   property but never compares it against anything -- there is no
   active enforcement path. ``iter_soft_membership_records`` accepts an
   optional ``expected_geometry_fingerprint`` and validates it eagerly
   (before the returned generator's first item, matching the same
   eager-validation contract Step 7's own ``iter_image_soft_
   memberships`` already uses for its other parameters).
3. Section 8 requires the raw source mass ``m_r`` to accumulate in
   float64 specifically (a numerical-stability requirement independent
   of whatever working dtype the pullback plan happens to use). Step
   7's ``component_source_mass`` sums in the *plan's* dtype, which is
   float64 by the module's own default but not guaranteed to be so.
   ``component_source_mass_float64`` upcasts before accumulating,
   guaranteeing float64 regardless of the plan's dtype -- still calling
   no interpolation or pullback math of its own.

No other Step 8 requirement needed new code: the formula, conservation,
chunking, memory bound, ``q`` reuse, strict validation, immutability,
and no-leakage requirements are already met -- and already
independently verified -- by the committed Step 7 implementation and
test suite (see ``docs/dfp_phase1a_stitching_pullback.md`` for the full
account; this module adds no duplicate documentation of that material).

**No real Phase 1A dataset diagnostic has been run by this step.**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from src.dfp_phase1a_stitching_pullback import (
    DfpPhase1aStitchingPullbackError,
    StitchingPullbackPlan,
    iter_image_soft_memberships,
)


class DfpPhase1aSoftMembershipError(DfpPhase1aStitchingPullbackError):
    """Raised on a Step 8 soft-membership-streaming invariant violation
    -- a subclass of Step 7's own domain error, since this module is a
    thin wrapper over Step 7's pullback machinery and never introduces a
    second, unrelated failure taxonomy. Always fail closed: never sort,
    coerce, clamp, merge, or repair a malformed input."""


@dataclass(frozen=True)
class MembershipChunkRecord:
    """One self-describing membership chunk for one canonical window:
    exactly what section 4 requires a streamed/chunked API to carry.
    ``membership`` has shape ``[component_id_stop - component_id_start,
    N_w]`` and is exactly the tensor Step 7's ``compute_window_soft_
    memberships`` returns for this (window, chunk) pair -- never
    recomputed, reshaped, or renormalized here."""

    window_index: int
    component_id_start: int
    component_id_stop: int
    membership: torch.Tensor
    geometry_fingerprint: str


def iter_soft_membership_records(
    component_ids: np.ndarray, component_count: int, *, window_plan: StitchingPullbackPlan, component_chunk_size: int,
    expected_geometry_fingerprint: str | None = None,
) -> Iterator[MembershipChunkRecord]:
    """Wrap Step 7's ``iter_image_soft_memberships`` to (a) reject a
    stale or mismatched ``window_plan.geometry_fingerprint`` before the
    first chunk is ever generated, and (b) re-package each ``(start,
    {window_index: tensor})`` pair into self-describing
    ``MembershipChunkRecord`` instances, one per window, in Step 5's
    canonical window order. Computes no numerator, no q, and no
    membership value of its own -- every number this yields comes
    directly from Step 7's own ``compute_window_soft_memberships``.

    Validation (including the fingerprint check) runs eagerly, exactly
    like Step 7's own functions: this is a plain function that
    validates then returns a generator, never a generator function
    itself."""
    if not isinstance(window_plan, StitchingPullbackPlan):
        raise DfpPhase1aSoftMembershipError("window_plan must be a StitchingPullbackPlan")
    if expected_geometry_fingerprint is not None:
        if type(expected_geometry_fingerprint) is not str:
            raise DfpPhase1aSoftMembershipError("expected_geometry_fingerprint must be an exact str")
        if window_plan.geometry_fingerprint != expected_geometry_fingerprint:
            raise DfpPhase1aSoftMembershipError(
                "window_plan.geometry_fingerprint disagrees with expected_geometry_fingerprint -- "
                "this pullback plan is stale or was built for a different image geometry"
            )

    try:
        body = iter_image_soft_memberships(
            component_ids, component_count, window_plan=window_plan, component_chunk_size=component_chunk_size,
        )
    except DfpPhase1aStitchingPullbackError as exc:
        raise DfpPhase1aSoftMembershipError(str(exc)) from exc

    return _records_from_body(body, window_plan.geometry_fingerprint)


def _records_from_body(
    body: Iterator[tuple[int, dict[int, torch.Tensor]]], geometry_fingerprint: str,
) -> Iterator[MembershipChunkRecord]:
    for start, chunk in body:
        for window_index in sorted(chunk):
            membership = chunk[window_index]
            yield MembershipChunkRecord(
                window_index=window_index, component_id_start=start, component_id_stop=start + membership.shape[0],
                membership=membership, geometry_fingerprint=geometry_fingerprint,
            )


def component_source_mass_float64(chunk_memberships: dict[int, torch.Tensor]) -> torch.Tensor:
    """``m_r = sum_{w,j} M[r,w,j]`` for one component chunk, accumulated
    in float64 regardless of the pullback plan's own working dtype --
    the one Step 7 gap section 8 requires closing. Never rounds,
    thresholds, or reinterprets the result as a pixel or
    patch-equivalent area; never used here to redefine or gate a
    component."""
    if not isinstance(chunk_memberships, dict) or not chunk_memberships:
        raise DfpPhase1aSoftMembershipError("chunk_memberships must be a non-empty dict")
    totals: torch.Tensor | None = None
    for tensor in chunk_memberships.values():
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise DfpPhase1aSoftMembershipError("each chunk_memberships value must be a 2-D torch.Tensor [K, N]")
        window_total = tensor.to(torch.float64).sum(dim=1)
        totals = window_total if totals is None else totals + window_total
    return totals.detach()


__all__ = [
    "DfpPhase1aSoftMembershipError",
    "MembershipChunkRecord",
    "iter_soft_membership_records",
    "component_source_mass_float64",
]
