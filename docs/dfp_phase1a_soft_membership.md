# DFP Phase 1A — Step 8: soft source memberships

The soft-membership formula, conservation identities, non-negativity,
no-clipping/no-renormalization guarantee, component chunking, geometry-
only `q` reuse, matrix-free memory bound, strict validation,
immutability, and ground-truth isolation are all already implemented,
tested, and documented by **Step 7**
(`src/dfp_phase1a_stitching_pullback.py`,
`docs/dfp_phase1a_stitching_pullback.md`). This document does not
repeat that material -- see Step 7's documentation for the authoritative
account of:

```
numerator[r,w,j] = (A_w^T 1_{Omega_r})[j]
q[w,j]           = (A_w^T 1)[j]
M[r,w,j]         = numerator[r,w,j] / q[w,j]
```

`src/dfp_phase1a_soft_membership.py` is a thin additive wrapper closing
three narrow gaps between Step 7's raw API and Step 8's own task text,
none of which touch Step 7's file or duplicate any of its computation:

## Self-describing chunk records

Step 7's `iter_image_soft_memberships` yields a bare `(component_id_
start, {window_index: membership_tensor})` pair. `iter_soft_membership_
records` wraps it, unchanged in every numeric respect, into
`MembershipChunkRecord(window_index, component_id_start,
component_id_stop, membership, geometry_fingerprint)` -- one record per
(chunk, window) pair, self-describing without requiring the caller to
retain a separate reference to the plan.

## Active geometry-fingerprint staleness rejection

Step 7 computes and stores `StitchingPullbackPlan.geometry_fingerprint`
but never compares it against anything. `iter_soft_membership_records`
accepts an optional `expected_geometry_fingerprint` and rejects a
mismatch **before** the first membership chunk is generated (zero
pullback/adjoint calls on rejection) -- closing the gap between "a
fingerprint exists" and "a stale plan is actively rejected."

## Float64 source-mass accumulation

Step 7's `component_source_mass` sums in whatever dtype the pullback
plan happens to use. `component_source_mass_float64` upcasts each
window's membership block to float64 before accumulating, guaranteeing
the numerical-stability property Step 8's own text requires
independently of the plan's working dtype. It remains, exactly as Step
7 already documents, a diagnostic-only quantity
(`source_copy_mass_is_diagnostic_only`, a frozen identity invariant):
never rounded, never thresholded, never reinterpreted as Step 6's pixel
area `|Ω_r|` or patch-equivalent area `|Ω_r|/p²`, and never used here to
redefine or gate a component.

## No new documentation elsewhere

Everything else this step's task text asks to be documented --
axis ordering, chunking mechanics, the conservation identities, the
fractional-boundary behavior of a patch straddling two components, the
absence of clipping/renormalization, the matrix-free memory bound, and
the absence of ground truth anywhere in the computation -- is already
accurate in `docs/dfp_phase1a_stitching_pullback.md` and is not repeated
here. **No real Phase 1A dataset diagnostic has been run by this
step.**
