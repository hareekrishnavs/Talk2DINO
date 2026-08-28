# DFP Phase 1A — Step 7: canonical stitching pullback and soft source memberships

`src/dfp_phase1a_stitching_pullback.py` computes the exact pullback
(adjoint) of Step 5's canonical stitching operator, and uses it to
compute the soft source membership of every window-patch copy in a
Step 6 global unary component.

## The exact forward operator `A_w`

Traced fresh from `src.dfp_phase1a_horizon_stitching.
stitch_dfp_phase1a_horizons` (never re-derived generically): for window
`w` and a class-agnostic scalar patch field `u` of shape `[N]`
(`N = patch_grid[0] * patch_grid[1]`),

```
A_w(u)(x) = interpolate_raw_scores(u, grid_hw, crop_hw)[local(x)]
            * weight[local(x)] / denominator(x)     for x in window w's accumulation_slice
          = 0                                        otherwise
```

`interpolate_raw_scores` and `build_stitch_weight` are reused
**unmodified** from Step 5. `denominator` is Step 5's own shared
per-pixel stitching denominator, reconstructed once per image geometry
via the identical accumulation Step 5 itself performs (it is not
exposed as a standalone function by Step 5, so this module recomputes it
using the exact same primitive and accumulation order).

## Why `A_w` is linear

Every stage is linear in `u`: bilinear interpolation
(`F.interpolate(mode="bilinear", align_corners=True)`) is a fixed convex
combination of at most 4 input values per output pixel, with weights
depending only on geometry; elementwise multiplication by `weight` and
division by `denominator` are diagonal (self-adjoint) linear maps, since
neither depends on `u`'s values; scattering into the window's region
(zero elsewhere) is a fixed linear embedding. No sigmoid, softmax,
argmax, threshold, or calibration appears anywhere -- Step 5's frozen
`raw_score_stage="pre_sigmoid_pre_upsample_pre_stitch"` guarantees this.

## The pullback

`A_w^T` is obtained as the exact vector-Jacobian product of
`interpolate_raw_scores` itself: for a pixel-space vector `v` restricted
to window `w`'s region and scaled by `weight/denominator` (self-adjoint,
so applied identically in both directions), a single
`torch.autograd.grad` call against a fresh, disconnected zero probe
tensor yields `A_w^T(v)` exactly -- no model graph, no retained graph
after the call, no parameter gradients (the probe is never a model
parameter). Because bilinear interpolation is genuinely linear, autograd
computes the *analytic*, exact adjoint of the literal function Step 5
calls -- not an approximation of a hand-derived formula. This is
independently verified (never by reusing this same pullback for both
sides of a check) via a tiny dense matrix built only from repeated
*forward* calls on patch-basis vectors.

## Soft membership formula

For component `Ω_r` (from Step 6's `component_ids`), window `w`, patch
node `j`:

```
M[r,w,j] = (A_w^T 1_{Ω_r})[j] / (A_w^T 1)[j]
```

## Axis order

- **Component axis**: Step 6's canonical `component_ids` order (`0..R-1`
  by ascending first row-major pixel index) -- never reordered, merged,
  or dropped.
- **Window axis**: Step 5's canonical `SlidingWindowPlan` row-major
  window order.
- **Patch-node axis**: the canonical patch-grid flattening
  (`row * grid_width + col`), identical to Step 4/5's own convention.

## Geometry-only `q`

`q[w,j] = (A_w^T 1)[j]` depends only on window geometry -- never on
horizon scores, class channels, or component content
(`structural_components.a_transpose_ones_cached_invariant ==
"geometry_only"`, a frozen identity invariant). It is computed exactly
once per `(image_size, contract)` pair inside `StitchingPullbackPlan`
and reused for every component chunk; two images sharing the same
geometry produce byte-identical `q`.

## No dense `A_w`

This module never materializes an array of shape `[image_pixels,
window_patch_nodes]` or any equivalent. Component indicator fields are
built one chunk at a time (`[chunk_size, H, W]`, never all `R`
components simultaneously); the bilinear-interpolation adjoint is
obtained via a `[1, N, chunk_size]` autograd probe, never a materialized
interpolation matrix. A dense reference matrix is constructed **only**
in tests, from repeated forward calls, and is never reachable from
production code.

## Chunking

`compute_window_soft_memberships`/`iter_image_soft_memberships` process
components in a caller-chosen `component_chunk_size` (an engineering
control, never a scientific parameter). Peak working memory for one
chunk is proportional to `chunk_size × image_pixels +
chunk_size × window_patch_nodes` plus fixed geometry metadata. Different
valid chunk sizes produce identical component order and numerically
equivalent memberships (verified for sizes 1, 2, a non-divisor of `R`,
`R` itself, and larger than `R`).

## Conservation identities and residual reporting

`sum_r numerator[r,w,j] == q[w,j]` and `sum_r M[r,w,j] == 1` hold up to
floating-point residual. Because a single component chunk is not
expected to sum to 1 by itself, conservation is checked by accumulating
column sums across every chunk (`accumulate_membership_column_sums`)
and validating the accumulated total once
(`validate_conservation`) -- this never requires materializing all `R`
components' memberships simultaneously. No runtime conservation
tolerance is invented: `validate_conservation` returns full-precision
residual diagnostics and never renormalizes, clips, or silently repairs
a failed check. `component_source_mass` (`m_r = sum_{w,j} M[r,w,j]`) is
exposed only as a diagnostic -- `source_copy_mass_is_diagnostic_only` is
a frozen identity invariant, and this step never thresholds, interprets,
or gates on it.

## Flush-edge handling

`SlidingWindowPlan`'s back-shifted clamp (reused, unmodified) already
guarantees every window's `accumulation_slice` reaches the true image
boundary with no gap and no out-of-bounds contribution. The pullback
uses `window.accumulation_slice` exactly as the plan returns it for
every window, including flush-right, flush-bottom, and flush-both
corner cases -- no window or patch is ever snapped to a new lattice.

## No ground truth, class-agnostic geometry

This module accepts no ground truth, ignore mask, validation annotation,
class name, text embedding, or oracle category anywhere in its API. It
never calls a model, backbone, graph builder, the recurrence, horizon
capture, a dataset loader, a metric, or Boundary IoU. The pullback
operates purely on Step 6's `component_ids` (geometry-derived, never
GT-derived) and window/patch geometry.

## Excluded from this step

No masked recurrence, no component-propagated score field, no signed
evidence `G_xr`, no semantic supercohort propagation, no k-star/phi, no
eta-net analysis, no Gate 1 oracle execution, no Gate 2 AUROC/AUPRC, no
final selector, no real-data artifact/checkpoint schema, no real-data
driver, no SLURM scripts, no GPU execution. **No real Phase 1A dataset
diagnostic has been run by this step.**
