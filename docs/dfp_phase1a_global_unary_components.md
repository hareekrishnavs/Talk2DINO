# DFP Phase 1A — Step 6: global unary components

`src/dfp_phase1a_global_unary_components.py` converts one image's
canonical stitched unary score field `S0` (Step 5's output) into a
deterministic, complete, disjoint global partition of the image domain
into maximal 8-connected same-unary-class components -- the structural
foundation the later soft-membership pullback consumes.

## Why components come from stitched S0

The frozen protocol identity's `[structural_components]` pins
`component_source = "canonical_stitched_unary_(T=0)_field"` and
`component_scope = "global_across_the_full_stitched_image_never_per_
sliding_window"`. Components are a property of the final stitched
image, never of any one sliding window: this module accepts only a full
`[class, height, width]` array (Step 5's `S0`, converted by the caller
to numpy) and never a per-window score tensor.

## The lowest-index tie rule

`[structural_components].tie_rule = "lowest_class_index"` is implemented
by reusing `src.dfp_phase1a_analysis.derive_horizon_labels` -- Step 3's
own explicit (never incidental-`argmax`) lowest-index tie-break, already
verified and used for T=0/T=320 dispute labels. This module adds no
second tie-break implementation; `derive_horizon_labels({0: S0})[0]`
*is* the unary label map.

## Global final-pixel coordinate system, 8-connectivity

Components are indexed in the exact pixel grid of `S0` -- the full
original-image domain Step 5 already restores exactly (no padding
beyond the image is ever part of the domain, since Step 5 returns the
precise original-image shape). `primary_connectivity = 8` (horizontal,
vertical, and both diagonal neighbors) is the *only* connectivity this
module produces; `sensitivity_connectivity = 4` is a different, later
diagnostic this module never computes.

## Deterministic component-ID ordering

Per-class labeling uses `scipy.ndimage.label` with an 8-connected
structuring element (one label pass per class *present* in the image,
never per window, never per pixel pair) -- the same connected-component
primitive `src.dfp_phase1a_analysis.compute_reversion_coherence` already
uses. scipy's own per-class label numbers are never treated as an
authority: every raw label is remapped to a canonical global ID by
sorting on the component's minimum row-major flat pixel index
(`np.unique(..., return_index=True)`'s documented first-occurrence
semantics), assigning `0..R-1` in ascending order. This ordering is
independent of `PYTHONHASHSEED`, per-class enumeration order, and array
contiguity -- verified directly by subprocess probes across multiple
hash seeds.

## Output fields and units

`GlobalUnaryComponents` (all array fields frozen read-only on
construction):

| field | shape | dtype | meaning |
|---|---|---|---|
| `unary_labels` | `[H,W]` | `int64` | lowest-index unary class per pixel |
| `component_ids` | `[H,W]` | `int64` | canonical global component id, `0..R-1` |
| `component_class_indices` | `[R]` | `int64` | unary class of each component |
| `component_pixel_areas` | `[R]` | `int64` | `\|Ω_r\|`, exact pixel count |
| `component_patch_equivalent_areas` | `[R]` | `float64` | `\|Ω_r\| / (patch_pitch_h * patch_pitch_w)` |
| `component_first_flat_indices` | `[R]` | `int64` | the ordering key itself |
| `component_count` | scalar `int` | -- | `R` |
| `per_class_component_counts` | `[C]` | `int64` | components per class, sums to `R` |

## Pixel area vs. patch-equivalent area

`[structural_component_area].formula = "area_patch_equivalent =
component_pixel_area / projected_patch_pitch_area"` is implemented
exactly, with no rounding, flooring, ceiling, or clamp. The projected
patch pitch is never re-derived here: it is reused, unmodified, from
`src.dfp_phase1a_horizon_stitching.dfp_stitching_contract_from_protocol`'s
own `patch_size` -- itself relationally bound to the matched-k11/k12
parent identity's live geometry (`14x14` pixels, `crop=448=patch_size*
patch_grid`). This is *not* the later soft source mass `m_r`, which
depends on window-patch memberships and is out of this step's scope.

## No ground truth, no ignore mask

Neither `[structural_components]` nor `[structural_component_area]`
references ground truth, an ignore index, or any annotation. This
module's entire public API accepts no such parameter under any name
(verified by a test that inspects every public function's signature).
Ground-truth ignore pixels are excluded later, during metric
computation -- never during this step's component construction.

## Partition invariants

`validate_global_unary_components_partition` independently re-checks,
without mutating its input: exact output keys/dtypes/shapes; contiguous
`0..R-1` component ids, each appearing at least once; pixel areas
reconciled against a direct recount of `component_ids`; every
component's class agrees with all of its pixels in `unary_labels`;
`component_first_flat_indices` reconciled against a direct
`np.argmax`-based reconstruction and strictly ascending; per-class
counts sum to `R`; total pixel area equals `H*W`; patch-equivalent areas
reconcile against pixel areas under one consistent patch pitch (and,
when a contract is supplied, against that contract's own authoritative
pitch). `compute_global_unary_components` calls this validator on its
own output before returning, as a self-check.

## Computational complexity

One `derive_horizon_labels` call, one `ndimage.label` call per class
*present* in the image (never per window, never per component, never a
dense pixel-by-pixel adjacency matrix or one full-size boolean mask per
component held simultaneously). Memory is proportional to image pixels
plus `O(R)` component metadata; `component_ids` is the sole compact
representation of the partition -- no other module holds a dense
per-component mask.

## Excluded from this step

This module returns a component partition only. It does not implement
soft source membership `M_{r,wj}`, the stitching-adjoint pullback
`A_w^T`, source mass `m_r`, supercohorts, masked forward propagation,
component evidence `G_{xr}`, k-star or phi, eta-net, horizon-persistence
analysis, oracle execution, Boundary IoU, artifact/checkpoint schemas, a
real-data driver, SLURM scripts, GPU work, or the final DFP selector.
**No Phase 1A dataset diagnostic has been run by this step.**
