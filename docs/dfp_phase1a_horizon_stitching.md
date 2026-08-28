# DFP Phase 1A — Step 5: canonical multi-horizon stitching

`src/dfp_phase1a_horizon_stitching.py` converts one image's per-window
captured trajectories at `T = 0, 16, 32, 64, 320` (from Step 4's
`capture_dfp_phase1a_trajectory`) into five full-image, pre-argmax,
pre-sigmoid stitched score fields `S0, S16, S32, S64, S320`, using
exactly the interpolation, window placement, accumulation order, overlap
weighting, denominator, and edge-window behavior the existing evaluator
already uses -- adding no new mathematics of its own beyond one small,
genuinely new accumulation primitive (see below).

## Authoritative stitching source

Traced from the live production code, not inferred from a generic
interpolation recipe:

- **Window enumeration/geometry**: `segmentation.evaluation.
  sliding_window_geometry.SlidingWindowPlan.build` -- pure, no torch
  dependency, reproduces the legacy `slide_inference` grid exactly
  (`grid = max(img - crop + stride - 1, 0) // stride + 1`, back-shifted
  clamp to image bounds, row-major flat-index window order). Reused
  unmodified.
- **Per-window raw-score interpolation**: `models.dinotext.cover_dr.
  stitching_control.interpolate_raw_scores` -- reshapes a `[B, N, C]`
  patch-score tensor to `[B, C, grid_h, grid_w]` and bilinear-interpolates
  (`align_corners=True`) to the window's pixel extent, deliberately
  *without* any sigmoid (unlike the probability-stage path, which
  sigmoids before interpolating). Reused unmodified.
- **Per-pixel weight**: `stitching_control.build_stitch_weight(kind=
  "uniform")` -- an all-ones `[1, 1, h, w]` map, matching DFP's frozen
  `averaging="uniform"` / `hann=false` policy exactly. Reused unmodified.
- **Identity source**: `evaluation_identities/e13_dfp_phase1a_protocol.
  toml`'s `[score_interpolation_stitching]` (`raw_score_stage`, `crop`,
  `stride`, `align_corners`, `averaging`, `window_enumeration`,
  `clamping_policy`, `hann`/`majority_vote`/`center_selection`/
  `sparse_delta_restitching`, `identical_across_horizons`) plus its
  `matched_k11_k12` parent identity's `[geometry]` (`patch_grid`,
  `patch_size`), cross-checked relationally.

## Why one new accumulation primitive was necessary

Neither existing accumulator fits DFP's exact contract. Both
`matched_power_evaluator.stitch_one_image` and `stitching_control.
StitchAccumulator` are single-numerator, single-denominator-per-variant.
`StitchingVariantSpec` additionally *forbids* a score-stage variant that
skips the post-stitch sigmoid -- but DFP's frozen
`raw_score_stage = "pre_sigmoid_pre_upsample_pre_stitch"` requires
exactly that: no sigmoid anywhere, ever, at any horizon. DFP also needs
five numerators (one per horizon) sharing *one* denominator, computed
once per window rather than once per horizon -- a shape no existing
accumulator supports.

Per this step's own instructions ("if the existing stitcher is too
tightly coupled ..., extract only the smallest pure reusable accumulation
primitive necessary"), this module owns exactly that one new primitive:
a plain `numerator[t][region] += crop_t * weight; denominator[region] +=
weight` loop, mathematically identical to `StitchAccumulator.
add_window`'s formula, generalized to several numerators over one shared,
horizon-independent denominator. Interpolation, geometry, and weighting
are never reimplemented -- only reused.

## The stitching equation

For each horizon `t` and output pixel `(x, c)`:

```
S_t(x, c) = sum_w W_w(x) * F_w(Y_w^t)(x, c)  /  sum_w W_w(x)
```

where `Y_w^t` is window `w`'s captured `[N, C]` score tensor at horizon
`t`, `F_w` is `interpolate_raw_scores` (patch-grid reshape + bilinear
`align_corners=True` interpolation to window `w`'s pixel extent), `W_w`
is `build_stitch_weight(kind="uniform")` (all ones), and the sum follows
the canonical row-major window order. `W_w` and the denominator
`sum_w W_w(x)` do not depend on `t` -- computed once, shared by all five
horizons (`identical_across_horizons=true` in the frozen identity).

## Shared geometry, one denominator

For N windows and H=5 horizons: the window plan is built once by the
caller (never rebuilt inside this module); the denominator receives
exactly N contributions (once per window, never once per horizon);
`interpolate_raw_scores` is called exactly H*N times; each horizon's
numerator is accumulated N times (H*N in total); zero recurrence, graph,
or model calls occur anywhere in this module.

## Endpoint invariance (T=0, T=320)

`torch.sigmoid(S0)` and `torch.sigmoid(S320)` are exactly (bit-for-bit)
equal to what the existing production `StitchAccumulator`/`UNIFORM_SCORE`
accumulator produces for the same window captures -- proven in
`tests/test_dfp_phase1a_horizon_stitching.py`. The only difference is
*where* sigmoid is applied (never, here; once, after stitching, there);
the interpolation, weighting, and accumulation arithmetic are identical,
so the pre-sigmoid stitched value is provably identical and sigmoiding
it reproduces the existing evaluator's finalized output exactly.

## Flush-edge handling

`SlidingWindowPlan.build`'s back-shifted clamp already guarantees every
window's accumulation slice reaches the true image boundary (`y0 = max(
y_end - h_crop, 0)`); this module does nothing extra for edge windows --
it accumulates into `window.accumulation_slice` exactly as returned by
the plan, for every window including the flush-right/flush-bottom/
flush-both corner case and the smaller-than-nominal-crop case.

## Strict input contract

Before any interpolation or accumulation: `window_captures` must be a
`list`/`tuple` of `WindowHorizonCaptures` whose length and per-entry
`window_index` exactly match the plan's window count and position (a
reordered, duplicated, missing, or extra window is rejected before the
first interpolation call); every window's `captures` dict must carry
exactly the contract's horizon set; every tensor must be exactly 2-D
`[N, C]` with `N` equal to `patch_grid[0]*patch_grid[1]` and `C` equal to
the caller's `class_count`; dtype and device must be identical across
every window and horizon; every value must be finite. Nothing is sorted,
reshaped, coerced, or silently repaired -- any violation raises
`DfpPhase1aHorizonStitchingError` before the first `interpolate_raw_
scores` call. `KeyboardInterrupt`/`SystemExit`/`MemoryError` are never
caught. Outputs are detached (no autograd graph retained), never alias
an input or another horizon, and mutating one horizon's output cannot
alter another's.

## What this step deliberately excludes

This module produces linear, pre-argmax, pre-sigmoid stitched score
fields only. It does not run `argmax`, V20/V21 remapping,
background-threshold insertion or cleaning, ignore masking, oracle
construction, horizon-persistence decisions, component labeling,
morphology, Boundary IoU, or metric accumulation -- ground truth is never
an input to this module. It does not run a real model or dataset driver,
does not implement artifact/checkpoint/resume schemas, component
extraction, soft source memberships, supercohort or masked propagation,
oracle execution, the final DFP selector, DenseCRF/PAMR/re-crop
baselines, or any SLURM/GPU execution. **This is infrastructure only: no
Phase 1A dataset result has been produced by this step.** The next step
consumes these stitched fields (converted to the numpy `[class, height,
width]` layout `src.dfp_phase1a_analysis.validate_stitched_inputs`
expects) as input to the existing CPU analysis core, which is out of
this step's scope.
