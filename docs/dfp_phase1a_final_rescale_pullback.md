# DFP Phase 1A — final-resolution analysis authority and composed final-rescale pullback

`src/dfp_phase1a_final_rescale_pullback.py` implements the composed
operator every downstream Phase 1A analysis (dispute, oracle,
persistence, confusion, global unary components, reversion coherence)
actually consumes, and freezes, additively, the authority that pins
which resolution that analysis happens in and exactly which rescale
operation it goes through first. **No real diagnostic run has occurred
under this authority.** The still-partial execution driver
(`src/dfp_phase1a_execute.py`) is unaffected by this module and remains
its own, separately blocked task.

## The final-resolution analysis domain

`evaluation_identities/e13_dfp_phase1a_protocol.toml`'s additive
`[final_resolution_analysis]` block freezes the canonical Phase 1A
analysis domain: `final_evaluator_output_at_ori_shape` — the ORIGINAL
image's own resolution (`ori_shape`), sourced exclusively from
authoritative image/pipeline metadata (`img_metas['ori_shape']`) and
**never** from ground truth. Every T0/T16/T32/T64/T320 stitched score
field is rescaled into this domain before argmax, dispute
classification, or component construction ever runs; ground truth is
required to already match this domain (checked, never silently
resized) and is read only after every prediction field is finalized.

The bound rescale operator `R` is `matched_power_evaluator.
finalize_prediction`'s own crop-then-bilinear-resize (file-content
SHA-256-pinned in the identity, cross-checked against the real bytes
on disk every time the identity loads): crop to `img_metas['img_shape']`
(removes sliding-window padding — a no-op in this pipeline, which has
no Pad transform), then `F.interpolate(mode="bilinear",
align_corners=True, antialias=False)` to `ori_shape`, strictly BEFORE
argmax. There is no alternate resize path.

## The composed operator

Calling Step 5's window-stitching operator `A_stitch` (patch sources →
stitched inference-resolution pixels) and the bound rescale `R`
(stitched inference-resolution pixels → final `ori_shape` pixels), every
downstream analysis needs the COMPOSITION:

```
A_final = R  composed with  A_stitch
A_final^T = A_stitch^T  composed with  R^T
```

`apply_final_rescale_forward` implements `R`, byte-identical to
`finalize_prediction`'s own formula (minus its trailing argmax — this
module needs linear scores). `apply_final_rescale_adjoint` implements
the EXACT `R^T` — Crop's adjoint (a trivial zero-pad scatter) composed
with the bilinear resize's own adjoint, computed via a local,
gradient-isolated `torch.autograd.grad` probe (see "Inference-context
safety" below) — never approximated, never a dense matrix.
`apply_final_window_stitching_pullback` composes `R^T` with Step 7's
own, completely unmodified `apply_window_stitching_pullback` to get
`A_final^T` for one window. `FinalStitchingPullbackPlan` precomputes
each window's geometry-only `q_final[w] = A_final^T @ 1` once per image,
exactly mirroring Step 7's own `per_window_q`, but through the composed
operator.

Global unary components (Step 6) are constructed directly from the
FINAL `ori_shape` stitched T0 field — never the raw, pre-rescale Step-5
output, and never resized afterward. Component IDs and masks are never
resized or nearest-neighbor-reconciled anywhere in this module (an
identity requirement, and statically proven by
`tests/test_dfp_phase1a_final_rescale_pullback.py::
test_no_nearest_neighbor_interpolation_in_module_source`). Soft
membership (`compute_window_final_soft_memberships` /
`iter_image_final_soft_memberships`) consumes those SAME
final-resolution component IDs directly, through the composed
`A_final^T` — exact partition-of-unity (`sum_r M[r,w,j] == 1`) and
source-mass conservation, never an approximation.

## Reversion coherence: T320 baseline, T0 fallback, `<20`-pixel threshold

`[reversion_coherence]` freezes the one diagnostic reverted-pixel-set
structural analysis (component count/sizes/perimeter/small-component
mass fraction) the per-image-stats-v2 schema's `reversion_component_*`
members require:

- **Baseline prediction: T320.** **Fallback prediction: T0.**
- The reversion set is EXACTLY the already-frozen
  `[dispute_categories].harmful_washout` definition, reused verbatim
  (relationally cross-checked, never reworded): `GT(x) == a(x) and
  GT(x) != b(x)` (`a` = T0, `b` = T320), evaluated in the final
  `ori_shape` domain, ignore pixels excluded (inherited from
  `[dispute_protocol].dispute_set_definition`'s own "non-ignore
  evaluation pixels" restriction — harmful-washout is a strict
  subcategory of that dispute set). Beneficial-correction and
  third-label-transition pixels are never part of this set.
- Connectivity: exactly 8 (relationally cross-checked against
  `[structural_components].primary_connectivity`).
- Component-size unit: pixel area, in the SAME final `ori_shape`
  domain everything else uses.
- **Small-component threshold: exactly the integer 20 pixels** — the
  pre-run formalization of the previously preregistered "components
  under approximately 20 px" diagnostic, frozen here BEFORE any Phase
  1A result exists. Comparison is strictly `< 20` (a component of size
  20 does NOT count as small; a component of size 19 does) — diagnostic
  descriptive evidence only, never a method parameter, gate, or filter
  applied to any prediction. Raw component sizes remain the primary
  persisted evidence; the small-component fraction is a derived summary
  statistic, not a replacement for them.
- Empty-set behavior matches `compute_reversion_coherence`'s own
  existing convention exactly: zero reverted pixels →
  `component_count=0`, `component_sizes=()`,
  `small_component_fraction=0.0`, `perimeter=0`,
  `perimeter_to_area_ratio=0.0`, `is_zero_reversion=True`.

T32 is a horizon-persistence baseline elsewhere in the protocol
(`[horizon_persistence].primary_horizon`) — it plays no role whatsoever
in reversion coherence, which is defined purely in terms of T0 and T320.

Boundary IoU remains its own, separate, still-`not_yet_bound` metric
authority (`[boundary_iou]`) — nothing in this module computes it; the
boundary-band diagnostic PRIMITIVE (`_official_mask_to_boundary`) is a
different, already-authorized tool used elsewhere for the
harmful-boundary/interior joint split, unrelated to reversion coherence.

## Inference-context safety

Both this module's `_bilinear_resize_adjoint` and Step 7's
`_apply_window_bilinear_adjoint` use a local, disconnected
`torch.autograd.grad` probe to compute an exact adjoint without ever
materializing a dense matrix. A real inference pipeline runs entirely
under an ambient `torch.no_grad()` / `torch.inference_mode()` — under
which a bare `requires_grad=True` probe produces no graph at all, and a
tensor already CREATED under `inference_mode()` carries a marker that
survives `.detach()` and can never be saved for backward at all. Both
functions therefore wrap the probe/forward/backward triple in a
narrowly scoped `torch.inference_mode(False)` + `torch.enable_grad()`
pair, and perform `.detach().clone()` on the incoming cotangent INSIDE
that scope (never on the caller's own tensor, never outside it) before
using it. Both context managers restore the caller's exact ambient
state on every exit path — success, a raised domain error, or any
`BaseException` (`KeyboardInterrupt`/`SystemExit`/`MemoryError`) — via
Python's own context-manager protocol. No model parameter is ever
referenced inside either block, and every returned tensor is detached
with no retained graph; see
`tests/test_dfp_phase1a_inference_context_adjoints.py` for the full
proof across all four ambient modes (normal, `no_grad()`,
`set_grad_enabled(False)`, `inference_mode()`), including tensors
created directly under `inference_mode()`.

## Status

This is authority and infrastructure only. No real dataset has been
touched, no model has been loaded, no CUDA device has been initialized,
and no Phase 1A diagnostic result exists yet. The production execution
driver that will eventually call this module remains a separate,
explicitly stopped task.
