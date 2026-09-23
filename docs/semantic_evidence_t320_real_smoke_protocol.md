# T320 Real Integration Smoke Contract

Task: `FREEZE_ISOLATED_T320_REAL_SMOKE_CONTRACT`.

Authority: `research_specs/semantic_evidence_t320_real_smoke_v1.toml`,
schema `semantic_evidence_t320_real_smoke_authority`, version `1.0.0`.

## What this is

This is a **two-image integration smoke test contract**, not an accuracy
experiment. It exists to answer one narrow question before intervention
work begins: does the committed, synthetically-verified isolated E3 adapter
(`src/semantic_evidence_e3_adapter.py`) and T320 target generator
(`src/semantic_evidence_t320.py`) actually work end-to-end against the real
E3 model, real external weights, real preprocessing, and real window
geometry — not just against fake models and hand-built synthetic tensors?

This document freezes the **contract**: what the future smoke runner and
launcher must do, what they must never do, and what a `PASS` result must
prove. It does not implement the runner. It does not execute anything.

## What this is not

- **Not an accuracy experiment.** It computes no IoU, mIoU, mAcc, or pixel
  accuracy, on any image, discovery or holdout.
- **Cannot validate the +1.5 mIoU publication target.** That requires the
  full 5,000-image evaluation and GT access, both explicitly unauthorized
  here.
- **Cannot establish full-evaluation runtime.** Two images' wall-clock
  timing on one job is not a valid estimator for 5,000 images; the
  telemetry contract explicitly prohibits using this smoke's timings that
  way.
- **Accesses no GT.** No COCO annotation, segmentation label, or ignore
  mask is ever read.
- **Not the 96-image feasibility study.** It uses exactly the first two
  canonical discovery-order image IDs from that study's own frozen 96-image
  manifest — never all 48 discovery images, never any holdout image, never
  the full 96.
- **Does not implement interventions.** No `CONTEXT_BLURRED`,
  `REGION_BLURRED`, or `AREA_MATCHED_CONTROL` view; no Gaussian blur; no
  region masking. Interventions remain unimplemented and unauthorized by
  this contract's mere existence.

## Why now

The isolated T320 implementation passed synthetic independent verification
(fake models, hand-built tensors, numerically-searched adversarial cases)
and has been committed
(`3a20c9ae8208a0df9ac9fa699ce0ff67a2221c8f`). But synthetic verification
cannot catch integration-boundary problems: a real DINOv2/CLIP backbone's
actual tensor shapes, dtypes, device placement, and the real
`image_transforms`/window-geometry pipeline have never touched this code.
Proving that boundary is sound — cheaply, on two images, with no accuracy
claim — is a prerequisite the project's own risk posture requires before
intervention implementation begins.

This contract may override the feasibility contract's general
no-real-T320-execution boundary **only** for: exactly two frozen discovery
images, GT-free integration checking, one bounded H100 smoke job, no
scientific accuracy claim. Every other prohibition in the feasibility
contract and the T320 source authority remains in full force.

## The one-forward design

```
real window
  -> exactly one ordinary E3 forward (extract_e3_window_snapshot)
  -> ordinary E3 reference output (bitwise preservation target)
     + immutable WindowSnapshot (dino_features, raw pre-sigmoid S0)
  -> T0 reconstruction from the SAME snapshot (internal_upsample(s0, ...))
  -> T320 from the SAME snapshot (build_t320_graph + propagate_t320 + internal_upsample)
```

`extract_e3_window_snapshot` calls `model.model.forward_features` exactly
once per window, and from that single pass computes both the ordinary E3
mask output (via `mask_pre_resize = sigmoid(simmap)`,
`F.interpolate(..., align_corners=True)`) and the `WindowSnapshot`
(L2-normalized DINO features, raw pre-sigmoid `S0` — captured directly as
`forward_seg`'s own `simmap`). `generate_t0_t320_targets` is a pure function
of `WindowSnapshot` objects; it performs no model inference itself. This
means the pure T0/T320 target generation is **repeatable from cached
snapshots** without repeating the model forward — the smoke's determinism
check (Part D of the scientific contract) exploits exactly this: the same
per-window snapshots are fed into `generate_t0_t320_targets` twice, with
zero additional real model forwards between the two repeats.

## Ordinary E3 preservation

For every window, the smoke must compare the ordinary E3 per-window
reference output (the adapter's own `ordinary_mask_output`, produced from
the same single real forward) against the T0 reconstruction from that same
forward's cached snapshot (`internal_upsample(snapshot.s0, grid_hw,
output_hw)`). Both are produced by the identical
`torch.sigmoid` + `F.interpolate(align_corners=True)` operation pair from
the identical underlying tensor, with no random or reduced-precision
operation between them and no cross-window reduction yet at this point —
so the contract freezes **bitwise equality**, not a tolerance. Any mismatch
fails the smoke; there is no warning-only bypass.

## Window count is deliberately runtime-derived

This contract task cannot access image pixels or image dimensions, so it
cannot and does not fabricate the two authorized images' window counts.
Crop (`[448, 448]`), stride (`[224, 224]`), and patch size (`[14, 14]`) are
frozen, matching the current-branch's own
`src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/stuff.py`
`test_cfg` and `src/local_weights.py`'s DINOv2 patch size — but the actual
window count `N` is `RUNTIME_DERIVED_AND_RECORDED`: only the future smoke's
own preflight step, after resolving the two authorized image files (no GT),
may generate the canonical window universe (reusing
`DINOTextSegInference.slide_inference`'s own row-major window-generation
logic, never a reimplementation), persist every ordered window record, and
compute a canonical ordered-window digest. No window may be dropped,
sampled, reordered, or replaced.

## T320 execution and determinism

Per window, per target-generation repeat: exactly one graph construction,
one 320-step recurrence trajectory operating on raw (pre-sigmoid) `S0`,
exactly one sigmoid applied after update 320, bilinear internal upsample
(`align_corners=True`), sigmoid-domain overlap-averaged stitching, bilinear
final rescale (`align_corners=False`), ordinary argmax. The pure target
generation is repeated exactly twice from the same cached, immutable
snapshots — no second model forward.

**Two count families, two different units.** `generate_t0_t320_targets` is
**image-scoped**: it takes a single scalar `canvas_h`/`canvas_w` pair and
explicitly asserts every snapshot passed to one call shares that canvas
(`src/semantic_evidence_t320.py:371-410`) — so one call processes exactly
one image's windows, never a multi-image campaign's. Two independent count
families follow from this:

- **Window-scoped** (one unit of work per window): model forwards, adapter
  snapshots, T0/T320 sigmoid conversions, T320 graph constructions,
  T320 trajectories, T320 recurrence updates. These scale with the total
  window count `N` across both images: for `R=2` repeats, `2N` T0 sigmoid
  conversions, `2N` T320 graph constructions and trajectories, `2N×320`
  recurrence updates, `2N` T320 sigmoid conversions.
- **Image-scoped** (one unit of work per image, per repeat): the T0 and
  T320 stitching and finalization calls, since each `generate_t0_t320_targets`
  call performs exactly one `stitch_windows` and one `finalize_prediction`
  per variant. These scale with the selected image count `M`, not with `N`:
  for `M=2` images and `R=2` repeats, the campaign totals are `M×R=4` T0
  stitch calls, `4` T0 finalize calls, `4` T320 stitch calls, and `4` T320
  finalize calls — **not 2**. (An earlier version of this contract froze
  these as an unqualified, unitless `2`; that was a defect, found by
  independent adversarial re-verification and repaired here. A flat `2`
  only holds "per image, per repeat" — it was never the correct campaign
  total, and calling `generate_t0_t320_targets` once for the whole
  two-image campaign to make `2` literally true is not a valid alternative:
  the function's own per-snapshot canvas-match assertion rejects mixing
  windows from two images with different real dimensions into one call,
  and even where dimensions coincidentally matched, doing so would corrupt
  stitching by overlapping unrelated images' content.)

Between the two repeats: identical window order, identical T0/T320
scientific digests, exact prediction equality, zero non-finite values,
unchanged snapshot bytes. This is same-job, same-device, same-input
repetition — never a cross-device bitwise-determinism claim.

## Scope isolation

Only the two authorized discovery image IDs' pixels may ever be accessed by
the future smoke. No COCO annotation, segmentation label, ignore mask,
class-correctness computation, IoU, mIoU, mAcc, pixel accuracy, or
discovery/holdout performance comparison is authorized. No holdout image,
no other discovery image, no larger population (96-image, 350-image BALP
development subset, 150-image confirmation subset, or the full 5,000-image
evaluation), no training, no interventions, no adapter learning, and no
publication-level accuracy claim is authorized by this contract.

## Resource and telemetry

One bounded job: 1×H100 80GB, 8 CPUs, ≤80GB host memory, 00:30:00 wall
time, account `rrg-yangw`, fresh process, no resume, no overwrite/force, no
self-submission. Preflight (including external-weight re-hashing) must run
before CUDA initialization, model construction, checkpoint loading, and
dataset image access. Telemetry (wall-clock per stage via
`time.perf_counter_ns`, `torch.cuda.synchronize()` around every measured
GPU region, peak host RSS, peak/reserved/total GPU memory) is recorded
operationally and must never be used to estimate full-dataset runtime, and
profiling must never alter the scientific digests being compared.

## Result and installation

The result artifact (`smoke_result.json`, scratch-only, never committed)
resolves to exactly one of two outcomes:
`PASS_REAL_E3_T320_INTEGRATION_SMOKE` or
`FAIL_REAL_E3_T320_INTEGRATION_SMOKE`. `PASS` requires every scientific,
count, determinism, scope, and artifact check to pass — there is no
warning-only bypass. Installation is staged in a sibling temporary
directory, validated before an atomic directory rename, then re-verified by
the public verifier after installation; a success message may not be
printed before that post-install verification completes; failure preserves
prior state.

## Authorization boundary

`PASS` on this contract's own verification permits proceeding to **Step 3
implementation** — i.e. writing the smoke runner and launcher — and nothing
else. Real execution remains unauthorized until: this contract passes
independent verification, is committed, the runner and launcher are
implemented and independently verified, and a separate committed-HEAD
readiness check passes. The operator submits the eventual job manually;
self-submission is prohibited by the contract itself.

## Machine validation coverage

All 19 top-level contract tables are machine-validated by
`src/semantic_evidence_t320_real_smoke_identity.py`, including the four
tables (`validity_contract`, `scope_isolation`, `telemetry_contract`,
`artifact_policy`) that an earlier version of the loader left present in
the TOML but never actually read — a second defect found and repaired
alongside the stitch/finalize count defect above. `verify_all` now proves
this at runtime (not merely by static claim): it wraps the loaded contract
in a dict subclass that records every top-level key actually read via
`__getitem__`, and refuses to report `verified=True` unless all 19 keys
were touched during that specific run. A dedicated coverage test
(`test_all_19_top_level_tables_validated_at_runtime`) exercises this
directly. Each of the four repaired tables also has its own adversarial
test coverage — every declared leaf value, every closed key set, and the
cross-table consistency checks (e.g. `scope_isolation`'s prohibitions must
agree with `gt_isolation`/`closure`/`status`; `artifact_policy`'s
overwrite/force prohibition must agree with `resource_contract`'s).

## New files (this task)

- `research_specs/semantic_evidence_t320_real_smoke_v1.toml` — the closed
  contract (identity/status, implementation binding with ancestry
  requirement, dependency bindings, external-weight bindings, two-image
  population, window geometry, scientific/determinism/scope/resource/
  telemetry/result-schema/artifact-policy sections, closure).
- `evaluation_subsets/semantic_evidence_t320_real_smoke_2.json` — the
  closed two-image population, deterministically derived from
  `evaluation_subsets/semantic_evidence_feasibility_96.json`'s first two
  already-ordered discovery-partition records (`000000567197`,
  `000000367680`).
- `src/semantic_evidence_t320_real_smoke_identity.py` — the stdlib-only,
  torch-free, CUDA-free loader/validator.
- `tests/test_semantic_evidence_t320_real_smoke_identity.py` — positive
  and adversarial tests covering all 19 top-level tables.
- this document.

No existing file is modified by this task. Nothing is staged or committed.
