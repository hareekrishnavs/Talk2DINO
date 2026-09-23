# T320 Isolated Implementation (Step 2)

Task: `IMPLEMENT_ISOLATED_T320_TARGET_GENERATOR_AND_E3_EXTRACTION_INTERFACE`,
repaired by `REPAIR_ISOLATED_T320_SIGMOID_AND_PUBLIC_API`.

This document explains the two new isolated modules created by this task,
what they do and do not do, and the exact boundary between "implemented and
tested" and "authorized to run on real data" — those are different things,
and this task only accomplishes the first.

## Repair note (score-domain sigmoid placement and public API closure)

An independent adversarial verification of the original implementation
found two defects, both now repaired in `src/semantic_evidence_t320.py`
only (`src/semantic_evidence_e3_adapter.py` and
`tests/test_semantic_evidence_e3_adapter.py` are byte-identical to the
original implementation throughout this repair):

1. **Score-domain defect (critical).** The original `internal_upsample`
   bilinear-interpolated raw (pre-sigmoid) scores directly. The historical
   source function it is bound to,
   `DINOText.masks_from_patch_scores` (commit
   `00392527937253f93b9f3fe36b76347b4a6f1eb0`), applies exactly one
   `torch.sigmoid` (via `masker.mask_from_similarity` →
   `DINOTextSim2Mask.forward`'s unconditional soft-mask branch) **before**
   its bilinear interpolation, and window stitching (`stitch_one_image`)
   accumulates that sigmoid-domain output directly. Since
   `mean(sigmoid(x)) != sigmoid(mean(x))` in general, stitching raw scores
   (as the original implementation did) does not reproduce the
   source-authorized behavior for any pixel covered by more than one
   window — the normal case for the sliding-window pipeline. `internal_upsample`
   now applies `torch.sigmoid` before interpolation, for both T0 and T320.
2. **Public API defect.** `build_t320_graph` originally accepted
   caller-supplied `k`/`affinity_power` keyword arguments on its public
   signature. It now accepts only `dino_features`; the frozen K=12,
   AFFINITY_POWER=3.0 are pinned internally and cannot be overridden. A
   private `_build_directed_topk_graph_with_params` helper retains the
   parameterized form for isolated unit testing only.

See `tests/test_semantic_evidence_t320.py::SigmoidDomainRepairRegressionTests`
for the regression tests added for this repair — independently confirmed
to fail against the pre-repair implementation (10 of 20 tests in that class
are specifically bug-detecting; the rest are structural/invariant checks
that correctly pass under both versions).

## Committed prerequisites

This implementation is built strictly on top of two already-committed,
already-verified authorities on branch `research/semantic-evidence-interventions`:

- `research_specs/semantic_evidence_feasibility_v1.toml` +
  `evaluation_subsets/semantic_evidence_feasibility_96.json` +
  `src/semantic_evidence_feasibility_identity.py` +
  `docs/semantic_evidence_feasibility_protocol.md` — the 96-image
  feasibility contract, which binds T0 to committed E3 unary inference and
  leaves T320 deferred until a successor authority exists.
- `research_specs/semantic_evidence_t320_source_v1.toml` +
  `research_specs/semantic_evidence_t320_source_manifest_v1.json` +
  `src/semantic_evidence_t320_source_identity.py` +
  `docs/semantic_evidence_t320_source_authority.md` — the T320 source
  authority, schema `semantic_evidence_t320_source_authority` version
  `1.0.0`, binding the historical source commit
  `00392527937253f93b9f3fe36b76347b4a6f1eb0` (branch
  `e12-connectivity-analysis`) and its seven-file scientific closure by
  Git blob hash and AST-verified required symbol.

Neither authority is modified by this task. This task adds exactly the
"later isolated implementation" both authorities explicitly anticipate and
authorize, once independently verified and committed — which is precisely
where this branch now stands.

## New files

- `src/semantic_evidence_e3_adapter.py` — the E3 extraction interface.
- `src/semantic_evidence_t320.py` — the isolated T0/T320 target generator.
- `tests/test_semantic_evidence_e3_adapter.py`,
  `tests/test_semantic_evidence_t320.py` — CPU-only oracle and fake-model
  tests, none of which touch a real model, checkpoint, dataset image, or GT.

## The E3 extraction interface

`extract_e3_window_snapshot(model, image, text_emb, *, ...)` returns
`(WindowSnapshot, ordinary_mask_output)` from **exactly one** call to
`model.model.forward_features` (the DINO backbone). It does this by calling
the *same* existing, committed methods `DINOText.generate_masks` already
calls internally — `model.image_transforms`, `model.model.forward_features`,
`model.proj.project_dino`/`project_visual`, `model.masker.forward_seg` — in
the same order, over the same single backbone pass. `generate_masks`
computes these same intermediate tensors but never returns them; this
adapter orchestrates the identical call sequence to make them visible,
without touching `dinotext.py` or `masker.py` and without reimplementing
any of their logic. The one non-method-call operation is recomputing the
L2-normalized features `forward_seg` uses internally, via
`torch.nn.functional.normalize(t, dim=1, eps=1e-6)` — the exact call the
repository's own `us.normalize` utility wraps, called directly only to
avoid an unrelated `mmcv`/`cv2` import chain, not as a reimplementation of
any E3 formula.

**Scope boundary**: `model.with_bg_clean=True` (a background-cleaning
post-process, off by default, off in the verified Step 0 baseline
configuration) is out of scope for T0/T320 generation. The adapter raises
`E3AdapterError` if the supplied model has `with_bg_clean=True`, rather
than silently producing an "ordinary output" value that would not actually
match `generate_masks` in that configuration.

### One-forward invariant

Verified by test (`test_exactly_one_backbone_forward_per_window`,
`test_no_second_forward_across_repeated_calls_is_caller_controlled` in
`tests/test_semantic_evidence_e3_adapter.py`) using a deterministic fake
model with a call-counting backbone: exactly one `forward_features` call
per `extract_e3_window_snapshot` call, never more, regardless of how many
windows are processed in sequence.

### Output preservation

`sigmoid(S0)`, resized with `align_corners=True`, equals the adapter's
returned `ordinary_mask_output` exactly (`test_sigmoid_s0_reconstructs_ordinary_mask_scores_exactly`),
because `DINOTextSim2Mask.forward`'s soft-mask branch is plain,
unconditional `torch.sigmoid(x)` — no gumbel noise on that path — confirmed
by reading `masker.py` directly (committed branch, not historical e12).
`test_adapter_on_and_ordinary_path_predictions_identical` additionally
proves, on a fake model, that the adapter's output matches an
independently-written reference implementation of `generate_masks`'s own
resize step, computed by calling the same fake sub-methods a second time
(a test-only comparison; the adapter itself still does exactly one forward
per call).

## S0 definition and the full score-domain order

`S0 = einsum("b c h w, n c -> b n h w", L2_normalize(dino_features, dim=1,
eps=1e-6), text_emb)` — text embeddings are **not** normalized (matching
`DINOTextMasker.forward_seg`'s own commented-out
`# text_emb = us.normalize(text_emb, dim=-1)` line, confirmed unapplied on
the committed branch). `S0` is pre-sigmoid; it is captured directly as
`forward_seg`'s own `simmap` return value, never computed by this module's
own formula.

The exact score-domain order, confirmed directly from historical source and
identical for T0 and T320 except for the graph/recurrence step in between:

```
T0:    raw pre-sigmoid S0
         -> exactly one sigmoid (internal_upsample)
         -> bilinear internal upsample, align_corners=True (internal_upsample)
         -> overlap averaging across windows, IN SIGMOID DOMAIN (stitch_windows)
         -> bilinear final rescale, align_corners=False (finalize_prediction)
         -> argmax (finalize_prediction)

T320:  raw pre-sigmoid S0
         -> frozen graph construction (build_t320_graph)
         -> exactly 320 propagation updates, IN RAW-SCORE SPACE (propagate_t320)
         -> exactly one sigmoid, applied AFTER all 320 updates (internal_upsample)
         -> bilinear internal upsample, align_corners=True (internal_upsample)
         -> overlap averaging across windows, IN SIGMOID DOMAIN (stitch_windows)
         -> bilinear final rescale, align_corners=False (finalize_prediction)
         -> argmax (finalize_prediction)
```

**Non-commutativity warning**: window-wise sigmoid must occur before
overlap averaging. Applying sigmoid after stitching is not equivalent --
`mean(sigmoid(x)) != sigmoid(mean(x))` in general -- and was the exact
defect this implementation's repair fixed (see "Repair note" above).
Sigmoid never occurs anywhere else: not before recurrence (recurrence
operates entirely on raw, unbounded scores -- confirmed structurally, since
sigmoid-domain values are bounded in `(0,1)` and recurrence output is not),
never a second time before stitching, never after stitching, and never
after final rescaling.

## Isolated T320 module

`src/semantic_evidence_t320.py` implements, from validated `WindowSnapshot`
objects only, exactly the source-authorized pipeline:

1. **Graph construction** (`build_t320_graph`) — directed top-k affinity
   graph, `k=12`, `affinity_power=3.0`, cosine similarity of L2-normalized
   features, negative affinities clamped to zero, self excluded via a
   `-inf` diagonal, top-k selected by a **stable** descending argsort
   (never `torch.topk`) so exact ties resolve to the lower original column
   index, row-normalized, a self-loop-of-weight-1 fallback for any
   all-zero row. Directed; never symmetrized.
2. **Finite-step recurrence** (`propagate_t320` / private
   `_finite_step_propagate`) — `P(0)=S0`; `P(t+1) = alpha*(A@P(t)) +
   (1-alpha)*S0` for exactly `steps=320` completed updates at `alpha=0.98`;
   no convergence test, no early stop, no tolerance, no adaptive step
   count; `S0` is defensively cloned once and never mutated; governed
   float32.
3. **Internal per-window upsample** (`internal_upsample`) — applies exactly
   one `torch.sigmoid` to the raw per-window patch scores (matching
   `masks_from_patch_scores`'s own `mask_from_similarity` → `sigmoid` step),
   then bilinear-interpolates the resulting sigmoid-domain mask with
   `align_corners=True`, patch-grid → window-pixel resolution. Applied once
   per window, to both T0's raw S0 and T320's post-320-step recurrence
   output, before stitching. See "S0 definition and the full score-domain
   order" above.
4. **Window stitching** (`stitch_windows`) — uniform averaging by overlap
   count only; fails closed (`T320Error`) on any uncovered canvas pixel.
5. **Final rescale + prediction** (`finalize_prediction`) — bilinear,
   `align_corners=False` (matching the committed
   `DINOTextSegInference.align_corners=False`, the sole such assignment in
   the historical source tree — never the historical 350-image DFP
   identity's `align_corners=True`), then plain `torch.argmax(dim=1)` — not
   the later `lowest_index_argmax` helper; PyTorch's own tie-break
   behavior on that primitive is used and is not independently guaranteed
   by this module.
6. **T0 generation** — the original view only: `S0` itself, sigmoid-transformed,
   upsampled, stitched, and rescaled exactly like T320's pipeline but with
   **no graph and no recurrence** (`t=0`, before any propagation). T0's
   predicted class at a node equals that node's own `argmax(S0)` (sigmoid
   is monotonic and does not change a single node's own argmax; it matters
   only once cross-window stitching averages multiple windows' outputs),
   matching q1 by construction.
7. **Top-level generator** (`generate_t0_t320_targets`) — accepts only
   validated `WindowSnapshot` objects and target geometry (`canvas_h`,
   `canvas_w`, `ori_h`, `ori_w`). No public function in either module
   exposes `k`, `alpha`, `steps`, `affinity_power`, `align_corners`,
   sigmoid placement, class-remapping, debug-subset, or
   candidate-specific-recurrence as a parameter (verified by
   `test_G_no_public_function_in_either_module_exposes_scientific_overrides`
   and `test_no_scientific_override_parameters_exposed`), and no GT or
   annotation parameter (`test_H_no_gt_dependency_in_public_api`,
   `test_no_gt_parameter_exists`). For N windows: N graph constructions, N
   recurrence trajectories (N×320 updates total), 2N sigmoid applications
   (N for T0, N for T320), zero candidate-specific recurrence, zero
   component/masked-forward calls — verified by
   `test_graph_and_recurrence_call_counts_for_n_windows` and
   `test_B_sigmoid_call_count_and_placement_for_n_windows`.

Every function's docstring cites the authority schema/version, the
historical source commit, and the relevant source-manifest role/required
symbol it reproduces.

## GT isolation

Neither new module references ground truth, annotations, correctness
labels, or intervention outputs anywhere — confirmed both by design (no
such parameter exists on any function) and by a dedicated AST-based test
(`ProhibitedImportAuditTests.test_e3_adapter_module_no_ground_truth_reference`)
that scans every identifier in `semantic_evidence_e3_adapter.py` for
GT-like names, not merely greps for a substring.

## Prohibited dependencies

An AST-based import scan (not substring-only grep) over both new modules
confirms their only imports are `__future__`, `dataclasses`, `typing`,
`torch`, `torch.nn.functional`, and (for `semantic_evidence_t320.py`) the
sibling `semantic_evidence_e3_adapter` module. Neither module imports
`window_cache`, any CGLS/RWR implementation, `cover_dr.inference`,
`cover_dr.coco_object_evaluator`, `cover_dr.native_edge_support`,
`cover_dr.stitching_control`, any `dfp_phase1a`/`dfp_phase1b`/BALP module,
a dataset provider, a model builder, or the historical `e12` branch at
runtime.

## CPU-only test evidence

All tests run on CPU, with small synthetic or hand-constructed tensors,
via a CPU-only PyTorch installation (`/scratch/haree/venv/talk2dino-a100`,
confirmed `torch.cuda.is_available() == False` in this environment). No
GPU was requested or used. 83 new tests pass (14 adapter + 69 T320-module,
the latter including the 20-test `SigmoidDomainRepairRegressionTests` class
added by the sigmoid/public-API repair), plus 25 pre-existing
torch-dependent E3-adjacent tests (`tests/test_paired_soft_routing.py`,
`tests/test_validation_feature_selection.py`) continue to pass unmodified,
plus both committed authorities' own official suites (292 tests total)
remain at their prior baseline.

- **Bit-exact** comparisons: `sigmoid(S0) == ordinary_mask_output` (up to
  the source's own floating-point behavior); recurrence steps 1–3 against
  an independent explicit-loop oracle; the hand-computed 5-node graph's
  neighbor indices and transition weights; the internal-vs-final
  `align_corners` divergence; `internal_upsample` against an independent
  `sigmoid`-then-`interpolate` oracle; a full independent T0 and T320
  end-to-end oracle (including a two-overlapping-window T320 case built to
  materially diverge from the pre-repair raw-score behavior).
- **Tolerance-based**: floating-point comparisons use `torch.allclose`/
  `assertAlmostEqual` at 1e-5–1e-6 precision, consistent with ordinary
  float32 accumulation behavior, never claiming bitwise cross-device
  determinism beyond what the source authority itself guarantees.
- **Structural**: call-count invariants (one forward per window, N graph
  constructions for N windows), shape/dtype/finiteness validation, and the
  AST-based prohibited-import and locator-grammar audits.

No real DINOText model, checkpoint, or dataset image was used in deriving
any of these comparisons; the fake model in
`tests/test_semantic_evidence_e3_adapter.py` is a small, independently-written
deterministic stand-in that matches the current E3 API surface, not a
model of DINOText's actual learned behavior.

## What this task does not do

- **No real model inference has been run.** Every test uses either purely
  synthetic tensors or a deterministic fake model; no checkpoint was
  loaded, no real image was processed, and no GPU or CUDA context was
  initialized.
- **No real T320 target has been generated.** `generate_t0_t320_targets`
  is implemented and tested against synthetic/fake inputs only.
- **The historical 350-image result (+1.9405pp T0→T320 mIoU gain) is not
  matched evidence for this implementation.** That run used a different
  final-rescale function with `align_corners=True`; this implementation
  freezes `align_corners=False`, per the committed T320 source authority.
- **No expected mIoU is claimed** for this implementation, on any dataset,
  real or synthetic.
- **Interventions are not implemented.** No `CONTEXT_BLURRED`/
  `REGION_BLURRED`/`AREA_MATCHED_CONTROL` view, no Gaussian blur, no region
  masking exists anywhere in these two new modules.
- **The 96-image feasibility study remains unauthorized.** Nothing in this
  task changes `[gates.outcome].feasibility_execution_authorized = false`
  in the committed feasibility contract, or any authorization-boundary
  flag in the committed T320 source authority. Real dataset access, model
  construction, checkpoint loading, GT access, training, confirmation, and
  full evaluation all remain unauthorized by this implementation's mere
  existence.
