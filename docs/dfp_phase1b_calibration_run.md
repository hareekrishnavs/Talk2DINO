# DFP Phase 1B: conservation-calibration runner

This document describes `src/dfp_phase1b_calibration_run.py` and
`diagnostics/run_dfp_phase1b_calibration.py`: the runner authorized by
the committed E16 successor authority
(`evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml`)
to measure raw conservation residuals over exactly 200 deterministically
selected windows of the frozen COCO-Stuff-500 subset.

**This is calibration only.** It does not run full Phase 1B provenance,
does not select or apply a numeric conservation tolerance, and a
successful, complete artifact does **not** authorize the full Phase 1B
run. See "What happens after this" below for the required next steps.

## What this runner does and does not do

Does:

- Reconstructs the frozen COCO-Stuff-500 candidate-window universe and
  deterministically selects exactly 200 windows via the committed E16
  selector.
- For every selected image, runs canonical (unmasked) and exact
  component-masked k12 propagation in the same job, reusing the exact
  same `DirectedTopKGraph` instance for both.
- Measures `rho_M`, `rho_0`, `rho_320`, and the primary final-resolution
  `rho_g` residual for every selected window, reduces them to FP64 raw
  summary statistics, and can assemble/atomically install the resulting
  `phase1b_conservation_calibration.json` artifact.

Does not:

- Choose, infer, or record a numeric conservation tolerance anywhere.
- Read ground truth at any point in candidate selection, component
  construction, membership computation, or residual measurement.
- Provide a `--device cpu` execution path for `execute` (the E16
  authority requires `device_type="cuda"` for a genuine, complete
  artifact; a CPU run could never be that).
- Implement resume/checkpoint semantics (E16 does not authorize them --
  see "Fresh-only execution" below).
- Submit SLURM, run the full 200-window calibration against real data,
  or authorize the full Phase 1B run.

## E16 binding

Every execution-sensitive value this runner uses is either read
directly from the loaded, hash-verified E16 identity
(`src.dfp_phase1b_calibration_identity.load_identity`) or from one of
E16's own already-hash-bound dependencies (the E15 protocol/artifacts,
the exact masked-forward core, the frozen subset manifest, the
canonical sliding-window-plan implementation, the stitching/final-
resolution pullback implementations). `run_preflight` (Phase A) loads
and relationally re-verifies all of this before any heavy import.

One value -- the component chunk size -- is authorized by E16 to be
*recorded* in the artifact's `propagation_precision_record.chunk_size`
field, but is not itself frozen to one number. This runner therefore
never silently defaults it: `execute --component-chunk-size N` requires
an explicit, positive integer on every invocation, and
`build_propagation_precision_record` rejects anything else.

## Candidate enumeration and selection (Phase B)

`build_candidate_universe` enumerates every canonical sliding window
for every one of the 500 frozen manifest images, in manifest order,
using only image/pipeline geometry (image size, patch size, the
frozen crop/stride from `e13_dfp_phase1a_protocol.toml`'s own
`[score_interpolation_stitching]` block) -- never GT, never a
prediction, never a model forward pass. `select_and_verify` then runs
the committed E16 selector (`select_calibration_windows`) exactly once
over that universe and asserts exactly 200 rows come back. Selection
depends only on the candidate records' own content (image id, window
geometry) hashed under E16's frozen namespace -- never on GT,
predictions, filesystem/hash iteration order, worker count, or GPU.

## Selected-image execution strategy (Phase C)

Windows are selected individually, but full-image canonical
reconstruction (T0/T320, global unary components, disputed-pixel set)
requires the COMPLETE stitched image, which in turn requires **every**
canonical window of a selected image, not only the selected ones. This
runner therefore groups selected windows by image and, for each
selected image, processes ALL of that image's canonical windows once
(`_process_one_window_for_calibration`, mirroring but never importing
`src.dfp_phase1a_execute`'s own private per-window helper), retaining
the built `DirectedTopKGraph` instance only for windows that were
actually selected.

## Same-graph guarantee

For a selected window, canonical (unmasked) propagation and masked
propagation must never use two different graph objects, even if
scientifically equivalent -- a rebuilt graph is a different (if
numerically close) top-k selection under floating point, and the E16
residuals are defined against the SAME transition operator for both
terms. `_process_one_window_for_calibration` builds the graph exactly
once per window and returns it (only when the window was selected)
alongside the canonical trajectory it was used to produce;
`process_selected_image` later passes that exact same object into
`propagate_masked_seed_chunk`. This is exercised directly by
`tests/test_dfp_phase1b_calibration_run.py::test_same_graph_instance_used_for_canonical_and_masked`,
which compares `id(graph)` between the canonical-capture call and every
masked-propagation call for the same window.

## Exact residuals

All four residual families are computed by calling the committed,
unmodified `src.dfp_phase1b_masked_forward` core
(`compute_membership_partition_residual`,
`compute_seed_reconstruction_residual`,
`compute_propagated_reconstruction_residual`,
`compute_signed_margin_conservation_residual`,
`compute_signed_contributions`, `stitch_and_rescale_component_scores`)
-- this module never re-derives any of these formulas independently.

Two design decisions were necessary because no committed function
directly answers them; both are also documented at length in
`src/dfp_phase1b_calibration_run.py`'s own module docstring:

1. **Q_w derivation.** `dfp_phase1b_masked_forward.py`'s own docstring
   names `src.dfp_phase1b_compute_estimator.compute_window_query_classes`
   as the frozen Q_w derivation, but that function structurally requires
   its `a`/`b`/`dispute` arrays at `window_plan.image_size` -- the
   stitched, pre-final-rescale (inference) resolution -- while E16's own
   `rho_g` domain is final (ori) resolution. `compute_window_query_classes`
   is also one of the functions
   `tests/test_dfp_phase1a_execute.py::test_driver_source_contains_no_deleted_temporary_paths`
   documents as deliberately superseded. This runner instead reproduces
   `src.dfp_phase1a_execute._process_one_image`'s CURRENT,
   final-resolution-consistent inline Q_w logic (component class index
   of each active component, unioned with the disputed b-values seen
   anywhere in that component), driven by
   `src.dfp_phase1a_final_rescale_pullback.iter_image_final_soft_memberships`
   -- the same membership authority the masked-forward core's own
   docstring names.

2. **Window-restricted numerator, full-image denominator.** E16 defines
   `rho_g`/`delta_m^(w)` as the selected window's own contribution,
   stitched against the complete (never reduced) canonical window plan.
   This runner realizes that literally: both the masked numerator and
   the canonical numerator are stitched by calling the unmodified
   `stitch_and_rescale_component_scores` with a `propagated_by_window`
   mapping containing only window w's own chunk, while `window_plan`/
   `final_plan` remain the image's complete, all-window plan. The
   canonical numerator is packaged as a `PropagatedChunk` with
   `component_count=1` purely as a container -- `stitch_and_rescale_
   component_scores` is already generic in component count, so no new
   formula is introduced.

3. **Selected window's own final-resolution support.** No committed
   function directly computes "which final(ori)-resolution pixels does
   window w's own crop project onto." This runner derives it
   (`_window_final_support_mask`) by feeding an all-ones patch
   indicator through the unmodified `apply_window_stitching_forward`,
   embedding the result into a zero inference-resolution canvas at
   `window.accumulation_slice`, and applying the unmodified
   `apply_final_rescale_forward` -- a coverage probe of an existing
   linear operator, not a new geometry formula.

A real, subtle bug was caught in early development by this module's own
synthetic end-to-end fixture (not by static review): `compute_signed_
contributions` requires its `disputed_pixel_a`/`disputed_pixel_b`
arguments to be **global** class ids (it performs its own local
conversion internally against its own `query_class_union` argument).
An earlier draft passed already-localized indices instead, which is
silently wrong whenever a window's own `Q_w` is not the contiguous
range `0..len(Q_w)-1` -- confirmed by a genuine `DfpPhase1bMaskedForwardError`
surfaced only once real random synthetic data was pushed through the
full pipeline. Fixed by passing the raw global `a(x)`/`b(x)` values to
`compute_signed_contributions`, while continuing to use the pre-
localized indices only for directly indexing the already-query-class-
ordered `m0_w`/`m320_w` tensors (a different, correctly-local-indexed
use).

## FP32 storage / FP64 recurrence and accumulation

`DirectedTopKGraph` storage remains FP32 (matching every other
production caller of the k12 graph). All propagation
(`MaskedForwardContract.propagation_dtype`) and all residual
accumulation (`accumulation_dtype`, `summarize_residual_tensor`) are
FP64, matching E16's `component_sum_accumulation_dtype == "fp64"`
requirement. No autocast, no mixed precision, anywhere.

## Chunk-size handling

`component_chunk_size` only changes how many global components are
processed per `build_masked_seed_chunk`/`propagate_masked_seed_chunk`
call -- never the output. `tests/test_dfp_phase1b_calibration_run.py::test_end_to_end_chunk_size_invariance`
confirms identical per-window sample counts and mean-absolute-residual
values across chunk sizes 1, 2, and "one chunk covers everything" on
the same synthetic image.

## Zero-sample behavior

A selected window can legitimately have zero active components, zero
query classes, or zero disputed pixels within its own final-resolution
support (E16 explicitly authorizes recording this as a valid outcome,
never a fabricated nonzero value). `summarize_residual_tensor` returns
`sample_count=0` with every float field `None` in that case; the
artifact schema's own `validate_residual_summary` requires exactly
that convention.

## Artifact lifecycle

`assemble_calibration_artifact` is a pure function: given a
`PreflightReport`, a `SelectionOutcome`, a list of selected-window rows
(each merged with its own `completed` flag and four residual-family
summaries), and device/software/precision records, it builds the exact
E16 closed schema in memory -- including the one additive aggregate
reconstruction via the committed `aggregate_residual_summaries` (never
a second, parallel aggregation).

`install_calibration_artifact` performs the required atomic install
flow: validate the in-memory document -> serialize deterministically
(`json.dumps(..., sort_keys=True, allow_nan=False)`) -> write to a
temp sibling file in the destination's own directory -> `fsync` ->
read the temp file back from disk -> independently re-validate the
read-back document via the same committed validator -> `os.replace`
(atomic on POSIX) -> on an ambiguous `os.replace` failure, check
whether the destination already matches the intended content before
concluding the install failed. The temp file is always removed on
every failure path; a caller-supplied `allow_overwrite` flag is
re-checked immediately before the write (closing the time-of-check-to-
time-of-use window since Phase A's own preflight check, which could
otherwise be stale after a long GPU run).

**Output-path policy.** `--output` may be any absolute or relative
path (the real calibration run uses an explicit absolute
`/scratch/.../phase1b_conservation_calibration.json` destination) with
one restriction: no path component may be the literal `..`. This is a
lexical check on the path as written, performed before any
resolution, directory check, or filesystem mutation, and shared by
both `run_preflight` and `install_calibration_artifact` (via
`_validate_output_path`) -- it never assumes preflight ran first. It
is not an authorized-output-root policy (no such root is frozen by
E16); it only rejects a `..` parent-directory reference, whether
relative or embedded inside an otherwise absolute path, so a typo
cannot silently resolve the artifact destination outside the intended
directory.

## Device and software records

`real_device_record`/`real_software_record` populate the artifact's
`device_record`/`software_record` fields from the live CUDA/torch/numpy/
git state at call time -- never a cached or hardcoded value.
`real_device_record` requires `torch.cuda.is_available()` and never
provides a CPU-shaped fallback. Both are ordinary functions (not
required to go through a `Protocol`), but the runner's Phase C
inference/image-preparation dependencies ARE `typing.Protocol`-typed
(`InferenceProvider`, `ImagePreparer`, `GeometryProvider`) specifically
so tests can inject small synthetic implementations while production
wires real Phase 1A execution-context objects.

## Resume/fresh-only decision

E16 (`evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml`,
and its loader/artifact modules) contains no "resume" or "checkpoint"
field anywhere. Per this task's own governing rule ("if E16 neither
freezes nor authorizes a value, do not invent it"), this runner
implements **fresh-only** execution: `_require_fresh_mode` rejects any
`mode` other than `"fresh"` with a message citing this exact absence,
and the CLI's `--mode` argument only accepts the literal `"fresh"`.

## No GT, ever

Candidate selection, component construction, membership computation,
and every residual measurement complete without reading ground truth
at any point -- confirmed by `src.dfp_phase1a_execute._process_one_image`'s
own structure (GT is read only *after* all of the equivalent Phase 1A
computation) and independently re-confirmed here: no public function in
this module accepts a GT-shaped parameter
(`test_no_public_function_accepts_a_gt_shaped_parameter`), and
`process_selected_image` itself has no `dataset`/`gt` parameter at all.

## No numeric tolerance, ever

No production code path in this module contains an epsilon, atol,
rtol, tolerance, residual threshold, percentile cutoff, safety factor,
or pass/fail verdict. The word "tolerance" appears in this document and
in the module's own docstrings only to describe what is deliberately
**not yet decided** -- the eventual conservation-tolerance authority is
a separate, future, explicitly-out-of-scope pre-registration step.

## No real calibration has been executed by this implementation

This runner has been synthetically tested against real, unmodified
scientific code (real k12 graphs, real 320-step FP64 recurrence, real
stitching/rescale/component/membership/masked-forward functions) using
small hand-built geometry fixtures. It has **not** been run against the
real COCO-Stuff-500 dataset, real model weights, or a real GPU. No
`phase1b_conservation_calibration.json` produced by a real run exists.
`CUDA_VISIBLE_DEVICES=""` was set for every test in this suite, and
`torch.cuda.is_initialized()` is confirmed `False` both before and
after the full test run.

## What happens after this

1. Independently verify this runner (a second reviewer/session reading
   the code and re-running the test suite) before any commit.
2. Commit this runner to a tracked branch.
3. Run `preflight` against the committed HEAD on a real login/compute
   node to confirm the authority/dependency chain still verifies.
4. Separately request and receive GPU execution authorization (this
   task does not grant it).
5. Run the real 200-window calibration (`execute --device cuda
   --component-chunk-size <N>`) on an authorized GPU allocation,
   producing the real `phase1b_conservation_calibration.json`.
6. Independently verify the resulting artifact
   (`verify-artifact --geometry-json ...`).
7. Only then does a separate, future pre-registration step select and
   freeze a numeric conservation tolerance from the resulting raw
   residual distribution -- never chosen or implied by this runner.

## Production provider wiring

`diagnostics/run_dfp_phase1b_calibration.py`'s `execute` subcommand
wires real `GeometryProvider`/`ImagePreparer`/`InferenceProvider`
implementations, in `src/dfp_phase1b_calibration_production.py`, by
reusing -- never reimplementing -- the exact production entry points
DFP Phase 1A's own execution driver
(`src/dfp_phase1a_execute.py::execute_run`) already uses:

- `diagnostics.run_k11_k12_stability._build_dataset_only` /
  `_build_inference` -- real canonical-dataset and real model/weight
  construction, called exactly once each per `execute` invocation;
- `diagnostics.run_k11_k12_stability._extract_prepared_image` -- the
  canonical mmseg test-pipeline run for one image;
- `src.dfp_phase1a_dataset_view.build_manifest_execution_view` /
  `verify_position_identity` -- the frozen-manifest-order -> native-
  dataset-index adapter that already closed the exact 500-vs-5000
  positional-index defect this module must not reintroduce.

**Ordering.** `_cmd_execute` runs the exact same `run_preflight` the
`preflight` subcommand uses -- mode/device/chunk-size validation, E16
loading, output-path freshness -- entirely before importing
`src.dfp_phase1b_calibration_production` or any of its heavy (torch/
mmcv/model/dataset) dependencies. Candidate-universe construction and
the exact-200 selection (Phase B) run next, using only real image/
pipeline geometry (`_extract_prepared_image`, discarding each decoded
tensor immediately after reading its height/width) -- no model has
been constructed yet, so no model output can influence selection. Only
after selection completes is the real model built
(`build_production_inference`) and scientific execution
(`process_selected_image`, unmodified) run for every image containing
at least one selected window. Output freshness is re-checked
immediately before model construction and again immediately before
the final atomic install, closing the time-of-check-to-time-of-use
window a real multi-hour run leaves open.

## Weight resolution

Real model construction resolves the DINOv2 backbone and CLIP weights
via `src.local_weights.DEFAULT_WEIGHT_DIR`
(`os.environ.get("TALK2DINO_WEIGHT_DIR", "weights")`, read once at
that module's own import time). Unlike the E13-bound bridge checkpoint
(`weights/vitb_mlp_infonce_paired_soft_routing_tau010.pth`, hash-pinned
in `evaluation_identities/e13_dfp_phase1a_artifacts.toml`), the
committed E3 identity
(`evaluation_identities/e3_paired_soft_routing.toml`) names the
backbone/CLIP checkpoint files only by filename
(`[model].backbone_checkpoint_name`/`clip_checkpoint_name`) -- **no
SHA-256 binding exists anywhere in the committed authorities for
either file.** `verify_backbone_weights_present` therefore checks only
that the named files exist at `TALK2DINO_WEIGHT_DIR`; it never invents
a digest binding no committed authority defines. The calibration
launcher requires `TALK2DINO_WEIGHT_DIR` as an explicit environment
variable (never a hardcoded path) for exactly this reason.

## Synthetic testing versus real execution

`tests/test_dfp_phase1b_calibration_production.py` proves this wiring
end to end by driving the actual `_cmd_execute` CLI function after a
real preflight against the real, committed 500-image frozen manifest
-- only the dataset/model-construction entry points
(`_build_dataset_only`/`_build_inference`/`_extract_prepared_image`)
are replaced with small synthetic stand-ins (dependency injection at
exactly the real/synthetic boundary, never inside the CLI or the
scientific core). Candidate-universe construction, the exact-200
selection algorithm, `process_selected_image`, and artifact assembly/
install/validation all run unmodified. The synthetic fixture uses the
real, unweakened k=12/T=320/171-class/1024-node production scale for a
small number of selected windows (a test-only authority override of
the *window count* only, applied via monkeypatching, restored after
the test, and never touching the production `WINDOW_COUNT` default)
so this real computation stays appropriate for a shared login node.
This implementation has not accessed the real COCO-Stuff dataset, has
not loaded real weights, and has not initialized CUDA at any point.
