# Semantic-evidence discovery execution runner

This document explains the implementation authorized by
`research_specs/semantic_evidence_discovery_execution_v1.toml` under
`TASK: IMPLEMENT_48_IMAGE_SEMANTIC_EVIDENCE_DISCOVERY_RUNNER`, and repaired
under `TASK: REPAIR_48_IMAGE_DISCOVERY_RUNNER_STATISTICAL_WIRING_AND_REAL_
LOADER_SHAPE`: the 7 files that actually run the committed 48-image
discovery experiment.

## What this is, and is not

This is **infrastructure**, not a completed experiment, and not a claim of
discovery effectiveness. Implementing and repairing these 7 files did not
execute anything: every test in
`tests/test_semantic_evidence_intervention_features.py` and
`tests/test_semantic_evidence_discovery_execute.py` runs against synthetic
data through fake providers, never a real model, checkpoint, image, or GT
annotation, and never initializes CUDA. No job was submitted by this task
or its repair, and this repair does not authorize real execution or GPU
submission by itself -- the same authorization sequence in
`docs/semantic_evidence_discovery_execution.md` still applies.

## What the repair changed

Independent verification of the first implementation found five mechanical
defects, all now repaired:

1. **Real image loader shape**: `_default_real_image_loader` now returns a
   contiguous `[1, 3, H, W]` tensor (the batch dimension is applied exactly
   once) plus `(height, width)`, matching what `process_window_gt_free`'s
   4-index crop slicing requires. It validates rank/batch-size/channel-
   count/finiteness before returning, rejects unauthorized IDs via
   `validate_discovery_image_id` before any filesystem access, and
   correctly propagates `MemoryError`/`KeyboardInterrupt`/`SystemExit`
   (the ordinary-exception branch now calls `_reraise_if_fatal` before
   translating anything into `DiscoveryExecutionError`, matching the
   sibling GT loader's own pattern).
2. **Per-row GT-free feature persistence**: the `gt_free_features`
   artifact's already-declared `intervention_view_scores` and
   `control_features` containers (frozen by the committed authority's own
   `artifact_contract`, never a new field) now hold the complete per-row
   content needed to reconstruct the frozen Model-A/Model-B design
   matrices: raw intervention scores, the exact 7-feature Model-A vector
   (controls 1,2,3,4,5,6,8), intervention reliability, anchor grid
   position, and -- for propagation-eligible anchors -- T0/T320
   predicted-class identifiers. `intervention_reliability` changed from a
   bare, unlinked list to a dict keyed the same way, so every value is
   traceable to its exact row.
3. **Four frozen logistic fits**: `run_discovery` now actually builds the
   four design matrices from the persisted per-row features and calls
   `fit_logistic_model` exactly four times (query-correctness Model A/B,
   propagation-outcome Model A/B), persisting `model_fit_summaries` with
   task/model IDs, feature order, row/class counts, coefficients,
   intercept, and convergence status. Fitting fails closed (raises inside
   a caught boundary, leaving `model_fit_summaries: []` and a
   `DISCOVERY_INCONCLUSIVE` decision) on an empty or single-class
   population, a wrong feature-vector length, a non-finite feature value,
   solver non-convergence, or non-finite fitted parameters -- never
   fabricates a result.
4. **Gate A accounting**: `compute_gate_a` takes `total_anchors` and
   `structurally_invalid_anchor_count` (both anchor-level; there is no
   image-count parameter for a caller to substitute by mistake) and a
   `non_finite_scores_count` that is always an *observed* aggregate from
   `process_window_gt_free`'s own per-window counters
   (`raw_score_non_finite_count`, `raw_score_out_of_range_count`,
   `derived_score_invalid_count`, `attempted_anchor_count`,
   `accepted_anchor_count`, `excluded_no_valid_control_count`,
   `structurally_invalid_anchor_count`, `query_row_count`,
   `propagation_row_count`) -- never a literal constant.
5. **Strict nested artifact validation**: `validate_gt_free_artifact` and
   `validate_gt_label_artifact` now recursively check every nested
   key set, type (rejecting bool-as-int), numeric range, ID format, and
   row-key linkage, and cross-verify the persisted digest against a
   digest recomputed directly from the artifact's own content. A new
   `validate_gt_pair_linkage` checks that every GT-label row traces back
   to a real GT-free row and that every GT-free row is either labeled or
   marked ambiguous. All of this runs at every stage the committed
   authority requires: before staging, after the staged reopen, after
   atomic installation, and through the public `verify-result` path.

## The 7 files

1. `src/semantic_evidence_intervention_features.py` -- pure, torch-using
   (no model, no CUDA) scientific/geometry logic: anchor selection, query
   construction, the four frozen intervention views (`ORIGINAL`,
   `CONTEXT_BLURRED`, `REGION_BLURRED`, `AREA_MATCHED_CONTROL`), Gaussian
   blur, the score-validity check, the derived-feature formulas, all 7
   available control features (id 7 is `NOT_AVAILABLE_IN_E3`, never
   implemented), row assembly, and the GT-free feature digest. Every
   constant carries an inline comment citing the exact TOML key it comes
   from.
2. `src/semantic_evidence_discovery_execute.py` -- the execution engine:
   torch-free preflight, population loading (from the committed authority's
   own dependency-bound manifest, never an operator-supplied image list),
   real and fake provider interfaces, the Phase A (GT-free) / Phase B
   (GT-dependent) per-image pipeline, the fresh/resume/finalize_only
   checkpoint state machine, artifact schema validators, the 4 frozen
   logistic-regression fits, Gate A/E computation, the discovery-only
   decision, and atomic staged-write/atomic-rename/reopen-and-reverify
   installation.
3. `diagnostics/run_semantic_evidence_discovery.py` -- the CLI:
   `verify-protocol`, `preflight`, `execute`, `verify-result`. No
   scientific-parameter, holdout, debug, or overwrite/force flag exists on
   any subcommand.
4. `semantic_evidence_discovery_h100.sbatch` -- the launcher: 1x H100 80GB,
   `--time=02:00:00` (matching the frozen Gate E ceiling), no
   self-submission, same venv/module-load precedent as the committed smoke
   launcher.
5. `tests/test_semantic_evidence_intervention_features.py` -- 49
   independent-oracle tests (formulas recomputed with separate arithmetic,
   never by calling the function under test to produce its own "expected"
   value).
6. `tests/test_semantic_evidence_discovery_execute.py` -- fake-provider
   tests covering population/ID validation, checkpoint construction and
   validation, the real image loader's shape/authorization/exception-
   propagation behavior (via small locally-generated synthetic JPEGs),
   Gate A boundaries and anchor/image unit-consistency, the four frozen
   fits (including a direct sklearn cross-check and fail-closed cases),
   GT majority-vote labeling, propagation-category taxonomy, strict
   nested artifact validation with adversarial mutations, cross-artifact
   GT-free/GT-label linkage, atomic install/overwrite-refusal, full
   `run_discovery` integration (fresh, interrupted+resume, per-phase
   resume after a checkpoint- or label-write failure, finalize_only,
   GT-isolation), and defect-reinstatement tests for the repair's own
   five findings.
7. This file.

## How Phase A (GT-free) and Phase B (GT-dependent) are wired

Phase A and Phase B are now two separate functions, `_run_phase_a` and
`_run_phase_b`, connected only through the already-validated, persisted
`gt_free_features` artifact -- `_run_phase_b` takes that artifact (freshly
reloaded from JSON, or handed straight from Phase A within one process)
plus a GT loader, and nothing else. It never receives an image loader,
model, text embedding, or forward function, so it cannot recompute
original/intervention views, E3 features, T0/T320, controls, or
intervention reliability even by accident -- there is no plumbing for any
of that to arrive through. This split is also what makes per-image
resumability possible (see below).

- **Phase A** (`_run_phase_a`): load the image; compute its canonical
  windows (reusing `semantic_evidence_t320_real_smoke.py`'s own
  `compute_canonical_windows` tiling, reimplemented identically here since
  that module is a real-smoke runner this authority does not depend on);
  for each window, perform exactly **one** canonical E3 forward via the
  committed, unmodified `semantic_evidence_e3_adapter.extract_e3_window_
  snapshot` (the reused ORIGINAL forward, per `[one_forward_reuse_
  contract]`); select up to 8 anchors and their queries; for each accepted
  anchor, perform exactly **3** additional E3 forwards, one per
  non-`ORIGINAL` intervention view (never reused across anchors, per the
  same contract); compute raw scores, derived features, and all available
  control features, tracking a primary rejection-reason counter for every
  anchor/row that does not produce a valid feature vector; call the
  committed, unmodified `semantic_evidence_t320.generate_t0_t320_targets`
  once per image (using only the canonical ORIGINAL-view snapshots) to
  obtain the image's T0/T320 predicted-class canvases -- a model
  prediction, not a GT comparison, so this stays inside Phase A; for every
  propagation-eligible anchor, read the **single center pixel** of its
  14x14 footprint from each of the T0/T320 canvases as that anchor's
  `t0_predicted_class`/`t320_predicted_class` identifier (a center-pixel
  read, not a majority vote -- model predictions are defined at every
  pixel, so there is no "ambiguous prediction" concept the way there is
  for GT's ignore label; a majority-vote reduction applied to i.i.d.
  per-pixel predictions has no reliable winner and must not be used here).
  The complete GT-free artifact -- raw scores, feature vectors, window
  boxes, T0/T320 identifiers, and the Gate-A counters -- is assembled,
  digested (the digest is computed directly from the persisted content, so
  it can be independently recomputed from a reloaded JSON file alone),
  validated, and returned. Nothing in this phase reads GT.
- **Phase B** (`_run_phase_b`): takes an already-validated `gt_free_
  features` artifact and a GT loader. Opens the image's GT raster (a
  COCO-Stuff-style semantic PNG, `_default_real_gt_loader`) only now. For
  each persisted row, its anchor's single-patch (14x14 pixel) footprint is
  mapped through the window box already recorded in the GT-free artifact
  (`footprint_pixel_bounds`) -- windows are already expressed in
  full-canvas pixel coordinates, so no separate rescale is needed.
  `compute_gt_majority` applies the frozen `min_valid_footprint_
  fraction=0.75` / `min_majority_share=0.80` rule to the **GT** raster only
  (`AMBIGUOUS_GT` anchors are excluded from both tasks). Query-correctness
  labels compare the row's query class against the GT majority class.
  Propagation-outcome categories apply `compute_propagation_category`'s
  exact 5-way MECE taxonomy (`beneficial` / `harmful` / `unchanged_correct`
  / `changed_but_same_correctness` / `unchanged_incorrect`) to the
  already-persisted T0/T320 predicted-class identifiers against the GT
  majority class -- Phase B never touches the T0/T320 canvases themselves,
  only the two small integers Phase A already reduced them to.

A `PhaseEventLog` records `phase_a_start`/`phase_a_complete`/
`gt_access_start`/`phase_b_complete` events per image and asserts, per
image, that no `gt_access*` event was recorded between that image's own
`phase_a_start` and `phase_a_complete` -- catching any future refactor that
accidentally moves GT access earlier. Independently of that runtime check,
`process_window_gt_free` (Phase A's own per-window function) has **no**
GT-related parameter in its signature at all, so Phase A code has no
plumbing through which a GT loader could arrive even if a caller tried.

## Checkpoint / resume / finalize-only

`fresh` mode creates a `<output_root>.inprogress` checkpoint directory
(refusing to start if either that directory or `output_root` itself
already exists) and processes discovery images in the frozen population
order. For each not-yet-completed image, the per-image loop now checks
**per phase**, not just per image, before doing any real work:

1. If a valid `gt_free_<id>.json` already exists on disk (left over from a
   prior interrupted attempt -- e.g. a crash between finishing that file
   and updating `checkpoint.json`), it is reused and Phase A is skipped
   entirely: no image loader or model call for that image.
2. If a valid `gt_label_<id>.json` also already exists (and its linkage to
   the current `gt_free_<id>.json` checks out -- a stale label from a
   since-regenerated GT-free artifact is detected and discarded), it is
   reused too and Phase B is skipped: no GT loader call.
3. Only the phases that are actually missing or invalid are (re)run, each
   written with an atomic rename, before `checkpoint.json` is atomically
   updated with the new completed-ID prefix.

The real model/text-embedding provider is now constructed lazily -- only
the first time some image in the loop actually needs Phase A -- so a
resume where every remaining image's Phase A is already on disk never
constructs a real model at all.

`resume` mode requires the checkpoint directory to exist and `output_root`
to be absent, reloads the existing checkpoint, and continues from the
first not-yet-completed image using the per-phase reuse logic above.
`finalize_only` mode requires every discovery image already marked
complete in the checkpoint and never touches the image/GT/model
providers -- verified directly by tests whose fake providers raise if
called. Fitting/gating/final-artifact assembly happens identically
regardless of which mode reached that point: it operates only on the
persisted, reopened, re-validated per-image JSON files, so `finalize_only`
rebuilds the four fits purely from files already on disk. Installation is
always the same staged-write, staged-reopen-and-validate (now including
every `gt_free`/`gt_label` artifact and their cross-linkage, not just the
top-level result/decision documents), atomic-rename,
reopen-and-publicly-reverify sequence (`install_discovery_result`), which
also refuses to install over an existing `output_root`.

An independent test (`test_interrupt_then_resume_matches_uninterrupted_fresh`)
runs a fresh execution that is force-interrupted after the first of two
images, resumes it, and asserts the resulting `gt_free_feature_digest` and
`gt_dependent_label_digest` are byte-identical to an uninterrupted fresh
run over the same two images -- the committed authority's own
`determinism_identity` requirement, checked directly rather than assumed.

## Statistical fitting

`run_discovery` aggregates every completed image's persisted GT-free
feature vectors and GT-dependent labels into exactly two row populations:
`query_correctness` (every non-ambiguous query-correctness row, both `q1`
and `q2`) and `propagation_outcome` (every `q1`-only propagation row whose
category is `beneficial` or `harmful` -- `unchanged_correct`,
`unchanged_incorrect`, and `changed_but_same_correctness` rows are excluded
from the binary fit, since there is no beneficial/harmful winner to
predict for them). For each population it calls `fit_logistic_model`
twice, once with the frozen 7-feature Model-A vector and once with that
same vector plus `intervention_reliability` appended (Model B) -- exactly
four calls, never per image, per checkpoint generation, or per bootstrap
replicate. Every fit uses the frozen `solver="liblinear"`, `C=1.0`,
`penalty="l2"`, `class_weight="balanced"`, `max_iter=1000`,
`random_state=0` settings and standardizes on the fitted rows only, and
fails closed (raising, caught by `run_discovery`, leaving
`model_fit_summaries: []` and forcing `DISCOVERY_INCONCLUSIVE`) on an
empty or single-class population, a row/feature-vector length mismatch, a
duplicate row key, a non-finite feature value, solver non-convergence, or
a non-finite fitted coefficient/intercept/prediction. No holdout row is
ever read, and no holdout performance estimate is ever computed or
persisted -- only the fitted discovery-side parameters.

## Gates

Only Gate A (intervention/structural validity) and Gate E (compute
feasibility, projected from this run's own Slurm elapsed time to a
96-image equivalent, never from the smoke's already-disclosed-defective
profiling) are computed. Gate A's accounting unit is the **anchor**,
never the image: `total_anchors` is the sum, across every window of every
completed image, of `process_window_gt_free`'s own observed
`attempted_anchor_count`; `structurally_invalid_anchor_count` is the same
sum of anchors that were excluded for having no valid control region, or
that produced zero valid query-correctness rows after raw/derived-score
validation. `non_finite_scores_count` is likewise the observed sum of
`raw_score_non_finite_count` across every window -- never a literal
constant. Gates B, C, and D are always recorded as
`NOT_COMPUTABLE_WITHOUT_HOLDOUT` in the installed decision document --
`validate_decision_document` refuses to install any decision document
where they are not exactly that literal status string, so a future bug
that accidentally computes them cannot silently reach an installed
artifact. The discovery decision
(`DISCOVERY_FEATURES_READY_FOR_HOLDOUT_AUTHORITY` /
`DISCOVERY_STOP_GATE_A_FAILED` / `DISCOVERY_INCONCLUSIVE`) is never the
feasibility contract's own `[gates.outcome]` proceed/stop decision, and it
is never reachable from Gates B/C/D -- `overall_gates_outcome_status` is
always installed as `PENDING_HOLDOUT_EVALUATION`. A successor holdout
authority remains a separate, later, explicitly-authorized task; nothing
in this repair changes that boundary.

## What this task did not do

No job was submitted. No real model was constructed, no real checkpoint
was loaded, no real image or GT annotation was read, and CUDA was never
initialized, anywhere in this implementation task -- confirmed by every
test in the two new test files using only fake providers and synthetic
tensors, and by `verify-protocol`/`preflight` never importing torch at
all. A discovery execution requires an operator to independently decide to
run `sbatch semantic_evidence_discovery_h100.sbatch` after re-verifying
committed-HEAD readiness; this document does not constitute that decision
and does not claim any discovery result exists.
