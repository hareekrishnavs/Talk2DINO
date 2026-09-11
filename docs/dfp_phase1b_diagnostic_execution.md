# DFP Phase 1B Task 6B — 500-Image Diagnostic Execution Contract (E20)

```text
THIS IS AN OFFLINE MECHANISTIC/PROVENANCE DIAGNOSTIC.
IT IS NOT THE DEPLOYED TEST-TIME METHOD.
EXACT MASKED-FORWARD PROVENANCE ON ALL 5,000 IMAGES IS NOT AUTHORIZED.
```

## 1. Purpose and scientific boundary

E20 (`evaluation_identities/e20_dfp_phase1b_diagnostic_execution.toml`)
freezes the **contract** — population, scientific-execution binding,
profiling contract, checkpoint/resume schema, and final-result bundle
schema — for one future, single-GPU, offline mechanistic/provenance
diagnostic run of the exact masked-forward provenance core over the
already-frozen 500-image COCO-Stuff subset.

It is an **additive successor authority only**. It:

* implements no execution driver, profiler, checkpoint writer, result
  writer, CLI execution command, or SLURM launcher;
* runs no inference and submits no GPU job;
* does **not** authorize the future 500-image run — only its contract;
* never redefines a scientific quantity already frozen by
  `e15-dfp-phase1b-protocol` / `e15-dfp-phase1b-artifacts` / the exact
  masked-forward core — it **binds** those schemas by content hash and
  adds only population-specific values (already fixed by E15/E16/E18/E19)
  and new **operational** (non-scientific) profiling/checkpoint-progress
  fields.

## 2. Exact dependencies

E20 relationally hash-verifies, on every load:

* `e15-dfp-phase1b-protocol`, `e15-dfp-phase1b-artifacts` — the Stage-A/
  Stage-B scientific schemas and checkpoint/path/atomic-write policy
  E20 binds rather than reinvents;
* `e16-dfp-phase1b-conservation-calibration` — the verified calibration
  artifact's own candidate-window-universe geometry;
* `e17-dfp-phase1b-chunk-benchmark`, `e18-dfp-phase1b-chunk-selection` —
  the frozen operational chunk size (16);
* `e19-dfp-phase1b-conservation-tolerance` — the frozen conservation
  tolerance (`9.5367431640625e-07` = `2**-20`);
* the frozen `coco_stuff_500` subset manifest;
* every scientific kernel module the masked-forward path calls: the
  sliding-window geometry module, the directed top-k graph module, the
  finite-step recurrence module, the recurrence-capture/stitching-
  pullback/final-rescale-pullback/horizon-stitching/global-unary-
  components/soft-membership modules, the dataset-view adapter, the
  masked-forward core, and the Phase 1B artifact-schema and calibration-
  artifact-validator modules;
* the verified calibration artifact and the installed Task 4 selection
  artifact + report, bound by content hash only (scratch-hosted, never
  a private path).

## 3. Population: 500 images / 1,090 windows

* **Images**: exactly 500, in the frozen `coco_stuff_500` subset
  manifest's own canonical (lexicographic fixed-width-ID) `selected_ids`
  order. Native-filesystem-index reconciliation reuses the same frozen
  `src/dfp_phase1a_dataset_view.py` adapter Task 4's own execution
  context relies on — never a fresh ad hoc mapping. No filesystem-order,
  dict-order, or hash-seed dependence.
* **Windows**: exactly 1,090 canonical sliding-window candidates,
  reconstructed from the **already-verified** calibration artifact's own
  `candidate_window_count` / `candidate_universe_digest` fields — the
  same fields E16's `[candidate_scope]` records for the identical
  500-image subset, produced by a pure per-image-canvas-dimension
  geometric computation (`crop=[448,448]`/`stride=[224,224]`) with GT/
  annotation/prediction/dispute/model-output influence forbidden. The
  200 calibration windows are a size-200 sample drawn from this same
  1,090-window universe (175 distinct images touched — consistent with
  a mean of ~2.18 candidate windows per image across all 500). This
  contract binds that persisted, digest-protected fact rather than
  re-deriving it from raw images, which would require dataset access
  this task never performs.

## 4. Frozen scientific execution binding

Reuses the exact masked-forward provenance core **unmodified**:
graph `k=12`, `320` recurrence steps, `alpha=0.98` (from E13, transitively
bound), canonical/masked recurrence dtype `float64`, graph storage
`float32`, residual accumulation `float64` (from the verified calibration
artifact's own recorded execution environment), component chunk size
`16` (E18), signed-contribution convention and final-resolution pullback
operator unchanged from E16's `rho_g` definition, and **conservation
tolerance `9.5367431640625e-07` (`2**-20`), inclusive comparison
`absolute_residual <= conservation_tolerance`** (E19). None of
component definition, query-class selection, prediction variants,
recurrence, contribution sign, resizing, stitching, the conservation
formula, the tolerance, or the chunk size is changed by this identity.

## 5. Profiling stages and boundaries

18 stages, each bound to an exact function/module boundary where one
already exists in this repository (see E20's
`[profiling_stage_contract.function_bindings]`). Two stages — the raw
model forward and model construction/checkpoint loading — have no
concrete function yet (Task 6B implements no execution driver) and are
explicitly **authority-bound** (to the E12/E3 authority chain) rather
than function-bound; every other stage binds to a real function:

| stage | scope | binding |
|---|---|---|
| authority_preflight_verification | run | this identity's own loader |
| dataset_index_and_manifest_adapter_construction | run | `dfp_phase1a_dataset_view.py` |
| model_construction_and_checkpoint_loading | run | authority-bound (E12/E3) |
| final_result_validation_and_installation | run | E15 `[atomic_write_policy].result_install` |
| per_image_checkpoint_serialization_and_atomic_installation | image | E15 `[checkpoint_resume_schema]` + `[atomic_write_policy].checkpoint_install` |
| input_preparation | window | `sliding_window_geometry.py` + dataset-view |
| unary_model_forward | window | authority-bound (E12/E3) |
| graph_construction | window | `graph.py::build_directed_topk_graph` |
| canonical_k12_recurrence | window | `dfp_phase1a_recurrence_capture.py::capture_dfp_phase1a_trajectory` |
| final_resolution_stitching_rescale_for_canonical_field | window | `apply_window_stitching_forward` + `apply_final_rescale_forward` |
| global_unary_component_construction | window | `compute_global_unary_components` |
| final_resolution_component_membership_pullback_construction | window | `build_final_stitching_pullback_plan` + `compute_window_final_soft_memberships` |
| component_masked_seed_construction | component_chunk | `dfp_phase1b_masked_forward.py::build_masked_seed_chunk` |
| masked_k12_recurrence | component_chunk | `propagate_masked_seed_chunk` |
| signed_component_contribution_computation | component_chunk | `compute_signed_contributions` |
| component_score_stitch_final_rescale | component_chunk | `stitch_and_rescale_component_scores` |
| conservation_residual_computation_and_validation | window | `build_conservation_measurements` + `validate_stage_a_flat_arrays` |
| scientific_row_array_assembly | window | schema-bound (E15 Stage-A member layout) |

**No double-counting.** Leaf-stage elapsed times are non-overlapping;
a parent/group aggregate (window total, image total, run total) is
inclusive and must never be summed with its own leaf totals it already
contains. Unattributed overhead = total wall time for a scope minus the
sum of that scope's own non-overlapping leaf-stage times; negative
overhead is invalid and rejected, never clamped to zero.

## 6. Timing semantics

CPU/end-to-end timing: `time.perf_counter_ns()`, integer nanoseconds,
measured immediately before stage entry and immediately after the
stage's required completion. CUDA timing: CUDA events on the active
execution stream, recorded immediately before/after the bound CUDA
work, synchronized only at the measurement boundary (never mid-stage,
never substituted with unsynchronized CPU wall time); elapsed time is
persisted as the float64 milliseconds the CUDA event API returns
directly, unrounded (a derived nanosecond integer, if also needed, uses
round-half-even and is recorded **alongside**, never instead of, the
float64 milliseconds). Warm-up CUDA work is recorded separately, never
merged into steady-state timing. Synchronization overhead is disclosed
as its own field. A CPU-only stage records `cuda_elapsed_ms = null`
(unavailable), never a fabricated `0.0`. Total run wall time is measured
independently of any stage-total reconstruction.

## 7. Host/GPU memory semantics

Host: `VmRSS` before/after each stage (`/proc/self/status`, exact
`bytes = KiB * 1024` integer conversion), `VmHWM` as the process's
cumulative high-water mark (never called a "stage peak"), and
`resource.getrusage(...).ru_maxrss` for the final process peak with an
explicitly recorded platform conversion. A field that could not be read
is `null` (unavailable), never `0`; a negative/non-finite/non-integer
parsed value is rejected. No background sampling interval is
authorized.

GPU (detailed profiling only): allocated/reserved bytes before and
after each stage, peak allocated/peak reserved bytes during the stage
(`torch.cuda.reset_peak_memory_stats` at stage entry,
`torch.cuda.max_memory_allocated`/`max_memory_reserved` at the defined
measurement boundary), device index and model. Resetting counters never
frees or moves memory; values are operational metadata only. CPU-only
stages record `null`, never `0`. `nvidia-smi` process-wide values, if
ever also recorded, are a distinct, separately-labeled field — never a
silent replacement for the allocator-reported values.

## 8. Profiling aggregation

`count, sum, mean, min, max, median, p90, p95, p99`, with a
library-independent **nearest-rank** percentile: sort ascending,
`rank = max(1, ceil(p * n))`, `value = sorted[rank - 1]`. Zero records:
`count=0`, every statistic `null`. One record: every statistic equals
that value. A record with `cuda_elapsed_ms = null` is excluded from
CUDA-timing aggregates but still counted. Image-scoped, window-scoped,
and component_chunk-scoped records for the same stage are **never**
averaged together — aggregation groups strictly by `(stage_name,
scope)`. NaN/±infinity are rejected at record-validation time. Two
records sharing the same canonical identity `(stage_name, scope,
scope_covariate_key)` are rejected as duplicates. Workload covariates
(image/window index and ID, windows-per-image, component/query-class/
chunk counts, node count, disputed-pixel count, chunk size) are
recorded for interpretation only — no new scientific statistic is
introduced for profiling's sake.

## 9. Checkpoint / resume

The checkpoint schema (`talk2dino-dfp-phase1b-diagnostic-checkpoint-v1`)
binds E15's own `[checkpoint_resume_schema]`, `[path_policy]`, and
`[atomic_write_policy]` rather than inventing an alternative, and adds
identity bindings (E20/E15/E18/E19/subset/window-universe hashes,
run mode, device contract, chunk size, tolerance), progress
(`expected_image_count=500`, `expected_window_count=1090`,
`next_canonical_image_index`, `completed_image_ids` as an exact frozen-
manifest-order prefix, `completed_window_count`,
`total_processed_component_chunks`, `generation`, `complete`,
`final_result_produced`), scientific state (Stage-A artifact filename/
schema/row-count/hash/digest — never alternative scientific data), and
profiling state (kept fully separate; **never** part of the scientific
digest).

Resume rules: `fresh` requires no existing checkpoint/artifacts unless
separately authorized to overwrite; `resume` requires a valid incomplete
checkpoint whose bindings match exactly and whose `completed_image_ids`
is an **exact prefix** of the frozen manifest (no skip/duplicate/
reorder/cross-feed); partial generations, corrupted hashes, and
unknown/v1 schemas are rejected; a complete-but-unfinalized checkpoint
may enter a finalize-only path; an already-finalized run is rejected
for resume; scientific-state restoration is exact.

## 10. Fresh/resume scientific equivalence

On the same deterministic fixture, fresh scientific arrays must equal
checkpoint-then-resume scientific arrays bit-for-bit where the existing
schema promises bit identity — row order, selected IDs, window order,
offsets, flat arrays, the scientific collection digest, and the final
scientific result document excluding explicitly operational fields.
Profiling records may differ in wall time/memory but must preserve
unique record identities, no duplicate completed-stage records after
resume, correct interrupted-attempt labeling, and aggregate
reconstruction from accepted (non-incomplete) records only. An
interrupted partial stage record is stored as an explicit `incomplete`
record — never discarded silently, never promoted to a completed
timing sample.

## 11. Profiling-on/off scientific equivalence

`scientific_output_collection_digest(profiling=off) ==
scientific_output_collection_digest(profiling=detailed)`. Profiling
fields, timestamps, memory measurements, device metadata, and execution
durations are excluded from the scientific digest. Profiling may
observe execution but must never change inputs, chunking, operation
order, precision, autocast, deterministic settings, seeds, scientific
serialization, row selection/rejection based on timing, or checkpoint
scientific state.

## 12. Schemas and filenames

* `talk2dino-dfp-phase1b-diagnostic-checkpoint-v1`
* `talk2dino-dfp-phase1b-diagnostic-profiling-v1` (`operational_only=true`, `included_in_scientific_digest=false`)
* `talk2dino-dfp-phase1b-diagnostic-result-v1`

Canonical filenames (safe basenames under one run-specific output root
— never an absolute or private path in any authority document):
`phase1b_diagnostic_checkpoint.json`, `phase1b_diagnostic_stage_a.npz`,
`phase1b_diagnostic_profiling.json`, `phase1b_diagnostic_result.json`.

## 13. Atomicity

Binds E15's own `[path_policy]` (lexical `..` rejected, physical
containment required, symlink-escape rejected, no private cluster/user
path) and `[atomic_write_policy]` verbatim: exclusive fresh creation by
default, same-filesystem staging, fsync before atomic rename, verified
rollback for multi-file replacement, no orphaned partial final bundle,
source checkpoint/scientific artifacts immutable during finalization.
No installer is implemented by this task — only its required behavior
and schemas are frozen.

## 14. The 8-hour allocation field

`within_8_hour_allocation = total_job_wall_time_seconds <= 28800`. This
is an **operational disclosure**, not a scientific acceptance gate — a
scientifically valid result does not become numerically invalid merely
because runtime exceeded the budget. The future launcher must
separately enforce the 8-hour allocation limit at the SLURM level.

## 15. No 5,000-image exact provenance

This contract governs the 500-image subset only. It explicitly does
**not** authorize exact masked-forward provenance on the full 5,000-
image validation set, nor a full 5,000-image mIoU evaluation, nor
selection of a lightweight final method. `E20 CLI` documents state
`offline_mechanistic_analysis=true`, `deployed_inference_method=false`,
`full_5000_image_evaluation=false` in every valid final-result document.

## 16. CLI

```text
verify-protocol             verify the E20 identity
verify-checkpoint-document  validate a checkpoint JSON document
verify-profiling-document   validate a profiling JSON document
verify-result-document      validate a final-result JSON document
show-contract                print the frozen population/device/tolerance/chunk-size/8-hour contract
```

CPU-only; never imports torch; never initializes CUDA; never accesses a
dataset image; never loads a model checkpoint. Runs from repository root
or any unrelated working directory; exits 0 on valid documents, exits 2
with a clean domain-specific message and no traceback on ordinary
invalid input.

## 17. Future implementation and authorization steps

This identity is a contract only. Implementing it — the execution
driver, profiler, checkpoint writer, and result writer — and then
explicitly, separately authorizing the 500-image diagnostic execution
itself, are both later, separately-authorized tasks. The 5,000-image
exact-provenance question and the full 5,000-image mIoU evaluation are
never implied or authorized by this identity.
