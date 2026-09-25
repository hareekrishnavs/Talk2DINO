# Semantic-evidence discovery execution authority

This document explains `research_specs/semantic_evidence_discovery_execution_v1.toml`
and its loader, `src/semantic_evidence_discovery_execution_identity.py`, created by
`FREEZE_48_IMAGE_SEMANTIC_EVIDENCE_DISCOVERY_EXECUTION_AUTHORITY`.

## What this is

This is a **bounded discovery experiment authority**, not an implementation and
not an execution. It converts three already-committed, already-verified pieces
of work into one mechanical execution contract for a future discovery-phase
runner:

1. `research_specs/semantic_evidence_feasibility_v1.toml` -- the frozen
   feasibility contract: population, anchors, queries, interventions, Gaussian
   blur, score formulas, statistical design, and Gates A-E.
2. `research_specs/semantic_evidence_t320_source_v1.toml` +
   `src/semantic_evidence_e3_adapter.py` + `src/semantic_evidence_t320.py` --
   the historical T320 source authority and its isolated, real-model-verified
   implementation.
3. Job `22640114`'s installed `smoke_result.json` -- the independently
   verified real E3/T320 integration smoke that fulfills the feasibility
   contract's own `[propagation_target_authority].required_successor_action`.

No discovery execution, image access, GT access, or job submission occurred
in the task that created this authority. Creating this authority means
exactly one thing: the scientific contract for a discovery-phase run now
exists, hash-bound and mechanically verifiable, and nothing more.

## Why the two-image smoke is sufficient to end infrastructure validation

Six independent real-path defects were found and repaired across the smoke
validation effort: launcher path resolution under real `sbatch`, the wrong
Python interpreter, an 80-class/171-class import bug, an E3 projection
dispatch fallthrough, a backbone-canvas/source-window geometry conflation,
and a tensor-contiguity divergence in `internal_upsample`. Job `22640114` is
the first run in that sequence to complete with `PASS_REAL_E3_T320_INTEGRATION_SMOKE`,
bitwise-exact T0 preservation on all four windows, and deterministic repeat
predictions. Each defect it exercised was a real-model-only failure mode,
invisible to every synthetic/fake-model test that preceded it. A further
two-image (or larger) smoke would exercise the same already-repaired code
paths without covering any new failure mode; the two smoke images are, not
coincidentally, discovery-population rank 0 and rank 1, so this smoke is
already evidence the discovery split itself is real-model-executable.

## Why smoke profiling is not suitable for runtime claims

The smoke authority's own `telemetry_contract` requires `torch.cuda.synchronize()`
immediately before and after every measured GPU region, and
`torch.cuda.reset_peak_memory_stats()` at a frozen reset point. Neither call
exists anywhere in the smoke runner. `graph_construction_wall_time_ns` and
`recurrence_wall_time_ns` are an arbitrary 50/50 split of one combined timer;
`stitching_finalization_wall_time_ns` and `artifact_validation_install_wall_time_ns`
are hardcoded to `0`; `gpu_model`/`cuda_runtime_version`/`cuda_driver_version`
are hardcoded empty strings. These were found and disclosed, never repaired,
by the independent verification of job 22640114's result. This authority
records `smoke_profiling_status = "NOT_AUTHORITY_FOR_RUNTIME_OR_THROUGHPUT_CLAIMS"`
and explicitly does not inherit these defects: the future discovery runner's
own resource contract (see below) retains only Slurm elapsed time as
non-scientific operational metadata, never a scientific field, never a gate
input.

## The exact 48/48 split boundary

The frozen 96-image population (`evaluation_subsets/semantic_evidence_feasibility_96.json`)
splits by `selection_rank`: ranks 0-47 are the **discovery** population (48
images), ranks 48-95 are the **locked holdout** (48 images), with zero
overlap -- independently re-verified by this authority's own loader against
freshly recomputed SHA-256 digests, not trusted from any prior report.

This authority's `execution_population = "discovery_only"`. The locked
holdout's image IDs are never enumerated in this document or in the TOML;
only its digest and count are bound. The future runner, CLI, and launcher
this authority permits must have no code path capable of selecting the
holdout, all 96 images, a debug/max-images subset, or the full 5,000-image
validation set.

## Intervention hypothesis

Can controlled region-preserving and region-removing image interventions
(`ORIGINAL`, `CONTEXT_BLURRED`, `REGION_BLURRED`, `AREA_MATCHED_CONTROL`)
identify whether a local visual region provides reliable evidence for a text
concept, beyond ordinary confidence/margin/entropy/neighbor-agreement
features -- particularly for identifying anchors where T0 (the unmodified
E3 prediction) is wrong but T320 (the 320-step graph-propagated prediction)
is correct? This is a **feasibility experiment**, not a full method
evaluation and not an mIoU claim.

## Two statistical tasks

- **query_correctness** (unit: anchor-query, up to 2 rows per accepted
  anchor): does `intervention_reliability` improve prediction of whether a
  query class is correct, beyond 7 ordinary control features?
- **propagation_outcome** (unit: anchor, 1 row per eligible anchor, `q1`
  only): does `intervention_reliability` improve prediction of whether T320
  is a *beneficial* correction of T0 (T0 wrong, T320 correct) versus
  *harmful* (T0 correct, T320 wrong)?

Both tasks fit four logistic-regression models (control-only and
control-plus-intervention, per task) exactly as frozen by the feasibility
contract's `[statistics]` tables -- `C=1.0`, `l2`, `balanced`, `liblinear`,
`max_iter=1000`, `random_state=0`, standardized on discovery rows only.

## GT isolation

A strict two-phase boundary is frozen. Phase A (GT-free): image load, window
construction, one E3 forward per window, T0/T320 targets, anchor/query
selection, all four intervention views, all scores and
`intervention_reliability`, persisted and digested. Phase B (GT-dependent):
opened only after Phase A for that image is complete and immutable; computes
correctness labels and propagation-outcome categories. A GT-free feature
digest and a GT-dependent label digest are required as separate fields; a GT
mutation must never change the GT-free digest.

## Artifact lifecycle

Fresh / resume / finalize-only modes; no overwrite, no force; exact
completed-ID prefix validation; staged write, staged reopen and validation,
atomic directory-rename installation, post-install reopen, and public
re-verification before any success state. `clean fresh == interrupted+resume
== complete-but-unfinalized+finalize-only` for every scientific digest and
decision field.

## Gates A-E and why this authority cannot reach a go/stop decision

Gates B, C, and D, and the feasibility contract's overall
`[gates.outcome]` proceed/stop decision, all structurally require
locked-holdout evaluation (`[gates.B]`/`[gates.C]` explicitly evaluate on
`locked_holdout` rows; `[statistics.fit_evaluate].evaluate_partition =
"locked_holdout"`). Holdout access is out of scope for this authority.

This authority therefore freezes only Gate A (intervention validity) and
Gate E (compute feasibility) as computable from discovery-side data now.
Gates B, C, and D are recorded as `NOT_COMPUTABLE_WITHOUT_HOLDOUT` -- never
silently passed, skipped, or assumed. All four logistic-regression models
*are* fit on the discovery partition by the future implementation this
authority permits; only their holdout-side *evaluation* is deferred.

This authority's own discovery-phase decision
(`DISCOVERY_FEATURES_READY_FOR_HOLDOUT_AUTHORITY` /
`DISCOVERY_STOP_GATE_A_FAILED` / `DISCOVERY_INCONCLUSIVE`) is **not** the
feasibility contract's `[gates.outcome]` proceed/stop decision. A discovery
PASS authorizes only the creation of a later, separate successor holdout
authority -- never holdout execution, never the full 96-image gate decision,
never any accuracy or mIoU claim, never a paper claim. A discovery STOP ends
this research direction under the current frozen hypothesis.

## Compute ceiling

One H100 80GB only, inherited from the feasibility contract's Gate E
(`max_h100_hours_96_image_feasibility_run = 2.0`,
`max_h100_hours_1000_image_projection = 8.0`) and the smoke launcher's own
resource precedent. Multi-GPU, alternate GPU types, login-node execution,
distributed execution, training, and any larger population are prohibited.

## Authorization sequence

1. This authority is independently verified (`READY_FOR_INDEPENDENT_REVERIFY`
   -> `PASS`) and committed.
2. A future task implements the discovery runner/CLI/launcher named in
   `[future_implementation_files]`, and that implementation is itself
   independently verified and committed.
3. Committed-head production readiness is re-verified.
4. An operator explicitly submits the job.

None of these four conditions is satisfied by this task. This task creates
exactly four files and executes nothing.

## Explicitly prohibited by this authority

The locked 48-image holdout, all 96 images, the full 5,000-image validation
set, arbitrary/debug/max-images subsets, confirmation images, training,
parameter updates, multi-GPU or non-H100 execution, login-node execution,
overwrite/force, and launcher self-submission are all frozen as
unauthorized, both for this task and for the future implementation this
authority permits.
