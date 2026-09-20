# DFP Phase 1B: E27 T64 early-exit benchmark execution authority

E27 is a narrow **execution-only** successor to E26. It authorizes
exactly one future H100 wall-clock benchmark campaign using the
already-implemented, already-independently-verified, already-committed
runner/CLI/launcher (bound below by exact SHA-256 to commit
`af92d4acc64254674d2156a13b55bf4f5d368d3d`). It defines no new
science, no new artifact schema, and no new implementation requirement
-- E25 remains the sole source of the benchmark's variants, population,
schedule, timing/memory/digest contracts, and decision rule; E25's
artifacts identity remains the sole source of the trial/result/decision
schemas; E26 remains the sole source of implementation requirements.
E13 through E26 are unchanged by this authority.

## 1. Why E27 exists, and what it does not claim

The committed CLI and launcher never read E26's own `authorization_boundary`
at runtime, and they never read this file either -- there is **no
code-level gate inside the existing, unmodified launcher or CLI** that
consults an authorization flag before executing. This was confirmed
directly during this authority's own creation: `grep -rn
authorization_boundary src/dfp_phase1b_t64_early_exit_benchmark.py
diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py
dfp_phase1b_t64_early_exit_benchmark_h100.sbatch` returns nothing.

E27 does not, and in this task cannot, change that: the launcher and
CLI are existing, committed files this task does not modify. Instead,
E27 adds a **mandatory, separate, read-only, torch-free verification
command** (`verify_execution_readiness`, exposed both as a Python
function and as this module's own `python3 -m
src.dfp_phase1b_t64_early_exit_benchmark_execution_identity` entry
point) that an **operator** must run, and receive a PASS from, before
manually invoking the unmodified launcher. This is an operator-process
prerequisite, never a runtime check embedded in the launcher itself --
this document and E27's own TOML say so explicitly, so no reader
mistakes the arrangement for automatic runtime enforcement.

Ordinary public loading (`load_identity`) can inspect an uncommitted
E27 authority's own internal schema/hash consistency at any time --
this alone never means execution is "activated"; see
`activation_prerequisites` below.

## 2. Files

| File | Role |
|---|---|
| `evaluation_identities/e27_dfp_phase1b_t64_early_exit_benchmark_execution.toml` | The frozen E27 identity: 21 dependency bindings (including the committed runner/CLI/launcher bytes, E25/E26's own identities/loaders, and the BALP split-loader module plus its two source data files -- see below), the exact resource/variant/population/schedule/campaign-workload execution scope, the dataset-access scope with an explicit population-overlap disclosure, the existing output layout/environment-variable bindings, re-pinned measurement/decision bindings, the authorization boundary, the activation-prerequisite chain, and the execution-readiness contract. |
| `src/dfp_phase1b_t64_early_exit_benchmark_execution_identity.py` | Strict loader/validator for E27 (closed schema, exact-value checks, relational hash reverification, repository-ancestor gate, no-circular-self-hash check) plus `verify_execution_readiness`, the mandatory separate readiness command. Torch-free throughout. |
| `tests/test_dfp_phase1b_t64_early_exit_benchmark_execution_identity.py` | Exhaustive E27 identity tests: closed schemas, dependency hashes (including tampered-bytes and symlink-escape fixtures), path containment, exact resource/population/schedule values, authorization-boundary exactness (all 21 flags individually toggled), activation-prerequisite exactness, minimal-git-fixture ancestor/dirty-tree/staged-change tests, malformed types, domain-error/fatal-exception behavior. |
| `docs/dfp_phase1b_t64_early_exit_benchmark_execution.md` | This document. |

## 3. What E27 conditionally authorizes, and what remains forbidden

Once its own `activation_prerequisites` are all satisfied (see Â§6),
E27 authorizes exactly: scoped GT-free dataset/image access limited to
the 25 manifest image IDs, bound model construction and checkpoint
loading, CUDA initialization, the one H100 benchmark campaign
described below, and artifact verification/merge.

Never authorized, in this task or by E27 itself: agent job submission
during this task, GT access, training, weight modification,
scientific-kernel modification, alternate populations, extra trials,
confirmation accuracy evaluation, full 5,000-image evaluation, BALP
candidate evaluation, exact masked-forward provenance, overwrite/force,
E25/E26 modification, or creating another authority merely to bypass
these prerequisites.

## 4. Execution scope

**Resources** (copied verbatim from the committed launcher's own
`#SBATCH` header and the existing GPU-model-matching convention already
used by the E17 CLI's own real-trial check):

```
1 x H100 80GB, 8 CPU, 80G host memory, wall time 02:00:00, account rrg-yangw
```

**Variants**: `T64_early_exit` (64 recurrence steps), `T320_reference`
(320 recurrence steps) -- re-pinned from E25, never redefined.

**Population**: the exact, already-frozen E17-bound 24-window measured
population, in canonical `benchmark_slot` order. **Every trial --
warm-up or measured -- processes the same full 24-window population.**
This is confirmed directly against the committed implementation
(`src.dfp_phase1b_t64_early_exit_benchmark.run_trial`): it calls
`records = benchmark_records_in_canonical_order(manifest)`
unconditionally, and its per-window loop iterates all 24 `records`
regardless of `is_warmup`. `is_warmup` only changes the trial
document's own classification field and whether `merge_trials`
includes it in summary statistics -- it never reduces which or how
many windows are processed. (An earlier version of this authority's
own `execution_scope.population` table used a single field,
`warmup_record_count = 1`, positioned in a way that could be misread
as "each warm-up trial processes 1 window" -- this was a
mischaracterization, corrected below and in
`execution_scope.population`/`execution_scope.campaign_workload`.)

The frozen E17 manifest additionally carries one auxiliary record of
its own, `warmup_record` -- a single image ID used **only** to extend
the CLI's dataset-provider ID-resolution set (see
`diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py`:
`selected_ids = sorted({warmup_record["image_id"]} | {r["image_id"]
for r in benchmark_records})`). This auxiliary ID is never processed
as its own one-window pass: `build_manifest_execution_view`
(`src/dfp_phase1a_dataset_view.py`) performs dataset-index/path
metadata resolution only for it (its own docstring: "before any image
is authorized to be processed"), and the runner's own per-window loop
never iterates the raw `warmup_record`, only the 24 `benchmark_records`.
See §5 below for the full three-way ID-population disclosure this
distinction implies.

No manifest override, alternate population, debug subset, or
`--max-images` flag is authorized -- none of these exist in the
committed CLI in the first place.

**Schedule**: warm-up `T64_early_exit` then `T320_reference` (2 warm-up
trials, one per variant, each a full 24-window run, excluded only from
summary statistics), then the frozen counterbalanced measured order (6
trials, also each a full 24-window run):

```
T320_reference repeat 0, T64_early_exit repeat 0, T64_early_exit repeat 1,
T320_reference repeat 1, T320_reference repeat 2, T64_early_exit repeat 2
```

Each trial: a fresh Python process, the same allocated GPU/node,
identical precision/settings, immediate trial verification, fail-fast.
No automatic retry, no extra repeat, no adaptive horizon, no threshold
change.

**Campaign workload counts** (`execution_scope.campaign_workload` --
scheduled repeated-processing counts, never evidence that execution has
occurred, never unique-window or unique-image counts): 4 trials per
variant (1 warm-up + 3 measured), each processing all 24 windows --
8 full trials total across both variants, 192 window executions
(48 warm-up + 144 measured). Recurrence-step counts, derived the same
way: 6,144 T64 steps (4 trials x 24 windows x 64 steps) and 30,720 T320
steps (4 trials x 24 windows x 320 steps), 36,864 total. **These are
scheduled workload counts, not evidence that execution has happened**
-- and two hours is the frozen maximum wall-clock allocation for this
whole campaign, not a measured completion-time estimate. A fresh
Python process per trial (§4 above, "process isolation") also means no
trial can rely on another trial's warm CPU/GPU caches persisting --
process isolation does not by itself guarantee warm-cache conditions
for any given trial.

## 5. Dataset-access scope and population-overlap disclosure

E27's own dataset-access scope distinguishes **three different image-ID
populations**, traced directly against the committed CLI's own
set-construction rule (`selected_ids = sorted({warmup_record["image_id"]}
| {r["image_id"] for r in benchmark_records})`,
`diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py`). An earlier
version of this authority conflated these into a single "24 measured +
1 warm-up = 25" image decode/inference figure; that conflation is
corrected here.

- **`processed_ids`** (24 images) -- the canonical `benchmark_records`'
  own image IDs. These are actually decoded and run through real model
  inference, once per window, in every trial (warm-up or measured).
- **`resolution_only_ids`** (1 image) -- the frozen manifest's own
  single auxiliary `warmup_record` image ID. Confirmed disjoint from
  `processed_ids`. This ID is metadata-resolved only (dataset
  index/path lookup via `build_manifest_execution_view`, confirmed by
  direct inspection of its source and docstring) -- it is never decoded
  or run through inference by the committed runner.
- **`provider_resolution_ids`** (25 images) -- `processed_ids` UNION
  `resolution_only_ids`, exactly the CLI's own `selected_ids`: the full
  set the production provider resolves metadata for at
  dataset-construction time.

E27 authorizes GT-free image decode and geometry crop for exactly the
24 `processed_ids` -- no other image is decoded or run through
inference. GT annotations remain forbidden throughout. Metadata
indexing is limited to what the already-bound production provider
requires to resolve `provider_resolution_ids` (25 IDs) to native
dataset positions -- no broader indexing.

**Overlap disclosure** (independently reconstructed from the E17
manifest JSON plus BALP's frozen 350-image development / 150-image
confirmation split, via the committed, unmodified
`src.dfp_phase1b_lightweight_boundary_identity.load_and_verify_split`
loader -- never merely trusted from a declared TOML value):

| | count | development overlap | confirmation overlap | outside frozen population |
|---|---|---|---|---|
| Development split size | 350 | -- | -- | -- |
| Confirmation split size | 150 | -- | -- | -- |
| `processed_ids` (actually decoded/inferred) | 24 | 18 | 6 | 0 |
| `resolution_only_ids` (metadata-resolution only) | 1 | 1 | 0 | 0 |
| `provider_resolution_ids` (union) | 25 | 19 | 6 | 0 |

**This distinction matters and is stated precisely**: 6 of the 24
actually-processed (`processed_ids`) images -- and 6 of the 25
`provider_resolution_ids` -- *are* members of the 150-image
confirmation split. Their pixels/geometry are read for wall-clock
timing only (or, for the single `resolution_only_ids` member, only its
dataset index/path is resolved, never its pixels) -- this is **GT-free
benchmark input access**, not **confirmation accuracy evaluation**. The
latter remains categorically unauthorized: no GT label, no accuracy
metric, and no candidate decision is ever computed or used anywhere in
this campaign. It would be false to claim confirmation-split images are
"never accessed" by this benchmark; it is accurate to say their GT is
never accessed and no confirmation-accuracy conclusion is ever drawn
from this campaign.

### 5.1 Dependency integrity for the BALP split reconstruction (REPAIR_E27_SPLIT_DEPENDENCY_BINDINGS)

The overlap reconstruction above depends on three things this authority
directly, individually SHA-256 pins in `dependency_bindings` -- the
same convention already used for `e25_loader`/`e26_loader`, extended
here because this is the first place E27 itself directly executes a
*third* loader module's code:

- **`balp_split_loader`** (`src/dfp_phase1b_lightweight_boundary_identity.py`)
  -- the module `load_and_verify_split` is defined in. **Direct**
  binding: unlike E22/E23/E24 (pinned elsewhere in this chain only as
  opaque TOML byte blobs inside E25's own bindings -- E25's and E26's
  own `load_identity()` functions never execute E22's own loader), this
  module's bytes are hash-verified, and then the module is loaded by
  its exact verified on-disk path (`_import_verified_module`, never a
  bare `import` statement) -- so the file whose bytes were hashed and
  the file that actually executes can never diverge, regardless of
  what else is cached on `sys.path`/`sys.modules` in the calling
  process.
- **`balp_split_artifact`** (`evaluation_subsets/dfp_phase1b_lightweight_boundary_500_split.json`)
  and **`balp_subset_manifest`** (`evaluation_subsets/dfp_phase1a_coco_stuff_500.json`)
  -- **direct** byte-anchor bindings, added as defense-in-depth
  alongside (never instead of) `load_and_verify_split`'s own internal
  reconstruction check. That internal check already independently
  detects a *single* one of these two files being altered in isolation
  (the persisted split must exactly match a deterministic
  recomputation from the subset manifest); it does not by itself anchor
  either file's bytes against an externally frozen value, so a
  *coordinated* tamper of both files together (kept mutually
  self-consistent) would not be caught by that check alone. The direct
  bindings close that residual gap.

Verification order: `_verify_dependency_bindings` (which hash-checks
all 21 bindings, including these three) always completes before
`_verify_dataset_access_scope` imports or calls anything from the BALP
split loader -- a failed dependency check prevents the loader from
ever being imported.

## 6. Activation prerequisites

None of the following is satisfied by this task. All six are required
before any of E27's conditional "true" authorization flags may be
exercised for real:

1. E27 independently verified
2. E27 committed
3. clean committed-HEAD production-readiness PASS
4. verified implementation and dependency hashes
5. fresh operator-selected destination
6. explicit operator submission

## 7. Output lifecycle (binds the existing layout; introduces nothing new)

Required launcher environment variables (read directly from the
committed launcher source): `DFP_REPO_ROOT`, `DFP_VENV_ACTIVATE_PATH`,
`COCO_STUFF_REAL_DATA_ROOT`, `TALK2DINO_WEIGHT_DIR`,
`DFP_BENCHMARK_OUTPUT_ROOT` (optional: `DFP_OPENCV_MODULE`). Run output
root: `$DFP_BENCHMARK_OUTPUT_ROOT/${SLURM_JOB_ID:-local}`. Trial
artifacts: `warmup_<variant>.json` (2), `trial_<index>_<variant>_repeat<repeat>.json`
(6). Result bundle: a `result/` subdirectory containing exactly
`merged_result.json` and `decision.json`, installed via one atomic
directory rename (never a sequential two-file install), reopened and
publicly verified before any success is reported. No overwrite, no
force, no new artifact schema/sidecar/CLI flag. Campaign provenance
beyond the installed bundle is limited to the CLI's own readiness
report (stdout) and the SLURM job log (`%x-%j.out`) -- no dedicated
provenance schema exists, and this limitation is stated explicitly
rather than silently assumed away.

## 8. Measurement and decision bindings (re-pinned, never redefined)

Primary timing field: `measured_window_loop_wall_ns`. Expected counts
per measured trial: 24 model forwards, 24 graph constructions, 1536
T64 recurrence steps, 7680 T320 recurrence steps.

`window_throughput_per_second` measures the fixed 24-window benchmark
loop only. **It is not image throughput and does not by itself
establish the full 5,000-image evaluation runtime.** Any later runtime
projection is a separate, explicitly labeled, unmeasured estimate
outside this campaign's own scientific output contract, exactly as E25
already requires.

Decision: `primary_speedup = median(T320_reference measured_window_loop_wall_ns)
/ median(T64_early_exit measured_window_loop_wall_ns)`. Proceed only if
`speedup >= 1.25` and every other E25 gate passes. Outcomes:
`PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL` /
`STOP_T64_EARLY_EXIT_TRACK`. A proceed outcome authorizes only freezing
a *separate*, later T64 confirmation protocol authority -- never
confirmation execution, never the full 5,000-image evaluation.

## 9. Remaining sequence

```
independent E27 verification
  -> commit E27
  -> committed-HEAD production readiness
  -> operator manually submits the verified launcher
  -> independent benchmark-result verification
```

No step in this sequence is performed by this task. No `sbatch`
command is provided by this task.
