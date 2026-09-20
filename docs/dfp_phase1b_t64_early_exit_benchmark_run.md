# DFP Phase 1B: T64-versus-T320 early-exit benchmark -- implementation

Implements the five files E26 authorizes: the runner (`src/
dfp_phase1b_t64_early_exit_benchmark.py`), the CLI (`diagnostics/
run_dfp_phase1b_t64_early_exit_benchmark.py`), its tests (`tests/
test_dfp_phase1b_t64_early_exit_benchmark.py`), this document, and the
launcher (`dfp_phase1b_t64_early_exit_benchmark_h100.sbatch`). The
scientific/operational contract remains frozen, unmodified, by E25;
this implementation defines no new scientific value, threshold,
schema, or authorization flag.

## 1. What this measures

Wall-clock time for 24 fixed windows (E17's own committed manifest)
processed under two variants that differ in exactly one respect -- the
finite-step recurrence horizon:

- **T64_early_exit**: `finite_step_propagate(graph, s0, alpha=0.98, steps=64, snapshot_steps=(64,))`, called directly.
- **T320_reference**: the identical function, `steps=320, snapshot_steps=(320,)`.

Because `finite_step_propagate`'s own loop is `for step in range(1,
steps + 1)`, step 65 is structurally unreachable for the T64 call --
never merely unread output of a longer run. T64 never calls
`src.dfp_phase1a_recurrence_capture.capture_dfp_phase1a_trajectory`
(permanently bound to E13's frozen `steps=320`).

## 2. Per-window pipeline

`input_preparation -> unary_model_forward -> graph_construction ->
canonical_recurrence -> stitching_and_rescale -> output_validation`,
identical between variants except the recurrence horizon. Reuses,
completely unmodified: `models.dinotext.cover_dr.graph.
build_directed_topk_graph` (k=12, affinity_power=3.0), `models.
dinotext.cover_dr.finite_step_regime.finite_step_propagate`, `src.
dfp_phase1a_horizon_stitching.stitch_dfp_phase1a_horizons` (forward
stitching only -- never the masked-forward/pullback module), and `src.
dfp_phase1a_final_rescale_pullback.apply_final_rescale_forward`.

Stitching runs against a synthetic single-window `SlidingWindowPlan`
sized exactly to the window's own crop (`image_size == crop_size ==
stride`), so the plan has exactly one fully-covering window and needs
no sibling windows from the same source image -- the fixed 24-window
population deliberately need not include every window of any one
image. The real E13-frozen crop/stride/patch_grid/patch_size/
align_corners geometry is reused verbatim (via
`resources.contracts.stitching`); only its `horizons` field is
narrowed, via `dataclasses.replace`, to the one captured step.

No exact masked-forward, component masking, membership pullback,
component contribution, solver, BALP candidate, or GT path is imported
or called anywhere in the runner.

## 3. Population, schedule, counts

Exactly E17's committed 24-window manifest, verified by file SHA-256
and both internal digests against E25's `population_binding` before
any trial. Two warm-up trials (one per variant, full 24-window runs,
excluded only from summary statistics) followed by the six frozen
measured trials in E25's exact counterbalanced order. Per trial: 1
dataset construction, 1 model construction, 24 unary forwards, 24
graph constructions, 1536 (T64) or 7680 (T320) total recurrence steps.

## 4. Timing, memory, digests

`measured_window_loop_wall_ns` is measured only around the 24-window
loop (CUDA-synchronized on both boundaries) and is the sole primary
decision quantity; `process_total_wall_ns` is computed as the exact
sum of the other six phase fields, so the accounting identity holds by
construction. Per-window stage timings (`stage_timings_ns`) are
recorded but never substitute for the primary field. Memory telemetry
(`host_peak_rss_bytes`, `gpu_peak_memory_allocated_bytes`,
`gpu_peak_memory_reserved_bytes`, `gpu_total_memory_bytes`) uses only
public `torch.cuda` accessors and the committed `/proc/self/status`
VmHWM parser -- never the removed private `torch._C.
_cuda_getDriverVersion` symbol; if a driver version is needed, the
already-verified `nvidia-smi`-based query
(`src.dfp_phase1b_calibration_run._query_cuda_driver_version`) is
reused. Per-window and collection scientific-output digests reuse the
repository's existing canonical-encoding/length-prefixed-digest
convention (`src.dfp_phase1b_chunk_benchmark.canonical_encode` /
`digest_over_records`).

## 5. Artifacts and CLI

Trial/merged-result/decision documents are built and verified against
E25's own closed artifact schemas (`evaluation_identities/
e25_dfp_phase1b_t64_early_exit_benchmark_artifacts.toml`), never a
competing schema. Individual trial artifacts install via
`install_artifact_atomically`: stage in a uniquely-named sibling
directory, self-verify the staged bytes, atomically rename into an
absent destination, reopen, and re-verify publicly.

The merged result and its decision install together as one
**pair-atomic bundle directory** (`install_result_decision_bundle_
atomically`): both `merged_result.json` and `decision.json` are
written and cross-verified inside one staging directory, then
installed via exactly one atomic directory rename. After that rename,
the destination is either completely absent or a complete,
cross-verified pair -- a result can never exist without its decision,
or vice versa. Cross-document consistency reuses E25's own frozen
`decision_schema.merged_result_sha256` field exclusively (bound to the
merged result's actual on-disk bytes, never a value merely echoed from
memory); no new or competing binding field is introduced. If
post-install re-verification fails after the rename, the installed
pair is deliberately left in place for forensic/public re-verification,
but the command still fails and never reports success.

No overwrite, no force mode anywhere in either installation path: a
pre-existing destination (trial file, or result/decision bundle
directory) is always rejected, both up front during `preflight`/before
the merge itself, and again immediately before the final rename
(closing the race window between the two checks). There is no
`--force` flag, `allow_overwrite` parameter, or any other override
anywhere in this implementation.

CLI subcommands: `verify-protocol`, `preflight`, `run-trial`,
`verify-trial`, `merge` (installs the merged result + decision bundle
via `--output-dir`), `verify-result`, `verify-decision` (both take a
plain file path, so they can inspect either file inside an installed
bundle directory). Every subcommand except `run-trial` never imports
torch, never initializes CUDA, and never accesses the real dataset or
model.

## 6. Launcher

`dfp_phase1b_t64_early_exit_benchmark_h100.sbatch`: 1x H100 80GB, 8
CPU, 80G, wall time 02:00:00, account `rrg-yangw`. Requires repository
root, venv activation path, real dataset root, real weight directory,
and a fresh benchmark output root as environment variables. Runs CPU
preflight, then the two warm-up trials, then the six measured trials
(each a fresh process, each verified immediately), then merges and
verifies the result and decision. Contains no `sbatch`/self-submission
call anywhere, no overwrite/force path, and no confirmation or
full-5000-image command. Only syntax-checked (`bash -n`) by this
implementation task; never submitted.

## 7. What a PROCEED outcome does and does not authorize

`PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL` authorizes only creating
a SEPARATE, later confirmation authority. It does not itself run the
150-image confirmation evaluation, select a final method, run the full
5,000-image evaluation, or make any deployment/latency claim beyond
this measured H100 contract -- exactly as E25's own decision-artifact
schema's `proceed_does_not_authorize` list states.

## 8. Deferred next steps

1. Independently verify this implementation.
2. Committed-HEAD production-readiness verification.
3. User manually submits the H100 job.
4. Independently verify the resulting benchmark artifacts.
5. Only if the outcome is `PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL`, create a separate confirmation authority.

No `sbatch` command is provided by this task.
