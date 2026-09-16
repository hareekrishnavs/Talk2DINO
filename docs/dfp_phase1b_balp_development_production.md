# DFP Phase 1B: E23 BALP production execution wiring

This document describes the real-provider CLI and one-H100 launcher
implementing E23's authorized real 350-image BALP development
execution
(`evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml`).
**This implementation task does not run real inference or submit the
real GPU job** — it wires the committed BALP runner to the committed
real dataset/geometry/model provider and verifies the wiring with
deterministic fake providers only.

## 1. E23 authorization interpretation

E23's `authorization_boundary.real_execute_cli_implementation_authorized_by_this_identity = false`
and the matching `prohibited.list` entry
`"real_execute_cli_implementation_in_this_identity"` are both
explicitly scoped ("`_by_this_identity`" / "`_in_this_identity`") to
E23's own document — they state that E23 *itself* does not implement a
real `execute` CLI, not that no successor task may ever do so. E23's
own `dependency_bindings.balp_cli.binds` text says this directly: *"a
future real `execute` implementation is a SEPARATE, later task this
identity does not perform."* Combined with
`gpu_execution_authorized_by_this_identity = true` and
`development_execution_authorized = true`, and E23's own docs
(`docs/dfp_phase1b_balp_development_execution.md`, "What comes after
E23") explicitly listing "a real `execute` CLI implementation bound to
E23's frozen provider/resource/mode contract" as the first prerequisite
toward real execution — this is unambiguously interpretation (1) from
the authorizing task: a descriptive statement of E23's own scope, with
successor implementation explicitly authorized as the next step.

## 2. Files

| File | Role |
|---|---|
| `src/dfp_phase1b_balp_development_production.py` | Orchestration-only production adapter: torch-free authority/population loading, identity-binding construction, `BalpContracts` narrowing, and the three real-provider builders, wired into the committed `execute_development`. |
| `diagnostics/run_dfp_phase1b_balp_development.py` | CLI: `verify-protocol`, `verify-source-binding`, `verify-model-binding`, `preflight` (all torch-free), `execute`, `verify-checkpoint`, `verify-development-result`. |
| `tests/test_dfp_phase1b_balp_development_production.py` | Fake-provider tests, including a full 350-image/769-window wiring test through the real `execute_development`. |
| `dfp_phase1b_balp_development_coco_stuff_350.sbatch` | The E23-authorized 1x H100 80GB, ≤05:00:00 launcher. Not submitted by this task. |
| `docs/dfp_phase1b_balp_development_production.md` | This document. |

No file whose exact hash E23 binds was modified. No BALP scientific
formula, checkpoint format, artifact schema, metric formula, window-
construction geometry, or graph/recurrence code was reimplemented —
every one is reused unmodified from its E23-bound source.

## 3. A real, load-bearing defect found and fixed during this task

`src/dfp_phase1b_development_window_universe.py` (E23-bound
`window_universe_materializer_module`) imports
`dfp_stitching_contract_from_protocol` from
`src.dfp_phase1a_horizon_stitching` at module scope, and that module
imports `torch` at its own top level — so importing the window-universe
module at all (even just to call its pure `verify_window_universe_document`
function) transitively imports torch. Since that file is E23-bound and
could not be modified, `load_development_window_index` in the
production adapter reimplements only the generic canonical-JSON-digest
primitive (the same convention already established in
`src.dfp_phase1a_dataset_view.compute_mapping_digest`) instead of
importing the heavier module, keeping the whole preflight path
genuinely torch-free.

A second instance of the same class of bug was found and fixed in this
module's own new CLI: `_all_variant_ids()` originally imported
`src.dfp_phase1b_lightweight_boundary` (the torch-dependent BALP
kernel) merely to read the 16 candidate-ID strings. It now uses
`e22.reconstruct_candidate_ids()` — the exact same torch-free helper
the *existing* committed CLI
(`diagnostics/run_dfp_phase1b_lightweight_boundary.py`) already uses
for this — confirmed byte-identical to `balp.CANDIDATE_IDS`.

Both fixes were verified empirically: a fresh subprocess that imports
this module and runs the entire torch-free preflight path (including
the real `preflight` CLI command against the real dataset/model-weight
presence checks) never has `torch` in `sys.modules` afterward.

## 4. Production provider mapping

| E23 role | Real function |
|---|---|
| dataset/geometry provider | `src.dfp_phase1b_calibration_production.build_production_dataset_resources` |
| model/inference builder | `src.dfp_phase1b_calibration_production.build_production_inference` |
| manifest-order/native-index adapter | `src.dfp_phase1a_dataset_view.build_manifest_execution_view` / `verify_position_identity` (via `resources.execution_view`) |
| image preparation | `diagnostics.run_k11_k12_stability._extract_prepared_image` |
| per-window scientific processing | `src.dfp_phase1b_lightweight_boundary_execute.process_development_image` (unmodified; production never overrides it) |
| checkpoint/resume/finalize-only | `src.dfp_phase1b_lightweight_boundary_execute.execute_development` (unmodified) |
| checkpoint/result validators | `src.dfp_phase1b_lightweight_boundary_artifact` (unmodified) |

`BalpContracts` (recurrence/stitching/pullback/final_resolution) is
built by narrowing the real `ComponentContractsBundle` the production
provider module returns, deliberately discarding its `component` and
`masked_forward` fields — verified by a test (`_FakeSixFieldContractsBundle`)
that these two fields are simply absent (`AttributeError` on access)
from the narrowed object.

## 5. 350/769 enforcement

`load_development_window_index` loads the committed manifest,
independently re-verifies its internal digest/count consistency, and
cross-checks it against E23's own pinned `image_count=350`/
`window_count=769`/digest. `verify_population_isolation` then requires
the manifest's 350 IDs to be byte-identical, in order, to E22's own
frozen development split, and disjoint from the 150-image confirmation
split — before any dataset/model access. `reject_out_of_population_id`
is an additional runtime gate inside the real (and fake, in tests)
image-provider closure: any canonical ID not in the 350-ID allow-list
(covering confirmation IDs and all other 4,500 images alike) raises
before any image is prepared. Both the confirmation-ID and other-4500
rejection paths, and the full 350/769 wiring, are exercised by tests
at true scale.

## 6. Operation counts

Enforced entirely by the unmodified `process_development_image`: one
model forward, one graph construction, and one 320-step k12 recurrence
trajectory per selected window (never per image, never per candidate);
all five horizon snapshots (T0/T16/T32/T64/T320) are read from that
single trajectory; all 16 candidates and 5 controls are evaluated from
those five score fields with zero additional recurrence. This
production adapter never reimplements or duplicates that loop — it
only supplies the real `inference`/`contracts`/`prepared`/
`selected_windows_for_image` arguments `process_development_image`
already accepts.

## 7. GT isolation

The real `image_provider_fn` resolves `prepared` (via
`_extract_prepared_image`, GT-free) and `ground_truth` (via
`dataset.get_gt_seg_map_by_idx`, the same real accessor
`src/dfp_phase1a_execute.py`'s own production driver uses) as two
independent fields of `DevelopmentImageProvider`. `execute_development`
passes only `prepared` to the scientific processor and passes
`ground_truth` to `accumulate_image_metrics` strictly after every
prediction for that image is already finalized — this adapter changes
none of that structural separation.

## 8. Checkpoint/resume/finalize-only

Entirely delegated to the unmodified `execute_development` /
`src.dfp_phase1b_lightweight_boundary_artifact` (immutable generations,
atomic installation, orphan-successor recovery, finalize-only
recovery, post-install public reverification) — already exhaustively
tested at full 350-image scale by the existing committed suite
(`tests/test_dfp_phase1b_lightweight_boundary_execute.py`). This task's
own tests do not re-prove that engine's correctness; they prove only
that this adapter supplies it with correct real (or fake, for testing)
inputs.

## 9. Profiling

`src.dfp_phase1b_lightweight_boundary_execute.ProfilingRecorder`
(already committed, unmodified) is threaded through
`run_development_execution`'s `recorder` parameter. Raw measurements
only; never affects the scientific digest (the committed engine's own
architecture already guarantees this — profiling fields never enter
`aggregate_confusion`/`per_image_reconstruction_statistics`).

## 10. CLI

`diagnostics/run_dfp_phase1b_balp_development.py`: `verify-protocol`,
`verify-source-binding`, `verify-model-binding`, `preflight` (all
torch-free), `execute` (requires `--mode fresh|resume --device cuda
--output-root <absolute path>`; `--device` accepts only `"cuda"`),
`verify-checkpoint`, `verify-development-result`. No
`--max-images`/`--debug-subset`/`--image-ids`/`--dataset`/
`--allow-overwrite`/`--force`/`--candidate`/`--confirmation`/
`--full-validation`/`--steps`/`--k`/`--crop-size`/`--stride` flag
exists. Failures use `DFP PHASE 1B BALP FAIL [<stage>]: <message>`
with exit code 2.

## 11. Launcher

`dfp_phase1b_balp_development_coco_stuff_350.sbatch`: 1x H100 80GB,
`--cpus-per-task=8 --mem=80G --account=rrg-yangw` (reusing
`dfp_phase1b_chunk_benchmark_h100.sbatch`'s own established H100-class
resource precedent), `--time=05:00:00`. Runs, in order: print HEAD,
`verify-protocol`, `verify-source-binding`, `verify-model-binding`,
`preflight`, a HEAD-unchanged check, `execute`, `verify-checkpoint`,
`verify-development-result`. Contains no `sbatch` invocation (never
self-submits), no private path/username, no `--force`, no debug/
max-images option, no confirmation/full-validation command. **Not
submitted by this task.**

## 12. Scope note for independent verification

This task's test suite proves, at true 350-image/769-window scale with
fake providers: population/window isolation (including confirmation-ID
and other-4500 rejection), identity-binding construction, contracts
narrowing, the torch-free preflight path (empirically, via a fresh
subprocess), and correct end-to-end wiring into `execute_development`
(producing a valid, publicly-reverified finalized result). It does
**not** re-run the existing committed suite's own exhaustive
fresh/resume-at-every-cut-point/orphan-recovery/finalize-only/
atomicity-injection tests for `execute_development` itself — those
already exist, already pass at full scale, and this task must not
duplicate them. Independent verification should confirm this scoping
choice is acceptable, and may wish to add CLI-level subprocess tests
(`--help`, corrupted checkpoint, wrong HEAD, wrong E23 hash, output
traversal, completed-result overwrite attempt, unauthorized device) as
a further hardening pass, which this task did not have budget to add
exhaustively.
