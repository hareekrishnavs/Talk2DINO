# DFP Phase 1B: E26 T64 early-exit benchmark implementation authority

Authorizes **implementation only** of the T64-versus-T320 early-exit
benchmark runner, CLI, tests, documentation, and SLURM launcher whose
scientific/operational contract is already frozen, unmodified, by E25.
This task creates no such files itself and executes nothing.

## 1. Why E26 exists

E25's own `authorization_boundary` explicitly sets both
`runner_implementation_authorized_by_e25 = false` and
`real_gpu_benchmark_execution_authorized_by_e25 = false`. E25 froze the
*contract* (variants, horizons, population, schedule, timing/memory/
digest rules, decision rule) but authorizes no code that implements it
and no real execution. E26 closes exactly the implementation gap, for
exactly five named files, and authorizes no execution of any kind.

## 2. Files

| File | Role |
|---|---|
| `evaluation_identities/e26_dfp_phase1b_t64_early_exit_benchmark_implementation.toml` | The frozen E26 identity: 14 dependency bindings (E25 + its own recurrence/graph/stitching/provider/artifact-schema dependencies), the exact 5-file authorized future set, the required CLI subcommand surface, runner population/construction contracts, process isolation, launcher resource contract, timing/telemetry/output-count implementation requirements, implementation-test requirements, and the authorization boundary. |
| `src/dfp_phase1b_t64_early_exit_benchmark_implementation_identity.py` | Strict loader/validator for E26 (closed schema, exact-value checks, relational hash reverification, repository-ancestor gate, no-circular-self-hash check). Torch-free. |
| `tests/test_dfp_phase1b_t64_early_exit_benchmark_implementation_identity.py` | Exhaustive E26 identity tests: closed schemas, dependency hashes, path containment, exact authorized-file list, exact subcommands, exact variants/horizons/population/schedule/resource contract, authorization-boundary exactness, adversarial mutations, fatal-exception propagation. |
| `docs/dfp_phase1b_t64_early_exit_benchmark_implementation.md` | This document. |

## 3. What E26 authorizes, and what it does not

Authorized: **writing** `src/dfp_phase1b_t64_early_exit_benchmark.py`,
`diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py`,
`tests/test_dfp_phase1b_t64_early_exit_benchmark.py`,
`docs/dfp_phase1b_t64_early_exit_benchmark_run.md`, and
`dfp_phase1b_t64_early_exit_benchmark_h100.sbatch`, plus CPU-only
synthetic testing of that code. `authorized_future_files.no_other_new_production_file_authorized = true` —
no other new production file may be created under E26.

Not authorized by E26: real dataset access, real model construction,
weight loading, CUDA initialization, a real GPU trial, SLURM
submission, confirmation evaluation, full-5,000-image evaluation, GT
access, scientific-kernel modification, E25 modification, or exact
masked-forward provenance. All twelve of these are frozen `false` in
`authorization_boundary`.

## 4. Required CLI subcommand surface

```
verify-protocol   -- torch-free, CUDA-free
preflight         -- torch-free, CUDA-free
run-trial         -- may contain the real execution path; never invoked by E26 itself
verify-trial
merge
verify-result
verify-decision
```

## 5. Runner contract (re-pinned from E25, never redefined)

Exactly the committed E17 24-window population (reject alternate
manifests, reordering, missing/duplicate/extra windows, any GT-bearing
input). Dataset and model constructed exactly once per trial. Exactly
one unary model forward and one graph construction per window. T64
calls `finite_step_propagate` directly with `steps=64` (never via
`capture_dfp_phase1a_trajectory`, which cannot produce a genuine early
exit); T320 with `steps=320`. Never a 320-step trajectory sliced for
T64. No candidate-specific recurrence. No exact masked-forward/
component provenance call. All trial outputs validated against E25's
own closed artifact schemas.

## 6. Process isolation and schedule (re-pinned from E25)

Fresh Python process per trial. Warm-up: `T64_early_exit` ×1,
`T320_reference` ×1 (validated, excluded from summaries). Measured, in
this exact order:

```
T320_reference(0), T64_early_exit(0), T64_early_exit(1),
T320_reference(1), T320_reference(2), T64_early_exit(2)
```

The launcher fails immediately on any trial or verification failure;
no partial merged result may ever be produced.

## 7. Launcher resource contract

1× H100 80GB, 8 CPU, 80G host memory, wall time 02:00:00, account
`rrg-yangw`. No self-submission, no literal `sbatch` invocation inside
the launcher script itself. Requires environment variables for
repository root, venv activation path, dataset root, and output root.
Preflight must run before any model/CUDA operation. Fresh output
destination only — no overwrite, no force mode. Every trial is
verified immediately after creation; the merge step requires exactly
the six measured trials plus warm-up evidence; the final result and
decision are verified again after installation. **Not submitted by
this or any prior task.**

## 8. Timing, telemetry, output-count requirements

Re-pinned verbatim from E25: the seven `..._wall_ns` fields and their
accounting identity; `measured_window_loop_wall_ns` as the sole primary
decision quantity (`time.perf_counter_ns`, CUDA-synchronized
boundaries, canonical window order); the four memory-telemetry fields
with `0 <= allocated <= reserved <= total`, CUDA peak stats reset
immediately before the measured loop, fail-closed on missing/malformed
telemetry; `model_forward_count = graph_construction_count = 24`,
`T64 recurrence_steps_executed = 1536`, `T320 recurrence_steps_executed = 7680`;
within-variant digest determinism (cross-variant equality not
required); all outputs finite; profiling never affects the scientific
digest.

## 9. Implementation-test requirements

The future implementation's own test suite must prove, using CPU-only
synthetic fixtures and no real dataset/model/weights/CUDA: genuine T64
early exit, step-64 equivalence to a controlled 320-step reference
prefix, step-65 unreachability, exact counts/schedule, fresh-process
orchestration, deterministic and within-variant-identical digests
(cross-variant equality not required), timing/accounting and memory
validation, strict artifact schemas, atomic installation with no
overwrite, public post-install verification, clean failure on
malformed input, fatal-exception propagation, and the absence of any
GT, confirmation/full-5000, or exact-masked-forward path.

## 10. Deferred next steps

1. Independently verify E26.
2. Commit E26.
3. Implement the five authorized files.
4. Independently verify the implementation.
5. Committed-HEAD production-readiness verification.
6. User manually submits the H100 job.
7. Independently verify the benchmark artifacts.
8. Only if the outcome is `PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL`,
   create a separate confirmation authority.

No `sbatch` command is provided by this task.
