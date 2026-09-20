# DFP Phase 1B: E25 T64 early-exit benchmark protocol

Freezes the scientific and operational contract for a future
T64_early_exit-versus-T320_reference wall-clock benchmark. This is an
authority-freeze task only: it implements no benchmark runner, no CLI
execute path, no sbatch launcher, and authorizes no real GPU execution.

## 1. Why this exists

The completed 350-image BALP development result (SLURM job `22071646`,
closed by E24) measured:

```text
T64 mIoU  = 30.859592058320246
T320 mIoU = 30.934420103426838
T64 - T320 = -0.074828045106592 pp
```

T64 uses 64 of T320's 320 recurrence steps (an 80% reduction). BALP
itself is closed (0/16 eligible candidates, `NO_BALP_CANDIDATE_PROCEEDS`,
frozen by E24) -- this task is **not** a continuation of BALP candidate
evaluation. It asks a different, narrower question: does the
already-measured T64 control offer a useful accuracy/latency tradeoff
as an **operational** shortcut for T320? T64 is always labeled a
`post_development_selected_operational_candidate` -- never a
preregistered method, never a new accuracy winner, not yet validated on
confirmation or the full validation set.

## 2. Files

| File | Role |
|---|---|
| `evaluation_identities/e25_dfp_phase1b_t64_early_exit_benchmark.toml` | The frozen E25 identity: dependency bindings, recurrence-execution contract, variant semantics, fixed population binding, execution schedule, timing/memory/scientific-output contracts, summary formulas, decision rule, authorization boundary. |
| `evaluation_identities/e25_dfp_phase1b_t64_early_exit_benchmark_artifacts.toml` | The closed trial/merged-result/decision artifact schemas (storage-representation-only, like E22 artifacts). |
| `src/dfp_phase1b_t64_early_exit_benchmark_identity.py` | Strict loader/validator for both E25 identities, plus pure schema/identity validation helpers (timing accounting identity, count relations, memory relations, scientific-output finiteness, within-variant digest equality, exact median, exact speedup, decision-rule application). No runner, no CLI, no torch/CUDA. |
| `tests/test_dfp_phase1b_t64_early_exit_benchmark_identity.py` | 102 tests: real-identity load, dependency-hash recomputation, exact-value checks, pure-helper-function tests (including threshold-boundary cases), and an adversarial mutation battery. |
| `docs/dfp_phase1b_t64_early_exit_benchmark.md` | This document. |

## 3. The genuine-early-exit mechanism

The critical technical fact this identity is built on: the committed,
unmodified `finite_step_propagate` (in
`src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py`)
runs `for step in range(1, steps + 1)`. Calling it with `steps=64`
makes step 65 structurally unreachable for that call -- not merely
unread output from a longer run. This is the **only** mechanism E25
authorizes for a genuine T64 early exit: call it directly with
`steps=64, snapshot_steps=(64,)`.

The existing `capture_dfp_phase1a_trajectory` wrapper (in
`src/dfp_phase1a_recurrence_capture.py`) is **not** usable for
T64_early_exit: it always reads `propagation.steps` from E13 (frozen at
320) and rejects any `requested_horizons` that isn't exactly E13's
frozen `[0, 16, 32, 64, 320]` list -- it has no shortened-steps code
path at all (verified directly against its committed source). It
**may** be reused, unmodified, for T320_reference only, exactly as the
committed BALP runner already does.

This is why E25 explicitly prohibits "treating a 320-step snapshot as a
genuine T64 early exit" -- that failure mode is not hypothetical; it's
exactly what the existing horizon-capture convention (used for
*accuracy* measurement) does, and it is the wrong tool for a *timing*
benchmark.

## 4. Fixed population

Reuses E17's already-frozen, already-independently-audited 24-window
benchmark manifest (`evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json`)
verbatim -- no re-selection, no GT field (confirmed directly from its
committed record field set), independently self-verified via the
existing committed `src.dfp_phase1b_chunk_benchmark.verify_benchmark_manifest`.

## 5. Variants, schedule, decision rule

Two variants (`T64_early_exit`, `T320_reference`), everything identical
between them except recurrence horizon (64 vs. 320 steps). 1 warm-up +
3 measured repeats per variant (6 measured trials total), fresh process
per trial, counterbalanced order:

```text
T320_reference(0), T64_early_exit(0), T64_early_exit(1),
T320_reference(1), T320_reference(2), T64_early_exit(2)
```

Primary timing: `measured_window_loop_wall_ns`. Decision rule produces
exactly one of `PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL` /
`STOP_T64_EARLY_EXIT_TRACK`, requiring (among other gates) end-to-end
speedup `>= 1.25x` and the accuracy disclosure to remain exactly
`-0.074828045106592` pp. No adaptive threshold. Passing does not itself
authorize confirmation, full-5000-image evaluation, candidate
selection, a deployment claim, or a latency claim outside the measured
H100 contract.

## 6. Authorization boundary

All ten of `real_gpu_benchmark_execution_authorized_by_e25`,
`runner_implementation_authorized_by_e25`, `confirmation_evaluation_authorized`,
`full_5000_image_evaluation_authorized`, `gt_access_authorized`,
`training_authorized`, `model_weight_modification_authorized`,
`scientific_kernel_modification_authorized`,
`exact_masked_forward_provenance_authorized`,
`balp_candidate_evaluation_authorized` are frozen `false`.

## 7. Future order

1. Independently verify E25.
2. Commit E25.
3. Implement the benchmark runner and launcher.
4. Independently verify the implementation.
5. Committed-HEAD production-readiness verification.
6. User manually submits the H100 job.
7. Independently verify the benchmark artifacts.
8. Only if the outcome is `PROCEED_TO_FREEZE_T64_CONFIRMATION_PROTOCOL`,
   create a separate confirmation authority.

No `sbatch` command is provided by this task.
