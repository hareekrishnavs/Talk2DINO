# Locked 48-image holdout evaluation runner

This document explains the implementation authorized by
`research_specs/semantic_evidence_holdout_evaluation_v1.toml` under
`TASK: IMPLEMENT_HOLDOUT_EVALUATION_RUNNER_WITH_SYNTHETIC_TESTS`: the 5
files that will evaluate the locked 48-image holdout (Gates B, C, and D)
once a separate, explicit successor authority grants execution.

## What this is, and is not

This is **infrastructure**, not permission to access the holdout. Every
test in `tests/test_semantic_evidence_holdout_evaluate.py` runs against
synthetic data through fake providers and a temporary, fully synthetic
directory tree, never real holdout images, GT annotations, a real
checkpoint, or a real model, and never initializes CUDA. No job was
submitted by this task. The runner's two real-access entry points,
`preflight_holdout` and `run_holdout_evaluation`, check
`[authorization.frozen_flags].holdout_execution_authorized` in the
committed holdout authority as their first action and raise
`HoldoutEvaluationError` immediately if it is not `True` -- it is `False`
in the currently committed authority, so both are unconditionally blocked
right now, before touching any holdout path.

## Files

- `src/semantic_evidence_holdout_evaluate.py` -- the runner: torch-free
  `verify_protocol_holdout`/`preflight_holdout`, lazy-torch
  `run_holdout_evaluation`, frozen-fit application, AUROC/AP computation,
  image-paired bootstrap, Gates B/C/D, and holdout-specific artifact
  install/verify wrappers.
- `diagnostics/run_semantic_evidence_holdout_evaluation.py` -- CLI with
  the same four-command shape as the discovery CLI
  (`verify-protocol`/`preflight`/`execute`/`verify-result`), no
  debug/max-image/force/alternate-population flag.
- `semantic_evidence_holdout_evaluation_h100.sbatch` -- H100 launcher,
  non-self-submitting, same resource ceiling as the discovery launcher
  (1x H100 80GB, 8 CPUs, 80GB, 02:00:00).
- `tests/test_semantic_evidence_holdout_evaluate.py` -- synthetic-only
  test suite (see below).

## Reuse boundary

The runner imports, and never copies, the following from
`semantic_evidence_discovery_execute.py` (unmodified by this task):
canonical-window tiling and clamped-window geometry (reached through
`_run_phase_a`/`_run_phase_b`), GT-phase isolation (`PhaseEventLog`,
`_run_phase_a`/`_run_phase_b` themselves), checkpoint/resume
(`build_checkpoint_metadata`, `validate_checkpoint_metadata`,
`_atomic_write_json`, `_load_valid_artifact_or_none`), atomic install
(`atomic_install`), per-image artifact validators
(`validate_gt_free_artifact`, `validate_gt_label_artifact`,
`validate_gt_pair_linkage`), the generic closed-schema validator
(`_validate_closed`), and the default real image/GT/model-and-text
providers. None of these functions embed a "discovery" identity in their
logic -- they operate on whatever `image_id`s and loaders their caller
supplies -- so reuse required no changes to that file.

Two small, holdout-specific adapters exist because the corresponding
discovery function's own docstring/output embeds discovery-only identity,
not because the underlying algorithm differs: `validate_holdout_image_id`
(the same regex/traversal/whitelist check as
`validate_discovery_image_id`, whose own stated purpose is to *reject*
holdout IDs) and the public `holdout_result.json`/`holdout_decision.json`
schemas (new field names; the transient, internal-only `.inprogress`
checkpoint still reuses the discovery module's checkpoint schema
verbatim, including its `discovery_image_count` field name, which holds
the holdout count in this context -- a documented, harmless naming quirk
confined to a file that is deleted after finalization and never a public
artifact).

## Gate C: defined and computable, not executable

The committed holdout authority resolves Gate C's scientific-definition
prerequisite for its own scope (see
`research_specs/semantic_evidence_holdout_evaluation_v1.toml`'s
`[gate_c_authorization]`), so `compute_gate_c` is fully implemented here.
But `execution_authorized_by_this_document=false` in that same table, and
`compute_gate_c` checks this flag as its own first action, returning
`INCONCLUSIVE` (never `PASS`, never `FAIL`, never silently skipped) if it
is not set -- an additional, narrower safeguard inside a function that is
in any case unreachable without first passing the outer
`holdout_execution_authorized` gate.

## Known limitation: Gate D subgroup assignment

`compute_gate_d` implements the frozen rule exactly (three named
subgroups, 100-record minimum, -0.01 maximum negative AUROC difference,
`NO_QUALIFYING_SUBGROUPS` rather than an implied `PASS` when nothing
qualifies), but `_run_holdout_evaluation_core` does not yet derive real
subgroup membership (`predicted_boundary_vs_interior`,
`thing_vs_stuff_gt_category`, `low_vs_high_unary_margin`) from the
persisted rows -- it passes an empty subgroup list, which correctly
yields `NO_QUALIFYING_SUBGROUPS`. Deriving real subgroup membership is
scoped to a later task, after execution authorization exists and there is
real data to assign into subgroups against.

## Authorization sequence still required before any real run

1. Implementation on fake/synthetic data (this task).
2. Independent code and scientific-contract verification.
3. A separate, explicit successor authority setting
   `holdout_execution_authorized=true`.
4. Read-only, committed-head production readiness check against real
   paths.
5. The locked 48-image holdout execution itself (explicit operator
   `sbatch`).
6. A final Gates B/C/D continue-or-stop review of the completed result.

Nothing in this implementation shortens or bypasses this sequence.
