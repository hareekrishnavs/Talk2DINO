# DFP Phase 1B: E23 BALP development-execution authorization

This document describes **E23**
(`evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml`,
loader `src/dfp_phase1b_balp_development_execution_identity.py`), the
successor authority to E22
(`evaluation_identities/e22_dfp_phase1b_lightweight_boundary_preservation.toml`)
that authorizes **real execution** of the already-frozen BALP method over
the already-frozen 350-image development split.

E22 explicitly "implements no method, runs no inference, selects no
candidate, and authorizes no GPU execution" — it freezes definitions,
schemas, and gates only. E23 is the identity that turns the key: it
binds the real production dataset/geometry/model provider path, the
resource contract, and the checkpoint/resume mode authorization, so a
future `execute` implementation has something concrete to point at.
**E23 itself implements no execute CLI, submits no GPU job, constructs
no model, and accesses no image pixel or GT annotation.**

## 1. What E23 adds beyond E22

E22 already fully defines the BALP method (16 candidates + 5 controls,
boundary/stability/late-flip formulas), the 350/150 development/
confirmation split, GT isolation, and (via its paired artifacts
identity) the closed checkpoint/development-result/selection/profiling
schemas and the staged-write/atomic-install/post-install-reverify
policy. E23 re-pins every one of these by exact path + SHA-256 and
introduces no competing definition. E23's own additions are:

1. **The real provider binding.** `src/dfp_phase1b_calibration_production.py`
   (dataset/geometry/model construction), `src/dfp_phase1a_dataset_view.py`
   (manifest-to-native-index resolution), and
   `src/open_vocabulary_segmentation/segmentation/evaluation/sliding_window_geometry.py`
   (window construction) — none of these were E22 dependencies, since
   E22 runs no inference.
2. **The frozen 350-image development window universe.** A prior task
   found the exact per-image window count/digest for the 350
   development images **unreconstructable** from any committed source
   (`BLOCKED: DEVELOPMENT WINDOW-UNIVERSE AUTHORITY IS INCOMPLETE`).
   That blocker is resolved by real, independently-verified
   materialization (`src/dfp_phase1b_development_window_universe.py`,
   `evaluation_subsets/dfp_phase1b_development_window_universe.json`):
   **350 images, 769 windows total**,
   `canonical_record_digest = b4672854e38199deef26c154a30e45c1835c201a4dbcf046aa29e7fdf12df9e6`.
   E23's `dependency_bindings.development_window_universe` binds this
   file by path, SHA-256, and its own `image_count`/`window_count`/
   `canonical_record_digest` fields — cross-checked at load time against
   the manifest's own recorded values, not merely trusted.
3. **The GPU/wall-time resource contract**: exactly 1× H100 80GB,
   `cuda`, wall time at most `05:00:00` (a hard operational ceiling — a
   timeout must leave a valid resumable checkpoint, never silently
   relax the limit). `cpus_per_task`/`host_memory`/account convention
   reuse the repository's own established H100-class launcher
   precedent (`dfp_phase1b_chunk_benchmark_h100.sbatch`:
   `--account=rrg-yangw`, `--cpus-per-task=8`, `--mem=80G`).
4. **Checkpoint/resume mode authorization**: exactly `fresh`, `resume`,
   `finalize_only_via_resume` (the E22-artifacts checkpoint state
   machine's own four states, collapsed to three authorized entry
   modes). Explicitly forbidden: debug subset, max-images override,
   alternate ID order, force-overwrite of a completed result, manual
   generation skipping, silent checkpoint reset.
5. **An explicit prohibition list**, strictly no broader than E22's own
   authorization boundary: no confirmation-split access, no other-4500
   access, no full 5,000-image evaluation, no exact masked-forward
   provenance, no candidate-specific recurrence, no overwrite of a
   completed development result, and no real-execute-CLI or
   GPU-job-submission authorization by this identity itself.

## 2. Files

| File | Role |
|---|---|
| `evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml` | The frozen E23 identity. |
| `src/dfp_phase1b_balp_development_execution_identity.py` | Closed-schema loader: exact-value validation of every nested table, SHA-256 re-verification of all 23 dependency bindings (including cross-checking the window-universe binding against the real manifest's own recorded fields), E22 cross-hash consistency, and repository ancestor-commit gate. |
| `tests/test_dfp_phase1b_balp_development_execution_identity.py` | Loads the real committed identity, plus 30+ adversarial mutations (wrong hash, path traversal, absolute path, bool-as-int, wrong population counts, wrong GPU count/model, wall time over 5h, every authorization-boundary field individually flipped true, tampered window-universe fields, forbidden checkpoint mode injected, wrong ancestor/successor). |
| `docs/dfp_phase1b_balp_development_execution.md` | This document. |

## 3. What E23 does **not** do

- Does not implement a real `execute` CLI command. The committed CLI's
  `execute` subcommand (`diagnostics/run_dfp_phase1b_lightweight_boundary.py`)
  still always fails closed with `EXECUTE IS NOT AUTHORIZED` — a real
  execute path is a separate, later implementation task.
- Does not submit, or contain a real invocation of, any GPU `sbatch`
  job.
- Does not construct a model, initialize CUDA, decode an image, or read
  a GT annotation.
- Does not select a BALP candidate, evaluate the 150-image confirmation
  split, or access any of the other 4,500 COCO-Stuff val2017 images.
- Does not authorize a full 5,000-image evaluation under any condition.

## 4. Dependency bindings (23 total)

Re-pinned from E22 (never redefined): `e22_preservation`, `e22_artifacts`,
`e13_protocol`, `matched_evaluator`, `subset_manifest`, `balp_split_artifact`,
`graph_module`, `finite_step_regime_module`, `recurrence_capture_module`,
`horizon_stitching_module`, `stitching_pullback_module`,
`final_rescale_module`, `analysis_module`, `identity_module`.

New in E23: `balp_scientific_kernel`, `balp_artifact_validator`,
`balp_execution_runner`, `balp_cli`, `production_provider_module`,
`dataset_view_module`, `sliding_window_geometry_module`,
`window_universe_materializer_module`, `development_window_universe`.

E15 (`e15-dfp-phase1b-protocol`/`-artifacts`), E16, E19, E20, and E21 are
explicitly excluded as scientific dependencies, identical in reasoning
to E22's own `[historical_exclusions]` — a real BALP execution neither
runs masked-forward propagation, constructs components, nor depends on
the discontinued exact-provenance runner. The shared production-adapter
infrastructure (`_load_production_identities`) does load an E15 and E3
identity internally as part of code it reuses from DFP Phase 1A's own
driver, but those loaded values are never read by BALP's scientific
formulas — see `[historical_exclusions].note_on_shared_adapter_identity_loads`
in the E23 TOML.

## 5. Verifying E23

```
module load opencv/4.14.0
source /scratch/haree/venv/talk2dino-a100/bin/activate
python -m pytest tests/test_dfp_phase1b_balp_development_execution_identity.py -q
```

or programmatically:

```python
from pathlib import Path
import src.dfp_phase1b_balp_development_execution_identity as e23

doc = e23.load_identity(repo_root=Path("."))
```

`load_identity` performs closed-schema validation of every nested
table, exact-value validation of every frozen field, SHA-256
re-verification of all 23 dependency bindings against current file
bytes, a relational cross-check of the E22 hash pinned here against
E22's own artifacts-identity `protocol_binding.protocol_identity_sha256`,
and (given a 40-hex commit) `verify_repository_gate` to confirm
`required_ancestor_commit` is an ancestor of the caller-supplied HEAD.

## 6. What comes after E23

Real 350-image BALP development execution requires, in order: (1) a
real `execute` CLI implementation bound to E23's frozen provider/
resource/mode contract, (2) independent implementation verification,
(3) committed-HEAD verification, (4) the actual GPU job (1× H100 80GB,
≤05:00:00, `def-yangw-ab`/`rrg-yangw`-style account per the established
convention), (5) an independently-verified completed 350-image
development result, before any candidate selection, confirmation
evaluation, or full 5,000-image execution can even be considered — each
of which remains gated behind its own later successor authority.
