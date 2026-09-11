# DFP Phase 1B — Stage-A Provenance-Feature Scientific Kernel (E21)

```text
SCIENTIFIC KERNEL ONLY.  NO RUNNER, PROFILER, CHECKPOINT/RESULT WRITER, OR CLI EXECUTION COMMAND.
NO REAL INFERENCE.  NO COCO-STUFF IMAGE ACCESS.  NO CUDA INITIALIZED MERELY ON IMPORT.
500-IMAGE EXECUTION AND 5,000-IMAGE EXACT PROVENANCE REMAIN NOT AUTHORIZED.
```

## 1. Purpose and scientific boundary

E21 (`evaluation_identities/e21_dfp_phase1b_provenance_features.toml`)
freezes the **numerical kernel** that computes the eight E15 Stage-A
component-evidence provenance features:

`n_positive_components`, `positive_evidence_sum`, `negative_evidence_sum`,
`top_positive_component_id`, `top_positive_evidence`,
`top1_positive_share`, `k_star`, `phi`

from the already-frozen, already-committed signed component-evidence
matrix `G[x,r]` produced by the exact masked-forward provenance core
(`src/dfp_phase1b_masked_forward.py`).

These eight fields were previously frozen **only as TEXT authority** in
`evaluation_identities/e15_dfp_phase1b_protocol.toml`'s
`[provenance_features.component_evidence]` block, with **no
implementation anywhere in the repository** — `dfp_phase1b_masked_
forward.py`'s own module docstring explicitly disclaims computing
`k_star`/`phi`. This gap blocked implementation of the profiled 500-image
diagnostic runner (bound by E20). E21 repairs exactly this gap and
nothing else.

E21 is an **additive successor authority only**. It:

* implements no execution driver, runner, profiler, checkpoint writer,
  result writer, CLI execution command, or SLURM launcher;
* runs no real inference, accesses no COCO-Stuff images or annotations,
  and initializes no CUDA context merely on import;
* does **not** authorize the 500-image diagnostic execution or the
  5,000-image exact masked-forward provenance question;
* does **not** resume implementation of the blocked runner itself —
  that remains a separate, later, separately-authorized task;
* never redefines a scientific quantity already frozen by
  `e15-dfp-phase1b-protocol` / `e15-dfp-phase1b-artifacts` — it **reuses**
  those formulas verbatim, never re-derives, re-selects, or
  reinterprets them, and invents no new scientific convention to fill
  a gap in E15.

## 2. Authority-completeness audit (why this is not BLOCKED)

Every semantic decision the kernel needed was either **directly stated**
by committed E15 text, or a **necessary mathematical consequence** of
other already-frozen E15 invariants (never an invented convention):

| Decision | Resolution | Source |
|---|---|---|
| Component evidence value | `G[x,r]` | e15 `signed_component_evidence_formula` |
| Positive/negative evidence sums | `sum_r max(G,0)` / `sum_r min(G,0)` | e15 `positive_evidence_sum_formula` / `negative_evidence_sum_formula` |
| Strict positivity | `G[x,r] > 0` (not `>=`) | e15 `g_plus_definition` |
| Ranking / tie-break | descending value, ties by ascending component ID | e15 `sort_order` |
| `k_star` formula | `min k` s.t. `m320(x) - sum_{q=1..k} G_sorted[q] <= 0` | e15 `k_star_definition` (uses `m320`, not `delta_m`, exactly as written) |
| Crossing tolerance | exact `<= 0`, no epsilon | e15 `phi_epsilon_forbidden = true`; no epsilon in the formula text |
| `phi` / `top1_positive_share` | e15 `phi_formula` / `top1_positive_share_formula`, verbatim | e15 |
| "No positive components" / "insufficient evidence" / "already crossed at k=0" / `positive_evidence_sum == 0` edge cases | **mathematically impossible** for a valid disputed pixel, given frozen `m0(x) <= 0`, `m320(x) > 0`, and the conservation identity `sum_r G[x,r] = delta_m(x)` — derivation in `[domain_invariants]` below; runtime violation is rejected, never fabricated | e15 `[margins]`, `[component_evidence_decomposition]`; `k_star_non_finite_case_behavior` ("reject ... rather than inventing infinity") generalized |
| Accumulation dtype | `float64` | e15 `component_sum_accumulation_dtype = "fp64"` |
| Output dtypes | `int64` (`k_star`, `n_positive_components`, `top_positive_component_id`); `float64` (rest) | `src/dfp_phase1b_artifact_schema.py` Stage-A schema (already committed, unmodified) |
| Non-finite / malformed input | reject the whole call | repo-wide fail-closed convention; `validate_stage_a_flat_arrays`'s own already-committed behavior |

No item required an invented convention. The audit **passed**.

## 3. Exact dependencies

E21 relationally hash-verifies, on every load:

* `e15-dfp-phase1b-protocol`, `e15-dfp-phase1b-artifacts` — the
  formula/dtype authority this kernel reuses verbatim;
* `e20-dfp-phase1b-diagnostic-execution` — the predecessor contract this
  kernel unblocks;
* `src/dfp_phase1b_diagnostic_execution.py` — E20's own loader module;
* `src/dfp_phase1b_masked_forward.py` — the exact masked-forward core
  that produces `G[x,r]` (this kernel never recomputes it);
* `src/dfp_phase1b_artifact_schema.py` — the Stage-A relational
  validator this kernel's output is designed to satisfy;
* `src/dfp_phase1b_provenance_features.py` — the new numerical kernel
  this identity freezes;
* its own loader (`src/dfp_phase1b_provenance_features_identity.py`)
  and CLI (`diagnostics/run_dfp_phase1b_provenance_features.py`) modules
  (self-referenced, non-circular: this TOML records their hash; those
  modules never record this TOML's own hash).

It never modifies any of E13–E20, the verified calibration artifact, the
installed Task 4 selection artifact, or the masked-forward core.

## 4. Kernel API

```python
from src.dfp_phase1b_provenance_features import (
    ComponentEvidenceFeatures,
    DfpPhase1bProvenanceFeaturesError,
    compute_component_evidence_features,
)

features = compute_component_evidence_features(G, m320, component_ids=component_ids)
```

* `G` — `[disputed_pixel_count, component_count]` `float64`: the
  already-computed signed component-evidence matrix.
* `m320` — `[disputed_pixel_count]` `float64`: the per-pixel propagated
  margin (`m320(x)`; e15's own `propagated_margin = m320` alias noted,
  but the kernel uses the literal `m320` name the `k_star` formula uses).
* `component_ids` — `[component_count]` `int64`: the canonical,
  deterministic component identity per column of `G`, independent of
  column position; no duplicates, no negative values.

Returns a frozen `ComponentEvidenceFeatures` dataclass with one
`[disputed_pixel_count]` tensor per Stage-A field, on the caller's
device, never a view into an input tensor.

The kernel never mutates its inputs, never silently moves a tensor to
another device, and never initializes CUDA merely to run — it operates
on whatever device the caller's tensors are already on.

## 5. Domain invariants relied upon

Every "impossible edge case" excluded by frozen E15 authority (not
invented here):

* `m320(x) > 0` and `m0(x) <= 0` are frozen (e15 `[margins]`).
* The conservation identity `sum_r G[x,r] = delta_m(x) = m320(x) - m0(x)`
  is frozen (e15 `[component_evidence_decomposition]`).
* Therefore `positive_evidence_sum = delta_m(x) - negative_evidence_sum
  >= delta_m(x) >= m320(x)` for every valid disputed pixel (since
  `negative_evidence_sum <= 0` and `m0(x) <= 0`) — so `n_positive_
  components >= 1`, `positive_evidence_sum > 0`, and a `k_star` in
  `[1, n_positive_components]` satisfying the crossing condition always
  exists.

If any of these is observed violated at runtime, the kernel raises
`DfpPhase1bProvenanceFeaturesError` rather than fabricating a value —
this generalizes e15's own `k_star_non_finite_case_behavior` ("reject an
invalid non-finite case rather than inventing infinity").

## 6. CLI usage

```sh
python3 diagnostics/run_dfp_phase1b_provenance_features.py verify-protocol
python3 diagnostics/run_dfp_phase1b_provenance_features.py show-contract
```

Both commands are CPU-only, torch-free, and perform no execution. There
is no `run`, `resume`, `submit`, or `benchmark` command in this CLI.

## 7. Verification commands

```sh
python3 -m py_compile src/dfp_phase1b_provenance_features.py \
    src/dfp_phase1b_provenance_features_identity.py \
    diagnostics/run_dfp_phase1b_provenance_features.py
python3 -m pytest tests/test_dfp_phase1b_provenance_features.py -q
python3 diagnostics/run_dfp_phase1b_provenance_features.py verify-protocol
python3 diagnostics/run_dfp_phase1b_provenance_features.py show-contract
```

## 8. Authorization boundary

* `scientific_kernel_only = true`
* `runner_implemented_by_this_task = false`
* `real_inference_executed = false`
* `gpu_execution_performed = false`
* `phase1b_500_image_execution_authorized = false`
* `phase1b_5000_image_exact_provenance_authorized = false`

## 9. Next steps (explicitly deferred, never invented here)

* Independently re-verify this E21 identity, kernel, and its loader/CLI
  module against the committed HEAD.
* Resume implementation of the profiled, resumable 500-image diagnostic
  runner bound by E20 — now unblocked by this kernel, but a separate,
  later, separately-authorized task.
* Explicitly authorize the 500-image diagnostic execution itself.
* The 5,000-image exact masked-forward provenance question.
