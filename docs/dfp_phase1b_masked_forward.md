# DFP Phase 1B: Exact Masked-Forward Provenance Core

Implements `evaluation_identities/e15_dfp_phase1b_protocol.toml`'s
`[component_evidence_decomposition]` block: the exact, GT-free,
component-masked forward propagation and signed-evidence extraction
that will underlie the Phase 1B diagnostic features (`k_star`, `phi`,
`top1_positive_share`, ...) once real execution is authorized. This
document is a narrow operational summary; the e15 TOML identity is the
sole scientific authority.

**This is scientific-core implementation and synthetic verification
only.** No real-data execution driver, calibration run, Stage-B
modeling, or Gate-2 decision logic is implemented here. **This module
measures conservation residuals but does not decide whether they
pass. No Phase 1B provenance has been run on real data. No Gate-2
result exists. No GPU execution is authorized by this
implementation.**

## Frozen authority and dependency files

Read directly, never re-derived, before writing this module:

- `evaluation_identities/e15_dfp_phase1b_protocol.toml` /
  `src/dfp_phase1b_identity.py` — the formulas, the `fp64`
  accumulation requirement, `recurrence_variant_required = "k12"`, and
  every `*_forbidden`/`*_required` boolean this module's contract
  extraction re-checks.
- `src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py`
  — `finite_step_propagate`, the single unmodified finite-step k12
  kernel every production caller already uses.
- `src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py`
  — `DirectedTopKGraph`, whose `.matmul(rhs)` already accepts an
  arbitrary `[N, R]` right-hand side, which is what makes reusing the
  kernel unmodified for `R_chunk * |Q_w|` channels possible.
- `src/dfp_phase1a_recurrence_capture.py` — the single canonical
  `alpha`/`steps` extractor (`dfp_recurrence_contract_from_protocol`),
  reused rather than re-derived, so masked and canonical propagation
  are provably driven by the identical two numbers.
- `src/dfp_phase1a_stitching_pullback.py` — `apply_window_stitching_
  forward` (the canonical per-window `A_w`, already batched-width
  generic) and `StitchingPullbackPlan`.
- `src/dfp_phase1a_final_rescale_pullback.py` — `apply_final_rescale_
  forward` (`R`) and `iter_image_final_soft_memberships` (the
  committed, composed `A_final = R ∘ A_stitch` soft-membership source
  this module's seeds must be built from).
- `src/dfp_phase1b_compute_estimator.py` — `compute_window_query_
  classes`, the single frozen `Q_w` derivation this module's callers
  are expected to reuse (this module accepts `Q_w` as an input; it
  never re-derives it).

## Formulas (verbatim from the frozen identity)

```
Y0^(r)[w,j,c]  = M[r,w,j] * Y0[w,j,c]                                  (seed)
Y_{t+1,w}^(r)  = alpha * P_w @ Y_{t,w}^(r) + (1-alpha) * Y0^(r)         (t = 0..steps-1)
S_t^{(r)}      = A_final(Y_t^{(r)})            , A_final = R ∘ A_stitch  (t in {0, steps})
g_{x,r}        = [S_final^{(r)}(x,b) - S_final^{(r)}(x,a)]
                 - [S_0^{(r)}(x,b) - S_0^{(r)}(x,a)]
sum_r g_{x,r}  = delta_m(x) = m320(x) - m0(x)                          (conservation identity)
```

Sign convention: `m0(x) = S0(x,b) - S0(x,a)`, `m320(x) = S320(x,b) -
S320(x,a)`, `delta_m(x) = m320(x) - m0(x)`. Never inverted anywhere in
this module.

## Public API and tensor shapes

- `MaskedForwardContract(alpha, steps, propagation_dtype, accumulation_dtype)`
  — `dfp_masked_forward_contract_from_protocol(phase1a_protocol_identity, phase1b_identity)`
  re-validates `alpha`/`steps` via the single canonical Phase 1A
  authority and every `[component_evidence_decomposition]` boolean/
  string flag, before any graph construction.
- `MaskedSeedChunk.seed_flat`: `[node_count, component_count *
  query_class_count]`, **component-major then query-class-major**
  (column `r_local * Q + c_local`) — `build_masked_seed_chunk(...)`.
  `.unflatten(tensor)` inverts this exactly.
- `PropagatedChunk.t0` / `.t_final`: same shape as `seed_flat` —
  `propagate_masked_seed_chunk(graph, seed, contract=...)`, one call
  to the unmodified `finite_step_propagate` with `snapshot_steps=(0,
  contract.steps)`.
- `stitch_and_rescale_component_scores(...)` → `(S0, S_final)`, each
  `[1, component_count * len(query_class_union), ori_h, ori_w]`.
- `SignedContributionChunk.values`: `[disputed_pixel_count,
  component_count]` — **rows** ordered by increasing final-image flat
  index (`row * ori_width + col`), **columns** ascending global
  component id from `component_id_start` — `compute_signed_
  contributions(...)`.
- `ConservationMeasurements` bundles four `ResidualSummary` records
  (signed array, absolute array, max/mean absolute, finite/non-finite/
  compared-element counts) — no pass/fail field anywhere.

## Component-masked seed behavior

`build_masked_seed_chunk` is a strict pure function: exact elementwise
multiply only. It rejects boolean-as-float tensors, non-finite values,
device mismatches, shape disagreements, non-ascending or non-
contiguous global component ids, and non-ascending or duplicate query
class indices — all before constructing anything. It never
normalizes, clamps, thresholds, binarizes, rounds, or epsilon-adjusts
`membership_rows`; never silently drops a zero-evidence component;
never merges components; never creates a residual component; and its
signature contains no GT/target/oracle-shaped parameter.

## Precision architecture: FP32 graph storage, FP64 recurrence

`DirectedTopKGraph` stores `transition_weights` and `edge_affinities`
as FP32, as required by its committed validation contract
(`DirectedTopKGraph.validate()` raises `TypeError` on any other
dtype). The masked and canonical recurrence inputs -- the seed, the
propagated state, and `t0`/`t_final` -- are FP64
(`MaskedForwardContract.propagation_dtype`/`accumulation_dtype`).
During `graph.matmul`, the stored FP32 graph values are converted to
the caller tensor's dtype (`self.transition_weights.to(dtype=rhs.dtype)`)
*before* multiplication; consequently the recurrence arithmetic and
states this module produces are FP64 throughout, never FP32. Canonical
and masked runs use the same graph instance and this identical
conversion path -- there is no separate masked-only or canonical-only
precision handling anywhere in this module. This satisfies
`altered_precision_between_canonical_and_masked_runs_forbidden`.

FP32 graph storage does not mean the graph was recomputed or
approximated for Phase 1B: it is the same, unmodified, already-
committed `DirectedTopKGraph` storage contract every other production
caller already uses; this module never constructs, converts, or
re-quantizes a graph of its own. Signed evidence and conservation sums
remain FP64 throughout (see "Raw conservation measurements" below).
This precision architecture introduces no tolerance, no epsilon, no
autocast, no TF32, and no mixed-precision arithmetic -- FP32 describes
only the graph's own frozen storage, never the recurrence, evidence,
or conservation dtype.

## Recurrence reuse and exact update counts

`propagate_masked_seed_chunk` calls `finite_step_propagate` exactly
once per (window, component-chunk) pair, with `snapshot_steps=(0,
contract.steps)` — one continuous pass, `contract.steps` completed
`alpha * graph.matmul(p) + (1-alpha) * s0` updates (independently
confirmed via `graph.matmul` call-count instrumentation in the test
suite), never a second propagation, never an early stop. No graph is
ever constructed, symmetrized, or re-selected inside this module —
`graph` is always caller-supplied, and its `k == 12` is checked, not
assumed.

## Chunk-size equivalence

`iter_component_chunks(component_count, component_chunk_size)` covers
every component exactly once, in ascending order, with an explicit,
non-data-dependent positive integer chunk size. The test suite
independently confirms that reassembling a component-chunked run
(sizes 1, 2, 3) reproduces the single-chunk (size = full count) result
exactly, for both the seed and the propagated field.

## Canonical stitching / final-rescale behavior

`stitch_and_rescale_component_scores` reuses `apply_window_stitching_
forward` (per window, per snapshot) and `apply_final_rescale_forward`
(once per snapshot) unmodified — no nearest-neighbor path, no
alternative stitching convention, no dense final-pixels × patch-
sources matrix. Because different windows can have different frozen
`Q_w` sets, this function aligns every window's contribution onto the
**ascending union** of the participating windows' `Q_w`s
(`compute_query_class_union`) before accumulating: a window whose own
`Q_w` omits a class contributes an exact zero for that class in the
shared accumulator — mathematically correct, since the frozen `Q_w`
definition guarantees every window covering a given disputed pixel
already includes that pixel's `a(x)`/`b(x)` in its own `Q_w`. This is
a documented, exact (never approximate) alignment step, not a second
`Q_w` derivation.

## Signed contribution formula and orientation

`compute_signed_contributions` gathers `S0`/`S_final` at exactly the
`a(x)`/`b(x)` columns (resolved against the shared query-class union)
for every listed disputed pixel and computes `g_{x,r}` per the formula
above. A required class absent from the union fails closed. Disputed
pixels must already be supplied ordered by increasing final-image
flat index — this function validates that order rather than silently
sorting.

## Raw conservation measurements — no tolerance

Four residuals, computed on already-given values only:

- `rho_M(j) = sum_r M[r,w,j] - 1` (membership partition)
- `rho_0(j,c) = sum_r Y0^(r)[w,j,c] - Y0[w,j,c]` (seed reconstruction)
- `rho_320(j,c) = sum_r Y_final^(r)[w,j,c] - Y_final[w,j,c]`
  (propagated reconstruction, compared against a canonical reference
  whose provenance is checked by `require_same_job_canonical_
  reference` — never a previously serialized tensor loaded from disk
  as the sole comparison authority)
- `rho_g(x) = sum_r g_{x,r} - delta_m(x)` (signed-margin conservation)

**`DO NOT DEFINE A TOLERANCE`** — enforced structurally: no boolean
pass/fail field exists anywhere in `ResidualSummary` or
`ConservationMeasurements`; no `1e-*` constant, no machine-epsilon
multiplier, no relative tolerance, and no `torch.allclose`-based
decision appears anywhere in this module (independently confirmed by
an AST-identifier scan in the test suite — immune to this document's
own prose, which necessarily *mentions* these concepts to explain
their absence). Constructing a residual never raises on a large or
nonzero value — only non-finite (NaN/Infinity) **input** is rejected,
which is a structural precondition failure, not a conservation
decision. **The future 200-window FP64 calibration step establishes
the tolerance this module deliberately does not define.**

## Memory complexity

`estimate_chunk_memory` reports shapes and estimated byte counts only
— never a pass/fail resource decision, never an automatic smaller
chunk size. Peak masked recurrence state is `O(node_count *
component_count * query_class_count)`; stitched working memory is
bounded by one component chunk's own channel width, never all
components, never `N × N`, never a final-pixels × total-patch-sources
matrix.

## GT isolation

No function in this module's public API accepts a GT/target/harmful/
beneficial/oracle-shaped parameter (checked via `inspect.signature`
over every `__all__` entry, and via `dataclasses.fields` over every
exported dataclass); no GT/annotation/dataset/model/checkpoint loader
is ever referenced (checked via an AST scan of every `Call` node); and
no Stage-B target-category, `harmful_binary_label`, or diagnostic-
model construct (`k_star`, `phi`, `top1_positive_share`, logistic
regression) appears anywhere as executable code (checked via an AST
identifier scan, which — unlike a raw text search — is immune to this
very document's and the module's own explanatory prose).

## Exclusions from this step

No masked-propagation execution on real data. No real image, GT, or
annotation access. No PAMR, text-distance, RGB, or boundary feature.
No `k_star`/`phi`/top-1-share/Gini/entropy computation. No Stage-B
target construction. No logistic regression, bootstrap, or Gate-2
decision logic. No execution driver or SLURM launcher. No GPU
allocation. Every synthetic test in `tests/test_dfp_phase1b_masked_
forward.py` runs CPU-only against hand-built fixtures.

A subset of the required synthetic tests (recurrence exactness,
canonical-geometry, and the parts of the contract-extraction test that
call into `src.dfp_phase1a_recurrence_capture`) need the real,
committed `models.dinotext.cover_dr` / `src.dfp_phase1a_stitching_
pullback` / `src.dfp_phase1a_final_rescale_pullback` machinery, which
in turn needs a working `cv2` import. Some default/login environments
on this cluster either lack an OpenCV module load entirely (`cv2` not
importable at all) or have one whose NumPy ABI predates the installed
NumPy 2.x (`_ARRAY_API`/`numpy.core.multiarray failed to import`) — the
same gap that can make `tests/test_dfp_phase1a_stitching_pullback.py`,
`tests/test_dfp_phase1a_final_rescale_pullback.py`, and others
uncollectible there, entirely unrelated to this module's own code.
Those specific tests detect the exact known failure signature and
`pytest.skip` with a precise, attributable reason rather than error
opaquely or silently pass.

### Compatible environment

This repository's own existing SLURM launchers (e.g.
`dfp_phase1a_preflight.sbatch`, `dfp_phase1a_execute_coco_stuff_500.sbatch`)
already establish a working module order: load an OpenCV module built
against this environment's NumPy version *before* activating the
project's existing Python virtual environment. In a login shell:

```bash
module load opencv/4.14.0
source <existing-project-environment>/bin/activate
export CUDA_VISIBLE_DEVICES=""
python -c "import cv2; print(cv2.__version__)"
```

With this module order, `cv2` imports cleanly and the real Phase 1A
geometry modules (`src.dfp_phase1a_horizon_stitching`,
`src.dfp_phase1a_stitching_pullback`,
`src.dfp_phase1a_final_rescale_pullback`, and the rest of
`_load_dependencies()`'s bundle) import and execute normally, with no
stubbed OpenCV, no replacement erosion, and no `importlib`-based import
bypass. Loading an OpenCV module only resolves this runtime import
dependency — it does not modify, recompute, or approximate any
scientific result. Using this environment, canonical multi-window
final-geometry stitching and rescale (`stitch_and_rescale_component_
scores`), signed contributions, and the conservation identity have all
been independently executed numerically end-to-end (not merely
inspected as source) and matched an independently hand-built reference
to floating-point machine precision; source inspection alone is no
longer the only evidence for those results.

None of this means real Phase 1B provenance has been run on real data,
that a Gate-2 result exists, or that GPU execution is authorized by
this module — those remain unchanged (see the top of this document).
Conservation residuals are still measured and reported raw, with no
hard-coded tolerance defined anywhere in this module (see "Raw
conservation measurements" above).

## Calibration is the next step

The next, separately authorized step is the deterministic 200-window
FP64 conservation-calibration runner (`phase1b_conservation_
calibration.json`), which will exercise this module's real recurrence
and stitching paths against real data and establish the tolerance this
module intentionally leaves undefined. This document and this module
freeze the exact, tolerance-free measurement core that step will
consume — nothing more.
