# Matched k=11 vs k=12 finite-step (T=320) connectivity identity

A pre-registered, fail-closed identity for a **connectivity dose-response**
experiment: it compares matched k=11 and k=12 directed row-stochastic RWR
graphs under T=320 finite-step propagation, with every other inference
condition held fixed. This commit adds the identity, its static preflight,
and its structured-result verification -- it does not run the experiment,
implement the power-iteration evaluator, or add a production YAML config.

This is a connectivity dose-response experiment, not COVER-DR, DCR, SUR,
semantic-consensus repair, or learned graph reweighting -- see
`[prohibited]` in the identity TOML.

## The question

> What is the effect of removing exactly the twelfth ranked outgoing edge
> from every non-fallback graph row, with every other inference condition
> held fixed?

## Why k=11 is formed from the canonical top-12 prefix

The existing canonical graph builder selects each row's top-12 neighbours
by a stable descending argsort over `ReLU(cosine)^3` affinity
(`order = argsort(affinity, descending=True, stable=True)`, ties won by the
lower patch index), then takes `order[:, :12]`. **k=11 is defined as
`order[:, :11]`** -- the first 11 entries of that exact same ordering, then
independently row-renormalized. This is a **matched-prefix intervention,
equivalent to deleting the rank-12 edge and renormalizing** -- never an
independently selected k=11 graph. `graph.k11_construction_method` in the
identity TOML pins this policy, and a submitted result's
`graph_contract.k11_construction_method` must equal it exactly; a result
claiming `independently_selected_k11_graph` is rejected.

## Why this isolates one-edge-per-row removal

Because k=11 is a literal prefix of the k=12 selection sharing the same
similarity matrix, the same argsort, and the same tie-break rule, the two
graphs differ *only* in whether the twelfth-ranked edge exists (and its
consequent row renormalization). No other degree of freedom (which edges,
what order, what affinity) can vary between the two variants -- the
experiment measures the marginal effect of exactly one edge per row.

### The zero-affinity fallback is k-invariant

The canonical graph builder overrides any row whose entries are all
non-positive affinity (a "fallback row") with a self-loop of weight 1.
Because the argsort is descending, a row's twelfth entry can only be
positive if every earlier entry is also positive; a row is a fallback row
under k=12 if and only if it is a fallback row under k=11. Fallback rows
are therefore identical between the two variants by construction, and a
submitted result's `variant_results.k11.fallback_row_count` must equal
`variant_results.k12.fallback_row_count` exactly.

## Why k11 and k12 must share the backbone and finite-step backend

The identity's `[execution]` section requires one shared E3 backbone
forward pass, one shared snapshot of pre-sigmoid patch scores, one shared
set of DINO features, one shared affinity matrix, and one shared top-12
selection -- only the row truncation and its renormalization differ. A
result's `runtime_telemetry.backbone_forward_count`,
`snapshot_count`, `affinity_build_count`, and `top12_selection_count` must
each equal the window count exactly (never twice that count), and
`second_backbone_pass_count` must be 0. This is what makes the k=11 vs
k=12 comparison a genuine matched pair rather than two independent runs
that could differ for unrelated reasons (sampling noise, a different
backbone forward, a different affinity computation).

## Why T=320 is finite-step propagation, not an exact equilibrium

The recurrence `P^(t+1) = alpha * A_k * P^(t) + (1-alpha) * S0` run for
exactly 320 completed updates is a **finite-step approximation**, not a
solve to convergence (that is what the canonical CGLS solver does, in the
separate canonical RWR identity). The historical T=320 k=12 reference this
identity carries is explicitly labeled `finite_step_reference` -- never
"exact" -- and its extracted metrics
(`aAcc=48.528725957459656, mIoU=29.87719634810705, mAcc=54.13708897297259`
at 5000 images) are visibly close to, but numerically distinct from, the
canonical converged CGLS result in `evaluation_identities/e3_canonical_directed_rwr.toml`.
That gap is exactly the finite-step/equilibrium distinction, not a
reproduction error.

## Why the experiment measures connectivity dose near k=12

k=12 is the canonical, already-validated operating point (see
`evaluation_identities/e3_canonical_directed_rwr.toml`). Comparing it to a
matched k=11 -- one edge fewer per row, everything else fixed -- gives a
local dose-response reading: does the marginal rank-12 edge help, hurt, or
make no detectable difference to finite-step propagation quality?

## Interpreting the paired delta

`paired_delta = mIoU_k11 - mIoU_k12`, always recomputed from the reported
per-variant confusion statistics and never accepted as a value supplied
independently of that recomputation.

- **Positive**: removing the rank-12 edge improved finite-step mIoU --
  evidence the marginal edge was net-harmful (e.g. injecting noisy
  long-tail similarity) at this k.
- **Negative**: removing the rank-12 edge hurt finite-step mIoU -- evidence
  the marginal edge carried net-useful connectivity.
- **Null** (delta indistinguishable from zero): the marginal rank-12 edge
  makes no detectable difference to finite-step propagation quality at
  T=320.

## Why a null result remains scientifically useful

A null delta is not "no result" -- it directly answers the dose-response
question by locating a plateau: propagation quality at T=320 does not
depend sensitively on whether the eleventh or twelfth ranked edge is
present. That is informative for downstream graph-pruning or
graph-repair design choices, independent of whether the answer is
"more/fewer edges help." The identity's result schema is deliberately
symmetric: a null, positive, or negative delta are all structurally valid
results, and none is required or preferred by validation.

## Why result verification uses confusion statistics

`aAcc`/`mIoU`/`mAcc` are always recomputed dataset-level from each
variant's `intersection`/`union`/`predicted_pixels`/`ground_truth_pixels`
arrays (171 classes) using the exact mmseg dataset-level definitions
(micro-average for aAcc, macro nanmean over classes with a defined
denominator for mIoU/mAcc) -- never accepted as a bare reported number, and
never accepted from a rounded PrettyTable-style display value. This is the
same fail-closed posture used throughout this repository's other
identities: the sufficient statistics are the source of truth, and a
reported summary metric is only as trustworthy as its agreement with them.

## Closed unit and dtype vocabularies

**Metric unit**: `metrics.unit` is closed to exactly `"percent_0_100"` --
metrics are percent on `[0, 100]`, never a `[0, 1]` fraction and never a
bare `"percent"` (which does not by itself disambiguate the two).
**Compute dtype**: `propagation.compute_dtype` is closed to exactly
`"float32"` -- the finite-step recurrence is computed in FP32, matching
the dtype the existing E3/RWR affinity and propagation code already uses
(raw patch scores, `knn_weights`, and `propagate_scores`'s `alpha`/weight
tensors are all explicitly `torch.float32`). **Output dtype**:
`propagation.output_dtype` is closed to exactly `"float32"` for the same
reason -- the propagated result is never silently downcast or upcast.

All three (`SUPPORTED_METRIC_UNIT`, `SUPPORTED_COMPUTE_DTYPE`,
`SUPPORTED_OUTPUT_DTYPE`) are checked by exact string equality at load
time, with no normalization, case-folding, or whitespace-stripping, and no
alias table -- `"fraction_0_1"`, `"Percent_0_100"`, `"percent_0_100 "`,
`"fp32"`, `"Float32"`, and `"torch.float32"` are all rejected exactly like
any other unsupported value. These are schema vocabulary, not
experimental result values, so they live in this module as Python
constants rather than in the TOML.

**Window enumeration**: `geometry.window_enumeration` is closed to exactly
`sliding_window_geometry.SlidingWindowPlan.build (legacy clamped
slide_inference grid, row-major flat index order)` -- the legacy
**clamped** sliding-window grid, enumerated in **row-major flat-index**
order, and nothing else. A different callable, a different clamping
policy, or a different enumeration order each change which windows exist
and in what order, so each defines a different experiment identity, not a
cosmetic restatement of this one.

These fields, together with every other closed-vocabulary field this
identity defines (graph self-edge/fallback-row policy, affinity function,
tie-break rule, propagation initial iterate and recurrence formula,
convergence tolerance, checkpoint-resume contract, stitching averaging and
crop order, the DINO feature stage, the geometry clamping policy, the
window enumeration algorithm, and the metric precision source -- see
`SUPPORTED_*` in `src/matched_k11_k12_identity.py`) are **frozen
experiment-identity fields**. Changing any of them to a different, even
semantically equivalent, value does not describe a variant of this
experiment -- it describes a *different* experiment identity, and must be
pre-registered as one (a new TOML, a new `identity_name`, a new commit)
rather than edited in place.

## Why paired bootstrap later resamples images and recomputes dataset mIoU

The per-image per-class sufficient-statistic artifact this identity
pre-registers (`[metrics].per_image_per_class_artifact_required`,
`validate_per_image_statistics_artifact_metadata`,
`validate_per_image_statistics_additive_aggregation`) exists to support a
**future** paired bootstrap: resampling images with replacement and
recomputing the *dataset-level* mIoU from the resampled images' summed
confusion statistics, for both variants on the same resample. This is
explicitly **not** implemented in this commit, and the schema explicitly
never stores a per-image mIoU scalar as an authoritative estimator --
per-image mIoU is not additive and does not equal the dataset mIoU, so
averaging per-image mIoU values would silently produce a different
(wrong) quantity than the bootstrap this artifact is meant to support.

## Why this experiment is separate from stopped semantic COVER-DR

`docs/t4_signal_alignment_report.md` (the previous commit) concluded that
the evidence does not support semantic-consensus-guided COVER-DR
specifically, while explicitly leaving graph-topology and structural
connectivity questions open. This identity is exactly that follow-up: a
**structural, connectivity-only** dose-response probe (one edge per row,
matched pair, no semantic consensus, no cross-view agreement signal
anywhere in its contract) -- see `[prohibited]` in the TOML, which
excludes semantic consensus, T4 targeting, DCR, SUR, Sherman-Morrison
updates, edge-influence gradients, learned weights, and any independently
selected k=11 graph.

## Running the preflight

```sh
python verify_matched_k11_k12.py preflight --repo-root .
```

Optionally add `--check-checkpoint` to also require and hash the shared
canonical projection checkpoint. Preflight validates: branch ancestry
(`identity.required_ancestor_commit`), the identity's own closed TOML
schema and exact types, the existing E3 identity
(`evaluation_identities/e3_paired_soft_routing.toml`), the existing
canonical RWR identity (`evaluation_identities/e3_canonical_directed_rwr.toml`),
both parent identity files' SHA-256, relational equality of every field
this identity shares with its parents (dataset, crop/stride, affinity
power, top-k, alpha, score stage), the historical finite-step reference's
Git provenance (commit/path/blob/SHA-256) and its typed row selection, and
the pinned window-count reference's Git provenance. It requires no result
values -- there is no result to validate before the experiment has run.
Run with whichever interpreter has this repository's dependencies
installed (e.g. `<python-environment>/bin/python`); no particular
environment path is required by the tool itself.

## Verifying a future result

```sh
python verify_matched_k11_k12.py verify-result --result PATH --repo-root .
```

Validates a `talk2dino-matched-k11-k12-t320-result-v1` structured result:
both variants' schema, recomputed metrics, cross-variant invariants
(identical GT arrays, identical fallback-row counts, identical image/window
counts), the recomputed paired delta, and the full telemetry contract
(exactly one shared backbone/snapshot/affinity/top-12-selection pass per
window, zero prefix mismatches, no early termination, no solver fallback,
no CGLS call, no second backbone pass). Exits `0` on success and `2` with a
concise diagnostic on `stderr` on any validation failure; it never
partially validates a result as passing.
