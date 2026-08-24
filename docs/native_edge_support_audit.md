# Native cross-view directed-edge support audit

## Scientific question

For a directed k12 graph edge `i -> j` built inside one sliding-window
crop, **among the OTHER windows of the same transformed image whose native
ViT patch grid contains an exact counterpart of both `i` and `j`, how many
independently reconstruct the same directed edge `i_v -> j_v` in their own
top-12 graph?**

This measures whether support is *defined*, how often edges recur across
views, how support varies by source rank and geometry, and whether
support-defined rows overlap the locations where the current diffusion
already fails (`source_row_misclassified`/`canonical_stitched_misclassified`).
**It does not test whether pruning those edges improves segmentation** —
this stage is read-only structural reachability measurement, never an
intervention, never an efficacy claim.

## What this is not

Never selects a T4/T4-prime target, never builds crop-consensus semantic
labels, never touches DCR/SUR, never votes (majority/unanimity), never
prunes (random/lowest-cosine/structural), never runs a Sherman–Morrison
counterfactual or adjoint-gradient solve, never symmetrizes the graph,
never introduces a learned threshold or confidence/entropy/margin gate,
never transports DINO features (nearest-neighbor or bilinear), never
interpolates graph topology, and never uses centrality as an inference-time
filter. It never deletes, reweights, or edits a single graph edge.

## Native patch-coordinate authority

Uses discrete native ViT patch coordinates only — never a float centre
comparison. For patch-grid node `i = row * 32 + col` (patch size `p=14`,
crop origin `(y0, x0)`), another window `v` with origin `(yv, xv)` has an
exact counterpart of `i` only when `(y0 - yv)` and `(x0 - xv)` are each
**exactly** divisible by `p`, mapping

```
row_v = row + (y0 - yv) / p
col_v = col + (x0 - xv) / p
```

and the mapped `(row_v, col_v)` falls inside `v`'s 32x32 grid. The
canonical, non-clamped stride (224) maps through exact ±16-patch offsets.
Clamped terminal windows may or may not be aligned depending on the exact
clamp shift — never rounded, quantized, or given a tolerance radius; an
unaligned pair contributes nothing (see
`native_window_pair_offset`/`map_native_node` in
[`native_edge_support.py`](../src/open_vocabulary_segmentation/models/dinotext/cover_dr/native_edge_support.py)).

This module never imports `sliding_window_geometry` (mirroring the
dependency-light, duck-typed pattern already used by
`matched_power_evaluator.py`/`stitching_control.py`); it implements the
same offset-and-bounds arithmetic as
`PatchGridSpec.native_patch_node` directly, and
`tests/test_native_edge_support_api.py` cross-validates the vectorized
implementation against that scalar reference exhaustively across several
window-plan configurations (aligned, clamped-aligned, clamped-unaligned,
non-square).

**Vectorization insight**: the origin displacement between two windows —
and hence the constant integer patch-unit offset it implies — is the same
for *every* node in the grid; only whether a given node's mapped
coordinate stays inside the target's bounds varies per node. This lets
`compute_window_support` compute observer/support counts for all `1024 x
12` edges of a source window against every other window with a handful of
vectorized tensor ops per window pair, never a per-node Python loop, and
never rebuilding or rerunning a graph.

## Directed support definition

Support is **topology-only**: an edge counts as supported through window
`v` only if `v`'s independently-built top-12 graph contains the directed
edge `i_v -> j_v`. The reverse edge `j_v -> i_v` never counts. The source
window itself is never counted as an observer. Support is never weighted
by confidence, crop position, distance, or graph weight — `observer_count`
and `support_count` are exact integer tallies; `support_fraction` is
reported descriptively alongside them, never used to gate or filter
anything internally.

## Deterministic descriptive ranking

`rank_edges_for_row` never edits anything — it only identifies which edge
*would* be structurally least supported, under a frozen lexicographic
order: defined before undefined; lower support fraction; lower support
count; larger observer count (zero of more observations is stronger
evidence than zero of fewer); lower source graph weight; larger neighbor
rank; lower destination node index as the final tie-break. Recorded
verbatim in `evaluation_identities/e12_native_edge_support_audit.toml`
`[ranking]` and cross-checked against `result.ranking_definition` by
`src.native_edge_support_report`.

## GT reachability diagnostics

Computed **strictly after** every row's support record is frozen — GT
never feeds back into support computation or ranking
(`gt_diagnostics.influences_support_or_ranking = false` in the identity,
enforced by construction: `build_row_correctness_record` takes an already-
finalized row summary's classification as input, never the reverse). Two
per-row indicators, both nearest-neighbor-sampled at the row's own
physical native patch centre — `centre = origin + patch_size*index +
patch_size // 2` (`native_patch_centre_pixel`; for `patch_size=14` this is
exactly the identity-declared `origin + 14*index + 7`, integer-only, never
a float/rounded comparison) — and rescaled into the dataset's `ori_shape`
pixel space with the same `align_corners=True` linear convention
`finalize_prediction` already uses (`rescale_pixel_align_corners`):

- `source_row_misclassified`: the source window's own post-k12
  finite-propagation class at node `i` (a direct per-node argmax — sigmoid
  is monotonic per class channel and never changes the argmax, so it is
  never actually applied here) vs. GT.
- `canonical_stitched_misclassified`: the canonical `uniform_probability`
  stitched-and-finalized label at the same physical pixel vs. GT (reuses
  `StitchAccumulator`/`build_stitch_weight`/`finalize_prediction` from
  `stitching_control.py` directly — never a re-derivation of the
  production stitching path).

Both respect `dataset.ignore_index`, tallied separately
(`ignored_gt_count`). Four cross-tabulations (support-defined vs.
source-correct, all-edges-defined vs. source-correct, and the same two
against stitched-correct) are reported as raw counts — never described as
an accuracy improvement or pruning headroom.

## Shared execution

Per window: exactly one backbone/snapshot pass, one directed top-12 graph
build (both always), and — only when diagnostics are needed for the GT
indicators — one T=320 finite-step propagation and one sigmoid+
interpolation. Propagation is **never required for support computation
itself** (`propagation_required_for_diagnostics_only = true` in the
identity); the mechanics20 evaluator always requests diagnostics, so in
practice `propagation_calls == windows_processed_total` for every produced
result, but the API supports skipping it entirely. Cross-view comparisons
scale with window pairs but never invoke the model or the graph builder —
see `process_one_window_for_audit`/`compute_window_support` in
`native_edge_support.py`.

## Aggregate funnel

A 13-step row/edge survival funnel (images → windows → graph rows →
directed edges → rows with another crop → rows with an aligned observer →
edges with an observer covering both endpoints → rows any/all
support-defined → rows with a defined zero-support edge → misclassified
rows → misclassified rows any/all support-defined), plus window/pair- and
diagnostic-level counters: `aligned_window_pairs`/`unaligned_window_pairs`
(each unordered window pair in an image, counted once), `clamped_windows`/
`non_clamped_windows`, `unanimous_support_edges` (support_count ==
observer_count > 0), `direction_reversal_only_cases` (a defined edge with
zero forward support where at least one of its own observers contains the
*reverse* edge instead — informational only, computed from the exact same
adjacency lookup as the forward check, never substituted for it),
`rows_with_unique_least_support`/`rows_tied_for_least_support` (whether
more than one defined edge shares the ranked-first edge's primary evidence
tuple — fraction, support count, observer count — before the
weight/rank/destination tiebreaks that exist only to pick one answer), and
`correct_rows_any_defined` (paired with `misclassified_rows_any_defined`
to compute the decision-output risk ratio below). Every subset/count
invariant is checked exactly by `validate_funnel_invariants`
(`src.native_edge_support_checkpoint`) — never approximated, never
skipped.

## Geometry stratification

Fixed, identity-declared bins (`["0","1","2","3-4","5-7",">=8"]` in patch
units), decided in advance rather than discovered after seeing results:
`crop_edge_band_histogram` (source patch distance from its own crop's
edge), `image_edge_band_histogram` (distance from the transformed image's
own boundary, using the same patch-centre pixel the GT indicators sample),
`displacement_band_histogram` (rounded Euclidean patch distance between an
edge's source and destination node), and `affinity_rank_histogram` (source
edge rank 1–12, i.e. the graph's own top-12 ordering).

## Decision output

`classify_reachability` (`src.native_edge_support_report`) labels the
completed funnel — never selects, prunes, or edits anything — into one of
four identity-declared outcomes (`identity["decision"]["outcomes"]`),
applied in order:

1. **`INCONCLUSIVE`** if `misclassified_rows` is below
   `decision.minimum_misclassified_rows_for_conclusive`.
2. **`ALIGNMENT_LIMITED`** if the undefined-edge fraction
   (`1 - edges_with_observer/directed_edges`) meets or exceeds
   `decision.alignment_limited_min_undefined_edge_fraction`.
3. **`REACHABLE`** if the risk ratio of support-defined coverage among
   wrong rows vs. among correct rows exceeds
   `decision.reachable_min_risk_ratio`.
4. **`STRUCTURALLY_UNREACHABLE`** otherwise.

No threshold is a bare Python literal — every one is read from
`evaluation_identities/e12_native_edge_support_audit.toml` `[decision]`.
The result records both `decision_output` and a human-readable
`decision_rationale`; `verify_record` independently recomputes the
classification from the result's own `funnel` and rejects a mismatch. This
stage never claims efficacy — the classification is a reachability read
only, feeding the separately-defined 20-image gate that follows.

## Artifacts

Strict, versioned JSON only (no per-class NPZ arrays are needed for this
stage's aggregate/histogram/cross-tab schema): a per-image-stats manifest
recording each image's own funnel-subset row (used to reconcile against
the checkpoint's running totals on resume, byte-exact), a checkpoint, and
a self-verified result. Never persists full DINO feature tensors, full
score volumes, all observer lists for all edges (only a small, explicitly
disabled-by-default, identity-locked bounded-debug sample is permitted),
or private/full image paths.

## Checkpoint semantics

Reuses the same hardened pattern as every other E12 evaluator
(`src.native_edge_support_checkpoint`, structurally parallel to
`src.stitching_control_checkpoint`): exact canonical image prefix, no
duplicate/skipped/reordered image, strict JSON, exact identity/run-mode
binding, atomic writes, one-complete-image checkpointing, and checkpoint
↔ artifact reconciliation. A checkpoint recorded under a different
identity (including one from the stitching-control suite or the matched
evaluator) is rejected outright.

## Run modes

Only `mechanics20` (exactly 20 canonical images) exists at this stage
(`scripts/slurm/e12_native_edge_support_audit_mechanics20_h100.sbatch`) —
structural audit only, no efficacy claim is made or authorized here, so no
mechanics100/full5000 script exists yet.

## Explicit exclusions

See `evaluation_identities/e12_native_edge_support_audit.toml`
`[prohibited]` for the full, identity-locked list (T4/T4-prime selection,
crop-consensus labels, DCR, SUR, majority/unanimity voting, semantic proxy
losses, edge deletion/reweighting, graph symmetrization, k11-construction-
as-intervention, random/lowest-cosine/structural pruning, Sherman–
Morrison counterfactuals, adjoint gradients, dense inverses, learned
thresholds, confidence/entropy/margin gates, DINO-feature transport,
graph-topology interpolation, centrality-as-filter, PAMR, a changed `k`,
changed propagation, changed crop/stride, a second backbone/graph pass per
observer, and any pruning-efficacy claim from this evaluator).
`tests/test_native_edge_support_api.py`'s scientific-isolation test
statically (AST-identifier, never raw text) confirms none of these appear
as an identifier in `native_edge_support.py`.

## Limitations

- The shared-execution CLI is verified only via CPU/synthetic fixtures in
  this stage (per HPC login-node policy) — a real 32x32-grid, T=320
  propagation, multi-window (2x2) synthetic run, but not the real GPU
  model/dataset. It has not been run on the real GPU/dataset as part of
  this implementation.
- `dataset.get_gt_seg_map_by_idx` is assumed available (mmseg's own
  contract, the same method `dataset.pre_eval` itself calls internally) —
  a dataset without it fails closed.
- This stage produces the reachability measurement only; whether pruning
  low-support edges would help is explicitly out of scope, deferred to a
  later, separate stage that this audit's artifacts are designed to feed.
