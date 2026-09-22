# T320 Target-Source Authority

Task: `FREEZE_T320_TARGET_SOURCE_AUTHORITY`.
Frozen by: `research_specs/semantic_evidence_t320_source_v1.toml` +
`research_specs/semantic_evidence_t320_source_manifest_v1.json`, loaded and
independently re-verified by `src/semantic_evidence_t320_source_identity.py`.

This document explains what this authority is, what it freezes, and what it
explicitly does **not** authorize. It does not implement, port, or execute
any T320 code.

## What this authority is

The committed semantic-evidence feasibility contract
(`research_specs/semantic_evidence_feasibility_v1.toml`) already binds T0 to
committed E3 unary inference, and deliberately leaves T320 —
the prediction after exactly 320 completed steps of a finite-step graph
propagation — `DEFERRED_REQUIRES_SUCCESSOR_AUTHORITY`. This authority is
that successor: it binds the exact historical Git source, blob identity, and
scientific formulas that define T320, and the exact conditions a future,
separately-verified isolated implementation must reproduce.

It freezes **provenance and scientific semantics only**. It copies no code,
modifies no existing file, and executes nothing.

## Why e12-connectivity-analysis is the selected lineage

`e12-connectivity-analysis` (tip commit `00392527937253f93b9f3fe36b76347b4a6f1eb0`)
is a descendant of the reviewed E3 base and contains a complete, working
implementation of the finite-step T320 kernel: `graph.py`
(directed top-k affinity graph construction), `finite_step_regime.py`
(the `finite_step_propagate` recurrence itself), and
`matched_power_evaluator.py` (the recommended driver's window stitching and
final-prediction assembly). All three, together with `dinotext.py`,
`masker.py`, `dinotext_seg.py`, and `sliding_window_geometry.py`, form a
complete, closed, seven-file source closure — no partial or synthetic
reconstruction was needed.

## Why e11-cover-dr-1 is rejected

`e11-cover-dr-1` is **not** rejected because its files are "present and
empty." A prior audit pass made exactly that claim, based on piping a
failed `git show` (which writes an error to stderr and nothing to stdout)
into `sha256sum` — which happens to produce the same SHA-256 as a genuine
empty file, if the pipeline's exit code is never checked. Re-derived
directly in this task's prerequisite gate: `git ls-tree e11-cover-dr-1 --
<path>` returns **no tree entry at all** for `finite_step_regime.py`,
`stitching_control.py`, `matched_power_evaluator.py`,
`native_edge_support.py`, and `coco_object_evaluator.py`, and `git show
e11-cover-dr-1:<path>` exits with status 128 ("does not exist"). These
files are **absent from the tree**, not present-and-empty. Either way, the
practical conclusion is the same — `finite_step_propagate` is not defined
anywhere on `e11-cover-dr-1`, so it cannot supply the T320 recurrence — but
this document states the precise, re-verified reason rather than repeating
the earlier, imprecise one.

## Why later dfp-phase1a-diagnostics work is unnecessary

`git log --oneline <e12-tip>..dfp-phase1a-diagnostics -- .../cover_dr/` is
empty: zero commits touch any `cover_dr/` file between e12's tip and
`dfp-phase1a-diagnostics`'s tip. The scientific kernel this authority
freezes is already complete and unchanged at e12's tip; the ~90 additional
Phase-1B/BALP-development commits on the dfp lineage add a downstream
research-analysis layer (BALP candidate computation, calibration,
conservation-tolerance QA, profiling) and a *different* final-rescale /
stitching / prediction-extraction wrapper, none of which are needed to
define T320 itself. Pulling in that lineage would add unnecessary source
surface and a divergent `align_corners` convention (see below) without
adding anything this authority needs.

## The exact seven-file closure

| # | Role | Path |
|---|---|---|
| 1 | Graph construction | `src/open_vocabulary_segmentation/models/dinotext/cover_dr/graph.py` |
| 2 | Finite-step recurrence | `src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py` |
| 3 | Matched T320 driver / stitch / finalization | `src/open_vocabulary_segmentation/models/dinotext/cover_dr/matched_power_evaluator.py` |
| 4 | Window geometry | `src/open_vocabulary_segmentation/segmentation/evaluation/sliding_window_geometry.py` |
| 5 | E3 model / unary integration | `src/open_vocabulary_segmentation/models/dinotext/dinotext.py` |
| 6 | E3 unary scoring | `src/open_vocabulary_segmentation/models/dinotext/masker.py` |
| 7 | Inference / class-order / final-align-corners boundary | `src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py` |

Every path, Git blob id, and SHA-256 is recorded in
`research_specs/semantic_evidence_t320_source_manifest_v1.json` and bound to
a single `canonical_source_closure_digest`, computed with a length-prefixed,
whitespace/order/locale-independent encoding (see
`[source_manifest_binding]` in the TOML and
`compute_canonical_source_closure_digest` in the loader). The v2 encoding
covers every field of each record, including `required` and
`reason_required` — an earlier v1 encoding omitted those two fields from
the digest (their integrity relied on schema validation alone); v2 closes
that gap and is now the sole encoding this authority uses.
`manifest.canonical_encoding` must equal exactly `"length_prefixed_utf8_v2"`
(case-sensitive, no surrounding whitespace, string type only) — the loader
implements only that one version and never selects encoding behavior
dynamically from an untrusted label; `"length_prefixed_utf8_v1"`,
`"length_prefixed_utf8_v3"`, any differently-cased or padded variant, and
any non-string value are all rejected before the digest is ever recomputed.

`dinotext.py`, `masker.py`, and `dinotext_seg.py` are all modified relative
to the reviewed E3 base — this is disclosed, not hidden. Every modification
follows a verified-additive pattern: the original E3 code path is preserved
byte-identically as one branch of a new conditional or extracted method
(confirmed for `dinotext_seg.py`'s `encode_decode` RWR-disabled branch), and
`DINOTextSegInference.align_corners = False` remains unconditionally
hardcoded regardless of which branch executes.

### The dinotext_seg.py record is narrowed to one attribute, not the whole class

Record 7's `required_symbols` is exactly `["DINOTextSegInference.align_corners"]`
— not the broad class name. `dinotext_seg.py` contains an RWR-enabled
`encode_decode` branch that imports `RWRInferenceConfig`, `RWRRuntimeSummary`,
and `apply_rwr_to_e3_snapshot` from `models.dinotext.cover_dr` (the latter
defined in `cover_dr/inference.py`). That branch is present in the
historical file, but it is **never reached** by the recommended T320 path:
the recommended driver never constructs `DINOTextSegInference` with an
`rwr=` override, and `matched_power_evaluator.process_one_window` calls
`inference.model.masks_from_patch_scores` directly — never
`DINOTextSegInference.encode_decode`. The only load-bearing surface of this
file for T320 is the single `self.align_corners = False` assignment in the
class's own initialization path, verified by AST parsing (never an import)
via `verify_align_corners_locator` in the loader: it confirms the class
exists exactly once, the assignment exists exactly once and is a **direct
statement of `DINOTextSegInference.__init__` specifically** — not merely
somewhere in the class body — and that its value is the literal constant
`False`. Failing closed on a missing class, a missing or duplicate
assignment, an assignment in any other method (`configure`, `encode_decode`,
a property setter), a nested helper defined inside `__init__`, class-level
scope, or any other value. `apply_rwr_to_e3_snapshot` and its providing
files (`cover_dr/inference.py`, `cover_dr/rwr.py`) remain correctly excluded
from the seven-file closure — a future isolated implementation must not
import or reproduce that branch.

### Required-symbol locator grammar (all seven records)

Every entry in every record's `required_symbols` is now AST-verified
against its bound historical Git blob — not only record 7's. The grammar is
closed to exactly two forms:

- **Bare symbol** (`SymbolName`, no dot): resolves to exactly one
  `class`/`def`/`async def` named `SymbolName`, found either at module
  level or as a direct method of some class (the two pools are combined and
  must total exactly one match). A nested function, a name that appears
  only in a comment or docstring, or a name that is only imported (never
  locally defined) never counts as a match. This form covers all of records
  1–6's symbols, e.g. `build_directed_topk_graph` (a top-level function in
  `graph.py`) and `mask_from_similarity` (a method of `DINOTextMasker` in
  `masker.py`, unique across that file).
- **Instance-attribute locator** (`ClassName.attribute_name`, exactly one
  dot): `ClassName` resolves to exactly one top-level class; within it,
  exactly one `self.attribute_name = <value>` assignment must exist, and it
  must be a direct statement of that class's own `__init__` — never a
  nested helper, another method, class-level scope, or a property setter.
  No other assignment to the same attribute may exist anywhere else in the
  class. This form is currently used only by record 7's
  `DINOTextSegInference.align_corners`; `__init__` is the only enclosing
  method the grammar supports, because that is the only case this
  authority's seven records require — the grammar is not extended further.

No wildcard, call, indexing, or relative-import syntax is accepted, and
more than one dot is rejected. Source blobs are parsed with `ast.parse`
only; they are never imported or executed. This symbol verification
strengthens provenance — it proves the manifest's semantic-symbol claims
correspond to the bound source, not merely that the file's bytes are
genuine — without changing any scientific formula, source commit, source
path, blob hash, or the canonical digest itself.

## Graph and recurrence formulas

Graph (`build_directed_topk_graph`, k=12, affinity_power=3.0): cosine
similarity of L2-normalized DINO patch features, negative affinities
clamped to 0 before the power, self-similarity excluded via
`fill_diagonal_(-inf)`, a **stable** descending top-k sort (ties broken by
lowest original column index), row-normalized (not symmetrized) transition
weights, and a self-loop-with-weight-1 fallback for any node whose top-k
affinities sum to exactly zero.

Recurrence (`finite_step_propagate`): `P(0) = S0`; `P(t+1) = 0.98 * A @ P(t)
+ 0.02 * S0`, for exactly `t = 0..319` (320 completed updates), with no
convergence test, no tolerance, no early stop, and no adaptive step count —
the function's own docstring states this explicitly. `S0` is defensively
cloned once and never mutated. T320 is the snapshot at the final step, 320,
not an intermediate one.

Full formula detail — including matmul orientation, dtype behavior,
zero-degree handling, and non-finite failure policy — is frozen field-by-
field in `[graph_contract]` and `[recurrence_contract]` in the TOML, each
statement pointing to its exact source location.

## Internal upsample (true) vs. final rescale (false)

These are **two separate, independently hardcoded** bilinear interpolation
operations, not one `align_corners` value read twice:

- **Operation A** (internal): `DINOText.masks_from_patch_scores` upsamples
  each window's propagated per-patch scores to crop-pixel resolution,
  hardcoded `align_corners=True`. This runs once per window, before
  stitching.
- **Operation B** (final): `matched_power_evaluator.finalize_prediction`
  rescales the fully-stitched canvas to the original image resolution,
  immediately before the trailing argmax. Its `align_corners` value is read
  live off the production `DINOTextSegInference` object
  (`align_corners = inference.align_corners`), which this authority
  confirmed — via a fresh `git grep` across the entire source-commit tree —
  is the **sole** `.align_corners =` assignment anywhere in that tree:
  `self.align_corners = False`, hardcoded unconditionally in
  `dinotext_seg.py`. The recommended e12 driver's own in-source comment
  exists specifically to prevent a future reader from conflating these two
  values.

This authority freezes final rescale `align_corners = False` because that
is E3's own convention, read live rather than chosen to match any result.

## sliding_window_geometry.py inclusion / window_cache.py exclusion

`sliding_window_geometry.py` is a pure-geometry, stdlib-only, torch-free
module directly imported by the recommended driver
(`SlidingWindowPlan.build`, `SpatialSize`) to enumerate window placement —
it is part of the closure. `window_cache.py` imports torch and serves the
excluded CGLS/canonical-RWR solver path; an AST-based parse of every import
statement in `matched_power_evaluator.py` (fixing an earlier, naive
text-filter check that false-positived on a module-docstring prose line)
confirms zero actual references to `window_cache` anywhere in the T320 call
chain — its four appearances in that file's source are comments and
docstrings only. Adding it to the source closure is rejected by the loader.

## The historical 350-image divergence, precisely

A completed DFP Phase-1B BALP-development run over 350 COCO-Stuff images
(execution commit `804a774588b97e43e0347343089a0209ae28a5c9`, artifact
digest `6d75c370bcb03e10698429467053c3a19bc42f1c7405ceb9fccff83e97c8af80`)
used the **same, byte-identical** graph, recurrence, and geometry kernels as
this authority — but a **different** stitching function
(`dfp_phase1a_horizon_stitching.stitch_dfp_phase1a_horizons` rather than
`stitch_one_image`), a **different** final-rescale function
(`dfp_phase1a_final_rescale_pullback.apply_final_rescale_forward`, using
`align_corners=True`, frozen by a separate, DFP-specific identity —
`evaluation_identities/e13_dfp_phase1a_protocol.toml`'s
`rescale_align_corners = true`), and a **different** prediction-extraction
function (`lowest_index_argmax`, which explicitly guarantees a lowest-index
tie-break, unlike the recommended path's plain, tie-break-undocumented
`torch.argmax`).

## Why +1.9405pp is contextual only

The 350-image run measured a T0→T320 mIoU gain of +1.9404718pp — but it did
so using `align_corners=True` for the final rescale, not the `False` value
this authority freezes. Citing that gain as expected or matched evidence
for the `False`-setting authority would conflate two different final-rescale
paths that happen to share an otherwise-identical kernel. This authority
therefore records the 350-image figures purely as historical, contextual
disclosure — `matched_evidence_for_this_authority = false` — and freezes no
expected T320 mIoU of its own. `src/semantic_evidence_t320_source_identity.py`
rejects any attempt to mark this evidence as matched, to claim complete
end-to-end path identity, or to cite the gain as expected performance.

## GT isolation

T320, as frozen, depends only on original image/model inputs, the unary
score field, image/window geometry, the graph, the recurrence, and
stitching/rescale/class-ordering — never on annotations, GT labels, ignore
masks, correctness targets, or any evaluation outcome. GT may be loaded only
by a later feasibility-analysis layer, strictly after T0 and T320
predictions are finalized and immutable. None of the seven closure files
reference ground truth or annotations (confirmed by two independent prior
audits, modulo one `from __future__ import annotations` false-positive
substring hit).

## Future isolated-port boundary

A later, separate task may add new isolated source files that reproduce
this exact frozen contract — reusing E3 unary inference read-only,
generating T0/T320 targets without GT, exposing pure testable
graph/recurrence/geometry/finalization functions, and including
differential tests against this authority's independently reconstructed
source behavior. That task is authorized only **after** this authority is
independently verified and committed. It must not yet include a real-dataset
CLI, a GPU launcher, intervention generation, GT loading, 96-image
execution, training, confirmation, or full evaluation.

## Continuing prohibition on the 96-image run

Nothing in this authority changes `[gates.outcome].feasibility_execution_authorized
= false` in the committed feasibility contract. The 96-image study remains
unauthorized until: (1) a future implementation task reproduces this exact
T320 contract, (2) that implementation is independently verified, and
(3) a separate, explicit authorization task is run. This authority does not,
by its own existence, authorize execution of any kind.

## What this document does not claim

This authority makes no causal claim, no world-model claim, and no accuracy
claim. It does not claim to replicate the 350-image result, and it does not
claim execution readiness. It is a provenance and scientific-semantics
freeze, nothing more.
