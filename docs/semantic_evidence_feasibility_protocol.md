# Semantic-Evidence Feasibility Protocol (v1)

This document explains, in prose, the contract frozen in
[`research_specs/semantic_evidence_feasibility_v1.toml`](../research_specs/semantic_evidence_feasibility_v1.toml)
and materialized in
[`evaluation_subsets/semantic_evidence_feasibility_96.json`](../evaluation_subsets/semantic_evidence_feasibility_96.json).
Both files are loaded and strictly validated by the Torch-free identity
loader at
[`src/semantic_evidence_feasibility_identity.py`](../src/semantic_evidence_feasibility_identity.py),
which is itself covered by
[`tests/test_semantic_evidence_feasibility_identity.py`](../tests/test_semantic_evidence_feasibility_identity.py).

This document is prose explanation. Where it and the TOML disagree, the
TOML is authoritative — this file does not carry any frozen numeric or
formula value that isn't also in the TOML, so any disagreement is a bug in
this document, to be fixed to match the TOML, not the other way around.

**Revision note.** This revision (schema version 1.1.0) repairs five
findings from an independent adversarial verification of the first version
of this contract: (1) the T0/T320 propagation-target source authority was
undefined — T0 is now bound, T320 is now explicitly **DEFERRED** and
non-computable (§9); (2) the statistical unit of analysis was ambiguous
between anchor-level and anchor-query-level rows — the contract now freezes
two distinct tasks and exactly four logistic-regression fits (§11); (3) the
frozen score formulas could silently launder a `NaN` input into a finite,
Gate-A-passing output — an explicit fail-closed input/output precondition
is now frozen (§8); (4) the Gaussian-blur padding/border behavior was
unspecified — it is now frozen as `reflect` padding, independently verified
against the installed `torchvision==0.27.0+computecanada` source (§7); (5)
the external projection-checkpoint path field name risked being read as a
git-tracked-repository claim — it is now named
`external_untracked_checkout_relative_path` with explicit `git_tracked =
false` evidence (§4). **The 96-image feasibility study remains
unauthorized after this repair** — see §12 and §13.

## 1. Why this study precedes any final method

The repository already contains, on other branches, substantial
implementation work toward propagation- and intervention-based
segmentation refinement (graph construction, RWR/CGLS solvers, BALP,
learned retrieval prototypes — see the `e6`–`e12` and
`dfp-phase1a-diagnostics`/`dfp-phase1b-*` branches and their leftover
`__pycache__` residue, documented in the Step 0 audit). None of that
machinery has been shown to answer a much more basic question first:
**does a controlled image intervention even carry information about
region-level evidence reliability that isn't already available more
cheaply from confidence, similarity, and geometry?** If the answer is no,
building a full propagation/intervention pipeline on top of E3 would be
building on a signal that doesn't exist. This contract exists to answer
that narrower question on a small, fixed, statistically-bounded population
*before* anyone writes a runner, touches a GPU, or trains anything.

## 2. The clean E3 baseline this contract is anchored to

Everything here is bound to one exact, previously-audited state:

- Branch `research/semantic-evidence-interventions`, commit
  `9f314dc973657719abb15cc33473dafc185444c4` ("paired soft routing") —
  verified clean (no staged/untracked changes, `git diff --check` passing)
  at both the Step 0 audit and this contract-freezing task.
- The E3 evaluation path itself (`DINOText` → `DINOTextMasker.forward_seg`
  → `DINOTextSegInference`) was read end-to-end in Step 0: it is a frozen
  DINOv2 ViT-B/14-reg backbone, a frozen CLIP ViT-B/16 text tower, one
  trained linear→tanh→linear text-only projection, and a plain
  cosine-similarity + sigmoid mask. No graph propagation, RWR/CGLS, BALP,
  PAMR, or intervention logic is reachable from it — confirmed by both a
  `git grep` across all tracked source at HEAD and by direct call-graph
  reading (Step 0 `baseline_report.md` §2–3), then independently
  re-confirmed by a real two-image GPU smoke test in the Step 0 closure
  (job `22357843`, `PASS` on every assertion including bit-identical
  repeated inference).
- Every source file this contract depends on is bound by repository-relative
  path **and** SHA-256 in `[[dependency]]` — see §4 below. The identity
  loader recomputes every one of these hashes against the *current* files
  before anything else in this contract is trusted.

## 3. `TALK2DINO_WEIGHT_DIR` and why an absolute path is not scientific identity

The E3 config resolves the DINOv2 backbone and CLIP text-encoder
checkpoints through the environment variable `TALK2DINO_WEIGHT_DIR`
(`src/local_weights.py:11`); the E3 projection checkpoint itself is
resolved independently, via a hardcoded `weights/` join
(`dinotext.py:109`) that ignores this variable entirely. On the current
cluster, `TALK2DINO_WEIGHT_DIR=/scratch/haree/weights` — but `/scratch/haree`
is this user's personal scratch mount on this specific Alliance cluster. It
will not exist, or will point at something else entirely, on a different
node allocation, a different user's account, or a different HPC site.

If this contract bound its scientific identity to that path, the contract
would silently become unverifiable (or, worse, silently verify against the
*wrong* file) the moment the operational environment changed. Instead, the
contract binds each external checkpoint by **filename and SHA-256**
(`[external_weights]`), and separately records that the *mechanism* by
which those bytes get found is `TALK2DINO_WEIGHT_DIR`. A future runner
task is responsible for setting that variable correctly for wherever it
runs; this contract's identity loader does not need to know or care what
its value is, only that the bytes it eventually resolves to match the
frozen hash.

## 4. Dependency bindings

`[[dependency]]` in the TOML lists ten repository-relative files — the E3
primary and inherited configs, the COCO-Stuff dataset subconfig,
`local_weights.py`, the `DINOText`/masker/evaluator/builder/projection
implementations — each with its current SHA-256. `[external_weights]`
separately lists the three checkpoint files (E3 projection, DINOv2
backbone, CLIP text encoder) by filename and SHA-256, with the E3
projection hash matching the historical value independently verified in
Step 0 (`9632117c18613674eeddcb35f86193d37da203a9cbd837cf500c4121b5fa9942`,
**unchanged and unmoved** by this revision). None of these external
checkpoint bytes, and none of the Step 0 closure's scratch-filesystem
artifacts, are copied into the repository — only their hashes are.

The E3 projection checkpoint physically sits under this checkout's
`weights/` directory, but that directory is listed in `.gitignore` and the
file is **not** git-tracked (`git ls-files
weights/vitb_mlp_infonce_paired_soft_routing_tau010.pth` returns nothing).
Its path field is named `external_untracked_checkout_relative_path` —
deliberately not `path` or anything resembling `[[dependency]]`'s
git-verified field names — precisely so it cannot be mistaken for evidence
that git commits these bytes. Its scientific identity, like the other two
checkpoints, is its filename plus SHA-256 alone.

## 5. The 96-image population and its 48/48 split

Selection is a pure function of the 5,000 canonical COCO-Stuff val2017
image IDs (12-digit, zero-padded, extension-stripped) and one fixed string,
the namespace `talk2dino-semantic-evidence-feasibility-v1`. For every
candidate ID we compute

```
selection_key = SHA256( UTF8(namespace) || 0x00 || ASCII(canonical_image_id) )
```

sort all 5,000 `(selection_key, canonical_image_id)` pairs ascending, and
take the first 96. Ranks 0–47 become `discovery` (used to *fit* the
statistical models); ranks 48–95 become `locked_holdout` (used to
*evaluate* them, exactly once). Because the sort key is a keyed hash of the
ID alone, this selection cannot be influenced by GT, image content, prior
model output, or anything else — reproducing it requires only the
namespace string and the 5,000-ID universe, both frozen. The identity
loader's `reconstruct_selection()` recomputes the entire thing from
scratch and the test suite cross-checks it with a second, independently
written implementation, not by calling the production function on itself.

Four digests (`canonical_source_digest`, `selected_record_digest`,
`discovery_digest`, `locked_holdout_digest`) are frozen in both files so
that any future re-derivation — or any tampering — is immediately
detectable without needing to inspect all 96 records by eye.

## 6. Anchor strategy

Within each E3 crop (448×448 pixels, DINOv2 patch size 14 → a 32×32 patch
grid, both facts pulled from source rather than assumed — see
`[e3_geometry]`), the outer one-patch ring is excluded from anchor
eligibility, leaving a 30×30 = 900-patch eligible interior. For every
eligible patch we compute its **unary** classification — the raw,
pre-sigmoid cosine similarity to all 171 class text embeddings, evaluated
in a single window in isolation, *not* the sliding-window-accumulated
whole-image prediction used for mIoU — and from that, its top-1/top-2
class and margin. A patch is `boundary` if any of its eight grid neighbors
(always exactly eight, since eligible patches never touch the grid edge —
see `e3_geometry.eligible_patches_always_have_8_grid_neighbors`) disagrees
with it on unary top-1; otherwise it's `interior`. Boundary and interior
populations are each split at their own median margin into low/high
(ties go low), giving four strata. Two anchors per stratum are chosen by
ascending `SHA256(anchor_namespace || id || window || patch)` order — a
second, independent hash-based lottery, deliberately different from the
image-selection namespace, so that patch-level and image-level selection
can't accidentally correlate. Eight anchors per window, never GT-informed.

## 7. The four interventions

Every anchor gets a 3×3-patch (42×42 pixel) core. Four views of the same
crop are defined:

1. **ORIGINAL** — untouched.
2. **CONTEXT_BLURRED** — the core is bit-exact; everything else is
   Gaussian-blurred.
3. **REGION_BLURRED** — everything else is bit-exact; the core is
   Gaussian-blurred.
4. **AREA_MATCHED_CONTROL** — the anchor's core is untouched; a
   *different*, non-overlapping 3×3-patch region (chosen by the same
   hash-order principle, applied to the control candidate) is blurred
   instead. If no valid non-overlapping control region exists for a given
   anchor, that anchor fails closed and is excluded — it is not silently
   dropped without a trace.

The blur itself is fully specified and deterministic:
`torchvision.transforms.functional.gaussian_blur`, kernel `[31,31]`, sigma
`[8.0,8.0]`, **padding mode `reflect`** — no randomness, no color jitter, no
resizing, no inpainting, no generative model, no feathering, no GT. The
padding mode is frozen explicitly, independent of any library default,
because a 31-pixel kernel needs padding within roughly 15 pixels of the
448×448 crop's outer edge, and different padding modes there would produce
different, non-reproducible blurred pixels at the crop boundary (never
inside the patch-aligned anchor/control cores, which per `[e3_geometry]`
are always interior). `reflect` was chosen to match, not because it is
convenient, but because it is independently confirmed as what the actual
target environment already does: a static read (not an import) of
`torchvision==0.27.0+computecanada`'s installed source at
`_functional_tensor.py:746-764` shows its tensor-path implementation
calling `torch_pad(img, padding, mode="reflect")` before convolution — and
0.27.0 is the exact TorchVision version recorded in the Step 0 GPU
smoke-test log, confirming it's the real target environment, not an
unrelated one. This contract *defines* these four views; it does not
generate them. `intervention_runner_implementation_authorized` is
explicitly `false`.

## 8. The reliability formula

For each anchor and each of its two queries (its own unary top-1 and
top-2 class — never a GT class), four raw cosine scores are read off at
the anchor's center patch across the four views, and combined into a
single number in `[0, 1]`:

```
context_change   = |s_context - s_original|
region_drop       = max(0, s_original - s_region)
control_change    = |s_control - s_original|

context_stability = 1 - min(1, context_change / 2)
specific_region_drop = max(0, region_drop - control_change)
region_necessity   = min(1, specific_region_drop / 2)

intervention_reliability = context_stability * region_necessity
```

Intuition: a region's evidence for a text concept is judged *reliable* if
removing everything **except** that region barely moves the score
(`context_stability` high — the region alone still supports the
prediction) **and** removing **only** that region drops the score by more
than an equally-sized, equally-blurred control region would
(`region_necessity` high — the drop is specific to this region, not just
an artifact of blurring something, anything). No threshold, temperature,
learned coefficient, or per-class tuning is introduced anywhere in this
formula — it is pure arithmetic on raw cosine scores bounded by the known
`[-1, 1]` cosine range.

**Fail-closed input/output validity (`[scores.validity]`).** Before any of
the arithmetic above runs, all four raw scores
(`s_original(q)`, `s_context(q)`, `s_region(q)`, `s_control(q)`) must each
be checked finite and within `[-1.0, 1.0]`. This precondition exists
because of a genuine, verified hazard: Python's builtin `min(1, x)` and
`max(0, x)` — exactly the argument order used in `context_stability` and
`region_necessity` above — do **not** propagate `NaN` (`min(1, float('nan'))
== 1`, `max(0, float('nan')) == 0`, confirmed against CPython 3.11). A
corrupted `s_original(q)` of `NaN` would otherwise reach a perfectly finite,
in-range, Gate-A-passing `intervention_reliability` of exactly `0.0` — a
"legitimate-looking" zero-reliability anchor that Gate A's output-only
`max_non_finite_scores = 0` check would never catch. The input check must
therefore run *before*, not instead of, the formulas; a record whose inputs
fail is marked `STRUCTURALLY_INVALID` and counted against Gate A directly —
it is never clamped, imputed, or silently laundered into a finite value.
The *output* (`intervention_reliability(q)`) is separately checked finite
and in `[0.0, 1.0]` afterward, as a second, independent safety net against
arithmetic-level failures — not a substitute for the input check.

## 9. GT isolation, and the T0/T320 split

GT is loaded **last**, only after every one of the above — selection,
all four views, all unary scores, `intervention_reliability`, and every
non-GT control feature — is already finalized for a given image. GT is
used only to compute two *evaluation targets*, never to influence
selection or scores:

- **Primary correctness**: was the anchor's query class actually right,
  per majority GT vote in its footprint (≥75% of footprint pixels valid,
  ≥80% majority share, else the anchor is `AMBIGUOUS_GT` and excluded from
  this metric — never fabricated).
- **Propagation outcome**: a T0→T320 five-way classification (see below).

**T0 is bound; T320 is deliberately not.** `[propagation_target_authority]`
freezes this asymmetry explicitly. **T0** is simply the anchor's own
original-view unary top-1 prediction (`q1`) — it needs no new machinery,
it's already fully defined by `[e3_geometry].unary_top1_top2_definition`
and bound to the same committed, pre-sigmoid cosine score and argmax
tie-breaking used everywhere else in this contract. **T320** — "the
prediction after 320 steps of some future finite-step propagation
process" — is a *concept only*. This contract does not specify, implement,
import, or authorize any graph-construction, recurrence, stitching, or
target-extraction process that would actually produce a T320 prediction.

This isn't a hypothetical caveat. A read-only, no-checkout inspection of
`dfp-phase1a-diagnostics`, `e11-cover-dr-1`, and `e12-connectivity-analysis`
(`git ls-tree`/`git grep`) shows a real `cover_dr` graph/RWR/finite-step
module family (`graph.py`, `rwr.py`, `finite_step_regime.py`,
`stitching_control.py`, ...) that is almost certainly *some* prior version
of what "T320" was meant to mean. On **this** branch, at this HEAD, that
source is completely absent — the only trace is orphaned, gitignored
`__pycache__/*.pyc` bytecode sitting at
`src/open_vocabulary_segmentation/models/dinotext/cover_dr/__pycache__/`,
left over from a different branch once being checked out locally in this
same working directory. `[propagation_target_authority]` records this
disclosure explicitly and states, in a machine-checked field, that this
bytecode residue is **not** treated as authority
(`ignored_bytecode_is_authority = false`), that no cross-branch source was
copied or imported by this revision, and that a successor task must freeze
an exact graph/recurrence/stitching/extraction implementation — bound by
path, commit, and SHA-256 — before any propagation-outcome row can exist.

**The five outcome categories** (not four — this revision adds one). Given
T0's and T320's predicted classes and GT:

- `beneficial`: T0 wrong, T320 correct.
- `harmful`: T0 correct, T320 wrong.
- `unchanged_correct`: T0 and T320 both correct (which necessarily means
  they predicted the *same* class — both equal the GT majority class).
- `changed_but_same_correctness`: T0 and T320 both **wrong**, but predicted
  **different** classes. This category did not exist in the first version
  of this contract, where it was silently indistinguishable from
  `unchanged_incorrect` — a real ambiguity an adversarial review caught.
- `unchanged_incorrect`: T0 and T320 both wrong, predicting the **same**
  class.

`changed_but_same_correctness` is excluded from the beneficial-vs-harmful
binary fit (it's neither) but is still counted and reported — never
silently dropped.

None of this makes the propagation-outcome target computable today. It
only ensures that when T320 *is* eventually bound, there is no ambiguity
left for whoever implements it to resolve on their own.

## 10. Controls, and why one is `NOT_AVAILABLE_IN_E3`

Seven non-GT, non-intervention control features are frozen: the anchor's
own top-1 score, its top1–top2 margin, its 171-way entropy, its DINO
cosine similarity to its neighbor-mean, its normalized distance to the
nearest boundary patch, its local neighbor-agreement fraction, and the
area-matched control's own score change. An eighth slot — "does an
*ordinary* deterministic augmentation used by E3's real evaluation
pipeline itself move the score" — was checked against source
(`stuff.py`'s `test_pipeline`) and found to contain only a `RandomFlip`
constructed with `flip=False`, i.e. an unconditional no-op. Rather than
invent a new augmentation to populate this slot, it is frozen as
`NOT_AVAILABLE_IN_E3` and simply excluded from both statistical models.

## 11. Two tasks, two row populations, four model fits

The first version of this contract had a real ambiguity here, caught by
adversarial review: `query_correctness` naturally produces **two rows per
accepted anchor** (one for `q1`, one for `q2`), while `propagation_outcome`
naturally produces **one row per anchor** (there's only ever one T0, so
only `q1` — T0's own predicted class — is meaningful). A single shared
`CONTROL_MODEL`/`CONTROL_PLUS_INTERVENTION_MODEL` pair fit across both
targets doesn't make sense once you notice this: `sklearn`'s
`LogisticRegression` fits one binary target at a time, and the two targets
don't even live on the same row population. This revision resolves it by
freezing **two explicit tasks**, each with its own row population and its
own feature parameterization, and **exactly four model fits**:

| task | unit of analysis | rows | model fit ids |
|---|---|---|---|
| `query_correctness` | `anchor_query` (2 rows/accepted anchor) | `q1` and `q2` | 1 (control-only), 2 (control+intervention) |
| `propagation_outcome` | `anchor` (1 row/eligible anchor) | `q1` only | 3 (control-only), 4 (control+intervention) |

Controls 1 (`top1_unary_score`) and 8 (`area_matched_control_change`) were
previously mis-specified: control 1 was hardcoded to `s_original(query_1)`
regardless of which row it fed, and control 8 was an anchor-level *average*
over both queries. Both are now parameterized by "the row's own query `q`"
— `q1` or `q2` for a `query_correctness` row, always `q1` for a
`propagation_outcome` row. Every other control (margin, entropy, DINO
neighbor similarity, boundary distance, local agreement) was already
anchor-level and query-independent, so it's unchanged.

Each of the 4 fits uses the identical, fully frozen logistic-regression
recipe (`C=1.0`, `l2`, `class_weight=balanced`, `solver=liblinear`,
`max_iter=1000`, `random_state=0`, **no** hyperparameter search),
standardized on its own task's `discovery` rows only, fit once on
`discovery`, and evaluated exactly once on `locked_holdout`. The 4 primary
metrics are each pinned to one task, one row population, one target, and
one pair of producing model-fit ids — `query_correctness_auroc`/`_ap` come
from fits 1 vs 2; `beneficial_dominant_vs_harmful_dominant_auroc`/`_ap`
come from fits 3 vs 4 — so no metric can silently draw from an
unspecified, generic "record" population.

The bootstrap is **image-level**, not row-level: 10,000 seeded, paired
resamples of `locked_holdout` *images* (with replacement), where every
eligible row belonging to a sampled image — for *both* tasks — is carried
along with it, preserving multiplicity if an image is sampled more than
once. All four models are fit exactly **once**, before any bootstrap
replicate begins; the replicates only resample which already-fitted
predictions get aggregated into each replicate's metric, never refit
anything. A replicate missing either class of a metric's target is
`NOT_A_NUMBER` for that metric and excluded from its percentile CI; if
fewer than 90% of replicates are valid for a metric, that metric's CI is
`INCONCLUSIVE`.

## 12. Gates A–E: what PASS and STOP mean

Five gates, all frozen, all evaluated on `locked_holdout` only:

- **A** — did the intervention machinery itself behave (≥95% structural
  validity, zero GT leakage into scores, all reliability values finite and
  in `[0,1]`)?
- **B** (task `query_correctness`, fits 1 vs 2) — does `intervention_reliability`
  add ≥0.03 AUROC for predicting whether the anchor-query's query is
  *correct*, with the bootstrap CI's lower bound above zero, and AP not
  worse? **Computable now.**
- **C** (task `propagation_outcome`, fits 3 vs 4) — same question, but for
  predicting whether propagation would be *beneficial rather than harmful*
  at that anchor. **Not computable now** — Gate C cannot be computed,
  passed, or silently skipped while `[propagation_target_authority].t320_status
  = DEFERRED_REQUIRES_SUCCESSOR_AUTHORITY`. A missing T320 binding is a
  Gate C **FAIL**, full stop.
- **D** — does the improvement hold up, or at least not reverse badly
  (`< -0.01` AUROC), in every large-enough subgroup (boundary/interior,
  thing/stuff, low/high margin)?
- **E** — is the actual 96-image run's compute (≤2 H100-hours) and the
  *projected* cost of generating supervision at 1,000-image scale (≤8
  H100-hours, with the projection formula and its uncertainty disclosed)
  within a defensible budget — without claiming anything about eventual
  deployment runtime?

If all five pass: `PROCEED_TO_DEPLOYABLE_SIGNAL_SPECIFICATION` — a
*specification* task, still not an implementation. If any fail:
`STOP_SEMANTIC_EVIDENCE_INTERVENTION_TRACK`. `INCONCLUSIVE` (a minimum-data
requirement not met) counts as STOP unless the shortfall is traceable to a
fixable implementation bug, not a scientific or population choice — and
even then, the population itself may never be retroactively enlarged.
Thresholds may not be adjusted after results are observed; that would
defeat the entire point of freezing them now, before anyone has looked at
the holdout set.

Because Gate C is structurally not computable while T320 is deferred, the
overall `proceed_condition` ("Gates A, B, C, D, and E all pass") **cannot
currently be satisfied by construction** — not as a probabilistic outcome
of running the study, but as a fact about this contract today.
`[gates.outcome].feasibility_execution_authorized = false` states this
directly and is checked by the loader independently of, and in addition
to, every flag in `[authorization]`: even if every `[authorization]` flag
were somehow satisfied, the 96-image study is still not authorized to run
until a successor task binds T320.

## 13. What this contract does not authorize

`[authorization]` is the literal, machine-checked source of truth (the
identity loader hard-fails if any required-`true` flag is `false`, or any
required-`false` flag is `true`), but in prose: this task may write a
closed specification, materialize the 96-image list, and write/test a
loader for both. It may not write an intervention runner, touch a GPU,
construct the model, load any checkpoint tensor, run graph propagation,
train anything, build a deployable predictor, search hyperparameters, run
any full-dataset evaluation (original or E3), touch a world model or SAM,
use GT for selection, or overwrite any existing artifact. The Step 0 GPU
smoke-test authorization does **not** carry forward here — a future task
that wants to run the intervention runner needs its own, separate
authorization.

## 14. Original Talk2DINO comparator: still unresolved

Both the Step 0 audit and its closure independently concluded
`ORIGINAL_MATCHED_BASELINE: NOT_AVAILABLE` — the original (non-E3)
Talk2DINO checkpoint (`weights/vitb_mlp_infonce.pth`) exists locally, and
an uncommitted `segmentation_results/vitb_mlp_infonce.json` shows a number
suggestively close to the published 28.1 mIoU, but no git-hash-stamped run
log ties that number to a specific commit or environment, unlike E3's own
28.48 figure (which *is* tied to HEAD via `output/e3-paired-soft-routing-tau010/log.txt`'s
embedded `Git hash: 9f314dc` line). This contract does not attempt to
close that gap — it is explicitly required, and explicitly still open,
before any later publication-level comparison between E3 and the original
published method can be made honestly.

## 15. What this contract explicitly does not claim

Per `[research_question]`: this is a feasibility study asking whether a
signal *exists*, using the vocabulary "controlled intervention",
"intervention-derived reliability", and "evidence reliability" throughout.
It does not claim, and passing Gates A–E would not retroactively justify
claiming, a final segmentation method, a causal effect, an accuracy
improvement, a specific mIoU delta, a deployable adapter, a propagation
replacement, a world-model method, that training occurred, that a full
evaluation occurred, publication novelty, or a runtime improvement. In
particular, no causal language may be used for this signal unless and
until a future successor task supplies and independently verifies the
causal assumptions (no unmeasured confounding, consistency, positivity)
that this contract does not establish and does not attempt to establish.
