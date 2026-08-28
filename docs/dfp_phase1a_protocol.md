# DFP Phase 1A protocol pre-registration

## Purpose

DFP ("Directed Finite-step Propagation") Phase 1A is a **diagnostic
pre-registration**, not a method implementation. It freezes what will be
measured about the population of pixels where the stitched T=0 (unary,
E3) prediction disagrees with the stitched T=320 (fully propagated)
prediction, on two already-frozen, already-verified evaluation
protocols. **Phase 1A implements no evaluator, generates no predictions,
creates no 500-image subsets, selects no Gate-3 inference method, and
does not itself validate DFP as a deployable method.**

Base commit: `00392527937253f93b9f3fe36b76347b4a6f1eb0`
("fix: resolve ADE20K --data-root through the shared root-resolution
logic"). Working branch: `dfp-phase1a-diagnostics`.

## Authoritative parents

| Authority | Identity | What it freezes |
|---|---|---|
| Graph/propagation/COCO-Stuff dataset | `e12-matched-k11-k12-t320` | directed top-12 ReLU(cosine)³ graph, α=0.98, T=320 finite-step recurrence, crop/stride/geometry, COCO-Stuff val2017 (5000 images, 171 classes) |
| Bridge/model | `e3-paired-soft-routing` | frozen `DINOText` bridge, `vitb_mlp_infonce_paired_soft_routing_tau010` projection |
| RWR/checkpoint | `e3-canonical-directed-rwr` | bridge checkpoint SHA-256, historical CGLS reproduction anchor (reference only — DFP itself never uses CGLS) |
| ADE20K dataset | `e12-ade20k-dataset-source` | ADE20K validation (2000 images, 150 classes, `reduce_zero_label=True`) |
| ADE20K evaluator | `e12-ade20k-matched-evaluator` | ADE20K-specific bridge checkpoint binding, no-background C150 finalization |

No separate COCO-Stuff dataset-source identity file exists in this
repository — COCO-Stuff's validation facts (5000 images, 171 classes,
no background class) live inside `e12-matched-k11-k12-t320`'s own
`[dataset]` section, and this identity binds to that section directly
rather than inventing a nonexistent file.

## Frozen recurrence and horizons

Unchanged from `e12-matched-k11-k12-t320`: directed top-12 row-stochastic
graph, ReLU(cosine)³ affinity, α=0.98, finite power iteration for exactly
320 completed updates, no early stopping, no CGLS/GMRES/dense solve/
fallback. DFP's canonical propagated field **P** uses the **k12**
variant — the unmodified top-12 graph (k11 is itself derived from k12 as
a matched prefix, so k12 is the prior, unablated choice; k11 remains
available only as the existing connectivity-sensitivity comparator, not
DFP's primary field).

Horizons captured: **T = [0, 16, 32, 64, 320]**. T=16/32/64 are
snapshots taken from **one continuous** 320-step recurrence via the
existing `finite_step_propagate` snapshot mechanism — never separate
propagations restarted from S0. Every horizon shares the identical
canonical interpolation and stitching.

## Post-stitching disputes

After final canonical interpolation and stitching, for every non-ignore
evaluation pixel `x`:

```
a(x) = lowest-index argmax of the stitched T=0  prediction
b(x) = lowest-index argmax of the stitched T=320 prediction
D = { x : a(x) != b(x) }
```

Exact argmax ties select the lowest class index — never random —
identically in NumPy and PyTorch.

## Four transition categories

| Category | Condition |
|---|---|
| harmful washout | `GT(x) == a(x)` and `GT(x) != b(x)` |
| beneficial correction | `GT(x) == b(x)` and `GT(x) != a(x)` |
| third-label transition | `GT(x)` not in `{a(x), b(x)}` |
| ignore | `GT(x)` equals the dataset's authoritative ignore value (255) |

The primary harmful-vs-beneficial AUROC/AUPRC population is restricted
to `GT(x) in {a(x), b(x)}`. Third-label and ignore pixels are never
silently folded into the negative class.

## Oracle definitions

- **P** = stitched T=320 label map. **U** = stitched T=0 label map.
- **O_P**: begin with P; wherever U is correct and P is wrong, replace P
  with U; retain P everywhere else (including third-label pixels).
- `G_recover = metric(O_P) - metric(P)`
- `G_oracle|unary = metric(O_P) - metric(U)`
- Relational identity: `G_oracle|unary = G_recover + (metric(P) - metric(U))`
- **Third-label diagnostic map** (GT-dependent, never deployable): begin
  with P; only where GT is neither a nor b, replace b with a; retain P
  elsewhere.
- **Horizon persistence** (T ∈ {16, 32, 64}): for a T=320 dispute a→b,
  retain b only if the stitched T=t winner is also b, otherwise use
  unary a; outside T=320 disputes, retain the T=320 prediction
  unchanged. **T=32 is primary; T=16/T=64 are preregistered sensitivity
  analyses.**

## Metrics and gates

Dataset-level sum-then-divide reconstruction from full-precision integer
sufficient statistics (intersection/union/predicted/ground-truth pixels)
for aAcc, mAcc, mIoU, and Boundary IoU. Metrics in `percent_0_100`,
differences in `percentage_points`. Mean-of-per-image mIoU, summed
per-pixel mIoU contributions, union-weighted proxy gains, and rounded
values labeled full precision are all prohibited.

**Boundary IoU**: no implementation exists anywhere in this repository
or its installed mmseg/mmcv dependencies (independently searched, zero
hits). Phase 1A freezes only the mathematical convention (Cheng et al.
2021: IoU restricted to a normalized boundary band around each region's
contour, with 1/2/4-pixel sensitivity diagnostics) — binding a concrete
implementation is an explicit **Step 2** decision, never invented or
locally reimplemented here.

**Gates**, evaluated per dataset `d`:

```
G_oracle|unary(d) >= 2.0 percentage points
G_recover(d)      >= 1.0 percentage point
```

- Neither dataset passes both → stop DFP as an inference-method project.
- Exactly one dataset passes both → continue only with a regime-scoped
  claim restricted to that dataset.
- Both datasets pass both → a cross-dataset method claim remains
  possible, but still requires independent per-dataset evaluation (a
  pass on one dataset never substitutes for the other).

**Soft (non-blocking) expectations**, distinguished from hard gates:
dispute fraction 5–25% expected (< 2% → addressable set too small, stop;
> 40% → reframe beyond rare flips); `GT ∈ {a,b}` coverage expected
≥ 40%; T=32 persistence expected to recover < 60% of oracle `G_recover`
(if ≥ 60%, a future provenance-based Phase 1B must beat it decisively to
justify its added complexity); if > 70% of harmful-washout mass lies in
the primary boundary band, Boundary IoU becomes the primary improvement
metric and mIoU becomes secondary/preservation.

## Structural component preregistration (Phase 1A only — no propagation)

Components are extracted from the canonical stitched **unary** (T=0)
field, globally over the full stitched image (never per sliding
window), 8-connected primary / 4-connected sensitivity, ties broken by
lowest class index. Soft source membership:

```
M[r,w,j] = (sum over x in component r of A[w,x,j]) / (sum over all x of A[w,x,j])
```

using the existing canonical interpolation/stitching operator `A` —
never a dense pixel-by-patch matrix. `A` must satisfy the adjoint
identity; memberships are non-negative and sum to one per window-patch
source; flush-edge windows are included; `Aᵀ1` is cached as a
geometry-only invariant.

Physical component area: `area_patch_equivalent = component_pixel_area /
projected_patch_pitch_area`, with the projected pitch taken from exact
production geometry. Primary microcomponent threshold: **< 4 patch
equivalents**; sensitivity thresholds: **[1, 2, 4, 8]**. Source-copy
mass is a diagnostic quantity only — never physical component area.
**Phase 1A never propagates these component masks.**

## Query and compute-projection contract

```
Q_w = { a(x), b(x) : x is a stitched T=320 dispute and window w contributes to x }
```

Phase 1A reports mean/median/p90/p95/p99/max `|Q_w|`, components with
nonzero soft membership per window, projected width `R_w * |Q_w|`, total
and worst-window projected width, and projected peak memory for a future
Phase 1B. **No adjoint/masked-forward router is defined in this task.**

## Numerical validation policy

Canonical and decomposed operations must share device and precision;
conservation summaries accumulate in float64; validation occurs at
seed/window/stitched-field/margin levels with both absolute and relative
residuals. The tolerance is **not invented here** — it must be
calibrated from a separate, frozen 200-window numerical-calibration
artifact, versioned before full diagnostic execution.

## Phase 1A / 1S / 1B separation

- **Phase 1A** (this step): freezes what will be measured — dispute
  categories, oracle definitions, metrics, gates, structural-component
  preregistration, query/compute projection. No predictions, no
  subsets, no execution.
- **Phase 1S** (Step 2, not this task): 500-image deterministic subset
  manifests for `coco_stuff_500` and `ade20k_500`, the 200-window
  numerical-calibration artifact, and the Boundary IoU implementation
  binding.
- **Phase 1B** (later, not this task): the adjoint/masked-forward
  router and provenance computation.

## Exact exclusions (this task)

No training/fine-tuning, no optimizer/gradient use, no target feature
extraction as a separate stage, no label-derived prediction
construction, no masked component propagation, no semantic-supercohort
propagation, no Phase 1B provenance computation, no Gate-3 selection, no
learned gates, no morphology, no tuned confidence thresholds, no
model/checkpoint creation, no changes to graph or recurrence
mathematics, no CGLS/GMRES/dense-solve/fallback, no PAMR, no creating
the 500-image subsets, no implementing Phase 1A execution, no masked
propagation, no inference, no CUDA initialization, no dataset access or
modification.

Ground truth is used **only** for evaluation, stratification/class-
presence metadata, oracle diagnostics, and this preregistered analysis
— never for prediction construction.

## Go/no-go outcomes

Summarized in `[gate_contract]`: the joint `G_oracle|unary ≥ 2.0pp` and
`G_recover ≥ 1.0pp` gate, evaluated independently per dataset, is the
sole basis for continuing DFP as an inference-method project, narrowing
to a regime-scoped claim, or stopping.

**This document does not claim that Phase 1A implements or validates
DFP as a method.** It defines what a later execution step will measure.
