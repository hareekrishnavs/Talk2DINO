# Reusable stitching control suite

## Scientific purpose

Determines whether crop artifacts and apparent graph "failures" observed in
the matched k11/k12 evaluation can be explained or mitigated by standard
output-stitching policy, **holding the shared k12 finite-step propagation
configuration fixed** (directed top-12 graph, ReLU(cosine)^3 affinity,
alpha=0.98, exactly 320 finite propagation iterations, the same crop
geometry and row-major crop order) and varying **only** the crop-to-image
aggregation rule.

**This tests stitching protocol, not graph repair.** It does not use
cross-view semantic consensus, does not modify the graph, does not select
T4 targets, and does not touch DCR/SUR/pruning/counterfactual solving. True
DINO feature stitching is outside this suite — every variant here operates
on the already-computed E3/RWR score or probability crop, never on raw DINO
features.

## The four frozen variants

All four consume the exact same immutable per-window k12 output (one
shared backbone/snapshot pass, one shared graph build, one shared T=320
propagation per window — see
[`stitching_control.process_one_window_shared`](../src/open_vocabulary_segmentation/models/dinotext/cover_dr/stitching_control.py)).
Only the aggregation differs:

| Variant | Stage | Weighting | Sigmoid |
|---|---|---|---|
| `uniform_probability` | probability | uniform | before interpolation |
| `hann_probability` | probability | Hann | before interpolation |
| `uniform_score` | score | uniform | after stitching (once) |
| `hann_score` | score | Hann | after stitching (once) |

**`uniform_probability` reproduces the existing matched k12 evaluator's
stitching bitwise-exactly** — this is the load-bearing identity control
(`tests/test_stitching_control_api.py::test_uniform_probability_reproduces_legacy_k12_stitching_exactly`),
verified against `matched_power_evaluator.stitch_one_image`'s k12 path
directly, never against a re-derivation.

**Probability-space vs. score-space**: probability-space variants sigmoid
per window then interpolate then aggregate (matching the existing
evaluator's `masks_from_patch_scores` contract exactly). Score-space
variants interpolate the *raw* patch scores first, aggregate in
score-space, then apply sigmoid exactly once to the stitched result. These
are not interchangeable in general — `average(sigmoid(x)) !=
sigmoid(average(x))` for nonlinear sigmoid — which is precisely the
question this suite investigates.

## The Hann formula

Pixel-centred, parameter-free, separable:

```
h_N(x) = 0.5 - 0.5*cos(2*pi*(x+0.5)/N)   for x = 0, ..., N-1
W(y,x) = h_H(y) * h_W(x)
```

The `+0.5` pixel-centre offset keeps the argument of `cos` strictly inside
`(0, 2*pi)` for every finite `N`, so `cos` never reaches exactly `1` and `h`
never reaches exactly `0` — **strictly positive everywhere, including at
crop boundaries, with no epsilon or floor hyperparameter**. At `N=1` the
formula itself evaluates to exactly `1.0` (`cos(pi) = -1`), a natural
consequence rather than a special-cased branch.

## Shared-execution design

One call each of backbone/snapshot, graph build, and finite-step
propagation **per window**, never per variant — see
`operation_telemetry.backbone_snapshot_calls == windows_processed_total`
(never `windows * 4`) in every produced result. At most **two**
interpolation calls per window: one shared sigmoid+interpolated
probability crop (reused by `uniform_probability`/`hann_probability`), one
shared interpolated raw-score crop (reused by
`uniform_score`/`hann_score`).

```
image/model forward
  -> immutable per-window k12 output (snapshot, graph, propagation)
  -> {shared probability crop, shared score crop}
  -> four independent StitchAccumulator instances (own numerator +
     denominator, FP32, no storage aliasing)
  -> finalize all four -> per-image/per-class sufficient statistics
```

## Metric units and artifacts

- Metrics (`aAcc`, `mIoU`, `mAcc`): `percent_0_100`.
- Paired deltas (`delta_mIoU_percentage_points_vs_uniform_probability`):
  `percentage_points`, reported for `hann_probability`, `uniform_score`,
  `hann_score` against `uniform_probability` as the reference.
- Raw per-image/per-class sufficient statistics in the NPZ
  (`intersect_<variant>`, `union_<variant>`, `pred_<variant>`, one shared
  `label`): `count`.
- The evaluator never bootstraps — sufficient statistics are preserved for
  a later paired-bootstrap analysis stage (reusing
  `src.k11_k12_full_result_analysis.bootstrap_paired_delta`'s bounded-memory
  algorithm against these NPZ arrays, out of scope for this stage's own
  CLI).

## Checkpoint semantics

Reuses the matched evaluator's hardened checkpoint pattern
(`src.stitching_control_checkpoint`, structurally parallel to
`src.k11_k12_power_evaluation_checkpoint`, never a weaker reimplementation):
exact canonical image prefix, no duplicate/skipped/reordered images,
class-count agreement, atomic writes, strict JSON. Additionally
**variant-set identity** is checked on every resume — a checkpoint recorded
under a different variant set, identity, or run mode is rejected, never
silently reinterpreted.

## Pilot ladder

`pilot20` -> `pilot100` -> `full5000`. **The full5000 run is not authorized
until both pilot20 and pilot100 pass** —
`scripts/slurm/e12_stitching_control_full5000_h100.sbatch` enforces this
mechanically: it requires `PILOT20_RESULT`/`PILOT100_RESULT` environment
variables pointing at already-produced result files and independently
re-verifies both (`verify_stitching_control_suite.py verify-result`)
before any full5000 GPU work begins.

## Explicit exclusions

Never implements: T4 target selection, consensus labels, DCR, SUR,
cross-view directed-edge support, graph pruning, Sherman–Morrison
counterfactuals, adjoint gradients, structural reachability, a changed
`k`, changed RWR propagation, changed crop/stride, PAMR, or any learned
parameter. `tests/test_stitching_control_api.py`'s scientific-isolation
tests statically confirm no such identifier appears in
`stitching_control.py`.

## Limitations

- The shared-execution CLI is verified only via CPU/synthetic fixtures in
  this stage (per HPC login-node policy); it has not been run on the real
  GPU/dataset as part of this implementation.
- `dataset.pre_eval` is assumed available on the dataset object (mmseg's
  own contract); a dataset without it fails closed rather than falling
  back to a hand-rolled statistic computation.
- Paired-bootstrap analysis of the four variants' sufficient statistics is
  a separate, later stage — this evaluator only produces the artifacts.
