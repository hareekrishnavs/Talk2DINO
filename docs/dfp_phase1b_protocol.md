# DFP Phase 1B Scientific Protocol

Frozen scientific pre-registration for DFP Phase 1B, governed by
`evaluation_identities/e15_dfp_phase1b_protocol.toml` and loaded by
`src/dfp_phase1b_identity.py`. This document is a narrow operational
summary; the TOML identity is the sole scientific authority.

**This is protocol authoring only.** No provenance value has been
computed, no masked propagation has run, no diagnostic model has been
fit, and no Gate-2 decision exists. See *Exclusions* below.

## Why Phase 1B, and why now

The DFP Phase 1A Gate-1 boundary-mass reframe trigger **fired** on the
finalized COCO-Stuff-500 result (see `docs/dfp_phase1a_gate1_analysis.md`):
97.65% of harmful-washout pixels fall inside the frozen boundary band,
exceeding the preregistered 70% threshold. Gate-1 also passed the
oracle/recovery headroom gate (`GATE-1 VERDICT = PROCEED_TO_PHASE_1B`,
image count 500, run `coco_stuff_500`). Per the frozen consequence,
Boundary IoU became the required primary Phase-1B metric, and Phase 1B
evaluation was blocked pending its binding — `docs/dfp_phase1b_boundary_iou.md`
performed that binding (`evaluation_identities/e14_dfp_phase1b_boundary_iou.toml`).
This protocol is the next prerequisite: the scientific hypothesis, the
GT-free feature decomposition, and the Gate-2 statistical procedure,
frozen before any provenance computation.

## The exact scientific hypothesis

> Does component-level diffusion evidence provenance provide
> statistically and practically meaningful information for
> distinguishing harmful T0→T320 washouts from beneficial T0→T320
> corrections, beyond unary confidence, propagation margin, text
> similarity and early-horizon persistence?

Phase 1B is diagnostic. It does **not** define the final deployable
selector — no fitted diagnostic model produced under this protocol may
ever be reported as the training-free final DFP method.

## Component-evidence provenance decomposition

For each disputed pixel, the frozen masked-forward decomposition
attributes the propagation margin gain to individual same-class
connected components of the T0 argmax field:

```
Y0^(r)[w,j,c] = M[r,w,j] * Y0[w,j,c]
S0^(r)   = A_final(Y0^(r))
S320^(r) = A_final(K320(P) Y0^(r))
G[x,r] = [S320^(r)(x,b) - S320^(r)(x,a)] - [S0^(r)(x,b) - S0^(r)(x,a)]
sum_r G[x,r] = m320(x) - m0(x) = delta_m(x)   (conservation)
```

Exact masked forward propagation only (no adjoint approximation, no
dense influence kernel, no top-K truncation, no residual
supercomponent), the same k12 graph/recurrence as canonical T320,
fp64 component-sum accumulation, no early stopping, no equilibrium
solver. Per-window class carry-through is restricted to
`Q_w = {a(x), b(x) : x disputed, window w nonzero-weight}` — never the
image-global class set, never raw patch argmax classes.

## Frozen GT-free features

Margins/controls: `unary_resistance = -m0`, `propagated_margin = m320`,
`margin_change = delta_m`, `eta_net = m320 / (m320 - m0)` (denominator
always strictly positive since `m320 > 0, m0 <= 0` — no epsilon).
Horizon controls: T16/T32/T64 win-state indicators (T32 is the primary
cheap baseline). Text control: `text_cosine_ab = cosine(t_a, t_b)`.
Component evidence: `n_positive_components`, `k_star` (minimum k such
that the top-k positive contributions exceed `m320`), `phi` (fraction
of positive evidence explained by the top-`k_star` components, no
epsilon), `top1_positive_share`, `positive_evidence_sum`,
`negative_evidence_sum`. No Gini/entropy/additional features may be
added after seeing results without an explicit successor protocol.

## GT-isolation architecture

Two logically separate, sequentially-sealed artifact stages:

- **Stage A** (GT-free): image identifiers, dispute offsets, pixel
  indices, a/b classes, margins, horizon states, text cosine, component
  evidence features, conservation diagnostics. **Must not** contain any
  GT-derived field. Finalized and content-hashed before Stage B begins.
  The driver/API must not accept GT arrays or dataset annotation
  objects; tests must poison GT access during Stage A.
- **Stage B** (evaluation labels): binds the exact Stage-A digest, may
  add GT class, target category, boundary/interior indicator,
  deterministic fold ID, confusion statistics, diagnostic model
  outputs.

## Deterministic fold assignment

5 image-grouped folds (100 images each), pixels never split across
folds: `key(image_id) = SHA256(namespace + NUL + canonical_image_id)`,
sorted by `(key, canonical_image_id)`, `fold_id = sorted_rank mod 5`.
Namespace frozen exactly (`talk2dino-dfp-phase1b-fold-assignment-v1`).
No random library, filesystem order, or Python hash may affect folds.

## Diagnostic feature evaluation (Gate 2)

Out-of-fold predictions only. Primary binary task: harmful vs
beneficial (third-label pixels excluded from fitting/AUROC/AUPRC, but
their feature/risk distributions are reported separately). A fixed
logistic-regression diagnostic model — L2, `C=1.0`, intercept enabled,
`lbfgs`, 1000 max iterations, tolerance `1e-4`, float64, training-fold-
only standardization, no class balancing, no hyperparameter search —
compares a cheap-control feature set against a provenance-augmented
set. **This model is diagnostic only and may never become the final
DFP selector.** Deterministic training subsampling (SHA-256 lowest-key
retention, 4096 pixels per image per binary category) reduces fitting
cost; held-out evaluation pixels are never subsampled.

## Image-level paired bootstrap

10,000 replicates, resampling unit = image, 500 image IDs with
replacement, paired (identical sampled images for both models),
percentile 95% interval (quantiles 0.025/0.975, NumPy `linear` method),
a SHA-256-derived fixed seed. AUROC/AUPRC computed per replicate only
when both classes are present; fails if fewer than 95% of replicates
are valid.

## Gate-2 criteria

Three outcomes: `PROCEED_TO_GATE_3`, `STOP_DFP`, `INCONCLUSIVE`. Eleven
mandatory PROCEED conditions (conservation/artifact validation;
`-phi`/`-k_star`/`top1_positive_share` single-feature AUROC lower bound
`> 0.5`; provenance AUROC `>= 0.65`; AUROC/AUPRC improvement `>= 0.02`
absolute on the `fraction_0_1` scale (= 2.0 percentage points) each
with a strictly-positive paired bootstrap lower bound; `>= 95%` valid
bootstrap replicates; exact computation within the resource ceiling;
Boundary IoU sufficient statistics for every baseline variant; valid
PAMR statistics present for both `t0_pamr` and `t320_pamr`). STOP if
provenance AUROC `< 0.60`, the improvement upper bound is `<= 0`,
conservation fails after confirmed correctness, or exact computation is
infeasible without an unapproved approximation. INCONCLUSIVE for every
intermediate case, including missing or invalid PAMR statistics.
Thresholds are frozen before any Phase 1B result and may never be
loosened after seeing results.

### AUROC/AUPRC units

`auroc`/`auprc` are always stored and reported on the raw `fraction_0_1`
scale (e.g. `0.65`, never `65.0` and never `65%`). A *change* in
AUROC/AUPRC (`delta_auroc`, `delta_auprc`, and every paired-bootstrap
interval endpoint/paired difference) is an `absolute_fraction_0_1`
value on that same 0–1 scale — `0.02` absolute is `2.0` percentage
points, and must never itself be written as `"0.02 pp"`. This wording
is frozen in `[gate2_metrics.units]` and `[bootstrap]` in the protocol
identity precisely so the required improvement threshold (`0.02`) is
never misread as `0.02` percentage points (i.e. `0.0002` absolute).

### PAMR's role at Gate 2 vs. Gate 3

PAMR confusion/Boundary-IoU statistics for `t0_pamr`/`t320_pamr` are
**required inputs to Gate-2 completion** (missing or invalid PAMR
statistics make the gate `INCONCLUSIVE`), but PAMR is **not** a feature
of the diagnostic classifier and **not** compared numerically against
the provenance-augmented model at Gate 2 — that comparison (PAMR vs.
the eventual final DFP selector) is explicitly deferred to Gate 3.
There is no PAMR-superiority numerical threshold at Gate 2.

## Boundary IoU: primary; mIoU: secondary

Bound to `evaluation_identities/e14_dfp_phase1b_boundary_iou.toml`.
Mean Boundary IoU is the primary Phase-1B segmentation metric; mIoU is
secondary/preservation. Per-variant (all 12, including `t0_pamr`/
`t320_pamr`), per-image, per-class
`boundary_intersection`/`boundary_union`/`predicted_boundary_pixels`/
`ground_truth_boundary_pixels` (sum-then-divide dataset aggregation),
in parallel with standard semantic confusion statistics. Boundary IoU
is never computed from the completed Phase 1A sufficient statistics
(raw prediction maps were not retained), and Phase 1A is never rerun by
this or any other schema-freezing task. This protocol-freeze task did
not compute Boundary IoU for any variant, including the two PAMR
variants.

## Fixed baselines (12, frozen order)

Required, in this exact order: T0, T16, T32, T64, T320, oracle O_P, and
all four T32 persistence subvariants (full/harmful-only/beneficial-
only/third-label-only) — the original ten, byte-identical to the
Phase 1A baseline set — followed by two appended PAMR variants,
`t0_pamr` and `t320_pamr`. The oracle variants are GT-conditioned
analysis tools, never deployable methods. DenseCRF and zoom/re-crop are
explicitly deferred Gate-3 baselines, not required for Gate 2.

### RGB/local-boundary baseline: bound to the historical PAMR operator

The RGB/local-boundary baseline is PAMR (Pixel-Adaptive Mask
Refinement), bound to the **only** `PAMR(...)` instantiation site in
this repository:

- **Implementation**: `src/open_vocabulary_segmentation/models/dinotext/pamr.py`
  (the vendored, training-free, parameter-free local-affinity smoothing
  operator).
- **Configuration / hardcoded parameters**:
  `src/open_vocabulary_segmentation/models/dinotext/dinotext.py`
  (`apply_pamr`), which hardcodes `num_iter=10`,
  `dilations=[1,2,4,8,12,24]` independent of any YAML — confirmed via
  an exhaustive `grep -rn "PAMR("` across the whole repository.
- **Activation authority**: `README.md`'s own officially-documented
  ViT-Base COCO-Stuff reproduction command (`# COCOStuff` section),
  which passes
  `--eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff_pamr.yml`
  — the inherited configuration bound alongside the implementation.
- **Insertion point**: applied to the soft, per-class score tensor
  after bilinear upsample to final output resolution, strictly before
  argmax; PAMR's own RGB input is the pipeline's own already-loaded,
  already (single-pass) normalized image — never re-decoded, never
  re-normalized, never restored to raw pixel values.
- **Resolved-parameter digest**: every scientifically meaningful PAMR
  parameter (operator identity, iterations, dilations, kernel,
  affinity, insertion point, image input/normalization,
  resize/interpolation, scope, argmax order, determinism,
  training-free status) is hashed into a single
  `resolved_parameter_digest`, computed directly from the same
  `[rgb_local_boundary_baseline]` table it freezes (see
  `compute_resolved_pamr_parameter_digest` in `src/dfp_phase1b_identity.py`)
  — mechanically self-verifying against any future edit to those
  fields.

**`T0_PAMR`** applies this fixed operator to DFP's own already-computed
T0 score tensor; **`T320_PAMR`** applies the *identical* operator,
image, normalization, parameters, scope, and output handling to DFP's
own T320 score tensor, at the same logical insertion point. No
parameter differs between the two (`same_pamr_parameters_for_both`,
`same_rgb_input_for_both`, `same_output_geometry_for_both`,
`same_argmax_convention_for_both` are all frozen `true`) — the pair
exists to test whether conventional local-image refinement explains
boundary improvements that provenance also explains, without
contaminating the causal T0/T320 chain: PAMR never redefines `a(x)` or
`b(x)`, never affects `G`/`k_star`/`phi`/`eta_net`/components/soft
membership/`Q_w`, and is never used to construct components or select
disputed pixels. **No PAMR result exists yet for the frozen 500-image
Phase 1B run** — the historical reference value quoted during
investigation (mIoU with PAMR enabled on a full-dataset COCO-Stuff run)
is context only and is explicitly **not** a Phase 1B result and was not
reproduced by this protocol-freeze task.

## Conservation calibration prerequisite

No numeric conservation tolerance is invented in this protocol. A
required `phase1b_conservation_calibration.json` artifact — 200
deterministically selected windows, same GPU/device/precision,
canonical and masked propagation in the same job, fp64 accumulation,
stitched-field residual measurement — must be generated, independently
verified, and bound into the execution identity before any full Phase
1B run is authorized.

## Resource ceiling

Planning authority: the bound Gate-1 `compute_projection`
(reconstructed exactly, never hardcoded rounded: relative cost
≈19.8313×, worst-case peak bytes ≈113,860,608). Single GPU only, exact
chunking via deterministic memory preflight, peak memory `<80%` of
physical GPU memory, no silently-approximating OOM retry, checkpoint
after every image, byte-identical resume.

## Exclusions (this task)

No masked propagation ran. `G`, `k_star`, and `phi` were never computed
on real data. No diagnostic classifier result exists. No real image or
GT was accessed. No PAMR execution occurred and no real PAMR result
(mIoU or Boundary IoU, for either `t0_pamr` or `t320_pamr`) was
computed on the frozen 500-image Phase 1B run. No real Boundary IoU
value was computed for any variant. No provenance result exists. No
final selector was added. No threshold was tuned. `e13`/`e14`
identities and Phase 1A artifacts were not modified. The PAMR
implementation and configuration files were bound read-only and were
not modified. No Phase 1B execution driver or SLURM launcher was
created. No GPU execution was authorized. No job was submitted.

## Future Gate-3 work

DenseCRF and zoom/re-crop baselines; `phase1b_conservation_calibration.json`
generation; the Phase 1B execution driver; real diagnostic model
fitting; real PAMR execution for `t0_pamr`/`t320_pamr`; the Gate-2
decision on real data; the Gate-3 protocol (if Gate 2 proceeds),
including the deferred PAMR-vs-final-selector comparison.
