# DFP Phase 1A Gate-1 Analysis

Deterministic, CPU-only, derived-statistics reconstruction of the
preregistered Gate-1 decision from an already-finalized DFP Phase 1A
result bundle. See `src/dfp_phase1a_gate1_analysis.py`'s own module
docstring for the full scientific/reuse rationale; this document is a
narrow operational summary.

## What this is

A separate analysis layer on top of the frozen result schema (Step 10)
— it never adds fields to `result.json`, never modifies the v2 NPZ
schema, and never re-runs inference, model construction, dataset
access, or CUDA. It reads only an already-verified, already-finalized
result bundle (`complete=true`, `final=true`, schema v2) and computes
sums/ratios/percentiles of already-persisted per-image sufficient
statistics.

## Command

```bash
python diagnostics/analyze_dfp_phase1a_gate1.py \
  --result-bundle /scratch/haree/dfp_phase1a_coco_stuff_500/coco_stuff_500/result \
  --output-dir /scratch/haree/dfp_phase1a_coco_stuff_500_analysis/coco_stuff_500_gate1
```

Exits 0 on success, prints one summary line. Exits 2 with a concise,
traceback-free `DFP PHASE 1A GATE1 FAIL: ...` diagnostic on any
ordinary failure (malformed source, already-existing destination, non-
v2 schema, a violated relational invariant). `KeyboardInterrupt`,
`SystemExit`, and `MemoryError` are never caught.

`--expected-image-count` defaults to 500 (the frozen `coco_stuff_500`/
`ade20k_500` subset size) and exists only so this module's own test
suite can exercise small synthetic fixtures; production runs never
need to pass it.

## Outputs (written atomically to `--output-dir`)

- `gate1_analysis.json` — every reconstructed statistic, strict JSON
  (no NaN/Infinity; undefined values are `null`), deterministic key
  order.
- `gate1_report.md` — human-readable narrative covering all 14 required
  sections (executive verdict through scope statement).
- `per_class_metrics.csv` — per-(variant, class) intersection/union/
  predicted/ground-truth pixel counts and IoU/accuracy.
- `component_summary.csv` — percentile summaries of component/reversion
  distributions.
- `qualitative_selection.json` — the deterministic qualitative-example
  selection (reused verbatim from `src/dfp_phase1a_qualitative_
  selection.py`), for every category reconstructable from persisted
  per-image scalars. Panel rendering is always deferred (raw
  prediction/label pixel data is excluded by design from the v2
  schema); `revert_set_fragmentation` eligibility is always deferred
  (requires a live pixel-level quantity never persisted).
- `analysis_provenance.json` — repository branch/HEAD, every source
  artifact's SHA-256, schema binding, selected-IDs digest, the exact
  command, this analysis code's own SHA-256, a UTC timestamp explicitly
  marked non-scientific, and every generated output's SHA-256.

## Boundary-mass reframe trigger

`gate1_analysis.json`'s `boundary_reframe_trigger` (and the "Boundary
Reframing Trigger" section of `gate1_report.md`) enforces — never
computes — the frozen `[gate_expectations].boundary_mass_reframe_threshold`
/ `boundary_mass_reframe_consequence` pair, read live from
`evaluation_identities/e13_dfp_phase1a_protocol.toml` (never a hardcoded
literal; a tampered threshold, wrong type, or altered consequence string
is rejected by `src.dfp_phase1a_identity.load_identity` before this
analysis ever runs).

- **Numerator/denominator**: `sum(harmful_boundary_pixel_count) /
  sum(harmful_pixel_count)` across the dataset, both already-persisted v2
  scalars. `harmful_boundary_pixel_count + harmful_interior_pixel_count ==
  harmful_pixel_count` is verified (by `build_boundary_concentration`)
  before the trigger is ever evaluated. This is the *harmful-washout*
  boundary concentration specifically — never derived from the
  all-disputes `disputed_boundary_pixel_count`.
- **Strict comparison**: the frozen consequence text reads "if >70% ...",
  so the comparison is strict `observed_fraction > threshold` — a fraction
  exactly equal to the threshold does **not** fire.
- **Three statuses**: `FIRED` (fraction > threshold), `NOT_FIRED` (fraction
  <= threshold), `UNVERIFIABLE` (see zero-harmful below). These are the
  only values `trigger_status` ever takes.
- **Zero-harmful behavior**: if the dataset-level total of
  `harmful_pixel_count` is 0, the fraction is mathematically undefined —
  the trigger never divides by zero, never emits NaN/Infinity anywhere in
  `gate1_analysis.json` (`harmful_washout_boundary_fraction_of_harmful` is
  JSON `null` in this case), sets `trigger_status = "UNVERIFIABLE"`, and
  sets `phase1b_evaluation_ready = false`.
- **Never reverses the Gate-1 decision**: `affects_oracle_headroom_gate`
  is always `false`. The oracle/recovery headroom gate
  (`G_oracle_unary >= 2.0pp AND G_recover >= 1.0pp`) is computed from
  `[gate_contract]` alone and is never read or altered by this trigger —
  `PROCEED_TO_PHASE_1B` stands regardless of the boundary-trigger outcome.
- **Why Phase-1B evaluation is nevertheless blocked when fired**: the
  frozen consequence is that Boundary IoU "becomes the primary improvement
  metric and mIoU becomes secondary/preservation" once the trigger fires.
  Since `[boundary_iou].implementation_source_status` remains
  `not_yet_bound` (this repair implements no Boundary IoU value, chooses
  no boundary width, and binds nothing), evaluating Phase 1B against its
  own required primary metric is not yet possible — hence
  `phase1b_evaluation_ready = false` and `required_next_step =
  "BIND_BOUNDARY_IOU"` whenever the trigger fires. When it does not fire,
  mIoU alone remains sufficient for evaluation and
  `phase1b_evaluation_ready = true`.
- **mIoU stays valid, just secondary when fired**: nothing about this
  trigger revises any previously reported mIoU/horizon/oracle value —
  those numbers are unchanged (see the *Known, explicitly disclosed
  limitations* below for what remains unreported). Once fired, mIoU is
  reframed as a secondary/preservation measurement for Phase 1B, not
  discarded or recomputed.

This document does not define — and this repair does not implement — any
Boundary IoU morphology, boundary width, padding, or ignore-handling
convention; those remain an explicit, separate, not-yet-authorized task.

## Known, explicitly disclosed limitations

- **Boundary IoU is never computed** — `[boundary_iou].
  implementation_source_status` is frozen as `not_yet_bound`. Only
  boundary/interior pixel-COUNT ratios are reported (disputed and
  harmful-washout only; beneficial/third-label boundary splits are not
  persisted separately). See *Boundary-mass reframe trigger* above for how
  the 70% harmful-washout boundary-concentration threshold is enforced and
  reported without computing an IoU value.
- **Q_w distributions are percentiles of per-image summaries**, not a
  true pooled per-window percentile (only each image's own mean/
  median/p90/p95/p99/maximum of `|Q_w|` is persisted, never raw
  per-window values).
- **Active-component count R_w in isolation is not reconstructable**
  (only its product with `|Q_w|`, the projected width, is persisted).
- **The "T32-persistence recovery fraction of oracle G_recover"** is
  reported as a descriptive ratio, never a hard gate:
  `[gate_expectations].expectations_are_soft_not_hard_validity_checks
  = true` in the frozen protocol.
- **`[gate_contract]` is defined over a pair of datasets**
  (`coco_stuff_500`, `ade20k_500`); this analysis covers `coco_stuff_500`
  only. The Gate-1 decision is therefore always dataset-scoped, never a
  cross-dataset claim, and never a claim of a deployable method — only
  measured headroom on a stratified 500-image feasibility subset.
