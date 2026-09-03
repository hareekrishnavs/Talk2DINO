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

## Known, explicitly disclosed limitations

- **Boundary IoU is never computed** — `[boundary_iou].
  implementation_source_status` is frozen as `not_yet_bound`. Only
  boundary/interior pixel-COUNT ratios are reported (disputed and
  harmful-washout only; beneficial/third-label boundary splits are not
  persisted separately).
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
