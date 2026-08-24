# K11/K12 full-result reconciliation and paired-uncertainty analysis

`analyze_k11_k12_full_result.py` (backed by `src/k11_k12_full_result_analysis.py`)
is an offline, CPU-only analysis stage that runs **after** a completed matched
k11-vs-k12 finite-step power evaluation (`diagnostics/run_matched_k11_k12_evaluation.py
--run-mode full`). It never initializes CUDA, never loads the model or
projection checkpoint, never constructs the dataset, and never reruns any
evaluation. It consumes only the four artifacts a completed run already wrote
(result JSON, checkpoint JSON, per-image-stats manifest JSON, per-image-stats
NPZ) plus git-archived historical records, and produces one deterministic
JSON report.

## Why this stage exists

The first full run (job 20345886) reported `mIoU_k12 ≈ 30.144`, roughly
+0.266 above the finite-step historical anchor recorded in
`evaluation_identities/e12_matched_k11_k12_t320.toml`
(`k12_mIoU = 29.87719634810705`). That discrepancy had to be explained before
the k11/k12 delta itself could be interpreted. This tool:

1. Never trusts the result JSON's own metrics — it independently recomputes
   `aAcc`/`mIoU`/`mAcc` from the archived per-image sufficient statistics
   using the exact same float64, nanmean-over-valid-classes formula as
   `compute_full_precision_metrics` (never reimplemented, only mirrored and
   cross-verified against it).
2. Computes a bounded-memory paired image bootstrap (≥10,000 replicates) to
   determine whether the k11-vs-k12 delta is statistically resolved.
3. Fetches the historical finite-step reference directly from its committed
   git blob (hash-verified against the identity), and reconciles the
   discrepancy against it: metric-formula variants, protocol comparison,
   and an evidence-backed anchor-status decision — never inventing a new
   canonical anchor.

## Usage

```
python analyze_k11_k12_full_result.py \
  --repo-root . \
  --result /scratch/haree/e12_k11_k12_evaluation/result-full-<job>.json \
  --checkpoint /scratch/haree/e12_k11_k12_evaluation/checkpoint-full-<job>.json \
  --per-image-stats /scratch/haree/e12_k11_k12_evaluation/per-image-stats-full-<job>.json \
  --pilot20-result ... --pilot20-checkpoint ... --pilot20-per-image-stats ... \
  --pilot100-result ... --pilot100-checkpoint ... --pilot100-per-image-stats ... \
  --output /scratch/haree/e12_k11_k12_evaluation/analysis-full-<job>.json
```

Pilot arguments are optional (enables the pilot/full nesting audit,
Section 7). The NPZ path for each artifact set defaults to the manifest's
own `npz_filename`, resolved next to the manifest; override with
`--*-per-image-stats-npz` if needed. Pass `--overwrite` to replace an
existing report at `--output`; without it, an existing report is left
untouched and the tool exits 2.

Exit code 2 with a concise `K11/K12 FULL RESULT ANALYSIS FAIL: ...` message
(never a traceback) on any validation, integrity, or computation failure —
`KeyboardInterrupt`/`SystemExit` are never swallowed.

## Report schema

The output is `src.k11_k12_full_result_analysis.SCHEMA_NAME`
(`talk2dino-k11-k12-full-result-analysis-v1`), with top-level sections:
`source_artifact_identities`, `validation_summary`,
`independently_reconstructed_metrics`, `aggregate_sufficient_statistics`,
`gt_and_pairing_consistency`, `paired_bootstrap`,
`equivalence_margin_sensitivity`, `per_class_results`, `pilot_full_nesting`,
`canonical_protocol_comparison`, `metric_reduction_variants`,
`label_statistics_reconciliation`, `anchor_decision`,
`scientific_interpretation`, `limitations`, `stop_proceed_recommendation`.
All metric fields carry explicit units (`percent_0_100`, `fraction_0_1`,
`percentage_points`, `count`) — never inferred from magnitude.

## Paired bootstrap: bounded-memory algorithm

For each replicate: draw `n_images` image indices with replacement, convert
to a per-image draw-count (weight) vector via `np.bincount`, and compute
resampled per-class sums as `weights @ arr` — algebraically identical to
gathering and summing the resampled rows (proven in
`test_bincount_weight_matmul_equals_gather_and_sum`), but never materializing
an array of shape `[replicates, images, classes]`. The **same** per-replicate
weight vector is applied to both k11 and k12 (the paired-sampling
requirement). Replicates are processed in bounded-size chunks (default 200);
`test_bootstrap_chunked_matches_simple_reference` proves a chunk size of 1
(the simple reference) and a large chunk size produce bit-identical results
for the same seed. On the real 5000×171 data, 10,000 replicates complete in
under 5 seconds, single-threaded.

## Canonical anchor decision

One of `ANCHOR_MATCH`, `METRIC_REDUCTION_MISMATCH`, `PROTOCOL_MISMATCH`,
`PREDICTION_MISMATCH_UNEXPLAINED`, or `INSUFFICIENT_PROVENANCE` — see
`determine_anchor_decision`. The historical finite-step reference
(`evaluation_identities/e12_matched_k11_k12_t320.toml [historical_reference]`)
is explicitly marked `sanity_anchor_only = true` with only a 0.01-point
reproduction tolerance; this tool never treats it as a bit-exact target and
never creates or overwrites a canonical anchor — a new anchor is only ever a
*recommendation* in the report, left for a human to act on.
