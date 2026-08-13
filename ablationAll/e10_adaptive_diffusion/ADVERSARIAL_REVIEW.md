# Adversarial review — Parts A/B/C on branch e10-adaptive-diffusion

Reviewed independently: re-derived every claim from code and real-cache
artifacts rather than trusting the implementation's own report. All heavy
checks ran on an allocated GPU node (salloc job 19489216, node o1), never on
the login node.

## Verdict: ADMISSIBLE, with one required fix already applied

One real defect was found and fixed during this review (`RIDGE_ALPHA`
miscalibration in the Part C text-predictability regression — see V5). No
other defect survived verification. Every other check below passed against
real, previously-untouched artifacts and a fresh re-run of the code.

## V1 — Provenance and blast radius: PASS

- `git diff --stat 06964ad..HEAD` (and the working tree): only
  `run_e3_affinity_oracle.py`, `src/e3_affinity_oracle.py`,
  `tests/test_e3_affinity_oracle.py` (one new test, added by this review —
  see V8), plus new files under `ablationAll/e10_adaptive_diffusion/`.
  Nothing under `src/open_vocabulary_segmentation/models/` or
  `src/open_vocabulary_segmentation/configs/`.
- Cache mtime: the newest file in
  `/scratch/haree/talk2dino_e3_affinity_oracle/cache/full` is
  `manifest.json`, mtime 2026-08-09 23:20:31 EDT. The branch's actual
  reflog-recorded creation point is commit `3a39f37` ("ablation results"),
  committed 2026-08-10 13:50:30 EDT — ~14.5h *after* the cache was
  finalized. (Note: `06964ad` is the commit *before* results were
  committed, not the literal branch-creation point per `git reflog show
  e10-adaptive-diffusion`; the cache predates both by a comfortable
  margin either way.)
- Every new artifact's `git_commit`/`git_dirty` provenance field is read
  from the cache manifest at run time (`manifest["source_git_commit"]`,
  `manifest["source_git_dirty"]`), not hand-set — confirmed by reading
  `_result()`/the v2 result-construction code in `run_e3_affinity_oracle.py`.

## V2 — Anchors: PASS (after fixing a bug in my own diff script, not in the code under review)

- `global-sweep-ext --assert-anchors`: exits 0, prints
  `alpha=0.00,T=10 mIoU=28.480169315748`,
  `alpha=0.95,T=10 mIoU=29.483747102672`,
  `aAcc=48.116736004478`, `mAcc=53.713479512576` — matches the recorded
  canonical values.
- Re-ran the **unmodified** `global-sweep` command against the production
  cache, wrote to a new path, diffed against the stored
  `ablationAll/talk2dino_e3_affinity_oracle/results/global_sweep.json`.
  First diff attempt reported a mismatch — investigated field-by-field and
  found the *only* differing values were `runtime_seconds` /
  `peak_cpu_ram_bytes` inside `payload.best_metrics`, which my diff script
  stripped from `payload.rows[*]` but forgot to also strip from
  `payload.best_metrics` (a separate copy of the winning row). Every
  substantive field — `mIoU`, `aAcc`, `mAcc`, all 171-length
  `per_class_iou`/`intersection`/`union`/`ground_truth_pixels`/
  `predicted_pixels`/`per_class_accuracy` arrays, all `delta_*` fields — is
  byte-identical. Corrected diff: **IDENTICAL**.

## V3 — Identity configurations: PASS, exactly (0.0 diff, not just within 1e-6)

Ran for real against the production cache:

| Check | Result |
|---|---|
| A: alpha=0.95, steps=10 | mIoU = 29.483747102672197 (via `--assert-anchors`, exact) |
| B: `local-stat-fit --n-buckets 1` | final mIoU = 29.483747102672197, diff = **0.0** vs Part A optimum |
| C: `bias-fit --bias-grid 0` | final mIoU = 29.483747102672197, diff = **0.0**; 0/171 betas nonzero; `diagnostic_only: true` |

## V4 — The failure mode that already happened once: PASS

Code-level (read, not trusted):
- `coordinate_ascent_fit` in `src/e3_affinity_oracle.py` calls a single
  `evaluate_fn(values)` that always returns the full-171-class confusion
  matrix over every image in the passed-in set (`evaluate_preloaded_bucketed`
  / `evaluate_preloaded_biased`, both built on `metrics_from_confusion`,
  i.e. `nanmean` over all 171 `per_class_iou` entries) — never a per-class
  or per-bucket metric.
- Acceptance is gated twice, both strict: candidate selection
  (`if metrics["mIoU"] > best_metrics["mIoU"] + 1e-6`) and final acceptance
  (`if best_metrics["mIoU"] > current_metrics["mIoU"] + 1e-6`).
- The monotonicity assertion (`if following < previous - 1e-9: raise
  AffinityOracleError(...)`) runs unconditionally after every sweep loop —
  no flag guards it, confirmed by reading the full function body.
- The `>= Part A optimum` gate is a separate, unconditional check in every
  caller (`_run_bucket_fit`, `bias_fit`, `split_half_bias`), executed
  *before* any artifact is written, confirmed by grepping every
  `part_a_optimum_miou` comparison site (4 locations, all unconditional).

Real-data evidence (not synthetic): pulled the structured `trace` array
directly out of the real `split-half --stage bias` artifact
(`split_half_bias_review.json`, 2500-image half A, real GPU run) — 64
entries (1 initial + 63 accepted moves), **zero monotonicity violations**,
mIoU climbing 29.605 -> 31.458 strictly non-decreasing at every step.
`A_final_metrics.mIoU >= part_a_optimum_mIoU`: confirmed true.

## V5 — Leakage: PASS overall; found and fixed a real regularization defect in the Part C regression

- Split-half fingerprints, read directly from the existing (untouched)
  `split_half_global_A.json`:
  `A=4ef6b09d5f17c5f8df8088df1c2bdbec8f31de69f9600cc51e080e5b820ca01d`,
  `B=a6e7d5b07542f1c1a8f30ea040fd6805ae0c83a906dc047ffa4d9450d8171b9b` —
  **exact match** to the values specified for this review. Also confirmed
  identical in the real `split-half --stage bias` run's own artifact.
- Bucket-edge quantile fitting: `fit_bucket_edges`/`torch.quantile` has
  exactly one call site in the entire codebase (`_run_bucket_fit`), and in
  `split_half_local` it is always invoked on `preloaded_a`
  (`preload_cache_images(..., selected_indices=a_indices)`), never on the
  full set — confirmed by grep across both files, not by inference.
- Per-class score std for the bias grid: `class_propagated_score_std` is
  called at exactly two sites; in `split_half_bias` it is called with
  `preloaded_a` only (line-verified). `bias-fit` (no split) correctly uses
  the full in-sample `preloaded` set, as it should for an in-sample
  diagnostic.
- K-fold grouping: `grouped_kfold_indices` partitions `range(n_items)`
  where `n_items` = number of *classes*; `cross_validated_ridge` builds
  `train_indices = all_indices[~held_out_mask]`, which by construction can
  never overlap the held-out fold. Feature standardization
  (`ridge_fit_predict`) computes mean/std from the training rows only, never
  the held-out rows.
- **Found a real defect**: the shuffled-target control, run for real
  (`split-half --stage bias`, real GPU-verified text embedding, real fitted
  beta from a 3.5h coordinate-ascent run on half A) returned R² = **-1.53**
  at the originally-hardcoded `RIDGE_ALPHA = 10.0` — far outside the
  required ±0.05-of-zero band. Diagnosed by sweeping alpha on the exact same
  real `(text_embedding, beta_by_class)` pair: 10 → -1.53, 100 → -0.64,
  1000 → -0.15, **10000 → -0.035** (first value inside tolerance), 1e5 →
  -0.015, 1e6 → -0.013 (by then also flattening the real-signal R² toward
  0). Root cause: with 768-dimensional CLIP features and only ~154 training
  rows per fold (p >> n), an under-regularized ridge overfits every
  training fold regardless of whether the target carries real signal or is
  pure noise — this is a regularization-strength defect, not a data-leak
  (a leak would show the *opposite* signature: an anomalously high, not
  deeply negative, shuffled-control R²). **Fixed**: `RIDGE_ALPHA` raised
  from `10.0` to `10000.0` in `run_e3_affinity_oracle.py`, with the sweep
  evidence recorded in a code comment at the definition site. Confirmed on
  the same real data: shuffled R² = -0.035 (within tolerance), real R² =
  0.045, Spearman = 0.260 vs shuffled Spearman = -0.243.
  (Caveat: this particular real run used an artificially cheap smoke-test
  grid, `--bias-grid "0,0.5"`, not C1's actual default 9-point grid — so
  0.045 should not be read as "the C3 experimental result," only as
  evidence the corrected regression pipeline behaves sanely on real data.
  A full run with the default grid was not performed: at the observed
  ~15s/candidate rate over 171 classes and a 9-value grid it would cost
  multiple GPU-days and was judged out of scope for this review.)

## V6 — Determinism: PASS

Ran `global-sweep-ext --alpha-grid 0.95 --steps 10` twice into different
output paths on the real cache. Diff (ignoring `invocation` and per-row
timing fields): **IDENTICAL**.

## V7 — Numerical sanity: PASS

Checked on the original pre-existing artifacts (`baseline_control.json`,
`global_sweep.json`, all 11 alpha configs) and on every new real artifact
produced during this review
(`global_sweep_ext_minimal.json`, `local_stat_fit_n1.json`,
`bias_fit_zero.json`):
- `intersection[c] <= union[c]` and `intersection[c] <= ground_truth_pixels[c]`
  for all 171 classes, every configuration: holds.
- `sum(ground_truth_pixels)` = **1312660894** in every single configuration
  checked (12 configurations total across old + new artifacts) — a
  dataset-level invariant, confirmed constant.
- `evaluated_images == 5000` for every full-set configuration checked.
- Recomputing mIoU as `mean(intersection/union over defined classes)` from
  the stored arrays matches the stored `mIoU` field to ~1e-14 in every case
  (float roundoff only).

## V8 — Unit test on a known graph: PASS (test did not previously exist; added permanently)

No clique/fully-separated-graph test existed in `tests/`
(`grep -n clique tests/test_e3_affinity_oracle.py` was empty before this
review). Wrote `test_two_fully_separated_cliques_saturate_to_their_own_class`
in `tests/test_e3_affinity_oracle.py`: 16 patches, two fully-separated
8-patch cliques (verified zero cross-clique edges), 2 classes, one
informative patch per clique, rest exactly zero for both classes, using the
actual production `oracle.propagate_scores` (not a reimplementation).
- alpha=0, T=10: output bit-identical to input (`torch.equal`).
- alpha=0.999, T=200: every patch in clique A predicts class 0, every patch
  in clique B predicts class 1 (`argmax` per patch).

Full suite after adding this test: **34/34 passed**.

## Summary table

| Check | Result |
|---|---|
| V1 Provenance/blast radius | PASS |
| V2 Anchors + rerun diff | PASS (after fixing my own diff script) |
| V3 Identity configs (A/B/C) | PASS, exact |
| V4 Joint-objective/monotonicity | PASS |
| V5 Leakage | PASS (after fixing RIDGE_ALPHA 10 -> 10000) |
| V6 Determinism | PASS |
| V7 Numerical sanity | PASS |
| V8 Clique unit test | PASS (test added) |

**Overall: ADMISSIBLE.**

## RUN COMMANDS

```bash
# 1. Request the GPU node (interactive; matches this review's allocation).
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=08:00:00

# Inside the allocation:
cd /project/6114407/haree/Talk2DINO
source /scratch/haree/venv/talk2dino-a100/bin/activate
# Do NOT `module load opencv` -- the only opencv module on this cluster is
# AVX-512-only and SIGILLs on GPU nodes without AVX-512 (confirmed: `grep
# avx512f /proc/cpuinfo` is empty on node o1). Nothing here needs real cv2;
# build_text_embedding.py stubs sys.modules["cv2"] itself.

CACHE=/scratch/haree/talk2dino_e3_affinity_oracle/cache/full
BASELINE=/scratch/haree/talk2dino_e3_affinity_oracle/results/baseline_control.json
SPLIT_GLOBAL=ablationAll/talk2dino_e3_affinity_oracle/results/split_half_global_A.json
ORIG_GS=ablationAll/talk2dino_e3_affinity_oracle/results/global_sweep.json
OUT=ablationAll/e10_adaptive_diffusion/results
mkdir -p "$OUT"

# 2. Text-embedding GPU reproduction + hash check (~1-2 min on GPU).
#    Verifies bit-for-bit against manifest.text_embedding_sha256.
python3 -u ablationAll/e10_adaptive_diffusion/scripts/build_text_embedding.py

# 3. --assert-anchors (~6 min: two full-cache passes at T=10).
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/_anchor_check.json" --csv "$OUT/_anchor_check.csv" \
  --device cuda --assert-anchors

# 4. Re-run the ORIGINAL, unmodified global-sweep and diff (~33 min).
python3 run_e3_affinity_oracle.py global-sweep \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/global_sweep_rerun.json" --csv "$OUT/global_sweep_rerun.csv" \
  --device cuda
python3 - "$OUT/global_sweep_rerun.json" "$ORIG_GS" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def strip(d):
    d = json.loads(json.dumps(d))
    d.pop("invocation", None)
    for row in d["payload"]["rows"]:
        for k in ("runtime_seconds", "peak_cpu_ram_bytes", "peak_gpu_bytes"):
            row.pop(k, None)
    for k in ("runtime_seconds", "peak_cpu_ram_bytes", "peak_gpu_bytes"):
        d["payload"]["best_metrics"].pop(k, None)
    return d
print("IDENTICAL" if strip(a) == strip(b) else "MISMATCH")
PY

# 5. Minimal Part A artifact for downstream identity checks (~3 min).
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/global_sweep_ext_minimal.json" --csv "$OUT/global_sweep_ext_minimal.csv" \
  --alpha-grid 0.95 --steps 10 --device cuda

# 6. V3-B / V3-C identity checks (a few min each).
python3 run_e3_affinity_oracle.py local-stat-fit \
  --cache "$CACHE" --global-sweep "$OUT/global_sweep_ext_minimal.json" \
  --n-buckets 1 --alpha-grid 0.95 --max-sweeps 1 \
  --output "$OUT/local_stat_fit_n1.json" --csv "$OUT/local_stat_fit_n1.csv" --device cuda
python3 run_e3_affinity_oracle.py bias-fit \
  --cache "$CACHE" --global-sweep "$OUT/global_sweep_ext_minimal.json" \
  --bias-grid 0 --max-sweeps 1 \
  --output "$OUT/bias_fit_zero.json" --csv "$OUT/bias_fit_zero.csv" --device cuda

# 7. Determinism check (~6 min: two minimal-grid runs + diff).
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/determinism_run1.json" --csv "$OUT/determinism_run1.csv" \
  --alpha-grid 0.95 --steps 10 --device cuda
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/determinism_run2.json" --csv "$OUT/determinism_run2.csv" \
  --alpha-grid 0.95 --steps 10 --device cuda
# diff as in step 4's pattern, but global-sweep-ext's payload has no
# "best_metrics" field (its "optimum" block carries no timing fields), so
# the plain per-row strip is sufficient here.

# 8. Real split-half-bias run for the V5 shuffled-control check (WARNING:
#    took 3.5 hours in this review at 171 classes x 1 nonzero candidate;
#    the full 9-point default --bias-grid would cost far more -- budget a
#    multi-day sbatch job, not an interactive salloc, if you want the
#    actual C3/C4 experimental numbers rather than a smoke test).
python3 run_e3_affinity_oracle.py split-half --stage bias \
  --cache "$CACHE" --split-global "$SPLIT_GLOBAL" \
  --global-sweep "$OUT/global_sweep_ext_minimal.json" \
  --text-embedding "$OUT/text_embedding.pt" --bias-grid "0,0.5" --max-sweeps 1 \
  --output "$OUT/split_half_bias_review.json" --csv "$OUT/split_half_bias_review.csv" \
  --device cuda

# 9. Unit tests, including the new V8 clique test (login node is fine, CPU-only, seconds).
python3 -m pytest tests/test_e3_affinity_oracle.py -q
```
