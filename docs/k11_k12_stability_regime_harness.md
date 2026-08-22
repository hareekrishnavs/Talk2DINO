# k11/k12 finite-step stability and regime harness

A bounded numerical/runtime gate that must pass **before** the full matched
k11-vs-k12 finite-step evaluation
(`evaluation_identities/e12_matched_k11_k12_t320.toml`) is ever run. This is
not a method efficacy experiment, not a full-validation run, not COVER-DR,
not T4 semantic repair, not DCR/SUR, and not an equilibrium CGLS
reproduction -- see `[prohibited]` in
`evaluation_identities/e12_k11_k12_stability_gate.toml`.

## The six questions this gate answers

1. Is the matched graph construction correct?
2. Is FP32 finite-step propagation numerically reliable?
3. Is T=320 stable relative to T=160 and T=640?
4. Is the k11-k12 difference distinguishable from finite-step
   truncation/arithmetic noise?
5. Is runtime compatible with the planned full evaluation?
6. Can the later production evaluator safely reuse this finite-step
   kernel?

## Deliverables and why each lives where it does

- `evaluation_identities/e12_k11_k12_stability_gate.toml` -- the sole
  authority for every diagnostic-only setting (sample size, reference
  window counts, snapshot steps, tolerances, gate thresholds). It
  references and hashes the parent matched-experiment identity rather than
  duplicating any of its scientific values.
- `src/k11_k12_stability_gate_identity.py` -- identity loader/validator,
  following the exact schema-vocabulary pattern established by
  `src/matched_k11_k12_identity.py` (closed schema, exact types,
  `SUPPORTED_*` constants, relational parent-identity validation).
- `src/k11_k12_stability_report.py` -- structured result/checkpoint schema
  validation and `classify_regime()`, the sole place regime classification
  is derived from TOML-defined thresholds.
- `src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py`
  -- the reusable finite-step kernel, matched k11-from-k12 graph
  construction, and FP64 reference/diagnostic utilities. This lives inside
  the `cover_dr` package (re-exported from its `__init__.py`, exactly like
  `graph.py`/`inference.py`/`rwr.py`) rather than under `diagnostics/`,
  specifically so a **later** production evaluator commit can import it
  directly without depending on the diagnostics harness at all. It is pure
  math over `DirectedTopKGraph`/tensors: it never parses TOML, never loads
  a model, and never touches a dataset.
- `diagnostics/run_k11_k12_stability.py` -- the harness that enumerates
  real windows, extracts the existing immutable snapshot, and drives the
  finite-step kernel. This is the one genuinely new top-level entry point
  this commit adds; `diagnostics/` did not previously exist as a
  convention, but mirrors the repository's existing pattern of dedicated,
  single-purpose top-level scripts (`generate_t4_signal_alignment_report.py`,
  `verify_*.py`).
- `verify_k11_k12_stability.py` -- `preflight` / `verify-result` /
  `verify-checkpoint` CLI, following the exact pattern of
  `verify_matched_k11_k12.py` / `verify_rwr_reproduction.py` /
  `verify_e3_identity.py`.

## Sample selection

Exactly the first `sample_selection.canonical_window_count` (100) windows
in:

1. canonical COCO-Stuff validation image order (the dataset object's own
   order, as built by `segmentation.evaluation.build_seg_dataset` from the
   resolved E3 config -- never re-derived or re-sorted here);
2. canonical `SlidingWindowPlan.build` order within each image;
3. row-major flat window-index order.

No random sampling, no class-based selection, no GT-based selection, no
runtime-based selection, no skipping of "difficult" windows, and no
synthetic window substitution -- see `[prohibited]`. If the 100th window
ends partway through an image, that is expected and acceptable: this is a
per-window numerical gate, not a dataset-level metric evaluation.
`src.k11_k12_stability_manifest.build_bounded_manifest` records, for every
selected window, its dataset index, image ID, image dimensions, window
flat index, crop box, clamped/non-clamped status, patch grid, and
sample-order index, then hashes the whole manifest to a single SHA-256
digest that both the checkpoint and the final result carry. Crop and
stride are never hardcoded in this module: the caller
(`diagnostics.run_k11_k12_stability.run_gate`) derives them once via
`manifest_geometry_from_e3_identity` from the already hash-pinned,
preflighted E3 evaluation identity, and passes the resulting immutable
`ManifestGeometry` in explicitly -- the same authority
`verify_e3_identity.py` itself uses for the E3 evaluation's own crop/
stride. `build_bounded_manifest` depends only on validated image metadata,
that geometry, the canonical `SlidingWindowPlan`, and the canonical
window-count limit; it requires no mmcv, mmseg, CUDA, model construction,
or dataset initialization, and is covered directly by
`tests/test_k11_k12_stability_manifest.py` with hand-derived window
fixtures (never a reimplementation of the enumeration under test).

## Real snapshot extraction path

The harness reuses the existing verified E3 machinery exactly as
production evaluation does -- `models.build_model`,
`segmentation.evaluation.build_seg_dataset`,
`segmentation.evaluation.build_dinotext_seg_inference` -- and then calls
the same read-only snapshot API `window_cache.py`'s two-pass cache already
uses: `inference.model.generate_patch_snapshot(crop, inference.text_embedding)`,
which returns the immutable raw pre-sigmoid patch scores (S0) and DINO
patch features for one crop, from **one** backbone forward pass. The
harness never calls `solve_rwr_cgls`, never runs canonical CGLS to obtain
the snapshot, and never monkeypatches `multi_gpu_test`, `dataset.evaluate`,
or `DINOTextSegInference.slide_inference`.

## Matched graph construction

For each window, `build_directed_topk_graph` (the existing, unmodified
canonical graph builder) is called **exactly once** with `k=12`. The
matched k=11 graph is derived from that single k=12 result via
`finite_step_regime.build_matched_k11_from_k12` -- a literal first-11-column
prefix, independently re-normalized, with the zero-affinity fallback
policy re-derived (never copied) from the retained 11 affinities. Because
the affinity ordering is descending, the zero-affinity fallback condition
is provably k-invariant: `compute_matched_graph_diagnostics` raises if the
prefix or fallback-row sets ever disagree between k11 and k12, rather than
silently tolerating it.

## Finite-step kernel

One continuous 640-step recurrence
(`P(t+1) = alpha * A_k @ P(t) + (1-alpha) * S0`) per graph, capturing
snapshots at exactly the registered steps (160, 320, 640). There is no
convergence test, no tolerance, no early stop, no CGLS, no GMRES, no dense
solve in this production path, and no fallback solver anywhere in
`finite_step_regime.py` -- confirmed by an AST-level test that no call to
a forbidden solver name appears anywhere in the module.

## FP64 references

- **FP64 finite-step reference**: the same kernel, with the
  already-FP32-constructed graph weights and S0 promoted to FP64 (not
  recomputed from scratch in FP64) -- this isolates pure arithmetic/
  accumulation error from graph-construction-precision error.
- **Dense FP64 direct reference**: `K = I - alpha*A`, `B = (1-alpha)*S0`,
  solved via `torch.linalg.solve` (never `torch.linalg.inv`), independently
  verified via the true residual `B - K @ P`. This is never called
  "exact" anywhere in this codebase -- it is a **dense FP64 direct
  reference**, itself subject to floating-point rounding, reported
  alongside its own residual and a backward-error bound.
- **Condition diagnostics**: `sigma_min(K)`, `sigma_max(K)`, and
  `kappa_2(K) = sigma_max/sigma_min` computed from `torch.linalg.svdvals(K)`
  directly -- the true singular values of K, never the eigenvalues of
  `K^T @ K` mistaken for K's own condition number. A separately-labeled,
  illustrative-only Henrici-style departure-from-normality measure is also
  reported; it is not a certified index and gates nothing.

## Gate classification

`classify_regime()` (in `src/k11_k12_stability_report.py`) derives exactly
one of `CLEAR_MATCHED_SIGNAL`, `NUMERICALLY_STABLE_BUT_EFFECT_NEAR_NOISE`,
`NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE`, `TRUNCATION_SENSITIVE`, or
`INVALID` **exclusively** from the TOML-defined `[gate_thresholds]` -- it
never claims k11 helps or hurts mIoU, and a null (near-noise) result is
recorded as honestly useful, not as a failure. `gate_thresholds` requires
`effect_near_noise_relative_delta_norm_max` to be strictly less than
`clear_signal_relative_delta_norm_min`, leaving an intentional gap between
them; an observed effect size that lands strictly inside that gap is
reported as `NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE` -- honestly
neither near-noise nor clear-signal -- rather than being folded into
whichever label happens to be checked first. The same fail-closed rule
applies to non-finite diagnostics: a NaN/Infinity effect size or
truncation-change value, or `numerical_validity_passed` not being exactly
`True`, always classifies as `INVALID`, never as a specific (and
potentially false) regime. A submitted result's `gate_classification` is
independently recomputed from its own reported diagnostics at verification
time and rejected if it disagrees with the TOML-threshold-derived
recomputation.

## Determinism

The harness reruns a small subset of windows (bounded by the smallest
registered reference-window count) `tolerances.determinism_replay_count`
times and compares the sample manifest, graph indices, graph weights,
P160/P320/P640, and argmax digests across replays, excluding runtime and
timestamp fields. Any disagreement sets `drift_detected = true` and is
never silently tolerated with a numeric epsilon unless the gate identity
explicitly defines one for that specific comparison.

## Report and checkpoint

Schemas `talk2dino-k11-k12-stability-gate-v1` and
`talk2dino-k11-k12-stability-checkpoint-v1`. Checkpoints are written via
atomic replacement (`src.k11_k12_stability_report.write_checkpoint_atomically`:
write to a sibling `.tmp` file, then `os.replace`), so a crash mid-write
never leaves a corrupt/partial checkpoint. `resume_window_start()` raises
if a checkpoint is already marked `complete`, so a completed checkpoint can
never be resumed as though it were incomplete.

## Running the preflight and verifiers

```sh
python verify_k11_k12_stability.py preflight --repo-root .
```

```sh
python verify_k11_k12_stability.py verify-result --result PATH --repo-root .
```

```sh
python verify_k11_k12_stability.py verify-checkpoint --checkpoint PATH --repo-root .
```

Run with whichever interpreter has this repository's dependencies
installed (e.g. `<python-environment>/bin/python`). All three exit `0` on
success and `2` with a concise diagnostic on `stderr` on any validation
failure, and never partially write an output file on failure.

## Running the real gate (GPU required, not run by this commit)

```sh
python diagnostics/run_k11_k12_stability.py \
    --output /path/to/scratch/e12_k11_k12_stability_result.json \
    --checkpoint /path/to/scratch/e12_k11_k12_stability_checkpoint.json \
    --device cuda
```

Add `--resume` to continue from an existing, incomplete checkpoint at the
same `--checkpoint` path, or `--dry-run-manifest-only` to build and print
just the bounded window manifest (dataset/model construction still
required) without running the gate itself. `--output`/`--checkpoint` are
rejected outright if they resolve inside the tracked repository tree.

This harness was implemented and unit-tested entirely on CPU with
synthetic graphs; its GPU/dataset/model-construction path reuses the
existing, already-verified `build_model` / `build_seg_dataset` /
`build_dinotext_seg_inference` / `generate_patch_snapshot` entry points
exactly as production evaluation does, but has not itself been executed
against a real GPU allocation as part of this commit -- see
`scripts/slurm/e12_k11_k12_stability_h100.sbatch` for the prepared (not
submitted) job.
