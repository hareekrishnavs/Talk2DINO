# Matched k11/k12 finite-step power evaluator

A dedicated, single-GPU evaluator that computes **both** the matched k=11
and k=12 finite-step (T=320) variants from exactly the same per-window E3
snapshot and canonical top-12 graph selection, and reports paired
full-precision aAcc/mIoU/mAcc plus per-image/per-class sufficient
statistics. This is the *scientific* evaluator; it is gated behind the
bounded numerical/runtime stability harness in
`docs/k11_k12_stability_regime_harness.md` -- the stability gate answers
"is the numerical machinery trustworthy," this evaluator answers "what do
the two matched variants actually predict."

This is **not** `main.py --eval`. It never calls `multi_gpu_test`, never
monkeypatches production inference, and drives its own bounded per-image
loop. It reuses exactly the same entry points production evaluation and
the stability-gate harness already use for dataset/model construction, and
mmseg's own `dataset.pre_eval` for per-image sufficient statistics.

## Required stability-gate binding

Before any dataset or model construction, the evaluator reads
`--stability-result` (no default path is ever hardcoded -- it must be
passed explicitly) and:

1. Structurally verifies it via the stability gate's own strict verifier
   (`src.k11_k12_stability_report.verify_record`) -- never a
   reimplementation of that schema/threshold logic.
2. Requires `complete=true`, `final=true`, `numerical_validity_passed=true`,
   `failure_reason=null` (the gate's own verifier permits but does not
   require `null`; the evaluator additionally requires it), and exactly
   the registered gate window count (100) processed.
3. Requires `gate_classification` to be one of `CLEAR_MATCHED_SIGNAL`,
   `NUMERICALLY_STABLE_BUT_EFFECT_NEAR_NOISE`, or
   `NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE` -- never `INVALID` or
   `TRUNCATION_SENSITIVE`. This only authorizes *numerical execution*; it
   never claims k11 helps or hurts mIoU either way.
4. Requires `matched_identity_sha256` to match this evaluator's own parent
   matched identity -- the gate and the evaluator must be bound to the
   exact same scientific identity file.
5. Requires the finite-step kernel (`finite_step_regime.py`) and graph
   construction (`graph.py`) source, as they existed at the git commit the
   gate result recorded, to be byte-identical to the currently-imported
   source -- proven by hashing both the live file and the file at that
   commit via `git show <commit>:<path>` and requiring equality, not by
   trusting a recorded hash alone.

Any violation fails closed with `K11K12PowerEvaluationError` before a
dataset object, model, or CUDA context is ever created --
`diagnostics/run_matched_k11_k12_evaluation.py`'s own ordering puts this
binding check ahead of `_build_inference`. This binding has been proven
end to end against the real GPU job `20300858`'s result
(`/scratch/haree/e12_k11_k12_stability/result-20300858.json`), which
passes every one of the above checks with
`gate_classification=NUMERICALLY_STABLE_BUT_REGIME_INCONCLUSIVE`.

## Scientific values are never duplicated

`evaluation_identities/e12_k11_k12_power_evaluation.toml` registers only
execution-contract facts specific to this evaluator binary: run-mode image
counts (20/100/5000), checkpoint granularity, result/checkpoint/artifact
schema names, and the stability-gate acceptance policy above. It
deliberately does **not** redeclare alpha, T=320 steps, crop/stride,
affinity_power, or align_corners -- the evaluator loads those directly,
at runtime, from the parent matched identity
(`evaluation_identities/e12_matched_k11_k12_t320.toml`), the same file the
stability gate is itself bound to. A single authoritative source of truth
is preserved end to end; nothing is copied into Python, tests, or the
SBATCH scripts.

## Matched per-window execution

For every selected crop, exactly:

- **one** E3 patch snapshot (`inference.model.generate_patch_snapshot`) --
  raw pre-sigmoid S0 and immutable DINO features, shared between variants;
- **one** canonical directed top-12 graph build
  (`build_directed_topk_graph`);
- **one** matched-prefix k11 derivation
  (`build_matched_k11_from_k12` -- the literal first-11-column prefix of
  the k12 selection, independently re-normalized; k11 is never
  independently top-k-selected);
- **two** independent graph normalizations (k11's own, k12's own);
- **two** independent finite-step propagations, each **exactly 320**
  updates, using `models.dinotext.cover_dr.finite_step_regime
  .finite_step_propagate` imported unmodified -- the exact kernel already
  verified by GPU stability-gate job `20300858`, never copied or
  reimplemented in the evaluator (`tests/test_matched_power_evaluator.py`
  proves this via AST inspection: no forbidden solver name appears
  anywhere in `matched_power_evaluator.py`, and `finite_step_propagate` is
  imported, never redefined locally);
- **two** independent sigmoid+bilinear-interpolation downstream transforms
  (`inference.model.masks_from_patch_scores`, the exact function
  `window_cache.py`'s two-pass RWR cache and
  `DINOTextSegInference.encode_decode` both already call) -- no PAMR.

Never: a second backbone pass for k12, an independently-selected k11
graph, `solve_rwr_cgls`, a dense solve/inverse in the production path,
early termination, solver fallback, T4/DCR/SUR semantic repair, or
Hann/majority/center-selection/sparse-delta stitching.

## Stitching

`models.dinotext.cover_dr.matched_power_evaluator.stitch_one_image`
processes every window of one image's `SlidingWindowPlan` in row-major
flat-index order, accumulating each variant's sigmoid+interpolated mask
into its own FP32 numerator (`preds11`, `preds12`) against **one shared**
FP32 coverage/count map -- incremented exactly once per window, never once
per variant, since both variants share identical window geometry. Division
happens only after every window of the image has been accumulated
(`preds / count_mat`), mirroring `DINOTextSegInference.slide_inference`'s
own accumulation arithmetic exactly (the `F.pad`-based accumulation step
is duplicated verbatim from that function and `window_cache.py`'s
`_stitch_step`, never independently re-derived). `stitch_one_image` raises
if any pixel is left with zero coverage.

`finalize_prediction` then mirrors `DINOTextSegInference._rescale`/
`simple_test` exactly: crop the stitched prediction to
`img_meta['img_shape']` (a no-op on this pipeline, which has no `Pad`
transform), bilinear-interpolate to `img_meta['ori_shape']` using the
production `inference.align_corners` value (never hardcoded), then take
the per-pixel class argmax -- scores are never rounded or converted first.
Production's intervening image-level softmax is deliberately omitted:
softmax is monotonic per pixel across the class axis, so it cannot change
which class attains the argmax, and omitting it changes no reported
prediction while avoiding redundant floating-point work.

## Metrics

Per-image sufficient statistics (`intersect`, `union`, `predicted_pixels`,
`ground_truth_pixels`) come from mmseg's own
`dataset.pre_eval(pred, dataset_index)` -- never a reimplementation of
`intersect_and_union`. Each array's length is the evaluator's validated
`class_count` (see "Class count", below) -- 171 for the real, canonical
COCO-Stuff task, but never a hardcoded literal. Aggregate aAcc/mIoU/mAcc are reduced from those
per-image tuples by `models.dinotext.cover_dr.compute_full_precision_metrics`,
the same already-verified float64-throughout reducer used to reproduce the
canonical 29.877 mIoU RWR result -- never mmseg's own
`total_area_to_metrics`, which accumulates in float32 and rounds to two
decimal places. Metrics are reported as percentages (`percent_0_100`), and
`delta_mIoU_percentage_points = mIoU_k11 - mIoU_k12` is always recomputed
from the two metrics dicts, never carried as an independent unchecked
scalar. The evaluator never renders an efficacy PASS/FAIL verdict itself.

## Class count

Resolved exactly once, immediately after `_build_inference` returns, from
the live `inference.num_classes` -- never a hardcoded module constant.
The resolved value is cross-checked against the identity's own registered
`metrics.class_count` and, where the dataset exposes it, `dataset.CLASSES`'
length; any disagreement fails closed before the per-image loop starts.
The single validated value is then threaded explicitly into every
downstream consumer -- the per-image-stats artifact builder, its manifest,
the checkpoint, and the result record -- never re-resolved inside the loop.
For the real, canonical COCO-Stuff task this always resolves to 171, but
relationally (through the identity and the live model), not as a literal
written anywhere in this evaluator.

## Per-image artifact

`--per-image-stats` writes a small JSON manifest (schema, image IDs in
canonical order, image-order digest, sibling NPZ filename and its SHA256)
next to a compact NPZ of **exact integer** arrays
(`dataset_indices`, `label`, `intersect_k11/union_k11/pred_k11`,
`intersect_k12/union_k12/pred_k12`), written with `allow_pickle=False` --
no Python object arrays anywhere. `label` (ground truth) is stored once,
shared between variants, rather than duplicated per variant, since it is
architecturally identical for both by construction (both variants are
evaluated against the same `dataset.pre_eval` call site for the same
`dataset_index`); the evaluator additionally asserts the two variants'
recorded GT areas agree before ever appending a row, failing closed on any
divergence.

## Run modes

Exactly three registered modes -- `pilot20` (first 20 canonical images),
`pilot100` (first 100), `full` (all 5000) -- each with its own result
schema name, so a pilot result can never be silently accepted where a full
result is required (`src.k11_k12_power_evaluation_report.verify_record`
checks the recorded schema against the schema registered for the recorded
`run_mode`, and separately requires `final=true` only for `full`). There is
no arbitrary `--max-images` debug escape hatch on this CLI at all.

## Checkpoint / resume

Checkpointed atomically (temp file + `os.replace`) after every **complete**
image (both variants finished, sufficient statistics appended) -- never
mid-image. The checkpoint records `next_dataset_index`,
`completed_image_ids`, `class_count`, the image-order digest,
identity/stability-result/kernel hashes, and windows processed so far; no
GPU tensor is ever serialized.

All checkpoint reading and validation goes through the single authoritative
module `src.k11_k12_power_evaluation_checkpoint` -- the evaluator's resume
path and the standalone `verify_k11_k12_power_evaluation.py
verify-checkpoint` call the exact same functions, never a parallel or
weaker reimplementation. JSON parsing itself
(`parse_strict_json_document`) fails closed on every malformed-input case
(missing file, directory, invalid UTF-8, empty/truncated/trailing-garbage
JSON, duplicate keys, `NaN`/`Infinity`, a non-object root) with a concise
domain error -- never an uncaught traceback.

Validation runs in two phases. **Phase A** (`validate_checkpoint_structure`,
`validate_checkpoint_against_artifact`) needs only the checkpoint document
and the per-image-stats artifact already on disk, so it runs *before*
`_build_inference` -- before any dataset, model, or CUDA construction. It
enforces, among other things, the checkpoint's core self-consistency
invariant: `next_dataset_index` must equal `len(completed_image_ids)`
exactly (never more, never fewer) -- a checkpoint that disagrees here could
otherwise cause an image to be silently skipped or reprocessed on resume,
which is exactly what an earlier version of this evaluator did before this
invariant was added. It also checks that the per-image-stats artifact's
recorded dataset indices are the *exact* canonical prefix
`[0, ..., next_dataset_index - 1]` -- no gap, reordering, skip, or future
index -- and that the k11/k12 row counts agree. **Phase B**
(`validate_checkpoint_against_canonical_order`) needs the freshly-resolved
canonical dataset image order and so only runs once `_build_inference` has
returned; it confirms `completed_image_ids` is the exact canonical
image-ID prefix for the resumed dataset, and separately that the live
`inference.num_classes` agrees with `checkpoint["class_count"]`.

A checkpoint already marked `complete` can never be resumed as though
incomplete. Every checkpoint the evaluator itself writes -- including its
own intermediate per-image writes -- is self-validated through
`validate_checkpoint_structure` before being persisted.

Before writing the final `--result` or printing a `PASS` message, the
evaluator enforces its own **completion contract**: processed image count,
`next_dataset_index`, and `completed_image_ids` length must all equal the
run mode's expected image count; the processed dataset indices must be the
exact full range `[0, image_count)`; and the constructed result is written
to a temporary sibling path, read back, and passed through the exact same
`verify_record` validator the standalone verifier uses -- only if that
self-check passes is the temporary file atomically renamed into
`--result` and the checkpoint promoted to `complete=true`. A result that
fails self-validation is never written to `--result` (a previous valid
result there is never overwritten), and the evaluator exits 2 with a
concise diagnostic rather than printing `PASS`.

## CLI

`diagnostics/run_matched_k11_k12_evaluation.py --repo-root . --identity ...
--stability-result <path> --run-mode {pilot20,pilot100,full} --checkpoint
... --result ... --per-image-stats ... [--resume] --device cuda`. Output
paths (`--result`, `--checkpoint`, `--per-image-stats`, and its sibling
`.npz`) are rejected outright if they resolve inside the tracked
repository (reusing `diagnostics.run_k11_k12_stability
._reject_tracked_output_path` directly). All identity/stability-binding/
path validation happens before CUDA is ever touched; `--help` requires no
heavy dependency.

`verify_k11_k12_power_evaluation.py` provides `preflight`,
`verify-stability-binding`, `verify-result`, and `verify-checkpoint`
subcommands, mirroring `verify_matched_k11_k12.py`'s CLI shape.

## SLURM scripts

Three scripts under `scripts/slurm/`
(`e12_k11_k12_eval_pilot20_h100.sbatch`,
`e12_k11_k12_eval_pilot100_h100.sbatch`,
`e12_k11_k12_eval_full_h100.sbatch`) are prepared but were never submitted
by the commit that introduced them. Each: requires a clean tracked
worktree and the correct branch (mirroring the stability gate's own
guard), requires every harness file to be committed, runs the E3/RWR/
matched/stability/evaluator preflights, passes `--stability-result`
explicitly (no default), runs only its own registered `--run-mode`,
propagates the evaluator's exit status, and never self-submits another
job.
