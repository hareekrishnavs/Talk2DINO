# VOC2012 V20/V21 matched evaluator (E3 vs k11 vs k12, T=320)

## Scope

This is a GPU evaluator for the shared PASCAL VOC 2012 segmentation
validation split, built on top of the already-committed
[`e12-voc2012-dataset-source`](../evaluation_identities/e12_voc2012_dataset_source.toml)
dataset-source identity (`data: add immutable VOC2012 V20/V21 source
identity and preflight`, independently verified and committed). It
produces six matched result variants -- **V20** (20 foreground classes,
no background) and **V21** (21 classes, background included) each under
**E3** (raw snapshot), **k11**, and **k12** (T=320 finite-step
propagation) -- from **one shared model/graph/propagation pass per
image**. It does not train, fine-tune, or pre-extract target-dataset
features.

## Shared V20/V21 execution

V20 and V21 are **not** two separate evaluations. `dinotext_builder.py`'s
`build_dinotext_seg_inference` derives the text query classnames as
`dataset.CLASSES[1:]` whenever `dataset.CLASSES[0] == "background"`
(V21) and `dataset.CLASSES` verbatim otherwise (V20) -- and V21's
`CLASSES[1:]` is *textually identical*, same order, to V20's own
`CLASSES` (registered in the committed VOC2012 source identity's
`class_contract.class_names_relationship`). Background is never
text-encoded or learned: it is injected **after** the foreground scores
are computed, as a spatially-uniform constant channel (see
[Background protocol](#background-protocol) below). So one shared
snapshot → top-12 graph → k11 prefix → k11/k12 propagation pass per
image, over the shared 20 foreground classes, is sufficient for both
protocols; V20 and V21 diverge **only** at finalization:

- **V20**: argmax the shared 20-channel stitched score tensor directly
  (`models.dinotext.cover_dr.matched_power_evaluator.finalize_prediction`,
  imported unmodified -- no background channel).
- **V21**: prepend the canonical constant background channel first, then
  argmax over 21 channels
  (`models.dinotext.cover_dr.coco_object_evaluator.finalize_prediction` +
  `apply_background_channel`, imported unmodified).

Neither the model nor the graph/propagation kernel is ever invoked a
second time for V21. This is enforced structurally, not just
documented: `stitch_one_image_with_e3` is called **exactly once** per
image in the evaluator driver (`diagnostics/run_voc2012_matched_evaluation.py`;
tested via AST inspection in
`tests/test_voc2012_matched_evaluator_scientific_contract.py`), and
`WindowOperationTelemetryE3` fail-closed-enforces the exact per-window
operation counts below.

## Per-window scientific call-count contract

| Operation | Count per window |
|---|---|
| Snapshot/backbone extraction | 1 |
| Top-12 graph construction | 1 |
| k11 prefix construction | 1 |
| Graph row-normalizations (k11 + k12) | 2 |
| k11 finite-step propagation | 1 (320 updates) |
| k12 finite-step propagation | 1 (320 updates) |
| E3 propagation | 0 |
| CGLS / dense-solve / inverse / GMRES / fallback | 0 |

## Exact label mappings

Both authoritative, from the committed VOC2012 source identity and the
live `mmseg`/`PascalVOCDataset20` source (`reduce_zero_label`, applied by
mmseg's own `pre_eval`/`intersect_and_union`, never reimplemented here):

- **V20** (`PascalVOCDataset20`, `reduce_zero_label=True`): raw pixel
  `0` (background) → `255` (ignore, excluded); raw `1..20` → `0..19`;
  raw `255` (void) stays `255`. Background is never evaluated as a V20
  class.
- **V21** (stock `mmseg.datasets.PascalVOCDataset`, `reduce_zero_label=False`):
  raw pixel `0..20` used directly (`0` = background); raw `255` stays
  `255` (ignored).
- `ignore_index = 255` for both.

## Background protocol

V21's background decision reproduces
`segmentation/evaluation/dinotext_seg.py::DINOTextSegInference.encode_decode`'s
own formula exactly:

```python
background_channel = torch.full([B, 1, H, W], bg_thresh)
masks = torch.cat([background_channel, foreground_masks], dim=1)
prediction = masks.argmax(dim=1)  # background wins at bg_thresh > every foreground score
```

`bg_thresh = 0.55`, sourced independently from this evaluator's own
authoritative V21 eval config
(`src/open_vocabulary_segmentation/configs/voc_bg/eval_voc_bg.yml`) --
**not** copied from the COCO-Object evaluator. It happens to equal
COCO-Object's own `bg_thresh` (also `0.55`,
`configs/coco_object/eval_coco_object.yml`) because both share the same
ViT-B/`dinov2_vitb14_reg`/`vitb_mlp_infonce` model family; this is a
verified coincidence, not an assumption -- see
`[background_protocol].provenance_note` in
`evaluation_identities/e12_voc2012_matched_evaluator.toml`. Other
`bg_thresh` values present elsewhere in
`src/open_vocabulary_segmentation/configs/voc_bg/` (e.g. `0.01` in the
generic `voc_bg/eval.yml`, `0.54` in the ViT-L variants) are **not**
used by this identity's config chain.

Background competes in the argmax like any other channel, at index 0
(matching `PascalVOCDataset.CLASSES[0] == 'background'`). Ignore pixels
(`255`) are excluded from the argmax/statistics entirely, via mmseg's own
`pre_eval`.

## Frozen COCO-2017-trained bridge; no target-dataset training

The evaluator loads `weights/vitb_mlp_infonce.pth` (SHA256-pinned in the
identity) read-only via `CheckpointLoader.load_checkpoint` +
`model.load_state_dict(..., strict=False)`. It is the same frozen
Talk2DINO projection head already used, identically, across the E3/RWR/
matched-k11-k12/COCO-Object evaluations in this repository. This
evaluator never runs a training step, never fine-tunes, and never
pre-extracts or caches target-dataset (VOC2012) features for later
reuse -- every DINO/CLIP forward pass happens inline, once per window,
during evaluation. This evaluator makes **no claim** that this
checkpoint reproduces any official COCO-2014 training protocol or
result; a possible future COCO-2014 official-comparison track (training/
evaluating strictly under the official COCO-2014 protocol for a
head-to-head comparison) is out of scope here and would need its own,
separately-provenanced identity -- never conflated with this evaluator's
"COCO-2017-trained" bridge label, which describes the checkpoint already
in use repository-wide, not a reproduction claim.

## Source-manifest dependency

The evaluator refuses to run before a real VOC2012 dataset-source
manifest is independently re-verified against the live dataset via
`verify_voc2012_dataset.py verify-manifest` (a real subprocess
invocation of the existing, already-verified CLI -- never a
reimplemented or partial check), and before the bridge checkpoint bytes
and every parent identity are hash-verified. All of this happens before
any CUDA/model initialization (`verification.require_source_manifest_verified_before_cuda`
/ `require_checkpoint_bytes_verified_before_cuda` in the identity).

## Run modes

| Mode | Images | Prefix |
|---|---|---|
| `pilot20` | 20 | first 20 canonical validation images |
| `pilot100` | 100 | first 100 canonical validation images |
| `full` | 1449 | all canonical validation images |

Image order is the canonical VOC2012 validation-split order (`val.txt`
line order, from the committed source identity) -- never re-sorted,
never independently derived. A pilot result's `schema` field is
distinct from the full-result schema and `final=false`; only a `full`
result has `final=true`. A pilot result must never be presented as a
full scientific conclusion.

## Artifacts

- **Result** (`--result`): strict JSON, `talk2dino-voc2012-matched-evaluator-{pilot20,pilot100,full}-result-v1`.
  Six `metrics_<variant>` blocks (`v20_e3`, `v20_k11`, `v20_k12`,
  `v21_e3`, `v21_k11`, `v21_k12`), each `{aAcc, mIoU, mAcc}` in
  **percent_0_100**. Six delta fields in **percentage points**
  (`delta_mIoU_v20_k11_minus_k12_percentage_points`, etc.), each checked
  against its own metric blocks to within `1e-6`.
- **Per-image-stats manifest + NPZ** (`--per-image-stats`, sibling
  `.npz`): exact integer (`int64`, `allow_pickle=False`) sufficient
  statistics -- `label_v20`, `label_v21`, and
  `intersect_<variant>`/`union_<variant>`/`pred_<variant>` for all six
  variants -- SHA256-bound into both the manifest and the result.
  Metrics are computed by summing these sufficient statistics across
  *all* processed images first (full float64 precision, never mean-of-
  per-image mIoU), then reducing once -- see
  `models.dinotext.cover_dr.compute_full_precision_metrics`.
- **Checkpoint** (`--checkpoint`): resumable, atomically written
  (temp-sibling + `os.replace`), one entry per fully-completed image
  covering all six variants.

## Checkpoint/resume

`--resume` requires the checkpoint's `completed_image_ids` to be an
exact prefix of the canonical image order, `next_dataset_index ==
len(completed_image_ids)`, and every identity/checkpoint/source-manifest
SHA256 binding to match the currently-loaded identity -- a checkpoint
from a different run mode, class count, or identity is rejected before
any model/dataset work. A corrupted checkpoint (malformed JSON,
duplicate keys, NaN/Infinity) fails via `parse_strict_json_document`
before the model or a dataset sample is ever touched. A run is never
marked `complete=true`/`final=true` until the exact expected image
count has been processed, both V20 and V21 statistics for all three
variants are present and finite, and the freshly-built result
self-verifies (`verify_record`) before atomic installation.

## Metric units

- Metrics (`aAcc`, `mIoU`, `mAcc`): **percent_0_100** (i.e. `0..100`,
  never `0..1`).
- Deltas (`delta_mIoU_*_percentage_points`): **percentage points** (a
  difference of two `percent_0_100` values) -- never confused with the
  metric scale itself.

## Pilot-to-full decision order

1. Run `pilot20`. Review the result (six metric blocks, six deltas,
   operation telemetry) via `verify_voc2012_matched_evaluation.py
   verify-result`.
2. Run `pilot100`. Same review.
3. Only after both pilots are reviewed and accepted, submit the `full`
   (1449-image) job. The full SLURM script's header comment states this
   explicitly; it is a documented human-review gate (matching this
   repository's established convention for every other e12 evaluator),
   not an automated block.

## Environment variables and commands

No tracked file in this evaluator embeds a private absolute path. Set:

```sh
export VOC2012_REPO_ROOT="/path/to/your/Talk2DINO/checkout"
export TALK2DINO_VENV="/path/to/your/virtualenv"
export VOC2012_REAL_DATA_ROOT="/path/to/VOCdevkit"
export VOC2012_SOURCE_MANIFEST="/path/to/a/verified/voc2012_manifest.json"
export VOC2012_EVAL_OUTPUT_DIR="/path/to/writable/output/dir"
export TALK2DINO_WEIGHT_DIR="/path/to/weights"
```

Preflight and verification (CPU-only, safe on a login node):

```sh
python verify_voc2012_dataset.py preflight --repo-root "$VOC2012_REPO_ROOT" --data-root "$VOC2012_REAL_DATA_ROOT"
python verify_voc2012_matched_evaluation.py preflight --repo-root "$VOC2012_REPO_ROOT"
python verify_voc2012_matched_evaluation.py verify-source-binding \
    --repo-root "$VOC2012_REPO_ROOT" --data-root "$VOC2012_REAL_DATA_ROOT" --source-manifest "$VOC2012_SOURCE_MANIFEST"
```

Evaluation (GPU job; see `scripts/slurm/e12_voc2012_eval_{pilot20,pilot100,full1449}_h100.sbatch`):

```sh
python diagnostics/run_voc2012_matched_evaluation.py \
    --repo-root "$VOC2012_REPO_ROOT" \
    --data-root "$VOC2012_REAL_DATA_ROOT" \
    --source-manifest "$VOC2012_SOURCE_MANIFEST" \
    --run-mode pilot20 \
    --checkpoint "$VOC2012_EVAL_OUTPUT_DIR/checkpoint-pilot20.json" \
    --result "$VOC2012_EVAL_OUTPUT_DIR/result-pilot20.json" \
    --per-image-stats "$VOC2012_EVAL_OUTPUT_DIR/per-image-stats-pilot20.json" \
    --device cuda

python verify_voc2012_matched_evaluation.py verify-result --repo-root "$VOC2012_REPO_ROOT" --result "$VOC2012_EVAL_OUTPUT_DIR/result-pilot20.json"
```

`--data-root` falls back to `$VOC2012_REAL_DATA_ROOT` if `--data-root`
is omitted; one of the two must be set.

## Files

- `evaluation_identities/e12_voc2012_matched_evaluator.toml` -- sole new
  authority for this evaluator's scientific/execution contract; binds
  relationally to the committed VOC2012 source identity and the matched
  k11/k12 identity, never redeclaring their values.
- `src/voc2012_matched_evaluator_identity.py` -- strict identity loading/
  validation, parent-identity binding, model/checkpoint file-hash
  binding.
- `src/voc2012_matched_evaluator_checkpoint.py` -- checkpoint schema/
  resume-invariant validation.
- `src/voc2012_matched_evaluator_report.py` -- result schema/invariant
  validation.
- `diagnostics/run_voc2012_matched_evaluation.py` -- the evaluator
  driver (`--run-mode pilot20|pilot100|full`).
- `verify_voc2012_matched_evaluation.py` -- standalone verifier CLI
  (`preflight` / `verify-source-binding` / `verify-checkpoint` /
  `verify-result`).
- `scripts/slurm/e12_voc2012_eval_{pilot20,pilot100,full1449}_h100.sbatch`.

No production code in this evaluator is COCO-Object-specific: the
per-window snapshot/graph/propagation/stitching primitives
(`models.dinotext.cover_dr.coco_object_evaluator`,
`models.dinotext.cover_dr.matched_power_evaluator`) are imported
unmodified and are already parametrized by `class_count`; only the
numeric `bg_thresh` value is dataset-specific, and it is independently
sourced and verified above.
