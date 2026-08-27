# ADE20K C150 matched evaluator

## Cross-dataset evaluation, not a same-data protocol confirmation

Unlike the COCO-Object protocol confirmation (which reuses the same COCO
images/annotations already seen by COCO-Stuff), ADE20K is a genuinely
independent dataset: the frozen COCO-2017-trained Talk2DINO bridge has
never seen ADE20K images or annotations in training. This is a true
zero-shot cross-dataset evaluation of the same E3/matched-k11/k12
scientific method already verified on VOC2012 and COCO-Object.

## Dataset layout and provenance

Standard `ADEChallengeData2016` release layout, used unmodified — no
annotation conversion:

```
ADEChallengeData2016/
  images/{training,validation}/*.jpg
  annotations/{training,validation}/*.png
```

Validation split: exactly **2,000** images (`images/validation`), each
with a matching `.png` mask (`annotations/validation`) — independently
confirmed against the real dataset (`/scratch/.../ADEChallengeData2016`,
not tracked here), not merely assumed from documentation. Canonical
order: sorted validation image filenames — no split file is configured
for this dataset (matches `mmseg.datasets.CustomDataset.load_annotations`'s
own directory-listing sort). Training set (20,210 images) confirmed
disjoint from validation.

## C150 mapping (no C151 variant)

Traced from the live, installed `mmseg.datasets.ade.ADE20KDataset`
(version 0.30.0), not inferred from the numbers 150/151:

- **150 classes**, `reduce_zero_label` **fixed `True`** in
  `ADE20KDataset.__init__` — not config-overridable.
- Raw pixel `0` (unlabeled/background) → `ignore_index` **255**.
- Raw pixel `k` in `[1, 150]` → evaluation label `k - 1`.
- **No background class is evaluated.** No background channel is
  synthesized, no background threshold is applied. This is the
  standard ADE20K protocol — there is deliberately no ADE
  "C151"/background-inclusive variant, unlike VOC's V20/V21 or
  COCO-Object's explicit-background design.

## Shared execution, reused unmodified

The scientific core is imported **unmodified** from the already-verified
`models.dinotext.cover_dr.coco_object_evaluator` module
(`stitch_one_image_with_e3` / `process_one_window_with_e3` /
`WindowOperationTelemetryE3` / `aggregate_telemetry`) — the same shared
E3+k11+k12 per-window orchestration used by COCO-Object — but the
background-channel step (`apply_background_channel` /
`coco_object_evaluator.finalize_prediction`) is **never invoked**.
Finalization instead reuses
`models.dinotext.cover_dr.matched_power_evaluator.finalize_prediction`
— the plain, background-free per-pixel argmax over exactly
`class_count=150` channels, the same finalizer already verified for
VOC2012's V20 (no-background) protocol.

No line of `graph.py`, `finite_step_regime.py`,
`matched_power_evaluator.py`, or `coco_object_evaluator.py` was
modified to build this adapter.

Per-window contract (identical to VOC2012/COCO-Object, enforced by
`WindowOperationTelemetryE3`'s own fail-closed constructor):

| Operation | Count per window |
|---|---|
| snapshot/backbone | 1 |
| top-12 graph build | 1 |
| k11 construction (literal top-12 prefix) | 1 |
| k11 propagation | 1 (320 completed updates) |
| k12 propagation | 1 (320 completed updates) |
| E3 propagation | 0 |
| prediction conversions (E3, k11, k12) | 3 |

ReLU(cosine)³ affinity, directed stable top-12 selection, α=0.98, exactly
320 completed finite-step updates, no early stop, no CGLS/GMRES/dense
solve/fallback — all sourced from `evaluation_identities/e12_matched_k11_k12_t320.toml`,
never duplicated as a Python default.

## Frozen bridge checkpoint

`weights/vitb_mlp_infonce_paired_soft_routing_tau010.pth` — the **same**
checkpoint used by the COCO-Object protocol confirmation (the most
recently established cross-dataset convention in this repository), not
VOC2012's older `vitb_mlp_infonce.pth`. Pinned by SHA-256 in
`evaluation_identities/e12_ade20k_matched_evaluator.toml` and verified
against the actual checkpoint bytes on disk before every run (never
merely format-checked). Frozen throughout: no training, no fine-tuning,
no ADE-specific feature extraction. Ground truth flows only into metric
accumulation (`dataset.pre_eval`), never into prediction construction.

## Logical vs. physical identity

Same separation already hardened for VOC2012/COCO-Object:
`src.dataset_image_identity.reconcile_canonical_image_id` compares the
pipeline-resolved physical path against
`dataset_root / canonical_relative_image_path` — the canonical relative
ID (e.g. `ADE_val_00000001.jpg`) is never derived circularly from
`dataset.img_infos` or the pipeline's own resolved filename; it is read
directly from the real dataset's own `img_infos[i]["filename"]` (the
production authoritative source) and reconciled against the physical
path independently. Adversarially tested (real data): missing suffix,
double suffix, absolute paths, traversal all rejected.

## Run modes

| Mode | Images | Result never represents |
|---|---|---|
| `pilot20` | first 20 (canonical order) | a full conclusion |
| `pilot100` | first 100 | a full conclusion |
| `full` | all 2,000 | — |

Counts and canonical order come from the identity and source manifest —
never a Python literal duplicated at the call site.

## Artifacts

Three variants: **C150 E3**, **C150 k11**, **C150 k12**. Per-image
`int64` sufficient statistics (`intersect`/`union`/`pred`/`label`,
shape `[N, 150]`) persisted in a strict NPZ + JSON manifest pair,
SHA-256-bound into every checkpoint and result document alongside the
evaluation identity, source identity, source manifest, and bridge
checkpoint hashes. Metrics: `percent_0_100`; deltas:
`percentage_points`; dataset-level aAcc/mIoU/mAcc reconstructed by
summing integer statistics first, never averaging per-image mIoU.

## Checkpoint/resume (Phase A)

Same hardened design as VOC2012/COCO-Object: every resume artifact
(checkpoint, per-image-stats manifest + NPZ, identity/source/checkpoint
hashes, run mode, canonical ID prefix) is validated **before** dataset
construction, model construction, bridge-checkpoint loading, or CUDA
initialization. `next_dataset_index == len(completed_image_ids) ==`
every per-image array's row count is enforced. An already-complete
checkpoint is rejected as a resume target.

## Pilot → full decision order

1. Run and independently verify `pilot20` (`scripts/slurm/e12_ade20k_eval_pilot20_h100.sbatch`).
2. Only then run `pilot100` (`scripts/slurm/e12_ade20k_eval_pilot100_h100.sbatch`).
3. Only after both pilots are reviewed, run `full` (`scripts/slurm/e12_ade20k_eval_full2000_h100.sbatch`).

**GPU submission requires separate, explicit authorization beyond
implementing this adapter.** None of these scripts self-submit.

## Portable environment variables

`ADE20K_REPO_ROOT`, `TALK2DINO_VENV`, `ADE20K_REAL_DATA_ROOT`,
`ADE20K_SOURCE_MANIFEST`, `ADE20K_EVAL_OUTPUT_DIR`,
`TALK2DINO_WEIGHT_DIR` — no private path is hardcoded in any tracked
script or identity file.
