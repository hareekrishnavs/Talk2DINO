# COCO-Object protocol confirmation

## Same-data protocol, not an independent dataset

COCO-Stuff remains the primary benchmark. COCO-Object uses the **same
COCO images/annotations** already used for COCO-Stuff, under an
object-centric protocol: 80 "thing" categories plus one explicit
background class, instead of COCO-Stuff's ~171 stuff/thing categories.
This is **not** independent-dataset validation — it tests whether the
COCO-Stuff local-connectivity finding transfers under a different
category regime, explicit background handling, and generally smaller
foreground objects. Genuine independent-dataset confirmation remains a
later roadmap stage.

## The transferred finding

COCO-Stuff (completed): k11 mIoU 30.122544, k12 mIoU 30.143997,
k11−k12 = −0.02145293 pp, paired 95% CI [−0.039102, −0.003700] —
uniform removal of the 12th directed edge causes a small, statistically
resolved degradation. Stitching controls showed no positive evidence;
native cross-view support stopped at `ALIGNMENT_LIMITED`.

## Authoritative dataset/class/background contract

- Dataset class: `COCOObjectDataset` (`img_suffix='.jpg'`,
  `seg_map_suffix='_instanceTrainIds.png'`) — unmodified.
- Dataset config: `coco.py` (`test_cfg: stride=(224,224),
  crop_size=(448,448)`) — unmodified.
- 81 classes: `'background'` (index 0) + 80 thing categories, in the
  exact order `COCOObjectDataset.CLASSES` declares.
- `ignore_index = 255` (mmseg default); the materialized masks never
  actually contain a 255-valued pixel (raw crowd/unlabeled folds to
  background at materialization time — see
  `docs/coco_object_val_materialization.md`), so ignore and stuff are
  never conflated in the metric.
- Background is an **explicit constant-threshold channel**, not a
  learned logit or an implicit complement — see below.

## Background mechanism (load-bearing)

The canonical background rule, reproduced verbatim (not redesigned)
from `segmentation/evaluation/dinotext_seg.py`'s
`DINOTextSegInference.encode_decode`:

```python
background = torch.full([B, 1, H, W], bg_thresh, dtype=torch.float, device=masks.device)
masks = torch.cat([background, masks], dim=1)
```

`bg_thresh = 0.55` (from `eval_coco_object.yml`, resolved and hash-bound
via `resolve_complete_e3_configuration`). Background sits at channel 0,
**competes in the final per-pixel argmax** exactly like every foreground
channel, and this is the official protocol's own mIoU/mAcc — no separate
foreground-only metric is introduced as a new primary.

**Production injects this per-window, before stitching accumulation;
this evaluator injects it once, after stitching, on the already-averaged
foreground scores.** These are mathematically identical: the background
channel is a spatially-uniform constant, and averaging any uniform
constant over a positive per-pixel coverage count reproduces that exact
constant (`(bg_thresh * count) / count == bg_thresh` for any `count >
0`) — independently confirmed in
`tests/test_coco_object_evaluator.py::test_background_channel_equivalence_under_uniform_averaging`.
This equivalence is what lets every variant (E3, k11, k12) share
`matched_power_evaluator`-style accumulation completely unmodified over
the 80 foreground-only channels, with background handled as one small,
independently-testable, shared post-stitch step.

### `with_bg_clean` is registered but inert here

COCO-Object's canonical model config sets `with_bg_clean: true` (unlike
COCO-Stuff's `false`). This flag only affects
`DINOText._generate_masks`'s `similarity_assignment_weighted` step — a
method this evaluator's E3/k11/k12 pipeline **never calls**. The pipeline
exclusively uses `generate_patch_snapshot` (raw similarity, no
`with_bg_clean`) and `masks_from_patch_scores` (sigmoid+interpolate, no
`with_bg_clean`). The flag is recorded in the identity for provenance
only, with this fact documented explicitly rather than silently ignored.

### An existing, orthogonal constraint this evaluator never triggers

`DINOTextSegInference.__init__` raises if `with_bg=True` and
`rwr_config.enabled=True` are both set ("canonical RWR evaluation
requires no background class"). This evaluator never calls
`DINOTextSegInference.encode_decode`/`inference()`/`simple_test()` at
all — like the COCO-Stuff matched k11/k12 power evaluator, it drives
`inference.model.generate_patch_snapshot`/`masks_from_patch_scores`
directly, so this constructor-time guard is never reached regardless of
`with_bg`.

## Exact E3 vs k11 vs k12 comparison

All three variants share, from one snapshot per window:

- the same transformed image and E3 snapshot (`generate_patch_snapshot`,
  called exactly once per window);
- the same immutable DINO features and top-12 directed affinity graph
  (`build_directed_topk_graph`, k=12, called exactly once);
- the same crop geometry (448 crop / 224 stride, from the matched
  identity, never redeclared as a literal);
- the same canonical uniform-probability stitching;
- the same background protocol and metric implementation.

k11 is the literal first-11 prefix of the same top-12 ranking,
independently renormalized (`build_matched_k11_from_k12`) — never an
independent top-11 selection. Both k11 and k12 propagate for exactly
T=320 finite-step iterations at alpha=0.98 (from the matched identity),
no early stopping. **E3 is the same raw unary snapshot converted via the
same `masks_from_patch_scores` sigmoid+interpolation, with zero graph
propagation** (`e3_propagations` is a required, validated `0` in every
checkpoint/result).

## Shared evaluator: required per-window call counts

| Operation | Count |
|---|---|
| snapshot/model call | 1 |
| top-12 selection | 1 |
| k12 graph build | 1 |
| k11 prefix construction | 1 |
| k11 propagation (320 steps) | 1 |
| k12 propagation (320 steps) | 1 |
| **E3 propagation** | **0** |
| sigmoid+interpolation | 3 (one per variant) |

Enforced structurally by `WindowOperationTelemetryE3.__post_init__`
(`src/open_vocabulary_segmentation/models/dinotext/cover_dr/coco_object_evaluator.py`)
and cross-checked again at the aggregate level by
`coco_object_protocol_confirmation_report.verify_record`.

## Data-root override: the only permitted deviation from canonical

`coco.py`'s own `data_root = "./data/coco_stuff164k"` does not contain
the COCO-Object masks (they were materialized separately — see
`docs/coco_object_val_materialization.md`). The evaluator loads the
canonical, unedited `coco.py` config, asserts its `data_root` still
equals the identity's registered `canonical_configured_root`, and
overrides **only** that one field in memory before calling
`build_dataset` — dataset class, annotation suffix, class order,
mapping, background rule, threshold, crop, and stride are read
unmodified from the file. Independently confirmed against the real
materialized data (CPU-only, no CUDA): dataset length 5000, `CLASSES`
81-tuple starting `('background', 'person', 'bicycle', ...)`,
`ignore_index=255`, canonical image order matching the materialization
stage's own digest exactly, GT label set for a real sample
(`{0,1,57,59,61,63,69,73,74,75,76}`) within the allowed `{0..80,255}`
range and identical to the label set independently computed during
materialization for the same image.

## Materialization verification before CUDA

`materialize_coco_object_val.py`/`verify_coco_object_val_materialization.py`
are never invoked by this evaluator (no conversion from the evaluator).
Before any `import torch`/CUDA-availability check, the evaluator runs
`verify_coco_object_val_materialization.py verify-output` as a real
subprocess against the supplied manifest/data-root/source paths, and
requires `complete=true`, `final=true`, `image_count=5000` on the
manifest itself. Independently confirmed: a tampered manifest
(`complete=False`) is rejected before reaching the CUDA check; the real,
unmodified production manifest passes and the pipeline proceeds to
(and, on this CUDA-less environment, cleanly stops at) the CUDA
availability check.

## Checkpoint / run ladder

`pilot20` → `pilot100` → `full` (5000), each with its own schema name
and required image count. Checkpointing follows the established E12
pattern: atomic JSON writes, resumable at the exact next unprocessed
image, every already-completed image's provenance re-validated on
resume, a COCO-Stuff (or any wrong-identity) checkpoint rejected via
`matched_identity_sha256`/`class_count` mismatch. Only `pilot20` may be
submitted after independent verification; `pilot100`/`full` require
separate, later authorization.

## Exclusions

No T4/semantic-crop-consensus, DCR/SUR, Hann/alternative stitching,
cross-view edge support, graph pruning beyond the matched k11 prefix,
learned edge weights, k-sweep, adaptive/class-specific alpha,
Sherman–Morrison, adjoint gradients, CGLS/dense-equilibrium solve,
approximate alignment, new training, annotation generation, or
independent-dataset claims. Paired bootstrap is never computed inside
the GPU evaluator — only per-image sufficient statistics (intersection/
union/pred/GT per class, for all three variants) are persisted, for
offline analysis.

## Interpretation contract (for the later offline analysis, not this stage)

- k11−k12 CI below zero: the 12th edge is useful on average in both
  protocols.
- CI includes zero: the local connectivity result does not clearly
  transfer.
- CI above zero: a protocol-dependent reversal — investigate the
  background/object regime.
- Also report whether k11/k12's gains over E3 transfer.
- No conclusion is drawn from `pilot20` alone.

## Next independent-dataset stage

This protocol confirmation, once complete, remains same-data (COCO
images/annotations). A genuine independent-dataset confirmation is a
separate, later roadmap stage this document does not authorize.

## Verification status of this implementation

Every CPU-testable layer (identity, background-channel helper and its
uniform-averaging equivalence proof, checkpoint/result schema, the full
pre-CUDA path of the evaluator CLI, and real-data dataset instantiation
with the data-root override) has been independently exercised against
the real, already-verified materialized COCO-Object data. The GPU
window-processing/propagation/inference path
(`stitch_one_image_with_e3`/`process_one_window_with_e3`) reuses
`matched_power_evaluator`'s exact, separately-verified primitives
(`build_directed_topk_graph`, `build_matched_k11_from_k12`,
`finite_step_propagate`) plus one new, CPU-tested background step; it
has not itself been executed end-to-end, since doing so requires CUDA
and real model inference, both explicitly out of scope for this
implementation phase. Full correctness of the actual propagated
mIoU/mAcc/aAcc numbers can only be established once `pilot20` is
authorized and run.

## Reproduction commands (not run during implementation)

```
module load opencv/4.14.0
source /scratch/haree/venv/talk2dino-a100/bin/activate

python verify_coco_object_protocol_confirmation.py preflight \
    --repo-root . \
    --materialization-manifest /scratch/haree/coco_object_protocol/manifests/manifest-20443250.json \
    --data-root /scratch/haree/coco_object_protocol \
    --source-masks /scratch/haree/coco_stuff164k/annotations/val2017 \
    --source-images /scratch/haree/coco_stuff164k/images/val2017

python diagnostics/run_coco_object_protocol_confirmation.py \
    --repo-root . \
    --materialization-manifest /scratch/haree/coco_object_protocol/manifests/manifest-20443250.json \
    --data-root /scratch/haree/coco_object_protocol \
    --source-masks /scratch/haree/coco_stuff164k/annotations/val2017 \
    --source-images /scratch/haree/coco_stuff164k/images/val2017 \
    --run-mode pilot20 \
    --checkpoint /scratch/haree/e12_coco_object_evaluation/checkpoint-manual.json \
    --result /scratch/haree/e12_coco_object_evaluation/result-manual.json \
    --per-image-stats /scratch/haree/e12_coco_object_evaluation/per-image-stats-manual.json \
    --device cuda
```

Or, on the cluster: `sbatch scripts/slurm/e12_coco_object_eval_pilot20_h100.sbatch`
(not submitted during implementation; only `pilot20` may be authorized
after independent verification).
