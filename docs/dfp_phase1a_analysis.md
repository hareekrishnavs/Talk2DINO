# DFP Phase 1A — Step 3: pure-CPU analysis core

`src/dfp_phase1a_analysis.py` implements the dataset-independent,
model-independent, CUDA-independent analysis functions Phase 1A's future
execution driver will call once it has already produced stitched score
arrays and ground-truth label maps elsewhere. This module never decodes
an image or mask, never resolves a dataset root, never constructs a
graph or runs propagation, never interpolates or stitches windows, and
never extracts semantic components/soft membership -- it operates
exclusively on already-stitched NumPy arrays and on primitive values the
caller reads from the frozen `e13-dfp-phase1a-protocol` identity.

## Runtime contract

`AnalysisContract` is built only from a live, already-validated protocol
identity (`src.dfp_phase1a_identity.load_identity`) via
`load_analysis_contract_from_protocol(identity, dataset_key="coco_stuff"
| "ade20k")`. No horizon, tie policy, metric unit, ignore policy, class
count, or connectivity value is ever hardcoded in this module; every
analysis function takes these as explicit parameters. The loader itself
rejects (rather than silently proceeding) if the identity's own frozen
`tie_rule` or `horizon_persistence.rule` text no longer matches the one
algorithm this module actually implements.

## Public API

- `validate_stitched_inputs` -- strict, non-mutating validation of one
  image's score dict (`{horizon: array[class, height, width]}`) and
  ground-truth label map.
- `derive_horizon_labels` -- explicit lowest-index-argmax label map per
  horizon (never relies on `np.argmax`'s incidental tie behavior).
- `classify_disputes` -- the four-way harmful/beneficial/third_label/
  ignore partition of the T=0-vs-T=320 dispute set.
- `construct_oracle_maps` -- the GT-conditioned `O_P` (harmful-recovery)
  diagnostic plus the beneficial-undo and third-label one-sided
  diagnostics. **Never a deployable prediction** -- ground truth is used
  only to build these.
- `construct_horizon_persistence_maps` -- the T=16/32/64 persistence
  selector (full + three one-sided variants) per the frozen
  `horizon_persistence.rule`.
- `compute_confusion_statistics` / `aggregate_confusion_statistics` --
  exact per-image and dataset-level intersection/union/predicted/
  ground-truth pixel vectors and confusion matrix, int64, ignore
  excluded.
- `compute_semantic_metrics` -- aAcc/mIoU/mAcc in `percent_0_100`,
  float64, undefined classes excluded from the mean (never coerced to
  0), all-ignore aggregations rejected outright.
- `compute_gain_summary` / `compute_persistence_gain_summary` -- the
  frozen `propagation_gain` / `G_recover` / `G_oracle_unary` triple (with
  their relational identity holding to float64 noise-floor precision)
  and the four independently-computed persistence-horizon deltas.
- `compute_reversion_coherence` -- deterministic connected-component
  statistics for a caller-supplied boolean reverted-pixel mask, using
  caller-supplied connectivity (4 or 8) and a caller-supplied small-
  component threshold **in pixel units** (converting the protocol's
  patch-equivalent threshold to pixels requires production stitching
  geometry this step does not have; that conversion is the caller's
  responsibility).
- `analyze_phase1a_image` -- single-image orchestration wiring the
  above together; does not compute metrics itself (those require
  dataset-level aggregation first) and does not itself touch Boundary
  IoU.
- `load_boundary_iou_authority` / `compute_boundary_statistics` /
  `aggregate_boundary_statistics` / `compute_boundary_metrics` --
  Boundary IoU, bound to the additive authority supplement (see below).

## Boundary IoU

Bound to `evaluation_identities/e13_dfp_phase1a_boundary_iou.toml`
(`src/dfp_phase1a_boundary_iou_identity.py`) -- an **additive**
supplement that resolves, without ever rewriting, the committed
protocol identity's own deliberately deferred
`[boundary_iou].implementation_source_status = "not_yet_bound"` (both
frozen Step-2 subset manifests pin the protocol identity's exact
SHA-256, so it can never be edited in place). None of the three
Boundary IoU functions is operational without an explicit,
already-verified `BoundaryIouAuthority` (`load_boundary_iou_authority
(repo_root=...)`, or `load_analysis_contract_from_protocol(...,
bind_boundary_iou=True, repo_root=...)`) -- there is no default and no
silent fallback.

**Official primitive, exact pinned source.** The binary mask-to-boundary
primitive (`_official_mask_to_boundary` / `_boundary_fixed_width`, both
sharing one `_erode_and_subtract` core) is a line-for-line reproduction
of the pinned official `mask_to_boundary` function from Cheng et al.,
*"Boundary IoU: Improving Object-Centric Image Segmentation Evaluation"*
(CVPR 2021, arXiv:2103.16562, DOI 10.1109/CVPR46437.2021.01508),
`https://github.com/bowenc0221/boundary-iou-api`, commit
`37d25586a677b043ed585f10e5c42d4e80176ea9`,
`boundary_iou/utils/boundary_utils.py`, SHA-256
`a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c`. See
`licenses/boundary_iou_api_LICENSE.txt` for the required BSD notice this
reproduction is bound by. Behavior is preserved exactly: the identical
`round()` convention, one-pixel constant-zero padding, 3×3 uint8 ones
kernel, `cv2.erode` iteration count, crop, and `mask - eroded_mask`
formula. `tests/test_dfp_phase1a_boundary_metrics.py` independently
re-fetches this pinned commit into a fresh `/tmp` directory, verifies
its hash, extracts *only* `mask_to_boundary` via `ast` (never imports
the whole upstream package, which depends on `pycocotools`), and
requires bit-identical output across exhaustive small masks, randomized
larger masks, and edge cases (empty/full/singleton/border-touching/thin-
line/multi-component/near-half-rounding-boundary).

**Fixed 1/2/4-pixel sensitivity widths** reuse the identical
padding/kernel/crop/subtraction core with an exact fixed iteration
count -- never reinterpreted as a ratio.

**Semantic aggregation is a repository-defined extension, not an
official definition.** The upstream boundary-iou-api package defines
only the binary primitive above, never a classwise semantic-segmentation
aggregation; this fact is itself frozen in the supplement's own
`[semantic_extension]` block (`is_official_semantic_definition =
false`), and must never be attributed to the upstream paper or
repository. For every image and class `c`: `gt_c = (gt == c)` (already
ignore-safe, since `ignore_index` is guaranteed outside `[0,
class_count)`); `pred_c = (pred == c) & valid` (explicitly ignore-safe,
since a prediction may legally take any class value at an ignored
pixel); both classes' boundary masks are additionally re-masked by
`valid` after extraction as an explicit safety net -- prediction values
inside ignored pixels never affect any statistic. `boundary_intersection`
/ `boundary_union` accumulate as exact int64 across images
(`aggregate_boundary_statistics`) before any division
(`compute_boundary_metrics`) -- never a mean of per-image bIoU. Class
bIoU is undefined (NaN) where total `boundary_union == 0`; mean bIoU
averages only classes with nonzero total union; an aggregation where
every class has zero total union is rejected outright. Output is
float64 `percent_0_100`.
