# DFP Phase 1B Boundary IoU

Frozen authority, strict loader, and CPU-only metric implementation for
the Boundary IoU metric required before any DFP Phase 1B evaluation.
This document covers authority-freezing and implementation only — see
*No artifact integration yet* below for what is explicitly out of scope.

## Why this is required

The DFP Phase 1A Gate-1 boundary-mass reframe trigger (see
`docs/dfp_phase1a_gate1_analysis.md`) **fired** on the finalized
COCO-Stuff-500 result: 97.65% of harmful-washout pixels fall inside the
frozen boundary band, exceeding the preregistered 70% threshold
(`[gate_expectations].boundary_mass_reframe_threshold`). Per that
threshold's frozen consequence, Boundary IoU must become the primary
Phase-1B improvement metric (mIoU becomes secondary/preservation), and
Phase-1B evaluation cannot begin until Boundary IoU is bound. This task
performs that binding — implementation only, no evaluation.

## Published Boundary IoU source

```bibtex
@inproceedings{cheng2021boundary,
  title={Boundary {IoU}: Improving Object-Centric Image Segmentation Evaluation},
  author={Bowen Cheng and Ross Girshick and Piotr Doll{\'a}r and Alexander C. Berg and Alexander Kirillov},
  booktitle={CVPR},
  year={2021}
}
```

arXiv: [2103.16562](https://arxiv.org/abs/2103.16562).

## Pinned upstream commit and hashes

- Repository: `https://github.com/bowenc0221/boundary-iou-api`
- Pinned commit: `37d25586a677b043ed585f10e5c42d4e80176ea9` (2021-04-04)
- Canonical source: `boundary_iou/utils/boundary_utils.py`, function
  `mask_to_boundary`
- Upstream source SHA-256:
  `a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c`
- Upstream `license.txt` SHA-256:
  `6da6c74bb714b3f7fe7eb6b817e42dd59ebfca15f292f54a9a7ad33a6a872863`
  (BSD-2-Clause, with additional bundled license blocks for COCOAPI,
  LVISAPI, PANOPTICAPI, and the Cityscapes Dataset that this repository
  neither uses nor redistributes further)

Vendored, byte-identical to the pinned commit, under
`third_party/boundary_iou_api/`:
`UPSTREAM_COMMIT`, `license.txt`,
`boundary_iou/__init__.py`, `boundary_iou/utils/__init__.py`,
`boundary_iou/utils/boundary_utils.py`. The vendored source is **never
imported by production DFP code** (it requires cv2 and pycocotools,
neither of which this repository's production path depends on) — it
exists solely as a pinned, independently-hashed reference oracle for
equivalence testing.

`evaluation_identities/e14_dfp_phase1b_boundary_iou.toml` binds all of
the above (plus the parent Phase 1A protocol's exact path and SHA-256)
under closed-schema, exact-type validation
(`src/dfp_phase1b_boundary_iou_identity.py`).

### Relationship to the pre-existing e13 supplement

This repository already contains a separate, pre-existing additive
supplement, `evaluation_identities/e13_dfp_phase1a_boundary_iou.toml`
(committed at `c042d07`, long before the Gate-1 work), which pins the
*same* upstream commit and source hash and is consumed by
`src/dfp_phase1a_analysis.py`'s `compute_boundary_statistics` (activated
only when `bind_boundary_iou=True` is explicitly passed to
`load_analysis_contract_from_protocol` — it was **never** activated for
the actual completed COCO-Stuff-500 run; only the underlying
`_official_mask_to_boundary` primitive was reused there, for the
diagnostic `harmful_boundary_pixel_count` / `disputed_boundary_pixel_count`
pixel-count split, never for an intersection/union Boundary IoU value).

That supplement's own ignore-handling masks GT/prediction to false at
ignored pixels **before** erosion and re-masks the boundary **after** —
which can manufacture a spurious internal boundary ring around an
ignore hole embedded inside a class region. This e14 identity freezes a
different, ignore-**neutral** erosion construction (see below) that
does not have that failure mode, specifically for DFP Phase 1B.

**The two identities are intentionally generationally separate and
coexist by design.** e14 does not modify, deprecate, or claim to
supersede e13; DFP Phase 1B code binds only to e14. Neither identity
retroactively claims Boundary IoU was available during the completed
Phase 1A run.

## Official binary boundary definition (frozen exactly)

For a binary `uint8` mask of height `H`, width `W`, containing only 0/1:

```
image_diagonal = sqrt(H^2 + W^2)
dilation = int(round(0.02 * image_diagonal))
dilation = max(dilation, 1)
```

1. Pad the mask by exactly 1 pixel on all four sides, constant value 0.
2. Erode with a 3×3 all-ones `uint8` kernel (8-connectivity), for
   exactly `dilation` iterations.
3. Crop the 1-pixel padding.
4. `boundary(mask) = mask - eroded(mask)`.

The zero padding is intentional: a foreground mask truncated by the
image border is treated as having a boundary at that edge. There is no
tunable boundary width in this repository — the ratio `0.02` and the
minimum of `1` are frozen exactly.

## DFP classwise semantic-segmentation adaptation

The official `boundary-iou-api` package supplies only instance/panoptic
integrations built on `mask_to_boundary` — it does not define a
complete classwise semantic-mIoU aggregation with an ignore label. Every
formula below is a **repository-defined DFP extension**, never
attributable to the upstream paper or repository.

### Ignore handling

Ignore pixels are *unknown*, not class-negative. Naively zeroing them
before erosion (as the pre-existing e13 supplement does) can manufacture
an artificial internal boundary around an ignore hole. This identity
instead defines *ignore-neutral erosion*:

```
V = (G != ignore_index)
G_c = (G == c) AND V
P_c = (P == c) AND V

E_V(M) = erosion(M OR NOT V)
B_V(M) = M AND V AND NOT E_V(M)
```

An ignore hole is filled in (treated as foreground) before erosion, so
it can never create a spurious internal boundary; the true image
exterior remains zero-padded, so real image-edge boundaries are still
counted. This reduces exactly to the plain official boundary extraction
when every pixel is valid (`tests/test_dfp_phase1b_boundary_iou.py::
test_all_valid_reduces_to_official_binary_extraction`).

### Dataset aggregation

Per image, per class `c`:

```
boundary_intersection_c = count(B_V(G_c) AND B_V(P_c))
boundary_union_c        = count(B_V(G_c) OR  B_V(P_c))
gt_boundary_pixels_c    = count(B_V(G_c))
pred_boundary_pixels_c  = count(B_V(P_c))
```

Dataset-level, sum-then-divide (never mean-of-per-image):

```
I_c = sum over images of boundary_intersection_c
U_c = sum over images of boundary_union_c
BoundaryIoU_c = I_c / U_c        (only when U_c > 0)
mBoundaryIoU  = mean over classes with U_c > 0 of BoundaryIoU_c
```

No epsilon is ever added to a denominator. If no class has positive
boundary union across the complete evaluated dataset, the aggregate is
rejected as undefined (a named error), never a fabricated value. An
all-ignore image contributes all-zero per-class statistics and does not
independently fail a multi-image aggregate.

### Class inclusion

A class is included in `mBoundaryIoU` exactly when its **dataset-level
aggregate** `U_c > 0` — never based on any single image's own union.

### Units

`percent_0_100` for the metric; `percentage_points` for differences —
identical convention to every other DFP Phase 1A/1B metric.

## CPU implementation and complexity

`src/dfp_phase1b_boundary_iou.py` uses only NumPy and
`scipy.ndimage.binary_erosion` (already an approved CPU dependency
elsewhere in this repository — see
`src/dfp_phase1a_global_unary_components.py`). No OpenCV, no
pycocotools, no torch, no CUDA in the production path. The erosion is
`O(H*W*dilation)` per class per image with SciPy's iterative erosion (or
equivalently expressible as a single Chebyshev distance-transform
threshold — see the module docstring); `dilation` scales with the image
diagonal (`0.02 * diagonal`), so cost grows with image resolution, not
with the number of classes beyond the per-class multiplicative factor.

### Erosion equivalence

The official primitive's own zero-pad is exactly 1 pixel, and
`cv2.erode` is called with no explicit border argument — OpenCV's
default erosion border does not further constrain the result beyond
that explicit pad. This is reproduced as `scipy.ndimage.binary_erosion(
structure=ones((3,3)), iterations=dilation, border_value=1)` on the same
one-pixel-zero-padded array. This equivalence is independently proven
(never merely assumed) in `tests/test_dfp_phase1b_boundary_iou.py`
against a from-scratch, dependency-free brute-force multi-source-BFS
Chebyshev-distance oracle, across many geometries and dilation values
including multi-iteration (`dilation > 1`) cases — and, wherever OpenCV
happens to be importable in the test environment, directly against the
vendored official function itself (skipped, never silently passed, when
OpenCV is unavailable — it was unavailable in the environment this
implementation was developed and verified in).

## Limitations

- No Boundary IoU value has been computed by this task. `src/
  dfp_phase1b_boundary_iou.py` is the metric implementation itself,
  frozen and tested in isolation against synthetic fixtures only.
- Completed Phase 1A artifacts (`evaluation_identities/
  e13_dfp_phase1a_protocol.toml` and everything it binds, plus the
  finalized COCO-Stuff-500 result and both Gate-1 reports) remain
  governed exclusively by the old, unbound
  `[boundary_iou].implementation_source_status = "not_yet_bound"`
  authority. This task changes none of them.
- No Phase 1B result schema, selector, or artifact-integration exists
  yet — that is explicitly a separate, later, independently-authorized
  task.
- Real-OpenCV equivalence testing was not exercised in the environment
  this was implemented in (no working `cv2` import was available); the
  dependency-free brute-force oracle comparison was exhaustive instead.
  An independent verifier with real OpenCV available should re-run
  `tests/test_dfp_phase1b_boundary_iou.py` to additionally exercise the
  (currently skipped) direct-vendored-function comparisons.
- **Phase 1B evaluation remains blocked** until a separate, independent
  verifier confirms this authority and implementation. This document and
  the code it describes do not themselves authorize Phase 1B evaluation.
