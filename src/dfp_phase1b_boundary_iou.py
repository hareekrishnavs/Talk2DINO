"""DFP Phase 1B Boundary IoU: CPU-only, deterministic reconstruction of
the pinned Cheng et al. (2021) binary mask-to-boundary primitive, and a
DFP classwise semantic-segmentation adaptation on top of it (ignore-
neutral erosion), governed exclusively by
`evaluation_identities/e14_dfp_phase1b_boundary_iou.toml` (see
`src.dfp_phase1b_boundary_iou_identity`).

**No Boundary IoU value is computed by this module in production use
yet** -- this is the metric implementation itself, frozen and tested in
isolation. Nothing here reads a Phase 1A result bundle, the COCO-Stuff
dataset, a prediction map, or a checkpoint; nothing here writes a file.
Phase 1B artifact integration (a selector, a result schema, a real
evaluation run) is explicitly out of scope and belongs to a separate,
later, independently-authorized task.

**Dependencies.** NumPy plus SciPy's `scipy.ndimage.binary_erosion`
(already an approved CPU dependency elsewhere in this repository, see
`src.dfp_phase1a_global_unary_components`) -- no OpenCV, no torch, no
CUDA, no pycocotools. The vendored official reference implementation
(`third_party/boundary_iou_api/boundary_iou/utils/boundary_utils.py`,
which itself requires cv2 and pycocotools) is never imported by this
module; it exists only as a pinned, independently-hashed reference
oracle for equivalence testing.

**Erosion equivalence.** The official primitive pads a binary mask by
exactly one constant-zero pixel on every side, then calls
`cv2.erode(kernel=ones((3,3)), iterations=dilation)` with no explicit
border argument of its own. OpenCV's own default erode border treatment
does not further constrain the result beyond that explicit one-pixel
pad (an erosion border default that never forces an additional local
minimum). This is reproduced here as `scipy.ndimage.binary_erosion(
structure=ones((3,3)), iterations=dilation, border_value=1)` on the same
explicitly one-pixel-zero-padded array -- `border_value=1` means
"treat anything beyond this finite padded array as foreground", i.e.
non-constraining, matching that default. This equivalence is
independently proven (never merely assumed) by
`tests/test_dfp_phase1b_boundary_iou.py` against a from-scratch,
dependency-free brute-force multi-source-BFS Chebyshev-distance oracle
implemented directly in the test module (not reused here), across many
geometries and dilation values including `dilation > 1` (multi-
iteration) cases, and -- if OpenCV happens to be importable in the test
environment -- directly against the vendored official function itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from src.dfp_phase1b_boundary_iou_identity import (
    BoundaryIoUContract,
    load_boundary_iou_contract,
)

__all__ = [
    "DfpPhase1bBoundaryIouError",
    "BoundaryIoUContract",
    "BoundaryConfusionStatistics",
    "BoundaryMetricResult",
    "load_boundary_iou_contract",
    "compute_boundary_width",
    "binary_mask_to_boundary",
    "compute_boundary_confusion_statistics",
    "aggregate_boundary_confusion_statistics",
    "compute_boundary_metrics",
]


class DfpPhase1bBoundaryIouError(ValueError):
    """Raised on any DFP Phase 1B Boundary IoU invariant violation.
    Always fail closed. Never raised for, and never catches,
    KeyboardInterrupt, SystemExit, or MemoryError -- those always
    propagate unmodified."""


_EROSION_STRUCTURE = np.ones((3, 3), dtype=bool)


def _require_contract(contract: Any) -> BoundaryIoUContract:
    if not isinstance(contract, BoundaryIoUContract):
        raise DfpPhase1bBoundaryIouError(
            "an already-verified BoundaryIoUContract is required (see load_boundary_iou_contract) -- there is no default"
        )
    return contract


def compute_boundary_width(height: int, width: int, contract: BoundaryIoUContract) -> int:
    """`dilation = int(round(dilation_ratio * sqrt(height**2 + width**2)))`,
    floored at the contract's frozen minimum (1). `height`/`width` must
    be exact positive Python ints (never a bool, never a NumPy scalar)."""
    _require_contract(contract)
    if type(height) is not int or type(width) is not int:
        raise DfpPhase1bBoundaryIouError("height and width must be exact Python int")
    if height <= 0 or width <= 0:
        raise DfpPhase1bBoundaryIouError("height and width must be positive")
    image_diagonal = float(np.sqrt(float(height) ** 2 + float(width) ** 2))
    dilation = int(round(contract.dilation_ratio * image_diagonal))
    if dilation < contract.minimum_dilation:
        dilation = contract.minimum_dilation
    return dilation


def _validate_binary_mask(mask: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise DfpPhase1bBoundaryIouError(f"{label} must be a numpy.ndarray")
    if mask.ndim != 2:
        raise DfpPhase1bBoundaryIouError(f"{label} must be exactly 2-D, observed {mask.ndim}-D")
    if mask.dtype != np.uint8:
        raise DfpPhase1bBoundaryIouError(f"{label} must have dtype uint8, observed {mask.dtype}")
    unique = np.unique(mask)
    if unique.size and (unique.min() < 0 or unique.max() > 1):
        raise DfpPhase1bBoundaryIouError(f"{label} must contain only 0/1 values, observed values including {unique.tolist()}")
    return mask


def _erode(padded_bool: np.ndarray, *, iterations: int) -> np.ndarray:
    """`iterations` sequential 3x3-all-ones (8-connected) binary
    erosions of an already one-pixel-padded boolean array, treating
    anything beyond the finite padded array as foreground (non-
    constraining) -- see the module docstring's Erosion equivalence
    note. Never mutates `padded_bool`; scipy always returns a fresh
    array here since `output` is left at its default (None)."""
    return ndimage.binary_erosion(
        padded_bool, structure=_EROSION_STRUCTURE, iterations=iterations, border_value=1, brute_force=True,
    )


def binary_mask_to_boundary(mask: np.ndarray, *, contract: BoundaryIoUContract, dilation: int | None = None) -> np.ndarray:
    """The official Cheng et al. (2021) binary mask-to-boundary
    primitive, reproduced without any ignore-awareness (see
    `_ignore_neutral_boundary` for the DFP semantic adaptation).
    `mask` must be a 2-D uint8 array containing only 0/1 -- never
    mutated (an internal padded copy is used for the erosion). Accepts
    read-only and non-contiguous arrays. If `dilation` is not given, it
    is derived from `mask.shape` via `compute_boundary_width` using the
    contract's frozen primary ratio; passing it explicitly lets callers
    reuse one dilation across many same-shaped per-class masks without
    recomputing it every time."""
    _require_contract(contract)
    mask = _validate_binary_mask(mask, "mask")
    h, w = mask.shape
    if dilation is None:
        dilation = compute_boundary_width(int(h), int(w), contract)
    elif type(dilation) is not int or dilation < contract.minimum_dilation:
        raise DfpPhase1bBoundaryIouError(f"dilation must be an exact int >= {contract.minimum_dilation}")

    padded = np.pad(
        mask.astype(bool, copy=True), pad_width=contract.padding_width, mode="constant", constant_values=bool(contract.padding_value),
    )
    eroded_padded = _erode(padded, iterations=dilation)
    pw = contract.padding_width
    eroded = eroded_padded[pw : pw + h, pw : pw + w]
    boundary = mask.astype(np.uint8, copy=True)
    boundary[eroded] = 0
    return boundary


def _ignore_neutral_erode(mask_bool: np.ndarray, valid_bool: np.ndarray, *, dilation: int, contract: BoundaryIoUContract) -> np.ndarray:
    """`E_V(M) = erosion(M OR NOT V)` -- an ignore hole is filled in
    (treated as foreground) before erosion so it can never manufacture
    an internal boundary; the true image exterior remains zero-padded
    (via `binary_mask_to_boundary`'s own padding step, replicated here
    identically) so true image-edge boundaries are still counted."""
    h, w = mask_bool.shape
    filled = mask_bool | (~valid_bool)
    padded = np.pad(filled, pad_width=contract.padding_width, mode="constant", constant_values=bool(contract.padding_value))
    eroded_padded = _erode(padded, iterations=dilation)
    pw = contract.padding_width
    return eroded_padded[pw : pw + h, pw : pw + w]


def _ignore_neutral_boundary(mask_bool: np.ndarray, valid_bool: np.ndarray, *, dilation: int, contract: BoundaryIoUContract) -> np.ndarray:
    """`B_V(M) = M AND V AND NOT E_V(M)`. Reduces exactly to
    `binary_mask_to_boundary`'s `mask - eroded` when `valid_bool` is
    all-True (see the equivalence test in
    tests/test_dfp_phase1b_boundary_iou.py)."""
    e_v = _ignore_neutral_erode(mask_bool, valid_bool, dilation=dilation, contract=contract)
    return mask_bool & valid_bool & (~e_v)


@dataclass(frozen=True)
class BoundaryConfusionStatistics:
    """Per-image (or already dataset-aggregated) classwise Boundary IoU
    sufficient statistics. All four arrays are int64, shape
    `(class_count,)`. Never mutate the arrays of an instance returned by
    this module -- they are not writably shared, but are not defensively
    copied on every access either."""

    boundary_intersection: np.ndarray
    boundary_union: np.ndarray
    gt_boundary_pixels: np.ndarray
    pred_boundary_pixels: np.ndarray


def compute_boundary_confusion_statistics(
    pred: np.ndarray, gt: np.ndarray, *, class_count: int, ignore_index: int, contract: BoundaryIoUContract,
) -> BoundaryConfusionStatistics:
    """Classwise Boundary IoU sufficient statistics for ONE image, per
    the frozen DFP semantic adaptation in the e14 identity: `V = (G !=
    ignore_index)`; for each class c, `G_c = (G == c) & V`, `P_c = (P ==
    c) & V`; boundaries are extracted with ignore-neutral erosion
    (`_ignore_neutral_boundary`), never the plain official primitive,
    so an ignore hole embedded inside a class region cannot manufacture
    a spurious internal boundary. Does not mutate `pred` or `gt`;
    accepts read-only and non-contiguous arrays. int64 sufficient
    statistics throughout; no float division happens in this function."""
    _require_contract(contract)
    if not isinstance(pred, np.ndarray) or not isinstance(gt, np.ndarray):
        raise DfpPhase1bBoundaryIouError("pred and gt must be numpy.ndarray")
    if pred.shape != gt.shape:
        raise DfpPhase1bBoundaryIouError(f"pred shape {pred.shape} disagrees with gt shape {gt.shape}")
    if pred.ndim != 2:
        raise DfpPhase1bBoundaryIouError(f"pred/gt must be exactly 2-D, observed {pred.ndim}-D")
    for name, arr in (("pred", pred), ("gt", gt)):
        if not np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
            raise DfpPhase1bBoundaryIouError(f"{name} must have an exact (non-boolean) integer dtype, observed {arr.dtype}")
    if type(class_count) is not int or class_count <= 0:
        raise DfpPhase1bBoundaryIouError("class_count must be a positive exact int")
    if type(ignore_index) is not int:
        raise DfpPhase1bBoundaryIouError("ignore_index must be an exact int")
    if 0 <= ignore_index < class_count:
        raise DfpPhase1bBoundaryIouError(f"ignore_index {ignore_index} must not fall inside the valid class range [0, {class_count})")
    gt_is_class = (gt >= 0) & (gt < class_count)
    gt_is_ignore = gt == ignore_index
    if np.any(~(gt_is_class | gt_is_ignore)):
        raise DfpPhase1bBoundaryIouError("gt contains a value that is neither a valid class in [0, class_count) nor exactly ignore_index")
    if np.any(pred < 0) or np.any(pred >= class_count):
        raise DfpPhase1bBoundaryIouError(f"pred contains a value outside [0, {class_count})")

    h, w = gt.shape
    valid = np.asarray(gt != ignore_index)
    dilation = compute_boundary_width(int(h), int(w), contract)

    boundary_intersection = np.zeros(class_count, dtype=np.int64)
    boundary_union = np.zeros(class_count, dtype=np.int64)
    gt_boundary_pixels = np.zeros(class_count, dtype=np.int64)
    pred_boundary_pixels = np.zeros(class_count, dtype=np.int64)

    if not np.any(valid):
        # An all-ignore image: every class's masks are empty, and
        # _ignore_neutral_boundary would trivially return all-zero
        # boundaries below anyway -- short-circuiting only skips
        # class_count redundant erosion calls, it changes no result.
        return BoundaryConfusionStatistics(
            boundary_intersection=boundary_intersection, boundary_union=boundary_union,
            gt_boundary_pixels=gt_boundary_pixels, pred_boundary_pixels=pred_boundary_pixels,
        )

    for c in range(class_count):
        gt_c = np.asarray(gt == c)
        pred_c = np.asarray(pred == c) & valid

        gt_boundary = _ignore_neutral_boundary(gt_c, valid, dilation=dilation, contract=contract)
        pred_boundary = _ignore_neutral_boundary(pred_c, valid, dilation=dilation, contract=contract)

        boundary_intersection[c] = int(np.count_nonzero(gt_boundary & pred_boundary))
        boundary_union[c] = int(np.count_nonzero(gt_boundary | pred_boundary))
        gt_boundary_pixels[c] = int(np.count_nonzero(gt_boundary))
        pred_boundary_pixels[c] = int(np.count_nonzero(pred_boundary))

    return BoundaryConfusionStatistics(
        boundary_intersection=boundary_intersection, boundary_union=boundary_union,
        gt_boundary_pixels=gt_boundary_pixels, pred_boundary_pixels=pred_boundary_pixels,
    )


def aggregate_boundary_confusion_statistics(stats: Any) -> BoundaryConfusionStatistics:
    """Exact element-wise int64 sum across images -- dataset-level
    Boundary IoU must be computed only after this summation, never
    averaged per image (`compute_boundary_metrics` enforces this by
    only ever accepting the aggregate, never a per-image list)."""
    stats = list(stats)
    if not stats:
        raise DfpPhase1bBoundaryIouError("cannot aggregate an empty sequence of boundary confusion statistics")
    for s in stats:
        if not isinstance(s, BoundaryConfusionStatistics):
            raise DfpPhase1bBoundaryIouError("every element must be a BoundaryConfusionStatistics")
    return BoundaryConfusionStatistics(
        boundary_intersection=np.sum([s.boundary_intersection for s in stats], axis=0, dtype=np.int64),
        boundary_union=np.sum([s.boundary_union for s in stats], axis=0, dtype=np.int64),
        gt_boundary_pixels=np.sum([s.gt_boundary_pixels for s in stats], axis=0, dtype=np.int64),
        pred_boundary_pixels=np.sum([s.pred_boundary_pixels for s in stats], axis=0, dtype=np.int64),
    )


@dataclass(frozen=True)
class BoundaryMetricResult:
    per_class_boundary_iou: np.ndarray  # float64, percent_0_100, NaN where undefined
    mBoundaryIoU: float  # float64, percent_0_100
    classes_with_defined_iou: int


def compute_boundary_metrics(aggregated: BoundaryConfusionStatistics) -> BoundaryMetricResult:
    """Dataset-level sum-then-divide Boundary IoU from an already-
    aggregated `BoundaryConfusionStatistics` (never a per-image one --
    `aggregate_boundary_confusion_statistics` must run first). Class c
    is included in the mean exactly when `boundary_union[c] > 0`; no
    epsilon is ever added to any denominator. Rejects (rather than
    returning a fabricated value) if no class has positive boundary
    union across the whole aggregate."""
    if not isinstance(aggregated, BoundaryConfusionStatistics):
        raise DfpPhase1bBoundaryIouError("aggregated must be a BoundaryConfusionStatistics (the output of aggregate_boundary_confusion_statistics)")
    intersection = aggregated.boundary_intersection.astype(np.int64)
    union = aggregated.boundary_union.astype(np.int64)
    if intersection.shape != union.shape or intersection.ndim != 1:
        raise DfpPhase1bBoundaryIouError("boundary_intersection/boundary_union must be 1-D arrays of equal shape")

    defined = union > 0
    if not np.any(defined):
        raise DfpPhase1bBoundaryIouError(
            "no class has positive boundary union across the complete evaluated aggregate -- Boundary IoU is undefined, refusing to return a fabricated value"
        )

    per_class = np.full(union.shape, np.nan, dtype=np.float64)
    per_class[defined] = (intersection[defined].astype(np.float64) / union[defined].astype(np.float64)) * 100.0
    mean = float(np.mean(per_class[defined]))
    return BoundaryMetricResult(
        per_class_boundary_iou=per_class, mBoundaryIoU=mean, classes_with_defined_iou=int(np.count_nonzero(defined)),
    )
