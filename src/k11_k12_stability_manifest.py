"""Pure, CPU-only construction of the bounded k11/k12 stability gate's
window manifest.

Depends only on validated image metadata, a validated crop/stride geometry
pair, the canonical :class:`SlidingWindowPlan`, and an explicit canonical
window-count limit -- never on mmcv, mmseg, CUDA, model construction, or
dataset initialization. Crop/stride are supplied by the caller, derived
from the authoritative E3 evaluation identity via
:func:`manifest_geometry_from_e3_identity`; this module never invents or
hardcodes a crop/stride pair of its own.

``SlidingWindowPlan``/``SpatialSize`` live in
``segmentation.evaluation.sliding_window_geometry``, a pure-Python module
with no mmcv/mmseg dependency of its own -- but importing it by its real
dotted path requires the ``segmentation.evaluation`` *package* to already
be importable, and that package's own ``__init__.py`` eagerly imports
mmcv-dependent submodules. ``_load_sliding_window_geometry_module`` below
resolves that module safely: if ``segmentation.evaluation`` is already
loaded (the real production path -- dataset/model construction always
imports it first), the already-loaded module is reused; otherwise a
minimal path-only stub package is installed so the geometry module loads
under its real canonical dotted name (sharing class identity with any
other code that imports it the same way), without ever executing the real
mmcv-eager ``__init__.py``. This mirrors the stub technique already
established in ``tests/test_image_window_cache.py``, formalized here so
every caller -- production and tests alike -- gets the same safe import.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVALUATION_PACKAGE_DIR = _REPO_ROOT / "src/open_vocabulary_segmentation/segmentation/evaluation"


@dataclass(frozen=True)
class ManifestGeometry:
    """An immutable, validated crop/stride pair for manifest construction.

    Never constructed with a default; every field must be supplied and is
    checked to be a positive exact integer (bool, float, and string values
    are all rejected)."""

    crop_height: int
    crop_width: int
    stride_height: int
    stride_width: int

    def __post_init__(self) -> None:
        for name in ("crop_height", "crop_width", "stride_height", "stride_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise K11K12StabilityGateError(
                    f"ManifestGeometry.{name} must be a positive exact integer, observed {value!r}"
                )


def manifest_geometry_from_e3_identity(e3_identity: Mapping[str, Any]) -> ManifestGeometry:
    """Derive validated manifest geometry from an already-loaded,
    already-validated E3 identity mapping (see
    ``src.e3_evaluation_identity.load_identity``, the sole authority for
    crop/stride). Only reads the mapping's already-typed integer pairs;
    never mutates it."""
    crop = e3_identity["evaluation"]["crop"]
    stride = e3_identity["evaluation"]["stride"]
    return ManifestGeometry(
        crop_height=crop[0], crop_width=crop[1],
        stride_height=stride[0], stride_width=stride[1],
    )


def _load_sliding_window_geometry_module():
    module_name = "segmentation.evaluation.sliding_window_geometry"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if "segmentation" not in sys.modules:
        stub = types.ModuleType("segmentation")
        stub.__path__ = [str(_EVALUATION_PACKAGE_DIR.parent)]
        sys.modules["segmentation"] = stub
    if "segmentation.evaluation" not in sys.modules:
        stub = types.ModuleType("segmentation.evaluation")
        stub.__path__ = [str(_EVALUATION_PACKAGE_DIR)]
        sys.modules["segmentation.evaluation"] = stub
    return importlib.import_module(module_name)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_bounded_manifest(
    dataset: Any, *, canonical_window_count: int, geometry: ManifestGeometry
) -> tuple[list[dict], str]:
    """Enumerate exactly the first ``canonical_window_count`` windows in
    canonical image order, then ``SlidingWindowPlan.build`` row-major
    flat-index order. Returns ``(manifest_entries, manifest_sha256)``.

    ``dataset`` must already be the canonical, unmodified COCO-Stuff
    validation dataset object (as built by
    ``segmentation.evaluation.build_seg_dataset`` from the resolved E3
    config) -- this function never invents image order, crop, or stride;
    ``geometry`` (crop/stride) is supplied by the caller from the
    authoritative E3 identity, and image order/dimensions come from the
    dataset object itself.
    """
    if isinstance(canonical_window_count, bool) or not isinstance(canonical_window_count, int) or canonical_window_count < 1:
        raise K11K12StabilityGateError("canonical_window_count must be a positive exact integer")
    if not isinstance(geometry, ManifestGeometry):
        raise K11K12StabilityGateError("geometry must be a validated ManifestGeometry")

    geometry_module = _load_sliding_window_geometry_module()
    SlidingWindowPlan = geometry_module.SlidingWindowPlan
    SpatialSize = geometry_module.SpatialSize

    target = canonical_window_count
    manifest: list[dict] = []
    sample_order_index = 0
    for dataset_index in range(len(dataset)):
        if sample_order_index >= target:
            break
        img_info = (
            dataset.img_infos[dataset_index]
            if hasattr(dataset, "img_infos")
            else dataset.data_infos[dataset_index]
        )
        height = int(img_info["height"])
        width = int(img_info["width"])
        image_id = str(img_info.get("filename", dataset_index))
        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(height, width),
            crop_size=SpatialSize(geometry.crop_height, geometry.crop_width),
            stride=SpatialSize(geometry.stride_height, geometry.stride_width),
        )
        for window in plan.windows:
            if sample_order_index >= target:
                break
            manifest.append(
                {
                    "sample_order_index": sample_order_index,
                    "dataset_index": dataset_index,
                    "image_id": image_id,
                    "image_height": height,
                    "image_width": width,
                    "window_flat_index": window.index,
                    "crop_origin": [window.origin.row, window.origin.col],
                    "crop_end": [window.end.row, window.end.col],
                    "clamped": bool(window.clamped_vertical or window.clamped_horizontal),
                    "patch_grid": [geometry.crop_height // 14, geometry.crop_width // 14],
                }
            )
            sample_order_index += 1

    if len(manifest) != target:
        raise K11K12StabilityGateError(
            f"bounded manifest has {len(manifest)} windows, expected exactly {target} -- "
            "the dataset does not contain enough windows for the registered sample size"
        )
    manifest_bytes = json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8")
    return manifest, _sha256_bytes(manifest_bytes)


__all__ = [
    "ManifestGeometry",
    "build_bounded_manifest",
    "manifest_geometry_from_e3_identity",
]
