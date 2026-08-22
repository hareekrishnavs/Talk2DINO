"""Pure, CPU-only construction of the bounded k11/k12 stability gate's
window manifest.

Depends only on validated per-image inference geometry records, a
validated crop/stride geometry pair, the canonical
:class:`SlidingWindowPlan`, and an explicit canonical window-count limit
-- never on mmcv, mmseg, torch, CUDA, model construction, or dataset
initialization. Crop/stride are supplied by the caller, derived from the
authoritative E3 evaluation identity via
:func:`manifest_geometry_from_e3_identity`; this module never invents or
hardcodes a crop/stride pair of its own.

Per-image inference height/width are supplied by the caller as
:class:`ImageGeometryRecord` values, derived from the *actual processed
inference tensor* the real mmseg test pipeline produces for that image
(see ``diagnostics.run_k11_k12_stability._extract_prepared_image``) --
never from ``dataset.img_infos``/``dataset.data_infos``, which for the
real COCOStuffDataset never carry height/width at all (only ``filename``
and ``ann``; dimensions only exist once an image has actually been loaded
and resized by the pipeline). This module has no way to fabricate that
distinction itself -- it trusts the geometry records it is given were
derived from the real inference tensor, and the adapter that builds them
is responsible for that verification.

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
from typing import Any, Iterable, Mapping

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


@dataclass(frozen=True)
class ImageGeometryRecord:
    """Authoritative, already-verified per-image inference geometry: the
    exact spatial dimensions the real canonical test pipeline produced for
    this dataset index, never raw ``img_infos`` metadata or a PIL/header
    read. ``source_shape_provenance`` records which independent sources
    (the processed tensor, ``img_meta['img_shape']``,
    ``img_meta['pad_shape']``) agreed to produce this height/width, purely
    for audit purposes -- it never affects manifest construction itself."""

    dataset_index: int
    image_id: str
    inference_height: int
    inference_width: int
    source_shape_provenance: str

    def __post_init__(self) -> None:
        if isinstance(self.dataset_index, bool) or not isinstance(self.dataset_index, int) or self.dataset_index < 0:
            raise K11K12StabilityGateError(
                f"ImageGeometryRecord.dataset_index must be a non-negative exact integer, observed {self.dataset_index!r}"
            )
        if type(self.image_id) is not str or not self.image_id:
            raise K11K12StabilityGateError(
                f"ImageGeometryRecord.image_id must be an exact non-empty string, observed {self.image_id!r}"
            )
        for name in ("inference_height", "inference_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise K11K12StabilityGateError(
                    f"ImageGeometryRecord.{name} must be a positive exact integer, observed {value!r}"
                )
        if type(self.source_shape_provenance) is not str or not self.source_shape_provenance:
            raise K11K12StabilityGateError(
                "ImageGeometryRecord.source_shape_provenance must be an exact non-empty string, "
                f"observed {self.source_shape_provenance!r}"
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
    image_geometries: Iterable[ImageGeometryRecord],
    *,
    canonical_window_count: int,
    geometry: ManifestGeometry,
) -> tuple[list[dict], str]:
    """Enumerate exactly the first ``canonical_window_count`` windows in
    canonical image order, then ``SlidingWindowPlan.build`` row-major
    flat-index order. Returns ``(manifest_entries, manifest_sha256)``.

    ``image_geometries`` must yield :class:`ImageGeometryRecord` values in
    canonical, non-decreasing ``dataset_index`` order -- typically a lazy
    generator over the real dataset that runs the canonical test pipeline
    exactly once per image, so images beyond what is needed to reach
    ``canonical_window_count`` are never visited (this function stops
    pulling from the iterable as soon as the target is reached). This
    function never invents image order, crop, stride, or per-image
    dimensions; ``geometry`` (crop/stride) is supplied by the caller from
    the authoritative E3 identity, and per-image dimensions come from the
    already-verified ``image_geometries`` records themselves.
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
    last_dataset_index: int | None = None
    _EXHAUSTED = object()
    iterator = iter(image_geometries)

    while sample_order_index < target:
        # Checking the target BEFORE pulling the next record (rather than
        # `for record in image_geometries:` with a break inside the loop
        # body) is deliberate: a plain `for` loop always calls `next()` to
        # test for exhaustion before the body runs, which would pull one
        # image beyond what is needed whenever the target lands exactly on
        # an image boundary -- and for a lazy, pipeline-driving generator,
        # that means running the real, expensive test pipeline on an image
        # that is never actually used.
        record = next(iterator, _EXHAUSTED)
        if record is _EXHAUSTED:
            break
        if not isinstance(record, ImageGeometryRecord):
            raise K11K12StabilityGateError(
                f"image_geometries must yield ImageGeometryRecord values, observed {type(record).__name__}"
            )
        if last_dataset_index is not None and record.dataset_index <= last_dataset_index:
            raise K11K12StabilityGateError(
                "image_geometries must be in strictly increasing dataset_index order "
                f"(saw {record.dataset_index!r} after {last_dataset_index!r})"
            )
        last_dataset_index = record.dataset_index

        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(record.inference_height, record.inference_width),
            crop_size=SpatialSize(geometry.crop_height, geometry.crop_width),
            stride=SpatialSize(geometry.stride_height, geometry.stride_width),
        )
        for window in plan.windows:
            if sample_order_index >= target:
                break
            manifest.append(
                {
                    "sample_order_index": sample_order_index,
                    "dataset_index": record.dataset_index,
                    "image_id": record.image_id,
                    "image_height": record.inference_height,
                    "image_width": record.inference_width,
                    "window_flat_index": window.index,
                    "crop_origin": [window.origin.row, window.origin.col],
                    "crop_end": [window.end.row, window.end.col],
                    "clamped": bool(window.clamped_vertical or window.clamped_horizontal),
                    "patch_grid": [geometry.crop_height // 14, geometry.crop_width // 14],
                    "source_shape_provenance": record.source_shape_provenance,
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
    "ImageGeometryRecord",
    "ManifestGeometry",
    "build_bounded_manifest",
    "manifest_geometry_from_e3_identity",
]
