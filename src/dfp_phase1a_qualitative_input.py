"""Verified in-memory input contract and per-image category analysis for
DFP Phase 1A Step 11 qualitative reporting.

Ground truth (GT) is used ONLY here, and ONLY for post-hoc category
labeling and panel visualization -- never to construct S0/S320
predictions, never touching graph/recurrence/component/membership code,
never influencing model input. This module accepts only already-computed
in-memory arrays; it never resolves or scans a dataset root.

All category logic is delegated to the already-committed, already-frozen
``src/dfp_phase1a_analysis.py`` (Step 3): ``analyze_phase1a_image``,
``_official_mask_to_boundary`` (the pinned Boundary IoU primitive, reused
unmodified against the dispute mask itself), and
``compute_reversion_coherence``. Nothing here redefines a category
formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.dfp_phase1a_qualitative_selection import (
    CATEGORY_ORDER,
    DfpPhase1aQualitativeReportError,
    category_eligibility_from_pixel_counts,
    compute_category_pixel_counts,
)


@dataclass(frozen=True)
class QualitativeInputRecord:
    """A single verified in-memory example. Contains only what qualitative
    reporting needs -- never a dataset root, never a file path."""

    canonical_image_id: str
    rgb: np.ndarray  # [H, W, 3] uint8, RGB order, full [0, 255] range
    gt: np.ndarray  # [H, W] int64
    scores: Mapping[int, np.ndarray]  # horizon -> [C, H, W] float32/float64
    source_hashes: Mapping[str, str]  # e.g. {"result_sha256": ..., "checkpoint_sha256": ...}


def validate_input_record(record: QualitativeInputRecord, *, class_count: int, ignore_index: int, horizons: Sequence[int]) -> None:
    """Strict validation of one input record. Rejects mismatched H×W
    dimensions, wrong RGB shape/dtype/range, wrong label shape/dtype/
    range, non-finite scores, unknown score horizons, and non-SHA-256
    source hashes. Never resizes a mismatched array. Never mutates
    `record`."""
    if not isinstance(record, QualitativeInputRecord):
        raise DfpPhase1aQualitativeReportError("record must be a QualitativeInputRecord")
    if type(record.canonical_image_id) is not str or not record.canonical_image_id:
        raise DfpPhase1aQualitativeReportError("canonical_image_id must be an exact non-empty string")

    if not isinstance(record.rgb, np.ndarray) or record.rgb.dtype != np.uint8 or record.rgb.ndim != 3 or record.rgb.shape[2] != 3:
        raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: rgb must be a numpy.ndarray of shape [H, W, 3] and dtype uint8")
    height, width = record.rgb.shape[:2]

    if not isinstance(record.gt, np.ndarray) or record.gt.dtype != np.int64 or record.gt.ndim != 2:
        raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: gt must be a numpy.ndarray of shape [H, W] and dtype int64")
    if record.gt.shape != (height, width):
        raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: gt shape {record.gt.shape} disagrees with rgb's [H, W] {(height, width)}")
    gt_valid = record.gt != ignore_index
    if gt_valid.any() and ((record.gt[gt_valid] < 0).any() or (record.gt[gt_valid] >= class_count).any()):
        raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: gt contains a non-ignore value outside [0, class_count)")

    if not isinstance(record.scores, Mapping) or set(record.scores) != set(horizons):
        raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: scores must have exactly one entry per contract horizon {sorted(horizons)}")
    for horizon, arr in record.scores.items():
        if not isinstance(arr, np.ndarray) or arr.dtype not in (np.float32, np.float64):
            raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: scores[{horizon}] must be a numpy.ndarray of dtype float32 or float64")
        if arr.ndim != 3 or arr.shape[0] != class_count or arr.shape[1:] != (height, width):
            raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: scores[{horizon}] must have shape [class_count, H, W] matching rgb's [H, W]")
        if not np.isfinite(arr).all():
            raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: scores[{horizon}] contains a non-finite value (NaN or Infinity)")

    if not isinstance(record.source_hashes, Mapping):
        raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: source_hashes must be a mapping")
    for key, value in record.source_hashes.items():
        if type(key) is not str or type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise DfpPhase1aQualitativeReportError(f"{record.canonical_image_id}: source_hashes[{key!r}] must be a lowercase 64-hex-character SHA-256")


def validate_input_record_sequence(records: Sequence[QualitativeInputRecord], *, expected_image_ids: Sequence[str]) -> None:
    """Requires `records` to be in EXACTLY `expected_image_ids` order --
    no reordering, no missing, no extra, no duplicate."""
    observed = [r.canonical_image_id for r in records]
    if observed != list(expected_image_ids):
        raise DfpPhase1aQualitativeReportError(
            "input record order disagrees with the expected canonical image-ID order "
            f"(expected {list(expected_image_ids)[:5]}..., observed {observed[:5]}...)"
        )


@dataclass(frozen=True)
class QualitativeExampleAnalysis:
    image_id: str
    analysis: Any  # src.dfp_phase1a_analysis.Phase1aImageAnalysis
    boundary_mask: np.ndarray
    reversion_mask: np.ndarray
    reversion_coherence: Any  # src.dfp_phase1a_analysis.ReversionCoherenceStatistics
    category_pixel_counts: dict[str, int]
    category_eligibility: dict[str, bool]


def build_example_analysis(record: QualitativeInputRecord, *, contract: Any, boundary_iou_authority: Any) -> QualitativeExampleAnalysis:
    """The sole per-image orchestration point: validates the input
    record, then delegates ALL category computation to Step 3's own
    ``analyze_phase1a_image``, ``_official_mask_to_boundary``, and
    ``compute_reversion_coherence`` -- never redefining a category
    formula. GT is used only via `record.gt`, passed straight into Step
    3's own already-verified ``analyze_phase1a_image``; nothing here
    otherwise touches GT."""
    from src.dfp_phase1a_analysis import _official_mask_to_boundary, analyze_phase1a_image, compute_reversion_coherence

    validate_input_record(record, class_count=contract.class_count, ignore_index=contract.ignore_index, horizons=contract.horizons)

    analysis = analyze_phase1a_image(record.scores, record.gt, contract=contract)
    dispute_mask_u8 = analysis.dispute.dispute.astype(np.uint8)
    boundary_mask = _official_mask_to_boundary(dispute_mask_u8, dilation_ratio=boundary_iou_authority.dilation_ratio_primary).astype(bool)

    primary_persistence = analysis.persistence[contract.primary_persistence_horizon]
    reversion_mask = analysis.dispute.dispute & (primary_persistence.full != analysis.dispute.b)
    coherence = compute_reversion_coherence(reversion_mask, connectivity=contract.primary_connectivity, small_component_threshold=1.0)

    pixel_counts = compute_category_pixel_counts(
        dispute=analysis.dispute, boundary_mask=boundary_mask, persistence_reverts=reversion_mask,
        reversion_component_count=coherence.component_count,
    )
    eligibility = category_eligibility_from_pixel_counts(pixel_counts)

    return QualitativeExampleAnalysis(
        image_id=record.canonical_image_id, analysis=analysis, boundary_mask=boundary_mask, reversion_mask=reversion_mask,
        reversion_coherence=coherence, category_pixel_counts=pixel_counts, category_eligibility=eligibility,
    )


# ---------------------------------------------------------------------------
# Synthetic fixture-bundle loading -- CPU/synthetic only, never a real
# dataset root. A fixture bundle is a directory containing
# `fixture_manifest.json` (dataset_key, class_count, ignore_index,
# image_ids, npz_filename) and one NPZ holding every image's rgb/gt/
# scores stacked along axis 0 (all images in one bundle share H, W).
# ---------------------------------------------------------------------------

FIXTURE_MANIFEST_KEYS = frozenset({"dataset_key", "class_count", "ignore_index", "horizons", "image_ids", "npz_filename"})


def load_fixture_bundle(bundle_dir) -> list["QualitativeInputRecord"]:
    """Load a synthetic fixture bundle from `bundle_dir` into an ordered
    list of `QualitativeInputRecord`. Never scans or resolves a real
    dataset root -- `bundle_dir` must already contain exactly
    `fixture_manifest.json` and the NPZ it names."""
    import json as _json
    from pathlib import Path as _Path

    import numpy as _np

    bundle_dir = _Path(bundle_dir)
    manifest_path = bundle_dir / "fixture_manifest.json"
    try:
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DfpPhase1aQualitativeReportError(f"cannot read fixture manifest {manifest_path}: {error}") from error
    if set(manifest) != FIXTURE_MANIFEST_KEYS:
        raise DfpPhase1aQualitativeReportError(f"fixture manifest {manifest_path} has an unexpected schema")

    npz_path = bundle_dir / manifest["npz_filename"]
    try:
        with _np.load(npz_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except (OSError, ValueError) as error:
        raise DfpPhase1aQualitativeReportError(f"cannot read fixture NPZ {npz_path}: {error}") from error

    image_ids = manifest["image_ids"]
    horizons = manifest["horizons"]
    if "rgb" not in arrays or "gt" not in arrays:
        raise DfpPhase1aQualitativeReportError(f"fixture NPZ {npz_path} must contain 'rgb' and 'gt' arrays")
    if arrays["rgb"].shape[0] != len(image_ids) or arrays["gt"].shape[0] != len(image_ids):
        raise DfpPhase1aQualitativeReportError(f"fixture NPZ {npz_path} array row count disagrees with fixture_manifest.image_ids")
    for horizon in horizons:
        key = f"scores_{horizon}"
        if key not in arrays:
            raise DfpPhase1aQualitativeReportError(f"fixture NPZ {npz_path} is missing {key!r} for horizon {horizon}")

    records = []
    for i, image_id in enumerate(image_ids):
        scores = {horizon: arrays[f"scores_{horizon}"][i] for horizon in horizons}
        records.append(
            QualitativeInputRecord(
                canonical_image_id=image_id, rgb=arrays["rgb"][i], gt=arrays["gt"][i], scores=scores,
                source_hashes={"fixture_npz_sha256": _sha256_of_file(npz_path)},
            )
        )
    return records


def _sha256_of_file(path) -> str:
    import hashlib as _hashlib
    from pathlib import Path as _Path

    return _hashlib.sha256(_Path(path).read_bytes()).hexdigest()


__all__ = [
    "QualitativeInputRecord",
    "validate_input_record",
    "validate_input_record_sequence",
    "QualitativeExampleAnalysis",
    "build_example_analysis",
    "FIXTURE_MANIFEST_KEYS",
    "load_fixture_bundle",
]
