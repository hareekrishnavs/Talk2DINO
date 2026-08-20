"""Harness logic for running the T4 trust/centrality diagnostics across a
full or bounded slice of the evaluation dataset, with genuinely observed
segmentation metrics (aAcc/mIoU/mAcc) recorded from the SAME predictions
already produced by the diagnostic-enabled path -- never a second model
forward.

This module is pure orchestration/bookkeeping logic (run-mode decision,
parity-check scheduling, a streaming segmentation-metric accumulator,
canonical-identity attribution, and checkpoint serialization). It has no
GPU dependency and is fully unit-testable; the actual GPU-driving script
(outside the repository, alongside the other bounded pilots) imports this
module and supplies real predictions/tensors from the live evaluation
loop.

Identity attribution
---------------------
Canonical **E3** metrics (``aAcc=46.614213``, ``mIoU=28.480169``,
``mAcc=52.077968``) and canonical **directed-RWR** metrics
(``rwr_aAcc=48.52867057377273``, ``rwr_mIoU=29.877244374599126``,
``rwr_mAcc=54.13703466982515``) live in two separate identity TOMLs and
are loaded through two separate, non-interchangeable functions
(:func:`load_e3_reference_metrics` / :func:`load_rwr_reference_metrics`).
Neither this module nor its tests hardcode these numbers; they are always
read from ``evaluation_identities/e3_paired_soft_routing.toml`` and
``evaluation_identities/e3_canonical_directed_rwr.toml`` respectively.

Observed vs. reference
-----------------------
:class:`ObservedRunMetrics` holds only metrics actually computed from this
run's own predictions via :class:`StreamingSegmentationMetricAccumulator`.
:class:`CanonicalReferenceMetrics` holds only metrics loaded from an
identity TOML. The two are never merged into one object, and reference
metrics are never substituted into a run's observed fields -- see
:func:`build_observed_metrics_report`.

Full vs. partial
-----------------
:func:`determine_run_mode` decides ``"full_dataset"`` (the requested image
limit is absent or >= dataset length) vs. ``"partial"`` (bounded). Only a
``full_dataset`` run may report a non-``None`` full-dataset ``mIoU``;
partial runs record why full-dataset metrics are unavailable rather than
silently omitting the field or reporting canonical reference numbers under
an observed key.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

CHECKPOINT_SCHEMA_VERSION_V1 = "talk2dino-trust-centrality-harness-checkpoint-v1"
CHECKPOINT_SCHEMA_VERSION = "talk2dino-trust-centrality-harness-checkpoint-v2"

RUN_MODE_FULL = "full_dataset"
RUN_MODE_PARTIAL = "partial"
RUN_STATUS_PARTIAL = "partial"
RUN_STATUS_COMPLETE = "complete"

METRIC_SOURCE_STREAMING = "streaming_mmseg_intersect_and_union"

# ---------------------------------------------------------------------------
# Explicit metric-unit contract (never inferred from numeric magnitude).
#
#   FRACTION_0_TO_1    -- mmseg's natural evaluate() summary; trust
#                         accuracies/fractions/survival rates/support
#                         fractions; the scientific Delta_trust estimand
#                         and its bootstrap CI.
#   PERCENT_0_TO_100   -- full-precision streaming segmentation metrics;
#                         canonical E3/RWR reference metrics; the
#                         normalized natural-evaluate percentage twin.
#   PERCENTAGE_POINTS  -- gain-over-E3; percentage-point display twins of
#                         fraction-unit trust quantities (e.g.
#                         Delta_trust * 100), never a replacement for the
#                         underlying fraction value.
#   FRACTION_DIFFERENCE -- Delta_trust and its CI, in their native
#                         (unscaled) fractional-difference unit.
#   DIMENSIONLESS      -- normalized centrality, Delta_c, normalized
#                         crop-edge distances: no percent/fraction
#                         relationship to declare at all.
#
# A value being < 1.0 does NOT imply it is a fraction -- a legitimate
# percentage metric can itself be under 1.0 -- so unit identity is always
# declared by the field/function that produced the value, never guessed
# from its magnitude.
# ---------------------------------------------------------------------------
UNIT_FRACTION_0_TO_1 = "fraction_0_to_1"
UNIT_PERCENT_0_TO_100 = "percent_0_to_100"
UNIT_PERCENTAGE_POINTS = "percentage_points"
UNIT_FRACTION_DIFFERENCE = "fraction_difference"
UNIT_DIMENSIONLESS = "dimensionless"
UNIT_COUNT = "count"


class TrustCentralityHarnessError(ValueError):
    """Raised on any harness validation or contract violation. Fail closed."""


def _require_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrustCentralityHarnessError(f"{name} must be a positive exact integer")
    return value


def _require_nonneg_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrustCentralityHarnessError(f"{name} must be a non-negative exact integer")
    return value


# ---------------------------------------------------------------------------
# Run-mode decision
# ---------------------------------------------------------------------------


def determine_run_mode(image_limit: Optional[int], dataset_length: int) -> str:
    """``full_dataset`` iff ``image_limit`` is absent or >= dataset_length;
    ``partial`` otherwise. ``dataset_length`` must be determined from the
    real dataset (e.g. ``len(data_loader.dataset)``) before calling this."""
    _require_positive_int(dataset_length, "dataset_length")
    if image_limit is None:
        return RUN_MODE_FULL
    _require_positive_int(image_limit, "image_limit")
    return RUN_MODE_FULL if image_limit >= dataset_length else RUN_MODE_PARTIAL


# ---------------------------------------------------------------------------
# Parity-check scheduling
# ---------------------------------------------------------------------------


def should_stop_after_image(
    *, mode: str, images_processed: int, image_limit: Optional[int], window_floor_satisfied: bool,
) -> bool:
    """Full-dataset runs NEVER stop themselves early -- they let mmseg's own
    evaluation loop run to natural completion and reach ``dataset.evaluate()``.
    Partial (bounded pilot) runs stop as soon as both the requested image
    limit and window floor are satisfied, exactly as the original bounded
    pilots always did."""
    if mode not in (RUN_MODE_FULL, RUN_MODE_PARTIAL):
        raise TrustCentralityHarnessError("mode must be 'full_dataset' or 'partial'")
    _require_nonneg_int(images_processed, "images_processed")
    if not isinstance(window_floor_satisfied, bool):
        raise TrustCentralityHarnessError("window_floor_satisfied must be a bool")
    if mode == RUN_MODE_FULL:
        return False
    if image_limit is None:
        raise TrustCentralityHarnessError("partial mode requires an explicit image_limit")
    _require_positive_int(image_limit, "image_limit")
    return images_processed >= image_limit and window_floor_satisfied


def is_dataset_fully_covered(processed_image_ids: Sequence[str], dataset_length: int) -> bool:
    """True iff every one of ``dataset_length`` images was processed exactly
    once (no missing, no duplicate) -- required before a run may be labeled
    ``complete``."""
    _require_positive_int(dataset_length, "dataset_length")
    if len(set(processed_image_ids)) != len(processed_image_ids):
        return False
    return len(processed_image_ids) == dataset_length


def parity_check_indices(dataset_length: int, interval: int) -> frozenset[int]:
    """Indices (0-based, into the dataset) that must undergo the
    audit-disabled canonical parity check: always the first image, every
    ``interval``-th image, and always the final image."""
    _require_positive_int(dataset_length, "dataset_length")
    _require_positive_int(interval, "parity check interval")
    indices = set(range(0, dataset_length, interval))
    indices.add(0)
    indices.add(dataset_length - 1)
    return frozenset(indices)


# ---------------------------------------------------------------------------
# Observed metrics (from THIS run's own predictions only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedRunMetrics:
    scope: str
    complete: bool
    evaluated_images: int
    unique_images: int
    classes: int
    metric_source: str
    aAcc: Optional[float]
    mIoU: Optional[float]
    mAcc: Optional[float]
    per_class_iou: Optional[tuple[float, ...]]
    per_class_acc: Optional[tuple[float, ...]]
    unavailable_reason: Optional[str]

    def __post_init__(self) -> None:
        if self.scope not in (RUN_MODE_FULL, RUN_MODE_PARTIAL):
            raise TrustCentralityHarnessError("scope must be 'full_dataset' or 'partial'")
        if not isinstance(self.complete, bool):
            raise TrustCentralityHarnessError("complete must be a bool")
        _require_nonneg_int(self.evaluated_images, "evaluated_images")
        _require_nonneg_int(self.unique_images, "unique_images")
        _require_positive_int(self.classes, "classes")
        if self.unique_images != self.evaluated_images:
            raise TrustCentralityHarnessError(
                "unique_images must equal evaluated_images (duplicates must have failed closed earlier)"
            )
        metrics_present = (self.aAcc, self.mIoU, self.mAcc)
        if self.complete:
            if any(m is None for m in metrics_present):
                raise TrustCentralityHarnessError("complete run must have all observed metrics populated")
            if self.unavailable_reason is not None:
                raise TrustCentralityHarnessError("complete run must not carry an unavailable_reason")
        else:
            if any(m is not None for m in metrics_present):
                raise TrustCentralityHarnessError(
                    "incomplete/partial run must not report observed aAcc/mIoU/mAcc"
                )
            if not self.unavailable_reason:
                raise TrustCentralityHarnessError(
                    "an incomplete run must state an explicit unavailable_reason"
                )


def unavailable_observed_metrics(*, scope: str, evaluated_images: int, classes: int, reason: str) -> ObservedRunMetrics:
    return ObservedRunMetrics(
        scope=scope, complete=False, evaluated_images=evaluated_images, unique_images=evaluated_images,
        classes=classes, metric_source=METRIC_SOURCE_STREAMING,
        aAcc=None, mIoU=None, mAcc=None, per_class_iou=None, per_class_acc=None,
        unavailable_reason=reason,
    )


# ---------------------------------------------------------------------------
# Typed fraction->percent normalization (section 2 of the metric-unit
# contract repair). The single place this conversion is ever performed.
# ---------------------------------------------------------------------------


def _is_bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return True
    # numpy boolean scalar type names vary by numpy version ("bool_",
    # "bool8", or -- in newer numpy -- simply "bool", distinct from
    # Python's own bool type object but sharing its __name__).
    return type(value).__name__ in ("bool", "bool_", "bool8")


def normalize_mmseg_fraction_to_percent(value: object, *, name: str) -> float:
    """Input contract: ``UNIT_FRACTION_0_TO_1``. Output contract:
    ``UNIT_PERCENT_0_TO_100``. ``normalized = value * 100.0``.

    Strict by construction: rejects bool (Python or numpy), str, non-finite
    values, and anything outside ``[0, 1]`` -- the last check is what
    prevents double-scaling (an already-percent value >1 is rejected
    outright rather than silently re-scaled) and catches gross unit
    errors, though it cannot by itself distinguish a genuine small
    fraction from a genuine small percentage; that ambiguity is why this
    function must only ever be called on a value whose provenance is
    already known to be ``UNIT_FRACTION_0_TO_1`` (mmseg's natural
    evaluate() summary), never on a value of unknown/inferred unit."""
    if _is_bool_like(value):
        raise TrustCentralityHarnessError(f"{name} must not be a bool (fraction_0_to_1 contract)")
    if isinstance(value, str):
        raise TrustCentralityHarnessError(f"{name} must not be a str (fraction_0_to_1 contract)")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise TrustCentralityHarnessError(f"{name} must be a finite real number (got {value!r})")
    if not math.isfinite(as_float):
        raise TrustCentralityHarnessError(f"{name} must be finite, not NaN/Infinity (got {value!r})")
    if not (0.0 <= as_float <= 1.0):
        raise TrustCentralityHarnessError(
            f"{name}={as_float!r} is outside the fraction_0_to_1 range [0,1]; refusing to guess "
            "units from magnitude -- this value's unit contract must be verified at its source, "
            "not silently re-scaled here"
        )
    return as_float * 100.0


# ---------------------------------------------------------------------------
# Redesigned natural-evaluation-result section (section 3): explicit raw
# (fraction) and normalized (percent) numeric fields, never a JSON string.
# ---------------------------------------------------------------------------

NATURAL_RESULT_SOURCE_PRECISION = "rounded_2dp_percent_expressed_as_fraction"


def build_natural_evaluation_result(
    raw_fraction: Mapping[str, object], *,
    source: str = "dataset.dataset.evaluate() natural path (mmseg CustomDataset.evaluate summary dict)",
) -> dict:
    """Builds the unambiguous, correctly-typed natural-evaluation-result
    section. ``raw_fraction`` must carry aAcc/mIoU/mAcc in
    ``UNIT_FRACTION_0_TO_1`` (mmseg's own convention -- its printed table
    shows percentages, but ``dataset.evaluate()``'s returned dict divides
    back to fractions). Every numeric value in the result is a real
    Python float, never a JSON string."""
    raw = {}
    normalized = {}
    for key in ("aAcc", "mIoU", "mAcc"):
        if key not in raw_fraction:
            raise TrustCentralityHarnessError(f"raw_fraction is missing required key {key!r}")
        percent = normalize_mmseg_fraction_to_percent(raw_fraction[key], name=f"raw_fraction[{key!r}]")
        raw[key] = float(raw_fraction[key])
        normalized[key] = percent
    return {
        "captured": True,
        "source": source,
        "source_precision": NATURAL_RESULT_SOURCE_PRECISION,
        "source_unit": UNIT_FRACTION_0_TO_1,
        "raw_fraction": raw,
        "normalized_unit": UNIT_PERCENT_0_TO_100,
        "normalized_percent": normalized,
        "display_decimals": 2,
    }


def uncaptured_natural_evaluation_result(*, reason: str) -> dict:
    return {"captured": False, "reason": reason}


# ---------------------------------------------------------------------------
# Fixed reconciliation (section 4): the ONLY entry point a caller should
# use to compare mmseg's natural fractions against the full-precision
# streaming percentage -- normalization happens INSIDE this function, so
# a caller can no longer forget it and compare mismatched units.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconciliationResult:
    status: str  # "success" | "failed"
    natural_normalized_percent: Mapping[str, float]
    streaming_percent: Mapping[str, float]
    decimals: int
    failure_category: Optional[str]
    reason: Optional[str]

    def __post_init__(self) -> None:
        if self.status not in ("success", "failed"):
            raise TrustCentralityHarnessError("ReconciliationResult.status must be 'success' or 'failed'")
        if self.status == "failed" and not (self.failure_category and self.reason):
            raise TrustCentralityHarnessError("a failed ReconciliationResult must carry failure_category and reason")
        if self.status == "success" and (self.failure_category is not None or self.reason is not None):
            raise TrustCentralityHarnessError("a successful ReconciliationResult must not carry a failure reason")


def reconcile_natural_percent_vs_streaming(
    raw_fraction: Mapping[str, object], streaming: "ObservedRunMetrics", *, decimals: int = 2,
) -> ReconciliationResult:
    """Normalizes ``raw_fraction`` (``UNIT_FRACTION_0_TO_1``) to percent
    internally, then compares against ``streaming`` (already
    ``UNIT_PERCENT_0_TO_100``) at the same rounding -- the caller never
    handles raw fractions directly, eliminating the unit-mismatch defect
    where a caller compared an unconverted fraction against a percentage
    and reported a false reconciliation failure."""
    if not streaming.complete:
        return ReconciliationResult(
            status="failed", natural_normalized_percent={}, streaming_percent={}, decimals=decimals,
            failure_category="streaming_metrics_incomplete",
            reason="cannot reconcile: streaming metrics are not complete",
        )
    try:
        natural_percent = {
            key: normalize_mmseg_fraction_to_percent(raw_fraction[key], name=f"raw_fraction[{key!r}]")
            for key in ("aAcc", "mIoU", "mAcc") if key in raw_fraction
        }
    except TrustCentralityHarnessError as exc:
        return ReconciliationResult(
            status="failed", natural_normalized_percent={}, streaming_percent={}, decimals=decimals,
            failure_category="natural_fraction_invalid", reason=str(exc),
        )
    streaming_percent = {"aAcc": streaming.aAcc, "mIoU": streaming.mIoU, "mAcc": streaming.mAcc}
    for key in ("aAcc", "mIoU", "mAcc"):
        if key not in natural_percent:
            return ReconciliationResult(
                status="failed", natural_normalized_percent=natural_percent, streaming_percent=streaming_percent,
                decimals=decimals, failure_category="natural_fraction_missing",
                reason=f"natural evaluate result missing {key!r}; cannot reconcile",
            )
        if round(natural_percent[key], decimals) != round(float(streaming_percent[key]), decimals):
            return ReconciliationResult(
                status="failed", natural_normalized_percent=natural_percent, streaming_percent=streaming_percent,
                decimals=decimals, failure_category="percent_mismatch",
                reason=(
                    f"{key}: natural normalized percent {round(natural_percent[key], decimals)} "
                    f"!= streaming percent {round(float(streaming_percent[key]), decimals)} "
                    f"at {decimals} decimal places"
                ),
            )
    return ReconciliationResult(
        status="success", natural_normalized_percent=natural_percent, streaming_percent=streaming_percent,
        decimals=decimals, failure_category=None, reason=None,
    )


# ---------------------------------------------------------------------------
# Streaming segmentation-metric accumulator
# ---------------------------------------------------------------------------


class StreamingSegmentationMetricAccumulator:
    """Exact streaming aAcc/mIoU/mAcc accumulator built on mmseg's own
    ``intersect_and_union`` (never a reimplementation of its semantics),
    fed one already-computed final prediction per image. Never triggers a
    model forward, a graph build, or a CGLS solve -- it only consumes
    tensors the caller already produced elsewhere."""

    def __init__(
        self, *, num_classes: int, ignore_index: int,
        reduce_zero_label: bool = False, label_map: Optional[Mapping[int, int]] = None,
    ) -> None:
        self.num_classes = _require_positive_int(num_classes, "num_classes")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise TrustCentralityHarnessError("ignore_index must be an exact integer")
        self.ignore_index = ignore_index
        self.reduce_zero_label = bool(reduce_zero_label)
        self.label_map: dict[int, int] = dict(label_map) if label_map else {}
        self._total_intersect = torch.zeros(self.num_classes, dtype=torch.float64)
        self._total_union = torch.zeros(self.num_classes, dtype=torch.float64)
        self._total_pred = torch.zeros(self.num_classes, dtype=torch.float64)
        self._total_label = torch.zeros(self.num_classes, dtype=torch.float64)
        self._seen_image_ids: set[str] = set()

    def image_count(self) -> int:
        return len(self._seen_image_ids)

    def unique_image_ids(self) -> frozenset[str]:
        return frozenset(self._seen_image_ids)

    def absorb(self, image_id: str, prediction: torch.Tensor, gt: torch.Tensor) -> None:
        if not isinstance(image_id, str) or not image_id:
            raise TrustCentralityHarnessError("image_id must be a non-empty str")
        if image_id in self._seen_image_ids:
            raise TrustCentralityHarnessError(
                f"duplicate image_id absorbed into the metric accumulator: {image_id!r}"
            )
        if not torch.is_tensor(prediction) or prediction.ndim != 2:
            raise TrustCentralityHarnessError("prediction must be a [H, W] tensor")
        if not torch.is_tensor(gt) or gt.ndim != 2:
            raise TrustCentralityHarnessError("gt must be a [H, W] tensor")
        if tuple(prediction.shape) != tuple(gt.shape):
            raise TrustCentralityHarnessError(
                f"prediction shape {tuple(prediction.shape)} does not match gt shape {tuple(gt.shape)}"
            )
        pred_f = prediction.to(torch.float64)
        gt_f = gt.to(torch.float64)
        if not torch.isfinite(pred_f).all():
            raise TrustCentralityHarnessError("prediction contains non-finite values")
        if not torch.isfinite(gt_f).all():
            raise TrustCentralityHarnessError("gt contains non-finite values")

        from mmseg.core.evaluation.metrics import intersect_and_union

        pred_np = prediction.to(torch.int64).cpu().numpy()
        gt_np = gt.to(torch.int64).cpu().numpy()
        area_intersect, area_union, area_pred, area_label = intersect_and_union(
            pred_np, gt_np, self.num_classes, self.ignore_index,
            label_map=self.label_map, reduce_zero_label=self.reduce_zero_label,
        )
        self._total_intersect += area_intersect.to(torch.float64)
        self._total_union += area_union.to(torch.float64)
        self._total_pred += area_pred.to(torch.float64)
        self._total_label += area_label.to(torch.float64)
        self._seen_image_ids.add(image_id)

    def finalize(self, *, scope: str, complete: bool) -> ObservedRunMetrics:
        count = self.image_count()
        if count == 0 or not complete:
            reason = "no images absorbed" if count == 0 else "run did not reach complete dataset coverage"
            return unavailable_observed_metrics(
                scope=scope, evaluated_images=count, classes=self.num_classes, reason=reason,
            )
        label_total = self._total_intersect.new_tensor(0.0) + self._total_label.sum()
        if label_total.item() <= 0:
            return unavailable_observed_metrics(
                scope=scope, evaluated_images=count, classes=self.num_classes,
                reason="total GT area across all absorbed images was zero (all-ignore dataset?)",
            )
        all_acc = float((self._total_intersect.sum() / label_total).item())
        iou = self._total_intersect / self._total_union
        acc = self._total_intersect / self._total_label
        miou = float(torch.nanmean(iou).item())
        macc = float(torch.nanmean(acc).item())
        return ObservedRunMetrics(
            scope=scope, complete=True, evaluated_images=count, unique_images=count,
            classes=self.num_classes, metric_source=METRIC_SOURCE_STREAMING,
            aAcc=all_acc * 100.0, mIoU=miou * 100.0, mAcc=macc * 100.0,
            per_class_iou=tuple((iou * 100.0).tolist()), per_class_acc=tuple((acc * 100.0).tolist()),
            unavailable_reason=None,
        )

    def state_dict(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "reduce_zero_label": self.reduce_zero_label,
            "label_map": dict(self.label_map),
            "total_intersect": self._total_intersect.tolist(),
            "total_union": self._total_union.tolist(),
            "total_pred": self._total_pred.tolist(),
            "total_label": self._total_label.tolist(),
            "seen_image_ids": sorted(self._seen_image_ids),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "StreamingSegmentationMetricAccumulator":
        acc = cls(
            num_classes=state["num_classes"], ignore_index=state["ignore_index"],
            reduce_zero_label=state["reduce_zero_label"], label_map=state.get("label_map") or {},
        )
        for name, target in (
            ("total_intersect", "_total_intersect"), ("total_union", "_total_union"),
            ("total_pred", "_total_pred"), ("total_label", "_total_label"),
        ):
            values = state[name]
            if len(values) != acc.num_classes:
                raise TrustCentralityHarnessError(f"checkpoint {name} length does not match num_classes")
            setattr(acc, target, torch.tensor(values, dtype=torch.float64))
        ids = state["seen_image_ids"]
        if len(set(ids)) != len(ids):
            raise TrustCentralityHarnessError("checkpoint seen_image_ids contains duplicates")
        acc._seen_image_ids = set(ids)
        return acc


# ---------------------------------------------------------------------------
# Canonical reference metrics (E3 and RWR identities, never conflated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalReferenceMetrics:
    identity_name: str
    source: str  # "e3" | "rwr" -- never mixed
    aAcc: float
    mIoU: float
    mAcc: float
    tolerance_absolute_mIoU: float

    def __post_init__(self) -> None:
        if self.source not in ("e3", "rwr"):
            raise TrustCentralityHarnessError("source must be 'e3' or 'rwr'")
        for name in ("aAcc", "mIoU", "mAcc", "tolerance_absolute_mIoU"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrustCentralityHarnessError(f"{name} must be numeric")


def load_rwr_reference_metrics(*, repo_root: Path) -> CanonicalReferenceMetrics:
    """The ONLY function permitted to read the canonical directed-RWR
    identity's expected_metrics. Field names are prefixed (``rwr_aAcc`` /
    ``rwr_mIoU`` / ``rwr_mAcc``) in the source TOML precisely so they can
    never be silently confused with the E3 identity's unprefixed fields."""
    from src.rwr_reproduction_identity import load_identity as _load_rwr_identity

    identity = _load_rwr_identity(repo_root=repo_root)
    expected = identity["expected_metrics"]
    acceptance = identity["acceptance"]
    return CanonicalReferenceMetrics(
        identity_name=identity["identity_name"], source="rwr",
        aAcc=float(expected["rwr_aAcc"]), mIoU=float(expected["rwr_mIoU"]), mAcc=float(expected["rwr_mAcc"]),
        tolerance_absolute_mIoU=float(acceptance["structured_absolute_mIoU"]),
    )


def load_e3_reference_metrics(*, repo_root: Path) -> CanonicalReferenceMetrics:
    """The ONLY function permitted to read the E3 identity's
    expected_metrics. These are E3 (pre-RWR) numbers -- never the RWR
    numbers -- and must never be reported as this run's RWR reference."""
    from src.e3_evaluation_identity import load_identity as _load_e3_identity

    identity = _load_e3_identity(repo_root=repo_root)
    expected = identity["expected_metrics"]
    tolerances = identity["tolerances"]
    return CanonicalReferenceMetrics(
        identity_name=identity["identity_name"], source="e3",
        aAcc=float(expected["aAcc"]), mIoU=float(expected["mIoU"]), mAcc=float(expected["mAcc"]),
        tolerance_absolute_mIoU=float(tolerances["structured_absolute"]),
    )


@dataclass(frozen=True)
class ComparisonResult:
    metric: str
    observed: float
    reference: float
    delta: float
    tolerance: float
    status: str  # "pass" | "fail"

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise TrustCentralityHarnessError("status must be 'pass' or 'fail'")


def compare_to_reference(observed_mIoU: float, reference: CanonicalReferenceMetrics) -> ComparisonResult:
    if isinstance(observed_mIoU, bool) or not isinstance(observed_mIoU, (int, float)):
        raise TrustCentralityHarnessError("observed_mIoU must be numeric")
    delta = float(observed_mIoU) - reference.mIoU
    status = "pass" if abs(delta) <= reference.tolerance_absolute_mIoU else "fail"
    return ComparisonResult(
        metric="mIoU", observed=float(observed_mIoU), reference=reference.mIoU,
        delta=delta, tolerance=reference.tolerance_absolute_mIoU, status=status,
    )


# ---------------------------------------------------------------------------
# Parity evidence (never described as more coverage than it is)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityEvidence:
    checked_image_count: int
    checked_image_ids: tuple[str, ...]
    total_images: int
    coverage_fraction: float
    all_checked_identical: Optional[bool]
    first_mismatch_image_id: Optional[str]
    final_pass2_vs_pass1_checked_count: int
    final_pass2_vs_pass1_total_count: int
    final_pass2_vs_pass1_all_identical: Optional[bool]

    def __post_init__(self) -> None:
        _require_nonneg_int(self.checked_image_count, "checked_image_count")
        _require_positive_int(self.total_images, "total_images")
        if len(set(self.checked_image_ids)) != len(self.checked_image_ids):
            raise TrustCentralityHarnessError("checked_image_ids must not contain duplicates")
        if len(self.checked_image_ids) != self.checked_image_count:
            raise TrustCentralityHarnessError("checked_image_ids length must equal checked_image_count")
        if not (0.0 <= self.coverage_fraction <= 1.0):
            raise TrustCentralityHarnessError("coverage_fraction must be in [0,1]")
        expected_fraction = self.checked_image_count / self.total_images
        if abs(self.coverage_fraction - expected_fraction) > 1e-9:
            raise TrustCentralityHarnessError("coverage_fraction is inconsistent with checked/total counts")
        if self.checked_image_count == 0 and self.all_checked_identical is not None:
            raise TrustCentralityHarnessError("all_checked_identical must be None when nothing was checked")
        if self.first_mismatch_image_id is not None and self.all_checked_identical is not False:
            raise TrustCentralityHarnessError("a recorded first_mismatch implies all_checked_identical is False")


def build_parity_evidence(
    *, checked_image_ids: Sequence[str], total_images: int, mismatches: Sequence[str],
    final_pass2_checked: int, final_pass2_total: int, final_pass2_all_identical: Optional[bool],
) -> ParityEvidence:
    checked = tuple(checked_image_ids)
    return ParityEvidence(
        checked_image_count=len(checked), checked_image_ids=checked, total_images=total_images,
        coverage_fraction=(len(checked) / total_images) if total_images else 0.0,
        all_checked_identical=(len(mismatches) == 0) if checked else None,
        first_mismatch_image_id=(mismatches[0] if mismatches else None),
        final_pass2_vs_pass1_checked_count=final_pass2_checked,
        final_pass2_vs_pass1_total_count=final_pass2_total,
        final_pass2_vs_pass1_all_identical=final_pass2_all_identical,
    )


# ---------------------------------------------------------------------------
# Checkpoint state (atomic save/resume, with compatibility validation)
# ---------------------------------------------------------------------------


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


@dataclass(frozen=True)
class CheckpointCompatibilityContext:
    """Everything that must match between save-time and resume-time for a
    checkpoint to be considered safe to resume; any mismatch fails closed.

    Deliberately includes every provenance/configuration input that could
    silently change the scientific meaning of the resumed run's per-image
    sufficient statistics or its bootstrap replicates (source code, git
    identity, dataset/class/label contract, canonical identities, and the
    diagnostic bin/bootstrap definitions the trust/centrality accumulator's
    populations are keyed by) -- see Section 4's fail-closed conditions."""

    git_head: str
    git_branch: str
    canonical_config_sha256: str
    identity_name: str
    e3_identity_sha256: str
    rwr_identity_sha256: str
    tracked_diff_sha256: str
    untracked_source_sha256: str
    num_classes: int
    ignore_index: int
    reduce_zero_label: bool
    parity_check_interval: int
    dataset_length: int
    bootstrap_resamples: int
    bootstrap_seed: int
    centrality_bin_edges: tuple[float, ...]
    edge_band_labels: tuple[str, ...]
    diagnostic_schema_version: str

    def __post_init__(self) -> None:
        for name in (
            "git_head", "git_branch", "canonical_config_sha256", "identity_name",
            "e3_identity_sha256", "rwr_identity_sha256", "tracked_diff_sha256",
            "untracked_source_sha256", "diagnostic_schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TrustCentralityHarnessError(f"{name} must be a non-empty str")
        _require_positive_int(self.num_classes, "num_classes")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TrustCentralityHarnessError("ignore_index must be an exact integer")
        if not isinstance(self.reduce_zero_label, bool):
            raise TrustCentralityHarnessError("reduce_zero_label must be a bool")
        _require_positive_int(self.parity_check_interval, "parity_check_interval")
        _require_positive_int(self.dataset_length, "dataset_length")
        _require_positive_int(self.bootstrap_resamples, "bootstrap_resamples")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int):
            raise TrustCentralityHarnessError("bootstrap_seed must be an exact integer")
        if not isinstance(self.centrality_bin_edges, tuple) or not all(
            isinstance(v, float) for v in self.centrality_bin_edges
        ):
            raise TrustCentralityHarnessError("centrality_bin_edges must be a tuple of float")
        if len(self.centrality_bin_edges) < 2:
            raise TrustCentralityHarnessError("centrality_bin_edges must contain at least 2 edges")
        if not isinstance(self.edge_band_labels, tuple) or not all(
            isinstance(v, str) for v in self.edge_band_labels
        ):
            raise TrustCentralityHarnessError("edge_band_labels must be a tuple of str")


@dataclass(frozen=True)
class HarnessCheckpoint:
    schema_version: str
    run_status: str  # "partial" | "complete"
    next_index: int
    processed_image_ids: tuple[str, ...]
    processed_count: int
    accumulator_state: Mapping
    context: CheckpointCompatibilityContext
    diagnostic_settings: Mapping[str, object]
    parity_checked_image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise TrustCentralityHarnessError(
                f"unsupported checkpoint schema_version {self.schema_version!r}; expected {CHECKPOINT_SCHEMA_VERSION!r}"
            )
        if self.run_status not in (RUN_STATUS_PARTIAL, RUN_STATUS_COMPLETE):
            raise TrustCentralityHarnessError("run_status must be 'partial' or 'complete'")
        _require_nonneg_int(self.next_index, "next_index")
        _require_nonneg_int(self.processed_count, "processed_count")
        if len(set(self.processed_image_ids)) != len(self.processed_image_ids):
            raise TrustCentralityHarnessError("processed_image_ids must not contain duplicates")
        if len(self.processed_image_ids) != self.processed_count:
            raise TrustCentralityHarnessError("processed_image_ids length must equal processed_count")
        if self.next_index != self.processed_count:
            # This harness always processes the dataset sequentially from
            # index 0 with no gaps, so next_index and processed_count are
            # definitionally the same quantity; a mismatch means either an
            # image was silently skipped (next_index advanced without a
            # corresponding processed_image_ids entry) or the checkpoint
            # is otherwise corrupted -- fail closed rather than resume
            # from an inconsistent position.
            raise TrustCentralityHarnessError(
                f"next_index ({self.next_index}) != processed_count ({self.processed_count}) -- "
                "checkpoint is inconsistent (a skipped image or corrupted state), refusing to resume"
            )
        if not isinstance(self.context, CheckpointCompatibilityContext):
            raise TrustCentralityHarnessError("context must be a CheckpointCompatibilityContext")


def _checkpoint_to_json(checkpoint: HarnessCheckpoint) -> dict:
    return {
        "schema_version": checkpoint.schema_version,
        "run_status": checkpoint.run_status,
        "next_index": checkpoint.next_index,
        "processed_image_ids": list(checkpoint.processed_image_ids),
        "processed_count": checkpoint.processed_count,
        "accumulator_state": checkpoint.accumulator_state,
        "context": {
            "git_head": checkpoint.context.git_head,
            "git_branch": checkpoint.context.git_branch,
            "canonical_config_sha256": checkpoint.context.canonical_config_sha256,
            "identity_name": checkpoint.context.identity_name,
            "e3_identity_sha256": checkpoint.context.e3_identity_sha256,
            "rwr_identity_sha256": checkpoint.context.rwr_identity_sha256,
            "tracked_diff_sha256": checkpoint.context.tracked_diff_sha256,
            "untracked_source_sha256": checkpoint.context.untracked_source_sha256,
            "num_classes": checkpoint.context.num_classes,
            "ignore_index": checkpoint.context.ignore_index,
            "reduce_zero_label": checkpoint.context.reduce_zero_label,
            "parity_check_interval": checkpoint.context.parity_check_interval,
            "dataset_length": checkpoint.context.dataset_length,
            "bootstrap_resamples": checkpoint.context.bootstrap_resamples,
            "bootstrap_seed": checkpoint.context.bootstrap_seed,
            "centrality_bin_edges": list(checkpoint.context.centrality_bin_edges),
            "edge_band_labels": list(checkpoint.context.edge_band_labels),
            "diagnostic_schema_version": checkpoint.context.diagnostic_schema_version,
        },
        "diagnostic_settings": dict(checkpoint.diagnostic_settings),
        "parity_checked_image_ids": list(checkpoint.parity_checked_image_ids),
    }


def save_checkpoint_atomic(checkpoint: HarnessCheckpoint, path: Path) -> None:
    """Write via a temp file in the same directory, then ``os.replace()`` --
    a reader can never observe a partially-written checkpoint."""
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    payload = json.dumps(_checkpoint_to_json(checkpoint), indent=2, sort_keys=True)
    tmp_path.write_text(payload)
    os.replace(tmp_path, path)


def load_checkpoint(path: Path) -> HarnessCheckpoint:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TrustCentralityHarnessError(f"cannot load checkpoint {path}: {error}") from error
    schema_version = raw.get("schema_version")
    if schema_version == CHECKPOINT_SCHEMA_VERSION_V1:
        raise TrustCentralityHarnessError(
            f"checkpoint {path} uses the older {CHECKPOINT_SCHEMA_VERSION_V1!r} schema, which recorded "
            "only segmentation-metric/stage-image-count/primary-solver state -- it has no trust/"
            "centrality per-image sufficient statistics and cannot be resumed under the current "
            f"{CHECKPOINT_SCHEMA_VERSION!r} contract. Start a fresh run instead of resuming this checkpoint."
        )
    try:
        raw_context = dict(raw["context"])
        for tuple_field in ("centrality_bin_edges", "edge_band_labels"):
            raw_context[tuple_field] = tuple(raw_context[tuple_field])
        context = CheckpointCompatibilityContext(**raw_context)
        checkpoint = HarnessCheckpoint(
            schema_version=raw["schema_version"], run_status=raw["run_status"],
            next_index=raw["next_index"], processed_image_ids=tuple(raw["processed_image_ids"]),
            processed_count=raw["processed_count"], accumulator_state=raw["accumulator_state"],
            context=context, diagnostic_settings=raw["diagnostic_settings"],
            parity_checked_image_ids=tuple(raw["parity_checked_image_ids"]),
        )
    except KeyError as error:
        raise TrustCentralityHarnessError(f"checkpoint {path} is missing required field {error}") from error
    return checkpoint


def validate_checkpoint_compatibility(
    checkpoint: HarnessCheckpoint, expected: CheckpointCompatibilityContext,
) -> None:
    """Fail closed rather than silently resuming an incompatible run."""
    if checkpoint.context != expected:
        mismatched = [
            name for name in expected.__dataclass_fields__
            if getattr(checkpoint.context, name) != getattr(expected, name)
        ]
        raise TrustCentralityHarnessError(
            f"checkpoint is incompatible with the current run context; mismatched fields: {mismatched}"
        )


# The full v2 accumulator_state bag a resumable full-run checkpoint must
# carry -- one key per accumulator/telemetry source needed to reconstruct
# the complete final scientific JSON. A checkpoint missing any of these
# (e.g. an old v1 checkpoint, or one written by a partial/bounded run that
# never reached a state worth resuming as a full run) fails closed rather
# than silently resuming with the missing state treated as empty/zero.
REQUIRED_ACCUMULATOR_STATE_KEYS = (
    "metric_accumulator", "t4_accumulator", "trust_accumulator",
    "primary_solver", "parity_solver", "windows_processed",
)


def validate_accumulator_state_complete(accumulator_state: Mapping) -> None:
    """Fail closed if ``accumulator_state`` is missing any of the
    per-accumulator sub-states required to resume a full run and
    reconstruct the complete final scientific JSON. Does not validate the
    internal shape of each sub-state (each accumulator's own
    ``from_state_dict`` does that); this only guarantees nothing was left
    out of the checkpoint entirely."""
    if not hasattr(accumulator_state, "get"):
        raise TrustCentralityHarnessError("accumulator_state must be a mapping")
    missing = [key for key in REQUIRED_ACCUMULATOR_STATE_KEYS if key not in accumulator_state]
    if missing:
        raise TrustCentralityHarnessError(
            f"checkpoint accumulator_state is missing required state: {sorted(missing)} -- "
            "an incomplete-schema checkpoint must never be silently resumed as if the missing "
            "state were empty/zero"
        )


def validate_checkpoint_internal_consistency(checkpoint: "HarnessCheckpoint") -> None:
    """Cross-field checks that no single accumulator's own validation can
    catch on its own: the metric accumulator's processed-image count must
    match the checkpoint's own processed_count, and the checkpoint's
    windows_processed must match the primary solver telemetry's own
    window_count -- exactly the same independent reconciliation a
    completed run's final report requires (see
    validate_solver_phase_separation), so a checkpoint can never silently
    carry mismatched sub-states through a resume."""
    validate_accumulator_state_complete(checkpoint.accumulator_state)
    state = checkpoint.accumulator_state
    metric_seen = state["metric_accumulator"].get("seen_image_ids")
    if metric_seen is not None and len(metric_seen) != checkpoint.processed_count:
        raise TrustCentralityHarnessError(
            f"checkpoint metric_accumulator has {len(metric_seen)} images but "
            f"processed_count is {checkpoint.processed_count} -- inconsistent checkpoint"
        )
    primary_window_count = state["primary_solver"].get("window_count")
    windows_processed = state["windows_processed"]
    if primary_window_count is not None and primary_window_count != windows_processed:
        raise TrustCentralityHarnessError(
            f"checkpoint primary_solver.window_count ({primary_window_count}) != "
            f"windows_processed ({windows_processed}) -- inconsistent checkpoint"
        )


# ---------------------------------------------------------------------------
# Primary vs. parity solver-phase accounting (section 5)
#
# Both summaries are pure REDUCTIONS over telemetry the real primary solve
# (wc.run_pass_one, via FirstPassImageContext.pass_summary) and the real
# parity-check solve (the original slide_inference path's
# seg_model.rwr_runtime) already produced. Neither ever re-runs CGLS.
# ---------------------------------------------------------------------------

PRIMARY_SOLVER_SUMMARY_FIELDS = (
    "window_count", "total_iterations", "minimum_iterations", "maximum_iterations",
    "total_restarts", "nonzero_restart_windows", "total_residual_replacements",
    "total_fallback_rows", "maximum_scaled_residual",
)


def _field(source, name: str):
    return getattr(source, name) if hasattr(source, name) else source[name]


class PrimarySolverSummaryAccumulator:
    """Aggregates one ``WindowSolverTelemetrySummary`` (or equivalent
    mapping) per image -- the real per-window CGLS telemetry the primary
    (``wc.run_pass_one``) path already computed for every image -- into a
    single full-run summary. Never invokes the solver itself."""

    def __init__(self) -> None:
        self.window_count = 0
        self.total_iterations = 0
        self.minimum_iterations: Optional[int] = None
        self.maximum_iterations = 0
        self.total_restarts = 0
        self.nonzero_restart_windows = 0
        self.total_residual_replacements = 0
        self.total_fallback_rows = 0
        self.maximum_scaled_residual = 0.0
        self._images = 0

    def absorb(self, pass_summary) -> None:
        window_count = _field(pass_summary, "window_count")
        _require_nonneg_int(window_count, "pass_summary.window_count")
        if window_count == 0:
            self._images += 1
            return
        self.window_count += window_count
        self.total_iterations += _field(pass_summary, "total_iterations")
        minimum = _field(pass_summary, "minimum_iterations")
        self.minimum_iterations = minimum if self.minimum_iterations is None else min(self.minimum_iterations, minimum)
        self.maximum_iterations = max(self.maximum_iterations, _field(pass_summary, "maximum_iterations"))
        self.total_restarts += _field(pass_summary, "total_restarts")
        self.nonzero_restart_windows += _field(pass_summary, "nonzero_restart_windows")
        self.total_residual_replacements += _field(pass_summary, "total_residual_replacements")
        self.total_fallback_rows += _field(pass_summary, "total_fallback_rows")
        self.maximum_scaled_residual = max(self.maximum_scaled_residual, _field(pass_summary, "maximum_scaled_residual"))
        self._images += 1

    def as_dict(self) -> dict:
        return {
            "phase": "primary_inference",
            "window_count": self.window_count,
            "total_iterations": self.total_iterations,
            "minimum_iterations": self.minimum_iterations or 0,
            "maximum_iterations": self.maximum_iterations,
            "total_restarts": self.total_restarts,
            "nonzero_restart_windows": self.nonzero_restart_windows,
            "total_residual_replacements": self.total_residual_replacements,
            "total_fallback_rows": self.total_fallback_rows,
            "maximum_scaled_residual": self.maximum_scaled_residual,
            "images_contributing": self._images,
        }

    def state_dict(self) -> dict:
        """Exact checkpoint state -- every field is a running integer/float
        total over already-computed per-image telemetry, so (like
        T4AuditAccumulator) this needs no per-image record to resume: the
        running totals alone are sufficient to continue absorbing the
        remaining images' pass_summary objects identically. Unlike
        as_dict() (a display summary), this keeps minimum_iterations as a
        literal ``None`` (JSON ``null``) when unset, rather than collapsing
        it to 0 -- collapsing it here would make a resumed run's minimum
        wrong if a later-absorbed window's minimum were larger than the
        true (but forgotten) minimum of 0."""
        state = self.as_dict()
        state["minimum_iterations"] = self.minimum_iterations
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "PrimarySolverSummaryAccumulator":
        acc = cls()
        try:
            acc.window_count = _require_nonneg_int(state["window_count"], "window_count")
            acc.total_iterations = _require_nonneg_int(state["total_iterations"], "total_iterations")
            raw_minimum = state["minimum_iterations"]
            acc.minimum_iterations = None if raw_minimum is None else _require_nonneg_int(raw_minimum, "minimum_iterations")
            acc.maximum_iterations = _require_nonneg_int(state["maximum_iterations"], "maximum_iterations")
            acc.total_restarts = _require_nonneg_int(state["total_restarts"], "total_restarts")
            acc.nonzero_restart_windows = _require_nonneg_int(state["nonzero_restart_windows"], "nonzero_restart_windows")
            acc.total_residual_replacements = _require_nonneg_int(state["total_residual_replacements"], "total_residual_replacements")
            acc.total_fallback_rows = _require_nonneg_int(state["total_fallback_rows"], "total_fallback_rows")
            acc.maximum_scaled_residual = float(state["maximum_scaled_residual"])
            acc._images = _require_nonneg_int(state["images_contributing"], "images_contributing")
        except KeyError as error:
            raise TrustCentralityHarnessError(
                f"primary solver checkpoint state is missing required field {error}"
            ) from error
        return acc


def build_parity_solver_summary(rwr_runtime_dict: Mapping[str, object]) -> dict:
    """Labels the pre-existing seg_model.rwr_runtime telemetry (which only
    accumulates during audit-disabled parity-check forwards in this
    harness) explicitly as a parity-phase summary -- never presented as
    the full-run/primary summary."""
    return {"phase": "parity_check", **{k: rwr_runtime_dict[k] for k in sorted(rwr_runtime_dict)}}


def validate_solver_phase_separation(*, primary: Mapping, windows_processed: int) -> None:
    """Fail closed if the primary solver summary's window_count does not
    equal the actual number of primary windows processed -- the exact
    reconciliation the spec requires for a completed run."""
    if primary["window_count"] != windows_processed:
        raise TrustCentralityHarnessError(
            f"primary_solver_summary.window_count ({primary['window_count']}) != "
            f"windows_processed ({windows_processed})"
        )


# ---------------------------------------------------------------------------
# Gain-over-E3 (section 4) and natural-vs-streaming reconciliation (section 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GainOverE3:
    metric: str
    rwr_observed_full_precision: float
    e3_reference_full_precision: float
    absolute_gain: float
    calculation_precision: str

    def __post_init__(self) -> None:
        if self.calculation_precision != "full":
            raise TrustCentralityHarnessError("gain_over_e3 must be computed at full precision")
        expected = self.rwr_observed_full_precision - self.e3_reference_full_precision
        if abs(expected - self.absolute_gain) > 1e-9:
            raise TrustCentralityHarnessError("gain_over_e3.absolute_gain does not reconcile with its own operands")


def compute_gain_over_e3(rwr_observed_mIoU: float, e3_reference: CanonicalReferenceMetrics) -> GainOverE3:
    if e3_reference.source != "e3":
        raise TrustCentralityHarnessError("compute_gain_over_e3 requires an E3 CanonicalReferenceMetrics")
    if isinstance(rwr_observed_mIoU, bool) or not isinstance(rwr_observed_mIoU, (int, float)):
        raise TrustCentralityHarnessError("rwr_observed_mIoU must be numeric")
    rwr_observed_mIoU = float(rwr_observed_mIoU)
    return GainOverE3(
        metric="mIoU", rwr_observed_full_precision=rwr_observed_mIoU,
        e3_reference_full_precision=e3_reference.mIoU,
        absolute_gain=rwr_observed_mIoU - e3_reference.mIoU, calculation_precision="full",
    )


def reconcile_natural_vs_streaming(
    natural_values: Mapping[str, Optional[float]], streaming: ObservedRunMetrics, *, decimals: int = 2,
) -> None:
    """Raises if mmseg's own rounded-to-``decimals``-dp natural evaluate()
    summary disagrees with the streaming accumulator's full-precision
    values at that same rounding -- required before a completed run may be
    accepted."""
    if not streaming.complete:
        raise TrustCentralityHarnessError("cannot reconcile: streaming metrics are not complete")
    for key, streaming_value in (("aAcc", streaming.aAcc), ("mIoU", streaming.mIoU), ("mAcc", streaming.mAcc)):
        natural_value = natural_values.get(key)
        if natural_value is None:
            raise TrustCentralityHarnessError(f"natural evaluate result missing {key}; cannot reconcile")
        if round(float(natural_value), decimals) != round(float(streaming_value), decimals):
            raise TrustCentralityHarnessError(
                f"natural (rounded) {key}={natural_value} disagrees with streaming {key}={streaming_value} "
                f"at {decimals} decimal places"
            )


# ---------------------------------------------------------------------------
# Complete trust/centrality report schema (section 6)
#
# Pure serialization + explicit reconciliation over already-computed
# t4_audit.T4AuditSummary / trust_centrality_diagnostics.TrustCentralityReport
# objects -- no new statistics are computed here beyond simple derived
# fractions/rankings; every underlying count was already produced by the
# (independently tested) T4 audit and trust/centrality diagnostics layers.
# ---------------------------------------------------------------------------

TRUST_CENTRALITY_REPORT_SCHEMA_VERSION = "talk2dino-trust-centrality-full-report-v3"

# Outer (pilot-script) report schema. Bumped to v4 for the metric-unit-
# contract repair: natural_evaluate_result is now a typed raw_fraction/
# normalized_percent structure (never a JSON string), final/complete
# semantics now always agree (final=True implies complete=True), and a
# new finalization_attempted/failure_category pair distinguishes an
# attempted-and-failed finalization from a still-in-progress periodic
# checkpoint dump. The v3 tag remains readable as a legacy/failed
# artifact -- see parse_legacy_v3_report_readonly().
FULL_REPORT_SCHEMA_VERSION_V3 = "talk2dino-trust-centrality-report-v3"
FULL_REPORT_SCHEMA_VERSION_V4 = "talk2dino-trust-centrality-report-v4"


@dataclass(frozen=True)
class LegacyV3ReportSummary:
    schema_version: str
    final: bool
    complete: bool
    reconciliation_error: Optional[str]
    images_processed: int
    windows_processed: int


def parse_legacy_v3_report_readonly(report: Mapping) -> LegacyV3ReportSummary:
    """Read-only summary of a ``talk2dino-trust-centrality-report-v3``
    JSON artifact (the schema this repair's predecessor produced, under
    the buggy fraction-vs-percent unit contract). Never mutates,
    reinterprets, or upgrades its fields to v4 semantics -- a v3 artifact
    that failed under the old contract remains inspectable as exactly
    what it is: a legacy failure, not silently treated as a v4 report."""
    if not hasattr(report, "get"):
        raise TrustCentralityHarnessError("legacy v3 report must be a mapping")
    schema_version = report.get("schema_version")
    if schema_version != FULL_REPORT_SCHEMA_VERSION_V3:
        raise TrustCentralityHarnessError(
            f"not a v3 legacy report: schema_version={schema_version!r}, expected {FULL_REPORT_SCHEMA_VERSION_V3!r}"
        )
    return LegacyV3ReportSummary(
        schema_version=schema_version,
        final=bool(report.get("final")),
        complete=bool(report.get("complete")),
        reconciliation_error=report.get("reconciliation_error"),
        images_processed=int(report.get("images_processed") or 0),
        windows_processed=int(report.get("windows_processed") or 0),
    )
BASE_TRUST_STAGES = ("t2", "t3", "actionable", "t4", "t4_prime")
CENTRALITY_REPORT_STAGES = ("actionable", "t4", "t4_prime")
EDGE_BAND_LABELS_HARNESS = ("<1", "[1,2)", "[2,4)", ">=4")
SHARED_UNARY_GROUPS_HARNESS = ("all", "some", "none", "unavailable")


def _serialize_stage_accuracy(acc, *, stage: str) -> dict:
    if acc.total:
        delta_trust_a = (acc.y_correct - acc.d_correct) / acc.total
        delta_trust_b = (acc.y_correct_d_wrong - acc.y_wrong_d_correct) / acc.total
        if abs(delta_trust_a - delta_trust_b) > 1e-9:
            raise TrustCentralityHarnessError(
                f"stage {stage!r}: Delta_trust reconciliation failed: "
                f"(y_correct-d_correct)/total={delta_trust_a} != (y_cdw-y_wdc)/total={delta_trust_b}"
            )
        delta_trust = delta_trust_a
        third_label_fraction = acc.gt_third_class / acc.total
    else:
        delta_trust = None
        third_label_fraction = None
    return {
        "valid_gt": acc.total, "ignored_gt": acc.ignored,
        "consensus_correct": acc.y_correct, "dissent_correct": acc.d_correct,
        "unary_correct": acc.u_correct, "stitched_valid": acc.g_total, "stitched_correct": acc.g_correct,
        "y_correct_d_wrong": acc.y_correct_d_wrong, "y_wrong_d_correct": acc.y_wrong_d_correct,
        "both_wrong": acc.both_wrong, "gt_third_label": acc.gt_third_class,
        "consensus_accuracy": acc.y_accuracy(), "dissent_accuracy": acc.d_accuracy(),
        "unary_accuracy": acc.u_accuracy(), "stitched_accuracy": acc.g_accuracy(),
        "accuracy_unit": UNIT_FRACTION_0_TO_1,
        "delta_trust": delta_trust, "delta_trust_unit": UNIT_FRACTION_DIFFERENCE,
        "delta_trust_percentage_points": None if delta_trust is None else delta_trust * 100.0,
        "third_label_fraction": third_label_fraction, "third_label_fraction_unit": UNIT_FRACTION_0_TO_1,
    }


def _pp(value: Optional[float]) -> Optional[float]:
    """Percentage-point display twin of a fraction_difference value --
    never a replacement for the underlying fraction, only an additional
    display-convenience field."""
    return None if value is None else value * 100.0


def _serialize_bootstrap_result(result) -> dict:
    observed = result.observed
    return {
        "population": result.population, "status": result.status, "unavailable_reason": result.unavailable_reason,
        "bootstrap_unit": result.settings.unit, "resamples_requested": result.settings.resamples,
        "seed": result.settings.seed, "confidence_level": result.settings.confidence_level,
        "quantile_method": "linear_interpolation_percentile",
        "valid_replicate_count": result.valid_replicate_count, "invalid_replicate_count": result.invalid_replicate_count,
        "invalid_replicate_fraction": result.invalid_replicate_rate,
        # point_estimate/ci_low/ci_high/image_macro_estimate are all
        # Delta_trust quantities: fraction_difference by construction
        # (PairedCounts.delta_trust() is an unscaled ratio difference).
        "point_estimate": result.point_estimate, "ci_low": result.ci_low, "ci_high": result.ci_high,
        "image_macro_estimate": result.image_macro_estimate,
        "delta_trust_estimate_unit": UNIT_FRACTION_DIFFERENCE,
        "point_estimate_percentage_points": _pp(result.point_estimate),
        "ci95_fraction_difference": [result.ci_low, result.ci_high],
        "ci95_percentage_points": [_pp(result.ci_low), _pp(result.ci_high)],
        "image_macro_estimate_percentage_points": _pp(result.image_macro_estimate),
        "contributing_images": result.images_in_population, "zero_target_images": result.images_with_zero_records,
        "observations": observed.count, "ignored_gt": observed.ignored,
        "consensus_correct": observed.consensus_correct, "dissent_correct": observed.dissent_correct,
        "y_correct_d_wrong": observed.y_correct_d_wrong, "y_wrong_d_correct": observed.y_wrong_d_correct,
        "both_correct": observed.both_correct, "both_wrong": observed.both_wrong, "third_label": observed.third_label,
        "accuracy_unit": UNIT_FRACTION_0_TO_1,
        "consensus_accuracy": observed.consensus_accuracy(), "dissent_accuracy": observed.dissent_accuracy(),
        "delta_trust": observed.delta_trust(), "delta_trust_unit": UNIT_FRACTION_DIFFERENCE,
        "delta_trust_percentage_points": _pp(observed.delta_trust()),
        "third_label_fraction": observed.third_label_fraction(), "third_label_fraction_unit": UNIT_FRACTION_0_TO_1,
    }


def _serialize_descriptive_stats(stats) -> dict:
    return {
        "count": stats.count, "mean": stats.mean, "std": stats.std, "median": stats.median,
        "min": stats.minimum, "max": stats.maximum,
        "quantiles": {str(q): v for q, v in stats.quantiles.items()},
    }


def build_full_trust_centrality_section(
    *, t4_summary, trust_report, stage_image_counts: Mapping[str, int], images_processed: int,
) -> dict:
    """Assembles the complete section-6 report from an already-built
    ``t4_audit.T4AuditSummary`` and ``trust_centrality_diagnostics.
    TrustCentralityReport``, plus externally-tracked per-stage
    contributing-image counts (``{"t0","t1","t2","t3","actionable","t4",
    "t4_prime"} -> image count``). Raises on any reconciliation failure."""
    if not t4_summary.gt_evaluated:
        raise TrustCentralityHarnessError("t4_summary must have gt_evaluated=True to build the full report")

    fc = t4_summary.funnel_counts
    if not (fc.t4 <= fc.actionable <= fc.t3 <= fc.t2 <= fc.t1 <= fc.t0):
        raise TrustCentralityHarnessError("funnel subset invariant T4<=actionable<=T3<=T2<=T1<=T0 violated")
    if fc.t4_prime > fc.actionable:
        raise TrustCentralityHarnessError("T4_prime <= actionable invariant violated")

    stage_accuracies = {
        "t2": t4_summary.gt_t2, "t3": t4_summary.gt_t3, "actionable": t4_summary.gt_actionable,
        "t4": t4_summary.gt_t4, "t4_prime": t4_summary.gt_t4_prime,
    }
    stage_counts = {"t2": fc.t2, "t3": fc.t3, "actionable": fc.actionable, "t4": fc.t4, "t4_prime": fc.t4_prime}

    funnel_stages = {}
    for stage, count in stage_counts.items():
        images_here = stage_image_counts.get(stage, 0)
        funnel_stages[stage] = {
            "count": count, "contributing_images": images_here,
            "images_with_zero_records": images_processed - images_here,
            "total_anchor_fraction": (count / fc.t0) if fc.t0 else None,
            "valid_gt": stage_accuracies[stage].total, "ignored_gt": stage_accuracies[stage].ignored,
        }
    funnel_stages["t0"] = {
        "count": fc.t0, "contributing_images": images_processed, "images_with_zero_records": 0,
        "total_anchor_fraction": 1.0 if fc.t0 else None, "valid_gt": None, "ignored_gt": None,
    }
    t1_images = stage_image_counts.get("t1", 0)
    funnel_stages["t1"] = {
        "count": fc.t1, "contributing_images": t1_images, "images_with_zero_records": images_processed - t1_images,
        "total_anchor_fraction": (fc.t1 / fc.t0) if fc.t0 else None, "valid_gt": None, "ignored_gt": None,
    }
    survival = dict(t4_summary.survival_rates)
    survival["t4_prime_given_actionable"] = (fc.t4_prime / fc.actionable) if fc.actionable else None

    trust_by_stage = {stage: _serialize_stage_accuracy(stage_accuracies[stage], stage=stage) for stage in stage_counts}

    bootstrap_section = {name: _serialize_bootstrap_result(result) for name, result in trust_report.bootstrap_results.items()}

    per_class_totals = trust_report.per_class_t4_totals
    class_deltas = [pc.delta_trust() for pc in per_class_totals.values() if pc.delta_trust() is not None]
    class_macro_estimate = (sum(class_deltas) / len(class_deltas)) if class_deltas else None
    total_t4_targets = sum(pc.count for pc in per_class_totals.values())
    ranked_by_count = sorted(per_class_totals.items(), key=lambda kv: kv[1].count, reverse=True)
    with_delta = [(cls, pc) for cls, pc in per_class_totals.items() if pc.delta_trust() is not None]
    top_beneficial = sorted(with_delta, key=lambda kv: kv[1].delta_trust(), reverse=True)[:10]
    top_harmful = sorted(with_delta, key=lambda kv: kv[1].delta_trust())[:10]
    per_class_section = {
        str(cls): {
            "class_id": cls, "observations": pc.count, "contributing_images": None,
            "consensus_correct": pc.consensus_correct, "dissent_correct": pc.dissent_correct,
            "delta_trust": pc.delta_trust(), "y_correct_d_wrong": pc.y_correct_d_wrong,
            "y_wrong_d_correct": pc.y_wrong_d_correct, "both_wrong": pc.both_wrong, "third_label": pc.third_label,
        }
        for cls, pc in per_class_totals.items()
    }
    concentration = {
        f"top_{n}_fraction": ((sum(pc.count for _, pc in ranked_by_count[:n]) / total_t4_targets) if total_t4_targets else None)
        for n in (1, 5, 10, 20)
    }

    centrality_bin_count = len(trust_report.centrality_bin_edges) - 1
    centrality_section = {}
    for stage in CENTRALITY_REPORT_STAGES:
        centrality_section[stage] = {
            "population_count": trust_report.centrality_delta_descriptive[stage].count,
            "delta_c": _serialize_descriptive_stats(trust_report.centrality_delta_descriptive[stage]),
            "source_centrality": _serialize_descriptive_stats(trust_report.centrality_source_descriptive[stage]),
            "jury_centrality": _serialize_descriptive_stats(trust_report.centrality_jury_descriptive[stage]),
            "bin_edges": list(trust_report.centrality_bin_edges),
            "bin_histogram": list(trust_report.centrality_bin_histogram[stage]),
            "strata": {
                stratum: _serialize_bootstrap_result(trust_report.bootstrap_results[f"{stage}_centrality_{stratum}"])
                for stratum in ("negative", "zero", "positive")
                if f"{stage}_centrality_{stratum}" in trust_report.bootstrap_results
            },
            # Fixed-width, predeclared bins over the theoretical Delta_c
            # range -- full paired-outcome trust stratification per bin
            # (not just the raw bin_histogram count above), additive to
            # the sign strata, never used to admit/reject/weight anything.
            "bins": [
                {
                    "bin_index": bin_idx,
                    "bin_lower": trust_report.centrality_bin_edges[bin_idx],
                    "bin_upper": trust_report.centrality_bin_edges[bin_idx + 1],
                    **_serialize_bootstrap_result(trust_report.bootstrap_results[f"{stage}_centrality_bin_{bin_idx}"]),
                }
                for bin_idx in range(centrality_bin_count)
                if f"{stage}_centrality_bin_{bin_idx}" in trust_report.bootstrap_results
            ],
        }

    edge_section = {}
    for stage in CENTRALITY_REPORT_STAGES:
        bands = {
            band: _serialize_bootstrap_result(trust_report.bootstrap_results[f"{stage}_edge_{band}"])
            for band in EDGE_BAND_LABELS_HARNESS
            if f"{stage}_edge_{band}" in trust_report.bootstrap_results
        }
        band_obs = {band: entry["observations"] for band, entry in bands.items()}
        total_band_obs = sum(band_obs.values())
        within_two = band_obs.get("<1", 0) + band_obs.get("[1,2)", 0)
        edge_section[stage] = {
            "edge_distance_patches": _serialize_descriptive_stats(trust_report.edge_patches_descriptive[stage]),
            "unavailable_count": trust_report.edge_unavailable_count.get(stage, 0),
            "bands": bands,
            "within_two_patch_spacings_count": within_two,
            "within_two_patch_spacings_fraction": (within_two / total_band_obs) if total_band_obs else None,
        }

    shared_unary_section = {
        group: _serialize_bootstrap_result(trust_report.bootstrap_results[f"t4_shared_unary_{group}"])
        for group in SHARED_UNARY_GROUPS_HARNESS
        if f"t4_shared_unary_{group}" in trust_report.bootstrap_results
    }

    g_equals_y_count = fc.t4 - trust_report.g_equals_d_count_t4 - trust_report.g_is_third_label_count_t4
    if g_equals_y_count != 0:
        raise TrustCentralityHarnessError(
            f"g_equals_y_count={g_equals_y_count}, but actionable's own definition (g != y) "
            "guarantees this must be exactly 0 for every strict-T4 record"
        )
    stitched_section = {
        "g_equals_d_count": trust_report.g_equals_d_count_t4,
        "g_equals_d_fraction": (trust_report.g_equals_d_count_t4 / fc.t4) if fc.t4 else None,
        "g_is_third_label_count": trust_report.g_is_third_label_count_t4,
        "g_is_third_label_fraction": (trust_report.g_is_third_label_count_t4 / fc.t4) if fc.t4 else None,
        "g_equals_y_count": g_equals_y_count,
    }

    t4_result = trust_report.bootstrap_results["t4"]
    positive_result = trust_report.bootstrap_results.get("t4_centrality_positive")
    mechanism_assessments = {
        "overall_trust_result": trust_report.trust_interpretation_t4,
        "more_central_dissenter_trust_result": None if positive_result is None else _interpret_bootstrap(positive_result),
        "shared_unary_noise_description": (
            "diagnostic only; see shared_unary section for per-group Delta_trust -- "
            "does not by itself confirm or rule out correlated E3 unary noise"
        ),
        "crop_edge_concentration_description": (
            "see crop_edge[stage].within_two_patch_spacings_fraction for concentration near crop boundaries"
        ),
        "empirical_starvation_description": (
            f"strict T4 = {fc.t4} of {fc.t0} total anchors "
            f"({(fc.t4 / fc.t0 * 100.0) if fc.t0 else 0.0:.4f}% of all evaluated anchors); "
            "a rare, narrowly-scoped diagnostic population by construction"
        ),
        "caveats": [
            "T4 identifies operator-attributed diffusion reversals, not guaranteed ground-truth errors.",
            "Target-anchor accuracy/Delta_trust is a diagnostic quantity, not mIoU.",
            "This report alone cannot justify graph repair; it motivates, but does not substitute for, "
            "the upcoming actuator/oracle experiments.",
        ],
    }

    return {
        "schema_version": TRUST_CENTRALITY_REPORT_SCHEMA_VERSION,
        "funnel": {**funnel_stages, "survival_rates": survival},
        "trust_by_stage": trust_by_stage,
        "bootstrap": bootstrap_section,
        "centrality": centrality_section,
        "crop_edge": edge_section,
        "shared_unary": shared_unary_section,
        "per_class_t4": per_class_section,
        "class_macro_estimate_exploratory": class_macro_estimate,
        "class_concentration": {
            "classes_represented": len(per_class_totals),
            "top_classes_by_count": [{"class_id": c, "count": pc.count} for c, pc in ranked_by_count[:20]],
            "top_beneficial_classes": [{"class_id": c, "delta_trust": pc.delta_trust()} for c, pc in top_beneficial],
            "top_harmful_classes": [{"class_id": c, "delta_trust": pc.delta_trust()} for c, pc in top_harmful],
            **concentration,
        },
        "stitched_label_categories": stitched_section,
        "trust_interpretation_t4": trust_report.trust_interpretation_t4,
        "load_bearing_result": _serialize_bootstrap_result(trust_report.load_bearing_result()),
        "mechanism_assessments": mechanism_assessments,
    }


REQUIRED_TRUST_CENTRALITY_SECTION_KEYS = (
    "schema_version", "funnel", "trust_by_stage", "bootstrap", "centrality", "crop_edge",
    "shared_unary", "per_class_t4", "class_macro_estimate_exploratory", "class_concentration",
    "stitched_label_categories", "trust_interpretation_t4", "load_bearing_result", "mechanism_assessments",
)
REQUIRED_FUNNEL_STAGE_KEYS = ("t0", "t1", "t2", "t3", "actionable", "t4", "t4_prime", "survival_rates")
REQUIRED_CENTRALITY_STAGE_KEYS = (
    "population_count", "delta_c", "source_centrality", "jury_centrality",
    "bin_edges", "bin_histogram", "strata", "bins",
)


def validate_trust_centrality_section_complete(section: Mapping) -> None:
    """Raises unless ``section`` (the dict returned by
    ``build_full_trust_centrality_section``) contains every required
    top-level and nested key. A report whose outer wrapper marks
    ``final: True`` must call this (on a status=="available" section)
    before being accepted as complete -- see the pilot script's
    write_report(). Checks presence/shape only, not numeric values (those
    are covered by build_full_trust_centrality_section's own internal
    reconciliation checks and by the dedicated numeric regression tests)."""
    if not hasattr(section, "get"):
        raise TrustCentralityHarnessError("trust_centrality section must be a mapping")
    missing = [k for k in REQUIRED_TRUST_CENTRALITY_SECTION_KEYS if k not in section]
    if missing:
        raise TrustCentralityHarnessError(f"trust_centrality section missing required keys: {sorted(missing)}")

    funnel_missing = [k for k in REQUIRED_FUNNEL_STAGE_KEYS if k not in section["funnel"]]
    if funnel_missing:
        raise TrustCentralityHarnessError(f"funnel section missing required keys: {sorted(funnel_missing)}")

    for stage in CENTRALITY_REPORT_STAGES:
        if stage not in section["centrality"]:
            raise TrustCentralityHarnessError(f"centrality section missing stage {stage!r}")
        stage_missing = [k for k in REQUIRED_CENTRALITY_STAGE_KEYS if k not in section["centrality"][stage]]
        if stage_missing:
            raise TrustCentralityHarnessError(f"centrality[{stage!r}] missing required keys: {sorted(stage_missing)}")
        bin_edges = section["centrality"][stage]["bin_edges"]
        expected_bin_count = len(bin_edges) - 1
        actual_bins = section["centrality"][stage]["bins"]
        if len(actual_bins) != expected_bin_count:
            raise TrustCentralityHarnessError(
                f"centrality[{stage!r}].bins has {len(actual_bins)} entries, "
                f"expected exactly {expected_bin_count} (derived from bin_edges, never hardcoded)"
            )
        for stratum in ("negative", "zero", "positive"):
            if stratum not in section["centrality"][stage]["strata"]:
                raise TrustCentralityHarnessError(f"centrality[{stage!r}].strata missing stratum {stratum!r}")

    for stage in CENTRALITY_REPORT_STAGES:
        if stage not in section["crop_edge"]:
            raise TrustCentralityHarnessError(f"crop_edge section missing stage {stage!r}")
    for group in SHARED_UNARY_GROUPS_HARNESS:
        if group not in section["shared_unary"]:
            raise TrustCentralityHarnessError(f"shared_unary section missing group {group!r}")


def _require_percent_range(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrustCentralityHarnessError(f"{name} must be numeric (percent_0_to_100 contract)")
    if not math.isfinite(value) or not (0.0 <= float(value) <= 100.0):
        raise TrustCentralityHarnessError(f"{name}={value!r} is outside the percent_0_to_100 range [0,100]")


def validate_report_v4_unit_contract(report: Mapping) -> None:
    """Exact schema/unit validation for a v4 outer report's top-level
    percent- and fraction-labeled fields (section 7 of the metric-unit-
    contract repair): confirms declared units match declared value
    ranges and that every checked numeric field is a real number, never
    a JSON string. Does not re-validate the trust_centrality subsection's
    own internal fraction-unit fields (see validate_trust_centrality_
    section_complete for structural completeness of that subsection)."""
    if not hasattr(report, "get"):
        raise TrustCentralityHarnessError("report must be a mapping")
    if report.get("schema_version") != FULL_REPORT_SCHEMA_VERSION_V4:
        raise TrustCentralityHarnessError(
            f"validate_report_v4_unit_contract requires schema_version={FULL_REPORT_SCHEMA_VERSION_V4!r}, "
            f"got {report.get('schema_version')!r}"
        )

    observed = report.get("observed_metrics") or {}
    if observed.get("complete"):
        if observed.get("unit") != UNIT_PERCENT_0_TO_100:
            raise TrustCentralityHarnessError(f"observed_metrics.unit must be {UNIT_PERCENT_0_TO_100!r}")
        for key in ("aAcc", "mIoU", "mAcc"):
            _require_percent_range(observed.get(key), f"observed_metrics.{key}")

    natural = report.get("natural_evaluate_result") or {}
    if natural.get("captured"):
        if natural.get("source_unit") != UNIT_FRACTION_0_TO_1:
            raise TrustCentralityHarnessError(f"natural_evaluate_result.source_unit must be {UNIT_FRACTION_0_TO_1!r}")
        if natural.get("normalized_unit") != UNIT_PERCENT_0_TO_100:
            raise TrustCentralityHarnessError(f"natural_evaluate_result.normalized_unit must be {UNIT_PERCENT_0_TO_100!r}")
        for key in ("aAcc", "mIoU", "mAcc"):
            raw = natural.get("raw_fraction", {}).get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TrustCentralityHarnessError(f"natural_evaluate_result.raw_fraction.{key} must be numeric")
            if not math.isfinite(raw) or not (0.0 <= float(raw) <= 1.0):
                raise TrustCentralityHarnessError(f"natural_evaluate_result.raw_fraction.{key}={raw!r} outside [0,1]")
            _require_percent_range(natural.get("normalized_percent", {}).get(key), f"natural_evaluate_result.normalized_percent.{key}")

    reference = report.get("reference_metrics") or {}
    for section_name in ("rwr", "e3_for_reference_only"):
        block = reference.get(section_name)
        if not block:
            continue
        if block.get("unit") != UNIT_PERCENT_0_TO_100:
            raise TrustCentralityHarnessError(f"reference_metrics.{section_name}.unit must be {UNIT_PERCENT_0_TO_100!r}")
        for key in ("aAcc", "mIoU", "mAcc"):
            _require_percent_range(block.get(key), f"reference_metrics.{section_name}.{key}")

    gain = reference.get("gain_over_e3")
    if gain:
        if gain.get("unit") != UNIT_PERCENTAGE_POINTS:
            raise TrustCentralityHarnessError(f"reference_metrics.gain_over_e3.unit must be {UNIT_PERCENTAGE_POINTS!r}")
        if isinstance(gain.get("absolute_gain"), bool) or not isinstance(gain.get("absolute_gain"), (int, float)):
            raise TrustCentralityHarnessError("reference_metrics.gain_over_e3.absolute_gain must be numeric")


def _interpret_bootstrap(result) -> str:
    if result.status != "available" or result.point_estimate is None:
        return "UNAVAILABLE"
    if result.point_estimate <= 0.0:
        return "CONTRADICTED"
    if result.ci_low is not None and result.ci_low > 0.0:
        return "SUPPORTED"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Deterministic scientific projection (section 3): an uninterrupted run and
# an interrupted/checkpointed/resumed run must produce byte-identical
# reports after removing only explicitly volatile fields.
# ---------------------------------------------------------------------------

DEFAULT_VOLATILE_REPORT_FIELD_PATHS: tuple[tuple[str, ...], ...] = (
    ("provenance", "start_time_unix"),
    ("provenance", "end_time_unix"),
    ("provenance", "elapsed_seconds"),
    ("provenance", "gpu_model"),
    ("provenance", "python_version"),
    ("provenance", "torch_version"),
    ("provenance", "cuda_version"),
    ("peak_gpu_bytes",),
    ("median_diagnostic_seconds_per_image",),
    ("checkpoint_history",),
)


def _delete_path(node: dict, path: tuple[str, ...]) -> None:
    for key in path[:-1]:
        if not isinstance(node, dict) or key not in node:
            return
        node = node[key]
    if isinstance(node, dict):
        node.pop(path[-1], None)


def canonical_scientific_projection(
    report: Mapping, *, extra_volatile_field_paths: Sequence[tuple[str, ...]] = (),
) -> dict:
    """Returns a deep copy of ``report`` with only explicitly volatile
    fields removed (timestamps, elapsed/runtime measurements, machine/GPU
    identity and memory measurements, checkpoint/resume history) -- every
    remaining scientific field must be identical between an uninterrupted
    run and a checkpoint-interrupted-and-resumed run of the same
    configuration. Never removes a field on any other basis."""
    if not hasattr(report, "get"):
        raise TrustCentralityHarnessError("report must be a mapping")
    projection = copy.deepcopy(dict(report))
    for path in tuple(DEFAULT_VOLATILE_REPORT_FIELD_PATHS) + tuple(extra_volatile_field_paths):
        _delete_path(projection, path)
    return projection


def canonical_scientific_projection_bytes(report: Mapping, **kwargs) -> bytes:
    """Deterministic JSON encoding of :func:`canonical_scientific_projection`
    -- sorted keys, so encoding order never depends on dict insertion
    order (image/class/stratum ordering determinism is the caller's
    responsibility, via processing images and building dicts in canonical
    dataset order, as this harness's accumulators already do)."""
    projection = canonical_scientific_projection(report, **kwargs)
    return json.dumps(projection, sort_keys=True, default=str).encode("utf-8")


def sha256_of_scientific_projection(report: Mapping, **kwargs) -> str:
    return hashlib.sha256(canonical_scientific_projection_bytes(report, **kwargs)).hexdigest()


def assert_scientific_projections_identical(uninterrupted: Mapping, resumed: Mapping) -> None:
    """Raises with a pointer to the first differing top-level section if
    the two reports' canonical scientific projections are not
    byte-identical; otherwise returns silently."""
    left = canonical_scientific_projection_bytes(uninterrupted)
    right = canonical_scientific_projection_bytes(resumed)
    if left == right:
        return
    left_proj = canonical_scientific_projection(uninterrupted)
    right_proj = canonical_scientific_projection(resumed)
    differing_keys = sorted(
        key for key in (set(left_proj) | set(right_proj))
        if left_proj.get(key) != right_proj.get(key)
    )
    raise TrustCentralityHarnessError(
        "uninterrupted and resumed scientific projections are not byte-identical; "
        f"differing top-level sections: {differing_keys}"
    )


# ---------------------------------------------------------------------------
# Full-run failure behavior (section 5): a requested full-dataset run must
# never terminate successfully with an incomplete report.
# ---------------------------------------------------------------------------


class TrustCentralityFullRunIncompleteError(TrustCentralityHarnessError):
    """Raised when a REQUESTED full-dataset run's final report cannot be
    certified complete. The caller (the GPU-driving pilot script) must
    catch this, write an atomic failure report recording the reason, and
    exit nonzero -- a full run must never terminate successfully with an
    incomplete report. Never raised for a partial/bounded run, which is
    permitted to be final=True/complete=False with an explicit reason."""


def require_full_run_complete(report: Mapping) -> None:
    """Gates on ``finalization_attempted`` (was THIS write_report() call
    the caller's attempt to produce the final artifact?), never on
    ``final`` -- ``final`` now means "genuinely complete" (see section 5
    of the metric-unit-contract repair: final=True is reserved
    exclusively for complete=True reports), so a failed finalization
    attempt correctly has final=False and would otherwise be invisible to
    this gate if it were still keyed off ``final``. A periodic/
    intermediate checkpoint dump mid-run (finalization_attempted=False)
    is never held to this bar, regardless of its own complete value."""
    if not hasattr(report, "get"):
        raise TrustCentralityHarnessError("report must be a mapping")
    if report.get("run_mode") != RUN_MODE_FULL or not report.get("finalization_attempted"):
        return  # only an attempted finalization of a full-dataset run is held to this bar
    if not report.get("complete"):
        raise TrustCentralityFullRunIncompleteError(
            "requested full-dataset run attempted finalization but complete=False "
            f"(failure_category={report.get('failure_category')!r}): "
            f"reconciliation_error={report.get('reconciliation_error')!r}"
        )
    if report.get("final") is not True:
        raise TrustCentralityFullRunIncompleteError(
            "requested full-dataset run has complete=True but final is not True -- "
            "final and complete must agree for a genuinely successful report"
        )
    trust_centrality = report.get("trust_centrality")
    if not hasattr(trust_centrality, "get") or trust_centrality.get("status") != "available":
        raise TrustCentralityFullRunIncompleteError(
            "requested full-dataset run marked complete=True but its trust_centrality "
            f"section is not available: {trust_centrality!r}"
        )
    try:
        validate_trust_centrality_section_complete(trust_centrality)
    except TrustCentralityHarnessError as exc:
        raise TrustCentralityFullRunIncompleteError(
            f"requested full-dataset run marked complete=True but its trust_centrality "
            f"section fails completeness validation: {exc}"
        ) from exc
    observed = report.get("observed_metrics")
    if not hasattr(observed, "get") or observed.get("mIoU") is None:
        raise TrustCentralityFullRunIncompleteError(
            "requested full-dataset run marked complete=True but observed_metrics.mIoU is missing"
        )
    natural = report.get("natural_evaluate_result")
    if not hasattr(natural, "get") or not natural.get("captured"):
        raise TrustCentralityFullRunIncompleteError(
            "requested full-dataset run marked complete=True but natural_evaluate_result was not captured"
        )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION_V1",
    "DEFAULT_VOLATILE_REPORT_FIELD_PATHS",
    "FULL_REPORT_SCHEMA_VERSION_V3",
    "FULL_REPORT_SCHEMA_VERSION_V4",
    "METRIC_SOURCE_STREAMING",
    "NATURAL_RESULT_SOURCE_PRECISION",
    "PRIMARY_SOLVER_SUMMARY_FIELDS",
    "REQUIRED_ACCUMULATOR_STATE_KEYS",
    "REQUIRED_TRUST_CENTRALITY_SECTION_KEYS",
    "TRUST_CENTRALITY_REPORT_SCHEMA_VERSION",
    "RUN_MODE_FULL",
    "RUN_MODE_PARTIAL",
    "RUN_STATUS_COMPLETE",
    "RUN_STATUS_PARTIAL",
    "UNIT_COUNT",
    "UNIT_DIMENSIONLESS",
    "UNIT_FRACTION_0_TO_1",
    "UNIT_FRACTION_DIFFERENCE",
    "UNIT_PERCENTAGE_POINTS",
    "UNIT_PERCENT_0_TO_100",
    "CanonicalReferenceMetrics",
    "CheckpointCompatibilityContext",
    "ComparisonResult",
    "GainOverE3",
    "HarnessCheckpoint",
    "LegacyV3ReportSummary",
    "ObservedRunMetrics",
    "ParityEvidence",
    "PrimarySolverSummaryAccumulator",
    "ReconciliationResult",
    "StreamingSegmentationMetricAccumulator",
    "TrustCentralityFullRunIncompleteError",
    "TrustCentralityHarnessError",
    "assert_scientific_projections_identical",
    "build_full_trust_centrality_section",
    "build_natural_evaluation_result",
    "build_parity_evidence",
    "build_parity_solver_summary",
    "canonical_scientific_projection",
    "canonical_scientific_projection_bytes",
    "compare_to_reference",
    "compute_gain_over_e3",
    "determine_run_mode",
    "git_head",
    "load_checkpoint",
    "load_e3_reference_metrics",
    "load_rwr_reference_metrics",
    "is_dataset_fully_covered",
    "normalize_mmseg_fraction_to_percent",
    "parity_check_indices",
    "parse_legacy_v3_report_readonly",
    "reconcile_natural_percent_vs_streaming",
    "reconcile_natural_vs_streaming",
    "require_full_run_complete",
    "save_checkpoint_atomic",
    "sha256_of_scientific_projection",
    "should_stop_after_image",
    "uncaptured_natural_evaluation_result",
    "unavailable_observed_metrics",
    "validate_accumulator_state_complete",
    "validate_checkpoint_compatibility",
    "validate_checkpoint_internal_consistency",
    "validate_report_v4_unit_contract",
    "validate_solver_phase_separation",
    "validate_trust_centrality_section_complete",
]
