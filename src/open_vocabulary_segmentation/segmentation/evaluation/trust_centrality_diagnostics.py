"""Opt-in, read-only trust and centrality diagnostics over the committed
T4 leave-one-out consensus audit.

This module is a **diagnostic side channel** on top of an already-frozen
:class:`~segmentation.evaluation.t4_audit.T4ImageSignal` and its GT
evaluation. It never rebuilds windows, reruns the backbone, rebuilds the
directed top-k graph, reruns CGLS, or changes stitching/pass-2 replay; with
diagnostics enabled or disabled, the final segmentation output is
byte-identical. It never mutates a ``ConsensusObservation`` (frozen by
construction) and never feeds GT into the T4 signal builder -- the only GT
primitive it calls is the already-audited, GT-isolated
``t4_audit.sample_gt_nearest_neighbor``, exactly the same nearest-neighbor
sampler the committed evaluator itself uses. It answers two scientific
questions without ever changing T0-T4/T4' membership, stitching, cached
scores, or the canonical E3/RWR identities:

1. **Trust alignment** -- on operator-attributed diffusion reversals
   (strict T4), is the other-window consensus label ``y`` more often
   correct than the source window's own post-RWR dissent label ``d``?
2. **Centrality / mechanism** -- is the dissenting source window
   systematically more central (better contextualized) than the
   unanimous agreeing jury, which would weaken or invert the
   "graph-pathology" reading of T4?

Primary trust quantity
-----------------------
For a non-ignored evaluated observation::

    v = 1[y == gt] - 1[d == gt]

so ``v = +1`` means consensus right / dissent wrong, ``v = -1`` means
consensus wrong / dissent right, and ``v = 0`` covers both-correct and
both-wrong (reported separately, never assumed to be zero-count -- see
:class:`PairedCounts`). ``Delta_trust = mean(v) = accuracy(y) -
accuracy(d)``. The primary scientific gate is **strict T4**; T2/T3/
actionable/T4' get the same *descriptive* quantities without a bootstrap
confidence interval or the pass/fail interpretation label (see
:func:`interpret_trust`).

Centrality sign convention
---------------------------
``centrality = 1 - source_normalized_center_distance`` (0 at the crop
corner, 1 at the crop center; range enforced from the already-committed,
already-audited ``ConsensusObservation.source_normalized_center_distance``
/ ``agreeing_window_center_distances`` fields -- no second coordinate
convention is introduced here)::

    c_source     = 1 - source_normalized_center_distance
    c_jury_mean  = mean(1 - d for d in agreeing_window_center_distances)
    Delta_c      = c_source - c_jury_mean

``Delta_c > 0``: the dissenting source window is *more* central than its
unanimous agreeing jury. ``Delta_c < 0``: the source is *less* central.
``Delta_c == 0`` is reported and stratified exactly, with no epsilon band
around zero for membership purposes (display rounding, if any, never
feeds back into the comparison).

Crop-edge bands (fixed, declared up front, diagnostic only)
-------------------------------------------------------------
Edge distance is converted from raw local pixel coordinates
(``ConsensusObservation.local_anchor`` / ``window_extent``, both already
committed and exact) into patch-spacing units using the image's
``FirstPassImageContext.common_patch_grid_shape`` (``None`` for images
with heterogeneous per-window grid shapes, in which case patch-spacing
conversion is reported as unavailable for that image rather than silently
approximated). Bands: ``EDGE_BAND_LABELS`` below, half-open
(``[lo, hi)``), with the last band unbounded above.

Shared-unary stratification
------------------------------
Uses the already-committed ``other_window_unary_labels`` /
``unary_other_agree_fraction`` fields (never the source's own unary,
which is trivially always ``y``-agreeing at strict T4 by definition) to
split strict T4 into ``all`` / ``some`` / ``none`` / ``unavailable``
(the last only if a record has zero sampled other-window unaries, which
cannot happen for a real T2+ record but is handled defensively rather
than assumed impossible).

Bootstrap
---------
A deterministic, image-level clustered percentile bootstrap
(:data:`CANONICAL_BOOTSTRAP_SETTINGS`): resamples whole images (with
replacement) from the *complete* evaluated image list -- including
images that contributed zero records to the population being
bootstrapped -- never resamples individual records within an image. All
registered populations for one run share a single resampling stream per
replicate (paired resampling), both for statistical coherence across
strata and for performance. A replicate with zero pooled records for a
population is marked invalid and excluded from that population's
quantiles (never coerced to zero); the invalid-replicate count/rate is
always reported. See :func:`run_clustered_bootstrap`.

Memory
------
Nothing here retains raw, dataset-wide ``ConsensusObservation`` records
or GT tensors across images. Per image, only compact, already-audited
sufficient statistics (:class:`PairedCounts`, small scalar tuples for
descriptive quantiles) are retained; the accumulator's per-image state is
bounded by (number of processed images) x (number of registered
populations), not by total patch count. The bounded diagnostic pilots
this module is designed for (<=100 images, never the full 5000-image
validation) keep this trivially small; see module-level docstring notes
in the pilot script for the explicit cap.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import torch

from .sliding_window_geometry import SpatialSize
from .t4_audit import (
    ConsensusObservation,
    T4ImageGTEvaluation,
    T4ImageSignal,
    T4StageAccuracy,
    evaluate_t4_signal_against_gt,
    sample_gt_nearest_neighbor,
)


class TrustCentralityDiagnosticsError(ValueError):
    """Raised on any trust/centrality diagnostics validation or contract
    violation. Fail closed."""


def _require_type(value, expected_type, name: str):
    if isinstance(value, bool) and expected_type is not bool:
        raise TrustCentralityDiagnosticsError(f"{name} must be {expected_type.__name__}, got bool")
    if not isinstance(value, expected_type):
        raise TrustCentralityDiagnosticsError(f"{name} must be {expected_type.__name__}")
    return value


def _require_nonneg_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrustCentralityDiagnosticsError(f"{name} must be a non-negative exact integer")
    return value


# ---------------------------------------------------------------------------
# Canonical, declared-up-front diagnostic settings (never tuned on GT)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapSettings:
    """Every field here is a scientific decision, declared once, serialized
    into every diagnostic result -- never a hidden default."""

    unit: str = "image"
    resamples: int = 10_000
    confidence_level: float = 0.95
    seed: int = 20260818

    def __post_init__(self) -> None:
        _require_type(self.unit, str, "unit")
        if self.unit != "image":
            raise TrustCentralityDiagnosticsError("bootstrap unit must be 'image' (records are never resampled independently)")
        if isinstance(self.resamples, bool) or not isinstance(self.resamples, int) or self.resamples < 1:
            raise TrustCentralityDiagnosticsError("resamples must be a positive exact integer")
        if isinstance(self.confidence_level, bool) or not isinstance(self.confidence_level, float):
            raise TrustCentralityDiagnosticsError("confidence_level must be a float")
        if not (0.0 < self.confidence_level < 1.0):
            raise TrustCentralityDiagnosticsError("confidence_level must lie in (0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TrustCentralityDiagnosticsError("seed must be an exact integer")


CANONICAL_BOOTSTRAP_SETTINGS = BootstrapSettings()

# Below this many valid replicates for a given population, the CI is
# reported as unavailable rather than computed from a thin sample. Declared
# up front; never tuned on observed results.
MINIMUM_VALID_BOOTSTRAP_REPLICATES = 100

EDGE_BAND_LABELS = ("<1", "[1,2)", "[2,4)", ">=4")
CENTRALITY_STRATA = ("negative", "zero", "positive")
SHARED_UNARY_GROUPS = ("all", "some", "none", "unavailable")
EXTENDED_STAGES = ("actionable", "t4", "t4_prime")  # centrality + edge-band scope
BASE_STAGES = ("t2", "t3", "actionable", "t4", "t4_prime")

QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Fixed-width bins over the theoretical Delta_c range [-1, 1]. Declared up
# front, serialized in the result, descriptive only -- never used to admit,
# reject, or weight anything.
DELTA_C_RANGE = (-1.0, 1.0)
DELTA_C_BIN_COUNT = 10
DELTA_C_BIN_EDGES = tuple(
    DELTA_C_RANGE[0] + i * (DELTA_C_RANGE[1] - DELTA_C_RANGE[0]) / DELTA_C_BIN_COUNT
    for i in range(DELTA_C_BIN_COUNT + 1)
)


def delta_c_bin_index(delta_c: float) -> int:
    if not math.isfinite(delta_c) or not (DELTA_C_RANGE[0] <= delta_c <= DELTA_C_RANGE[1]):
        raise TrustCentralityDiagnosticsError(f"delta_c out of the theoretical range {DELTA_C_RANGE}: {delta_c}")
    # Looked up directly against the precomputed edges (never re-derived
    # via interval arithmetic on delta_c itself) so that a value exactly
    # equal to a declared edge always lands in the bin that edge is the
    # *lower*, closed bound of -- e.g. delta_c_bin_index(-0.8) == 1, not 0
    # -- regardless of float64 rounding in how that edge was computed.
    # bisect_right places delta_c after any edge equal to it, so
    # bisect_right(...) - 1 is exactly that bin's index; the final bin's
    # upper edge (1.0) is the one case that is closed on both ends, so the
    # result is clamped to the last bin rather than an out-of-range 10th.
    idx = bisect.bisect_right(DELTA_C_BIN_EDGES, delta_c) - 1
    return min(max(idx, 0), DELTA_C_BIN_COUNT - 1)


# ---------------------------------------------------------------------------
# Paired-outcome sufficient statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedCounts:
    """Sufficient statistics for one (population, GT) pairing. Exact
    integer counters only; never coerced from bool/float."""

    count: int = 0
    ignored: int = 0
    consensus_correct: int = 0
    dissent_correct: int = 0
    y_correct_d_wrong: int = 0
    y_wrong_d_correct: int = 0
    both_correct: int = 0
    both_wrong: int = 0
    third_label: int = 0

    _INT_FIELDS = (
        "count", "ignored", "consensus_correct", "dissent_correct",
        "y_correct_d_wrong", "y_wrong_d_correct", "both_correct", "both_wrong",
        "third_label",
    )

    def __post_init__(self) -> None:
        for name in self._INT_FIELDS:
            _require_nonneg_int(getattr(self, name), name)
        if self.consensus_correct > self.count:
            raise TrustCentralityDiagnosticsError("consensus_correct cannot exceed count")
        if self.dissent_correct > self.count:
            raise TrustCentralityDiagnosticsError("dissent_correct cannot exceed count")
        if self.third_label > self.count:
            raise TrustCentralityDiagnosticsError("third_label cannot exceed count")
        partition = self.y_correct_d_wrong + self.y_wrong_d_correct + self.both_correct + self.both_wrong
        if partition != self.count:
            raise TrustCentralityDiagnosticsError(
                "paired-outcome categories (y_correct_d_wrong + y_wrong_d_correct + "
                f"both_correct + both_wrong = {partition}) must exactly partition count ({self.count})"
            )
        if self.consensus_correct != self.y_correct_d_wrong + self.both_correct:
            raise TrustCentralityDiagnosticsError(
                "consensus_correct must equal y_correct_d_wrong + both_correct"
            )
        if self.dissent_correct != self.y_wrong_d_correct + self.both_correct:
            raise TrustCentralityDiagnosticsError(
                "dissent_correct must equal y_wrong_d_correct + both_correct"
            )

    def __add__(self, other: "PairedCounts") -> "PairedCounts":
        if not isinstance(other, PairedCounts):
            return NotImplemented
        return PairedCounts(**{name: getattr(self, name) + getattr(other, name) for name in self._INT_FIELDS})

    def consensus_accuracy(self) -> Optional[float]:
        return (self.consensus_correct / self.count) if self.count else None

    def dissent_accuracy(self) -> Optional[float]:
        return (self.dissent_correct / self.count) if self.count else None

    def delta_trust(self) -> Optional[float]:
        return ((self.consensus_correct - self.dissent_correct) / self.count) if self.count else None

    def third_label_fraction(self) -> Optional[float]:
        return (self.third_label / self.count) if self.count else None

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self._INT_FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping) -> "PairedCounts":
        try:
            return cls(**{name: data[name] for name in cls._INT_FIELDS})
        except KeyError as error:
            raise TrustCentralityDiagnosticsError(f"PairedCounts checkpoint state is missing field {error}") from error


EMPTY_PAIRED_COUNTS = PairedCounts()


def _accumulate_paired_counts(
    observations: Sequence[ConsensusObservation], gt: torch.Tensor, image_size: SpatialSize, ignore_label: int,
) -> PairedCounts:
    """The one place this module samples GT for paired-outcome bookkeeping.
    Mirrors t4_audit._accumulate_stage's exact sampling call (same public,
    already-audited ``sample_gt_nearest_neighbor``), extended to also
    partition both-correct/both-wrong explicitly rather than assuming
    both-correct is impossible."""
    count = ignored = cc = dc = ycdw = ywdc = bc = bw = tl = 0
    for obs in observations:
        label = sample_gt_nearest_neighbor(gt, obs.global_anchor[0], obs.global_anchor[1], image_size)
        if label == ignore_label:
            ignored += 1
            continue
        count += 1
        y_ok = label == obs.consensus_label
        d_ok = label == obs.source_dissent_label
        cc += int(y_ok)
        dc += int(d_ok)
        if y_ok and not d_ok:
            ycdw += 1
        elif d_ok and not y_ok:
            ywdc += 1
        elif y_ok and d_ok:
            bc += 1
        else:
            bw += 1
        if label != obs.consensus_label and label != obs.source_dissent_label:
            tl += 1
    return PairedCounts(
        count=count, ignored=ignored, consensus_correct=cc, dissent_correct=dc,
        y_correct_d_wrong=ycdw, y_wrong_d_correct=ywdc, both_correct=bc, both_wrong=bw,
        third_label=tl,
    )


def _require_matches_official(pc: PairedCounts, official: T4StageAccuracy, stage: str) -> None:
    """Defensive, always-on cross-check: this module's own per-record GT
    sampling, summed over an entire stage, must reproduce the committed
    evaluator's own aggregate exactly. This can only fail if the two
    sampling paths ever drift apart; it never fires in normal operation."""
    if (pc.count, pc.ignored, pc.consensus_correct, pc.dissent_correct, pc.third_label) != (
        official.total, official.ignored, official.y_correct, official.d_correct, official.gt_third_class,
    ):
        raise TrustCentralityDiagnosticsError(
            f"stage {stage!r}: diagnostics' own per-record GT tally disagrees with the committed "
            f"evaluator's aggregate T4StageAccuracy -- refusing to report a possibly-inconsistent result"
        )


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------


def centrality_from_distance(distance: float) -> float:
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise TrustCentralityDiagnosticsError("normalized center distance must be numeric")
    distance = float(distance)
    if not math.isfinite(distance) or not (0.0 <= distance <= 1.0):
        raise TrustCentralityDiagnosticsError(f"normalized center distance out of [0,1] or nonfinite: {distance}")
    return 1.0 - distance


def compute_delta_c(obs: ConsensusObservation) -> tuple[float, float, float]:
    """Returns ``(delta_c, c_source, c_jury_mean)``. ``c_jury_mean`` is the
    mean centrality of every window in ``obs.agreeing_window_center_
    distances`` -- by construction (a record only exists for T2+ nodes,
    where all "other" covering windows are unanimous) this is exactly the
    agreeing jury, and it never includes the source."""
    if not obs.agreeing_window_center_distances:
        raise TrustCentralityDiagnosticsError(
            "observation has no agreeing-window centrality metadata (T2+ records must have >=2 others)"
        )
    c_source = centrality_from_distance(obs.source_normalized_center_distance)
    jury = [centrality_from_distance(d) for d in obs.agreeing_window_center_distances]
    c_jury_mean = sum(jury) / len(jury)
    return c_source - c_jury_mean, c_source, c_jury_mean


def centrality_stratum(delta_c: float) -> str:
    if not math.isfinite(delta_c):
        raise TrustCentralityDiagnosticsError("delta_c must be finite")
    if delta_c > 0.0:
        return "positive"
    if delta_c < 0.0:
        return "negative"
    return "zero"


# ---------------------------------------------------------------------------
# Crop-edge (patch-spacing) bands
# ---------------------------------------------------------------------------


def patch_spacing_pixels(window_extent: tuple[int, int], grid_shape: tuple[int, int]) -> Optional[float]:
    """Minimum per-axis patch spacing in pixels, reusing the committed
    align_corners=True node spacing (``(extent - 1) / (grid - 1)``) already
    used by the geometry/t4-audit stencil machinery. ``None`` if neither
    axis has more than one grid node (spacing undefined)."""
    height, width = window_extent
    grid_h, grid_w = grid_shape
    spacings = []
    if grid_h > 1:
        spacings.append((height - 1) / (grid_h - 1))
    if grid_w > 1:
        spacings.append((width - 1) / (grid_w - 1))
    if not spacings:
        return None
    return min(spacings)


def edge_distance_patches(obs: ConsensusObservation, grid_shape: Optional[tuple[int, int]]) -> Optional[float]:
    """Distance from the source anchor to the nearest crop edge, in
    patch-spacing units. ``None`` (reported as "unavailable", never
    silently binned) when ``grid_shape`` is unknown (heterogeneous grids
    for this image) or spacing is degenerate."""
    if grid_shape is None:
        return None
    spacing = patch_spacing_pixels(obs.window_extent, grid_shape)
    if spacing is None or spacing <= 0:
        return None
    local_row, local_col = obs.local_anchor
    height, width = obs.window_extent
    edge_px = min(local_row, (height - 1) - local_row, local_col, (width - 1) - local_col)
    edge_px = max(0.0, edge_px)
    return edge_px / spacing


def edge_band(patches: float) -> str:
    if not math.isfinite(patches) or patches < 0:
        raise TrustCentralityDiagnosticsError(f"edge distance in patches must be finite and non-negative: {patches}")
    if patches < 1.0:
        return "<1"
    if patches < 2.0:
        return "[1,2)"
    if patches < 4.0:
        return "[2,4)"
    return ">=4"


# ---------------------------------------------------------------------------
# Shared-unary stratification (strict T4 only)
# ---------------------------------------------------------------------------


def shared_unary_group(obs: ConsensusObservation) -> str:
    if not obs.other_window_unary_labels:
        return "unavailable"
    fraction = obs.unary_other_agree_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, float) or not (0.0 <= fraction <= 1.0):
        raise TrustCentralityDiagnosticsError("unary_other_agree_fraction must be a float in [0,1]")
    if fraction == 1.0:
        return "all"
    if fraction == 0.0:
        return "none"
    return "some"


# ---------------------------------------------------------------------------
# Descriptive statistics (bounded per-image scalar samples only)
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = q * (n - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


@dataclass(frozen=True)
class DescriptiveStats:
    count: int
    mean: Optional[float]
    std: Optional[float]
    median: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]
    quantiles: Mapping[float, float]

    def __post_init__(self) -> None:
        _require_nonneg_int(self.count, "count")
        if self.count == 0:
            if any(v is not None for v in (self.mean, self.std, self.median, self.minimum, self.maximum)) or self.quantiles:
                raise TrustCentralityDiagnosticsError("an empty DescriptiveStats must have all fields None/empty")
        else:
            for name in ("mean", "std", "median", "minimum", "maximum"):
                if getattr(self, name) is None:
                    raise TrustCentralityDiagnosticsError(f"{name} must be populated when count > 0")


EMPTY_DESCRIPTIVE_STATS = DescriptiveStats(count=0, mean=None, std=None, median=None, minimum=None, maximum=None, quantiles={})


def describe(values: Sequence[float]) -> DescriptiveStats:
    n = len(values)
    if n == 0:
        return EMPTY_DESCRIPTIVE_STATS
    for v in values:
        if not math.isfinite(v):
            raise TrustCentralityDiagnosticsError("descriptive stats require finite values")
    ordered = sorted(values)
    mean = sum(ordered) / n
    variance = sum((v - mean) ** 2 for v in ordered) / n
    std = math.sqrt(variance)
    quantiles = {q: _percentile(ordered, q) for q in QUANTILE_LEVELS}
    return DescriptiveStats(count=n, mean=mean, std=std, median=quantiles[0.5], minimum=ordered[0], maximum=ordered[-1], quantiles=quantiles)


# ---------------------------------------------------------------------------
# Per-image diagnostic statistics (GT consumed here; nothing above this
# point in the per-record pipeline needs it beyond the single sampling
# call already covered by _accumulate_paired_counts / *_stratify_by*).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageDiagnosticStatistics:
    image_id: str
    base: Mapping[str, PairedCounts]
    centrality_strata: Mapping[str, Mapping[str, PairedCounts]]
    centrality_bins: Mapping[str, Mapping[int, PairedCounts]]
    edge_bands: Mapping[str, Mapping[str, PairedCounts]]
    shared_unary: Mapping[str, PairedCounts]
    per_class_t4: Mapping[int, PairedCounts]
    delta_c_samples: Mapping[str, tuple[float, ...]]
    c_source_samples: Mapping[str, tuple[float, ...]]
    c_jury_mean_samples: Mapping[str, tuple[float, ...]]
    edge_patches_samples: Mapping[str, tuple[float, ...]]
    edge_unavailable_count: Mapping[str, int]
    g_equals_d_count_t4: int
    g_is_third_label_count_t4: int

    def __post_init__(self) -> None:
        _require_type(self.image_id, str, "image_id")
        for name in ("g_equals_d_count_t4", "g_is_third_label_count_t4"):
            _require_nonneg_int(getattr(self, name), name)
        for stage in BASE_STAGES:
            if stage not in self.base:
                raise TrustCentralityDiagnosticsError(f"missing base stage {stage!r}")


def _stratify_by(
    observations: Sequence[ConsensusObservation], gt: torch.Tensor, image_size: SpatialSize, ignore_label: int,
    key_fn,
) -> dict:
    buckets: dict = {}
    for obs in observations:
        key = key_fn(obs)
        buckets.setdefault(key, []).append(obs)
    return {key: _accumulate_paired_counts(obs_list, gt, image_size, ignore_label) for key, obs_list in buckets.items()}


def _stratify_by_gt_class(
    observations: Sequence[ConsensusObservation], gt: torch.Tensor, image_size: SpatialSize, ignore_label: int,
) -> dict:
    buckets: dict = {}
    for obs in observations:
        label = sample_gt_nearest_neighbor(gt, obs.global_anchor[0], obs.global_anchor[1], image_size)
        if label == ignore_label:
            continue
        buckets.setdefault(label, []).append(obs)
    return {key: _accumulate_paired_counts(obs_list, gt, image_size, ignore_label) for key, obs_list in buckets.items()}


def build_image_diagnostic_statistics(
    signal: T4ImageSignal,
    gt: torch.Tensor,
    *,
    ignore_label: int,
    image_size: SpatialSize,
    common_patch_grid_shape: Optional[tuple[int, int]],
) -> ImageDiagnosticStatistics:
    """The only function in this module permitted to combine a frozen
    ``T4ImageSignal`` with a GT tensor. Never mutates ``signal``; never
    calls the T4 signal builder; never touches the cache. Cross-validates
    its own base-stage GT tallies against the committed evaluator's own
    aggregate before returning anything (see _require_matches_official)."""
    if not isinstance(signal, T4ImageSignal):
        raise TrustCentralityDiagnosticsError("signal must be a T4ImageSignal")
    evaluation = evaluate_t4_signal_against_gt(signal, gt, ignore_label=ignore_label, image_size=image_size)

    stage_observations = {
        "t2": [o for o in signal.observations if o.t2],
        "t3": [o for o in signal.observations if o.t3],
        "actionable": [o for o in signal.observations if o.actionable],
        "t4": [o for o in signal.observations if o.t4],
        "t4_prime": [o for o in signal.observations if o.t4_prime],
    }
    official = {"t2": evaluation.t2, "t3": evaluation.t3, "actionable": evaluation.actionable, "t4": evaluation.t4, "t4_prime": evaluation.t4_prime}

    base: dict[str, PairedCounts] = {}
    for stage in BASE_STAGES:
        pc = _accumulate_paired_counts(stage_observations[stage], gt, image_size, ignore_label)
        _require_matches_official(pc, official[stage], stage)
        base[stage] = pc

    centrality_strata: dict[str, dict[str, PairedCounts]] = {}
    centrality_bins: dict[str, dict[int, PairedCounts]] = {}
    edge_bands_out: dict[str, dict[str, PairedCounts]] = {}
    delta_c_samples: dict[str, tuple[float, ...]] = {}
    c_source_samples: dict[str, tuple[float, ...]] = {}
    c_jury_mean_samples: dict[str, tuple[float, ...]] = {}
    edge_patches_samples: dict[str, tuple[float, ...]] = {}
    edge_unavailable_count: dict[str, int] = {}

    for stage in EXTENDED_STAGES:
        obs_list = stage_observations[stage]

        def _centrality_key(o: ConsensusObservation) -> str:
            delta_c, _, _ = compute_delta_c(o)
            return centrality_stratum(delta_c)

        centrality_strata[stage] = _stratify_by(obs_list, gt, image_size, ignore_label, _centrality_key)

        def _centrality_bin_key(o: ConsensusObservation) -> int:
            delta_c, _, _ = compute_delta_c(o)
            return delta_c_bin_index(delta_c)

        centrality_bins[stage] = _stratify_by(obs_list, gt, image_size, ignore_label, _centrality_bin_key)

        deltas, sources, juries = [], [], []
        for o in obs_list:
            dc, cs, cj = compute_delta_c(o)
            deltas.append(dc)
            sources.append(cs)
            juries.append(cj)
        delta_c_samples[stage] = tuple(deltas)
        c_source_samples[stage] = tuple(sources)
        c_jury_mean_samples[stage] = tuple(juries)

        def _edge_key(o: ConsensusObservation) -> Optional[str]:
            patches = edge_distance_patches(o, common_patch_grid_shape)
            return None if patches is None else edge_band(patches)

        buckets: dict = {}
        unavailable = 0
        edge_values = []
        for o in obs_list:
            patches = edge_distance_patches(o, common_patch_grid_shape)
            if patches is None:
                unavailable += 1
                continue
            edge_values.append(patches)
            buckets.setdefault(edge_band(patches), []).append(o)
        edge_bands_out[stage] = {
            key: _accumulate_paired_counts(obs, gt, image_size, ignore_label) for key, obs in buckets.items()
        }
        edge_patches_samples[stage] = tuple(edge_values)
        edge_unavailable_count[stage] = unavailable

    t4_observations = stage_observations["t4"]
    shared_unary = _stratify_by(t4_observations, gt, image_size, ignore_label, shared_unary_group)
    per_class_t4 = _stratify_by_gt_class(t4_observations, gt, image_size, ignore_label)
    g_equals_d_count_t4 = sum(1 for o in t4_observations if o.g_equals_d)
    g_is_third_label_count_t4 = sum(1 for o in t4_observations if o.g_is_third_label)

    return ImageDiagnosticStatistics(
        image_id=signal.image_id,
        base=base,
        centrality_strata=centrality_strata,
        centrality_bins=centrality_bins,
        edge_bands=edge_bands_out,
        shared_unary=shared_unary,
        per_class_t4=per_class_t4,
        delta_c_samples=delta_c_samples,
        c_source_samples=c_source_samples,
        c_jury_mean_samples=c_jury_mean_samples,
        edge_patches_samples=edge_patches_samples,
        edge_unavailable_count=edge_unavailable_count,
        g_equals_d_count_t4=g_equals_d_count_t4,
        g_is_third_label_count_t4=g_is_third_label_count_t4,
    )


# ---------------------------------------------------------------------------
# Image-cluster clustered percentile bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    population: str
    settings: BootstrapSettings
    status: str  # "available" | "unavailable"
    unavailable_reason: Optional[str]
    point_estimate: Optional[float]
    image_macro_estimate: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    valid_replicate_count: int
    invalid_replicate_count: int
    invalid_replicate_rate: float
    images_in_population: int
    images_with_zero_records: int
    observed: PairedCounts

    def __post_init__(self) -> None:
        _require_type(self.population, str, "population")
        if not isinstance(self.observed, PairedCounts):
            raise TrustCentralityDiagnosticsError("observed must be a PairedCounts")
        if not isinstance(self.settings, BootstrapSettings):
            raise TrustCentralityDiagnosticsError("settings must be BootstrapSettings")
        if self.status not in ("available", "unavailable"):
            raise TrustCentralityDiagnosticsError("status must be 'available' or 'unavailable'")
        if self.status == "unavailable" and not self.unavailable_reason:
            raise TrustCentralityDiagnosticsError("unavailable_reason is required when status is 'unavailable'")
        if self.status == "available":
            for name in ("point_estimate", "ci_low", "ci_high"):
                if getattr(self, name) is None:
                    raise TrustCentralityDiagnosticsError(f"{name} required when status is 'available'")
        _require_nonneg_int(self.valid_replicate_count, "valid_replicate_count")
        _require_nonneg_int(self.invalid_replicate_count, "invalid_replicate_count")
        _require_nonneg_int(self.images_in_population, "images_in_population")
        _require_nonneg_int(self.images_with_zero_records, "images_with_zero_records")


def interpret_trust(result: BootstrapResult) -> str:
    """Deterministic, report-only label. Never influences inference or
    future target selection."""
    if result.status != "available" or result.point_estimate is None:
        return "UNAVAILABLE"
    if result.point_estimate <= 0.0:
        return "CONTRADICTED"
    if result.ci_low is not None and result.ci_low > 0.0:
        return "SUPPORTED"
    return "INCONCLUSIVE"


def run_clustered_bootstrap(
    populations: Mapping[str, Mapping[str, PairedCounts]],
    all_image_ids: Sequence[str],
    *,
    settings: BootstrapSettings = CANONICAL_BOOTSTRAP_SETTINGS,
) -> dict[str, BootstrapResult]:
    """One shared image-resampling stream drives every registered
    population's bootstrap replicate, for statistical coherence and to
    avoid re-drawing N*resamples image indices per population.

    ``populations``: population name -> {image_id: PairedCounts}. An
    image absent from a population's mapping is treated as contributing
    zero records to that population (never dropped from the resampling
    pool -- see module docstring).
    """
    if not isinstance(settings, BootstrapSettings):
        raise TrustCentralityDiagnosticsError("settings must be BootstrapSettings")
    if not isinstance(all_image_ids, (list, tuple)) or not all(isinstance(i, str) for i in all_image_ids):
        raise TrustCentralityDiagnosticsError("all_image_ids must be a list/tuple of str")
    n = len(all_image_ids)
    if n == 0:
        raise TrustCentralityDiagnosticsError("cannot bootstrap over an empty image list")
    for name, mapping in populations.items():
        if not isinstance(mapping, dict):
            raise TrustCentralityDiagnosticsError(f"population {name!r} must map image_id -> PairedCounts")
        for image_id, pc in mapping.items():
            if image_id not in all_image_ids:
                raise TrustCentralityDiagnosticsError(
                    f"population {name!r} references image_id {image_id!r} not in all_image_ids"
                )
            if not isinstance(pc, PairedCounts):
                raise TrustCentralityDiagnosticsError(f"population {name!r}: values must be PairedCounts")

    # Point estimate and image-macro estimate: computed once from the full
    # (non-resampled) observed data, independent of the bootstrap RNG/seed.
    observed_totals: dict[str, PairedCounts] = {}
    image_macro: dict[str, Optional[float]] = {}
    images_in_population: dict[str, int] = {}
    images_with_zero: dict[str, int] = {}
    for name, mapping in populations.items():
        total = EMPTY_PAIRED_COUNTS
        macro_values = []
        containing = 0
        for image_id, pc in mapping.items():
            total = total + pc
            if pc.count > 0:
                containing += 1
                dt = pc.delta_trust()
                if dt is not None:
                    macro_values.append(dt)
        observed_totals[name] = total
        image_macro[name] = (sum(macro_values) / len(macro_values)) if macro_values else None
        images_in_population[name] = containing
        images_with_zero[name] = n - containing

    # Precompute (count, consensus_correct, dissent_correct) per image per
    # population as flat arrays indexed identically to all_image_ids, for
    # fast resampling.
    flat: dict[str, list[tuple[int, int, int]]] = {}
    for name, mapping in populations.items():
        flat[name] = [
            (
                mapping[image_id].count if image_id in mapping else 0,
                mapping[image_id].consensus_correct if image_id in mapping else 0,
                mapping[image_id].dissent_correct if image_id in mapping else 0,
            )
            for image_id in all_image_ids
        ]

    rng = random.Random(settings.seed)
    valid_deltas: dict[str, list[float]] = {name: [] for name in populations}
    invalid_counts: dict[str, int] = {name: 0 for name in populations}

    for _ in range(settings.resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        for name, arr in flat.items():
            total_count = total_cc = total_dc = 0
            for idx in indices:
                c, cc, dc = arr[idx]
                total_count += c
                total_cc += cc
                total_dc += dc
            if total_count == 0:
                invalid_counts[name] += 1
            else:
                valid_deltas[name].append((total_cc - total_dc) / total_count)

    alpha = 1.0 - settings.confidence_level
    lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0

    results: dict[str, BootstrapResult] = {}
    for name in populations:
        deltas = sorted(valid_deltas[name])
        valid_count = len(deltas)
        invalid_count = invalid_counts[name]
        rate = invalid_count / settings.resamples
        point_estimate = observed_totals[name].delta_trust()
        if point_estimate is None:
            results[name] = BootstrapResult(
                population=name, settings=settings, status="unavailable",
                unavailable_reason="zero observed records for this population across all evaluated images",
                point_estimate=None, image_macro_estimate=image_macro[name],
                ci_low=None, ci_high=None,
                valid_replicate_count=valid_count, invalid_replicate_count=invalid_count,
                invalid_replicate_rate=rate,
                images_in_population=images_in_population[name], images_with_zero_records=images_with_zero[name],
                observed=observed_totals[name],
            )
        elif valid_count < MINIMUM_VALID_BOOTSTRAP_REPLICATES:
            results[name] = BootstrapResult(
                population=name, settings=settings, status="unavailable",
                unavailable_reason=(
                    f"only {valid_count} valid bootstrap replicates (< {MINIMUM_VALID_BOOTSTRAP_REPLICATES} minimum); "
                    "too many resamples had zero pooled records for this population"
                ),
                point_estimate=point_estimate, image_macro_estimate=image_macro[name],
                ci_low=None, ci_high=None,
                valid_replicate_count=valid_count, invalid_replicate_count=invalid_count,
                invalid_replicate_rate=rate,
                images_in_population=images_in_population[name], images_with_zero_records=images_with_zero[name],
                observed=observed_totals[name],
            )
        else:
            ci_low = _percentile(deltas, lower_q)
            ci_high = _percentile(deltas, upper_q)
            results[name] = BootstrapResult(
                population=name, settings=settings, status="available", unavailable_reason=None,
                point_estimate=point_estimate, image_macro_estimate=image_macro[name],
                ci_low=ci_low, ci_high=ci_high,
                valid_replicate_count=valid_count, invalid_replicate_count=invalid_count,
                invalid_replicate_rate=rate,
                images_in_population=images_in_population[name], images_with_zero_records=images_with_zero[name],
                observed=observed_totals[name],
            )
    return results


# ---------------------------------------------------------------------------
# Streaming accumulator and final report
# ---------------------------------------------------------------------------


class TrustCentralityAccumulator:
    """Mutable, streaming accumulator: absorbs one image's
    ``ImageDiagnosticStatistics`` at a time. Retains only per-image
    sufficient statistics (PairedCounts) and bounded scalar samples
    (Delta_c / edge-patches) needed for the bootstrap and descriptive
    quantiles -- never raw ``ConsensusObservation`` records or GT
    tensors."""

    def __init__(self) -> None:
        self._image_ids: list[str] = []
        self._seen_image_ids: set[str] = set()
        self._base_by_image: dict[str, dict[str, PairedCounts]] = {}
        self._centrality_by_image: dict[str, dict[str, dict[str, PairedCounts]]] = {}
        self._centrality_bins_by_image: dict[str, dict[str, dict[int, PairedCounts]]] = {}
        self._edge_by_image: dict[str, dict[str, dict[str, PairedCounts]]] = {}
        self._shared_unary_by_image: dict[str, dict[str, PairedCounts]] = {}
        self._per_class_by_image: dict[str, dict[int, PairedCounts]] = {}
        self._delta_c_pool: dict[str, list[float]] = {stage: [] for stage in EXTENDED_STAGES}
        self._c_source_pool: dict[str, list[float]] = {stage: [] for stage in EXTENDED_STAGES}
        self._c_jury_pool: dict[str, list[float]] = {stage: [] for stage in EXTENDED_STAGES}
        self._edge_patches_pool: dict[str, list[float]] = {stage: [] for stage in EXTENDED_STAGES}
        self._edge_unavailable_total: dict[str, int] = {stage: 0 for stage in EXTENDED_STAGES}
        self._g_equals_d_total = 0
        self._g_third_label_total = 0

    def absorb_image(self, stats: ImageDiagnosticStatistics) -> None:
        if not isinstance(stats, ImageDiagnosticStatistics):
            raise TrustCentralityDiagnosticsError("stats must be an ImageDiagnosticStatistics")
        if stats.image_id in self._seen_image_ids:
            raise TrustCentralityDiagnosticsError(f"image_id {stats.image_id!r} absorbed more than once")
        self._seen_image_ids.add(stats.image_id)
        self._image_ids.append(stats.image_id)
        self._base_by_image[stats.image_id] = dict(stats.base)
        self._centrality_by_image[stats.image_id] = {k: dict(v) for k, v in stats.centrality_strata.items()}
        self._centrality_bins_by_image[stats.image_id] = {k: dict(v) for k, v in stats.centrality_bins.items()}
        self._edge_by_image[stats.image_id] = {k: dict(v) for k, v in stats.edge_bands.items()}
        self._shared_unary_by_image[stats.image_id] = dict(stats.shared_unary)
        self._per_class_by_image[stats.image_id] = dict(stats.per_class_t4)
        for stage in EXTENDED_STAGES:
            self._delta_c_pool[stage].extend(stats.delta_c_samples.get(stage, ()))
            self._c_source_pool[stage].extend(stats.c_source_samples.get(stage, ()))
            self._c_jury_pool[stage].extend(stats.c_jury_mean_samples.get(stage, ()))
            self._edge_patches_pool[stage].extend(stats.edge_patches_samples.get(stage, ()))
            self._edge_unavailable_total[stage] += stats.edge_unavailable_count.get(stage, 0)
        self._g_equals_d_total += stats.g_equals_d_count_t4
        self._g_third_label_total += stats.g_is_third_label_count_t4

    def image_ids(self) -> tuple[str, ...]:
        return tuple(self._image_ids)

    def summary(self, *, settings: BootstrapSettings = CANONICAL_BOOTSTRAP_SETTINGS) -> "TrustCentralityReport":
        if not self._image_ids:
            raise TrustCentralityDiagnosticsError("no images have been absorbed; cannot summarize")

        base_totals: dict[str, PairedCounts] = {stage: EMPTY_PAIRED_COUNTS for stage in BASE_STAGES}
        for per_image in self._base_by_image.values():
            for stage in BASE_STAGES:
                base_totals[stage] = base_totals[stage] + per_image[stage]

        populations: dict[str, dict[str, PairedCounts]] = {}
        populations["t4"] = {img: self._base_by_image[img]["t4"] for img in self._image_ids}

        for stage in EXTENDED_STAGES:
            for stratum in CENTRALITY_STRATA:
                populations[f"{stage}_centrality_{stratum}"] = {
                    img: self._centrality_by_image[img].get(stage, {}).get(stratum, EMPTY_PAIRED_COUNTS)
                    for img in self._image_ids
                }

        for stage in EXTENDED_STAGES:
            for bin_idx in range(DELTA_C_BIN_COUNT):
                populations[f"{stage}_centrality_bin_{bin_idx}"] = {
                    img: self._centrality_bins_by_image[img].get(stage, {}).get(bin_idx, EMPTY_PAIRED_COUNTS)
                    for img in self._image_ids
                }

        for stage in EXTENDED_STAGES:
            for band in EDGE_BAND_LABELS:
                populations[f"{stage}_edge_{band}"] = {
                    img: self._edge_by_image[img].get(stage, {}).get(band, EMPTY_PAIRED_COUNTS)
                    for img in self._image_ids
                }

        for group in SHARED_UNARY_GROUPS:
            populations[f"t4_shared_unary_{group}"] = {
                img: self._shared_unary_by_image[img].get(group, EMPTY_PAIRED_COUNTS) for img in self._image_ids
            }

        bootstrap_results = run_clustered_bootstrap(populations, self._image_ids, settings=settings)

        centrality_descriptive = {stage: describe(self._delta_c_pool[stage]) for stage in EXTENDED_STAGES}
        c_source_descriptive = {stage: describe(self._c_source_pool[stage]) for stage in EXTENDED_STAGES}
        c_jury_descriptive = {stage: describe(self._c_jury_pool[stage]) for stage in EXTENDED_STAGES}
        edge_descriptive = {stage: describe(self._edge_patches_pool[stage]) for stage in EXTENDED_STAGES}

        centrality_bin_histogram = {stage: self._bin_delta_c(self._delta_c_pool[stage]) for stage in EXTENDED_STAGES}

        per_class_totals: dict[int, PairedCounts] = {}
        for per_image in self._per_class_by_image.values():
            for cls, pc in per_image.items():
                per_class_totals[cls] = per_class_totals.get(cls, EMPTY_PAIRED_COUNTS) + pc

        trust_interpretation = interpret_trust(bootstrap_results["t4"])

        return TrustCentralityReport(
            images_processed=len(self._image_ids),
            settings=settings,
            base_stage_totals=base_totals,
            bootstrap_results=bootstrap_results,
            trust_interpretation_t4=trust_interpretation,
            centrality_delta_descriptive=centrality_descriptive,
            centrality_source_descriptive=c_source_descriptive,
            centrality_jury_descriptive=c_jury_descriptive,
            centrality_bin_edges=DELTA_C_BIN_EDGES,
            centrality_bin_histogram=centrality_bin_histogram,
            edge_patches_descriptive=edge_descriptive,
            edge_unavailable_count=dict(self._edge_unavailable_total),
            per_class_t4_totals=per_class_totals,
            g_equals_d_count_t4=self._g_equals_d_total,
            g_is_third_label_count_t4=self._g_third_label_total,
        )

    @staticmethod
    def _bin_delta_c(values: Sequence[float]) -> tuple[int, ...]:
        counts = [0] * DELTA_C_BIN_COUNT
        for v in values:
            counts[delta_c_bin_index(v)] += 1
        return tuple(counts)

    def image_order_digest(self) -> str:
        """SHA256 over the exact, ordered sequence of absorbed image_ids.
        The clustered bootstrap resamples by indexing into this same
        sequence, so an uninterrupted run and a resumed run must restore
        it byte-for-byte identically, not merely as an unordered set, for
        their bootstrap replicates to agree."""
        return hashlib.sha256("\n".join(self._image_ids).encode("utf-8")).hexdigest()

    def state_dict(self) -> dict:
        """Exact per-image sufficient statistics -- the bootstrap unit is
        image, so (unlike T4AuditAccumulator's pure running totals) this
        accumulator's per-image PairedCounts records and per-image
        continuous samples must be individually restorable, in the exact
        original image order, for a resumed run's clustered bootstrap to
        reproduce identical replicates. Never stores raw predictions/GT."""
        return {
            "image_ids": list(self._image_ids),
            "base_by_image": {
                img: {stage: pc.to_dict() for stage, pc in stages.items()}
                for img, stages in self._base_by_image.items()
            },
            "centrality_by_image": {
                img: {stage: {stratum: pc.to_dict() for stratum, pc in strata.items()} for stage, strata in stages.items()}
                for img, stages in self._centrality_by_image.items()
            },
            "centrality_bins_by_image": {
                img: {
                    stage: {str(bin_idx): pc.to_dict() for bin_idx, pc in bins.items()}
                    for stage, bins in stages.items()
                }
                for img, stages in self._centrality_bins_by_image.items()
            },
            "edge_by_image": {
                img: {stage: {band: pc.to_dict() for band, pc in bands.items()} for stage, bands in stages.items()}
                for img, stages in self._edge_by_image.items()
            },
            "shared_unary_by_image": {
                img: {group: pc.to_dict() for group, pc in groups.items()}
                for img, groups in self._shared_unary_by_image.items()
            },
            "per_class_by_image": {
                img: {str(cls): pc.to_dict() for cls, pc in classes.items()}
                for img, classes in self._per_class_by_image.items()
            },
            "delta_c_pool": {stage: list(values) for stage, values in self._delta_c_pool.items()},
            "c_source_pool": {stage: list(values) for stage, values in self._c_source_pool.items()},
            "c_jury_pool": {stage: list(values) for stage, values in self._c_jury_pool.items()},
            "edge_patches_pool": {stage: list(values) for stage, values in self._edge_patches_pool.items()},
            "edge_unavailable_total": dict(self._edge_unavailable_total),
            "g_equals_d_total": self._g_equals_d_total,
            "g_third_label_total": self._g_third_label_total,
            "image_order_digest": self.image_order_digest(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "TrustCentralityAccumulator":
        acc = cls()
        try:
            image_ids = list(state["image_ids"])
            if len(set(image_ids)) != len(image_ids):
                raise TrustCentralityDiagnosticsError("checkpoint image_ids contains duplicates")
            acc._image_ids = image_ids
            acc._seen_image_ids = set(image_ids)
            acc._base_by_image = {
                img: {stage: PairedCounts.from_dict(pc) for stage, pc in stages.items()}
                for img, stages in state["base_by_image"].items()
            }
            acc._centrality_by_image = {
                img: {stage: {stratum: PairedCounts.from_dict(pc) for stratum, pc in strata.items()} for stage, strata in stages.items()}
                for img, stages in state["centrality_by_image"].items()
            }
            acc._centrality_bins_by_image = {
                img: {
                    stage: {int(bin_idx): PairedCounts.from_dict(pc) for bin_idx, pc in bins.items()}
                    for stage, bins in stages.items()
                }
                for img, stages in state["centrality_bins_by_image"].items()
            }
            acc._edge_by_image = {
                img: {stage: {band: PairedCounts.from_dict(pc) for band, pc in bands.items()} for stage, bands in stages.items()}
                for img, stages in state["edge_by_image"].items()
            }
            acc._shared_unary_by_image = {
                img: {group: PairedCounts.from_dict(pc) for group, pc in groups.items()}
                for img, groups in state["shared_unary_by_image"].items()
            }
            acc._per_class_by_image = {
                img: {int(cls): PairedCounts.from_dict(pc) for cls, pc in classes.items()}
                for img, classes in state["per_class_by_image"].items()
            }
            for pool_name, target in (
                ("delta_c_pool", "_delta_c_pool"), ("c_source_pool", "_c_source_pool"),
                ("c_jury_pool", "_c_jury_pool"), ("edge_patches_pool", "_edge_patches_pool"),
            ):
                pool = {stage: list(values) for stage, values in state[pool_name].items()}
                if set(pool) != set(EXTENDED_STAGES):
                    raise TrustCentralityDiagnosticsError(f"checkpoint {pool_name} has an unexpected stage key set")
                setattr(acc, target, pool)
            edge_unavailable = {stage: int(v) for stage, v in state["edge_unavailable_total"].items()}
            if set(edge_unavailable) != set(EXTENDED_STAGES):
                raise TrustCentralityDiagnosticsError("checkpoint edge_unavailable_total has an unexpected stage key set")
            acc._edge_unavailable_total = edge_unavailable
            acc._g_equals_d_total = _require_nonneg_int(state["g_equals_d_total"], "g_equals_d_total")
            acc._g_third_label_total = _require_nonneg_int(state["g_third_label_total"], "g_third_label_total")
        except KeyError as error:
            raise TrustCentralityDiagnosticsError(
                f"trust/centrality accumulator checkpoint state is missing required field {error}"
            ) from error
        expected_digest = state.get("image_order_digest")
        if expected_digest is not None and acc.image_order_digest() != expected_digest:
            raise TrustCentralityDiagnosticsError(
                "restored image_ids do not reproduce the checkpoint's own recorded image_order_digest "
                "-- refusing to resume from a state that cannot reproduce identical bootstrap replicates"
            )
        return acc


@dataclass(frozen=True)
class TrustCentralityReport:
    """Final, immutable, aggregate-only report. No per-patch or per-image
    raw records are retained here."""

    images_processed: int
    settings: BootstrapSettings
    base_stage_totals: Mapping[str, PairedCounts]
    bootstrap_results: Mapping[str, BootstrapResult]
    trust_interpretation_t4: str
    centrality_delta_descriptive: Mapping[str, DescriptiveStats]
    centrality_source_descriptive: Mapping[str, DescriptiveStats]
    centrality_jury_descriptive: Mapping[str, DescriptiveStats]
    centrality_bin_edges: tuple[float, ...]
    centrality_bin_histogram: Mapping[str, tuple[int, ...]]
    edge_patches_descriptive: Mapping[str, DescriptiveStats]
    edge_unavailable_count: Mapping[str, int]
    per_class_t4_totals: Mapping[int, PairedCounts]
    g_equals_d_count_t4: int
    g_is_third_label_count_t4: int

    def __post_init__(self) -> None:
        _require_nonneg_int(self.images_processed, "images_processed")
        if self.trust_interpretation_t4 not in ("SUPPORTED", "CONTRADICTED", "INCONCLUSIVE", "UNAVAILABLE"):
            raise TrustCentralityDiagnosticsError("trust_interpretation_t4 must be a canonical label")
        _require_nonneg_int(self.g_equals_d_count_t4, "g_equals_d_count_t4")
        _require_nonneg_int(self.g_is_third_label_count_t4, "g_is_third_label_count_t4")

    def load_bearing_result(self) -> BootstrapResult:
        """``Delta_trust`` on strict T4 restricted to ``Delta_c > 0``
        (dissenting source window more central than its jury) -- the
        quantity spec section 7 identifies as load-bearing."""
        return self.bootstrap_results["t4_centrality_positive"]


# ---------------------------------------------------------------------------
# Integration entry point
# ---------------------------------------------------------------------------


def run_trust_centrality_diagnostics_for_image(
    signal: T4ImageSignal,
    gt: torch.Tensor,
    *,
    ignore_label: int,
    image_size: SpatialSize,
    common_patch_grid_shape: Optional[tuple[int, int]],
) -> ImageDiagnosticStatistics:
    """Call AFTER the T4 signal has been built (and, if desired, evaluated
    against GT) and BEFORE pass 2 / cache.close() -- or any time after,
    since this never touches the cache at all. Read-only: never mutates
    ``signal``. Entirely opt-in; a caller that never calls this pays zero
    cost and sees zero behavior change (this module has no reference from
    ``window_cache.py`` or ``t4_audit.py`` -- see
    test_disabled_diagnostics_zero_overhead_source)."""
    return build_image_diagnostic_statistics(
        signal, gt, ignore_label=ignore_label, image_size=image_size, common_patch_grid_shape=common_patch_grid_shape,
    )


__all__ = [
    "BASE_STAGES",
    "BootstrapResult",
    "BootstrapSettings",
    "CANONICAL_BOOTSTRAP_SETTINGS",
    "CENTRALITY_STRATA",
    "DELTA_C_BIN_COUNT",
    "DELTA_C_BIN_EDGES",
    "DescriptiveStats",
    "EDGE_BAND_LABELS",
    "EXTENDED_STAGES",
    "ImageDiagnosticStatistics",
    "MINIMUM_VALID_BOOTSTRAP_REPLICATES",
    "PairedCounts",
    "QUANTILE_LEVELS",
    "SHARED_UNARY_GROUPS",
    "TrustCentralityAccumulator",
    "TrustCentralityDiagnosticsError",
    "TrustCentralityReport",
    "build_image_diagnostic_statistics",
    "centrality_from_distance",
    "centrality_stratum",
    "compute_delta_c",
    "delta_c_bin_index",
    "describe",
    "edge_band",
    "edge_distance_patches",
    "interpret_trust",
    "patch_spacing_pixels",
    "run_clustered_bootstrap",
    "run_trust_centrality_diagnostics_for_image",
    "shared_unary_group",
]
