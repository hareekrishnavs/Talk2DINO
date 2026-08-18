"""Opt-in, read-only leave-one-window-out consensus audit and T0-T4 funnel.

This module is a **diagnostic side channel**. It consumes an already-sealed
:class:`~segmentation.evaluation.window_cache.ImageWindowCache` produced by
pass 1 of the two-pass window cache, and never reruns the backbone, the
directed top-k graph builder, or the CGLS solver. It never mutates the
cache, never changes pass-2 replay, and never changes the final stitched
segmentation output. With the audit enabled or disabled, production output
is byte-identical -- see ``tests/test_t4_audit.py`` for the parity proof.

Why "leave-one-window-out" consensus, not independent samples
---------------------------------------------------------------
Overlapping crop windows are *not* independent observations of a scene:
they share most of their pixels, most of their DINO features (up to crop
boundary effects), and were produced by the same frozen backbone and
projection head. Calling them a "jury" is a convenience; the correct
framing is that the *other* windows are alternative **context views** of
the same underlying content -- each one re-derives a full RWR solve from a
differently-cropped neighborhood, so unanimity among them is evidence that
the crop-independent signal favors one label, not evidence of statistical
independence. Every docstring and record field below uses "context view"
in place of "sample" for exactly this reason.

Semantic score-grid anchor vs. native DINO patch center
----------------------------------------------------------
:mod:`sliding_window_geometry` already offers
``PatchGridSpec.patch_center_doubled_local`` / ``native_patch_node``, which
locate the *feature* lattice's patch centers and map them natively between
windows when origins are exact multiples of the patch size. This module
uses a **different** geometric object: the *semantic score-grid anchor*
defined in section 3.1 of the design -- the continuous pixel-index position
that a post-RWR score-grid node ``(r, c)`` maps to under the exact
``align_corners=True`` convention already used by
``patch_scores_to_masks``/``masks_from_patch_scores`` in production. This
anchor is defined for *every* window pair, aligned or not, because it uses
real (Fraction-exact) interpolation rather than requiring an integer patch
lattice coincidence. Native patch-center alignment remains reserved for a
later, purely structural graph-comparison stage; it is not used here.

Sigmoid before bilinear sampling
------------------------------------
Consensus operates on ``Q_w = sigmoid(P_w)``, computed *before* any
interpolation, exactly mirroring production's own order of operations in
``patch_scores_to_masks``/``masks_from_patch_scores`` (sigmoid the raw
post-RWR patch grid, then bilinearly upsample). Sampling raw ``P_w`` and
applying sigmoid afterward would not commute with bilinear interpolation
and would silently diverge from what production actually stitches.

Terminology: "operator-attributed diffusion reversal"
----------------------------------------------------------
T4 proves that: every other context view unanimously retains label ``y``;
the frozen pre-RWR E3 unary at the source node also selects ``y``; the
source window's own post-RWR score selects a different label ``d``; and
production's ordinary uniform stitching does not restore ``y`` either. This
is strong evidence that the RWR *diffusion operator itself*, running on the
source crop's particular neighborhood graph, is the cause of the observed
label change -- hence "operator-attributed diffusion reversal". It is
*not* evidence that ``y`` is the ground-truth label; a shared systematic
error in the unary/context-view signal (see "shared-unary risk" below)
could make every one of these anchors agree while still being wrong. This
module never claims T4 is error detection, and never repairs anything.

Exact T0-T4/T4' funnel definitions
----------------------------------------
For a source window ``w`` and node ``i`` with post-RWR grid ``P_w``, unary
grid ``S0_w``, and consensus label ``y`` sampled unanimously from ``w``'s
*other* covering windows (never ``w`` itself):

* **T0**: every valid cached patch-grid node in every valid source window.
* **T1**: at least two OTHER windows cover the anchor (total coverage,
  including the source, is therefore at least three).
* **T2**: all T1-qualifying other windows agree on exactly one label ``y``
  (strict unanimity -- no majority rule, no threshold of any kind).
* **T3**: ``d := argmax P_w(i) != y`` (the source's own post-RWR label
  dissents from the unanimous other-window consensus).
* **actionable**: ``g := argmax G(anchor) != y``, where ``G(anchor)`` is the
  arithmetic mean of ``sigmoid(P)`` sampled at the anchor from *every*
  covering window including the source -- i.e. production's own uniform
  stitching still does not recover ``y`` either.
* **T4**: all actionable conditions plus ``u := argmax S0_w(i) == y`` (the
  frozen pre-RWR unary already agreed with the context views).
* **T4'** (diagnostic only, computed on the actionable subset): strict
  ``S0_w(i, y) > S0_w(i, d)``; never influences T4, repair, or membership.

Shared-unary-noise risk
----------------------------
Section 6 fields record whether the *other* windows' pre-RWR unaries also
agree with ``y`` (and whether the source's own unary agrees). If the E3
unary head has a systematic bias for some class/context combination, that
bias could appear identically in the source window and every context view,
making the whole T0-T4 funnel agree with a wrong label for a reason that
has nothing to do with RWR. These fields are recorded for that later
diagnostic; they never participate in T4 membership.

Why centrality/edge-distance metadata is recorded but unused
-----------------------------------------------------------------
Crop-edge and crop-center distances are recorded per section 7 so a later
commit can test whether disagreements cluster near crop boundaries (where
DINO features and the directed graph are most affected by cropping
artifacts). Using that metadata to filter or weight consensus *now* would
be a centrality-based heuristic, which this commit explicitly excludes.

No repair
-------------
This module never edits a score, a label, a graph edge, or the final
stitched output. It only classifies and counts. Any later "acceptance
rule" or repair mechanism is out of scope by design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Mapping, Optional, Sequence

import torch

from .sliding_window_geometry import (
    BilinearStencil,
    SlidingWindowGeometryError,
    SlidingWindowPlan,
    SpatialSize,
    WindowGeometry,
    _axis_stencil,
    _combine_axis_stencils,
)
from .window_cache import (
    CachedWindowState,
    FirstPassImageContext,
    ImageWindowCache,
    WindowCacheState,
)


class T4AuditError(ValueError):
    """Raised on any T4 audit validation or contract violation. Fail closed."""


# ---------------------------------------------------------------------------
# Exact-rational bilinear stencil: reuses the geometry module's own private
# per-axis stencil math (not reimplemented) for an arbitrary exact rational
# coordinate, generalizing bilinear_stencil/bilinear_stencil_from_doubled
# (denominator 1 / 2 only) to the general anchors this module needs.
# ---------------------------------------------------------------------------


def _axis_stencil_safe(
    numerator: int, denominator: int, output_extent: int, grid_extent: int
) -> tuple[int, int, float]:
    """Like ``sliding_window_geometry._axis_stencil``, but also handles a
    length-1 *output* extent (a 1-pixel-tall/wide effective crop), which
    that function does not itself handle (it raises). Mirrors that
    function's own length-1 *grid* handling: when an axis has only one
    valid pixel position, the only well-defined mapping is to grid index 0
    with full weight, regardless of the requested coordinate."""
    if output_extent < 2:
        return 0, 0, 0.0
    return _axis_stencil(numerator, denominator, output_extent, grid_extent)


def _bilinear_stencil_from_fraction(
    row: Fraction,
    col: Fraction,
    effective_extent: tuple[int, int],
    grid_shape: tuple[int, int],
) -> BilinearStencil:
    """Exact-rational generalization of ``bilinear_stencil``/
    ``bilinear_stencil_from_doubled``, reusing the same private per-axis
    helpers so the interpolation math is defined in exactly one place."""
    row_low, row_high, row_fraction = _axis_stencil_safe(
        row.numerator, row.denominator, effective_extent[0], grid_shape[0]
    )
    col_low, col_high, col_fraction = _axis_stencil_safe(
        col.numerator, col.denominator, effective_extent[1], grid_shape[1]
    )
    return _combine_axis_stencils(
        row_low, row_high, row_fraction, col_low, col_high, col_fraction
    )


def _local_anchor_component(
    index: int, grid_extent: int, effective_extent: int
) -> Fraction:
    """Section 3.1's ``local_y``/``local_x`` formula, as an exact Fraction.

    Handles one-element grid/extent dimensions explicitly:
    * ``effective_extent <= 1``: only pixel position 0 is valid.
    * ``grid_extent <= 1``: the single grid node covers the whole extent,
      so its anchor is the extent's midpoint.
    """
    if effective_extent <= 1:
        return Fraction(0)
    if grid_extent <= 1:
        return Fraction(effective_extent - 1, 2)
    return Fraction(index * (effective_extent - 1), grid_extent - 1)


def source_score_grid_anchor(
    window: WindowGeometry, patch_grid_shape: tuple[int, int], node_row: int, node_col: int
) -> tuple[Fraction, Fraction]:
    """Section 3.1: the exact global semantic score-grid anchor of source
    node ``(node_row, node_col)``. Maps exactly back to that node under the
    committed ``align_corners=True`` convention (see
    ``test_source_anchor_maps_exactly_back_to_its_own_node``)."""
    height, width = window.extent.as_tuple()
    grid_h, grid_w = patch_grid_shape
    local_y = _local_anchor_component(node_row, grid_h, height)
    local_x = _local_anchor_component(node_col, grid_w, width)
    return (window.origin.row + local_y, window.origin.col + local_x)


# ---------------------------------------------------------------------------
# Continuous coverage: the exact real-valued generalization of the existing
# committed half-open integer coverage rule (contains_point,
# windows_covering_point). For integers, "origin <= p < end" is identical to
# "origin <= p <= end - 1"; the closed form is the one that generalizes
# correctly to a continuous align_corners=True pixel-index coordinate, since
# that convention's own valid range is the closed interval
# [0, extent - 1]. This is not a competing coverage rule -- it is the same
# rule, generalized to the coordinate type this module actually needs.
# ---------------------------------------------------------------------------


def covers_continuous_point(window: WindowGeometry, y: Fraction, x: Fraction) -> bool:
    return (
        window.origin.row <= y <= window.end.row - 1
        and window.origin.col <= x <= window.end.col - 1
    )


def _row_band_covers(window: WindowGeometry, y: Fraction) -> bool:
    return window.origin.row <= y <= window.end.row - 1


def _col_band_covers(window: WindowGeometry, x: Fraction) -> bool:
    return window.origin.col <= x <= window.end.col - 1


def _covering_grid_rows(plan: SlidingWindowPlan, y: Fraction) -> tuple[int, ...]:
    """Grid rows whose shared row band covers ``y``.

    All windows in the same grid row share one row band -- exactly the
    same structural fact ``SlidingWindowPlan._row_bands`` already exploits
    for integer coverage, reused here at the coarser grid-index granularity
    so per-node coverage lookup is O(1) after one O(grid_rows + grid_cols)
    precomputation pass per window, not O(grid_rows * grid_cols) per node.
    """
    return tuple(
        grid_row for grid_row in range(plan.grid_rows)
        if _row_band_covers(plan.window_at(grid_row, 0), y)
    )


def _covering_grid_cols(plan: SlidingWindowPlan, x: Fraction) -> tuple[int, ...]:
    """Grid columns whose shared column band covers ``x``; see
    :func:`_covering_grid_rows`."""
    return tuple(
        grid_col for grid_col in range(plan.grid_cols)
        if _col_band_covers(plan.window_at(0, grid_col), x)
    )


# ---------------------------------------------------------------------------
# Argmax with explicit tie detection (production convention: first/lowest
# index wins ties, matching torch.argmax's documented behavior; a
# *separate* boolean records whether a tie existed at all).
# ---------------------------------------------------------------------------


def _argmax_with_tie(scores: torch.Tensor) -> tuple[int, bool]:
    if scores.ndim != 1:
        raise T4AuditError("argmax input must be a 1-D class-score vector")
    if not bool(torch.isfinite(scores).all()):
        raise T4AuditError("argmax input must be finite")
    top_value = scores.max()
    label = int(scores.argmax().item())
    tie = bool((scores == top_value).sum().item() > 1)
    return label, tie


def _sample_bilinear(
    grid_scores: torch.Tensor, grid_shape: tuple[int, int], stencil: BilinearStencil
) -> torch.Tensor:
    """Vectorized (no per-class Python loop) bilinear gather of a [N, C]
    grid at one stencil's four neighbor nodes."""
    width = grid_shape[1]
    idx00 = stencil.row_low * width + stencil.col_low
    idx01 = stencil.row_low * width + stencil.col_high
    idx10 = stencil.row_high * width + stencil.col_low
    idx11 = stencil.row_high * width + stencil.col_high
    weights = torch.tensor(
        stencil.weights(), dtype=grid_scores.dtype, device=grid_scores.device
    )
    gathered = grid_scores[[idx00, idx01, idx10, idx11]]  # [4, C]
    return (weights.unsqueeze(-1) * gathered).sum(dim=0)  # [C]


# ---------------------------------------------------------------------------
# Immutable per-anchor record
# ---------------------------------------------------------------------------


def _require_type(value, expected_type, label: str):
    if not isinstance(value, expected_type):
        raise T4AuditError(f"{label} must be {expected_type}, got {type(value)}")
    return value


def _require_finite_float(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T4AuditError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise T4AuditError(f"{label} must be finite")
    return value


@dataclass(frozen=True)
class ConsensusObservation:
    """One immutable leave-one-window-out consensus record for a single
    source ``(window, node)`` anchor. Ground-truth-free by construction:
    no field here is ever set from a GT map."""

    image_id: str
    source_window_id: int
    node_index: int
    node_row: int
    node_col: int
    local_anchor: tuple[float, float]
    global_anchor: tuple[float, float]
    window_origin: tuple[int, int]
    window_extent: tuple[int, int]

    total_coverage: int
    other_covering_window_ids: tuple[int, ...]

    other_window_labels: tuple[int, ...]
    other_window_unary_labels: tuple[int, ...]

    source_dissent_label: int
    source_unary_label: int

    consensus_label: Optional[int]
    stitched_label: Optional[int]
    g_equals_d: Optional[bool]
    g_is_third_label: Optional[bool]

    unary_other_agree_count: int
    unary_other_agree_fraction: float
    unary_all_other_agree_y: bool
    unary_all_covering_agree_y: bool

    t0: bool
    t1: bool
    t2: bool
    t3: bool
    actionable: bool
    t4: bool
    t4_prime: bool

    t4_prime_source_score_y: Optional[float]
    t4_prime_source_score_d: Optional[float]
    t4_prime_source_rank_y: Optional[int]
    t4_prime_source_rank_d: Optional[int]

    other_window_tie: bool
    source_post_rwr_tie: bool
    source_unary_tie: bool
    stitched_tie: Optional[bool]

    source_normalized_edge_distance: float
    source_normalized_center_distance: float
    agreeing_window_edge_distances: tuple[float, ...]
    agreeing_window_center_distances: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_type(self.image_id, str, "image_id")
        for name in ("source_window_id", "node_index", "node_row", "node_col",
                     "total_coverage", "source_dissent_label", "source_unary_label",
                     "unary_other_agree_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise T4AuditError(f"{name} must be a non-negative exact integer")
        for name in ("t0", "t1", "t2", "t3", "actionable", "t4", "t4_prime",
                     "unary_all_other_agree_y", "unary_all_covering_agree_y",
                     "other_window_tie", "source_post_rwr_tie", "source_unary_tie"):
            _require_type(getattr(self, name), bool, name)
        if not (0.0 <= self.unary_other_agree_fraction <= 1.0):
            raise T4AuditError("unary_other_agree_fraction must be in [0,1]")

        # Funnel subset invariants: T4 subset actionable subset T3 subset T2
        # subset T1 subset T0; T4_prime subset actionable.
        if self.t1 and not self.t0:
            raise T4AuditError("T1 implies T0")
        if self.t2 and not self.t1:
            raise T4AuditError("T2 implies T1")
        if self.t3 and not self.t2:
            raise T4AuditError("T3 implies T2")
        if self.actionable and not self.t3:
            raise T4AuditError("actionable implies T3")
        if self.t4 and not self.actionable:
            raise T4AuditError("T4 implies actionable")
        if self.t4_prime and not self.actionable:
            raise T4AuditError("T4_prime implies actionable")

        if self.t2 and self.consensus_label is None:
            raise T4AuditError("T2 requires a defined consensus_label")
        if not self.t2 and self.consensus_label is not None and self.t1:
            # T1-but-not-T2 nodes may still report a would-be majority for
            # diagnostics elsewhere, but this module defines consensus_label
            # ONLY when unanimity (T2) holds.
            raise T4AuditError("consensus_label must be None unless T2 holds")
        # g/g_equals_d/g_is_third_label are computed as soon as T3 holds
        # (the stitched score G(anchor) must be evaluated to even determine
        # whether a node is actionable), so they are defined whenever T3
        # holds, not only once a node turns out to be actionable.
        if self.t3 and self.stitched_label is None:
            raise T4AuditError("T3 requires a defined stitched_label (g is computed to test actionable)")
        if not self.t3 and self.stitched_label is not None:
            raise T4AuditError("stitched_label must be None before T3 is reached")
        if self.t4 and self.source_unary_label != self.consensus_label:
            raise T4AuditError("T4 requires source unary u == consensus label y")
        if self.t3 and self.source_dissent_label == self.consensus_label:
            raise T4AuditError("T3 requires source dissent d != consensus label y")
        if self.actionable and self.stitched_label == self.consensus_label:
            raise T4AuditError("actionable requires stitched label g != consensus label y")

        if self.t4_prime:
            if self.t4_prime_source_score_y is None or self.t4_prime_source_score_d is None:
                raise T4AuditError("T4_prime requires recorded S0 scores for y and d")
            if not (self.t4_prime_source_score_y > self.t4_prime_source_score_d):
                raise T4AuditError("T4_prime requires strict S0(y) > S0(d)")

        object.__setattr__(self, "local_anchor", tuple(float(v) for v in self.local_anchor))
        object.__setattr__(self, "global_anchor", tuple(float(v) for v in self.global_anchor))
        object.__setattr__(
            self, "other_covering_window_ids", tuple(int(v) for v in self.other_covering_window_ids)
        )
        object.__setattr__(
            self, "other_window_labels", tuple(int(v) for v in self.other_window_labels)
        )
        object.__setattr__(
            self, "other_window_unary_labels", tuple(int(v) for v in self.other_window_unary_labels)
        )


# ---------------------------------------------------------------------------
# T0-only coverage tally (no full record; kept for the coverage histogram)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class T4FunnelCounts:
    """Exact per-image funnel counts. Immutable; combinable via ``+``."""

    t0: int = 0
    t1: int = 0
    t2: int = 0
    t3: int = 0
    actionable: int = 0
    t4: int = 0
    t4_prime: int = 0

    def __post_init__(self) -> None:
        for name in ("t0", "t1", "t2", "t3", "actionable", "t4", "t4_prime"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise T4AuditError(f"{name} must be a non-negative exact integer")
        if not (self.t4 <= self.actionable <= self.t3 <= self.t2 <= self.t1 <= self.t0):
            raise T4AuditError(
                "funnel counts violate T4<=actionable<=T3<=T2<=T1<=T0: "
                f"{(self.t4, self.actionable, self.t3, self.t2, self.t1, self.t0)}"
            )
        if self.t4_prime > self.actionable:
            raise T4AuditError("T4_prime count must not exceed the actionable count")

    def __add__(self, other: "T4FunnelCounts") -> "T4FunnelCounts":
        if not isinstance(other, T4FunnelCounts):
            return NotImplemented
        return T4FunnelCounts(
            t0=self.t0 + other.t0, t1=self.t1 + other.t1, t2=self.t2 + other.t2,
            t3=self.t3 + other.t3, actionable=self.actionable + other.actionable,
            t4=self.t4 + other.t4, t4_prime=self.t4_prime + other.t4_prime,
        )

    def survival_rates(self) -> dict[str, float]:
        def rate(numerator: int, denominator: int) -> float:
            return (numerator / denominator) if denominator else 0.0

        return {
            "t0_to_t1": rate(self.t1, self.t0),
            "t1_to_t2": rate(self.t2, self.t1),
            "t2_to_t3": rate(self.t3, self.t2),
            "t3_to_actionable": rate(self.actionable, self.t3),
            "actionable_to_t4": rate(self.t4, self.actionable),
        }


# ---------------------------------------------------------------------------
# Centrality metadata (recorded only; never used to filter or weight)
# ---------------------------------------------------------------------------


def _normalized_edge_distance(local_row: Fraction, local_col: Fraction, extent: tuple[int, int]) -> float:
    """Distance (in [0, 1]) from the nearest crop edge; 0 = on the edge,
    ~1 = as far from every edge as this crop allows. Recorded for later
    trust/centrality diagnostics only -- never used here to filter, weight,
    or otherwise influence consensus."""
    height, width = extent
    row_span = max(height - 1, 1)
    col_span = max(width - 1, 1)
    row_distance = min(float(local_row), float(height - 1) - float(local_row)) / row_span
    col_distance = min(float(local_col), float(width - 1) - float(local_col)) / col_span
    return max(0.0, min(1.0, min(row_distance, col_distance)))


def _normalized_center_distance(local_row: Fraction, local_col: Fraction, extent: tuple[int, int]) -> float:
    """Euclidean distance from the crop center, normalized so the crop
    corner is distance 1. Recorded only; never used as a filter."""
    height, width = extent
    center_row = (height - 1) / 2.0
    center_col = (width - 1) / 2.0
    max_distance = math.hypot(center_row, center_col) or 1.0
    distance = math.hypot(float(local_row) - center_row, float(local_col) - center_col)
    return max(0.0, min(1.0, distance / max_distance))


# ---------------------------------------------------------------------------
# GT-free signal builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class T4ImageSignal:
    """GT-free output of the signal builder for one image. Contains
    everything the (separate) GT evaluator needs, and nothing GT-derived.
    ``observations`` holds only T1+ records (nodes with fewer than two
    other covering windows are tallied into ``coverage_histogram`` and
    ``funnel_counts.t0`` without building a full record)."""

    image_id: str
    windows_processed: int
    funnel_counts: T4FunnelCounts
    coverage_histogram: Mapping[int, int]
    observations: tuple[ConsensusObservation, ...]
    tie_counts: Mapping[str, int]
    class_counts_y: Mapping[int, int]
    class_counts_d: Mapping[int, int]
    class_counts_u: Mapping[int, int]
    class_counts_g: Mapping[int, int]
    g_equals_d_count: int
    g_third_label_count: int
    unary_rank_histogram_y: Mapping[int, int]
    unary_rank_histogram_d: Mapping[int, int]
    shared_unary_support_histogram: Mapping[int, int]
    unique_global_anchor_count: int

    def __post_init__(self) -> None:
        _require_type(self.image_id, str, "image_id")
        if not isinstance(self.funnel_counts, T4FunnelCounts):
            raise T4AuditError("funnel_counts must be a T4FunnelCounts")
        object.__setattr__(self, "observations", tuple(self.observations))
        for observation in self.observations:
            if not isinstance(observation, ConsensusObservation):
                raise T4AuditError("observations must contain only ConsensusObservation")
        if len(self.observations) != self.funnel_counts.t2:
            raise T4AuditError(
                "observations count must equal funnel_counts.t2 -- one record is built "
                "for every node with a defined consensus label, whether or not it goes "
                f"on to T3/actionable/T4 (observations={len(self.observations)}, "
                f"t2={self.funnel_counts.t2})"
            )
        object.__setattr__(self, "coverage_histogram", dict(self.coverage_histogram))
        object.__setattr__(self, "tie_counts", dict(self.tie_counts))
        object.__setattr__(self, "class_counts_y", dict(self.class_counts_y))
        object.__setattr__(self, "class_counts_d", dict(self.class_counts_d))
        object.__setattr__(self, "class_counts_u", dict(self.class_counts_u))
        object.__setattr__(self, "class_counts_g", dict(self.class_counts_g))
        object.__setattr__(self, "unary_rank_histogram_y", dict(self.unary_rank_histogram_y))
        object.__setattr__(self, "unary_rank_histogram_d", dict(self.unary_rank_histogram_d))
        object.__setattr__(
            self, "shared_unary_support_histogram", dict(self.shared_unary_support_histogram)
        )


class _Histogram:
    """Tiny mutable int-key counter; converted to a plain dict at the end."""

    __slots__ = ("_counts",)

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}

    def add(self, key: int) -> None:
        self._counts[key] = self._counts.get(key, 0) + 1

    def as_dict(self) -> dict[int, int]:
        return dict(self._counts)


def build_t4_signal_for_image(
    cache: ImageWindowCache,
    context: FirstPassImageContext,
    *,
    image_id: str,
) -> T4ImageSignal:
    """Build the GT-free leave-one-window-out consensus signal for one
    image. Inputs are restricted, by this function's own signature, to the
    sealed cache, the immutable first-pass context, and ``image_id``: there
    is no parameter through which ground truth could reach this function,
    and it captures no GT from any enclosing scope.

    Never reruns the backbone, the graph builder, or CGLS: every score used
    here (``S0``, ``P``) is read directly from the already-sealed cache.
    """
    _require_type(image_id, str, "image_id")
    if cache.state not in (WindowCacheState.SEALED, WindowCacheState.REPLAYING):
        raise T4AuditError(
            f"the two-pass cache must be sealed before auditing, got {cache.state.value}"
        )
    plan = context.plan

    t0 = t1 = t2 = t3 = actionable_count = t4 = t4_prime = 0
    coverage_histogram = _Histogram()
    tie_counts = {"other_window": 0, "source_post_rwr": 0, "source_unary": 0, "stitched": 0}
    class_counts_y = _Histogram()
    class_counts_d = _Histogram()
    class_counts_u = _Histogram()
    class_counts_g = _Histogram()
    g_equals_d_count = 0
    g_third_label_count = 0
    unary_rank_histogram_y = _Histogram()
    unary_rank_histogram_d = _Histogram()
    shared_unary_support_histogram = _Histogram()
    observations: list[ConsensusObservation] = []
    seen_anchors: set[tuple[int, int]] = set()
    anchor_rounding = 1_000_000  # micropixel precision for anchor de-duplication only

    windows = tuple(cache.windows_in_order())

    # Q_v = sigmoid(P_v) is computed once per window (sigmoid on the WHOLE
    # score grid, before any interpolation), then reused across every node
    # that samples from that window -- both for correctness (sigmoid must
    # happen before bilinear sampling, never after) and to avoid recomputing
    # it once per (source node, covering window) pair.
    sigmoid_scores_by_window: dict[int, torch.Tensor] = {}

    def sigmoid_scores_for(window_index: int) -> torch.Tensor:
        cached = sigmoid_scores_by_window.get(window_index)
        if cached is None:
            cached = torch.sigmoid(cache.get(window_index).propagated_scores)
            sigmoid_scores_by_window[window_index] = cached
        return cached

    for source in windows:
        grid_h, grid_w = source.patch_grid_shape
        window = source.geometry
        height, width = window.extent.as_tuple()

        # Precompute, once per window, which grid rows/cols cover each of
        # this window's own row/col anchors -- O(grid_h + grid_w) lookups
        # against O(plan.grid_rows + plan.grid_cols) bands, not O(N) against
        # O(plan.grid_rows * plan.grid_cols).
        row_locals = [_local_anchor_component(r, grid_h, height) for r in range(grid_h)]
        col_locals = [_local_anchor_component(c, grid_w, width) for c in range(grid_w)]
        row_globals = [window.origin.row + local for local in row_locals]
        col_globals = [window.origin.col + local for local in col_locals]
        row_covering = [_covering_grid_rows(plan, y) for y in row_globals]
        col_covering = [_covering_grid_cols(plan, x) for x in col_globals]

        source_d = source.propagated_scores.argmax(dim=1)  # [N]
        source_u = source.s0.argmax(dim=1)  # [N]
        source_d_ties = (
            source.propagated_scores == source.propagated_scores.max(dim=1, keepdim=True).values
        ).sum(dim=1) > 1
        source_u_ties = (
            source.s0 == source.s0.max(dim=1, keepdim=True).values
        ).sum(dim=1) > 1

        for r in range(grid_h):
            for c in range(grid_w):
                node_index = r * grid_w + c
                global_y, global_x = row_globals[r], col_globals[c]
                anchor_key = (
                    round(float(global_y) * anchor_rounding),
                    round(float(global_x) * anchor_rounding),
                )
                seen_anchors.add(anchor_key)

                other_windows = [
                    (grid_row, grid_col)
                    for grid_row in row_covering[r]
                    for grid_col in col_covering[c]
                    if plan.window_at(grid_row, grid_col).index != window.index
                ]
                total_coverage = len(other_windows) + 1
                coverage_histogram.add(total_coverage)
                t0 += 1

                if len(other_windows) < 2:
                    continue  # T1 requires >= 2 OTHER covering windows
                t1 += 1

                d = int(source_d[node_index].item())
                u = int(source_u[node_index].item())

                other_labels: list[int] = []
                other_unary_labels: list[int] = []
                other_window_ids: list[int] = []
                other_tie_any = False
                agreeing_edge_distances: list[float] = []
                agreeing_center_distances: list[float] = []
                q_terms = []  # sampled sigmoid(P) per covering window, including source later

                for grid_row, grid_col in other_windows:
                    other = cache.get(plan.window_at(grid_row, grid_col).index)
                    other_extent = other.geometry.extent.as_tuple()
                    other_grid = other.patch_grid_shape
                    local_y = global_y - other.geometry.origin.row
                    local_x = global_x - other.geometry.origin.col
                    stencil = _bilinear_stencil_from_fraction(
                        local_y, local_x, other_extent, other_grid
                    )
                    sampled_p = _sample_bilinear(other.propagated_scores, other_grid, stencil)
                    sampled_label, sampled_tie = _argmax_with_tie(sampled_p)
                    other_tie_any = other_tie_any or sampled_tie
                    other_labels.append(sampled_label)
                    other_window_ids.append(other.window_index)

                    sampled_unary = _sample_bilinear(other.s0, other_grid, stencil)
                    other_unary_label, _unary_tie = _argmax_with_tie(sampled_unary)
                    other_unary_labels.append(other_unary_label)

                    # sigmoid BEFORE sampling: gather from the already-
                    # sigmoided grid, not sigmoid(sampled raw P) -- those are
                    # not equal, since sigmoid does not commute with the
                    # linear bilinear-interpolation weighted sum.
                    other_q = sigmoid_scores_for(other.window_index)
                    q_terms.append(_sample_bilinear(other_q, other_grid, stencil))

                    edge_distance = _normalized_edge_distance(local_y, local_x, other_extent)
                    center_distance = _normalized_center_distance(local_y, local_x, other_extent)
                    agreeing_edge_distances.append(edge_distance)
                    agreeing_center_distances.append(center_distance)

                if other_tie_any:
                    tie_counts["other_window"] += 1
                if bool(source_d_ties[node_index].item()):
                    tie_counts["source_post_rwr"] += 1
                if bool(source_u_ties[node_index].item()):
                    tie_counts["source_unary"] += 1

                unanimous = len(set(other_labels)) == 1
                if not unanimous:
                    continue  # T2 fails: no defined consensus label; no record built
                t2 += 1
                y = other_labels[0]
                class_counts_y.add(y)

                other_unary_agree = sum(1 for label in other_unary_labels if label == y)
                unary_all_other_agree_y = other_unary_agree == len(other_unary_labels)
                unary_all_covering_agree_y = unary_all_other_agree_y and (u == y)
                shared_unary_support_histogram.add(other_unary_agree)

                class_counts_d.add(d)
                class_counts_u.add(u)

                # From here on, EVERY T2 node gets one record (this fixes an
                # earlier draft that only recorded the actionable subset,
                # which made per-stage GT reporting for T2/T3 impossible).
                # T3/actionable/T4/T4' fields are populated progressively
                # and left at their "not yet reached" default otherwise.
                t3_flag = d != y
                if t3_flag:
                    t3 += 1

                g: Optional[int] = None
                g_tie: Optional[bool] = None
                g_equals_d: Optional[bool] = None
                g_is_third: Optional[bool] = None
                actionable_flag = False
                if t3_flag:
                    # G(anchor) must be computed to even test the actionable
                    # condition, so it (and g_equals_d/g_is_third_label) are
                    # defined for every T3 node, not only actionable ones --
                    # "g == y" for a T3 node is itself a reportable outcome
                    # (uniform stitching already recovered y).
                    q_source = sigmoid_scores_for(window.index)[node_index]
                    stacked = torch.stack(q_terms + [q_source], dim=0)
                    g_scores = stacked.mean(dim=0)
                    g, g_tie = _argmax_with_tie(g_scores)
                    if g_tie:
                        tie_counts["stitched"] += 1
                    class_counts_g.add(g)
                    g_equals_d = g == d
                    g_is_third = (g != y) and (g != d)
                    if g_equals_d:
                        g_equals_d_count += 1
                    if g_is_third:
                        g_third_label_count += 1
                    actionable_flag = g != y
                    if actionable_flag:
                        actionable_count += 1

                t4_flag = False
                t4_prime_flag = False
                s0_y = s0_d = None
                rank_y = rank_d = None
                if actionable_flag:
                    t4_flag = u == y
                    if t4_flag:
                        t4 += 1
                    s0_y = float(source.s0[node_index, y].item())
                    s0_d = float(source.s0[node_index, d].item())
                    _sorted_scores, sorted_indices = torch.sort(source.s0[node_index], descending=True)
                    rank_y = int((sorted_indices == y).nonzero(as_tuple=True)[0].item())
                    rank_d = int((sorted_indices == d).nonzero(as_tuple=True)[0].item())
                    t4_prime_flag = s0_y > s0_d
                    if t4_prime_flag:
                        t4_prime += 1
                        unary_rank_histogram_y.add(rank_y)
                        unary_rank_histogram_d.add(rank_d)

                local_row_fraction = global_y - window.origin.row
                local_col_fraction = global_x - window.origin.col
                observations.append(
                    ConsensusObservation(
                        image_id=image_id,
                        source_window_id=window.index,
                        node_index=node_index,
                        node_row=r,
                        node_col=c,
                        local_anchor=(float(local_row_fraction), float(local_col_fraction)),
                        global_anchor=(float(global_y), float(global_x)),
                        window_origin=window.origin.as_tuple(),
                        window_extent=(height, width),
                        total_coverage=total_coverage,
                        other_covering_window_ids=tuple(other_window_ids),
                        other_window_labels=tuple(other_labels),
                        other_window_unary_labels=tuple(other_unary_labels),
                        source_dissent_label=d,
                        source_unary_label=u,
                        consensus_label=y,
                        stitched_label=g,
                        g_equals_d=g_equals_d,
                        g_is_third_label=g_is_third,
                        unary_other_agree_count=other_unary_agree,
                        unary_other_agree_fraction=(
                            other_unary_agree / len(other_unary_labels) if other_unary_labels else 0.0
                        ),
                        unary_all_other_agree_y=unary_all_other_agree_y,
                        unary_all_covering_agree_y=unary_all_covering_agree_y,
                        t0=True, t1=True, t2=True, t3=t3_flag,
                        actionable=actionable_flag, t4=t4_flag, t4_prime=t4_prime_flag,
                        t4_prime_source_score_y=s0_y,
                        t4_prime_source_score_d=s0_d,
                        t4_prime_source_rank_y=rank_y,
                        t4_prime_source_rank_d=rank_d,
                        other_window_tie=other_tie_any,
                        source_post_rwr_tie=bool(source_d_ties[node_index].item()),
                        source_unary_tie=bool(source_u_ties[node_index].item()),
                        stitched_tie=g_tie,
                        source_normalized_edge_distance=_normalized_edge_distance(
                            local_row_fraction, local_col_fraction, (height, width)
                        ),
                        source_normalized_center_distance=_normalized_center_distance(
                            local_row_fraction, local_col_fraction, (height, width)
                        ),
                        agreeing_window_edge_distances=tuple(agreeing_edge_distances),
                        agreeing_window_center_distances=tuple(agreeing_center_distances),
                    )
                )

    funnel_counts = T4FunnelCounts(
        t0=t0, t1=t1, t2=t2, t3=t3, actionable=actionable_count, t4=t4, t4_prime=t4_prime
    )
    return T4ImageSignal(
        image_id=image_id,
        windows_processed=len(windows),
        funnel_counts=funnel_counts,
        coverage_histogram=coverage_histogram.as_dict(),
        observations=tuple(observations),
        tie_counts=tie_counts,
        class_counts_y=class_counts_y.as_dict(),
        class_counts_d=class_counts_d.as_dict(),
        class_counts_u=class_counts_u.as_dict(),
        class_counts_g=class_counts_g.as_dict(),
        g_equals_d_count=g_equals_d_count,
        g_third_label_count=g_third_label_count,
        unary_rank_histogram_y=unary_rank_histogram_y.as_dict(),
        unary_rank_histogram_d=unary_rank_histogram_d.as_dict(),
        shared_unary_support_histogram=shared_unary_support_histogram.as_dict(),
        unique_global_anchor_count=len(seen_anchors),
    )


# ---------------------------------------------------------------------------
# GT isolation, layer 2: the audit evaluator.
#
# Everything above this point never sees a GT tensor. The functions below
# are the ONLY code in this module permitted to read one, and they only
# ever READ an already-frozen T4ImageSignal -- they cannot mutate it
# (dataclasses are frozen) and therefore cannot let GT influence y, d, u,
# g, coverage, unanimity, or any T0-T4/T4' membership flag: those were all
# fixed before this function was ever called.
# ---------------------------------------------------------------------------


def _require_gt_matches_image_size(gt: torch.Tensor, image_size: SpatialSize) -> None:
    """Shared, eager GT-shape validation.

    Reused by every GT-consuming entry point (both the per-anchor sampler
    below and the image-level evaluator) so the checks cannot diverge, and
    so a caller is rejected up front -- before any observation is
    inspected or iterated -- even when there happen to be zero
    observations to sample against.
    """
    if not isinstance(image_size, SpatialSize):
        raise T4AuditError("image_size must be a SpatialSize")
    if not torch.is_tensor(gt) or gt.ndim != 2:
        raise T4AuditError("gt must be a [H, W] label map")
    if tuple(gt.shape) != image_size.as_tuple():
        raise T4AuditError("gt shape does not match the image size")


def sample_gt_nearest_neighbor(
    gt: torch.Tensor, y: float, x: float, image_size: SpatialSize
) -> int:
    """Deterministic nearest-neighbor GT lookup at a continuous anchor.

    Rounding rule: round-half-up (``floor(v + 0.5)``), independently per
    axis, then clamp to the valid image extent. Never bilinearly
    interpolates a label.
    """
    _require_gt_matches_image_size(gt, image_size)
    if not (math.isfinite(y) and math.isfinite(x)):
        raise T4AuditError("anchor coordinates must be finite")
    row = int(math.floor(y + 0.5))
    col = int(math.floor(x + 0.5))
    row = max(0, min(image_size.height - 1, row))
    col = max(0, min(image_size.width - 1, col))
    return int(gt[row, col].item())


@dataclass(frozen=True)
class T4StageAccuracy:
    """GT-correctness breakdown for one funnel stage's observations."""

    total: int = 0
    ignored: int = 0
    y_correct: int = 0
    d_correct: int = 0
    u_correct: int = 0
    g_total: int = 0
    g_correct: int = 0
    y_correct_d_wrong: int = 0
    y_wrong_d_correct: int = 0
    both_wrong: int = 0
    gt_third_class: int = 0

    def __post_init__(self) -> None:
        for name in (
            "total", "ignored", "y_correct", "d_correct", "u_correct", "g_total",
            "g_correct", "y_correct_d_wrong", "y_wrong_d_correct", "both_wrong",
            "gt_third_class",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise T4AuditError(f"{name} must be a non-negative exact integer")
        if self.y_correct > self.total or self.d_correct > self.total or self.u_correct > self.total:
            raise T4AuditError("per-label correct counts cannot exceed the total")
        if self.g_correct > self.g_total:
            raise T4AuditError("g_correct cannot exceed g_total")

    def __add__(self, other: "T4StageAccuracy") -> "T4StageAccuracy":
        if not isinstance(other, T4StageAccuracy):
            return NotImplemented
        return T4StageAccuracy(
            total=self.total + other.total, ignored=self.ignored + other.ignored,
            y_correct=self.y_correct + other.y_correct, d_correct=self.d_correct + other.d_correct,
            u_correct=self.u_correct + other.u_correct, g_total=self.g_total + other.g_total,
            g_correct=self.g_correct + other.g_correct,
            y_correct_d_wrong=self.y_correct_d_wrong + other.y_correct_d_wrong,
            y_wrong_d_correct=self.y_wrong_d_correct + other.y_wrong_d_correct,
            both_wrong=self.both_wrong + other.both_wrong,
            gt_third_class=self.gt_third_class + other.gt_third_class,
        )

    def y_accuracy(self) -> Optional[float]:
        return (self.y_correct / self.total) if self.total else None

    def d_accuracy(self) -> Optional[float]:
        return (self.d_correct / self.total) if self.total else None

    def u_accuracy(self) -> Optional[float]:
        return (self.u_correct / self.total) if self.total else None

    def g_accuracy(self) -> Optional[float]:
        return (self.g_correct / self.g_total) if self.g_total else None


def _accumulate_stage(
    observations: Sequence[ConsensusObservation],
    gt: torch.Tensor,
    *,
    ignore_label: int,
    image_size: SpatialSize,
) -> T4StageAccuracy:
    total = ignored = y_correct = d_correct = u_correct = g_total = g_correct = 0
    y_correct_d_wrong = y_wrong_d_correct = both_wrong = gt_third_class = 0
    for observation in observations:
        label = sample_gt_nearest_neighbor(
            gt, observation.global_anchor[0], observation.global_anchor[1], image_size
        )
        if label == ignore_label:
            ignored += 1
            continue
        total += 1
        y_ok = label == observation.consensus_label
        d_ok = label == observation.source_dissent_label
        u_ok = label == observation.source_unary_label
        y_correct += int(y_ok)
        d_correct += int(d_ok)
        u_correct += int(u_ok)
        if observation.stitched_label is not None:
            g_total += 1
            g_correct += int(label == observation.stitched_label)
        if y_ok and not d_ok:
            y_correct_d_wrong += 1
        if d_ok and not y_ok:
            y_wrong_d_correct += 1
        if not y_ok and not d_ok:
            both_wrong += 1
        if label != observation.consensus_label and label != observation.source_dissent_label:
            gt_third_class += 1
    return T4StageAccuracy(
        total=total, ignored=ignored, y_correct=y_correct, d_correct=d_correct,
        u_correct=u_correct, g_total=g_total, g_correct=g_correct,
        y_correct_d_wrong=y_correct_d_wrong, y_wrong_d_correct=y_wrong_d_correct,
        both_wrong=both_wrong, gt_third_class=gt_third_class,
    )


@dataclass(frozen=True)
class T4ImageGTEvaluation:
    """Per-image GT correctness, reported separately per funnel stage.

    The actionable-subset consensus precision (``actionable.y_accuracy()``)
    is deliberately a *different* object from the general T2 precision
    (``t2.y_accuracy()``); they must never be combined.
    """

    image_id: str
    t2: T4StageAccuracy
    t3: T4StageAccuracy
    actionable: T4StageAccuracy
    t4: T4StageAccuracy
    t4_prime: T4StageAccuracy

    def __post_init__(self) -> None:
        _require_type(self.image_id, str, "image_id")
        for name in ("t2", "t3", "actionable", "t4", "t4_prime"):
            if not isinstance(getattr(self, name), T4StageAccuracy):
                raise T4AuditError(f"{name} must be a T4StageAccuracy")


def evaluate_t4_signal_against_gt(
    signal: T4ImageSignal,
    gt: torch.Tensor,
    *,
    ignore_label: int,
    image_size: SpatialSize,
) -> T4ImageGTEvaluation:
    """Layer 2 of GT isolation: reads an already-built, frozen signal plus
    GT, and returns correctness statistics. Cannot alter ``signal`` (it is
    immutable) and therefore cannot influence y/d/u/g/coverage/unanimity/
    membership, all of which were already fixed by the GT-free builder."""
    if not isinstance(signal, T4ImageSignal):
        raise T4AuditError("signal must be a T4ImageSignal")
    if isinstance(ignore_label, bool) or not isinstance(ignore_label, int):
        raise T4AuditError("ignore_label must be an exact integer")
    # Eager, unconditional: validated here regardless of how many (if any)
    # observations exist, so a malformed/mismatched gt can never slip
    # through unnoticed on an image with zero T2 observations -- see
    # _require_gt_matches_image_size's docstring.
    _require_gt_matches_image_size(gt, image_size)

    t3_observations = [o for o in signal.observations if o.t3]
    actionable_observations = [o for o in signal.observations if o.actionable]
    t4_observations = [o for o in signal.observations if o.t4]
    t4_prime_observations = [o for o in signal.observations if o.t4_prime]

    return T4ImageGTEvaluation(
        image_id=signal.image_id,
        t2=_accumulate_stage(signal.observations, gt, ignore_label=ignore_label, image_size=image_size),
        t3=_accumulate_stage(t3_observations, gt, ignore_label=ignore_label, image_size=image_size),
        actionable=_accumulate_stage(
            actionable_observations, gt, ignore_label=ignore_label, image_size=image_size
        ),
        t4=_accumulate_stage(t4_observations, gt, ignore_label=ignore_label, image_size=image_size),
        t4_prime=_accumulate_stage(
            t4_prime_observations, gt, ignore_label=ignore_label, image_size=image_size
        ),
    )


# ---------------------------------------------------------------------------
# Streaming accumulator and final summary
# ---------------------------------------------------------------------------


def _merge_int_histogram(target: dict[int, int], addition: Mapping[int, int]) -> None:
    for key, value in addition.items():
        target[key] = target.get(key, 0) + value


@dataclass(frozen=True)
class T4AuditSummary:
    """Final, immutable, streamed-aggregate report. No per-patch records
    are retained here; only counts/distributions accumulated image by
    image."""

    images_processed: int
    windows_processed: int
    unique_global_anchor_count: int
    funnel_counts: T4FunnelCounts
    survival_rates: Mapping[str, float]
    coverage_histogram: Mapping[int, int]
    tie_counts: Mapping[str, int]
    class_counts_y: Mapping[int, int]
    class_counts_d: Mapping[int, int]
    class_counts_u: Mapping[int, int]
    class_counts_g: Mapping[int, int]
    g_equals_d_count: int
    g_third_label_count: int
    unary_rank_histogram_y: Mapping[int, int]
    unary_rank_histogram_d: Mapping[int, int]
    shared_unary_support_histogram: Mapping[int, int]
    gt_evaluated: bool
    ignored_gt_count: int
    gt_t2: Optional[T4StageAccuracy]
    gt_t3: Optional[T4StageAccuracy]
    gt_actionable: Optional[T4StageAccuracy]
    gt_t4: Optional[T4StageAccuracy]
    gt_t4_prime: Optional[T4StageAccuracy]
    delta_trust_actionable: Optional[float]

    def __post_init__(self) -> None:
        for name in ("images_processed", "windows_processed", "unique_global_anchor_count",
                     "g_equals_d_count", "g_third_label_count", "ignored_gt_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise T4AuditError(f"{name} must be a non-negative exact integer")
        if not isinstance(self.funnel_counts, T4FunnelCounts):
            raise T4AuditError("funnel_counts must be a T4FunnelCounts")
        _require_type(self.gt_evaluated, bool, "gt_evaluated")
        if self.gt_evaluated:
            for name in ("gt_t2", "gt_t3", "gt_actionable", "gt_t4", "gt_t4_prime"):
                if not isinstance(getattr(self, name), T4StageAccuracy):
                    raise T4AuditError(f"{name} must be a T4StageAccuracy when gt_evaluated is True")
        else:
            if any(
                getattr(self, name) is not None
                for name in ("gt_t2", "gt_t3", "gt_actionable", "gt_t4", "gt_t4_prime", "delta_trust_actionable")
            ):
                raise T4AuditError("GT fields must be None when gt_evaluated is False")


class T4AuditAccumulator:
    """Mutable, streaming accumulator: absorbs one image's signal (and,
    optionally, one image's GT evaluation) at a time, and releases
    per-image state immediately -- only running counts/histograms are
    retained between images."""

    def __init__(self) -> None:
        self._images = 0
        self._windows = 0
        self._unique_anchors = 0
        self._funnel = T4FunnelCounts()
        self._coverage_histogram: dict[int, int] = {}
        self._tie_counts = {"other_window": 0, "source_post_rwr": 0, "source_unary": 0, "stitched": 0}
        self._class_counts_y: dict[int, int] = {}
        self._class_counts_d: dict[int, int] = {}
        self._class_counts_u: dict[int, int] = {}
        self._class_counts_g: dict[int, int] = {}
        self._g_equals_d_count = 0
        self._g_third_label_count = 0
        self._unary_rank_histogram_y: dict[int, int] = {}
        self._unary_rank_histogram_d: dict[int, int] = {}
        self._shared_unary_support_histogram: dict[int, int] = {}
        self._gt_evaluated = False
        self._gt_t2 = T4StageAccuracy()
        self._gt_t3 = T4StageAccuracy()
        self._gt_actionable = T4StageAccuracy()
        self._gt_t4 = T4StageAccuracy()
        self._gt_t4_prime = T4StageAccuracy()

    def absorb_signal(self, signal: T4ImageSignal) -> None:
        if not isinstance(signal, T4ImageSignal):
            raise T4AuditError("signal must be a T4ImageSignal")
        self._images += 1
        self._windows += signal.windows_processed
        self._unique_anchors += signal.unique_global_anchor_count
        self._funnel = self._funnel + signal.funnel_counts
        _merge_int_histogram(self._coverage_histogram, signal.coverage_histogram)
        for key in self._tie_counts:
            self._tie_counts[key] += signal.tie_counts.get(key, 0)
        _merge_int_histogram(self._class_counts_y, signal.class_counts_y)
        _merge_int_histogram(self._class_counts_d, signal.class_counts_d)
        _merge_int_histogram(self._class_counts_u, signal.class_counts_u)
        _merge_int_histogram(self._class_counts_g, signal.class_counts_g)
        self._g_equals_d_count += signal.g_equals_d_count
        self._g_third_label_count += signal.g_third_label_count
        _merge_int_histogram(self._unary_rank_histogram_y, signal.unary_rank_histogram_y)
        _merge_int_histogram(self._unary_rank_histogram_d, signal.unary_rank_histogram_d)
        _merge_int_histogram(self._shared_unary_support_histogram, signal.shared_unary_support_histogram)

    def absorb_gt_evaluation(self, evaluation: T4ImageGTEvaluation) -> None:
        if not isinstance(evaluation, T4ImageGTEvaluation):
            raise T4AuditError("evaluation must be a T4ImageGTEvaluation")
        self._gt_evaluated = True
        self._gt_t2 = self._gt_t2 + evaluation.t2
        self._gt_t3 = self._gt_t3 + evaluation.t3
        self._gt_actionable = self._gt_actionable + evaluation.actionable
        self._gt_t4 = self._gt_t4 + evaluation.t4
        self._gt_t4_prime = self._gt_t4_prime + evaluation.t4_prime

    def summary(self) -> T4AuditSummary:
        delta_trust = None
        if self._gt_evaluated:
            y_acc = self._gt_actionable.y_accuracy()
            d_acc = self._gt_actionable.d_accuracy()
            if y_acc is not None and d_acc is not None:
                delta_trust = y_acc - d_acc
        return T4AuditSummary(
            images_processed=self._images,
            windows_processed=self._windows,
            unique_global_anchor_count=self._unique_anchors,
            funnel_counts=self._funnel,
            survival_rates=self._funnel.survival_rates(),
            coverage_histogram=dict(self._coverage_histogram),
            tie_counts=dict(self._tie_counts),
            class_counts_y=dict(self._class_counts_y),
            class_counts_d=dict(self._class_counts_d),
            class_counts_u=dict(self._class_counts_u),
            class_counts_g=dict(self._class_counts_g),
            g_equals_d_count=self._g_equals_d_count,
            g_third_label_count=self._g_third_label_count,
            unary_rank_histogram_y=dict(self._unary_rank_histogram_y),
            unary_rank_histogram_d=dict(self._unary_rank_histogram_d),
            shared_unary_support_histogram=dict(self._shared_unary_support_histogram),
            gt_evaluated=self._gt_evaluated,
            ignored_gt_count=self._gt_t2.ignored if self._gt_evaluated else 0,
            gt_t2=self._gt_t2 if self._gt_evaluated else None,
            gt_t3=self._gt_t3 if self._gt_evaluated else None,
            gt_actionable=self._gt_actionable if self._gt_evaluated else None,
            gt_t4=self._gt_t4 if self._gt_evaluated else None,
            gt_t4_prime=self._gt_t4_prime if self._gt_evaluated else None,
            delta_trust_actionable=delta_trust,
        )


# ---------------------------------------------------------------------------
# Integration entry point (Section 11)
#
# Call this AFTER run_pass_one() has sealed the cache and BEFORE run_pass_two()
# / cache.close(). It never touches pass-2 replay, never mutates the cache,
# and is entirely opt-in: a caller that never calls this function pays zero
# cost and sees zero behavior change, because window_cache.py itself has no
# reference to this module (see test_disabled_audit_zero_overhead).
# ---------------------------------------------------------------------------


def run_t4_audit_for_image(
    cache: ImageWindowCache,
    context: FirstPassImageContext,
    *,
    image_id: str,
    gt: Optional[torch.Tensor] = None,
    ignore_label: Optional[int] = None,
) -> tuple[T4ImageSignal, Optional[T4ImageGTEvaluation]]:
    """Build the GT-free signal for one image and, only if ``gt`` is
    supplied, separately evaluate it against GT. Read-only: the cache and
    ``context`` are never modified."""
    signal = build_t4_signal_for_image(cache, context, image_id=image_id)
    if gt is None:
        return signal, None
    if ignore_label is None:
        raise T4AuditError("ignore_label is required whenever gt is provided")
    evaluation = evaluate_t4_signal_against_gt(
        signal, gt, ignore_label=ignore_label, image_size=context.image_size
    )
    return signal, evaluation


__all__ = [
    "ConsensusObservation",
    "T4AuditAccumulator",
    "T4AuditError",
    "T4AuditSummary",
    "T4FunnelCounts",
    "T4ImageGTEvaluation",
    "T4ImageSignal",
    "T4StageAccuracy",
    "build_t4_signal_for_image",
    "covers_continuous_point",
    "evaluate_t4_signal_against_gt",
    "run_t4_audit_for_image",
    "sample_gt_nearest_neighbor",
    "source_score_grid_anchor",
]
