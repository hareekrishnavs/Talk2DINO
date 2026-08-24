"""Native cross-view directed-edge support audit.

Measures whether directed k12 graph edges recur across native overlapping
crop views. For a directed edge ``i -> j`` built inside one source window,
this module asks: among the OTHER windows of the same transformed image
whose native ViT patch grids contain an exact counterpart of both ``i`` and
``j``, how many independently reconstruct the SAME directed edge in their
own top-12 graph? See ``evaluation_identities/e12_native_edge_support_audit.toml``
for the sole authoritative definition of every constant and policy this
module implements.

This module never deletes, reweights, or edits a graph edge, never selects
a T4/T4-prime target, never builds crop-consensus semantic labels, never
touches DCR/SUR, never votes, never prunes (random/lowest-cosine/
structural), never runs a counterfactual/adjoint/Sherman-Morrison solve,
never symmetrizes the graph, never introduces a learned threshold or
confidence/entropy/margin gate, and never interpolates DINO features or
graph topology. Support is topology-only: whether an edge exists, never
weighted by confidence, position, distance, or graph weight. It measures
reachability of a later decision, never evaluates pruning efficacy, and
never renders a pass/fail verdict on segmentation accuracy.

Native coordinate mapping between windows is pure offset-and-bounds
arithmetic identical in effect to
``segmentation.evaluation.sliding_window_geometry.PatchGridSpec.
native_patch_node`` (see that method's docstring for the scalar reference
definition) -- this module never imports that geometry module (mirroring
the duck-typed, dependency-light pattern already used by
``matched_power_evaluator.py`` and ``stitching_control.py`` in this
package), but ``tests/test_native_edge_support_api.py`` cross-validates the
vectorized implementation below against the scalar reference exhaustively.
The key fact this module exploits for vectorization: the displacement
between two window origins is constant across every node in the grid, so
alignment (divisibility by the patch size) and the resulting integer
row/column offset are properties of the WINDOW PAIR, not of the individual
node -- only whether the offset keeps a given node's mapped coordinate
inside the target's grid varies node-to-node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


class NativeEdgeSupportError(RuntimeError):
    """Raised on any native-edge-support-audit invariant violation. Always
    fail closed: never silently substitute a fallback mapping, tolerance,
    or interpolated coordinate."""


UNDEFINED_REASON_SINGLE_WINDOW_IMAGE = "single_window_image"
UNDEFINED_REASON_NO_ALIGNED_OBSERVER = "no_exactly_aligned_observer_covering_both_endpoints"
UNDEFINED_REASONS: tuple[str, ...] = (
    UNDEFINED_REASON_SINGLE_WINDOW_IMAGE,
    UNDEFINED_REASON_NO_ALIGNED_OBSERVER,
)

CROP_EDGE_BAND_LABELS: tuple[str, ...] = ("0", "1", "2", "3-4", "5-7", ">=8")


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise NativeEdgeSupportError(f"{label} must be a positive exact integer")
    return value


def crop_edge_band(distance_patches: int) -> str:
    """Identity-locked crop-edge distance band, in patch units."""
    if type(distance_patches) is not int or distance_patches < 0:
        raise NativeEdgeSupportError("distance_patches must be a non-negative exact integer")
    if distance_patches == 0:
        return "0"
    if distance_patches == 1:
        return "1"
    if distance_patches == 2:
        return "2"
    if distance_patches <= 4:
        return "3-4"
    if distance_patches <= 7:
        return "5-7"
    return ">=8"


# ---------------------------------------------------------------------------
# Native node coordinate / mapping records (Section 4 conceptual types)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeNodeCoordinate:
    """One discrete native ViT patch-grid node inside one window."""

    window_index: int
    patch_row: int
    patch_col: int
    grid_size: int

    def __post_init__(self) -> None:
        if type(self.window_index) is not int or self.window_index < 0:
            raise NativeEdgeSupportError("window_index must be a non-negative exact integer")
        _require_positive_int(self.grid_size, "grid_size")
        for name in ("patch_row", "patch_col"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < self.grid_size:
                raise NativeEdgeSupportError(f"{name} must be an exact integer in [0, grid_size)")

    @property
    def flat_index(self) -> int:
        return self.patch_row * self.grid_size + self.patch_col


@dataclass(frozen=True)
class NativeNodeMapping:
    """The result of mapping one source node into one target window's
    native patch grid: either an exact counterpart node, or ``None`` with
    a persisted reason. Never a float centre comparison, never a nearest
    or interpolated node."""

    source: NativeNodeCoordinate
    target_window_index: int
    target: NativeNodeCoordinate | None
    aligned: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source, NativeNodeCoordinate):
            raise NativeEdgeSupportError("source must be a NativeNodeCoordinate")
        if type(self.target_window_index) is not int or self.target_window_index < 0:
            raise NativeEdgeSupportError("target_window_index must be a non-negative exact integer")
        if self.aligned and self.target is None:
            raise NativeEdgeSupportError("aligned mapping must carry a target node")
        if not self.aligned and self.target is not None:
            raise NativeEdgeSupportError("unaligned mapping must not carry a target node")
        if self.target is not None and not isinstance(self.target, NativeNodeCoordinate):
            raise NativeEdgeSupportError("target must be a NativeNodeCoordinate or None")


def native_window_pair_offset(
    source_origin_row: int, source_origin_col: int, target_origin_row: int, target_origin_col: int,
    *, patch_size: int,
) -> tuple[int, int] | None:
    """Scalar reference: the constant integer ``(row, col)`` patch-unit
    offset mapping every node of the source window into the target
    window's grid, or ``None`` when the origin displacement is not an
    exact multiple of ``patch_size`` along both axes. Equivalent to
    ``PatchGridSpec.native_patch_node``'s alignment test, expressed once
    per window pair (never per node) since the displacement is constant
    across every node in the grid."""
    _require_positive_int(patch_size, "patch_size")
    displacement_row = source_origin_row - target_origin_row
    displacement_col = source_origin_col - target_origin_col
    if displacement_row % patch_size != 0 or displacement_col % patch_size != 0:
        return None
    return displacement_row // patch_size, displacement_col // patch_size


def map_native_node(
    source: NativeNodeCoordinate, *, target_window_index: int,
    source_origin_row: int, source_origin_col: int, target_origin_row: int, target_origin_col: int,
    patch_size: int,
) -> NativeNodeMapping:
    """Scalar per-node mapping, used for unit tests and the bounded-debug
    sample only -- the bulk audit uses the vectorized offset arithmetic in
    :func:`compute_window_support`, which is exhaustively cross-validated
    against this function in ``tests/test_native_edge_support_api.py``."""
    offset = native_window_pair_offset(
        source_origin_row, source_origin_col, target_origin_row, target_origin_col, patch_size=patch_size,
    )
    if offset is None:
        return NativeNodeMapping(source=source, target_window_index=target_window_index, target=None, aligned=False)
    dr, dc = offset
    target_row, target_col = source.patch_row + dr, source.patch_col + dc
    if not (0 <= target_row < source.grid_size and 0 <= target_col < source.grid_size):
        return NativeNodeMapping(source=source, target_window_index=target_window_index, target=None, aligned=False)
    target = NativeNodeCoordinate(
        window_index=target_window_index, patch_row=target_row, patch_col=target_col, grid_size=source.grid_size,
    )
    return NativeNodeMapping(source=source, target_window_index=target_window_index, target=target, aligned=True)


# ---------------------------------------------------------------------------
# Directed edge support record and deterministic descriptive ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectedEdgeSupport:
    """One directed top-12 edge's cross-view support record. Topology-only:
    ``support_fraction`` is never used to gate, weight, or filter anything
    inside this module -- it is a descriptive statistic reported alongside
    the edge, nothing more."""

    source_window_index: int
    source_node: int
    neighbor_rank: int
    destination_node: int
    source_graph_weight: float
    observer_count: int
    support_count: int
    support_fraction: float | None
    defined: bool
    undefined_reason: str | None

    def __post_init__(self) -> None:
        if type(self.source_window_index) is not int or self.source_window_index < 0:
            raise NativeEdgeSupportError("source_window_index must be a non-negative exact integer")
        if type(self.source_node) is not int or self.source_node < 0:
            raise NativeEdgeSupportError("source_node must be a non-negative exact integer")
        if type(self.neighbor_rank) is not int or not 0 <= self.neighbor_rank < 12:
            raise NativeEdgeSupportError("neighbor_rank must be an exact integer in [0, 12)")
        if type(self.destination_node) is not int or self.destination_node < 0:
            raise NativeEdgeSupportError("destination_node must be a non-negative exact integer")
        if not isinstance(self.source_graph_weight, float) or not math.isfinite(self.source_graph_weight):
            raise NativeEdgeSupportError("source_graph_weight must be a finite float")
        if type(self.observer_count) is not int or self.observer_count < 0:
            raise NativeEdgeSupportError("observer_count must be a non-negative exact integer")
        if type(self.support_count) is not int or self.support_count < 0:
            raise NativeEdgeSupportError("support_count must be a non-negative exact integer")
        if self.support_count > self.observer_count:
            raise NativeEdgeSupportError("support_count must not exceed observer_count")
        if type(self.defined) is not bool:
            raise NativeEdgeSupportError("defined must be an exact boolean")
        if self.defined != (self.observer_count > 0):
            raise NativeEdgeSupportError("defined must be exactly (observer_count > 0)")
        if self.defined:
            if self.undefined_reason is not None:
                raise NativeEdgeSupportError("a defined edge must not carry an undefined_reason")
            if self.support_fraction is None:
                raise NativeEdgeSupportError("a defined edge must carry a support_fraction")
            expected = self.support_count / self.observer_count
            if abs(self.support_fraction - expected) > 1e-9:
                raise NativeEdgeSupportError("support_fraction disagrees with support_count / observer_count")
        else:
            if self.support_fraction is not None:
                raise NativeEdgeSupportError("an undefined edge must not carry a support_fraction")
            if self.undefined_reason not in UNDEFINED_REASONS:
                raise NativeEdgeSupportError(f"undefined_reason must be one of {UNDEFINED_REASONS}")


_UNDEFINED_FRACTION_SORT_KEY = 2.0  # strictly greater than any real fraction in [0, 1]
_UNDEFINED_SUPPORT_COUNT_SORT_KEY = 1 << 30


def _rank_sort_key(edge: DirectedEdgeSupport) -> tuple:
    return (
        0 if edge.defined else 1,
        edge.support_fraction if edge.defined else _UNDEFINED_FRACTION_SORT_KEY,
        edge.support_count if edge.defined else _UNDEFINED_SUPPORT_COUNT_SORT_KEY,
        -edge.observer_count,
        edge.source_graph_weight,
        -edge.neighbor_rank,
        edge.destination_node,
    )


def rank_edges_for_row(edges: Sequence[DirectedEdgeSupport]) -> tuple[DirectedEdgeSupport, ...]:
    """Frozen lexicographic descriptive ranking (see module docstring /
    ``evaluation_identities/e12_native_edge_support_audit.toml``
    ``[ranking]``): defined before undefined; lower support fraction
    first; lower support count first; larger observer count first (zero
    of more observations is stronger evidence than zero of fewer); lower
    source weight first; larger neighbor rank first; lower destination
    node index as the final deterministic tie-break. Descriptive only --
    never used to select an edge for deletion or any other edit."""
    if not edges:
        raise NativeEdgeSupportError("rank_edges_for_row requires at least one edge")
    return tuple(sorted(edges, key=_rank_sort_key))


# ---------------------------------------------------------------------------
# Row / window / image support summaries (Section 4)
# ---------------------------------------------------------------------------

SUPPORT_FRACTION_BUCKETS: tuple[str, ...] = (
    "0.0", "(0.0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]",
    "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]", "(0.8,0.9]", "(0.9,1.0]",
)


def _support_fraction_bucket(fraction: float) -> str:
    if fraction == 0.0:
        return "0.0"
    index = min(int(math.ceil(fraction * 10)), 10)
    return SUPPORT_FRACTION_BUCKETS[index]


@dataclass(frozen=True)
class RowSupportSummary:
    """One source-window patch-grid row's (node ``i``'s) outgoing top-12
    edge population, summarized. Never deletes the least-supported edge --
    only reports which one it would be under the frozen descriptive
    ranking."""

    window_index: int
    source_node: int
    total_outgoing_edges: int
    num_support_defined: int
    num_undefined: int
    observer_count_min: int
    observer_count_max: int
    observer_count_mean: float
    support_count_histogram: Mapping[int, int]
    support_fraction_histogram: Mapping[str, int]
    any_defined: bool
    all_defined: bool
    any_defined_zero_support: bool
    least_supported_edge_rank: int | None
    least_supported_edge_destination_node: int | None
    least_support_tied: bool | None

    def __post_init__(self) -> None:
        if self.total_outgoing_edges != 12:
            raise NativeEdgeSupportError("total_outgoing_edges must be exactly 12")
        if self.num_support_defined + self.num_undefined != 12:
            raise NativeEdgeSupportError("num_support_defined + num_undefined must equal 12")
        if self.any_defined != (self.num_support_defined > 0):
            raise NativeEdgeSupportError("any_defined must be exactly (num_support_defined > 0)")
        if self.all_defined != (self.num_support_defined == 12):
            raise NativeEdgeSupportError("all_defined must be exactly (num_support_defined == 12)")
        if self.any_defined and self.least_support_tied is None:
            raise NativeEdgeSupportError("least_support_tied must be set (not None) whenever any_defined is true")
        if not self.any_defined and self.least_support_tied is not None:
            raise NativeEdgeSupportError("least_support_tied must be None when no edge is support-defined")


def build_row_summary(edges: Sequence[DirectedEdgeSupport]) -> RowSupportSummary:
    if len(edges) != 12:
        raise NativeEdgeSupportError(f"build_row_summary requires exactly 12 edges, observed {len(edges)}")
    window_indices = {e.source_window_index for e in edges}
    source_nodes = {e.source_node for e in edges}
    if len(window_indices) != 1 or len(source_nodes) != 1:
        raise NativeEdgeSupportError("all 12 edges of one row must share the same window_index and source_node")
    ranks = sorted(e.neighbor_rank for e in edges)
    if ranks != list(range(12)):
        raise NativeEdgeSupportError("a row's 12 edges must cover neighbor_rank 0..11 exactly once each")

    defined_edges = [e for e in edges if e.defined]
    num_defined = len(defined_edges)
    observer_counts = [e.observer_count for e in edges]

    support_count_histogram = {n: 0 for n in range(13)}
    for e in defined_edges:
        support_count_histogram[e.support_count] += 1

    support_fraction_histogram = {bucket: 0 for bucket in SUPPORT_FRACTION_BUCKETS}
    for e in defined_edges:
        support_fraction_histogram[_support_fraction_bucket(e.support_fraction)] += 1

    ranked = rank_edges_for_row(edges)
    least_supported = ranked[0]

    least_support_tied: bool | None = None
    if num_defined > 0:
        # Tied on the primary evidence criteria (fraction, support_count,
        # observer_count) -- i.e. before the weight/rank/destination
        # tiebreaks that only exist to produce a single deterministic
        # answer, never because the underlying evidence actually differs.
        primary_key = (least_supported.support_fraction, least_supported.support_count, least_supported.observer_count)
        tied_count = sum(
            1 for e in defined_edges
            if (e.support_fraction, e.support_count, e.observer_count) == primary_key
        )
        least_support_tied = tied_count > 1

    return RowSupportSummary(
        window_index=next(iter(window_indices)),
        source_node=next(iter(source_nodes)),
        total_outgoing_edges=12,
        num_support_defined=num_defined,
        num_undefined=12 - num_defined,
        observer_count_min=min(observer_counts),
        observer_count_max=max(observer_counts),
        observer_count_mean=sum(observer_counts) / 12.0,
        support_count_histogram=support_count_histogram,
        support_fraction_histogram=support_fraction_histogram,
        any_defined=num_defined > 0,
        all_defined=num_defined == 12,
        any_defined_zero_support=any(e.defined and e.support_count == 0 for e in edges),
        least_supported_edge_rank=least_supported.neighbor_rank,
        least_supported_edge_destination_node=least_supported.destination_node,
        least_support_tied=least_support_tied,
    )


@dataclass(frozen=True)
class WindowSupportSummary:
    """One window's aggregate over all of its patch-grid rows."""

    window_index: int
    is_clamped: bool
    aligned_observer_window_count: int
    num_rows: int
    num_rows_any_defined: int
    num_rows_all_defined: int
    num_rows_any_defined_zero_support: int

    def __post_init__(self) -> None:
        if self.num_rows_any_defined > self.num_rows or self.num_rows_all_defined > self.num_rows_any_defined:
            raise NativeEdgeSupportError("window support-row subset counts violate the containment invariant")


def build_window_summary(
    *, window_index: int, is_clamped: bool, aligned_observer_window_count: int, row_summaries: Sequence[RowSupportSummary],
) -> WindowSupportSummary:
    if not row_summaries:
        raise NativeEdgeSupportError("build_window_summary requires at least one row summary")
    return WindowSupportSummary(
        window_index=window_index,
        is_clamped=is_clamped,
        aligned_observer_window_count=aligned_observer_window_count,
        num_rows=len(row_summaries),
        num_rows_any_defined=sum(1 for r in row_summaries if r.any_defined),
        num_rows_all_defined=sum(1 for r in row_summaries if r.all_defined),
        num_rows_any_defined_zero_support=sum(1 for r in row_summaries if r.any_defined_zero_support),
    )


# ---------------------------------------------------------------------------
# Vectorized cross-view support computation (Section 3 / Section 7)
# ---------------------------------------------------------------------------


def compute_dense_adjacency(graph: Any) -> torch.Tensor:
    """``[N, N]`` boolean adjacency from one window's own
    :class:`DirectedTopKGraph`, via its own ``to_dense()`` (explicitly
    documented there as safe for diagnostics -- one 32x32-grid graph's
    dense form is 1024x1024, a few MB, never materialized dataset-wide)."""
    return graph.to_dense() > 0


def compute_window_support(
    source_index: int, *, window_origins: Sequence[tuple[int, int]], graphs: Sequence[Any],
    dense_adjacencies: Sequence[torch.Tensor], grid_size: int, patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[bool, ...]]:
    """Vectorized cross-view support for every one of ``source_index``'s
    outgoing edges (shape ``[N, 12]``). ``window_origins[i] = (row, col)``
    pixel origin of window ``i``. Returns ``(observer_count, support_count,
    reverse_support_count, aligned_mask)``: the first three are int64
    tensors of shape ``[N, 12]``; ``aligned_mask`` is a tuple of length
    ``len(window_origins)`` marking which OTHER windows are natively
    aligned with the source window's origin (regardless of whether any
    particular edge's endpoints fall inside the aligned sub-region).

    ``reverse_support_count`` tallies, for informational/diagnostic
    purposes only (the "direction-reversal-only" funnel case -- never used
    to define or gate support itself), how many observers contain the
    REVERSE edge ``j_v -> i_v`` -- computed from the exact same adjacency
    lookups already performed for the forward direction, just swapping the
    two index tensors, so it costs nothing extra to derive.

    Exploits that the origin displacement -- and hence the constant
    integer patch-unit offset mapping every node -- is the same for every
    node in the source window's grid; only per-edge boundary containment
    varies. Never rebuilds or reruns a graph: this function only reads the
    already-built ``graphs``/``dense_adjacencies`` passed in."""
    graph = graphs[source_index]
    n_windows = len(window_origins)
    if not (len(graphs) == len(dense_adjacencies) == n_windows):
        raise NativeEdgeSupportError("window_origins, graphs, and dense_adjacencies must have the same length")
    num_nodes, k = graph.num_nodes, graph.k
    if num_nodes != grid_size * grid_size:
        raise NativeEdgeSupportError("graph.num_nodes disagrees with grid_size * grid_size")
    device = graph.neighbor_indices.device

    i_flat = torch.arange(num_nodes, device=device).unsqueeze(1).expand(num_nodes, k)
    j_flat = graph.neighbor_indices
    i_row, i_col = torch.div(i_flat, grid_size, rounding_mode="floor"), i_flat % grid_size
    j_row, j_col = torch.div(j_flat, grid_size, rounding_mode="floor"), j_flat % grid_size

    observer_count = torch.zeros((num_nodes, k), dtype=torch.int64, device=device)
    support_count = torch.zeros((num_nodes, k), dtype=torch.int64, device=device)
    reverse_support_count = torch.zeros((num_nodes, k), dtype=torch.int64, device=device)
    aligned_mask: list[bool] = [False] * n_windows

    source_origin_row, source_origin_col = window_origins[source_index]
    for t in range(n_windows):
        if t == source_index:
            continue
        target_origin_row, target_origin_col = window_origins[t]
        offset = native_window_pair_offset(
            source_origin_row, source_origin_col, target_origin_row, target_origin_col, patch_size=patch_size,
        )
        if offset is None:
            continue
        aligned_mask[t] = True
        dr, dc = offset

        i_row_v, i_col_v = i_row + dr, i_col + dc
        j_row_v, j_col_v = j_row + dr, j_col + dc
        i_valid = (i_row_v >= 0) & (i_row_v < grid_size) & (i_col_v >= 0) & (i_col_v < grid_size)
        j_valid = (j_row_v >= 0) & (j_row_v < grid_size) & (j_col_v >= 0) & (j_col_v < grid_size)
        both_valid = i_valid & j_valid
        observer_count += both_valid.to(torch.int64)

        adjacency_t = dense_adjacencies[t]
        i_v_flat = i_row_v.clamp(0, grid_size - 1) * grid_size + i_col_v.clamp(0, grid_size - 1)
        j_v_flat = j_row_v.clamp(0, grid_size - 1) * grid_size + j_col_v.clamp(0, grid_size - 1)
        supported = adjacency_t[i_v_flat, j_v_flat] & both_valid
        support_count += supported.to(torch.int64)
        reverse_supported = adjacency_t[j_v_flat, i_v_flat] & both_valid
        reverse_support_count += reverse_supported.to(torch.int64)

    if torch.any(support_count > observer_count) or torch.any(reverse_support_count > observer_count):
        raise NativeEdgeSupportError("computed support_count exceeds observer_count -- internal invariant violated")
    return observer_count, support_count, reverse_support_count, tuple(aligned_mask)


def build_edge_records_for_row(
    *, window_index: int, source_node: int, neighbor_indices_row: Sequence[int], weights_row: Sequence[float],
    observer_count_row: Sequence[int], support_count_row: Sequence[int], window_count: int,
) -> tuple[DirectedEdgeSupport, ...]:
    if not (len(neighbor_indices_row) == len(weights_row) == len(observer_count_row) == len(support_count_row) == 12):
        raise NativeEdgeSupportError("row arrays must all have length 12")
    edges = []
    for rank in range(12):
        observer_count = int(observer_count_row[rank])
        support_count = int(support_count_row[rank])
        defined = observer_count > 0
        if defined:
            fraction: float | None = support_count / observer_count
            reason: str | None = None
        else:
            fraction = None
            reason = UNDEFINED_REASON_SINGLE_WINDOW_IMAGE if window_count <= 1 else UNDEFINED_REASON_NO_ALIGNED_OBSERVER
        edges.append(
            DirectedEdgeSupport(
                source_window_index=window_index, source_node=source_node, neighbor_rank=rank,
                destination_node=int(neighbor_indices_row[rank]), source_graph_weight=float(weights_row[rank]),
                observer_count=observer_count, support_count=support_count, support_fraction=fraction,
                defined=defined, undefined_reason=reason,
            )
        )
    return tuple(edges)


# ---------------------------------------------------------------------------
# GT reachability diagnostics (Section 8): computed strictly AFTER support
# records are frozen. Never influences support computation or ranking.
# ---------------------------------------------------------------------------


def node_class_from_scores(patch_scores_row: torch.Tensor) -> int:
    """The class at one patch node directly from its own propagated score
    row: ``argmax_c(sigmoid(scores[c]))``. Sigmoid is monotonic per class
    channel and does not depend on any other channel, so it never changes
    which class attains the argmax -- applying it here is purely
    documentary of the production probability-space contract, not
    numerically required. No interpolation: this is the node's own score,
    never a neighboring node's or a blended value."""
    return int(torch.argmax(patch_scores_row).item())


def native_patch_centre_pixel(
    window_origin_row: int, window_origin_col: int, patch_row: int, patch_col: int, *, patch_size: int,
) -> tuple[int, int]:
    """The physical pixel-index centre of one patch-grid node's footprint,
    in the transformed (processed/window) image's own pixel space:
    ``centre = origin + patch_size * index + patch_size // 2`` -- the
    identity-declared convention (``patch_size=14`` gives exactly
    ``origin + 14*index + 7``), integer-only, never a float/rounded
    comparison."""
    return (
        window_origin_row + patch_row * patch_size + patch_size // 2,
        window_origin_col + patch_col * patch_size + patch_size // 2,
    )


def rescale_pixel_align_corners(
    row: int, col: int, *, from_hw: tuple[int, int], to_hw: tuple[int, int],
) -> tuple[int, int]:
    """Map one integer pixel-index coordinate from the processed/window
    image's pixel space (``from_hw``) into the dataset's ``ori_shape``
    pixel space (``to_hw``), using the SAME ``align_corners=True`` linear
    convention already applied by ``finalize_prediction``'s bilinear
    rescale -- never a newly-invented mapping. Used only to locate the GT
    pixel for a diagnostic reachability indicator; never used for support
    computation, ranking, or stitching."""
    from_h, from_w = from_hw
    to_h, to_w = to_hw

    def _axis(p: int, from_extent: int, to_extent: int) -> int:
        if from_extent <= 1 or to_extent <= 1:
            return 0
        scaled = p * (to_extent - 1) / (from_extent - 1)
        return min(max(round(scaled), 0), to_extent - 1)

    return _axis(row, from_h, to_h), _axis(col, from_w, to_w)


@dataclass(frozen=True)
class RowCorrectnessRecord:
    """GT-derived correctness indicators for one source-window row,
    computed and joined AFTER the corresponding :class:`RowSupportSummary`
    is already frozen -- GT never feeds back into support computation or
    the descriptive ranking."""

    window_index: int
    source_node: int
    gt_ignored: bool
    source_row_misclassified: bool | None
    canonical_stitched_misclassified: bool | None

    def __post_init__(self) -> None:
        if self.gt_ignored and (self.source_row_misclassified is not None or self.canonical_stitched_misclassified is not None):
            raise NativeEdgeSupportError("an ignored-GT row must not carry a correctness verdict")
        if not self.gt_ignored and (self.source_row_misclassified is None or self.canonical_stitched_misclassified is None):
            raise NativeEdgeSupportError("a non-ignored row must carry both correctness verdicts")


def build_row_correctness_record(
    *, window_index: int, source_node: int, window_origin_row: int, window_origin_col: int,
    patch_row: int, patch_col: int, patch_size: int, source_patch_scores_row: torch.Tensor,
    canonical_stitched_label_map: torch.Tensor, gt_label_map: torch.Tensor, ignore_index: int,
    processed_hw: tuple[int, int], ori_hw: tuple[int, int],
) -> RowCorrectnessRecord:
    processed_row, processed_col = native_patch_centre_pixel(
        window_origin_row, window_origin_col, patch_row, patch_col, patch_size=patch_size,
    )
    ori_row, ori_col = rescale_pixel_align_corners(processed_row, processed_col, from_hw=processed_hw, to_hw=ori_hw)
    gt_class = int(gt_label_map[ori_row, ori_col])
    if gt_class == ignore_index:
        return RowCorrectnessRecord(
            window_index=window_index, source_node=source_node, gt_ignored=True,
            source_row_misclassified=None, canonical_stitched_misclassified=None,
        )
    source_class = node_class_from_scores(source_patch_scores_row)
    stitched_class = int(canonical_stitched_label_map[ori_row, ori_col])
    return RowCorrectnessRecord(
        window_index=window_index, source_node=source_node, gt_ignored=False,
        source_row_misclassified=source_class != gt_class,
        canonical_stitched_misclassified=stitched_class != gt_class,
    )


# ---------------------------------------------------------------------------
# Shared per-window execution (Section 7): one backbone/snapshot pass, one
# k12 graph build, and -- only when diagnostics are requested -- one T=320
# propagation and one probability interpolation, reusing
# ``build_directed_topk_graph``/``finite_step_propagate`` unmodified from
# this package. Never a second graph or propagation for the same window.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeEdgeSupportWindowTelemetry:
    backbone_snapshot_calls: int
    graph_builds: int
    propagation_calls: int
    probability_interpolation_calls: int

    def __post_init__(self) -> None:
        fixed = {"backbone_snapshot_calls": 1, "graph_builds": 1}
        for name, expected in fixed.items():
            observed = getattr(self, name)
            if observed != expected:
                raise NativeEdgeSupportError(f"NativeEdgeSupportWindowTelemetry.{name} must be exactly {expected}, observed {observed!r}")
        for name in ("propagation_calls", "probability_interpolation_calls"):
            observed = getattr(self, name)
            if observed not in (0, 1):
                raise NativeEdgeSupportError(f"NativeEdgeSupportWindowTelemetry.{name} must be 0 or 1, observed {observed!r}")


@dataclass(frozen=True)
class WindowAuditContext:
    """One window's already-computed, immutable state needed for cross-view
    support comparison and (optionally) GT reachability diagnostics. Never
    rebuilt: retained only for the lifetime of the image that produced it,
    then released -- see :mod:`diagnostics.run_native_edge_support_audit`,
    which discards this after each image completes."""

    window_index: int
    origin: tuple[int, int]
    is_clamped: bool
    graph: Any
    grid_hw: tuple[int, int]
    patch_scores: torch.Tensor | None
    probability_crop: torch.Tensor | None


def process_one_window_for_audit(
    inference: Any, image_tensor: torch.Tensor, window: Any, *,
    k: int, alpha: float, steps: int, affinity_power: float, need_diagnostics: bool,
) -> tuple[WindowAuditContext, NativeEdgeSupportWindowTelemetry]:
    """Run the shared per-window flow exactly once: one backbone/snapshot
    pass, one canonical directed top-k graph build. When
    ``need_diagnostics`` is true, additionally runs one finite-step
    propagation and one sigmoid+interpolation to produce the probability
    crop this window contributes to the canonical uniform-probability
    stitched map (used only for the ``canonical_stitched_misclassified``
    reachability indicator, computed after support is already frozen).
    When false, propagation is skipped entirely: it is never required for
    support computation itself."""
    from .graph import build_directed_topk_graph

    crop_rows, crop_cols = window.crop_slice
    crop = image_tensor[:, :, crop_rows, crop_cols]

    snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
    dino_features = snapshot.dino_features[0]
    grid_hw = snapshot.grid_hw

    graph12 = build_directed_topk_graph(dino_features, k=k, affinity_power=affinity_power)

    patch_scores: torch.Tensor | None = None
    probability_crop: torch.Tensor | None = None
    propagation_calls = 0
    if need_diagnostics:
        from .finite_step_regime import finite_step_propagate

        s0 = snapshot.unary_scores[0]
        trace = finite_step_propagate(graph12, s0, alpha=alpha, steps=steps, snapshot_steps=(steps,))
        patch_scores = trace.snapshots[steps]
        crop_hw = window.extent.as_tuple()
        probability_crop = inference.model.masks_from_patch_scores(patch_scores.unsqueeze(0), grid_hw, crop_hw)
        propagation_calls = 1

    context = WindowAuditContext(
        window_index=int(window.index), origin=(int(window.origin.row), int(window.origin.col)),
        is_clamped=bool(window.clamped_vertical or window.clamped_horizontal),
        graph=graph12, grid_hw=tuple(grid_hw), patch_scores=patch_scores, probability_crop=probability_crop,
    )
    telemetry = NativeEdgeSupportWindowTelemetry(
        backbone_snapshot_calls=1, graph_builds=1, propagation_calls=propagation_calls,
        probability_interpolation_calls=1 if need_diagnostics else 0,
    )
    return context, telemetry


__all__ = [
    "CROP_EDGE_BAND_LABELS",
    "DirectedEdgeSupport",
    "NativeEdgeSupportError",
    "NativeNodeCoordinate",
    "NativeNodeMapping",
    "NativeEdgeSupportWindowTelemetry",
    "RowCorrectnessRecord",
    "RowSupportSummary",
    "SUPPORT_FRACTION_BUCKETS",
    "UNDEFINED_REASONS",
    "UNDEFINED_REASON_NO_ALIGNED_OBSERVER",
    "UNDEFINED_REASON_SINGLE_WINDOW_IMAGE",
    "WindowAuditContext",
    "WindowSupportSummary",
    "build_edge_records_for_row",
    "build_row_correctness_record",
    "build_row_summary",
    "build_window_summary",
    "compute_dense_adjacency",
    "compute_window_support",
    "crop_edge_band",
    "map_native_node",
    "native_patch_centre_pixel",
    "native_window_pair_offset",
    "node_class_from_scores",
    "process_one_window_for_audit",
    "rank_edges_for_row",
    "rescale_pixel_align_corners",
]
