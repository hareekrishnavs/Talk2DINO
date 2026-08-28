"""DFP Phase 1A Step 9: a pure, static estimator for the arithmetic
width, operation count, and working-memory requirements of the
*proposed* Phase 1B exact chunked masked-forward propagation.

This module never runs the propagation it estimates. It consumes
already-derived structural metadata (Step 5's canonical stitched score
fields, Step 6's global component partition, Step 8's soft
memberships, and caller-supplied per-window graph metadata) and
produces exact integer counts, a declared FLOP-counting convention, a
transparent byte-counting model, and hardware-independent feasibility
against an optional memory budget. It fabricates no runtime, no
hardware calibration, and no real-dataset percentile.

Dispute set D and query classes Q_w
-------------------------------------
The frozen identity's ``[dispute_protocol]`` defines
``a(x) = lowest_index_argmax_of_S0_stitched_prediction``,
``b(x) = lowest_index_argmax_of_S320_stitched_prediction``, and
``D = {x : a(x) != b(x)}, restricted to non-ignore evaluation pixels``.
That restriction concerns the *metric-computation* use of D, where
ground truth is available; this estimator has no ground-truth input by
design (per this step's own explicit instructions) and therefore
computes D over the *complete* stitched-image domain, without any
ignore-mask restriction. Since dropping that restriction can only ever
*add* pixels to D (never remove any), this module's D is a safe,
conservative superset of the metric-restricted dispute set -- exactly
the direction a capacity estimator should err in. a/b reuse Step 3's
own verified ``derive_horizon_labels`` lowest-index tie-break
unmodified.

``Q_w = {a(x), b(x) : x in D and window w contributes to x}``. "Window
w contributes to x" is derived from the *validated Step 7 geometry*
(``window.accumulation_slice``, the exact rectangular region Step 7's
own forward/pullback functions read and write), cross-checked against
Step 7's own weight and denominator fields being strictly positive
there (already an invariant Step 7 itself enforces at plan-construction
time) -- never an independently assumed bounding box.

Active components R_w
-----------------------
``C_w = {r : exists j such that M[r,w,j] > 0}``, using Step 8's raw,
un-thresholded, un-epsilon'd soft memberships directly.
``R_w = |C_w|``.

Projected width
-----------------
``K_w = R_w * |Q_w|`` (the frozen ``query_compute_projection.
projected_width_formula``). If ``Q_w`` is empty, ``K_w = 0`` regardless
of ``R_w``.

FLOP convention (declared explicitly, traced from the live recurrence)
--------------------------------------------------------------------------
Sparse transition matmul, per recurrence update, for a given width:
``spmm_flops(width) = 2 * E_w * width`` (one multiply, one add, per
stored transition/class pair -- exactly this step's own convention).

Elementwise restart-recurrence work, traced from ``finite_step_regime.
finite_step_propagate``'s literal update line ``p = alpha_value *
graph.matmul(p) + one_minus_alpha * s0_owned``: two scalar multiplies
(``alpha_value * matmul_result``, ``one_minus_alpha * s0_owned``) and
one scalar add, per (node, column) pair -- ``elementwise_flops(width)
= 3 * N_w * width``, never assumed without this trace.

Total recurrence work for T steps at a given width:
``T * (spmm_flops(width) + elementwise_flops(width)) = T * width *
(2*E_w + 3*N_w)``. This is *linear* in width, so the sum of per-chunk
totals over any partition of a width into chunks equals the single
full-width total exactly -- chunking changes peak width and invocation
count, never the total component-class arithmetic work.

Byte-counting model (declared explicitly)
--------------------------------------------
Graph storage: ``E_w * (index_dtype_bytes + weight_dtype_bytes)``
(one stored index and one stored weight per nonzero transition).
Membership chunk: ``b * N_w * dtype_bytes``. Query-class source scores
(the raw class-restricted score column set the masked seed would be
built from): ``|Q_w| * N_w * dtype_bytes``. Working recurrence state,
conservatively counted as four *simultaneously live* width_chunk-sized
tensors per the literal update line above (current state, the matmul
output before combination, the freshly-combined next state, and the
persistent masked seed referenced every step): ``4 * width_chunk *
N_w * dtype_bytes``. Capture buffers, mirroring Step 4's five frozen
horizons (T=0/16/32/64/320): ``5 * width_chunk * N_w * dtype_bytes``.
Fixed geometry metadata is reported as a minor, explicitly-labeled
index-array-sized term, never folded into the width-scaled terms.
Allocator/workspace overhead is unknown and excluded -- no percentage
multiplier is invented for it. Two headline numbers are always reported
separately and never conflated: an *arithmetic tensor lower bound*
(the minimal live-tensor count assuming perfect in-place reuse) and a
*planned live-tensor working-set estimate* (the four-simultaneous-
tensor model above, based on the actual, traced recurrence API) --
neither is ever labeled "measured GPU peak memory."

No runtime, no real-subset numbers
-------------------------------------
This module never converts a FLOP or byte count into seconds,
images/second, GPU-hours, or a completion date. Every estimate's
``runtime_status`` is the fixed string
``"not_estimated_without_measured_hardware_calibration"`` unless an
explicit, separately-measured ``HardwareCalibration`` record is
supplied (never fabricated here). This module also never fabricates
R_w, Q_w, graph nonzero counts, or memory distributions for the real
COCO-Stuff/ADE20K 500-image subsets -- every fixture in this step's own
test suite is synthetic, and this module's own capability (not a real
Phase 1B feasibility claim) is all that is verified here.

Excluded from this step
--------------------------
No masked seed construction, no masked forward recurrence, no
component evidence, no G_xr, no supercohort propagation, no k-star/phi,
no eta-net, no Gate 1/2 analysis, no GPU microbenchmark, no real-data
driver, no result artifact, no checkpoint/resume, no SLURM script, no
final selector. **No real Phase 1B work is executed by this step.**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.dfp_phase1a_analysis import derive_horizon_labels
from src.dfp_phase1a_global_unary_components import DfpPhase1aGlobalUnaryComponentsError
from src.dfp_phase1a_horizon_stitching import DfpPhase1aHorizonStitchingError
from src.dfp_phase1a_identity import (
    DfpPhase1aIdentityError,
    validate_dispute_protocol_block,
    validate_graph_block,
    validate_propagation_block,
    validate_query_compute_projection_block,
    validate_score_interpolation_stitching_block,
    validate_structural_components_block,
)
from src.dfp_phase1a_soft_membership import DfpPhase1aSoftMembershipError
from src.dfp_phase1a_stitching_pullback import DfpPhase1aStitchingPullbackError, StitchingPullbackPlan


class DfpPhase1bComputeEstimatorError(ValueError):
    """Raised on any DFP Phase 1B compute-estimator invariant
    violation. Always fail closed: never coerce, sort a malformed
    sequence, clamp, infer a missing value, or silently repair a
    record."""


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DfpPhase1bComputeEstimatorError(f"{label} must be an exact non-boolean integer")
    return value


def _require_positive_exact_int(value: Any, label: str) -> int:
    value = _require_exact_int(value, label)
    if value <= 0:
        raise DfpPhase1bComputeEstimatorError(f"{label} must be positive")
    return value


def _require_non_negative_exact_int(value: Any, label: str) -> int:
    value = _require_exact_int(value, label)
    if value < 0:
        raise DfpPhase1bComputeEstimatorError(f"{label} must be non-negative")
    return value


# ---------------------------------------------------------------------------
# Identity-bound compute-projection contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DfpComputeProjectionContract:
    """Independently validated compute-projection parameters extracted
    from the live protocol identity -- never hardcoded module
    constants. Delegates to the single existing canonical authorities:
    ``validate_propagation_block`` (recurrence steps), ``validate_
    dispute_protocol_block`` (D/a/b/tie-rule definitions),
    ``validate_query_compute_projection_block`` (Q_w/projected-width
    formula and required statistics), ``validate_structural_components_
    block`` (component source/scope/tie-rule), ``validate_graph_block``
    (canonical graph variant), and ``validate_score_interpolation_
    stitching_block`` (class-score stage)."""

    recurrence_steps: int
    required_statistics: tuple[str, ...]


def dfp_compute_projection_contract_from_protocol(protocol_identity: Mapping[str, Any]) -> DfpComputeProjectionContract:
    """Strictly extract and validate the compute-projection-relevant
    fields of an already-loaded protocol identity. Any failure from any
    delegate is re-raised as this module's own
    ``DfpPhase1bComputeEstimatorError`` (chained via ``from exc``)."""
    try:
        propagation = protocol_identity["propagation"]
        dispute_protocol = protocol_identity["dispute_protocol"]
        query_compute_projection = protocol_identity["query_compute_projection"]
        structural_components = protocol_identity["structural_components"]
        graph = protocol_identity["graph"]
        score_interpolation_stitching = protocol_identity["score_interpolation_stitching"]
        validate_propagation_block(propagation)
        validate_dispute_protocol_block(dispute_protocol)
        validate_query_compute_projection_block(query_compute_projection)
        validate_structural_components_block(structural_components)
        validate_graph_block(graph)
        validate_score_interpolation_stitching_block(score_interpolation_stitching)
    except DfpPhase1aIdentityError as exc:
        raise DfpPhase1bComputeEstimatorError(str(exc)) from exc
    except KeyError as exc:
        raise DfpPhase1bComputeEstimatorError(f"protocol_identity is missing a required section: {exc}") from exc

    return DfpComputeProjectionContract(
        recurrence_steps=propagation["steps"], required_statistics=tuple(query_compute_projection["required_statistics"]),
    )


# ---------------------------------------------------------------------------
# Dispute set D and query-class sets Q_w
# ---------------------------------------------------------------------------


def compute_stitched_dispute_set(s0: np.ndarray, s320: np.ndarray, *, class_count: int) -> np.ndarray:
    """Independently re-derive the lowest-index unary labels for S0 and
    S320 (reusing Step 3's own ``derive_horizon_labels``, never a second
    tie-break implementation) and return the boolean dispute mask
    ``D[x] = (a(x) != b(x))`` over the *complete* stitched-image domain
    -- no ignore mask, no GT, no per-window or per-patch argmax, no T16/
    T32/T64 anywhere in this function."""
    for name, arr in (("s0", s0), ("s320", s320)):
        if not isinstance(arr, np.ndarray):
            raise DfpPhase1bComputeEstimatorError(f"{name} must be a numpy.ndarray")
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            raise DfpPhase1bComputeEstimatorError(f"{name} dtype must be exactly float32 or float64, observed {arr.dtype}")
        if arr.ndim != 3:
            raise DfpPhase1bComputeEstimatorError(f"{name} must be exactly 3-D [class, height, width], observed {arr.ndim}-D")
        if arr.shape[0] != class_count:
            raise DfpPhase1bComputeEstimatorError(f"{name} class dimension {arr.shape[0]} disagrees with class_count {class_count}")
        if not np.all(np.isfinite(arr)):
            raise DfpPhase1bComputeEstimatorError(f"{name} contains a non-finite value (NaN or Infinity)")
    if s0.shape != s320.shape:
        raise DfpPhase1bComputeEstimatorError(f"s0 shape {s0.shape} disagrees with s320 shape {s320.shape}")

    labels = derive_horizon_labels({0: s0, 320: s320})
    a = labels[0]
    b = labels[320]
    dispute = a != b
    return a, b, dispute


def compute_window_query_classes(
    a: np.ndarray, b: np.ndarray, dispute: np.ndarray, *, window_plan: StitchingPullbackPlan, window_index: int,
) -> tuple[int, ...]:
    """Q_w for one window: the ascending, deduplicated set of ``a(x)``/
    ``b(x)`` values over every disputed pixel x that window w
    contributes to. "Contributes to" is derived from Step 7's own
    validated window geometry (``accumulation_slice``) and the actual
    positivity of Step 7's own weight/denominator fields there -- never
    an independently assumed bounding box."""
    if not isinstance(window_plan, StitchingPullbackPlan):
        raise DfpPhase1bComputeEstimatorError("window_plan must be a StitchingPullbackPlan")
    window_index = _require_exact_int(window_index, "window_index")
    if window_index < 0 or window_index >= len(window_plan.plan.windows):
        raise DfpPhase1bComputeEstimatorError(f"window_index {window_index} out of range")
    for name, arr in (("a", a), ("b", b), ("dispute", dispute)):
        if not isinstance(arr, np.ndarray) or tuple(arr.shape) != window_plan.image_size:
            raise DfpPhase1bComputeEstimatorError(f"{name} shape must exactly match the plan's image_size {window_plan.image_size}")

    window = window_plan.plan.windows[window_index]
    rows, cols = window.accumulation_slice
    denom_region = window_plan.denominator[0, 0, rows, cols]
    if not bool((denom_region > 0).all()):
        raise DfpPhase1bComputeEstimatorError(f"window {window_index}: Step 7's own denominator is not strictly positive throughout its accumulation region")

    region_dispute = dispute[rows, cols]
    if not region_dispute.any():
        return ()

    region_a = a[rows, cols][region_dispute]
    region_b = b[rows, cols][region_dispute]
    combined = np.union1d(region_a, region_b)
    return tuple(int(c) for c in combined)


# ---------------------------------------------------------------------------
# Active components R_w
# ---------------------------------------------------------------------------


def compute_window_active_components(
    component_ids: np.ndarray, component_count: int, *, window_plan: StitchingPullbackPlan, window_index: int,
    component_chunk_size: int,
) -> tuple[int, ...]:
    """C_w for one window: the ascending, deduplicated set of component
    ids r for which some patch node j has strictly positive membership
    ``M[r,w,j] > 0`` -- raw, un-thresholded, no epsilon, reusing Step
    8's own ``iter_soft_membership_records`` directly."""
    from src.dfp_phase1a_soft_membership import iter_soft_membership_records

    try:
        records = iter_soft_membership_records(
            component_ids, component_count, window_plan=window_plan, component_chunk_size=component_chunk_size,
        )
    except (DfpPhase1aSoftMembershipError, DfpPhase1aStitchingPullbackError) as exc:
        raise DfpPhase1bComputeEstimatorError(str(exc)) from exc

    active: set[int] = set()
    for record in records:
        if record.window_index != window_index:
            continue
        positive_rows = (record.membership > 0).any(dim=1)
        for offset, is_positive in enumerate(positive_rows.tolist()):
            if is_positive:
                active.add(record.component_id_start + offset)
    return tuple(sorted(active))


# ---------------------------------------------------------------------------
# Per-window graph metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowGraphMetadata:
    """Validated per-window graph metadata this estimator consumes --
    never constructs a graph itself. ``graph_variant`` must match the
    frozen canonical k-variant string bound by the protocol identity's
    ``[graph]`` block (validated relationally, not re-hardcoded)."""

    node_count: int
    nonzero_count: int
    index_dtype_bytes: int
    weight_dtype_bytes: int
    fallback_included: bool
    graph_variant: str

    def __post_init__(self) -> None:
        _require_positive_exact_int(self.node_count, "node_count")
        _require_non_negative_exact_int(self.nonzero_count, "nonzero_count")
        _require_positive_exact_int(self.index_dtype_bytes, "index_dtype_bytes")
        _require_positive_exact_int(self.weight_dtype_bytes, "weight_dtype_bytes")
        if type(self.fallback_included) is not bool:
            raise DfpPhase1bComputeEstimatorError("fallback_included must be an exact bool")
        if type(self.graph_variant) is not str or not self.graph_variant:
            raise DfpPhase1bComputeEstimatorError("graph_variant must be a non-empty exact str")
        # every stored nonzero must be physically representable: at most
        # node_count^2 directed edges can exist between node_count nodes
        max_possible = self.node_count * self.node_count
        if self.nonzero_count > max_possible:
            raise DfpPhase1bComputeEstimatorError(
                f"nonzero_count {self.nonzero_count} exceeds the maximum possible {max_possible} for node_count {self.node_count} (impossible edge count)"
            )


def validate_window_graph_metadata(
    metadata: Sequence[WindowGraphMetadata], *, window_plan: StitchingPullbackPlan, expected_graph_variant: str,
) -> None:
    """Relationally verify a full window-ordered graph-metadata sequence
    against the canonical window plan and the frozen graph-variant
    string. Rejects missing/extra/duplicate/reordered windows and a
    node-count mismatch against the plan's own patch-grid product."""
    if type(metadata) not in (list, tuple):
        raise DfpPhase1bComputeEstimatorError("metadata must be a list or tuple")
    if len(metadata) != len(window_plan.plan.windows):
        raise DfpPhase1bComputeEstimatorError(
            f"metadata has {len(metadata)} entries, plan has {len(window_plan.plan.windows)} windows"
        )
    expected_node_count = window_plan.patch_grid[0] * window_plan.patch_grid[1]
    for position, entry in enumerate(metadata):
        if not isinstance(entry, WindowGraphMetadata):
            raise DfpPhase1bComputeEstimatorError(f"metadata[{position}] must be a WindowGraphMetadata")
        if entry.node_count != expected_node_count:
            raise DfpPhase1bComputeEstimatorError(
                f"metadata[{position}].node_count {entry.node_count} disagrees with the plan's patch_grid product {expected_node_count}"
            )
        if entry.graph_variant != expected_graph_variant:
            raise DfpPhase1bComputeEstimatorError(
                f"metadata[{position}].graph_variant {entry.graph_variant!r} disagrees with the frozen canonical variant {expected_graph_variant!r}"
            )


# ---------------------------------------------------------------------------
# Declared FLOP-counting convention (named constants, never hidden magic
# numbers -- see the module docstring for the derivation of each)
# ---------------------------------------------------------------------------

SPMM_FLOPS_PER_NONZERO_PER_WIDTH = 2
"""One multiply + one add, per stored transition/class pair, per
recurrence update -- this step's own declared spmm convention."""

ELEMENTWISE_FLOPS_PER_NODE_PER_WIDTH = 3
"""Two scalar multiplies + one scalar add per (node, column), traced
literally from finite_step_regime.finite_step_propagate's update line
``p = alpha_value * graph.matmul(p) + one_minus_alpha * s0_owned``."""

WORKING_STATE_TENSOR_MULTIPLIER = 4
"""Four simultaneously-live width_chunk x N_w-sized tensors per
recurrence step, traced from the same update line: the current state
being read, the matmul output before combination, the freshly-combined
next state, and the persistent masked seed referenced every step."""

CAPTURE_HORIZON_COUNT = 5
"""T=0, 16, 32, 64, 320 -- mirroring Step 4's frozen horizon count."""

MASKING_FLOPS_PER_NODE_PER_WIDTH = 1
"""Optional seed-construction/membership-masking work, reported
separately from recurrence work: a single elementwise multiply (mask *
score) per (node, column) is the minimal plausible masking operation --
not a claim about the eventual Phase 1B implementation's exact cost."""


def spmm_flops(nonzero_count: int, width: int) -> int:
    """``2 * E_w * width`` -- exact, arbitrary-precision Python int."""
    return SPMM_FLOPS_PER_NONZERO_PER_WIDTH * nonzero_count * width


def elementwise_flops(node_count: int, width: int) -> int:
    """``3 * N_w * width`` -- traced, not assumed; see module docstring."""
    return ELEMENTWISE_FLOPS_PER_NODE_PER_WIDTH * node_count * width


def recurrence_flops(node_count: int, nonzero_count: int, width: int, steps: int) -> int:
    """``T * (spmm_flops + elementwise_flops)`` for a given width."""
    return steps * (spmm_flops(nonzero_count, width) + elementwise_flops(node_count, width))


def masking_flops(node_count: int, width: int) -> int:
    """Optional, separately-reported seed-construction/masking estimate;
    see ``MASKING_FLOPS_PER_NODE_PER_WIDTH``."""
    return MASKING_FLOPS_PER_NODE_PER_WIDTH * node_count * width


# ---------------------------------------------------------------------------
# Declared byte-counting model
# ---------------------------------------------------------------------------


def graph_storage_bytes(nonzero_count: int, index_dtype_bytes: int, weight_dtype_bytes: int) -> int:
    return nonzero_count * (index_dtype_bytes + weight_dtype_bytes)


def membership_chunk_bytes(chunk_component_count: int, node_count: int, dtype_bytes: int) -> int:
    return chunk_component_count * node_count * dtype_bytes


def query_class_source_score_bytes(query_class_count: int, node_count: int, dtype_bytes: int) -> int:
    return query_class_count * node_count * dtype_bytes


def working_state_bytes(width_chunk: int, node_count: int, dtype_bytes: int) -> int:
    return WORKING_STATE_TENSOR_MULTIPLIER * width_chunk * node_count * dtype_bytes


def capture_buffer_bytes(width_chunk: int, node_count: int, dtype_bytes: int) -> int:
    return CAPTURE_HORIZON_COUNT * width_chunk * node_count * dtype_bytes


def fixed_geometry_metadata_bytes(node_count: int, index_dtype_bytes: int) -> int:
    """A minor, explicitly-labeled term (patch-node index bookkeeping),
    never folded into the width-scaled terms above."""
    return node_count * index_dtype_bytes


def arithmetic_tensor_lower_bound_bytes(
    *, nonzero_count: int, index_dtype_bytes: int, weight_dtype_bytes: int,
    chunk_component_count: int, node_count: int, width_chunk: int, query_class_count: int, dtype_bytes: int,
) -> int:
    """Minimal live-tensor count assuming perfect in-place reuse: graph
    storage + the membership chunk + exactly one width_chunk-sized state
    tensor + the query-class source scores. Never labeled "measured"."""
    return (
        graph_storage_bytes(nonzero_count, index_dtype_bytes, weight_dtype_bytes)
        + membership_chunk_bytes(chunk_component_count, node_count, dtype_bytes)
        + width_chunk * node_count * dtype_bytes
        + query_class_source_score_bytes(query_class_count, node_count, dtype_bytes)
    )


def planned_working_set_bytes(
    *, nonzero_count: int, index_dtype_bytes: int, weight_dtype_bytes: int,
    chunk_component_count: int, node_count: int, width_chunk: int, query_class_count: int, dtype_bytes: int,
) -> int:
    """Realistic peak: graph storage (persists for the whole window) +
    the membership chunk + the four-simultaneous-tensor working state +
    the query-class source scores + the capture buffers + fixed
    geometry metadata. Every included term temporally overlaps the
    recurrence in the traced update line; allocator/workspace overhead
    is excluded (unknown, no invented percentage)."""
    return (
        graph_storage_bytes(nonzero_count, index_dtype_bytes, weight_dtype_bytes)
        + membership_chunk_bytes(chunk_component_count, node_count, dtype_bytes)
        + working_state_bytes(width_chunk, node_count, dtype_bytes)
        + query_class_source_score_bytes(query_class_count, node_count, dtype_bytes)
        + capture_buffer_bytes(width_chunk, node_count, dtype_bytes)
        + fixed_geometry_metadata_bytes(node_count, index_dtype_bytes)
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_component_counts(active_component_count: int, chunk_size: int) -> tuple[int, ...]:
    """Exact contiguous chunk sizes covering ``active_component_count``
    components at ``chunk_size`` per chunk; the final chunk may be
    smaller. Never invented -- this is the same ``ceil(R_w/B)``
    partition Step 8's own chunked API already uses."""
    chunk_size = _require_positive_exact_int(chunk_size, "chunk_size")
    active_component_count = _require_non_negative_exact_int(active_component_count, "active_component_count")
    if active_component_count == 0:
        return ()
    counts = []
    remaining = active_component_count
    while remaining > 0:
        take = min(chunk_size, remaining)
        counts.append(take)
        remaining -= take
    return tuple(counts)


# ---------------------------------------------------------------------------
# Runtime-status contract (section 13): no fabricated wall-clock estimate
# ---------------------------------------------------------------------------

RUNTIME_NOT_ESTIMATED_STATUS = "not_estimated_without_measured_hardware_calibration"


@dataclass(frozen=True)
class HardwareCalibration:
    """A separately, explicitly measured calibration input -- never
    fabricated by this module. Supplying one is the ONLY way a caller
    may ever attach a runtime figure to an estimate produced here."""

    hardware_identifier: str
    software_environment: str
    measured_effective_throughput_flops_per_second: float
    calibration_workload_definition: str
    artifact_digest: str

    def __post_init__(self) -> None:
        for name in ("hardware_identifier", "software_environment", "calibration_workload_definition", "artifact_digest"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise DfpPhase1bComputeEstimatorError(f"HardwareCalibration.{name} must be a non-empty exact str")
        if type(self.measured_effective_throughput_flops_per_second) is not float or not (self.measured_effective_throughput_flops_per_second > 0):
            raise DfpPhase1bComputeEstimatorError("HardwareCalibration.measured_effective_throughput_flops_per_second must be a positive exact float")


# ---------------------------------------------------------------------------
# Per-window structural record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DfpPhase1bWindowEstimate:
    """Exact structural, FLOP, and byte estimate for one canonical
    window's proposed Phase 1B chunked masked-forward propagation. Every
    count is a Python arbitrary-precision int -- never int32/int64,
    never at risk of aggregate overflow."""

    window_index: int
    node_count: int
    nonzero_count: int
    class_count: int
    active_component_ids: tuple[int, ...]
    active_component_count: int
    query_class_indices: tuple[int, ...]
    query_class_count: int
    projected_width: int
    chunk_size: int
    chunk_count: int
    chunk_component_counts: tuple[int, ...]
    chunk_widths: tuple[int, ...]
    baseline_recurrence_flops: int
    projected_recurrence_flops: int
    chunked_total_recurrence_flops: int
    projected_masking_flops: int
    baseline_state_bytes: int
    projected_state_bytes: int
    chunk_peak_bytes: tuple[int, ...]
    chunk_lower_bound_bytes: tuple[int, ...]
    peak_planned_bytes: int
    compute_ratio_numerator: int
    compute_ratio_denominator: int
    memory_budget_bytes: int | None
    memory_budget_feasible: tuple[bool, ...] | None
    memory_budget_excess_bytes: tuple[int, ...] | None
    largest_feasible_chunk_size: int | None
    runtime_status: str

    def __post_init__(self) -> None:
        if len(self.chunk_component_counts) != self.chunk_count:
            raise DfpPhase1bComputeEstimatorError("chunk_component_counts length disagrees with chunk_count")
        # When Q_w is empty, K_w = R_w * 0 = 0 regardless of R_w -- zero
        # Phase 1B invocations are required, so chunk_component_counts is
        # legitimately empty even though active_component_count (the
        # exact, truthfully-reported R_w) may still be positive.
        if self.query_class_count > 0 and sum(self.chunk_component_counts) != self.active_component_count:
            raise DfpPhase1bComputeEstimatorError("chunk_component_counts does not sum to active_component_count")
        if self.chunked_total_recurrence_flops != self.projected_recurrence_flops:
            raise DfpPhase1bComputeEstimatorError(
                "chunked_total_recurrence_flops disagrees with projected_recurrence_flops -- "
                "chunking must never change the total mathematical work"
            )


def estimate_window_compute(
    *,
    window_index: int,
    graph_metadata: WindowGraphMetadata,
    class_count: int,
    active_component_ids: tuple[int, ...],
    query_class_indices: tuple[int, ...],
    recurrence_steps: int,
    dtype_bytes: int,
    component_chunk_size: int,
    memory_budget_bytes: int | None = None,
    hardware_calibration: HardwareCalibration | None = None,
) -> DfpPhase1bWindowEstimate:
    """Build the full per-window estimate. Consumes only already-derived
    structural counts -- constructs no graph, runs no recurrence, reads
    no dataset or ground truth. Every returned count is an exact Python
    int; every FLOP/byte formula is one of the module-level named
    functions above, never inlined arithmetic with an undeclared
    coefficient."""
    window_index = _require_non_negative_exact_int(window_index, "window_index")
    if not isinstance(graph_metadata, WindowGraphMetadata):
        raise DfpPhase1bComputeEstimatorError("graph_metadata must be a WindowGraphMetadata")
    class_count = _require_positive_exact_int(class_count, "class_count")
    recurrence_steps = _require_positive_exact_int(recurrence_steps, "recurrence_steps")
    dtype_bytes = _require_positive_exact_int(dtype_bytes, "dtype_bytes")
    component_chunk_size = _require_positive_exact_int(component_chunk_size, "component_chunk_size")
    if type(active_component_ids) not in (list, tuple):
        raise DfpPhase1bComputeEstimatorError("active_component_ids must be a list or tuple")
    if type(query_class_indices) not in (list, tuple):
        raise DfpPhase1bComputeEstimatorError("query_class_indices must be a list or tuple")
    active_component_ids = tuple(_require_non_negative_exact_int(v, "active_component_ids element") for v in active_component_ids)
    query_class_indices = tuple(_require_non_negative_exact_int(v, "query_class_indices element") for v in query_class_indices)
    if list(active_component_ids) != sorted(set(active_component_ids)):
        raise DfpPhase1bComputeEstimatorError("active_component_ids must be strictly ascending with no duplicates")
    if list(query_class_indices) != sorted(set(query_class_indices)):
        raise DfpPhase1bComputeEstimatorError("query_class_indices must be strictly ascending with no duplicates")
    for v in query_class_indices:
        if v >= class_count:
            raise DfpPhase1bComputeEstimatorError(f"query_class_indices contains {v}, out of range for class_count {class_count}")
    if memory_budget_bytes is not None:
        memory_budget_bytes = _require_positive_exact_int(memory_budget_bytes, "memory_budget_bytes")
    if hardware_calibration is not None and not isinstance(hardware_calibration, HardwareCalibration):
        raise DfpPhase1bComputeEstimatorError("hardware_calibration must be a HardwareCalibration or None")

    node_count = graph_metadata.node_count
    nonzero_count = graph_metadata.nonzero_count

    active_component_count = len(active_component_ids)
    query_class_count = len(query_class_indices)
    projected_width = active_component_count * query_class_count  # K_w = R_w * |Q_w|; 0 if either is 0

    counts = chunk_component_counts(active_component_count if query_class_count > 0 else 0, component_chunk_size)
    chunk_widths = tuple(c * query_class_count for c in counts)

    baseline_recurrence_flops = recurrence_flops(node_count, nonzero_count, class_count, recurrence_steps)
    projected_recurrence_flops = recurrence_flops(node_count, nonzero_count, projected_width, recurrence_steps)
    chunked_total_recurrence_flops = sum(
        (recurrence_flops(node_count, nonzero_count, w, recurrence_steps) for w in chunk_widths), start=0,
    )
    projected_masking_flops = masking_flops(node_count, projected_width)

    baseline_state_bytes = class_count * node_count * dtype_bytes
    projected_state_bytes = projected_width * node_count * dtype_bytes

    chunk_peak_bytes = tuple(
        planned_working_set_bytes(
            nonzero_count=nonzero_count, index_dtype_bytes=graph_metadata.index_dtype_bytes,
            weight_dtype_bytes=graph_metadata.weight_dtype_bytes, chunk_component_count=c, node_count=node_count,
            width_chunk=w, query_class_count=query_class_count, dtype_bytes=dtype_bytes,
        )
        for c, w in zip(counts, chunk_widths)
    )
    chunk_lower_bound_bytes = tuple(
        arithmetic_tensor_lower_bound_bytes(
            nonzero_count=nonzero_count, index_dtype_bytes=graph_metadata.index_dtype_bytes,
            weight_dtype_bytes=graph_metadata.weight_dtype_bytes, chunk_component_count=c, node_count=node_count,
            width_chunk=w, query_class_count=query_class_count, dtype_bytes=dtype_bytes,
        )
        for c, w in zip(counts, chunk_widths)
    )
    peak_planned_bytes = max(chunk_peak_bytes) if chunk_peak_bytes else 0

    memory_budget_feasible = None
    memory_budget_excess_bytes = None
    largest_feasible_chunk_size = None
    if memory_budget_bytes is not None:
        memory_budget_feasible = tuple(b <= memory_budget_bytes for b in chunk_peak_bytes)
        memory_budget_excess_bytes = tuple(max(0, b - memory_budget_bytes) for b in chunk_peak_bytes)
        # exact largest feasible chunk size under the declared memory model,
        # found by exact integer search (never estimated/approximated)
        if query_class_count > 0 and active_component_count > 0:
            largest_feasible_chunk_size = 0
            for candidate in range(1, active_component_count + 1):
                candidate_width = candidate * query_class_count
                candidate_bytes = planned_working_set_bytes(
                    nonzero_count=nonzero_count, index_dtype_bytes=graph_metadata.index_dtype_bytes,
                    weight_dtype_bytes=graph_metadata.weight_dtype_bytes, chunk_component_count=candidate,
                    node_count=node_count, width_chunk=candidate_width, query_class_count=query_class_count,
                    dtype_bytes=dtype_bytes,
                )
                if candidate_bytes <= memory_budget_bytes:
                    largest_feasible_chunk_size = candidate
                else:
                    break

    runtime_status = RUNTIME_NOT_ESTIMATED_STATUS if hardware_calibration is None else "estimated_from_supplied_calibration"

    return DfpPhase1bWindowEstimate(
        window_index=window_index, node_count=node_count, nonzero_count=nonzero_count, class_count=class_count,
        active_component_ids=active_component_ids, active_component_count=active_component_count,
        query_class_indices=query_class_indices, query_class_count=query_class_count, projected_width=projected_width,
        chunk_size=component_chunk_size, chunk_count=len(counts), chunk_component_counts=counts, chunk_widths=chunk_widths,
        baseline_recurrence_flops=baseline_recurrence_flops, projected_recurrence_flops=projected_recurrence_flops,
        chunked_total_recurrence_flops=chunked_total_recurrence_flops, projected_masking_flops=projected_masking_flops,
        baseline_state_bytes=baseline_state_bytes, projected_state_bytes=projected_state_bytes,
        chunk_peak_bytes=chunk_peak_bytes, chunk_lower_bound_bytes=chunk_lower_bound_bytes,
        peak_planned_bytes=peak_planned_bytes,
        compute_ratio_numerator=projected_recurrence_flops, compute_ratio_denominator=baseline_recurrence_flops,
        memory_budget_bytes=memory_budget_bytes, memory_budget_feasible=memory_budget_feasible,
        memory_budget_excess_bytes=memory_budget_excess_bytes, largest_feasible_chunk_size=largest_feasible_chunk_size,
        runtime_status=runtime_status,
    )


# ---------------------------------------------------------------------------
# Aggregation over an image / dataset of per-window records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DfpPhase1bAggregateEstimate:
    """Pure aggregation over per-window records. Exact integer totals
    plus raw vectors sufficient to reconstruct every reported
    descriptive statistic -- never a lossy summary alone."""

    total_windows: int
    zero_dispute_windows: int
    total_disputed_pixels: int
    total_node_count: int
    total_nonzero_count: int
    total_baseline_recurrence_flops: int
    total_projected_recurrence_flops: int
    total_chunk_invocations: int
    maximum_projected_width: int
    maximum_peak_planned_bytes: int
    aggregate_compute_ratio_numerator: int
    aggregate_compute_ratio_denominator: int
    active_component_counts: tuple[int, ...]
    query_class_counts: tuple[int, ...]
    projected_widths: tuple[int, ...]
    chunk_counts: tuple[int, ...]
    peak_bytes_per_window: tuple[int, ...]
    projected_flops_per_window: tuple[int, ...]


def aggregate_window_estimates(
    estimates: Sequence[DfpPhase1bWindowEstimate], *, total_disputed_pixels: int,
) -> DfpPhase1bAggregateEstimate:
    if type(estimates) not in (list, tuple):
        raise DfpPhase1bComputeEstimatorError("estimates must be a list or tuple")
    if not estimates:
        raise DfpPhase1bComputeEstimatorError("estimates must be non-empty")
    total_disputed_pixels = _require_non_negative_exact_int(total_disputed_pixels, "total_disputed_pixels")

    zero_dispute_windows = sum(1 for e in estimates if e.query_class_count == 0)
    total_node_count = sum(e.node_count for e in estimates)
    total_nonzero_count = sum(e.nonzero_count for e in estimates)
    total_baseline = sum(e.baseline_recurrence_flops for e in estimates)
    total_projected = sum(e.projected_recurrence_flops for e in estimates)
    total_chunks = sum(e.chunk_count for e in estimates)
    max_width = max(e.projected_width for e in estimates)
    max_bytes = max(e.peak_planned_bytes for e in estimates)

    return DfpPhase1bAggregateEstimate(
        total_windows=len(estimates), zero_dispute_windows=zero_dispute_windows,
        total_disputed_pixels=total_disputed_pixels, total_node_count=total_node_count,
        total_nonzero_count=total_nonzero_count, total_baseline_recurrence_flops=total_baseline,
        total_projected_recurrence_flops=total_projected, total_chunk_invocations=total_chunks,
        maximum_projected_width=max_width, maximum_peak_planned_bytes=max_bytes,
        aggregate_compute_ratio_numerator=total_projected, aggregate_compute_ratio_denominator=total_baseline,
        active_component_counts=tuple(e.active_component_count for e in estimates),
        query_class_counts=tuple(e.query_class_count for e in estimates),
        projected_widths=tuple(e.projected_width for e in estimates),
        chunk_counts=tuple(e.chunk_count for e in estimates),
        peak_bytes_per_window=tuple(e.peak_planned_bytes for e in estimates),
        projected_flops_per_window=tuple(e.projected_recurrence_flops for e in estimates),
    )


# ---------------------------------------------------------------------------
# Descriptive percentile statistics (documented method, raw values retained)
# ---------------------------------------------------------------------------

PERCENTILE_METHOD = "linear_interpolation_inclusive_0_100 (numpy.percentile default, method='linear')"


def descriptive_statistics(values: Sequence[int]) -> dict[str, float | tuple[int, ...]]:
    """mean/median/p90/p95/p99/maximum over ``values`` -- exactly the
    six statistics ``query_compute_projection.required_statistics``
    pins, using numpy's default linear-interpolation percentile method
    (documented, never an unstated convention), labeled descriptive,
    with the raw input values always retained alongside."""
    if type(values) not in (list, tuple) or not values:
        raise DfpPhase1bComputeEstimatorError("values must be a non-empty list or tuple")
    for v in values:
        _require_non_negative_exact_int(v, "values element")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90, method="linear")),
        "p95": float(np.percentile(arr, 95, method="linear")),
        "p99": float(np.percentile(arr, 99, method="linear")),
        "maximum": float(np.max(arr)),
        "percentile_method": PERCENTILE_METHOD,
        "raw_values": tuple(int(v) for v in values),
    }


__all__ = [
    "DfpPhase1bComputeEstimatorError",
    "DfpComputeProjectionContract",
    "dfp_compute_projection_contract_from_protocol",
    "compute_stitched_dispute_set",
    "compute_window_query_classes",
    "compute_window_active_components",
    "WindowGraphMetadata",
    "validate_window_graph_metadata",
    "SPMM_FLOPS_PER_NONZERO_PER_WIDTH",
    "ELEMENTWISE_FLOPS_PER_NODE_PER_WIDTH",
    "WORKING_STATE_TENSOR_MULTIPLIER",
    "CAPTURE_HORIZON_COUNT",
    "MASKING_FLOPS_PER_NODE_PER_WIDTH",
    "spmm_flops",
    "elementwise_flops",
    "recurrence_flops",
    "masking_flops",
    "graph_storage_bytes",
    "membership_chunk_bytes",
    "query_class_source_score_bytes",
    "working_state_bytes",
    "capture_buffer_bytes",
    "fixed_geometry_metadata_bytes",
    "arithmetic_tensor_lower_bound_bytes",
    "planned_working_set_bytes",
    "chunk_component_counts",
    "RUNTIME_NOT_ESTIMATED_STATUS",
    "HardwareCalibration",
    "DfpPhase1bWindowEstimate",
    "estimate_window_compute",
    "DfpPhase1bAggregateEstimate",
    "aggregate_window_estimates",
    "PERCENTILE_METHOD",
    "descriptive_statistics",
]
