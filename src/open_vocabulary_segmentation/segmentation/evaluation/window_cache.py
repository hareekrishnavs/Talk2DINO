"""Image-scoped two-pass sliding-window cache for canonical RWR crops.

This module adds caching and orchestration only: it stores an immutable,
CPU-owned snapshot of the per-window state that canonical RWR already
computes (E3 pre-sigmoid scores, DINO features, the directed top-k graph,
and the RWR-propagated scores), so that a second pass over the same image
can be replayed without rerunning the backbone, the graph builder, or the
CGLS solver. The only production "processor" in this commit is the
identity/no-op replay, which must reproduce the first-pass stitched output
exactly. No consensus, repair, edge editing, or counterfactual logic lives
here.

Dependency policy
------------------
The core lifecycle/storage types below (``GraphSnapshot``,
``CachedWindowState``, ``ImageWindowCache``, ``FirstPassImageContext``) use
only :mod:`sliding_window_geometry`, PyTorch tensors, and small immutable
metadata types defined in this file — they never import the model package.
The orchestration functions at the bottom (``run_pass_one``,
``run_pass_two``, ``run_two_pass_slide_inference``) do need the real model,
graph builder, and solver; they import those lazily, inside the function
body, so merely importing this module (e.g. to construct cache objects in
a test with synthetic tensors) never pulls in mmcv/mmseg/the full model.

Coordinate/ownership conventions match ``sliding_window_geometry.py``:
``(row, col)`` throughout, half-open windows, and — additionally here —
every cached tensor is CPU-owned, detached, cloned, and contiguous; public
accessors return a fresh clone (or device-materialized copy) on every call
rather than exposing internal storage by reference.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import astuple, dataclass
from enum import Enum
from typing import Callable, Iterator, Mapping, Optional

import torch
import torch.nn.functional as F

from .sliding_window_geometry import SpatialSize, SlidingWindowPlan, WindowGeometry


class WindowCacheError(RuntimeError):
    """Raised on any two-pass cache lifecycle, ownership, or validation
    violation. Always fail closed: never silently fall back to one pass."""


class WindowCacheState(Enum):
    OPEN = "open"
    SEALED = "sealed"
    REPLAYING = "replaying"
    CLOSED = "closed"


def _require_exact_int(value, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WindowCacheError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise WindowCacheError(f"{label} must be >= {minimum}")
    return value


def _require_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise WindowCacheError(f"{label} must be a torch.Tensor")
    return value


def _owned_cpu_clone(value: torch.Tensor) -> torch.Tensor:
    """Detach, move to CPU, clone, and make contiguous.

    This is the sole tensor-ownership boundary: it guarantees the cache
    never aliases storage with a caller-supplied tensor (mutating the
    caller's original tensor afterward cannot affect the cached copy) and
    strips any autograd history/CUDA storage before the tensor is retained.
    """
    return value.detach().to("cpu").clone().contiguous()


# ---------------------------------------------------------------------------
# Small immutable metadata types (byte accounting, solver telemetry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedWindowByteAccounting:
    """Exact per-field and total byte counts for one cached window."""

    s0_bytes: int
    features_bytes: int
    graph_indices_bytes: int
    graph_weights_bytes: int
    graph_metadata_bytes: int
    propagated_scores_bytes: int
    metadata_overhead_bytes: int
    total_bytes: int

    def human_readable(self) -> str:
        def mib(count: int) -> str:
            return f"{count / (1024 * 1024):.4f} MiB"

        return (
            f"s0={mib(self.s0_bytes)} features={mib(self.features_bytes)} "
            f"graph_indices={mib(self.graph_indices_bytes)} "
            f"graph_weights={mib(self.graph_weights_bytes)} "
            f"graph_metadata={mib(self.graph_metadata_bytes)} "
            f"propagated={mib(self.propagated_scores_bytes)} "
            f"overhead={mib(self.metadata_overhead_bytes)} "
            f"total={mib(self.total_bytes)}"
        )


@dataclass(frozen=True)
class WindowSolverTelemetry:
    """Immutable per-window solver diagnostics, independent of the cover_dr
    package's own ``RWRWindowDiagnostics`` type (mirrors its fields) so the
    core cache module never needs to import it."""

    iterations: int
    work_count: int
    restarts: int
    residual_replacements: int
    fallback_rows: int
    maximum_scaled_residual: float
    working_maximum_scaled_residual: float = 0.0
    certified_maximum_scaled_residual: float = 0.0
    certificate_dtype: str = ""
    fp64_certificate_checks: int = 0
    fp64_certified_rhs: int = 0
    fp64_certificate_rejections: int = 0
    fp64_certificate_restart_count: int = 0
    fp64_certificate_work: int = 0


@dataclass(frozen=True)
class WindowSolverTelemetrySummary:
    """Bounded aggregate over one image's windows; no tensors retained."""

    window_count: int
    total_iterations: int
    minimum_iterations: int
    maximum_iterations: int
    total_restarts: int
    nonzero_restart_windows: int
    total_residual_replacements: int
    total_fallback_rows: int
    maximum_scaled_residual: float


def _summarize_telemetry(
    entries: list[WindowSolverTelemetry],
) -> WindowSolverTelemetrySummary:
    if not entries:
        return WindowSolverTelemetrySummary(0, 0, 0, 0, 0, 0, 0, 0, 0.0)
    iterations = [entry.iterations for entry in entries]
    return WindowSolverTelemetrySummary(
        window_count=len(entries),
        total_iterations=sum(iterations),
        minimum_iterations=min(iterations),
        maximum_iterations=max(iterations),
        total_restarts=sum(entry.restarts for entry in entries),
        nonzero_restart_windows=sum(1 for entry in entries if entry.restarts > 0),
        total_residual_replacements=sum(entry.residual_replacements for entry in entries),
        total_fallback_rows=sum(entry.fallback_rows for entry in entries),
        maximum_scaled_residual=max(entry.maximum_scaled_residual for entry in entries),
    )


def compute_window_checksums(
    s0: torch.Tensor,
    dino_features: torch.Tensor,
    propagated_scores: torch.Tensor,
    graph_indices: torch.Tensor,
    graph_weights: torch.Tensor,
) -> dict[str, str]:
    """Optional sha256 checksums for diagnostics, computed before caching."""

    def digest(tensor: torch.Tensor) -> str:
        contiguous = tensor.detach().to("cpu").contiguous()
        return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()

    return {
        "s0": digest(s0),
        "dino_features": digest(dino_features),
        "propagated_scores": digest(propagated_scores),
        "graph_indices": digest(graph_indices),
        "graph_weights": digest(graph_weights),
    }


# ---------------------------------------------------------------------------
# Directed sparse graph snapshot (pure tensors; no cover_dr dependency)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSnapshot:
    """Owned CPU copy of everything needed to exactly reconstruct one
    directed top-k graph, mirroring ``DirectedTopKGraph``'s own field set
    (indices, weights, raw selected affinities, fallback rows, k,
    affinity_power) without importing that class."""

    neighbor_indices: torch.Tensor
    transition_weights: torch.Tensor
    edge_affinities: torch.Tensor
    self_loop_fallback: torch.Tensor
    num_nodes: int
    k: int
    affinity_power: float

    def __post_init__(self) -> None:
        num_nodes = _require_exact_int(self.num_nodes, "num_nodes", minimum=2)
        k = _require_exact_int(self.k, "k", minimum=1)
        if not k < num_nodes:
            raise WindowCacheError("k must satisfy 0 < k < num_nodes")
        if isinstance(self.affinity_power, bool) or not isinstance(
            self.affinity_power, (int, float)
        ):
            raise WindowCacheError("affinity_power must be numeric")
        affinity_power = float(self.affinity_power)
        if affinity_power != affinity_power or affinity_power <= 0:  # NaN-safe
            raise WindowCacheError("affinity_power must be finite and positive")
        object.__setattr__(self, "affinity_power", affinity_power)

        expected_edges = (num_nodes, k)
        owned: dict[str, torch.Tensor] = {}
        for name, expected_dtype in (
            ("neighbor_indices", torch.int64),
            ("transition_weights", torch.float32),
            ("edge_affinities", torch.float32),
        ):
            value = _require_tensor(getattr(self, name), name)
            if tuple(value.shape) != expected_edges:
                raise WindowCacheError(
                    f"{name} must have shape {expected_edges}, got {tuple(value.shape)}"
                )
            if value.dtype != expected_dtype:
                raise WindowCacheError(f"{name} must have dtype {expected_dtype}")
            owned[name] = _owned_cpu_clone(value)

        fallback = _require_tensor(self.self_loop_fallback, "self_loop_fallback")
        if tuple(fallback.shape) != (num_nodes,):
            raise WindowCacheError("self_loop_fallback must have shape (num_nodes,)")
        if fallback.dtype != torch.bool:
            raise WindowCacheError("self_loop_fallback must have dtype torch.bool")
        owned["self_loop_fallback"] = _owned_cpu_clone(fallback)

        if not bool(torch.isfinite(owned["transition_weights"]).all()):
            raise WindowCacheError("transition_weights must be finite")
        if not bool(torch.isfinite(owned["edge_affinities"]).all()):
            raise WindowCacheError("edge_affinities must be finite")

        for name, value in owned.items():
            object.__setattr__(self, name, value)

    def indices_copy(self, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.neighbor_indices if device is None else self.neighbor_indices.to(device)
        return value.clone()

    def weights_copy(self, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.transition_weights if device is None else self.transition_weights.to(device)
        return value.clone()

    def edge_affinities_copy(self, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.edge_affinities if device is None else self.edge_affinities.to(device)
        return value.clone()

    def self_loop_fallback_copy(self, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.self_loop_fallback if device is None else self.self_loop_fallback.to(device)
        return value.clone()

    def byte_sizes(self) -> tuple[int, int, int]:
        """Return ``(indices_bytes, weights_bytes, metadata_bytes)``."""
        indices_bytes = self.neighbor_indices.element_size() * self.neighbor_indices.nelement()
        weights_bytes = self.transition_weights.element_size() * self.transition_weights.nelement()
        metadata_bytes = (
            self.edge_affinities.element_size() * self.edge_affinities.nelement()
            + self.self_loop_fallback.element_size() * self.self_loop_fallback.nelement()
        )
        return indices_bytes, weights_bytes, metadata_bytes


def reconstruct_directed_topk_graph(
    snapshot: GraphSnapshot, *, device: torch.device | str | None = None
):
    """Reconstruct a real ``DirectedTopKGraph`` from a ``GraphSnapshot``.

    This never rebuilds the graph from DINO features; it only moves the
    already-selected indices/weights/affinities/fallback rows to the
    requested device and re-wraps them. Imported lazily so importing this
    module never requires the model package.
    """
    from models.dinotext.cover_dr import DirectedTopKGraph

    return DirectedTopKGraph(
        neighbor_indices=snapshot.indices_copy(device),
        transition_weights=snapshot.weights_copy(device),
        edge_affinities=snapshot.edge_affinities_copy(device),
        self_loop_fallback=snapshot.self_loop_fallback_copy(device),
        num_nodes=snapshot.num_nodes,
        k=snapshot.k,
        affinity_power=snapshot.affinity_power,
    )


# ---------------------------------------------------------------------------
# Immutable cached window state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedWindowState:
    """Immutable, CPU-owned snapshot of one window's pass-1 state.

    Sufficient for exact graph reconstruction and future consensus/repair
    stages. Deliberately excludes dense window predictions, source images,
    ground truth, per-pixel argmax maps, autograd graphs, CUDA tensor
    references, and model objects.
    """

    geometry: WindowGeometry
    window_index: int
    patch_grid_shape: tuple[int, int]
    class_count: int
    s0: torch.Tensor
    dino_features: torch.Tensor
    graph: GraphSnapshot
    propagated_scores: torch.Tensor
    solver_summary: WindowSolverTelemetry
    checksums: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, WindowGeometry):
            raise WindowCacheError("geometry must be a WindowGeometry")
        object.__setattr__(
            self, "window_index", _require_exact_int(self.window_index, "window_index", minimum=0)
        )
        if (
            not isinstance(self.patch_grid_shape, tuple)
            or len(self.patch_grid_shape) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0
                for v in self.patch_grid_shape
            )
        ):
            raise WindowCacheError("patch_grid_shape must be a pair of positive exact integers")
        object.__setattr__(
            self, "class_count", _require_exact_int(self.class_count, "class_count", minimum=1)
        )
        if not isinstance(self.graph, GraphSnapshot):
            raise WindowCacheError("graph must be a GraphSnapshot")
        if not isinstance(self.solver_summary, WindowSolverTelemetry):
            raise WindowCacheError("solver_summary must be a WindowSolverTelemetry")
        if self.checksums is not None:
            if not isinstance(self.checksums, Mapping) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in self.checksums.items()
            ):
                raise WindowCacheError("checksums must be a mapping of str to str")
            object.__setattr__(self, "checksums", dict(self.checksums))

        raw: dict[str, torch.Tensor] = {}
        for name in ("s0", "dino_features", "propagated_scores"):
            value = _require_tensor(getattr(self, name), name)
            if not value.is_floating_point():
                raise WindowCacheError(f"{name} must be floating point")
            if value.ndim != 2:
                raise WindowCacheError(f"{name} must have shape [N, D_or_C], got {tuple(value.shape)}")
            if not bool(torch.isfinite(value).all()):
                raise WindowCacheError(f"{name} must be finite")
            raw[name] = value

        # Checked on the CALLER-SUPPLIED tensors, before cloning: cloning
        # below would unconditionally de-alias everything anyway, so this
        # catches a genuine caller mistake (e.g. passing the same tensor
        # object for two conceptually distinct fields) instead of silently
        # absorbing it.
        if (
            raw["s0"].nelement() > 0
            and raw["s0"].data_ptr() == raw["propagated_scores"].data_ptr()
        ):
            raise WindowCacheError("s0 and propagated_scores must not alias storage")
        if (
            raw["s0"].nelement() > 0
            and raw["s0"].data_ptr() == raw["dino_features"].data_ptr()
        ):
            raise WindowCacheError("s0 and dino_features must not alias storage")
        if (
            raw["dino_features"].nelement() > 0
            and raw["dino_features"].data_ptr() == raw["propagated_scores"].data_ptr()
        ):
            raise WindowCacheError("dino_features and propagated_scores must not alias storage")

        owned = {name: _owned_cpu_clone(value) for name, value in raw.items()}
        for name, value in owned.items():
            object.__setattr__(self, name, value)

        num_nodes = self.s0.shape[0]
        if self.dino_features.shape[0] != num_nodes:
            raise WindowCacheError("dino_features node count does not match s0")
        if self.propagated_scores.shape[0] != num_nodes:
            raise WindowCacheError("propagated_scores node count does not match s0")
        if self.graph.num_nodes != num_nodes:
            raise WindowCacheError("graph node count does not match s0/features/propagated_scores")
        if self.s0.shape[1] != self.class_count:
            raise WindowCacheError("s0 class dimension does not match class_count")
        if self.propagated_scores.shape[1] != self.class_count:
            raise WindowCacheError("propagated_scores class dimension does not match class_count")
        if self.patch_grid_shape[0] * self.patch_grid_shape[1] != num_nodes:
            raise WindowCacheError("patch_grid_shape does not match node count")

    # -- safe accessors: every call returns an independent tensor -------

    def s0_copy(self, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.s0 if device is None else self.s0.to(device)
        return value.clone()

    def dino_features_copy(self, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.dino_features if device is None else self.dino_features.to(device)
        return value.clone()

    def propagated_scores_copy(self, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.propagated_scores if device is None else self.propagated_scores.to(device)
        return value.clone()

    def graph_copy(self) -> GraphSnapshot:
        return GraphSnapshot(
            neighbor_indices=self.graph.neighbor_indices.clone(),
            transition_weights=self.graph.transition_weights.clone(),
            edge_affinities=self.graph.edge_affinities.clone(),
            self_loop_fallback=self.graph.self_loop_fallback.clone(),
            num_nodes=self.graph.num_nodes,
            k=self.graph.k,
            affinity_power=self.graph.affinity_power,
        )

    def verify_checksums(self) -> bool:
        if self.checksums is None:
            raise WindowCacheError("no checksums were recorded for this window")
        observed = compute_window_checksums(
            self.s0, self.dino_features, self.propagated_scores,
            self.graph.neighbor_indices, self.graph.transition_weights,
        )
        return observed == dict(self.checksums)

    def byte_accounting(self) -> CachedWindowByteAccounting:
        s0_bytes = self.s0.element_size() * self.s0.nelement()
        features_bytes = self.dino_features.element_size() * self.dino_features.nelement()
        propagated_bytes = self.propagated_scores.element_size() * self.propagated_scores.nelement()
        indices_bytes, weights_bytes, graph_metadata_bytes = self.graph.byte_sizes()
        overhead_bytes = sum(
            sys.getsizeof(field) for field in astuple(self.solver_summary)
        )
        total = (
            s0_bytes + features_bytes + indices_bytes + weights_bytes
            + graph_metadata_bytes + propagated_bytes + overhead_bytes
        )
        return CachedWindowByteAccounting(
            s0_bytes=s0_bytes,
            features_bytes=features_bytes,
            graph_indices_bytes=indices_bytes,
            graph_weights_bytes=weights_bytes,
            graph_metadata_bytes=graph_metadata_bytes,
            propagated_scores_bytes=propagated_bytes,
            metadata_overhead_bytes=overhead_bytes,
            total_bytes=total,
        )


# ---------------------------------------------------------------------------
# Image-level cache lifecycle
# ---------------------------------------------------------------------------


class ImageWindowCache:
    """Explicit OPEN -> SEALED -> REPLAYING(-> SEALED) -> CLOSED lifecycle
    for one image's cached windows.

    Exactly one instance is created per image (by the orchestration
    functions below); nothing here is module-level or shared across
    images. Intended to be used as a context manager so cleanup always
    happens, even on exception.
    """

    def __init__(self, plan: SlidingWindowPlan):
        if not isinstance(plan, SlidingWindowPlan):
            raise WindowCacheError("plan must be a SlidingWindowPlan")
        self._plan = plan
        self._expected_count = plan.window_count
        self._entries: dict[int, CachedWindowState] = {}
        self._order: list[int] = []
        self._state = WindowCacheState.OPEN

    @property
    def state(self) -> WindowCacheState:
        return self._state

    @property
    def plan(self) -> SlidingWindowPlan:
        return self._plan

    @property
    def expected_window_count(self) -> int:
        return self._expected_count

    @property
    def cached_window_count(self) -> int:
        return len(self._entries)

    def append(self, state: CachedWindowState) -> None:
        if self._state is not WindowCacheState.OPEN:
            raise WindowCacheError(
                f"cannot append a window while the cache is {self._state.value}"
            )
        if not isinstance(state, CachedWindowState):
            raise WindowCacheError("state must be a CachedWindowState")
        index = state.window_index
        if index in self._entries:
            raise WindowCacheError(f"duplicate window index {index}")
        if index != len(self._order):
            raise WindowCacheError(
                f"out-of-order window append: expected index {len(self._order)}, got {index}"
            )
        if index >= self._expected_count:
            raise WindowCacheError(
                f"window index {index} exceeds the plan's window count {self._expected_count}"
            )
        expected_geometry = self._plan.windows[index]
        if state.geometry != expected_geometry:
            raise WindowCacheError(
                f"cached window {index} geometry does not match the SlidingWindowPlan"
            )
        self._entries[index] = state
        self._order.append(index)

    def seal(self) -> None:
        if self._state is not WindowCacheState.OPEN:
            raise WindowCacheError(f"cannot seal the cache from state {self._state.value}")
        if len(self._entries) != self._expected_count:
            raise WindowCacheError(
                "cannot seal: expected "
                f"{self._expected_count} windows, got {len(self._entries)}"
            )
        if self._order != list(range(self._expected_count)):
            raise WindowCacheError("cannot seal: windows are not in contiguous row-major order")
        self._state = WindowCacheState.SEALED

    def begin_replay(self) -> None:
        if self._state is not WindowCacheState.SEALED:
            raise WindowCacheError(
                f"cannot begin replay from state {self._state.value}; the cache must be sealed first"
            )
        self._state = WindowCacheState.REPLAYING

    def end_replay(self) -> None:
        if self._state is not WindowCacheState.REPLAYING:
            raise WindowCacheError(f"cannot end replay from state {self._state.value}")
        self._state = WindowCacheState.SEALED

    def get(self, index: int) -> CachedWindowState:
        if self._state not in (WindowCacheState.SEALED, WindowCacheState.REPLAYING):
            raise WindowCacheError(
                f"cannot access cached windows while the cache is {self._state.value}"
            )
        if index not in self._entries:
            raise WindowCacheError(f"no cached window at index {index}")
        return self._entries[index]

    def windows_in_order(self) -> Iterator[CachedWindowState]:
        if self._state not in (WindowCacheState.SEALED, WindowCacheState.REPLAYING):
            raise WindowCacheError(
                f"cannot iterate cached windows while the cache is {self._state.value}"
            )
        for index in self._order:
            yield self._entries[index]

    def total_bytes(self) -> int:
        if self._state is WindowCacheState.CLOSED:
            raise WindowCacheError("cannot query cache bytes after the cache is closed")
        return sum(entry.byte_accounting().total_bytes for entry in self._entries.values())

    def close(self) -> None:
        """Release all cached tensors. Idempotent; safe to call more than once."""
        self._entries.clear()
        self._order.clear()
        self._state = WindowCacheState.CLOSED

    def __enter__(self) -> "ImageWindowCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


# ---------------------------------------------------------------------------
# Image-level first-pass context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstPassImageContext:
    """Immutable image-level context produced by pass 1.

    Holds the already-required stitched accumulator (one owned copy, not
    duplicated) plus lightweight geometry/coverage/byte/telemetry metadata.
    This commit does not interpret ``stitched_scores`` semantically; it is
    made available for later stages to consume.
    """

    image_size: SpatialSize
    plan: SlidingWindowPlan
    class_count: int
    common_patch_grid_shape: Optional[tuple[int, int]]
    expected_window_count: int
    cached_window_count: int
    min_coverage: int
    max_coverage: int
    stitched_scores: torch.Tensor
    cache_total_bytes: int
    pass_summary: WindowSolverTelemetrySummary

    def __post_init__(self) -> None:
        if not isinstance(self.image_size, SpatialSize):
            raise WindowCacheError("image_size must be a SpatialSize")
        if not isinstance(self.plan, SlidingWindowPlan):
            raise WindowCacheError("plan must be a SlidingWindowPlan")
        object.__setattr__(
            self, "class_count", _require_exact_int(self.class_count, "class_count", minimum=1)
        )
        object.__setattr__(
            self,
            "expected_window_count",
            _require_exact_int(self.expected_window_count, "expected_window_count", minimum=1),
        )
        object.__setattr__(
            self,
            "cached_window_count",
            _require_exact_int(self.cached_window_count, "cached_window_count", minimum=0),
        )
        object.__setattr__(
            self, "min_coverage", _require_exact_int(self.min_coverage, "min_coverage", minimum=0)
        )
        object.__setattr__(
            self, "max_coverage", _require_exact_int(self.max_coverage, "max_coverage", minimum=0)
        )
        object.__setattr__(
            self,
            "cache_total_bytes",
            _require_exact_int(self.cache_total_bytes, "cache_total_bytes", minimum=0),
        )
        if not isinstance(self.pass_summary, WindowSolverTelemetrySummary):
            raise WindowCacheError("pass_summary must be a WindowSolverTelemetrySummary")
        stitched = _require_tensor(self.stitched_scores, "stitched_scores")
        if stitched.ndim != 4:
            raise WindowCacheError("stitched_scores must have shape [1, C, H, W]")
        object.__setattr__(
            self, "stitched_scores", stitched.detach().clone().contiguous()
        )

    def stitched_scores_copy(self, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.stitched_scores if device is None else self.stitched_scores.to(device)
        return value.clone()


# ---------------------------------------------------------------------------
# Orchestration: opt-in two-pass pipeline
# ---------------------------------------------------------------------------

SecondPassProcessor = Callable[[int, WindowGeometry, torch.Tensor], torch.Tensor]


def identity_second_pass_processor(
    window_index: int, geometry: WindowGeometry, propagated_scores: torch.Tensor
) -> torch.Tensor:
    """No-op replay processor: returns the cached propagated scores
    unchanged. Deliberately has no access to ground truth or evaluator
    labels; the only production processor in this commit."""
    del window_index, geometry
    return propagated_scores


def _stitch_step(
    preds: torch.Tensor, count_mat: torch.Tensor, masks: torch.Tensor, window: WindowGeometry
) -> None:
    """Shared accumulation step, used identically by pass 1 and pass 2 so
    their arithmetic is guaranteed to match; mirrors the committed
    ``DINOTextSegInference.slide_inference`` accumulation exactly."""
    accum_rows, accum_cols = window.accumulation_slice
    preds += F.pad(
        masks,
        (
            accum_cols.start,
            preds.shape[3] - accum_cols.stop,
            accum_rows.start,
            preds.shape[2] - accum_rows.stop,
        ),
    )
    count_mat[:, :, accum_rows, accum_cols] += 1


def run_pass_one(
    inference,
    img: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    want_checksums: bool = False,
) -> tuple[torch.Tensor, ImageWindowCache, FirstPassImageContext]:
    """Run pass 1: one snapshot, one graph build, one CGLS solve, one
    downstream transform per window; accumulate the first-pass stitch; cache
    every window's immutable state; seal the cache; finalize the context.

    Canonical RWR itself is unchanged: this calls the same
    ``build_directed_topk_graph``/``solve_rwr_cgls`` functions with the same
    arguments the (unmodified) ``apply_rwr_to_e3_snapshot`` uses, just
    retaining the intermediate graph object for caching instead of
    discarding it, since building the graph must happen exactly once.
    """
    from models.dinotext.cover_dr.graph import build_directed_topk_graph
    from models.dinotext.cover_dr.inference import RWRInferenceConfig
    from models.dinotext.cover_dr.rwr import solve_rwr_cgls

    config = inference.rwr_config
    if not isinstance(config, RWRInferenceConfig):
        raise WindowCacheError("inference.rwr_config must be an RWRInferenceConfig")
    config.validate()
    if not config.enabled:
        raise WindowCacheError("two-pass window caching requires canonical RWR to be enabled")
    if config.alpha == 0:
        raise WindowCacheError("two-pass window caching requires a nonzero RWR alpha")
    if inference.with_bg:
        raise WindowCacheError(
            "two-pass window caching requires with_bg=False, matching canonical RWR"
        )
    if not torch.is_tensor(img) or img.ndim != 4 or img.shape[0] != 1:
        raise WindowCacheError(
            "two-pass window caching requires exactly one image (batch size 1)"
        )

    h_stride, w_stride = inference.test_cfg.stride
    h_crop, w_crop = inference.test_cfg.crop_size
    _, _, h_img, w_img = img.shape
    plan = SlidingWindowPlan.build(
        image_size=SpatialSize(h_img, w_img),
        crop_size=SpatialSize(h_crop, w_crop),
        stride=SpatialSize(h_stride, w_stride),
    )
    class_count = _require_exact_int(inference.num_classes, "inference.num_classes", minimum=1)
    output_device = img.device if device is None else device

    preds = img.new_zeros((1, class_count, h_img, w_img))
    count_mat = img.new_zeros((1, 1, h_img, w_img))

    cache = ImageWindowCache(plan)
    telemetry: list[WindowSolverTelemetry] = []
    common_grid_shape: tuple[int, int] | None = None
    heterogeneous_grid = False

    try:
        for window in plan.windows:
            crop_rows, crop_cols = window.crop_slice
            crop = img[:, :, crop_rows, crop_cols]

            snapshot = inference.model.generate_patch_snapshot(crop, inference.text_embedding)
            scores = getattr(snapshot, "unary_scores", None)
            features = getattr(snapshot, "dino_features", None)
            grid_hw = getattr(snapshot, "grid_hw", None)
            if not torch.is_tensor(scores) or scores.ndim != 3:
                raise WindowCacheError("snapshot.unary_scores must have shape [1,N,C]")
            if not torch.is_tensor(features) or features.ndim != 3:
                raise WindowCacheError("snapshot.dino_features must have shape [1,N,D]")
            if scores.shape[0] != 1 or features.shape[0] != 1:
                raise WindowCacheError("two-pass window caching requires batch size 1 per crop")
            if scores.shape[1] != features.shape[1]:
                raise WindowCacheError("snapshot score/feature patch count mismatch")
            if scores.shape[2] != config.expected_class_count:
                raise WindowCacheError("canonical RWR class-count mismatch in pass 1")
            if (
                not isinstance(grid_hw, tuple)
                or len(grid_hw) != 2
                or any(type(value) is not int or value <= 0 for value in grid_hw)
                or grid_hw[0] * grid_hw[1] != scores.shape[1]
            ):
                raise WindowCacheError("snapshot grid_hw does not match its patch count")
            if not scores.is_floating_point() or not features.is_floating_point():
                raise WindowCacheError("snapshot scores/features must be floating point")
            if not bool(torch.isfinite(scores).all()):
                raise WindowCacheError("snapshot scores must be finite")
            if not bool(torch.isfinite(features).all()):
                raise WindowCacheError("snapshot features must be finite")
            if config.top_k >= scores.shape[1]:
                raise WindowCacheError("RWR top_k must be smaller than the patch count")

            if common_grid_shape is None:
                common_grid_shape = grid_hw
            elif common_grid_shape != grid_hw:
                heterogeneous_grid = True

            graph = build_directed_topk_graph(
                features[0], k=config.top_k, affinity_power=config.affinity_power,
            )
            result = solve_rwr_cgls(
                graph, scores[0], alpha=config.alpha, rtol=config.solver_rtol,
                atol=config.solver_atol, max_iter=config.solver_max_iterations,
            )
            propagated = result.scores  # [N, C]

            masks = inference.model.masks_from_patch_scores(
                propagated.unsqueeze(0), grid_hw, tuple(crop.shape[-2:]),
            )
            _stitch_step(preds, count_mat, masks, window)

            window_telemetry = WindowSolverTelemetry(
                iterations=result.iterations,
                work_count=result.work_count,
                restarts=result.total_restart_count,
                residual_replacements=result.total_residual_replacement_count,
                fallback_rows=int(graph.self_loop_fallback.sum().item()),
                maximum_scaled_residual=result.maximum_scaled_residual,
                working_maximum_scaled_residual=result.working_maximum_scaled_residual,
                certified_maximum_scaled_residual=result.certified_maximum_scaled_residual,
                certificate_dtype=result.certificate_dtype,
                fp64_certificate_checks=result.fp64_certificate_checks,
                fp64_certified_rhs=result.fp64_certified_rhs,
                fp64_certificate_rejections=result.fp64_certificate_rejections,
                fp64_certificate_restart_count=result.fp64_certificate_restart_count,
                fp64_certificate_work=result.fp64_certificate_work,
            )
            telemetry.append(window_telemetry)

            graph_snapshot = GraphSnapshot(
                neighbor_indices=graph.neighbor_indices,
                transition_weights=graph.transition_weights,
                edge_affinities=graph.edge_affinities,
                self_loop_fallback=graph.self_loop_fallback,
                num_nodes=graph.num_nodes,
                k=graph.k,
                affinity_power=graph.affinity_power,
            )
            checksums = None
            if want_checksums:
                checksums = compute_window_checksums(
                    scores[0], features[0], propagated,
                    graph.neighbor_indices, graph.transition_weights,
                )
            cache.append(
                CachedWindowState(
                    geometry=window,
                    window_index=window.index,
                    patch_grid_shape=grid_hw,
                    class_count=class_count,
                    s0=scores[0],
                    dino_features=features[0],
                    graph=graph_snapshot,
                    propagated_scores=propagated,
                    solver_summary=window_telemetry,
                    checksums=checksums,
                )
            )

        if torch.any(count_mat == 0):
            raise WindowCacheError("two-pass pass 1 left uncovered pixels")
        cache.seal()
    except Exception:
        cache.close()
        raise

    stitched = (preds / count_mat).to(output_device).detach().clone().contiguous()

    row_counts = plan.row_coverage_counts()
    col_counts = plan.col_coverage_counts()
    context = FirstPassImageContext(
        image_size=SpatialSize(h_img, w_img),
        plan=plan,
        class_count=class_count,
        common_patch_grid_shape=None if heterogeneous_grid else common_grid_shape,
        expected_window_count=plan.window_count,
        cached_window_count=cache.cached_window_count,
        min_coverage=min(row_counts) * min(col_counts),
        max_coverage=max(row_counts) * max(col_counts),
        stitched_scores=stitched,
        cache_total_bytes=cache.total_bytes(),
        pass_summary=_summarize_telemetry(telemetry),
    )
    return stitched, cache, context


def run_pass_two(
    inference,
    cache: ImageWindowCache,
    context: FirstPassImageContext,
    *,
    processor: Optional[SecondPassProcessor] = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Run pass 2: replay cached windows in the same row-major order,
    applying ``processor`` (identity by default) to each window's cached
    propagated scores, without rerunning the backbone, graph builder, or
    solver, and accumulate into a fresh second-pass stitch."""
    if processor is None:
        processor = identity_second_pass_processor
    if cache.state is not WindowCacheState.SEALED:
        raise WindowCacheError(
            f"cannot begin pass 2 from cache state {cache.state.value}; the cache must be sealed"
        )

    output_device = context.stitched_scores.device if device is None else device
    dtype = context.stitched_scores.dtype
    h_img, w_img = context.image_size.as_tuple()
    preds = torch.zeros((1, context.class_count, h_img, w_img), dtype=dtype, device=output_device)
    count_mat = torch.zeros((1, 1, h_img, w_img), dtype=dtype, device=output_device)

    cache.begin_replay()
    try:
        for window in context.plan.windows:
            cached = cache.get(window.index)
            propagated = cached.propagated_scores_copy(device=output_device)
            processed = processor(window.index, cached.geometry, propagated)
            if not torch.is_tensor(processed) or tuple(processed.shape) != tuple(propagated.shape):
                raise WindowCacheError(
                    "second-pass processor must return a tensor matching the cached P shape"
                )
            if not bool(torch.isfinite(processed).all()):
                raise WindowCacheError("second-pass processor output must be finite")
            masks = inference.model.masks_from_patch_scores(
                processed.unsqueeze(0), cached.patch_grid_shape, window.extent.as_tuple(),
            )
            _stitch_step(preds, count_mat, masks, window)
    finally:
        cache.end_replay()

    if torch.any(count_mat == 0):
        raise WindowCacheError("two-pass pass 2 left uncovered pixels")
    return preds / count_mat


@dataclass(frozen=True)
class TwoPassResult:
    first_pass_output: torch.Tensor
    second_pass_output: torch.Tensor
    context: FirstPassImageContext


def run_two_pass_slide_inference(
    inference,
    img: torch.Tensor,
    *,
    processor: Optional[SecondPassProcessor] = None,
    device: torch.device | str | None = None,
    want_checksums: bool = False,
) -> TwoPassResult:
    """Opt-in convenience entry point: run pass 1, then pass 2, and always
    release the cache afterward (success or exception). Two-pass caching is
    only ever exercised by explicitly calling this function or its
    building blocks; the canonical ``DINOTextSegInference.slide_inference``
    is never touched by this module and remains byte-for-byte unchanged.
    """
    first_pass_output, cache, context = run_pass_one(
        inference, img, device=device, want_checksums=want_checksums
    )
    try:
        second_pass_output = run_pass_two(inference, cache, context, processor=processor, device=device)
    finally:
        cache.close()
    return TwoPassResult(
        first_pass_output=first_pass_output,
        second_pass_output=second_pass_output,
        context=context,
    )


__all__ = [
    "CachedWindowByteAccounting",
    "CachedWindowState",
    "FirstPassImageContext",
    "GraphSnapshot",
    "ImageWindowCache",
    "SecondPassProcessor",
    "TwoPassResult",
    "WindowCacheError",
    "WindowCacheState",
    "WindowSolverTelemetry",
    "WindowSolverTelemetrySummary",
    "compute_window_checksums",
    "identity_second_pass_processor",
    "reconstruct_directed_topk_graph",
    "run_pass_one",
    "run_pass_two",
    "run_two_pass_slide_inference",
]
