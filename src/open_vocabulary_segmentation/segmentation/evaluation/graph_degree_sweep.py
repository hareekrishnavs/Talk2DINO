"""Same-feature, same-unary directed graph-degree sweep (Section 9).

Rebuilds a fresh directed top-k graph from the SAME cached, immutable DINO
patch features and re-solves canonical RWR from the SAME cached unary
scores ``S0``, for each requested ``k``. Zero backbone forwards happen
here: the backbone already ran exactly once when the two-pass window
cache was built, and this module only ever reads that cache's owned
``s0``/``dino_features`` tensors.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .sliding_window_geometry import SpatialSize
from .stitching_baselines import (
    STITCH_MODE_UNIFORM,
    StitchResult,
    WindowProbabilityMap,
    stitch_windows,
)
from .window_cache import CachedWindowState, ImageWindowCache


class GraphDegreeSweepError(ValueError):
    """Raised when graph-degree sweep inputs are invalid."""


CANONICAL_K = 12
DEFAULT_K_VALUES: tuple[int, ...] = (4, 6, 8, 10, 11, 12, 16, 32)


def _require_positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GraphDegreeSweepError(f"{label} must be a positive exact integer")
    return value


def run_k_sweep_for_window(
    state: CachedWindowState,
    *,
    k_values: Sequence[int],
    affinity_power: float,
    alpha: float,
    rtol: float | None,
    atol: float | None,
    max_iter: int,
) -> dict[int, object]:
    """Rebuild the graph and re-solve RWR at every requested ``k`` for one
    cached window, reusing the same cached ``s0``/``dino_features``.

    One graph build and one CGLS solve happen per ``k``; the backbone is
    never invoked here (its output is read from the already-sealed cache).
    Returns a ``{k: RWRSolveResult}`` mapping, so solver telemetry (from
    each ``RWRSolveResult``'s own fields) stays separated by ``k``.
    """
    from models.dinotext.cover_dr.graph import build_directed_topk_graph
    from models.dinotext.cover_dr.rwr import solve_rwr_cgls

    if not isinstance(state, CachedWindowState):
        raise GraphDegreeSweepError("state must be a CachedWindowState")
    if not k_values:
        raise GraphDegreeSweepError("k_values must be non-empty")
    seen: set[int] = set()
    for k in k_values:
        _require_positive_int(k, "k")
        if k in seen:
            raise GraphDegreeSweepError(f"duplicate k value {k}")
        seen.add(k)

    features = state.dino_features_copy()
    s0 = state.s0_copy()
    results: dict[int, object] = {}
    for k in k_values:
        graph = build_directed_topk_graph(features, k=k, affinity_power=affinity_power)
        results[k] = solve_rwr_cgls(
            graph, s0, alpha=alpha, rtol=rtol, atol=atol, max_iter=max_iter
        )
    return results


def stitch_k_sweep_for_image(
    cache: ImageWindowCache,
    *,
    image_size: SpatialSize,
    class_count: int,
    k_values: Sequence[int],
    affinity_power: float,
    alpha: float,
    rtol: float | None,
    atol: float | None,
    max_iter: int,
    mode: str = STITCH_MODE_UNIFORM,
) -> tuple[dict[int, StitchResult], dict[int, dict[int, object]]]:
    """Evaluate every ``k`` for one image before releasing its cache.

    Returns ``(stitched_results_by_k, solve_results_by_window_by_k)``. Never
    persists raw per-window features beyond the caller-owned ``cache``.
    """
    from models.dinotext.cover_dr.inference import patch_scores_to_masks

    per_k_windows: dict[int, list[WindowProbabilityMap]] = {k: [] for k in k_values}
    solve_results_by_window: dict[int, dict[int, object]] = {}
    for state in cache.windows_in_order():
        sweep = run_k_sweep_for_window(
            state,
            k_values=k_values,
            affinity_power=affinity_power,
            alpha=alpha,
            rtol=rtol,
            atol=atol,
            max_iter=max_iter,
        )
        solve_results_by_window[state.window_index] = sweep
        extent = state.geometry.extent.as_tuple()
        for k, solved in sweep.items():
            scores = solved.scores.unsqueeze(0)
            masks = patch_scores_to_masks(scores, state.patch_grid_shape, extent)[0]
            per_k_windows[k].append(
                WindowProbabilityMap(
                    geometry=state.geometry,
                    window_index=state.window_index,
                    probabilities=masks,
                )
            )

    stitched: dict[int, StitchResult] = {}
    for k, windows in per_k_windows.items():
        stitched[k] = stitch_windows(
            windows,
            image_size=image_size,
            class_count=class_count,
            mode=mode,
            score_source=f"rwr_k{k}_q",
        )
    return stitched, solve_results_by_window


__all__ = [
    "GraphDegreeSweepError",
    "CANONICAL_K",
    "DEFAULT_K_VALUES",
    "run_k_sweep_for_window",
    "stitch_k_sweep_for_image",
]
