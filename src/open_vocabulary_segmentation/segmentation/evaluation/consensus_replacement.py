"""Direct Consensus Replacement, Selective Unary Reversion, and their
strict-safe sequential variants (Sections 10-13).

Every function here consumes only the already-sealed, GT-free two-pass
window cache and a :class:`~.t4_audit.T4ImageSignal` built from it (itself
GT-free). Ground truth never reaches target selection, replacement-vector
computation, or safe acceptance -- it is used, if at all, only afterward by
:func:`compute_gt_accounting`, which is a pure measurement function over
already-finalized predictions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .sliding_window_geometry import SpatialSize
from .stitching_baselines import (
    STITCH_MODE_UNIFORM,
    StitchResult,
    WindowProbabilityMap,
    stitch_windows,
)
from .t4_audit import ConsensusObservation, T4ImageSignal, source_score_grid_anchor
from .t4_audit import _argmax_with_tie, _bilinear_stencil_from_fraction, _sample_bilinear
from .window_cache import ImageWindowCache


class ConsensusReplacementError(ValueError):
    """Raised when replacement inputs or acceptance state are invalid."""


DCR_HARD = "dcr_hard"
DCR_JURY_MEAN = "dcr_jury_mean"
SUR = "sur"
REPLACEMENT_KINDS = (DCR_HARD, DCR_JURY_MEAN, SUR)

TARGET_POPULATION_STRICT_T4 = "strict_t4"
TARGET_POPULATION_T4_PRIME = "t4_prime"
TARGET_POPULATIONS = (TARGET_POPULATION_STRICT_T4, TARGET_POPULATION_T4_PRIME)


class _SigmoidScoreCache:
    """Memoizes ``sigmoid(propagated_scores)`` per window, computed once."""

    def __init__(self, cache: ImageWindowCache):
        self._cache = cache
        self._sigmoid: dict[int, torch.Tensor] = {}

    def get(self, window_index: int) -> torch.Tensor:
        cached = self._sigmoid.get(window_index)
        if cached is None:
            cached = torch.sigmoid(self._cache.get(window_index).propagated_scores)
            self._sigmoid[window_index] = cached
        return cached


def _select_targets(signal: T4ImageSignal, target_population: str) -> tuple[ConsensusObservation, ...]:
    if target_population == TARGET_POPULATION_STRICT_T4:
        return tuple(obs for obs in signal.observations if obs.t4)
    if target_population == TARGET_POPULATION_T4_PRIME:
        return tuple(obs for obs in signal.observations if obs.t4_prime)
    raise ConsensusReplacementError(f"unknown target_population {target_population!r}")


def _one_hot(label: int, class_count: int) -> torch.Tensor:
    vector = torch.zeros(class_count, dtype=torch.float32)
    vector[label] = 1.0
    return vector


def _sample_juror_probability(
    cache: ImageWindowCache,
    sigmoid_cache: _SigmoidScoreCache,
    juror_window_id: int,
    global_y,
    global_x,
) -> torch.Tensor:
    juror_state = cache.get(juror_window_id)
    extent = juror_state.geometry.extent.as_tuple()
    local_y = global_y - juror_state.geometry.origin.row
    local_x = global_x - juror_state.geometry.origin.col
    stencil = _bilinear_stencil_from_fraction(local_y, local_x, extent, juror_state.patch_grid_shape)
    return _sample_bilinear(sigmoid_cache.get(juror_window_id), juror_state.patch_grid_shape, stencil)


def compute_dcr_jury_mean_vector(
    cache: ImageWindowCache, sigmoid_cache: _SigmoidScoreCache, obs: ConsensusObservation
) -> torch.Tensor:
    """Average the frozen sampled post-RWR probability vector from each
    agreeing OTHER window at the source's own global anchor. The source is
    structurally excluded: ``obs.other_covering_window_ids`` never includes
    the source window (see ``t4_audit.build_t4_signal_for_image``)."""
    if obs.source_window_id in obs.other_covering_window_ids:
        raise ConsensusReplacementError("jury must exclude the source window")
    if not obs.other_covering_window_ids:
        raise ConsensusReplacementError("DCR-JuryMean requires at least one juror window")
    source_state = cache.get(obs.source_window_id)
    global_y, global_x = source_score_grid_anchor(
        source_state.geometry, source_state.patch_grid_shape, obs.node_row, obs.node_col
    )
    samples = [
        _sample_juror_probability(cache, sigmoid_cache, juror_id, global_y, global_x)
        for juror_id in obs.other_covering_window_ids
    ]
    return torch.stack(samples, dim=0).mean(dim=0)


@dataclass(frozen=True)
class FrozenReplacement:
    """One immutable, pre-computed replacement vector for a target row.

    Computed exclusively from the original immutable first pass; frozen
    before any replacement (including this one) is applied to any window.
    """

    image_id: str
    source_window_id: int
    node_index: int
    node_row: int
    node_col: int
    consensus_label: int
    kind: str
    vector: torch.Tensor

    def __post_init__(self) -> None:
        if self.kind not in REPLACEMENT_KINDS:
            raise ConsensusReplacementError(f"unknown replacement kind {self.kind!r}")
        if not torch.is_tensor(self.vector) or self.vector.ndim != 1:
            raise ConsensusReplacementError("vector must be a 1-D tensor")
        if not bool(torch.isfinite(self.vector).all()):
            raise ConsensusReplacementError("vector must be finite")
        object.__setattr__(self, "vector", self.vector.detach().clone())


def build_frozen_replacements(
    cache: ImageWindowCache,
    signal: T4ImageSignal,
    *,
    kind: str,
    image_id: str,
    class_count: int,
    target_population: str = TARGET_POPULATION_STRICT_T4,
) -> tuple[FrozenReplacement, ...]:
    """Freeze all target records and replacement vectors before any
    replacement is applied. No parameter here can carry ground truth."""
    if kind not in REPLACEMENT_KINDS:
        raise ConsensusReplacementError(f"unknown replacement kind {kind!r}")
    targets = _select_targets(signal, target_population)
    sigmoid_cache = _SigmoidScoreCache(cache)
    frozen: list[FrozenReplacement] = []
    for obs in targets:
        if kind == DCR_HARD:
            vector = _one_hot(obs.consensus_label, class_count)
        elif kind == DCR_JURY_MEAN:
            vector = compute_dcr_jury_mean_vector(cache, sigmoid_cache, obs)
        else:  # SUR
            source_state = cache.get(obs.source_window_id)
            vector = torch.sigmoid(source_state.s0[obs.node_index])
        frozen.append(
            FrozenReplacement(
                image_id=image_id,
                source_window_id=obs.source_window_id,
                node_index=obs.node_index,
                node_row=obs.node_row,
                node_col=obs.node_col,
                consensus_label=obs.consensus_label,
                kind=kind,
                vector=vector,
            )
        )
    return tuple(frozen)


_PROBABILITY_RANGE_TOLERANCE = 1e-5


def _validate_probability_grid(grid: torch.Tensor, *, class_count: int, label: str) -> torch.Tensor:
    """Fail closed if ``grid`` is not already a probability grid.

    This is the raw-score/probability-domain boundary: a grid that still
    carries raw (pre-sigmoid) scores will typically fall far outside
    ``[0,1]`` and is rejected here, so a raw-score tensor cannot silently
    flow into :func:`probability_grid_to_mask`.
    """
    if not torch.is_tensor(grid) or grid.ndim != 2:
        raise ConsensusReplacementError(f"{label} must have shape [N, C]")
    if grid.shape[1] != class_count:
        raise ConsensusReplacementError(
            f"{label} class dimension mismatch: expected {class_count}, got {grid.shape[1]}"
        )
    if not grid.is_floating_point():
        raise ConsensusReplacementError(f"{label} must be floating point")
    if not bool(torch.isfinite(grid).all()):
        raise ConsensusReplacementError(f"{label} must be finite")
    minimum = float(grid.min().item())
    maximum = float(grid.max().item())
    if minimum < -_PROBABILITY_RANGE_TOLERANCE or maximum > 1.0 + _PROBABILITY_RANGE_TOLERANCE:
        raise ConsensusReplacementError(
            f"{label} must be a probability grid in [0,1] (tolerance {_PROBABILITY_RANGE_TOLERANCE}); "
            f"observed range [{minimum}, {maximum}] -- this looks like a raw score/logit grid, "
            "not a probability grid; use patch_scores_to_masks for raw pre-sigmoid scores instead"
        )
    return grid.clamp(0.0, 1.0)


def probability_grid_to_mask(
    probability_grid: torch.Tensor,
    grid_hw: tuple[int, int],
    output_hw: tuple[int, int],
    *,
    class_count: int,
) -> torch.Tensor:
    """Bilinearly upsample an ALREADY-sigmoided ``[N, C]`` probability grid
    to pixel resolution, using the same ``align_corners=True`` convention as
    production's ``patch_scores_to_masks``.

    Performs interpolation ONLY: no sigmoid, no softmax, no class-score
    normalization, no clamping beyond the input-validity check, no logit
    conversion, and no reordering of classes. Use ``patch_scores_to_masks``
    instead for raw, pre-sigmoid score grids -- the two must never be
    interchanged, which is exactly the domain-confusion bug this function
    exists to make impossible: DCR/SUR replacement vectors (one-hot,
    jury-mean, or ``sigmoid(S0)``) are already probabilities, and applying
    the internal sigmoid inside ``patch_scores_to_masks`` to them a second
    time would silently corrupt them (e.g. an exact one-hot ``[1,0,0]``
    would become ``[0.731, 0.5, 0.5]``).
    """
    grid = _validate_probability_grid(probability_grid, class_count=class_count, label="probability_grid")
    if math.prod(grid_hw) != grid.shape[0]:
        raise ConsensusReplacementError("probability grid node count does not match grid_hw")
    if len(output_hw) != 2 or any(type(value) is not int or value <= 0 for value in output_hw):
        raise ConsensusReplacementError("output_hw must contain two positive integers")
    reshaped = grid.reshape(1, *grid_hw, class_count).permute(0, 3, 1, 2)
    return F.interpolate(reshaped, output_hw, mode="bilinear", align_corners=True)[0]


@dataclass(frozen=True)
class ReplacementApplicationReport:
    target_rows: int
    unique_windows: int
    duplicate_global_anchor_count: int
    changed_windows: tuple[int, ...]


def apply_replacements_and_stitch(
    cache: ImageWindowCache,
    replacements: Sequence[FrozenReplacement],
    *,
    image_size: SpatialSize,
    class_count: int,
    mode: str = STITCH_MODE_UNIFORM,
    score_source: str,
) -> tuple[StitchResult, ReplacementApplicationReport]:
    """Make one mutable copy of each probability grid, replace all targeted
    rows simultaneously using the frozen vectors, stitch once.

    Sigmoid is applied exactly once, via ``_SigmoidScoreCache``, to
    reconstruct each window's baseline probability grid from its raw
    ``propagated_scores``. After that, only probability-domain values ever
    flow through this function (the original ``sigmoid(P)`` rows and the
    frozen replacement vectors, which are already probabilities) -- upsampling
    uses :func:`probability_grid_to_mask`, which performs interpolation only.
    """
    sigmoid_cache = _SigmoidScoreCache(cache)

    by_window: dict[int, dict[int, torch.Tensor]] = {}
    for replacement in replacements:
        by_window.setdefault(replacement.source_window_id, {})[replacement.node_index] = replacement.vector

    anchor_rounding = 1_000_000
    seen_anchors: set[tuple[int, int]] = set()
    for replacement in replacements:
        state = cache.get(replacement.source_window_id)
        global_y, global_x = source_score_grid_anchor(
            state.geometry, state.patch_grid_shape, replacement.node_row, replacement.node_col
        )
        seen_anchors.add((round(float(global_y) * anchor_rounding), round(float(global_x) * anchor_rounding)))

    windows_out: list[WindowProbabilityMap] = []
    for state in cache.windows_in_order():
        q = sigmoid_cache.get(state.window_index).clone()
        overrides = by_window.get(state.window_index)
        if overrides:
            for node_index, vector in overrides.items():
                q[node_index] = vector
        extent = state.geometry.extent.as_tuple()
        masks = probability_grid_to_mask(q, state.patch_grid_shape, extent, class_count=class_count)
        windows_out.append(
            WindowProbabilityMap(geometry=state.geometry, window_index=state.window_index, probabilities=masks)
        )

    result = stitch_windows(
        windows_out, image_size=image_size, class_count=class_count, mode=mode, score_source=score_source
    )
    report = ReplacementApplicationReport(
        target_rows=len(replacements),
        unique_windows=len(by_window),
        duplicate_global_anchor_count=len(replacements) - len(seen_anchors),
        changed_windows=tuple(sorted(by_window)),
    )
    return result, report


# ---------------------------------------------------------------------------
# Section 15: GT isolation and off-target accounting. Everything above this
# point never sees a GT tensor; every function below only ever READS
# already-finalized before/after predictions plus GT, purely for
# measurement -- GT can never influence target membership, replacement
# vectors, safe acceptance, stitching weights, or candidate order.
# ---------------------------------------------------------------------------


def _nearest_patch_node_map(grid_shape: tuple[int, int], extent: tuple[int, int]) -> torch.Tensor:
    """For each pixel in a window's own extent, the nearest patch-grid node
    under the committed align_corners=True mapping -- an explicit,
    deterministic full-pixel-resolution definition of which patch node
    "owns" a given output pixel."""

    def axis_map(grid_extent: int, out_extent: int) -> torch.Tensor:
        if out_extent <= 1 or grid_extent <= 1:
            return torch.zeros(out_extent, dtype=torch.int64)
        positions = torch.arange(out_extent, dtype=torch.float64) * (grid_extent - 1) / (out_extent - 1)
        return torch.round(positions).to(torch.int64).clamp(0, grid_extent - 1)

    grid_h, grid_w = grid_shape
    ext_h, ext_w = extent
    row_idx = axis_map(grid_h, ext_h)
    col_idx = axis_map(grid_w, ext_w)
    return row_idx[:, None] * grid_w + col_idx[None, :]


def compute_target_pixel_mask(
    cache: ImageWindowCache, replacements: Sequence[FrozenReplacement], image_size: SpatialSize
) -> torch.Tensor:
    """Deterministic full-pixel target/off-target partition: a pixel is
    "target" iff its nearest patch-grid node (under align_corners=True) in
    some window was replaced."""
    by_window: dict[int, set[int]] = {}
    for replacement in replacements:
        by_window.setdefault(replacement.source_window_id, set()).add(replacement.node_index)

    h_img, w_img = image_size.as_tuple()
    mask = torch.zeros((h_img, w_img), dtype=torch.bool)
    for state in cache.windows_in_order():
        replaced_nodes = by_window.get(state.window_index)
        if not replaced_nodes:
            continue
        extent = state.geometry.extent.as_tuple()
        node_map = _nearest_patch_node_map(state.patch_grid_shape, extent)
        replaced_tensor = torch.tensor(sorted(replaced_nodes), dtype=torch.int64)
        window_mask = torch.isin(node_map, replaced_tensor)
        rows, cols = state.geometry.accumulation_slice
        mask[rows, cols] |= window_mask
    return mask


@dataclass(frozen=True)
class GTAccounting:
    target_gt_gains: int
    target_gt_losses: int
    off_target_gt_gains: int
    off_target_gt_losses: int
    net_changed_correct_pixels: int
    changed_pixel_count: int
    per_class_changes: Mapping[int, Mapping[str, int]] = field(default_factory=dict)


def compute_gt_accounting(
    before_probabilities: torch.Tensor,
    after_probabilities: torch.Tensor,
    gt: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    ignore_index: int,
) -> GTAccounting:
    if before_probabilities.shape != after_probabilities.shape:
        raise ConsensusReplacementError("before/after probability shapes must match")
    if gt.shape != target_mask.shape or gt.shape != before_probabilities.shape[1:]:
        raise ConsensusReplacementError("gt/target_mask/prediction spatial shapes must match")
    before_label = before_probabilities.argmax(dim=0)
    after_label = after_probabilities.argmax(dim=0)
    valid = gt != ignore_index
    changed = (before_label != after_label) & valid
    before_correct = (before_label == gt) & valid
    after_correct = (after_label == gt) & valid
    gains = (~before_correct) & after_correct & changed
    losses = before_correct & (~after_correct) & changed

    target_gains = int((gains & target_mask).sum().item())
    target_losses = int((losses & target_mask).sum().item())
    off_target_gains = int((gains & ~target_mask).sum().item())
    off_target_losses = int((losses & ~target_mask).sum().item())

    per_class: dict[int, dict[str, int]] = {}
    for class_id in torch.unique(gt[valid]).tolist():
        class_mask = gt == class_id
        class_gains = int((gains & class_mask).sum().item())
        class_losses = int((losses & class_mask).sum().item())
        if class_gains or class_losses:
            per_class[int(class_id)] = {"gains": class_gains, "losses": class_losses}

    return GTAccounting(
        target_gt_gains=target_gains,
        target_gt_losses=target_losses,
        off_target_gt_gains=off_target_gains,
        off_target_gt_losses=off_target_losses,
        net_changed_correct_pixels=(target_gains + off_target_gains) - (target_losses + off_target_losses),
        changed_pixel_count=int(changed.sum().item()),
        per_class_changes=per_class,
    )


# ---------------------------------------------------------------------------
# Section 12: strict-safe sequential DCR/SUR.
# ---------------------------------------------------------------------------


class _MutableQState:
    """Per-image mutable ``sigmoid(P)`` grids used only by the strict-safe
    sequential acceptance loop; independent of ``apply_replacements_and_stitch``."""

    def __init__(self, cache: ImageWindowCache, sigmoid_cache: _SigmoidScoreCache):
        self._cache = cache
        self._grids: dict[int, torch.Tensor] = {
            state.window_index: sigmoid_cache.get(state.window_index).clone()
            for state in cache.windows_in_order()
        }

    def row(self, window_index: int, node_index: int) -> torch.Tensor:
        return self._grids[window_index][node_index].clone()

    def grid(self, window_index: int) -> torch.Tensor:
        return self._grids[window_index]

    def apply(self, window_index: int, node_index: int, vector: torch.Tensor) -> None:
        self._grids[window_index][node_index] = vector


def evaluate_stitched_label_at_anchor(
    cache: ImageWindowCache, obs: ConsensusObservation, state: _MutableQState
) -> int:
    """The uniform-stitching argmax G(anchor) under the CURRENT (possibly
    partially mutated) state, sampled exactly at ``obs``'s own anchor."""
    source_state = cache.get(obs.source_window_id)
    global_y, global_x = source_score_grid_anchor(
        source_state.geometry, source_state.patch_grid_shape, obs.node_row, obs.node_col
    )
    samples = [state.row(obs.source_window_id, obs.node_index)]
    for juror_id in obs.other_covering_window_ids:
        juror_state = cache.get(juror_id)
        extent = juror_state.geometry.extent.as_tuple()
        local_y = global_y - juror_state.geometry.origin.row
        local_x = global_x - juror_state.geometry.origin.col
        stencil = _bilinear_stencil_from_fraction(local_y, local_x, extent, juror_state.patch_grid_shape)
        samples.append(_sample_bilinear(state.grid(juror_id), juror_state.patch_grid_shape, stencil))
    label, _tie = _argmax_with_tie(torch.stack(samples, dim=0).mean(dim=0))
    return label


@dataclass(frozen=True)
class StrictSafeReport:
    candidates: int
    accepted: int
    rejected: int
    acceptance_rate: float
    violations_fixed: int
    new_violations: int
    targets_resolved: int
    runtime_seconds: float
    accepted_candidates: tuple[FrozenReplacement, ...] = ()
    rejected_candidates: tuple[FrozenReplacement, ...] = ()


def run_strict_safe_sequential(
    cache: ImageWindowCache,
    signal: T4ImageSignal,
    replacements: Sequence[FrozenReplacement],
) -> StrictSafeReport:
    """Strict-safe sequential acceptance: a candidate is accepted iff the
    protected-anchor violation set strictly shrinks (``E(after) ⊂
    E(before)``) -- no cardinality-only fallback, no threshold, no GT."""
    start = time.perf_counter()
    sigmoid_cache = _SigmoidScoreCache(cache)
    state = _MutableQState(cache, sigmoid_cache)
    protected = signal.observations

    def violation_set() -> frozenset[tuple[int, int]]:
        return frozenset(
            (obs.source_window_id, obs.node_index)
            for obs in protected
            if evaluate_stitched_label_at_anchor(cache, obs, state) != obs.consensus_label
        )

    initial_violations = violation_set()
    current_violations = initial_violations
    candidates = sorted(
        replacements, key=lambda r: (r.image_id, r.source_window_id, r.node_index, r.kind)
    )
    accepted: list[FrozenReplacement] = []
    rejected: list[FrozenReplacement] = []
    for candidate in candidates:
        previous_row = state.row(candidate.source_window_id, candidate.node_index)
        state.apply(candidate.source_window_id, candidate.node_index, candidate.vector)
        candidate_violations = violation_set()
        if candidate_violations < current_violations:
            accepted.append(candidate)
            current_violations = candidate_violations
        else:
            state.apply(candidate.source_window_id, candidate.node_index, previous_row)
            rejected.append(candidate)

    new_violations = len(current_violations - initial_violations)
    if new_violations != 0:
        raise ConsensusReplacementError(
            "strict-safe acceptance invariant violated: new protected violations were introduced"
        )
    resolved_targets = sum(
        1
        for candidate in candidates
        if (candidate.source_window_id, candidate.node_index) not in current_violations
    )
    total = len(candidates)
    return StrictSafeReport(
        candidates=total,
        accepted=len(accepted),
        rejected=len(rejected),
        acceptance_rate=(len(accepted) / total) if total else 0.0,
        violations_fixed=len(initial_violations) - len(current_violations),
        new_violations=new_violations,
        targets_resolved=resolved_targets,
        runtime_seconds=time.perf_counter() - start,
        accepted_candidates=tuple(accepted),
        rejected_candidates=tuple(rejected),
    )


__all__ = [
    "ConsensusReplacementError",
    "DCR_HARD",
    "DCR_JURY_MEAN",
    "SUR",
    "REPLACEMENT_KINDS",
    "TARGET_POPULATION_STRICT_T4",
    "TARGET_POPULATION_T4_PRIME",
    "TARGET_POPULATIONS",
    "FrozenReplacement",
    "build_frozen_replacements",
    "compute_dcr_jury_mean_vector",
    "probability_grid_to_mask",
    "ReplacementApplicationReport",
    "apply_replacements_and_stitch",
    "compute_target_pixel_mask",
    "GTAccounting",
    "compute_gt_accounting",
    "evaluate_stitched_label_at_anchor",
    "StrictSafeReport",
    "run_strict_safe_sequential",
]
