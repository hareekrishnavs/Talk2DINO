"""Reusable, score-source-agnostic sliding-window stitching baselines.

Implements the parameter-free stitching protocols compared against the
canonical uniform-probability-average stitch: hard majority vote, a
strictly-positive half-sample Hann weighted average, and center-select
(winner-take-all by Hann weight). Every mode consumes the same immutable
per-window probability grids and geometry, never ground truth, and every
mode fails closed on any uncovered pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .sliding_window_geometry import SpatialSize, WindowGeometry


class StitchingError(ValueError):
    """Raised when stitching inputs or coverage are invalid."""


STITCH_MODE_UNIFORM = "uniform_probability_average"
STITCH_MODE_MAJORITY = "hard_majority_vote"
STITCH_MODE_HANN = "half_sample_hann"
STITCH_MODE_CENTER_SELECT = "center_select"
STITCH_MODES = (
    STITCH_MODE_UNIFORM,
    STITCH_MODE_MAJORITY,
    STITCH_MODE_HANN,
    STITCH_MODE_CENTER_SELECT,
)

SCORE_SOURCE_E3_UNARY = "e3_unary_q0"
SCORE_SOURCE_RWR_K12 = "rwr_k12_q"

# Section 7.5: the reusable stitcher must support at least this matrix of
# (score_source, mode) combinations. Baselines must never silently compare
# results carrying different score_source labels.
REQUIRED_SCORE_SOURCE_MODE_MATRIX = (
    (SCORE_SOURCE_E3_UNARY, STITCH_MODE_UNIFORM),
    (SCORE_SOURCE_E3_UNARY, STITCH_MODE_HANN),
    (SCORE_SOURCE_RWR_K12, STITCH_MODE_UNIFORM),
    (SCORE_SOURCE_RWR_K12, STITCH_MODE_MAJORITY),
    (SCORE_SOURCE_RWR_K12, STITCH_MODE_HANN),
    (SCORE_SOURCE_RWR_K12, STITCH_MODE_CENTER_SELECT),
)


def _require_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise StitchingError(f"{label} must be a torch.Tensor")
    return value


@dataclass(frozen=True)
class WindowProbabilityMap:
    """One immutable per-window dense probability grid ready for stitching.

    ``probabilities`` has shape ``[C, extent_h, extent_w]`` (the window's own
    pixel extent, matching clamped terminal windows) and must already be a
    probability in ``[0, 1]`` -- i.e. already sigmoided, never a raw logit.
    """

    geometry: WindowGeometry
    window_index: int
    probabilities: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, WindowGeometry):
            raise StitchingError("geometry must be a WindowGeometry")
        if isinstance(self.window_index, bool) or not isinstance(self.window_index, int):
            raise StitchingError("window_index must be an exact integer")
        if self.window_index < 0:
            raise StitchingError("window_index must be non-negative")
        probabilities = _require_tensor(self.probabilities, "probabilities")
        if probabilities.ndim != 3:
            raise StitchingError("probabilities must have shape [C, H, W]")
        if not probabilities.is_floating_point():
            raise StitchingError("probabilities must be floating point")
        if not bool(torch.isfinite(probabilities).all()):
            raise StitchingError("probabilities must be finite")
        extent = self.geometry.extent.as_tuple()
        if tuple(probabilities.shape[1:]) != extent:
            raise StitchingError(
                "probabilities spatial shape does not match window extent: "
                f"expected {extent}, got {tuple(probabilities.shape[1:])}"
            )
        minimum = float(probabilities.min().item())
        maximum = float(probabilities.max().item())
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise StitchingError(
                f"probabilities must lie in [0,1], observed range [{minimum}, {maximum}]"
            )
        object.__setattr__(
            self,
            "probabilities",
            probabilities.clamp(0.0, 1.0).detach().clone().contiguous(),
        )


@dataclass(frozen=True)
class StitchDiagnostics:
    mode: str
    score_source: str
    window_count: int
    min_coverage: int
    max_coverage: int
    tie_count: int = 0
    tie_fraction: float = 0.0
    min_weight_denominator: float = 0.0
    max_weight_denominator: float = 0.0


@dataclass(frozen=True)
class StitchResult:
    image: torch.Tensor  # [C, H, W], float32, in [0, 1] (or one-hot for majority)
    diagnostics: StitchDiagnostics


def _hann_weight_1d(extent: int, *, device, dtype) -> torch.Tensor:
    if extent < 1:
        raise StitchingError("Hann weight extent must be positive")
    n = torch.arange(extent, dtype=dtype, device=device)
    return torch.sin(math.pi * (n + 0.5) / extent) ** 2


def hann_window_weight(
    extent: tuple[int, int], *, device=None, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Strictly positive, center-weighted, separable half-sample Hann weight.

    ``h(n; N) = sin^2(pi * (n + 0.5) / N)``; no epsilon/floor parameter, no
    validation tuning, and (being strictly positive everywhere) no zero-
    weight boundary holes.
    """
    height, width = extent
    row = _hann_weight_1d(height, device=device, dtype=dtype)
    col = _hann_weight_1d(width, device=device, dtype=dtype)
    weight = row[:, None] * col[None, :]
    if not bool(torch.all(weight > 0)):
        raise StitchingError("Hann weights must be strictly positive")
    return weight


def _validate_common(
    windows: Sequence[WindowProbabilityMap], *, image_size: SpatialSize, class_count: int
) -> list[WindowProbabilityMap]:
    if not windows:
        raise StitchingError("stitch_windows requires at least one window")
    if not isinstance(image_size, SpatialSize):
        raise StitchingError("image_size must be a SpatialSize")
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count < 1:
        raise StitchingError("class_count must be a positive exact integer")
    ordered = sorted(windows, key=lambda item: item.window_index)
    seen: set[int] = set()
    for window in ordered:
        if window.probabilities.shape[0] != class_count:
            raise StitchingError(
                "window class dimension does not match class_count: "
                f"expected {class_count}, got {window.probabilities.shape[0]}"
            )
        if window.window_index in seen:
            raise StitchingError(f"duplicate window_index {window.window_index}")
        seen.add(window.window_index)
    return ordered


def _coverage_count_map(
    windows: Sequence[WindowProbabilityMap], h_img: int, w_img: int, device
) -> torch.Tensor:
    count = torch.zeros((h_img, w_img), dtype=torch.int64, device=device)
    for window in windows:
        rows, cols = window.geometry.accumulation_slice
        count[rows, cols] += 1
    return count


def _stitch_uniform(
    windows: Sequence[WindowProbabilityMap], class_count: int, h_img: int, w_img: int, dtype, device
) -> tuple[torch.Tensor, tuple[int, int]]:
    numerator = torch.zeros((class_count, h_img, w_img), dtype=dtype, device=device)
    count = torch.zeros((h_img, w_img), dtype=dtype, device=device)
    for window in windows:
        rows, cols = window.geometry.accumulation_slice
        numerator[:, rows, cols] += window.probabilities.to(dtype)
        count[rows, cols] += 1
    if bool((count == 0).any()):
        raise StitchingError("uniform stitching left uncovered pixels")
    result = numerator / count.unsqueeze(0)
    return result, (int(count.min().item()), int(count.max().item()))


def _stitch_majority(
    windows: Sequence[WindowProbabilityMap], class_count: int, h_img: int, w_img: int, device
) -> tuple[torch.Tensor, tuple[int, int], int, float]:
    vote_counts = torch.zeros((class_count, h_img, w_img), dtype=torch.int32, device=device)
    count = torch.zeros((h_img, w_img), dtype=torch.int64, device=device)
    for window in windows:
        rows, cols = window.geometry.accumulation_slice
        argmax_map = window.probabilities.argmax(dim=0)
        one_hot = F.one_hot(argmax_map, num_classes=class_count).permute(2, 0, 1).to(torch.int32)
        vote_counts[:, rows, cols] += one_hot
        count[rows, cols] += 1
    if bool((count == 0).any()):
        raise StitchingError("majority-vote stitching left uncovered pixels")
    # torch.max's documented tie behavior returns the first (lowest) index
    # of the maximal value, matching the required lowest-class-ID tie rule.
    max_votes, winner = vote_counts.max(dim=0)
    tie_mask = (vote_counts == max_votes.unsqueeze(0)).sum(dim=0) > 1
    tie_count = int(tie_mask.sum().item())
    tie_fraction = tie_count / float(h_img * w_img)
    result = torch.zeros((class_count, h_img, w_img), dtype=torch.float64, device=device)
    result.scatter_(0, winner.unsqueeze(0), 1.0)
    return result, (int(count.min().item()), int(count.max().item())), tie_count, tie_fraction


def _stitch_hann(
    windows: Sequence[WindowProbabilityMap], class_count: int, h_img: int, w_img: int, dtype, device
) -> tuple[torch.Tensor, tuple[int, int], float, float]:
    numerator = torch.zeros((class_count, h_img, w_img), dtype=dtype, device=device)
    denominator = torch.zeros((h_img, w_img), dtype=dtype, device=device)
    count = torch.zeros((h_img, w_img), dtype=torch.int64, device=device)
    for window in windows:
        rows, cols = window.geometry.accumulation_slice
        weight = hann_window_weight(window.geometry.extent.as_tuple(), device=device, dtype=dtype)
        numerator[:, rows, cols] += weight.unsqueeze(0) * window.probabilities.to(dtype)
        denominator[rows, cols] += weight
        count[rows, cols] += 1
    if bool((count == 0).any()):
        raise StitchingError("Hann stitching left uncovered pixels")
    if bool((denominator <= 0).any()):
        raise StitchingError("Hann stitching produced a non-positive accumulated denominator")
    result = numerator / denominator.unsqueeze(0)
    coverage = (int(count.min().item()), int(count.max().item()))
    return result, coverage, float(denominator.min().item()), float(denominator.max().item())


def _stitch_center_select(
    windows: Sequence[WindowProbabilityMap], class_count: int, h_img: int, w_img: int, dtype, device
) -> tuple[torch.Tensor, tuple[int, int], float, float]:
    best_weight = torch.full((h_img, w_img), -1.0, dtype=dtype, device=device)
    result = torch.zeros((class_count, h_img, w_img), dtype=dtype, device=device)
    count = torch.zeros((h_img, w_img), dtype=torch.int64, device=device)
    # Windows are already ordered ascending by window_index; a strict '>'
    # improvement test means the first (lowest-ordinal) window wins ties.
    for window in windows:
        rows, cols = window.geometry.accumulation_slice
        weight = hann_window_weight(window.geometry.extent.as_tuple(), device=device, dtype=dtype)
        count[rows, cols] += 1
        current_best = best_weight[rows, cols]
        improve = weight > current_best
        sub_result = result[:, rows, cols]
        result[:, rows, cols] = torch.where(improve.unsqueeze(0), window.probabilities.to(dtype), sub_result)
        best_weight[rows, cols] = torch.where(improve, weight, current_best)
    if bool((count == 0).any()):
        raise StitchingError("center-select stitching left uncovered pixels")
    coverage = (int(count.min().item()), int(count.max().item()))
    return result, coverage, float(best_weight.min().item()), float(best_weight.max().item())


def stitch_windows(
    windows: Sequence[WindowProbabilityMap],
    *,
    image_size: SpatialSize,
    class_count: int,
    mode: str,
    score_source: str,
) -> StitchResult:
    """Stitch one image's per-window probability grids under ``mode``.

    Never reads ground truth: no parameter through which it could arrive.
    ``score_source`` is a caller-supplied label recorded on the result so
    downstream comparisons cannot silently mix different score stages.
    """
    if mode not in STITCH_MODES:
        raise StitchingError(f"unknown stitching mode {mode!r}")
    if not isinstance(score_source, str) or not score_source:
        raise StitchingError("score_source must be a non-empty string")
    ordered = _validate_common(windows, image_size=image_size, class_count=class_count)
    h_img, w_img = image_size.as_tuple()
    device = ordered[0].probabilities.device
    dtype = torch.float64

    tie_count = 0
    tie_fraction = 0.0
    min_weight = max_weight = 0.0

    if mode == STITCH_MODE_UNIFORM:
        result, coverage = _stitch_uniform(ordered, class_count, h_img, w_img, dtype, device)
    elif mode == STITCH_MODE_MAJORITY:
        result, coverage, tie_count, tie_fraction = _stitch_majority(
            ordered, class_count, h_img, w_img, device
        )
    elif mode == STITCH_MODE_HANN:
        result, coverage, min_weight, max_weight = _stitch_hann(
            ordered, class_count, h_img, w_img, dtype, device
        )
    elif mode == STITCH_MODE_CENTER_SELECT:
        result, coverage, min_weight, max_weight = _stitch_center_select(
            ordered, class_count, h_img, w_img, dtype, device
        )
    else:  # pragma: no cover - guarded by the membership check above
        raise StitchingError(f"unhandled stitching mode {mode!r}")

    min_coverage, max_coverage = coverage
    diagnostics = StitchDiagnostics(
        mode=mode,
        score_source=score_source,
        window_count=len(ordered),
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        tie_count=tie_count,
        tie_fraction=tie_fraction,
        min_weight_denominator=min_weight,
        max_weight_denominator=max_weight,
    )
    return StitchResult(image=result.to(torch.float32), diagnostics=diagnostics)


__all__ = [
    "StitchingError",
    "STITCH_MODE_UNIFORM",
    "STITCH_MODE_MAJORITY",
    "STITCH_MODE_HANN",
    "STITCH_MODE_CENTER_SELECT",
    "STITCH_MODES",
    "SCORE_SOURCE_E3_UNARY",
    "SCORE_SOURCE_RWR_K12",
    "REQUIRED_SCORE_SOURCE_MODE_MATRIX",
    "WindowProbabilityMap",
    "StitchDiagnostics",
    "StitchResult",
    "hann_window_weight",
    "stitch_windows",
]
