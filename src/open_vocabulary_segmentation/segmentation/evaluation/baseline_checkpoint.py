"""Typed, versioned checkpoint/resume for the stitching/k-sweep/DCR/SUR
baseline suite (Section 6 of the baseline-repair spec).

Checkpoints are only ever written after a complete image transaction: every
declared baseline (E3 controls, RWR controls, the full k-sweep, DCR/SUR,
and strict-safe variants) has finished for that image before
``save_checkpoint_atomic`` is called. If a run is interrupted mid-image, no
checkpoint reflecting that partial image was ever written, so resume simply
re-processes that one image from scratch against the last complete
checkpoint -- there is no partial per-image state to discard because none
is ever persisted.

Reuses :class:`~.trust_centrality_harness.StreamingSegmentationMetricAccumulator`
for full-precision per-baseline metrics (the same verified, mmseg-backed
accumulator the trust/centrality harness already uses) rather than
reimplementing area-statistics bookkeeping.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .trust_centrality_harness import StreamingSegmentationMetricAccumulator


class BaselineCheckpointError(ValueError):
    """Raised on any baseline-checkpoint validation or compatibility failure."""


CHECKPOINT_SCHEMA_VERSION = "talk2dino-baseline-suite-checkpoint-v1"
RUN_STATUS_PARTIAL = "partial"
RUN_STATUS_COMPLETE = "complete"

# -- canonical baseline names -------------------------------------------

BASELINE_E3_UNIFORM = "e3_unary_q0__uniform_probability_average"
BASELINE_E3_HANN = "e3_unary_q0__half_sample_hann"
BASELINE_RWR_K12_UNIFORM = "rwr_k12_q__uniform_probability_average"
BASELINE_RWR_K12_MAJORITY = "rwr_k12_q__hard_majority_vote"
BASELINE_RWR_K12_HANN = "rwr_k12_q__half_sample_hann"
BASELINE_RWR_K12_CENTER_SELECT = "rwr_k12_q__center_select"
BASELINE_DCR_HARD = "dcr_hard"
BASELINE_DCR_JURY_MEAN = "dcr_jury_mean"
BASELINE_SUR = "sur"
BASELINE_STRICT_SAFE_DCR_HARD = "strict_safe_dcr_hard"
BASELINE_STRICT_SAFE_DCR_JURY_MEAN = "strict_safe_dcr_jury_mean"
BASELINE_STRICT_SAFE_SUR = "strict_safe_sur"

assert BASELINE_RWR_K12_UNIFORM == "rwr_k12_q__uniform_probability_average"

# BASELINE_RWR_K12_UNIFORM is deliberately NOT listed here: it is the same
# baseline as k_sweep_baseline_name(12) (canonical k=12 uniform stitching
# IS one point of the k-sweep), so declared_baseline_names() below adds it
# exactly once, via the k-sweep loop, rather than twice under two names.
_FIXED_BASELINE_NAMES = (
    BASELINE_E3_UNIFORM,
    BASELINE_E3_HANN,
    BASELINE_RWR_K12_MAJORITY,
    BASELINE_RWR_K12_HANN,
    BASELINE_RWR_K12_CENTER_SELECT,
    BASELINE_DCR_HARD,
    BASELINE_DCR_JURY_MEAN,
    BASELINE_SUR,
    BASELINE_STRICT_SAFE_DCR_HARD,
    BASELINE_STRICT_SAFE_DCR_JURY_MEAN,
    BASELINE_STRICT_SAFE_SUR,
)

REPLACEMENT_VARIANT_NAMES = (
    BASELINE_DCR_HARD,
    BASELINE_DCR_JURY_MEAN,
    BASELINE_SUR,
    BASELINE_STRICT_SAFE_DCR_HARD,
    BASELINE_STRICT_SAFE_DCR_JURY_MEAN,
    BASELINE_STRICT_SAFE_SUR,
)


def k_sweep_baseline_name(k: int) -> str:
    return f"rwr_k{k}_q__uniform_probability_average"


def declared_baseline_names(k_values: Sequence[int]) -> tuple[str, ...]:
    """The complete, exact set of baselines a full checkpoint must cover."""
    names = list(_FIXED_BASELINE_NAMES)
    names.extend(k_sweep_baseline_name(k) for k in k_values)
    if len(set(names)) != len(names):
        raise BaselineCheckpointError("declared baseline names must be unique")
    return tuple(names)


# -- small validation helpers --------------------------------------------


def _require_str(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise BaselineCheckpointError(f"{label} must be an exact non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise BaselineCheckpointError(f"{label} must be an exact boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise BaselineCheckpointError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise BaselineCheckpointError(f"{label} must be at least {minimum}")
    return value


def _require_nonneg_int(value: Any, label: str) -> int:
    return _require_int(value, label, minimum=0)


def _require_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineCheckpointError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BaselineCheckpointError(f"{label} must be finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    token = _require_str(value, label)
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise BaselineCheckpointError(f"{label} must be a lowercase SHA256")
    return token


# ---------------------------------------------------------------------------
# Provenance / compatibility context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineCheckpointCompatibilityContext:
    """Everything that must match between save-time and resume-time.

    Any mismatch fails closed: a checkpoint written under a different
    baseline identity, k-set, stitching definition, DCR/SUR definition,
    safe-acceptance rule, or source tree must never be silently resumed.
    """

    git_head: str
    git_branch: str
    baseline_identity_name: str
    baseline_manifest_sha256: str
    e3_identity_sha256: str
    rwr_identity_sha256: str
    canonical_config_sha256: str
    checkpoint_sha256: str
    tracked_diff_sha256: str
    untracked_source_sha256: str
    crop: tuple[int, int]
    stride: tuple[int, int]
    k_values: tuple[int, ...]
    stitching_modes: tuple[str, ...]
    dcr_variants: tuple[str, ...]
    sur_definition: str
    strict_safe_rule: str
    class_count: int
    ignore_label: int
    metric_unit_contract: str
    dataset_length: int

    def __post_init__(self) -> None:
        for name in (
            "git_head", "git_branch", "baseline_identity_name", "baseline_manifest_sha256",
            "e3_identity_sha256", "rwr_identity_sha256", "canonical_config_sha256",
            "checkpoint_sha256", "tracked_diff_sha256", "untracked_source_sha256",
            "sur_definition", "strict_safe_rule", "metric_unit_contract",
        ):
            _require_str(getattr(self, name), name)
        for pair_name in ("crop", "stride"):
            pair = getattr(self, pair_name)
            if (
                not isinstance(pair, tuple) or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in pair)
            ):
                raise BaselineCheckpointError(f"{pair_name} must be a pair of positive exact integers")
        if not isinstance(self.k_values, tuple) or not self.k_values:
            raise BaselineCheckpointError("k_values must be a non-empty exact tuple")
        if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in self.k_values):
            raise BaselineCheckpointError("k_values elements must be positive exact integers")
        if len(set(self.k_values)) != len(self.k_values):
            raise BaselineCheckpointError("k_values must not contain duplicates")
        if not isinstance(self.stitching_modes, tuple) or not all(isinstance(v, str) for v in self.stitching_modes):
            raise BaselineCheckpointError("stitching_modes must be a tuple of str")
        if not isinstance(self.dcr_variants, tuple) or not all(isinstance(v, str) for v in self.dcr_variants):
            raise BaselineCheckpointError("dcr_variants must be a tuple of str")
        _require_int(self.class_count, "class_count", minimum=1)
        _require_nonneg_int(self.ignore_label, "ignore_label")
        _require_int(self.dataset_length, "dataset_length", minimum=1)


# ---------------------------------------------------------------------------
# Call accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallAccounting:
    model_forwards: int = 0
    feature_extractions: int = 0
    graph_builds: int = 0
    solver_calls: int = 0
    sigmoid_calls: int = 0
    interpolation_calls: int = 0
    stitching_calls: int = 0

    def __post_init__(self) -> None:
        for name in (
            "model_forwards", "feature_extractions", "graph_builds", "solver_calls",
            "sigmoid_calls", "interpolation_calls", "stitching_calls",
        ):
            _require_nonneg_int(getattr(self, name), name)

    def __add__(self, other: "CallAccounting") -> "CallAccounting":
        if not isinstance(other, CallAccounting):
            return NotImplemented
        return CallAccounting(
            model_forwards=self.model_forwards + other.model_forwards,
            feature_extractions=self.feature_extractions + other.feature_extractions,
            graph_builds=self.graph_builds + other.graph_builds,
            solver_calls=self.solver_calls + other.solver_calls,
            sigmoid_calls=self.sigmoid_calls + other.sigmoid_calls,
            interpolation_calls=self.interpolation_calls + other.interpolation_calls,
            stitching_calls=self.stitching_calls + other.stitching_calls,
        )

    def state_dict(self) -> dict:
        return {
            "model_forwards": self.model_forwards, "feature_extractions": self.feature_extractions,
            "graph_builds": self.graph_builds, "solver_calls": self.solver_calls,
            "sigmoid_calls": self.sigmoid_calls, "interpolation_calls": self.interpolation_calls,
            "stitching_calls": self.stitching_calls,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "CallAccounting":
        return cls(**{name: _require_nonneg_int(state[name], name) for name in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Per-k solver telemetry
# ---------------------------------------------------------------------------


class KSweepTelemetryAccumulator:
    """Running per-k totals over ``RWRSolveResult``-shaped solves, one
    instance per k. Never invokes the solver itself."""

    def __init__(self) -> None:
        self.graph_count = 0
        self.solve_count = 0
        self.converged_window_count = 0
        self.total_iterations = 0
        self.total_work = 0
        self.total_residual_replacements = 0
        self.total_restarts = 0
        self.total_fallback_rows = 0
        self.maximum_scaled_residual = 0.0

    def absorb(self, solve_result: Any, *, fallback_rows: int = 0) -> None:
        self.graph_count += 1
        self.solve_count += 1
        if bool(getattr(solve_result, "converged", False)):
            self.converged_window_count += 1
        self.total_iterations += int(getattr(solve_result, "iterations", 0))
        self.total_work += int(getattr(solve_result, "work_count", 0))
        self.total_residual_replacements += int(getattr(solve_result, "total_residual_replacement_count", 0))
        self.total_restarts += int(getattr(solve_result, "total_restart_count", 0))
        self.total_fallback_rows += _require_nonneg_int(fallback_rows, "fallback_rows")
        self.maximum_scaled_residual = max(
            self.maximum_scaled_residual, float(getattr(solve_result, "maximum_scaled_residual", 0.0))
        )

    def state_dict(self) -> dict:
        return {
            "graph_count": self.graph_count, "solve_count": self.solve_count,
            "converged_window_count": self.converged_window_count,
            "total_iterations": self.total_iterations, "total_work": self.total_work,
            "total_residual_replacements": self.total_residual_replacements,
            "total_restarts": self.total_restarts, "total_fallback_rows": self.total_fallback_rows,
            "maximum_scaled_residual": self.maximum_scaled_residual,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "KSweepTelemetryAccumulator":
        acc = cls()
        acc.graph_count = _require_nonneg_int(state["graph_count"], "graph_count")
        acc.solve_count = _require_nonneg_int(state["solve_count"], "solve_count")
        if acc.graph_count != acc.solve_count:
            raise BaselineCheckpointError(
                "k-sweep telemetry graph_count must equal solve_count (one graph, one solve, per k per window)"
            )
        acc.converged_window_count = _require_nonneg_int(state["converged_window_count"], "converged_window_count")
        if acc.converged_window_count > acc.solve_count:
            raise BaselineCheckpointError("converged_window_count must not exceed solve_count")
        acc.total_iterations = _require_nonneg_int(state["total_iterations"], "total_iterations")
        acc.total_work = _require_nonneg_int(state["total_work"], "total_work")
        acc.total_residual_replacements = _require_nonneg_int(state["total_residual_replacements"], "total_residual_replacements")
        acc.total_restarts = _require_nonneg_int(state["total_restarts"], "total_restarts")
        acc.total_fallback_rows = _require_nonneg_int(state["total_fallback_rows"], "total_fallback_rows")
        acc.maximum_scaled_residual = _require_finite_float(state["maximum_scaled_residual"], "maximum_scaled_residual")
        if acc.maximum_scaled_residual < 0:
            raise BaselineCheckpointError("maximum_scaled_residual must be non-negative")
        return acc


# ---------------------------------------------------------------------------
# Stitching diagnostics
# ---------------------------------------------------------------------------


class StitchingDiagnosticsAccumulator:
    """Running per-mode coverage/weight/tie diagnostics, one instance per
    stitching mode. Consumes ``StitchDiagnostics`` objects only."""

    def __init__(self) -> None:
        self.window_count = 0
        self.image_count = 0
        self.min_coverage: int | None = None
        self.max_coverage = 0
        self.min_weight_denominator: float | None = None
        self.max_weight_denominator = 0.0
        self.majority_tie_count = 0
        self.center_select_tie_count = 0
        self.uncovered_pixel_count = 0

    def absorb(self, diagnostics: Any, *, mode: str) -> None:
        self.image_count += 1
        self.window_count += int(getattr(diagnostics, "window_count", 0))
        min_cov = int(getattr(diagnostics, "min_coverage", 0))
        max_cov = int(getattr(diagnostics, "max_coverage", 0))
        self.min_coverage = min_cov if self.min_coverage is None else min(self.min_coverage, min_cov)
        self.max_coverage = max(self.max_coverage, max_cov)
        min_w = float(getattr(diagnostics, "min_weight_denominator", 0.0))
        max_w = float(getattr(diagnostics, "max_weight_denominator", 0.0))
        if min_w > 0:
            self.min_weight_denominator = min_w if self.min_weight_denominator is None else min(self.min_weight_denominator, min_w)
        self.max_weight_denominator = max(self.max_weight_denominator, max_w)
        if mode == "hard_majority_vote":
            self.majority_tie_count += int(getattr(diagnostics, "tie_count", 0))
        elif mode == "center_select":
            self.center_select_tie_count += int(getattr(diagnostics, "tie_count", 0))

    def state_dict(self) -> dict:
        return {
            "window_count": self.window_count, "image_count": self.image_count,
            "min_coverage": self.min_coverage, "max_coverage": self.max_coverage,
            "min_weight_denominator": self.min_weight_denominator, "max_weight_denominator": self.max_weight_denominator,
            "majority_tie_count": self.majority_tie_count, "center_select_tie_count": self.center_select_tie_count,
            "uncovered_pixel_count": self.uncovered_pixel_count,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "StitchingDiagnosticsAccumulator":
        acc = cls()
        acc.window_count = _require_nonneg_int(state["window_count"], "window_count")
        acc.image_count = _require_nonneg_int(state["image_count"], "image_count")
        acc.min_coverage = None if state["min_coverage"] is None else _require_nonneg_int(state["min_coverage"], "min_coverage")
        acc.max_coverage = _require_nonneg_int(state["max_coverage"], "max_coverage")
        acc.min_weight_denominator = (
            None if state["min_weight_denominator"] is None
            else _require_finite_float(state["min_weight_denominator"], "min_weight_denominator")
        )
        acc.max_weight_denominator = _require_finite_float(state["max_weight_denominator"], "max_weight_denominator")
        acc.majority_tie_count = _require_nonneg_int(state["majority_tie_count"], "majority_tie_count")
        acc.center_select_tie_count = _require_nonneg_int(state["center_select_tie_count"], "center_select_tie_count")
        acc.uncovered_pixel_count = _require_nonneg_int(state["uncovered_pixel_count"], "uncovered_pixel_count")
        return acc


# ---------------------------------------------------------------------------
# DCR/SUR/strict-safe diagnostics
# ---------------------------------------------------------------------------


class ReplacementDiagnosticsAccumulator:
    """Running per-variant DCR/SUR/strict-safe diagnostics, one instance
    per replacement variant name."""

    def __init__(self) -> None:
        self.target_rows = 0
        self.unique_target_windows = 0
        self.candidate_count = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.violations_fixed = 0
        self.new_violations = 0
        self.targets_resolved = 0
        self.changed_pixel_count = 0
        self.target_gt_gains = 0
        self.target_gt_losses = 0
        self.off_target_gt_gains = 0
        self.off_target_gt_losses = 0
        self.per_class_changes: dict[int, dict[str, int]] = {}

    def absorb_application(self, report: Any) -> None:
        self.target_rows += int(getattr(report, "target_rows", 0))
        self.unique_target_windows += int(getattr(report, "unique_windows", 0))

    def absorb_strict_safe(self, report: Any) -> None:
        self.candidate_count += int(getattr(report, "candidates", 0))
        self.accepted_count += int(getattr(report, "accepted", 0))
        self.rejected_count += int(getattr(report, "rejected", 0))
        self.violations_fixed += int(getattr(report, "violations_fixed", 0))
        new_violations = int(getattr(report, "new_violations", 0))
        if new_violations != 0:
            raise BaselineCheckpointError(
                "strict-safe acceptance must never introduce new protected violations "
                f"(observed new_violations={new_violations})"
            )
        self.new_violations += new_violations
        self.targets_resolved += int(getattr(report, "targets_resolved", 0))

    def absorb_gt_accounting(self, accounting: Any) -> None:
        self.changed_pixel_count += int(getattr(accounting, "changed_pixel_count", 0))
        self.target_gt_gains += int(getattr(accounting, "target_gt_gains", 0))
        self.target_gt_losses += int(getattr(accounting, "target_gt_losses", 0))
        self.off_target_gt_gains += int(getattr(accounting, "off_target_gt_gains", 0))
        self.off_target_gt_losses += int(getattr(accounting, "off_target_gt_losses", 0))
        for class_id, changes in dict(getattr(accounting, "per_class_changes", {})).items():
            entry = self.per_class_changes.setdefault(int(class_id), {"gains": 0, "losses": 0})
            entry["gains"] += int(changes.get("gains", 0))
            entry["losses"] += int(changes.get("losses", 0))

    def state_dict(self) -> dict:
        return {
            "target_rows": self.target_rows, "unique_target_windows": self.unique_target_windows,
            "candidate_count": self.candidate_count, "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count, "violations_fixed": self.violations_fixed,
            "new_violations": self.new_violations, "targets_resolved": self.targets_resolved,
            "changed_pixel_count": self.changed_pixel_count,
            "target_gt_gains": self.target_gt_gains, "target_gt_losses": self.target_gt_losses,
            "off_target_gt_gains": self.off_target_gt_gains, "off_target_gt_losses": self.off_target_gt_losses,
            "per_class_changes": {str(k): dict(v) for k, v in self.per_class_changes.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "ReplacementDiagnosticsAccumulator":
        acc = cls()
        for name in (
            "target_rows", "unique_target_windows", "candidate_count", "accepted_count", "rejected_count",
            "violations_fixed", "new_violations", "targets_resolved", "changed_pixel_count",
            "target_gt_gains", "target_gt_losses", "off_target_gt_gains", "off_target_gt_losses",
        ):
            setattr(acc, name, _require_nonneg_int(state[name], name))
        if acc.new_violations != 0:
            raise BaselineCheckpointError("checkpoint reports a nonzero new_violations count; refusing to resume")
        if acc.accepted_count + acc.rejected_count != acc.candidate_count:
            raise BaselineCheckpointError("accepted_count + rejected_count must equal candidate_count")
        per_class = {}
        for key, value in dict(state["per_class_changes"]).items():
            class_id = _require_nonneg_int(int(key), "per_class_changes key")
            per_class[class_id] = {
                "gains": _require_nonneg_int(value["gains"], "per_class gains"),
                "losses": _require_nonneg_int(value["losses"], "per_class losses"),
            }
        acc.per_class_changes = per_class
        return acc


# ---------------------------------------------------------------------------
# Top-level checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineCheckpoint:
    schema_version: str
    run_status: str
    next_index: int
    processed_image_ids: tuple[str, ...]
    processed_count: int
    metric_accumulator_state: Mapping[str, dict]
    ksweep_telemetry_state: Mapping[str, dict]
    stitching_diagnostics_state: Mapping[str, dict]
    replacement_diagnostics_state: Mapping[str, dict]
    call_accounting_state: dict
    context: BaselineCheckpointCompatibilityContext

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise BaselineCheckpointError(
                f"unsupported checkpoint schema_version {self.schema_version!r}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
            )
        if self.run_status not in (RUN_STATUS_PARTIAL, RUN_STATUS_COMPLETE):
            raise BaselineCheckpointError("run_status must be 'partial' or 'complete'")
        _require_nonneg_int(self.next_index, "next_index")
        _require_nonneg_int(self.processed_count, "processed_count")
        if len(set(self.processed_image_ids)) != len(self.processed_image_ids):
            raise BaselineCheckpointError("processed_image_ids must not contain duplicates -- duplicate image detected")
        if len(self.processed_image_ids) != self.processed_count:
            raise BaselineCheckpointError("processed_image_ids length must equal processed_count")
        if self.next_index != self.processed_count:
            raise BaselineCheckpointError(
                f"next_index ({self.next_index}) != processed_count ({self.processed_count}) -- "
                "checkpoint reflects an incomplete image transaction; refusing to resume"
            )
        if not isinstance(self.context, BaselineCheckpointCompatibilityContext):
            raise BaselineCheckpointError("context must be a BaselineCheckpointCompatibilityContext")
        if self.run_status == RUN_STATUS_COMPLETE and self.processed_count != self.context.dataset_length:
            raise BaselineCheckpointError(
                "run_status is 'complete' but processed_count does not equal dataset_length"
            )
        declared = set(declared_baseline_names(self.context.k_values))
        missing_metrics = declared - set(self.metric_accumulator_state)
        if missing_metrics:
            raise BaselineCheckpointError(f"checkpoint is missing metric accumulators for: {sorted(missing_metrics)}")
        missing_ksweep = {str(k) for k in self.context.k_values} - set(self.ksweep_telemetry_state)
        if missing_ksweep:
            raise BaselineCheckpointError(f"checkpoint is missing k-sweep telemetry for k in: {sorted(missing_ksweep)}")
        missing_stitching = set(self.context.stitching_modes) - set(self.stitching_diagnostics_state)
        if missing_stitching:
            raise BaselineCheckpointError(f"checkpoint is missing stitching diagnostics for modes: {sorted(missing_stitching)}")
        missing_replacement = set(REPLACEMENT_VARIANT_NAMES) - set(self.replacement_diagnostics_state)
        if missing_replacement:
            raise BaselineCheckpointError(f"checkpoint is missing replacement diagnostics for: {sorted(missing_replacement)}")


def _checkpoint_to_json(checkpoint: BaselineCheckpoint) -> dict:
    context = checkpoint.context
    return {
        "schema_version": checkpoint.schema_version,
        "run_status": checkpoint.run_status,
        "next_index": checkpoint.next_index,
        "processed_image_ids": list(checkpoint.processed_image_ids),
        "processed_count": checkpoint.processed_count,
        "metric_accumulator_state": dict(checkpoint.metric_accumulator_state),
        "ksweep_telemetry_state": dict(checkpoint.ksweep_telemetry_state),
        "stitching_diagnostics_state": dict(checkpoint.stitching_diagnostics_state),
        "replacement_diagnostics_state": dict(checkpoint.replacement_diagnostics_state),
        "call_accounting_state": dict(checkpoint.call_accounting_state),
        "context": {
            "git_head": context.git_head, "git_branch": context.git_branch,
            "baseline_identity_name": context.baseline_identity_name,
            "baseline_manifest_sha256": context.baseline_manifest_sha256,
            "e3_identity_sha256": context.e3_identity_sha256, "rwr_identity_sha256": context.rwr_identity_sha256,
            "canonical_config_sha256": context.canonical_config_sha256, "checkpoint_sha256": context.checkpoint_sha256,
            "tracked_diff_sha256": context.tracked_diff_sha256, "untracked_source_sha256": context.untracked_source_sha256,
            "crop": list(context.crop), "stride": list(context.stride), "k_values": list(context.k_values),
            "stitching_modes": list(context.stitching_modes), "dcr_variants": list(context.dcr_variants),
            "sur_definition": context.sur_definition, "strict_safe_rule": context.strict_safe_rule,
            "class_count": context.class_count, "ignore_label": context.ignore_label,
            "metric_unit_contract": context.metric_unit_contract, "dataset_length": context.dataset_length,
        },
    }


def save_checkpoint_atomic(checkpoint: BaselineCheckpoint, path: Path) -> None:
    """Write through a temporary sibling file, flush/fsync, then
    ``os.replace()`` -- a reader can never observe a partially-written
    checkpoint, and an interrupted write leaves the previous valid
    checkpoint (if any) untouched."""
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    payload = json.dumps(_checkpoint_to_json(checkpoint), indent=2, sort_keys=True)
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, path)


def _require_closed_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineCheckpointError(f"{label} must be a JSON object")
    return value


def load_checkpoint(path: Path) -> BaselineCheckpoint:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineCheckpointError(f"cannot load baseline checkpoint {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise BaselineCheckpointError("checkpoint must be a JSON object")
    expected_keys = {
        "schema_version", "run_status", "next_index", "processed_image_ids", "processed_count",
        "metric_accumulator_state", "ksweep_telemetry_state", "stitching_diagnostics_state",
        "replacement_diagnostics_state", "call_accounting_state", "context",
    }
    if set(raw) != expected_keys:
        raise BaselineCheckpointError(
            f"checkpoint has an unexpected top-level schema; unknown={sorted(set(raw) - expected_keys)}, "
            f"missing={sorted(expected_keys - set(raw))}"
        )
    context_raw = _require_closed_mapping(raw["context"], "context")
    context = BaselineCheckpointCompatibilityContext(
        git_head=context_raw["git_head"], git_branch=context_raw["git_branch"],
        baseline_identity_name=context_raw["baseline_identity_name"],
        baseline_manifest_sha256=context_raw["baseline_manifest_sha256"],
        e3_identity_sha256=context_raw["e3_identity_sha256"], rwr_identity_sha256=context_raw["rwr_identity_sha256"],
        canonical_config_sha256=context_raw["canonical_config_sha256"], checkpoint_sha256=context_raw["checkpoint_sha256"],
        tracked_diff_sha256=context_raw["tracked_diff_sha256"], untracked_source_sha256=context_raw["untracked_source_sha256"],
        crop=tuple(context_raw["crop"]), stride=tuple(context_raw["stride"]), k_values=tuple(context_raw["k_values"]),
        stitching_modes=tuple(context_raw["stitching_modes"]), dcr_variants=tuple(context_raw["dcr_variants"]),
        sur_definition=context_raw["sur_definition"], strict_safe_rule=context_raw["strict_safe_rule"],
        class_count=context_raw["class_count"], ignore_label=context_raw["ignore_label"],
        metric_unit_contract=context_raw["metric_unit_contract"], dataset_length=context_raw["dataset_length"],
    )
    return BaselineCheckpoint(
        schema_version=raw["schema_version"], run_status=raw["run_status"], next_index=raw["next_index"],
        processed_image_ids=tuple(raw["processed_image_ids"]), processed_count=raw["processed_count"],
        metric_accumulator_state=raw["metric_accumulator_state"], ksweep_telemetry_state=raw["ksweep_telemetry_state"],
        stitching_diagnostics_state=raw["stitching_diagnostics_state"],
        replacement_diagnostics_state=raw["replacement_diagnostics_state"],
        call_accounting_state=raw["call_accounting_state"], context=context,
    )


def validate_checkpoint_compatibility(
    checkpoint: BaselineCheckpoint, expected: BaselineCheckpointCompatibilityContext
) -> None:
    """Fail closed on any provenance/configuration mismatch between the
    checkpoint and the current run's expected context."""
    observed = checkpoint.context
    mismatches = []
    for name in (
        "baseline_identity_name", "baseline_manifest_sha256", "e3_identity_sha256", "rwr_identity_sha256",
        "canonical_config_sha256", "checkpoint_sha256", "tracked_diff_sha256", "untracked_source_sha256",
        "crop", "stride", "sur_definition", "strict_safe_rule", "class_count", "ignore_label",
        "metric_unit_contract", "dataset_length",
    ):
        if getattr(observed, name) != getattr(expected, name):
            mismatches.append(name)
    if set(observed.k_values) != set(expected.k_values):
        mismatches.append("k_values")
    if set(observed.stitching_modes) != set(expected.stitching_modes):
        mismatches.append("stitching_modes")
    if set(observed.dcr_variants) != set(expected.dcr_variants):
        mismatches.append("dcr_variants")
    if mismatches:
        raise BaselineCheckpointError(
            f"checkpoint is incompatible with the current run: mismatched fields {sorted(mismatches)}"
        )


def validate_checkpoint_internal_consistency(checkpoint: BaselineCheckpoint) -> None:
    """Cross-check accumulator state without needing to reconstruct every
    accumulator: type/shape/sign checks plus cross-field agreement."""
    for baseline_name, state in checkpoint.metric_accumulator_state.items():
        if not isinstance(state, Mapping):
            raise BaselineCheckpointError(f"metric accumulator state for {baseline_name!r} must be a mapping")
        StreamingSegmentationMetricAccumulator.from_state_dict(state)
    for k_name, state in checkpoint.ksweep_telemetry_state.items():
        KSweepTelemetryAccumulator.from_state_dict(state)
    for mode_name, state in checkpoint.stitching_diagnostics_state.items():
        StitchingDiagnosticsAccumulator.from_state_dict(state)
    for variant_name, state in checkpoint.replacement_diagnostics_state.items():
        ReplacementDiagnosticsAccumulator.from_state_dict(state)
    CallAccounting.from_state_dict(checkpoint.call_accounting_state)

    call_accounting = checkpoint.call_accounting_state
    total_ksweep_graph_builds = sum(state["graph_count"] for state in checkpoint.ksweep_telemetry_state.values())
    total_ksweep_solves = sum(state["solve_count"] for state in checkpoint.ksweep_telemetry_state.values())
    if total_ksweep_graph_builds != total_ksweep_solves:
        raise BaselineCheckpointError("total k-sweep graph builds must equal total k-sweep solves")
    if call_accounting["graph_builds"] < total_ksweep_graph_builds:
        raise BaselineCheckpointError(
            "call_accounting.graph_builds is inconsistent with the sum of per-k graph_count "
            f"({call_accounting['graph_builds']} < {total_ksweep_graph_builds})"
        )
    if call_accounting["solver_calls"] < total_ksweep_solves:
        raise BaselineCheckpointError(
            "call_accounting.solver_calls is inconsistent with the sum of per-k solve_count "
            f"({call_accounting['solver_calls']} < {total_ksweep_solves})"
        )


__all__ = [
    "BaselineCheckpointError",
    "CHECKPOINT_SCHEMA_VERSION",
    "RUN_STATUS_PARTIAL",
    "RUN_STATUS_COMPLETE",
    "BASELINE_E3_UNIFORM",
    "BASELINE_E3_HANN",
    "BASELINE_RWR_K12_UNIFORM",
    "BASELINE_RWR_K12_MAJORITY",
    "BASELINE_RWR_K12_HANN",
    "BASELINE_RWR_K12_CENTER_SELECT",
    "BASELINE_DCR_HARD",
    "BASELINE_DCR_JURY_MEAN",
    "BASELINE_SUR",
    "BASELINE_STRICT_SAFE_DCR_HARD",
    "BASELINE_STRICT_SAFE_DCR_JURY_MEAN",
    "BASELINE_STRICT_SAFE_SUR",
    "REPLACEMENT_VARIANT_NAMES",
    "k_sweep_baseline_name",
    "declared_baseline_names",
    "CallAccounting",
    "KSweepTelemetryAccumulator",
    "StitchingDiagnosticsAccumulator",
    "ReplacementDiagnosticsAccumulator",
    "BaselineCheckpointCompatibilityContext",
    "BaselineCheckpoint",
    "save_checkpoint_atomic",
    "load_checkpoint",
    "validate_checkpoint_compatibility",
    "validate_checkpoint_internal_consistency",
]
