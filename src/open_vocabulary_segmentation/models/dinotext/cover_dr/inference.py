"""Opt-in canonical RWR integration at the immutable E3 patch boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.rwr_reproduction_identity import (
    SUPPORTED_CANONICAL_GRAPH_MODE,
    SUPPORTED_CANONICAL_SOLVER_METHOD,
)

from .graph import build_directed_topk_graph
from .rwr import solve_rwr_cgls


_ENABLED_KEYS = frozenset(
    {"enabled", "identity_path"}
)


class RWRInferenceConfigError(ValueError):
    """Raised when the opt-in evaluation configuration is not canonical."""


@dataclass(frozen=True)
class RWRInferenceConfig:
    enabled: bool = False
    identity_path: str | None = None
    graph_mode: str | None = None
    alpha: float | None = None
    top_k: int | None = None
    affinity_power: float | None = None
    solver: str | None = None
    solver_rtol: float | None = None
    solver_atol: float | None = None
    solver_max_iterations: int | None = None
    expected_class_count: int | None = None
    config_path: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "RWRInferenceConfig":
        if value is None:
            return cls()
        if not hasattr(value, "items"):
            raise RWRInferenceConfigError("evaluate.rwr must be a mapping")
        raw = dict(value.items())
        unknown = set(raw) - _ENABLED_KEYS
        if unknown:
            raise RWRInferenceConfigError(
                f"unknown evaluate.rwr settings: {sorted(unknown)}"
            )
        enabled = raw.get("enabled", False)
        if type(enabled) is not bool:
            raise RWRInferenceConfigError("evaluate.rwr.enabled must be boolean")
        if not enabled:
            if set(raw) - {"enabled"}:
                raise RWRInferenceConfigError(
                    "disabled RWR accepts only the enabled=false setting"
                )
            return cls(enabled=False)
        missing = _ENABLED_KEYS - set(raw)
        if missing:
            raise RWRInferenceConfigError(
                f"enabled RWR configuration is incomplete: {sorted(missing)}"
            )
        identity_path = raw["identity_path"]
        if type(identity_path) is not str or not identity_path:
            raise RWRInferenceConfigError(
                "evaluate.rwr.identity_path must be a non-empty string"
            )
        try:
            from src.rwr_reproduction_identity import (
                load_identity,
                repository_root,
                validate_static_configuration,
            )

            root = repository_root()
            validate_static_configuration(
                repo_root=root,
                identity_path=root / identity_path,
            )
            identity = load_identity(root / identity_path, repo_root=root)
        except Exception as error:
            raise RWRInferenceConfigError(
                f"canonical RWR identity preflight failed: {error}"
            ) from error
        config = cls(
            enabled=identity["rwr"]["enabled"],
            identity_path=identity_path,
            graph_mode=identity["rwr"]["graph_mode"],
            alpha=identity["rwr"]["alpha"],
            top_k=identity["rwr"]["top_k"],
            affinity_power=identity["rwr"]["affinity_power"],
            solver=identity["solver"]["method"],
            solver_rtol=identity["solver"]["rtol"],
            solver_atol=identity["solver"]["atol"],
            solver_max_iterations=identity["solver"]["max_iterations"],
            expected_class_count=identity["dataset"]["classes"],
            config_path=identity["canonical_config_path"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise RWRInferenceConfigError("enabled must be boolean")
        if not self.enabled:
            return
        if (
            type(self.graph_mode) is not str
            or self.graph_mode != SUPPORTED_CANONICAL_GRAPH_MODE
        ):
            raise RWRInferenceConfigError(
                "rwr.graph_mode capability mismatch: expected "
                f"{SUPPORTED_CANONICAL_GRAPH_MODE!r}, observed {self.graph_mode!r}"
            )
        if (
            type(self.alpha) is not float
            or not math.isfinite(self.alpha)
            or not 0 <= self.alpha < 1
        ):
            raise RWRInferenceConfigError(
                "RWR alpha must be finite and satisfy 0 <= alpha < 1"
            )
        if (
            type(self.top_k) is not int
            or self.top_k <= 0
        ):
            raise RWRInferenceConfigError("RWR top_k must be a positive integer")
        if (
            type(self.affinity_power) is not float
            or not math.isfinite(self.affinity_power)
            or self.affinity_power <= 0
        ):
            raise RWRInferenceConfigError(
                "RWR affinity_power must be finite and positive"
            )
        if (
            type(self.solver) is not str
            or self.solver != SUPPORTED_CANONICAL_SOLVER_METHOD
        ):
            raise RWRInferenceConfigError(
                "solver.method capability mismatch: expected "
                f"{SUPPORTED_CANONICAL_SOLVER_METHOD!r}, observed {self.solver!r}"
            )
        for name, value in (
            ("solver_rtol", self.solver_rtol),
            ("solver_atol", self.solver_atol),
        ):
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0
            ):
                raise RWRInferenceConfigError(
                    f"{name} must be a finite non-negative number"
                )
        if (
            type(self.solver_max_iterations) is not int
            or self.solver_max_iterations <= 0
        ):
            raise RWRInferenceConfigError(
                "solver_max_iterations must be a positive integer"
            )
        if (
            type(self.expected_class_count) is not int
            or self.expected_class_count <= 0
        ):
            raise RWRInferenceConfigError(
                "expected_class_count must be a positive integer"
            )
        if type(self.config_path) is not str or not self.config_path:
            raise RWRInferenceConfigError("canonical RWR config_path is required")
        if type(self.identity_path) is not str or not self.identity_path:
            raise RWRInferenceConfigError("canonical RWR identity_path is required")


@dataclass(frozen=True)
class RWRWindowDiagnostics:
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
class RWRInferenceOutput:
    patch_scores: torch.Tensor
    windows: tuple[RWRWindowDiagnostics, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "patch_scores",
            self.patch_scores.detach().clone().contiguous(),
        )


class RWRRuntimeSummary:
    """Bounded aggregate diagnostics; no crop tensors or predictions retained."""

    def __init__(self) -> None:
        self.window_count = 0
        self.converged_window_count = 0
        self.total_iterations = 0
        self.minimum_iterations: int | None = None
        self.maximum_iterations = 0
        self.total_restarts = 0
        self.nonzero_restart_windows = 0
        self.total_residual_replacements = 0
        self.total_fallback_rows = 0
        self.maximum_scaled_residual = 0.0

    def add(self, output: RWRInferenceOutput) -> None:
        for item in output.windows:
            self.window_count += 1
            self.converged_window_count += 1
            self.total_iterations += item.iterations
            self.minimum_iterations = (
                item.iterations
                if self.minimum_iterations is None
                else min(self.minimum_iterations, item.iterations)
            )
            self.maximum_iterations = max(self.maximum_iterations, item.iterations)
            self.total_restarts += item.restarts
            self.nonzero_restart_windows += int(item.restarts > 0)
            self.total_residual_replacements += item.residual_replacements
            self.total_fallback_rows += item.fallback_rows
            self.maximum_scaled_residual = max(
                self.maximum_scaled_residual, item.maximum_scaled_residual
            )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "window_count": self.window_count,
            "converged_window_count": self.converged_window_count,
            "total_iterations": self.total_iterations,
            "minimum_iterations": self.minimum_iterations or 0,
            "maximum_iterations": self.maximum_iterations,
            "total_restarts": self.total_restarts,
            "nonzero_restart_windows": self.nonzero_restart_windows,
            "total_residual_replacements": self.total_residual_replacements,
            "total_fallback_rows": self.total_fallback_rows,
            "maximum_scaled_residual": self.maximum_scaled_residual,
        }


@torch.no_grad()
def apply_rwr_to_e3_snapshot(
    snapshot: Any,
    config: RWRInferenceConfig,
) -> RWRInferenceOutput:
    """Diffuse owned E3 ``[B,N,C]`` patch scores, independently per crop."""
    if not isinstance(config, RWRInferenceConfig):
        raise TypeError("config must be an RWRInferenceConfig")
    config.validate()
    scores = getattr(snapshot, "unary_scores", None)
    if not torch.is_tensor(scores) or scores.ndim != 3:
        raise ValueError("E3 snapshot unary_scores must have shape [B,N,C]")
    # The disabled helper is an owned identity and has no graph-side contract.
    # The production E3 path bypasses this helper altogether (tested below).
    if not config.enabled:
        return RWRInferenceOutput(scores, ())

    features = getattr(snapshot, "dino_features", None)
    grid_hw = getattr(snapshot, "grid_hw", None)
    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError("E3 snapshot dino_features must have shape [B,N,D]")
    if scores.device != features.device:
        raise ValueError("E3 snapshot scores and features must share a device")
    if scores.shape[:2] != features.shape[:2]:
        raise ValueError(
            "E3 snapshot score/feature batch or patch count does not match"
        )
    if scores.shape[2] != config.expected_class_count:
        raise ValueError(
            "canonical RWR class-count mismatch: "
            f"expected {config.expected_class_count}, observed {scores.shape[2]}"
        )
    if (
        not isinstance(grid_hw, tuple)
        or len(grid_hw) != 2
        or any(type(item) is not int or item <= 0 for item in grid_hw)
        or math.prod(grid_hw) != scores.shape[1]
    ):
        raise ValueError("E3 snapshot grid does not match its patch count")
    if not scores.is_floating_point() or not features.is_floating_point():
        raise TypeError("E3 snapshot scores and features must be floating point")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("E3 snapshot scores must be finite")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("E3 snapshot features must be finite")

    # Alpha zero is the exact identity and deliberately avoids graph/solver work.
    if config.alpha == 0:
        return RWRInferenceOutput(scores, ())

    if config.top_k >= scores.shape[1]:
        raise ValueError("RWR top_k must be smaller than the patch count")
    outputs: list[torch.Tensor] = []
    diagnostics: list[RWRWindowDiagnostics] = []
    for batch_index in range(scores.shape[0]):
        graph = build_directed_topk_graph(
            features[batch_index],
            k=config.top_k,
            affinity_power=config.affinity_power,
        )
        result = solve_rwr_cgls(
            graph,
            scores[batch_index],
            alpha=config.alpha,
            rtol=config.solver_rtol,
            atol=config.solver_atol,
            max_iter=config.solver_max_iterations,
        )
        outputs.append(result.scores)
        diagnostics.append(
            RWRWindowDiagnostics(
                iterations=result.iterations,
                work_count=result.work_count,
                restarts=result.total_restart_count,
                residual_replacements=result.total_residual_replacement_count,
                fallback_rows=int(graph.self_loop_fallback.sum().item()),
                maximum_scaled_residual=result.maximum_scaled_residual,
                working_maximum_scaled_residual=(
                    result.working_maximum_scaled_residual
                ),
                certified_maximum_scaled_residual=(
                    result.certified_maximum_scaled_residual
                ),
                certificate_dtype=result.certificate_dtype,
                fp64_certificate_checks=result.fp64_certificate_checks,
                fp64_certified_rhs=result.fp64_certified_rhs,
                fp64_certificate_rejections=result.fp64_certificate_rejections,
                fp64_certificate_restart_count=(
                    result.fp64_certificate_restart_count
                ),
                fp64_certificate_work=result.fp64_certificate_work,
            )
        )
    return RWRInferenceOutput(torch.stack(outputs), tuple(diagnostics))


@torch.no_grad()
def patch_scores_to_masks(
    patch_scores: torch.Tensor,
    grid_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> torch.Tensor:
    """Apply E3's sole sigmoid and bilinear crop upsampling after RWR."""
    if not torch.is_tensor(patch_scores) or patch_scores.ndim != 3:
        raise ValueError("patch_scores must have shape [B,N,C]")
    if math.prod(grid_hw) != patch_scores.shape[1]:
        raise ValueError("patch score count does not match grid_hw")
    if len(output_hw) != 2 or any(type(item) is not int or item <= 0 for item in output_hw):
        raise ValueError("output_hw must contain two positive integers")
    if not bool(torch.isfinite(patch_scores).all()):
        raise ValueError("patch_scores must be finite")
    batch, _patches, classes = patch_scores.shape
    logits = patch_scores.reshape(batch, *grid_hw, classes).permute(0, 3, 1, 2)
    masks = torch.sigmoid(logits)
    return F.interpolate(masks, output_hw, mode="bilinear", align_corners=True)


def build_rwr_structured_record(
    *,
    config: RWRInferenceConfig,
    canonical_identity: Any,
    metrics: Any,
    image_count: int,
    class_count: int,
    crop: tuple[int, int],
    stride: tuple[int, int],
    pamr: bool,
    background_class: bool,
    checkpoint_path: str,
    source_git_commit: str,
    source_git_branch: str,
    source_git_dirty: bool,
    gpu_model: str,
    torch_version: str,
    cuda_version: str | None,
    elapsed_seconds: float,
    solver_summary: dict[str, int | float],
) -> dict[str, Any]:
    """Build the single closed, percentage-scale canonical result record."""
    config.validate()
    if not hasattr(metrics, "items"):
        raise ValueError("evaluation metrics must be a mapping")

    def exact_int(value: Any, name: str, *, positive: bool = False) -> int:
        if type(value) is not int or (positive and value <= 0) or value < 0:
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be an exact {qualifier} integer")
        return value

    def exact_float(value: Any, name: str, *, non_negative: bool = False) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{name} must be an exact finite float")
        if non_negative and value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def exact_bool(value: Any, name: str) -> bool:
        if type(value) is not bool:
            raise ValueError(f"{name} must be an exact boolean")
        return value

    def exact_string(value: Any, name: str) -> str:
        if type(value) is not str or not value:
            raise ValueError(f"{name} must be an exact non-empty string")
        return value

    def exact_pair(value: Any, name: str) -> list[int]:
        if (
            type(value) is not tuple
            or len(value) != 2
            or any(type(item) is not int or item <= 0 for item in value)
        ):
            raise ValueError(f"{name} must be an exact tuple of two positive integers")
        return [value[0], value[1]]

    def percentage(name: str) -> float:
        if name not in metrics:
            raise ValueError(f"evaluation result is missing {name}")
        value = exact_float(metrics[name], f"evaluation metric {name}")
        if not 0 <= value <= 1:
            raise ValueError(
                f"raw evaluator metric {name} must be a fraction in [0,1]"
            )
        return 100.0 * value

    observed_miou = percentage("mIoU")
    expected = canonical_identity
    if not hasattr(expected, "items"):
        raise ValueError("canonical_identity must be a mapping")
    exact_int(image_count, "image_count", positive=True)
    exact_int(class_count, "class_count", positive=True)
    exact_bool(pamr, "pamr")
    exact_bool(background_class, "background_class")
    exact_bool(source_git_dirty, "source_git_dirty")
    exact_string(checkpoint_path, "checkpoint_path")
    exact_string(source_git_commit, "source_git_commit")
    exact_string(source_git_branch, "source_git_branch")
    exact_string(gpu_model, "gpu_model")
    exact_string(torch_version, "torch_version")
    exact_string(cuda_version, "cuda_version")
    exact_float(elapsed_seconds, "elapsed_seconds", non_negative=True)
    if type(solver_summary) is not dict:
        raise ValueError("solver_summary must be an exact dictionary")
    integer_summary = {
        "window_count", "converged_window_count", "total_iterations",
        "minimum_iterations", "maximum_iterations", "total_restarts",
        "nonzero_restart_windows", "total_residual_replacements",
        "total_fallback_rows",
    }
    expected_summary = integer_summary | {"maximum_scaled_residual"}
    if set(solver_summary) != expected_summary:
        raise ValueError("solver_summary has an unexpected schema")
    for name in sorted(integer_summary):
        exact_int(solver_summary[name], f"solver_summary.{name}")
    exact_float(
        solver_summary["maximum_scaled_residual"],
        "solver_summary.maximum_scaled_residual",
        non_negative=True,
    )
    record = {
        "format_version": "talk2dino-canonical-rwr-result-v2",
        "identity_name": expected["identity_name"],
        "image_count": image_count,
        "class_count": class_count,
        "aAcc": percentage("aAcc"),
        "mIoU": observed_miou,
        "mAcc": percentage("mAcc"),
        "metrics_precision": "full",
        "gain_over_e3_miou": observed_miou - expected["expected_metrics"]["e3_mIoU"],
        "rwr_enabled": config.enabled,
        "alpha": config.alpha,
        "top_k": config.top_k,
        "affinity_power": config.affinity_power,
        "graph_mode": config.graph_mode,
        "solver": config.solver,
        "solver_rtol": config.solver_rtol,
        "solver_atol": config.solver_atol,
        "solver_max_iterations": config.solver_max_iterations,
        "crop": exact_pair(crop, "crop"),
        "stride": exact_pair(stride, "stride"),
        "pamr": pamr,
        "background_class": background_class,
        "config_path": config.config_path,
        "config_sha256": expected["canonical_config_sha256"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": expected["checkpoint"]["sha256"],
        "source_e10_commit": expected["source_e10_commit"],
        "cache_source_commit": expected["cache_source_commit"],
        "cache_manifest_sha256": expected["cache_manifest_sha256"],
        "cache_manifest_evidence": expected["cache_manifest_evidence"],
        "cache_manifest_attestation_commit": expected[
            "cache_manifest_attestation_commit"
        ],
        "cache_manifest_attestation_path": expected[
            "cache_manifest_attestation_path"
        ],
        "cache_manifest_attestation_blob_sha256": expected[
            "cache_manifest_attestation_blob_sha256"
        ],
        "cache_manifest_archived": expected["cache_manifest_archived"],
        "historical_cache_used_by_current_run": expected[
            "historical_cache_used_by_current_run"
        ],
        "source_git_commit": source_git_commit,
        "source_git_branch": source_git_branch,
        "source_git_dirty": source_git_dirty,
        "gpu_model": gpu_model,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "elapsed_seconds": elapsed_seconds,
        "solver_summary": {name: solver_summary[name] for name in sorted(solver_summary)},
    }
    return record


__all__ = [
    "RWRInferenceConfig",
    "RWRInferenceConfigError",
    "RWRInferenceOutput",
    "RWRRuntimeSummary",
    "RWRWindowDiagnostics",
    "apply_rwr_to_e3_snapshot",
    "build_rwr_structured_record",
    "patch_scores_to_masks",
]
