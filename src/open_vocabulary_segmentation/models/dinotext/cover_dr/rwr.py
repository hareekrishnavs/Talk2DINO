"""Verified sparse solvers for directed random walk with restart.

``K = I - alpha*A`` is nonsymmetric because ``A`` is directed. Ordinary
conjugate gradient must not be applied directly to ``K``. CGLS is valid
because it uses ``K`` and ``K.T`` implicitly, although the normal-equation
form can square the effective condition number. The fixed-point method is the
contraction reference for a row-stochastic graph and ``0 <= alpha < 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import torch

from .graph import DirectedTopKGraph


_CGLS_PERIODIC_RESTART_STEPS = 128
_CGLS_STAGNATION_STEPS = 512
_CGLS_STAGNATION_RATIO = 2.0
_CGLS_PROGRESS_EPS_MULTIPLIER = 64.0
_CGLS_MAX_RECOVERY_RESTARTS = 3


class RWRSolverError(RuntimeError):
    """Base exception for RWR solve failures."""


class RWRInputError(ValueError, RWRSolverError):
    """Raised when an RWR input violates the public contract."""


class RWRNonConvergenceError(RWRSolverError):
    """Raised instead of returning an unconverged approximation."""


class RWRNumericalBreakdownError(RWRSolverError):
    """Raised when CGLS cannot make a numerically valid update."""


class RWRNonFiniteError(RWRNumericalBreakdownError):
    """Raised when non-finite solver arithmetic is detected."""


def _validate_alpha(alpha: float) -> float:
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0 <= float(alpha) < 1
    ):
        raise RWRInputError(
            f"alpha must be finite and satisfy 0 <= alpha < 1, got {alpha!r}"
        )
    return float(alpha)


def _validate_tolerance(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise RWRInputError(f"{name} must be finite and non-negative")
    return float(value)


def _validate_max_iter(max_iter: int) -> int:
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise RWRInputError("max_iter must be a positive integer")
    return max_iter


def _solve_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.float64
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    raise RWRInputError(
        "unary_scores must use float16, bfloat16, float32, or float64"
    )


def _default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return numerical, dtype-derived residual tolerances.

    Float32 uses tolerances comfortably above accumulated sparse-reduction
    rounding. Float64 uses stricter reference tolerances. These are solver
    accuracy settings and are independent of evaluation metrics.
    """
    if dtype == torch.float64:
        return 1e-10, 1e-12
    return 1e-5, 1e-7


def _require_finite(value: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RWRNonFiniteError(f"non-finite arithmetic in {context}")


def _cgls_reduction_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use device-local FP64 accumulation only for CGLS scalar reductions."""
    return torch.float64 if dtype == torch.float32 else dtype


def _cgls_squared_column_norm(value: torch.Tensor) -> torch.Tensor:
    return (value * value).sum(
        dim=0, dtype=_cgls_reduction_dtype(value.dtype)
    )


def _cgls_column_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(
        dim=0, dtype=_cgls_reduction_dtype(left.dtype)
    )


def _scaled_residual_quantities(
    residual: torch.Tensor,
    right_hand_side: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_scale = residual.abs().amax(dim=0)
    rhs_scale = right_hand_side.abs().amax(dim=0)
    safe_residual_scale = torch.where(
        residual_scale > 0, residual_scale, torch.ones_like(residual_scale)
    )
    safe_rhs_scale = torch.where(
        rhs_scale > 0, rhs_scale, torch.ones_like(rhs_scale)
    )
    residual_norm = torch.where(
        residual_scale > 0,
        residual_scale
        * torch.linalg.vector_norm(residual / safe_residual_scale, dim=0),
        torch.zeros_like(residual_scale),
    )
    rhs_norm = torch.where(
        rhs_scale > 0,
        rhs_scale
        * torch.linalg.vector_norm(
            right_hand_side / safe_rhs_scale, dim=0
        ),
        torch.zeros_like(rhs_scale),
    )
    threshold = atol * math.sqrt(residual.shape[0]) + rtol * rhs_norm
    scaled = torch.where(
        threshold > 0,
        residual_norm / threshold,
        torch.where(
            residual_norm == 0,
            torch.zeros_like(residual_norm),
            torch.full_like(residual_norm, torch.inf),
        ),
    )
    return residual_norm, threshold, scaled


def _raise_solver_failure(
    exception_type: type[RWRSolverError],
    *,
    method: str,
    iteration: int,
    rtol: float,
    atol: float,
    reason: str,
    operator: SparseRWROperator | None = None,
    solution: torch.Tensor | None = None,
    right_hand_side: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    failing_mask: torch.Tensor | None = None,
    stage: str | None = None,
    tensor: str | None = None,
    detail: str | None = None,
    breakdown_value: float | None = None,
    restart_counts: torch.Tensor | None = None,
    residual_replacement_counts: torch.Tensor | None = None,
    restart_reason_counts: dict[str, int] | None = None,
    latest_restart_reason: str | None = None,
    work_count: int | None = None,
    certified_residual: torch.Tensor | None = None,
    certified_scaled_residual: torch.Tensor | None = None,
    certificate_dtype: str | None = None,
    fp64_certificate_checks: int = 0,
    fp64_certified_rhs: int = 0,
    fp64_certificate_rejections: int = 0,
    fp64_certificate_restart_count: int = 0,
    fp64_certificate_work: int = 0,
) -> NoReturn:
    """Raise one complete diagnostic for an iterative solver failure.

    ``iteration=0`` means no update completed; otherwise ``iteration=t``
    means exactly ``t`` valid solver updates completed.
    """
    working_primal_residual = None
    current_scores = None
    working_absolute_residual_inf = math.inf
    working_maximum_scaled_residual = math.inf
    if solution is not None:
        current_scores = solution.detach().clone().contiguous()
    if operator is not None and solution is not None and right_hand_side is not None:
        working_primal_residual = (
            right_hand_side - operator._matmul_prepared(solution)
        ).detach()
        if bool(torch.isfinite(working_primal_residual).all()):
            diagnostic_residual = working_primal_residual
            diagnostic_rhs = right_hand_side
            if active_mask is not None and bool(torch.any(active_mask)):
                diagnostic_residual = diagnostic_residual[:, active_mask]
                diagnostic_rhs = diagnostic_rhs[:, active_mask]
            working_absolute_residual_inf = float(
                diagnostic_residual.abs().max().item()
            )
            _norm, _threshold, scaled = _scaled_residual_quantities(
                diagnostic_residual,
                diagnostic_rhs,
                rtol=rtol,
                atol=atol,
            )
            if bool(torch.isfinite(scaled).all()):
                working_maximum_scaled_residual = float(scaled.max().item())

    certified_absolute_residual_inf = math.inf
    certified_maximum_scaled_residual = math.inf
    if certified_residual is not None:
        diagnostic_certified_residual = certified_residual
        diagnostic_certified_scaled = certified_scaled_residual
        if active_mask is not None and bool(torch.any(active_mask)):
            diagnostic_certified_residual = diagnostic_certified_residual[
                :, active_mask
            ]
            if diagnostic_certified_scaled is not None:
                diagnostic_certified_scaled = diagnostic_certified_scaled[active_mask]
        if bool(torch.isfinite(diagnostic_certified_residual).all()):
            certified_absolute_residual_inf = float(
                diagnostic_certified_residual.abs().max().item()
            )
        if (
            diagnostic_certified_scaled is not None
            and bool(torch.isfinite(diagnostic_certified_scaled).all())
        ):
            certified_maximum_scaled_residual = float(
                diagnostic_certified_scaled.max().item()
            )
    absolute_residual_inf = (
        certified_absolute_residual_inf
        if certified_residual is not None
        else working_absolute_residual_inf
    )
    maximum_scaled_residual = (
        certified_maximum_scaled_residual
        if certified_residual is not None
        else working_maximum_scaled_residual
    )

    failing_rhs: tuple[int, ...] = ()
    if failing_mask is not None:
        failing_rhs = tuple(
            int(value) for value in failing_mask.nonzero().flatten().tolist()
        )
    active_rhs: tuple[int, ...] = ()
    if active_mask is not None:
        active_rhs = tuple(
            int(value) for value in active_mask.nonzero().flatten().tolist()
        )
    restarts_per_rhs: tuple[int, ...] = ()
    if restart_counts is not None:
        restarts_per_rhs = tuple(int(value) for value in restart_counts.tolist())
    residual_replacements_per_rhs: tuple[int, ...] = ()
    if residual_replacement_counts is not None:
        residual_replacements_per_rhs = tuple(
            int(value) for value in residual_replacement_counts.tolist()
        )
    total_restart_count = sum(restarts_per_rhs)
    max_restarts_per_rhs = max(restarts_per_rhs, default=0)
    total_residual_replacement_count = sum(residual_replacements_per_rhs)
    ordered_restart_reasons = tuple(sorted((restart_reason_counts or {}).items()))
    fields = [
        "RWR solver failure:",
        f"method={method}",
        f"iteration={iteration}",
        f"abs_primal_residual_inf={absolute_residual_inf:.17g}",
        f"max_scaled_primal_residual={maximum_scaled_residual:.17g}",
        f"working_abs_primal_residual_inf={working_absolute_residual_inf:.17g}",
        f"working_max_scaled_primal_residual={working_maximum_scaled_residual:.17g}",
        f"certified_abs_primal_residual_inf={certified_absolute_residual_inf:.17g}",
        f"certified_max_scaled_primal_residual={certified_maximum_scaled_residual:.17g}",
        f"rtol={rtol:.17g}",
        f"atol={atol:.17g}",
        f"reason={reason}",
    ]
    for name, value in (
        ("failing_rhs", list(failing_rhs) if failing_rhs else None),
        ("active_rhs", list(active_rhs) if active_rhs else None),
        ("stage", stage),
        ("tensor", tensor),
        ("breakdown_value", breakdown_value),
        ("total_restart_count", total_restart_count),
        ("max_restarts_per_rhs", max_restarts_per_rhs),
        ("total_residual_replacement_count", total_residual_replacement_count),
        ("restart_counts", list(restarts_per_rhs) if restarts_per_rhs else None),
        ("restart_reasons", dict(ordered_restart_reasons) if ordered_restart_reasons else None),
        ("latest_restart_reason", latest_restart_reason),
        ("work_count", work_count),
        ("certificate_dtype", certificate_dtype),
        ("fp64_certificate_checks", fp64_certificate_checks),
        ("fp64_certified_rhs", fp64_certified_rhs),
        ("fp64_certificate_rejections", fp64_certificate_rejections),
        ("fp64_certificate_restart_count", fp64_certificate_restart_count),
        ("fp64_certificate_work", fp64_certificate_work),
        ("detail", detail),
    ):
        if value is not None:
            fields.append(f"{name}={value}")
    error = exception_type(" ".join(fields))
    error.method = method
    error.iteration = iteration
    error.abs_primal_residual_inf = absolute_residual_inf
    error.max_scaled_primal_residual = maximum_scaled_residual
    error.working_abs_primal_residual_inf = working_absolute_residual_inf
    error.working_max_scaled_primal_residual = working_maximum_scaled_residual
    error.certified_abs_primal_residual_inf = certified_absolute_residual_inf
    error.certified_max_scaled_primal_residual = certified_maximum_scaled_residual
    error.rtol = rtol
    error.atol = atol
    error.reason = reason
    error.failing_rhs = failing_rhs
    error.active_rhs = active_rhs
    error.stage = stage
    error.tensor = tensor
    error.detail = detail
    error.breakdown_value = breakdown_value
    error.total_restart_count = total_restart_count
    error.max_restarts_per_rhs = max_restarts_per_rhs
    error.total_residual_replacement_count = total_residual_replacement_count
    error.restarts_per_rhs = restarts_per_rhs
    error.residual_replacements_per_rhs = residual_replacements_per_rhs
    error.restart_reason_counts = ordered_restart_reasons
    error.latest_restart_reason = latest_restart_reason
    error.work_count = work_count
    error.certificate_dtype = certificate_dtype
    error.fp64_certificate_checks = fp64_certificate_checks
    error.fp64_certified_rhs = fp64_certified_rhs
    error.fp64_certificate_rejections = fp64_certificate_rejections
    error.fp64_certificate_restart_count = fp64_certificate_restart_count
    error.fp64_certificate_work = fp64_certificate_work
    error.current_scores = current_scores
    error.primal_residual = (
        None
        if certified_residual is None and working_primal_residual is None
        else (
            certified_residual
            if certified_residual is not None
            else working_primal_residual
        ).clone().contiguous()
    )
    error.working_primal_residual = (
        None
        if working_primal_residual is None
        else working_primal_residual.clone().contiguous()
    )
    error.certified_primal_residual = (
        None
        if certified_residual is None
        else certified_residual.clone().contiguous()
    )
    raise error


def _validate_operator_rhs(
    graph: DirectedTopKGraph,
    rhs: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(rhs, torch.Tensor):
        raise RWRInputError("operator RHS must be a torch.Tensor")
    if rhs.ndim not in (1, 2):
        raise RWRInputError(
            f"operator RHS must have shape [N] or [N, R], got {tuple(rhs.shape)}"
        )
    if rhs.shape[0] != graph.num_nodes:
        raise RWRInputError(
            "operator node mismatch: "
            f"graph={graph.num_nodes}, rhs={rhs.shape[0]}"
        )
    if rhs.numel() == 0:
        raise RWRInputError("operator RHS must be nonempty")
    if rhs.device != graph.transition_weights.device:
        raise RWRInputError("operator RHS and graph must be on the same device")
    if not rhs.is_floating_point():
        raise RWRInputError("operator RHS must be floating point")
    solve_dtype = _solve_dtype(rhs.dtype)
    value = rhs.detach().to(dtype=solve_dtype)
    _require_finite(value, "operator input")
    return value


@dataclass(frozen=True)
class SparseRWROperator:
    """Implicit nonsymmetric ``K = I - alpha*A`` sparse operator."""

    graph: DirectedTopKGraph
    alpha: float = 0.98

    def __post_init__(self) -> None:
        if not isinstance(self.graph, DirectedTopKGraph):
            raise RWRInputError("graph must be a DirectedTopKGraph")
        self.graph.validate()
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))

    def _matmul_prepared(self, value: torch.Tensor) -> torch.Tensor:
        return value - self.alpha * self.graph.matmul(value)

    def _transpose_matmul_prepared(self, value: torch.Tensor) -> torch.Tensor:
        return value - self.alpha * self.graph.transpose_matmul(value)

    @torch.no_grad()
    def matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        """Apply ``K @ rhs = rhs - alpha*(A @ rhs)``."""
        value = _validate_operator_rhs(self.graph, rhs)
        result = self._matmul_prepared(value)
        _require_finite(result, "forward operator")
        return result.detach()

    @torch.no_grad()
    def transpose_matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        """Apply ``K.T @ rhs`` without symmetrizing ``A`` or ``K``."""
        value = _validate_operator_rhs(self.graph, rhs)
        result = self._transpose_matmul_prepared(value)
        _require_finite(result, "transpose operator")
        return result.detach()


@dataclass(frozen=True)
class _FP64ResidualCertificate:
    """Device-local FP64 residuals for selected columns of the stored system."""

    columns: torch.Tensor
    residual: torch.Tensor
    residual_norm: torch.Tensor
    threshold: torch.Tensor
    scaled_residual: torch.Tensor


@torch.no_grad()
def _sparse_fp64_residual_certificate(
    graph: DirectedTopKGraph,
    alpha: float,
    solution: torch.Tensor,
    right_hand_side: torch.Tensor,
    *,
    columns: torch.Tensor | None = None,
    rtol: float,
    atol: float,
) -> _FP64ResidualCertificate:
    """Certify selected columns of the exact quantized production system.

    The graph indices are used unchanged. Stored graph weights, the already
    constructed RHS, and the candidate FP32 iterate are cast to FP64 on their
    current device. No dense graph, transpose application, CPU transfer, or
    autograd graph is created.
    """
    if solution.ndim != 2 or right_hand_side.ndim != 2:
        raise RWRInputError("certificate operands must have shape [N, R]")
    if solution.shape != right_hand_side.shape:
        raise RWRInputError("certificate solution and RHS shapes must match")
    if solution.shape[0] != graph.num_nodes:
        raise RWRInputError("certificate operands and graph node count must match")
    if solution.device != graph.transition_weights.device:
        raise RWRInputError("certificate operands and graph must share a device")
    if right_hand_side.device != solution.device:
        raise RWRInputError("certificate solution and RHS must share a device")
    if columns is None:
        selected_columns = torch.arange(
            solution.shape[1], dtype=torch.long, device=solution.device
        )
    else:
        if (
            not isinstance(columns, torch.Tensor)
            or columns.ndim != 1
            or columns.dtype != torch.long
            or columns.device != solution.device
        ):
            raise RWRInputError(
                "certificate columns must be a device-local int64 vector"
            )
        if columns.numel() == 0:
            raise RWRInputError("certificate columns must be nonempty")
        if bool(torch.any(columns < 0)) or bool(
            torch.any(columns >= solution.shape[1])
        ):
            raise RWRInputError("certificate column index is out of range")
        if torch.unique(columns).numel() != columns.numel():
            raise RWRInputError("certificate columns must be unique")
        selected_columns = columns.detach().clone()

    candidate64 = solution.index_select(1, selected_columns).to(torch.float64)
    rhs64 = right_hand_side.index_select(1, selected_columns).to(torch.float64)
    indices = graph.neighbor_indices
    weights64 = graph.transition_weights.to(torch.float64)
    gathered = candidate64[indices]
    spread = (weights64.unsqueeze(-1) * gathered).sum(dim=1)
    residual = rhs64 - (candidate64 - float(alpha) * spread)
    _require_finite(residual, "FP64 residual certificate")
    residual_norm, threshold, scaled = _scaled_residual_quantities(
        residual,
        rhs64,
        rtol=rtol,
        atol=atol,
    )
    _require_finite(residual_norm, "FP64 certificate residual norm")
    _require_finite(threshold, "FP64 certificate threshold")
    return _FP64ResidualCertificate(
        columns=selected_columns,
        residual=residual,
        residual_norm=residual_norm,
        threshold=threshold,
        scaled_residual=scaled,
    )


@dataclass(frozen=True)
class RWRSolveResult:
    """Owned scores and finite diagnostics from a successful RWR solve."""

    scores: torch.Tensor
    method: str
    alpha: float
    converged: bool
    iterations: int
    absolute_residual_inf: float
    maximum_scaled_residual: float
    fixed_point_delta_inf: float
    work_count: int = 0
    total_restart_count: int = 0
    max_restarts_per_rhs: int = 0
    total_residual_replacement_count: int = 0
    restarts_per_rhs: tuple[int, ...] = ()
    residual_replacements_per_rhs: tuple[int, ...] = ()
    restart_reason_counts: tuple[tuple[str, int], ...] = ()
    working_absolute_residual_inf: float = 0.0
    working_maximum_scaled_residual: float = 0.0
    certified_absolute_residual_inf: float = 0.0
    certified_maximum_scaled_residual: float = 0.0
    certificate_dtype: str = ""
    fp64_certificate_checks: int = 0
    fp64_certified_rhs: int = 0
    fp64_certificate_rejections: int = 0
    fp64_certificate_restart_count: int = 0
    fp64_certificate_work: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.scores, torch.Tensor):
            raise TypeError("scores must be a torch.Tensor")
        if not self.scores.is_floating_point() or self.scores.numel() == 0:
            raise ValueError("scores must be a nonempty floating-point tensor")
        if not bool(torch.isfinite(self.scores).all()):
            raise ValueError("scores must be finite")
        object.__setattr__(
            self,
            "scores",
            self.scores.detach().clone().contiguous(),
        )
        if self.method not in ("cgls", "fixed_point"):
            raise ValueError(f"unknown RWR result method: {self.method!r}")
        if not isinstance(self.converged, bool) or not self.converged:
            raise ValueError("public RWR results must be converged")
        _validate_alpha(self.alpha)
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 0
        ):
            raise ValueError("iterations must be a non-negative integer")
        if (
            isinstance(self.work_count, bool)
            or not isinstance(self.work_count, int)
            or self.work_count < self.iterations
        ):
            raise ValueError(
                "work_count must be an integer greater than or equal to iterations"
            )
        for name in (
            "absolute_residual_inf",
            "maximum_scaled_residual",
            "fixed_point_delta_inf",
            "working_absolute_residual_inf",
            "working_maximum_scaled_residual",
            "certified_absolute_residual_inf",
            "certified_maximum_scaled_residual",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "total_restart_count",
            "max_restarts_per_rhs",
            "total_residual_replacement_count",
            "fp64_certificate_checks",
            "fp64_certified_rhs",
            "fp64_certificate_rejections",
            "fp64_certificate_restart_count",
            "fp64_certificate_work",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("restarts_per_rhs", "residual_replacements_per_rhs"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in value
            ):
                raise ValueError(f"{name} must contain non-negative integers")
        if self.total_restart_count != sum(self.restarts_per_rhs):
            raise ValueError("total_restart_count does not match restarts_per_rhs")
        if self.max_restarts_per_rhs != max(self.restarts_per_rhs, default=0):
            raise ValueError("max_restarts_per_rhs does not match restarts_per_rhs")
        if self.total_residual_replacement_count != sum(
            self.residual_replacements_per_rhs
        ):
            raise ValueError(
                "total_residual_replacement_count does not match "
                "residual_replacements_per_rhs"
            )
        if not isinstance(self.restart_reason_counts, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 0
            for item in self.restart_reason_counts
        ):
            raise ValueError("restart_reason_counts has invalid entries")
        if tuple(sorted(self.restart_reason_counts)) != self.restart_reason_counts:
            raise ValueError("restart_reason_counts must be sorted")
        if not isinstance(self.certificate_dtype, str) or not self.certificate_dtype:
            raise ValueError("certificate_dtype must be a nonempty string")
        if self.fp64_certificate_work != self.fp64_certificate_checks:
            raise ValueError(
                "fp64_certificate_work must match sparse certificate checks"
            )
        rhs_count = self.scores.shape[1] if self.scores.ndim == 2 else 1
        if self.fp64_certified_rhs > rhs_count:
            raise ValueError("fp64_certified_rhs exceeds the RHS count")


@dataclass(frozen=True)
class _PreparedProblem:
    graph: DirectedTopKGraph
    scores: torch.Tensor
    scores_matrix: torch.Tensor
    was_vector: bool
    alpha: float
    rtol: float
    atol: float
    max_iter: int


def _prepare_problem(
    graph: DirectedTopKGraph,
    unary_scores: torch.Tensor,
    method: str,
    alpha: float,
    rtol: float | None,
    atol: float | None,
    max_iter: int,
) -> _PreparedProblem:
    if not isinstance(graph, DirectedTopKGraph):
        raise RWRInputError("graph must be a DirectedTopKGraph")
    graph.validate()
    alpha = _validate_alpha(alpha)
    rtol = _validate_tolerance("rtol", rtol)
    atol = _validate_tolerance("atol", atol)
    max_iter = _validate_max_iter(max_iter)
    if not isinstance(unary_scores, torch.Tensor):
        raise RWRInputError("unary_scores must be a torch.Tensor")
    if unary_scores.ndim not in (1, 2):
        raise RWRInputError(
            "unary_scores must have shape [N] or [N, C], got "
            f"{tuple(unary_scores.shape)}"
        )
    if unary_scores.shape[0] != graph.num_nodes:
        raise RWRInputError(
            "unary score node mismatch: "
            f"graph={graph.num_nodes}, scores={unary_scores.shape[0]}"
        )
    if unary_scores.numel() == 0:
        raise RWRInputError("unary_scores must be nonempty")
    if unary_scores.device != graph.transition_weights.device:
        raise RWRInputError("unary_scores and graph must be on the same device")
    if not unary_scores.is_floating_point():
        raise RWRInputError("unary_scores must be floating point")
    dtype = _solve_dtype(unary_scores.dtype)
    scores = unary_scores.detach().to(dtype=dtype).clone().contiguous()
    defaults = _default_tolerances(dtype)
    rtol = defaults[0] if rtol is None else rtol
    atol = defaults[1] if atol is None else atol
    if not bool(torch.isfinite(scores).all()):
        _raise_solver_failure(
            RWRNonFiniteError,
            method=method,
            iteration=0,
            rtol=rtol,
            atol=atol,
            reason="non_finite_input",
            tensor="unary_scores",
            detail="non-finite unary_scores input",
        )
    was_vector = scores.ndim == 1
    scores_matrix = scores[:, None] if was_vector else scores
    return _PreparedProblem(
        graph=graph,
        scores=scores,
        scores_matrix=scores_matrix,
        was_vector=was_vector,
        alpha=alpha,
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
    )


def _residual_state(
    operator: SparseRWROperator,
    solution: torch.Tensor,
    right_hand_side: torch.Tensor,
    *,
    method: str,
    iteration: int,
    active_mask: torch.Tensor | None = None,
    rtol: float,
    atol: float,
    restart_counts: torch.Tensor | None = None,
    residual_replacement_counts: torch.Tensor | None = None,
    restart_reason_counts: dict[str, int] | None = None,
    latest_restart_reason: str | None = None,
    certificate_dtype: str | None = None,
    fp64_certificate_checks: int = 0,
    fp64_certified_rhs: int = 0,
    fp64_certificate_rejections: int = 0,
    fp64_certificate_restart_count: int = 0,
    fp64_certificate_work: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = operator._matmul_prepared(solution) - right_hand_side
    residual_norm, threshold, scaled = _scaled_residual_quantities(
        residual,
        right_hand_side,
        rtol=rtol,
        atol=atol,
    )
    finite_residual_rhs = torch.isfinite(residual).all(dim=0) & torch.isfinite(
        residual_norm
    )
    failing = ~finite_residual_rhs
    if active_mask is not None:
        failing = failing & active_mask
    if not bool(torch.all(finite_residual_rhs)):
        _raise_solver_failure(
            RWRNonFiniteError,
            method=method,
            iteration=iteration,
            rtol=rtol,
            atol=atol,
            reason="non_finite_primal_residual",
            operator=operator,
            solution=solution,
            right_hand_side=right_hand_side,
            active_mask=active_mask,
            failing_mask=failing,
            tensor="primal_residual",
            detail="true primal residual B-KP is non-finite",
            restart_counts=restart_counts,
            residual_replacement_counts=residual_replacement_counts,
            restart_reason_counts=restart_reason_counts,
            latest_restart_reason=latest_restart_reason,
            certificate_dtype=certificate_dtype,
            fp64_certificate_checks=fp64_certificate_checks,
            fp64_certified_rhs=fp64_certified_rhs,
            fp64_certificate_rejections=fp64_certificate_rejections,
            fp64_certificate_restart_count=fp64_certificate_restart_count,
            fp64_certificate_work=fp64_certificate_work,
        )
    failing = ~torch.isfinite(threshold)
    if active_mask is not None:
        failing = failing & active_mask
    if not bool(torch.isfinite(threshold).all()):
        _raise_solver_failure(
            RWRNonFiniteError,
            method=method,
            iteration=iteration,
            rtol=rtol,
            atol=atol,
            reason="non_finite_residual_threshold",
            operator=operator,
            solution=solution,
            right_hand_side=right_hand_side,
            active_mask=active_mask,
            failing_mask=failing,
            tensor="residual_threshold",
            detail="scaled primal-residual threshold is non-finite",
            restart_counts=restart_counts,
            residual_replacement_counts=residual_replacement_counts,
            restart_reason_counts=restart_reason_counts,
            latest_restart_reason=latest_restart_reason,
            certificate_dtype=certificate_dtype,
            fp64_certificate_checks=fp64_certificate_checks,
            fp64_certified_rhs=fp64_certified_rhs,
            fp64_certificate_rejections=fp64_certificate_rejections,
            fp64_certificate_restart_count=fp64_certificate_restart_count,
            fp64_certificate_work=fp64_certificate_work,
        )
    converged = residual_norm <= threshold
    return residual, converged, scaled


def _make_result(
    problem: _PreparedProblem,
    operator: SparseRWROperator,
    solution: torch.Tensor,
    *,
    method: str,
    iterations: int,
    work_count: int | None = None,
    fixed_point_delta_inf: float | None = None,
    restart_counts: torch.Tensor | None = None,
    residual_replacement_counts: torch.Tensor | None = None,
    restart_reason_counts: dict[str, int] | None = None,
    latest_restart_reason: str | None = None,
    certified_residual: torch.Tensor | None = None,
    certified_scaled_residual: torch.Tensor | None = None,
    certificate_dtype: str | None = None,
    fp64_certificate_checks: int = 0,
    fp64_certified_rhs: int = 0,
    fp64_certificate_rejections: int = 0,
    fp64_certificate_restart_count: int = 0,
    fp64_certificate_work: int = 0,
) -> RWRSolveResult:
    if work_count is None:
        work_count = iterations
    right_hand_side = (1 - problem.alpha) * problem.scores_matrix
    residual, converged, scaled = _residual_state(
        operator,
        solution,
        right_hand_side,
        method=method,
        iteration=iterations,
        rtol=problem.rtol,
        atol=problem.atol,
        restart_counts=restart_counts,
        residual_replacement_counts=residual_replacement_counts,
        restart_reason_counts=restart_reason_counts,
        latest_restart_reason=latest_restart_reason,
    )
    if not bool(torch.all(converged)):
        _raise_solver_failure(
            RWRNonConvergenceError,
            method=method,
            iteration=iterations,
            rtol=problem.rtol,
            atol=problem.atol,
            reason="final_residual_verification_failed",
            operator=operator,
            solution=solution,
            right_hand_side=right_hand_side,
            active_mask=~converged,
            failing_mask=~converged,
            stage="final_verification",
            restart_counts=restart_counts,
            residual_replacement_counts=residual_replacement_counts,
            restart_reason_counts=restart_reason_counts,
            latest_restart_reason=latest_restart_reason,
            work_count=work_count,
        )
    if certified_residual is not None:
        if certified_scaled_residual is None:
            raise RWRInputError(
                "certified_scaled_residual is required with certified_residual"
            )
        certified_converged = certified_scaled_residual <= 1
        if not bool(torch.all(certified_converged)):
            _raise_solver_failure(
                RWRNonConvergenceError,
                method=method,
                iteration=iterations,
                rtol=problem.rtol,
                atol=problem.atol,
                reason="final_fp64_certificate_failed",
                operator=operator,
                solution=solution,
                right_hand_side=right_hand_side,
                active_mask=~certified_converged,
                failing_mask=~certified_converged,
                stage="final_fp64_certificate",
                restart_counts=restart_counts,
                residual_replacement_counts=residual_replacement_counts,
                restart_reason_counts=restart_reason_counts,
                latest_restart_reason=latest_restart_reason,
                work_count=work_count,
                certified_residual=certified_residual,
                certified_scaled_residual=certified_scaled_residual,
                certificate_dtype=certificate_dtype,
                fp64_certificate_checks=fp64_certificate_checks,
                fp64_certified_rhs=fp64_certified_rhs,
                fp64_certificate_rejections=fp64_certificate_rejections,
                fp64_certificate_restart_count=fp64_certificate_restart_count,
                fp64_certificate_work=fp64_certificate_work,
            )
    if fixed_point_delta_inf is None:
        update = right_hand_side + problem.alpha * problem.graph.matmul(solution)
        fixed_point_delta = (update - solution).abs().max()
        if not bool(torch.isfinite(fixed_point_delta)):
            _raise_solver_failure(
                RWRNonFiniteError,
                method=method,
                iteration=iterations,
                rtol=problem.rtol,
                atol=problem.atol,
                reason="non_finite_fixed_point_diagnostic",
                operator=operator,
                solution=solution,
                right_hand_side=right_hand_side,
                stage="final_verification",
                tensor="fixed_point_delta",
                restart_counts=restart_counts,
                residual_replacement_counts=residual_replacement_counts,
                restart_reason_counts=restart_reason_counts,
                latest_restart_reason=latest_restart_reason,
                work_count=work_count,
            )
        fixed_point_delta_inf = float(fixed_point_delta.item())
    working_absolute_residual_inf = float(residual.abs().max().item())
    working_maximum_scaled_residual = float(scaled.max().item())
    if certified_residual is None:
        certified_residual = residual
        certified_scaled_residual = scaled
    certified_absolute_residual_inf = float(certified_residual.abs().max().item())
    certified_maximum_scaled_residual = float(
        certified_scaled_residual.max().item()
    )
    absolute_residual_inf = certified_absolute_residual_inf
    maximum_scaled_residual = certified_maximum_scaled_residual
    if certificate_dtype is None:
        certificate_dtype = str(certified_residual.dtype).removeprefix("torch.")
    output = solution[:, 0] if problem.was_vector else solution
    restarts_per_rhs = (
        ()
        if restart_counts is None
        else tuple(int(value) for value in restart_counts.tolist())
    )
    residual_replacements_per_rhs = (
        ()
        if residual_replacement_counts is None
        else tuple(int(value) for value in residual_replacement_counts.tolist())
    )
    return RWRSolveResult(
        scores=output,
        method=method,
        alpha=problem.alpha,
        converged=True,
        iterations=iterations,
        absolute_residual_inf=absolute_residual_inf,
        maximum_scaled_residual=maximum_scaled_residual,
        fixed_point_delta_inf=fixed_point_delta_inf,
        work_count=work_count,
        total_restart_count=sum(restarts_per_rhs),
        max_restarts_per_rhs=max(restarts_per_rhs, default=0),
        total_residual_replacement_count=sum(residual_replacements_per_rhs),
        restarts_per_rhs=restarts_per_rhs,
        residual_replacements_per_rhs=residual_replacements_per_rhs,
        restart_reason_counts=tuple(sorted((restart_reason_counts or {}).items())),
        working_absolute_residual_inf=working_absolute_residual_inf,
        working_maximum_scaled_residual=working_maximum_scaled_residual,
        certified_absolute_residual_inf=certified_absolute_residual_inf,
        certified_maximum_scaled_residual=certified_maximum_scaled_residual,
        certificate_dtype=certificate_dtype,
        fp64_certificate_checks=fp64_certificate_checks,
        fp64_certified_rhs=fp64_certified_rhs,
        fp64_certificate_rejections=fp64_certificate_rejections,
        fp64_certificate_restart_count=fp64_certificate_restart_count,
        fp64_certificate_work=fp64_certificate_work,
    )


@torch.no_grad()
def solve_rwr_fixed_point(
    graph: DirectedTopKGraph,
    unary_scores: torch.Tensor,
    *,
    alpha: float = 0.98,
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int = 5000,
) -> RWRSolveResult:
    """Solve RWR by contraction, accepting only the true primal residual."""
    problem = _prepare_problem(
        graph, unary_scores, "fixed_point", alpha, rtol, atol, max_iter
    )
    operator = SparseRWROperator(problem.graph, problem.alpha)
    solution = problem.scores_matrix.clone()
    right_hand_side = (1 - problem.alpha) * problem.scores_matrix
    if problem.alpha == 0:
        return _make_result(problem, operator, solution, method="fixed_point", iterations=0)

    residual, converged, _scaled = _residual_state(
        operator,
        solution,
        right_hand_side,
        method="fixed_point",
        iteration=0,
        rtol=problem.rtol,
        atol=problem.atol,
    )
    del residual
    if bool(torch.all(converged)):
        return _make_result(
            problem,
            operator,
            solution,
            method="fixed_point",
            iterations=0,
        )

    fixed_point_delta = torch.full(
        (), math.inf, dtype=solution.dtype, device=solution.device
    )
    for iteration in range(1, problem.max_iter + 1):
        updated = right_hand_side + problem.alpha * problem.graph.matmul(solution)
        if not bool(torch.isfinite(updated).all()):
            _raise_solver_failure(
                RWRNonFiniteError,
                method="fixed_point",
                iteration=iteration - 1,
                rtol=problem.rtol,
                atol=problem.atol,
                reason="non_finite_fixed_point_iterate",
                operator=operator,
                solution=solution,
                right_hand_side=right_hand_side,
                stage="fixed_point_update",
                tensor="updated_scores",
            )
        fixed_point_delta = (updated - solution).abs().max()
        if not bool(torch.isfinite(fixed_point_delta)):
            _raise_solver_failure(
                RWRNonFiniteError,
                method="fixed_point",
                iteration=iteration,
                rtol=problem.rtol,
                atol=problem.atol,
                reason="non_finite_fixed_point_delta",
                operator=operator,
                solution=updated,
                right_hand_side=right_hand_side,
                stage="fixed_point_update",
                tensor="fixed_point_delta",
            )
        solution = updated
        _residual, converged, _scaled = _residual_state(
            operator,
            solution,
            right_hand_side,
            method="fixed_point",
            iteration=iteration,
            rtol=problem.rtol,
            atol=problem.atol,
        )
        if bool(torch.all(converged)):
            return _make_result(
                problem,
                operator,
                solution,
                method="fixed_point",
                iterations=iteration,
                fixed_point_delta_inf=float(fixed_point_delta.item()),
            )
    _raise_solver_failure(
        RWRNonConvergenceError,
        method="fixed_point",
        iteration=problem.max_iter,
        rtol=problem.rtol,
        atol=problem.atol,
        reason="max_iterations_exhausted",
        operator=operator,
        solution=solution,
        right_hand_side=right_hand_side,
        active_mask=~converged,
        failing_mask=~converged,
        stage="convergence_check",
    )


@torch.no_grad()
def solve_rwr_cgls(
    graph: DirectedTopKGraph,
    unary_scores: torch.Tensor,
    *,
    alpha: float = 0.98,
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int = 5000,
) -> RWRSolveResult:
    """Solve nonsymmetric RWR with reliable-update, restarted CGLS.

    The recursive residual preserves the CGLS conjugacy recurrence between
    reliable replacements.  The independently recomputed true primal
    residual remains the sole convergence authority after every finite
    solution update.  Each active RHS periodically rebuilds its Krylov state
    from ``B - K@P`` after 128 updates. A column is also restored to its best
    finite iterate and restarted after severe stagnation or a recoverable
    recurrence breakdown. Restarts are column-local and deterministic.
    ``max_iter`` counts completed CGLS updates, matching the historical
    contract; ``work_count`` reports updates plus restart/recovery work.
    Device-local sparse FP64 certificate work is reported independently and
    never consumes a completed iteration or ``work_count`` unit.
    """
    problem = _prepare_problem(
        graph, unary_scores, "cgls", alpha, rtol, atol, max_iter
    )
    operator = SparseRWROperator(problem.graph, problem.alpha)
    solution = problem.scores_matrix.clone()
    right_hand_side = (1 - problem.alpha) * problem.scores_matrix
    rhs_count = problem.scores_matrix.shape[1]
    restart_counts = torch.zeros(
        rhs_count, dtype=torch.int64, device=solution.device
    )
    residual_replacement_counts = torch.zeros_like(restart_counts)
    restart_reason_counts: dict[str, int] = {}
    latest_restart_reason: str | None = None
    use_quantized_fp64_certificate = solution.dtype == torch.float32
    fp64_certificate_checks = 0
    fp64_certificate_rejections = 0
    fp64_certificate_restart_count = 0
    fp64_certificate_work = 0
    certified_rhs = torch.zeros(
        rhs_count, dtype=torch.bool, device=solution.device
    )

    def telemetry() -> dict[str, object]:
        return {
            "restart_counts": restart_counts,
            "residual_replacement_counts": residual_replacement_counts,
            "restart_reason_counts": restart_reason_counts,
            "latest_restart_reason": latest_restart_reason,
            "certificate_dtype": (
                "float64_quantized_fp32_system"
                if use_quantized_fp64_certificate
                else "float64_native_solver"
            ),
            "fp64_certificate_checks": fp64_certificate_checks,
            "fp64_certified_rhs": int(certified_rhs.count_nonzero().item()),
            "fp64_certificate_rejections": fp64_certificate_rejections,
            "fp64_certificate_restart_count": fp64_certificate_restart_count,
            "fp64_certificate_work": fp64_certificate_work,
        }

    def certify_candidates(
        candidate_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        _FP64ResidualCertificate | None,
    ]:
        """Return passing/rejected global masks for working-gate candidates."""
        nonlocal fp64_certificate_checks, fp64_certificate_rejections
        nonlocal fp64_certificate_work
        passed = torch.zeros_like(candidate_mask)
        rejected = torch.zeros_like(candidate_mask)
        if not bool(torch.any(candidate_mask)):
            return passed, rejected, None
        if not use_quantized_fp64_certificate:
            passed[candidate_mask] = True
            certified_rhs[candidate_mask] = True
            return passed, rejected, None
        columns = candidate_mask.nonzero().flatten()
        certificate = _sparse_fp64_residual_certificate(
            problem.graph,
            problem.alpha,
            solution,
            right_hand_side,
            columns=columns,
            rtol=problem.rtol,
            atol=problem.atol,
        )
        fp64_certificate_checks += 1
        fp64_certificate_work += 1
        local_passed = certificate.scaled_residual <= 1
        passed[columns[local_passed]] = True
        rejected[columns[~local_passed]] = True
        certified_rhs[passed] = True
        certified_rhs[rejected] = False
        fp64_certificate_rejections += int(rejected.count_nonzero().item())
        return passed, rejected, certificate

    def replace_rejected_residual(
        working_residual: torch.Tensor,
        rejected: torch.Tensor,
        certificate: _FP64ResidualCertificate | None,
    ) -> None:
        """Install B-KP from the FP64 certificate in the K@P-B recurrence."""
        nonlocal latest_restart_reason, fp64_certificate_restart_count
        if not bool(torch.any(rejected)):
            return
        if certificate is None:
            raise AssertionError("certificate rejection lacks FP64 residual")
        local_rejected = certificate.scaled_residual > 1
        rejected_columns = certificate.columns[local_rejected]
        working_residual[:, rejected_columns] = -certificate.residual[
            :, local_rejected
        ].to(working_residual.dtype)
        restart_counts[rejected] += 1
        residual_replacement_counts[rejected] += 1
        rejection_count = int(rejected.count_nonzero().item())
        restart_reason_counts["fp64_certificate_rejection"] = (
            restart_reason_counts.get("fp64_certificate_rejection", 0)
            + rejection_count
        )
        latest_restart_reason = "fp64_certificate_rejection"
        fp64_certificate_restart_count += rejection_count
        # Certificate rejection is not a numerical breakdown. It may recur
        # while a valid FP32 iterate approaches the FP64 boundary, so it is
        # bounded by the unchanged completed-update limit rather than the
        # three-strike recovery counter used for arithmetic breakdowns.
        solution_compensation[:, rejected] = 0
        steps_since_restart[rejected] = 0
        steps_since_progress[rejected] = 0

    def final_fp64_certificate() -> _FP64ResidualCertificate | None:
        """Defensively certify every FP32 RHS immediately before return."""
        nonlocal fp64_certificate_checks, fp64_certificate_work
        if not use_quantized_fp64_certificate:
            return None
        certificate = _sparse_fp64_residual_certificate(
            problem.graph,
            problem.alpha,
            solution,
            right_hand_side,
            rtol=problem.rtol,
            atol=problem.atol,
        )
        fp64_certificate_checks += 1
        fp64_certificate_work += 1
        return certificate

    solution_compensation = torch.zeros_like(solution)
    steps_since_restart = torch.zeros_like(restart_counts)
    steps_since_progress = torch.zeros_like(restart_counts)
    recovery_restart_streak = torch.zeros_like(restart_counts)
    completed_iterations = 0
    budget_used = 0

    if problem.alpha == 0:
        certificate = final_fp64_certificate()
        if certificate is not None:
            certified_rhs[:] = certificate.scaled_residual <= 1
        return _make_result(
            problem,
            operator,
            solution,
            method="cgls",
            iterations=0,
            certified_residual=(
                None if certificate is None else certificate.residual
            ),
            certified_scaled_residual=(
                None if certificate is None else certificate.scaled_residual
            ),
            **telemetry(),
        )

    residual, converged, scaled = _residual_state(
        operator,
        solution,
        right_hand_side,
        method="cgls",
        iteration=0,
        rtol=problem.rtol,
        atol=problem.atol,
        **telemetry(),
    )
    initial_candidates = converged.clone()
    passed, rejected, certificate = certify_candidates(initial_candidates)
    active = ~passed
    replace_rejected_residual(residual, rejected, certificate)
    if not bool(torch.any(active)):
        certificate = final_fp64_certificate()
        if certificate is not None and not bool(
            torch.all(certificate.scaled_residual <= 1)
        ):
            raise AssertionError(
                "candidate certificate changed before final certification"
            )
        return _make_result(
            problem,
            operator,
            solution,
            method="cgls",
            iterations=0,
            certified_residual=(
                None if certificate is None else certificate.residual
            ),
            certified_scaled_residual=(
                None if certificate is None else certificate.scaled_residual
            ),
            **telemetry(),
        )

    normal_residual = operator._transpose_matmul_prepared(-residual)
    normal_residual[:, ~active] = 0
    direction = normal_residual.clone()
    gamma = _cgls_squared_column_norm(normal_residual)
    tiny = torch.finfo(solution.dtype).tiny
    failing = active & ~torch.isfinite(gamma)
    if bool(torch.any(failing)):
        _raise_solver_failure(
            RWRNonFiniteError,
            method="cgls",
            iteration=0,
            rtol=problem.rtol,
            atol=problem.atol,
            reason="non_finite_initial_normal_residual_norm",
            operator=operator,
            solution=solution,
            right_hand_side=right_hand_side,
            active_mask=active,
            failing_mask=failing,
            stage="initialization",
            tensor="normal_residual_norm_squared",
            **telemetry(),
        )
    failing = active & (gamma <= tiny)
    if bool(torch.any(failing)):
        _raise_solver_failure(
            RWRNumericalBreakdownError,
            method="cgls",
            iteration=0,
            rtol=problem.rtol,
            atol=problem.atol,
            reason="initial_normal_residual_vanished",
            operator=operator,
            solution=solution,
            right_hand_side=right_hand_side,
            active_mask=active,
            failing_mask=failing,
            stage="initialization",
            tensor="normal_residual_norm_squared",
            breakdown_value=float(gamma[failing].max().item()),
            **telemetry(),
        )

    best_solution = solution.clone()
    # Neumaier/Kahan-style compensation prevents small FP32 CGLS increments
    # from being discarded when accumulated into a much larger current score.
    # It remains in the working dtype and stores no Krylov history.
    best_scaled = scaled.clone()
    working_epsilon = torch.finfo(solution.dtype).eps

    def raise_cgls_failure(
        exception_type: type[RWRSolverError],
        *,
        reason: str,
        failing_mask: torch.Tensor,
        stage: str,
        tensor: str,
        breakdown_value: float | None = None,
        detail: str | None = None,
    ) -> NoReturn:
        _raise_solver_failure(
            exception_type,
            method="cgls",
            iteration=completed_iterations,
            rtol=problem.rtol,
            atol=problem.atol,
            reason=reason,
            operator=operator,
            solution=solution,
            right_hand_side=right_hand_side,
            active_mask=active,
            failing_mask=failing_mask,
            stage=stage,
            tensor=tensor,
            breakdown_value=breakdown_value,
            detail=detail,
            work_count=budget_used,
            **telemetry(),
        )

    def restart_from_true_residual(
        restart_mask: torch.Tensor,
        *,
        reason: str,
        restore_best: bool,
        failure_type: type[RWRSolverError],
        tensor: str,
    ) -> None:
        nonlocal residual, normal_residual, gamma, active
        nonlocal scaled, latest_restart_reason
        repeated = restart_mask & (
            recovery_restart_streak >= _CGLS_MAX_RECOVERY_RESTARTS
        )
        if bool(torch.any(repeated)):
            latest_restart_reason = reason
            raise_cgls_failure(
                failure_type,
                reason="repeated_recovery_failure",
                failing_mask=repeated,
                stage="reliable_restart",
                tensor=tensor,
                detail=f"latest breakdown reason: {reason}",
            )
        if restore_best:
            solution[:, restart_mask] = best_solution[:, restart_mask]
            solution_compensation[:, restart_mask] = 0

        restart_counts[restart_mask] += 1
        residual_replacement_counts[restart_mask] += 1
        restart_reason_counts[reason] = restart_reason_counts.get(reason, 0) + int(
            restart_mask.count_nonzero().item()
        )
        latest_restart_reason = reason
        if reason != "periodic_residual_replacement":
            recovery_restart_streak[restart_mask] += 1

        residual, converged_now, scaled = _residual_state(
            operator,
            solution,
            right_hand_side,
            method="cgls",
            iteration=completed_iterations,
            active_mask=active,
            rtol=problem.rtol,
            atol=problem.atol,
            **telemetry(),
        )
        candidates = restart_mask & active & converged_now
        passed, rejected, certificate = certify_candidates(candidates)
        replace_rejected_residual(residual, rejected, certificate)
        active = active & ~passed
        rebuild_mask = restart_mask & active
        if not bool(torch.any(rebuild_mask)):
            direction[:, restart_mask] = 0
            normal_residual[:, restart_mask] = 0
            gamma[restart_mask] = 0
            return

        rebuilt_normal = operator._transpose_matmul_prepared(-residual)
        invalid_normal = rebuild_mask & ~torch.isfinite(rebuilt_normal).all(dim=0)
        if bool(torch.any(invalid_normal)):
            raise_cgls_failure(
                RWRNonFiniteError,
                reason="non_finite_restarted_normal_residual",
                failing_mask=invalid_normal,
                stage="reliable_restart",
                tensor="restarted_normal_residual",
            )
        rebuilt_gamma = _cgls_squared_column_norm(rebuilt_normal)
        invalid_gamma = rebuild_mask & ~torch.isfinite(rebuilt_gamma)
        if bool(torch.any(invalid_gamma)):
            raise_cgls_failure(
                RWRNonFiniteError,
                reason="non_finite_restarted_normal_residual_norm",
                failing_mask=invalid_gamma,
                stage="reliable_restart",
                tensor="restarted_normal_residual_norm_squared",
            )
        vanished = rebuild_mask & (rebuilt_gamma <= tiny)
        if bool(torch.any(vanished)):
            raise_cgls_failure(
                RWRNumericalBreakdownError,
                reason="restarted_normal_residual_vanished",
                failing_mask=vanished,
                stage="reliable_restart",
                tensor="restarted_normal_residual_norm_squared",
                breakdown_value=float(rebuilt_gamma[vanished].max().item()),
            )
        normal_residual[:, rebuild_mask] = rebuilt_normal[:, rebuild_mask]
        direction[:, rebuild_mask] = rebuilt_normal[:, rebuild_mask]
        gamma[rebuild_mask] = rebuilt_gamma[rebuild_mask]
        steps_since_restart[restart_mask] = 0
        if reason != "periodic_residual_replacement":
            steps_since_progress[restart_mask] = 0

    while completed_iterations < problem.max_iter:
        direction_finite = torch.isfinite(direction).all(dim=0)
        direction_alignment = _cgls_column_inner(normal_residual, direction)
        failing = active & (
            ~direction_finite | ~torch.isfinite(direction_alignment)
        )
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="invalid_search_direction_alignment",
                restore_best=True,
                failure_type=RWRNumericalBreakdownError,
                tensor="normal_residual_search_direction_inner_product",
            )
            budget_used += 1
            continue

        forward_direction = operator._matmul_prepared(direction)
        failing = active & ~torch.isfinite(forward_direction).all(dim=0)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="non_finite_search_direction_operator_output",
                restore_best=True,
                failure_type=RWRNonFiniteError,
                tensor="K_direction",
            )
            budget_used += 1
            continue
        denominator = _cgls_squared_column_norm(forward_direction)
        failing = active & ~torch.isfinite(denominator)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="non_finite_search_direction_operator_norm",
                restore_best=True,
                failure_type=RWRNonFiniteError,
                tensor="search_direction_operator_norm_squared",
            )
            budget_used += 1
            continue
        failing = active & (denominator <= tiny)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="search_direction_operator_norm_vanished",
                restore_best=True,
                failure_type=RWRNumericalBreakdownError,
                tensor="search_direction_operator_norm_squared",
            )
            budget_used += 1
            continue
        step = torch.zeros(
            gamma.shape, dtype=solution.dtype, device=solution.device
        )
        step[active] = (gamma[active] / denominator[active]).to(solution.dtype)
        failing = active & ~torch.isfinite(step)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="non_finite_step_coefficient",
                restore_best=True,
                failure_type=RWRNonFiniteError,
                tensor="step",
            )
            budget_used += 1
            continue
        increment = direction * step[None, :]
        compensated_increment = increment - solution_compensation
        updated_solution = solution + compensated_increment
        updated_compensation = (
            updated_solution - solution
        ) - compensated_increment
        failing = active & ~torch.isfinite(updated_solution).all(dim=0)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="non_finite_solution_update",
                restore_best=True,
                failure_type=RWRNonFiniteError,
                tensor="updated_solution",
            )
            budget_used += 1
            continue
        solution = updated_solution
        solution_compensation = updated_compensation
        budget_used += 1
        completed_iterations += 1

        previous_active = active.clone()
        # Preserve the CGLS recurrence between reliable replacements. Feeding
        # a freshly recomputed FP32 residual into the unrestarted direction
        # recurrence destroys conjugacy and caused the canonical first crop to
        # stagnate.  The separate true residual below remains authoritative
        # for convergence, progress, rollback, and diagnostics.
        residual = residual + forward_direction * step[None, :]
        true_residual, converged_now, scaled = _residual_state(
            operator,
            solution,
            right_hand_side,
            method="cgls",
            iteration=completed_iterations,
            active_mask=active,
            rtol=problem.rtol,
            atol=problem.atol,
            **telemetry(),
        )
        improved = previous_active & (scaled < best_scaled)
        meaningful_progress = previous_active & (
            scaled
            < best_scaled
            * (1 - _CGLS_PROGRESS_EPS_MULTIPLIER * working_epsilon)
        )
        best_solution[:, improved] = solution[:, improved]
        best_scaled[improved] = scaled[improved]
        steps_since_restart[previous_active] += 1
        steps_since_progress[previous_active] += 1
        steps_since_progress[meaningful_progress] = 0
        recovery_restart_streak[meaningful_progress] = 0
        candidates = previous_active & converged_now
        passed, rejected, candidate_certificate = certify_candidates(candidates)
        replace_rejected_residual(residual, rejected, candidate_certificate)
        certificate_restarted = rejected.clone()
        active = active & ~passed
        solution_compensation[:, ~active] = 0
        if not bool(torch.any(active)):
            final_certificate = final_fp64_certificate()
            if final_certificate is not None:
                final_passed = final_certificate.scaled_residual <= 1
                if not bool(torch.all(final_passed)):
                    final_rejected = ~final_passed
                    fp64_certificate_rejections += int(
                        final_rejected.count_nonzero().item()
                    )
                    certified_rhs[final_rejected] = False
                    replace_rejected_residual(
                        residual,
                        final_rejected,
                        final_certificate,
                    )
                    certificate_restarted |= final_rejected
                    active = final_rejected
                else:
                    certified_rhs[:] = True
            if not bool(torch.any(active)):
                return _make_result(
                    problem,
                    operator,
                    solution,
                    method="cgls",
                    iterations=completed_iterations,
                    work_count=budget_used,
                    certified_residual=(
                        None
                        if final_certificate is None
                        else final_certificate.residual
                    ),
                    certified_scaled_residual=(
                        None
                        if final_certificate is None
                        else final_certificate.scaled_residual
                    ),
                    **telemetry(),
                )

        next_normal_residual = operator._transpose_matmul_prepared(-residual)
        next_normal_residual[:, ~active] = 0
        failing = active & ~torch.isfinite(next_normal_residual).all(dim=0)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="non_finite_updated_normal_residual",
                restore_best=True,
                failure_type=RWRNonFiniteError,
                tensor="updated_normal_residual",
            )
            budget_used += 1
            continue
        next_gamma = _cgls_squared_column_norm(next_normal_residual)
        failing = active & ~torch.isfinite(next_gamma)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="non_finite_updated_normal_residual_norm",
                restore_best=True,
                failure_type=RWRNonFiniteError,
                tensor="updated_normal_residual_norm_squared",
            )
            budget_used += 1
            continue
        failing = active & (next_gamma <= tiny)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="updated_normal_residual_vanished",
                restore_best=True,
                failure_type=RWRNumericalBreakdownError,
                tensor="updated_normal_residual_norm_squared",
            )
            budget_used += 1
            continue
        coefficient = torch.zeros(
            gamma.shape, dtype=solution.dtype, device=solution.device
        )
        continuing = active & ~certificate_restarted
        coefficient[continuing] = (
            next_gamma[continuing] / gamma[continuing]
        ).to(
            solution.dtype
        )
        failing = active & ~torch.isfinite(coefficient)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="non_finite_direction_coefficient",
                restore_best=True,
                failure_type=RWRNonFiniteError,
                tensor="direction_coefficient",
            )
            budget_used += 1
            continue
        updated_direction = next_normal_residual + direction * coefficient[None, :]
        updated_direction[:, certificate_restarted] = next_normal_residual[
            :, certificate_restarted
        ]
        failing = active & ~torch.isfinite(updated_direction).all(dim=0)
        if bool(torch.any(failing)):
            restart_from_true_residual(
                failing,
                reason="non_finite_search_direction",
                restore_best=True,
                failure_type=RWRNonFiniteError,
                tensor="search_direction",
            )
            budget_used += 1
            continue
        direction = updated_direction
        direction[:, ~active] = 0
        normal_residual = next_normal_residual
        gamma = next_gamma

        severe_stagnation = active & (
            (steps_since_progress >= _CGLS_STAGNATION_STEPS)
            & (scaled > _CGLS_STAGNATION_RATIO * best_scaled)
        )
        periodic = active & (
            steps_since_restart >= _CGLS_PERIODIC_RESTART_STEPS
        ) & ~severe_stagnation
        if (
            bool(torch.any(severe_stagnation))
            and completed_iterations < problem.max_iter
        ):
            restart_from_true_residual(
                severe_stagnation,
                reason="severe_true_residual_stagnation",
                restore_best=True,
                failure_type=RWRNumericalBreakdownError,
                tensor="scaled_true_primal_residual",
            )
            budget_used += 1
        if bool(torch.any(periodic)) and completed_iterations < problem.max_iter:
            restart_from_true_residual(
                periodic,
                reason="periodic_residual_replacement",
                restore_best=False,
                failure_type=RWRNumericalBreakdownError,
                tensor="true_primal_residual",
            )
            budget_used += 1

    final_certificate = final_fp64_certificate()
    final_failing = active.clone()
    if final_certificate is not None:
        final_certificate_passed = final_certificate.scaled_residual <= 1
        certified_rhs[final_certificate_passed] = True
        certified_rhs[~final_certificate_passed] = False
        fp64_certificate_rejections += int(
            (~final_certificate_passed).count_nonzero().item()
        )
        final_failing |= ~final_certificate_passed
    _raise_solver_failure(
        RWRNonConvergenceError,
        method="cgls",
        iteration=completed_iterations,
        rtol=problem.rtol,
        atol=problem.atol,
        reason="max_iterations_exhausted",
        operator=operator,
        solution=solution,
        right_hand_side=right_hand_side,
        active_mask=final_failing,
        failing_mask=final_failing,
        stage="convergence_check",
        detail=f"completed_iteration_limit={problem.max_iter}",
        work_count=budget_used,
        certified_residual=(
            None if final_certificate is None else final_certificate.residual
        ),
        certified_scaled_residual=(
            None
            if final_certificate is None
            else final_certificate.scaled_residual
        ),
        **telemetry(),
    )


def solve_rwr(
    graph: DirectedTopKGraph,
    unary_scores: torch.Tensor,
    *,
    alpha: float = 0.98,
    method: str = "cgls",
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int = 5000,
) -> RWRSolveResult:
    """Dispatch to one of the two verified nonsymmetric RWR solvers."""
    if method == "cgls":
        return solve_rwr_cgls(
            graph,
            unary_scores,
            alpha=alpha,
            rtol=rtol,
            atol=atol,
            max_iter=max_iter,
        )
    if method == "fixed_point":
        return solve_rwr_fixed_point(
            graph,
            unary_scores,
            alpha=alpha,
            rtol=rtol,
            atol=atol,
            max_iter=max_iter,
        )
    raise RWRInputError(
        f"method must be 'cgls' or 'fixed_point', got {method!r}"
    )


__all__ = [
    "RWRInputError",
    "RWRNonConvergenceError",
    "RWRNonFiniteError",
    "RWRNumericalBreakdownError",
    "RWRSolveResult",
    "RWRSolverError",
    "SparseRWROperator",
    "solve_rwr",
    "solve_rwr_cgls",
    "solve_rwr_fixed_point",
]
