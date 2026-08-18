"""Opt-in components for COVER-DR experiments."""

from .graph import DirectedTopKGraph, build_directed_topk_graph
from .inference import (
    RWRInferenceConfig,
    RWRInferenceConfigError,
    RWRInferenceOutput,
    RWRRuntimeSummary,
    RWRWindowDiagnostics,
    apply_rwr_to_e3_snapshot,
    build_rwr_structured_record,
    patch_scores_to_masks,
)
from .rwr import (
    RWRInputError,
    RWRNonConvergenceError,
    RWRNonFiniteError,
    RWRNumericalBreakdownError,
    RWRSolveResult,
    RWRSolverError,
    SparseRWROperator,
    solve_rwr,
    solve_rwr_cgls,
    solve_rwr_fixed_point,
)

__all__ = [
    "DirectedTopKGraph",
    "RWRInferenceConfig",
    "RWRInferenceConfigError",
    "RWRInferenceOutput",
    "RWRRuntimeSummary",
    "RWRWindowDiagnostics",
    "RWRInputError",
    "RWRNonConvergenceError",
    "RWRNonFiniteError",
    "RWRNumericalBreakdownError",
    "RWRSolveResult",
    "RWRSolverError",
    "SparseRWROperator",
    "build_directed_topk_graph",
    "apply_rwr_to_e3_snapshot",
    "build_rwr_structured_record",
    "patch_scores_to_masks",
    "solve_rwr",
    "solve_rwr_cgls",
    "solve_rwr_fixed_point",
]
