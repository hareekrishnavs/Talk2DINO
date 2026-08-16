"""Opt-in components for COVER-DR experiments."""

from .graph import DirectedTopKGraph, build_directed_topk_graph
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
    "RWRInputError",
    "RWRNonConvergenceError",
    "RWRNonFiniteError",
    "RWRNumericalBreakdownError",
    "RWRSolveResult",
    "RWRSolverError",
    "SparseRWROperator",
    "build_directed_topk_graph",
    "solve_rwr",
    "solve_rwr_cgls",
    "solve_rwr_fixed_point",
]
