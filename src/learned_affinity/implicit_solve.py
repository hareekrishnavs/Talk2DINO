"""O(N*k + N*C) implicit fixed-point graph propagation with adjoint backward.

Solver history (see ADVERSARIAL_REVIEW_PARTF1.md, X3): the original solver
here was the Richardson/fixed-point recurrence
``S_{t+1} = (1-alpha)*S0 + alpha*A*S_t`` -- the same iteration
``src.e3_affinity_oracle.propagate_scores`` runs for a fixed T. An
adversarial review found that gradients computed from that solver at its
own DEFAULT tolerance (``tol=1e-6``) disagreed with finite differences by
30-70% relative error -- correct in principle at very tight tolerance
(``tol=1e-12``), but the shipped default was nowhere near tight enough, and
nothing in the test suite exercised the actual default.

The production solver is now **CGLS** (Conjugate Gradient for Least
Squares, i.e. CG applied to the normal equations M^T M x = M^T b without
ever forming M^T M) -- plain CG does not apply directly, because
``M = I - alpha*A`` is NOT symmetric (A is directed/row-stochastic; A^T !=
A in general). CGLS needs one application of M and one of M^T per
iteration (vs. Richardson's one application of A per iteration), but
converges in a number of iterations governed by roughly
sqrt(condition number of M) rather than the condition number itself, and
-- critically -- exposes a real, checkable residual at each step, so a
genuine convergence ASSERTION is possible: if a solve does not reach `tol`
within `max_iter`, it raises `SolverConvergenceError` rather than silently
returning an under-converged (and therefore under-differentiated) result.

The old Richardson solver is kept (renamed `*_richardson`) only for the
F1g/X7 comparison against `propagate_scores`, which itself is a fixed-T
Richardson iteration -- that comparison is only meaningful against the same
kind of solver. It is no longer used by `ImplicitPropagate`.

A is never materialised densely -- only ``indices``/``weights`` of shape
``[P,K]`` plus O(1) [P,C]-shaped iterates are ever held, so memory is
O(P*K + P*C) regardless of iteration count, for BOTH solvers. No iterate is
retained after either loop (both run under ``torch.no_grad()``): backprop
goes through the ANALYTIC adjoint formula, never through the iterations.
"""
from __future__ import annotations

from typing import Callable

import torch

# Diagnostic-only side channel (F1d/F1g/F1h ask to "report iterations
# actually needed" / "report measured peak memory" -- torch.autograd.
# Function's ctx is not readable after backward() returns, so the last
# forward/backward iteration counts are mirrored here for reporting.
# Never read by, or fed into, any loss or metric.
LAST_SOLVE_STATS: dict[str, int | float | bool] = {}


class SolverConvergenceError(RuntimeError):
    """Raised when a solve does not reach `tol` within `max_iter` and
    `raise_on_nonconvergence=True` (the default). An under-converged S* (or
    adjoint lambda) silently fed into the analytic gradient formula is
    exactly the failure mode that produced X3's 30-70% gradient error --
    this makes that failure loud instead of silent."""


def apply_knn(features: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``(A @ features)`` where ``A[p, indices[p,k]] = weights[p,k]``.
    features: [P,C], indices: [P,K] int64, weights: [P,K]. Returns [P,C]."""
    neighbours = features[indices]  # [P,K,C]
    return (neighbours * weights.unsqueeze(-1)).sum(dim=1)


def apply_knn_transpose(
    features: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, num_patches: int,
) -> torch.Tensor:
    """``(A^T @ features)``, via scatter-add -- never materialises A. Same
    shapes as apply_knn; num_patches is P (needed since index_add's output
    size cannot be inferred from indices alone)."""
    patches, k = indices.shape
    channels = features.shape[-1]
    src = features.unsqueeze(1) * weights.unsqueeze(-1)  # [P,K,C]
    out = torch.zeros(num_patches, channels, dtype=features.dtype, device=features.device)
    out.index_add_(0, indices.reshape(-1), src.reshape(patches * k, channels))
    return out


def _cgls_solve(
    b: torch.Tensor,
    apply_forward: Callable[[torch.Tensor], torch.Tensor],
    apply_transpose: Callable[[torch.Tensor], torch.Tensor],
    *, tol: float, max_iter: int,
) -> tuple[torch.Tensor, int, bool]:
    """Solve M x = b via CGLS (CG on the normal equations M^T M x = M^T b,
    never formed explicitly): one `apply_forward` (M@x) and one
    `apply_transpose` (M^T@x) per iteration. b, x: [P,C] -- C independent
    right-hand sides solved simultaneously (block CG: step sizes alpha/beta
    are per-column [C] vectors, not scalars). Stopping rule: relative
    residual on the ORIGINAL system, ``||b - M@x|| / (||b||+eps) < tol``,
    per column, ALL columns must satisfy it. Returns (x, iters, converged).
    No autograd tracking; caller is responsible for detaching inputs."""
    eps = torch.finfo(b.dtype).tiny
    with torch.no_grad():
        x = b.clone()  # same start point convention the old Richardson solver used
        r = b - apply_forward(x)
        z = apply_transpose(r)
        p = z.clone()
        z_norm_sq = (z * z).sum(dim=0)  # [C]
        b_norm = b.norm(dim=0).clamp_min(eps)  # [C]
        iters = 0
        converged = False
        for iters in range(1, max_iter + 1):
            w = apply_forward(p)
            w_norm_sq = (w * w).sum(dim=0).clamp_min(eps)  # [C]
            step = z_norm_sq / w_norm_sq  # [C]
            x = x + step[None, :] * p
            r = r - step[None, :] * w
            if torch.all(r.norm(dim=0) / b_norm < tol):
                converged = True
                break
            z_new = apply_transpose(r)
            z_norm_sq_new = (z_new * z_new).sum(dim=0)
            beta = z_norm_sq_new / z_norm_sq.clamp_min(eps)  # [C]
            p = z_new + beta[None, :] * p
            z_norm_sq = z_norm_sq_new
    return x, iters, converged


def solve_fixed_point(
    s0: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, alpha: float,
    *, tol: float = 1e-10, max_iter: int = 1000, raise_on_nonconvergence: bool = True,
) -> tuple[torch.Tensor, int]:
    """Production forward solve: (I - alpha*A) S* = (1-alpha)*S0, via CGLS.
    `raise_on_nonconvergence=False` is for controlled diagnostics only (e.g.
    forcing an exact iteration count to measure memory) -- production
    callers should never disable it."""
    num_patches = s0.shape[0]

    def apply_forward(x: torch.Tensor) -> torch.Tensor:
        return x - alpha * apply_knn(x, indices, weights)

    def apply_transpose(x: torch.Tensor) -> torch.Tensor:
        return x - alpha * apply_knn_transpose(x, indices, weights, num_patches)

    b = (1 - alpha) * s0
    x, iters, converged = _cgls_solve(b, apply_forward, apply_transpose, tol=tol, max_iter=max_iter)
    if not converged and raise_on_nonconvergence:
        raise SolverConvergenceError(
            f"forward CGLS solve did not converge: tol={tol}, max_iter={max_iter}, "
            f"alpha={alpha}. Increase max_iter or loosen tol explicitly if this is expected."
        )
    return x, iters


def solve_adjoint(
    grad_output: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, alpha: float,
    num_patches: int, *, tol: float = 1e-10, max_iter: int = 1000, raise_on_nonconvergence: bool = True,
) -> tuple[torch.Tensor, int]:
    """Production adjoint solve: (I - alpha*A^T) lambda = grad_output, via
    CGLS. M2 = I - alpha*A^T here, so M2^T = I - alpha*A -- the adjoint of
    the adjoint operator is just the forward operator, so apply_transpose
    below is literally apply_knn (not apply_knn_transpose)."""

    def apply_forward(x: torch.Tensor) -> torch.Tensor:
        return x - alpha * apply_knn_transpose(x, indices, weights, num_patches)

    def apply_transpose(x: torch.Tensor) -> torch.Tensor:
        return x - alpha * apply_knn(x, indices, weights)

    x, iters, converged = _cgls_solve(grad_output, apply_forward, apply_transpose, tol=tol, max_iter=max_iter)
    if not converged and raise_on_nonconvergence:
        raise SolverConvergenceError(
            f"adjoint CGLS solve did not converge: tol={tol}, max_iter={max_iter}, "
            f"alpha={alpha}. Increase max_iter or loosen tol explicitly if this is expected."
        )
    return x, iters


def solve_fixed_point_richardson(
    s0: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, alpha: float,
    *, tol: float = 1e-6, max_iter: int = 500,
) -> tuple[torch.Tensor, int]:
    """Original Richardson/fixed-point solver -- kept ONLY for the F1g/X7
    comparison against propagate_scores (itself a fixed-T Richardson
    iteration); no longer used by ImplicitPropagate. See module docstring."""
    eps = torch.finfo(s0.dtype).tiny
    with torch.no_grad():
        current = s0.clone()
        iters = 0
        for iters in range(1, max_iter + 1):
            spread = apply_knn(current, indices, weights)
            updated = (1 - alpha) * s0 + alpha * spread
            diff = (updated - current).norm()
            denom = current.norm().clamp_min(eps)
            current = updated
            if diff / denom < tol:
                break
    return current, iters


def solve_adjoint_richardson(
    grad_output: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, alpha: float,
    num_patches: int, *, tol: float = 1e-6, max_iter: int = 500,
) -> tuple[torch.Tensor, int]:
    """Original Richardson adjoint solver -- see solve_fixed_point_richardson."""
    eps = torch.finfo(grad_output.dtype).tiny
    with torch.no_grad():
        current = grad_output.clone()
        iters = 0
        for iters in range(1, max_iter + 1):
            spread = apply_knn_transpose(current, indices, weights, num_patches)
            updated = grad_output + alpha * spread
            diff = (updated - current).norm()
            denom = current.norm().clamp_min(eps)
            current = updated
            if diff / denom < tol:
                break
    return current, iters


class ImplicitPropagate(torch.autograd.Function):
    """Custom autograd.Function: forward runs the (untracked, O(1)-memory)
    CGLS solve; backward solves the adjoint system (also CGLS) and forms
    the gradient on the SPARSE kNN pattern only (never a dense [P,P]
    tensor).

    Inputs: s0 [P,C] (requires_grad ok), indices [P,K] int64 (no grad),
    weights [P,K] (requires_grad ok), alpha (python float; a
    requires_grad-tensor alpha is rejected explicitly rather than silently
    dropped, since alpha-gradients are out of F1's scope), tol, max_iter
    (python scalars).
    """

    @staticmethod
    def forward(ctx, s0, indices, weights, alpha, tol, max_iter, raise_on_nonconvergence):
        if torch.is_tensor(alpha):
            if alpha.requires_grad:
                raise ValueError(
                    "ImplicitPropagate does not implement d(loss)/d(alpha); "
                    "pass alpha as a plain float or a tensor with requires_grad=False"
                )
            alpha_value = float(alpha.item())
        else:
            alpha_value = float(alpha)
        s_star, iters_forward = solve_fixed_point(
            s0.detach(), indices, weights.detach(), alpha_value, tol=tol, max_iter=max_iter,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )
        ctx.save_for_backward(s_star, indices, weights)
        ctx.alpha = alpha_value
        ctx.tol = tol
        ctx.max_iter = max_iter
        ctx.num_patches = s0.shape[0]
        ctx.raise_on_nonconvergence = raise_on_nonconvergence
        LAST_SOLVE_STATS["forward_iters"] = iters_forward
        LAST_SOLVE_STATS["forward_converged"] = iters_forward < max_iter
        return s_star

    @staticmethod
    def backward(ctx, grad_output):
        s_star, indices, weights = ctx.saved_tensors
        alpha = ctx.alpha
        lam, iters_backward = solve_adjoint(
            grad_output.detach(), indices, weights.detach(), alpha, ctx.num_patches,
            tol=ctx.tol, max_iter=ctx.max_iter, raise_on_nonconvergence=ctx.raise_on_nonconvergence,
        )
        LAST_SOLVE_STATS["backward_iters"] = iters_backward
        LAST_SOLVE_STATS["backward_converged"] = iters_backward < ctx.max_iter

        grad_s0 = (1 - alpha) * lam if ctx.needs_input_grad[0] else None

        grad_weights = None
        if ctx.needs_input_grad[2]:
            # dL/dA_ij = alpha * lambda_i . S*_j for (i,j) in the kNN
            # pattern -- gathered directly, sparse pattern only, no dense
            # [P,P] tensor is ever formed.
            neighbours = s_star[indices]  # [P,K,C]: S*_j for each edge (i,k)
            grad_weights = alpha * (lam.unsqueeze(1) * neighbours).sum(dim=-1)  # [P,K]

        return grad_s0, None, grad_weights, None, None, None, None


def implicit_propagate(
    s0: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, alpha: float,
    *, tol: float = 1e-10, max_iter: int = 1000, raise_on_nonconvergence: bool = True,
) -> torch.Tensor:
    """Public entry point: S0 [P,C], indices/weights [P,K] (weights
    differentiable), alpha in [0,1). Returns S* [P,C].

    Defaults (tol=1e-10, max_iter=1000) are informed by a REAL measurement,
    not a guess: on 20 real production windows (P=1024, C=171, alpha=0.98,
    the highest-alpha operating point), CGLS needed 538-653 iterations to
    reach tol=1e-10 -- small synthetic test problems (N<100) converged in
    15-30 iterations and would have suggested a much smaller, WRONG default.
    Validated end-to-end against finite differences at the small-problem
    scale (see tests/test_learned_affinity.py's
    test_shipped_default_no_overrides) -- unlike the old Richardson
    defaults, calling this with NO keyword overrides is safe; at production
    scale the iteration count (not correctness) is what actually costs
    more than the original, unconverged Richardson timings implied.
    `raise_on_nonconvergence=False` is for controlled diagnostics (e.g.
    forcing an exact iteration count to measure memory) -- production
    callers should never disable it."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0,1)")
    return ImplicitPropagate.apply(s0, indices, weights, alpha, tol, max_iter, raise_on_nonconvergence)
