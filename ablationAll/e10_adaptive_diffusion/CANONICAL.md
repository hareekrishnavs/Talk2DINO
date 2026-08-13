# Canonical operating points — converged solve (post-F1 solver fix)

This supersedes the T=320-derived numbers in `RUN_PartF1.md`/`RUN_PartE1.md`
as the reference for anything measuring gain over the alpha=0.98 operating
point, following the adversarial review (`ADVERSARIAL_REVIEW_PARTF1.md`, X3)
and the solver change it triggered.

## What changed and why

`ADVERSARIAL_REVIEW_PARTF1.md`'s X3 finding: `ImplicitPropagate`'s original
solver was the Richardson/fixed-point iteration
(`S_{t+1} = (1-alpha)*S0 + alpha*A*S_t}`) — the same recurrence
`propagate_scores` runs for a fixed T=320. At its *own shipped default*
(`tol=1e-6, max_iter=500`), gradients disagreed with finite differences by
**30–70% relative error**, on both a fresh problem and the repo's own test
dimensions. The gradient math was correct in principle (confirmed at
`tol=1e-12`); the default just wasn't tight enough, and nothing in the test
suite exercised the actual default.

Fixes made in response:
1. **Solver switch: Richardson → CGLS** (Conjugate Gradient for Least
   Squares — CG on the normal equations `M^T M x = M^T b`, `M = I-alpha*A`,
   without ever forming `M^T M`). Plain CG does not apply directly because
   `A` is directed/row-stochastic, not symmetric (`A^T != A` in general).
   CGLS needs one application each of `M` and `M^T` per iteration (vs.
   Richardson's one application of `A`), but exposes a real, checkable
   residual — which is what makes point 2 possible.
2. **Convergence assertion**: `solve_fixed_point`/`solve_adjoint` now raise
   `SolverConvergenceError` if `tol` isn't reached within `max_iter`
   (default `True`; `raise_on_nonconvergence=False` is an explicit,
   documented escape hatch for controlled diagnostics only, e.g. forcing an
   exact iteration count to measure memory). An under-converged S*/lambda
   silently feeding the analytic gradient formula is exactly the X3 failure
   mode — this makes it loud instead of silent.
3. **Defaults corrected from a real measurement, not a guess.** Small
   synthetic test problems (N<100) converge in 15–30 CGLS iterations,
   which would have suggested `max_iter=200` — that value was tried first
   and immediately caught its own inadequacy via the new convergence
   assertion when run against real production-scale data. Measured on 20+
   real windows (P=1024, C=171, alpha=0.98 — the highest-alpha, hardest-to-
   converge operating point): CGLS needs **361–760 iterations** (mean 587)
   to reach `tol=1e-10`. Shipped default is now `tol=1e-10, max_iter=1000`
   — informed by that measurement, with headroom.
4. **New tests**: gradient check at r=0 and r=0.3 through the FULL
   differentiable chain (metric parameters → g(f) → graph → solve → loss,
   not just the solver in isolation) — MLP randomly initialised (not the
   default zero-init) so the check isn't trivially unfalsifiable; a test
   that calls `implicit_propagate` with **no overrides at all** (the
   natural thing F2 training code would write); a test that the
   convergence assertion actually raises. All in
   `tests/test_learned_affinity.py`, 8/8 passing; full suite 76/76 passing.

## Iteration counts and timing at alpha=0.98 (measured, H100, real windows)

| | forward solve | adjoint (backward) solve |
|---|---|---|
| iterations (50-window sample) | min=471 mean=555.1 max=636 | min=502 mean=576.9 max=657 |
| iterations (full 5000-image/11,075-window val, forward only) | min=361 mean=580.6 max=760 | — |
| wall-clock (50-window sample) | mean=92.9ms max=106.6ms | mean=92.4ms max=107.5ms |

Graph construction (`g(f)` → kNN graph) is cheap by comparison: mean 0.5ms.
The solve dominates a would-be training step by ~150-200x, not the ~100x
originally estimated from the (unconverged) Richardson timings — this is
the honest cost of actually reaching `tol=1e-10`, relevant for F2 batch-size
planning. Peak GPU memory at 100 vs 500 forced iterations: 23.267MB vs
23.967MB (3% spread) — still confirms O(P·K + P·C) memory, independent of
iteration count.

## Canonical numbers

| operating point | mIoU | aAcc | mAcc | source |
|---|---|---|---|---|
| α=0.00, T=10 | 28.480169315747716 | 46.614213 | 52.077968 | exact, unchanged (α=0 never touches the graph or solver) |
| α=0.98 — **old (T=320 Richardson, truncated)** | 29.87804875879269 | 48.528202364501915 | 54.13882134194957 | `RUN_PartE1.md`/`RUN_PartF1.md`, historical |
| α=0.98 — **new (CGLS, converged to tol=1e-10)** | **29.877244374599126** | **48.52867057377273** | **54.13703466982515** | this document — **use this going forward** |
| delta (converged − old T=320) | −0.000804 | +0.000468 | −0.001787 | |

The delta is small (all three metrics within ~0.002 percentage points),
confirming `propagate_scores`'s T=320 truncation was already a reasonably
close approximation of the true fixed point — the practical numbers barely
move. What changes is that these new numbers are now the actual converged
fixed point (not a 320-step approximation of it), and — separately — that
the *solver used to reach them* now has a working gradient at its own
default settings, which the old one did not.

**All subsequent experiments, including training the learned affinity
metric (F2), should be measured against the converged α=0.98 numbers above,
not the historical T=320 ones.** The α=0.00 baseline is unchanged and
exact either way.

Full metrics (per-class IoU, confusion counts) are saved at
`ablationAll/e10_adaptive_diffusion/results/canonical_cg_metrics.json`.

## Reproduction

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import src.e3_affinity_oracle as oracle
from src.learned_affinity.implicit_solve import solve_fixed_point
from pathlib import Path
import torch

CACHE = Path('/scratch/haree/talk2dino_e3_affinity_oracle/cache/full')

def propagate_scores_via_cg(raw_scores, knn_indices, knn_weights, alpha, *, propagation_steps=10, alpha_dim='class'):
    if float(alpha) == 0:
        return raw_scores
    indices64 = knn_indices.to(device=raw_scores.device, dtype=torch.int64)
    weights32 = knn_weights.to(device=raw_scores.device, dtype=torch.float32)
    s0_pc = raw_scores.float().T.contiguous()
    s_star, _ = solve_fixed_point(s0_pc, indices64, weights32, float(alpha))
    return s_star.T.contiguous()

oracle.propagate_scores = propagate_scores_via_cg
metrics = oracle.evaluate_cache(CACHE, 0.98, device='cuda', propagation_steps=320)
print(metrics['mIoU'], metrics['aAcc'], metrics['mAcc'])
"
# Expected: 29.877244374599126 48.52867057377273 54.13703466982515
# Measured wall-clock: 1258.9s (~21 min) for the full 5000-image/11,075-window val set.
```
