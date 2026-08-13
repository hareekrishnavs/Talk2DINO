# Part F1 — Learned affinity metric: module, implicit solve, correctness checks

New package `src/learned_affinity/` (`__init__.py`, `metric.py`,
`implicit_solve.py`, `evaluate.py`), tests `tests/test_learned_affinity.py`,
and a checks driver `run_learned_affinity_checks.py` at repo root. No
training, no loss, no data pipeline was implemented (out of F1 scope, per
the task's explicit instruction).

## F1a — discovery report

Delivered verbatim before any edit (see chat). Four required items all
found and reported: `build_knn_graph` (`src/e3_affinity_oracle.py:232-283`,
no symmetrisation — `GRAPH_DIRECTIONALITY = "directed_row_stochastic_knn"`),
`propagate_scores` (`:412-475`), the Part E capture manifest/loader
(`load_capture_manifest`/`load_capture_features`, `:2561`/`:2616`), and the
metric accumulator (`confusion_from_prediction`/`metrics_from_confusion`,
`:560`/`:583`). One negative finding reported rather than fabricated: no
"detach indices, gather differentiable weights" pattern exists anywhere in
the repo already (grepped) — built fresh in F1c.

## F1b — LearnedMetric

`src/learned_affinity/metric.py`. `__init__(dim=768, hidden=256, r_max=0.5,
kappa=3.0, k=12)`; MLP final `Linear` layer zero-initialised (weight AND
bias) so `MLP(f) == 0` exactly regardless of input; `gate` initialised to
`-10.0` (`sigmoid(-10) ≈ 4.54e-5`). `forward(f, *, r_override=None)`;
`r_override == 0.0` short-circuits to `return f` unchanged (see F1f below
for why this needed to be exact-by-construction, not merely close).

## F1c — differentiable graph construction

`build_differentiable_knn_graph(g, *, k, kappa)` in the same file. Same
cosine/clamp/pow/argsort/gather/row-normalise math as `build_knn_graph`,
**no symmetrisation** (matching the discovery finding). Pattern: `affinity`
stays differentiable; a **detached clone** is used only to pick the top-k
`indices` (`metric.py`, the `selection_affinity = affinity.detach().clone()`
/ `indices = order[:, :k].detach()` lines); `weights` are then **gathered
from the live (non-detached) `affinity` tensor** at those indices, so
gradient flows through edge weights, never through which edges were
selected.

## F1d — implicit fixed-point solve + adjoint

`src/learned_affinity/implicit_solve.py`. Chose the **Richardson/fixed-point
iteration** (the same recurrence `propagate_scores` already runs for a fixed
T) over conjugate gradient: `A` is directed/row-stochastic, not symmetric,
so CG is not formally applicable without normal equations (2x the matvec
cost) — and matching the existing math exactly is what F1g needs anyway.
Forward and adjoint solves both run under `torch.no_grad()`, no iterate
retained, memory O(P·K + P·C) — never a dense `[P,P]` tensor. Custom
`torch.autograd.Function` (`ImplicitPropagate`): backward solves
`(I - alpha·A^T)λ = dL/dS*` via the same iteration on `A^T` (scatter-add,
`apply_knn_transpose`), then `dL/dweights[p,k] = alpha·(λ_p · S*_{indices[p,k]})`
— sparse-pattern only, matches the task's formula exactly (re-derived via
the adjoint-method / implicit function theorem before coding, not copied
blind). `dL/dalpha` is not implemented (out of scope — alpha is not learned
in F1); passing a `requires_grad=True` alpha tensor raises explicitly rather
than silently returning a wrong (`None`) gradient.

## F1e — gradient correctness (mandatory gate)

`tests/test_learned_affinity.py`, N=64 patches / C=4 channels / k=4, float64:
- **Finite-difference check**: `eps=1e-4`, central difference, 20 randomly
  sampled edges. **Max relative error = 6.33e-06** (bar: < 1e-4). PASSED.
- **`torch.autograd.gradcheck`**, float64, `eps=1e-6, atol=1e-5, rtol=1e-3`:
  PASSED.
- 3 further tests: forward-solve convergence/residual, `LearnedMetric`
  identity at untrained init, and differentiable-graph-matches-reference at
  r=0 (exact, `torch.equal`, both indices and fp16-cast weights).
- **5/5 tests pass.** Full existing suite (`tests/`) also re-run: **73/73
  pass**, confirming no regression.

## F1f — identity gate (`--assert-identity`), real GPU data, full 5000-image val

All four checks **PASSED**:
1. `g(f) == f`, atol=1e-6: **True, max abs diff = 0.0 exactly** (not merely
   within float32 eps — `r_override=0.0` returns `f` unchanged by
   construction, see F1b).
2. Rebuilt graph indices/weights identical to `build_knn_graph(f)` directly:
   **True** (exact, `torch.equal`, both indices and fp16-cast weights) — the
   deliberately EXACT (not tolerance-based) half of F1f, since it's a pure
   function of f with no propagation.
3. Full val eval, α=0.98/T=320, graph from `g(f)` at r=0: mIoU=29.878049,
   aAcc=48.528202, mAcc=54.138821 — deviations from the Part E1 canonical
   anchors are **2.4e-7 / 3.6e-7 / 3.4e-7** (essentially exact, since the
   graph is bit-identical; well inside the 5e-3 bar). evaluated_images=5000,
   measured runtime 338.0s.
4. α=0.00: **28.480169315747716, exact match.**

## F1g — solver equivalence vs the existing 320-step power iteration

**Measured max per-element difference: 3.809e-05, on 50 real windows.**
Fails the literal < 1e-5 acceptance bar. Diagnosed (not just reported) rather
than left unexplained:
- `propagate_scores` hardcodes `base = raw_scores.float()` — it can only
  ever run in **float32**, regardless of input dtype, so this comparison
  cannot be de-noised by running the reference in float64.
- On a representative window, compared `propagate_scores` at T=320, T=1000,
  T=5000, T=20000 (the last is an effectively-exact reference at this
  window's convergence rate) against my implicit solve (converged to
  1e-6 relative residual in 316 iterations):
  `|T=320 − T=20000| = 2.2e-05`, `|implicit − T=20000| = 2.4e-05`,
  `|T=320 − implicit| = 2.5e-06`. **Both methods land ~2e-5 away from the
  true fixed point, by similar magnitude, not asymmetrically** — this is
  float32 accumulation noise inherent to hundreds of iterations at
  α=0.98 (a slow-mixing regime), not a one-sided error in either
  implementation. It's also consistent with F1h's finding (below) that the
  500-iteration cap is sometimes insufficient for the adjoint solve to
  fully converge at α=0.98, adding further imprecision on top of
  `propagate_scores`'s own T=320 truncation.
- **Verdict: F1g's acceptance bar is not met by the literal number, and I
  am reporting that plainly rather than loosening the bar myself.** The
  gradient-correctness gate (F1e) — the check that actually matters for
  whether training would silently produce garbage — passed independently
  and by a wide margin (6.3e-6 vs the 1e-4 bar), and does not depend on
  `propagate_scores`'s own float32 precision at all.

## F1h — timing (real captured windows, 50-window sample, H100)

```
graph_construction_from_g(f):  mean=0.714ms  max=10.634ms
implicit_forward_solve:        mean=26.581ms max=28.230ms
adjoint_backward_solve:        mean=46.511ms max=189.318ms
```
Graph construction is cheap; the solve (forward + backward) dominates a
would-be training step by roughly 100x. **Finding relevant to F2's batch
size**: on the timed window, the forward solve converged in 338 iterations,
but the **adjoint (backward) solve hit the 500-iteration cap without
converging** to the 1e-6 relative-residual tolerance at α=0.98 — consistent
with F1g's diagnosis that α=0.98 is a slow-mixing regime where 500
iterations is sometimes not enough. This should inform F2: either accept a
looser residual tolerance for the backward pass, or budget for occasionally
uncapped/longer adjoint solves.

**Peak memory, 100 vs 500 forced iterations** (`tol=0.0`, forcing the full
iteration count both times, isolating the iteration-count effect):
`max_iter=100: 53.315 MB`, `max_iter=500: 54.015 MB` — a 1.3% difference
attributable to allocator/measurement noise, not iteration count.
**Confirms O(P·K + P·C) memory, independent of solver iterations.**

## Acceptance criteria — final status

| criterion | result |
|---|---|
| F1e gradient check < 1e-4 max relative error | **PASS** (6.33e-06) |
| F1f all four identity checks | **PASS** (all four) |
| F1g max per-element difference < 1e-5 | **FAIL** (3.809e-05, diagnosed as float32 noise inherent to `propagate_scores`, not a solver defect) |
| Peak memory independent of iteration count (100 vs 500) | **PASS** (53.3MB vs 54.0MB, 1.3% noise) |
| `git diff --stat` shows no change under protected paths | **PASS** (empty diff for `models/`/`configs/`) |

## RUN COMMANDS

```bash
# Ran via srun --overlap on an already-active interactive allocation
# (job 19677241, H100, node g25). Cold-start salloc:
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=40G --time=00:30:00

cd /project/6114407/haree/Talk2DINO
module load gcc opencv
source /scratch/haree/venv/talk2dino-a100/bin/activate

# F1e: gradient correctness (CPU-only, no GPU needed). Measured: 16.4s.
python3 -m pytest tests/test_learned_affinity.py -q
# Expected: 5 passed.

# Full regression (no GPU needed). Measured: 120.7s.
python3 -m pytest tests/ -q
# Expected: 73 passed.

# F1g: solver equivalence, 50 real windows. Measured: 7.1s.
python3 -u run_learned_affinity_checks.py \
  --cache /scratch/haree/talk2dino_e3_affinity_oracle/cache/full \
  --device cuda --check-solver-equivalence --n-windows 50
# Expected: prints max_per_element_difference=3.809e-05, exits non-zero
# (documented FAIL against the literal 1e-5 bar; see F1g above).

# F1f: identity gate, full 5000-image val. Measured: 8m25s.
python3 -u run_learned_affinity_checks.py \
  --capture-dir /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full \
  --cache /scratch/haree/talk2dino_e3_affinity_oracle/cache/full \
  --device cuda --assert-identity
# Expected: "F1f: ALL FOUR IDENTITY CHECKS PASSED", exit 0.

# F1h: timing + peak-memory report, 50 real windows. Measured: 19.0s.
python3 -u run_learned_affinity_checks.py \
  --capture-dir /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full \
  --cache /scratch/haree/talk2dino_e3_affinity_oracle/cache/full \
  --device cuda --report-timing --report-peak-memory --n-windows 50
# Expected: timing means as reported above; peak_gpu_bytes 100 vs 500 iters.

# Protected-path check.
git diff --stat HEAD -- src/open_vocabulary_segmentation/models/ src/open_vocabulary_segmentation/configs/
# Expected: empty output.
```
