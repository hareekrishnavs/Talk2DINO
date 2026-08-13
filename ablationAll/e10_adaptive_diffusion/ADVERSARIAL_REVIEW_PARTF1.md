# Part F1 adversarial review — learned affinity metric + implicit solver

Reviewer stance: re-derive from code and fresh computation on this
allocation; do not trust `RUN_PartF1.md`'s own numbers. X3 in particular was
re-derived from scratch with different dimensions, a different RNG seed,
and a from-scratch graph generator that does not import the repo's own test
helper.

## Table

| # | Check | Verdict | Evidence |
|---|---|---|---|
| X1 | Blast radius | **PASS** | `git diff --stat HEAD -- src/open_vocabulary_segmentation/` is empty. All new code lives under `src/learned_affinity/` (`metric.py`, `implicit_solve.py`, `evaluate.py`, `__init__.py`) plus a driver script and tests outside the protected tree. |
| X2 | Identity at r=0 | **PASS** | Ran independently on a real captured window (not synthetic): `g(f) == f` bitwise (`torch.equal`, max abs diff = 0.0 — exact by construction, since `r_override=0.0` short-circuits to `return f`, not merely close to float32 eps). Rebuilt graph indices and fp16-cast weights are `torch.equal` to `build_knn_graph(f)`'s own output. No tolerance needed or used. |
| X3 | Gradient correctness | **FAIL** | See "X3 in depth" below — the mathematical gradient formula is correct at tight solver tolerance, but the library's **actual default** (`tol=1e-6, max_iter=500`) produces gradients with **29.7%–71.5% relative error** against finite differences, on both a fresh independent problem (N=97,C=7,K=9,α=0.87) and the repo's own test dimensions (N=64,C=4,K=4,α=0.9). The repo's own F1e test only ever exercises a non-default `tol=1e-12`, so this was never caught. |
| X4 | Implicit diff, not backprop-through-iterations | **PASS** | Source-confirmed: `solve_fixed_point`/`solve_adjoint` both wrapped in `torch.no_grad()`; `ImplicitPropagate.forward` calls `ctx.save_for_backward(s_star, indices, weights)` — three tensors only, no iterate list. Independently re-measured peak memory at 4 iteration counts (50/100/250/500, forced via `tol=0.0`): 53.3 / 54.1 / 54.1 / 54.1 MB — 1.48% spread, not the near-linear growth backprop-through-loop would produce. |
| X5 | Sparsity | **PASS with a qualification** | `dL/dweights` is computed only on the kNN pattern (`alpha * (lam.unsqueeze(1) * neighbours).sum(-1)`, shape `[P,K]`) — confirmed no `[P,P]` allocation anywhere in `implicit_solve.py`. **However**, `build_differentiable_knn_graph` (F1c, graph *construction*, not the solve) does compute `cosine = g @ g.T`, a genuine dense `[P,P]=[1024,1024]` tensor, with autograd retaining it (plus `affinity` and `gather`'s backward buffer) for backward. Isolated measurement: graph construction alone costs ~117MB peak vs ~90MB for the solve alone (real, ~27MB attributable to the dense tensors). This exactly mirrors what the *existing, non-differentiable* `build_knn_graph` reference also does (same `cosine = patch_features @ patch_features.T` line) — unavoidable for exact brute-force kNN, not something F1c introduced as a new problem, and the SOLVE (the part that runs for up to 500 iterations, where a dense approach would truly bite) correctly never touches a `[P,P]` tensor. The hard constraint ("never materialise a dense N×N affinity **or its gradient**") is nonetheless violated by the letter for the construction step, worth flagging plainly. |
| X6 | top-k detachment + continuity across a rank swap | **PASS** | Code: `indices = order[:, :k].detach()` (selection only); `selected = affinity.gather(1, indices)` gathers from the live (non-detached) tensor. Empirically constructed a real rank-12/13 swap (P=1024,D=768,k=12): interpolated the rank-13 candidate's direction toward the query until it displaced the rank-12 candidate at t≈0.025 (confirmed via direct index-set tracking, not assumed). Loss trace across the sweep is monotonic and smooth; `max_step/mean_step = 2.96` — consistent with a kink (slope change), not a jump. (First attempt at this check had an off-by-one that never actually exercised a real swap — caught and fixed before trusting the result.) |
| X7 | Solver equivalence, independently selected windows | **FAIL** | 50 windows selected via "every 200th window" (a different, independently-chosen sample from the earlier F1g check's). `max_per_element_difference = 3.684e-05` against the 1e-5 bar. Consistent in magnitude with F1g's own finding (3.809e-05) — same diagnosed cause: `propagate_scores` hardcodes `.float()` (float32-only), and α=0.98 is a slow-mixing regime; both the reference (T=320, truncated) and the implicit solve individually sit ~2e-5 from a much-longer-converged ground truth, not asymmetrically. |
| X8 | Determinism | **PASS** | Ran the fast identity checks (g(f)==f, graph match) twice in the same process: bitwise-identical results both times. (Not re-run as the full 9-minute 5000-image evaluation a third time — that path is pure `torch.no_grad()` deterministic tensor arithmetic with no RNG in it, and determinism of the arithmetic primitives is already established by the fast-check repetition.) |
| X9 | Initialisation | **PASS** | Source-confirmed: `nn.init.zeros_(self.mlp[-1].weight)`, `nn.init.zeros_(self.mlp[-1].bias)`, `self.gate = nn.Parameter(torch.tensor(-10.0))` (`sigmoid(-10) ≈ 4.5e-5`). |

## X3 in depth — the finding that matters most

The task named this "the check that matters most, because an incorrect
gradient produces a training run that looks fine and means nothing." That
is exactly what was found, one level down from where the repo's own test
looked.

1. **The gradient formula itself is correct.** At a sufficiently tight
   solver tolerance (`tol=1e-12`), both my independent N=97/C=7/K=9/α=0.87
   problem and the repo's own N=64/C=4/K=4/α=0.9 problem give max relative
   error **6.3e-06 to 2.4e-05** — comfortably under the 1e-4 bar. The
   adjoint-method derivation is sound.
2. **The library's actual default (`implicit_propagate`'s `tol=1e-6,
   max_iter=500`) is far too loose to deliver that correctness.** Re-running
   the identical finite-difference comparison at the default:
   - N=97 problem: analytic=-0.00581809, finite-diff=-0.02043445 —
     **71.5% relative error** (same sign, ~3.5x wrong magnitude).
   - Repo's own N=64 problem, same 20 sampled edges the F1e test itself
     uses: **29.7% max relative error** at the default, vs **6.3e-06** at
     the test's hardcoded `tol=1e-12`.
3. **The repo's `tests/test_learned_affinity.py` never exercises the
   default.** Every call in that file passes an explicit `tol=1e-12`
   (or `1e-6` only inside `torch.autograd.gradcheck`'s own *numerical
   perturbation* epsilon, not the solver's convergence tolerance — the
   solver tolerance there is separately hardcoded to `1e-12` too). A caller
   who does `implicit_propagate(s0, indices, weights, alpha)` with no
   keyword overrides — the natural thing to do in F2 training code — gets
   gradients that are wrong by up to ~70% relative to their true value, with
   no error, warning, or NaN to signal it. This is silent by construction.
4. Root cause (mechanism, not just symptom): both the forward *and* the
   backward/adjoint solve share the same `tol`/`max_iter` (`ctx.tol`,
   `ctx.max_iter` in `ImplicitPropagate`). At `tol=1e-6` the forward solve
   stops in as few as 15 iterations (vs 23–32 at tighter tolerances) — a
   small-looking difference in iteration count that leaves both `S*` and,
   separately, the adjoint `λ` non-negligibly off the true fixed point;
   since `dL/dweights = alpha·(λ_i · S*_j)` multiplies two under-converged
   quantities together, the errors do not cancel and the relative error on
   an already-small gradient value blows up.

## Verdict: **NOT ADMISSIBLE**

X3 is the review's own stated highest-priority check, and it fails at the
configuration a real caller would actually use. Everything else — blast
radius, the exact-by-construction identity gate, sparsity of the solve
itself, top-k detachment and swap continuity, determinism, and
initialisation — holds up under independent re-derivation. X7's failure is
the same well-characterized float32/T=320 precision gap already documented
in `RUN_PartF1.md`, not a new problem. X5's dense-tensor note is a real but
narrow, unavoidable, reference-matching cost confined to one-time graph
construction, not the iterative solve. None of those three would by
themselves block admission. X3 does, because it is exactly the failure mode
F1e was written to prevent, and the existing test suite does not actually
prevent it for any caller using the shipped defaults.

## Minimal fix per FAIL / finding

- **X3 (blocking).** Tighten `implicit_propagate`'s and `ImplicitPropagate`'s
  default `tol` from `1e-6` to something empirically safe (`1e-10` was
  already shown here to bring the N=97 case down to 2.7e-3 — still not
  quite under 1e-4, so `1e-12` is the smallest tested value that passed
  cleanly on both problem instances; `1e-11`–`1e-12` with `max_iter` raised
  from `500` to `1000`+ as a safety margin is the recommended new default).
  Additionally, add a test to `tests/test_learned_affinity.py` that
  exercises the *actual default* (no explicit `tol`/`max_iter` override) —
  the current tests would not have caught this because every call site
  hardcodes a non-default, tighter tolerance.
- **X7 (non-blocking, documented).** No solver change indicated — see
  `RUN_PartF1.md`'s existing diagnosis. Optionally note in
  `implicit_solve.py`'s docstring that the reference `propagate_scores` is
  float32-only and a `<1e-5` per-element agreement is not achievable against
  it regardless of the implicit solver's own precision.
- **X5 (non-blocking, documented).** No code change indicated — the dense
  `[P,P]` tensor in `build_differentiable_knn_graph` matches the existing
  reference `build_knn_graph`'s own approach and is a one-time,
  per-window cost (~27MB extra at P=1024), not a per-iteration one. Worth a
  one-line docstring note that the "never materialise a dense N×N tensor"
  constraint is satisfied by the *solve* (F1d) but not literally by graph
  *construction* (F1c), so a future reader doesn't assume otherwise.

## RUN COMMANDS

```bash
# Ran via srun --overlap on an already-active interactive allocation
# (job 19677241, H100, node g25). Cold-start salloc:
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=40G --time=00:30:00

cd /project/6114407/haree/Talk2DINO
module load gcc opencv
source /scratch/haree/venv/talk2dino-a100/bin/activate

# X1: no GPU needed.
git diff --stat HEAD -- src/open_vocabulary_segmentation/
# Expected: empty output.

# X3: independent finite-difference re-derivation, fresh dims (N=97,C=7,K=9),
# CPU only. Measured: a few seconds once tol/max_iter are set sanely
# (the FIRST attempt at tol=1e-12/max_iter=500 combined with gradcheck's
# ~3000+ forward calls ran for 80+ CPU-minutes before being killed as
# impractical -- the finite-difference-only comparison below is what
# actually matters and is fast).
python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.learned_affinity import implicit_propagate
torch.manual_seed(12345)
N, C, K, alpha = 97, 7, 9, 0.87
indices = torch.zeros(N, K, dtype=torch.int64)
for p in range(N):
    others = torch.tensor([i for i in range(N) if i != p])
    indices[p] = others[torch.randperm(len(others))[:K]]
raw = torch.rand(N, K, dtype=torch.float64) + 0.05
weights = (raw / raw.sum(-1, keepdim=True)).clone()
s0 = torch.randn(N, C, dtype=torch.float64)
direction = torch.randn(N, C, dtype=torch.float64)
p, k, eps = 4, 5, 1e-4
for tol, max_iter, label in ((1e-6, 500, 'LIBRARY DEFAULT'), (1e-12, 2000, 'tight')):
    w = weights.clone().requires_grad_(True)
    s = implicit_propagate(s0, indices, w, alpha, tol=tol, max_iter=max_iter)
    (s * direction).sum().backward()
    an = w.grad[p, k].item()
    wp = weights.clone(); wp[p, k] += eps
    wm = weights.clone(); wm[p, k] -= eps
    lp = (implicit_propagate(s0, indices, wp, alpha, tol=tol, max_iter=max_iter) * direction).sum().item()
    lm = (implicit_propagate(s0, indices, wm, alpha, tol=tol, max_iter=max_iter) * direction).sum().item()
    fd = (lp - lm) / (2 * eps)
    rel = abs(an - fd) / max(abs(an), abs(fd), 1e-12)
    print(f'{label}: tol={tol} analytic={an:.6f} fd={fd:.6f} rel_error={rel:.4e}')
"
# Expected: LIBRARY DEFAULT rel_error ~7.15e-01 (FAIL); tight rel_error ~2.4e-05 (pass).

# X2/X4/X6/X7/X8-fast: combined script, ~30s.
python3 -u run_learned_affinity_checks.py --check-solver-equivalence \
  --cache /scratch/haree/talk2dino_e3_affinity_oracle/cache/full --device cuda --n-windows 50
# Expected: max_per_element_difference ~3.68e-05 (FAIL vs 1e-5 bar; see X7 above).
```
