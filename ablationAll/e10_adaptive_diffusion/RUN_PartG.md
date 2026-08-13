# Part G — pairwise edge gate (EdgeGate), implementation and evaluation

## G7 FAILED — read this before anything else below

**The pairwise redesign moves the graph LESS than the unary design it was
built to replace, measured with the identical metric under the identical
perturbation.** Using total-variation distance between old and new
row-normalised edge distributions, on real captured features, with random
final-layer weights of magnitude 0.5 on BOTH designs (matching magnitude,
not just matching intent):

| Design | K | Perturbation | Mean TV distance |
|---|---|---|---|
| Unary (`metric.py`) | 12 | both MLP layers, magnitude 0.5 | **0.1461** |
| Pairwise (`edge_gate.py`) | 12 | both MLP layers, magnitude 0.5 | **0.0650** |
| Pairwise (`edge_gate.py`) | 32 | both MLP layers, magnitude 0.5 | **0.0800** |

The pairwise gate's graph movement is roughly **half** the unary design's
under a directly comparable scramble. **G6 (overfit-one-batch) corroborates
this independently**: EdgeGate fails the same test the unary design failed,
dropping L1 by only 2.84% over 500 dedicated steps against a required 20%
— despite gradients that are unambiguously healthier than the unary
design's ever were (`last_layer_grad` reaches 7.9e-2 here vs ~1e-4-6e-4 for
the fixed unary design; the gate's own values visibly diversify over
training, std growing from 0.0014 to 0.245). The mechanism is not blocked
— F1's original deadlock is genuinely gone (G3 confirms no residual scale
anywhere) — but the objective still does not improve. **This means the
redesign did not solve the problem it was built for.** Everything below is
reported for completeness and because G8 required both G6 and G7 to be run
regardless of outcome, but no result below overturns this finding.

One important caveat, found and corrected during this review, not swept
under the rug: the FIRST attempt at G7 (literal spec — randomising only the
final layer, leaving `mlp[0]` at PyTorch's tiny default init) showed
essentially zero graph movement (TV ~0.005) and was **confounded** —
`mlp[0]`'s default init (Kaiming-uniform, std~=0.012) gives pre-GELU hidden
activations with std~=0.023, too small for even a magnitude-0.5 final
layer to produce a strongly-varying `e_ij` (measured std only ~0.07,
dominated by the bias term). The deconfounded re-run above, scaling BOTH
layers to 0.5 (matching the unary design's own Y4b test convention
exactly), is the number that should be trusted — and it still fails to
show substantial movement, so the confound's correction did not rescue the
design; it just made the negative finding legitimate rather than an
artifact of an under-powered test.

## G1 — Discovery (reported before any code was written)

- **Graph builder**: `src/e3_affinity_oracle.py:232`,
  `build_knn_graph(patch_features, *, knn_k=12, affinity_power=3.0) ->
  (indices[int16,P,K], weights[float16,P,K], zero_row_count)`. Cosine ->
  ReLU -> `pow(affinity_power)` -> fill diagonal `-inf` -> stable descending
  argsort -> top-k gather -> row-normalise, with a self-loop fallback for
  all-zero rows.
- **Symmetrisation — the spec's assumption does not hold.**
  `GRAPH_DIRECTIONALITY = "directed_row_stochastic_knn"`
  (`e3_affinity_oracle.py:38`); a fresh repo-wide grep for `symmetri`
  returns only prior reports documenting its ABSENCE. There is no
  symmetrisation step anywhere in the existing pipeline. `EdgeGate`
  therefore implements `symmetrize(w)` as a **named no-op** — the only
  choice consistent with G4's bitwise-identity requirement against the
  actual production graph, which is directed. This is documented in
  `edge_gate.py`'s module docstring, not silently omitted.
- **kappa** applied at `cosine.clamp_min(0).pow(power)`, before top-k
  selection, in every graph-construction variant found.
- **Fixed-point solver**: `implicit_solve.py:275`,
  `implicit_propagate(s0, indices, weights, alpha, *, tol=1e-10,
  max_iter=1000, raise_on_nonconvergence=True) -> s_star`. Generic over K
  — no solver changes needed for K=32.
- **`training_step`**: `train.py:57`, unary-design-specific (calls
  `metric(features)` then `build_differentiable_knn_graph` internally) —
  not reused directly; G6 has its own self-contained training loop.
- **Candidate indices, currently**: in the unary design, computed fresh
  every forward call from `g` (not frozen `f`), inside
  `build_differentiable_knn_graph`, with `.detach()` on the selection only.
  No existing "compute once from frozen features, reuse for training"
  mechanism — that pattern only existed on the eval/cache side. New for G.

Nothing was undeterminable.

## G2 — `src/learned_affinity/edge_gate.py`

`EdgeGate(dim=768, hidden=256, kappa=3.0, K=32)`, `forward(f, cand_idx, *,
return_gate=False) -> weights[P,K]` (or `(weights, gate)` if
`return_gate=True`, needed for G6's logging). `metric.py` is untouched —
confirmed via file mtime (last modified during the earlier F1 fix, 14:10,
unchanged since) and `git diff --stat HEAD` showing no changes to it.

**Memory**, measured directly (not assumed) at production scale (P=1024,
K=32, dim=768, H100): **611.6 MB peak** for one window's forward+backward.
The 2305-dim MLP input (`f_i, f_j, f_i*f_j, cos_ij` concatenated) is NOT a
memory problem at this scale — no narrowing (e.g. dropping `f_i*f_j`) was
needed or applied.

**Detach audit**: there is no `.detach()` call anywhere in `EdgeGate.forward`
itself. `cand_idx` arrives as a plain, already-computed LongTensor with no
`grad_fn` — confirmed empirically
(`cand_idx.requires_grad == False`, `cand_idx.grad_fn is None`). The one
`.detach()` in the file is on `zero_rows` (a boolean mask for the
self-loop fallback, mirroring `build_differentiable_knn_graph`'s own
convention) — not on anything in the candidate-selection or weight path.
`build_frozen_candidate_set` runs its top-k entirely under
`torch.no_grad()`; `torch.argsort` never produces a `grad_fn` regardless,
so no explicit detach is needed there either. This confirms the spec's
"second benefit" claim structurally: there is no non-differentiable
top-k inside the gradient path anywhere in this design.

## G3 — No hidden residual scale

Grepped `edge_gate.py` for `self.r`, `self.gamma`, `self.alpha_gate`,
`nn.Parameter`: **zero matches** — `EdgeGate` has no trainable parameter of
its own outside `self.mlp`. `2*sigmoid(e_ij)` is applied directly to the
MLP's own output, not to a separately-initialised scale. At the shipped
zero-init, `e=0` gives `2*sigmoid(0)=1.0` (a constant, not a trainable
near-zero value), with local gradient `d(gate)/de|_{e=0} = 0.5` — a healthy
O(1) multiplier, not an attenuator. Directly measured at true shipped
init: `last_layer_grad = 0.0227` at step 1 (random-seed check) — large
compared to the unary design's original ~2.3e-5-scaled gradients, and
comparable to or larger than the F1-fixed unary design's ~1e-4 range.
`first_layer_grad` IS exactly `0.0` at literal step 1 — but this is the
SAME structural chain-rule fact as the unary design (any zero-final-layer
MLP has this property, independent of scale), not a residual-scale
deadlock; it clears once the final layer's healthy gradient moves it.

## G4 — Identity gate, K=12 — PASS (full 5000 images)

**Check 1** (real captured features, no propagation): candidate indices
EXACT match `build_knn_graph`'s own top-12. Weights: **not** bitwise-equal
at float16 storage precision (9/12288 entries, 0.073%, differ by one
float16 ULP) — but comparing BOTH sides at native float32, before
`build_knn_graph`'s own float16 storage cast, the max absolute difference
is **4.917e-07** (float32-epsilon agreement; the float16 mismatches are an
expected artifact of computing cosine via matmul in the reference vs
elementwise-multiply-then-sum in `EdgeGate` — two equally valid but
non-bitwise-associative float32 code paths straddling float16 rounding
boundaries at the ~5e-7 noise level, not a formula defect). G4's own
wording accepts "bitwise **or** float32 epsilon" — the float32-epsilon bar
is met cleanly.

**Check 2**, run at the **full 5000 images**, converged CG fixed point
(not an approximation): **mIoU = 29.878098700019645**, canonical =
29.877244374599126, **deviation = 0.000854** (tolerance 5e-3): **PASS**.
43m10s wall-clock.

## G5 — K=32 untrained reference, full 5000 images

**mIoU = 29.44410036953941**, aAcc = 47.47864493021151, mAcc =
51.97338126507353 — **0.433144 below** the K=12 canonical
(29.877244374599126), confirming that widening the candidate set alone,
even with a neutral gate, materially changes the graph. Any later K=32
training run must be judged against **29.444100**, not against
29.877244. 33m5s wall-clock.

## G6 — Overfit-one-batch — FAIL

Identical protocol and thresholds to the unary design's failed run: one
fixed batch (batch_size 4, seed 0), fixed mask seed, L2=L3=0.0, 500 steps,
`cand_idx` computed once (K=32) and never recomputed.

| step | l1_masked_ce | gap_vs_chance | first_layer_grad | last_layer_grad | gate_mean | gate_std |
|---|---|---|---|---|---|---|
| 0 | 3.277216 | -0.435760 | -- | -- | -- | -- |
| 25 | 3.277016 | -0.435960 | 2.37e-04 | 4.93e-03 | 0.998072 | 0.001425 |
| 100 | 3.270304 | -0.442673 | 1.69e-03 | 3.27e-02 | 0.951887 | 0.047271 |
| 175 | 3.242900 | -0.470077 | 4.27e-03 | 7.87e-02 | 0.712210 | 0.180734 |
| 250 | 3.216907 | -0.496070 | 2.97e-03 | 4.82e-02 | 0.479161 | 0.241570 |
| 350 | 3.202234 | -0.510742 | 2.22e-03 | 2.87e-02 | 0.341950 | 0.235672 |
| 425 | 3.193516 | -0.519461 | 2.26e-03 | 2.77e-02 | 0.253369 | 0.207547 |
| 500 | 3.184276 | -0.528700 | 2.28e-03 | 2.82e-02 | 0.190322 | 0.188514 |

`gate_mean` falls from 0.998 to 0.190 and `gate_std` rises from 0.001 to a
peak of 0.245 (step 275-300) — the gate is unambiguously learning,
suppressing edges substantially and non-uniformly, unlike the unary
design's `r` which barely moved. Gradients never vanish. **And still:**
step0->step500 change is **-2.84%**, against a required **-20%**. **FAIL.**
This is the same symptom as the unary design's failure, now demonstrated
under a parameterisation with no structural deadlock and no evidence of an
under-powered gradient — consistent with G7's finding that the graph
itself just isn't moving enough to matter for this objective, not that
training is blocked from moving it.

## G8 — No full training was run

Only G6 (one fixed batch, 500 steps) and G7 (direct weight-diagnostic
computation, no training) were executed. No `run_training`/`run_pilot`
equivalent for EdgeGate exists or was invoked.

## Blast radius

```
git diff --stat HEAD -- src/open_vocabulary_segmentation/ configs/   ->  empty
```
New code confined to `src/learned_affinity/edge_gate.py`,
`src/learned_affinity/edge_gate_evaluate.py`, and
`ablationAll/e10_adaptive_diffusion/scripts/edge_gate_*.py`. `metric.py`
untouched (file mtime unchanged since the earlier F1 fix).

## Recommendation

Do not proceed to train EdgeGate at scale. The expressiveness diagnostic
(G7, deconfounded, apples-to-apples with the unary design's own test) and
the overfit test (G6) agree: this parameterisation, despite fixing the
structural gradient deadlock, does not move the graph enough to make the
masked-reconstruction objective learnable, at least not with `kappa=3.0`
holding the base `ReLU(cos)^kappa` term as dominant as it is. Worth
investigating before any further redesign: what fraction of each row's
total weight is captured by the top-1 candidate under the current
`kappa=3.0` (a high `kappa` could be structurally suppressing the gate's
ability to reorder a row's dominant candidate, regardless of how much
`e_ij` varies) — this was not measured here and would be a cheap,
CPU-only, real-feature diagnostic to run before deciding whether the
pairwise idea itself is dead or whether `kappa` (unchanged per the hard
constraints here) is fighting it.

## RUN COMMANDS

```bash
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=40G --time=02:00:00

cd /project/6114407/haree/Talk2DINO
module load gcc opencv
source /scratch/haree/venv/talk2dino-a100/bin/activate

# G2 memory measurement (~5s). Expected: ~611.6 MB peak at P=1024,K=32.
python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.learned_affinity.edge_gate import EdgeGate, build_frozen_candidate_set
torch.cuda.reset_peak_memory_stats('cuda')
f = torch.nn.functional.normalize(torch.randn(1024, 768, device='cuda'), dim=-1)
cand_idx = build_frozen_candidate_set(f, K=32)
gate = EdgeGate(K=32).to('cuda')
loss = (gate(f, cand_idx) ** 2).sum()
loss.backward()
torch.cuda.synchronize()
print(f'peak: {torch.cuda.max_memory_allocated(\"cuda\")/1e6:.1f} MB')
"

# G7: expressiveness diagnostic (CPU-only, ~10s). THE decisive check.
python3 ablationAll/e10_adaptive_diffusion/scripts/edge_gate_g7_expressiveness.py
# Expected: G7b (deconfounded) mean TV at K=32 ~= 0.08, vs unary design's ~0.146
# under the identical scramble -- pairwise design moves the graph LESS.

# G4: identity gate at K=12, full 5000 images, converged CG (~43 min measured).
python3 ablationAll/e10_adaptive_diffusion/scripts/edge_gate_assert_identity.py \
  --device cuda --assert-identity
# Expected: check 1 float32-epsilon PASS (4.917e-07), check 2 mIoU=29.8781,
# deviation=0.000854 within 5e-3 -- PASS.

# G5: K=32 untrained reference, full 5000 images (~33 min measured).
python3 ablationAll/e10_adaptive_diffusion/scripts/edge_gate_g5_k32_reference.py --device cuda
# Expected: mIoU=29.444100, delta vs K=12 canonical = -0.433144.

# G6: overfit-one-batch, 500 steps (~6 min measured).
python3 ablationAll/e10_adaptive_diffusion/scripts/edge_gate_g6_overfit_one_batch.py \
  --device cuda --steps 500 --log-every 25 --batch-size 4 --seed 0
# Expected: step0=3.277216, step500=3.184276, change=-2.84% -- FAIL (requires <=-20%).

# Protected-path check.
git diff --stat HEAD -- src/open_vocabulary_segmentation/ configs/
# Expected: empty output.
```
