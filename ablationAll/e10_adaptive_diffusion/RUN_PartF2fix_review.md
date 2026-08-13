# Part F2fix — adversarial review (Y1–Y10)

**Overall verdict: ADMISSIBLE, with two required minimal fixes before Gate 2
is retried / Gate 3 is run.** Y4b (the critical evaluation-path check) is
confirmed PASS via direct code evidence, not just the empirical proxy the
task specified — see Y4b below for why the empirical test alone was
inconclusive and had to be corroborated. Two real, non-fatal findings
(Y4i, Y10) were uncovered that were not previously reported and should be
fixed before more GPU time is spent.

This review does not trust `RUN_PartF2fix.md`'s claims — every verdict
below was re-derived from the code and from live runs on job 19748752
(H100, node g8), and two of my own first-draft verification scripts turned
out to have bugs (a row-invariant loss in Y4g's first attempt, a
float64/float32 precision mismatch in Y7's first attempt, and an
uncalibrated threshold in Y4b's first attempt) — caught and corrected
before trusting their output, not before reporting them.

## Y1 — Blast radius

**PASS**, with a precision caveat worth stating explicitly. `git diff
--stat` against `main`'s merge-base (`18ddc21d`) DOES show changes under
`src/open_vocabulary_segmentation/models/` and `configs/` — but these come
from 7 commits (`3a39f37` "ablation results" back through `87824ce` "mmcv
issue resolved") that predate this task entirely; `git log --all -- 
src/learned_affinity/` returns nothing, confirming the learned-affinity
work has never been committed and sits entirely in the working tree. The
correct check for THIS task's blast radius is `git diff --stat HEAD --
src/open_vocabulary_segmentation/ configs/`, which is **empty** — confirmed
also via `git status --porcelain` on the same paths. New code from this
task's work is confined to `src/learned_affinity/`,
`ablationAll/e10_adaptive_diffusion/`, and new files under `tests/`; the
three pre-existing modified files (`run_e3_affinity_oracle.py`,
`src/e3_affinity_oracle.py`, `tests/test_e3_affinity_oracle.py`) were
already modified before this task began (present in the session's very
first `git status` snapshot).

## Y2 — The fix, as written

**PASS.** Read `metric.py` directly and constructed a fresh module live:

```
r = 0.10000000149011612  requires_grad = True  is_leaf = True
mlp[-1].weight sum: 0.0   mlp[-1].bias sum: 0.0
has 'gate' attr: False
```

`r` is `0.1` to float32 precision (the value differs from mathematical 0.1
by ~1.5e-9 — the nearest representable float32 value, not a defect). No
`gate`/sigmoid anywhere. Clamp is `torch.clamp(self.r, 0.0, self.r_max)` —
isolated verification: `d(clamp)/dr = 1.0` at `r=0.1` (inside range), `=0.0`
at `r=0.9` and `r=-0.3` (outside range) — a genuine differentiable
subgradient, not a detach (a detach would give `grad=None`, not `0.0`).
MLP final layer zero-init confirmed by direct weight/bias sums.

## Y3 — Identity at initialisation

**PASS**, on real captured DINOv2 features (window 0 of image 0 from
`feature_capture_val_full`, not synthetic data). `g = metric(f, 
r_override=0.0)`: `max|g-f| = 0.0` exactly (short-circuit returns `f`
directly — `g is f_real` is `True`). kNN graph built from `g` vs. built
directly from `f` via the production `build_knn_graph`: indices EXACT
match, weights EXACT match at float16 storage precision. No tolerance to
hide behind, and none needed.

## Y4 — Wiring audit

**Y4a — PASS.** Same batch, same seed, real image: loss at r=0.3 (randomised
final layer) = 3.331566; loss at r=0.0 = 3.317748; difference = 0.013819,
well above noise. Training loss depends on `g`.

**Y4b — PASS (the critical one). Full reasoning, since the first empirical
attempt was inconclusive and needed a second pass to resolve correctly:**

1. First attempt: scrambled metric (r=0.5, random weights magnitude 0.5) on
   500 real images gave mIoU=29.1106 vs. canonical 29.8772 (deviation
   0.767) — below my own ad hoc "1.0 point" threshold, provisionally FAIL.
2. Before trusting that, I built a control: the pure IDENTITY metric (r=0,
   no scrambling) on the SAME 500 images gave mIoU=29.2242 — already 0.653
   points off the full-5000 canonical from subset-sampling noise ALONE. The
   scrambled metric's deviation from this identity-on-same-subset baseline
   was only 0.114 — just 0.17x the noise floor. A blind threshold would
   have called this inconclusive-to-failing.
3. Rather than accept that, I checked whether the scrambling was actually
   perturbing the graph at the patch level, independent of the noisy mIoU
   aggregate: on the same real features, `cosine(g,f)` averaged 0.04 (g is
   essentially uncorrelated with f in direction — the scramble is real),
   yet kNN neighbour-set overlap with the identity graph stayed at 87% on
   average. This is a Johnson–Lindenstrauss-type effect: the SAME random
   (though largely linear-then-GELU) transformation applied to all 1024
   correlated patches tends to approximately preserve *relative* neighbour
   structure even while randomising each patch's *absolute* direction — so
   this particular scrambling method was not adversarial enough to move
   mIoU by a large margin, independent of whether the evaluation path is
   wired correctly.
4. This meant the empirical mIoU test alone could not distinguish "wired
   correctly, weak scramble" from "wired incorrectly." I settled it by
   reading `evaluate.py`'s actual evaluation loop (`evaluate_with_learned_
   metric`, lines 90–95):
   ```python
   g = metric(features32, r_override=r_override)
   indices, weights = build_differentiable_knn_graph(g, k=metric.k, kappa=metric.kappa)
   row = dict(cache_row)
   row["knn_indices"] = indices.to(torch.int16).cpu()
   row["knn_weights"] = weights.detach().to(torch.float16).cpu()
   ```
   For every window of every evaluated image, `g` is computed from the
   passed-in metric and a FRESH graph is built from it, explicitly
   overwriting `row["knn_indices"]`/`row["knn_weights"]` (which start as a
   copy of the cache's original row) before being handed to
   `replay_cached_image`. There is no code path here that could silently
   fall back to the cache's original graph. Combined with the graph
   diagnostic (87% overlap, not 100% — the rebuilt graph DOES differ, just
   not enough to move mIoU past a naive threshold under this specific
   scrambling method) and `evaluate_full_val_converged` /
   `evaluate_with_learned_metric` being the single evaluation code path
   used by both the pilot's per-checkpoint eval and `final_eval.py`'s E1
   (no second, differently-wired path exists), this is decisive: **the
   evaluation path uses `g`.**

Lesson for future scrambling-based tests here: use a scrambling method that
also disrupts *relative* patch geometry (e.g. per-patch independent noise,
not a shared MLP), not just absolute magnitude, if a cleaner large-margin
mIoU signal is wanted.

**Y4c — PASS.** Fresh module's 5 parameters (`r`, `mlp.0.weight`,
`mlp.0.bias`, `mlp.2.weight`, `mlp.2.bias`) all present with
`requires_grad=True` before optimizer construction; optimizer's own
`param_groups` hold exactly those 5 tensors by identity
(`opt_param_ids == model_param_ids`).

**Y4d — PASS.** Saved a module with `r=0.37` and non-zero (including a
deliberately non-zero final layer, breaking the usual zero-init) MLP
weights; loaded into a FRESH module. Every `state_dict` key bitwise equal
(`torch.equal`, not `allclose`). `load_state_dict` called with no
`strict=` argument (i.e. `strict=True`, the default) — confirmed via
`inspect.getsource`. Loaded `r=0.37`, not the fresh module's own init value.

**Y4e — PASS.** `CACHE` is a single module-level constant in `pilot.py`;
`final_eval.py` imports it (`from .pilot import CACHE as _CACHE, 
CAPTURE_DIR as _CAPTURE_DIR`) rather than redefining it — no possibility of
drift by construction. `--checkpoint` is a REQUIRED CLI arg with no
default, so there is no stale hardcoded path to silently fall back to.
`pilot_run_v1` and `pilot_run_v2` both happen to have a `checkpoint_000300.
pt` file (different directories) — checked this isn't just an
absolute-path coincidence: `pilot_run_v2/checkpoint_000300.pt`'s saved `r`
(2.2886e-05) bitwise-matches `pilot_run_v2/pilot_summary.json`'s own
logged `r` at step 300, and `pilot_run_v1`'s same-named file has a
DIFFERENT `r` (2.2874e-05) — genuinely separate runs, self-consistent
content, no crossover.

**Y4f — PASS.** Grepped every `requires_grad_(False)`, `.eval()`,
`.detach()`, `torch.no_grad()` in `src/learned_affinity/`. Every site is
one of: (a) the frozen DINOv2/CLIP forward (`extract.py`'s
`@torch.no_grad()`-decorated `generate_masks` call, `train.py`'s
`model.eval()`) — matches the task's own expectation; (b) evaluation-only
code (`evaluate.py`'s `with torch.no_grad():` around the whole eval loop,
`pilot.py`'s `evaluate_held_out_caption_losses`) — never backprops, so this
is correct; (c) detaching individual scalar values for LOGGING only
(`losses.py`/`train.py`'s `.detach()` on returned component values — the
actual backprop target `total` is never detached); or (d) internal to
`ImplicitPropagate(torch.autograd.Function)`'s `forward`/`backward` — a
standard implicit-function-theorem custom autograd Function, where
detaching inputs INSIDE `forward` is required/correct (the class's own
`backward` computes the true gradient via the adjoint CGLS solve, not by
differentiating through the forward iteration) and does NOT block the
outer gradient path — confirmed empirically by Gate 1's real nonzero `r`
and MLP gradients, which flow through exactly this code.
`build_differentiable_knn_graph`'s three `.detach()` calls (`metric.py`)
are on the SELECTION only (which indices to keep); the returned `weights`
are gathered from the live, non-detached `affinity` tensor.

**Y4g — PASS**, after fixing my own first test. First attempt used
`weights.sum()` as the probe loss — this is a mathematically CONSTANT
quantity (≈ num_patches) because `build_differentiable_knn_graph`'s weights
are row-normalised to sum to 1 per row by construction, so its gradient was
~1e-16 (float64 noise floor) regardless of correctness — my bug, not the
code's. Redone with `(weights**2).sum()` (sensitive to weight
DISTRIBUTION, not just row sums): base gradient norm 3.14 (a real signal),
and perturbing one input feature by eps=0.01/0.3/1.0 changed the gradient
w.r.t. metric params by a relative 0.3%/7.1%/28.5% — clearly above any
noise floor and scaling with the perturbation size, as expected for a
non-detached, genuinely differentiable path.

**Y4h — PASS.** `training_step`'s body, in order: `optimizer.zero_grad()`
→ per-sample forward+loss+`.backward()` (accumulated over the batch) →
`clip_grad_norm_` → `optimizer.step()`. Grepped the whole file for
`try:`/`except`: zero matches — no exception-swallowing anywhere near the
step.

**Y4i — FLAGGED (real finding, not previously reported).** `torch.optim.
AdamW`'s default `weight_decay=0.01` (verified via `inspect.signature`);
all four optimizer-construction sites (`train.py`, `pilot.py`,
`grad_flow_check.py`, `gate2_short_schedule.py`) call `AdamW(metric.
parameters(), lr=...)` with no override — `r` is NOT excluded from decay.
Confirmed empirically, not just theoretically: at step 1, `r`'s own
gradient is structurally exactly zero (documented, expected — `dg/dr =
MLP(f) = 0` at init), so any movement at step 1 can only come from decay.
Predicted decay-only value: `0.1 * (1 - 1e-4*0.01) = 0.0999999`. Gate 1's
actual measured step-1 `r`: `0.0999999` (bitwise match to the printed
precision). This does not indicate a bug — the LATER steps' movement is
gradient-dominated (decay pulls toward 0; Gate 1/2 both show `r` moving
UP, away from 0, meaning the real gradient signal exceeds decay) — but it
is an unnecessary, easily-removed confound for a longer Gate 3 run.
**Minimal fix:** construct the optimizer with `weight_decay=0.0`, or move
`r` into a separate param group with `weight_decay=0.0` if decay on the
MLP weights is wanted.

## Y5 — Gradient magnitudes after the fix

**PASS.** Ran `grad_flow_check.py` live, 20 steps (job 19748752):

| step | r | r_grad | first_layer_grad | last_layer_grad |
|---|---|---|---|---|
| 1 | 9.999990e-02 | 0.000000e+00 | 0.000000e+00 | 1.737335e-04 |
| 2 | 1.000741e-01 | -9.194840e-06 | 9.515090e-07 | 1.106631e-04 |
| 3 | 1.001512e-01 | -3.223160e-05 | 2.732629e-06 | 1.727067e-04 |
| 20 | 1.017544e-01 | -4.304971e-05 | 5.100493e-06 | 5.880509e-05 |

`last_layer_grad` at step 1 (1.74e-04) > 1e-6. `first_layer_grad` exceeds
1e-6 at step 3 (within 20). `r_grad` is exactly 0.0 at step 1 (structural,
expected) and non-zero from step 2 onward — exactly when `first_layer_grad`
also first becomes non-zero, i.e. once `MLP(f)` itself becomes non-zero.
All three required conditions hold.

## Y6 — The check's verdict must be computed, not written

**PASS.** Read `grad_flow_check.py`: `passed = last_layer_nonzero_at_step1 
and first_layer_nonzero_step is not None`, both derived from the same
`rows` list the printed table is built from; `sys.exit(0)`/`sys.exit(1)`
branch directly on `passed`. Then forced an actual failure — monkeypatched
`LearnedMetric` so it constructs with `r=0.0` (reproducing the pre-F1
deadlock) and ran the REAL script's REAL `main()` under that condition:

```
step 1: r=0.0, r_grad=0.0, first_layer_grad=0.0, last_layer_grad=0.0
...
last_layer_grad at step 1: 0.000000e+00 (<= 1e-06: FAIL)
first_layer_grad never exceeded 1e-06 in 10 steps: FAIL
FAIL: last_layer_nonzero_at_step1=False, first_layer_nonzero_step=None.
```
Script exited with code 1. The specific old bug (printing PASS while the
table showed all zeros) is confirmed gone: this run's table and verdict
agree.

## Y7 — L1 reference line

**PASS**, after fixing my own first test (same precision-mismatch class of
error as Y4g — mixing a Python-float64 recomputation against a float32
tensor result gave a false "FAIL" the first time; redone replicating the
source's exact `tensor - python_float` operation gives a bitwise match).
Independently constructed a batch with a KNOWN `C=17`: `raw_scores.shape[0]
== 17 == result["num_classes"]`; `math.log(17)` exactly equals
`result["chance_ce"]` (both are plain Python floats via the same formula,
so this is an exact equality, not an approximation); `result["l1_masked_ce"]
- result["chance_ce"]` (replicating the source's own float32 subtraction
order) is `torch.equal` to `result["l1_masked_ce_gap_vs_chance"]`.

## Y8 — Checkpoint selection logic

**PASS.** Synthetic `pilot_log` where every one of 5 components is
negatively correlated with mIoU (Pearson -1.0 across the board):
`selection["no_positively_correlated_signal"] == True`,
`selected_checkpoint_step == 500` (the LAST checkpoint), `selection_signal
is None`, and the finding string states the fallback explicitly ("Falling
back to the LAST checkpoint trained (step 500), stated explicitly, not
chosen via any signal"). Sanity check with one positively-correlated
component (`l1_masked_ce`, Pearson +1.0) correctly selects on it instead.
`compute_correlations`'s component list structurally never includes
`full_val_mIoU` — validation mIoU cannot be a selection candidate by
construction, not just by convention.

## Y9 — Gate discipline

**PASS.** No `gate3`-named output directory exists anywhere under
`/scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/`.
`RUN_PartF2fix.md` explicitly and repeatedly states Gate 3 was not run.
`sacct` history for this session shows no separate multi-hour job or step
matching a Gate 3 launch — only the user's own interactive allocations
(themselves multi-hour, but that is allocation time, not a training run;
every individual `srun` step run against them this session measured in
single-digit minutes).

## Y10 — Determinism

**FAIL — confirmed root cause, not just a symptom.** Ran `grad_flow_check.
py --seed 0` twice in immediate succession on the same allocation. Tables
are qualitatively similar (same PASS verdict, same order of magnitude) but
NOT identical:

| step | run 1 last_layer_grad | run 2 last_layer_grad |
|---|---|---|
| 1 | 1.737335e-04 | 1.612791e-04 |
| 7 | 2.130522e-04 | 2.324530e-04 |

Root cause: `src/learned_affinity/crop_dataset.py:76`,
`CocoCaptionCropDataset.__getitem__`: `rng = random.Random()` — a genuinely
UNSEEDED RNG constructed fresh on every call. `self.seed` (stored at
`__init__`) is never referenced anywhere in the class; it has no effect on
per-item crop/flip augmentation. This is a deliberate design choice
(comment: "must vary across epochs, not be a fixed, memorised choice per
image") that is reasonable for real training but directly defeats
`--seed`'s reproducibility for the gate/diagnostic scripts. **Minimal fix:**
give `CocoCaptionCropDataset` an optional deterministic mode (e.g. an
`augmentation_seed: int | None = None` constructor parameter that, when
set, seeds a per-item RNG from `(augmentation_seed, index)` instead of
`random.Random()`), defaulting to the current unseeded behaviour so real
training's epoch-to-epoch diversity is unaffected; have `grad_flow_check.
py` and `gate2_short_schedule.py` pass their `--seed` through to it. Not
implemented here — this touches a shared file used by real training too,
and is reported rather than applied unilaterally.

## Summary table

| Check | Verdict | Evidence basis |
|---|---|---|
| Y1 | PASS | `git diff --stat HEAD` empty on protected paths; branch-history diff traced to 7 pre-existing commits |
| Y2 | PASS | live construction: r=0.1 (float32), no gate, clamp verified differentiable, MLP zero-init |
| Y3 | PASS | real features: exact identity, exact kNN graph match |
| Y4a | PASS | real batch: loss differs 3.3316 vs 3.3177 |
| Y4b | PASS | direct code evidence (evaluate.py:90-95) + graph-level diagnostic; naive mIoU threshold was confounded by scrambling method |
| Y4c | PASS | live: 5/5 params registered, requires_grad=True |
| Y4d | PASS | bitwise round-trip, strict=True, r=0.37 preserved |
| Y4e | PASS | single-source-of-truth paths; content cross-check rules out v1/v2 crossover |
| Y4f | PASS | every no_grad/detach site accounted for and justified |
| Y4g | PASS (after fixing own test) | corrected probe: gradient sensitivity confirmed above noise floor |
| Y4h | PASS | source order confirmed; zero try/except |
| Y4i | FLAGGED | AdamW default weight_decay=0.01 applies to r; confirmed via exact match to Gate 1's step-1 value |
| Y5 | PASS | live 20-step run, all three conditions hold |
| Y6 | PASS | verdict traced to source; forced-failure run confirms FAIL+exit(1) |
| Y7 | PASS (after fixing own test) | C, ln(C), gap independently bitwise-verified |
| Y8 | PASS | synthetic all-negative and positive-signal cases both correct |
| Y9 | PASS | no gate3 artifacts, no matching job history |
| Y10 | FAIL | two same-seed runs differ; root cause isolated to crop_dataset.py:76 |

**Verdict: ADMISSIBLE.** The fix itself (F1) is correct and the training
and evaluation paths are both genuinely wired through `g` — Y4b, the check
this review was told mattered most, passes on direct code evidence, not
just an inconclusive empirical proxy. Fix Y4i (weight_decay=0.0 for the
optimizer, one-line change) before retrying Gate 2. Y10's fix is
lower-priority (it doesn't affect correctness, only bit-reproducibility of
the diagnostic) but should be done before this gate infrastructure is
relied on for a "did anything change" comparison across runs.

## RUN COMMANDS

```bash
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=40G --time=01:00:00

cd /project/6114407/haree/Talk2DINO
module load gcc opencv
source /scratch/haree/venv/talk2dino-a100/bin/activate

# Y2: print r from a fresh module (~2s).
python3 -c "
import sys; sys.path.insert(0, '.')
from src.learned_affinity.metric import LearnedMetric
m = LearnedMetric()
print('r =', float(m.r.detach()), 'requires_grad =', m.r.requires_grad)
print('has gate attr:', hasattr(m, 'gate'))
"

# Y5/Y10: repaired gradient-flow check, run twice same seed (~1min each).
# Expect PASS both times; tables will be CLOSE but not identical (Y10 finding).
python3 ablationAll/e10_adaptive_diffusion/scripts/grad_flow_check.py --device cuda --steps 20 --seed 0
python3 ablationAll/e10_adaptive_diffusion/scripts/grad_flow_check.py --device cuda --steps 20 --seed 0

# Full test suite (~136s).
python3 -m pytest tests/ -q
# Expected: 100 passed.

# Protected-path check.
git diff --stat HEAD -- src/open_vocabulary_segmentation/ configs/
# Expected: empty output.
```
