# Part F2fix — LearnedMetric initialisation deadlock, fix and staged re-validation

**Status: Gate 1 PASSED. Gate 2 FAILED. Gate 3 NOT RUN, per the explicit
instruction to stop at the first gate failure.** F1-F5 are implemented,
unit-tested (100/100 passing), and Gate 1/2 were run for real on GPU. Gate
2's failure is a genuine, reproducible finding (confirmed via a
window-averaged trend check, not step-to-step noise) — see "Gate 2" below
for the data and what it implies. Do not run Gate 3 until this is
understood; see "Recommendation" at the end.

## Diagnosis (confirmed against the actual pilot_run_v2 artifacts)

`pilot_run_v2/pilot_summary.json` and `final_eval_v2.json` (the run this
task's diagnosis is based on) were re-read to confirm the numbers before
touching any code:

- `r` moved from 2.2886e-05 (step 300) to 2.6369e-05 (step 2400) — unchanged
  to two significant figures across 2100 real training steps.
- `full_val_mIoU` was flat at ~25.452 across **all ten** checkpoints
  (25.45216731409996 ... 25.45223480867973 — fourth-decimal noise only).
- E2's per-class deltas at the selected checkpoint were uniformly small
  (banana -0.0169, zebra -0.0025, stairs +0.0127, ... all |delta| < 0.02) —
  exactly what a model that has not moved from its near-identity init
  produces, not evidence any mechanism was exercised.
- The prior run's V3 selection picked `l3_anchor` at Pearson r=-0.2381 — a
  negative correlation, which F5 below now explicitly excludes from
  selection.

This confirms the root cause: `r = r_max * sigmoid(gate)` with `gate`
init'd to -10.0 put `r` at ~2.3e-5, multiplying against the MLP's
zero-initialised final layer (`MLP(f) = 0` at init) to produce a genuine
gradient deadlock — `dL/dr ∝ MLP(f) = 0` and `dL/d(final layer) ∝ r ≈
2.3e-5 ≈ 0`, so essentially nothing moved for 2100+ steps.

## F1 — LearnedMetric initialisation fix

`r` is now a plain `nn.Parameter` initialised to 0.1 (not gated through a
sigmoid), clamped to `[0, r_max]` inside `forward` with a gradient-preserving
`torch.clamp` (not detached/hard-thresholded). The MLP's final layer stays
zero-initialised, so `g(f) = f` exactly at init regardless of `r`'s value —
the identity guarantee is unaffected. `r_max` (0.5), `kappa` (3.0), `k`
(12), `dim` (768), `hidden` (256) all keep their existing defaults;
`build_differentiable_knn_graph` is byte-for-byte unchanged (confirmed via
diff against the pre-fix file).

Exact diff (`src/learned_affinity/metric.py`):

```diff
@@ class LearnedMetric.__init__
-        self.gate = nn.Parameter(torch.tensor(-10.0))
-
-    @property
-    def r(self) -> torch.Tensor:
-        return self.r_max * torch.sigmoid(self.gate)
+        self.r = nn.Parameter(torch.tensor(0.1))

@@ class LearnedMetric.forward
         residual = self.mlp(f)
-        r_value = self.r if r_override is None else r_override
+        # torch.clamp preserves gradient inside [0, r_max] (subgradient 0
+        # outside it) -- deliberately not detached/hard-thresholded, so r
+        # itself remains trainable through this clamp (F1).
+        r_value = torch.clamp(self.r, 0.0, self.r_max) if r_override is None else r_override
```

(Docstrings were also updated throughout to describe the new
parametrisation and the deadlock it replaces; no other line of executable
code in the file changed. `build_differentiable_knn_graph` untouched.)

## F3 — grad_flow_check.py rewritten

Verdict is now computed directly from measured thresholds
(`first_layer_grad > 1e-6` within N steps AND `last_layer_grad > 1e-6` at
step 1), printed with `sys.exit(1)` on failure, all gradients in scientific
notation, `r` and its gradient printed every step. The previous version's
bug (printing "PASS" while its own table showed `first_layer_grad =
0.0000000000` at every step) is gone — the verdict can no longer disagree
with the table because it is derived from the same values the table prints.

## F4 — ln(C) reference line

`compute_total_loss` (`losses.py`) now returns `num_classes` (`C =
raw_scores.shape[0]`), `chance_ce` (`ln(C)`), and
`l1_masked_ce_gap_vs_chance` (`l1_masked_ce - ln(C)`) alongside the existing
components. Threaded through `training_step`'s per-step log, the pilot's
held-out-loss aggregation, and the pilot's per-checkpoint print line. A
value at or above zero means reconstruction is no better than chance,
exactly as specified.

## F5 — checkpoint selection fixed

`select_checkpoint` (`pilot.py`) now requires **strictly positive** Pearson
correlation with mIoU (previously selected on the most *negative*
correlation, which is what let the prior run select `l3_anchor` at
Pearson -0.2381 — a negative correlation). If no component has positive
correlation, no checkpoint is selected via a signal; the return now
includes `no_positively_correlated_signal: true`, all measured
correlations, and falls back explicitly to the **last** checkpoint trained,
stated as a fallback, not a validated choice.

## Unit tests

All gate-dependent tests updated (`metric.gate` → `metric.r` throughout;
`r=0.3` now set directly rather than via sigmoid inversion). The old
`test_gradient_at_true_untrained_init_needs_larger_fd_eps` — which
documented the pre-fix bug's signature (a real but vanishingly tiny
gradient needing a larger FD `eps` to resolve) — is now obsolete under the
new init and was rewritten as
`test_gradient_at_true_untrained_init_final_layer_resolves_at_default_eps`:
it confirms the two REMAINING structural exact-zero gradients (first MLP
layer and `r` itself, both still gated by the zero-initialised final layer,
unaffected by F1) but now asserts the final layer's gradient resolves at
the **default** FD eps (it no longer needs a larger one) — a direct
regression test that the fix actually worked. **Full suite: 100/100
passing.**

## Gate 1 — PASSED

20 real training steps, fresh F1 init, real COCO images/captions, real
model, real CG solves. Measured (H100, job 19748752):

| step | r | first_layer_grad | last_layer_grad |
|---|---|---|---|
| 1 | 9.999990e-02 | 0.000000e+00 | 1.352108e-04 |
| 2 | 1.000740e-01 | 6.758331e-07 | 9.827336e-05 |
| 3 | 1.001462e-01 | 3.266993e-06 | 2.224313e-04 |
| 20 | 1.017631e-01 | 7.887706e-06 | 7.693170e-05 |

`last_layer_grad` at step 1 = 1.35e-04 (> 1e-6: **OK**) — confirms F1's core
claim directly: the final layer's gradient is no longer vanishing.
`first_layer_grad` is exactly `0.0` at step 1 (expected — still gated by the
zero-init final layer, unaffected by F1) but exceeds 1e-6 by step 3 (within
20: **OK**) — confirms the final layer moved enough, fast enough, to
unblock the rest of the MLP. `r` moved from 0.09999990 to 0.1017631 over 20
steps (visibly non-static). **PASS.** (65s wall-clock.)

## Gate 2 — FAILED

200 real training steps, batch size 4, logged every 10 steps, no full-val
eval. Measured (H100, job 19748752, 5m56s wall-clock):

| step | r | l1_masked_ce | gap_vs_chance | C |
|---|---|---|---|---|
| 10 | 0.100785 | 3.006232 | -0.693772 | 40.5 |
| 50 | 0.105373 | 2.844719 | -0.830711 | 39.5 |
| 100 | 0.111189 | 3.054747 | -0.600273 | 38.8 |
| 150 | 0.115208 | 2.873909 | -0.818908 | 40.0 |
| 200 | 0.119308 | 3.091947 | -0.570436 | 39.0 |

- `|r - 0.1| = 0.019308 > 0.005`: **OK** — r visibly moved (19% relative
  change over 200 steps), consistent with Gate 1's healthy gradients.
- `gap_vs_chance` negative at step 200 (-0.570436): **OK** — still
  better than chance.
- `gap_vs_chance` lower at step 200 than step 20 (-0.640490): **FAIL** —
  -0.570436 is *higher* (closer to chance) than step 20's -0.640490.

This is not step-to-step noise dressed up as a failure — a window-averaged
check (mean of steps 10-50 vs mean of steps 160-200, to average out
per-step sampling variance from the small batch and per-crop randomness)
shows the same direction:

```
early-window (steps 10-50) mean gap_vs_chance:  -0.753107
late-window  (steps 160-200) mean gap_vs_chance: -0.664689
```

The late window is measurably *closer to chance* than the early window.
**The reconstruction loss is trending the wrong way over 200 steps, even
though the gradient mechanism itself (Gate 1) is confirmed healthy.**
**FAIL. Gate 3 was NOT run**, per the explicit instruction to stop at the
first gate failure.

## What Gate 2's failure does and does not tell us

- It does **not** mean F1 is wrong — Gate 1's data is unambiguous: the
  deadlock is broken, gradients flow, `r` moves substantially and
  continuously. The mechanism works.
- It does mean that whatever is currently learned in the first 200 steps
  makes masked reconstruction (L1, the primary loss) mildly *worse* on
  average, not better. Candidate explanations, none yet distinguished:
  1. `lr=1e-4` combined with `r` now moving quickly (19%/200 steps) may be
     large enough to be genuinely destabilising L1 before the model finds a
     useful direction — plausible "gets worse before it gets better" early
     dynamics that 200 steps may be too early to see past.
  2. L2 (ranking, weight 0.1) or L3 (anchor, weight 0.05) could be pulling
     the shared MLP weights in a direction that trades against L1, even
     though L1 dominates the total loss numerically.
  3. 200 steps (800 total image samples, batch 4) may simply be too few to
     separate a real trend from per-sample variance in which crop/caption/
     mask is drawn — the window-average check argues against pure noise,
     but does not rule out "started noisy, will recover with more steps."
  4. r_max=0.5 allows a fairly large maximum deviation from identity;
     r growing unchecked by a floor/ceiling schedule at lr=1e-4 could
     overshoot early.

None of these are diagnosed yet — that is exactly why Task 5 gates before
Gate 3, and exactly why Gate 3 was not launched: 2000 steps / 10
checkpoints / 1000-image evals is a multi-hour GPU commitment that
should not be spent training further under a mechanism that is already
observed to be moving the wrong way at 200 steps.

## Gate 3 cost note (not run, flagged for when this is unblocked)

Independent of Gate 2's failure: the task's own estimate for Gate 3 is
"about 2-3 hours." Using this session's directly measured full-val-eval
cost with the learned metric (~0.96-1.4s/image, ~1.17s/image average from
two real 50-image runs), 10 checkpoints × 1000 images × ~1.17s ≈ 195
minutes of evaluation alone, plus ~49 minutes of training (2000 steps ×
~1.46s/step at batch 4) ≈ **~4 hours total**, not 2-3. Worth knowing before
this gate is greenlit, regardless of what unblocks it.

## Recommendation

Do not run Gate 3. Before retrying Gate 2, decide how to disambiguate the
candidate explanations above — e.g., rerun Gate 2 for more steps (does the
trend reverse or continue?), rerun with L2/L3 weights at 0 to isolate
whether they are implicated, or rerun at a lower `lr`. This is a real
finding worth your input on rather than a call I should make unilaterally.

## RUN COMMANDS

```bash
# All commands below assume an existing interactive allocation; substitute
# your own job id or salloc fresh. Gate 1 and Gate 2 were run against job
# 19748752 (H100, node g8) via srun --overlap from the login node.
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=40G --time=03:30:00

cd /project/6114407/haree/Talk2DINO
module load gcc opencv
source /scratch/haree/venv/talk2dino-a100/bin/activate

# Full test suite -- confirms F1/F3/F4/F5 didn't break anything else.
# Measured: 136.00s.
python3 -m pytest tests/ -q
# Expected: 100 passed. STOP if this fails.

# ============================================================
# Gate 1 (~1 min measured). PASS/FAIL computed and printed; exits 1 on FAIL.
# ============================================================
python3 ablationAll/e10_adaptive_diffusion/scripts/grad_flow_check.py --device cuda --steps 20
# Measured this session: PASS (last_layer_grad=1.35e-04 at step 1,
# first_layer_grad exceeded 1e-6 at step 3).

# ============================================================
# Gate 2 (~6 min measured). Exits 1 on FAIL -- do not proceed past this.
# ============================================================
python3 ablationAll/e10_adaptive_diffusion/scripts/gate2_short_schedule.py \
  --device cuda --steps 200 --log-every 10 --batch-size 4 --vocab-image-subset 2000
# Measured this session: FAIL (gap_vs_chance regressed from -0.753 to
# -0.665, window-averaged, over 200 steps). DO NOT PROCEED TO GATE 3 UNTIL
# THIS IS UNDERSTOOD -- see "Recommendation" above.

# ============================================================
# Gate 3 (task estimate ~2-3h; this session's measured-cost estimate is
# closer to ~4h -- see "Gate 3 cost note"). NOT YET AUTHORISED -- Gate 2
# failed. Included here for when Gate 2's failure is resolved, not as a
# next step to run now.
# ============================================================
python3 ablationAll/e10_adaptive_diffusion/scripts/run_pilot.py \
  --output-dir /scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/gate3_run \
  --device cuda \
  --total-steps 2000 \
  --checkpoint-every 200 \
  --batch-size 4 \
  --vocab-image-subset 2000 \
  --n-held-out-samples 50 \
  --full-val-max-images 1000 \
  --seed 0

# Protected-path check.
git diff --stat HEAD -- src/open_vocabulary_segmentation/ configs/
# Expected: empty output.
```
