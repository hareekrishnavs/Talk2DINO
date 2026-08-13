# Part F2 — learned affinity metric training infrastructure

Status: **prerequisite gate PASSED; infrastructure (D1–D5, L1–L4, T1–T4)
implemented and unit-tested; V1–V4 pilot pipeline and E1–E4 final evaluation
implemented and smoke-tested end-to-end on real GPU (real COCO images/
captions, real model, real checkpoints). The full-scale pilot run and final
evaluation have NOT been launched** — see "Pilot budget design" and "RUN
COMMANDS" below for the exact, ordered commands to run them within a 3-hour
GPU budget.

## Prerequisite gate (P1–P4)

Covered in full, with the diagnostic detail, in the earlier gate report
(same file, see git history / conversation) — summary:

- **P1/P2**: max relative error 1.04e-05 at r≈0 (true untrained init),
  1.41e-08 at r=0.3 — both PASS. The initial P1 failure was traced to the
  finite-difference test's step size, not the analytic gradient or the
  solver (confirmed by sweeping solver tolerance down to 1e-14, which did
  NOT close the gap, then sweeping FD `eps` instead, which did,
  monotonically). No solver change was made. Locked in as a permanent
  regression test
  (`test_gradient_at_true_untrained_init_needs_larger_fd_eps`).
- **P3**: `SolverConvergenceError` raises (not warns) on non-convergence —
  confirmed.
- **P4**: real production-scale timing at α=0.98 — mean ~555–581 CG
  iterations, ~93–107ms per solve (forward or backward), H100.

## D1 — image crop/augmentation

`src/learned_affinity/crop_dataset.py`. `load_random_crop_bgr`: PIL-loads
an image, pads up if smaller than 448×448, takes a random 448×448 crop,
random horizontal flip, returns `[3,448,448]` float32 **BGR**, raw `[0,255]`
values (not yet `/255`'d or normalised). BGR, not RGB, is deliberate:
`DINOTextInference.generate_masks` does its own `image[:, [2,1,0]]`
BGR→RGB flip internally before its own `image_transforms` (resize/÷255/
ImageNet-normalise) — feeding it BGR here means it ends up correctly RGB
after that flip, with zero reimplementation of resize/normalise math.
`CocoCaptionCropDataset` wraps this as a `torch.utils.data.Dataset`,
excluding any image with no vocabulary nouns present in its captions.

## D2 — on-the-fly DINO extraction (measured, not assumed)

`src/learned_affinity/extract.py`. `extract_training_sample` calls
`model.generate_masks(crop, text_embedding)` directly — the SAME function
the sliding-window evaluation calls once per window; a single 448×448 crop
*is* one window, so no sliding-window wrapper is needed. Patch features are
tapped via the same `affinity_oracle_observer` hook Part E's capture
pipeline uses (`model.masker.affinity_oracle_observer`); `S_0` (raw
text-patch scores) is `generate_masks`' own second return value (`simmap`),
already at the right stage (`SCORE_STAGE = "pre_sigmoid_pre_upsample_pre_stitch"`,
matching the E3 oracle's own convention exactly). `generate_masks` is
`@torch.no_grad()` — correct, since DINOv2/CLIP are frozen (T1); nothing
in this function needs gradients.

**Measured (H100, 20 real COCO training samples, real captions, real spaCy
nouns, real CLIP text encoding):**

| | mean | min | max |
|---|---|---|---|
| DINO forward (crop → features+S0) | 8.41ms | 8.19ms | 9.19ms |
| CG forward solve | 94.07ms | 77.71ms | 105.16ms |
| CG backward (adjoint) solve | 92.37ms | 76.17ms | 103.37ms |

**CG solve (fwd+bwd) / DINO forward ratio: 22.17x.** The solve dominates by
far more than the 2x threshold — in the *opposite* direction D2 was
checking for. **Recommendation: on-the-fly extraction, as built, is
correct; caching a fixed-crop subset would not meaningfully help training
throughput**, since DINO extraction was never the bottleneck. (Caching
would still trade ~8ms/step of compute for disk I/O and a fixed subset's
reduced diversity — not worth it here.)

## D3/D4 — noun vocabulary + frozen CLIP text encoding

`src/learned_affinity/coco_captions.py` (`extract_nouns`,
`build_noun_vocabulary`) + `src/learned_affinity/text_vocab.py`
(`encode_noun_vocabulary`, `load_or_build_vocabulary_embeddings`,
`VocabularyEmbeddings`). POS tagging via spaCy's `en_core_web_sm` —
installed from the Alliance wheelhouse (`pip install --no-index spacy
en_core_web_sm`), entirely offline (unlike NLTK, whose POS tagger needs a
separate internet-downloaded corpus via `nltk.download()`, which the
cluster's "avoid arbitrary internet installs" policy rules out; spaCy's
model ships as a regular installable wheel instead). Vocabulary nouns are
encoded ONCE with `model.build_dataset_class_tokens`/`build_text_embedding`
— the exact same calls, on the exact same frozen model (same
`build_merged_config` pattern as `capture_dino_features.py`), using
`cfg.evaluate.template` (`"subset"` → `sub_imagenet_template`, confirmed by
replaying the real cache-build command, not assumed) — the SAME template
the evaluation uses. D4's per-step class list (`sample_step_vocabulary`) is
a pure index-gather into the cached embeddings; no text is re-encoded at
step time, and there is no trainable parameter anywhere on the text side.

Real run on a 2000-image subset: 912-noun vocabulary built in 21.5s (POS
tagging), encoded in 1.8s (text encoder). Full-corpus (118,287 images)
vocabulary building was not run in this session (would scale to roughly
`21.5s × 118287/2000 ≈ 21 min`, dominated by spaCy CPU tagging, not GPU
work) — left for the pilot run itself.

## D5 — train/val split

`disjoint_by_image_split` in `coco_captions.py`. Shuffles image IDs with a
fixed seed, splits by a `val_fraction`, asserts the two id sets are
disjoint. No image contributes captions to both splits.

## L1–L4 — loss functions

`src/learned_affinity/losses.py`. `compute_total_loss` combines all four:

- **L1 (primary, masked reconstruction)**: samples a boolean patch mask
  (default 30%), zeros `S_0` there, solves the fixed point from the masked
  input, and scores soft cross-entropy between `softmax(S*_masked/τ)` and
  `softmax(S_0/τ)` (the undiffused local evidence) — **at the masked
  patches only** (the backprop target). The same quantity at unmasked
  patches is computed and logged separately (not backpropagated) as a
  sanity signal — verified in a unit test to be smaller than the masked
  loss (unmasked patches keep their true `S_0` in the propagation's own
  `(1-alpha)*S0` term, so reconstructing them is close to trivial; masked
  ones rely entirely on diffusion from neighbours).
- **L2 (anti-collapse regulariser, default weight 0.1)**: propagates the
  **full, unmasked** `S_0` (a second solve, sharing the same graph as L1
  but not its input — this is a deliberate design choice, documented in
  the module, over reusing L1's masked solve for compute savings, since L2
  is conceptually a different signal), max-pools over patches per class,
  and margin-ranks present-caption nouns against sampled-absent
  distractors. Kept at low weight per the project's prior finding that a
  caption-contrastive objective anti-correlated with mIoU (Pearson
  −0.727) — logged as a separate component precisely so its influence stays
  auditable, per the task's own caution.
- **L3 (anchor regulariser, default weight 0.05)**: `1 - cos(g(f), f)`,
  averaged over patches.
- **L4 (row-entropy floor guard, default weight 0.0)**: one-sided penalty
  on mean row entropy of the graph falling below a floor. `mean_row_entropy`
  is always computed and returned for logging regardless of L4's weight,
  per the task's instruction to watch for entropy collapse even when the
  guard itself is off.

18 unit tests (`tests/test_f2_data_and_losses.py`), including: gradient
flow through the full `compute_total_loss` chain to both the graph weights
and to `g` itself; the masked > unmasked reconstruction-difficulty sanity
check; L2's zero-loss-when-already-separated and both-classes-required
checks; L3 zero-at-identity / positive-when-different; L4's uniform-row
entropy matching the analytic `log(k)` value and zero at full concentration.

## T1–T4 — training loop

`src/learned_affinity/train.py`. `training_step` (the actual per-step
optimiser logic — forward, loss, backward, clip, step, log) is deliberately
decoupled from real image/text extraction, taking already-extracted
`(features, raw_scores, is_present)` samples — this is what makes it
unit-testable on synthetic CPU data without a live model or GPU. The real
glue (`build_step_sample`, `run_training`) produces those samples from real
COCO images/captions via the frozen model and is exercised end-to-end by
the D2 measurement script above, but `run_training`'s full multi-thousand-
step loop was not invoked.

- **T1**: only `metric.parameters()` are ever passed to the optimiser;
  `alpha` is a fixed float (0.98) threaded through every call, never a
  learnable tensor (and `implicit_propagate`/`ImplicitPropagate` already
  reject a `requires_grad=True` alpha explicitly, from the F1 solver work).
- **T2**: `AdamW`, `lr=1e-4` default, `cosine_lr_lambda` (warmup then
  cosine decay), gradient clipping via `torch.nn.utils.clip_grad_norm_` at
  1.0 default. Batch size is gradient-accumulated (loss averaged over the
  batch before one `backward()`/`step()`) rather than solved in a single
  batched CG call — the solver operates on one `[P,C]` system per image, so
  a "batch" of B images costs B sequential solve pairs; **at the measured
  ~186ms/pair, a batch of 4 costs roughly 750ms of solve time alone**. This
  sets a real, load-bearing constraint on `T2`'s "batch size set from the
  P4 measurement" instruction — worth confirming against your intended
  training wall-clock budget before picking a batch size for the pilot.
- **T3**: `training_step` returns every quantity asked for — all four loss
  components, `r`, mean row entropy, per-sample CG forward/backward
  iteration counts, and grad norm — every step.
- **T4**: `save_checkpoint`/`load_checkpoint` save `r` explicitly alongside
  the state dict and optimiser state; round-trip tested.

5 unit tests (`tests/test_f2_training_loop.py`): a real optimiser step
changes metric parameters and produces finite losses/logs on synthetic
data; the cosine schedule's warmup/peak/decay values; checkpoint
save/load round-trips `r` and the step count correctly.

## Full suite: 100/100 passing

(34 original E3 oracle tests + 7 Part E tests + 9 F1 learned-affinity tests
+ 18 F2 data/loss tests + 5 F2 training-loop tests + others.)

## V1–V4 pilot pipeline and E1–E4 final evaluation — implemented and smoke-tested

`src/learned_affinity/pilot.py` (V1 training+checkpoint+correlation loop,
V2 Pearson/Spearman correlation of each loss component against full-val
mIoU across checkpoints, V3/V4 checkpoint selection using ONLY a
positively-correlated component — or an explicit "no usable signal" finding
if none exists) and `src/learned_affinity/final_eval.py` (E1 full-val
mIoU/aAcc/mAcc, E2 per-class deltas vs canonical with the ten specified
previously-negative classes highlighted, E3 thing/stuff group means, E4
r-forced-to-0 identity regression against the CANONICAL.md anchor). Both
were run for real on GPU end-to-end (4-step, 300-image-vocab, 50-image-val
smoke test) — the full mechanism works: real training steps update `r`,
checkpoints save/load, correlations compute, selection logic picks a
checkpoint (or reports V4's "no signal" finding), and E1-E4 produce real
numbers. One bug was found and fixed during this smoke test: `final_eval.py`
crashed formatting `None` per-class IoU values (expected — a class with
zero ground-truth pixels in a small/capped evaluated set, per
`metrics_from_confusion`'s own convention, not a defect); fixed with an
explicit `N/A (no ground-truth pixels for this class in the evaluated set)`
fallback, re-verified crash-free on GPU.

`final_eval.py` also gained an independent `--e4-max-images` flag (defaults
to `--max-images` if omitted). This matters because of a cost asymmetry
discovered during smoke-testing (see below): E4 compares against the
CANONICAL_MIOU constant, which is itself a full-5000-image number — a
capped E4 subset will NOT reproduce it even with zero drift, simply from
subset-sampling variance (the smoke test's own E4 showed a 6.87-point
"failure" at 50 images purely from this effect, not a real problem). So E4
is only a literal tolerance check at full scale; at any smaller scale it is
a mechanism/no-drift sanity check only (this is stated in the script's own
output). Since E1 is the primary reported number, the two now have
independently sized budgets rather than forcing both to pay full-scale cost
together.

## Pilot budget design (measured costs, real numbers, H100)

| Operation | Measured cost |
|---|---|
| DINO forward (crop → features+S0) | 8.41ms/image |
| One `compute_total_loss` training step | ~365ms/sample (2 CG solves: L1 masked + L2 full — NOT 1; confirmed by direct timing, batch size does not change the per-sample rate since solves are not batched across images) |
| Full-val eval **with the learned metric** (`evaluate_with_learned_metric` / `evaluate_full_val_converged`, used for E1 and every pilot checkpoint) | ~0.96–1.4s/image (measured 1.386s and 0.958s/image on two 50-image runs; **not** the ~0.25s/image of the plain canonical cache replay — this path builds a fresh differentiable graph per window) → **full 5000-image E1/E4 costs ~80–115 minutes each**, not the ~21 minutes I originally assumed by analogy to the plain replay path. This is the single most important correction to plan around. |
| Setup (model + vocab build, 2000-image subset) | ~1–1.5 min (912-noun vocab from 2000 images measured at 21.5s tagging + 1.8s encoding, plus model load) |

Given a **3-hour (180 min) budget**, spending it on BOTH a full-scale E1
AND a full-scale E4 (~200 min combined) alone exceeds the budget before any
training happens. The design below spends the budget as:

- **Pilot (V1–V4), ~32 min**: 500 steps, 5 checkpoints, capped
  `full-val-max-images=150` per checkpoint (enough for 5 real correlation
  data points — the smoke test's 2-checkpoint "perfect" ±1.0 correlations
  are not meaningful; with only 2 points Pearson r is trivially ±1).
- **Final evaluation (E1–E4), ~112 min**: E1–E3 at full scale (5000 images,
  the literal, directly-comparable-to-CANONICAL.md number — this is the
  headline deliverable) + E4 capped to 600 images (a real but
  budget-conscious mechanism/no-drift check, not the literal 5e-3 tolerance
  test, which would need the full 5000).
- Leaves **~26–35 min buffer** out of 180 for setup overhead, reading the
  selected checkpoint, and slack against measurement variance (all the
  per-image costs above have ~±20% spread across the two real
  measurements taken).

If you have MORE than 3 hours, the highest-value upgrade is raising
`--e4-max-images` toward the full 5000 (each +1000 images costs ~20 min) so
E4 becomes a literal tolerance check rather than a sanity check; the second
highest-value upgrade is more pilot steps/checkpoints for a less noisy V2
correlation.

## RUN COMMANDS

```bash
# Fresh interactive allocation sized for the full pipeline below (~144 min
# of GPU compute + setup/buffer). Adjust --account/--partition/--gres to
# match what's available on the queue at the time.
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=40G --time=03:30:00

cd /project/6114407/haree/Talk2DINO
module load gcc opencv
source /scratch/haree/venv/talk2dino-a100/bin/activate

# One-time: spaCy + POS-tagger model, from the wheelhouse (offline).
# Already installed in this venv from earlier session work -- pip will just
# report "already satisfied". Measured (fresh install): ~15s.
pip install --no-index spacy en_core_web_sm

# Full test suite (P1-P4 permanent regression test + all F2 unit tests) as
# a final correctness gate before spending real GPU time. Measured: 125.4s.
python3 -m pytest tests/ -q
# Expected: 100 passed. STOP and do not proceed if this fails.

# ============================================================
# Phase 1 -- V1 pilot: real training + checkpoint sweep + V2/V3/V4.
# Estimated: ~32 min (setup ~1.5min + training ~12min + eval ~17.5min).
# ============================================================
python3 ablationAll/e10_adaptive_diffusion/scripts/run_pilot.py \
  --output-dir /scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/pilot_run_v1 \
  --device cuda \
  --total-steps 500 \
  --checkpoint-every 100 \
  --batch-size 4 \
  --vocab-image-subset 2000 \
  --n-held-out-samples 50 \
  --full-val-max-images 150 \
  --seed 0
# Expected: 5 checkpoints (steps 100,200,300,400,500), prints V2
# correlations per component, then a V3/V4 selection line at the end, e.g.
# "selected step N using <component> (Pearson r=... vs mIoU)" or, if no
# component correlates usefully, an explicit V4 "no signal" finding --
# report either outcome plainly, do not override it.

# Read the selected checkpoint (V3: selection MUST come from this file's
# "selection" field, not from eyeballing full_val_mIoU in the log).
python3 -c "
import json
d = json.load(open('/scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/pilot_run_v1/pilot_summary.json'))
print(d['selection'])
"
# Note the 'selected_checkpoint_step' value (or, if null, see V4's finding
# string -- in that case there is no validated checkpoint to select for the
# final eval below; that null result is itself the reportable outcome).

# ============================================================
# Phase 2 -- E1-E4 final evaluation on the selected checkpoint.
# Substitute the step number from above into CKPT_STEP.
# Estimated: ~112 min (E1-E3 full 5000 images ~100min + E4 @600 images ~12min).
# ============================================================
CKPT_STEP=500   # <-- replace with the step selected above, zero-padding not needed here (python does it)
python3 -m src.learned_affinity.final_eval \
  --checkpoint /scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/pilot_run_v1/checkpoint_$(printf '%06d' $CKPT_STEP).pt \
  --device cuda \
  --e4-max-images 600 \
  --output ablationAll/e10_adaptive_diffusion/results/final_eval_v1.json
# Expected: E4 result (mechanism check at 600 images, not the literal 5e-3
# tolerance test -- see the script's own printed NOTE), then E1 full_val
# mIoU/aAcc/mAcc with delta vs CANONICAL.md's 29.877244/48.528671/54.137035,
# E3 thing/stuff group-mean deltas (compare against fixed-diffusion's
# reported +1.3747 things / +1.4166 stuff at T=320), and E2's ten
# previously-negative classes with canonical/trained/delta (some may show
# "N/A" if that class has zero ground-truth pixels among the images
# evaluated -- expected at anything less than the full 5000, not a bug).
# Written to ablationAll/e10_adaptive_diffusion/results/final_eval_v1.json.

# Protected-path check -- confirm no protected file was touched anywhere
# in this session.
git diff --stat HEAD -- src/open_vocabulary_segmentation/ configs/
# Expected: empty output.
```
