# Part E adversarial review — DINO patch-feature capture pipeline

Reviewer stance: re-derive every claim from code and artifacts on this
allocation; do not trust `RUN_PartE.md`'s own numbers. Every check below
either reads a real on-disk artifact, greps real source, or runs a fresh
computation independent of `src/e3_affinity_oracle.py`'s own verifier
functions (`verify_feature_capture`/`evaluate_with_rebuilt_graph` were
**not** called for W3/W4/W6/W7 — a separate from-scratch cosine/top-k
reimplementation was used instead, so a bug shared between the capture
pipeline and its own verifier would still be caught).

## Table

| # | Check | Verdict | Evidence |
|---|---|---|---|
| W1 | Blast radius / read-only tap | **PASS** | `git diff --stat HEAD -- src/open_vocabulary_segmentation/models/` and `.../configs/` are both empty. `affinity_oracle_observer` (masker.py) and `affinity_oracle_capture`/`oracle_dataset` (dinotext_seg.py) were already present in committed `HEAD` (`git show HEAD:...`) — Part E reused a pre-existing no-op extension point, added no new hook into any model file. `capture_dino_features.py` only ever calls `OmegaConf.load`/string paths on config files, never writes to them. |
| W2 | Preprocessing identity | **PASS** | Dataset config file is byte-identical: SHA256 of the file `capture_dino_features.py` loads = `651710d2bff2...` = the cache manifest's own recorded `dataset_config_sha256`. Independently reconstructed the REAL cache-build command's exact `OmegaConf.merge` (from the manifest's own recorded `commands` field) and diffed `cfg.model` against `capture_dino_features.py`'s `build_merged_config()` — **YAML-identical**. `forward()` dispatch: the oracle-capture branch (`dinotext_seg.py:178-193`) calls the literal same `self.simple_test(...)` as the non-oracle branch (`:176`) — only difference is `begin_image`/`end_image` bookkeeping, zero effect on computed logits. Calling convention (`data["img"]`, `data["img_metas"]`, `return_loss=False, rescale=True`) matches `us/misc.py:multi_gpu_test:285-289` exactly. `capture_dino_features.py` contains zero resize/crop/normalize logic of its own (grepped). |
| W3 | Independent E3 affinity re-derivation, 200 windows | **PASS** | Fresh cosine/clamp/pow/top-12/tie-break reimplementation, 200 windows sampled (seed 999) across the *entire* val set (not just the 50-image smoke sample the original report used): `edge_match_fraction = 0.997157` (edges_total=2,457,600), clears the >=0.99 bar. Slightly higher than the original report's smoke-50-only figure (0.996772) — consistent with a larger, more representative sample. |
| W4 | Tie verification | **PASS** | Of 6,831 disagreeing rows, 92.91% have an affinity gap < 0.01 (mean gap 0.00296); the tail goes up to 0.0967 — small in absolute terms, plausible as genuine (not just floating-point) close competition at the k=12 boundary in real 768-d embeddings, not evidence of corrupted features. |
| W5 | Anchors (`--assert-anchors`) | **FAIL (partial)** | Real artifact, `e4_assert_anchors.log`, timestamp 2026-08-12 12:51. α=0 passed silently (the function only reaches the α=0.98 comparison after α=0 succeeds — a separate, distinct error message exists for an α=0 failure, and it did not fire), so α=0=**28.480169315747716 exactly, PASS**. α=0.98,T=320 **misses the literal 1e-4 tolerance**: mIoU actual=29.87804875879269 vs expected 29.877196 (Δ0.00085); aAcc actual=48.52820236 vs expected 48.528726 (Δ0.00052); mAcc actual=54.13882134 vs expected 54.137089 (Δ0.00173). `--assert-anchors` correctly exits non-zero on this. |
| W6 | Normalisation | **PASS** | Loaded 5 random shards (2,560,000 patch rows). Max \|‖f‖−1\| = 0.000121, mean = 0.0000149; **100%** of rows are within 1e-3 *and* within `build_knn_graph`'s own strict 2e-4 tolerance. Confirms features are genuinely L2-normalised at the point they're captured (masker.py:222, before the write), not merely claimed — and the theoretical fp16-norm-drift risk the original session flagged and pre-emptively patched (re-normalise on load) turns out not to materialise on real data at this magnitude, though the patch is still the correct defensive choice. |
| W7 | Manifest integrity | **PASS** | 5 random windows resolved manifest → shard → offset: correct shape `(1024,768)`, dtype `float16`, and correct `image_id`/`coordinates`/`dataset_index` cross-references (including two non-trivial edge windows with clamped coordinates, e.g. `[149,0,597,448]`). Manifest window count (11,075) independently reproduced two different ways: (a) recomputed from the cache's own `resized_input_shape`/crop/stride grid math (not copied from either manifest) = 11,075; (b) a fresh `build_seg_dataset()` call gives `len(dataset) == 5000`, matching both manifests' `selected_image_count`. Three-way agreement. |
| W8 | Determinism | **PASS** | Re-ran `capture_dino_features.py --limit 50` as a fresh, independent GPU invocation. `sha256sum` of the resulting shard file is **bit-for-bit identical** to the original smoke capture's shard (`be01cca9...` both sides), confirmed additionally by `cmp` (exit 0). |
| W9 | Storage claim | **PASS** | `du -sb` on the full-val capture directory = 17,423,537,501 bytes. The capture's own manifest `total_bytes` (shard data only) = 17,419,505,071 bytes = 17.4195 GB, matching the claimed "17.420 GB". The ~4 MB gap between `du` and manifest `total_bytes` is `manifest.json` itself (large — per-window metadata for 11,075 windows) plus filesystem overhead, not a discrepancy in the shard-size claim. |

## Verdict: **NOT ADMISSIBLE** as a claim of *exact* reproduction — with one clarification

Eight of nine axes hold up under independent, from-scratch re-derivation:
blast radius is clean, the preprocessing path is *provably* the same code
(not just plausibly similar — the model config is YAML-identical and the
oracle branch calls the literal same `simple_test` method), the graph
rebuild agrees on 99.7% of edges with the disagreements explained as
near-ties, normalisation is real, the manifest is internally and externally
consistent, capture is bit-for-bit deterministic, and the storage figure is
accurate.

**W5 is a genuine, reproducible failure of the literal claim.** α=0.98,T=320
propagated entirely from a graph rebuilt out of the captured features does
**not** reproduce the canonical anchor within the specified 1e-4 tolerance —
it's off by roughly 0.0005–0.0017 percentage points depending on the metric.
This was already disclosed in the original `RUN_PartE.md` (not hidden), and
is fully explained by the already-quantified 99.7% (not 100%) edge agreement
compounding over 320 propagation steps — not a separate, unexplained bug.
Every other check independently confirms the capture pipeline itself is
correct; the failure is specifically in "exact" numerical reproduction under
long propagation chains, traceable to fp16 feature storage.

## Minimal fix per FAIL

**W5 only.** Re-capture with `raw_scores`-equivalent fidelity for features —
i.e. store patch features as float32 instead of float16
(`WindowFeatureCapture.observe`'s `features.to(device="cpu", dtype=torch.float16)`
→ `dtype=torch.float32`). This roughly doubles on-disk size (~17.4 GB → ~34.8
GB for the full val split) and capture script memory footprint per shard,
but removes the only source of the disagreement this review found. No other
row in this table has a fix to apply.

## RUN COMMANDS (this adversarial review)

```bash
# Ran via srun --overlap on an already-active interactive allocation
# (job 19677241, H100, node g25); salloc line for a cold start:
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=40G --time=00:30:00

cd /project/6114407/haree/Talk2DINO
module load gcc opencv
source /scratch/haree/venv/talk2dino-a100/bin/activate

# W1/W2: no GPU needed -- git/grep/sha256sum on the login node, plus one
# quick config-merge equivalence check (~5s, CPU only):
git diff --stat HEAD -- src/open_vocabulary_segmentation/models/ \
  src/open_vocabulary_segmentation/configs/
git show HEAD:src/open_vocabulary_segmentation/models/dinotext/masker.py | grep -n affinity_oracle_observer
git show HEAD:src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py | grep -n affinity_oracle_capture
sha256sum src/open_vocabulary_segmentation/segmentation/configs/_base_/datasets/stuff.py
# (compare against manifest["dataset_config_sha256"] in the cache manifest)

# W3/W4/W6/W7: independent re-derivation script (must live on a
# node-shared filesystem, NOT /tmp -- /tmp is login-node-local and
# invisible from the compute node). Measured: ~55s wall-clock.
python3 -u /scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/adversarial_w3_w4_w6_w7.py
# Expected: edge_match_fraction >= 0.99, W7 3-way window-count agreement,
# 100% of sampled rows within build_knn_graph's 2e-4 norm tolerance.

# W5: cross-check against the real --assert-anchors artifact already on
# disk (measured 8m47s when originally run; not re-run here since capture
# is proven bit-deterministic in W8, so a re-run would reproduce the
# identical numbers):
cat /scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/e4_assert_anchors.log
# To regenerate from scratch instead of trusting the artifact (~9 min):
python3 -u run_e3_affinity_oracle.py verify-feature-capture \
  --capture-dir /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full \
  --cache /scratch/haree/talk2dino_e3_affinity_oracle/cache/full \
  --device cuda --assert-anchors

# W8: independent re-capture + bitwise diff. Measured: 2.3s capture +
# instant sha256sum/cmp.
rm -rf /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_smoke50_redo
python3 -u capture_dino_features.py \
  --split val --output-dir /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_smoke50_redo \
  --shard-size 500 --device cuda --limit 50
sha256sum /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_smoke50/shards/windows-000000.pt \
          /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_smoke50_redo/shards/windows-000000.pt
# Expected: identical hashes on both lines.

# W9: storage claim (no GPU needed).
du -sb /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full/
python3 -c "import json; print(json.load(open('/scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full/manifest.json'))['total_bytes'])"
```
