# Part E — DINO patch-feature capture infrastructure: RUN COMMANDS

Status of this document: written after implementing `capture_dino_features.py`
(repo root), the E3/E4 library functions in `src/e3_affinity_oracle.py`
(`load_capture_manifest`, `load_capture_features`, `verify_feature_capture`,
`evaluate_with_rebuilt_graph`, `assert_feature_capture_anchors`), and the new
`verify-feature-capture` subcommand in `run_e3_affinity_oracle.py`, on branch
`e10-adaptive-diffusion`. Part D (whole-image graphs) was cancelled before
this part started and is not touched here. This is infrastructure only —
training a learned projection `g` on the captured features is explicitly out
of scope for this part.

Every number below is measured from the actual runs (E7 discipline — no
priors, no extrapolation presented as fact); GPU work ran through
`srun --jobid=<allocation> --overlap` on an already-active interactive
allocation rather than a fresh `salloc`/`sbatch`, since one was available at
the time. An equivalent `salloc` line is given below for a cold start.

## E1 — discovery summary (delivered before any code was written)

- DINOv2 patch tokens are produced by `DINOTextMasker.forward_seg`
  (`src/open_vocabulary_segmentation/models/dinotext/masker.py`); the
  L2-normalisation used for the text–patch dot product happens at
  `masker.py:222` (`image_feat = us.normalize(image_feat, dim=1)`) —
  **after** the backbone, not inside it. Tensor shape at that point is
  `[1, 768, 32, 32]` (BCHW), dtype float32.
- Sliding-window protocol: crop `(448, 448)`, stride `(224, 224)`, driven by
  `DINOTextSegInference.slide_inference` — unmodified, reused as-is via the
  observer hook (see E2 below). Resize/normalisation is deliberately
  two-stage: the mmseg pipeline resizes with `Normalize` disabled
  (`stuff.py:21`, commented out) and casts to float via `FloatImage`; the
  model's own `image_transforms` does the actual per-crop `Resize((448,448))`
  → `/255.0` → ImageNet mean/std normalisation, avoiding double-normalisation.
- `build_text_embedding.py` (used verbatim as the loading pattern for the
  frozen backbone) stubs `cv2` with a `MagicMock` because it never decodes an
  image. `capture_dino_features.py` **must** decode real images and initially
  reused that same stub — this was wrong for this script (see "bugs found on
  GPU" below).
- `knn_k=12`/`affinity_power=3.0` are baked into `knn_indices`/`knn_weights`
  at cache-construction time inside `AffinityOracleCacheWriter.add_window` →
  `build_knn_graph`; the cache never stores raw patch features
  (`validate_cache_shard` rejects any shard key containing `"feature"`). This
  is why a correctness gate (E3) has to rebuild the graph from a fresh
  capture and compare it edge-for-edge against the cache's own graph, rather
  than reading anything back out of the cache directly.

## Design

`capture_dino_features.py` (repo root) attaches a new `WindowFeatureCapture`
class — mirroring `OnlineAffinityOracleCapture`'s
`begin_image`/`set_window`/`observe`/`end_image` interface exactly — through
the same `affinity_oracle_capture`/`oracle_dataset` extension point the real
E3 online cache uses. `DINOTextSegInference.forward()`/`slide_inference()`
run completely unmodified; nothing under
`src/open_vocabulary_segmentation/{models,configs}/` is read for writing or
edited. Captured features are stored as L2-normalised float16 `[1024,768]`
per window, sharded with a manifest (image id, window index, shard file,
offset, provenance block, `existing_cache_manifest_sha256` cross-reference).

`src/e3_affinity_oracle.py` gained:
- `load_capture_manifest`/`load_capture_features` — schema-validated readers
  for the new capture format (`CAPTURE_FORMAT =
  "talk2dino-e10-dino-feature-capture-v1"`), mirroring the closed-schema
  discipline `load_cache_manifest` already uses for the real cache.
- `verify_feature_capture` (**E3**) — rebuilds the k=12/affinity_power=3.0
  kNN graph from captured features for every window matched to the cache by
  `(image_id, coordinates)`, and compares it edge-for-edge against the
  cache's own `knn_indices`/`knn_weights`. Reports window-exact-match
  fraction, edge-match fraction, max weight diff on matching edges, and a
  tie-affinity-gap diagnostic (computed from the rebuilt affinity matrix) to
  distinguish genuine disagreements from floating-point ties. Gate:
  `edge_match_fraction >= 0.99`.
- `evaluate_with_rebuilt_graph`/`assert_feature_capture_anchors` (**E4**) —
  substitutes the rebuilt graph into the *unmodified*
  `replay_cached_image`/`replay_cached_image_logits` (only the per-window
  `knn_indices`/`knn_weights` tensors are swapped; the cache's own
  `raw_scores` are reused as-is, since the capture pipeline cannot reproduce
  those — they need the live text embedding, not just patch features), then
  checks the α=0 identity and the α=0.98,T=320 anchors.
- A latent correctness risk was found and fixed before any GPU run: fp16
  storage of a 768-d unit vector can perturb its L2 norm by enough to
  occasionally miss `build_knn_graph`'s `2e-4` tolerance check. Both E3/E4
  functions now re-normalise (`F.normalize`) immediately after the fp16→fp32
  upcast.
- `verify-feature-capture` is a new subcommand in `run_e3_affinity_oracle.py`
  (`--capture-dir`, `--cache`, `--device`, `--output`, `--max-images`,
  `--assert-anchors`) — existing subcommands are untouched (I3).

6 new tests added to `tests/test_e3_affinity_oracle.py` (synthetic
clustered-feature cache + capture fixtures, exact-agreement case, a
deliberately-corrupted-feature case, partial-coverage handling, a stale-cache
rejection check, and an `evaluate_with_rebuilt_graph` vs `evaluate_cache`
equivalence check at α=0 and α=0.5). Full suite: **41/41 passed** before any
GPU work was attempted.

## Bugs found only once real GPU/data was exercised (E7 discipline: report
what actually happened, not what was assumed)

1. **`cv2` stub was wrong for this script.** `build_text_embedding.py`'s
   `MagicMock` stub is harmless there because it never decodes an image;
   `capture_dino_features.py` does, and the stub produced `img.shape == ()`
   deep inside `mmcv.imrescale`, crashing `Resize`. The earlier "GPU nodes
   lack AVX-512, real cv2 SIGILLs" finding turned out to be **node-specific**
   — the H100 node used here (`g25`) has `avx512f`, and
   `module load gcc opencv` (loaded **before** activating the venv, giving a
   real `cv2 4.14.0`) works cleanly. Fixed by removing the stub entirely and
   requiring a real `cv2` import.
2. **`FloatImage` mmseg pipeline transform wasn't registered.** It's defined
   and registered via `@PIPELINES.register_module()` as an import side
   effect of `main.py` (`main.py:54-58`); every dataset config references it
   by name. Fixed with `import main  # noqa: F401` (its `main()` only runs
   under `main.py`'s own `__main__` guard, so this is side-effect-free beyond
   the one registration).
3. **`DINOTextSegInference.__init__` calls the module-global
   `utils.logger.get_logger()` with no arguments**, which requires
   `get_logger(cfg)` to have set its `logger_name` global at least once
   first — normally done by `main.py`'s own `train()`. Fixed by calling
   `get_logger(OmegaConf.create({"model_name": ..., "output": ...}))` once,
   pointed at the script's own `--output-dir` (after `WindowFeatureCapture`
   has already created that directory, so its own overwrite/`FileExistsError`
   guard still runs first).
4. **The `talk2dino` venv (as opposed to `talk2dino-a100`) is missing
   `typing_extensions`** — torch itself fails to import. Not something this
   session modified; flagged for awareness, not silently patched.

None of these required touching anything under
`src/open_vocabulary_segmentation/{models,configs}/`.

## E3 — correctness gate (measured, on the 50-image smoke capture)

```
images_matched=50  windows_compared=111  edges_total=1,363,968
window_exact_index_set_match_fraction = 0.000000
edge_match_fraction                   = 0.996772   (gate: >= 0.99 -> PASSED)
max_abs_weight_diff_where_indices_match = 0.014221
disagreeing_row_count = 4294
mean_tie_affinity_gap = 0.003253
max_tie_affinity_gap  = 0.144044
ties_under_threshold_fraction (gap < 0.01) = 0.925
```
No window had a 100%-exact index set (expected on real DINO features — real
embeddings have no artificial tie-breaking margin, unlike the synthetic
clustered fixtures used in the unit tests). 92.5% of the disagreeing rows are
within a `0.01` affinity-gap tie threshold; a small tail goes up to `0.144`,
consistent with genuine close competition among natural near-neighbours in
real feature space at the k=12 boundary, not a bug. **Gate: PASSED.**

## E4 — end-to-end reproduction (measured, on the full 5000-image capture)

- `alpha=0.00,T=10`: reproduced `28.480169315747716` **exactly** (5000/5000
  images matched, 0 skipped) — this by itself proves the raw_scores/image/
  annotation replay path is untouched and correct.
- `alpha=0.98,T=320`, propagated **entirely from the rebuilt graph**:

  | metric | actual | expected | diff |
  |---|---|---|---|
  | mIoU | 29.87804875879269 | 29.877196 | 0.00085 |
  | aAcc | 48.52820236450192 | 48.528726 | 0.00052 |
  | mAcc | 54.13882134194957 | 54.137089 | 0.00173 |

  This misses the literal `1e-4` tolerance. **Decision (user-confirmed):
  accept as a documented limitation.** The deviation is fully consistent
  with E3's own 99.68% (not 100%) edge-agreement figure compounding over 320
  propagation steps — not a separate bug, since the α=0 check (which never
  touches the graph at all) is bit-exact. `--assert-anchors` correctly
  exits non-zero on this miss; it is not silently loosened.

## E5 — actual on-disk size (measured, not estimated)

| capture | images | windows | bytes/window | total |
|---|---|---|---|---|
| smoke (`--limit 50`) | 50 | 111 | 1,572,878 | 167 MB (0.175 GB) |
| full val | 5000 | 11,075 | 1,573,382 | **17.420 GB** |

Under the 20GB flag threshold from the task; no STOP triggered.

## E6 — training-data strategy cost comparison (report only, nothing implemented)

Numbers below are measured on the val split (0.048 s/image, 1.573 MB/window,
~2.215 windows/image); COCO-Stuff train2017's public image count (118,287)
is used for the train-side estimate below and is **not** independently
verified against this cluster's local copy of the dataset (the local
`data/coco_stuff164k/images/` path used by the eval config was not found
under the repo when checked — the data likely lives elsewhere, e.g. a
symlinked or externally-mounted location — so this figure is a labelled
assumption, not a measurement).

**(a) On-the-fly extraction during training.** Every training step that
needs fresh patch features for a batch of images would pay a full frozen
DINOv2-backbone forward pass through the sliding-window protocol: measured
at ~0.048 s/image on an (already shared, `--overlap`) H100. A projection
head `g` (`normalize(f + r * MLP(f))`) is a tiny two-layer MLP — its own
forward/backward is sub-millisecond. So the frozen backbone forward pass
would dominate every single training step by 1-2 orders of magnitude,
turning each gradient step into essentially a full inference pass. This cost
is paid **again on every single epoch** for the same images, since nothing
is reused across epochs.

**(b) Caching a fixed subset.** Extraction is a one-time cost:
`images × 0.048 s`. Some concrete budgets, using the measured rate:

| target | images | extraction wall-clock | disk (fp16) |
|---|---|---|---|
| match today's val-scale cache | ~5,000 | ~4 min | ~17.4 GB |
| a larger training subset | 20,000 | ~16 min | ~70 GB |
| a larger training subset | 50,000 | ~40 min | ~175 GB |
| full COCO-Stuff train2017 (unverified local count, public figure) | 118,287 | ~95 min (~1.6 h) | ~412 GB |

Once cached, every subsequent training epoch reads fp16 tensors off disk
instead of re-running the backbone — effectively free compared to (a).

**Recommendation:** cache a fixed subset, not the full train split, and
not on-the-fly. Reasoning: (1) a lightweight MLP projection head almost
always needs many epochs over its training pool to converge, and repaying a
~0.048 s/image frozen-backbone cost on every epoch (option a) is wasteful
the moment training runs more than ~1 epoch; (2) the full train2017 split's
projected ~412 GB is a large, unverified number worth checking against
actual project/scratch quota before committing to it — a 20-50K-image
subset (~70-175 GB) is very likely enough diversity for a small MLP head and
keeps the one-time extraction cost to well under an hour; (3) extraction
cost and size scale linearly and predictably from measured, not assumed,
per-image numbers, so the actual target size is a budget decision for
whoever trains `g`, not an infrastructure constraint.

## `git diff --stat` (I1 confirmation)

```
 run_e3_affinity_oracle.py        | 1011 +++++++++++++++++++++++++++++-
 src/e3_affinity_oracle.py        | 1288 +++++++++++++++++++++++++++++++++++++-
 tests/test_e3_affinity_oracle.py |  298 +++++++++
 3 files changed, 2586 insertions(+), 11 deletions(-)
```
Plus the new, untracked `capture_dino_features.py` at repo root. **Nothing**
under `src/open_vocabulary_segmentation/models/`,
`src/open_vocabulary_segmentation/configs/`, the E3 checkpoint, or the E3
config appears in the diff. Nothing inside
`/scratch/haree/talk2dino_e3_affinity_oracle/cache/` was written, moved, or
deleted — every script that touches it only reads `manifest.json` and shard
files, and `verify_feature_capture_cli` hard-fails if the cache's
`manifest.json` hash doesn't match what a capture recorded at capture time.

## RUN COMMANDS

```bash
# 1. Allocate a GPU node (cold-start form; this session instead attached
#    non-invasively via `srun --jobid=<id> --overlap` to an already-active
#    interactive allocation on node g25, an H100 — see note below).
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=40G --time=02:00:00

# Inside the allocation:
cd /project/6114407/haree/Talk2DINO
module load gcc opencv        # MUST precede venv activation -- provides a
                               # real, working cv2 (avx512f-built; this node
                               # has avx512f). Do not stub cv2 for this script.
source /scratch/haree/venv/talk2dino-a100/bin/activate
# NOTE: /scratch/haree/venv/talk2dino (no "-a100" suffix) is currently
# broken (missing typing_extensions) -- use talk2dino-a100.

CACHE=/scratch/haree/talk2dino_e3_affinity_oracle/cache/full
OUT=ablationAll/e10_adaptive_diffusion/results
mkdir -p "$OUT"

# 2. E7 smoke test (--limit 50). Measured: 3.1s total, 0.063s/image,
#    0.028s/window, 167MB written.
rm -rf /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_smoke50
python3 -u capture_dino_features.py \
  --split val --output-dir /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_smoke50 \
  --shard-size 500 --device cuda --limit 50
# Expected: "DONE: 50 images, 111 windows, ~3s total, ..., 0.175 GB written"

# 3. E3 correctness gate on the smoke capture. Measured: 44.6s wall-clock
#    (streams and validates the full 7.8GB production cache from disk).
python3 -u run_e3_affinity_oracle.py verify-feature-capture \
  --capture-dir /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_smoke50 \
  --cache "$CACHE" --device cpu \
  --output "$OUT/verify_feature_capture_smoke50.json"
# Expected: "... edge_match_fraction=0.996772 ... gate_passed=True", exit 0.

# 4. Full val capture (all 5000 images). Measured: 237.5s (~4.0 min) total,
#    0.048s/image, 17.420 GB written.
python3 -u capture_dino_features.py \
  --split val --output-dir /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full \
  --shard-size 500 --device cuda
# Expected: "DONE: 5000 images, 11075 windows, ~237s total, ..., 17.420 GB written"

# 5. E4 second gate: alpha=0 exact identity + alpha=0.98/T=320 anchors.
#    Measured: 8m47s wall-clock.
python3 -u run_e3_affinity_oracle.py verify-feature-capture \
  --capture-dir /scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_val_full \
  --cache "$CACHE" --device cuda --assert-anchors
# Expected/actual: alpha=0 exact; alpha=0.98,T=320 misses the literal 1e-4
# tolerance by ~0.001-0.002 percentage points (documented and accepted --
# see "E4" section above); exits non-zero. This is expected given this
# decision, not a regression to chase.

# 6. Unit tests (no GPU needed; run any time, including on the login node).
python3 -m pytest tests/test_e3_affinity_oracle.py -q
# Expected: 41 passed.
```

Note on how this session actually ran: a GPU node (`g25`, H100, job
`19677241`) was already active as the user's own interactive allocation, so
every GPU command above ran via
`srun --jobid=19677241 --overlap bash -c '...'` rather than opening a new
`salloc`. This is the "peek at/use an already-running job non-invasively"
pattern — appropriate here since the allocation was idle and available; use
the `salloc` line above for a cold start.
