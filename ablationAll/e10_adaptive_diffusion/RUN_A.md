# Part A — extended alpha/T global sweep: RUN COMMANDS

Status of this document: written after implementing `global-sweep-ext` in
`run_e3_affinity_oracle.py` / `src/e3_affinity_oracle.py` on branch
`e10-adaptive-diffusion`. Verification performed before writing this file:

- Synthetic-cache unit-level checks (tiny 2-image cache, CPU, no GPU/queue
  needed): `propagation_steps=None` (default) is bit-identical to the old
  hard-coded `manifest["propagation_steps"]` path; `propagation_steps=T`
  actually changes the result and matches an independent
  `propagate_scores(..., propagation_steps=T)` call on the same cached
  graph; `global_sweep_ext` runs end-to-end and produces a schema-valid
  artifact; `--assert-anchors` raises on a cache that cannot match the real
  anchors. All 4 checks passed. The existing `tests/test_e3_affinity_oracle.py`
  suite (33 tests) still passes unchanged.
- Real-cache GPU checks (see below for the exact commands): re-ran the
  existing, unmodified `global-sweep` command against the production cache
  and diffed its payload against the stored
  `ablationAll/talk2dino_e3_affinity_oracle/results/global_sweep.json`
  (ignoring `invocation` and timing fields) — PASS, byte-identical. Ran
  `global-sweep-ext --assert-anchors` against the production cache — PASS.
- `git diff --stat` confirms only `run_e3_affinity_oracle.py` and
  `src/e3_affinity_oracle.py` changed; nothing under
  `src/open_vocabulary_segmentation/models/`,
  `src/open_vocabulary_segmentation/configs/`, the E3 checkpoint, the E3
  config, or `/scratch/haree/talk2dino_e3_affinity_oracle/cache/` was
  touched.

## A3 answer (required by the task)

`affinity_power` (kappa=3.0) and `knn_k` (=12) **are baked into the cached
`knn_weights`/`knn_indices`** at cache-construction time, inside
`AffinityOracleCacheWriter.add_window` -> `build_knn_graph`
(`src/e3_affinity_oracle.py:773-806`, `:232-283`). Raw normalized patch
features are never persisted in the cache — `validate_cache_shard` rejects
any shard key containing `"feature"` — so kappa/k cannot be recomputed or
swept by replaying this cache; `propagate_scores` has no `affinity_power`/
`knn_k` parameter at all and only ever applies `alpha` and
`propagation_steps` (T) at replay time. **Sweeping kappa or k requires
rebuilding the cache (~2311 s per rebuild)** with a different
`OracleProtocol(affinity_power=..., knn_k=...)`; per requirement A3 this is
explicitly out of scope for Part A and is not attempted here. `--affinity-power`
and `--knn-k` flags were **not** added to `global-sweep-ext`. This is
recorded verbatim as `payload.kappa_k_sweep_limitation` in every
`global-sweep-ext` result artifact.

## RUN COMMANDS

```bash
# 1. Request the GPU node (interactive allocation; matches the account/
#    partition/GPU type used for every prior run in this project's job
#    history). Everything below runs inside this allocation, never on the
#    login node.
salloc --account=rrg-yangw_gpu --partition=gpubase_interac \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=02:00:00

# Inside the allocation:
cd /project/6114407/haree/Talk2DINO
module load python/3.11
source /scratch/haree/venv/talk2dino-a100/bin/activate

CACHE=/scratch/haree/talk2dino_e3_affinity_oracle/cache/full
BASELINE=/scratch/haree/talk2dino_e3_affinity_oracle/results/baseline_control.json
OUT=ablationAll/e10_adaptive_diffusion/results
mkdir -p "$OUT"

# 2. Fast anchor gate (~6 min): confirms alpha=0.00,T=10 and alpha=0.95,T=10
#    reproduce the recorded E3/global-sweep numbers to 1e-6 through the
#    SAME evaluate_cache()/propagate_scores() code path used below, before
#    spending an hour on the full grid. Already run once during
#    implementation (real cache, PASS) -- rerun to reconfirm on your own
#    allocation if you want an independent check.
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/global_sweep_ext_anchor_check.json" \
  --csv "$OUT/global_sweep_ext_anchor_check.csv" \
  --device cuda --assert-anchors
# Expected: prints "global-sweep-ext anchors OK: ..." and exits 0.
# Wall clock: ~6 min (2 full-cache evaluate_cache passes at T=10, ~163-180s
# each, per the recorded per-alpha cost in global_sweep.csv).

# 3. The extended alpha/T sweep itself: default grid is
#    alpha in {0.95,0.96,0.97,0.98,0.99,0.995,0.999} x steps in {10,20,40}
#    = 21 full 5000-image evaluations.
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/global_sweep_ext.json" \
  --csv "$OUT/global_sweep_ext.csv" \
  --device cuda
# Expected wall clock: ~65-100 min. Estimated by decomposing the recorded
# global-sweep cost: alpha=0 (no propagation loop) took 163.4s and
# alpha=0.95,T=10 took 178.1s, i.e. ~15s of the 178s is the T=10 propagation
# loop itself and ~163s is fixed per-config overhead (streaming the 8.3GB
# cache + confusion accounting), which does not scale with T. Extrapolating
# the loop cost linearly in T: T=10 ~178s/config, T=20 ~193s/config,
# T=40 ~223s/config; 7 alphas x (178+193+223)s ~= 4158s ~= 69 min. Treat
# this as an estimate, not a guarantee -- give the allocation headroom
# (the salloc above requests 2h).
# Expected output:
#   ablationAll/e10_adaptive_diffusion/results/global_sweep_ext.json
#   ablationAll/e10_adaptive_diffusion/results/global_sweep_ext.csv
# Peak GPU/CPU RAM: expected in the same range as the original global-sweep
# (~2.2GB GPU, ~2.0GB CPU) -- evaluate_cache streams shards, it does not
# load the 8.3GB cache into memory at once.

# 4. Read the verdict:
python3 - << 'PY'
import json
d = json.load(open("ablationAll/e10_adaptive_diffusion/results/global_sweep_ext.json"))
p = d["payload"]
print("optimum:", p["optimum"])
print("at_grid_edge:", p["optimum"]["at_grid_edge"])
if p["warning"]:
    print("WARNING:", p["warning"])
PY
# If at_grid_edge is true, the artifact says so in payload["warning"] --
# extend --alpha-grid and/or --steps (e.g. push past 0.999, or try T=80)
# and re-run step 3 with the wider grids before treating any number here
# as the true optimum.
```

## Notes

- `--baseline` must point at a `baseline` artifact whose
  `canonical_reported_precision_matches` is `true` (same guard as the
  existing `global-sweep` command) -- the recorded
  `/scratch/haree/talk2dino_e3_affinity_oracle/results/baseline_control.json`
  satisfies this.
- All values in `--alpha-grid` must be in `[0,1)` (propagate_scores'
  existing contract); `0.999` is valid, `1.0` is not.
- Rerunning either command without `--overwrite` against an existing output
  path raises `FileExistsError` by design (same atomic-write contract as
  every other subcommand in this file).
- Nothing under `src/open_vocabulary_segmentation/models/`,
  `src/open_vocabulary_segmentation/configs/`, the E3 checkpoint/config, or
  `/scratch/haree/talk2dino_e3_affinity_oracle/cache/` is read for writing
  or modified by this part -- the cache is opened read-only via
  `load_cache_manifest`/`load_cache_shard`.
- Part A intentionally does not sweep kappa/knn_k (see A3 answer above) and
  does not fold in per-patch adaptive alpha (that is Part B, which consumes
  this command's `--global-sweep` output... note: Part A's own output here
  is `global-sweep-ext`, a *new* artifact; Part B's spec (not yet given)
  says it "consumes Part A's `--global-sweep` artifact to initialise every
  bucket at α*" -- confirm in Part B's spec whether that means the original
  `global-sweep` artifact (alpha* from the canonical 0-0.95 grid) or this
  `global-sweep-ext` artifact's optimum, since the wording is ambiguous and
  should not be guessed.
