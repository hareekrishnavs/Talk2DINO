#!/bin/bash
set -euo pipefail
cd /project/6114407/haree/Talk2DINO
source /scratch/haree/venv/talk2dino-a100/bin/activate

CACHE=/scratch/haree/talk2dino_e3_affinity_oracle/cache/full
BASELINE=/scratch/haree/talk2dino_e3_affinity_oracle/results/baseline_control.json
OUT=/project/6114407/haree/Talk2DINO/ablationAll/e10_adaptive_diffusion/results
mkdir -p "$OUT"

echo "=== [Part A artifact for downstream identity checks] minimal grid ==="
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/global_sweep_ext_minimal.json" --csv "$OUT/global_sweep_ext_minimal.csv" \
  --alpha-grid 0.95 --steps 10 --device cuda

echo "=== [V3-B] local-stat-fit --n-buckets 1 (must reproduce global optimum) ==="
python3 run_e3_affinity_oracle.py local-stat-fit \
  --cache "$CACHE" --global-sweep "$OUT/global_sweep_ext_minimal.json" \
  --n-buckets 1 --alpha-grid 0.95 --max-sweeps 1 \
  --output "$OUT/local_stat_fit_n1.json" --csv "$OUT/local_stat_fit_n1.csv" \
  --device cuda

echo "=== [V3-C] bias-fit --bias-grid 0 (must reproduce global optimum) ==="
python3 run_e3_affinity_oracle.py bias-fit \
  --cache "$CACHE" --global-sweep "$OUT/global_sweep_ext_minimal.json" \
  --bias-grid 0 --max-sweeps 1 \
  --output "$OUT/bias_fit_zero.json" --csv "$OUT/bias_fit_zero.csv" \
  --device cuda

echo "=== [V6] determinism: rerun global-sweep-ext minimal grid twice, diff ==="
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/determinism_run1.json" --csv "$OUT/determinism_run1.csv" \
  --alpha-grid 0.95 --steps 10 --device cuda
python3 run_e3_affinity_oracle.py global-sweep-ext \
  --cache "$CACHE" --baseline "$BASELINE" \
  --output "$OUT/determinism_run2.json" --csv "$OUT/determinism_run2.csv" \
  --alpha-grid 0.95 --steps 10 --device cuda
python3 - "$OUT/determinism_run1.json" "$OUT/determinism_run2.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
def strip(d):
    d = json.loads(json.dumps(d))
    d.pop("invocation", None)
    for row in d["payload"]["rows"]:
        for k in ("runtime_seconds", "peak_cpu_ram_bytes", "peak_gpu_bytes"):
            row.pop(k, None)
    return d
sa, sb = strip(a), strip(b)
if sa == sb:
    print("V6 DIFF RESULT: IDENTICAL (deterministic, modulo timing/invocation)")
else:
    print("V6 DIFF RESULT: MISMATCH -- nondeterministic")
    raise SystemExit(1)
PY

echo "ALL REMAINDER CHECKS COMPLETE"
