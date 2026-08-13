#!/bin/bash
set -euo pipefail
cd /project/6114407/haree/Talk2DINO
source /scratch/haree/venv/talk2dino-a100/bin/activate

CACHE=/scratch/haree/talk2dino_e3_affinity_oracle/cache/full
SPLIT_GLOBAL=/project/6114407/haree/Talk2DINO/ablationAll/talk2dino_e3_affinity_oracle/results/split_half_global_A.json
GLOBAL_SWEEP=/project/6114407/haree/Talk2DINO/ablationAll/e10_adaptive_diffusion/results/global_sweep_ext_minimal.json
TEXT_EMBEDDING=/project/6114407/haree/Talk2DINO/ablationAll/e10_adaptive_diffusion/results/text_embedding.pt
OUT=/project/6114407/haree/Talk2DINO/ablationAll/e10_adaptive_diffusion/results
mkdir -p "$OUT"

echo "=== [V5-C] split-half --stage bias, minimal grid (real cache, real GPU-verified text embedding) ==="
python3 run_e3_affinity_oracle.py split-half --stage bias \
  --cache "$CACHE" --split-global "$SPLIT_GLOBAL" --global-sweep "$GLOBAL_SWEEP" \
  --text-embedding "$TEXT_EMBEDDING" --bias-grid "0,0.5" --max-sweeps 1 \
  --output "$OUT/split_half_bias_review.json" --csv "$OUT/split_half_bias_review.csv" \
  --device cuda

python3 - "$OUT/split_half_bias_review.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d["payload"]
print("A_fingerprint:", p["A_fingerprint"])
print("B_fingerprint:", p["B_fingerprint"])
print("regression_class_count:", p["regression_class_count"])
print("excluded_classes_no_A_support:", p["excluded_classes_no_A_support"])
print("text_cv.r2:", p["text_cv"]["r2"], "spearman:", p["text_cv"]["spearman"])
print("shuffled_target_control_cv.r2:", p["shuffled_target_control_cv"]["r2"])
print("shuffled_control_near_zero:", p["shuffled_control_near_zero"])
print("gain_over_global_B:", p["gain_over_global_B"])
print("transfer_delta_mIoU:", p["transfer_delta_mIoU"])
print("decision:", p["decision"])
print("diagnostic_only:", p["diagnostic_only"])
PY
echo "V5-C CHECK COMPLETE"
