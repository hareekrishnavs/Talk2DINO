#!/bin/bash
#SBATCH --account=rrg-yangw_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=e10_partE_smoke
#SBATCH --output=/scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/partE_smoke_%j.log

set -euo pipefail
cd /project/6114407/haree/Talk2DINO
source /scratch/haree/venv/talk2dino-a100/bin/activate

OUT=/scratch/haree/talk2dino_e3_affinity_oracle/feature_capture_smoke50
rm -rf "$OUT"  # smoke test only, safe to clobber between attempts

python3 -u capture_dino_features.py \
  --split val --output-dir "$OUT" \
  --shard-size 500 --device cuda --limit 50

echo "SMOKE TEST COMPLETE"
