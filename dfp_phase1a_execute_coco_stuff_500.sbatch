#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=dfp-phase1a-execute-coco-stuff-500
#SBATCH --output=%x-%j.out
#
# DFP Phase 1A Step 11: production execution launcher for the real
# COCO-Stuff frozen 500-image diagnostic (real GPU model inference).
#
# Resource request mirrors the closest verified prior COCO job in this
# repository, scripts/slurm/e12_k11_k12_eval_full_h100.sbatch (5000
# images, dual k11+k12 propagation per window, cpus=8/mem=80G/gpus=h100:1/
# time=5h) -- this run covers 1/10th as many images with a single
# propagation per window (DFP has no k11 comparator), so the same time
# budget leaves generous headroom for this driver's additional per-image
# analysis/component/compute-projection work. --account is a normal
# #SBATCH directive and, like any Slurm job, can be overridden per the
# scheduler's own mechanism at submission time without editing this file.
#
# This script never submits itself or any other job -- it contains no
# call to the scheduler at all. It is not submitted by this task; only
# validated with `bash -n` (syntax check).
#
# Required environment (never hardcoded in this file):
#   DFP_REPO_ROOT              path to this repository checkout
#   DFP_VENV_ACTIVATE_PATH     path to the Python virtualenv's
#                              bin/activate script (opencv module loaded
#                              separately, see DFP_OPENCV_MODULE below)
#   COCO_STUFF_REAL_DATA_ROOT  path to the real COCO-Stuff-164K dataset
#                              root (consumed directly by the execution
#                              driver's own preflight -- never read or
#                              interpreted by this script)
#   DFP_OUTPUT_ROOT            scratch output root; this run's checkpoint/
#                              result/qualitative artifacts are written
#                              under "$DFP_OUTPUT_ROOT/coco_stuff_500"
#                              (the same per-dataset subpath convention
#                              dfp_phase1a_preflight.sbatch already uses)
# Optional environment:
#   DFP_OPENCV_MODULE          environment-module name to load (default:
#                              opencv/4.14.0, the module every verified
#                              prior job in this repository loads)
#
# Required positional argument: exactly "fresh" or "resume" -- this
# script never picks one automatically and never falls back from one to
# the other. Example (submission mechanics deliberately omitted; use
# your site's normal scheduler submission command with this script as
# the target and MODE as its one argument):
#
#   export DFP_REPO_ROOT=/path/to/Talk2DINO
#   export DFP_VENV_ACTIVATE_PATH=/path/to/venv/bin/activate
#   export COCO_STUFF_REAL_DATA_ROOT=/path/to/coco_stuff164k
#   export DFP_OUTPUT_ROOT=/path/to/scratch/output
#   <scheduler submit command> dfp_phase1a_execute_coco_stuff_500.sbatch fresh

set -euo pipefail

MODE="${1:?usage: $0 <fresh|resume> -- exactly one required argument, no default}"
if [ "$MODE" != "fresh" ] && [ "$MODE" != "resume" ]; then
    echo "FATAL: mode must be exactly 'fresh' or 'resume' (no aliases), got: $MODE" >&2
    exit 1
fi

for var in DFP_REPO_ROOT DFP_VENV_ACTIVATE_PATH COCO_STUFF_REAL_DATA_ROOT DFP_OUTPUT_ROOT; do
    if [ -z "${!var:-}" ]; then
        echo "FATAL: \$$var is not set in the environment this job was submitted from." >&2
        echo "Export it and resubmit -- see the header comment in this script." >&2
        exit 1
    fi
done

cd "$DFP_REPO_ROOT"

module load "${DFP_OPENCV_MODULE:-opencv/4.14.0}"
source "$DFP_VENV_ACTIVATE_PATH"

RUN_OUTPUT_ROOT="$DFP_OUTPUT_ROOT/coco_stuff_500"
mkdir -p "$RUN_OUTPUT_ROOT"

echo "=== CPU preflight (COCO-Stuff-500, mode=$MODE) ==="
python diagnostics/run_dfp_phase1a.py preflight \
    --run coco_stuff_500 \
    --mode "$MODE" \
    --output-root "$RUN_OUTPUT_ROOT" \
    --probe-output-writeable

echo
echo "=== Execution (COCO-Stuff-500, mode=$MODE, device=cuda) ==="
python diagnostics/run_dfp_phase1a.py execute \
    --run coco_stuff_500 \
    --mode "$MODE" \
    --output-root "$RUN_OUTPUT_ROOT" \
    --device cuda

echo
echo "DFP PHASE 1A COCO-STUFF-500 EXECUTION LAUNCHER COMPLETE (mode=$MODE)"
