#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=dfp-phase1b-balp-development-350
#SBATCH --output=%x-%j.out
#
# DFP Phase 1B: the real 350-image BALP development execution,
# E23-authorized (evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml).
#
# *** ONLY the frozen 350-image development split is authorized. ***
# This launcher does NOT run the 150-image confirmation split, does NOT
# touch the other 4,500 validation images, does NOT run a full
# 5,000-image evaluation, does NOT run exact masked-forward provenance,
# does NOT run candidate-specific recurrence, and NEVER overwrites a
# completed development result. GPU: exactly 1x H100 80GB. Wall time:
# at most 05:00:00 (a hard ceiling -- SLURM may terminate this job at
# that limit; the last completed checkpoint generation remains valid
# and the next authorized run must use --mode resume).
#
# This script never submits itself or any other job -- it contains no
# "sbatch" invocation anywhere in it, and is NOT submitted by the task
# that created it; see the implementation report for the current
# authorization state before submitting this for real.
#
# CPU/host-memory resources reuse the repository's own established
# H100-class launcher precedent (dfp_phase1b_chunk_benchmark_h100.sbatch:
# --cpus-per-task=8, --mem=80G, --account=rrg-yangw) -- never invented.
#
# Required environment (never hardcoded here):
#   DFP_REPO_ROOT               path to this repository checkout, at the
#                               exact committed HEAD this run is authorized
#                               against (this script requires a clean
#                               committed checkout -- see the check below)
#   DFP_VENV_ACTIVATE_PATH      path to the Python virtualenv's
#                               bin/activate script
#   COCO_STUFF_REAL_DATA_ROOT   path to the real COCO-Stuff-164K dataset
#                               root (containing images/val2017 and
#                               annotations/val2017)
#   TALK2DINO_WEIGHT_DIR        path to the real DINOv2 backbone/CLIP
#                               checkpoint directory (same convention
#                               already used by every other committed
#                               production launcher in this repository)
#   DFP_BALP_OUTPUT_ROOT        absolute scratch output root for this
#                               development run -- an EXISTING completed
#                               (finalized) result is never overwritten;
#                               resume against the SAME root to continue
#                               an interrupted run
#   DFP_BALP_MODE                "fresh" for a first run, "resume" to
#                               continue an interrupted run (never
#                               defaulted -- must be set explicitly)
# Optional environment:
#   DFP_OPENCV_MODULE           environment-module name to load
#                               (default: opencv/4.14.0)
#
# Example (submission mechanics deliberately omitted; use your site's
# normal scheduler submission command with this script as the target):
#
#   export DFP_REPO_ROOT=/path/to/Talk2DINO
#   export DFP_VENV_ACTIVATE_PATH=/path/to/venv/bin/activate
#   export COCO_STUFF_REAL_DATA_ROOT=/path/to/coco_stuff164k
#   export TALK2DINO_WEIGHT_DIR=/path/to/weights
#   export DFP_BALP_OUTPUT_ROOT=/path/to/scratch/balp_development_output
#   export DFP_BALP_MODE=fresh
#   <scheduler submit command> dfp_phase1b_balp_development_coco_stuff_350.sbatch

set -euo pipefail

for var in DFP_REPO_ROOT DFP_VENV_ACTIVATE_PATH COCO_STUFF_REAL_DATA_ROOT TALK2DINO_WEIGHT_DIR DFP_BALP_OUTPUT_ROOT DFP_BALP_MODE; do
    if [ -z "${!var:-}" ]; then
        echo "FATAL: \$$var is not set in the environment this job was submitted from." >&2
        echo "Export it and resubmit -- see the header comment in this script." >&2
        exit 1
    fi
done

if [ "$DFP_BALP_MODE" != "fresh" ] && [ "$DFP_BALP_MODE" != "resume" ]; then
    echo "FATAL: \$DFP_BALP_MODE must be exactly 'fresh' or 'resume', observed '$DFP_BALP_MODE'" >&2
    exit 1
fi

case "$DFP_BALP_OUTPUT_ROOT" in
    /*) ;;
    *) echo "FATAL: \$DFP_BALP_OUTPUT_ROOT must be an absolute path" >&2; exit 1 ;;
esac

cd "$DFP_REPO_ROOT"

echo "=== Repository / authority identity ==="
HEAD_AT_SUBMIT_TIME_COMMIT="$(git -C "$DFP_REPO_ROOT" rev-parse HEAD)"
echo "committed HEAD: $HEAD_AT_SUBMIT_TIME_COMMIT"
echo "branch: $(git -C "$DFP_REPO_ROOT" rev-parse --abbrev-ref HEAD)"
echo "authority: evaluation_identities/e23_dfp_phase1b_balp_development_execution.toml"
echo
echo "*** ONLY the frozen 350-image BALP development split is authorized. ***"
echo "*** No confirmation split, no other-4500 images, no 5,000-image evaluation. ***"

if [ -n "$(git -C "$DFP_REPO_ROOT" status --porcelain)" ]; then
    echo "FATAL: repository checkout at \$DFP_REPO_ROOT is not clean -- this launcher requires a clean committed checkout." >&2
    git -C "$DFP_REPO_ROOT" status --porcelain >&2
    exit 1
fi

module load "${DFP_OPENCV_MODULE:-opencv/4.14.0}"
source "$DFP_VENV_ACTIVATE_PATH"

CLI="python diagnostics/run_dfp_phase1b_balp_development.py"

echo
echo "=== 1/7: print exact committed HEAD ==="
echo "$HEAD_AT_SUBMIT_TIME_COMMIT"

echo
echo "=== 2/7: verify E23 protocol (torch-free) ==="
$CLI verify-protocol

echo
echo "=== 3/7: verify real source binding (torch-free) ==="
$CLI verify-source-binding

echo
echo "=== 4/7: verify real model binding (no torch, no model construction) ==="
$CLI verify-model-binding

echo
echo "=== 5/7: $DFP_BALP_MODE preflight (torch-free) ==="
$CLI preflight --output-root "$DFP_BALP_OUTPUT_ROOT"

CURRENT_HEAD="$(git -C "$DFP_REPO_ROOT" rev-parse HEAD)"
if [ "$CURRENT_HEAD" != "$HEAD_AT_SUBMIT_TIME_COMMIT" ]; then
    echo "FATAL: repository HEAD changed during preflight ($HEAD_AT_SUBMIT_TIME_COMMIT -> $CURRENT_HEAD) -- refusing to execute." >&2
    exit 1
fi

echo
echo "=== 6/7: execute (--mode $DFP_BALP_MODE --device cuda) ==="
$CLI execute --mode "$DFP_BALP_MODE" --device cuda --output-root "$DFP_BALP_OUTPUT_ROOT"

echo
echo "=== 7a/7: verify checkpoint ==="
$CLI verify-checkpoint --output-root "$DFP_BALP_OUTPUT_ROOT"

echo
echo "=== 7b/7: verify final development result ==="
$CLI verify-development-result --output-root "$DFP_BALP_OUTPUT_ROOT"

echo
echo "DFP PHASE 1B BALP 350-IMAGE DEVELOPMENT EXECUTION LAUNCHER COMPLETE"
echo "output root: $DFP_BALP_OUTPUT_ROOT"
echo "This authorized ONLY the 350-image development split. It did not run the"
echo "150-image confirmation split, did not touch the other 4,500 validation"
echo "images, and did not run a full 5,000-image evaluation."
