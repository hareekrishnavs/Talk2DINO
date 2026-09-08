#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=dfp-phase1b-chunk-benchmark
#SBATCH --output=%x-%j.out
#
# DFP Phase 1B Task 3: fixed component-chunk-size (16/32/64) performance
# benchmark launcher, E17-authorized.
#
# **This launches an OPERATIONAL PERFORMANCE BENCHMARK ONLY.** It
# selects no chunk size, freezes no numeric conservation tolerance, and
# a successful merged result does NOT authorize the full Phase 1B run
# or the held 5,000-image evaluation -- see
# docs/dfp_phase1b_chunk_benchmark.md for the complete account.
#
# This script never submits itself or any other job -- it contains no
# call to the scheduler at all, and no literal "sbatch" invocation
# anywhere in it. It is not submitted by the implementation task that
# created it; only validated with `bash -n` (syntax check).
#
# No full calibration, full Phase 1B, or 5,000-image command appears
# anywhere in this script.
#
# Required environment (never hardcoded in this file):
#   DFP_REPO_ROOT               path to this repository checkout, at
#                               the exact committed HEAD this benchmark
#                               is authorized against (this script
#                               requires a clean committed checkout --
#                               see the check below)
#   DFP_VENV_ACTIVATE_PATH      path to the Python virtualenv's
#                               bin/activate script (opencv module
#                               loaded separately, see DFP_OPENCV_MODULE)
#   COCO_STUFF_REAL_DATA_ROOT   path to the real COCO-Stuff-164K dataset
#                               root (same *_REAL_DATA_ROOT convention
#                               the calibration launcher already uses)
#   TALK2DINO_WEIGHT_DIR        path to the real DINOv2 backbone/CLIP
#                               checkpoint directory (same convention as
#                               the calibration launcher)
#   DFP_CALIBRATION_ARTIFACT_PATH
#                               path to the completed, independently
#                               verified phase1b_conservation_calibration.json
#   DFP_WORKLOAD_BUNDLE_DIR      path to the independently-verified Task 2
#                               v2 workload-report bundle directory
#   DFP_BENCHMARK_MANIFEST_PATH  path to the frozen 24-window benchmark
#                               manifest (normally
#                               evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json
#                               inside $DFP_REPO_ROOT)
#   DFP_BENCHMARK_OUTPUT_ROOT    scratch output root; a FRESH subdirectory
#                               is created under this root for this run
#                               (job ID suffixed) -- an existing
#                               destination is rejected (see below)
# Optional environment:
#   DFP_OPENCV_MODULE           environment-module name to load
#                               (default: opencv/4.14.0)
#
# Resume policy: this is a small, six-trial benchmark -- no resume
# policy is implemented (none has been separately tested). An existing
# output destination is always rejected; re-run into a fresh
# DFP_BENCHMARK_OUTPUT_ROOT/<job id> directory instead.
#
# Example (submission mechanics deliberately omitted; use your site's
# normal scheduler submission command with this script as the target):
#
#   export DFP_REPO_ROOT=/path/to/Talk2DINO
#   export DFP_VENV_ACTIVATE_PATH=/path/to/venv/bin/activate
#   export COCO_STUFF_REAL_DATA_ROOT=/path/to/coco_stuff164k
#   export TALK2DINO_WEIGHT_DIR=/path/to/weights
#   export DFP_CALIBRATION_ARTIFACT_PATH=/path/to/phase1b_conservation_calibration.json
#   export DFP_WORKLOAD_BUNDLE_DIR=/path/to/coco_stuff_200_workload_stats_v2
#   export DFP_BENCHMARK_MANIFEST_PATH=/path/to/Talk2DINO/evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json
#   export DFP_BENCHMARK_OUTPUT_ROOT=/path/to/scratch/benchmark_output
#   <scheduler submit command> dfp_phase1b_chunk_benchmark_h100.sbatch

set -euo pipefail

for var in DFP_REPO_ROOT DFP_VENV_ACTIVATE_PATH COCO_STUFF_REAL_DATA_ROOT TALK2DINO_WEIGHT_DIR \
           DFP_CALIBRATION_ARTIFACT_PATH DFP_WORKLOAD_BUNDLE_DIR DFP_BENCHMARK_MANIFEST_PATH \
           DFP_BENCHMARK_OUTPUT_ROOT; do
    if [ -z "${!var:-}" ]; then
        echo "FATAL: \$$var is not set in the environment this job was submitted from." >&2
        echo "Export it and resubmit -- see the header comment in this script." >&2
        exit 1
    fi
done

cd "$DFP_REPO_ROOT"

echo "=== Repository / authority identity ==="
BRANCH="$(git -C "$DFP_REPO_ROOT" rev-parse --abbrev-ref HEAD)"
HEAD_COMMIT="$(git -C "$DFP_REPO_ROOT" rev-parse HEAD)"
echo "branch: $BRANCH"
echo "commit: $HEAD_COMMIT"
echo "authority: evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml"
echo "This run measures operational performance only -- it does not select"
echo "or apply a numeric conservation tolerance, does not choose a chunk"
echo "size, and does not authorize the full Phase 1B run or the 5,000-image"
echo "evaluation."

if [ -n "$(git -C "$DFP_REPO_ROOT" status --porcelain)" ]; then
    echo "FATAL: repository checkout at \$DFP_REPO_ROOT is not clean -- this launcher requires a clean committed checkout." >&2
    git -C "$DFP_REPO_ROOT" status --porcelain >&2
    exit 1
fi

module load "${DFP_OPENCV_MODULE:-opencv/4.14.0}"
source "$DFP_VENV_ACTIVATE_PATH"

CLI="python diagnostics/run_dfp_phase1b_chunk_benchmark.py"

echo
echo "=== CPU preflight (E17 protocol + benchmark manifest + source"
echo "    calibration artifact verification) -- no torch, no CUDA,"
echo "    no dataset access ==="
$CLI preflight \
    --artifact "$DFP_CALIBRATION_ARTIFACT_PATH" \
    --manifest "$DFP_BENCHMARK_MANIFEST_PATH" \
    --workload-bundle-dir "$DFP_WORKLOAD_BUNDLE_DIR"

RUN_OUTPUT_ROOT="$DFP_BENCHMARK_OUTPUT_ROOT/${SLURM_JOB_ID:-local}"
if [ -e "$RUN_OUTPUT_ROOT" ]; then
    echo "FATAL: output destination $RUN_OUTPUT_ROOT already exists -- refusing to overwrite (no resume policy for this small benchmark)." >&2
    exit 1
fi
mkdir -p "$RUN_OUTPUT_ROOT"

echo
echo "=== Six fresh-process trials, frozen counterbalanced schedule ==="

declare -a TRIAL_INDEX=(0 1 2 3 4 5)
declare -a REPEAT=(0 0 0 1 1 1)
declare -a CHUNK_SIZE=(16 32 64 64 32 16)

TRIAL_FILES=()
for i in 0 1 2 3 4 5; do
    ti="${TRIAL_INDEX[$i]}"
    rp="${REPEAT[$i]}"
    cs="${CHUNK_SIZE[$i]}"
    trial_output="$RUN_OUTPUT_ROOT/trial_${ti}_repeat${rp}_chunk${cs}.json"
    echo
    echo "--- trial $ti: repeat=$rp chunk_size=$cs ---"
    $CLI run-trial \
        --artifact "$DFP_CALIBRATION_ARTIFACT_PATH" \
        --manifest "$DFP_BENCHMARK_MANIFEST_PATH" \
        --workload-bundle-dir "$DFP_WORKLOAD_BUNDLE_DIR" \
        --device cuda \
        --trial-index "$ti" \
        --repeat "$rp" \
        --chunk-size "$cs" \
        --output "$trial_output"
    echo
    echo "--- independently verifying trial $ti ---"
    $CLI verify-trial --trial-file "$trial_output"
    TRIAL_FILES+=("$trial_output")
done

echo
echo "=== Merging all six independently-verified trials (CPU-only) ==="
MERGE_ARGS=()
for f in "${TRIAL_FILES[@]}"; do
    MERGE_ARGS+=(--trial "$f")
done
MERGED_OUTPUT="$RUN_OUTPUT_ROOT/merged_result.json"
$CLI merge "${MERGE_ARGS[@]}" --output "$MERGED_OUTPUT"

echo
echo "=== Independently verifying the merged result ==="
$CLI verify-result --result "$MERGED_OUTPUT"

echo
echo "DFP PHASE 1B CHUNK BENCHMARK LAUNCHER COMPLETE"
echo "merged result: $MERGED_OUTPUT"
echo "This is an operational performance benchmark only. It selects no"
echo "chunk size, freezes no tolerance, and does NOT authorize the full"
echo "Phase 1B run or the 5,000-image evaluation -- see"
echo "docs/dfp_phase1b_chunk_benchmark.md."
