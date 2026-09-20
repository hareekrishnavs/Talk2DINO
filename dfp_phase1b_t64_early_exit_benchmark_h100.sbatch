#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=dfp-phase1b-t64-early-exit-benchmark
#SBATCH --output=%x-%j.out
#
# DFP Phase 1B: T64-versus-T320 early-exit wall-clock benchmark
# launcher, E26-authorized. The scientific/operational contract is
# frozen, unmodified, by E25.
#
# **This launches an OPERATIONAL WALL-CLOCK BENCHMARK ONLY.** It never
# accesses GT, never runs the 150-image confirmation evaluation, never
# runs the held 5,000-image evaluation, and a PROCEED outcome only
# authorizes freezing a SEPARATE T64 confirmation protocol -- it does
# not itself confirm T64, select a final method, or make a deployment
# claim. See docs/dfp_phase1b_t64_early_exit_benchmark_run.md for the
# complete account.
#
# This script never submits itself or any other job -- it contains no
# call to the scheduler at all, and no literal "sbatch" invocation
# anywhere in it. It is not submitted by the implementation task that
# created it; only validated with `bash -n` (syntax check).
#
# No confirmation, full-5000-image, or GT-access command appears
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
#                               root
#   TALK2DINO_WEIGHT_DIR        path to the real DINOv2 backbone/CLIP
#                               checkpoint directory
#   DFP_BENCHMARK_OUTPUT_ROOT   scratch output root; a FRESH
#                               subdirectory is created under this root
#                               for this run (job ID suffixed) -- an
#                               existing destination is rejected
# Optional environment:
#   DFP_OPENCV_MODULE           environment-module name to load
#                               (default: opencv/4.14.0)
#
# Resume policy: this is a small, eight-trial (2 warm-up + 6 measured)
# benchmark -- no resume policy is implemented. An existing output
# destination is always rejected; re-run into a fresh
# DFP_BENCHMARK_OUTPUT_ROOT/<job id> directory instead.
#
# Example (submission mechanics deliberately omitted; use your site's
# normal scheduler submission command with this script as the target):
#
#   export DFP_REPO_ROOT=/path/to/Talk2DINO
#   export DFP_VENV_ACTIVATE_PATH=/path/to/venv/bin/activate
#   export COCO_STUFF_REAL_DATA_ROOT=/path/to/coco_stuff164k
#   export TALK2DINO_WEIGHT_DIR=/path/to/weights
#   export DFP_BENCHMARK_OUTPUT_ROOT=/path/to/scratch/benchmark_output
#   <scheduler submit command> dfp_phase1b_t64_early_exit_benchmark_h100.sbatch

set -euo pipefail

for var in DFP_REPO_ROOT DFP_VENV_ACTIVATE_PATH COCO_STUFF_REAL_DATA_ROOT TALK2DINO_WEIGHT_DIR DFP_BENCHMARK_OUTPUT_ROOT; do
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
echo "authority: evaluation_identities/e25_dfp_phase1b_t64_early_exit_benchmark.toml (contract), e26 (implementation)"
echo "This run measures operational wall-clock performance only. It never"
echo "accesses GT, never runs the confirmation or full-5000-image evaluation,"
echo "and a PROCEED outcome only authorizes freezing a SEPARATE T64"
echo "confirmation protocol."

if [ -n "$(git -C "$DFP_REPO_ROOT" status --porcelain)" ]; then
    echo "FATAL: repository checkout at \$DFP_REPO_ROOT is not clean -- this launcher requires a clean committed checkout." >&2
    git -C "$DFP_REPO_ROOT" status --porcelain >&2
    exit 1
fi

module load "${DFP_OPENCV_MODULE:-opencv/4.14.0}"
source "$DFP_VENV_ACTIVATE_PATH"

CLI="python diagnostics/run_dfp_phase1b_t64_early_exit_benchmark.py"

echo
echo "=== CPU preflight (E25 + E26 protocol + benchmark manifest + git"
echo "    ancestry/clean-worktree verification) -- no torch, no CUDA,"
echo "    no dataset access ==="
$CLI preflight

RUN_OUTPUT_ROOT="$DFP_BENCHMARK_OUTPUT_ROOT/${SLURM_JOB_ID:-local}"
if [ -e "$RUN_OUTPUT_ROOT" ]; then
    echo "FATAL: output destination $RUN_OUTPUT_ROOT already exists -- refusing to overwrite (no resume policy for this small benchmark)." >&2
    exit 1
fi
mkdir -p "$RUN_OUTPUT_ROOT"

echo
echo "=== Two warm-up trials (one per variant), fresh processes, run"
echo "    before the measured matrix; excluded from summary statistics ==="

WARMUP_TRIAL_FILES=()
for variant in T64_early_exit T320_reference; do
    warmup_output="$RUN_OUTPUT_ROOT/warmup_${variant}.json"
    echo
    echo "--- warm-up: variant=$variant ---"
    $CLI run-trial --device cuda --variant "$variant" --repeat 0 --warmup --output "$warmup_output"
    echo
    echo "--- independently verifying warm-up trial ($variant) ---"
    $CLI verify-trial --trial-file "$warmup_output"
    WARMUP_TRIAL_FILES+=("$warmup_output")
done

echo
echo "=== Six fresh-process measured trials, frozen counterbalanced schedule ==="

declare -a TRIAL_INDEX=(0 1 2 3 4 5)
declare -a VARIANT=(T320_reference T64_early_exit T64_early_exit T320_reference T320_reference T64_early_exit)
declare -a REPEAT=(0 0 1 1 2 2)

MEASURED_TRIAL_FILES=()
for i in 0 1 2 3 4 5; do
    ti="${TRIAL_INDEX[$i]}"
    variant="${VARIANT[$i]}"
    rp="${REPEAT[$i]}"
    trial_output="$RUN_OUTPUT_ROOT/trial_${ti}_${variant}_repeat${rp}.json"
    echo
    echo "--- trial $ti: variant=$variant repeat=$rp ---"
    $CLI run-trial --device cuda --variant "$variant" --trial-index "$ti" --repeat "$rp" --output "$trial_output"
    echo
    echo "--- independently verifying trial $ti ---"
    $CLI verify-trial --trial-file "$trial_output"
    MEASURED_TRIAL_FILES+=("$trial_output")
done

echo
echo "=== Merging all eight independently-verified trials (CPU-only) and"
echo "    installing the merged result + decision as one pair-atomic"
echo "    bundle directory (never a result without its decision, or"
echo "    vice versa -- a single atomic directory rename either leaves"
echo "    the destination absent or a complete, cross-verified pair) ==="
MERGE_ARGS=()
for f in "${MEASURED_TRIAL_FILES[@]}"; do
    MERGE_ARGS+=(--trial "$f")
done
for f in "${WARMUP_TRIAL_FILES[@]}"; do
    MERGE_ARGS+=(--warmup-trial "$f")
done
BUNDLE_DIR="$RUN_OUTPUT_ROOT/result"
$CLI merge "${MERGE_ARGS[@]}" --output-dir "$BUNDLE_DIR"
MERGED_OUTPUT="$BUNDLE_DIR/merged_result.json"
DECISION_OUTPUT="$BUNDLE_DIR/decision.json"

echo
echo "=== Independently verifying the merged result ==="
$CLI verify-result --result "$MERGED_OUTPUT"

echo
echo "=== Independently verifying the decision artifact against the"
echo "    merged result ==="
$CLI verify-decision --decision "$DECISION_OUTPUT" --result "$MERGED_OUTPUT"

echo
echo "DFP PHASE 1B T64 EARLY-EXIT BENCHMARK LAUNCHER COMPLETE"
echo "merged result: $MERGED_OUTPUT"
echo "decision: $DECISION_OUTPUT"
echo
echo "=== Authorization boundary (E26) ==="
echo "runner_implementation_authorized=true cli_implementation_authorized=true"
echo "launcher_implementation_authorized=true cpu_only_synthetic_testing_authorized=true"
echo "real_dataset_access_authorized=false (this run itself is the authorized real access)"
echo "confirmation_execution_authorized=false full_5000_execution_authorized=false gt_access_authorized=false"
echo "A PROCEED outcome authorizes only freezing a separate T64 confirmation protocol -- it does"
echo "NOT itself run confirmation, select a final method, or make a deployment/latency claim beyond"
echo "this measured H100 contract. See docs/dfp_phase1b_t64_early_exit_benchmark_run.md."
