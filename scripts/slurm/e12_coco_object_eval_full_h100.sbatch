#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=e12-coco-object-eval-full
#SBATCH --output=/scratch/haree/e12_coco_object_evaluation/slurm-full-%j.out
#SBATCH --error=/scratch/haree/e12_coco_object_evaluation/slurm-full-%j.err
#
# COCO-Object protocol confirmation: E3 vs matched k11 vs k12, T=320
# finite-step, full run mode (all 5000 canonical val2017 images). Same-
# data protocol confirmation on the existing COCO images/annotations, not
# an independent dataset. Requires the verified COCO-Object val2017
# materialization manifest (MATERIALIZATION_MANIFEST below) -- never runs
# conversion itself. Never routes through main.py --eval/multi_gpu_test,
# never runs T4/DCR/SUR, cross-view edge support, structural pruning,
# alternative stitching, or learned parameters. Fails closed at every
# preflight step before touching the GPU. This script requires SEPARATE,
# later authorization beyond the pilot20 round -- do not submit until
# pilot20 and pilot100 have completed and been reviewed.
#
# Submission directory (/scratch/haree/e12_coco_object_evaluation) must
# already exist before `sbatch` is invoked -- this script does not create
# it, and never submits itself or any other job.

set -euo pipefail

echo "=== e12 COCO-Object protocol confirmation: full ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="/project/6114407/haree/Talk2DINO"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="/scratch/haree/venv/talk2dino-a100"
OUTPUT_DIR="/scratch/haree/e12_coco_object_evaluation"
MATERIALIZATION_MANIFEST="${MATERIALIZATION_MANIFEST:-/scratch/haree/coco_object_protocol/manifests/manifest-20443250.json}"
DATA_ROOT="/scratch/haree/coco_object_protocol"
SOURCE_MASKS="/scratch/haree/coco_stuff164k/annotations/val2017"
SOURCE_IMAGES="/scratch/haree/coco_stuff164k/images/val2017"

export TALK2DINO_WEIGHT_DIR="/scratch/haree/weights"

cd "${REPO_ROOT}"

echo "Git commit:    $(git rev-parse HEAD)"
echo "Git branch:    $(git branch --show-current)"

if [ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]; then
    echo "FATAL: expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)" >&2
    exit 1
fi

# Reject TRACKED changes only (staged or unstaged) -- unrelated untracked
# files elsewhere in the repo are fine and must not block submission. This
# job must run exactly the committed evaluator, never an uncommitted
# tracked edit.
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "FATAL: tracked worktree has uncommitted changes; commit or stash before submitting" >&2
    echo "Changed tracked files:" >&2
    git diff --name-only >&2
    git diff --cached --name-only >&2
    exit 1
fi

for required_file in \
    "evaluation_identities/e12_coco_object_protocol_confirmation.toml" \
    "evaluation_identities/e12_matched_k11_k12_t320.toml" \
    "evaluation_identities/e12_coco_object_val_materialization.toml" \
    "src/coco_object_protocol_confirmation_identity.py" \
    "src/coco_object_protocol_confirmation_checkpoint.py" \
    "src/coco_object_protocol_confirmation_report.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/coco_object_evaluator.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/matched_power_evaluator.py" \
    "diagnostics/run_coco_object_protocol_confirmation.py" \
    "verify_coco_object_protocol_confirmation.py" \
    "verify_coco_object_val_materialization.py"
do
    if ! git ls-files --error-unmatch "${required_file}" > /dev/null 2>&1; then
        echo "FATAL: ${required_file} is not committed; refusing to run an uncommitted evaluator" >&2
        exit 1
    fi
done

echo "=== activating environment ==="
module load opencv/4.14.0
source "${VENV}/bin/activate"
PYTHON="${VENV}/bin/python"

echo "Python:        $(${PYTHON} --version)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "Materialization manifest: ${MATERIALIZATION_MANIFEST}"
echo "Data root:     ${DATA_ROOT}"

mkdir -p "${OUTPUT_DIR}"

echo "=== preflights (static + real-data checks, before any CUDA/model work) ==="
"${PYTHON}" verify_e3_identity.py preflight
"${PYTHON}" verify_rwr_reproduction.py preflight
"${PYTHON}" verify_matched_k11_k12.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_stability.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_power_evaluation.py preflight --repo-root .
"${PYTHON}" verify_stitching_control_suite.py preflight --repo-root .
"${PYTHON}" verify_native_edge_support_audit.py preflight --repo-root .
"${PYTHON}" verify_native_edge_support_reachability_gate.py preflight --repo-root .
"${PYTHON}" verify_coco_object_val_materialization.py preflight \
    --repo-root . --source-masks "${SOURCE_MASKS}" --source-images "${SOURCE_IMAGES}" --output-root "${DATA_ROOT}"
"${PYTHON}" verify_coco_object_protocol_confirmation.py preflight \
    --repo-root . --materialization-manifest "${MATERIALIZATION_MANIFEST}" --data-root "${DATA_ROOT}" \
    --source-masks "${SOURCE_MASKS}" --source-images "${SOURCE_IMAGES}"

echo "=== running the full (5000-image) COCO-Object protocol confirmation ==="
RESULT_PATH="${OUTPUT_DIR}/result-full-${SLURM_JOB_ID:-manual}.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-full-${SLURM_JOB_ID:-manual}.json"
PER_IMAGE_STATS_PATH="${OUTPUT_DIR}/per-image-stats-full-${SLURM_JOB_ID:-manual}.json"

# Job-specific paths above already prevent accidental reuse of a previous
# job's checkpoint/result -- --resume is deliberately never passed here.
set +e
"${PYTHON}" diagnostics/run_coco_object_protocol_confirmation.py \
    --repo-root . \
    --materialization-manifest "${MATERIALIZATION_MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --source-masks "${SOURCE_MASKS}" \
    --source-images "${SOURCE_IMAGES}" \
    --run-mode full \
    --checkpoint "${CHECKPOINT_PATH}" \
    --result "${RESULT_PATH}" \
    --per-image-stats "${PER_IMAGE_STATS_PATH}" \
    --device cuda
EVAL_STATUS=$?
set -e

if [ ${EVAL_STATUS} -ne 0 ]; then
    echo "FATAL: full protocol confirmation exited with status ${EVAL_STATUS}" >&2
    echo "Checkpoint and any partial output are preserved at:" >&2
    echo "  ${CHECKPOINT_PATH}" >&2
    echo "  ${RESULT_PATH}" >&2
    exit "${EVAL_STATUS}"
fi

echo "=== verifying the result ==="
"${PYTHON}" verify_coco_object_protocol_confirmation.py verify-result --result "${RESULT_PATH}" --repo-root .

echo "=== runtime and peak memory ==="
"${PYTHON}" -c "
import json
with open('${RESULT_PATH}') as f:
    r = json.load(f)
print(f\"phase_runtime_seconds.total: {r['phase_runtime_seconds']['total']:.3f}\")
print(f\"peak_gpu_memory_bytes: {r['peak_gpu_memory_bytes']}\")
"

date -u +"End (UTC): %Y-%m-%dT%H:%M:%SZ"
echo "Result:        ${RESULT_PATH}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Per-image stats: ${PER_IMAGE_STATS_PATH}"
echo "=== done ==="
