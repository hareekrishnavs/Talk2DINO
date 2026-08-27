#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=e12-ade20k-eval-pilot20
#
# ADE20K C150 matched evaluator: E3 vs matched k11 vs k12, T=320
# finite-step, pilot20 run mode (first 20 canonical validation images).
# Cross-dataset evaluation of the frozen COCO-2017 Talk2DINO bridge
# against the standard 150-class ADE20K validation protocol -- no
# background class, no C151 variant, no annotation conversion. Requires
# a verified ADE20K dataset-source manifest (ADE20K_SOURCE_MANIFEST
# below) -- never runs conversion, never modifies the dataset. Never
# runs the full 2000-image evaluation, never routes through
# main.py --eval/multi_gpu_test, never runs T4/DCR/SUR, cross-view edge
# support, structural pruning, alternative stitching, or learned
# parameters. Fails closed at every preflight step before touching the
# GPU. Only this pilot20 script may be submitted after independent
# verification -- pilot100/full require separate, later authorization.
#
# --account is overridable at submission time: sbatch --account=<acct> ...
# --output/--error are intentionally left to SLURM's own default
# (slurm-<jobid>.out in the submission directory) so this tracked script
# never hardcodes a private filesystem path; override with
# sbatch --output=... --error=... if a different location is wanted.
#
# Required environment variables (no defaults inside this script, and no
# hardcoded private path anywhere in it):
#   ADE20K_REPO_ROOT             -- path to this repository checkout
#   TALK2DINO_VENV               -- path to the project virtual environment
#   ADE20K_REAL_DATA_ROOT        -- ADEChallengeData2016 root or its unambiguous
#                                    parent (containing images/{training,validation}
#                                    and annotations/{training,validation})
#   ADE20K_SOURCE_MANIFEST       -- verified ADE20K dataset-source manifest JSON
#   ADE20K_EVAL_OUTPUT_DIR       -- writable directory for checkpoint/result/per-image-stats
#   TALK2DINO_WEIGHT_DIR         -- weights directory (exported to the evaluator process)
#
# The submission directory and ADE20K_EVAL_OUTPUT_DIR must already exist
# before `sbatch` is invoked -- this script creates only the latter via
# mkdir -p, never assumes either is pre-populated with prior job output.

set -euo pipefail

: "${ADE20K_REPO_ROOT:?set ADE20K_REPO_ROOT to the repository root before submitting}"
: "${TALK2DINO_VENV:?set TALK2DINO_VENV to the project virtual environment before submitting}"
: "${ADE20K_REAL_DATA_ROOT:?set ADE20K_REAL_DATA_ROOT to the ADEChallengeData2016 root (or its parent) before submitting}"
: "${ADE20K_SOURCE_MANIFEST:?set ADE20K_SOURCE_MANIFEST to a verified dataset-source manifest before submitting}"
: "${ADE20K_EVAL_OUTPUT_DIR:?set ADE20K_EVAL_OUTPUT_DIR to a writable output directory before submitting}"
: "${TALK2DINO_WEIGHT_DIR:?set TALK2DINO_WEIGHT_DIR to the weights directory before submitting}"

echo "=== e12 ADE20K C150 matched evaluator: pilot20 ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="${ADE20K_REPO_ROOT}"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="${TALK2DINO_VENV}"
OUTPUT_DIR="${ADE20K_EVAL_OUTPUT_DIR}"
DATA_ROOT="${ADE20K_REAL_DATA_ROOT}"
SOURCE_MANIFEST="${ADE20K_SOURCE_MANIFEST}"

export TALK2DINO_WEIGHT_DIR

cd "${REPO_ROOT}"

echo "Git commit:    $(git rev-parse HEAD)"
echo "Git branch:    $(git branch --show-current)"

if [ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]; then
    echo "FATAL: expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)" >&2
    exit 1
fi

# Reject TRACKED changes only (staged or unstaged) -- unrelated untracked
# files elsewhere in the repo are fine and must not block submission.
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "FATAL: tracked worktree has uncommitted changes; commit or stash before submitting" >&2
    echo "Changed tracked files:" >&2
    git diff --name-only >&2
    git diff --cached --name-only >&2
    exit 1
fi

for required_file in \
    "evaluation_identities/e12_ade20k_dataset_source.toml" \
    "evaluation_identities/e12_ade20k_matched_evaluator.toml" \
    "evaluation_identities/e12_matched_k11_k12_t320.toml" \
    "src/ade20k_dataset_identity.py" \
    "src/ade20k_dataset_manifest.py" \
    "src/ade20k_matched_evaluator_identity.py" \
    "src/ade20k_matched_evaluator_checkpoint.py" \
    "src/ade20k_matched_evaluator_report.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/coco_object_evaluator.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/matched_power_evaluator.py" \
    "diagnostics/run_ade20k_matched_evaluation.py" \
    "verify_ade20k_dataset.py"
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
echo "Source manifest: ${SOURCE_MANIFEST}"
echo "Data root:     ${DATA_ROOT}"

mkdir -p "${OUTPUT_DIR}"

echo "=== preflights (static + real-data checks, before any CUDA/model work) ==="
"${PYTHON}" verify_e3_identity.py preflight
"${PYTHON}" verify_rwr_reproduction.py preflight
"${PYTHON}" verify_matched_k11_k12.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_stability.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_power_evaluation.py preflight --repo-root .
"${PYTHON}" verify_ade20k_dataset.py preflight --data-root "${DATA_ROOT}"
"${PYTHON}" verify_ade20k_dataset.py verify-manifest --data-root "${DATA_ROOT}" --manifest "${SOURCE_MANIFEST}"

echo "=== running the pilot20 ADE20K C150 matched evaluation (never the full 2000-image run) ==="
RESULT_PATH="${OUTPUT_DIR}/result-pilot20-${SLURM_JOB_ID:-manual}.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-pilot20-${SLURM_JOB_ID:-manual}.json"
PER_IMAGE_STATS_PATH="${OUTPUT_DIR}/per-image-stats-pilot20-${SLURM_JOB_ID:-manual}.json"

# Job-specific paths above already prevent accidental reuse of a previous
# job's checkpoint/result -- --resume is deliberately never passed here.
set +e
"${PYTHON}" diagnostics/run_ade20k_matched_evaluation.py \
    --repo-root . \
    --source-manifest "${SOURCE_MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --run-mode pilot20 \
    --checkpoint "${CHECKPOINT_PATH}" \
    --result "${RESULT_PATH}" \
    --per-image-stats "${PER_IMAGE_STATS_PATH}" \
    --device cuda
EVAL_STATUS=$?
set -e

if [ ${EVAL_STATUS} -ne 0 ]; then
    echo "FATAL: pilot20 evaluation exited with status ${EVAL_STATUS}" >&2
    echo "Checkpoint and any partial output are preserved at:" >&2
    echo "  ${CHECKPOINT_PATH}" >&2
    echo "  ${RESULT_PATH}" >&2
    exit "${EVAL_STATUS}"
fi

date -u +"End (UTC): %Y-%m-%dT%H:%M:%SZ"
echo "Result:        ${RESULT_PATH}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Per-image stats: ${PER_IMAGE_STATS_PATH}"
echo "=== done ==="
