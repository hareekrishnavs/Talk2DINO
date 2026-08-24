#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=e12-stitching-control-pilot20
#SBATCH --output=/scratch/haree/e12_stitching_control/slurm-pilot20-%j.out
#SBATCH --error=/scratch/haree/e12_stitching_control/slurm-pilot20-%j.err
#
# Reusable stitching-control suite: pilot20 run mode (first 20 canonical
# images). Shares one immutable per-window k12 finite-step output across
# four frozen aggregation variants (uniform_probability, hann_probability,
# uniform_score, hann_score). Tests stitching protocol only -- never a
# graph/propagation change, never T4/DCR/SUR/cross-view consensus, never
# PAMR, never a learned parameter. Requires an accepted k11/k12
# stability-gate result (via the power-evaluation identity chain), same as
# the matched evaluator. Fails closed at every preflight step before
# touching the GPU.
#
# Submission directory (/scratch/haree/e12_stitching_control) must already
# exist before `sbatch` is invoked -- this script does not create it.

set -euo pipefail

echo "=== e12 stitching control suite: pilot20 ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="/project/6114407/haree/Talk2DINO"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="/scratch/haree/venv/talk2dino-a100"
OUTPUT_DIR="/scratch/haree/e12_stitching_control"
STABILITY_RESULT="${STABILITY_RESULT:-/scratch/haree/e12_k11_k12_stability/result-20300858.json}"

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
# job must run exactly the committed suite, never an uncommitted tracked edit.
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "FATAL: tracked worktree has uncommitted changes; commit or stash before submitting" >&2
    echo "Changed tracked files:" >&2
    git diff --name-only >&2
    git diff --cached --name-only >&2
    exit 1
fi

for required_file in \
    "evaluation_identities/e12_stitching_control_suite.toml" \
    "evaluation_identities/e12_k11_k12_power_evaluation.toml" \
    "evaluation_identities/e12_matched_k11_k12_t320.toml" \
    "src/stitching_control_identity.py" \
    "src/stitching_control_checkpoint.py" \
    "src/stitching_control_report.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/stitching_control.py" \
    "diagnostics/run_stitching_control_evaluation.py" \
    "verify_stitching_control_suite.py"
do
    if ! git ls-files --error-unmatch "${required_file}" > /dev/null 2>&1; then
        echo "FATAL: ${required_file} is not committed; refusing to run an uncommitted suite" >&2
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
echo "Stability result: ${STABILITY_RESULT}"

mkdir -p "${OUTPUT_DIR}"

echo "=== preflights (static only, before any CUDA/model work) ==="
"${PYTHON}" verify_e3_identity.py preflight
"${PYTHON}" verify_rwr_reproduction.py preflight
"${PYTHON}" verify_matched_k11_k12.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_stability.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_power_evaluation.py preflight --repo-root .
"${PYTHON}" verify_stitching_control_suite.py preflight --repo-root .

echo "=== running the pilot20 stitching-control evaluation (never full5000) ==="
RESULT_PATH="${OUTPUT_DIR}/result-pilot20-${SLURM_JOB_ID:-manual}.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-pilot20-${SLURM_JOB_ID:-manual}.json"
PER_IMAGE_STATS_PATH="${OUTPUT_DIR}/per-image-stats-pilot20-${SLURM_JOB_ID:-manual}.json"

# Job-specific paths above already prevent accidental reuse of a previous
# job's checkpoint/result -- --resume is deliberately never passed here.
set +e
"${PYTHON}" diagnostics/run_stitching_control_evaluation.py \
    --repo-root . \
    --stability-result "${STABILITY_RESULT}" \
    --run-mode pilot20 \
    --checkpoint "${CHECKPOINT_PATH}" \
    --result "${RESULT_PATH}" \
    --per-image-stats "${PER_IMAGE_STATS_PATH}" \
    --device cuda
GATE_STATUS=$?
set -e

if [ ${GATE_STATUS} -ne 0 ]; then
    echo "FATAL: pilot20 stitching-control evaluation exited with status ${GATE_STATUS}" >&2
    echo "Checkpoint and any partial output are preserved at:" >&2
    echo "  ${CHECKPOINT_PATH}" >&2
    echo "  ${RESULT_PATH}" >&2
    exit "${GATE_STATUS}"
fi

echo "=== verifying the result ==="
"${PYTHON}" verify_stitching_control_suite.py verify-result --result "${RESULT_PATH}" --repo-root .

date -u +"End (UTC): %Y-%m-%dT%H:%M:%SZ"
echo "Result:        ${RESULT_PATH}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Per-image stats: ${PER_IMAGE_STATS_PATH}"
echo "=== done ==="
