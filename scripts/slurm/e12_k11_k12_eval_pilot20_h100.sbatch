#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=e12-k11k12-eval-pilot20
#SBATCH --output=/scratch/haree/e12_k11_k12_evaluation/slurm-pilot20-%j.out
#SBATCH --error=/scratch/haree/e12_k11_k12_evaluation/slurm-pilot20-%j.err
#
# Matched k11-vs-k12 power evaluator: pilot20 run mode (first 20 canonical
# images). Requires an accepted k11/k12 stability-gate result -- never a
# hardcoded default inside the evaluator CLI itself. Never runs the full
# 5000-image evaluation, never routes through main.py --eval /
# multi_gpu_test, never runs canonical CGLS. Fails closed at every
# preflight step before touching the GPU.
#
# Submission directory (/scratch/haree/e12_k11_k12_evaluation) must already
# exist before `sbatch` is invoked -- this script does not create it.

set -euo pipefail

echo "=== e12 k11/k12 power evaluator: pilot20 ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="/project/6114407/haree/Talk2DINO"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="/scratch/haree/venv/talk2dino-a100"
OUTPUT_DIR="/scratch/haree/e12_k11_k12_evaluation"
# The stability-gate result this evaluator's --stability-result binds
# against. Overridable via environment variable so a newer passing gate
# result can be used without editing this script; the evaluator CLI itself
# has no default and always requires this to be passed explicitly.
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
# files (e.g. historical local JSON artifacts elsewhere in the repo) are
# fine and must not block submission. This job must run exactly the
# committed evaluator, never an uncommitted tracked edit.
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "FATAL: tracked worktree has uncommitted changes; commit or stash before submitting" >&2
    echo "Changed tracked files:" >&2
    git diff --name-only >&2
    git diff --cached --name-only >&2
    exit 1
fi

# Require the evaluator (and the stability gate it binds against) to
# actually be committed, not merely present on disk as an untracked file.
for required_file in \
    "evaluation_identities/e12_k11_k12_power_evaluation.toml" \
    "evaluation_identities/e12_k11_k12_stability_gate.toml" \
    "evaluation_identities/e12_matched_k11_k12_t320.toml" \
    "src/k11_k12_power_evaluation_identity.py" \
    "src/k11_k12_power_evaluation_report.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/matched_power_evaluator.py" \
    "diagnostics/run_matched_k11_k12_evaluation.py" \
    "verify_k11_k12_power_evaluation.py"
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
echo "Stability result: ${STABILITY_RESULT}"

mkdir -p "${OUTPUT_DIR}"

echo "=== preflights ==="
"${PYTHON}" verify_e3_identity.py preflight
"${PYTHON}" verify_rwr_reproduction.py preflight
"${PYTHON}" verify_matched_k11_k12.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_stability.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_power_evaluation.py preflight --repo-root .

echo "=== running the pilot20 matched k11/k12 power evaluation (never the full 5000-image run) ==="
RESULT_PATH="${OUTPUT_DIR}/result-pilot20-${SLURM_JOB_ID:-manual}.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-pilot20-${SLURM_JOB_ID:-manual}.json"
PER_IMAGE_STATS_PATH="${OUTPUT_DIR}/per-image-stats-pilot20-${SLURM_JOB_ID:-manual}.json"

set +e
"${PYTHON}" diagnostics/run_matched_k11_k12_evaluation.py \
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
    echo "FATAL: pilot20 power evaluation exited with status ${GATE_STATUS}" >&2
    echo "Checkpoint and any partial output are preserved at:" >&2
    echo "  ${CHECKPOINT_PATH}" >&2
    echo "  ${RESULT_PATH}" >&2
    exit "${GATE_STATUS}"
fi

echo "=== verifying the result ==="
"${PYTHON}" verify_k11_k12_power_evaluation.py verify-result --result "${RESULT_PATH}" --repo-root .

date -u +"End (UTC): %Y-%m-%dT%H:%M:%SZ"
echo "Result:        ${RESULT_PATH}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Per-image stats: ${PER_IMAGE_STATS_PATH}"
echo "=== done ==="
