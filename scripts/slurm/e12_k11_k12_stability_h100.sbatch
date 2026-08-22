#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=e12-k11k12-stability
#SBATCH --output=/scratch/haree/e12_k11_k12_stability/slurm-%j.out
#SBATCH --error=/scratch/haree/e12_k11_k12_stability/slurm-%j.err
#
# Bounded k11/k12 finite-step numerical/runtime stability gate.
#
# This job runs ONLY the bounded gate (100 canonical windows) -- never the
# full 5000-image matched evaluation, never canonical CGLS, never
# COVER-DR/DCR/SUR/T4 semantic-repair machinery. It fails closed at every
# preflight step before touching the GPU.
#
# Submission directory (/scratch/haree/e12_k11_k12_stability) must already
# exist before `sbatch` is invoked -- this script does not create it,
# consistent with treating scratch paths as belonging to the invoking user,
# not to this script.

set -euo pipefail

echo "=== e12 k11/k12 stability gate ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="/project/6114407/haree/Talk2DINO"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="/scratch/haree/venv/talk2dino-a100"
OUTPUT_DIR="/scratch/haree/e12_k11_k12_stability"

cd "${REPO_ROOT}"

echo "Git commit:    $(git rev-parse HEAD)"
echo "Git branch:    $(git branch --show-current)"

if [ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]; then
    echo "FATAL: expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)" >&2
    exit 1
fi

# Reject a dirty TRACKED worktree (untracked scratch/output files are fine;
# uncommitted changes to tracked files are not -- this job must run exactly
# the committed harness, never an uncommitted local edit).
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "FATAL: tracked worktree is dirty; commit or stash before submitting" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

# Require the harness to actually be committed (not merely present on disk
# as an untracked file) before this job is allowed to run it.
for required_file in \
    "evaluation_identities/e12_k11_k12_stability_gate.toml" \
    "src/k11_k12_stability_gate_identity.py" \
    "src/k11_k12_stability_report.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/finite_step_regime.py" \
    "diagnostics/run_k11_k12_stability.py" \
    "verify_k11_k12_stability.py"
do
    if ! git ls-files --error-unmatch "${required_file}" > /dev/null 2>&1; then
        echo "FATAL: ${required_file} is not committed; refusing to run an uncommitted harness" >&2
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

mkdir -p "${OUTPUT_DIR}"

echo "=== preflights ==="
"${PYTHON}" verify_e3_identity.py preflight
"${PYTHON}" verify_rwr_reproduction.py preflight
"${PYTHON}" verify_matched_k11_k12.py preflight --repo-root .
"${PYTHON}" verify_k11_k12_stability.py preflight --repo-root .

echo "=== running the bounded stability gate (never the full evaluation) ==="
RESULT_PATH="${OUTPUT_DIR}/result-${SLURM_JOB_ID:-manual}.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-${SLURM_JOB_ID:-manual}.json"

set +e
"${PYTHON}" diagnostics/run_k11_k12_stability.py \
    --repo-root . \
    --output "${RESULT_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --device cuda
GATE_STATUS=$?
set -e

if [ ${GATE_STATUS} -ne 0 ]; then
    echo "FATAL: stability gate exited with status ${GATE_STATUS}" >&2
    echo "Checkpoint and any partial output are preserved at:" >&2
    echo "  ${CHECKPOINT_PATH}" >&2
    echo "  ${RESULT_PATH}" >&2
    exit "${GATE_STATUS}"
fi

echo "=== verifying the result ==="
"${PYTHON}" verify_k11_k12_stability.py verify-result --result "${RESULT_PATH}" --repo-root .

date -u +"End (UTC): %Y-%m-%dT%H:%M:%SZ"
echo "Result:        ${RESULT_PATH}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "=== done ==="
