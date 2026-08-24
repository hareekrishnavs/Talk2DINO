#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=e12-native-edge-support-audit-mechanics20
#SBATCH --output=/scratch/haree/e12_native_edge_support_audit/slurm-mechanics20-%j.out
#SBATCH --error=/scratch/haree/e12_native_edge_support_audit/slurm-mechanics20-%j.err
#
# Native cross-view directed-edge support audit: mechanics20 run mode (first 20
# canonical images). Read-only structural reachability measurement over the
# existing matched k12 directed top-12 graph -- never deletes, reweights,
# symmetrizes, or edits an edge; never selects a T4/T4-prime target; never
# builds crop-consensus semantic labels; never touches DCR/SUR; never
# votes; never prunes; never runs a counterfactual/adjoint/Sherman-Morrison
# solve; never introduces a learned threshold or confidence/entropy/margin
# gate. Never renders a segmentation-accuracy or pruning-efficacy verdict.
# Requires an accepted k11/k12 stability-gate result (via the
# power-evaluation identity chain, reached through the stitching-control
# identity), same as the matched/stitching-control evaluators. Fails closed
# at every preflight step before touching the GPU. Structural audit only:
# no pilot100/full5000 script exists yet and none is submitted here.
#
# Submission directory (/scratch/haree/e12_native_edge_support_audit) must
# already exist before `sbatch` is invoked -- this script does not create
# it, and never submits itself or any other job.

set -euo pipefail

echo "=== e12 native cross-view directed-edge support audit: mechanics20 ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="/project/6114407/haree/Talk2DINO"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="/scratch/haree/venv/talk2dino-a100"
OUTPUT_DIR="/scratch/haree/e12_native_edge_support_audit"
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
# job must run exactly the committed audit, never an uncommitted tracked edit.
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "FATAL: tracked worktree has uncommitted changes; commit or stash before submitting" >&2
    echo "Changed tracked files:" >&2
    git diff --name-only >&2
    git diff --cached --name-only >&2
    exit 1
fi

for required_file in \
    "evaluation_identities/e12_native_edge_support_audit.toml" \
    "evaluation_identities/e12_stitching_control_suite.toml" \
    "evaluation_identities/e12_k11_k12_power_evaluation.toml" \
    "evaluation_identities/e12_matched_k11_k12_t320.toml" \
    "src/native_edge_support_identity.py" \
    "src/native_edge_support_checkpoint.py" \
    "src/native_edge_support_report.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/native_edge_support.py" \
    "diagnostics/run_native_edge_support_audit.py" \
    "verify_native_edge_support_audit.py"
do
    if ! git ls-files --error-unmatch "${required_file}" > /dev/null 2>&1; then
        echo "FATAL: ${required_file} is not committed; refusing to run an uncommitted audit" >&2
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
"${PYTHON}" verify_native_edge_support_audit.py preflight --repo-root .

echo "=== running the mechanics20 native-edge-support audit (structural audit only) ==="
RESULT_PATH="${OUTPUT_DIR}/result-mechanics20-${SLURM_JOB_ID:-manual}.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-mechanics20-${SLURM_JOB_ID:-manual}.json"
PER_IMAGE_STATS_PATH="${OUTPUT_DIR}/per-image-stats-mechanics20-${SLURM_JOB_ID:-manual}.json"

# Job-specific paths above already prevent accidental reuse of a previous
# job's checkpoint/result -- --resume is deliberately never passed here.
set +e
"${PYTHON}" diagnostics/run_native_edge_support_audit.py \
    --repo-root . \
    --stability-result "${STABILITY_RESULT}" \
    --run-mode mechanics20 \
    --checkpoint "${CHECKPOINT_PATH}" \
    --result "${RESULT_PATH}" \
    --per-image-stats "${PER_IMAGE_STATS_PATH}" \
    --device cuda
GATE_STATUS=$?
set -e

if [ ${GATE_STATUS} -ne 0 ]; then
    echo "FATAL: mechanics20 native-edge-support audit exited with status ${GATE_STATUS}" >&2
    echo "Checkpoint and any partial output are preserved at:" >&2
    echo "  ${CHECKPOINT_PATH}" >&2
    echo "  ${RESULT_PATH}" >&2
    exit "${GATE_STATUS}"
fi

echo "=== verifying the result ==="
"${PYTHON}" verify_native_edge_support_audit.py verify-result --result "${RESULT_PATH}" --repo-root .

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
