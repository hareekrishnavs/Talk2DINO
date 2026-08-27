#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=e12-coco-object-eval-pilot20
#
# COCO-Object protocol confirmation: E3 vs matched k11 vs k12, T=320
# finite-step, pilot20 run mode (first 20 canonical val2017 images). Same-
# data protocol confirmation on the existing COCO images/annotations, not
# an independent dataset. Requires the verified COCO-Object val2017
# materialization manifest (COCO_OBJECT_MATERIALIZATION_MANIFEST below) --
# never runs conversion itself. Never runs the full 5000-image evaluation,
# never routes through main.py --eval/multi_gpu_test, never runs
# T4/DCR/SUR, cross-view edge support, structural pruning, alternative
# stitching, or learned parameters. Fails closed at every preflight step
# before touching the GPU. Only this pilot20 script may be submitted after
# independent verification -- pilot100/full require separate, later
# authorization.
#
# --account is overridable at submission time: sbatch --account=<acct> ...
# --output/--error are intentionally left to SLURM's own default
# (slurm-<jobid>.out in the submission directory) so this tracked script
# never hardcodes a private filesystem path; override with
# sbatch --output=... --error=... if a different location is wanted.
#
# Required environment variables (no defaults inside this script, and no
# hardcoded private path anywhere in it):
#   COCO_OBJECT_REPO_ROOT             -- path to this repository checkout
#   TALK2DINO_VENV                    -- path to the project virtual environment
#   COCO_OBJECT_REAL_DATA_ROOT        -- raw COCO-Stuff164k root (containing
#                                         annotations/val2017 and images/val2017)
#   COCO_OBJECT_MATERIALIZED_ROOT     -- root of the verified materialized
#                                         COCO-Object masks (evaluator --data-root)
#   COCO_OBJECT_MATERIALIZATION_MANIFEST -- verified materialization manifest JSON
#   COCO_OBJECT_EVAL_OUTPUT_DIR       -- writable directory for checkpoint/result/per-image-stats
#   TALK2DINO_WEIGHT_DIR              -- weights directory (exported to the evaluator process)
#
# The submission directory and COCO_OBJECT_EVAL_OUTPUT_DIR must already
# exist before `sbatch` is invoked -- this script creates only the latter
# via mkdir -p, never assumes either is pre-populated with prior job output.

set -euo pipefail

: "${COCO_OBJECT_REPO_ROOT:?set COCO_OBJECT_REPO_ROOT to the repository root before submitting}"
: "${TALK2DINO_VENV:?set TALK2DINO_VENV to the project virtual environment before submitting}"
: "${COCO_OBJECT_REAL_DATA_ROOT:?set COCO_OBJECT_REAL_DATA_ROOT to the raw COCO-Stuff164k root before submitting}"
: "${COCO_OBJECT_MATERIALIZED_ROOT:?set COCO_OBJECT_MATERIALIZED_ROOT to the verified materialized-masks root before submitting}"
: "${COCO_OBJECT_MATERIALIZATION_MANIFEST:?set COCO_OBJECT_MATERIALIZATION_MANIFEST to a verified materialization manifest before submitting}"
: "${COCO_OBJECT_EVAL_OUTPUT_DIR:?set COCO_OBJECT_EVAL_OUTPUT_DIR to a writable output directory before submitting}"
: "${TALK2DINO_WEIGHT_DIR:?set TALK2DINO_WEIGHT_DIR to the weights directory before submitting}"

echo "=== e12 COCO-Object protocol confirmation: pilot20 ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="${COCO_OBJECT_REPO_ROOT}"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="${TALK2DINO_VENV}"
OUTPUT_DIR="${COCO_OBJECT_EVAL_OUTPUT_DIR}"
MATERIALIZATION_MANIFEST="${COCO_OBJECT_MATERIALIZATION_MANIFEST}"
DATA_ROOT="${COCO_OBJECT_MATERIALIZED_ROOT}"
SOURCE_MASKS="${COCO_OBJECT_REAL_DATA_ROOT}/annotations/val2017"
SOURCE_IMAGES="${COCO_OBJECT_REAL_DATA_ROOT}/images/val2017"

export TALK2DINO_WEIGHT_DIR

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

echo "=== running the pilot20 COCO-Object protocol confirmation (never the full 5000-image run) ==="
RESULT_PATH="${OUTPUT_DIR}/result-pilot20-${SLURM_JOB_ID:-manual}.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-pilot20-${SLURM_JOB_ID:-manual}.json"
PER_IMAGE_STATS_PATH="${OUTPUT_DIR}/per-image-stats-pilot20-${SLURM_JOB_ID:-manual}.json"

# Job-specific paths above already prevent accidental reuse of a previous
# job's checkpoint/result -- --resume is deliberately never passed here.
set +e
"${PYTHON}" diagnostics/run_coco_object_protocol_confirmation.py \
    --repo-root . \
    --materialization-manifest "${MATERIALIZATION_MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --source-masks "${SOURCE_MASKS}" \
    --source-images "${SOURCE_IMAGES}" \
    --run-mode pilot20 \
    --checkpoint "${CHECKPOINT_PATH}" \
    --result "${RESULT_PATH}" \
    --per-image-stats "${PER_IMAGE_STATS_PATH}" \
    --device cuda
EVAL_STATUS=$?
set -e

if [ ${EVAL_STATUS} -ne 0 ]; then
    echo "FATAL: pilot20 protocol confirmation exited with status ${EVAL_STATUS}" >&2
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
