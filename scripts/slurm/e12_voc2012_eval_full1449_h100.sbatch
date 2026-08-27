#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=e12-voc2012-eval-full1449
#
# Shared VOC2012 V20/V21 matched evaluator: full run mode (all 1449
# canonical validation images). RUN ONLY AFTER REVIEWING BOTH THE
# PILOT20 AND PILOT100 RESULTS -- this is the full, authoritative
# 1449-image evaluation. One shared backbone/snapshot/graph/propagation
# pass per image feeds BOTH the V20 (no-background) and V21 (background)
# protocols -- never a second backbone pass, never independently-
# selected/recomputed foreground scores. Uses the existing frozen
# COCO-2017-trained Talk2DINO bridge checkpoint read-only -- never
# trains, fine-tunes, or pre-extracts target-dataset features. Never
# routes through main.py --eval / multi_gpu_test, never runs CGLS/
# dense-solve. Fails closed at every preflight step before touching the
# GPU.
#
# --account is overridable at submission time: sbatch --account=<acct> ...
# --output/--error are intentionally left to SLURM's own default
# (slurm-<jobid>.out in the submission directory) so this tracked script
# never hardcodes a private filesystem path; override with
# sbatch --output=... --error=... if a different location is wanted.
#
# Required environment variables (no defaults inside this script, and no
# hardcoded private path anywhere in it):
#   VOC2012_REPO_ROOT       -- path to this repository checkout
#   TALK2DINO_VENV          -- path to the project virtual environment
#   VOC2012_REAL_DATA_ROOT  -- VOC2012 dataset root (VOCdevkit or VOC2012 dir)
#   VOC2012_SOURCE_MANIFEST -- verified VOC2012 dataset-source manifest JSON
#   VOC2012_EVAL_OUTPUT_DIR -- writable directory for checkpoint/result/per-image-stats
#   TALK2DINO_WEIGHT_DIR    -- weights directory (exported to the evaluator process)
#
# The submission directory and VOC2012_EVAL_OUTPUT_DIR must already exist
# before `sbatch` is invoked -- this script creates only the latter via
# mkdir -p, never assumes either is pre-populated with prior job output.

set -euo pipefail

: "${VOC2012_REPO_ROOT:?set VOC2012_REPO_ROOT to the repository root before submitting}"
: "${TALK2DINO_VENV:?set TALK2DINO_VENV to the project virtual environment before submitting}"
: "${VOC2012_REAL_DATA_ROOT:?set VOC2012_REAL_DATA_ROOT to the VOC2012 dataset root before submitting}"
: "${VOC2012_SOURCE_MANIFEST:?set VOC2012_SOURCE_MANIFEST to a verified VOC2012 source manifest before submitting}"
: "${VOC2012_EVAL_OUTPUT_DIR:?set VOC2012_EVAL_OUTPUT_DIR to a writable output directory before submitting}"
: "${TALK2DINO_WEIGHT_DIR:?set TALK2DINO_WEIGHT_DIR to the weights directory before submitting}"

echo "=== e12 VOC2012 V20/V21 matched evaluator: full1449 ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="${VOC2012_REPO_ROOT}"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="${TALK2DINO_VENV}"
DATA_ROOT="${VOC2012_REAL_DATA_ROOT}"
SOURCE_MANIFEST="${VOC2012_SOURCE_MANIFEST}"
OUTPUT_DIR="${VOC2012_EVAL_OUTPUT_DIR}"

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
# This job must run exactly the committed evaluator, never an
# uncommitted tracked edit.
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "FATAL: tracked worktree has uncommitted changes; commit or stash before submitting" >&2
    echo "Changed tracked files:" >&2
    git diff --name-only >&2
    git diff --cached --name-only >&2
    exit 1
fi

# Require the evaluator (and the parent identities it binds against) to
# actually be committed, not merely present on disk as an untracked file.
for required_file in \
    "evaluation_identities/e12_voc2012_matched_evaluator.toml" \
    "evaluation_identities/e12_voc2012_dataset_source.toml" \
    "evaluation_identities/e12_matched_k11_k12_t320.toml" \
    "src/voc2012_matched_evaluator_identity.py" \
    "src/voc2012_matched_evaluator_checkpoint.py" \
    "src/voc2012_matched_evaluator_report.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/coco_object_evaluator.py" \
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/matched_power_evaluator.py" \
    "diagnostics/run_voc2012_matched_evaluation.py" \
    "verify_voc2012_matched_evaluation.py"
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
echo "Data root:     ${DATA_ROOT}"
echo "Source manifest: ${SOURCE_MANIFEST}"

mkdir -p "${OUTPUT_DIR}"

echo "=== preflights (before any CUDA/model initialization) ==="
"${PYTHON}" verify_voc2012_dataset.py preflight --repo-root . --data-root "${DATA_ROOT}"
"${PYTHON}" verify_voc2012_matched_evaluation.py preflight --repo-root .
"${PYTHON}" verify_voc2012_matched_evaluation.py verify-source-binding --repo-root . --data-root "${DATA_ROOT}" --source-manifest "${SOURCE_MANIFEST}"

echo "=== running the full1449 VOC2012 V20/V21 matched evaluation (full 1449-image evaluation -- run only after reviewing both the pilot20 and pilot100 results) ==="
RESULT_PATH="${OUTPUT_DIR}/result-full1449-${SLURM_JOB_ID:-manual}.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-full1449-${SLURM_JOB_ID:-manual}.json"
PER_IMAGE_STATS_PATH="${OUTPUT_DIR}/per-image-stats-full1449-${SLURM_JOB_ID:-manual}.json"

set +e
"${PYTHON}" diagnostics/run_voc2012_matched_evaluation.py \
    --repo-root . \
    --data-root "${DATA_ROOT}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --run-mode full \
    --checkpoint "${CHECKPOINT_PATH}" \
    --result "${RESULT_PATH}" \
    --per-image-stats "${PER_IMAGE_STATS_PATH}" \
    --device cuda
GATE_STATUS=$?
set -e

if [ ${GATE_STATUS} -ne 0 ]; then
    echo "FATAL: full1449 VOC2012 evaluation exited with status ${GATE_STATUS}" >&2
    echo "Checkpoint and any partial output are preserved at:" >&2
    echo "  ${CHECKPOINT_PATH}" >&2
    echo "  ${RESULT_PATH}" >&2
    exit "${GATE_STATUS}"
fi

echo "=== verifying the result ==="
"${PYTHON}" verify_voc2012_matched_evaluation.py verify-result --repo-root . --result "${RESULT_PATH}"

date -u +"End (UTC): %Y-%m-%dT%H:%M:%SZ"
echo "Result:        ${RESULT_PATH}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Per-image stats: ${PER_IMAGE_STATS_PATH}"
echo "=== done ==="
