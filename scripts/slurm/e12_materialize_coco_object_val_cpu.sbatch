#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=e12-coco-object-val-materialization
#
# CPU-only materialization of the 5000 COCO-Object val2017
# '_instanceTrainIds.png' masks. Reuses the repository's own
# convert_dataset/convert_coco_object.py mapping table verbatim (hash-
# checked before use) against the already-present raw COCO-Stuff
# per-pixel category masks -- never a redesigned class mapping, never a
# new background threshold, never an annotation-conversion algorithm
# invented for this stage. No CUDA, no model, no scientific evaluation:
# this only prepares a prerequisite data artifact the existing
# COCOObjectDataset/coco.py protocol already expects.
#
# --account is overridable at submission time: sbatch --account=<acct> ...
# --output/--error are intentionally left to SLURM's own default
# (slurm-<jobid>.out in the submission directory) so this tracked script
# never hardcodes a private filesystem path; override with
# sbatch --output=... --error=... if a different location is wanted.
#
# Required environment variables (no defaults inside this script, and no
# hardcoded private path anywhere in it):
#   COCO_OBJECT_REPO_ROOT           -- path to this repository checkout
#   TALK2DINO_VENV                  -- path to the project virtual environment
#   COCO_OBJECT_REAL_DATA_ROOT      -- raw COCO-Stuff164k root (containing
#                                       annotations/val2017 and images/val2017)
#   COCO_OBJECT_MATERIALIZED_ROOT   -- writable root for the materialized
#                                       masks/manifests/checkpoints this
#                                       script produces
#
# The submission directory must already exist before `sbatch` is invoked.
# This script creates only COCO_OBJECT_MATERIALIZED_ROOT (and its
# checkpoints/manifests subdirectories) via mkdir -p, never assumes either
# directory is pre-populated with prior job output.

set -euo pipefail

: "${COCO_OBJECT_REPO_ROOT:?set COCO_OBJECT_REPO_ROOT to the repository root before submitting}"
: "${TALK2DINO_VENV:?set TALK2DINO_VENV to the project virtual environment before submitting}"
: "${COCO_OBJECT_REAL_DATA_ROOT:?set COCO_OBJECT_REAL_DATA_ROOT to the raw COCO-Stuff164k root before submitting}"
: "${COCO_OBJECT_MATERIALIZED_ROOT:?set COCO_OBJECT_MATERIALIZED_ROOT to a writable materialized-output root before submitting}"

echo "=== e12 COCO-Object val2017 mask materialization ==="
echo "Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Hostname:      $(hostname)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR:-unknown}"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

REPO_ROOT="${COCO_OBJECT_REPO_ROOT}"
EXPECTED_BRANCH="e12-connectivity-analysis"
VENV="${TALK2DINO_VENV}"
SOURCE_MASKS="${COCO_OBJECT_REAL_DATA_ROOT}/annotations/val2017"
SOURCE_IMAGES="${COCO_OBJECT_REAL_DATA_ROOT}/images/val2017"
OUTPUT_ROOT="${COCO_OBJECT_MATERIALIZED_ROOT}"
CHECKPOINT="${OUTPUT_ROOT}/checkpoints/checkpoint-${SLURM_JOB_ID:-manual}.json"
MANIFEST="${OUTPUT_ROOT}/manifests/manifest-${SLURM_JOB_ID:-manual}.json"

cd "${REPO_ROOT}"

echo "Git commit:    $(git rev-parse HEAD)"
echo "Git branch:    $(git branch --show-current)"

if [ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]; then
    echo "FATAL: expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)" >&2
    exit 1
fi

# Reject TRACKED changes only (staged or unstaged) -- unrelated untracked
# files elsewhere in the repo are fine and must not block submission. This
# job must run exactly the committed materializer, never an uncommitted
# tracked edit.
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "FATAL: tracked worktree has uncommitted changes; commit or stash before submitting" >&2
    echo "Changed tracked files:" >&2
    git diff --name-only >&2
    git diff --cached --name-only >&2
    exit 1
fi

for required_file in \
    "evaluation_identities/e12_coco_object_val_materialization.toml" \
    "src/coco_object_val_materialization.py" \
    "src/coco_object_val_materialization_checkpoint.py" \
    "src/coco_object_val_materialization_report.py" \
    "src/coco_object_val_materialization_identity.py" \
    "src/open_vocabulary_segmentation/convert_dataset/convert_coco_object.py" \
    "src/open_vocabulary_segmentation/segmentation/datasets/coco_object.py" \
    "materialize_coco_object_val.py" \
    "verify_coco_object_val_materialization.py"
do
    if ! git ls-files --error-unmatch "${required_file}" > /dev/null 2>&1; then
        echo "FATAL: ${required_file} is not committed; refusing to run an uncommitted materializer" >&2
        exit 1
    fi
done

echo "=== activating environment ==="
module load opencv/4.14.0
source "${VENV}/bin/activate"
PYTHON="${VENV}/bin/python"

echo "Python:        $(${PYTHON} --version)"
echo "Source masks:  ${SOURCE_MASKS}"
echo "Source images: ${SOURCE_IMAGES}"
echo "Output root:   ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/manifests"

echo "=== preflights (static + real-data checks, before any conversion) ==="
"${PYTHON}" verify_coco_object_val_materialization.py preflight \
    --repo-root . --source-masks "${SOURCE_MASKS}" --source-images "${SOURCE_IMAGES}" --output-root "${OUTPUT_ROOT}"

echo "=== materializing the 5000 val2017 COCO-Object masks ==="
# Job-specific checkpoint/manifest paths above already prevent accidental
# reuse of a previous job's artifacts -- --resume is deliberately never
# passed here; a genuinely interrupted run should be resumed manually with
# an explicit, reviewed --resume invocation, not automatically resubmitted.
set +e
"${PYTHON}" materialize_coco_object_val.py \
    --repo-root . \
    --source-masks "${SOURCE_MASKS}" \
    --source-images "${SOURCE_IMAGES}" \
    --output-root "${OUTPUT_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}"
MATERIALIZE_STATUS=$?
set -e

if [ ${MATERIALIZE_STATUS} -ne 0 ]; then
    echo "FATAL: materialization exited with status ${MATERIALIZE_STATUS}" >&2
    echo "Checkpoint and any partial output are preserved at:" >&2
    echo "  ${CHECKPOINT}" >&2
    exit "${MATERIALIZE_STATUS}"
fi

echo "=== independently verifying the complete output ==="
"${PYTHON}" verify_coco_object_val_materialization.py verify-output \
    --repo-root . --manifest "${MANIFEST}" --output-root "${OUTPUT_ROOT}" \
    --source-masks "${SOURCE_MASKS}" --source-images "${SOURCE_IMAGES}"

date -u +"End (UTC): %Y-%m-%dT%H:%M:%SZ"
echo "Manifest:      ${MANIFEST}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "=== done ==="
