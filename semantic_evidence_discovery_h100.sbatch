#!/bin/bash
#SBATCH --job-name=semantic-evidence-discovery
#SBATCH --account=rrg-yangw
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# H100 launcher for the committed 48-image semantic-evidence discovery
# execution (research_specs/semantic_evidence_discovery_execution_v1.toml).
#
# THIS FILE DOES NOT SUBMIT ITSELF. It contains no `sbatch` invocation of
# its own, and no automation in this repository is authorized to call
# `sbatch` on it. Only a human operator may run:
#
#     sbatch semantic_evidence_discovery_h100.sbatch
#
# and only after every one of these conditions is independently true:
#   1. this authority (research_specs/semantic_evidence_discovery_execution_v1.toml)
#      passed independent verification and is committed;
#   2. this runner/CLI/launcher (the 7 files it authorizes) are implemented
#      and independently verified;
#   3. committed-HEAD readiness is re-verified;
#   4. the operator has explicitly decided to submit.
#
# Resource ceiling: exactly 1x H100 80GB, inherited from the feasibility
# contract's Gate E (max_h100_hours_96_image_feasibility_run=2.0) and the
# smoke launcher's own precedent -- see [resource_contract] in the
# committed authority. Multi-GPU, alternate GPU types, distributed
# execution, training, and any population beyond the frozen 48-image
# discovery split are prohibited, both by the authority and by this script
# never exposing a code path to request them.
#
# AUTHORIZATION BOUNDARY: a successful run of this launcher authorizes
# NOTHING beyond producing discovery-side features/labels and Gate A/E.
# It never accesses the locked 48-image holdout, never computes Gates
# B/C/D, never reaches the feasibility contract's own gates.outcome
# proceed/stop decision, and never authorizes a full evaluation or any
# accuracy/mIoU/paper claim.
#
# Required environment before submission (set by the operator, not this
# script, and never overridden here):
#   DISCOVERY_WEIGHT_DIR    DINOv2/CLIP checkpoint directory
#                            (resolves TALK2DINO_WEIGHT_DIR)
#   DISCOVERY_IMAGE_ROOT    COCO-Stuff val2017 images directory
#   DISCOVERY_GT_ROOT       COCO-Stuff semantic annotation directory
#                            (read only in Phase B, after Phase A for each
#                            image is complete and immutable)
#   DISCOVERY_OUTPUT_ROOT   destination for the installed discovery result
#                            (fresh mode: must not yet exist; resume/
#                            finalize_only: must already exist as an
#                            in-progress checkpoint directory)
#   DISCOVERY_MODE          fresh | resume | finalize_only (default: fresh)

set -euo pipefail

echo "=== 48-image semantic-evidence discovery execution: discovery-only, Gate A/E only ==="
echo "AUTHORIZATION BOUNDARY: this run authorizes nothing beyond discovery-side"
echo "feature/label production and Gate A/E -- no holdout, no Gates B/C/D, no"
echo "full evaluation, no accuracy/mIoU claim, no paper claim."

: "${DISCOVERY_WEIGHT_DIR:?DISCOVERY_WEIGHT_DIR must be set by the operator}"
: "${DISCOVERY_IMAGE_ROOT:?DISCOVERY_IMAGE_ROOT must be set by the operator}"
: "${DISCOVERY_GT_ROOT:?DISCOVERY_GT_ROOT must be set by the operator}"
: "${DISCOVERY_OUTPUT_ROOT:?DISCOVERY_OUTPUT_ROOT must be set by the operator}"
DISCOVERY_MODE="${DISCOVERY_MODE:-fresh}"

case "${DISCOVERY_MODE}" in
    fresh|resume|finalize_only) ;;
    *)
        echo "FATAL: DISCOVERY_MODE must be fresh, resume, or finalize_only (got '${DISCOVERY_MODE}')" >&2
        exit 1
        ;;
esac

if [ "${DISCOVERY_MODE}" = "fresh" ] && [ -e "${DISCOVERY_OUTPUT_ROOT}" ]; then
    echo "FATAL: fresh mode requires an absent DISCOVERY_OUTPUT_ROOT (overwrite prohibited): ${DISCOVERY_OUTPUT_ROOT}" >&2
    exit 1
fi

export TALK2DINO_WEIGHT_DIR="${DISCOVERY_WEIGHT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# See semantic_evidence_t320_real_smoke_h100.sbatch (this launcher's own
# dependency precedent, per [resource_contract].precedent) for why
# SLURM_SUBMIT_DIR is the correct repo-root resolution under real sbatch
# execution, rather than BASH_SOURCE.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CLI="${REPO_ROOT}/diagnostics/run_semantic_evidence_discovery.py"

if [ ! -f "${CLI}" ]; then
    echo "FATAL: resolved REPO_ROOT does not contain the discovery CLI: ${REPO_ROOT}" >&2
    echo "       (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
fi

# Same established venv and cv2/opencv module ordering as the smoke
# launcher -- see that script's own comment for the full rationale
# (numpy 2.x / opencv 4.14.0 compatibility, wheelhouse dummy opencv
# package).
module load opencv/4.14.0

PYTHON_BIN="/scratch/haree/venv/talk2dino-a100/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "FATAL: established venv python3 not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c "import cv2" >/dev/null 2>&1; then
    echo "FATAL: cv2 not importable through the venv python3 even after 'module load opencv/4.14.0'" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c "import sklearn" >/dev/null 2>&1; then
    echo "FATAL: scikit-learn not importable through the venv python3 (required for the 4 frozen logistic fits)" >&2
    exit 1
fi

echo "--- fresh process, one job, no self-submission, mode=${DISCOVERY_MODE} ---"
echo "--- Step 1/3: torch-free preflight ---"
"${PYTHON_BIN}" "${CLI}" preflight \
    --weight-dir "${DISCOVERY_WEIGHT_DIR}" \
    --image-root "${DISCOVERY_IMAGE_ROOT}" \
    --output-root "${DISCOVERY_OUTPUT_ROOT}" \
    --mode "${DISCOVERY_MODE}"

echo "--- Step 2/3: execute (lazy torch import, real discovery execution) ---"
"${PYTHON_BIN}" "${CLI}" execute \
    --weight-dir "${DISCOVERY_WEIGHT_DIR}" \
    --image-root "${DISCOVERY_IMAGE_ROOT}" \
    --gt-root "${DISCOVERY_GT_ROOT}" \
    --output-root "${DISCOVERY_OUTPUT_ROOT}" \
    --mode "${DISCOVERY_MODE}" \
    --slurm-job-id "${SLURM_JOB_ID:-}" \
    --node "${SLURMD_NODENAME:-$(hostname)}"

echo "--- Step 3/3: torch-free public result verification ---"
"${PYTHON_BIN}" "${CLI}" verify-result --result-path "${DISCOVERY_OUTPUT_ROOT}"

echo "=== discovery job complete; see ${DISCOVERY_OUTPUT_ROOT}/discovery_result.json and discovery_decision.json ==="
echo "AUTHORIZATION BOUNDARY (repeated): PASS on this discovery run authorizes"
echo "only the creation of a later, separate successor holdout authority --"
echo "never holdout execution, never Gates B/C/D, never a full evaluation,"
echo "never any accuracy/mIoU/paper claim."
