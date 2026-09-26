#!/bin/bash
#SBATCH --job-name=semantic-evidence-holdout-evaluation
#SBATCH --account=rrg-yangw
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# H100 launcher for the locked 48-image holdout evaluation (Gates B/C/D),
# per research_specs/semantic_evidence_holdout_evaluation_v1.toml.
#
# THIS FILE DOES NOT SUBMIT ITSELF. It contains no `sbatch` invocation of
# its own, and no automation in this repository is authorized to call
# `sbatch` on it. Only a human operator may run:
#
#     sbatch semantic_evidence_holdout_evaluation_h100.sbatch
#
# and only after every one of these conditions is independently true:
#   1. the holdout evaluation authority
#      (research_specs/semantic_evidence_holdout_evaluation_v1.toml) has
#      passed independent verification and is committed;
#   2. this runner/CLI/launcher has been implemented and independently
#      verified against synthetic data only;
#   3. a SEPARATE, EXPLICIT successor authority has set
#      [authorization.frozen_flags].holdout_execution_authorized=true --
#      it is False in the currently committed authority, and this
#      launcher's own runner (semantic_evidence_holdout_evaluate.py)
#      checks that flag as its first action and fails closed if it is not
#      True, before opening any holdout image or GT file;
#   4. committed-HEAD holdout readiness is separately re-verified;
#   5. the operator has explicitly decided to submit.
#
# Resource ceiling: identical to the discovery launcher's own precedent --
# exactly 1x H100 80GB, 8 CPUs, 80GB host memory, 02:00:00 wall time (the
# same 48-image scale as discovery; no refitting occurs, only per-image
# GT-free/GT-dependent feature production via the frozen E3/T320 pipeline,
# followed by CPU-side frozen-fit application, AUROC/AP, and a 10,000-
# replicate bootstrap over already-computed scores). Multi-GPU, alternate
# GPU types, distributed execution, training, and any population beyond
# the frozen 48-image locked holdout are prohibited, both by the authority
# and by this script never exposing a code path to request them.
#
# AUTHORIZATION BOUNDARY: a successful run of this launcher authorizes
# NOTHING beyond producing the locked-holdout Gate B/C/D result. It never
# accesses the discovery population's own images/GT again, never performs
# a combined 96-image run, never touches the full 5,000-image validation
# set, and never authorizes a full-evaluation or deployment claim beyond
# what [gates.outcome] in the feasibility contract itself defines.
#
# Required environment before submission (set by the operator, not this
# script, and never overridden here):
#   HOLDOUT_WEIGHT_DIR          DINOv2/CLIP checkpoint directory
#                                 (resolves TALK2DINO_WEIGHT_DIR)
#   HOLDOUT_IMAGE_ROOT          COCO-Stuff val2017 images directory
#   HOLDOUT_GT_ROOT             COCO-Stuff semantic annotation directory
#                                 (read only in Phase B, after Phase A for
#                                 each image is complete and immutable)
#   HOLDOUT_OUTPUT_ROOT         destination for the installed holdout
#                                 result (fresh mode: must not yet exist;
#                                 resume/finalize_only: must already exist
#                                 as an in-progress checkpoint directory)
#   HOLDOUT_DISCOVERY_RESULT    path to the finalized discovery_result.json
#                                 whose frozen fits this evaluation applies
#   HOLDOUT_MODE                fresh | resume | finalize_only (default: fresh)

set -euo pipefail

echo "=== 48-image locked-holdout evaluation: Gates B/C/D ==="
echo "AUTHORIZATION BOUNDARY: this run authorizes nothing beyond the"
echo "locked-holdout Gate B/C/D result -- no combined 96-image run, no"
echo "full-5000-image evaluation, no deployment/paper claim beyond"
echo "[gates.outcome] in the feasibility contract itself."
echo "This launcher's own runner fails closed unless a separate, explicit"
echo "successor authority has authorized holdout execution."

: "${HOLDOUT_WEIGHT_DIR:?HOLDOUT_WEIGHT_DIR must be set by the operator}"
: "${HOLDOUT_IMAGE_ROOT:?HOLDOUT_IMAGE_ROOT must be set by the operator}"
: "${HOLDOUT_GT_ROOT:?HOLDOUT_GT_ROOT must be set by the operator}"
: "${HOLDOUT_OUTPUT_ROOT:?HOLDOUT_OUTPUT_ROOT must be set by the operator}"
: "${HOLDOUT_DISCOVERY_RESULT:?HOLDOUT_DISCOVERY_RESULT must be set by the operator}"
HOLDOUT_MODE="${HOLDOUT_MODE:-fresh}"

case "${HOLDOUT_MODE}" in
    fresh|resume|finalize_only) ;;
    *)
        echo "FATAL: HOLDOUT_MODE must be fresh, resume, or finalize_only (got '${HOLDOUT_MODE}')" >&2
        exit 1
        ;;
esac

if [ "${HOLDOUT_MODE}" = "fresh" ] && [ -e "${HOLDOUT_OUTPUT_ROOT}" ]; then
    echo "FATAL: fresh mode requires an absent HOLDOUT_OUTPUT_ROOT (overwrite prohibited): ${HOLDOUT_OUTPUT_ROOT}" >&2
    exit 1
fi

export TALK2DINO_WEIGHT_DIR="${HOLDOUT_WEIGHT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# See semantic_evidence_discovery_h100.sbatch (this launcher's own
# dependency precedent) for why SLURM_SUBMIT_DIR is the correct repo-root
# resolution under real sbatch execution, rather than BASH_SOURCE.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CLI="${REPO_ROOT}/diagnostics/run_semantic_evidence_holdout_evaluation.py"

if [ ! -f "${CLI}" ]; then
    echo "FATAL: resolved REPO_ROOT does not contain the holdout evaluation CLI: ${REPO_ROOT}" >&2
    echo "       (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
fi

# Same established venv and cv2/opencv module ordering as the discovery
# and smoke launchers.
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
    echo "FATAL: scikit-learn not importable through the venv python3 (required for AUROC/AP computation)" >&2
    exit 1
fi

echo "--- fresh process, one job, no self-submission, mode=${HOLDOUT_MODE} ---"
echo "--- Step 1/3: torch-free preflight (fails closed unless authorized) ---"
"${PYTHON_BIN}" "${CLI}" preflight \
    --weight-dir "${HOLDOUT_WEIGHT_DIR}" \
    --image-root "${HOLDOUT_IMAGE_ROOT}" \
    --gt-root "${HOLDOUT_GT_ROOT}" \
    --output-root "${HOLDOUT_OUTPUT_ROOT}" \
    --discovery-result-path "${HOLDOUT_DISCOVERY_RESULT}" \
    --mode "${HOLDOUT_MODE}"

echo "--- Step 2/3: execute (lazy torch import, real holdout evaluation; fails closed unless authorized) ---"
"${PYTHON_BIN}" "${CLI}" execute \
    --weight-dir "${HOLDOUT_WEIGHT_DIR}" \
    --image-root "${HOLDOUT_IMAGE_ROOT}" \
    --gt-root "${HOLDOUT_GT_ROOT}" \
    --output-root "${HOLDOUT_OUTPUT_ROOT}" \
    --discovery-result-path "${HOLDOUT_DISCOVERY_RESULT}" \
    --mode "${HOLDOUT_MODE}" \
    --slurm-job-id "${SLURM_JOB_ID:-}" \
    --node "${SLURMD_NODENAME:-$(hostname)}"

echo "--- Step 3/3: torch-free public result verification ---"
"${PYTHON_BIN}" "${CLI}" verify-result --result-path "${HOLDOUT_OUTPUT_ROOT}"

echo "=== holdout evaluation job complete; see ${HOLDOUT_OUTPUT_ROOT}/holdout_result.json and holdout_decision.json ==="
echo "AUTHORIZATION BOUNDARY (repeated): this run authorizes only the"
echo "locked-holdout Gate B/C/D result -- never a combined 96-image run,"
echo "never the full 5,000-image validation set, never a deployment/paper"
echo "claim beyond [gates.outcome] in the feasibility contract itself."
