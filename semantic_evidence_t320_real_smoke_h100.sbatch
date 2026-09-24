#!/bin/bash
#SBATCH --job-name=t320-real-smoke
#SBATCH --account=rrg-yangw
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# H100 launcher for the committed two-image E3/T320 real-model integration
# smoke (research_specs/semantic_evidence_t320_real_smoke_v1.toml).
#
# THIS FILE DOES NOT SUBMIT ITSELF. It contains no `sbatch` invocation of
# its own, and no automation in this repository is authorized to call
# `sbatch` on it. Only a human operator may run:
#
#     sbatch semantic_evidence_t320_real_smoke_h100.sbatch
#
# and only after every one of these conditions is independently true:
#   1. this contract (research_specs/semantic_evidence_t320_real_smoke_v1.toml)
#      passed independent verification;
#   2. it is committed;
#   3. the runner and this launcher are implemented;
#   4. the runner passed independent verification;
#   5. committed-HEAD readiness passed (COMMITTED_HEAD_VERIFY_T320_REAL_SMOKE_CONTRACT).
#
# AUTHORIZATION BOUNDARY: a successful run of this launcher authorizes
# NOTHING beyond this exact two-image, GT-free integration smoke. It does
# not authorize GT access, accuracy/mIoU computation, interventions, the
# complete 96-image feasibility study, confirmation evaluation, full
# 5,000-image evaluation, or any full-dataset runtime projection from this
# smoke's timings.
#
# Required environment before submission (set by the operator, not this
# script, and never overridden here):
#   T320_SMOKE_WEIGHT_DIR   DINOv2/CLIP checkpoint directory
#                            (resolves TALK2DINO_WEIGHT_DIR)
#   T320_SMOKE_IMAGE_ROOT   COCO-Stuff val2017 images directory
#                            (must contain 000000567197.jpg and
#                            000000367680.jpg; no annotation directory is
#                            ever read by this smoke)
#   T320_SMOKE_OUTPUT_ROOT  a fresh, not-yet-existing scratch destination
#                            for smoke_result.json (never overwritten)

set -euo pipefail

echo "=== T320 real-model integration smoke: two images, GT-free, no accuracy claim ==="
echo "AUTHORIZATION BOUNDARY: this run authorizes nothing beyond this exact"
echo "two-image smoke -- no GT, no accuracy/mIoU, no interventions, no"
echo "96-image study, no confirmation, no full 5,000-image evaluation, no"
echo "full-dataset runtime projection."

: "${T320_SMOKE_WEIGHT_DIR:?T320_SMOKE_WEIGHT_DIR must be set by the operator}"
: "${T320_SMOKE_IMAGE_ROOT:?T320_SMOKE_IMAGE_ROOT must be set by the operator}"
: "${T320_SMOKE_OUTPUT_ROOT:?T320_SMOKE_OUTPUT_ROOT must be set by the operator}"

if [ -e "${T320_SMOKE_OUTPUT_ROOT}" ]; then
    echo "FATAL: T320_SMOKE_OUTPUT_ROOT already exists (overwrite prohibited): ${T320_SMOKE_OUTPUT_ROOT}" >&2
    exit 1
fi

export TALK2DINO_WEIGHT_DIR="${T320_SMOKE_WEIGHT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Under real sbatch execution, Slurm copies this script into a per-job
# spool directory and runs it from there, so BASH_SOURCE no longer points
# at the repository -- SLURM_SUBMIT_DIR (the directory `sbatch` was
# invoked from) is the correct resolution mechanism in that case. Fall
# back to BASH_SOURCE only for a direct, non-sbatch invocation.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CLI="${REPO_ROOT}/diagnostics/run_semantic_evidence_t320_real_smoke.py"

if [ ! -f "${CLI}" ]; then
    echo "FATAL: resolved REPO_ROOT does not contain the smoke CLI: ${REPO_ROOT}" >&2
    echo "       (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
fi

# The bare `python3` resolved from PATH on a compute node is the system/
# CVMFS interpreter, which does not have torch installed. The project's
# established CPU/GPU-capable virtualenv (built via `module load python`
# + `virtualenv --no-download` + wheelhouse installs, per docs/
# semantic_evidence_t320_implementation.md's own reference to this same
# venv) is self-contained and does not require an explicit `module load`
# to run (its interpreter resolves its own base install via absolute
# CVMFS paths recorded in pyvenv.cfg).
PYTHON_BIN="/scratch/haree/venv/talk2dino-a100/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "FATAL: established venv python3 not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "--- fresh process, one job, no self-submission, no resume ---"
echo "--- Step 1/3: torch-free preflight ---"
"${PYTHON_BIN}" "${CLI}" preflight \
    --weight-dir "${T320_SMOKE_WEIGHT_DIR}" \
    --image-root "${T320_SMOKE_IMAGE_ROOT}" \
    --output-root "${T320_SMOKE_OUTPUT_ROOT}"

echo "--- Step 2/3: execute (lazy torch import, model construction, real E3/T320 smoke) ---"
"${PYTHON_BIN}" "${CLI}" execute \
    --weight-dir "${T320_SMOKE_WEIGHT_DIR}" \
    --image-root "${T320_SMOKE_IMAGE_ROOT}" \
    --output-root "${T320_SMOKE_OUTPUT_ROOT}" \
    --slurm-job-id "${SLURM_JOB_ID:-}" \
    --node "${SLURMD_NODENAME:-$(hostname)}"

echo "--- Step 3/3: torch-free public result verification ---"
"${PYTHON_BIN}" "${CLI}" verify-result --result-path "${T320_SMOKE_OUTPUT_ROOT}/smoke_result.json"

echo "=== smoke job complete; see ${T320_SMOKE_OUTPUT_ROOT}/smoke_result.json for the closed result ==="
echo "AUTHORIZATION BOUNDARY (repeated): PASS on this smoke authorizes"
echo "nothing beyond itself -- proceeding to intervention implementation or"
echo "any broader evaluation requires a SEPARATE, later authorization."
