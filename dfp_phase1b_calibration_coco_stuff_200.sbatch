#!/bin/bash
#SBATCH --account=rrg-yangw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --gpus=h100:1
#SBATCH --job-name=dfp-phase1b-calibration-coco-stuff-200
#SBATCH --output=%x-%j.out
#
# DFP Phase 1B: conservation-calibration launcher for the E16-authorized
# 200-window COCO-Stuff-500 run.
#
# **This launches CALIBRATION MEASUREMENT ONLY.** It does not run full
# Phase 1B provenance, does not select or apply a numeric conservation
# tolerance, and a successful run does NOT authorize the full Phase 1B
# run -- see docs/dfp_phase1b_calibration_run.md for the complete
# account of what happens after a successful calibration artifact.
#
# Resource request mirrors the closest verified prior job in this
# repository, dfp_phase1a_execute_coco_stuff_500.sbatch: this run does
# strictly less per-image work (200 selected windows across at most 500
# images, no per-image analysis/component-report/qualitative-figure
# generation, no GT access at all) than that full Phase 1A execution
# driver, so the same resource envelope leaves generous headroom.
# --account is a normal #SBATCH directive and, like any Slurm job, can
# be overridden per the scheduler's own mechanism at submission time
# without editing this file.
#
# This script never submits itself or any other job -- it contains no
# call to the scheduler at all, and no literal "sbatch" invocation
# anywhere in it. It is not submitted by this task; only validated with
# `bash -n` (syntax check).
#
# Required environment (never hardcoded in this file):
#   DFP_REPO_ROOT              path to this repository checkout, at the
#                              exact committed HEAD this calibration run
#                              is authorized against
#   DFP_VENV_ACTIVATE_PATH     path to the Python virtualenv's
#                              bin/activate script (opencv module loaded
#                              separately, see DFP_OPENCV_MODULE below)
#   COCO_STUFF_REAL_DATA_ROOT  path to the real COCO-Stuff-164K dataset
#                              root (consumed directly by the real
#                              GeometryProvider/ImagePreparer wiring a
#                              future, authorized change supplies to
#                              `execute` -- never read or interpreted by
#                              this script)
#   DFP_CALIBRATION_OUTPUT_ROOT
#                              scratch output root; this run's
#                              `phase1b_conservation_calibration.json`
#                              is written under
#                              "$DFP_CALIBRATION_OUTPUT_ROOT/coco_stuff_200"
#   DFP_CALIBRATION_COMPONENT_CHUNK_SIZE
#                              explicit, required positive integer --
#                              E16 authorizes recording this value in the
#                              artifact but does not freeze it to one
#                              number; this script never silently
#                              defaults it
# Optional environment:
#   DFP_OPENCV_MODULE          environment-module name to load (default:
#                              opencv/4.14.0, the module every verified
#                              prior job in this repository loads)
#
# This launcher always runs Phase A+B preflight (including full 200-
# window candidate-universe construction and selection) before
# `execute`, and never overwrites an existing calibration artifact by
# default -- pass `--force` yourself via DFP_CALIBRATION_EXTRA_ARGS if
# an explicit, deliberate overwrite is intended.
#
# Only mode "fresh" is accepted: E16 defines no resume/checkpoint
# semantics anywhere (see docs/dfp_phase1b_calibration_run.md's "Resume/
# fresh-only decision" section), so this script never accepts or passes
# any other mode.
#
# Example (submission mechanics deliberately omitted; use your site's
# normal scheduler submission command with this script as the target):
#
#   export DFP_REPO_ROOT=/path/to/Talk2DINO
#   export DFP_VENV_ACTIVATE_PATH=/path/to/venv/bin/activate
#   export COCO_STUFF_REAL_DATA_ROOT=/path/to/coco_stuff164k
#   export DFP_CALIBRATION_OUTPUT_ROOT=/path/to/scratch/output
#   export DFP_CALIBRATION_COMPONENT_CHUNK_SIZE=8
#   <scheduler submit command> dfp_phase1b_calibration_coco_stuff_200.sbatch

set -euo pipefail

for var in DFP_REPO_ROOT DFP_VENV_ACTIVATE_PATH COCO_STUFF_REAL_DATA_ROOT \
           DFP_CALIBRATION_OUTPUT_ROOT DFP_CALIBRATION_COMPONENT_CHUNK_SIZE; do
    if [ -z "${!var:-}" ]; then
        echo "FATAL: \$$var is not set in the environment this job was submitted from." >&2
        echo "Export it and resubmit -- see the header comment in this script." >&2
        exit 1
    fi
done

case "$DFP_CALIBRATION_COMPONENT_CHUNK_SIZE" in
    ''|*[!0-9]*)
        echo "FATAL: \$DFP_CALIBRATION_COMPONENT_CHUNK_SIZE must be a positive integer, got: $DFP_CALIBRATION_COMPONENT_CHUNK_SIZE" >&2
        exit 1
        ;;
esac
if [ "$DFP_CALIBRATION_COMPONENT_CHUNK_SIZE" -lt 1 ]; then
    echo "FATAL: \$DFP_CALIBRATION_COMPONENT_CHUNK_SIZE must be at least 1, got: $DFP_CALIBRATION_COMPONENT_CHUNK_SIZE" >&2
    exit 1
fi

cd "$DFP_REPO_ROOT"

module load "${DFP_OPENCV_MODULE:-opencv/4.14.0}"
source "$DFP_VENV_ACTIVATE_PATH"

RUN_OUTPUT_ROOT="$DFP_CALIBRATION_OUTPUT_ROOT/coco_stuff_200"
mkdir -p "$RUN_OUTPUT_ROOT"
ARTIFACT_PATH="$RUN_OUTPUT_ROOT/phase1b_conservation_calibration.json"

echo "=== Repository / authority identity ==="
echo "commit: $(git -C "$DFP_REPO_ROOT" rev-parse HEAD)"
echo "authority: evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml"
echo "This run measures raw conservation residuals only -- it does not select"
echo "or apply a numeric conservation tolerance, and success does not"
echo "authorize the full Phase 1B run."

echo
echo "=== CPU preflight (Phase A + Phase B: authority/dependency/subset"
echo "    verification, 200-window candidate-universe construction and"
echo "    selection) -- no torch, no CUDA, no dataset access ==="
python diagnostics/run_dfp_phase1b_calibration.py preflight \
    --output "$ARTIFACT_PATH" \
    --mode fresh

echo
echo "=== Execution (Phase C: canonical + masked k12 propagation," \
     "device=cuda, chunk_size=$DFP_CALIBRATION_COMPONENT_CHUNK_SIZE) ==="
python diagnostics/run_dfp_phase1b_calibration.py execute \
    --output "$ARTIFACT_PATH" \
    --mode fresh \
    --device cuda \
    --component-chunk-size "$DFP_CALIBRATION_COMPONENT_CHUNK_SIZE"

echo
echo "=== Independent artifact verification ==="
python diagnostics/run_dfp_phase1b_calibration.py verify-artifact \
    --artifact "$ARTIFACT_PATH"

echo
echo "DFP PHASE 1B CALIBRATION COCO-STUFF-200 LAUNCHER COMPLETE"
echo "artifact: $ARTIFACT_PATH"
echo "This is calibration only. A successful, complete artifact does NOT"
echo "authorize the full Phase 1B run -- see"
echo "docs/dfp_phase1b_calibration_run.md's 'What happens after this'."
