#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --job-name=dfp-phase1a-preflight
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# DFP Phase 1A Step 12: unified CLI verification/preflight bundle.
#
# CPU-only. No CUDA, no model, no dataset construction -- these are all
# read-only verify/preflight commands (diagnostics/run_dfp_phase1a.py),
# so no GPU resource is requested.
#
# Requires COCO_STUFF_REAL_DATA_ROOT, ADE20K_REAL_DATA_ROOT, and
# DFP_OUTPUT_ROOT to already be exported in the environment this script
# is submitted from (sbatch inherits it by default) -- never hardcoded
# here. Submit with:
#
#   export COCO_STUFF_REAL_DATA_ROOT=/path/to/coco_stuff_root
#   export ADE20K_REAL_DATA_ROOT=/path/to/ade20k_root
#   export DFP_OUTPUT_ROOT=/path/to/output_root
#   sbatch dfp_phase1a_preflight.sbatch

set -uo pipefail

REPO_ROOT="/project/6114407/haree/Talk2DINO"
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT" >&2; exit 1; }

for var in COCO_STUFF_REAL_DATA_ROOT ADE20K_REAL_DATA_ROOT DFP_OUTPUT_ROOT; do
    if [ -z "${!var:-}" ]; then
        echo "FATAL: \$$var is not set in the environment this job was submitted from." >&2
        echo "Export it and resubmit -- see the header comment in this script." >&2
        exit 1
    fi
done

module load opencv/4.14.0
source /scratch/haree/venv/talk2dino-a100/bin/activate

mkdir -p "$DFP_OUTPUT_ROOT"

STATUS=0
run_step() {
    echo
    echo "=== $* ==="
    if ! "$@"; then
        echo "--- FAILED: $* ---" >&2
        STATUS=1
    fi
}

run_step python diagnostics/run_dfp_phase1a.py verify-source-binding --dataset coco_stuff
run_step python diagnostics/run_dfp_phase1a.py verify-source-binding --dataset ade20k

run_step python diagnostics/run_dfp_phase1a.py verify-model-binding --dataset coco_stuff
run_step python diagnostics/run_dfp_phase1a.py verify-model-binding --dataset ade20k

run_step python diagnostics/run_dfp_phase1a.py preflight \
    --run coco_stuff_500 \
    --mode fresh \
    --output-root "$DFP_OUTPUT_ROOT/coco_stuff_500" \
    --probe-output-writeable

run_step python diagnostics/run_dfp_phase1a.py preflight \
    --run ade20k_500 \
    --mode fresh \
    --output-root "$DFP_OUTPUT_ROOT/ade20k_500" \
    --probe-output-writeable

echo
if [ "$STATUS" -eq 0 ]; then
    echo "ALL 6 DFP PHASE 1A PREFLIGHT STEPS PASSED"
else
    echo "ONE OR MORE DFP PHASE 1A PREFLIGHT STEPS FAILED -- see above" >&2
fi
exit "$STATUS"
