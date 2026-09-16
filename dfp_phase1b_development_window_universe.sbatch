#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=dfp-phase1b-dev-window-universe
#SBATCH --output=%x-%j.out

# DFP Phase 1B: materialize + independently verify the frozen 350-image
# BALP development window universe.
#
# CPU-only, model-free, GT-free -- no CUDA, no model, no recurrence, no
# BALP evaluation. Only real image metadata/pixel reads through the
# already-committed production dataset/geometry adapter, scoped to
# exactly the 350 frozen development IDs. See
# src/dfp_phase1b_development_window_universe.py for the full boundary
# contract this job enforces by construction.
#
# Required environment (sbatch inherits the submitting shell's exported
# variables by default -- nothing here is hardcoded):
#   TALK2DINO_VENV   -- path to the project virtual environment
#     (defaults to /scratch/haree/venv/talk2dino-a100 if unset)
#
# Submit from the repository root with:
#   sbatch dfp_phase1b_development_window_universe.sbatch

set -euo pipefail

REPO_ROOT="/project/6114407/haree/Talk2DINO"
EXPECTED_BRANCH="dfp-phase1a-diagnostics"
VENV="${TALK2DINO_VENV:-/scratch/haree/venv/talk2dino-a100}"
OUTPUT="${REPO_ROOT}/evaluation_subsets/dfp_phase1b_development_window_universe.json"

cd "${REPO_ROOT}"

echo "=== DFP Phase 1B development window-universe materialization job ==="
echo "Job ID:     ${SLURM_JOB_ID:-unknown}"
echo "Hostname:   $(hostname)"
date -u +"Start (UTC): %Y-%m-%dT%H:%M:%SZ"

echo "Git commit: $(git rev-parse HEAD)"
echo "Git branch: $(git branch --show-current)"
if [ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]; then
    echo "FATAL: expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)" >&2
    exit 1
fi

if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
    echo "FATAL: tracked worktree has uncommitted changes; commit or stash before submitting" >&2
    git diff --name-only >&2
    git diff --cached --name-only >&2
    exit 1
fi

for required_file in \
    "src/dfp_phase1b_development_window_universe.py" \
    "materialize_dfp_phase1b_development_windows.py" \
    "verify_dfp_phase1b_development_windows.py" \
    "evaluation_subsets/dfp_phase1b_lightweight_boundary_500_split.json"
do
    if ! git ls-files --error-unmatch "${required_file}" > /dev/null 2>&1; then
        echo "FATAL: ${required_file} is not committed; refusing to run an uncommitted materializer" >&2
        exit 1
    fi
done

echo "=== activating environment ==="
module load StdEnv/2023 gcc/12.3 opencv/4.14.0
source "${VENV}/bin/activate"
export PYTHONPATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcc12/opencv/4.14.0/lib/python3.11/site-packages:${PYTHONPATH:-}"
PYTHON="${VENV}/bin/python"
echo "Python: $(${PYTHON} --version)"

echo "=== materializing the 350-image development window universe ==="
"${PYTHON}" materialize_dfp_phase1b_development_windows.py --repo-root "${REPO_ROOT}" --output "${OUTPUT}"

echo "=== independently reconstructing and verifying the manifest ==="
"${PYTHON}" verify_dfp_phase1b_development_windows.py --repo-root "${REPO_ROOT}" --manifest "${OUTPUT}"

date -u +"End (UTC): %Y-%m-%dT%H:%M:%SZ"
echo "Manifest: ${OUTPUT}"
