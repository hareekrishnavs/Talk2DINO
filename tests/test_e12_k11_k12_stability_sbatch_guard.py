"""Direct tests of the SLURM script's tracked-only clean-worktree guard.

Extracts the guard's actual bash lines from the real
``scripts/slurm/e12_k11_k12_stability_h100.sbatch`` (located by its own
marker comment, not duplicated/reimplemented here) and executes them
directly inside temporary, disposable Git repositories -- the real
repository is never touched or dirtied by these tests.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SBATCH_PATH = ROOT / "scripts/slurm/e12_k11_k12_stability_h100.sbatch"


def _extract_guard_snippet() -> str:
    """Pull out exactly the tracked-worktree-cleanliness guard block from
    the real SBATCH script, between its start marker comment and the
    matching ``fi``."""
    lines = SBATCH_PATH.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# Reject TRACKED changes only"))
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "fi")
    return "\n".join(lines[start:end + 1])


GUARD_SNIPPET = _extract_guard_snippet()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


@pytest.fixture
def temp_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked.txt").write_text("original\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _run_guard(repo: Path) -> subprocess.CompletedProcess:
    script = f"set -euo pipefail\ncd {repo}\n{GUARD_SNIPPET}\necho GUARD_PASSED\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_guard_extraction_found_a_nonempty_snippet():
    assert "git diff" in GUARD_SNIPPET
    assert "exit 1" in GUARD_SNIPPET


# ---------------------------------------------------------------------
# 1. Clean tracked state passes
# ---------------------------------------------------------------------


def test_1_clean_tracked_state_passes(temp_repo):
    result = _run_guard(temp_repo)
    assert result.returncode == 0, result.stderr
    assert "GUARD_PASSED" in result.stdout


# ---------------------------------------------------------------------
# 2. Untracked file alone does not fail the guard
# ---------------------------------------------------------------------


def test_2_untracked_file_alone_does_not_fail(temp_repo):
    (temp_repo / "untracked_local_artifact.json").write_text("{}\n")
    result = _run_guard(temp_repo)
    assert result.returncode == 0, result.stderr
    assert "GUARD_PASSED" in result.stdout


def test_2b_multiple_untracked_files_still_pass(temp_repo):
    (temp_repo / "a.json").write_text("{}\n")
    (temp_repo / "b.log").write_text("log\n")
    result = _run_guard(temp_repo)
    assert result.returncode == 0
    assert "GUARD_PASSED" in result.stdout


# ---------------------------------------------------------------------
# 3. Unstaged tracked modification fails
# ---------------------------------------------------------------------


def test_3_unstaged_tracked_modification_fails(temp_repo):
    (temp_repo / "tracked.txt").write_text("modified\n")
    result = _run_guard(temp_repo)
    assert result.returncode != 0
    assert "GUARD_PASSED" not in result.stdout
    assert "FATAL" in result.stderr
    assert "tracked.txt" in result.stderr


# ---------------------------------------------------------------------
# 4. Staged tracked modification fails
# ---------------------------------------------------------------------


def test_4_staged_tracked_modification_fails(temp_repo):
    (temp_repo / "tracked.txt").write_text("staged-change\n")
    _git(temp_repo, "add", "tracked.txt")
    result = _run_guard(temp_repo)
    assert result.returncode != 0
    assert "GUARD_PASSED" not in result.stdout
    assert "FATAL" in result.stderr
    assert "tracked.txt" in result.stderr


def test_4b_new_staged_tracked_file_fails(temp_repo):
    (temp_repo / "new_file.txt").write_text("new\n")
    _git(temp_repo, "add", "new_file.txt")
    result = _run_guard(temp_repo)
    assert result.returncode != 0
    assert "GUARD_PASSED" not in result.stdout


def test_diagnostic_does_not_dump_full_diff_content(temp_repo):
    (temp_repo / "tracked.txt").write_text("A" * 5000 + "\n")
    result = _run_guard(temp_repo)
    assert result.returncode != 0
    # only filenames (via --name-only), never the actual changed content
    assert "A" * 100 not in result.stderr


# ---------------------------------------------------------------------
# Real-script structural checks (5, 6, 7)
# ---------------------------------------------------------------------


def test_5_guard_appears_before_preflight_and_model_cuda_commands():
    lines = SBATCH_PATH.read_text().splitlines()
    guard_index = next(i for i, l in enumerate(lines) if l.startswith("# Reject TRACKED changes only"))
    preflight_index = next(i for i, l in enumerate(lines) if "verify_e3_identity.py preflight" in l)
    gate_index = next(i for i, l in enumerate(lines) if "run_k11_k12_stability.py" in l and "--device cuda" in "\n".join(lines[i:i+8]))
    assert guard_index < preflight_index
    assert guard_index < gate_index


def test_6_bash_n_passes():
    result = subprocess.run(["bash", "-n", str(SBATCH_PATH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_7_script_never_modifies_resets_or_stashes_the_worktree():
    source = SBATCH_PATH.read_text()
    forbidden = ["git reset", "git checkout", "git stash", "git clean", "git commit", "git add", "git rm"]
    found = [tok for tok in forbidden if tok in source]
    assert not found, f"SLURM script must never modify the worktree, found: {found}"


def test_branch_check_still_present():
    source = SBATCH_PATH.read_text()
    assert 'EXPECTED_BRANCH="e12-connectivity-analysis"' in source
    assert "git branch --show-current" in source


def test_git_commit_is_echoed_and_recorded_in_result():
    source = SBATCH_PATH.read_text()
    assert 'git rev-parse HEAD' in source
    runner_source = (ROOT / "diagnostics/run_k11_k12_stability.py").read_text()
    assert '"git_commit": git_commit' in runner_source


def test_required_harness_files_check_still_present():
    source = SBATCH_PATH.read_text()
    assert "is not committed; refusing to run an uncommitted harness" in source
