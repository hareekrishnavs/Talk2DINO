"""Named regression coverage for repository-hygiene invariants the trust/
centrality diagnostic work must never violate: no private filesystem path
leaking into tracked source, no raw per-target/prediction cache files
tracked in git, and the two known generated JSON report artifacts never
staged. These are lightweight, environment-independent checks over the
current working tree -- not GPU/mmseg-dependent."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]

PRIVATE_PATH_FRAGMENTS = ("/scratch/haree", "/home/haree", "/scratch/haree/venv")

GENERATED_ARTIFACT_NAMES = (
    "trust_centrality_full_fixed.json",
    "trust_centrality_pilot_5000.json",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


TRUST_CENTRALITY_TOUCHED_FILES = (
    "src/open_vocabulary_segmentation/main.py",
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/inference.py",
    "src/open_vocabulary_segmentation/models/dinotext/cover_dr/__init__.py",
    "src/open_vocabulary_segmentation/segmentation/evaluation/dinotext_seg.py",
    "src/rwr_reproduction_identity.py",
    "src/open_vocabulary_segmentation/segmentation/evaluation/trust_centrality_diagnostics.py",
    "src/open_vocabulary_segmentation/segmentation/evaluation/trust_centrality_harness.py",
    "src/open_vocabulary_segmentation/segmentation/evaluation/t4_audit.py",
)


def test_no_private_filesystem_path_in_the_files_this_repair_touched():
    # Scoped to exactly the files this trust/centrality repair edits or
    # adds -- not a blanket scan of the whole tree, which would also catch
    # unrelated pre-existing diagnostic scripts this task never touches.
    offenders = []
    for relative in TRUST_CENTRALITY_TOUCHED_FILES:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fragment in PRIVATE_PATH_FRAGMENTS:
            if fragment in text:
                offenders.append((relative, fragment))
    assert offenders == [], f"private filesystem paths leaked into tracked source: {offenders}"


def test_no_raw_prediction_or_cache_files_tracked_in_git():
    tracked = _git("ls-files").splitlines()
    forbidden_suffixes = (".pt", ".pth", ".pkl", ".npy", ".npz")
    offenders = [
        f for f in tracked
        if f.endswith(forbidden_suffixes) and "weights" not in f.split("/")
    ]
    assert offenders == [], f"raw prediction/cache-like files are tracked in git: {offenders}"


def test_generated_trust_centrality_json_artifacts_are_never_staged():
    staged = set(_git("diff", "--cached", "--name-only").splitlines())
    tracked = set(_git("ls-files").splitlines())
    for name in GENERATED_ARTIFACT_NAMES:
        assert name not in staged, f"generated artifact {name!r} must never be staged"
        assert name not in tracked, f"generated artifact {name!r} must never be committed"


def test_generated_trust_centrality_json_artifacts_stay_untracked_when_present():
    untracked = set(_git("ls-files", "--others", "--exclude-standard").splitlines())
    for name in GENERATED_ARTIFACT_NAMES:
        path = ROOT / name
        if path.exists():
            assert name in untracked, f"{name} exists but is not reported as untracked -- may have been added"
