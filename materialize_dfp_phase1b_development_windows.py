#!/usr/bin/env python3
"""Materialize and freeze the exact 350-image DFP Phase 1B BALP
development window universe.

CPU-only, model-free, GT-free: runs the already-committed production
dataset/geometry adapter (``src.dfp_phase1b_calibration_production``)
against the real 350 frozen development images only, and the canonical
``SlidingWindowPlan.build`` primitive for window construction. No CUDA,
no model, no recurrence, no BALP evaluation -- see
``src/dfp_phase1b_development_window_universe.py`` for the full
boundary contract this script enforces by construction.

Usage::

    python materialize_dfp_phase1b_development_windows.py \\
        --repo-root . --output evaluation_subsets/dfp_phase1b_development_window_universe.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1b_development_window_universe import (
    MANIFEST_RELATIVE_PATH,
    DfpPhase1bDevelopmentWindowUniverseError,
    materialize_development_window_universe,
)


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, check=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json_atomically(document: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f".staging-{os.getpid()}")
    payload = json.dumps(document, ensure_ascii=True, sort_keys=False, indent=2) + "\n"
    staging.write_text(payload, encoding="utf-8")
    os.replace(staging, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=str(_REPO_ROOT), help="repository root (default: this script's own directory)")
    parser.add_argument(
        "--output", default=str(MANIFEST_RELATIVE_PATH),
        help=f"output path, repo-root-relative or absolute (default: {MANIFEST_RELATIVE_PATH})",
    )
    parser.add_argument(
        "--allow-overwrite", action="store_true",
        help="allow overwriting an existing output file (refused by default -- this artifact is frozen once written)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    if output_path.exists() and not args.allow_overwrite:
        print(f"FATAL: {output_path} already exists; pass --allow-overwrite to replace it", file=sys.stderr)
        return 1

    print("=== DFP Phase 1B BALP development window-universe materialization ===")
    print(f"Repo root:   {repo_root}")
    print(f"Git commit:  {_git_commit(repo_root)}")
    print(f"Output:      {output_path}")
    print(f"Start (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    start = time.monotonic()

    def _progress(rank: int, canonical_id: str, window_count: int) -> None:
        if rank == 0 or (rank + 1) % 50 == 0:
            elapsed = time.monotonic() - start
            print(f"  [{rank + 1:>3}/350] {canonical_id} -> {window_count} windows ({elapsed:.1f}s elapsed)")

    try:
        document = materialize_development_window_universe(repo_root=repo_root, progress_callback=_progress)
    except DfpPhase1bDevelopmentWindowUniverseError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - start
    print(f"Materialized {document['development_count']} images, {document['aggregate_window_count']} windows total in {elapsed:.1f}s")
    print(f"canonical_record_digest = {document['canonical_record_digest']}")

    _write_json_atomically(document, output_path)
    print(f"Wrote {output_path}")
    print(f"End (UTC):   {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
