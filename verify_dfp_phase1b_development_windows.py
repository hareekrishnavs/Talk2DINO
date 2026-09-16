#!/usr/bin/env python3
"""Independently reconstruct and verify a persisted DFP Phase 1B BALP
development window-universe manifest, and explicitly demonstrate that
confirmation IDs and the other 4,500 COCO-Stuff val2017 images are
rejected by the same scoped adapter.

CPU-only, model-free, GT-free (see
``src/dfp_phase1b_development_window_universe.py`` for the full
boundary contract). This is a genuinely separate second pass: it
rebuilds the real dataset resources from scratch and recomputes every
image's geometry and window plan again, rather than re-hashing objects
already in memory from the run that produced the persisted document.

Usage::

    python verify_dfp_phase1b_development_windows.py \\
        --repo-root . --manifest evaluation_subsets/dfp_phase1b_development_window_universe.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1b_development_window_universe import (
    MANIFEST_RELATIVE_PATH,
    DfpPhase1bDevelopmentWindowUniverseError,
    demonstrate_out_of_scope_rejection,
    independently_reconstruct_and_verify,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=str(_REPO_ROOT), help="repository root (default: this script's own directory)")
    parser.add_argument(
        "--manifest", default=str(MANIFEST_RELATIVE_PATH),
        help=f"persisted manifest path, repo-root-relative or absolute (default: {MANIFEST_RELATIVE_PATH})",
    )
    parser.add_argument(
        "--skip-rejection-probe", action="store_true",
        help="skip the explicit confirmation-ID / other-image-ID rejection demonstration (runs by default)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    print("=== DFP Phase 1B BALP development window-universe independent verification ===")
    print(f"Repo root: {repo_root}")
    print(f"Manifest:  {manifest_path}")

    status = 0

    print("\n--- pass 1: independent reconstruction from scratch ---")
    start = time.monotonic()
    try:
        reconstructed = independently_reconstruct_and_verify(repo_root=repo_root, persisted_path=manifest_path)
    except DfpPhase1bDevelopmentWindowUniverseError as error:
        print(f"FATAL: independent reconstruction disagreed with the persisted document -- {error}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - start
    print(
        f"PASS: independently reconstructed {reconstructed['development_count']} images / "
        f"{reconstructed['aggregate_window_count']} windows in {elapsed:.1f}s; "
        f"canonical_record_digest={reconstructed['canonical_record_digest']} matches the persisted document byte-for-byte"
    )

    if not args.skip_rejection_probe:
        print("\n--- pass 2: explicit confirmation-ID / other-image-ID rejection demonstration ---")
        try:
            results = demonstrate_out_of_scope_rejection(repo_root=repo_root)
        except DfpPhase1bDevelopmentWindowUniverseError as error:
            print(f"FATAL: {error}", file=sys.stderr)
            return 1
        for label, outcome in results.items():
            print(f"PASS: {label} {outcome['probe_id']!r} correctly rejected -- {outcome['error']}")

    print("\nOVERALL VERDICT: PASS" if status == 0 else "\nOVERALL VERDICT: FAIL")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
