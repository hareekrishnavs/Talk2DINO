#!/usr/bin/env python3
"""CPU/static verifier for the DFP Phase 1A protocol pre-registration
identity.

Never initializes CUDA, never constructs a dataset, never accesses
dataset images, never loads a model, never deserializes the bridge
checkpoint as a model (only hashes its bytes). Never writes outside an
explicit temporary test path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1a_identity import (
    DfpPhase1aIdentityError,
    validate_static_configuration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the DFP Phase 1A protocol pre-registration identity")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--no-git-check", action="store_true", help="skip the required_ancestor_commit ancestry check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_static_configuration(
            repo_root=args.repo_root, identity_path=args.identity, check_git=not args.no_git_check,
        )
    except (DfpPhase1aIdentityError, OSError, ValueError) as error:
        print(f"DFP PHASE 1A PROTOCOL VERIFY FAIL: {error}", file=sys.stderr)
        return 2

    print("PASS: DFP Phase 1A protocol identity verified")
    print(
        f"# identity={result['identity_name']} ancestor={result['required_ancestor_commit']} "
        f"coco_stuff_images={result['coco_stuff_images']} ade20k_images={result['ade20k_images']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
