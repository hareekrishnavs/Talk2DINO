#!/usr/bin/env python3
"""Preflight and result/checkpoint verification CLI for the bounded k11/k12
finite-step stability gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.k11_k12_stability_gate_identity import (
    K11K12StabilityGateError,
    repository_root,
    validate_static_configuration,
)
from src.k11_k12_stability_report import (
    verify_checkpoint,
    verify_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the bounded k11/k12 finite-step stability gate"
    )
    parser.add_argument("--identity", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight", help="validate repository configuration without loading models or data"
    )
    preflight.add_argument("--repo-root", type=Path, default=repository_root())

    verify_result_parser = commands.add_parser(
        "verify-result", help="verify one completed stability-gate result"
    )
    verify_result_parser.add_argument("--result", type=Path, required=True)
    verify_result_parser.add_argument("--repo-root", type=Path, default=repository_root())

    verify_checkpoint_parser = commands.add_parser(
        "verify-checkpoint", help="verify one stability-gate checkpoint file"
    )
    verify_checkpoint_parser.add_argument("--checkpoint", type=Path, required=True)
    verify_checkpoint_parser.add_argument("--repo-root", type=Path, default=repository_root())

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_static_configuration(
                repo_root=args.repo_root, identity_path=args.identity, check_git=True
            )
            print(
                "K11/K12 STABILITY GATE PREFLIGHT PASS "
                f"identity={result['identity_name']} "
                f"matched={result['matched_identity']} "
                f"ancestor={result['required_ancestor_commit']} "
                f"canonical_window_count={result['canonical_window_count']}"
            )
        elif args.command == "verify-result":
            print(
                verify_result(
                    args.result, identity_path=args.identity, repo_root=args.repo_root
                )
            )
        else:
            print(
                verify_checkpoint(
                    args.checkpoint, identity_path=args.identity, repo_root=args.repo_root
                )
            )
    except K11K12StabilityGateError as error:
        print(f"K11/K12 STABILITY GATE VERIFICATION FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
