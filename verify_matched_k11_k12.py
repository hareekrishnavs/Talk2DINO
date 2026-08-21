#!/usr/bin/env python3
"""Preflight and result verification CLI for the matched k=11 versus k=12
finite-step (T=320) RWR connectivity dose-response identity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.matched_k11_k12_identity import (
    MatchedK11K12Error,
    repository_root,
    validate_static_configuration,
    verify_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the matched k11/k12 T=320 connectivity identity"
    )
    parser.add_argument("--identity", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "preflight",
        help="validate repository configuration without loading models or data",
    )
    preflight.add_argument("--repo-root", type=Path, default=repository_root())
    preflight.add_argument(
        "--check-checkpoint",
        action="store_true",
        help="also require the canonical projection checkpoint",
    )
    verify = commands.add_parser(
        "verify-result", help="verify one completed matched k11/k12 result"
    )
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, default=repository_root())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_static_configuration(
                repo_root=args.repo_root,
                identity_path=args.identity,
                check_checkpoint=args.check_checkpoint,
            )
            print(
                "MATCHED K11/K12 PREFLIGHT PASS "
                f"identity={result['identity_name']} "
                f"e3={result['e3_identity']} "
                f"rwr={result['rwr_identity']} "
                f"ancestor={result['required_ancestor_commit']} "
                f"window_count={result['window_count']}"
            )
        else:
            print(
                verify_result(
                    args.result,
                    identity_path=args.identity,
                    repo_root=args.repo_root,
                )
            )
    except MatchedK11K12Error as error:
        print(f"MATCHED K11/K12 VERIFICATION FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
