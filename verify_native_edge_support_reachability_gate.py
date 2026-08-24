#!/usr/bin/env python3
"""Preflight and result verification CLI for the offline 20-image
structural reachability gate. No CUDA/model/dataset loading in either
subcommand."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from src.native_edge_support_reachability_gate_identity import (
    NativeEdgeSupportReachabilityGateIdentityError,
    load_identity,
    repository_root,
    validate_static_configuration,
)
from src.native_edge_support_reachability_gate_report import verify_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the native-edge-support-reachability-gate identity and results"
    )
    parser.add_argument("--identity", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="validate repository configuration without loading models, data, or CUDA")
    preflight.add_argument("--repo-root", type=Path, default=repository_root())

    verify = commands.add_parser("verify-result", help="verify one completed reachability-gate result")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, default=repository_root())

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_static_configuration(repo_root=args.repo_root, identity_path=args.identity, check_git=True)
            print(
                "NATIVE EDGE SUPPORT REACHABILITY GATE PREFLIGHT PASS "
                f"identity={result['identity_name']} "
                f"native_audit={result['native_audit_identity']} "
                f"ancestor={result['required_ancestor_commit']} "
                f"decision_outcomes={','.join(result['decision_outcomes'])}"
            )
        else:
            print(verify_result(args.result, identity_path=args.identity, repo_root=args.repo_root))
    except NativeEdgeSupportReachabilityGateIdentityError as error:
        print(f"NATIVE EDGE SUPPORT REACHABILITY GATE VERIFICATION FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
