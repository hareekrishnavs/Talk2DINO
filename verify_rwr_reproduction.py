#!/usr/bin/env python3
"""Preflight and result verification CLI for canonical E3 directed RWR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.rwr_reproduction_identity import (
    RWRReproductionError,
    repository_root,
    validate_static_configuration,
    verify_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify canonical E3 RWR reproduction")
    parser.add_argument("--identity", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--repo-root", type=Path, default=repository_root())
    preflight.add_argument("--check-checkpoint", action="store_true")
    preflight.add_argument("--dataset-root", type=Path)
    preflight.add_argument("--weight-dir", type=Path)
    verify = commands.add_parser("verify-result")
    source = verify.add_mutually_exclusive_group(required=True)
    source.add_argument("--metrics-json", type=Path)
    source.add_argument("--log", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_static_configuration(
                repo_root=args.repo_root,
                identity_path=args.identity,
                check_checkpoint=args.check_checkpoint,
                dataset_root=args.dataset_root,
                weight_dir=args.weight_dir,
            )
            print(
                "RWR PREFLIGHT PASS "
                f"identity={result['identity_name']} "
                f"e3={result['e3_identity']} "
                f"source_e10={result['source_e10_commit']} "
                f"manifest_evidence={result['cache_manifest_evidence']} "
                f"manifest_archived={str(result['cache_manifest_archived']).lower()} "
                f"historical_cache_used={str(result['historical_cache_used_by_current_run']).lower()}"
            )
        else:
            path = args.metrics_json if args.metrics_json is not None else args.log
            kind = "json" if args.metrics_json is not None else "log"
            print(
                verify_result(
                    path,
                    source_kind=kind,
                    identity_path=args.identity,
                )
            )
    except RWRReproductionError as error:
        print(f"RWR REPRODUCTION FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
