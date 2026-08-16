#!/usr/bin/env python3
"""Preflight and result verification CLI for canonical E3 evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.e3_evaluation_identity import (
    E3IdentityError,
    IDENTITY_RELATIVE_PATH,
    repository_root,
    validate_static_configuration,
    verify_result,
)


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Validate the frozen Talk2DINO E3 evaluation identity."
    )
    root.add_argument(
        "--identity",
        type=Path,
        default=None,
        help=f"identity TOML (default: {IDENTITY_RELATIVE_PATH})",
    )
    commands = root.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight",
        help="validate repository configuration without loading models or data",
    )
    preflight.add_argument("--repo-root", type=Path, default=repository_root())
    preflight.add_argument("--eval-config", type=Path)
    preflight.add_argument("--eval-base-config", type=Path)
    preflight.add_argument(
        "--check-checkpoint",
        action="store_true",
        help="also require the canonical projection checkpoint",
    )
    preflight.add_argument(
        "--dataset-root",
        type=Path,
        help="also check images/val2017 and annotations/val2017 below this root",
    )
    preflight.add_argument(
        "--weight-dir",
        type=Path,
        help="also check the canonical DINOv2 and CLIP files in this directory",
    )

    verify = commands.add_parser(
        "verify-result", help="verify one completed evaluation result"
    )
    source = verify.add_mutually_exclusive_group(required=True)
    source.add_argument("--metrics-json", type=Path)
    source.add_argument("--log", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_static_configuration(
                repo_root=args.repo_root,
                identity_path=args.identity,
                eval_config=args.eval_config,
                eval_base_config=args.eval_base_config,
                check_checkpoint=args.check_checkpoint,
                dataset_root=args.dataset_root,
                weight_dir=args.weight_dir,
            )
            checks = ["configuration", "ancestry"]
            if result["checkpoint_checked"]:
                checks.append("projection-checkpoint")
            if result["dataset_checked"]:
                checks.append("dataset-root")
            if result["external_weights_checked"]:
                checks.append("backbone/CLIP-weights")
            print(
                f"E3 PREFLIGHT PASS identity={result['identity_name']} "
                f"checks={','.join(checks)}"
            )
        else:
            source_path = args.metrics_json if args.metrics_json is not None else args.log
            source_kind = "structured" if args.metrics_json is not None else "log"
            print(
                verify_result(
                    source_path,
                    source_kind=source_kind,
                    identity_path=args.identity,
                )
            )
    except E3IdentityError as error:
        print(f"E3 IDENTITY FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
