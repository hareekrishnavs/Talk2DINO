#!/usr/bin/env python3
"""Preflight and result/checkpoint verification CLI for the reusable
stitching control suite."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from src.stitching_control_checkpoint import (
    parse_strict_json_document,
    validate_checkpoint_structure,
)
from src.stitching_control_identity import (
    StitchingControlIdentityError,
    load_identity,
    repository_root,
    validate_static_configuration,
)
from src.stitching_control_report import verify_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the stitching control suite identity and results/checkpoints"
    )
    parser.add_argument("--identity", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="validate repository configuration without loading models or data")
    preflight.add_argument("--repo-root", type=Path, default=repository_root())

    verify = commands.add_parser("verify-result", help="verify one completed stitching-control result")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, default=repository_root())

    checkpoint = commands.add_parser("verify-checkpoint", help="verify one stitching-control checkpoint")
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--repo-root", type=Path, default=repository_root())

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_static_configuration(repo_root=args.repo_root, identity_path=args.identity, check_git=True)
            print(
                "STITCHING CONTROL SUITE PREFLIGHT PASS "
                f"identity={result['identity_name']} "
                f"power_evaluation={result['power_evaluation_identity']} "
                f"matched={result['matched_identity']} "
                f"ancestor={result['required_ancestor_commit']} "
                f"pilot20={result['pilot20_image_count']} "
                f"pilot100={result['pilot100_image_count']} "
                f"full5000={result['full5000_image_count']} "
                f"variants={','.join(result['variant_names'])}"
            )
        elif args.command == "verify-result":
            print(verify_result(args.result, identity_path=args.identity, repo_root=args.repo_root))
        else:
            identity = load_identity(args.identity, repo_root=args.repo_root)
            resolved_identity_path = (
                args.identity if args.identity is not None
                else args.repo_root / "evaluation_identities/e12_stitching_control_suite.toml"
            )
            identity_sha256 = hashlib.sha256(Path(resolved_identity_path).read_bytes()).hexdigest()
            checkpoint = parse_strict_json_document(args.checkpoint, label="structured checkpoint")
            validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)
            status = "complete" if checkpoint["complete"] else (
                f"resumable at image {checkpoint['next_dataset_index']}/{checkpoint['image_count_expected']}"
            )
            print(
                "STITCHING CONTROL SUITE CHECKPOINT PASS "
                f"run_mode={checkpoint['run_mode']} images_recorded={checkpoint['next_dataset_index']} status={status}"
            )
    except StitchingControlIdentityError as error:
        print(f"STITCHING CONTROL SUITE VERIFICATION FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
