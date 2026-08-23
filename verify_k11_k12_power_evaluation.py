#!/usr/bin/env python3
"""Preflight and result/checkpoint verification CLI for the matched
k11-vs-k12 finite-step power evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.k11_k12_power_evaluation_checkpoint import (
    parse_strict_json_document,
    validate_checkpoint_structure,
)
from src.k11_k12_power_evaluation_identity import (
    K11K12PowerEvaluationError,
    repository_root,
    validate_static_configuration,
    validate_stability_result_binding,
)
from src.k11_k12_power_evaluation_report import verify_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the matched k11/k12 power evaluator identity, stability-result binding, and results"
    )
    parser.add_argument("--identity", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight", help="validate repository configuration without loading models or data"
    )
    preflight.add_argument("--repo-root", type=Path, default=repository_root())

    binding = commands.add_parser(
        "verify-stability-binding",
        help="validate a --stability-result file's structural correctness and provenance binding",
    )
    binding.add_argument("--stability-result", type=Path, required=True)
    binding.add_argument("--repo-root", type=Path, default=repository_root())

    verify = commands.add_parser("verify-result", help="verify one completed power-evaluation result")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, default=repository_root())

    checkpoint = commands.add_parser("verify-checkpoint", help="verify one power-evaluation checkpoint")
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--repo-root", type=Path, default=repository_root())

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_static_configuration(repo_root=args.repo_root, identity_path=args.identity, check_git=True)
            print(
                "K11/K12 POWER EVALUATION PREFLIGHT PASS "
                f"identity={result['identity_name']} "
                f"matched={result['matched_identity']} "
                f"stability_gate={result['stability_gate_identity']} "
                f"ancestor={result['required_ancestor_commit']} "
                f"pilot20={result['pilot20_image_count']} "
                f"pilot100={result['pilot100_image_count']} "
                f"full={result['full_image_count']}"
            )
        elif args.command == "verify-stability-binding":
            from src.k11_k12_power_evaluation_identity import load_identity

            identity = load_identity(args.identity, repo_root=args.repo_root)
            binding = validate_stability_result_binding(
                args.stability_result, identity=identity, repo_root=args.repo_root, check_git=True
            )
            print(
                "K11/K12 POWER EVALUATION STABILITY BINDING PASS "
                f"classification={binding['gate_classification']} "
                f"gate_commit={binding['gate_git_commit']}"
            )
        elif args.command == "verify-result":
            print(verify_result(args.result, identity_path=args.identity, repo_root=args.repo_root))
        else:
            from src.k11_k12_power_evaluation_identity import load_identity

            identity = load_identity(args.identity, repo_root=args.repo_root)
            resolved_identity_path = (
                args.identity if args.identity is not None
                else args.repo_root / "evaluation_identities/e12_k11_k12_power_evaluation.toml"
            )
            import hashlib

            identity_sha256 = hashlib.sha256(Path(resolved_identity_path).read_bytes()).hexdigest()
            checkpoint = parse_strict_json_document(args.checkpoint, label="structured checkpoint")
            validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)
            status = "complete" if checkpoint["complete"] else (
                f"resumable at image {checkpoint['next_dataset_index']}/{checkpoint['image_count_expected']}"
            )
            print(
                "K11/K12 POWER EVALUATION CHECKPOINT PASS "
                f"run_mode={checkpoint['run_mode']} images_recorded={checkpoint['next_dataset_index']} status={status}"
            )
    except K11K12PowerEvaluationError as error:
        print(f"K11/K12 POWER EVALUATION VERIFICATION FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
