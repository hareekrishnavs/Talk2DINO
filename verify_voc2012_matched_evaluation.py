#!/usr/bin/env python3
"""Verifier for the shared VOC2012 V20/V21 matched-evaluator identity/
checkpoint/result. ``preflight`` validates identity/configuration/
parent-identity/checkpoint-bytes binding without CUDA/model.
``verify-source-binding`` additionally verifies a real VOC2012 dataset-
source manifest against a real data root (still no CUDA/model).
``verify-checkpoint`` validates an incomplete resumable checkpoint.
``verify-result`` validates one completed result. Execution (pilot20/
pilot100/full) lives in diagnostics/run_voc2012_matched_evaluation.py,
kept separate from verification per established repository convention."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

from src.native_edge_support_checkpoint import parse_strict_json_document
from src.voc2012_matched_evaluator_checkpoint import validate_checkpoint_structure
from src.voc2012_matched_evaluator_identity import (
    Voc2012MatchedEvaluatorIdentityError,
    load_identity,
    validate_static_configuration,
)
from src.voc2012_matched_evaluator_report import verify_record


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the shared VOC2012 V20/V21 matched-evaluator identity/checkpoint/result")
    parser.add_argument("--identity", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="validate identity/configuration/parent-identity/checkpoint-bytes binding without CUDA/model")
    preflight.add_argument("--repo-root", type=Path, default=_REPO_ROOT)

    binding = sub.add_parser("verify-source-binding", help="verify a real VOC2012 dataset-source manifest against a real data root, no CUDA/model")
    binding.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    binding.add_argument("--data-root", type=Path, required=True)
    binding.add_argument("--source-manifest", type=Path, required=True)

    verify_ckpt = sub.add_parser("verify-checkpoint", help="validate an incomplete resumable checkpoint")
    verify_ckpt.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    verify_ckpt.add_argument("--checkpoint", type=Path, required=True)

    verify_result = sub.add_parser("verify-result", help="validate one completed result")
    verify_result.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    verify_result.add_argument("--result", type=Path, required=True)

    return parser


def _cmd_preflight(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    result = validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)

    print(
        f"VOC2012 MATCHED EVALUATOR PREFLIGHT PASS identity={identity['identity']['name']} "
        f"voc2012_source={result['voc2012_source_identity_name']} matched={result['matched_identity_name']} "
        f"checks=configuration,voc2012_source_parent,matched_parent,model_and_checkpoint_binding"
    )
    return 0


def _cmd_verify_source_binding(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)

    manifest = parse_strict_json_document(args.source_manifest, label="VOC2012 source manifest")
    if manifest["image_count"] != identity["protocol"]["expected_image_count"]:
        raise Voc2012MatchedEvaluatorIdentityError("VOC2012 source manifest image_count disagrees with the identity's registered expected_image_count")

    verify_script = root / "verify_voc2012_dataset.py"
    proc = subprocess.run(
        [sys.executable, str(verify_script), "verify-manifest", "--repo-root", str(root), "--data-root", str(args.data_root), "--manifest", str(args.source_manifest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise Voc2012MatchedEvaluatorIdentityError(
            f"VOC2012 verify-manifest failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    print(
        f"VOC2012 MATCHED EVALUATOR SOURCE-BINDING PASS identity={identity['identity']['name']} "
        f"image_count={manifest['image_count']} checks=configuration,source_manifest_verify_manifest"
    )
    return 0


def _cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
    identity_sha256 = _sha256_file(identity_path)

    checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)

    print(
        f"VOC2012 MATCHED EVALUATOR CHECKPOINT PASS run_mode={checkpoint['run_mode']} "
        f"next_dataset_index={checkpoint['next_dataset_index']} "
        f"completed={len(checkpoint['completed_image_ids'])} complete={checkpoint['complete']}"
    )
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_voc2012_matched_evaluator.toml"
    identity_sha256 = _sha256_file(identity_path)

    record = parse_strict_json_document(args.result, label="result")
    verify_record(record, identity, identity_sha256=identity_sha256)

    print(
        f"VOC2012 MATCHED EVALUATOR RESULT PASS run_mode={record['run_mode']} "
        f"mIoU_v20_k11={record['metrics_v20_k11']['mIoU']:.6f} mIoU_v20_k12={record['metrics_v20_k12']['mIoU']:.6f} "
        f"mIoU_v21_k11={record['metrics_v21_k11']['mIoU']:.6f} mIoU_v21_k12={record['metrics_v21_k12']['mIoU']:.6f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "verify-source-binding":
            return _cmd_verify_source_binding(args)
        if args.command == "verify-checkpoint":
            return _cmd_verify_checkpoint(args)
        if args.command == "verify-result":
            return _cmd_verify_result(args)
        parser.error(f"unknown command {args.command!r}")
        return 2
    except (Voc2012MatchedEvaluatorIdentityError, ValueError) as error:
        print(f"VOC2012 MATCHED EVALUATOR VERIFY FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
