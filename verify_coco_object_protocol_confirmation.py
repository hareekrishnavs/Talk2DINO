#!/usr/bin/env python3
"""Verifier for the COCO-Object protocol-confirmation identity/checkpoint/
result. ``preflight`` validates identity/configuration/materialization
binding without CUDA/model. ``verify-checkpoint`` validates an incomplete
resumable checkpoint. ``verify-result`` validates one completed result."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

from src.coco_object_protocol_confirmation_checkpoint import validate_checkpoint_structure
from src.coco_object_protocol_confirmation_identity import (
    CocoObjectProtocolConfirmationIdentityError,
    load_identity,
    validate_static_configuration,
)
from src.coco_object_protocol_confirmation_report import verify_record
from src.native_edge_support_checkpoint import parse_strict_json_document


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the COCO-Object protocol-confirmation identity/checkpoint/result")
    parser.add_argument("--identity", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="validate identity/configuration/materialization-binding without CUDA/model")
    preflight.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    preflight.add_argument("--materialization-manifest", type=Path, default=None)
    preflight.add_argument("--data-root", type=Path, default=None)
    preflight.add_argument("--source-masks", type=Path, default=None)
    preflight.add_argument("--source-images", type=Path, default=None)

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

    checks = ["configuration", "matched_parent", "materialization_parent", "bridge_checkpoint_binding"]
    if args.materialization_manifest is not None and args.data_root is not None and args.source_masks is not None and args.source_images is not None:
        manifest = parse_strict_json_document(args.materialization_manifest, label="materialization manifest")
        verification = identity["verification"]
        if manifest["schema"] != verification["materialization_manifest_schema_name"]:
            raise CocoObjectProtocolConfirmationIdentityError("materialization manifest schema disagrees with the identity's registered contract")
        if manifest["complete"] is not True or manifest["final"] is not True:
            raise CocoObjectProtocolConfirmationIdentityError("materialization manifest must be complete and final")
        if manifest["image_count"] != verification["required_manifest_image_count"]:
            raise CocoObjectProtocolConfirmationIdentityError("materialization manifest.image_count disagrees with the identity's requirement")

        verify_script = root / "verify_coco_object_val_materialization.py"
        proc = subprocess.run(
            [
                sys.executable, str(verify_script), "verify-output",
                "--repo-root", str(root), "--manifest", str(args.materialization_manifest),
                "--output-root", str(args.data_root), "--source-masks", str(args.source_masks),
                "--source-images", str(args.source_images),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise CocoObjectProtocolConfirmationIdentityError(
                f"materialization verify-output failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        checks.append("materialization_verify_output")

        import tomllib
        dataset_config_path = root / identity["dataset"]["dataset_config_relative_path"]
        text = dataset_config_path.read_text()
        if identity["dataset_root_override"]["canonical_configured_root"] not in text:
            raise CocoObjectProtocolConfirmationIdentityError(
                "coco.py no longer contains the identity's registered canonical_configured_root"
            )
        checks.append("dataset_root_override_binding")

    print(
        f"COCO-OBJECT PROTOCOL CONFIRMATION PREFLIGHT PASS identity={identity['identity']['name']} "
        f"matched={result['matched_identity_name']} materialization={result['materialization_identity_name']} "
        f"checks={','.join(checks)}"
    )
    return 0


def _cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_coco_object_protocol_confirmation.toml"
    identity_sha256 = _sha256_file(identity_path)

    checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
    validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256)

    print(
        f"COCO-OBJECT PROTOCOL CONFIRMATION CHECKPOINT PASS run_mode={checkpoint['run_mode']} "
        f"next_dataset_index={checkpoint['next_dataset_index']} "
        f"completed={len(checkpoint['completed_image_ids'])} complete={checkpoint['complete']}"
    )
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)
    identity = load_identity(args.identity, repo_root=root)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_coco_object_protocol_confirmation.toml"
    identity_sha256 = _sha256_file(identity_path)

    record = parse_strict_json_document(args.result, label="result")
    verify_record(record, identity, identity_sha256=identity_sha256)

    print(
        f"COCO-OBJECT PROTOCOL CONFIRMATION RESULT PASS run_mode={record['run_mode']} "
        f"mIoU_E3={record['metrics_E3']['mIoU']:.6f} mIoU_k11={record['metrics_k11']['mIoU']:.6f} "
        f"mIoU_k12={record['metrics_k12']['mIoU']:.6f} "
        f"delta_k11_k12={record['delta_mIoU_k11_minus_k12_percentage_points']:.6f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "verify-checkpoint":
            return _cmd_verify_checkpoint(args)
        if args.command == "verify-result":
            return _cmd_verify_result(args)
        parser.error(f"unknown command {args.command!r}")
        return 2
    except (CocoObjectProtocolConfirmationIdentityError, ValueError) as error:
        print(f"COCO-OBJECT PROTOCOL CONFIRMATION VERIFY FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
