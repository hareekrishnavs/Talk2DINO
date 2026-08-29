#!/usr/bin/env python3
"""CPU-only standalone verifier for DFP Phase 1A Step 10 checkpoint/
artifact infrastructure.

Never accesses a real dataset root, never constructs a dataset, never
imports/initializes CUDA, never builds a model, never loads model weights
into torch, and never runs graph/recurrence code. Every subcommand is a
pure static/CPU validation of already-written JSON/NPZ artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1a_artifact_schema import (
    DfpPhase1aArtifactSchemaError,
    parse_strict_json_document,
    sha256_file,
    validate_static_configuration,
)
from src.dfp_phase1a_checkpoint import (
    DfpPhase1aCheckpointError,
    validate_checkpoint_against_per_image_stats,
    validate_checkpoint_structure,
    validate_per_image_stats_manifest_structure,
    validate_resume_state,
)
from src.dfp_phase1a_report import DfpPhase1aReportError, verify_result_bundle

_KNOWN_ERRORS = (DfpPhase1aArtifactSchemaError, DfpPhase1aCheckpointError, DfpPhase1aReportError, OSError, ValueError)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only verifier for DFP Phase 1A checkpoint/artifact infrastructure")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    schema_cmd = sub.add_parser("verify-schema", help="verify the artifact-schema authority identity")
    schema_cmd.add_argument("--identity", type=Path, default=None)

    ckpt_cmd = sub.add_parser("verify-checkpoint", help="verify one checkpoint document's structure and per-image-stats binding")
    ckpt_cmd.add_argument("--checkpoint", type=Path, required=True)

    resume_cmd = sub.add_parser("verify-resume", help="run the full ordered Phase-A resume validation against a checkpoints directory")
    resume_cmd.add_argument("--checkpoints-dir", type=Path, required=True)
    resume_cmd.add_argument("--checkpoint-filename", type=str, default="checkpoint.json")

    result_cmd = sub.add_parser("verify-result", help="verify an installed final-result bundle directory")
    result_cmd.add_argument("--bundle", type=Path, required=True)

    return parser


def _cmd_verify_schema(args: argparse.Namespace) -> int:
    result = validate_static_configuration(repo_root=args.repo_root, identity_path=args.identity)
    print("PASS: DFP Phase 1A artifact-schema authority verified")
    print(f"# identity={result['identity_name']} protocol_sha256={result['protocol_identity_sha256']}", file=sys.stderr)
    return 0


def _cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity
    import src.dfp_phase1a_artifact_schema as sch

    schema_summary = validate_static_configuration(repo_root=args.repo_root)
    artifact_identity = sch.load_identity(repo_root=args.repo_root)
    protocol = load_protocol_identity(args.repo_root / artifact_identity["protocol_binding"]["protocol_identity_relative_path"], repo_root=args.repo_root)

    run_mode = checkpoint.get("run_mode")
    manifest_sha256_key = "coco_stuff_manifest_sha256" if run_mode == "coco_stuff_500" else "ade20k_manifest_sha256"
    validate_checkpoint_structure(
        checkpoint,
        artifact_schema_identity_sha256=sch.sha256_file(args.repo_root / sch.IDENTITY_RELATIVE_PATH),
        protocol_identity_sha256=schema_summary["protocol_identity_sha256"],
        subset_manifest_sha256=schema_summary.get(manifest_sha256_key, ""),
        bridge_checkpoint_sha256=schema_summary["bridge_checkpoint_sha256"],
        required_ancestor_commit=protocol["identity"]["required_ancestor_commit"],
    )

    stats_manifest_path = args.checkpoint.parent / checkpoint["per_image_stats_manifest_relative_path"]
    stats_manifest = parse_strict_json_document(stats_manifest_path, label="per-image-stats manifest")
    validate_per_image_stats_manifest_structure(stats_manifest)
    if sha256_file(stats_manifest_path) != checkpoint["per_image_stats_manifest_sha256"]:
        raise DfpPhase1aCheckpointError("per-image-stats manifest SHA256 mismatch against the checkpoint's binding")
    validate_checkpoint_against_per_image_stats(checkpoint, stats_manifest)

    print("PASS: DFP Phase 1A checkpoint verified")
    print(f"# run_mode={checkpoint['run_mode']} next_dataset_index={checkpoint['next_dataset_index']} complete={checkpoint['complete']}", file=sys.stderr)
    return 0


def _cmd_verify_resume(args: argparse.Namespace) -> int:
    result = validate_resume_state(checkpoints_dir=args.checkpoints_dir, checkpoint_filename=args.checkpoint_filename, repo_root=args.repo_root)
    print("PASS: DFP Phase 1A resume state verified")
    print(f"# run_mode={result['run_mode']} next_dataset_index={result['next_dataset_index']} resumable={result['resumable']}", file=sys.stderr)
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    result = verify_result_bundle(args.bundle)
    print("PASS: DFP Phase 1A result bundle verified")
    print(f"# run_mode={result['result']['run_mode']} complete={result['result']['complete']} final={result['result']['final']}", file=sys.stderr)
    return 0


_DISPATCH = {
    "verify-schema": _cmd_verify_schema,
    "verify-checkpoint": _cmd_verify_checkpoint,
    "verify-resume": _cmd_verify_resume,
    "verify-result": _cmd_verify_result,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1A ARTIFACTS VERIFY FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
