#!/usr/bin/env python3
"""Diagnostic CLI for the 48-image semantic-evidence discovery execution.

Subcommands (`verify-protocol`, `preflight`, `verify-result` are torch-free;
`execute` lazily imports torch/model code only after preflight passes):

  verify-protocol   Re-run the committed discovery authority's own official
                     verifier (semantic_evidence_discovery_execution_identity).
  preflight          Torch-free readiness check (see
                     semantic_evidence_discovery_execute.preflight).
  execute            Run (or resume, or finalize) the real discovery
                     execution and install its result. NEVER submit this
                     directly -- it must be invoked only by the H100
                     launcher, by an operator, after every
                     authorization-boundary condition in the committed
                     authority is satisfied.
  verify-result      Torch-free reopen + re-verification of an already
                     installed discovery result.

No subcommand accepts a scientific-parameter override, a debug/max-image
option, an alternate image ID, the locked holdout, the full 96-image
population, or overwrite/force flags -- none exist on this CLI by design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import semantic_evidence_discovery_execution_identity as discovery_id  # noqa: E402
import semantic_evidence_discovery_execute as runner  # noqa: E402


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    try:
        result = runner.verify_protocol(REPO_ROOT)
    except discovery_id.DiscoveryExecutionContractError as exc:
        print(f"CONTRACT_VERIFICATION_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["verified"] else 1


def _cmd_preflight(args: argparse.Namespace) -> int:
    try:
        result = runner.preflight(
            REPO_ROOT,
            weight_dir=args.weight_dir,
            image_root=Path(args.image_root),
            gt_root=Path(args.gt_root),
            output_root=Path(args.output_root),
            mode=args.mode,
        )
    except runner.DiscoveryExecutionError as exc:
        print(f"PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    print(result["status"])
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    try:
        outcome = runner.run_discovery(
            REPO_ROOT,
            weight_dir=args.weight_dir,
            image_root=Path(args.image_root),
            gt_root=Path(args.gt_root),
            output_root=Path(args.output_root),
            mode=args.mode,
            slurm_job_id=args.slurm_job_id or "",
            node=args.node or "",
        )
    except runner.DiscoveryExecutionError as exc:
        print(f"EXECUTE_FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"installed: {outcome['path']}")
    print(f"outcome: {outcome['outcome']}")
    if not outcome["verified"]:
        return 1
    print("SUCCESS (post-install public verification passed)")
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    try:
        discovery_ids = runner.load_discovery_population(REPO_ROOT)
        verification = runner.verify_result(Path(args.result_path), discovery_ids)
    except runner.DiscoveryExecutionError as exc:
        print(f"VERIFY_RESULT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(verification, indent=2))
    return 0 if verification["verified"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="48-image semantic-evidence discovery execution CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_vp = sub.add_parser("verify-protocol", help="Torch-free: re-run the committed authority's official verifier")
    p_vp.set_defaults(func=_cmd_verify_protocol)

    p_pf = sub.add_parser("preflight", help="Torch-free readiness check")
    p_pf.add_argument("--weight-dir", required=True, help="TALK2DINO_WEIGHT_DIR value (DINOv2/CLIP checkpoint directory)")
    p_pf.add_argument("--image-root", required=True, help="COCO-Stuff val2017 images directory")
    p_pf.add_argument("--gt-root", required=True, help="COCO-Stuff semantic annotation directory (path/existence checked only, never decoded)")
    p_pf.add_argument("--output-root", required=True, help="Destination directory for the installed discovery result")
    p_pf.add_argument("--mode", choices=("fresh", "resume", "finalize_only"), default="fresh")
    p_pf.set_defaults(func=_cmd_preflight)

    p_ex = sub.add_parser("execute", help="Run/resume/finalize the real discovery execution (operator/launcher only)")
    p_ex.add_argument("--weight-dir", required=True)
    p_ex.add_argument("--image-root", required=True)
    p_ex.add_argument("--gt-root", required=True, help="COCO-Stuff semantic annotation directory")
    p_ex.add_argument("--output-root", required=True)
    p_ex.add_argument("--mode", choices=("fresh", "resume", "finalize_only"), default="fresh")
    p_ex.add_argument("--slurm-job-id", default="")
    p_ex.add_argument("--node", default="")
    p_ex.set_defaults(func=_cmd_execute)

    p_vr = sub.add_parser("verify-result", help="Torch-free: reopen and re-verify an installed discovery result")
    p_vr.add_argument("--result-path", required=True)
    p_vr.set_defaults(func=_cmd_verify_result)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
