#!/usr/bin/env python3
"""Diagnostic CLI for the locked 48-image holdout evaluation (Gates B/C/D).

Subcommands (`verify-protocol`, `preflight`, `verify-result` are torch-free;
`execute` lazily imports torch/model code only after preflight passes):

  verify-protocol   Re-verify the committed holdout authority's own
                     prerequisite/dependency/authorization-non-widening
                     state (semantic_evidence_holdout_evaluate.
                     verify_protocol_holdout).
  preflight          Torch-free readiness check. FAILS CLOSED, before any
                     holdout image/GT path is touched, unless a separate,
                     explicit successor authority has set
                     [authorization.frozen_flags].holdout_execution_
                     authorized=true in the committed holdout authority --
                     which it currently does not.
  execute            Run (or resume, or finalize) the real holdout
                     evaluation and install its result. Subject to the
                     SAME fail-closed authorization check as preflight.
                     NEVER submit this directly -- it must be invoked
                     only by the H100 launcher, by an operator, after
                     every authorization-boundary condition in the
                     committed authority is satisfied.
  verify-result      Torch-free reopen + re-verification of an already
                     installed holdout result.

No subcommand accepts a scientific-parameter override, a debug/max-image
option, an alternate image ID, the discovery population, the full
96-image population, the full 5,000-image validation set, or an
overwrite/force flag -- none exist on this CLI by design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import semantic_evidence_holdout_evaluate as runner  # noqa: E402
import semantic_evidence_discovery_execute as disc  # noqa: E402

# This CLI surfaces two sibling "expected domain failure" exception types:
# HoldoutEvaluationError (raised by semantic_evidence_holdout_evaluate's
# own code) and DiscoveryExecutionError (raised by the reused, unmodified
# discovery-module validators/checkpoint helpers it calls internally --
# e.g. validate_gt_free_artifact, validate_checkpoint_metadata). Both are
# caught identically here; neither is a bug, both are a clean, intended
# failure to report to the operator.
_EXPECTED_ERRORS = (runner.HoldoutEvaluationError, disc.DiscoveryExecutionError)


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    try:
        result = runner.verify_protocol_holdout(REPO_ROOT)
    except _EXPECTED_ERRORS as exc:
        print(f"CONTRACT_VERIFICATION_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["verified"] else 1


def _cmd_preflight(args: argparse.Namespace) -> int:
    try:
        result = runner.preflight_holdout(
            REPO_ROOT,
            weight_dir=args.weight_dir,
            image_root=Path(args.image_root),
            gt_root=Path(args.gt_root),
            output_root=Path(args.output_root),
            discovery_result_path=Path(args.discovery_result_path),
            mode=args.mode,
        )
    except _EXPECTED_ERRORS as exc:
        print(f"PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    print(result["status"])
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    try:
        outcome = runner.run_holdout_evaluation(
            REPO_ROOT,
            weight_dir=args.weight_dir,
            image_root=Path(args.image_root),
            gt_root=Path(args.gt_root),
            output_root=Path(args.output_root),
            discovery_result_path=Path(args.discovery_result_path),
            mode=args.mode,
            slurm_job_id=args.slurm_job_id or "",
            node=args.node or "",
        )
    except _EXPECTED_ERRORS as exc:
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
        holdout_ids = runner.load_holdout_population(REPO_ROOT)
        verification = runner.verify_holdout_result(Path(args.result_path), holdout_ids)
    except _EXPECTED_ERRORS as exc:
        print(f"VERIFY_RESULT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(verification, indent=2))
    return 0 if verification["verified"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locked 48-image holdout evaluation CLI (Gates B/C/D)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_vp = sub.add_parser("verify-protocol", help="Torch-free: re-verify the committed holdout authority's own state")
    p_vp.set_defaults(func=_cmd_verify_protocol)

    p_pf = sub.add_parser("preflight", help="Torch-free readiness check (fails closed unless execution is authorized)")
    p_pf.add_argument("--weight-dir", required=True, help="TALK2DINO_WEIGHT_DIR value (DINOv2/CLIP checkpoint directory)")
    p_pf.add_argument("--image-root", required=True, help="COCO-Stuff val2017 images directory")
    p_pf.add_argument("--gt-root", required=True, help="COCO-Stuff semantic annotation directory (path/existence checked only, never decoded)")
    p_pf.add_argument("--output-root", required=True, help="Destination directory for the installed holdout result")
    p_pf.add_argument("--discovery-result-path", required=True, help="Path to the finalized discovery_result.json whose frozen fits this evaluation applies")
    p_pf.add_argument("--mode", choices=("fresh", "resume", "finalize_only"), default="fresh")
    p_pf.set_defaults(func=_cmd_preflight)

    p_ex = sub.add_parser("execute", help="Run/resume/finalize the real holdout evaluation (operator/launcher only; fails closed unless authorized)")
    p_ex.add_argument("--weight-dir", required=True)
    p_ex.add_argument("--image-root", required=True)
    p_ex.add_argument("--gt-root", required=True, help="COCO-Stuff semantic annotation directory")
    p_ex.add_argument("--output-root", required=True)
    p_ex.add_argument("--discovery-result-path", required=True)
    p_ex.add_argument("--mode", choices=("fresh", "resume", "finalize_only"), default="fresh")
    p_ex.add_argument("--slurm-job-id", default="")
    p_ex.add_argument("--node", default="")
    p_ex.set_defaults(func=_cmd_execute)

    p_vr = sub.add_parser("verify-result", help="Torch-free: reopen and re-verify an installed holdout result")
    p_vr.add_argument("--result-path", required=True)
    p_vr.set_defaults(func=_cmd_verify_result)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
