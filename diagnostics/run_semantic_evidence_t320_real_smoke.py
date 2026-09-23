#!/usr/bin/env python3
"""Diagnostic CLI for the two-image E3/T320 real-model integration smoke.

Subcommands (`verify-protocol`, `preflight`, `verify-result` are torch-free;
`execute` lazily imports torch/model code only after preflight passes):

  verify-protocol   Re-run the committed smoke authority's own official
                     verifier (semantic_evidence_t320_real_smoke_identity).
  preflight          Torch-free readiness check (see
                     semantic_evidence_t320_real_smoke.preflight).
  execute            Rerun preflight, then run the real smoke and install
                     its result. NEVER submit this directly -- it must be
                     invoked only by the H100 launcher, by an operator, after
                     every authorization-boundary condition in the committed
                     authority is satisfied.
  verify-result      Torch-free reopen + re-verification of an already
                     installed smoke_result.json.

No subcommand accepts a scientific-parameter override, a debug/max-image
option, an alternate image ID, a GT/annotation path, or overwrite/force
flags -- none exist on this CLI by design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import semantic_evidence_t320_real_smoke_identity as smoke_id  # noqa: E402
import semantic_evidence_t320_real_smoke as runner  # noqa: E402


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    try:
        result = smoke_id.verify_all(REPO_ROOT)
    except smoke_id.T320RealSmokeContractError as exc:
        print(f"CONTRACT_VERIFICATION_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"verified": result["verified"], "accessed_top_level_tables": result["accessed_top_level_tables"]}, indent=2))
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    try:
        result = runner.preflight(
            REPO_ROOT,
            weight_dir=args.weight_dir,
            image_root=Path(args.image_root),
            output_root=Path(args.output_root),
        )
    except (smoke_id.T320RealSmokeContractError, runner.SmokeRunnerError) as exc:
        print(f"PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    print(result["status"])
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    try:
        outcome = runner.execute(
            REPO_ROOT,
            weight_dir=args.weight_dir,
            image_root=Path(args.image_root),
            output_root=Path(args.output_root),
            slurm_job_id=args.slurm_job_id or "",
            node=args.node or "",
        )
    except (smoke_id.T320RealSmokeContractError, runner.SmokeRunnerError) as exc:
        print(f"EXECUTE_FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"installed: {outcome['installed_path']}")
    print(f"outcome: {outcome['result']['final_smoke_outcome']}")
    if outcome["result"]["final_smoke_outcome"] != "PASS_REAL_E3_T320_INTEGRATION_SMOKE":
        return 1
    print("SUCCESS (post-install public verification passed)")
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    # Always supplies repo_root so the public verifier performs its full
    # git-based provenance reverification (Part 4), never just the
    # structural-only schema check.
    try:
        verification = runner.verify_installed_result(Path(args.result_path), repo_root=REPO_ROOT)
    except runner.SmokeRunnerError as exc:
        print(f"VERIFY_RESULT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(verification, indent=2))
    return 0 if verification["verified"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-image E3/T320 real-model integration smoke CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_vp = sub.add_parser("verify-protocol", help="Torch-free: re-run the committed authority's official verifier")
    p_vp.set_defaults(func=_cmd_verify_protocol)

    p_pf = sub.add_parser("preflight", help="Torch-free readiness check")
    p_pf.add_argument("--weight-dir", required=True, help="TALK2DINO_WEIGHT_DIR value (DINOv2/CLIP checkpoint directory)")
    p_pf.add_argument("--image-root", required=True, help="COCO-Stuff val2017 images directory")
    p_pf.add_argument("--output-root", required=True, help="Fresh, non-existent destination for smoke_result.json")
    p_pf.set_defaults(func=_cmd_preflight)

    p_ex = sub.add_parser("execute", help="Rerun preflight, then run and install the real smoke (operator/launcher only)")
    p_ex.add_argument("--weight-dir", required=True)
    p_ex.add_argument("--image-root", required=True)
    p_ex.add_argument("--output-root", required=True)
    p_ex.add_argument("--slurm-job-id", default="")
    p_ex.add_argument("--node", default="")
    p_ex.set_defaults(func=_cmd_execute)

    p_vr = sub.add_parser("verify-result", help="Torch-free: reopen and re-verify an installed smoke_result.json")
    p_vr.add_argument("--result-path", required=True)
    p_vr.set_defaults(func=_cmd_verify_result)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
