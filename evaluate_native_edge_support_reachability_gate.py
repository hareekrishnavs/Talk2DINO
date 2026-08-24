#!/usr/bin/env python3
"""Offline, CPU-only 20-image structural reachability gate.

Consumes one explicitly-supplied, already-verified mechanics20
native-edge-support-audit result and formally decides whether the roadmap
may proceed to structural pruning. Never initializes CUDA, never loads the
model or dataset, never reruns mechanics20, never implements pruning, and
never modifies the parent artifact. The gate reproduces the parent's own
decision via the parent's own classifier -- it never introduces a new
empirical threshold after observing the result.

This is deliberately a single-shot, non-checkpointed CLI (unlike the
per-image evaluators upstream): the entire computation is over one small,
already-materialized JSON file, so there is no long-running loop to
resume.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

from src.native_edge_support_checkpoint import parse_strict_json_document
from src.native_edge_support_reachability_gate import (
    NativeEdgeSupportReachabilityGateError,
    build_gate_report,
    select_and_validate_parent_artifact,
    sha256_file,
)
from src.native_edge_support_reachability_gate_identity import (
    NativeEdgeSupportReachabilityGateIdentityError,
    load_identity,
    validate_static_configuration,
)
from src.native_edge_support_reachability_gate_report import verify_record
from src.k11_k12_stability_report import write_checkpoint_atomically
from src.native_edge_support_identity import NativeEdgeSupportAuditIdentityError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline 20-image structural reachability gate: formalizes the mechanics20 "
        "native-edge-support audit decision into a roadmap authorization record."
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--audit-result", type=Path, default=None, help="explicit path to a verified mechanics20 native-edge-support-audit result JSON; never auto-selected. Required unless --list-candidates is used.")
    parser.add_argument("--result", type=Path, default=None, help="required unless --list-candidates is used")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--list-candidates", type=Path, default=None,
        help="informational only: list *.json files under this directory whose name contains "
        "'mechanics20' and 'result', without selecting or validating any of them. Never used to "
        "auto-select --audit-result.",
    )
    return parser


def _list_candidates(directory: Path) -> int:
    if not directory.is_dir():
        print(f"NATIVE EDGE SUPPORT REACHABILITY GATE FAIL: --list-candidates path is not a directory: {directory}", file=sys.stderr)
        return 2
    candidates = sorted(
        p for p in directory.glob("*.json")
        if "mechanics20" in p.name and "result" in p.name
    )
    if not candidates:
        print(f"no candidate mechanics20 result files found under {directory}")
        return 0
    print(f"candidate mechanics20 result files under {directory} (informational only -- supply one explicitly via --audit-result):")
    for candidate in candidates:
        print(f"  {candidate.name}")
    return 0


def _run(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)

    gate_identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_native_edge_support_reachability_gate.toml"
    gate_identity_sha256 = sha256_file(identity_path)

    if not args.overwrite and args.result.exists():
        raise NativeEdgeSupportReachabilityGateError(f"--result {args.result} already exists; pass --overwrite for explicit resume/overwrite behavior")
    args.result.parent.mkdir(parents=True, exist_ok=True)

    parent = select_and_validate_parent_artifact(args.audit_result, repo_root=root, gate_identity=gate_identity)

    created_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = build_gate_report(
        gate_identity=gate_identity, gate_identity_sha256=gate_identity_sha256, parent=parent,
        audit_result_path_reference=str(args.audit_result), created_at_utc=created_at_utc,
    )

    temp_result_path = args.result.with_name(args.result.name + f".selfcheck-{os.getpid()}.tmp")
    try:
        write_checkpoint_atomically(temp_result_path, report)
        reloaded = parse_strict_json_document(temp_result_path, label="self-check reachability-gate result")
        verify_record(reloaded, gate_identity, gate_identity_sha256=gate_identity_sha256)
    except Exception:
        temp_result_path.unlink(missing_ok=True)
        raise
    os.replace(temp_result_path, args.result)

    print(
        f"NATIVE EDGE SUPPORT REACHABILITY GATE PASS "
        f"parent_decision={report['parent_decision_reproduction']['parent_reported_decision']} "
        f"roadmap_action={report['stop_proceed_decision']} "
        f"next_stage={report['roadmap_authorization']['next_authorized_stage']!r} -> {args.result}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_candidates is not None:
        return _list_candidates(args.list_candidates)
    if args.audit_result is None or args.result is None:
        parser.error("--audit-result and --result are required unless --list-candidates is used")
    try:
        return _run(args)
    except (
        NativeEdgeSupportReachabilityGateError,
        NativeEdgeSupportReachabilityGateIdentityError,
        NativeEdgeSupportAuditIdentityError,
        OSError,
        ValueError,
    ) as error:
        print(f"NATIVE EDGE SUPPORT REACHABILITY GATE FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
