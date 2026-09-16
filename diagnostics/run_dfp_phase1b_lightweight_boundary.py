#!/usr/bin/env python3
"""CPU-only, non-executing CLI for the DFP Phase 1B lightweight BALP
350-image development runner (E22).

`verify-protocol`          -- load + closed-schema + relational-hash
                               verify the committed E22 identity, the
                               committed E22 artifacts identity, and the
                               frozen 350/150 development/confirmation
                               split; verify the required-ancestor gate
                               against the current repository HEAD.
`preflight`                -- everything `verify-protocol` does, plus:
                               working-branch match, clean-repository
                               check, output-root path-safety policy,
                               and mode-specific checkpoint state
                               (fresh / resume / complete-but-unfinalized
                               / already-finalized) for a given
                               `--output-root`.
`verify-checkpoint`        -- load + validate an installed development
                               checkpoint (metadata JSON + arrays NPZ)
                               under a given `--output-root` against the
                               closed checkpoint schema and every E22
                               relational rule.
`verify-development-result`-- load + validate an installed
                               `development_result/result.json` document
                               under a given `--output-root` against the
                               closed development-result schema.

This CLI implements no usable `execute` command. E22 authorizes no
inference, no GPU execution, no candidate selection, and no access to
the confirmation split or the other 4,500 images; a real 350-image
development run requires a later successor authority that explicitly
permits it. `verify-protocol` and `preflight` never import torch, never
initialize CUDA, never access a dataset image or annotation, and never
load a model checkpoint -- they only validate the committed identity,
the frozen split, and (for `preflight`) the local git/filesystem state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import src.dfp_phase1b_lightweight_boundary_artifact as art  # noqa: E402
import src.dfp_phase1b_lightweight_boundary_identity as e22  # noqa: E402
from src.dfp_phase1a_preflight import (  # noqa: E402
    DfpPhase1aPreflightError,
    validate_output_root_safety,
)


class DfpPhase1bLightweightBoundaryCliError(ValueError):
    """Raised for CLI-level failures (git state, path policy, argument
    errors) that are not already one of E22's own exception types."""


def _banner(command: str) -> None:
    print(f"DFP Phase 1B lightweight BALP development runner [{command}] -- E22 AUTHORITY ONLY.")
    print("  - Freezes the 16-candidate BALP arbitration family over already-frozen DFP")
    print("    Phase 1A canonical horizon score fields, and the 350/150 development/")
    print("    confirmation split of the frozen coco_stuff_500 subset.")
    print("  - This CLI implements no usable `execute` command. Runs no inference.")
    print("    Constructs no model. Initializes no CUDA. Submits no GPU job.")
    print("  - Does NOT select a candidate, evaluate the confirmation split, run an")
    print("    operational benchmark, or access the 5,000-image full population.")


def _run_git(repo_root: Path, *args: str) -> str:
    import subprocess

    result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise DfpPhase1bLightweightBoundaryCliError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _current_head_commit(repo_root: Path) -> str:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise DfpPhase1bLightweightBoundaryCliError(f"git rev-parse HEAD returned a non-canonical commit id: {commit!r}")
    return commit


def _current_branch(repo_root: Path) -> str:
    return _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")


def _repo_is_clean(repo_root: Path) -> bool:
    return _run_git(repo_root, "status", "--porcelain") == ""


def _all_variant_ids(identity: dict) -> list[str]:
    return list(e22.CANONICAL_CONTROLS) + list(e22.reconstruct_candidate_ids())


def _load_verified_protocol(repo_root: Path):
    identity = e22.load_identity(repo_root=repo_root)
    artifacts_identity = e22.load_artifacts_identity(repo_root=repo_root)
    e22.verify_candidate_family(identity)
    e22.verify_full_evaluation_not_authorized(identity)
    split = e22.load_and_verify_split(repo_root=repo_root)
    head = _current_head_commit(repo_root)
    e22.verify_repository_gate(repo_root, head)
    return identity, artifacts_identity, split, head


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    _banner("verify-protocol")
    repo_root = Path(args.repo_root)
    identity, artifacts_identity, split, head = _load_verified_protocol(repo_root)
    print("E22 identity verified (closed schema + exact values + relational dependency hashes).")
    print(f"required_ancestor_commit={identity['identity']['required_ancestor_commit']}  HEAD={head}  is_ancestor=True")
    print(f"working_branch={identity['identity']['working_branch']}")
    print(f"development_count={len(split['development_image_ids'])}  confirmation_count={len(split['confirmation_image_ids'])}")
    print(f"candidate_count={len(e22.reconstruct_candidate_ids())}  control_count={len(e22.CANONICAL_CONTROLS)}")
    ab = identity["authorization_boundary"]
    print(
        f"candidate_selected_by_this_identity={ab['candidate_selected_by_this_identity']}  "
        f"gpu_execution_authorized_by_this_identity={ab['gpu_execution_authorized_by_this_identity']}  "
        f"full_5000_image_evaluation_authorized={ab['full_5000_image_evaluation_authorized']}  "
        f"other_4500_image_access_authorized={ab['other_4500_image_access_authorized']}"
    )
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    _banner("preflight")
    repo_root = Path(args.repo_root)
    identity, artifacts_identity, split, head = _load_verified_protocol(repo_root)
    print("committed E22 protocol + artifacts + frozen split: OK")

    branch = _current_branch(repo_root)
    expected_branch = identity["identity"]["working_branch"]
    if branch != expected_branch:
        raise DfpPhase1bLightweightBoundaryCliError(
            f"current branch {branch!r} does not match the E22-frozen working_branch {expected_branch!r}"
        )
    print(f"branch check: OK ({branch})")

    clean = _repo_is_clean(repo_root)
    if not clean and not args.allow_dirty_repo:
        raise DfpPhase1bLightweightBoundaryCliError(
            "repository is not clean (untracked/modified/staged changes present); "
            "pass --allow-dirty-repo to override for local iteration only"
        )
    print(f"clean-repo check: {'OK' if clean else 'DIRTY (overridden via --allow-dirty-repo)'}")

    forbidden_roots = [
        repo_root / "src", repo_root / "evaluation_identities", repo_root / "evaluation_subsets",
        repo_root / "docs", repo_root / "tests", repo_root / "diagnostics", repo_root / ".git",
    ]
    output_root = validate_output_root_safety(output_root=Path(args.output_root), forbidden_roots=forbidden_roots)
    print(f"output-root path policy: OK ({output_root})")

    variant_ids = _all_variant_ids(identity)
    existing = art.load_full_checkpoint(output_root, variant_ids=variant_ids)
    if existing is None:
        mode_description = "fresh (no checkpoint present)"
    elif existing["final"]:
        mode_description = "already finalized -- a real execute would be REJECTED"
    elif existing["complete"]:
        mode_description = "complete-but-unfinalized -- a real execute would perform finalize-only recovery"
    else:
        n_completed = len(existing["progress"]["completed_image_ids"])
        mode_description = f"resume -- {n_completed}/{e22.DEVELOPMENT_COUNT} development images already completed"
    print(f"checkpoint state at output-root: {mode_description}")

    print(
        "PREFLIGHT PASSED. No image or annotation pixel accessed. No model constructed. "
        "No CUDA initialized. No inference executed."
    )
    return 0


def _cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    _banner("verify-checkpoint")
    repo_root = Path(args.repo_root)
    identity, artifacts_identity, split, head = _load_verified_protocol(repo_root)
    variant_ids = _all_variant_ids(identity)
    output_root = Path(args.output_root)
    doc = art.load_full_checkpoint(output_root, variant_ids=variant_ids)
    if doc is None:
        raise DfpPhase1bLightweightBoundaryCliError(f"no checkpoint found at --output-root {output_root}")
    art.validate_full_checkpoint(doc, development_ids=split["development_image_ids"], confirmation_ids=split["confirmation_image_ids"])
    progress = doc["progress"]
    print(f"checkpoint is schema-valid and relationally consistent with committed E22.")
    print(f"complete={doc['complete']}  final={doc['final']}  generation={progress['generation']}")
    print(f"completed_image_ids={len(progress['completed_image_ids'])}/{e22.DEVELOPMENT_COUNT}")
    return 0


def _cmd_verify_development_result(args: argparse.Namespace) -> int:
    _banner("verify-development-result")
    repo_root = Path(args.repo_root)
    identity, artifacts_identity, split, head = _load_verified_protocol(repo_root)
    variant_ids = _all_variant_ids(identity)
    output_root = Path(args.output_root)
    result_path = output_root / "development_result" / "result.json"
    if not result_path.is_file():
        raise DfpPhase1bLightweightBoundaryCliError(f"no development result found at {result_path}")
    doc = json.loads(result_path.read_bytes())
    art.validate_development_result(doc, variant_ids=variant_ids)
    print("development result is schema-valid and relationally consistent with committed E22.")
    print(f"control_results={len(doc['control_results'])}  candidate_results={len(doc['candidate_results'])}")
    print(f"scientific_output_digest={doc['scientific_output_digest']}")
    print("candidate_selected=false  confirmation_evaluated=false  full_5000_image_evaluation=false")
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    _banner("execute")
    print("EXECUTE IS NOT AUTHORIZED.", file=sys.stderr)
    print(
        "E22 authorizes no inference, no GPU execution, and no candidate selection. "
        "A real `execute` CLI path requires a later successor authority that explicitly "
        "permits the 350-image development run.",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_dfp_phase1b_lightweight_boundary.py")
    parser.add_argument("--repo-root", type=str, default=str(_REPO_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-protocol", help="verify the E22 identity, artifacts identity, and frozen 350/150 split")

    p = sub.add_parser("preflight", help="CPU-only preflight for a prospective development run at --output-root")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--allow-dirty-repo", action="store_true", help="permit a non-clean working tree (local iteration only)")

    p = sub.add_parser("verify-checkpoint", help="validate an installed development checkpoint at --output-root")
    p.add_argument("--output-root", type=str, required=True)

    p = sub.add_parser("verify-development-result", help="validate an installed development_result/result.json at --output-root")
    p.add_argument("--output-root", type=str, required=True)

    p = sub.add_parser("execute", help="NOT AUTHORIZED -- fails closed before importing torch or accessing data")

    args = parser.parse_args(argv)
    handlers = {
        "verify-protocol": _cmd_verify_protocol,
        "preflight": _cmd_preflight,
        "verify-checkpoint": _cmd_verify_checkpoint,
        "verify-development-result": _cmd_verify_development_result,
        "execute": _cmd_execute,
    }
    try:
        return handlers[args.command](args)
    except (
        DfpPhase1bLightweightBoundaryCliError,
        e22.DfpPhase1bLightweightBoundaryIdentityError,
        art.DfpPhase1bLightweightBoundaryArtifactError,
        DfpPhase1aPreflightError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"DFP PHASE 1B LIGHTWEIGHT BOUNDARY CLI FAIL [{args.command}]: {error}", file=sys.stderr)
        _banner(args.command)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
