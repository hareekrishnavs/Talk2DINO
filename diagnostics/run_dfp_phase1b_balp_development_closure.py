#!/usr/bin/env python3
"""E24-authorized CPU-only closure-analysis CLI for the completed E23
real 350-image BALP development execution (SLURM job 22071646).

`verify-authority` -- load + closed-schema/relationally verify E22,
                       E23, and E24. No source-artifact access.
`verify-source`    -- publicly reverify the source checkpoint/result at
                       `--source-output-root` against E24's pinned
                       hashes. Read-only.
`generate`         -- verify authority + source, independently
                       recompute every metric/eligibility flag from the
                       persisted confusion-matrix arrays, build the
                       closure JSON + Markdown, and atomically install
                       both into a FRESH `--destination-root` (no
                       overwrite). Never writes into the source output
                       root.
`verify-closure`   -- reopen + publicly reverify an already-installed
                       closure at `--destination-root`.

Runs no inference, constructs no model, initializes no CUDA, and never
writes into `--source-output-root`. `generate` performs exactly one
atomic install into an absent `--destination-root`; a non-fresh
destination is always rejected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.dfp_phase1b_balp_development_closure as closure  # noqa: E402
import src.dfp_phase1b_balp_development_closure_identity as e24  # noqa: E402
import src.dfp_phase1b_balp_development_execution_identity as e23  # noqa: E402
import src.dfp_phase1b_lightweight_boundary_identity as e22  # noqa: E402

_KNOWN_ERRORS = (
    e24.DfpPhase1bBalpDevelopmentClosureIdentityError,
    e23.DfpPhase1bBalpDevelopmentExecutionIdentityError,
    e22.DfpPhase1bLightweightBoundaryIdentityError,
    closure.DfpPhase1bBalpDevelopmentClosureError,
    OSError,
    ValueError,
)


def _banner(command: str) -> None:
    print(f"DFP Phase 1B BALP development closure [{command}] -- E24 AUTHORITY.")
    print("  - Read-only interpretation of the completed SLURM job 22071646. Runs no")
    print("    inference, constructs no model, initializes no CUDA, submits no job.")
    print("  - Selects no candidate, authorizes no confirmation, authorizes no")
    print("    5,000-image evaluation, fabricates no profiling value.")


def _cmd_verify_authority(args: argparse.Namespace) -> int:
    _banner("verify-authority")
    repo_root = Path(args.repo_root)
    e22.load_identity(repo_root=repo_root)
    e22.load_artifacts_identity(repo_root=repo_root)
    e23.load_identity(repo_root=repo_root)
    e24_doc = e24.load_identity(repo_root=repo_root)
    e24.verify_repository_gate(repo_root, e24.REQUIRED_ANCESTOR_COMMIT)
    print("E22 + E23 + E24 identities verified (closed schema + exact values + relational hashes).")
    print(f"frozen_decision.eligible_candidate_count={e24_doc['frozen_decision']['eligible_candidate_count']}  development_outcome={e24_doc['frozen_decision']['development_outcome']}")
    print(f"frozen_decision.profiling_status={e24_doc['frozen_decision']['profiling_status']}")
    return 0


def _cmd_verify_source(args: argparse.Namespace) -> int:
    _banner("verify-source")
    repo_root = Path(args.repo_root)
    checkpoint = closure.verify_source_result(Path(args.source_output_root), repo_root=repo_root)
    print(f"PASS: source checkpoint/result publicly reverified against E24's pinned hashes at {args.source_output_root}")
    print(f"checkpoint complete={checkpoint['complete']} final={checkpoint['final']} generation={checkpoint['progress']['generation']}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    _banner("generate")
    repo_root = Path(args.repo_root)

    # Mandatory clean-HEAD generation gate -- runs FIRST, before any
    # source access or output path is touched. Fails closed (raises,
    # creates/modifies nothing) unless the repository is clean, nothing
    # is staged, the closure implementation/CLI on-disk bytes match
    # their committed HEAD bytes exactly, E24 and its dependency hashes
    # verify, and HEAD descends from the required source-run ancestor.
    closure_generation = closure.verify_clean_head_generation_gate(repo_root)
    print(f"clean-HEAD generation gate: PASS (closure_generation.repository_commit={closure_generation['repository_commit']})")

    checkpoint = closure.verify_source_result(Path(args.source_output_root), repo_root=repo_root)
    print("source checkpoint/result: PASS (publicly reverified against E24's pinned hashes)")

    doc = closure.build_closure_document(
        checkpoint, repo_root=repo_root, source_output_root=Path(args.source_output_root), closure_generation=closure_generation,
    )
    print(f"independent reconstruction: eligible_candidate_count={doc['decision']['eligible_candidate_count']}  outcome={doc['decision']['outcome']}")
    markdown_text = closure.build_closure_markdown(doc)

    destination = Path(args.destination_root)
    if not destination.is_absolute():
        raise closure.DfpPhase1bBalpDevelopmentClosureError("--destination-root must be an absolute path")
    installed = closure.install_closure(destination, json_doc=doc, markdown_text=markdown_text)

    reloaded = closure.load_closure(installed)
    print(f"CLOSURE INSTALLED AND VERIFIED at {installed}")
    print(f"scientific_output_digest={reloaded['provenance']['source_run']['scientific_output_digest']}")
    print(f"source_run.repository_commit={reloaded['provenance']['source_run']['repository_commit']}  closure_generation.repository_commit={reloaded['provenance']['closure_generation']['repository_commit']}")
    print(f"eligible_candidate_count={reloaded['decision']['eligible_candidate_count']}/16  outcome={reloaded['decision']['outcome']}")
    return 0


def _cmd_verify_closure(args: argparse.Namespace) -> int:
    _banner("verify-closure")
    doc = closure.load_closure(Path(args.destination_root))
    if doc is None:
        raise closure.DfpPhase1bBalpDevelopmentClosureError(f"no closure.json found at --destination-root {args.destination_root}")
    print("closure document is schema-valid and internally consistent.")
    print(f"eligible_candidate_count={doc['decision']['eligible_candidate_count']}/16  outcome={doc['decision']['outcome']}")
    print(f"scientific_output_digest={doc['provenance']['source_run']['scientific_output_digest']}")
    print(f"profiling_status={doc['profiling_disclosure']['profiling_status']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_dfp_phase1b_balp_development_closure.py")
    parser.add_argument("--repo-root", type=str, default=str(_REPO_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-authority", help="verify E22/E23/E24 identities")

    p = sub.add_parser("verify-source", help="publicly reverify the source checkpoint/result against E24's pinned hashes")
    p.add_argument("--source-output-root", type=str, required=True)

    p = sub.add_parser("generate", help="build and atomically install the closure JSON+Markdown into a fresh --destination-root")
    p.add_argument("--source-output-root", type=str, required=True)
    p.add_argument("--destination-root", type=str, required=True)

    p = sub.add_parser("verify-closure", help="reopen and publicly reverify an installed closure")
    p.add_argument("--destination-root", type=str, required=True)

    args = parser.parse_args(argv)
    handlers = {
        "verify-authority": _cmd_verify_authority,
        "verify-source": _cmd_verify_source,
        "generate": _cmd_generate,
        "verify-closure": _cmd_verify_closure,
    }
    try:
        return handlers[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1B BALP CLOSURE FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
