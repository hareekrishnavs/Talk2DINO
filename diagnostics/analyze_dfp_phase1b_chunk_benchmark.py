#!/usr/bin/env python3
"""DFP Phase 1B Task 4: CLI for the E18-authorized deterministic
operational chunk-size selection (additive successor to E17).

**This is a CPU-only operational analysis/selection step.** It selects
an operational ``component_chunk_size`` value by applying a
deterministic, frozen-before-being-applied rule to the completed and
independently-verified E17 benchmark output. It defines no numeric
conservation tolerance and authorizes no full Phase 1B run or the held
5,000-image evaluation -- see ``docs/dfp_phase1b_chunk_selection.md``
for the complete account.

Subcommands
-----------
verify-protocol    Verify the E18 identity: closed schema + relational
                    SHA-256 hashes against every bound file, and
                    cross-checked against the live E17. No torch.
analyze            Independently re-verify a real completed Task 3
                    benchmark output directory (six raw trials +
                    merged result) and recompute the deterministic
                    selection over it. Reads the benchmark; runs no
                    inference. No torch.
verify-selection   Full boundary: verify-protocol + analyze, then
                    require the recomputed winner to exactly equal
                    E18's recorded selected_chunk_size. No torch.

No subcommand imports torch, mmcv, or any model/dataset module. No GPU
flag exists anywhere in this CLI. No SLURM submission. No 5,000-image
option.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.dfp_phase1b_chunk_selection import (
    DfpPhase1bChunkSelectionError,
    build_selection_artifact,
    install_selection_bundle,
    load_e18_identity,
    render_markdown_report,
    sha256_file,
    verify_e18_against_e17,
    verify_e18_dependency_bindings,
    verify_selection as _verify_selection_impl,
    write_markdown_report,
    write_selection_artifact_atomically,
)
from src.dfp_phase1b_chunk_benchmark import DfpPhase1bChunkBenchmarkError
from src.dfp_phase1b_chunk_benchmark_identity import DfpPhase1bChunkBenchmarkIdentityError

_KNOWN_ERRORS = (DfpPhase1bChunkSelectionError, DfpPhase1bChunkBenchmarkError, DfpPhase1bChunkBenchmarkIdentityError, OSError, ValueError)


def _print_banner(command: str) -> None:
    print(f"DFP Phase 1B chunk selection [{command}] -- OPERATIONAL ENGINEERING SELECTION ONLY.")
    print("  - Post-hoc selection over an already-completed, independently-verified benchmark.")
    print("  - Defines no numeric conservation tolerance.")
    print("  - Does NOT authorize the full Phase 1B run.")
    print("  - Does not touch the held 5,000-image evaluation.")


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    _print_banner("verify-protocol")
    repo_root = Path(args.repo_root)
    e18 = load_e18_identity(repo_root, path=Path(args.identity) if args.identity else None)
    if args.merged_result is not None:
        verify_e18_dependency_bindings(repo_root, e18, merged_result_path=Path(args.merged_result))
        print(f"merged_result binding verified against {args.merged_result}")
    else:
        print("merged_result binding NOT independently re-verified (no --merged-result supplied) -- repository-relative bindings only")
        for name in ("e17_identity", "benchmark_manifest", "benchmark_runner_module", "benchmark_cli_module",
                     "selector_module", "analysis_cli_module"):
            from src.dfp_phase1b_chunk_benchmark_identity import resolve_repository_relative_path
            binding = e18["dependency_bindings"][name]
            target = resolve_repository_relative_path(repo_root, binding["relative_path"], binding_name=name)
            actual = sha256_file(target)
            if actual != binding["sha256"]:
                raise DfpPhase1bChunkSelectionError(f"E18 dependency_bindings.{name} SHA-256 mismatch for {target}")
    verify_e18_against_e17(repo_root, e18)
    print("E18 identity verified (closed schema + relational hashes + cross-checked against live E17).")
    print(f"successor_of={e18['identity']['successor_of']}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    _print_banner("analyze")
    from src.dfp_phase1b_chunk_selection import analyze_benchmark
    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir)
    merged_result_path = Path(args.merged_result) if args.merged_result else None
    analysis = analyze_benchmark(repo_root=repo_root, output_dir=output_dir, merged_result_path=merged_result_path)

    for cs in sorted(analysis["measurements"]):
        m = analysis["measurements"][cs]
        s = analysis["safety"][cs]
        print(f"chunk_size={cs} elapsed_ns_by_repeat={m['elapsed_ns_by_repeat']} mean_elapsed_ns={m['mean_elapsed_ns']} safe={s['safe']} reasons={s['reasons']}")
    print(f"common_scientific_output_digest={analysis['common_digest']}")
    print(f"eligible={analysis['eligible']}")
    print(f"selected_chunk_size={analysis['winner']}")
    print(f"tie_break_path={analysis['tie_break_path']}")

    if args.output is not None or args.markdown_output is not None:
        e17_binding_path = _REPO_ROOT / "evaluation_identities/e17_dfp_phase1b_chunk_benchmark.toml"
        manifest_path = _REPO_ROOT / "evaluation_subsets/dfp_phase1b_chunk_benchmark_24.json"
        artifact = build_selection_artifact(
            analysis, e18_sha256="", e17_sha256=sha256_file(e17_binding_path), manifest_sha256=sha256_file(manifest_path),
            repository_commit=analysis["merged"]["code_provenance"]["repository_commit"],
            command="analyze_dfp_phase1b_chunk_benchmark.py analyze",
        )
        # --output / --markdown-output are SAFE BASENAMES ONLY, resolved
        # under --output-dir (the same directory the real benchmark
        # trials/merged result were read from) -- never an arbitrary
        # caller-supplied path. When both are requested, install them
        # as one atomic pair (Findings-repair Section 2/7).
        if args.output is not None and args.markdown_output is not None:
            json_dest, md_dest = install_selection_bundle(
                artifact, render_markdown_report(artifact), output_dir=output_dir,
                json_basename=args.output, markdown_basename=args.markdown_output, allow_overwrite=args.force,
            )
            print(f"Selection artifact written to {json_dest}")
            print(f"Markdown report written to {md_dest}")
        elif args.output is not None:
            json_dest = write_selection_artifact_atomically(artifact, output_dir=output_dir, basename=args.output, allow_overwrite=args.force)
            print(f"Selection artifact written to {json_dest}")
        else:
            md_dest = write_markdown_report(render_markdown_report(artifact), output_dir=output_dir, basename=args.markdown_output, allow_overwrite=args.force)
            print(f"Markdown report written to {md_dest}")
    return 0


def _cmd_verify_selection(args: argparse.Namespace) -> int:
    _print_banner("verify-selection")
    repo_root = Path(args.repo_root)
    analysis = _verify_selection_impl(
        repo_root=repo_root, e18_path=Path(args.identity) if args.identity else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        merged_result_path=Path(args.merged_result) if args.merged_result else None,
    )
    print(f"Recomputed selected_chunk_size={analysis['winner']} PASSES independent verification against E18's recorded value.")
    print(f"common_scientific_output_digest={analysis['common_digest']}")
    print(f"eligible={analysis['eligible']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DFP Phase 1B chunk-size selection CLI (operational engineering selection only, CPU-only)",
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-protocol", help="verify the E18 identity: closed schema + relational hashes, cross-checked against live E17")
    p.add_argument("--identity", type=str, default=None)
    p.add_argument("--merged-result", type=str, default=None, help="optional -- also relationally re-verify the merged_result binding against this real path")

    p = sub.add_parser("analyze", help="independently re-verify a real Task 3 benchmark output directory and recompute the selection")
    p.add_argument("--output-dir", type=str, required=True, help="directory containing the six trial_*.json files and merged_result.json")
    p.add_argument("--merged-result", type=str, default=None, help="override the merged result path (default: <output-dir>/merged_result.json)")
    p.add_argument("--output", type=str, default=None, metavar="BASENAME",
                    help="optional -- safe basename (no path) for the deterministic selection artifact JSON, "
                         "installed directly under --output-dir; must end with .json")
    p.add_argument("--markdown-output", type=str, default=None, metavar="BASENAME",
                    help="optional -- safe basename (no path) for the human-readable Markdown report, "
                         "installed directly under --output-dir; must end with .md")
    p.add_argument("--force", action="store_true", help="allow overwriting an existing --output/--markdown-output destination")

    p = sub.add_parser("verify-selection", help="full boundary: verify E18 + analyze + require the recomputed winner to equal E18's recorded winner")
    p.add_argument("--identity", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None, help="directory containing the six trial_*.json files and merged_result.json")
    p.add_argument("--merged-result", type=str, default=None, help="override the merged result path")

    return parser


_DISPATCH = {
    "verify-protocol": _cmd_verify_protocol,
    "analyze": _cmd_analyze,
    "verify-selection": _cmd_verify_selection,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1B CHUNK SELECTION FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
