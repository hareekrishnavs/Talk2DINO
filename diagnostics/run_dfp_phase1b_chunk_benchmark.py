#!/usr/bin/env python3
"""DFP Phase 1B Task 3: CLI for the E17-authorized fixed component-
chunk-size (16/32/64) performance benchmark.

**This is an operational performance benchmark only.** It selects no
chunk size, freezes no numeric conservation tolerance, and a
successful merged result does not authorize the full Phase 1B run or
the held 5,000-image evaluation -- see
``docs/dfp_phase1b_chunk_benchmark.md`` for the complete account.

Subcommands
-----------
verify-protocol   Verify the E17 identity: closed schema and relational
                   SHA-256 hashes against every bound file. No torch.
build-manifest     Build and atomically install the 24-window benchmark
                   manifest from a verified E16 calibration artifact.
                   No torch.
verify-manifest    Independently re-verify an installed benchmark
                   manifest against its source artifact. No torch.
preflight          Verify protocol + manifest + source artifact
                   together -- no torch, no CUDA, no dataset access.
run-trial          Run exactly one fresh-process trial. Requires
                   ``--device cuda``. The only subcommand that imports
                   torch/mmcv/model/dataset code, and only after
                   preflight succeeds.
verify-trial       Verify one trial artifact's closed schema and
                   authority/manifest bindings. No torch.
merge              Merge exactly six trial artifacts (the frozen
                   schedule) into one closed-schema merged result. No
                   torch -- pure CPU aggregation/verification.
verify-result      Independently re-verify an installed merged result.
                   No torch.

Every subcommand except ``run-trial`` performs no torch import, no CUDA
initialization, and no real dataset/model access. Every torch/model/
dataset import in this file and in ``src.dfp_phase1b_chunk_benchmark``
is deferred into a function body, never at module scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.dfp_phase1b_chunk_benchmark import (
    DfpPhase1bChunkBenchmarkError,
    build_benchmark_manifest,
    build_code_provenance,
    merge_trials,
    strict_structural_equal,
    verify_benchmark_manifest,
    verify_trial_document,
    write_merged_artifact_atomically,
    write_trial_document_atomically,
)
from src.dfp_phase1b_chunk_benchmark_identity import (
    CANONICAL_DEPENDENCY_RELATIVE_PATHS,
    DfpPhase1bChunkBenchmarkIdentityError,
    IDENTITY_RELATIVE_PATH,
    TRIAL_SCHEDULE,
    code_binding_hashes,
    load_identity,
    memory_guard_contract_from_identity,
    require_clean_worktree,
    verify_ancestor_relation,
)

_KNOWN_ERRORS = (DfpPhase1bChunkBenchmarkError, DfpPhase1bChunkBenchmarkIdentityError, OSError, ValueError)


def _print_banner(command: str) -> None:
    print(f"DFP Phase 1B chunk benchmark [{command}] -- OPERATIONAL PERFORMANCE MEASUREMENT ONLY.")
    print("  - Selects no chunk size, freezes no conservation tolerance.")
    print("  - A successful result does NOT authorize the full Phase 1B run.")
    print("  - Does not touch the held 5,000-image evaluation.")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise DfpPhase1bChunkBenchmarkError(f"cannot read {path}: {exc}") from exc


def _reject_lexical_path_traversal(path: Path) -> None:
    for part in Path(path).parts:
        if part == "..":
            raise DfpPhase1bChunkBenchmarkError(f"path {path} contains a lexical '..' component -- refused")


def _load_calibration_document(artifact_path: Path) -> dict:
    _reject_lexical_path_traversal(artifact_path)
    with open(artifact_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class ReadinessResult:
    """The result of one full repository-readiness check (Finding:
    ``preflight``/``run-trial`` shared-implementation repair) --
    everything a caller needs to proceed straight to (or report
    readiness for) real trial execution, with no further identity/
    manifest/git re-derivation."""

    identity_document: dict
    manifest: dict
    repository_commit: str
    code_provenance: dict


def perform_repository_readiness_check(
    *, repo_root: Path, artifact_path: Path, manifest_path: Path, workload_bundle_dir: Path,
    output_path: Path | None, allow_overwrite: bool,
) -> ReadinessResult:
    """The SOLE repository-readiness check, used by both ``preflight``
    and ``run-trial`` -- one implementation, never duplicated (Finding
    repair: the standalone ``preflight`` command previously performed
    only identity/manifest validation and silently omitted the git
    ancestry/clean-worktree check that ``run-trial`` itself DID
    perform, so a dirty checkout could receive a false ``PREFLIGHT
    PASS``). Performs, in this order, entirely before any torch/CUDA
    import, model/dataset construction, checkpoint loading, or output
    creation:

      1. repository root exists and resolves to a real directory;
      2. the E17 identity loads and every one of its hashes passes
         (including canonical dependency-path pinning and repository
         containment -- see ``resolve_repository_relative_path``);
      3. the benchmark manifest independently re-verifies against the
         source calibration artifact;
      4. the current HEAD is a valid, existing commit;
      5. E17's ``required_ancestor_commit`` exists as a real commit;
      6. that required ancestor is an ancestor of (or equal to) the
         current HEAD (never requiring exact equality);
      7. the working tree is completely clean -- unstaged tracked
         changes, staged changes, AND untracked files all reject;
      8. the requested output path (if any) is lexically safe and
         either does not yet exist or ``allow_overwrite`` is set.

    Because the real repository is still uncommitted (eight untracked
    implementation files), a real readiness check run against the
    current working tree is EXPECTED to fail at step 7 -- this is
    correct, intentional behavior and must never be weakened."""
    try:
        resolved_repo_root = Path(repo_root).resolve(strict=True)
    except OSError as exc:
        raise DfpPhase1bChunkBenchmarkError(f"repository root {repo_root} does not exist or cannot be resolved: {exc}") from exc
    if not resolved_repo_root.is_dir():
        raise DfpPhase1bChunkBenchmarkError(f"repository root {resolved_repo_root} is not a directory")

    identity_document = load_identity(
        repo_root=resolved_repo_root, calibration_artifact_path=artifact_path, workload_bundle_dir=workload_bundle_dir,
    )

    calibration_document = _load_calibration_document(artifact_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_benchmark_manifest(
        manifest, calibration_document=calibration_document,
        calibration_artifact_sha256=_sha256(artifact_path),
        e16_identity_sha256=_sha256(resolved_repo_root / CANONICAL_DEPENDENCY_RELATIVE_PATHS["e16_identity"]),
    )

    # Git ancestry / worktree-cleanliness -- required BEFORE any torch/
    # CUDA/model/dataset import or output creation. Covers current-HEAD
    # validity, required_ancestor_commit existence, the ancestor
    # relation itself, and full worktree cleanliness (steps 4-7 above).
    try:
        repository_commit = require_clean_worktree(resolved_repo_root)
        verify_ancestor_relation(
            resolved_repo_root, required_ancestor_commit=identity_document["identity"]["required_ancestor_commit"],
            repository_commit=repository_commit,
        )
    except DfpPhase1bChunkBenchmarkIdentityError as exc:
        raise DfpPhase1bChunkBenchmarkError(str(exc)) from exc

    if output_path is not None:
        _reject_lexical_path_traversal(output_path)
        if output_path.exists() and not allow_overwrite:
            raise DfpPhase1bChunkBenchmarkError(
                f"refusing to overwrite existing output {output_path} (pass --force to overwrite deliberately)"
            )

    code_hashes = code_binding_hashes(identity_document, repo_root=resolved_repo_root)
    code_provenance = build_code_provenance(
        identity_loader_sha256=code_hashes["identity_loader_sha256"], runner_sha256=code_hashes["runner_sha256"],
        cli_sha256=code_hashes["cli_sha256"], launcher_sha256=code_hashes["launcher_sha256"],
        repository_commit=repository_commit,
        required_ancestor_commit=identity_document["identity"]["required_ancestor_commit"],
        repository_worktree_clean=True,
    )

    return ReadinessResult(
        identity_document=identity_document, manifest=manifest,
        repository_commit=repository_commit, code_provenance=code_provenance,
    )


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    _print_banner("verify-protocol")
    load_identity(
        repo_root=Path(args.repo_root),
        calibration_artifact_path=Path(args.artifact) if args.artifact else None,
        workload_bundle_dir=Path(args.workload_bundle_dir) if args.workload_bundle_dir else None,
    )
    print("E17 identity verified: closed schema + relational hashes all pass.")
    return 0


def _cmd_build_manifest(args: argparse.Namespace) -> int:
    _print_banner("build-manifest")
    output_path = Path(args.output)
    _reject_lexical_path_traversal(output_path)
    if output_path.exists() and not args.force:
        raise DfpPhase1bChunkBenchmarkError(f"refusing to overwrite existing manifest {output_path} (pass --force to overwrite deliberately)")

    calibration_document = _load_calibration_document(Path(args.artifact))
    manifest = build_benchmark_manifest(
        calibration_document=calibration_document,
        calibration_artifact_sha256=_sha256(Path(args.artifact)),
        e16_identity_sha256=_sha256(Path(args.repo_root) / CANONICAL_DEPENDENCY_RELATIVE_PATHS["e16_identity"]),
    )
    write_trial_document_atomically(manifest, output_path, allow_overwrite=args.force)
    print(f"Benchmark manifest written to {output_path}")
    print(f"benchmark_record_digest={manifest['benchmark_record_digest']}")
    print(f"canonical_manifest_digest={manifest['canonical_manifest_digest']}")
    return 0


def _cmd_verify_manifest(args: argparse.Namespace) -> int:
    _print_banner("verify-manifest")
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    calibration_document = _load_calibration_document(Path(args.artifact)) if args.artifact else None
    verify_benchmark_manifest(
        manifest, calibration_document=calibration_document,
        calibration_artifact_sha256=_sha256(Path(args.artifact)) if args.artifact else None,
        e16_identity_sha256=_sha256(Path(args.repo_root) / CANONICAL_DEPENDENCY_RELATIVE_PATHS["e16_identity"]) if args.artifact else None,
    )
    print(f"Manifest {args.manifest} PASSES independent verification.")
    print(f"benchmark_record_digest={manifest['benchmark_record_digest']}")
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    _print_banner("preflight")
    perform_repository_readiness_check(
        repo_root=Path(args.repo_root), artifact_path=Path(args.artifact), manifest_path=Path(args.manifest),
        workload_bundle_dir=Path(args.workload_bundle_dir),
        output_path=Path(args.output) if getattr(args, "output", None) else None,
        allow_overwrite=bool(getattr(args, "force", False)),
    )
    print("E17 identity verified.")
    print("Benchmark manifest verified.")
    print("Repository ancestry and clean-worktree verified.")
    print("PREFLIGHT PASS -- no GPU/dataset/model access has occurred.")
    return 0


def _cmd_run_trial(args: argparse.Namespace) -> int:
    _print_banner("run-trial")
    if args.device != "cuda":
        raise DfpPhase1bChunkBenchmarkError(
            f"--device {args.device!r} is not authorized -- run-trial always requires --device cuda"
        )
    key = (args.trial_index, args.repeat, args.chunk_size)
    if key not in TRIAL_SCHEDULE:
        raise DfpPhase1bChunkBenchmarkError(
            f"(--trial-index {args.trial_index}, --repeat {args.repeat}, --chunk-size {args.chunk_size}) "
            f"is not a member of the frozen trial schedule {TRIAL_SCHEDULE}"
        )

    repo_root = Path(args.repo_root)
    output_path = Path(args.output)

    # Repository readiness -- the EXACT SAME shared check the
    # standalone `preflight` command uses (never a second, potentially
    # drifting implementation). Entirely before any heavy (torch/mmcv/
    # model/dataset) import below. Re-run here even though the launcher
    # may already have run `preflight` earlier -- never relies solely
    # on that having happened.
    readiness = perform_repository_readiness_check(
        repo_root=repo_root, artifact_path=Path(args.artifact), manifest_path=Path(args.manifest),
        workload_bundle_dir=Path(args.workload_bundle_dir),
        output_path=output_path, allow_overwrite=args.force,
    )
    identity_document = readiness.identity_document
    manifest = readiness.manifest
    code_provenance = readiness.code_provenance

    # The memory-guard contract is derived EXCLUSIVELY from the already-
    # verified E17 document -- no independent Python default exists
    # anywhere in this codebase (Finding 2 repair), and no CLI flag or
    # environment variable may override it.
    memory_guard_contract = memory_guard_contract_from_identity(identity_document)

    print("Preflight PASS -- beginning real trial execution.")

    # Only now -- after preflight has already fully succeeded -- import
    # the heavy (torch/mmcv/model/dataset) production stack.
    import torch

    if torch.cuda.device_count() != 1:
        raise DfpPhase1bChunkBenchmarkError(
            f"run-trial requires exactly one visible CUDA device, observed {torch.cuda.device_count()}"
        )
    observed_gpu_name = torch.cuda.get_device_name(0)
    if observed_gpu_name != "NVIDIA H100 80GB HBM3":
        raise DfpPhase1bChunkBenchmarkError(
            f"run-trial requires GPU model 'NVIDIA H100 80GB HBM3', observed {observed_gpu_name!r}"
        )

    from src.dfp_phase1b_calibration_production import (
        build_production_dataset_resources, build_production_image_preparer, build_production_inference,
    )
    from src.dfp_phase1b_calibration_run import process_selected_image, real_device_record, real_software_record
    import src.dfp_phase1b_masked_forward as masked_forward_module
    from src.dfp_phase1b_chunk_benchmark import GpuHostMemoryProvider, run_trial

    benchmark_records = manifest["benchmark_records"]
    warmup_record = manifest["warmup_record"]
    selected_ids = sorted({warmup_record["image_id"]} | {r["image_id"] for r in benchmark_records})

    def build_resources():
        return build_production_dataset_resources(repo_root=repo_root, selected_ids=selected_ids)

    log_dir = output_path.parent

    document = run_trial(
        chunk_size=args.chunk_size, trial_index=args.trial_index, repeat_index=args.repeat,
        manifest=manifest, authority_sha256=_sha256(repo_root / IDENTITY_RELATIVE_PATH),
        manifest_sha256=_sha256(Path(args.manifest)),
        build_resources=build_resources,
        build_image_preparer=build_production_image_preparer,
        build_inference=lambda resources: build_production_inference(resources, device="cuda", log_dir=log_dir),
        get_device_record=real_device_record,
        get_software_record=lambda: real_software_record(repo_root),
        masked_forward_module=masked_forward_module,
        process_selected_image_fn=process_selected_image,
        memory_provider=GpuHostMemoryProvider(),
        memory_guard_contract=memory_guard_contract,
        code_provenance=code_provenance,
        cuda_synchronize=torch.cuda.synchronize,
    )
    write_trial_document_atomically(document, output_path, allow_overwrite=args.force)
    print(f"Trial artifact written to {output_path}")
    print(f"primary_elapsed_ns={document['timing']['primary_elapsed_ns']}")
    print(f"scientific_output_collection_digest={document['scientific_output_digests']['collection_digest']}")
    return 0


def _cmd_verify_trial(args: argparse.Namespace) -> int:
    _print_banner("verify-trial")
    with open(args.trial_file, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    verify_trial_document(
        document,
        expected_authority_sha256=_sha256(Path(args.authority)) if args.authority else None,
        expected_manifest_sha256=_sha256(Path(args.manifest)) if args.manifest else None,
        verify_repo_root=Path(args.repo_root),
    )
    print(f"Trial artifact {args.trial_file} PASSES independent verification.")
    print(f"(trial_index={document['trial_index']}, repeat_index={document['repeat_index']}, chunk_size={document['chunk_size']})")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    _print_banner("merge")
    output_path = Path(args.output)
    _reject_lexical_path_traversal(output_path)
    if output_path.exists() and not args.force:
        raise DfpPhase1bChunkBenchmarkError(f"refusing to overwrite existing merged result {output_path} (pass --force to overwrite deliberately)")

    trials = []
    for trial_path in args.trial:
        with open(trial_path, "r", encoding="utf-8") as handle:
            trials.append(json.load(handle))
    merged = merge_trials(trials, verify_repo_root=Path(args.repo_root))
    write_merged_artifact_atomically(merged, output_path, allow_overwrite=args.force)
    print(f"Merged benchmark result written to {output_path}")
    print(f"scientific_output_digest={merged['scientific_output_digest']}")
    print("No winner or preferred chunk size is recorded by this merge.")
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    _print_banner("verify-result")
    with open(args.result, "r", encoding="utf-8") as handle:
        merged = json.load(handle)
    trials = merged.get("raw_trials")
    if not isinstance(trials, list):
        raise DfpPhase1bChunkBenchmarkError("merged result is missing raw_trials")
    recomputed = merge_trials(trials, verify_repo_root=Path(args.repo_root))
    if recomputed["scientific_output_digest"] != merged["scientific_output_digest"]:
        raise DfpPhase1bChunkBenchmarkError("merged result scientific_output_digest does not recompute")
    if recomputed["code_provenance"] != merged["code_provenance"]:
        raise DfpPhase1bChunkBenchmarkError("merged result code_provenance does not recompute")
    # Relational verification of the persisted descriptive summaries
    # (Finding A repair): merge_trials already recomputed
    # descriptive_summaries_by_chunk_size from the SAME raw_trials this
    # merged result carries -- require it to match the stored value
    # exactly (deterministic structural equality: no numeric tolerance,
    # no reordering/omission tolerance). This is a RELATIONAL check
    # against the existing recomputation path, never a new summary
    # formula and never a scientific digest -- E17 continues to treat
    # these summaries as descriptive, not scientific.
    stored_summary = merged.get("descriptive_summaries_by_chunk_size")
    recomputed_summary = recomputed.get("descriptive_summaries_by_chunk_size")
    if not strict_structural_equal(stored_summary, recomputed_summary):
        raise DfpPhase1bChunkBenchmarkError(
            "merged result descriptive_summaries_by_chunk_size does not recompute from raw_trials"
        )
    print(f"Merged result {args.result} PASSES independent verification.")
    print(f"scientific_output_digest={merged['scientific_output_digest']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DFP Phase 1B chunk-size benchmark CLI (operational performance measurement only)",
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-protocol", help="verify the E17 identity: closed schema + relational hashes")
    p.add_argument("--artifact", type=str, default=None)
    p.add_argument("--workload-bundle-dir", type=str, default=None)

    p = sub.add_parser("build-manifest", help="build and atomically install the 24-window benchmark manifest")
    p.add_argument("--artifact", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("verify-manifest", help="independently re-verify an installed benchmark manifest")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--artifact", type=str, default=None)

    p = sub.add_parser(
        "preflight",
        help="full repository-readiness check -- identity, manifest, git ancestry/clean-worktree; no GPU/dataset/model access",
    )
    p.add_argument("--artifact", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--workload-bundle-dir", type=str, required=True)
    p.add_argument("--output", type=str, default=None, help="optional -- also validate this future run-trial output path is safe/fresh")
    p.add_argument("--force", action="store_true", help="only meaningful together with --output")

    p = sub.add_parser("run-trial", help="run exactly one fresh-process trial (requires --device cuda)")
    p.add_argument("--artifact", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--workload-bundle-dir", type=str, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--trial-index", type=int, required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--chunk-size", type=int, choices=(16, 32, 64), required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("verify-trial", help="verify one trial artifact's closed schema and bindings")
    p.add_argument("--trial-file", type=str, required=True)
    p.add_argument("--authority", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)

    p = sub.add_parser("merge", help="merge exactly six trial artifacts into one closed-schema merged result")
    p.add_argument("--trial", type=str, action="append", required=True, help="repeat once per trial file (exactly six required)")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("verify-result", help="independently re-verify an installed merged result")
    p.add_argument("--result", type=str, required=True)

    return parser


_DISPATCH = {
    "verify-protocol": _cmd_verify_protocol,
    "build-manifest": _cmd_build_manifest,
    "verify-manifest": _cmd_verify_manifest,
    "preflight": _cmd_preflight,
    "run-trial": _cmd_run_trial,
    "verify-trial": _cmd_verify_trial,
    "merge": _cmd_merge,
    "verify-result": _cmd_verify_result,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1B CHUNK BENCHMARK FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
