#!/usr/bin/env python3
"""DFP Phase 1A Step 12: unified CLI for CPU-only preflight orchestration,
plus (Step 11) the production COCO-Stuff-500 execution driver.

Every subcommand except ``execute`` is verification/preflight only: no
model inference, no dataset/model object construction, no torch weight
load, no CUDA init, no job submission. ``execute`` is the one command
that performs real work -- but only after its own call into
``src.dfp_phase1a_execute.build_execution_context`` (the full ordered
Phase A/B/C preflight, plus this driver's additive v1-resume-rejection/
v2-only gate) has already returned successfully; see that module's
docstring for the exact heavy-import boundary.

Every check is delegated to the already-committed, already-verified
Step 1-11 validators; this file adds only CLI plumbing and the ordered
Phase A/B/C composition in ``src/dfp_phase1a_preflight.py`` /
``src/dfp_phase1a_execute.py``. Nothing here imports ``torch`` at module
scope -- importing ``src.dfp_phase1a_execute`` itself never imports
torch (every torch/model import in that module is deferred into a
function body, past the heavy-import boundary).

Subcommands
-----------
verify-protocol          verify the frozen scientific protocol identity.
verify-subset             verify one frozen 500-image subset manifest's
                          schema/relational bindings (no dataset access).
verify-source-binding     like verify-subset, plus independent
                          recomputation of the selection from the real
                          dataset (requires the dataset root env var).
verify-model-binding      verify the model/evaluator config-source-hash
                          and bridge-checkpoint raw-byte binding for one
                          dataset (no torch, no model construction).
verify-checkpoint         verify one checkpoint document's structure.
verify-result             verify an installed final-result bundle.
verify-qualitative-report verify an installed qualitative-report bundle.
preflight                 run the full ordered Phase A/B/C preflight for
                          one run and print a single PASS/READY summary.
execute                   run the production COCO-Stuff-500 execution
                          driver (fresh or resume); always runs the full
                          preflight first; performs real model inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ade20k_matched_evaluator_identity import Ade20kMatchedEvaluatorIdentityError
from src.dfp_phase1a_artifact_schema import DfpPhase1aArtifactSchemaError
from src.dfp_phase1a_checkpoint import (
    DfpPhase1aCheckpointError,
    validate_checkpoint_structure,
)
from src.dfp_phase1a_execute import (
    DfpPhase1aExecuteError,
    RUN_NAME as _EXECUTE_RUN_NAME,
    build_execution_context,
    execute_run,
)
from src.dfp_phase1a_identity import DfpPhase1aIdentityError
from src.dfp_phase1a_identity import validate_static_configuration as validate_protocol_configuration
from src.dfp_phase1a_preflight import (
    DfpPhase1aPreflightError,
    ENV_VAR_BY_RUN_MODE,
    EXECUTION_ACTION_FINALIZE_ONLY,
    run_preflight,
    validate_model_evaluator_binding,
    verify_frozen_subset_manifest,
)
from src.dfp_phase1a_qualitative_selection import DfpPhase1aQualitativeReportError
from src.dfp_phase1a_qualitative_selection import validate_static_configuration as validate_qualitative_configuration
from src.dfp_phase1a_report import DfpPhase1aReportError, verify_result_bundle
from src.dfp_phase1a_subset_manifest import (
    DATASET_CONTRACTS,
    DfpPhase1aSubsetManifestError,
    ade20k_class_presence,
    coco_stuff_canonical_validation_ids,
    coco_stuff_class_presence,
    parse_strict_json_document,
    require_env_root,
    resolve_coco_stuff_root,
    sha256_file,
    verify_manifest_reconciles_with_selection,
)
from src.matched_k11_k12_identity import MatchedK11K12Error

_KNOWN_ERRORS = (
    DfpPhase1aExecuteError,
    DfpPhase1aPreflightError,
    DfpPhase1aIdentityError,
    DfpPhase1aArtifactSchemaError,
    DfpPhase1aSubsetManifestError,
    DfpPhase1aCheckpointError,
    DfpPhase1aReportError,
    DfpPhase1aQualitativeReportError,
    MatchedK11K12Error,
    Ade20kMatchedEvaluatorIdentityError,
    OSError,
    ValueError,
)

DATASET_ARG_TO_RUN_MODE = {"coco_stuff": "coco_stuff_500", "ade20k": "ade20k_500"}
DEFAULT_MANIFEST_PATH = {
    "coco_stuff": Path("evaluation_subsets/dfp_phase1a_coco_stuff_500.json"),
    "ade20k": Path("evaluation_subsets/dfp_phase1a_ade20k_500.json"),
}


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    validate_protocol_configuration(repo_root=repo_root, check_git=not args.no_git_check)
    print("PASS: DFP Phase 1A protocol identity verified")
    return 0


def _cmd_verify_subset(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else (repo_root / DEFAULT_MANIFEST_PATH[args.dataset]).resolve()
    validate_protocol_configuration(repo_root=repo_root, check_git=not args.no_git_check)
    from src.dfp_phase1a_identity import IDENTITY_RELATIVE_PATH

    protocol_identity_sha256 = sha256_file(repo_root / IDENTITY_RELATIVE_PATH)
    verify_frozen_subset_manifest(
        repo_root=repo_root, dataset_key=args.dataset, protocol_identity_sha256=protocol_identity_sha256, manifest_path=manifest_path,
    )
    print("PASS: DFP Phase 1A subset manifest verified")
    print(f"# dataset={args.dataset} manifest={manifest_path}", file=sys.stderr)
    return 0


def _cmd_verify_source_binding(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else (repo_root / DEFAULT_MANIFEST_PATH[args.dataset]).resolve()
    validate_protocol_configuration(repo_root=repo_root, check_git=not args.no_git_check)
    from src.dfp_phase1a_identity import IDENTITY_RELATIVE_PATH

    protocol_identity_sha256 = sha256_file(repo_root / IDENTITY_RELATIVE_PATH)
    _, _, manifest = verify_frozen_subset_manifest(
        repo_root=repo_root, dataset_key=args.dataset, protocol_identity_sha256=protocol_identity_sha256, manifest_path=manifest_path,
    )

    root = require_env_root(ENV_VAR_BY_RUN_MODE[DATASET_ARG_TO_RUN_MODE[args.dataset]], override=args.data_root)
    if args.dataset == "coco_stuff":
        contract = DATASET_CONTRACTS["coco_stuff"]
        dataset_root = resolve_coco_stuff_root(root)
        canonical_ids = coco_stuff_canonical_validation_ids(dataset_root, expected_count=contract["source_validation_count"])
        presence = coco_stuff_class_presence(dataset_root, canonical_ids)
    else:
        from src.ade20k_dataset_identity import load_identity as load_ade_source
        from src.ade20k_dataset_manifest import canonical_validation_ids as ade_canonical_ids, resolve_dataset_root as ade_resolve_root

        ade_identity = load_ade_source(repo_root / DATASET_CONTRACTS["ade20k"]["source_identity_relative_path"], repo_root=repo_root)
        dataset_root = ade_resolve_root(root, ade_identity)
        canonical_ids = ade_canonical_ids(dataset_root, ade_identity)
        presence = ade20k_class_presence(dataset_root, ade_identity, canonical_ids)

    verify_manifest_reconciles_with_selection(manifest, canonical_ids=canonical_ids, presence=presence)
    print("PASS: DFP Phase 1A subset manifest verified against the real dataset source binding")
    print(f"# dataset={args.dataset} manifest={manifest_path}", file=sys.stderr)
    return 0


def _cmd_verify_model_binding(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    dataset_key = args.dataset
    identity_name = validate_model_evaluator_binding(repo_root, dataset_key, check_git=not args.no_git_check)
    print("PASS: DFP Phase 1A model/evaluator binding verified")
    print(f"# dataset={dataset_key} evaluator_identity={identity_name}", file=sys.stderr)
    return 0


def _cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    checkpoint = parse_strict_json_document(Path(args.checkpoint), label="checkpoint")
    from src.dfp_phase1a_artifact_schema import validate_static_configuration as validate_artifact_schema

    repo_root = Path(args.repo_root).resolve()
    schema_summary = validate_artifact_schema(repo_root=repo_root)
    from src.dfp_phase1a_identity import load_identity as load_protocol_identity

    protocol = load_protocol_identity(repo_root=repo_root)
    run_mode = checkpoint.get("run_mode")
    manifest_key = "coco_stuff_manifest_sha256" if run_mode == "coco_stuff_500" else "ade20k_manifest_sha256"
    from src.dfp_phase1a_artifact_schema import load_identity as load_artifact_identity

    artifact_identity = load_artifact_identity(repo_root=repo_root)
    class_count_key = "coco_stuff_class_count" if run_mode == "coco_stuff_500" else "ade20k_class_count"
    class_names_digest_key = "coco_stuff_class_names_digest" if run_mode == "coco_stuff_500" else "ade20k_class_names_digest"
    manifest_relative_path = (
        artifact_identity["subset_manifest_binding"]["coco_stuff_manifest_relative_path"]
        if run_mode == "coco_stuff_500"
        else artifact_identity["subset_manifest_binding"]["ade20k_manifest_relative_path"]
    )
    subset_manifest = parse_strict_json_document(repo_root / manifest_relative_path, label="subset manifest")
    validate_checkpoint_structure(
        checkpoint,
        artifact_schema_identity_sha256=sha256_file(repo_root / "evaluation_identities/e13_dfp_phase1a_artifacts.toml"),
        protocol_identity_sha256=schema_summary["protocol_identity_sha256"],
        subset_manifest_sha256=schema_summary[manifest_key],
        bridge_checkpoint_sha256=schema_summary["bridge_checkpoint_sha256"],
        required_ancestor_commit=protocol["identity"]["required_ancestor_commit"],
        run_mode=run_mode,
        image_count_expected=len(subset_manifest["selected_ids"]),
        class_count=artifact_identity["dataset_binding"][class_count_key],
        class_names_digest=artifact_identity["dataset_binding"][class_names_digest_key],
    )
    print("PASS: DFP Phase 1A checkpoint structure verified")
    print(f"# run_mode={run_mode}", file=sys.stderr)
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    result = verify_result_bundle(Path(args.bundle))
    print("PASS: DFP Phase 1A result bundle verified")
    print(f"# run_mode={result['result']['run_mode']} complete={result['result']['complete']} final={result['result']['final']}", file=sys.stderr)
    return 0


def _cmd_verify_qualitative_report(args: argparse.Namespace) -> int:
    from src.dfp_phase1a_qualitative_render import verify_qualitative_bundle

    result = verify_qualitative_bundle(Path(args.bundle))
    print("PASS: DFP Phase 1A qualitative-report bundle verified")
    print(f"# run_mode={result['selection_manifest']['run_mode']}", file=sys.stderr)
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else None
    ctx = run_preflight(
        repo_root=repo_root,
        run_mode=args.run,
        mode=args.mode,
        output_root=Path(args.output_root),
        data_root_override=args.data_root,
        checkpoints_dir=checkpoints_dir,
        checkpoint_filename=args.checkpoint_filename,
        allow_fresh_overwrite=args.allow_fresh_overwrite,
        probe_output_writeable=args.probe_output_writeable,
        check_git=not args.no_git_check,
    )
    summary = ctx.to_summary_dict()
    if summary["execution_action"] == EXECUTION_ACTION_FINALIZE_ONLY:
        print(
            "READY_TO_FINALIZE: DFP Phase 1A preflight passed all phases -- checkpoint is already complete, "
            f"images to process: 0 -- no execution has occurred"
        )
    else:
        print("READY: DFP Phase 1A preflight passed all phases -- no execution has occurred")
    print(
        f"# run_mode={summary['run_mode']} mode={summary['mode']} execution_action={summary['execution_action']} "
        f"expected_count={summary['expected_count']} class_count={summary['class_count']} "
        f"next_dataset_index={summary['next_dataset_index']} completed_count={summary['completed_count']} "
        f"images_to_process={summary['images_to_process']} evaluator_identity={summary['evaluator_identity_name']} "
        f"source_commit={summary['source_commit']}",
        file=sys.stderr,
    )
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    context = build_execution_context(
        run=args.run, mode=args.mode, output_root=Path(args.output_root), device=args.device, repo_root=repo_root,
    )
    summary = execute_run(context)
    status = "COMPLETE" if summary["complete"] else "OK"
    print(f"DFP PHASE 1A EXECUTE {status}: {summary['images_completed']}/{summary['expected_count']} images completed")
    print(f"# run={args.run} mode={args.mode} device={args.device} output_root={args.output_root}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified DFP Phase 1A CLI: verification and ordered preflight (every subcommand except 'execute'), "
        "plus the production COCO-Stuff-500 execution driver ('execute')",
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-protocol", help="verify the frozen scientific protocol identity")
    p.add_argument("--no-git-check", action="store_true")

    p = sub.add_parser("verify-subset", help="verify one frozen 500-image subset manifest")
    p.add_argument("--dataset", choices=("coco_stuff", "ade20k"), required=True)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--no-git-check", action="store_true")

    p = sub.add_parser("verify-source-binding", help="verify a subset manifest against the real dataset (requires *_REAL_DATA_ROOT)")
    p.add_argument("--dataset", choices=("coco_stuff", "ade20k"), required=True)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None, help="override for the *_REAL_DATA_ROOT environment variable")
    p.add_argument("--no-git-check", action="store_true")

    p = sub.add_parser("verify-model-binding", help="verify model/evaluator config-source-hash and bridge-checkpoint binding")
    p.add_argument("--dataset", choices=("coco_stuff", "ade20k"), required=True)
    p.add_argument("--no-git-check", action="store_true")

    p = sub.add_parser("verify-checkpoint", help="verify one checkpoint document's structure and bindings")
    p.add_argument("--checkpoint", type=Path, required=True)

    p = sub.add_parser("verify-result", help="verify an installed final-result bundle directory")
    p.add_argument("--bundle", type=Path, required=True)

    p = sub.add_parser("verify-qualitative-report", help="verify an installed qualitative-report bundle directory")
    p.add_argument("--bundle", type=Path, required=True)

    p = sub.add_parser(
        "preflight",
        help="run the full ordered Phase A/B/C preflight for one run (verification only -- performs no execution)",
    )
    p.add_argument("--run", choices=("coco_stuff_500", "ade20k_500"), required=True, help="exact frozen run/dataset name (no aliases)")
    p.add_argument("--mode", choices=("fresh", "resume"), required=True)
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None, help="override for the *_REAL_DATA_ROOT environment variable")
    p.add_argument("--checkpoints-dir", type=str, default=None, help="defaults to --output-root")
    p.add_argument("--checkpoint-filename", type=str, default="checkpoint.json")
    p.add_argument("--allow-fresh-overwrite", action="store_true", help="explicit authorization to start fresh over an existing checkpoint/result")
    p.add_argument("--probe-output-writeable", action="store_true", help="create+remove a uniquely named temp file to prove writeability")
    p.add_argument("--no-git-check", action="store_true")

    p = sub.add_parser(
        "execute",
        help="run the production COCO-Stuff-500 execution driver (real model inference; always preflights first)",
    )
    p.add_argument("--run", choices=(_EXECUTE_RUN_NAME,), required=True, help="exact frozen run name; only 'coco_stuff_500' is implemented")
    p.add_argument("--mode", choices=("fresh", "resume"), required=True)
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)

    return parser


_DISPATCH = {
    "verify-protocol": _cmd_verify_protocol,
    "verify-subset": _cmd_verify_subset,
    "verify-source-binding": _cmd_verify_source_binding,
    "verify-model-binding": _cmd_verify_model_binding,
    "verify-checkpoint": _cmd_verify_checkpoint,
    "verify-result": _cmd_verify_result,
    "verify-qualitative-report": _cmd_verify_qualitative_report,
    "preflight": _cmd_preflight,
    "execute": _cmd_execute,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1A FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
