#!/usr/bin/env python3
"""E23-authorized CLI for the real DFP Phase 1B BALP 350-image
development execution.

`verify-protocol`         -- load + closed-schema + relational-hash
                              verify the committed E23 identity, E22
                              identity/artifacts, and the frozen
                              350/150 split. Torch-free.
`verify-source-binding`    -- verify the real COCO-Stuff dataset root
                              against the frozen subset manifest
                              (reuses `src.dfp_phase1a_preflight`'s
                              existing verifier; requires
                              `COCO_STUFF_REAL_DATA_ROOT`). Torch-free.
`verify-model-binding`     -- verify the model/evaluator config-source-
                              hash and bridge-checkpoint binding (reuses
                              `src.dfp_phase1a_preflight.
                              validate_model_evaluator_binding`). No
                              torch, no model construction.
`preflight`                -- everything above, plus: the committed
                              350-image/769-window development window-
                              universe manifest, output-root path
                              safety, and checkpoint state (fresh /
                              resume / complete-but-unfinalized /
                              already-finalized) for a given
                              `--output-root`. Torch-free throughout.
`execute`                  -- THE real execution command. Requires
                              `--mode fresh|resume`, `--device cuda`,
                              `--output-root <absolute path>`. Runs the
                              full preflight first, then delegates all
                              checkpoint/resume/finalize-only
                              orchestration to the committed, E23-bound
                              `execute_development`.
`verify-checkpoint`        -- load + validate an installed checkpoint
                              under `--output-root`.
`verify-development-result`-- load + validate an installed
                              `development_result/result.json` under
                              `--output-root`.

Only real image/GT/weight/CUDA access this CLI ever performs happens
inside `execute`, and only after preflight has already passed. Every
other command is read-only verification.

Scientific and population values (candidate/control identity, split
membership, window universe, resource contract) come exclusively from
the committed E22/E23 identities and manifests -- this CLI exposes no
`--max-images`, `--debug-subset`, `--image-ids`, `--dataset`,
`--allow-overwrite`, `--force`, `--candidate`, `--confirmation`,
`--full-validation`, `--steps`, `--k`, `--crop-size`, or `--stride`
flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.dfp_phase1b_balp_development_execution_identity as e23  # noqa: E402
import src.dfp_phase1b_balp_development_production as prod  # noqa: E402
import src.dfp_phase1b_lightweight_boundary_artifact as art  # noqa: E402
import src.dfp_phase1b_lightweight_boundary_identity as e22  # noqa: E402
from src.dfp_phase1a_preflight import (  # noqa: E402
    DfpPhase1aPreflightError,
    validate_model_evaluator_binding,
    validate_output_root_safety,
    verify_frozen_subset_manifest,
)


class DfpPhase1bBalpDevelopmentCliError(ValueError):
    """Raised for CLI-level failures (argument/mode/path errors) that
    are not already one of the underlying modules' own domain
    exceptions."""


_KNOWN_ERRORS = (
    DfpPhase1bBalpDevelopmentCliError,
    e23.DfpPhase1bBalpDevelopmentExecutionIdentityError,
    e22.DfpPhase1bLightweightBoundaryIdentityError,
    art.DfpPhase1bLightweightBoundaryArtifactError,
    prod.DfpPhase1bBalpDevelopmentProductionError,
    DfpPhase1aPreflightError,
    OSError,
    ValueError,
    json.JSONDecodeError,
)


def _banner(command: str) -> None:
    print(f"DFP Phase 1B BALP development execution [{command}] -- E23 AUTHORITY.")
    print("  - Authorizes ONLY real BALP evaluation on the frozen 350-image development")
    print("    split (769 committed windows), via the committed BALP runner and the")
    print("    committed real dataset/geometry/model provider, on exactly 1x H100 80GB,")
    print("    wall time <= 05:00:00, in fresh/resume/finalize-only-via-resume modes.")
    print("  - Does NOT access the 150-image confirmation split or the other 4,500")
    print("    validation images. Does NOT run exact masked-forward provenance,")
    print("    component construction, or candidate-specific recurrence. Does NOT")
    print("    select a candidate, run a confirmation evaluation, or touch the full")
    print("    5,000-image population.")


def _all_variant_ids() -> list[str]:
    # Torch-free: e22.reconstruct_candidate_ids() derives the 16
    # candidate ID strings from pure closed-form formatting (identical
    # to the existing committed CLI's own _all_variant_ids), never by
    # importing the BALP scientific kernel module (which is
    # torch-dependent at module scope) merely to read a string list.
    return list(e22.CANONICAL_CONTROLS) + list(e22.reconstruct_candidate_ids())


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    _banner("verify-protocol")
    repo_root = Path(args.repo_root)
    e23_doc, e22_doc, split, _ = prod.load_verified_authority(repo_root)
    pop = e23_doc["population"]
    rc = e23_doc["resource_contract"]
    ab = e23_doc["authorization_boundary"]
    print("E23 identity verified (closed schema + exact values + relational dependency hashes, incl. E22).")
    print(f"development_images={pop['development_image_count']}  windows={pop['window_count']}  window_digest={pop['window_universe_digest']}")
    print(f"gpu={rc['gpu_count']}x{rc['gpu_model']}  max_wall_time={rc['max_wall_time']}")
    print(
        f"confirmation_execution_authorized={ab['confirmation_execution_authorized']}  "
        f"full_5000_image_evaluation_authorized={ab['full_5000_image_evaluation_authorized']}  "
        f"exact_masked_forward_provenance_authorized={ab['exact_masked_forward_provenance_authorized']}  "
        f"candidate_specific_recurrence_authorized={ab['candidate_specific_recurrence_authorized']}"
    )
    print(f"development_count(split)={len(split['development_image_ids'])}  confirmation_count(split)={len(split['confirmation_image_ids'])}")
    return 0


def _cmd_verify_source_binding(args: argparse.Namespace) -> int:
    _banner("verify-source-binding")
    repo_root = Path(args.repo_root)
    from src.dfp_phase1a_identity import IDENTITY_RELATIVE_PATH
    from src.dfp_phase1a_subset_manifest import (
        DfpPhase1aSubsetManifestError,
        coco_stuff_canonical_validation_ids,
        coco_stuff_class_presence,
        require_env_root,
        resolve_coco_stuff_root,
        sha256_file,
        verify_manifest_reconciles_with_selection,
    )
    from src.dfp_phase1a_preflight import DATASET_CONTRACTS, ENV_VAR_BY_RUN_MODE

    protocol_identity_sha256 = sha256_file(repo_root / IDENTITY_RELATIVE_PATH)
    _, _, manifest = verify_frozen_subset_manifest(
        repo_root=repo_root, dataset_key="coco_stuff", protocol_identity_sha256=protocol_identity_sha256,
        manifest_path=repo_root / "evaluation_subsets/dfp_phase1a_coco_stuff_500.json",
    )
    root = require_env_root(ENV_VAR_BY_RUN_MODE["coco_stuff_500"], override=args.data_root)
    contract = DATASET_CONTRACTS["coco_stuff"]
    dataset_root = resolve_coco_stuff_root(root)
    canonical_ids = coco_stuff_canonical_validation_ids(dataset_root, expected_count=contract["source_validation_count"])
    presence = coco_stuff_class_presence(dataset_root, canonical_ids)
    verify_manifest_reconciles_with_selection(manifest, canonical_ids=canonical_ids, presence=presence)
    print("PASS: real COCO-Stuff source binding verified against the frozen coco_stuff_500 subset manifest")
    return 0


def _cmd_verify_model_binding(args: argparse.Namespace) -> int:
    _banner("verify-model-binding")
    repo_root = Path(args.repo_root)
    identity_name = validate_model_evaluator_binding(repo_root, "coco_stuff", check_git=not args.no_git_check)
    print(f"PASS: model/evaluator binding verified (evaluator_identity={identity_name})")
    return 0


def _checkpoint_state_description(output_root: Path) -> tuple[str, dict | None]:
    variant_ids = _all_variant_ids()
    existing = art.load_full_checkpoint(output_root, variant_ids=variant_ids)
    if existing is None:
        return "fresh (no checkpoint present)", None
    if existing["final"]:
        return "already finalized -- execute would be REJECTED", existing
    if existing["complete"]:
        return "complete-but-unfinalized -- execute would perform finalize-only recovery", existing
    n = len(existing["progress"]["completed_image_ids"])
    return f"resume -- {n}/{prod.DEVELOPMENT_COUNT} development images already completed", existing


def _cmd_preflight(args: argparse.Namespace) -> int:
    _banner("preflight")
    repo_root = Path(args.repo_root)
    e23_doc, e22_doc, split, _ = prod.load_verified_authority(repo_root)
    print("committed E23 + E22 + frozen split: OK")

    window_index = prod.load_development_window_index(repo_root, e23_doc=e23_doc)
    prod.verify_population_isolation(window_index, development_ids=split["development_image_ids"], confirmation_ids=split["confirmation_image_ids"])
    print(f"development window universe: OK ({len(window_index.canonical_ids)} images, {window_index.aggregate_window_count} windows, digest={window_index.canonical_record_digest})")

    if not args.skip_source_binding:
        _cmd_verify_source_binding(args)
    if not args.skip_model_binding:
        _cmd_verify_model_binding(args)

    forbidden_roots = [
        repo_root / "src", repo_root / "evaluation_identities", repo_root / "evaluation_subsets",
        repo_root / "docs", repo_root / "tests", repo_root / "diagnostics", repo_root / ".git",
    ]
    output_root = validate_output_root_safety(output_root=Path(args.output_root), forbidden_roots=forbidden_roots)
    print(f"output-root path policy: OK ({output_root})")

    description, existing = _checkpoint_state_description(output_root)
    print(f"checkpoint state at output-root: {description}")
    if existing is not None:
        art.validate_full_checkpoint(existing, development_ids=split["development_image_ids"], confirmation_ids=split["confirmation_image_ids"])
        print("existing checkpoint: schema-valid and relationally consistent with committed E22/E23")

    print(
        "PREFLIGHT PASSED. No image or annotation pixel decoded by this command itself beyond the "
        "already-established source-binding verification above. No model constructed. No CUDA initialized."
    )
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    _banner("execute")
    if args.device != "cuda":
        raise DfpPhase1bBalpDevelopmentCliError(f"--device must be exactly 'cuda' (E23 authorizes GPU execution only), observed {args.device!r}")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        raise DfpPhase1bBalpDevelopmentCliError("--output-root must be an absolute path")

    repo_root = Path(args.repo_root)
    # Full preflight first -- identical checks to `preflight`, never skipped for a real execute.
    preflight_args = argparse.Namespace(
        repo_root=args.repo_root, output_root=args.output_root, data_root=args.data_root,
        no_git_check=args.no_git_check, skip_source_binding=False, skip_model_binding=False,
    )
    _cmd_preflight(preflight_args)

    _, existing = _checkpoint_state_description(output_root)
    if args.mode == "fresh" and existing is not None:
        raise DfpPhase1bBalpDevelopmentCliError(
            f"--mode fresh requested but a checkpoint already exists at {output_root} -- pass --mode resume, "
            "or use a fresh --output-root"
        )
    if args.mode == "resume" and existing is None:
        raise DfpPhase1bBalpDevelopmentCliError(f"--mode resume requested but no checkpoint exists at {output_root} -- use --mode fresh for a first run")

    log_dir = output_root / "logs"
    result = prod.run_development_execution(repo_root, output_root, device=args.device, log_dir=log_dir)
    print(f"EXECUTE COMPLETE. control_results={len(result['control_results'])} candidate_results={len(result['candidate_results'])}")
    print(f"scientific_output_digest={result['scientific_output_digest']}")
    return 0


def _cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    _banner("verify-checkpoint")
    repo_root = Path(args.repo_root)
    _, _, split, _ = prod.load_verified_authority(repo_root)
    variant_ids = _all_variant_ids()
    output_root = Path(args.output_root)
    doc = art.load_full_checkpoint(output_root, variant_ids=variant_ids)
    if doc is None:
        raise DfpPhase1bBalpDevelopmentCliError(f"no checkpoint found at --output-root {output_root}")
    art.validate_full_checkpoint(doc, development_ids=split["development_image_ids"], confirmation_ids=split["confirmation_image_ids"])
    progress = doc["progress"]
    print("checkpoint is schema-valid and relationally consistent with committed E22/E23.")
    print(f"complete={doc['complete']}  final={doc['final']}  generation={progress['generation']}")
    print(f"completed_image_ids={len(progress['completed_image_ids'])}/{prod.DEVELOPMENT_COUNT}")
    return 0


def _cmd_verify_result(args: argparse.Namespace) -> int:
    _banner("verify-result")
    variant_ids = _all_variant_ids()
    output_root = Path(args.output_root)
    result_path = output_root / "development_result" / "result.json"
    if not result_path.is_file():
        raise DfpPhase1bBalpDevelopmentCliError(f"no development result found at {result_path}")
    doc = json.loads(result_path.read_bytes())
    art.validate_development_result(doc, variant_ids=variant_ids)
    print("development result is schema-valid and relationally consistent with committed E22/E23.")
    print(f"control_results={len(doc['control_results'])}  candidate_results={len(doc['candidate_results'])}")
    print(f"scientific_output_digest={doc['scientific_output_digest']}")
    print("candidate_selected=false  confirmation_evaluated=false  full_5000_image_evaluation=false")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_dfp_phase1b_balp_development.py")
    parser.add_argument("--repo-root", type=str, default=str(_REPO_ROOT))
    parser.add_argument("--no-git-check", action="store_true", help="skip the git branch/commit ancestry check (local iteration only)")
    parser.add_argument("--data-root", type=str, default=None, help="override for the real COCO-Stuff root (else COCO_STUFF_REAL_DATA_ROOT)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-protocol", help="verify the E23/E22 identities and frozen 350/150 split")
    sub.add_parser("verify-source-binding", help="verify the real COCO-Stuff dataset root against the frozen subset manifest")
    sub.add_parser("verify-model-binding", help="verify the model/evaluator config-source-hash and bridge-checkpoint binding")

    p = sub.add_parser("preflight", help="torch-free preflight for a prospective development execution at --output-root")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--skip-source-binding", action="store_true", help="skip the real-dataset source-binding check (local iteration only)")
    p.add_argument("--skip-model-binding", action="store_true", help="skip the model/evaluator binding check (local iteration only)")

    p = sub.add_parser("execute", help="run the real 350-image BALP development execution")
    p.add_argument("--mode", choices=["fresh", "resume"], required=True)
    p.add_argument("--device", type=str, required=True)
    p.add_argument("--output-root", type=str, required=True)

    p = sub.add_parser("verify-checkpoint", help="validate an installed development checkpoint at --output-root")
    p.add_argument("--output-root", type=str, required=True)

    p = sub.add_parser("verify-development-result", help="validate an installed development_result/result.json at --output-root")
    p.add_argument("--output-root", type=str, required=True)

    args = parser.parse_args(argv)
    handlers = {
        "verify-protocol": _cmd_verify_protocol,
        "verify-source-binding": _cmd_verify_source_binding,
        "verify-model-binding": _cmd_verify_model_binding,
        "preflight": _cmd_preflight,
        "execute": _cmd_execute,
        "verify-checkpoint": _cmd_verify_checkpoint,
        "verify-development-result": _cmd_verify_result,
    }
    try:
        return handlers[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1B BALP FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
