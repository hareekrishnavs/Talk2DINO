#!/usr/bin/env python3
"""CPU-only verification CLI for the DFP Phase 1B Task 6B diagnostic-
execution contract (E20).

`verify-protocol`             -- load + closed-schema + relational-hash
                                  verify the committed E20 identity.
`verify-checkpoint-document`  -- validate a checkpoint JSON document
                                  against the closed checkpoint schema.
`verify-profiling-document`   -- validate a profiling JSON document
                                  against the closed profiling schema.
`verify-result-document`      -- validate a final-result JSON document
                                  against the closed result schema.
`show-contract`               -- print the frozen population/device/
                                  tolerance/chunk-size/8-hour contract.

This CLI implements no execution, run, resume, submit, or benchmark
command. It never imports torch, never initializes CUDA, never accesses
a dataset image, and never loads a model checkpoint -- it only validates
already-written documents and the E20 authority itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1b_diagnostic_execution import (  # noqa: E402
    DfpPhase1bDiagnosticExecutionError,
    IMAGE_COUNT,
    MAXIMUM_JOB_WALL_TIME_SECONDS,
    STAGE_NAME_ENUMERATION,
    WINDOW_COUNT,
    load_identity,
    validate_checkpoint_document,
    validate_profiling_document,
    validate_result_document,
)


def _banner(command: str) -> None:
    print(f"DFP Phase 1B diagnostic-execution contract [{command}] -- AUTHORITY ONLY, NO EXECUTION.")
    print("  - Freezes the 500-image / 1090-window population, scientific-execution binding,")
    print("    profiling contract, checkpoint schema, and final-result schema.")
    print("  - Implements no execution driver, profiler, checkpoint writer, result writer,")
    print("    CLI execution command, or SLURM launcher. Runs no inference. Submits no GPU job.")
    print("  - Does NOT authorize the 500-image diagnostic execution or 5,000-image exact provenance.")


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    _banner("verify-protocol")
    repo_root = Path(args.repo_root)
    identity = load_identity(repo_root=repo_root)
    print("E20 identity verified (closed schema + exact values + relational dependency hashes).")
    print(f"successor_of={identity['identity']['successor_of']}")
    pop = identity["population"]
    print(f"image_count={pop['image_count']}  window_count={pop['window_count']}  component_chunk_size={pop['component_chunk_size']}")
    print(f"gpu_count={pop['gpu_count']}  gpu_model={pop['gpu_model']!r}  maximum_job_wall_time_seconds={pop['maximum_job_wall_time_seconds']}")
    tol = identity["scientific_execution_binding"]["conservation_tolerance"]
    print(f"conservation_tolerance={tol!r}  ({tol.hex()})")
    print(f"stage_count={len(STAGE_NAME_ENUMERATION)}")
    return 0


def _cmd_verify_checkpoint_document(args: argparse.Namespace) -> int:
    _banner("verify-checkpoint-document")
    doc = json.loads(Path(args.document).read_text())
    validate_checkpoint_document(doc)
    print(f"checkpoint document is schema-valid: complete={doc['complete']} final={doc['final']}")
    print(f"completed_image_ids={len(doc['progress']['completed_image_ids'])}  completed_window_count={doc['progress']['completed_window_count']}")
    return 0


def _cmd_verify_profiling_document(args: argparse.Namespace) -> int:
    _banner("verify-profiling-document")
    doc = json.loads(Path(args.document).read_text())
    validate_profiling_document(doc)
    print(f"profiling document is schema-valid: mode={doc['profiling_mode']}  records={len(doc['records'])}")
    return 0


def _cmd_verify_result_document(args: argparse.Namespace) -> int:
    _banner("verify-result-document")
    doc = json.loads(Path(args.document).read_text())
    validate_result_document(doc)
    print(f"result document is schema-valid: image_count={doc['image_count']}  window_count={doc['window_count']}")
    print(f"within_8_hour_allocation={doc['within_8_hour_allocation']}  total_job_wall_time_seconds={doc['total_job_wall_time_seconds']}")
    print("offline_mechanistic_analysis=true  deployed_inference_method=false  full_5000_image_evaluation=false")
    return 0


def _cmd_show_contract(args: argparse.Namespace) -> int:
    _banner("show-contract")
    repo_root = Path(args.repo_root)
    identity = load_identity(repo_root=repo_root)
    pop = identity["population"]
    seb = identity["scientific_execution_binding"]
    print(f"run_mode                     = {pop['run_mode']}")
    print(f"image_count                  = {pop['image_count']}")
    print(f"window_count                 = {pop['window_count']}")
    print(f"component_chunk_size         = {pop['component_chunk_size']}")
    print(f"device_type                  = {pop['device_type']}")
    print(f"gpu_count                    = {pop['gpu_count']}")
    print(f"gpu_model                    = {pop['gpu_model']}")
    print(f"maximum_job_wall_time_seconds= {pop['maximum_job_wall_time_seconds']}")
    print(f"conservation_tolerance       = {seb['conservation_tolerance']!r}  ({seb['comparison_operator']})")
    print(f"stage_name_enumeration       = {len(STAGE_NAME_ENUMERATION)} stages")
    print("POST-CALIBRATION, POST-TOLERANCE, PRE-EXECUTION.")
    print("OFFLINE MECHANISTIC/PROVENANCE ANALYSIS.  NOT THE DEPLOYED TEST-TIME METHOD.")
    print("EXACT MASKED-FORWARD PROVENANCE ON ALL 5,000 IMAGES IS NOT AUTHORIZED.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_dfp_phase1b_diagnostic_execution.py")
    parser.add_argument("--repo-root", type=str, default=str(_REPO_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-protocol", help="verify the E20 identity: closed schema + exact values + relational hashes")

    p = sub.add_parser("verify-checkpoint-document", help="validate a checkpoint JSON document against the closed checkpoint schema")
    p.add_argument("--document", type=str, required=True)

    p = sub.add_parser("verify-profiling-document", help="validate a profiling JSON document against the closed profiling schema")
    p.add_argument("--document", type=str, required=True)

    p = sub.add_parser("verify-result-document", help="validate a final-result JSON document against the closed result schema")
    p.add_argument("--document", type=str, required=True)

    sub.add_parser("show-contract", help="print the frozen population/device/tolerance/chunk-size/8-hour contract")

    args = parser.parse_args(argv)
    handlers = {
        "verify-protocol": _cmd_verify_protocol,
        "verify-checkpoint-document": _cmd_verify_checkpoint_document,
        "verify-profiling-document": _cmd_verify_profiling_document,
        "verify-result-document": _cmd_verify_result_document,
        "show-contract": _cmd_show_contract,
    }
    try:
        return handlers[args.command](args)
    except DfpPhase1bDiagnosticExecutionError as error:
        print(f"DFP PHASE 1B DIAGNOSTIC EXECUTION FAIL [{args.command}]: {error}", file=sys.stderr)
        _banner(args.command)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"DFP PHASE 1B DIAGNOSTIC EXECUTION FAIL [{args.command}]: {error}", file=sys.stderr)
        _banner(args.command)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
