#!/usr/bin/env python3
"""CPU-only verification CLI for the DFP Phase 1B Stage-A provenance-
feature scientific-kernel contract (E21).

`verify-protocol`  -- load + closed-schema + relational-hash verify the
                       committed E21 identity.
`show-contract`    -- print the frozen kernel function/formula/dtype
                       contract.

This CLI implements no execution, run, resume, submit, or benchmark
command. It never imports torch, never initializes CUDA, never accesses
a dataset image, and never loads a model checkpoint -- it only
relationally hash-verifies the E21 authority and its bound files. It
does not import `src.dfp_phase1b_provenance_features` (the numerical
kernel) -- only its already-frozen SHA-256, exactly like the identity
loader it calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1b_provenance_features_identity import (  # noqa: E402
    DfpPhase1bProvenanceFeaturesIdentityError,
    load_identity,
)


def _banner(command: str) -> None:
    print(f"DFP Phase 1B provenance-feature kernel contract [{command}] -- AUTHORITY ONLY, NO EXECUTION.")
    print("  - Freezes the numerical kernel computing the eight E15 Stage-A component-evidence")
    print("    provenance features (k_star, phi, top1_positive_share, positive_evidence_sum,")
    print("    negative_evidence_sum, n_positive_components, top_positive_component_id,")
    print("    top_positive_evidence) from the already-frozen signed component-evidence matrix G[x,r].")
    print("  - Implements no execution driver, runner, profiler, checkpoint writer, result writer,")
    print("    CLI execution command, or SLURM launcher. Runs no inference. Submits no GPU job.")
    print("  - Does NOT authorize the 500-image diagnostic execution or 5,000-image exact provenance.")
    print("  - Does NOT resume implementation of the blocked profiled 500-image runner.")


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    _banner("verify-protocol")
    repo_root = Path(args.repo_root)
    identity = load_identity(repo_root=repo_root)
    print("E21 identity verified (closed schema + exact values + relational dependency hashes).")
    print(f"successor_of={identity['identity']['successor_of']}")
    kb = identity["kernel_binding"]
    print(f"module={kb['module']}  result_type={kb['result_type']}  function={kb['function']}")
    print(f"output_fields={kb['output_fields']}")
    print(f"scientific_kernel_only={identity['operational_scope']['scientific_kernel_only']}")
    print(f"runner_implemented_by_this_task={identity['operational_scope']['runner_implemented_by_this_task']}")
    return 0


def _cmd_show_contract(args: argparse.Namespace) -> int:
    _banner("show-contract")
    repo_root = Path(args.repo_root)
    identity = load_identity(repo_root=repo_root)
    kb = identity["kernel_binding"]
    ks = identity["kernel_interface_schema"]
    print(f"function_signature   = {kb['function_signature']}")
    print(f"g_input              = {kb['g_input_name']}: {ks['g_shape']} {ks['g_dtype']}")
    print(f"m320_input           = {kb['m320_input_name']}: {ks['m320_shape']} {ks['m320_dtype']}")
    print(f"component_ids_input  = {kb['component_ids_input_name']}: {ks['component_ids_shape']} {ks['component_ids_dtype']}")
    print(f"output_fields        = {kb['output_fields']}")
    print(f"accumulation_dtype   = {kb['accumulation_dtype']}  (source: {kb['accumulation_dtype_source']})")
    print(f"epsilon_used         = {kb['tolerance_or_epsilon_used_in_feature_construction']}")
    print(f"reused_e15_formulas  = {kb['reused_e15_formulas_verbatim']}")
    print("SCIENTIFIC KERNEL ONLY.  NO RUNNER, PROFILER, CHECKPOINT/RESULT WRITER, OR CLI EXECUTION COMMAND.")
    print("500-IMAGE EXECUTION AND 5,000-IMAGE EXACT PROVENANCE REMAIN NOT AUTHORIZED.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_dfp_phase1b_provenance_features.py")
    parser.add_argument("--repo-root", type=str, default=str(_REPO_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-protocol", help="verify the E21 identity: closed schema + exact values + relational hashes")
    sub.add_parser("show-contract", help="print the frozen kernel function/formula/dtype contract")

    args = parser.parse_args(argv)
    handlers = {
        "verify-protocol": _cmd_verify_protocol,
        "show-contract": _cmd_show_contract,
    }
    try:
        return handlers[args.command](args)
    except DfpPhase1bProvenanceFeaturesIdentityError as error:
        print(f"DFP PHASE 1B PROVENANCE FEATURES FAIL [{args.command}]: {error}", file=sys.stderr)
        _banner(args.command)
        return 2
    except (OSError, ValueError) as error:
        print(f"DFP PHASE 1B PROVENANCE FEATURES FAIL [{args.command}]: {error}", file=sys.stderr)
        _banner(args.command)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
