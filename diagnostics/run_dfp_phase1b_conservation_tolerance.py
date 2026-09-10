#!/usr/bin/env python3
"""DFP Phase 1B Task 6A: CPU-only verification CLI for the E19
conservation-TOLERANCE successor authority
(`evaluation_identities/e19_dfp_phase1b_conservation_tolerance.toml`).

POST-CALIBRATION, PRE-EVALUATION. This CLI defines nothing and
executes nothing scientific. It only validates the frozen E19 identity
and re-derives the single numeric conservation tolerance from the
completed, independently-verified 200-window
`phase1b_conservation_calibration.json`.

Subcommands
-----------
verify-protocol            Closed-schema + exact-value validation of
                           E19 and relational SHA-256 verification of
                           every bound dependency. No calibration
                           artifact needed.
verify-calibration-binding Read the supplied calibration artifact,
                           hash-verify it against E19, run the
                           committed official E16 calibration-artifact
                           validator, reconstruct every residual-family
                           maximum from the 200 rows, apply the exact
                           power-of-two rule, and confirm exact
                           agreement with the frozen E19 tolerance.
show-tolerance             Print the frozen tolerance, governed family,
                           comparison operator, source-artifact digest,
                           and the POST-CALIBRATION / PRE-EVALUATION
                           labels.

No subcommand imports torch, initializes CUDA, loads model weights or
dataset files, submits a job, or provides any execution/evaluation
path. Ordinary validation failures exit 2 with a clean domain message
and no traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1b_conservation_tolerance import (  # noqa: E402
    DfpPhase1bConservationToleranceError,
    frozen_tolerance,
    load_identity,
    verify_calibration_binding,
)


def _banner(command: str) -> None:
    print(f"DFP Phase 1B conservation tolerance [{command}] -- POST-CALIBRATION, PRE-EVALUATION.")
    print("  - Derived only from the verified 200-window calibration artifact.")
    print("  - Defines a numeric conservation threshold, not an accuracy metric.")
    print("  - Does NOT authorize the 500-image diagnostic, full Phase 1B, or the 5,000-image evaluation.")


def _cmd_verify_protocol(args: argparse.Namespace) -> int:
    _banner("verify-protocol")
    identity = load_identity(repo_root=Path(args.repo_root))
    tol = identity["tolerance"]
    print("E19 identity verified (closed schema + exact values + relational dependency hashes).")
    print(f"successor_of={identity['identity']['successor_of']}")
    print(f"governed_family={identity['governed_residual']['governed_family']}  tolerance_structure={identity['governed_residual']['tolerance_structure']}")
    print(f"conservation_tolerance={tol['conservation_tolerance']!r}  (2**{tol['conservation_tolerance_binary_exponent']})  comparison_operator={tol['comparison_operator']!r}")
    return 0


def _cmd_verify_calibration_binding(args: argparse.Namespace) -> int:
    _banner("verify-calibration-binding")
    summary = verify_calibration_binding(
        calibration_artifact_path=Path(args.artifact),
        repo_root=Path(args.repo_root),
    )
    print("Calibration artifact hash-verified against E19 and PASSES the committed E16 validator.")
    print(f"rows={summary['row_count']}  calibration_artifact_sha256={summary['calibration_artifact_sha256']}")
    print("row-level reconstructed maximum absolute residual per family:")
    for fam in ("rho_M", "rho_0", "rho_320", "rho_g"):
        gov = " (GOVERNED)" if fam == summary["governed_family"] else ""
        print(f"  {fam}: {summary['reconstructed_family_maxima'][fam]!r}  samples={summary['reconstructed_family_sample_counts'][fam]}{gov}")
    print(f"governed observed maximum (M) = {summary['governed_observed_maximum']!r}")
    print(f"recomputed conservation_tolerance = {summary['conservation_tolerance']!r}  (2**{summary['conservation_tolerance_binary_exponent']})")
    print(f"acceptance: absolute_residual {summary['comparison_operator']} conservation_tolerance  (inclusive)")
    print("recomputed tolerance EXACTLY matches the frozen E19 value.")
    return 0


def _cmd_show_tolerance(args: argparse.Namespace) -> int:
    _banner("show-tolerance")
    info = frozen_tolerance(repo_root=Path(args.repo_root))
    print(f"conservation_tolerance = {info['conservation_tolerance']!r}  (2**{info['conservation_tolerance_binary_exponent']})")
    print(f"governed_family        = {info['governed_family']}")
    print(f"comparison_operator    = {info['comparison_operator']}  ({info['acceptance_relation']})")
    print(f"recorded_but_ungoverned_families = {info['recorded_but_ungoverned_families']}")
    print(f"observed_maximum_absolute_residual = {info['observed_maximum_absolute_residual']!r}")
    print(f"source_calibration_artifact_sha256 = {info['source_calibration_artifact_sha256']}")
    print(f"{info['label_post_calibration']}  {info['label_pre_evaluation']}")
    return 0


_DISPATCH = {
    "verify-protocol": _cmd_verify_protocol,
    "verify-calibration-binding": _cmd_verify_calibration_binding,
    "show-tolerance": _cmd_show_tolerance,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DFP Phase 1B conservation-tolerance verification CLI (POST-CALIBRATION, PRE-EVALUATION; CPU-only; defines and executes nothing scientific).",
    )
    parser.add_argument("--repo-root", type=str, default=str(_REPO_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-protocol", help="closed-schema + exact-value + relational-hash validation of E19")

    p = sub.add_parser("verify-calibration-binding", help="reconstruct the tolerance from the 200-window calibration artifact and confirm it matches E19")
    p.add_argument("--artifact", type=str, required=True, help="path to phase1b_conservation_calibration.json")

    sub.add_parser("show-tolerance", help="print the frozen tolerance, governed family, operator, source digest, and labels")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except DfpPhase1bConservationToleranceError as error:
        print(f"DFP PHASE 1B CONSERVATION TOLERANCE FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"DFP PHASE 1B CONSERVATION TOLERANCE FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
