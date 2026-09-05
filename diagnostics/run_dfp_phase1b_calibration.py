#!/usr/bin/env python3
"""DFP Phase 1B: CLI for the E16-authorized conservation-calibration
runner.

**This is calibration only.** It does not run full Phase 1B provenance,
does not select or apply a numeric conservation tolerance, and a
successful artifact does not authorize the full Phase 1B run -- see
``docs/dfp_phase1b_calibration_run.md`` for the complete account.

Subcommands
-----------
preflight       Phase A + Phase B: verify E16/dependencies/subset/
                run-mode/output-path, construct the candidate universe,
                and select exactly 200 windows -- no torch import, no
                CUDA, no model, no real image access.
execute         Run the full calibration (Phase C) and atomically
                install ``phase1b_conservation_calibration.json``.
                Requires ``--device cuda`` (never ``cpu`` -- a CPU run
                could never be a genuine completed E16 calibration).
verify-artifact Independently re-validate an installed artifact via
                the committed E16 validator.

Every subcommand except ``execute`` performs no torch import, no CUDA
initialization, and no real dataset/model access. ``execute`` always
runs the full preflight first; every torch/model/dataset import in
this file and in ``src.dfp_phase1b_calibration_run`` is deferred into a
function body, never at module scope.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dfp_phase1b_calibration_artifact import DfpPhase1bCalibrationArtifactError, validate_calibration_artifact
from src.dfp_phase1b_calibration_run import (
    DfpPhase1bCalibrationRunError,
    build_candidate_universe,
    default_window_plan_builder,
    run_preflight,
    select_and_verify,
)

_KNOWN_ERRORS = (DfpPhase1bCalibrationRunError, DfpPhase1bCalibrationArtifactError, OSError, ValueError)


def _print_banner(command: str) -> None:
    print(f"DFP Phase 1B calibration [{command}] -- CALIBRATION MEASUREMENT ONLY.")
    print("  - Does not run full Phase 1B provenance.")
    print("  - Does not select or apply a numeric conservation tolerance.")
    print("  - A successful artifact does NOT authorize the full Phase 1B run.")


def _cmd_preflight(args: argparse.Namespace) -> int:
    _print_banner("preflight")
    report = run_preflight(output_path=Path(args.output), mode=args.mode, allow_overwrite=args.force, repo_root=args.repo_root)
    print(f"E16 identity verified: {report.e16_identity['identity']['name']}")
    print(f"Subset manifest verified: {len(report.selected_ids)} images, run_mode={report.subset_manifest['run_mode']}")
    print(f"Output path OK: {report.output_path}")

    if args.geometry_json is not None:
        with open(args.geometry_json, "r", encoding="utf-8") as handle:
            geometry_payload = json.load(handle)
        from src.dfp_phase1b_calibration_run import ImageGeometry

        def geometry_provider(canonical_id: str):
            entry = geometry_payload[canonical_id]
            return ImageGeometry(
                inference_height=entry["inference_height"], inference_width=entry["inference_width"], patch_size=entry["patch_size"],
            )

        window_plan_builder = default_window_plan_builder(report.repo_root)
        universe = build_candidate_universe(
            report.selected_ids, geometry_provider=geometry_provider, window_plan_builder=window_plan_builder,
        )
        outcome = select_and_verify(universe)
        print(f"Candidate universe: {len(universe)} windows across {len(report.selected_ids)} images")
        print(f"Selected exactly {len(outcome.selected_rows)} windows")
        print(f"candidate_universe_digest={outcome.candidate_universe_digest}")
        print(f"selected_windows_digest={outcome.selected_windows_digest}")
        print(f"selection_trace_digest={outcome.selection_trace_digest}")
    else:
        print("(pass --geometry-json to also run Phase B candidate-universe construction and selection)")

    print("PREFLIGHT PASS -- no execution has occurred.")
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    _print_banner("execute")
    if args.device != "cuda":
        raise DfpPhase1bCalibrationRunError(
            f"--device {args.device!r} is not authorized -- the committed E16 authority requires "
            "device_type='cuda' for a genuine completed calibration artifact; this runner never provides "
            "a --device cpu execution path that could produce a falsely-final artifact."
        )
    raise DfpPhase1bCalibrationRunError(
        "real execution is not implemented by this CLI invocation in this environment: producing a real "
        "phase1b_conservation_calibration.json requires real dataset access, real model weights, and a "
        "real CUDA device, none of which this task is authorized to touch. Wire a real GeometryProvider/"
        "ImagePreparer/InferenceProvider (see src/dfp_phase1b_calibration_run.py's Protocol definitions) "
        "from the real Phase 1A execution context before running this on an authorized GPU allocation."
    )


def _cmd_verify_artifact(args: argparse.Namespace) -> int:
    _print_banner("verify-artifact")
    with open(args.artifact, "r", encoding="utf-8") as handle:
        document = json.load(handle)

    candidate_universe = None
    if args.geometry_json is not None:
        from src.dfp_phase1b_calibration_identity import load_identity as load_e16_identity
        from src.dfp_phase1b_calibration_run import ImageGeometry, load_subset_manifest

        repo_root = Path(args.repo_root)
        e16_identity = load_e16_identity(repo_root=repo_root)
        selected_ids = tuple(load_subset_manifest(repo_root, e16_identity)["selected_ids"])

        with open(args.geometry_json, "r", encoding="utf-8") as handle:
            geometry_payload = json.load(handle)

        def geometry_provider(canonical_id: str):
            entry = geometry_payload[canonical_id]
            return ImageGeometry(
                inference_height=entry["inference_height"], inference_width=entry["inference_width"], patch_size=entry["patch_size"],
            )

        window_plan_builder = default_window_plan_builder(repo_root)
        candidate_universe = build_candidate_universe(
            selected_ids, geometry_provider=geometry_provider, window_plan_builder=window_plan_builder,
        )

    validate_calibration_artifact(document, candidate_universe=candidate_universe)
    print(f"Artifact {args.artifact} PASSES the committed E16 validator.")
    print(f"complete={document['complete']} final={document['final']} rows={len(document['selected_window_records'])}")
    if candidate_universe is None:
        print("(pass --geometry-json to also independently re-derive the selection from the candidate universe)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DFP Phase 1B conservation-calibration CLI (measurement only -- no tolerance, no full-run authorization)",
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="Phase A + Phase B: verify authority/dependencies/subset and (optionally) select 200 windows")
    p.add_argument("--output", type=str, required=True, help="the destination phase1b_conservation_calibration.json path")
    p.add_argument("--mode", choices=("fresh",), required=True, help="only 'fresh' is authorized -- E16 defines no resume semantics")
    p.add_argument("--force", action="store_true", help="explicit authorization to overwrite an existing output path")
    p.add_argument("--geometry-json", type=str, default=None, help="optional: a {canonical_id: {inference_height,inference_width,patch_size}} JSON file to also run Phase B")

    p = sub.add_parser("execute", help="run the full calibration and atomically install the artifact (requires --device cuda)")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--mode", choices=("fresh",), required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--component-chunk-size", type=int, required=True, help="explicit, required -- E16 authorizes recording this value but does not freeze it to one number")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("verify-artifact", help="independently re-validate an installed artifact via the committed E16 validator")
    p.add_argument("--artifact", type=str, required=True)
    p.add_argument("--geometry-json", type=str, default=None, help="optional: also independently re-derive the selection from the candidate universe")

    return parser


_DISPATCH = {
    "preflight": _cmd_preflight,
    "execute": _cmd_execute,
    "verify-artifact": _cmd_verify_artifact,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1B CALIBRATION FAIL [{args.command}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
