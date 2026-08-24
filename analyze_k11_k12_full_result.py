#!/usr/bin/env python3
"""Offline, CPU-only reconciliation and paired-uncertainty analysis for a
completed matched k11-vs-k12 finite-step full evaluation.

Never initializes CUDA, never loads the model or projection checkpoint,
never constructs the dataset, and never runs inference -- see
:mod:`src.k11_k12_full_result_analysis` for the full contract. Consumes
only the four artifacts a completed evaluator run already wrote plus
git-archived historical records, and writes one deterministic JSON report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.k11_k12_power_evaluation_identity import K11K12PowerEvaluationError
from src.k11_k12_full_result_analysis import (
    K11K12AnalysisError,
    bootstrap_paired_delta,
    build_report,
    canonical_protocol_comparison_table,
    determine_anchor_decision,
    equivalence_margin_sensitivity,
    fetch_historical_sweep_record,
    label_statistics_reconciliation,
    load_full_artifacts,
    load_pilot_artifacts,
    metric_reduction_variants,
    per_class_analysis,
    pilot_nesting_audit,
    reconstruct_and_verify_metrics,
    scientific_interpretation,
    verify_pairing_consistency,
    write_report_atomically,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline reconciliation and paired-uncertainty analysis for a completed matched k11/k12 full evaluation."
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--result", type=Path, required=True, help="full-run result JSON")
    parser.add_argument("--checkpoint", type=Path, required=True, help="full-run checkpoint JSON")
    parser.add_argument("--per-image-stats", type=Path, required=True, help="full-run per-image-stats manifest JSON")
    parser.add_argument("--per-image-stats-npz", type=Path, default=None, help="full-run per-image-stats NPZ (defaults to the manifest's own npz_filename, resolved next to the manifest)")
    parser.add_argument("--pilot20-result", type=Path, default=None)
    parser.add_argument("--pilot20-checkpoint", type=Path, default=None)
    parser.add_argument("--pilot20-per-image-stats", type=Path, default=None)
    parser.add_argument("--pilot20-per-image-stats-npz", type=Path, default=None)
    parser.add_argument("--pilot100-result", type=Path, default=None)
    parser.add_argument("--pilot100-checkpoint", type=Path, default=None)
    parser.add_argument("--pilot100-per-image-stats", type=Path, default=None)
    parser.add_argument("--pilot100-per-image-stats-npz", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="deterministic JSON report output path")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting an existing report at --output")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20345886)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=200)
    parser.add_argument("--skip-historical-anchor", action="store_true", help="skip git-based historical sweep fetch (e.g. offline dev iteration)")
    return parser


def _resolve_npz(manifest_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    import json

    try:
        manifest = json.loads(manifest_path.read_text())
    except OSError as error:
        raise K11K12AnalysisError(f"cannot read manifest {manifest_path} to resolve its NPZ filename: {error}") from error
    except json.JSONDecodeError as error:
        raise K11K12AnalysisError(f"manifest {manifest_path} is not valid JSON: {error}") from error
    npz_filename = manifest.get("npz_filename")
    if not npz_filename:
        raise K11K12AnalysisError(f"manifest {manifest_path} has no npz_filename and no --per-image-stats-npz was given")
    return manifest_path.parent / npz_filename


def _run(args: argparse.Namespace) -> int:
    npz_path = _resolve_npz(args.per_image_stats, args.per_image_stats_npz)
    bundle = load_full_artifacts(
        result_path=args.result, checkpoint_path=args.checkpoint, manifest_path=args.per_image_stats,
        npz_path=npz_path, repo_root=args.repo_root,
    )
    metrics = reconstruct_and_verify_metrics(bundle)
    pairing = verify_pairing_consistency(bundle)
    if not pairing["all_passed"]:
        raise K11K12AnalysisError(f"GT/pairing consistency violations: {pairing['violations']}")

    bootstrap = bootstrap_paired_delta(
        bundle.arrays["intersect_k11"], bundle.arrays["union_k11"], bundle.arrays["intersect_k12"], bundle.arrays["union_k12"],
        observed_delta_percentage_points=metrics["delta_mIoU_percentage_points"],
        n_replicates=args.bootstrap_replicates, seed=args.bootstrap_seed, chunk_size=args.bootstrap_chunk_size,
    )
    equivalence = equivalence_margin_sensitivity(bootstrap.delta_replicates)
    per_class = per_class_analysis(bundle, metrics)

    pilot_bundles = {}
    if args.pilot20_result is not None:
        pilot_bundles["pilot20"] = load_pilot_artifacts(
            run_mode="pilot20", image_count=20, result_path=args.pilot20_result, checkpoint_path=args.pilot20_checkpoint,
            manifest_path=args.pilot20_per_image_stats,
            npz_path=_resolve_npz(args.pilot20_per_image_stats, args.pilot20_per_image_stats_npz),
            repo_root=args.repo_root,
        )
    if args.pilot100_result is not None:
        pilot_bundles["pilot100"] = load_pilot_artifacts(
            run_mode="pilot100", image_count=100, result_path=args.pilot100_result, checkpoint_path=args.pilot100_checkpoint,
            manifest_path=args.pilot100_per_image_stats,
            npz_path=_resolve_npz(args.pilot100_per_image_stats, args.pilot100_per_image_stats_npz),
            repo_root=args.repo_root,
        )
    nesting = pilot_nesting_audit(bundle, pilot_bundles)

    if args.skip_historical_anchor:
        historical = {"row": {"intersection": [0.0], "union": [0.0], "ground_truth_pixels": [0.0], "predicted_pixels": [0.0]}, "optimum": {}, "blob_sha1": None}
        protocol_table = canonical_protocol_comparison_table(bundle)
        variants = {"variants": {}, "historical_formula_reproduces_historical_reported_exactly": False, "conclusion": "skipped (--skip-historical-anchor)"}
        label_stats = {"same_gt_pixel_universe": False, "total_correct_pixel_diff_relative": float("nan"), "interpretation": "skipped (--skip-historical-anchor)", "e12_total_gt_pixels": 0.0}
        anchor = {"status": "INSUFFICIENT_PROVENANCE", "rationale": "historical anchor fetch skipped by flag", "material_protocol_differences": [], "unknown_protocol_fields": [], "no_new_canonical_anchor_created": True, "recommendation": "rerun without --skip-historical-anchor to reconcile the canonical anchor"}
    else:
        historical = fetch_historical_sweep_record(args.repo_root)
        protocol_table = canonical_protocol_comparison_table(bundle)
        variants = metric_reduction_variants(bundle, metrics, historical)
        label_stats = label_statistics_reconciliation(metrics, historical)
        anchor = determine_anchor_decision(protocol_table, variants, label_stats)

    interpretation = scientific_interpretation(bootstrap)

    report = build_report(
        bundle=bundle, metrics=metrics, pairing=pairing, bootstrap=bootstrap, equivalence=equivalence,
        per_class=per_class, nesting=nesting, protocol_table=protocol_table, variants=variants,
        label_stats=label_stats, anchor=anchor, interpretation=interpretation,
    )
    write_report_atomically(args.output, report, overwrite=args.overwrite)
    print(
        f"K11/K12 FULL RESULT ANALYSIS PASS delta_mIoU={metrics['delta_mIoU_percentage_points']:.6f} "
        f"ci=[{bootstrap.ci_low_2_5:.6f},{bootstrap.ci_high_97_5:.6f}] classification={bootstrap.classification} "
        f"anchor={anchor['status']} -> {args.output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # K11K12PowerEvaluationError is caught too: load_full_artifacts /
    # load_pilot_artifacts reuse verify_record, validate_checkpoint_structure,
    # and validate_checkpoint_against_artifact directly (never
    # reimplemented), and those raise this type for their own structural/
    # relational failures -- never independently wrapped into
    # K11K12AnalysisError, so both types must fail closed through this same
    # single reporting boundary. Both are ValueError subclasses but neither
    # this tuple nor either type is ever widened to Exception/BaseException.
    try:
        return _run(args)
    except (K11K12AnalysisError, K11K12PowerEvaluationError) as error:
        print(f"K11/K12 FULL RESULT ANALYSIS FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
