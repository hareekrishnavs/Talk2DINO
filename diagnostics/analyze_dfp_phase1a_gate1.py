#!/usr/bin/env python3
"""DFP Phase 1A Gate-1 analysis CLI: deterministic, CPU-only derived-
statistics reconstruction from an already-finalized Phase 1A result
bundle, and the exact preregistered Gate-1 decision.

Never accesses a real dataset root, never constructs a dataset, never
imports/initializes CUDA, never builds a model, never loads model
weights into torch, never runs graph/recurrence code, never mutates the
source result bundle. See src/dfp_phase1a_gate1_analysis.py for the full
scientific/reuse rationale.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import subprocess

from src.dfp_phase1a_gate1_analysis import (
    DfpPhase1aGate1AnalysisError,
    GATE_PROCEED, GATE_STOP,
    apply_gate1_decision,
    build_boundary_concentration,
    build_component_structure,
    build_compute_projection,
    build_dispute_accounting,
    build_horizon_table,
    build_oracle_analysis,
    build_persistence_baseline,
    build_qualitative_selection,
    build_reversion_coherence,
    build_variant_confusion,
    compute_all_variant_metrics,
    independent_numpy_cross_check,
    load_gate1_source,
)
from src.dfp_phase1a_artifact_schema import PREDICTION_VARIANT_NAMES_V2
from src.dfp_phase1a_identity import load_identity as load_protocol_identity
from src.dfp_phase1a_artifact_schema import sha256_file

_KNOWN_ERRORS = (DfpPhase1aGate1AnalysisError, OSError, ValueError)


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace float NaN/Infinity with None (null) -- the
    ONLY JSON-legal representation of an undefined statistic (e.g. a
    per-class IoU for a class with zero evaluated ground-truth pixels
    in this 500-image subset). json.dumps(allow_nan=False) then fails
    loudly on anything this function missed, rather than silently
    emitting a non-standard NaN/Infinity literal."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN != NaN
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _write_strict_json(path: Path, document: Mapping_Any) -> None:
    sanitized = _sanitize_for_json(document)
    text = json.dumps(sanitized, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


# typing alias kept simple/local to avoid a top-level Mapping import churn
Mapping_Any = dict


def _git_head_and_branch(repo_root: Path) -> tuple[str, str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
    return head, branch


def run_gate1_analysis(*, result_dir: Path, output_dir: Path, repo_root: Path, requested_count_per_category: int, expected_image_count: int = 500) -> dict[str, Any]:
    source = load_gate1_source(result_dir, expected_image_count=expected_image_count)

    protocol_identity = load_protocol_identity(repo_root=repo_root)

    variant_metrics = compute_all_variant_metrics(source.arrays, source.class_count)
    for variant in PREDICTION_VARIANT_NAMES_V2:
        independent_numpy_cross_check(source.arrays, variant, source.class_count)

    horizon_table = build_horizon_table(variant_metrics)
    dispute_accounting = build_dispute_accounting(source.arrays)
    oracle = build_oracle_analysis(variant_metrics)
    persistence = build_persistence_baseline(variant_metrics, source.arrays, oracle)
    components = build_component_structure(source.arrays)
    reversion = build_reversion_coherence(source.arrays)
    boundary = build_boundary_concentration(source.arrays)
    compute_projection = build_compute_projection(source.arrays)
    qualitative = build_qualitative_selection(source, requested_count_per_category=requested_count_per_category)
    gate = apply_gate1_decision(oracle, protocol_identity)

    per_class_rows = []
    for variant in PREDICTION_VARIANT_NAMES_V2:
        vm = variant_metrics[variant]
        for c in range(source.class_count):
            per_class_rows.append({
                "variant": variant, "class_index": c,
                "intersection": int(vm.confusion.intersection[c]), "union": int(vm.confusion.union[c]),
                "predicted_pixels": int(vm.confusion.predicted_pixels[c]), "ground_truth_pixels": int(vm.confusion.ground_truth_pixels[c]),
                "iou_pct": float(vm.metrics.per_class_iou[c]), "acc_pct": float(vm.metrics.per_class_accuracy[c]),
            })

    component_summary_rows = []
    for key, stats in components.items():
        if isinstance(stats, dict) and "mean" in stats:
            row = {"quantity": key}
            row.update(stats)
            component_summary_rows.append(row)

    analysis: dict[str, Any] = {
        "run_mode": "coco_stuff_500",
        "image_count": source.image_count,
        "class_count": source.class_count,
        "horizon_table": horizon_table,
        "dispute_accounting": dispute_accounting,
        "oracle_analysis": oracle,
        "persistence_baseline": persistence,
        "component_structure": components,
        "reversion_coherence": reversion,
        "boundary_concentration": boundary,
        "compute_projection": compute_projection,
        "gate1_decision": gate,
        "variant_dataset_level_metrics": {
            v: {"mIoU": variant_metrics[v].metrics.mIoU, "mAcc": variant_metrics[v].metrics.mAcc, "aAcc": variant_metrics[v].metrics.aAcc,
                "classes_with_defined_iou": variant_metrics[v].metrics.classes_with_defined_iou}
            for v in PREDICTION_VARIANT_NAMES_V2
        },
    }

    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise DfpPhase1aGate1AnalysisError(f"analysis destination {output_dir} already exists; refusing to overwrite a previous analysis")
    staging_dir = parent / f"{output_dir.name}.staging-{secrets.token_hex(8)}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    try:
        staging_dir.mkdir(parents=True)

        _write_strict_json(staging_dir / "gate1_analysis.json", analysis)
        _write_strict_json(staging_dir / "qualitative_selection.json", qualitative)

        with (staging_dir / "per_class_metrics.csv").open("w", encoding="utf-8") as f:
            f.write("variant,class_index,intersection,union,predicted_pixels,ground_truth_pixels,iou_pct,acc_pct\n")
            for row in per_class_rows:
                f.write(f"{row['variant']},{row['class_index']},{row['intersection']},{row['union']},{row['predicted_pixels']},{row['ground_truth_pixels']},{row['iou_pct']!r},{row['acc_pct']!r}\n")

        with (staging_dir / "component_summary.csv").open("w", encoding="utf-8") as f:
            f.write("quantity,mean,median,std,min,max,p50,p90,p95,p99\n")
            for row in component_summary_rows:
                f.write(f"{row['quantity']},{row['mean']!r},{row['median']!r},{row['std']!r},{row['min']!r},{row['max']!r},{row['p50']!r},{row['p90']!r},{row['p95']!r},{row['p99']!r}\n")

        report_md = _render_markdown_report(source, analysis, qualitative)
        (staging_dir / "gate1_report.md").write_text(report_md, encoding="utf-8")

        head, branch = _git_head_and_branch(repo_root)
        generated_files = ["gate1_analysis.json", "gate1_report.md", "per_class_metrics.csv", "component_summary.csv", "qualitative_selection.json"]
        provenance = {
            "repository_branch": branch, "repository_head": head,
            "source_result_path": str(source.result_dir),
            "source_artifact_sha256": dict(source.source_artifact_hashes),
            "per_image_stats_schema_name": source.stats_manifest["schema"],
            "per_image_stats_schema_version": source.stats_manifest["schema_version"],
            "selected_ids_digest": source.checkpoint["selected_ids_digest"],
            "image_count": source.image_count,
            "analysis_command": " ".join(sys.argv),
            "analysis_code_sha256": sha256_file(_REPO_ROOT / "src" / "dfp_phase1a_gate1_analysis.py"),
            "analysis_cli_sha256": sha256_file(Path(__file__).resolve()),
            "generated_at_utc_non_scientific_metadata": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generated_output_sha256": {name: sha256_file(staging_dir / name) for name in generated_files},
        }
        _write_strict_json(staging_dir / "analysis_provenance.json", provenance)
        provenance["generated_output_sha256"]["analysis_provenance.json"] = sha256_file(staging_dir / "analysis_provenance.json")
        _write_strict_json(staging_dir / "analysis_provenance.json", provenance)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    try:
        __import__("os").rename(staging_dir, output_dir)
    except OSError as error:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise DfpPhase1aGate1AnalysisError(f"cannot install analysis output at {output_dir}: {error}") from error

    return analysis


def _render_markdown_report(source, analysis: dict, qualitative: dict) -> str:
    gate = analysis["gate1_decision"]
    oracle = analysis["oracle_analysis"]
    dispute = analysis["dispute_accounting"]
    persistence = analysis["persistence_baseline"]
    boundary = analysis["boundary_concentration"]
    compute = analysis["compute_projection"]
    lines = []
    lines.append("# DFP Phase 1A Gate-1 Analysis: COCO-Stuff 500\n")
    lines.append("## 1. Executive verdict\n")
    lines.append(f"**{gate['decision']}**\n\n{gate['rationale']}\n\n{gate['scope_note']}\n")
    lines.append("## 2. Source verification\n")
    lines.append(f"- Result path: `{source.result_dir}`\n- Image count: {source.image_count}\n- Class count: {source.class_count}\n"
                  f"- Schema: `{source.stats_manifest['schema']}` v{source.stats_manifest['schema_version']}\n"
                  f"- Selected-IDs digest: `{source.checkpoint['selected_ids_digest']}`\n")
    lines.append("## 3. Horizon results\n")
    for v, row in analysis["horizon_table"]["rows"].items():
        lines.append(f"- {v}: mIoU={row['mIoU']:.4f}pp, aAcc={row['aAcc']:.4f}pp, mAcc={row['mAcc']:.4f}pp, "
                      f"delta_vs_t0={row['delta_vs_t0_pp']:.4f}pp, delta_vs_t320={row['delta_vs_t320_pp']:.4f}pp\n")
    lines.append(f"\nBest global horizon: **{analysis['horizon_table']['best_global_horizon']}**. "
                 f"T320 better than T0: {analysis['horizon_table']['t320_better_than_t0']} "
                 f"({analysis['horizon_table']['t320_minus_t0_pp']:.4f}pp).\n")
    lines.append("## 4. Dispute accounting\n")
    for k, v in dispute.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("## 5. Oracle headroom\n")
    lines.append(f"- mIoU(T0)={oracle['mIoU_t0']:.4f}, mIoU(T320)={oracle['mIoU_t320']:.4f}, mIoU(O_P)={oracle['mIoU_oracle_o_p']:.4f}\n"
                  f"- G_recover = {oracle['G_recover_pp']:.4f}pp\n- G_oracle_unary = {oracle['G_oracle_unary_pp']:.4f}pp\n"
                  f"- propagation net gain = {oracle['propagation_net_gain_pp']:.4f}pp\n"
                  f"- algebraic residual = {oracle['algebraic_identity_residual_pp']:.2e}\n\n{oracle['note']}\n")
    lines.append("## 6. Persistence baseline (primary horizon T32)\n")
    lines.append(f"- persistence(full) gain vs T320: {persistence['primary_persistence_gain_over_t320_pp']:.4f}pp\n"
                  f"- oracle G_recover: {persistence['oracle_G_recover_pp']:.4f}pp\n"
                  f"- fraction of G_recover explained by cheap persistence: {persistence['primary_persistence_recovery_fraction_of_oracle']!r}\n"
                  f"- beneficial-only gain vs T320: {persistence['beneficial_only_gain_vs_t320_pp']:.4f}pp\n"
                  f"- third-label-only gain vs T320: {persistence['third_label_only_gain_vs_t320_pp']:.4f}pp\n")
    lines.append("## 7. Component structure\n")
    cs = analysis["component_structure"]
    lines.append(f"- total components: {cs['total_component_count']}\n"
                  f"- microcomponent fraction (count): {cs['microcomponent_fraction_of_component_count']!r}\n"
                  f"- microcomponent fraction (pixel area): {cs['microcomponent_fraction_of_total_pixel_area']!r}\n"
                  f"- microcomponent fraction (source mass): {cs['microcomponent_fraction_of_total_source_mass']!r}\n")
    lines.append("## 8. Reversion coherence\n")
    rc = analysis["reversion_coherence"]
    lines.append(f"- harmful-washout pixels: {rc['harmful_washout_pixel_count']}\n"
                  f"- total reversion components: {rc['total_reversion_component_count']}\n"
                  f"- fraction of reverted mass in small (<20px) components: {rc['fraction_of_reverted_pixel_mass_in_small_components']!r}\n"
                  f"- empty-reversion images: {rc['empty_reversion_image_count']}\n")
    lines.append("## 9. Boundary-data limitation\n")
    lines.append(f"{boundary['limitation']}\n\n- harmful-washout boundary fraction: {boundary['harmful_washout_boundary_fraction_of_harmful']!r}\n"
                  f"- disputed boundary fraction: {boundary['disputed_boundary_fraction_of_disputed']!r}\n")
    lines.append("## 10. Phase-1B compute projection\n")
    lines.append(f"- total projected width: {compute['total_projected_width']}\n"
                  f"- dataset worst-case window projected width: {compute['dataset_worst_case_window_projected_width']}\n"
                  f"- dataset worst-case window peak planned bytes: {compute['dataset_worst_case_window_peak_planned_bytes']}\n"
                  f"- projected relative cost vs one standard propagation: {compute['projected_relative_cost_vs_one_standard_propagation']!r}\n")
    lines.append("## 11. Qualitative availability\n")
    for name, cat in qualitative["categories"].items():
        lines.append(f"- {name}: eligible={cat['eligible_count']}, selected={cat['selected_count']}, status={cat['shortage_status']}\n")
    lines.append(f"\nDeferred categories: {list(qualitative['deferred_categories'])}\n\nPanel rendering: {qualitative['panel_rendering']}\n")
    lines.append("## 12. Gate-1 decision table\n")
    for k, v in gate["criteria"].items():
        lines.append(f"- {k}: {v}\n")
    lines.append("## 13. Limitations\n")
    lines.append("- Boundary IoU is unavailable (`implementation_source_status = not_yet_bound`); only boundary/interior pixel-count ratios are reported.\n"
                 "- `revert_set_fragmentation` qualitative category is deferred (not reconstructable from persisted scalars).\n"
                 "- Qualitative panel rendering is deferred (raw prediction/label pixel data is excluded by design from the v2 schema).\n"
                 "- Q_w distributions are percentiles of per-image summaries, not a true pooled per-window percentile.\n"
                 "- Active-component count R_w in isolation is not reconstructable (only R_w*|Q_w| is persisted).\n"
                 "- This Gate-1 predicate is evaluated on coco_stuff_500 only; ade20k_500 has not been run, so no cross-dataset claim is made.\n")
    lines.append("## 14. Scope statement\n")
    lines.append("This is a stratified 500-image feasibility analysis (Phase 1A diagnostic), not the final 5,000-image paper evaluation. "
                 "No claim of a deployable method is made anywhere in this report.\n")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DFP Phase 1A Gate-1 analysis (CPU-only, derived-statistics-only)")
    p.add_argument("--result-bundle", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    p.add_argument("--requested-count-per-category", type=int, default=5)
    p.add_argument("--expected-image-count", type=int, default=500, help="test/diagnostic override only; production runs always use the frozen 500")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_gate1_analysis(
            result_dir=args.result_bundle, output_dir=args.output_dir, repo_root=Path(args.repo_root).resolve(),
            requested_count_per_category=args.requested_count_per_category, expected_image_count=args.expected_image_count,
        )
    except _KNOWN_ERRORS as error:
        print(f"DFP PHASE 1A GATE1 FAIL: {error}", file=sys.stderr)
        return 2
    print(f"DFP PHASE 1A GATE1 ANALYSIS COMPLETE: written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
