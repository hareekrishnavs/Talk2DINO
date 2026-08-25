#!/usr/bin/env python3
"""Offline, CPU-only paired-image bootstrap analysis for a completed
COCO-Object protocol confirmation full (5000-image) evaluation.

Never initializes CUDA, never loads the model or projection checkpoint,
never constructs the dataset, and never runs inference. Consumes only the
four artifacts a completed evaluator run already wrote (result,
checkpoint, per-image-stats manifest, per-image-stats NPZ) plus the
matching pilot20/pilot100 artifacts for the prefix-reproduction check.

Reuses the exact dataset-level mIoU and paired-image bootstrap primitives
already validated for the COCO-Stuff k11-vs-k12 analysis
(``src.k11_k12_full_result_analysis``) unmodified -- the algorithm
(recompute per-class intersection/union sums from the sampled images,
then mIoU from those summed class sums, never averaged per-image mIoUs)
is identical; only the artifact schema (E3/k11/k12 variants, 81 COCO-
Object classes, class names) differs from the COCO-Stuff module's own
constants, so this script adapts the schema layer without touching the
underlying numerics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.k11_k12_full_result_analysis import (
    K11K12AnalysisError,
    aggregate_class_sums,
    bootstrap_paired_delta,
    compute_metrics_from_class_sums,
)
from src.native_edge_support_checkpoint import parse_strict_json_document
from src.k11_k12_power_evaluation_checkpoint import K11K12PowerEvaluationError

SCHEMA_NAME = "talk2dino-coco-object-full-result-analysis-v1"
TOOL_VERSION = "1.0.0"

REQUIRED_RUN_MODE = "full"
REQUIRED_IMAGE_COUNT = 5000
REQUIRED_CLASS_COUNT = 81
BACKGROUND_CLASS_INDEX = 0

VARIANTS = ("E3", "k11", "k12")
_NPZ_SUFFIX = {"E3": "e3", "k11": "k11", "k12": "k12"}

# (label_for_the_comparison, first_variant, second_variant) -- delta is
# always first-minus-second, matching the result JSON's own field naming.
COMPARISONS = (
    ("k11_minus_k12", "k11", "k12"),
    ("k11_minus_E3", "k11", "E3"),
    ("k12_minus_E3", "k12", "E3"),
)

METRIC_RECONSTRUCTION_TOLERANCE_PERCENT = 1e-6


class CocoObjectFullResultAnalysisError(K11K12AnalysisError):
    """Domain error for this analysis. Subclasses K11K12AnalysisError (a
    ValueError) so the same fail-closed exception boundary already
    established for offline reconciliation tooling in this repository
    covers both without widening to Exception/BaseException."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_class_names(root: Path, class_count: int) -> list[str] | None:
    """Independently discovered via AST, never by importing mmseg/the
    dataset module (keeps this analysis CPU-only and CUDA/mmcv-free)."""
    source_path = root / "src/open_vocabulary_segmentation/segmentation/datasets/coco_object.py"
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    class_def = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "COCOObjectDataset"), None
    )
    if class_def is None:
        return None
    assign = next(
        (
            n for n in class_def.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CLASSES" for t in n.targets)
        ),
        None,
    )
    if assign is None:
        return None
    names = list(ast.literal_eval(assign.value))
    return names if len(names) == class_count else None


def load_artifact_bundle(*, result_path: Path, checkpoint_path: Path, manifest_path: Path, npz_path: Path) -> dict[str, Any]:
    result = parse_strict_json_document(result_path, label="result")
    checkpoint = parse_strict_json_document(checkpoint_path, label="checkpoint")
    manifest = parse_strict_json_document(manifest_path, label="per-image-stats manifest")

    if manifest["npz_sha256"] != _sha256_file(npz_path):
        raise CocoObjectFullResultAnalysisError(f"per-image-stats NPZ SHA256 does not match its manifest for {npz_path}")
    if result["per_image_stats_manifest_sha256"] != _sha256_file(manifest_path):
        raise CocoObjectFullResultAnalysisError(f"result.per_image_stats_manifest_sha256 does not match {manifest_path}")
    if checkpoint["identity_sha256"] != result["identity_sha256"]:
        raise CocoObjectFullResultAnalysisError("checkpoint.identity_sha256 disagrees with result.identity_sha256")
    if manifest["image_order_digest"] != result["image_order_digest"]:
        raise CocoObjectFullResultAnalysisError("manifest.image_order_digest disagrees with result.image_order_digest")

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    for variant in VARIANTS:
        suffix = _NPZ_SUFFIX[variant]
        for prefix in ("intersect", "union", "pred"):
            key = f"{prefix}_{suffix}"
            if key not in arrays:
                raise CocoObjectFullResultAnalysisError(f"per-image-stats NPZ is missing required array {key!r}")
    if "label" not in arrays:
        raise CocoObjectFullResultAnalysisError("per-image-stats NPZ is missing required array 'label'")

    n_images = arrays["label"].shape[0]
    n_classes = arrays["label"].shape[1]
    for key, array in arrays.items():
        if key == "dataset_indices":
            continue
        if array.shape[0] != n_images or array.shape[1] != n_classes:
            raise CocoObjectFullResultAnalysisError(f"array {key!r} shape {array.shape} disagrees with label shape {arrays['label'].shape}")

    return {"result": result, "checkpoint": checkpoint, "manifest": manifest, "arrays": arrays}


def _metrics_for_slice(
    arrays: dict[str, np.ndarray], *, variant: str, image_slice: slice | None = None, class_slice: slice | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    suffix = _NPZ_SUFFIX[variant]
    label = arrays["label"]
    intersect = arrays[f"intersect_{suffix}"]
    union = arrays[f"union_{suffix}"]
    if image_slice is not None:
        label = label[image_slice]
        intersect = intersect[image_slice]
        union = union[image_slice]
    if class_slice is not None:
        label = label[:, class_slice]
        intersect = intersect[:, class_slice]
        union = union[:, class_slice]
    sums = aggregate_class_sums(intersect, union, label)
    metrics = compute_metrics_from_class_sums(sums["intersect"], sums["union"], sums["label"])
    return metrics, sums


def reconcile_full_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    """Independently recompute E3/k11/k12 metrics from the raw NPZ class
    sums (dataset-level: sum intersections/unions over all 5000 images,
    THEN divide -- never average per-image mIoUs) and require they match
    the result JSON within tight float64-summation-order tolerance."""
    result = bundle["result"]
    reconstructed: dict[str, dict[str, Any]] = {}
    mismatches = []
    for variant in VARIANTS:
        metrics, sums = _metrics_for_slice(bundle["arrays"], variant=variant)
        reconstructed[variant] = {"metrics": metrics, "sums": sums}
        reported = result[f"metrics_{variant}"]
        for field, reported_key in (("mIoU", "mIoU_percent_0_100"), ("aAcc", "aAcc_percent_0_100"), ("mAcc", "mAcc_percent_0_100")):
            diff = abs(metrics[reported_key] - reported[field])
            if diff > METRIC_RECONSTRUCTION_TOLERANCE_PERCENT:
                mismatches.append({"variant": variant, "field": field, "reconstructed": metrics[reported_key], "reported": reported[field], "abs_diff": diff})
    if mismatches:
        raise CocoObjectFullResultAnalysisError(
            f"independently reconstructed full-run metrics disagree with result.json beyond "
            f"{METRIC_RECONSTRUCTION_TOLERANCE_PERCENT} percentage points: {mismatches}"
        )
    return reconstructed


def reproduce_pilot_prefix(
    *, full_bundle: dict[str, Any], pilot_bundle: dict[str, Any], prefix_length: int, run_mode: str,
) -> dict[str, Any]:
    """Slice the full run's own per-image arrays to the first
    `prefix_length` images (canonical dataset order) and confirm the
    independently-recomputed metrics reproduce the pilot run's own
    reported metrics exactly (within reconstruction tolerance) -- proving
    the full run's per-image data for that prefix is the same data the
    pilot run separately computed, not merely a rerun."""
    full_ids = full_bundle["manifest"]["image_ids"][:prefix_length]
    pilot_ids = pilot_bundle["manifest"]["image_ids"]
    if len(pilot_ids) != prefix_length:
        raise CocoObjectFullResultAnalysisError(f"{run_mode} manifest has {len(pilot_ids)} images, expected {prefix_length}")
    if full_ids != pilot_ids:
        raise CocoObjectFullResultAnalysisError(f"full run's first {prefix_length} image_ids disagree with {run_mode}'s own image_ids -- not the same canonical prefix")

    image_slice = slice(0, prefix_length)
    mismatches = []
    variant_metrics: dict[str, Any] = {}
    for variant in VARIANTS:
        metrics_from_full_prefix, _ = _metrics_for_slice(full_bundle["arrays"], variant=variant, image_slice=image_slice)
        reported_pilot = pilot_bundle["result"][f"metrics_{variant}"]
        variant_metrics[variant] = {
            "reconstructed_from_full_prefix_mIoU_percent": metrics_from_full_prefix["mIoU_percent_0_100"],
            "reported_by_pilot_run_mIoU_percent": reported_pilot["mIoU"],
        }
        diff = abs(metrics_from_full_prefix["mIoU_percent_0_100"] - reported_pilot["mIoU"])
        if diff > METRIC_RECONSTRUCTION_TOLERANCE_PERCENT:
            mismatches.append({"variant": variant, "abs_diff": diff})

    return {
        "run_mode": run_mode,
        "prefix_length": prefix_length,
        "image_ids_match": True,
        "variant_metrics": variant_metrics,
        "reproduces_exactly": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def per_class_gains_losses(bundle: dict[str, Any], class_names: list[str] | None) -> list[dict[str, Any]]:
    per_variant_iou: dict[str, np.ndarray] = {}
    per_variant_sums: dict[str, dict[str, np.ndarray]] = {}
    for variant in VARIANTS:
        metrics, sums = _metrics_for_slice(bundle["arrays"], variant=variant)
        per_variant_iou[variant] = metrics["iou_per_class_fraction_0_1"] * 100.0
        per_variant_sums[variant] = sums

    n_classes = per_variant_iou["k11"].shape[0]
    rows = []
    for c in range(n_classes):
        row: dict[str, Any] = {
            "class_id": c,
            "class_name": class_names[c] if class_names is not None else None,
            "is_background": c == BACKGROUND_CLASS_INDEX,
        }
        for variant in VARIANTS:
            valid = per_variant_sums[variant]["union"][c] > 0
            row[f"{variant}_iou_percent"] = float(per_variant_iou[variant][c]) if valid else None
        for label, a, b in COMPARISONS:
            va, vb = row[f"{a}_iou_percent"], row[f"{b}_iou_percent"]
            row[f"delta_{label}_iou_percent"] = (va - vb) if (va is not None and vb is not None) else None
        rows.append(row)
    return rows


def foreground_only_descriptive_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    """DESCRIPTIVE ONLY -- the official COCO-Object protocol metric
    includes the background class (identity.dataset.metric_includes_background
    = true); this excludes class 0 purely to characterize how much of the
    official delta is driven by background vs. foreground classes. Never
    used as a substitute for the official mIoU."""
    class_slice = slice(1, None)
    out: dict[str, Any] = {}
    metrics_by_variant = {}
    for variant in VARIANTS:
        metrics, _ = _metrics_for_slice(bundle["arrays"], variant=variant, class_slice=class_slice)
        metrics_by_variant[variant] = metrics
        out[f"{variant}_foreground_only_mIoU_percent"] = metrics["mIoU_percent_0_100"]
    for label, a, b in COMPARISONS:
        out[f"delta_{label}_foreground_only_percentage_points"] = (
            metrics_by_variant[a]["mIoU_percent_0_100"] - metrics_by_variant[b]["mIoU_percent_0_100"]
        )
    return out


def background_class_delta(bundle: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    iou_by_variant = {}
    for variant in VARIANTS:
        metrics, sums = _metrics_for_slice(bundle["arrays"], variant=variant)
        valid = sums["union"][BACKGROUND_CLASS_INDEX] > 0
        iou = float(metrics["iou_per_class_fraction_0_1"][BACKGROUND_CLASS_INDEX] * 100.0) if valid else None
        iou_by_variant[variant] = iou
        out[f"{variant}_background_iou_percent"] = iou
    for label, a, b in COMPARISONS:
        va, vb = iou_by_variant[a], iou_by_variant[b]
        out[f"delta_{label}_background_iou_percentage_points"] = (va - vb) if (va is not None and vb is not None) else None
    return out


def run_all_bootstraps(
    bundle: dict[str, Any], reconciled: dict[str, Any], *, n_replicates: int, seed: int, chunk_size: int,
) -> dict[str, Any]:
    arrays = bundle["arrays"]
    results = {}
    for label, a, b in COMPARISONS:
        observed_delta = (
            reconciled[a]["metrics"]["mIoU_percent_0_100"] - reconciled[b]["metrics"]["mIoU_percent_0_100"]
        )
        bootstrap = bootstrap_paired_delta(
            arrays[f"intersect_{_NPZ_SUFFIX[a]}"], arrays[f"union_{_NPZ_SUFFIX[a]}"],
            arrays[f"intersect_{_NPZ_SUFFIX[b]}"], arrays[f"union_{_NPZ_SUFFIX[b]}"],
            observed_delta_percentage_points=observed_delta,
            n_replicates=n_replicates, seed=seed, chunk_size=chunk_size,
        )
        results[label] = bootstrap
    return results


def build_report(
    *, bundle: dict[str, Any], reconciled: dict[str, Any], bootstraps: dict[str, Any],
    background_delta: dict[str, Any], foreground_descriptive: dict[str, Any],
    per_class: list[dict[str, Any]], prefix_checks: dict[str, Any], class_names: list[str] | None,
) -> dict[str, Any]:
    result = bundle["result"]
    comparisons_report = {}
    for label, a, b in COMPARISONS:
        bootstrap = bootstraps[label]
        comparisons_report[label] = {
            "variant_a": a, "variant_b": b,
            "observed_delta_percentage_points_full_precision": bootstrap.observed_delta_percentage_points,
            "bootstrap_mean_delta_percentage_points": bootstrap.bootstrap_mean_delta,
            "bootstrap_standard_error_percentage_points": bootstrap.bootstrap_standard_error,
            "ci_95_percentile_low": bootstrap.ci_low_2_5,
            "ci_95_percentile_high": bootstrap.ci_high_97_5,
            "probability_delta_gt_0": bootstrap.probability_delta_gt_0,
            "probability_delta_lt_0": bootstrap.probability_delta_lt_0,
            "replicate_count": bootstrap.replicate_count,
            "seed": bootstrap.seed,
            "chunk_size": bootstrap.chunk_size,
            "ci_method": bootstrap.ci_method,
        }
    primary = comparisons_report["k11_minus_k12"]
    if primary["ci_95_percentile_high"] < 0:
        primary_classification = "CI_BELOW_ZERO"
        primary_interpretation = "the 12th edge is useful on average in both COCO protocols (transfers from COCO-Stuff to COCO-Object)"
    elif primary["ci_95_percentile_low"] > 0:
        primary_classification = "CI_ABOVE_ZERO"
        primary_interpretation = "true COCO-Object reversal (k11 significantly outperforms k12, opposite sign from COCO-Stuff)"
    else:
        primary_classification = "CI_INCLUDES_ZERO"
        primary_interpretation = "the local-connectivity effect does not clearly transfer to the COCO-Object protocol"

    return {
        "schema": SCHEMA_NAME,
        "tool_version": TOOL_VERSION,
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_result_identity": result["identity"],
        "source_result_identity_sha256": result["identity_sha256"],
        "source_git_commit": result["git_commit"],
        "run_mode": result["run_mode"],
        "image_count": result["image_count_processed"],
        "class_count": result["class_count"],
        "class_names_source": "COCOObjectDataset.CLASSES (AST-discovered)" if class_names is not None else None,
        "reported_metrics": {v: result[f"metrics_{v}"] for v in VARIANTS},
        "reconstructed_metrics_from_npz": {
            v: {
                "mIoU_percent": reconciled[v]["metrics"]["mIoU_percent_0_100"],
                "aAcc_percent": reconciled[v]["metrics"]["aAcc_percent_0_100"],
                "mAcc_percent": reconciled[v]["metrics"]["mAcc_percent_0_100"],
            }
            for v in VARIANTS
        },
        "reconciliation_tolerance_percentage_points": METRIC_RECONSTRUCTION_TOLERANCE_PERCENT,
        "reported_deltas_full_precision": {
            "k11_minus_k12": result["delta_mIoU_k11_minus_k12_percentage_points"],
            "k11_minus_E3": result["delta_mIoU_k11_minus_E3_percentage_points"],
            "k12_minus_E3": result["delta_mIoU_k12_minus_E3_percentage_points"],
        },
        "diffusion_gains_full_precision": {
            "k11_minus_E3_percentage_points": result["delta_mIoU_k11_minus_E3_percentage_points"],
            "k12_minus_E3_percentage_points": result["delta_mIoU_k12_minus_E3_percentage_points"],
            "note": "both positive: graph diffusion (k11/k12) substantially outperforms the raw E3 unary snapshot with zero propagation",
        },
        "bootstrap": {
            "method": "paired image resampling with replacement; per-replicate weight vector shared across both compared variants (paired sampling); dataset-level mIoU recomputed from resampled class-sum intersections/unions, never from averaged per-image mIoUs",
            "comparisons": comparisons_report,
        },
        "primary_transfer_question": {
            "comparison": "k11_minus_k12",
            "classification": primary_classification,
            "interpretation": primary_interpretation,
            "classification_rule": "CI_BELOW_ZERO: 12th edge useful on average in both protocols. CI_INCLUDES_ZERO: effect does not clearly transfer. CI_ABOVE_ZERO: true COCO-Object reversal.",
        },
        "background_class_delta": background_delta,
        "foreground_only_descriptive_delta": foreground_descriptive,
        "per_class_gains_losses": per_class,
        "pilot_prefix_reproduction": prefix_checks,
        "pilot_direction_reversal_warning": {
            "pilot20_delta_k11_minus_k12_percentage_points": prefix_checks["pilot20"]["variant_metrics"]["k11"]["reported_by_pilot_run_mIoU_percent"] - prefix_checks["pilot20"]["variant_metrics"]["k12"]["reported_by_pilot_run_mIoU_percent"] if "pilot20" in prefix_checks else None,
            "pilot100_delta_k11_minus_k12_percentage_points": prefix_checks["pilot100"]["variant_metrics"]["k11"]["reported_by_pilot_run_mIoU_percent"] - prefix_checks["pilot100"]["variant_metrics"]["k12"]["reported_by_pilot_run_mIoU_percent"] if "pilot100" in prefix_checks else None,
            "full_delta_k11_minus_k12_percentage_points": result["delta_mIoU_k11_minus_k12_percentage_points"],
            "warning": (
                "pilot20 (+0.816 pp) and pilot100 (+0.275 pp) both showed k11 > k12 (opposite sign from the "
                "full 5000-image result). 20-100 images is not an adequate sample size for this class-balanced, "
                "small-object, 81-class metric -- neither pilot's sign or magnitude should be treated as "
                "indicative of the full-dataset result. No conclusion should ever be drawn from pilot20/pilot100 "
                "alone, exactly as the protocol's own interpretation contract states."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline paired-image bootstrap analysis for a completed COCO-Object protocol confirmation full evaluation.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--per-image-stats", type=Path, required=True)
    parser.add_argument("--per-image-stats-npz", type=Path, default=None)
    parser.add_argument("--pilot20-result", type=Path, required=True)
    parser.add_argument("--pilot20-checkpoint", type=Path, required=True)
    parser.add_argument("--pilot20-per-image-stats", type=Path, required=True)
    parser.add_argument("--pilot20-per-image-stats-npz", type=Path, default=None)
    parser.add_argument("--pilot100-result", type=Path, required=True)
    parser.add_argument("--pilot100-checkpoint", type=Path, required=True)
    parser.add_argument("--pilot100-per-image-stats", type=Path, required=True)
    parser.add_argument("--pilot100-per-image-stats-npz", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20345886)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=200)
    return parser


def _resolve_npz(manifest_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    manifest = json.loads(manifest_path.read_text())
    npz_filename = manifest.get("npz_filename")
    if not npz_filename:
        raise CocoObjectFullResultAnalysisError(f"manifest {manifest_path} has no npz_filename and no explicit NPZ path was given")
    return manifest_path.parent / npz_filename


def _run(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.overwrite:
        raise CocoObjectFullResultAnalysisError(f"refusing to overwrite existing report at {args.output} without --overwrite")

    full_bundle = load_artifact_bundle(
        result_path=args.result, checkpoint_path=args.checkpoint, manifest_path=args.per_image_stats,
        npz_path=_resolve_npz(args.per_image_stats, args.per_image_stats_npz),
    )
    if full_bundle["result"]["run_mode"] != REQUIRED_RUN_MODE:
        raise CocoObjectFullResultAnalysisError(f"--result run_mode is {full_bundle['result']['run_mode']!r}, expected {REQUIRED_RUN_MODE!r}")
    if full_bundle["result"]["image_count_processed"] != REQUIRED_IMAGE_COUNT:
        raise CocoObjectFullResultAnalysisError(f"--result image_count_processed is {full_bundle['result']['image_count_processed']}, expected {REQUIRED_IMAGE_COUNT}")
    if full_bundle["result"]["class_count"] != REQUIRED_CLASS_COUNT:
        raise CocoObjectFullResultAnalysisError(f"--result class_count is {full_bundle['result']['class_count']}, expected {REQUIRED_CLASS_COUNT}")

    pilot20_bundle = load_artifact_bundle(
        result_path=args.pilot20_result, checkpoint_path=args.pilot20_checkpoint, manifest_path=args.pilot20_per_image_stats,
        npz_path=_resolve_npz(args.pilot20_per_image_stats, args.pilot20_per_image_stats_npz),
    )
    pilot100_bundle = load_artifact_bundle(
        result_path=args.pilot100_result, checkpoint_path=args.pilot100_checkpoint, manifest_path=args.pilot100_per_image_stats,
        npz_path=_resolve_npz(args.pilot100_per_image_stats, args.pilot100_per_image_stats_npz),
    )

    reconciled = reconcile_full_metrics(full_bundle)
    bootstraps = run_all_bootstraps(
        full_bundle, reconciled, n_replicates=args.bootstrap_replicates, seed=args.bootstrap_seed, chunk_size=args.bootstrap_chunk_size,
    )
    class_names = _canonical_class_names(args.repo_root, full_bundle["result"]["class_count"])
    per_class = per_class_gains_losses(full_bundle, class_names)
    background_delta = background_class_delta(full_bundle)
    foreground_descriptive = foreground_only_descriptive_metrics(full_bundle)
    prefix_checks = {
        "pilot20": reproduce_pilot_prefix(full_bundle=full_bundle, pilot_bundle=pilot20_bundle, prefix_length=20, run_mode="pilot20"),
        "pilot100": reproduce_pilot_prefix(full_bundle=full_bundle, pilot_bundle=pilot100_bundle, prefix_length=100, run_mode="pilot100"),
    }
    for name, check in prefix_checks.items():
        if not check["reproduces_exactly"]:
            raise CocoObjectFullResultAnalysisError(f"{name} prefix reproduction failed: {check['mismatches']}")

    report = build_report(
        bundle=full_bundle, reconciled=reconciled, bootstraps=bootstraps,
        background_delta=background_delta, foreground_descriptive=foreground_descriptive,
        per_class=per_class, prefix_checks=prefix_checks, class_names=class_names,
    )

    temp_path = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, args.output)

    primary = report["primary_transfer_question"]
    k11_k12 = report["bootstrap"]["comparisons"]["k11_minus_k12"]
    print(
        f"COCO-OBJECT FULL RESULT ANALYSIS PASS "
        f"delta_k11_minus_k12={k11_k12['observed_delta_percentage_points_full_precision']:.6f} "
        f"ci=[{k11_k12['ci_95_percentile_low']:.6f},{k11_k12['ci_95_percentile_high']:.6f}] "
        f"classification={primary['classification']} "
        f"pilot20_prefix_ok={prefix_checks['pilot20']['reproduces_exactly']} "
        f"pilot100_prefix_ok={prefix_checks['pilot100']['reproduces_exactly']} "
        f"-> {args.output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (CocoObjectFullResultAnalysisError, K11K12AnalysisError, K11K12PowerEvaluationError) as error:
        print(f"COCO-OBJECT FULL RESULT ANALYSIS FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
