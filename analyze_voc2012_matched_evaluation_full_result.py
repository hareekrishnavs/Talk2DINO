#!/usr/bin/env python3
"""Offline, CPU-only paired-image bootstrap analysis for a completed
VOC2012 V20/V21 matched-evaluator full (1449-image) run.

Never initializes CUDA, never loads the model or projection checkpoint,
never constructs the mmseg dataset, and never runs inference. Consumes
only the artifacts a completed evaluator run already wrote (result,
checkpoint, per-image-stats manifest, per-image-stats NPZ) plus the
matching pilot20/pilot100 artifacts for the prefix-reproduction check.

Reuses the exact dataset-level mIoU and paired-image bootstrap
primitives already validated for the COCO-Stuff k11-vs-k12 analysis and
the COCO-Object full-result analysis (``src.k11_k12_full_result_analysis``)
unmodified -- the algorithm (recompute per-class intersection/union sums
from the sampled images, then mIoU from those summed class sums, never
averaged per-image mIoUs; confidence intervals from resampled RAW
full-precision integer sufficient statistics, never from the rounded
percent_0_100 values already stored in result.json) is identical; only
the schema (six V20/V21 x E3/k11/k12 variants, two class counts, one
shared background class in V21 only) differs.

Framing (per explicit request): propagation (k11 and k12) versus E3 is
the PRIMARY result for both V20 and V21 -- it answers "does graph
diffusion help over the raw unary snapshot." k11 versus k12 is the
SECONDARY, connectivity-sensitivity comparison -- it answers "how much
does the specific local-connectivity choice (11th vs 12th edge) matter,"
which is a narrower question than the primary one.
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

SCHEMA_NAME = "talk2dino-voc2012-matched-evaluator-full-result-analysis-v1"
TOOL_VERSION = "1.0.0"

REQUIRED_RUN_MODE = "full"
REQUIRED_IMAGE_COUNT = 1449
V20_CLASS_COUNT = 20
V21_CLASS_COUNT = 21
BACKGROUND_CLASS_INDEX = 0  # V21 only; V20 has no background channel.

VARIANT_NAMES = ("v20_e3", "v20_k11", "v20_k12", "v21_e3", "v21_k11", "v21_k12")
PROTOCOLS = ("v20", "v21")

# PRIMARY: propagation (k11, k12) vs the raw E3 unary snapshot -- for
# both protocols independently. This is the primary scientific question.
PRIMARY_COMPARISONS = (
    ("v20_k11_minus_e3", "v20_k11", "v20_e3", "v20"),
    ("v20_k12_minus_e3", "v20_k12", "v20_e3", "v20"),
    ("v21_k11_minus_e3", "v21_k11", "v21_e3", "v21"),
    ("v21_k12_minus_e3", "v21_k12", "v21_e3", "v21"),
)
# SECONDARY: connectivity sensitivity -- how much the local-connectivity
# choice (k11 vs k12) matters, independent of whether propagation helps.
SECONDARY_COMPARISONS = (
    ("v20_k11_minus_k12", "v20_k11", "v20_k12", "v20"),
    ("v21_k11_minus_k12", "v21_k11", "v21_k12", "v21"),
)
ALL_COMPARISONS = PRIMARY_COMPARISONS + SECONDARY_COMPARISONS

METRIC_RECONSTRUCTION_TOLERANCE_PERCENT = 1e-6


class Voc2012FullResultAnalysisError(K11K12AnalysisError):
    """Domain error for this analysis. Subclasses K11K12AnalysisError (a
    ValueError) so the same fail-closed exception boundary already
    established for offline reconciliation tooling in this repository
    covers both without widening to Exception/BaseException."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v20_class_names(root: Path) -> list[str] | None:
    """Independently discovered via AST, never by importing mmseg/the
    dataset module (keeps this analysis CPU-only and CUDA/mmcv-free)."""
    source_path = root / "src/open_vocabulary_segmentation/segmentation/datasets/pascal_voc.py"
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    class_def = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PascalVOCDataset20"), None)
    if class_def is None:
        return None
    assign = next(
        (n for n in class_def.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CLASSES" for t in n.targets)),
        None,
    )
    if assign is None:
        return None
    names = list(ast.literal_eval(assign.value))
    return names if len(names) == V20_CLASS_COUNT else None


def load_artifact_bundle(*, result_path: Path, checkpoint_path: Path, manifest_path: Path, npz_path: Path) -> dict[str, Any]:
    result = parse_strict_json_document(result_path, label="result")
    checkpoint = parse_strict_json_document(checkpoint_path, label="checkpoint")
    manifest = parse_strict_json_document(manifest_path, label="per-image-stats manifest")

    if manifest["npz_sha256"] != _sha256_file(npz_path):
        raise Voc2012FullResultAnalysisError(f"per-image-stats NPZ SHA256 does not match its manifest for {npz_path}")
    if result["per_image_stats_manifest_sha256"] != _sha256_file(manifest_path):
        raise Voc2012FullResultAnalysisError(f"result.per_image_stats_manifest_sha256 does not match {manifest_path}")
    if result["per_image_stats_npz_sha256"] != _sha256_file(npz_path):
        raise Voc2012FullResultAnalysisError(f"result.per_image_stats_npz_sha256 does not match {npz_path}")
    if checkpoint["identity_sha256"] != result["identity_sha256"]:
        raise Voc2012FullResultAnalysisError("checkpoint.identity_sha256 disagrees with result.identity_sha256")
    if manifest["image_order_digest"] != result["image_order_digest"]:
        raise Voc2012FullResultAnalysisError("manifest.image_order_digest disagrees with result.image_order_digest")

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    for variant in VARIANT_NAMES:
        for prefix in ("intersect", "union", "pred"):
            key = f"{prefix}_{variant}"
            if key not in arrays:
                raise Voc2012FullResultAnalysisError(f"per-image-stats NPZ is missing required array {key!r}")
    for label_key in ("label_v20", "label_v21"):
        if label_key not in arrays:
            raise Voc2012FullResultAnalysisError(f"per-image-stats NPZ is missing required array {label_key!r}")

    n_images = arrays["label_v20"].shape[0]
    if arrays["label_v20"].shape[1] != V20_CLASS_COUNT:
        raise Voc2012FullResultAnalysisError(f"label_v20 has {arrays['label_v20'].shape[1]} columns, expected {V20_CLASS_COUNT}")
    if arrays["label_v21"].shape != (n_images, V21_CLASS_COUNT):
        raise Voc2012FullResultAnalysisError(f"label_v21 shape {arrays['label_v21'].shape} disagrees with expected ({n_images}, {V21_CLASS_COUNT})")
    for variant in VARIANT_NAMES:
        expected_classes = V20_CLASS_COUNT if variant.startswith("v20_") else V21_CLASS_COUNT
        for prefix in ("intersect", "union", "pred"):
            key = f"{prefix}_{variant}"
            if arrays[key].shape != (n_images, expected_classes):
                raise Voc2012FullResultAnalysisError(f"array {key!r} shape {arrays[key].shape} disagrees with expected ({n_images}, {expected_classes})")

    return {"result": result, "checkpoint": checkpoint, "manifest": manifest, "arrays": arrays}


def _label_key(variant: str) -> str:
    return "label_v20" if variant.startswith("v20_") else "label_v21"


def _metrics_for_slice(arrays: dict[str, np.ndarray], *, variant: str, image_slice: slice | None = None, class_slice: slice | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    label = arrays[_label_key(variant)]
    intersect = arrays[f"intersect_{variant}"]
    union = arrays[f"union_{variant}"]
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
    """Independently recompute all six variants' metrics from the raw NPZ
    class sums (dataset-level: sum intersections/unions over all 1449
    images, THEN divide -- never average per-image mIoUs) and require
    they match the result JSON within tight float64-summation-order
    tolerance."""
    result = bundle["result"]
    reconstructed: dict[str, dict[str, Any]] = {}
    mismatches = []
    for variant in VARIANT_NAMES:
        metrics, sums = _metrics_for_slice(bundle["arrays"], variant=variant)
        reconstructed[variant] = {"metrics": metrics, "sums": sums}
        reported = result[f"metrics_{variant}"]
        for field, reported_key in (("mIoU", "mIoU_percent_0_100"), ("aAcc", "aAcc_percent_0_100"), ("mAcc", "mAcc_percent_0_100")):
            diff = abs(metrics[reported_key] - reported[field])
            if diff > METRIC_RECONSTRUCTION_TOLERANCE_PERCENT:
                mismatches.append({"variant": variant, "field": field, "reconstructed": metrics[reported_key], "reported": reported[field], "abs_diff": diff})
    if mismatches:
        raise Voc2012FullResultAnalysisError(
            f"independently reconstructed full-run metrics disagree with result.json beyond "
            f"{METRIC_RECONSTRUCTION_TOLERANCE_PERCENT} percentage points: {mismatches}"
        )
    return reconstructed


def reproduce_pilot_prefix(*, full_bundle: dict[str, Any], pilot_bundle: dict[str, Any], prefix_length: int, run_mode: str) -> dict[str, Any]:
    """Slice the full run's own per-image arrays to the first
    `prefix_length` images (canonical dataset order) and confirm the
    independently-recomputed metrics reproduce the pilot run's own
    reported metrics exactly (within reconstruction tolerance) -- proving
    the full run's per-image data for that prefix is the same data the
    pilot run separately computed, not merely a rerun."""
    full_ids = full_bundle["manifest"]["image_ids"][:prefix_length]
    pilot_ids = pilot_bundle["manifest"]["image_ids"]
    if len(pilot_ids) != prefix_length:
        raise Voc2012FullResultAnalysisError(f"{run_mode} manifest has {len(pilot_ids)} images, expected {prefix_length}")
    if full_ids != pilot_ids:
        raise Voc2012FullResultAnalysisError(f"full run's first {prefix_length} image_ids disagree with {run_mode}'s own image_ids -- not the same canonical prefix")

    image_slice = slice(0, prefix_length)
    mismatches = []
    variant_metrics: dict[str, Any] = {}
    for variant in VARIANT_NAMES:
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
        "run_mode": run_mode, "prefix_length": prefix_length, "image_ids_match": True,
        "variant_metrics": variant_metrics, "reproduces_exactly": len(mismatches) == 0, "mismatches": mismatches,
    }


def per_class_gains_losses(bundle: dict[str, Any], v20_class_names: list[str] | None) -> dict[str, list[dict[str, Any]]]:
    v21_class_names = (["background"] + v20_class_names) if v20_class_names is not None else None
    out: dict[str, list[dict[str, Any]]] = {}
    for protocol, class_names, n_classes in (("v20", v20_class_names, V20_CLASS_COUNT), ("v21", v21_class_names, V21_CLASS_COUNT)):
        per_variant_iou: dict[str, np.ndarray] = {}
        per_variant_sums: dict[str, dict[str, np.ndarray]] = {}
        for suffix in ("e3", "k11", "k12"):
            variant = f"{protocol}_{suffix}"
            metrics, sums = _metrics_for_slice(bundle["arrays"], variant=variant)
            per_variant_iou[suffix] = metrics["iou_per_class_fraction_0_1"] * 100.0
            per_variant_sums[suffix] = sums

        rows = []
        for c in range(n_classes):
            row: dict[str, Any] = {
                "class_id": c, "class_name": class_names[c] if class_names is not None else None,
                "is_background": (protocol == "v21" and c == BACKGROUND_CLASS_INDEX),
            }
            for suffix in ("e3", "k11", "k12"):
                valid = per_variant_sums[suffix]["union"][c] > 0
                row[f"{suffix}_iou_percent"] = float(per_variant_iou[suffix][c]) if valid else None
            e3, k11, k12 = row["e3_iou_percent"], row["k11_iou_percent"], row["k12_iou_percent"]
            row["delta_k11_minus_e3_percent"] = (k11 - e3) if (k11 is not None and e3 is not None) else None
            row["delta_k12_minus_e3_percent"] = (k12 - e3) if (k12 is not None and e3 is not None) else None
            row["delta_k11_minus_k12_percent"] = (k11 - k12) if (k11 is not None and k12 is not None) else None
            if k11 is not None and k12 is not None and e3 is not None:
                row["propagation_avg_minus_e3_percent"] = ((k11 + k12) / 2.0) - e3
            else:
                row["propagation_avg_minus_e3_percent"] = None
            rows.append(row)
        out[protocol] = rows
    return out


def broad_based_summary(per_class: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Confirms the propagation-vs-E3 gain is not driven by one class:
    reports the mean gain with and without the single top-gain class
    removed, and the count/list of classes that regress."""
    out: dict[str, Any] = {}
    for protocol, rows in per_class.items():
        gains = np.array([r["propagation_avg_minus_e3_percent"] for r in rows if r["propagation_avg_minus_e3_percent"] is not None], dtype=np.float64)
        names = [r["class_name"] for r in rows if r["propagation_avg_minus_e3_percent"] is not None]
        if gains.size == 0:
            out[protocol] = {"error": "no valid classes"}
            continue
        top_i = int(np.argmax(gains))
        min_i = int(np.argmin(gains))
        regressing = [names[i] if names[i] is not None else i for i in range(len(gains)) if gains[i] <= 0]
        out[protocol] = {
            "n_classes": int(gains.size),
            "mean_gain_percentage_points": float(np.mean(gains)),
            "std_gain_percentage_points": float(np.std(gains)),
            "mean_gain_excluding_top_class_percentage_points": float(np.mean(np.delete(gains, top_i))),
            "top_gain_class": names[top_i], "top_gain_percentage_points": float(gains[top_i]),
            "min_gain_class": names[min_i], "min_gain_percentage_points": float(gains[min_i]),
            "classes_with_nonpositive_gain": regressing,
            "classes_with_positive_gain_count": int(np.sum(gains > 0)),
            "improvement_broad_based": bool(np.mean(np.delete(gains, top_i)) > 0 and np.sum(gains > 0) >= (gains.size // 2)),
        }
        if protocol == "v21":
            bg_row = next(r for r in per_class["v21"] if r["is_background"])
            out[protocol]["background_gain_percentage_points"] = bg_row["propagation_avg_minus_e3_percent"]
            sorted_gains = sorted(gains)
            out[protocol]["background_gain_rank_from_smallest"] = (
                sorted_gains.index(bg_row["propagation_avg_minus_e3_percent"]) if bg_row["propagation_avg_minus_e3_percent"] in sorted_gains else None
            )
    return out


def run_all_bootstraps(bundle: dict[str, Any], reconciled: dict[str, Any], *, n_replicates: int, seed: int, chunk_size: int) -> dict[str, Any]:
    arrays = bundle["arrays"]
    results = {}
    for label, a, b, _protocol in ALL_COMPARISONS:
        observed_delta = reconciled[a]["metrics"]["mIoU_percent_0_100"] - reconciled[b]["metrics"]["mIoU_percent_0_100"]
        bootstrap = bootstrap_paired_delta(
            arrays[f"intersect_{a}"], arrays[f"union_{a}"], arrays[f"intersect_{b}"], arrays[f"union_{b}"],
            observed_delta_percentage_points=observed_delta, n_replicates=n_replicates, seed=seed, chunk_size=chunk_size,
        )
        results[label] = bootstrap
    return results


def background_class_delta(bundle: dict[str, Any]) -> dict[str, Any]:
    """V21 only -- V20 has no background channel."""
    iou_by_suffix: dict[str, float | None] = {}
    for suffix in ("e3", "k11", "k12"):
        metrics, sums = _metrics_for_slice(bundle["arrays"], variant=f"v21_{suffix}")
        valid = sums["union"][BACKGROUND_CLASS_INDEX] > 0
        iou_by_suffix[suffix] = float(metrics["iou_per_class_fraction_0_1"][BACKGROUND_CLASS_INDEX] * 100.0) if valid else None
    out: dict[str, Any] = {f"{s}_background_iou_percent": iou_by_suffix[s] for s in ("e3", "k11", "k12")}
    for label, a, b in (("k11_minus_e3", "k11", "e3"), ("k12_minus_e3", "k12", "e3"), ("k11_minus_k12", "k11", "k12")):
        va, vb = iou_by_suffix[a], iou_by_suffix[b]
        out[f"delta_{label}_background_iou_percentage_points"] = (va - vb) if (va is not None and vb is not None) else None
    return out


def build_report(*, bundle: dict[str, Any], reconciled: dict[str, Any], bootstraps: dict[str, Any], background_delta: dict[str, Any], per_class: dict[str, list[dict[str, Any]]], broad_based: dict[str, Any], prefix_checks: dict[str, Any], v20_class_names: list[str] | None) -> dict[str, Any]:
    result = bundle["result"]
    comparisons_report = {}
    for label, a, b, protocol in ALL_COMPARISONS:
        bootstrap = bootstraps[label]
        comparisons_report[label] = {
            "protocol": protocol, "variant_a": a, "variant_b": b,
            "observed_delta_percentage_points_full_precision": bootstrap.observed_delta_percentage_points,
            "bootstrap_mean_delta_percentage_points": bootstrap.bootstrap_mean_delta,
            "bootstrap_standard_error_percentage_points": bootstrap.bootstrap_standard_error,
            "ci_95_percentile_low": bootstrap.ci_low_2_5, "ci_95_percentile_high": bootstrap.ci_high_97_5,
            "probability_delta_gt_0": bootstrap.probability_delta_gt_0, "probability_delta_lt_0": bootstrap.probability_delta_lt_0,
            "classification": bootstrap.classification,
            "replicate_count": bootstrap.replicate_count, "seed": bootstrap.seed, "chunk_size": bootstrap.chunk_size, "ci_method": bootstrap.ci_method,
        }

    primary_summary = {}
    for protocol in PROTOCOLS:
        k11_e3 = comparisons_report[f"{protocol}_k11_minus_e3"]
        k12_e3 = comparisons_report[f"{protocol}_k12_minus_e3"]
        both_above_zero = k11_e3["classification"] == "CI_ABOVE_ZERO" and k12_e3["classification"] == "CI_ABOVE_ZERO"
        primary_summary[protocol] = {
            "k11_minus_e3": {"observed": k11_e3["observed_delta_percentage_points_full_precision"], "ci": [k11_e3["ci_95_percentile_low"], k11_e3["ci_95_percentile_high"]], "classification": k11_e3["classification"]},
            "k12_minus_e3": {"observed": k12_e3["observed_delta_percentage_points_full_precision"], "ci": [k12_e3["ci_95_percentile_low"], k12_e3["ci_95_percentile_high"]], "classification": k12_e3["classification"]},
            "propagation_significantly_beats_e3": both_above_zero,
            "interpretation": (
                "graph diffusion (both k11 and k12) significantly outperforms the raw E3 unary snapshot"
                if both_above_zero else
                "propagation vs E3 does not show a clear significant advantage in both connectivity variants"
            ),
        }

    secondary_summary = {}
    for protocol in PROTOCOLS:
        cmp = comparisons_report[f"{protocol}_k11_minus_k12"]
        secondary_summary[protocol] = {
            "observed": cmp["observed_delta_percentage_points_full_precision"], "ci": [cmp["ci_95_percentile_low"], cmp["ci_95_percentile_high"]],
            "classification": cmp["classification"],
            "interpretation": {
                "CI_BELOW_ZERO": "k12 (12th edge) significantly outperforms k11 -- the specific connectivity choice matters, favoring k12",
                "CI_ABOVE_ZERO": "k11 significantly outperforms k12 -- the specific connectivity choice matters, favoring k11",
                "CI_INCLUDES_ZERO": "no significant difference between k11 and k12 -- low sensitivity to this connectivity choice",
            }[cmp["classification"]],
        }

    return {
        "schema": SCHEMA_NAME, "tool_version": TOOL_VERSION,
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_result_identity": result["identity"], "source_result_identity_sha256": result["identity_sha256"],
        "source_git_commit": result["git_commit"], "run_mode": result["run_mode"],
        "image_count": result["image_count_processed"], "v20_class_count": result["v20_class_count"], "v21_class_count": result["v21_class_count"],
        "class_names_source": "PascalVOCDataset20.CLASSES (AST-discovered); V21 = ('background',) + V20 (verified live invariant)" if v20_class_names is not None else None,
        "reported_metrics": {v: result[f"metrics_{v}"] for v in VARIANT_NAMES},
        "reconstructed_metrics_from_npz": {v: {"mIoU_percent": reconciled[v]["metrics"]["mIoU_percent_0_100"], "aAcc_percent": reconciled[v]["metrics"]["aAcc_percent_0_100"], "mAcc_percent": reconciled[v]["metrics"]["mAcc_percent_0_100"]} for v in VARIANT_NAMES},
        "reconciliation_tolerance_percentage_points": METRIC_RECONSTRUCTION_TOLERANCE_PERCENT,
        "reported_deltas_full_precision": {
            "v20_k11_minus_e3": result["delta_mIoU_v20_k11_minus_e3_percentage_points"],
            "v20_k12_minus_e3": result["delta_mIoU_v20_k12_minus_e3_percentage_points"],
            "v20_k11_minus_k12": result["delta_mIoU_v20_k11_minus_k12_percentage_points"],
            "v21_k11_minus_e3": result["delta_mIoU_v21_k11_minus_e3_percentage_points"],
            "v21_k12_minus_e3": result["delta_mIoU_v21_k12_minus_e3_percentage_points"],
            "v21_k11_minus_k12": result["delta_mIoU_v21_k11_minus_k12_percentage_points"],
        },
        "bootstrap": {
            "method": "paired image resampling with replacement, 10000+ replicates, drawn from RAW full-precision per-image integer sufficient statistics (intersect/union pixel counts) in the per-image-stats NPZ -- never from the rounded percent_0_100 values in result.json; per-replicate weight vector shared across both compared variants (paired sampling); dataset-level mIoU recomputed from resampled class-sum intersections/unions, never from averaged per-image mIoUs",
            "comparisons": comparisons_report,
        },
        "primary_result_propagation_vs_e3": primary_summary,
        "secondary_result_connectivity_sensitivity_k11_vs_k12": secondary_summary,
        "background_class_delta_v21": background_delta,
        "per_class_gains_losses": per_class,
        "broad_based_improvement_check": broad_based,
        "pilot_prefix_reproduction": prefix_checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline paired-image bootstrap analysis for a completed VOC2012 V20/V21 matched-evaluator full evaluation.")
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
        raise Voc2012FullResultAnalysisError(f"manifest {manifest_path} has no npz_filename and no explicit NPZ path was given")
    return manifest_path.parent / npz_filename


def _run(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.overwrite:
        raise Voc2012FullResultAnalysisError(f"refusing to overwrite existing report at {args.output} without --overwrite")

    full_bundle = load_artifact_bundle(
        result_path=args.result, checkpoint_path=args.checkpoint, manifest_path=args.per_image_stats,
        npz_path=_resolve_npz(args.per_image_stats, args.per_image_stats_npz),
    )
    if full_bundle["result"]["run_mode"] != REQUIRED_RUN_MODE:
        raise Voc2012FullResultAnalysisError(f"--result run_mode is {full_bundle['result']['run_mode']!r}, expected {REQUIRED_RUN_MODE!r}")
    if full_bundle["result"]["image_count_processed"] != REQUIRED_IMAGE_COUNT:
        raise Voc2012FullResultAnalysisError(f"--result image_count_processed is {full_bundle['result']['image_count_processed']}, expected {REQUIRED_IMAGE_COUNT}")

    pilot20_bundle = load_artifact_bundle(
        result_path=args.pilot20_result, checkpoint_path=args.pilot20_checkpoint, manifest_path=args.pilot20_per_image_stats,
        npz_path=_resolve_npz(args.pilot20_per_image_stats, args.pilot20_per_image_stats_npz),
    )
    pilot100_bundle = load_artifact_bundle(
        result_path=args.pilot100_result, checkpoint_path=args.pilot100_checkpoint, manifest_path=args.pilot100_per_image_stats,
        npz_path=_resolve_npz(args.pilot100_per_image_stats, args.pilot100_per_image_stats_npz),
    )

    reconciled = reconcile_full_metrics(full_bundle)
    bootstraps = run_all_bootstraps(full_bundle, reconciled, n_replicates=args.bootstrap_replicates, seed=args.bootstrap_seed, chunk_size=args.bootstrap_chunk_size)
    v20_class_names = _v20_class_names(args.repo_root)
    per_class = per_class_gains_losses(full_bundle, v20_class_names)
    broad_based = broad_based_summary(per_class)
    background_delta = background_class_delta(full_bundle)
    prefix_checks = {
        "pilot20": reproduce_pilot_prefix(full_bundle=full_bundle, pilot_bundle=pilot20_bundle, prefix_length=20, run_mode="pilot20"),
        "pilot100": reproduce_pilot_prefix(full_bundle=full_bundle, pilot_bundle=pilot100_bundle, prefix_length=100, run_mode="pilot100"),
    }
    for name, check in prefix_checks.items():
        if not check["reproduces_exactly"]:
            raise Voc2012FullResultAnalysisError(f"{name} prefix reproduction failed: {check['mismatches']}")

    report = build_report(
        bundle=full_bundle, reconciled=reconciled, bootstraps=bootstraps, background_delta=background_delta,
        per_class=per_class, broad_based=broad_based, prefix_checks=prefix_checks, v20_class_names=v20_class_names,
    )

    temp_path = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, args.output)

    v20_bb = report["broad_based_improvement_check"]["v20"]
    v21_bb = report["broad_based_improvement_check"]["v21"]
    print(
        f"VOC2012 FULL RESULT ANALYSIS PASS "
        f"v20_propagation_vs_e3_broad_based={v20_bb['improvement_broad_based']} "
        f"v21_propagation_vs_e3_broad_based={v21_bb['improvement_broad_based']} "
        f"v20_k11_vs_k12={report['secondary_result_connectivity_sensitivity_k11_vs_k12']['v20']['classification']} "
        f"v21_k11_vs_k12={report['secondary_result_connectivity_sensitivity_k11_vs_k12']['v21']['classification']} "
        f"pilot20_prefix_ok={prefix_checks['pilot20']['reproduces_exactly']} "
        f"pilot100_prefix_ok={prefix_checks['pilot100']['reproduces_exactly']} "
        f"-> {args.output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (Voc2012FullResultAnalysisError, K11K12AnalysisError, K11K12PowerEvaluationError) as error:
        print(f"VOC2012 FULL RESULT ANALYSIS FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
