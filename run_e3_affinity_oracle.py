#!/usr/bin/env python3
"""Streaming offline runner for the E3 affinity-spread oracle."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path

import torch

from src.e3_affinity_oracle import (
    ALPHA_GRID,
    EXPERIMENT,
    PATCH_STATISTICS,
    RESULT_FORMAT,
    FEATURE_CAPTURE_VERIFY_FORMAT,
    AffinityOracleError,
    assert_feature_capture_anchors,
    atomic_json,
    cache_summary,
    class_propagated_score_std,
    collect_patch_statistic_values,
    coordinate_ascent_bias_fit,
    coordinate_ascent_bucket_fit,
    cross_validated_ridge,
    decision_from_transfer,
    dominant_classes_per_bucket,
    evaluate_cache,
    evaluate_cache_biased,
    evaluate_cache_bucketed,
    evaluate_preloaded,
    evaluate_with_rebuilt_graph,
    fit_bucket_edges,
    fitting_support,
    global_sweep,
    greedy_fit,
    load_cache_manifest,
    load_capture_manifest,
    ordered_fingerprint,
    peak_cpu_ram_bytes,
    preload_cache_images,
    sha256_file,
    spearman_correlation,
    split_balanced_halves,
    support_for_indices,
    verify_feature_capture,
    write_rows_csv,
)


RESULT_PAYLOAD_KEYS = {
    "baseline": {
        "label", "alpha", "metrics", "canonical_e3",
        "canonical_reported_precision_matches",
    },
    "global-sweep": {
        "rows", "alpha_global_star", "best_metrics", "delta_from_e3",
    },
    "greedy-fit": {
        "label", "subset_seed", "subset_size", "subset_indices",
        "subset_image_ids", "subset_fingerprint", "global_alpha",
        "image_support", "pixel_support", "per_class_fits",
        "alpha_by_class", "coupling", "other_classes",
        "class_order",
        "runtime_seconds", "peak_cpu_ram_bytes", "peak_gpu_bytes",
    },
    "joint-eval": {
        "label", "warning", "baseline_metrics", "global_metrics", "metrics",
        "alpha_by_class", "delta_from_e3", "delta_from_global",
        "per_class_delta", "alpha_histogram", "alpha_zero_count",
        "alpha_zero_fraction", "alpha_high_count", "alpha_high_fraction",
        "alpha_low_count", "alpha_low_fraction", "largest_20_gains",
        "largest_20_losses",
    },
    "split-half-global": {
        "warning", "seed", "fit_half", "evaluation_half", "A_indices",
        "B_indices", "A_image_ids", "B_image_ids", "A_fingerprint",
        "B_fingerprint", "class_image_support", "class_pixel_support",
        "A_class_image_support", "A_class_pixel_support",
        "B_class_image_support", "B_class_pixel_support",
        "rare_split_limitations",
        "alpha_global_A", "A_global_sweep", "runtime_seconds",
        "peak_cpu_ram_bytes", "peak_gpu_bytes",
    },
    "split-half-greedy": {
        "warning", "seed", "A_fingerprint", "alpha_global_A",
        "alpha_by_class", "A_fitting_diagnostics", "runtime_seconds",
        "peak_cpu_ram_bytes", "peak_gpu_bytes",
    },
    "split-half": {
        "warning", "seed", "fit_half", "evaluation_half", "A_indices",
        "B_indices", "A_image_ids", "B_image_ids", "A_fingerprint",
        "B_fingerprint", "class_image_support", "class_pixel_support",
        "A_class_image_support", "A_class_pixel_support",
        "B_class_image_support", "B_class_pixel_support",
        "rare_split_limitations",
        "alpha_global_A", "A_global_sweep", "alpha_by_class",
        "A_fitting_diagnostics", "baseline_B", "global_B", "transferred_B",
        "transfer_delta_mIoU", "oracle_overfit_gap",
        "global_fit_invocation", "greedy_fit_invocation", "phase_resources",
    },
}

BASE_METRIC_KEYS = {
    "aAcc", "mIoU", "mAcc", "per_class_iou", "per_class_accuracy",
    "intersection", "union", "predicted_pixels", "ground_truth_pixels",
}
EVALUATION_METRIC_KEYS = BASE_METRIC_KEYS | {
    "evaluated_images", "runtime_seconds", "peak_cpu_ram_bytes",
    "peak_gpu_bytes",
}
GLOBAL_ROW_KEYS = EVALUATION_METRIC_KEYS | {
    "alpha", "delta_aAcc", "delta_mIoU", "delta_mAcc",
}
FIT_ROW_KEYS = {
    "class_index", "class_name", "thing_stuff",
    "positive_fitting_image_count", "gt_pixel_count", "alpha_global_star",
    "candidates", "selected_alpha", "class_iou_before_selection",
    "fitted_class_iou", "apparent_gain", "tie_break_reason", "fit_status",
}
GAIN_ROW_KEYS = {
    "class_index", "class_name", "baseline_iou", "joint_iou", "delta",
}
SPLIT_LIMITATION_KEYS = {
    "class_index", "class_name", "A_image_support", "B_image_support",
    "limitation",
}

# Part A: alpha/steps extended sweep. This is a new result schema (v2), not
# a variant of the closed v1 RESULT_PAYLOAD_KEYS commands above, so it is
# validated and provenanced independently of _result()/_load_result().
GLOBAL_SWEEP_EXT_FORMAT = "talk2dino-e3-affinity-oracle-results-v2"
GLOBAL_SWEEP_EXT_ANCHOR_TOLERANCE = 1e-6
GLOBAL_SWEEP_EXT_ANCHORS = {
    "alpha=0.00,T=10 mIoU": 28.480169315747716,
    "alpha=0.95,T=10 mIoU": 29.483747102672197,
    "alpha=0.95,T=10 aAcc": 48.116736004477936,
    "alpha=0.95,T=10 mAcc": 53.7134795125758,
}
GLOBAL_SWEEP_EXT_ROW_KEYS = EVALUATION_METRIC_KEYS | {"alpha", "steps"}
GLOBAL_SWEEP_EXT_OPTIMUM_KEYS = {
    "alpha", "steps", "mIoU", "aAcc", "mAcc",
    "delta_mIoU_from_canonical_alpha_0.95_T10", "at_grid_edge",
}
GLOBAL_SWEEP_EXT_PAYLOAD_KEYS = {
    "label", "warning", "alpha_grid", "steps_grid", "rows", "optimum",
    "kappa_k_baked_at_construction_time", "kappa_k_sweep_limitation",
}
KAPPA_K_SWEEP_LIMITATION = (
    "affinity_power (kappa=3.0) and knn_k (=12) are baked into the cached "
    "knn_weights/knn_indices at cache-construction time inside "
    "AffinityOracleCacheWriter.add_window, which calls build_knn_graph("
    "features, knn_k=protocol.knn_k, affinity_power=protocol.affinity_power) "
    "(src/e3_affinity_oracle.py:773-806 and :232-283). Raw normalized patch "
    "features are never persisted in the cache -- validate_cache_shard "
    "rejects any shard key containing 'feature' -- so kappa/k cannot be "
    "recomputed or swept by replaying this cache; propagate_scores only ever "
    "applies alpha and propagation_steps (T) at replay time and has no "
    "affinity_power/knn_k parameter at all. Sweeping kappa or k requires "
    "rebuilding the cache (~2311s per rebuild) with a different "
    "OracleProtocol(affinity_power=..., knn_k=...) and is out of scope for "
    "this part."
)

# Part B: per-patch-bucket alpha. Shares the v2 result envelope/format with
# Part A (same convention v1 already used: one format_version shared across
# several "command" values).
DEGREE_STAT_LIMITATION = (
    "Every cached knn_weights row is ALREADY row-stochastic (validate_graph "
    "enforces sum(-1) == 1 to 2e-3 as a cache-shard invariant), so a literal "
    "reading of 'degree = sum of the row's own (outgoing) weights' is a "
    "structural constant (~1.0 for every patch) and carries no information "
    "-- it is not a bug in this run, it is a property of every cache this "
    "code can ever read. The pre-normalization row magnitude (row_sums in "
    "build_knn_graph, src/e3_affinity_oracle.py:269) is discarded before "
    "caching and is not one of SHARD_KEYS, so it cannot be recovered by "
    "replaying the cache either. 'degree' is therefore computed as the "
    "graph-standard alternative that IS fully recoverable from the cached "
    "knn_indices/knn_weights alone: weighted in-degree, i.e. the total "
    "incoming edge weight a patch receives from the rest of its window's "
    "directed knn graph (patch_weighted_in_degree in src/e3_affinity_oracle."
    "py). This is a genuine, non-degenerate local-structure statistic."
)
LOCAL_STAT_FIT_PAYLOAD_KEYS = {
    "label", "stat", "stat_definition_note", "n_buckets", "seed",
    "global_alpha", "global_propagation_steps", "bucket_edges",
    "bucket_alphas", "bucket_order", "patches_per_bucket",
    "dominant_classes_per_bucket", "mean_baseline_iou_of_dominant_classes",
    "bucket_index_vs_alpha_spearman", "alpha_grid", "trace", "sweeps_run",
    "converged", "final_metrics", "part_a_optimum_mIoU",
    "delta_mIoU_from_part_a_optimum", "monotonicity_assertion_passed",
    "final_at_least_part_a_optimum_assertion_passed",
}
SPLIT_HALF_LOCAL_PAYLOAD_KEYS = {
    "label", "warning", "stat", "stat_definition_note", "n_buckets", "seed",
    "A_fingerprint", "B_fingerprint", "global_alpha",
    "global_propagation_steps", "bucket_edges", "bucket_alphas",
    "bucket_order", "patches_per_bucket", "dominant_classes_per_bucket",
    "mean_baseline_iou_of_dominant_classes_A",
    "bucket_index_vs_alpha_spearman", "alpha_grid", "trace", "sweeps_run",
    "converged", "A_final_metrics", "part_a_optimum_mIoU",
    "monotonicity_assertion_passed",
    "final_at_least_part_a_optimum_assertion_passed",
    "baseline_B", "global_B", "local_B", "transfer_delta_mIoU",
    "gain_over_global_B",
}

# Part C: per-class additive post-propagation bias. DIAGNOSTIC ONLY -- see
# DIAGNOSTIC_ONLY_NOTE. Never applied to any reported mIoU/aAcc/mAcc outside
# these two artifacts.
DIAGNOSTIC_ONLY_NOTE = (
    "This fitted per-class additive bias beta_c is a DIAGNOSTIC measurement "
    "of cross-class competition headroom ONLY. This project forbids learned "
    "per-COCO-class embeddings and class-name-specific tuning; beta_c must "
    "NEVER be folded into any reported mIoU/aAcc/mAcc, and no downstream "
    "pipeline in this codebase applies it as a correction. Its only purpose "
    "is to measure whether it is predictable from the CLIP text embedding: "
    "only a text-conditional function would generalise to an unseen "
    "vocabulary, beta_c itself does not."
)
# Calibrated empirically against a real split-half-bias fit (171 classes,
# 768-d CLIP text features, ~154 training rows/fold): alpha=10 left the
# shuffled-target control at R^2=-1.53 (should be ~0, i.e. no leak but also
# no spurious fit), because with p=768 >> n~154 an under-regularized ridge
# overfits every training fold, real signal or not. Sweeping alpha on that
# same real data: 10->-1.53, 100->-0.64, 1e3->-0.15, 1e4->-0.035 (first
# value inside the +-0.05 tolerance), 1e5->-0.015, 1e6->-0.013 (by then also
# flattening the real-signal R^2 toward 0, i.e. over-regularized). 1e4 is
# the smallest alpha that clears the shuffled-control bar.
RIDGE_ALPHA = 10000.0
TEXT_PREDICT_KFOLD = 10
BIAS_FIT_PAYLOAD_KEYS = {
    "label", "diagnostic_only", "diagnostic_only_note", "bias_grid_multipliers",
    "per_class_score_std", "global_alpha", "global_propagation_steps",
    "beta_by_class", "class_visit_order", "trace", "sweeps_run", "converged",
    "final_metrics", "part_a_optimum_mIoU", "delta_mIoU_from_part_a_optimum",
    "monotonicity_assertion_passed", "final_at_least_part_a_optimum_assertion_passed",
}
CV_RESULT_KEYS = {"r2", "spearman", "predictions", "folds"}
SPLIT_HALF_BIAS_PAYLOAD_KEYS = {
    "label", "warning", "diagnostic_only", "diagnostic_only_note",
    "bias_grid_multipliers", "per_class_score_std_A", "seed",
    "A_fingerprint", "B_fingerprint", "global_alpha", "global_propagation_steps",
    "beta_by_class", "class_visit_order", "trace", "sweeps_run", "converged",
    "A_final_metrics", "part_a_optimum_mIoU",
    "monotonicity_assertion_passed",
    "final_at_least_part_a_optimum_assertion_passed",
    "baseline_B", "global_B", "bias_B", "transfer_delta_mIoU",
    "gain_over_global_B",
    "text_embedding_identity", "text_embedding_dimension",
    "regression_class_count", "excluded_classes_no_A_support",
    "ridge_alpha", "kfold_k", "shuffle_seed",
    "text_cv", "pixel_frequency_cv", "image_frequency_cv",
    "baseline_iou_cv", "text_plus_trivial_cv", "shuffled_target_control_cv",
    "shuffled_control_near_zero", "decision",
}


def _decision_from_bias(gain_over_global_b: float, held_out_text_r2: float) -> tuple[str, str]:
    if gain_over_global_b > 0.50 and held_out_text_r2 > 0.30:
        return (
            "BUILD_TEXT_CONDITIONAL_CALIBRATION_HEAD",
            "gain_over_global_B > 0.50 and held-out text R^2 > 0.30",
        )
    if gain_over_global_b > 0.50:
        return (
            "HEADROOM_NOT_TEXT_REACHABLE",
            "gain_over_global_B > 0.50 but held-out text R^2 <= 0.30 -- real "
            "headroom exists but is not reachable from text; do not build "
            "a text-conditional head",
        )
    return (
        "CALIBRATION_REJECTED",
        "gain_over_global_B <= 0.50",
    )


def _closed_nested(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise AffinityOracleError(
            f"closed {label} mismatch: expected={sorted(expected)}, got={actual}"
        )
    return value


def _validate_metric(value: object, expected: set[str], label: str) -> None:
    metric = _closed_nested(value, expected, label)
    for key in (
        "per_class_iou", "per_class_accuracy", "intersection", "union",
        "predicted_pixels", "ground_truth_pixels",
    ):
        if not isinstance(metric[key], list) or len(metric[key]) != 171:
            raise AffinityOracleError(f"{label}.{key} must contain 171 values")
    for key in ("per_class_iou", "per_class_accuracy"):
        if any(
            item is not None
            and (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
            )
            for item in metric[key]
        ):
            raise AffinityOracleError(f"{label}.{key} contains a nonnumeric value")
    for key in ("intersection", "union", "predicted_pixels", "ground_truth_pixels"):
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in metric[key]
        ):
            raise AffinityOracleError(f"{label}.{key} must contain counts")


def _validate_global_rows(rows: object, label: str) -> None:
    if not isinstance(rows, list) or len(rows) != len(ALPHA_GRID):
        raise AffinityOracleError(f"{label} must contain the fixed alpha grid")
    for index, row in enumerate(rows):
        _validate_metric(row, GLOBAL_ROW_KEYS, f"{label}[{index}]")


def _validate_fits(rows: object, label: str) -> None:
    if not isinstance(rows, list) or len(rows) != 171:
        raise AffinityOracleError(f"{label} must contain 171 class fits")
    for index, value in enumerate(rows):
        row = _closed_nested(value, FIT_ROW_KEYS, f"{label}[{index}]")
        if row["class_index"] != index:
            raise AffinityOracleError(f"{label} class order changed")
        candidates = row["candidates"]
        if not isinstance(candidates, list) or len(candidates) != len(ALPHA_GRID):
            raise AffinityOracleError(f"{label}[{index}] candidate grid changed")
        for candidate_index, candidate in enumerate(candidates):
            candidate = _closed_nested(
                candidate, {"alpha", "class_iou"},
                f"{label}[{index}].candidates[{candidate_index}]",
            )
            if candidate["alpha"] != ALPHA_GRID[candidate_index]:
                raise AffinityOracleError(f"{label}[{index}] alpha order changed")


def _validate_payload(command: str, payload: object) -> None:
    from src.e3_affinity_oracle import _require_finite_tree
    expected = RESULT_PAYLOAD_KEYS.get(command)
    if expected is None or not isinstance(payload, dict) or set(payload) != expected:
        actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise AffinityOracleError(
            f"closed {command} payload mismatch: expected={sorted(expected or ())}, "
            f"got={actual}"
        )
    _require_finite_tree(payload, f"{command} result")
    if command == "baseline":
        _validate_metric(payload["metrics"], EVALUATION_METRIC_KEYS, "baseline metrics")
        _closed_nested(
            payload["canonical_e3"], {"aAcc", "mIoU", "mAcc"},
            "canonical E3 metrics",
        )
    elif command == "global-sweep":
        _validate_global_rows(payload["rows"], "global sweep rows")
        _validate_metric(payload["best_metrics"], GLOBAL_ROW_KEYS, "best metrics")
        _closed_nested(
            payload["delta_from_e3"], {"aAcc", "mIoU", "mAcc"},
            "global delta",
        )
    elif command == "greedy-fit":
        _validate_fits(payload["per_class_fits"], "greedy fits")
        if len(payload["alpha_by_class"]) != 171 or len(payload["class_order"]) != 171:
            raise AffinityOracleError("greedy class vectors must contain 171 values")
    elif command == "joint-eval":
        _validate_metric(payload["baseline_metrics"], EVALUATION_METRIC_KEYS, "joint baseline")
        _validate_metric(payload["global_metrics"], GLOBAL_ROW_KEYS, "joint global")
        _validate_metric(payload["metrics"], EVALUATION_METRIC_KEYS, "joint metrics")
        for label in ("largest_20_gains", "largest_20_losses"):
            rows = payload[label]
            if not isinstance(rows, list) or len(rows) > 20:
                raise AffinityOracleError(f"{label} must contain at most 20 rows")
            for index, row in enumerate(rows):
                _closed_nested(row, GAIN_ROW_KEYS, f"{label}[{index}]")
    elif command == "split-half-global":
        _validate_global_rows(payload["A_global_sweep"], "half-A global rows")
        for index, row in enumerate(payload["rare_split_limitations"]):
            _closed_nested(row, SPLIT_LIMITATION_KEYS, f"split limitation[{index}]")
    elif command == "split-half-greedy":
        _validate_fits(payload["A_fitting_diagnostics"], "half-A fits")
    elif command == "split-half":
        _validate_global_rows(payload["A_global_sweep"], "frozen half-A global rows")
        _validate_fits(payload["A_fitting_diagnostics"], "frozen half-A fits")
        for label in ("baseline_B", "global_B", "transferred_B"):
            _validate_metric(payload[label], EVALUATION_METRIC_KEYS, label)
        for index, row in enumerate(payload["rare_split_limitations"]):
            _closed_nested(row, SPLIT_LIMITATION_KEYS, f"split limitation[{index}]")
        _closed_nested(
            payload["phase_resources"],
            {
                "global_A_seconds", "greedy_A_seconds", "transfer_B_seconds",
                "peak_cpu_ram_bytes", "peak_gpu_bytes",
            },
            "split phase resources",
        )


def _result(cache: Path, command: str, payload: dict) -> dict:
    manifest = load_cache_manifest(cache, verify_shards=False)
    _validate_payload(command, payload)
    return {
        "format_version": RESULT_FORMAT,
        "experiment": EXPERIMENT,
        "command": command,
        "invocation": " ".join(shlex.quote(argument) for argument in sys.argv),
        "cache_manifest_sha256": __import__(
            "src.e3_affinity_oracle", fromlist=["sha256_file"]
        ).sha256_file(cache / "manifest.json"),
        "class_order_sha256": manifest["class_order_sha256"],
        "dataset_config_sha256": manifest["dataset_config_sha256"],
        "payload": payload,
    }


def _load_result(path: Path, command: str) -> dict:
    value = json.loads(path.read_text())
    required = {
        "format_version", "experiment", "command", "invocation",
        "cache_manifest_sha256",
        "class_order_sha256", "dataset_config_sha256", "payload",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise AffinityOracleError(f"closed result schema mismatch: {path}")
    if value["format_version"] != RESULT_FORMAT or value["command"] != command:
        raise AffinityOracleError(f"result identity mismatch: {path}")
    if value["experiment"] != EXPERIMENT:
        raise AffinityOracleError(f"result experiment mismatch: {path}")
    _validate_payload(command, value["payload"])
    return value


def _compatible(cache: Path, result: dict, identity=None):
    from src.e3_affinity_oracle import sha256_file
    if identity is None:
        manifest = load_cache_manifest(cache)
        identity = (manifest, sha256_file(cache / "manifest.json"))
    else:
        manifest = identity[0]
    if (
        result["cache_manifest_sha256"] != identity[1]
        or result["class_order_sha256"] != manifest["class_order_sha256"]
        or result["dataset_config_sha256"] != manifest["dataset_config_sha256"]
    ):
        raise AffinityOracleError("cache/result identities are incompatible")
    return identity


def baseline(args: argparse.Namespace) -> None:
    metrics = evaluate_cache(args.cache, 0.0, device=args.device)
    payload = {
        "label": "alpha_zero_control", "alpha": 0.0, "metrics": metrics,
        "canonical_e3": {"aAcc": 46.61, "mIoU": 28.48, "mAcc": 52.08},
        "canonical_reported_precision_matches": (
            round(metrics["aAcc"], 2) == 46.61
            and round(metrics["mIoU"], 2) == 28.48
            and round(metrics["mAcc"], 2) == 52.08
        ),
    }
    atomic_json(args.output, _result(args.cache, "baseline", payload), overwrite=args.overwrite)


def sweep(args: argparse.Namespace) -> None:
    cache_manifest = load_cache_manifest(args.cache, verify_shards=False)
    if (
        cache_manifest["score_dtype"] == "float16"
        and not cache_manifest["fp16_suitable"]
    ):
        raise AffinityOracleError(
            "FP16 pilot control failed; rebuild with score_dtype=float32"
        )
    baseline_result = _load_result(args.baseline, "baseline")
    _compatible(args.cache, baseline_result)
    if not baseline_result["payload"]["canonical_reported_precision_matches"]:
        raise AffinityOracleError("alpha=0 does not reproduce canonical E3; stopping sweep")
    rows, best = global_sweep(args.cache, device=args.device)
    best_metrics = next(row for row in rows if row["alpha"] == best)
    payload = {
        "rows": rows,
        "alpha_global_star": best,
        "best_metrics": best_metrics,
        "delta_from_e3": {
            "aAcc": best_metrics["delta_aAcc"],
            "mIoU": best_metrics["delta_mIoU"],
            "mAcc": best_metrics["delta_mAcc"],
        },
    }
    atomic_json(args.output, _result(args.cache, "global-sweep", payload), overwrite=args.overwrite)
    csv_rows = [
        {key: row[key] for key in (
            "alpha", "aAcc", "mIoU", "mAcc", "delta_aAcc", "delta_mIoU",
            "delta_mAcc", "runtime_seconds", "peak_cpu_ram_bytes", "peak_gpu_bytes",
        )}
        for row in rows
    ]
    write_rows_csv(args.csv, csv_rows, overwrite=args.overwrite)


def fit(args: argparse.Namespace) -> None:
    started = time.monotonic()
    sweep_result = _load_result(args.global_sweep, "global-sweep")
    _compatible(args.cache, sweep_result)
    classes, image_counts, pixel_counts = fitting_support(args.cache)
    subset = set(
        __import__("src.e3_affinity_oracle", fromlist=["coverage_subset"])
        .coverage_subset(classes, min(args.subset_size, len(classes)), seed=42)
    )
    global_alpha = sweep_result["payload"]["alpha_global_star"]
    fits = greedy_fit(
        args.cache, global_alpha=global_alpha, selected_indices=subset,
        device=args.device,
    )
    manifest = load_cache_manifest(args.cache, verify_shards=False)
    by_index = {row["dataset_index"]: row for row in manifest["images"]}
    subset_records = [
        {"dataset_index": index, "image_id": by_index[index]["image_id"]}
        for index in sorted(subset)
    ]
    payload = {
        "label": "one_pass_coupled_greedy_approximation",
        "subset_seed": 42, "subset_size": len(subset),
        "subset_indices": sorted(subset),
        "subset_image_ids": [row["image_id"] for row in subset_records],
        "subset_fingerprint": ordered_fingerprint(subset_records),
        "global_alpha": global_alpha, "image_support": image_counts,
        "pixel_support": pixel_counts, "per_class_fits": fits,
        "alpha_by_class": [row["selected_alpha"] for row in fits],
        "class_order": manifest["class_names"],
        "coupling": "best_other_argmax_competition",
        "other_classes": "held_at_global_optimum",
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": _cuda_peak(args.device),
    }
    atomic_json(args.output, _result(args.cache, "greedy-fit", payload), overwrite=args.overwrite)
    write_rows_csv(args.csv, [{
        "class_index": row["class_index"], "class_name": row["class_name"],
        "thing_stuff": row["thing_stuff"],
        "positive_fitting_image_count": row["positive_fitting_image_count"],
        "gt_pixel_count": row["gt_pixel_count"],
        "alpha_global_star": row["alpha_global_star"],
        "candidates": json.dumps(row["candidates"], separators=(",", ":")),
        "selected_alpha": row["selected_alpha"],
        "class_iou_before_selection": row["class_iou_before_selection"],
        "fitted_class_iou": row["fitted_class_iou"],
        "apparent_gain": row["apparent_gain"],
        "tie_break_reason": row["tie_break_reason"],
        "fit_status": row["fit_status"],
    } for row in fits], overwrite=args.overwrite)


def joint(args: argparse.Namespace) -> None:
    baseline_result = _load_result(args.baseline, "baseline")
    sweep_result = _load_result(args.global_sweep, "global-sweep")
    fit_result = _load_result(args.greedy_fit, "greedy-fit")
    identity = None
    for value in (baseline_result, sweep_result, fit_result):
        identity = _compatible(args.cache, value, identity)
    vector = torch.tensor(fit_result["payload"]["alpha_by_class"], dtype=torch.float32)
    metrics = evaluate_cache(args.cache, vector, device=args.device)
    global_row = next(
        row for row in sweep_result["payload"]["rows"]
        if row["alpha"] == sweep_result["payload"]["alpha_global_star"]
    )
    baseline_metrics = baseline_result["payload"]["metrics"]
    deltas = [
        (new - old) if new is not None and old is not None else None
        for new, old in zip(metrics["per_class_iou"], baseline_metrics["per_class_iou"])
    ]
    manifest = identity[0]
    ranked = sorted(
        (
            {
                "class_index": index,
                "class_name": manifest["class_names"][index],
                "baseline_iou": baseline_metrics["per_class_iou"][index],
                "joint_iou": metrics["per_class_iou"][index],
                "delta": delta,
            }
            for index, delta in enumerate(deltas) if delta is not None
        ),
        key=lambda row: (-row["delta"], row["class_index"]),
    )
    histogram = {str(alpha): int((vector == alpha).sum()) for alpha in ALPHA_GRID}
    payload = {
        "label": "in_sample_full_validation_oracle",
        "warning": "IN-SAMPLE ORACLE — NOT A GENERALIZATION RESULT",
        "baseline_metrics": baseline_metrics, "global_metrics": global_row,
        "metrics": metrics, "alpha_by_class": vector.tolist(),
        "delta_from_e3": metrics["mIoU"] - baseline_metrics["mIoU"],
        "delta_from_global": metrics["mIoU"] - global_row["mIoU"],
        "per_class_delta": deltas,
        "alpha_histogram": histogram,
        "alpha_zero_count": int((vector == 0).sum()),
        "alpha_zero_fraction": float((vector == 0).float().mean()),
        "alpha_high_count": int((vector >= 0.7).sum()),
        "alpha_high_fraction": float((vector >= 0.7).float().mean()),
        "alpha_low_count": int((vector <= 0.2).sum()),
        "alpha_low_fraction": float((vector <= 0.2).float().mean()),
        "largest_20_gains": ranked[:20],
        "largest_20_losses": list(reversed(ranked[-20:])),
    }
    atomic_json(args.output, _result(args.cache, "joint-eval", payload), overwrite=args.overwrite)


def _cuda_peak(device: str) -> int:
    value = torch.device(device)
    return (
        int(torch.cuda.max_memory_allocated(value))
        if value.type == "cuda" and torch.cuda.is_available() else 0
    )


def split_half_global(args: argparse.Namespace) -> None:
    started = time.monotonic()
    classes, image_counts, pixel_counts = fitting_support(args.cache)
    a, b = split_balanced_halves(classes, seed=42)
    a_image_support, a_pixel_support = support_for_indices(args.cache, set(a))
    b_image_support, b_pixel_support = support_for_indices(args.cache, set(b))
    rows_a, alpha_a = global_sweep(args.cache, device=args.device, selected_indices=set(a))
    manifest = load_cache_manifest(args.cache, verify_shards=False)
    by_index = {row["dataset_index"]: row for row in manifest["images"]}
    a_records = [{"dataset_index": index, "image_id": by_index[index]["image_id"]} for index in a]
    b_records = [{"dataset_index": index, "image_id": by_index[index]["image_id"]} for index in b]
    payload = {
        "warning": "SPLIT-HALF TRANSFER — PRIMARY DECISION METRIC",
        "seed": 42, "fit_half": "A", "evaluation_half": "B",
        "A_indices": a, "B_indices": b,
        "A_image_ids": [row["image_id"] for row in a_records],
        "B_image_ids": [row["image_id"] for row in b_records],
        "A_fingerprint": ordered_fingerprint(a_records),
        "B_fingerprint": ordered_fingerprint(b_records),
        "class_image_support": image_counts, "class_pixel_support": pixel_counts,
        "A_class_image_support": a_image_support,
        "A_class_pixel_support": a_pixel_support,
        "B_class_image_support": b_image_support,
        "B_class_pixel_support": b_pixel_support,
        "rare_split_limitations": [
            {
                "class_index": index,
                "class_name": manifest["class_names"][index],
                "A_image_support": a_image_support[index],
                "B_image_support": b_image_support[index],
                "limitation": "class_not_supported_in_both_halves",
            }
            for index in range(manifest["class_count"])
            if a_image_support[index] == 0 or b_image_support[index] == 0
        ],
        "alpha_global_A": alpha_a, "A_global_sweep": rows_a,
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": max(row["peak_gpu_bytes"] for row in rows_a),
    }
    atomic_json(
        args.output, _result(args.cache, "split-half-global", payload),
        overwrite=args.overwrite,
    )


def split_half_greedy(args: argparse.Namespace) -> None:
    if args.split_global is None:
        raise AffinityOracleError("split-half --stage greedy requires --split-global")
    started = time.monotonic()
    global_result = _load_result(args.split_global, "split-half-global")
    _compatible(args.cache, global_result)
    global_payload = global_result["payload"]
    fits = greedy_fit(
        args.cache, global_alpha=global_payload["alpha_global_A"],
        selected_indices=set(global_payload["A_indices"]), device=args.device,
    )
    payload = {
        "warning": "SPLIT-HALF TRANSFER — PRIMARY DECISION METRIC",
        "seed": 42, "A_fingerprint": global_payload["A_fingerprint"],
        "alpha_global_A": global_payload["alpha_global_A"],
        "alpha_by_class": [row["selected_alpha"] for row in fits],
        "A_fitting_diagnostics": fits,
        "runtime_seconds": time.monotonic() - started,
        "peak_cpu_ram_bytes": peak_cpu_ram_bytes(),
        "peak_gpu_bytes": _cuda_peak(args.device),
    }
    atomic_json(
        args.output, _result(args.cache, "split-half-greedy", payload),
        overwrite=args.overwrite,
    )


def split_half_transfer(args: argparse.Namespace) -> None:
    if args.split_global is None or args.split_greedy is None or args.joint is None:
        raise AffinityOracleError(
            "split-half --stage transfer requires --split-global, "
            "--split-greedy and --joint"
        )
    started = time.monotonic()
    global_result = _load_result(args.split_global, "split-half-global")
    greedy_result = _load_result(args.split_greedy, "split-half-greedy")
    joint_result = _load_result(args.joint, "joint-eval")
    identity = None
    for result in (global_result, greedy_result, joint_result):
        identity = _compatible(args.cache, result, identity)
    global_payload = global_result["payload"]
    greedy_payload = greedy_result["payload"]
    if greedy_payload["A_fingerprint"] != global_payload["A_fingerprint"]:
        raise AffinityOracleError("split-half greedy/global split identities differ")
    b = set(global_payload["B_indices"])
    vector = torch.tensor(greedy_payload["alpha_by_class"], dtype=torch.float32)
    baseline_b = evaluate_cache(args.cache, 0.0, device=args.device, selected_indices=b)
    global_b = evaluate_cache(
        args.cache, global_payload["alpha_global_A"], device=args.device,
        selected_indices=b,
    )
    transfer_b = evaluate_cache(args.cache, vector, device=args.device, selected_indices=b)
    transfer = transfer_b["mIoU"] - baseline_b["mIoU"]
    payload = {
        "warning": "SPLIT-HALF TRANSFER — PRIMARY DECISION METRIC",
        "seed": 42, "fit_half": "A", "evaluation_half": "B",
        "A_indices": global_payload["A_indices"],
        "B_indices": global_payload["B_indices"],
        "A_image_ids": global_payload["A_image_ids"],
        "B_image_ids": global_payload["B_image_ids"],
        "A_fingerprint": global_payload["A_fingerprint"],
        "B_fingerprint": global_payload["B_fingerprint"],
        "class_image_support": global_payload["class_image_support"],
        "class_pixel_support": global_payload["class_pixel_support"],
        "A_class_image_support": global_payload["A_class_image_support"],
        "A_class_pixel_support": global_payload["A_class_pixel_support"],
        "B_class_image_support": global_payload["B_class_image_support"],
        "B_class_pixel_support": global_payload["B_class_pixel_support"],
        "rare_split_limitations": global_payload["rare_split_limitations"],
        "alpha_global_A": global_payload["alpha_global_A"],
        "A_global_sweep": global_payload["A_global_sweep"],
        "alpha_by_class": greedy_payload["alpha_by_class"],
        "A_fitting_diagnostics": greedy_payload["A_fitting_diagnostics"],
        "baseline_B": baseline_b, "global_B": global_b,
        "transferred_B": transfer_b, "transfer_delta_mIoU": transfer,
        "oracle_overfit_gap": joint_result["payload"]["delta_from_e3"] - transfer,
        "global_fit_invocation": global_result["invocation"],
        "greedy_fit_invocation": greedy_result["invocation"],
        "phase_resources": {
            "global_A_seconds": global_payload["runtime_seconds"],
            "greedy_A_seconds": greedy_payload["runtime_seconds"],
            "transfer_B_seconds": time.monotonic() - started,
            "peak_cpu_ram_bytes": max(
                global_payload["peak_cpu_ram_bytes"],
                greedy_payload["peak_cpu_ram_bytes"], peak_cpu_ram_bytes(),
            ),
            "peak_gpu_bytes": max(
                global_payload["peak_gpu_bytes"], greedy_payload["peak_gpu_bytes"],
                baseline_b["peak_gpu_bytes"], global_b["peak_gpu_bytes"],
                transfer_b["peak_gpu_bytes"],
            ),
        },
    }
    atomic_json(
        args.output, _result(args.cache, "split-half", payload),
        overwrite=args.overwrite,
    )


def split_half(args: argparse.Namespace) -> None:
    if args.stage == "global":
        split_half_global(args)
    elif args.stage == "greedy":
        split_half_greedy(args)
    elif args.stage == "local":
        split_half_local(args)
    elif args.stage == "bias":
        split_half_bias(args)
    else:
        split_half_transfer(args)


def report(args: argparse.Namespace) -> None:
    baseline_result = _load_result(args.baseline, "baseline")
    sweep_result = _load_result(args.global_sweep, "global-sweep")
    fit_result = _load_result(args.greedy_fit, "greedy-fit")
    joint_result = _load_result(args.joint, "joint-eval")
    split_result = _load_result(args.split_half, "split-half")
    identity = None
    for value in (baseline_result, sweep_result, fit_result, joint_result, split_result):
        identity = _compatible(args.cache, value, identity)
    split_payload = split_result["payload"]
    joint_payload = joint_result["payload"]
    gap = split_payload["oracle_overfit_gap"]
    verdict, reason = decision_from_transfer(split_payload["transfer_delta_mIoU"])
    manifest = identity[0]
    fits = fit_result["payload"]["per_class_fits"]
    transfer_fits = split_payload["A_fitting_diagnostics"]
    baseline_iou = split_payload["baseline_B"]["per_class_iou"]
    transferred_iou = split_payload["transferred_B"]["per_class_iou"]
    image_support = [row["positive_fitting_image_count"] for row in transfer_fits]
    pixel_support = [row["gt_pixel_count"] for row in transfer_fits]
    from src.e3_affinity_oracle import spearman_correlation
    alphas = [row["selected_alpha"] for row in transfer_fits]
    finite_rows = [
        index for index, value in enumerate(baseline_iou) if value is not None
    ]
    def group_summary(indices):
        deltas = [
            transferred_iou[index] - baseline_iou[index]
            for index in indices
            if transferred_iou[index] is not None and baseline_iou[index] is not None
        ]
        bases = [baseline_iou[index] for index in indices if baseline_iou[index] is not None]
        return {
            "classes": [manifest["class_names"][index] for index in indices],
            "count": len(indices),
            "percentage": 100.0 * len(indices) / manifest["class_count"],
            "thing_count": None, "stuff_count": None,
            "mean_baseline_iou": sum(bases) / len(bases) if bases else None,
            "mean_transferred_iou_change": sum(deltas) / len(deltas) if deltas else None,
            "fitting_image_support": sum(image_support[index] for index in indices),
            "fitting_pixel_support": sum(pixel_support[index] for index in indices),
        }
    low_indices = [index for index, alpha in enumerate(alphas) if alpha <= 0.2]
    medium_indices = [index for index, alpha in enumerate(alphas) if 0.2 < alpha < 0.7]
    high_indices = [index for index, alpha in enumerate(alphas) if alpha >= 0.7]
    groups = {
        "low_alpha_classes": [manifest["class_names"][index] for index in low_indices],
        "medium_alpha_classes": [manifest["class_names"][index] for index in medium_indices],
        "high_alpha_classes": [manifest["class_names"][index] for index in high_indices],
        "group_statistics": {
            "low_alpha": group_summary(low_indices),
            "medium_alpha": group_summary(medium_indices),
            "high_alpha": group_summary(high_indices),
        },
        "thing_stuff": None,
        "named_class_checks": {
            name: {
                "selected_alpha": alphas[manifest["class_names"].index(name)],
                "group": (
                    "high" if alphas[manifest["class_names"].index(name)] >= 0.7
                    else "low" if alphas[manifest["class_names"].index(name)] <= 0.2
                    else "medium"
                ),
            }
            for name in ("sky", "grass", "road", "wall", "racket", "fork", "bottle", "tie")
            if name in manifest["class_names"]
        },
        "correlations": {
            "alpha_vs_baseline_iou": spearman_correlation(
                [alphas[i] for i in finite_rows],
                [baseline_iou[i] for i in finite_rows],
            ),
            "alpha_vs_pixel_frequency": spearman_correlation(alphas, pixel_support),
            "alpha_vs_image_frequency": spearman_correlation(alphas, image_support),
            "alpha_vs_average_component_area": None,
        },
    }
    protocol = {
        "raw_score_location": "DINOTextMasker.forward_seg normalized text-patch dot product",
        "score_stage": "pre_sigmoid_pre_upsample_pre_stitch",
        "crop_size": manifest["protocol"]["crop_size"],
        "stride": manifest["protocol"]["stride"],
        "patch_grid": manifest["patch_grid"],
        "knn_k": manifest["knn_k"],
        "affinity_power": manifest["affinity_power"],
        "propagation_steps": manifest["propagation_steps"],
        "propagation_equation": manifest["propagation_equation"],
        "alpha_grid": manifest["alpha_grid"],
        "pamr": manifest["protocol"]["pamr"],
        "background": {
            "with_background": manifest["protocol"]["with_background"],
            "threshold": manifest["protocol"]["background_threshold"],
        },
    }
    summary = {
        "format_version": RESULT_FORMAT, "experiment": EXPERIMENT,
        "status": "complete",
        "identities": {
            "git": {
                "commit": manifest["source_git_commit"],
                "dirty": manifest["source_git_dirty"],
                "diff_sha256": manifest["source_git_diff_sha256"],
            },
            "e3_config": manifest["e3_config_sha256"],
            "e3_checkpoint": manifest["e3_checkpoint_sha256"],
            "dino": {
                "identity": manifest["dino_identity"],
                "checkpoint_sha256": manifest["dino_checkpoint_sha256"],
            },
            "clip_checkpoint": manifest["clip_checkpoint_sha256"],
            "text_embedding": manifest["text_embedding_sha256"],
            "dataset": manifest["dataset_config_sha256"],
            "class_order": manifest["class_order_sha256"],
        },
        "protocol": protocol,
        "cache": {
            "format_version": manifest["format_version"],
            "complete": manifest["complete"],
            "pilot": manifest["pilot"],
            "images": manifest["selected_image_count"],
            "windows": manifest["selected_window_count"],
            "shards": len(manifest["shards"]),
            "bytes": manifest["total_cache_bytes"],
            "fp16_suitable": manifest["fp16_suitable"],
            "zero_neighbour_rows": manifest["zero_neighbour_rows"],
            "path": str(args.cache),
            "dtypes": {
                "scores": manifest["score_dtype"],
                "indices": manifest["graph_index_dtype"],
                "weights": manifest["graph_weight_dtype"],
            },
            "fp16_control": manifest["fp16_control"],
        },
        "baseline": baseline_result["payload"]["metrics"],
        "global_sweep": sweep_result["payload"], "greedy": fit_result["payload"],
        "joint_oracle": joint_payload,
        "split_half": {**split_payload, "oracle_overfit_gap": gap},
        "semantic_analysis": groups,
        "decision": {"source_metric": "split_half.transfer_delta_mIoU", "verdict": verdict, "reason": reason},
        "resources": {
            "cache_construction_seconds": (
                manifest["construction_finished_at"]
                - manifest["construction_started_at"]
            ),
            "sweep_seconds": sum(
                row["runtime_seconds"] for row in sweep_result["payload"]["rows"]
            ),
            "greedy_seconds": fit_result["payload"]["runtime_seconds"],
            "joint_seconds": joint_payload["metrics"]["runtime_seconds"],
            "split_half_seconds": sum(
                split_payload["phase_resources"][key]
                for key in (
                    "global_A_seconds", "greedy_A_seconds", "transfer_B_seconds"
                )
            ),
            "peak_cpu_ram_bytes": max(
                max(
                    row["peak_cpu_ram_bytes"]
                    for row in sweep_result["payload"]["rows"]
                ),
                fit_result["payload"]["peak_cpu_ram_bytes"],
                joint_payload["metrics"]["peak_cpu_ram_bytes"],
                split_payload["phase_resources"]["peak_cpu_ram_bytes"],
            ),
            "peak_gpu_bytes": max(
                max(row["peak_gpu_bytes"] for row in sweep_result["payload"]["rows"]),
                fit_result["payload"]["peak_gpu_bytes"],
                joint_payload["metrics"]["peak_gpu_bytes"],
                split_payload["phase_resources"]["peak_gpu_bytes"],
            ),
        },
        "commands": [
            *manifest["commands"],
            baseline_result["invocation"], sweep_result["invocation"],
            fit_result["invocation"], joint_result["invocation"],
            split_payload["global_fit_invocation"],
            split_payload["greedy_fit_invocation"],
            split_result["invocation"],
            " ".join(shlex.quote(argument) for argument in sys.argv),
        ],
    }
    expected_summary_keys = {
        "format_version", "experiment", "status", "identities", "protocol",
        "cache", "baseline", "global_sweep", "greedy", "joint_oracle",
        "split_half", "semantic_analysis", "decision", "resources", "commands",
    }
    if set(summary) != expected_summary_keys:
        raise AffinityOracleError("closed summary schema mismatch")
    from src.e3_affinity_oracle import _require_finite_tree
    _require_finite_tree(summary, "summary")
    atomic_json(args.output, summary, overwrite=args.overwrite)
    global_rows = sweep_result["payload"]["rows"]
    alpha_table = [
        "| alpha | aAcc | mIoU | mAcc | delta mIoU |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {row['alpha']:.2f} | {row['aAcc']:.3f} | {row['mIoU']:.3f} | "
            f"{row['mAcc']:.3f} | {row['delta_mIoU']:+.3f} |"
            for row in global_rows
        ],
    ]
    class_table = [
        "| index | class | alpha | fit IoU | apparent gain | support images |",
        "|---:|:---|---:|---:|---:|---:|",
        *[
            f"| {row['class_index']} | {row['class_name']} | "
            f"{row['selected_alpha']:.2f} | {row['fitted_class_iou']:.3f} | "
            f"{row['apparent_gain']:+.3f} | {row['positive_fitting_image_count']} |"
            for row in fits
        ],
    ]
    lines = [
        "# E3 Affinity-Spread Oracle", "",
        "**IN-SAMPLE ORACLE — NOT A GENERALIZATION RESULT**", "",
        "**SPLIT-HALF TRANSFER — PRIMARY DECISION METRIC**", "",
        f"## 1. Executive verdict\n\n{verdict}: {reason}.", "",
        "## 2. E3 baseline/control reproduction", "",
        f"aAcc {baseline_result['payload']['metrics']['aAcc']:.4f}, mIoU "
        f"{baseline_result['payload']['metrics']['mIoU']:.4f}, mAcc "
        f"{baseline_result['payload']['metrics']['mAcc']:.4f}.", "",
        "## 3. Cache size and FP16 fidelity", "",
        f"{manifest['total_cache_bytes']} bytes; control: `{json.dumps(manifest['fp16_control'], sort_keys=True)}`.", "",
        "## 4. Global alpha table", "", *alpha_table, "",
        f"## 5. Global optimum\n\nalpha={sweep_result['payload']['alpha_global_star']}.", "",
        "## 6. Complete per-class alpha table", "", *class_table, "",
        "## 7. Exact full joint oracle metrics", "",
        f"mIoU {joint_payload['metrics']['mIoU']:.4f}; delta {joint_payload['delta_from_e3']:+.4f}.", "",
        "## 8. Split-half transfer metrics", "",
        f"B mIoU {split_payload['transferred_B']['mIoU']:.4f}; transfer delta {split_payload['transfer_delta_mIoU']:+.4f}.", "",
        f"## 9. Full-oracle versus transfer overfit gap\n\n{gap:.4f} points.", "",
        "## 10. High-/medium-/low-alpha groups", "",
        f"Low: {', '.join(groups['low_alpha_classes'])}.",
        f"Medium: {', '.join(groups['medium_alpha_classes'])}.",
        f"High: {', '.join(groups['high_alpha_classes'])}.", "",
        *[
            f"{label}: {row['count']} classes ({row['percentage']:.2f}%), "
            f"mean B-baseline IoU {row['mean_baseline_iou']}, mean transferred "
            f"IoU change {row['mean_transferred_iou_change']}, A fitting support "
            f"{row['fitting_image_support']} images / {row['fitting_pixel_support']} pixels."
            for label, row in groups["group_statistics"].items()
        ], "",
        "## 11. Thing/stuff breakdown", "", "Unavailable in the canonical dataset metadata; recorded as null.", "",
        "## 12. Largest gains and losses", "",
        "Largest gains: " + ", ".join(
            f"{row['class_name']} ({row['delta']:+.3f})"
            for row in joint_payload["largest_20_gains"]
        ) + ".",
        "Largest losses: " + ", ".join(
            f"{row['class_name']} ({row['delta']:+.3f})"
            for row in joint_payload["largest_20_losses"]
        ) + ".", "",
        "## 13. Resource usage", "",
        f"Cache {summary['resources']['cache_construction_seconds']:.1f}s; global sweep "
        f"{summary['resources']['sweep_seconds']:.1f}s; greedy "
        f"{summary['resources']['greedy_seconds']:.1f}s; joint "
        f"{summary['resources']['joint_seconds']:.1f}s; split-half "
        f"{summary['resources']['split_half_seconds']:.1f}s. Peak CPU RAM: "
        f"{summary['resources']['peak_cpu_ram_bytes']} bytes; peak GPU: "
        f"{summary['resources']['peak_gpu_bytes']} bytes.", "",
        "## 14. Scientific limitations", "",
        "The joint result uses validation labels to choose 171 values. It is not a generalization result.",
        "", "## 15. Decision-rule verdict", "", f"{verdict}: {reason}.",
        "", "## 16. Exact reproduction commands", "",
        *[f"- `{command}`" for command in summary["commands"]],
    ]
    from src.e3_affinity_oracle import _atomic_bytes
    _atomic_bytes(args.markdown, ("\n".join(lines) + "\n").encode(), overwrite=args.overwrite)


def _parse_float_csv(value: str, *, label: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise AffinityOracleError(f"{label} must be a comma-separated float list") from error
    if not values:
        raise AffinityOracleError(f"{label} must contain at least one value")
    for item in values:
        if not 0.0 <= item < 1.0:
            raise AffinityOracleError(f"{label} value {item} is outside [0,1)")
    return sorted(set(values))


def _parse_int_csv(value: str, *, label: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise AffinityOracleError(f"{label} must be a comma-separated integer list") from error
    if not values:
        raise AffinityOracleError(f"{label} must contain at least one value")
    for item in values:
        if item <= 0:
            raise AffinityOracleError(f"{label} value {item} must be positive")
    return sorted(set(values))


def _parse_signed_float_csv(value: str, *, label: str) -> list[float]:
    """Like _parse_float_csv but for --bias-grid: multipliers of a per-class
    score std, so negative values and values >1 are both valid."""

    try:
        values = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise AffinityOracleError(f"{label} must be a comma-separated float list") from error
    if not values:
        raise AffinityOracleError(f"{label} must contain at least one value")
    if not all(math.isfinite(item) for item in values):
        raise AffinityOracleError(f"{label} values must be finite")
    return sorted(set(values))


def _assert_global_sweep_ext_anchors(cache: Path, device: str) -> None:
    zero = evaluate_cache(cache, 0.0, device=device, propagation_steps=10)
    point95 = evaluate_cache(cache, 0.95, device=device, propagation_steps=10)
    actual = {
        "alpha=0.00,T=10 mIoU": zero["mIoU"],
        "alpha=0.95,T=10 mIoU": point95["mIoU"],
        "alpha=0.95,T=10 aAcc": point95["aAcc"],
        "alpha=0.95,T=10 mAcc": point95["mAcc"],
    }
    failures = {
        name: {"actual": actual[name], "expected": expected}
        for name, expected in GLOBAL_SWEEP_EXT_ANCHORS.items()
        if abs(actual[name] - expected) > GLOBAL_SWEEP_EXT_ANCHOR_TOLERANCE
    }
    if failures:
        raise AffinityOracleError(f"global-sweep-ext anchor mismatch: {failures}")
    print(
        "global-sweep-ext anchors OK: "
        + ", ".join(f"{name}={value:.12f}" for name, value in actual.items())
    )


def _validate_global_sweep_ext_payload(
    payload: dict, alpha_values: list[float], steps_values: list[int],
) -> None:
    from src.e3_affinity_oracle import _require_finite_tree
    _closed_nested(payload, GLOBAL_SWEEP_EXT_PAYLOAD_KEYS, "global-sweep-ext payload")
    expected_rows = len(alpha_values) * len(steps_values)
    if not isinstance(payload["rows"], list) or len(payload["rows"]) != expected_rows:
        raise AffinityOracleError("global-sweep-ext row count mismatch")
    for index, row in enumerate(payload["rows"]):
        _validate_metric(row, GLOBAL_SWEEP_EXT_ROW_KEYS, f"global-sweep-ext rows[{index}]")
    _closed_nested(
        payload["optimum"], GLOBAL_SWEEP_EXT_OPTIMUM_KEYS, "global-sweep-ext optimum"
    )
    if payload["optimum"]["at_grid_edge"] and not payload["warning"]:
        raise AffinityOracleError("grid-edge optimum must carry a top-level warning")
    _require_finite_tree(payload, "global-sweep-ext payload")


def global_sweep_ext(args: argparse.Namespace) -> None:
    if args.assert_anchors:
        _assert_global_sweep_ext_anchors(args.cache, args.device)
        return
    baseline_result = _load_result(args.baseline, "baseline")
    manifest, cache_sha = _compatible(args.cache, baseline_result)
    if not baseline_result["payload"]["canonical_reported_precision_matches"]:
        raise AffinityOracleError(
            "alpha=0 does not reproduce canonical E3; stopping extended sweep"
        )
    alpha_values = _parse_float_csv(args.alpha_grid, label="--alpha-grid")
    steps_values = _parse_int_csv(args.steps, label="--steps")
    rows: list[dict] = []
    for steps in steps_values:
        for alpha in alpha_values:
            metrics = evaluate_cache(
                args.cache, alpha, device=args.device, propagation_steps=steps,
            )
            rows.append({"alpha": alpha, "steps": steps, **metrics})
            print(
                f"global-sweep-ext done: alpha={alpha:.4f} steps={steps} "
                f"mIoU={metrics['mIoU']:.6f} aAcc={metrics['aAcc']:.6f} "
                f"mAcc={metrics['mAcc']:.6f} runtime={metrics['runtime_seconds']:.1f}s"
            )
    best = min(rows, key=lambda row: (-row["mIoU"], row["steps"], row["alpha"]))
    max_alpha, max_steps = max(alpha_values), max(steps_values)
    at_grid_edge = best["alpha"] == max_alpha or best["steps"] == max_steps
    optimum = {
        "alpha": best["alpha"], "steps": best["steps"], "mIoU": best["mIoU"],
        "aAcc": best["aAcc"], "mAcc": best["mAcc"],
        "delta_mIoU_from_canonical_alpha_0.95_T10": (
            best["mIoU"] - GLOBAL_SWEEP_EXT_ANCHORS["alpha=0.95,T=10 mIoU"]
        ),
        "at_grid_edge": at_grid_edge,
    }
    warning = (
        f"OPTIMUM AT GRID EDGE (alpha={best['alpha']:.4f}, steps={best['steps']}) "
        "-- extend --alpha-grid and/or --steps and re-run; the true optimum "
        "has not been bracketed."
        if at_grid_edge else None
    )
    if warning:
        print(f"WARNING: {warning}")
    payload = {
        "label": "extended_alpha_steps_global_sweep",
        "warning": warning,
        "alpha_grid": alpha_values, "steps_grid": steps_values,
        "rows": rows, "optimum": optimum,
        "kappa_k_baked_at_construction_time": {"affinity_power": True, "knn_k": True},
        "kappa_k_sweep_limitation": KAPPA_K_SWEEP_LIMITATION,
    }
    _validate_global_sweep_ext_payload(payload, alpha_values, steps_values)
    result = {
        "format_version": GLOBAL_SWEEP_EXT_FORMAT,
        "experiment": EXPERIMENT,
        "command": "global-sweep-ext",
        "invocation": " ".join(shlex.quote(argument) for argument in sys.argv),
        "seed": None,
        "git_commit": manifest["source_git_commit"],
        "git_dirty": manifest["source_git_dirty"],
        "cache_manifest_sha256": cache_sha,
        "class_order_sha256": manifest["class_order_sha256"],
        "dataset_config_sha256": manifest["dataset_config_sha256"],
        "payload": payload,
    }
    atomic_json(args.output, result, overwrite=args.overwrite)
    csv_rows = [
        {
            "alpha": row["alpha"], "steps": row["steps"], "aAcc": row["aAcc"],
            "mIoU": row["mIoU"], "mAcc": row["mAcc"],
            "runtime_seconds": row["runtime_seconds"],
            "peak_cpu_ram_bytes": row["peak_cpu_ram_bytes"],
            "peak_gpu_bytes": row["peak_gpu_bytes"],
        }
        for row in rows
    ]
    write_rows_csv(args.csv, csv_rows, overwrite=args.overwrite)


def _load_global_sweep_ext_result(path: Path) -> dict:
    value = json.loads(path.read_text())
    required = {
        "format_version", "experiment", "command", "invocation", "seed",
        "git_commit", "git_dirty", "cache_manifest_sha256",
        "class_order_sha256", "dataset_config_sha256", "payload",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise AffinityOracleError(f"closed v2 result schema mismatch: {path}")
    if value["format_version"] != GLOBAL_SWEEP_EXT_FORMAT:
        raise AffinityOracleError(f"v2 result format mismatch: {path}")
    if value["command"] != "global-sweep-ext":
        raise AffinityOracleError(
            "split-half --stage local / local-stat-fit require a "
            f"global-sweep-ext (Part A) artifact, got command="
            f"{value['command']!r}: {path}"
        )
    if value["experiment"] != EXPERIMENT:
        raise AffinityOracleError(f"result experiment mismatch: {path}")
    return value


def verify_feature_capture_cli(args: argparse.Namespace) -> None:
    if args.assert_anchors:
        anchors = assert_feature_capture_anchors(
            args.capture_dir, args.cache, device=args.device,
        )
        metrics = {k: v for k, v in anchors.items() if k != "tolerance_rationale"}
        print(
            "feature-capture E4 anchors OK: "
            + ", ".join(f"{name}={value:.6f}" for name, value in metrics.items())
        )
        rationale = anchors["tolerance_rationale"]
        print(f"tolerance_used={rationale['tolerance_used']}")
        print(f"measured_deviation={rationale['measured_deviation']}")
        print(f"ratio_deviation_to_effect_size={rationale['ratio_deviation_to_effect_size']}")
        if args.output is not None:
            capture_manifest = load_capture_manifest(args.capture_dir)
            result = {
                "format_version": FEATURE_CAPTURE_VERIFY_FORMAT,
                "experiment": EXPERIMENT,
                "command": "verify-feature-capture --assert-anchors",
                "invocation": " ".join(shlex.quote(argument) for argument in sys.argv),
                "seed": capture_manifest["seed"],
                "git_commit": capture_manifest["source_git_commit"],
                "git_dirty": capture_manifest["source_git_dirty"],
                "cache_manifest_sha256": sha256_file(args.cache / "manifest.json"),
                "payload": anchors,
            }
            atomic_json(args.output, result, overwrite=args.overwrite)
        return
    if args.output is None:
        raise AffinityOracleError("--output is required unless --assert-anchors is set")

    capture_manifest = load_capture_manifest(args.capture_dir)
    cache_sha = sha256_file(args.cache / "manifest.json")
    if cache_sha != capture_manifest["existing_cache_manifest_sha256"]:
        raise AffinityOracleError(
            "cache manifest changed since capture_dino_features.py ran "
            f"(capture recorded {capture_manifest['existing_cache_manifest_sha256']}, "
            f"current cache is {cache_sha}); re-run capture against the current cache"
        )

    payload = verify_feature_capture(
        args.capture_dir, args.cache, device=args.device, max_images=args.max_images,
    )
    result = {
        "format_version": FEATURE_CAPTURE_VERIFY_FORMAT,
        "experiment": EXPERIMENT,
        "command": "verify-feature-capture",
        "invocation": " ".join(shlex.quote(argument) for argument in sys.argv),
        "seed": capture_manifest["seed"],
        "git_commit": capture_manifest["source_git_commit"],
        "git_dirty": capture_manifest["source_git_dirty"],
        "cache_manifest_sha256": cache_sha,
        "capture_manifest_existing_cache_sha256": capture_manifest["existing_cache_manifest_sha256"],
        "capture_split": capture_manifest["split"],
        "capture_limit": capture_manifest["limit"],
        "payload": payload,
    }
    atomic_json(args.output, result, overwrite=args.overwrite)
    print(
        "verify-feature-capture done: "
        f"images_matched={payload['images_matched']} "
        f"windows_compared={payload['windows_compared']} "
        f"window_exact_index_set_match_fraction={payload['window_exact_index_set_match_fraction']:.6f} "
        f"edge_match_fraction={payload['edge_match_fraction']:.6f} "
        f"max_abs_weight_diff_where_indices_match={payload['max_abs_weight_diff_where_indices_match']:.6f} "
        f"disagreeing_row_count={payload['disagreeing_row_count']} "
        f"max_tie_affinity_gap={payload['max_tie_affinity_gap']:.6f} "
        f"gate_passed={payload['gate_passed']}"
    )
    if not payload["gate_passed"]:
        raise AffinityOracleError(
            "E3 correctness gate FAILED: edge_match_fraction="
            f"{payload['edge_match_fraction']:.6f} < 0.99 "
            f"(see {args.output} for the full report before investigating)"
        )


def _mean_baseline_iou_of_dominant_classes(
    dominant: list[dict], baseline_per_class_iou: list,
) -> list:
    result = []
    for bucket_row in dominant:
        ious = [
            baseline_per_class_iou[entry["class_index"]]
            for entry in bucket_row["dominant_classes"]
            if baseline_per_class_iou[entry["class_index"]] is not None
        ]
        result.append(sum(ious) / len(ious) if ious else None)
    return result


def _run_bucket_fit(
    preloaded, manifest, *, stat: str, n_buckets: int, alpha_grid: list[float],
    global_alpha: float, global_steps: int, max_sweeps: int, seed: int,
    device: str, part_a_optimum_miou: float,
):
    from src.e3_affinity_oracle import assign_buckets

    values = collect_patch_statistic_values(preloaded, manifest, stat=stat)
    edges = fit_bucket_edges(values, n_buckets=n_buckets)
    fit = coordinate_ascent_bucket_fit(
        preloaded, manifest, stat=stat, edges=edges, alpha_grid=alpha_grid,
        global_alpha=global_alpha, n_buckets=n_buckets, max_sweeps=max_sweeps,
        seed=seed, propagation_steps=global_steps, device=device, log=print,
    )
    final_miou = fit["final_metrics"]["mIoU"]
    if final_miou < part_a_optimum_miou - 1e-9:
        raise AffinityOracleError(
            f"local-stat-fit final joint mIoU ({final_miou}) is below Part "
            f"A's global optimum ({part_a_optimum_miou}) -- the structural "
            "guarantee is violated; refusing to write an artifact"
        )
    bucket_indices = assign_buckets(values, edges)
    patches_per_bucket = [
        int((bucket_indices == bucket).sum()) for bucket in range(n_buckets)
    ]
    baseline = evaluate_preloaded(
        preloaded, manifest, alpha=0.0, device=device, propagation_steps=global_steps,
    )
    dominant = dominant_classes_per_bucket(
        preloaded, manifest, stat=stat, edges=edges, global_alpha=global_alpha,
        propagation_steps=global_steps, device=device, n_buckets=n_buckets,
    )
    mean_baseline_iou = _mean_baseline_iou_of_dominant_classes(
        dominant, baseline["per_class_iou"]
    )
    spearman = spearman_correlation(list(range(n_buckets)), fit["bucket_alphas"])
    return edges, fit, patches_per_bucket, dominant, mean_baseline_iou, spearman


def local_stat_fit(args: argparse.Namespace) -> None:
    if args.assert_anchors:
        _assert_global_sweep_ext_anchors(args.cache, args.device)
        return
    if args.stat not in PATCH_STATISTICS:
        raise AffinityOracleError(f"--stat must be one of {PATCH_STATISTICS}")
    if args.n_buckets <= 0:
        raise AffinityOracleError("--n-buckets must be positive")
    if args.max_sweeps <= 0:
        raise AffinityOracleError("--max-sweeps must be positive")
    global_sweep_result = _load_global_sweep_ext_result(args.global_sweep)
    manifest, cache_sha = _compatible(args.cache, global_sweep_result)
    optimum = global_sweep_result["payload"]["optimum"]
    global_alpha = optimum["alpha"]
    global_steps = optimum["steps"]
    part_a_optimum_miou = optimum["mIoU"]
    alpha_values = (
        sorted(set(ALPHA_GRID) | {global_alpha}) if args.alpha_grid is None
        else _parse_float_csv(args.alpha_grid, label="--alpha-grid")
    )
    preloaded = preload_cache_images(args.cache, manifest)
    edges, fit, patches_per_bucket, dominant, mean_baseline_iou, spearman = _run_bucket_fit(
        preloaded, manifest, stat=args.stat, n_buckets=args.n_buckets,
        alpha_grid=alpha_values, global_alpha=global_alpha,
        global_steps=global_steps, max_sweeps=args.max_sweeps, seed=args.seed,
        device=args.device, part_a_optimum_miou=part_a_optimum_miou,
    )
    payload = {
        "label": "local_stat_fit_coordinate_ascent",
        "stat": args.stat, "stat_definition_note": DEGREE_STAT_LIMITATION,
        "n_buckets": args.n_buckets, "seed": args.seed,
        "global_alpha": global_alpha, "global_propagation_steps": global_steps,
        "bucket_edges": edges.tolist(), "bucket_alphas": fit["bucket_alphas"],
        "bucket_order": fit["bucket_order"],
        "patches_per_bucket": patches_per_bucket,
        "dominant_classes_per_bucket": dominant,
        "mean_baseline_iou_of_dominant_classes": mean_baseline_iou,
        "bucket_index_vs_alpha_spearman": spearman,
        "alpha_grid": alpha_values, "trace": fit["trace"],
        "sweeps_run": fit["sweeps_run"], "converged": fit["converged"],
        "final_metrics": fit["final_metrics"],
        "part_a_optimum_mIoU": part_a_optimum_miou,
        "delta_mIoU_from_part_a_optimum": (
            fit["final_metrics"]["mIoU"] - part_a_optimum_miou
        ),
        "monotonicity_assertion_passed": True,
        "final_at_least_part_a_optimum_assertion_passed": True,
    }
    _closed_nested(payload, LOCAL_STAT_FIT_PAYLOAD_KEYS, "local-stat-fit payload")
    _validate_metric(payload["final_metrics"], EVALUATION_METRIC_KEYS, "final_metrics")
    from src.e3_affinity_oracle import _require_finite_tree
    _require_finite_tree(payload, "local-stat-fit payload")
    result = {
        "format_version": GLOBAL_SWEEP_EXT_FORMAT, "experiment": EXPERIMENT,
        "command": "local-stat-fit",
        "invocation": " ".join(shlex.quote(argument) for argument in sys.argv),
        "seed": args.seed, "git_commit": manifest["source_git_commit"],
        "git_dirty": manifest["source_git_dirty"],
        "cache_manifest_sha256": cache_sha,
        "class_order_sha256": manifest["class_order_sha256"],
        "dataset_config_sha256": manifest["dataset_config_sha256"],
        "payload": payload,
    }
    atomic_json(args.output, result, overwrite=args.overwrite)
    write_rows_csv(args.csv, [
        {key: row[key] for key in ("sweep", "bucket", "alpha", "joint_mIoU", "accepted")}
        for row in fit["trace"]
    ], overwrite=args.overwrite)


def split_half_local(args: argparse.Namespace) -> None:
    if args.assert_anchors:
        _assert_global_sweep_ext_anchors(args.cache, args.device)
        return
    if args.split_global is None:
        raise AffinityOracleError("split-half --stage local requires --split-global")
    if args.global_sweep is None:
        raise AffinityOracleError("split-half --stage local requires --global-sweep")
    if args.csv is None:
        raise AffinityOracleError("split-half --stage local requires --csv")
    if args.stat not in PATCH_STATISTICS:
        raise AffinityOracleError(f"--stat must be one of {PATCH_STATISTICS}")
    if args.n_buckets <= 0:
        raise AffinityOracleError("--n-buckets must be positive")
    if args.max_sweeps <= 0:
        raise AffinityOracleError("--max-sweeps must be positive")
    global_result = _load_result(args.split_global, "split-half-global")
    identity = _compatible(args.cache, global_result)
    global_sweep_result = _load_global_sweep_ext_result(args.global_sweep)
    manifest, cache_sha = _compatible(args.cache, global_sweep_result, identity)
    global_payload = global_result["payload"]
    a_indices = set(global_payload["A_indices"])
    b_indices = set(global_payload["B_indices"])
    optimum = global_sweep_result["payload"]["optimum"]
    global_alpha = optimum["alpha"]
    global_steps = optimum["steps"]
    part_a_optimum_miou = optimum["mIoU"]
    alpha_values = (
        sorted(set(ALPHA_GRID) | {global_alpha}) if args.alpha_grid is None
        else _parse_float_csv(args.alpha_grid, label="--alpha-grid")
    )
    # Halves are read from the frozen split-half-global artifact, never
    # re-derived from --seed here -- B5's core requirement.
    preloaded_a = preload_cache_images(args.cache, manifest, selected_indices=a_indices)
    edges, fit, patches_per_bucket, dominant, mean_baseline_iou, spearman = _run_bucket_fit(
        preloaded_a, manifest, stat=args.stat, n_buckets=args.n_buckets,
        alpha_grid=alpha_values, global_alpha=global_alpha,
        global_steps=global_steps, max_sweeps=args.max_sweeps, seed=args.seed,
        device=args.device, part_a_optimum_miou=part_a_optimum_miou,
    )
    bucket_alphas_tensor = torch.tensor(fit["bucket_alphas"], dtype=torch.float32)
    baseline_b = evaluate_cache(
        args.cache, 0.0, device=args.device, selected_indices=b_indices,
    )
    global_b = evaluate_cache(
        args.cache, global_alpha, device=args.device, selected_indices=b_indices,
        propagation_steps=global_steps,
    )
    local_b = evaluate_cache_bucketed(
        args.cache, stat=args.stat, edges=edges, bucket_alphas=bucket_alphas_tensor,
        device=args.device, selected_indices=b_indices, propagation_steps=global_steps,
    )
    transfer_delta = local_b["mIoU"] - baseline_b["mIoU"]
    gain_over_global = local_b["mIoU"] - global_b["mIoU"]
    payload = {
        "label": "split_half_local_stat_transfer",
        "warning": "SPLIT-HALF TRANSFER -- PRIMARY DECISION METRIC",
        "stat": args.stat, "stat_definition_note": DEGREE_STAT_LIMITATION,
        "n_buckets": args.n_buckets, "seed": args.seed,
        "A_fingerprint": global_payload["A_fingerprint"],
        "B_fingerprint": global_payload["B_fingerprint"],
        "global_alpha": global_alpha, "global_propagation_steps": global_steps,
        "bucket_edges": edges.tolist(), "bucket_alphas": fit["bucket_alphas"],
        "bucket_order": fit["bucket_order"],
        "patches_per_bucket": patches_per_bucket,
        "dominant_classes_per_bucket": dominant,
        "mean_baseline_iou_of_dominant_classes_A": mean_baseline_iou,
        "bucket_index_vs_alpha_spearman": spearman,
        "alpha_grid": alpha_values, "trace": fit["trace"],
        "sweeps_run": fit["sweeps_run"], "converged": fit["converged"],
        "A_final_metrics": fit["final_metrics"],
        "part_a_optimum_mIoU": part_a_optimum_miou,
        "monotonicity_assertion_passed": True,
        "final_at_least_part_a_optimum_assertion_passed": True,
        "baseline_B": baseline_b, "global_B": global_b, "local_B": local_b,
        "transfer_delta_mIoU": transfer_delta,
        "gain_over_global_B": {
            "value": gain_over_global, "primary_decision_metric": True,
        },
    }
    _closed_nested(payload, SPLIT_HALF_LOCAL_PAYLOAD_KEYS, "split-half-local payload")
    _closed_nested(
        payload["gain_over_global_B"], {"value", "primary_decision_metric"},
        "gain_over_global_B",
    )
    for label in ("baseline_B", "global_B", "local_B"):
        _validate_metric(payload[label], EVALUATION_METRIC_KEYS, label)
    from src.e3_affinity_oracle import _require_finite_tree
    _require_finite_tree(payload, "split-half-local payload")
    result = {
        "format_version": GLOBAL_SWEEP_EXT_FORMAT, "experiment": EXPERIMENT,
        "command": "split-half-local",
        "invocation": " ".join(shlex.quote(argument) for argument in sys.argv),
        "seed": args.seed, "git_commit": manifest["source_git_commit"],
        "git_dirty": manifest["source_git_dirty"],
        "cache_manifest_sha256": cache_sha,
        "class_order_sha256": manifest["class_order_sha256"],
        "dataset_config_sha256": manifest["dataset_config_sha256"],
        "payload": payload,
    }
    atomic_json(args.output, result, overwrite=args.overwrite)
    write_rows_csv(args.csv, [
        {key: row[key] for key in ("sweep", "bucket", "alpha", "joint_mIoU", "accepted")}
        for row in fit["trace"]
    ], overwrite=args.overwrite)


def _load_text_embedding(path: Path, manifest: dict) -> torch.Tensor:
    bundle = torch.load(path, weights_only=False)
    required = {
        "text_embedding", "class_names", "sha256", "expected_sha256",
        "hash_matched", "device_used", "template", "source_command",
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise AffinityOracleError(f"unrecognized text-embedding bundle: {path}")
    if list(bundle["class_names"]) != list(manifest["class_names"]):
        raise AffinityOracleError(
            "text-embedding class-name order does not match this cache's manifest"
        )
    if bundle["sha256"] != manifest["text_embedding_sha256"]:
        raise AffinityOracleError(
            "text-embedding identity does not match this cache's "
            f"manifest.text_embedding_sha256 (bundle sha256={bundle['sha256']}, "
            f"expected={manifest['text_embedding_sha256']}); rebuild the "
            "embedding against this exact cache/GPU before reusing it here"
        )
    embedding = bundle["text_embedding"]
    if (
        not torch.is_tensor(embedding) or embedding.ndim != 2
        or embedding.shape[0] != manifest["class_count"]
    ):
        raise AffinityOracleError("text-embedding tensor has an unexpected shape")
    return embedding.double()


def bias_fit(args: argparse.Namespace) -> None:
    if args.assert_anchors:
        _assert_global_sweep_ext_anchors(args.cache, args.device)
        return
    if args.max_sweeps <= 0:
        raise AffinityOracleError("--max-sweeps must be positive")
    global_sweep_result = _load_global_sweep_ext_result(args.global_sweep)
    manifest, cache_sha = _compatible(args.cache, global_sweep_result)
    optimum = global_sweep_result["payload"]["optimum"]
    global_alpha = optimum["alpha"]
    global_steps = optimum["steps"]
    part_a_optimum_miou = optimum["mIoU"]
    multipliers = _parse_signed_float_csv(args.bias_grid, label="--bias-grid")
    preloaded = preload_cache_images(args.cache, manifest)
    std = class_propagated_score_std(
        preloaded, manifest, alpha=global_alpha, propagation_steps=global_steps,
        device=args.device,
    )
    grid_per_class = [
        [float(multiplier) * float(std[class_index]) for multiplier in multipliers]
        for class_index in range(manifest["class_count"])
    ]
    fit = coordinate_ascent_bias_fit(
        preloaded, manifest, alpha=global_alpha, grid_per_class=grid_per_class,
        max_sweeps=args.max_sweeps, seed=args.seed, propagation_steps=global_steps,
        device=args.device, log=print,
    )
    final_miou = fit["final_metrics"]["mIoU"]
    if final_miou < part_a_optimum_miou - 1e-9:
        raise AffinityOracleError(
            f"bias-fit final joint mIoU ({final_miou}) is below Part A's "
            f"global optimum ({part_a_optimum_miou}) -- the structural "
            "guarantee is violated; refusing to write an artifact"
        )
    payload = {
        "label": "bias_fit_coordinate_ascent",
        "diagnostic_only": True, "diagnostic_only_note": DIAGNOSTIC_ONLY_NOTE,
        "bias_grid_multipliers": multipliers, "per_class_score_std": std.tolist(),
        "global_alpha": global_alpha, "global_propagation_steps": global_steps,
        "beta_by_class": fit["beta_by_class"],
        "class_visit_order": fit["class_visit_order"],
        "trace": fit["trace"], "sweeps_run": fit["sweeps_run"],
        "converged": fit["converged"], "final_metrics": fit["final_metrics"],
        "part_a_optimum_mIoU": part_a_optimum_miou,
        "delta_mIoU_from_part_a_optimum": final_miou - part_a_optimum_miou,
        "monotonicity_assertion_passed": True,
        "final_at_least_part_a_optimum_assertion_passed": True,
    }
    _closed_nested(payload, BIAS_FIT_PAYLOAD_KEYS, "bias-fit payload")
    _validate_metric(payload["final_metrics"], EVALUATION_METRIC_KEYS, "final_metrics")
    from src.e3_affinity_oracle import _require_finite_tree
    _require_finite_tree(payload, "bias-fit payload")
    result = {
        "format_version": GLOBAL_SWEEP_EXT_FORMAT, "experiment": EXPERIMENT,
        "command": "bias-fit",
        "invocation": " ".join(shlex.quote(argument) for argument in sys.argv),
        "seed": args.seed, "git_commit": manifest["source_git_commit"],
        "git_dirty": manifest["source_git_dirty"],
        "cache_manifest_sha256": cache_sha,
        "class_order_sha256": manifest["class_order_sha256"],
        "dataset_config_sha256": manifest["dataset_config_sha256"],
        "payload": payload,
    }
    atomic_json(args.output, result, overwrite=args.overwrite)
    write_rows_csv(args.csv, [
        {key: row[key] for key in ("sweep", "class_index", "beta", "joint_mIoU", "accepted")}
        for row in fit["trace"]
    ], overwrite=args.overwrite)


def split_half_bias(args: argparse.Namespace) -> None:
    if args.assert_anchors:
        _assert_global_sweep_ext_anchors(args.cache, args.device)
        return
    if args.split_global is None:
        raise AffinityOracleError("split-half --stage bias requires --split-global")
    if args.global_sweep is None:
        raise AffinityOracleError("split-half --stage bias requires --global-sweep")
    if args.text_embedding is None:
        raise AffinityOracleError("split-half --stage bias requires --text-embedding")
    if args.csv is None:
        raise AffinityOracleError("split-half --stage bias requires --csv")
    if args.max_sweeps <= 0:
        raise AffinityOracleError("--max-sweeps must be positive")

    global_result = _load_result(args.split_global, "split-half-global")
    identity = _compatible(args.cache, global_result)
    global_sweep_result = _load_global_sweep_ext_result(args.global_sweep)
    manifest, cache_sha = _compatible(args.cache, global_sweep_result, identity)
    global_payload = global_result["payload"]
    a_indices = set(global_payload["A_indices"])
    b_indices = set(global_payload["B_indices"])
    optimum = global_sweep_result["payload"]["optimum"]
    global_alpha = optimum["alpha"]
    global_steps = optimum["steps"]
    part_a_optimum_miou = optimum["mIoU"]

    text_embedding = _load_text_embedding(args.text_embedding, manifest)

    multipliers = _parse_signed_float_csv(args.bias_grid, label="--bias-grid")
    preloaded_a = preload_cache_images(args.cache, manifest, selected_indices=a_indices)
    std_a = class_propagated_score_std(
        preloaded_a, manifest, alpha=global_alpha, propagation_steps=global_steps,
        device=args.device,
    )
    grid_per_class = [
        [float(multiplier) * float(std_a[class_index]) for multiplier in multipliers]
        for class_index in range(manifest["class_count"])
    ]
    # B5's core requirement: reuse the frozen A/B split read above -- never
    # re-derive it from --seed. Fit on A only.
    fit = coordinate_ascent_bias_fit(
        preloaded_a, manifest, alpha=global_alpha, grid_per_class=grid_per_class,
        max_sweeps=args.max_sweeps, seed=args.seed, propagation_steps=global_steps,
        device=args.device, log=print,
    )
    a_final_miou = fit["final_metrics"]["mIoU"]
    if a_final_miou < part_a_optimum_miou - 1e-9:
        raise AffinityOracleError(
            f"split-half-bias half-A final joint mIoU ({a_final_miou}) is "
            f"below Part A's global optimum ({part_a_optimum_miou}) -- the "
            "structural guarantee is violated; refusing to write an artifact"
        )

    beta_tensor = torch.tensor(fit["beta_by_class"], dtype=torch.float32)
    baseline_a = evaluate_preloaded(
        preloaded_a, manifest, alpha=0.0, device=args.device,
        propagation_steps=global_steps,
    )
    baseline_b = evaluate_cache(
        args.cache, 0.0, device=args.device, selected_indices=b_indices,
    )
    global_b = evaluate_cache(
        args.cache, global_alpha, device=args.device, selected_indices=b_indices,
        propagation_steps=global_steps,
    )
    bias_b = evaluate_cache_biased(
        args.cache, alpha=global_alpha, beta=beta_tensor, device=args.device,
        selected_indices=b_indices, propagation_steps=global_steps,
    )
    transfer_delta = bias_b["mIoU"] - baseline_b["mIoU"]
    gain_over_global = bias_b["mIoU"] - global_b["mIoU"]

    # C4 trivial baselines, all measured on the fitting half (A), matching
    # the beta_c they are being regressed against.
    image_support_a, pixel_support_a = support_for_indices(args.cache, a_indices)
    total_pixels_a = sum(pixel_support_a)
    pixel_frequency_a = [
        (count / total_pixels_a) if total_pixels_a else 0.0 for count in pixel_support_a
    ]
    image_frequency_a = [count / len(a_indices) for count in image_support_a]

    supported = [
        class_index for class_index in range(manifest["class_count"])
        if baseline_a["per_class_iou"][class_index] is not None
    ]
    excluded = [c for c in range(manifest["class_count"]) if c not in supported]
    k_folds = min(TEXT_PREDICT_KFOLD, len(supported))
    if k_folds < 2:
        raise AffinityOracleError(
            "too few A-supported classes for grouped K-fold text-predictability CV"
        )

    beta_supported = torch.tensor(
        [fit["beta_by_class"][c] for c in supported], dtype=torch.float64
    )
    text_supported = text_embedding[supported]
    pixel_feature = torch.tensor(
        [[pixel_frequency_a[c]] for c in supported], dtype=torch.float64
    )
    image_feature = torch.tensor(
        [[image_frequency_a[c]] for c in supported], dtype=torch.float64
    )
    baseline_iou_feature = torch.tensor(
        [[baseline_a["per_class_iou"][c]] for c in supported], dtype=torch.float64
    )
    combined_feature = torch.cat(
        [text_supported, pixel_feature, image_feature, baseline_iou_feature], dim=-1
    )

    text_cv = cross_validated_ridge(
        text_supported, beta_supported, k=k_folds, seed=args.seed, alpha=RIDGE_ALPHA,
    )
    pixel_cv = cross_validated_ridge(
        pixel_feature, beta_supported, k=k_folds, seed=args.seed, alpha=RIDGE_ALPHA,
    )
    image_cv = cross_validated_ridge(
        image_feature, beta_supported, k=k_folds, seed=args.seed, alpha=RIDGE_ALPHA,
    )
    baseline_iou_cv = cross_validated_ridge(
        baseline_iou_feature, beta_supported, k=k_folds, seed=args.seed, alpha=RIDGE_ALPHA,
    )
    combined_cv = cross_validated_ridge(
        combined_feature, beta_supported, k=k_folds, seed=args.seed, alpha=RIDGE_ALPHA,
    )
    # Shuffled-target control: same CV machinery, target permuted with an
    # independent seed. A near-zero R^2 here is the sanity check that the
    # CV pipeline itself is not leaking.
    shuffle_seed = args.seed + 1
    shuffle_generator = torch.Generator().manual_seed(shuffle_seed)
    shuffled_beta = beta_supported[
        torch.randperm(len(supported), generator=shuffle_generator)
    ]
    shuffled_cv = cross_validated_ridge(
        text_supported, shuffled_beta, k=k_folds, seed=args.seed, alpha=RIDGE_ALPHA,
    )
    shuffled_control_near_zero = abs(shuffled_cv["r2"]) <= 0.05

    verdict, reason = _decision_from_bias(gain_over_global, text_cv["r2"])

    payload = {
        "label": "split_half_bias_text_predictability",
        "warning": "SPLIT-HALF TRANSFER -- PRIMARY DECISION METRIC",
        "diagnostic_only": True, "diagnostic_only_note": DIAGNOSTIC_ONLY_NOTE,
        "bias_grid_multipliers": multipliers,
        "per_class_score_std_A": std_a.tolist(), "seed": args.seed,
        "A_fingerprint": global_payload["A_fingerprint"],
        "B_fingerprint": global_payload["B_fingerprint"],
        "global_alpha": global_alpha, "global_propagation_steps": global_steps,
        "beta_by_class": fit["beta_by_class"],
        "class_visit_order": fit["class_visit_order"],
        "trace": fit["trace"], "sweeps_run": fit["sweeps_run"],
        "converged": fit["converged"], "A_final_metrics": fit["final_metrics"],
        "part_a_optimum_mIoU": part_a_optimum_miou,
        "monotonicity_assertion_passed": True,
        "final_at_least_part_a_optimum_assertion_passed": True,
        "baseline_B": baseline_b, "global_B": global_b, "bias_B": bias_b,
        "transfer_delta_mIoU": transfer_delta,
        "gain_over_global_B": {
            "value": gain_over_global, "primary_decision_metric": True,
        },
        "text_embedding_identity": manifest["text_embedding_sha256"],
        "text_embedding_dimension": int(text_embedding.shape[1]),
        "regression_class_count": len(supported),
        "excluded_classes_no_A_support": excluded,
        "ridge_alpha": RIDGE_ALPHA, "kfold_k": k_folds, "shuffle_seed": shuffle_seed,
        "text_cv": text_cv, "pixel_frequency_cv": pixel_cv,
        "image_frequency_cv": image_cv, "baseline_iou_cv": baseline_iou_cv,
        "text_plus_trivial_cv": combined_cv, "shuffled_target_control_cv": shuffled_cv,
        "shuffled_control_near_zero": shuffled_control_near_zero,
        "decision": {
            "source_metric": "gain_over_global_B,text_cv.r2",
            "verdict": verdict, "reason": reason,
        },
    }
    _closed_nested(payload, SPLIT_HALF_BIAS_PAYLOAD_KEYS, "split-half-bias payload")
    _closed_nested(
        payload["gain_over_global_B"], {"value", "primary_decision_metric"},
        "gain_over_global_B",
    )
    _closed_nested(
        payload["decision"], {"source_metric", "verdict", "reason"}, "decision",
    )
    for label in ("baseline_B", "global_B", "bias_B"):
        _validate_metric(payload[label], EVALUATION_METRIC_KEYS, label)
    _validate_metric(
        payload["A_final_metrics"], EVALUATION_METRIC_KEYS, "A_final_metrics"
    )
    for cv_key in (
        "text_cv", "pixel_frequency_cv", "image_frequency_cv", "baseline_iou_cv",
        "text_plus_trivial_cv", "shuffled_target_control_cv",
    ):
        _closed_nested(payload[cv_key], CV_RESULT_KEYS, cv_key)
    from src.e3_affinity_oracle import _require_finite_tree
    _require_finite_tree(payload, "split-half-bias payload")
    result = {
        "format_version": GLOBAL_SWEEP_EXT_FORMAT, "experiment": EXPERIMENT,
        "command": "split-half-bias",
        "invocation": " ".join(shlex.quote(argument) for argument in sys.argv),
        "seed": args.seed, "git_commit": manifest["source_git_commit"],
        "git_dirty": manifest["source_git_dirty"],
        "cache_manifest_sha256": cache_sha,
        "class_order_sha256": manifest["class_order_sha256"],
        "dataset_config_sha256": manifest["dataset_config_sha256"],
        "payload": payload,
    }
    atomic_json(args.output, result, overwrite=args.overwrite)
    write_rows_csv(args.csv, [
        {key: row[key] for key in ("sweep", "class_index", "beta", "joint_mIoU", "accepted")}
        for row in fit["trace"]
    ], overwrite=args.overwrite)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    verify_capture = commands.add_parser("verify-feature-capture")
    verify_capture.add_argument("--capture-dir", required=True, type=Path)
    verify_capture.add_argument("--cache", required=True, type=Path)
    verify_capture.add_argument("--device", default="cpu")
    verify_capture.add_argument("--output", type=Path, default=None)
    verify_capture.add_argument("--overwrite", action="store_true")
    verify_capture.add_argument("--max-images", type=int, default=None)
    verify_capture.add_argument("--assert-anchors", action="store_true")
    verify_capture.set_defaults(function=verify_feature_capture_cli)
    validate = commands.add_parser("validate-cache")
    validate.add_argument("cache", type=Path)
    validate.set_defaults(function=lambda a: print(json.dumps(cache_summary(a.cache), indent=2)))
    for name, function in (("baseline", baseline), ("global-sweep", sweep), ("greedy-fit", fit), ("joint-eval", joint), ("split-half", split_half), ("report", report), ("global-sweep-ext", global_sweep_ext), ("local-stat-fit", local_stat_fit), ("bias-fit", bias_fit)):
        command = commands.add_parser(name)
        command.add_argument("--cache", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--device", default="cpu")
        command.add_argument("--overwrite", action="store_true")
        command.set_defaults(function=function)
        if name == "global-sweep":
            command.add_argument("--baseline", required=True, type=Path)
            command.add_argument("--csv", required=True, type=Path)
        elif name == "greedy-fit":
            command.add_argument("--global-sweep", required=True, type=Path)
            command.add_argument("--subset-size", type=int, default=1000)
            command.add_argument("--csv", required=True, type=Path)
        elif name == "joint-eval":
            command.add_argument("--baseline", required=True, type=Path)
            command.add_argument("--global-sweep", required=True, type=Path)
            command.add_argument("--greedy-fit", required=True, type=Path)
        elif name == "report":
            command.add_argument("--baseline", required=True, type=Path)
            command.add_argument("--global-sweep", required=True, type=Path)
            command.add_argument("--greedy-fit", required=True, type=Path)
            command.add_argument("--joint", required=True, type=Path)
            command.add_argument("--split-half", required=True, type=Path)
            command.add_argument("--markdown", required=True, type=Path)
        elif name == "split-half":
            command.add_argument(
                "--stage", required=True,
                choices=("global", "greedy", "transfer", "local", "bias"),
            )
            command.add_argument("--split-global", type=Path)
            command.add_argument("--split-greedy", type=Path)
            command.add_argument("--joint", type=Path)
            # --stage local/bias only (B5/C2): reuses --split-global above
            # for the frozen A/B halves; these mirror local-stat-fit's and
            # bias-fit's own flags.
            command.add_argument("--global-sweep", type=Path)
            command.add_argument("--stat", choices=PATCH_STATISTICS, default="entropy")
            command.add_argument("--n-buckets", type=int, default=10)
            command.add_argument("--alpha-grid", default=None)
            command.add_argument(
                "--bias-grid",
                default="-1.0,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1.0",
            )
            command.add_argument("--text-embedding", type=Path)
            command.add_argument("--max-sweeps", type=int, default=3)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--csv", type=Path)
            command.add_argument("--assert-anchors", action="store_true")
        elif name == "global-sweep-ext":
            command.add_argument("--baseline", required=True, type=Path)
            command.add_argument(
                "--alpha-grid", default="0.95,0.96,0.97,0.98,0.99,0.995,0.999"
            )
            command.add_argument("--steps", default="10,20,40")
            command.add_argument("--csv", required=True, type=Path)
            command.add_argument("--assert-anchors", action="store_true")
        elif name == "local-stat-fit":
            command.add_argument("--global-sweep", required=True, type=Path)
            command.add_argument("--stat", choices=PATCH_STATISTICS, default="entropy")
            command.add_argument("--n-buckets", type=int, default=10)
            command.add_argument("--alpha-grid", default=None)
            command.add_argument("--max-sweeps", type=int, default=3)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--csv", required=True, type=Path)
            command.add_argument("--assert-anchors", action="store_true")
        elif name == "bias-fit":
            command.add_argument("--global-sweep", required=True, type=Path)
            command.add_argument(
                "--bias-grid",
                default="-1.0,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1.0",
            )
            command.add_argument("--max-sweeps", type=int, default=3)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--csv", required=True, type=Path)
            command.add_argument("--assert-anchors", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
