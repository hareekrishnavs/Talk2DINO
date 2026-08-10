#!/usr/bin/env python3
"""Streaming offline runner for the E3 affinity-spread oracle."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import torch

from src.e3_affinity_oracle import (
    ALPHA_GRID,
    EXPERIMENT,
    RESULT_FORMAT,
    AffinityOracleError,
    atomic_json,
    cache_summary,
    decision_from_transfer,
    evaluate_cache,
    fitting_support,
    global_sweep,
    greedy_fit,
    load_cache_manifest,
    ordered_fingerprint,
    peak_cpu_ram_bytes,
    split_balanced_halves,
    support_for_indices,
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-cache")
    validate.add_argument("cache", type=Path)
    validate.set_defaults(function=lambda a: print(json.dumps(cache_summary(a.cache), indent=2)))
    for name, function in (("baseline", baseline), ("global-sweep", sweep), ("greedy-fit", fit), ("joint-eval", joint), ("split-half", split_half), ("report", report)):
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
                "--stage", required=True, choices=("global", "greedy", "transfer")
            )
            command.add_argument("--split-global", type=Path)
            command.add_argument("--split-greedy", type=Path)
            command.add_argument("--joint", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
