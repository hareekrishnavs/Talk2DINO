#!/usr/bin/env python3
"""Reusable stitching-control suite evaluator.

Shares one immutable per-window k12 finite-step output (one shared
backbone/snapshot pass, one shared directed top-12 graph build, one shared
T=320 finite-step propagation -- every one of these reused unmodified from
``models.dinotext.cover_dr``, never rebuilt or rerun per variant) across
four frozen crop-to-image aggregation variants (uniform_probability,
hann_probability, uniform_score, hann_score -- see
``models.dinotext.cover_dr.stitching_control`` and
``evaluation_identities/e12_stitching_control_suite.toml``), then reports
paired per-variant full-precision aAcc/mIoU/mAcc plus per-image/per-class
sufficient statistics for all four variants.

This is NOT ``main.py --eval``: it drives its own bounded per-image loop,
never calls ``multi_gpu_test``, and never monkeypatches production
inference. Checkpoint reading/writing and validation go exclusively through
:mod:`src.stitching_control_checkpoint` -- this module never reimplements
checkpoint invariant checks.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.stitching_control_checkpoint import (
    CHECKPOINT_SCHEMA_NAME,
    parse_strict_json_document,
    resume_dataset_index,
    validate_checkpoint_against_artifact,
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
)
from src.stitching_control_identity import (
    CANONICAL_VARIANT_NAMES,
    RUN_MODE_IMAGE_COUNT_KEYS,
    RUN_MODE_SCHEMA_KEYS,
    StitchingControlIdentityError,
    load_identity,
    repository_root,
    validate_static_configuration,
)
from src.stitching_control_report import verify_record
from src.k11_k12_power_evaluation_identity import (
    load_identity as load_power_identity,
    validate_stability_result_binding,
)
from src.k11_k12_stability_report import write_checkpoint_atomically
from diagnostics.run_k11_k12_stability import (
    _build_inference,
    _extract_prepared_image,
    _git,
    _reject_tracked_output_path,
    _sha256_file,
)
from diagnostics.run_matched_k11_k12_evaluation import (
    _expected_image_ids,
    _image_order_digest,
    resolve_and_validate_class_count,
)


def _per_image_stats_npz_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".npz")


def _write_per_image_stats_atomically(
    manifest_path: Path, *, schema_name: str, class_count: int, dataset_indices: list[int],
    image_ids: list[str], label: np.ndarray, per_variant_arrays: dict[str, dict[str, np.ndarray]],
) -> str:
    """Atomically (re)write the per-image sufficient-statistics artifact
    for all four variants: a compact NPZ of exact integer arrays
    (``allow_pickle=False``) plus a small JSON manifest. GT (``label``) is
    stored once, shared by construction across every variant -- never
    duplicated or allowed to silently diverge."""
    arrays: dict[str, np.ndarray] = {"label": label}
    for variant in CANONICAL_VARIANT_NAMES:
        for suffix in ("intersect", "union", "pred"):
            arrays[f"{suffix}_{variant}"] = per_variant_arrays[variant][suffix]

    for name, array in arrays.items():
        if array.shape[0] != len(image_ids):
            raise StitchingControlIdentityError(f"per-image-stats array {name!r} row count disagrees with image_ids length")
        if array.shape[1] != class_count:
            raise StitchingControlIdentityError(f"per-image-stats array {name!r} has {array.shape[1]} columns, expected class_count={class_count}")

    npz_path = _per_image_stats_npz_path(manifest_path)
    temp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    np.savez(temp_npz, dataset_indices=np.asarray(dataset_indices, dtype=np.int64),
              **{name: array.astype(np.int64) for name, array in arrays.items()}, allow_pickle=False)
    written = temp_npz if temp_npz.suffix == ".npz" else temp_npz.with_suffix(temp_npz.suffix + ".npz")
    os.replace(written, npz_path)
    npz_sha256 = _sha256_file(npz_path)

    manifest = {
        "schema": schema_name, "npz_filename": npz_path.name, "npz_sha256": npz_sha256,
        "class_count": class_count, "image_count": len(image_ids), "dataset_indices": dataset_indices,
        "image_ids": image_ids, "image_order_digest": _image_order_digest(image_ids),
        "variant_names": list(CANONICAL_VARIANT_NAMES),
    }
    write_checkpoint_atomically(manifest_path, manifest)
    return npz_sha256


def _load_per_image_stats(manifest_path: Path) -> dict[str, Any]:
    manifest = parse_strict_json_document(manifest_path, label="per-image-stats manifest")
    npz_path = _per_image_stats_npz_path(manifest_path)
    if manifest["npz_sha256"] != _sha256_file(npz_path):
        raise StitchingControlIdentityError("per-image-stats NPZ SHA256 does not match its manifest; refusing to resume")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return {"manifest": manifest, "arrays": arrays}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable stitching-control suite evaluator: shares one immutable per-window k12 output across four frozen aggregation variants.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--stability-result", type=Path, required=True)
    parser.add_argument("--run-mode", choices=list(RUN_MODE_IMAGE_COUNT_KEYS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--per-image-stats", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _run_evaluation(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)

    # --- STATIC + STABILITY-GATE BINDING: fully validated BEFORE any
    # dataset/model construction or CUDA initialization. ---
    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_stitching_control_suite.toml"
    identity_sha256 = _sha256_file(identity_path)

    power_identity_path = root / identity["parent_identity"]["power_evaluation_identity_path"]
    power_identity = load_power_identity(power_identity_path, repo_root=root)
    binding = validate_stability_result_binding(args.stability_result, identity=power_identity, repo_root=root, check_git=True)

    image_count = identity["run_modes"][RUN_MODE_IMAGE_COUNT_KEYS[args.run_mode]]
    schema_name = identity["run_modes"][RUN_MODE_SCHEMA_KEYS[args.run_mode]]
    registered_class_count = identity["metrics"]["class_count"]

    if not args.overwrite and args.result.exists():
        raise StitchingControlIdentityError(f"--result {args.result} already exists; pass --overwrite for explicit resume/overwrite behavior")
    for path in (args.result, args.checkpoint, args.per_image_stats, _per_image_stats_npz_path(args.per_image_stats)):
        try:
            _reject_tracked_output_path(root, path)
        except Exception as error:  # noqa: BLE001 -- re-raised as this CLI's own error type
            raise StitchingControlIdentityError(str(error)) from error
    for path in (args.result, args.checkpoint, args.per_image_stats):
        path.parent.mkdir(parents=True, exist_ok=True)

    # --- PHASE A checkpoint validation: no dataset, model, or CUDA. ---
    if args.resume and args.checkpoint.exists():
        phase_a_checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
        validate_checkpoint_structure(phase_a_checkpoint, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode)
        resume_dataset_index(phase_a_checkpoint)
        phase_a_stats = _load_per_image_stats(args.per_image_stats)
        validate_checkpoint_against_artifact(phase_a_checkpoint, stats_manifest=phase_a_stats["manifest"], stats_arrays=phase_a_stats["arrays"])

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise StitchingControlIdentityError("--device cuda requested but CUDA is not available")

    from src.matched_k11_k12_identity import load_identity as load_matched_identity
    from src.e3_evaluation_identity import load_identity as load_e3_identity

    matched_identity = load_matched_identity(root / identity["parent_identity"]["matched_identity_path"], repo_root=root)
    e3_identity = load_e3_identity(root / matched_identity["parent_identities"]["e3_identity_path"], repo_root=root)

    alpha = matched_identity["propagation"]["alpha"]
    steps = matched_identity["propagation"]["steps"]
    affinity_power = matched_identity["graph"]["affinity_power"]
    if matched_identity["graph"]["maximum_rank"] != 12:
        raise StitchingControlIdentityError("matched_identity.graph.maximum_rank must be exactly 12")

    git_commit = _git(root, "rev-parse", "HEAD")
    git_branch = _git(root, "branch", "--show-current")

    inference, dataset = _build_inference(root, e3_identity, args.device, log_dir=args.result.parent)
    align_corners = inference.align_corners
    class_count = resolve_and_validate_class_count(inference, dataset, expected_class_count=registered_class_count)
    expected_image_ids = _expected_image_ids(dataset, image_count)
    image_order_digest = _image_order_digest(expected_image_ids)

    from models.dinotext.cover_dr.stitching_control import (
        CANONICAL_VARIANTS,
        finalize_prediction,
        stitch_one_image_multi_variant,
    )
    from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize

    started_at = time.monotonic()

    dataset_indices: list[int] = []
    image_ids: list[str] = []
    label_rows: list[np.ndarray] = []
    per_variant_rows: dict[str, dict[str, list[np.ndarray]]] = {
        v: {"intersect": [], "union": [], "pred": []} for v in CANONICAL_VARIANT_NAMES
    }
    completed_image_ids: list[str] = []
    windows_processed_total = 0
    next_dataset_index = 0

    if args.resume and args.checkpoint.exists():
        checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
        validate_checkpoint_structure(checkpoint, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode, class_count=class_count)
        validate_checkpoint_against_canonical_order(checkpoint, expected_image_ids, image_order_digest=image_order_digest)
        next_dataset_index = resume_dataset_index(checkpoint)
        stats = _load_per_image_stats(args.per_image_stats)
        validate_checkpoint_against_artifact(checkpoint, stats_manifest=stats["manifest"], stats_arrays=stats["arrays"])
        dataset_indices = list(stats["manifest"]["dataset_indices"])
        image_ids = list(stats["manifest"]["image_ids"])
        completed_image_ids = list(checkpoint["completed_image_ids"])
        label_rows = [row for row in stats["arrays"]["label"]]
        for variant in CANONICAL_VARIANT_NAMES:
            for suffix in ("intersect", "union", "pred"):
                per_variant_rows[variant][suffix] = [row for row in stats["arrays"][f"{suffix}_{variant}"]]
        windows_processed_total = checkpoint["windows_processed_total"]

    def _checkpoint_payload(*, complete: bool) -> dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "schema": CHECKPOINT_SCHEMA_NAME, "run_mode": args.run_mode, "identity": identity["identity"]["name"],
            "identity_sha256": identity_sha256,
            "power_evaluation_identity_sha256": identity["parent_identity"]["power_evaluation_identity_sha256"],
            "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
            "variant_names": list(CANONICAL_VARIANT_NAMES), "git_commit": git_commit, "class_count": class_count,
            "image_count_expected": image_count, "image_order_digest": image_order_digest,
            "next_dataset_index": next_dataset_index, "completed_image_ids": completed_image_ids,
            "images_completed_count": len(completed_image_ids), "windows_processed_total": windows_processed_total,
            "complete": complete, "created_at_utc": now, "updated_at_utc": now,
        }

    def _write_checkpoint_self_validated(*, complete: bool) -> None:
        payload = _checkpoint_payload(complete=complete)
        validate_checkpoint_structure(payload, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode, class_count=class_count)
        write_checkpoint_atomically(args.checkpoint, payload)

    telemetry_totals = {
        "sample_pulls": 0, "window_enumerations": 0, "backbone_snapshot_calls": 0, "graph_builds": 0,
        "propagation_calls": 0, "probability_interpolation_calls": 0, "score_interpolation_calls": 0,
        "accumulator_finalizations": 0,
    }
    if args.resume and args.checkpoint.exists() and windows_processed_total:
        telemetry_totals["sample_pulls"] = len(completed_image_ids)
        telemetry_totals["window_enumerations"] = windows_processed_total
        telemetry_totals["backbone_snapshot_calls"] = windows_processed_total
        telemetry_totals["graph_builds"] = windows_processed_total
        telemetry_totals["propagation_calls"] = windows_processed_total
        telemetry_totals["probability_interpolation_calls"] = windows_processed_total
        telemetry_totals["score_interpolation_calls"] = windows_processed_total
        telemetry_totals["accumulator_finalizations"] = len(completed_image_ids) * len(CANONICAL_VARIANT_NAMES)

    while next_dataset_index < image_count:
        dataset_index = next_dataset_index
        prepared = _extract_prepared_image(dataset, dataset_index, canonical_image_id=expected_image_ids[dataset_index])
        if prepared.image_id != expected_image_ids[dataset_index]:
            raise StitchingControlIdentityError(
                f"dataset[{dataset_index}] image_id {prepared.image_id!r} disagrees with the precomputed canonical order"
            )

        image_tensor = prepared.image_tensor.to(device=args.device).unsqueeze(0)
        h_img, w_img = prepared.inference_height, prepared.inference_width
        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(h_img, w_img),
            crop_size=SpatialSize(*matched_identity["geometry"]["crop"]),
            stride=SpatialSize(*matched_identity["geometry"]["stride"]),
        )

        stitched, window_count, window_telemetry = stitch_one_image_multi_variant(
            inference, image_tensor, plan, variants=CANONICAL_VARIANTS, class_count=class_count,
            alpha=alpha, steps=steps, affinity_power=affinity_power,
        )

        img_meta = {"img_shape": (h_img, w_img, 3), "ori_shape": (h_img, w_img, 3)}
        predictions = {name: finalize_prediction(tensor, img_meta, align_corners=align_corners) for name, tensor in stitched.items()}

        if not hasattr(dataset, "pre_eval"):
            raise StitchingControlIdentityError("dataset must provide pre_eval() for sufficient-statistic extraction")
        pre_eval_per_variant = {
            name: dataset.pre_eval(pred.cpu().numpy(), dataset_index) for name, pred in predictions.items()
        }

        area_label = None
        for name, stats in pre_eval_per_variant.items():
            area_intersect, area_union, area_pred, this_area_label = (t.numpy() for t in stats[0])
            if area_label is None:
                area_label = this_area_label
            elif not np.array_equal(this_area_label, area_label):
                raise StitchingControlIdentityError(f"{name}: GT sufficient statistics disagree with the shared reference -- GT must be identical by construction")
            per_variant_rows[name]["intersect"].append(area_intersect)
            per_variant_rows[name]["union"].append(area_union)
            per_variant_rows[name]["pred"].append(area_pred)

        label_rows.append(np.asarray(area_label))
        dataset_indices.append(dataset_index)
        image_ids.append(prepared.image_id)
        windows_processed_total += window_count
        completed_image_ids.append(prepared.image_id)
        next_dataset_index = dataset_index + 1

        telemetry_totals["sample_pulls"] += 1
        telemetry_totals["window_enumerations"] += window_count
        telemetry_totals["backbone_snapshot_calls"] += sum(t.backbone_snapshot_calls for t in window_telemetry)
        telemetry_totals["graph_builds"] += sum(t.graph_builds for t in window_telemetry)
        telemetry_totals["propagation_calls"] += sum(t.propagation_calls for t in window_telemetry)
        telemetry_totals["probability_interpolation_calls"] += sum(t.probability_interpolation_calls for t in window_telemetry)
        telemetry_totals["score_interpolation_calls"] += sum(t.score_interpolation_calls for t in window_telemetry)
        telemetry_totals["accumulator_finalizations"] += len(CANONICAL_VARIANT_NAMES)

        _write_per_image_stats_atomically(
            args.per_image_stats, schema_name=identity["run_modes"]["per_image_stats_manifest_schema_name"],
            class_count=class_count, dataset_indices=dataset_indices, image_ids=image_ids,
            label=np.stack(label_rows), per_variant_arrays={
                v: {suffix: np.stack(per_variant_rows[v][suffix]) for suffix in ("intersect", "union", "pred")}
                for v in CANONICAL_VARIANT_NAMES
            },
        )
        _write_checkpoint_self_validated(complete=False)

        del image_tensor, stitched, predictions
        if args.device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - started_at

    if len(image_ids) != image_count or next_dataset_index != image_count or len(completed_image_ids) != image_count:
        raise StitchingControlIdentityError("processed image count does not equal the expected image count")
    if dataset_indices != list(range(image_count)):
        raise StitchingControlIdentityError("processed dataset indices are not the exact canonical full range [0, image_count)")

    from src.k11_k12_full_result_analysis import compute_metrics_from_class_sums, aggregate_class_sums

    per_variant_metrics_fraction: dict[str, dict[str, float]] = {}
    for variant in CANONICAL_VARIANT_NAMES:
        intersect = np.stack(per_variant_rows[variant]["intersect"])
        union = np.stack(per_variant_rows[variant]["union"])
        sums = aggregate_class_sums(intersect, union, np.stack(label_rows))
        metrics = compute_metrics_from_class_sums(sums["intersect"], sums["union"], sums["label"])
        per_variant_metrics_fraction[variant] = metrics

    metrics_percent = {
        variant: {name: 100.0 * per_variant_metrics_fraction[variant][f"{name}_fraction_0_1"] for name in ("aAcc", "mIoU", "mAcc")}
        for variant in CANONICAL_VARIANT_NAMES
    }
    reference_mIoU = metrics_percent["uniform_probability"]["mIoU"]
    deltas = {
        variant: metrics_percent[variant]["mIoU"] - reference_mIoU
        for variant in CANONICAL_VARIANT_NAMES if variant != "uniform_probability"
    }

    result = {
        "schema": schema_name, "run_mode": args.run_mode, "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "power_evaluation_identity_sha256": identity["parent_identity"]["power_evaluation_identity_sha256"],
        "matched_identity_sha256": identity["parent_identity"]["matched_identity_sha256"],
        "git_commit": git_commit, "complete": True, "final": args.run_mode == "full5000",
        "device": args.device, "gpu_model": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "torch_version": torch.__version__, "cuda_version": getattr(torch.version, "cuda", None) or "none",
        "image_count_expected": image_count, "image_count_processed": len(image_ids),
        "image_order_digest": image_order_digest, "windows_processed_total": windows_processed_total,
        "class_count": class_count, "variant_names": list(CANONICAL_VARIANT_NAMES), "metrics": metrics_percent,
        "delta_mIoU_percentage_points_vs_uniform_probability": deltas, "metric_unit": "percent_0_100",
        "metric_source": identity["metrics"]["precision_source"],
        "per_image_stats_manifest_path": str(args.per_image_stats),
        "per_image_stats_manifest_sha256": _sha256_file(args.per_image_stats),
        "per_image_stats_npz_sha256": _sha256_file(_per_image_stats_npz_path(args.per_image_stats)),
        "operation_telemetry": telemetry_totals,
        "phase_runtime_seconds": {"total": elapsed, "shared": elapsed, "per_variant": {v: 0.0 for v in CANONICAL_VARIANT_NAMES}},
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
        "resumed_from_checkpoint": args.resume, "source_git_branch": git_branch, "failure_reason": None,
    }

    temp_result_path = args.result.with_name(args.result.name + f".selfcheck-{os.getpid()}.tmp")
    try:
        write_checkpoint_atomically(temp_result_path, result)
        reloaded = parse_strict_json_document(temp_result_path, label="self-check result")
        verify_record(reloaded, identity, identity_sha256=identity_sha256)
    except Exception:
        temp_result_path.unlink(missing_ok=True)
        raise
    os.replace(temp_result_path, args.result)

    _write_checkpoint_self_validated(complete=True)
    print(
        f"STITCHING CONTROL SUITE {args.run_mode.upper()} PASS images={len(image_ids)} "
        f"uniform_probability_mIoU={metrics_percent['uniform_probability']['mIoU']:.6f} "
        f"deltas={deltas} -> {args.result}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from models.dinotext.cover_dr.stitching_control import StitchingControlError
    from src.k11_k12_power_evaluation_identity import K11K12PowerEvaluationError
    from src.k11_k12_stability_gate_identity import K11K12StabilityGateError

    try:
        return _run_evaluation(args)
    except (StitchingControlIdentityError, StitchingControlError, K11K12PowerEvaluationError, K11K12StabilityGateError, ValueError) as error:
        print(f"STITCHING CONTROL SUITE FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
