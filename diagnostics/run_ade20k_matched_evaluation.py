#!/usr/bin/env python3
"""ADE20K C150 matched evaluator: E3 vs matched k11 vs k12, T=320
finite-step, cross-dataset evaluation of the frozen COCO-2017 Talk2DINO
bridge against the standard 150-class ADE20K validation split.

Reuses the exact finite-step kernel, E3/graph primitives, and shared
per-window E3+k11+k12 orchestration already verified by the COCO-Object
protocol confirmation (``models.dinotext.cover_dr.coco_object_evaluator``,
imported unmodified) -- but never invokes its background-channel step:
ADE20K's standard protocol evaluates no background class and synthesizes
no background channel. Finalization instead reuses
``models.dinotext.cover_dr.matched_power_evaluator.finalize_prediction``,
the plain background-free 150-channel argmax finalizer already verified
for VOC2012's V20 (no-background) protocol.

Requires an already-verified ADE20K dataset-source manifest, verified via
``verify_ade20k_dataset.py verify-manifest`` BEFORE any CUDA/model
initialization. Never converts annotations. Never modifies the dataset.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OVS_ROOT = _REPO_ROOT / "src/open_vocabulary_segmentation"
for _path in (_REPO_ROOT, _OVS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.ade20k_matched_evaluator_checkpoint import (
    resume_dataset_index,
    validate_checkpoint_against_canonical_order,
    validate_checkpoint_structure,
)
from src.ade20k_matched_evaluator_identity import (
    Ade20kMatchedEvaluatorIdentityError,
    load_identity,
    repository_root,
    validate_live_class_order,
    validate_static_configuration,
)
from src.ade20k_matched_evaluator_report import verify_record
from src.ade20k_dataset_identity import load_identity as load_ade20k_source_identity
from src.ade20k_dataset_manifest import resolve_dataset_root as resolve_ade20k_dataset_root
from src.k11_k12_stability_report import write_checkpoint_atomically
from src.matched_k11_k12_identity import MatchedK11K12Error
from src.matched_k11_k12_identity import load_identity as load_matched_identity
from src.native_edge_support_checkpoint import parse_strict_json_document


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Ade20kMatchedEvaluatorIdentityError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ADE20K C150 matched evaluator: E3 vs matched k11 vs k12, T=320 finite-step.")
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, required=True, help="verified ADE20K dataset-source manifest")
    parser.add_argument("--data-root", type=Path, required=True, help="resolved ADE20K root (ADEChallengeData2016 or its unambiguous parent)")
    parser.add_argument("--run-mode", required=True, choices=("pilot20", "pilot100", "full"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--per-image-stats", type=Path, required=True, help="JSON manifest path; a sibling .npz is written alongside it")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _per_image_stats_npz_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".npz")


def _image_order_digest(image_ids: list[str]) -> str:
    payload = json.dumps(image_ids, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_source_before_cuda(root: Path, identity: Mapping[str, Any], manifest_path: Path, data_root: Path) -> dict[str, Any]:
    """Run verify_ade20k_dataset.py verify-manifest BEFORE any CUDA/model
    initialization, as an actual subprocess invocation of the real,
    independent verifier CLI -- never a reimplemented or partial check."""
    manifest = parse_strict_json_document(manifest_path, label="ade20k source manifest")
    verification = identity["verification"]
    if manifest["schema"] != verification["source_manifest_schema_name"]:
        raise Ade20kMatchedEvaluatorIdentityError("ade20k source manifest schema disagrees with the identity's registered contract")
    if manifest["image_count"] != verification["required_manifest_image_count"]:
        raise Ade20kMatchedEvaluatorIdentityError("ade20k source manifest.image_count disagrees with the identity's requirement")

    verify_script = root / "verify_ade20k_dataset.py"
    proc = subprocess.run(
        [sys.executable, str(verify_script), "verify-manifest", "--repo-root", str(root), "--data-root", str(data_root), "--manifest", str(manifest_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise Ade20kMatchedEvaluatorIdentityError(
            f"ade20k source verify-manifest failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return manifest


def _run_evaluation(args: argparse.Namespace) -> int:
    root = Path(args.repo_root)

    identity = load_identity(args.identity, repo_root=root)
    validate_static_configuration(repo_root=root, identity_path=args.identity, check_git=True)
    identity_path = args.identity if args.identity is not None else root / "evaluation_identities/e12_ade20k_matched_evaluator.toml"
    identity_sha256 = _sha256_file(identity_path)

    _verify_source_before_cuda(root, identity, args.source_manifest, args.data_root)
    source_manifest_sha256 = _sha256_file(args.source_manifest)

    image_count = identity["run_modes"][f"{args.run_mode}_images"]
    schema_name = identity["run_modes"][f"{args.run_mode}_schema_name"]
    class_count = identity["dataset"]["class_count"]

    if not args.overwrite and args.result.exists():
        raise Ade20kMatchedEvaluatorIdentityError(f"--result {args.result} already exists; pass --overwrite for explicit resume/overwrite behavior")
    for path in (args.result, args.checkpoint, args.per_image_stats, _per_image_stats_npz_path(args.per_image_stats)):
        path.parent.mkdir(parents=True, exist_ok=True)

    phase_a_checkpoint: dict[str, Any] | None = None
    if args.resume and args.checkpoint.exists():
        phase_a_checkpoint = parse_strict_json_document(args.checkpoint, label="checkpoint")
        validate_checkpoint_structure(phase_a_checkpoint, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode)
        resume_dataset_index(phase_a_checkpoint)  # raises if already complete

    matched_identity_path = root / identity["parent_identities"]["matched_identity_path"]
    try:
        matched_identity = load_matched_identity(matched_identity_path, repo_root=root)
    except MatchedK11K12Error as error:
        raise Ade20kMatchedEvaluatorIdentityError(f"matched parent identity failed validation: {error}") from error
    if matched_identity["identity"]["name"] != identity["parent_identities"]["matched_identity_name"]:
        raise Ade20kMatchedEvaluatorIdentityError("matched parent identity name mismatch")
    alpha = matched_identity["propagation"]["alpha"]
    steps = matched_identity["propagation"]["steps"]
    affinity_power = matched_identity["graph"]["affinity_power"]
    crop = tuple(matched_identity["geometry"]["crop"])
    stride = tuple(matched_identity["geometry"]["stride"])

    from mmcv import Config as MMCVConfig
    from mmseg.datasets import build_dataset
    import main  # noqa: F401 -- registers the custom FloatImage transform, side-effect only

    dataset_config_path = root / identity["dataset"]["dataset_config_relative_path"]
    dataset_cfg = MMCVConfig.fromfile(str(dataset_config_path))
    original_data_root = dataset_cfg.data.test.data_root
    if original_data_root != identity["dataset_root_override"]["canonical_configured_root"]:
        raise Ade20kMatchedEvaluatorIdentityError(
            "ade20k.py's own data_root disagrees with the identity's registered canonical_configured_root -- "
            "refusing to override a config that no longer matches the bound canonical authority"
        )
    # ade20k.py's own img_dir/ann_dir already embed the 'ADEChallengeData2016/'
    # prefix, so the config's data_root override must be the PARENT of that
    # directory -- resolve --data-root through the exact same acceptance
    # logic verify_ade20k_dataset.py's preflight/verify-manifest already
    # applied (either the ADEChallengeData2016 directory itself or its
    # unambiguous parent), so the two never disagree on which form is
    # required.
    ade_source_identity_path = root / identity["parent_identities"]["ade20k_source_identity_path"]
    ade_source_identity = load_ade20k_source_identity(ade_source_identity_path, repo_root=root)
    resolved_ade_root = resolve_ade20k_dataset_root(args.data_root, ade_source_identity)
    dataset_cfg.data.test.data_root = str(resolved_ade_root.parent)
    dataset = build_dataset(dataset_cfg.data.test)

    live_class_names_digest = validate_live_class_order(dataset, identity)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise Ade20kMatchedEvaluatorIdentityError("--device cuda requested but CUDA is not available")

    git_commit = _git(root, "rev-parse", "HEAD")
    git_branch = _git(root, "branch", "--show-current")

    from utils.config import load_config
    from utils.logger import get_logger
    from models import build_model
    from mmcv.runner import CheckpointLoader
    from segmentation.evaluation import build_dinotext_seg_inference
    from torch.utils.data import Subset

    eval_config_path = root / identity["e3_config"]["eval_config_relative_path"]
    cfg = load_config(str(eval_config_path))

    model = build_model(cfg.model)
    checkpoint_path = root / identity["e3_config"]["projection_checkpoint_relative_path"]
    checkpoint = CheckpointLoader.load_checkpoint(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    if args.device == "cuda":
        model.cuda()
    model.eval()

    log_dir = Path(args.result.parent)
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.output = str(log_dir)
    get_logger(cfg)

    inference = build_dinotext_seg_inference(model, Subset(dataset, range(len(dataset))), cfg, str(dataset_config_path))
    inference.reset_evaluation_state()
    align_corners = inference.align_corners

    live_class_count = getattr(inference, "num_classes", None)
    if live_class_count != class_count:
        raise Ade20kMatchedEvaluatorIdentityError(
            f"live inference.num_classes ({live_class_count}) disagrees with the registered class count ({class_count})"
        )

    infos = dataset.img_infos if hasattr(dataset, "img_infos") else dataset.data_infos
    if len(infos) < image_count:
        raise Ade20kMatchedEvaluatorIdentityError(f"dataset has only {len(infos)} images, fewer than the requested {image_count}")
    expected_image_ids = [str(infos[i]["filename"]) for i in range(image_count)]
    image_order_digest = _image_order_digest(expected_image_ids)

    from models.dinotext.cover_dr.coco_object_evaluator import (
        WindowOperationTelemetryE3,
        aggregate_telemetry,
        stitch_one_image_with_e3,
    )
    from models.dinotext.cover_dr.matched_power_evaluator import finalize_prediction
    from segmentation.evaluation.sliding_window_geometry import SlidingWindowPlan, SpatialSize
    from diagnostics.run_k11_k12_stability import _extract_prepared_image

    started_at = time.monotonic()

    next_dataset_index = 0
    completed_image_ids: list[str] = []
    dataset_indices: list[int] = []
    image_ids: list[str] = []
    label_rows: list[np.ndarray] = []
    intersect_e3_rows: list[np.ndarray] = []
    union_e3_rows: list[np.ndarray] = []
    pred_e3_rows: list[np.ndarray] = []
    intersect_k11_rows: list[np.ndarray] = []
    union_k11_rows: list[np.ndarray] = []
    pred_k11_rows: list[np.ndarray] = []
    intersect_k12_rows: list[np.ndarray] = []
    union_k12_rows: list[np.ndarray] = []
    pred_k12_rows: list[np.ndarray] = []
    windows_processed_total = 0

    if phase_a_checkpoint is not None:
        checkpoint_doc = phase_a_checkpoint
        validate_checkpoint_against_canonical_order(checkpoint_doc, expected_image_ids, image_order_digest=image_order_digest)
        next_dataset_index = checkpoint_doc["next_dataset_index"]
        completed_image_ids = list(checkpoint_doc["completed_image_ids"])
        windows_processed_total = checkpoint_doc["windows_processed_total"]
        stats = _load_per_image_stats(args.per_image_stats)
        dataset_indices = list(int(i) for i in stats["manifest"]["dataset_indices"])
        image_ids = list(stats["manifest"]["image_ids"])
        arrays = stats["arrays"]
        label_rows = list(arrays["label"])
        intersect_e3_rows = list(arrays["intersect_E3"])
        union_e3_rows = list(arrays["union_E3"])
        pred_e3_rows = list(arrays["pred_E3"])
        intersect_k11_rows = list(arrays["intersect_k11"])
        union_k11_rows = list(arrays["union_k11"])
        pred_k11_rows = list(arrays["pred_k11"])
        intersect_k12_rows = list(arrays["intersect_k12"])
        union_k12_rows = list(arrays["union_k12"])
        pred_k12_rows = list(arrays["pred_k12"])

    def _checkpoint_payload(*, complete: bool) -> dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "schema": identity["checkpoint"]["schema_name"],
            "run_mode": args.run_mode,
            "identity": identity["identity"]["name"],
            "identity_sha256": identity_sha256,
            "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
            "ade20k_source_identity_sha256": identity["parent_identities"]["ade20k_source_identity_sha256"],
            "ade20k_source_manifest_sha256": source_manifest_sha256,
            "git_commit": git_commit,
            "class_count": class_count,
            "live_class_names_digest": live_class_names_digest,
            "image_count_expected": image_count,
            "image_order_digest": image_order_digest,
            "next_dataset_index": next_dataset_index,
            "completed_image_ids": completed_image_ids,
            "images_completed_count": len(completed_image_ids),
            "windows_processed_total": windows_processed_total,
            "complete": complete,
            "created_at_utc": now,
            "updated_at_utc": now,
        }

    def _write_checkpoint_self_validated(*, complete: bool) -> None:
        payload = _checkpoint_payload(complete=complete)
        validate_checkpoint_structure(payload, identity=identity, identity_sha256=identity_sha256, run_mode=args.run_mode, class_count=class_count)
        write_checkpoint_atomically(args.checkpoint, payload)

    while next_dataset_index < image_count:
        dataset_index = next_dataset_index
        prepared = _extract_prepared_image(dataset, dataset_index, canonical_image_id=expected_image_ids[dataset_index])
        if prepared.image_id != expected_image_ids[dataset_index]:
            raise Ade20kMatchedEvaluatorIdentityError(
                f"dataset[{dataset_index}] image_id {prepared.image_id!r} disagrees with the canonical order {expected_image_ids[dataset_index]!r}"
            )

        image_tensor = prepared.image_tensor
        if args.device == "cuda":
            image_tensor = image_tensor.cuda()
        batched = image_tensor.unsqueeze(0)

        plan = SlidingWindowPlan.build(
            image_size=SpatialSize(prepared.inference_height, prepared.inference_width),
            crop_size=SpatialSize(*crop), stride=SpatialSize(*stride),
        )
        stitched = stitch_one_image_with_e3(
            inference, batched, plan, class_count=class_count, alpha=alpha, steps=steps, affinity_power=affinity_power,
        )

        pred_e3 = finalize_prediction(stitched.e3_stitched, prepared.img_metas, align_corners=align_corners)
        pred_k11 = finalize_prediction(stitched.k11_stitched, prepared.img_metas, align_corners=align_corners)
        pred_k12 = finalize_prediction(stitched.k12_stitched, prepared.img_metas, align_corners=align_corners)
        pred_e3_np = pred_e3[0].detach().cpu().numpy()
        pred_k11_np = pred_k11[0].detach().cpu().numpy()
        pred_k12_np = pred_k12[0].detach().cpu().numpy()

        pre_eval_e3 = dataset.pre_eval(pred_e3_np, dataset_index)[0]
        pre_eval_k11 = dataset.pre_eval(pred_k11_np, dataset_index)[0]
        pre_eval_k12 = dataset.pre_eval(pred_k12_np, dataset_index)[0]
        intersect_e3, union_e3, area_pred_e3, area_label_e3 = (t.numpy() for t in pre_eval_e3)
        intersect_k11, union_k11, area_pred_k11, area_label_k11 = (t.numpy() for t in pre_eval_k11)
        intersect_k12, union_k12, area_pred_k12, area_label_k12 = (t.numpy() for t in pre_eval_k12)
        if not (np.array_equal(area_label_e3, area_label_k11) and np.array_equal(area_label_k11, area_label_k12)):
            raise Ade20kMatchedEvaluatorIdentityError(
                f"dataset[{dataset_index}]: ground-truth area arrays diverged between E3, k11 and k12 variants"
            )

        dataset_indices.append(dataset_index)
        image_ids.append(prepared.image_id)
        label_rows.append(area_label_e3)
        intersect_e3_rows.append(intersect_e3); union_e3_rows.append(union_e3); pred_e3_rows.append(area_pred_e3)
        intersect_k11_rows.append(intersect_k11); union_k11_rows.append(union_k11); pred_k11_rows.append(area_pred_k11)
        intersect_k12_rows.append(intersect_k12); union_k12_rows.append(union_k12); pred_k12_rows.append(area_pred_k12)
        windows_processed_total += stitched.window_count
        completed_image_ids.append(prepared.image_id)
        next_dataset_index = dataset_index + 1

        _write_per_image_stats_atomically(
            args.per_image_stats, schema_name=f"{identity['identity']['name']}-per-image-stats-v1",
            class_count=class_count, live_class_names_digest=live_class_names_digest,
            dataset_indices=dataset_indices, image_ids=image_ids,
            label=np.stack(label_rows),
            intersect_e3=np.stack(intersect_e3_rows), union_e3=np.stack(union_e3_rows), pred_e3=np.stack(pred_e3_rows),
            intersect_k11=np.stack(intersect_k11_rows), union_k11=np.stack(union_k11_rows), pred_k11=np.stack(pred_k11_rows),
            intersect_k12=np.stack(intersect_k12_rows), union_k12=np.stack(union_k12_rows), pred_k12=np.stack(pred_k12_rows),
        )
        _write_checkpoint_self_validated(complete=False)

        del image_tensor, batched, stitched, pred_e3, pred_k11, pred_k12
        if args.device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - started_at

    if len(image_ids) != image_count or next_dataset_index != image_count or dataset_indices != list(range(image_count)):
        raise Ade20kMatchedEvaluatorIdentityError("processed images do not form the exact canonical full range")

    from models.dinotext.cover_dr import compute_full_precision_metrics

    def _metrics(intersect_rows, union_rows, pred_rows) -> dict[str, float]:
        fraction = compute_full_precision_metrics(list(zip(intersect_rows, union_rows, pred_rows, label_rows)))
        return {name: 100.0 * value for name, value in fraction.items()}

    metrics_e3 = _metrics(intersect_e3_rows, union_e3_rows, pred_e3_rows)
    metrics_k11 = _metrics(intersect_k11_rows, union_k11_rows, pred_k11_rows)
    metrics_k12 = _metrics(intersect_k12_rows, union_k12_rows, pred_k12_rows)

    npz_sha256 = _sha256_file(_per_image_stats_npz_path(args.per_image_stats))
    per_image_manifest_sha256 = _sha256_file(args.per_image_stats)

    canonical_window_telemetry = WindowOperationTelemetryE3(
        backbone_snapshot_calls=1, dino_feature_extractions=1, topk_selection_calls=1,
        graph_normalizations=2, finite_step_propagations=2, e3_propagations=0,
        k11_updates=320, k12_updates=320, sigmoid_calls=3, interpolation_calls=3,
    )
    telemetry_totals = aggregate_telemetry([canonical_window_telemetry] * windows_processed_total)

    def _peak_gpu_memory_bytes() -> int:
        return int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0

    result = {
        "schema": schema_name,
        "run_mode": args.run_mode,
        "identity": identity["identity"]["name"],
        "identity_sha256": identity_sha256,
        "matched_identity_sha256": identity["parent_identities"]["matched_identity_sha256"],
        "ade20k_source_identity_sha256": identity["parent_identities"]["ade20k_source_identity_sha256"],
        "ade20k_source_manifest_sha256": source_manifest_sha256,
        "git_commit": git_commit,
        "complete": True,
        "final": args.run_mode == "full",
        "device": args.device,
        "gpu_model": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None) or "none",
        "image_count_expected": image_count,
        "image_count_processed": len(image_ids),
        "image_order_digest": image_order_digest,
        "windows_processed_total": windows_processed_total,
        "class_count": class_count,
        "live_class_names_digest": live_class_names_digest,
        "metrics_E3": metrics_e3,
        "metrics_k11": metrics_k11,
        "metrics_k12": metrics_k12,
        "delta_mIoU_k11_minus_k12_percentage_points": metrics_k11["mIoU"] - metrics_k12["mIoU"],
        "delta_mIoU_k11_minus_E3_percentage_points": metrics_k11["mIoU"] - metrics_e3["mIoU"],
        "delta_mIoU_k12_minus_E3_percentage_points": metrics_k12["mIoU"] - metrics_e3["mIoU"],
        "metric_unit": "percent_0_100",
        "metric_source": "full_precision_area_statistics_from_mmseg_pre_eval",
        "per_image_stats_manifest_path": str(args.per_image_stats),
        "per_image_stats_manifest_sha256": per_image_manifest_sha256,
        "per_image_stats_npz_sha256": npz_sha256,
        "operation_telemetry": telemetry_totals,
        "phase_runtime_seconds": {"total": elapsed},
        "peak_gpu_memory_bytes": _peak_gpu_memory_bytes(),
        "resumed_from_checkpoint": args.resume,
        "source_git_branch": git_branch,
        "failure_reason": None,
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
        f"ADE20K MATCHED EVALUATOR {args.run_mode.upper()} PASS images={len(image_ids)} "
        f"mIoU_E3={metrics_e3['mIoU']:.6f} mIoU_k11={metrics_k11['mIoU']:.6f} mIoU_k12={metrics_k12['mIoU']:.6f} "
        f"delta_k11_k12={result['delta_mIoU_k11_minus_k12_percentage_points']:.6f} -> {args.result}"
    )
    return 0


def _write_per_image_stats_atomically(manifest_path: Path, *, schema_name: str, class_count: int, live_class_names_digest: str, dataset_indices, image_ids, label, **variant_arrays) -> None:
    arrays = {"label": label, **variant_arrays}
    for name, array in arrays.items():
        if array.shape[0] != len(image_ids):
            raise Ade20kMatchedEvaluatorIdentityError(f"per-image-stats array {name!r} row count disagrees with image_ids length")
        if array.shape[1] != class_count:
            raise Ade20kMatchedEvaluatorIdentityError(f"per-image-stats array {name!r} has {array.shape[1]} columns, expected class_count={class_count}")

    npz_path = _per_image_stats_npz_path(manifest_path)
    temp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    save_arrays = {"dataset_indices": np.asarray(dataset_indices, dtype=np.int64)}
    for name, array in arrays.items():
        save_arrays[name] = array.astype(np.int64)
    np.savez(temp_npz, allow_pickle=False, **save_arrays)
    written = temp_npz if temp_npz.suffix == ".npz" else temp_npz.with_suffix(temp_npz.suffix + ".npz")
    os.replace(written, npz_path)
    npz_sha256 = _sha256_file(npz_path)

    manifest = {
        "schema": schema_name, "npz_filename": npz_path.name, "npz_sha256": npz_sha256,
        "class_count": class_count, "live_class_names_digest": live_class_names_digest,
        "image_count": len(image_ids), "dataset_indices": list(dataset_indices),
        "image_ids": list(image_ids), "image_order_digest": _image_order_digest(list(image_ids)),
    }
    write_checkpoint_atomically(manifest_path, manifest)


def _load_per_image_stats(manifest_path: Path) -> dict[str, Any]:
    manifest = parse_strict_json_document(manifest_path, label="per-image-stats manifest")
    npz_path = _per_image_stats_npz_path(manifest_path)
    if manifest["npz_sha256"] != _sha256_file(npz_path):
        raise Ade20kMatchedEvaluatorIdentityError("per-image-stats NPZ SHA256 does not match its manifest; refusing to resume")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return {"manifest": manifest, "arrays": arrays}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run_evaluation(args)
    except (Ade20kMatchedEvaluatorIdentityError, MatchedK11K12Error, ValueError, OSError) as error:
        print(f"ADE20K MATCHED EVALUATOR FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
